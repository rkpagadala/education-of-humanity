# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/universal_transformer.py
# Paper:   "The Long Childhood: On the Convergence of Humanity"
#
# Produces:
#   The universal-biology architecture. Same prediction function applied
#   to every country — no per-country embedding, no country identity in
#   the model. If the paper's claim is biological (same mechanism in all
#   populations), this is the right architecture. Niger and Korea differ
#   only in their feature values, not in any learned per-country offset.
#
#   Contrast with the prior PanelTransformer build, which included an
#   8-or-64-dim country embedding equivalent to country fixed effects.
#   That architecture absorbed each country's baseline level and could
#   not generalize across countries — saturating Niger's features still
#   left the embedding saying "this is Niger, baseline TFR ~7."
#
# Inputs:
#   scripts/ml/data_loader.load_panel(mode=...)
#
# Outputs:
#   scripts/ml/checkpoints/universal_fold{k}.pt
#   scripts/ml/checkpoints/universal_summary.json
# =============================================================================
"""
universal_transformer.py

Encoder-only transformer over feature sequences ONLY — no country
identity. The model learns a single prediction function
f(features_t-15, features_t-10, features_t-5, features_t) -> (LE, TFR, U5MR)
applied universally to every (country, year).

Country IDs are still threaded through so country-holdout cross-
validation works, but the model never receives them.
"""

import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, SCRIPT_DIR)

from data_loader import load_panel, country_holdout_mask

CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEFAULT_HP = dict(
    d_model=64,
    nhead=4,
    num_layers=2,
    ff_mult=2,
    dropout=0.25,
    batch_size=256,
    epochs=400,
    lr=3e-3,
    weight_decay=1e-3,
    patience=60,
    seed=42,
    num_threads=8,
)


class UniversalTransformer(nn.Module):
    """No country embedding. Pure feature-based prediction.

    Same forward function applied to every country. The country identity
    is unavailable to the model by construction. If the paper's mechanism
    is universal across human populations, this architecture should still
    achieve high R²; if R² collapses, the country-embedding model was
    relying on per-country memorization.
    """

    def __init__(self, n_features, window, d_model, nhead, num_layers,
                 ff_mult, n_outputs=3, dropout=0.1):
        super().__init__()
        self.feature_proj = nn.Linear(n_features, d_model)
        self.position_emb = nn.Embedding(window, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_outputs),
        )

    def forward(self, X, mask):
        """X: [B, window, F]   mask: [B, window]
        Returns: [B, n_outputs]
        """
        B, W, _ = X.shape
        proj = self.feature_proj(X)                     # [B, W, d_model]
        pos = self.position_emb.weight.unsqueeze(0)     # [1, W, d_model]
        h = proj + pos
        # src_key_padding_mask: True where padded (mask=0 means padded)
        key_pad = (mask == 0)
        h = self.encoder(h, src_key_padding_mask=key_pad)
        # Mean-pool over the window, weighted by the validity mask
        m = mask.unsqueeze(-1)                          # [B, W, 1]
        h_pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)
        return self.head(h_pooled)


def _standardize_y(y):
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (y - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def _r2(y_true, y_pred):
    """Per-target R² over axis 0."""
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def train_one_fold(panel, fold, hp, n_folds=5, verbose=True, log_every=40):
    torch.manual_seed(hp["seed"] + fold)
    np.random.seed(hp["seed"] + fold)
    torch.set_num_threads(hp["num_threads"])

    is_val = country_holdout_mask(panel, n_folds=n_folds, fold=fold, seed=hp["seed"])
    X = panel["X"]
    y = panel["y"]
    mask = panel["mask"]

    X_train, X_val = X[~is_val], X[is_val]
    y_train, y_val = y[~is_val], y[is_val]
    m_train, m_val = mask[~is_val], mask[is_val]

    y_train_std, y_mean, y_std = _standardize_y(y_train)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train_std, dtype=torch.float32)
    mt = torch.tensor(m_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    mv = torch.tensor(m_val, dtype=torch.float32)

    ds = TensorDataset(Xt, mt, yt)
    dl = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True)

    n_features = X.shape[-1]
    window = panel["window"]

    model = UniversalTransformer(
        n_features=n_features, window=window,
        d_model=hp["d_model"], nhead=hp["nhead"],
        num_layers=hp["num_layers"], ff_mult=hp["ff_mult"],
        n_outputs=y.shape[1], dropout=hp["dropout"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                             weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()

    best_val_r2_mean = -np.inf
    best_state = None
    best_epoch = -1
    patience = hp["patience"]
    no_improve = 0
    for epoch in range(hp["epochs"]):
        model.train()
        epoch_loss = 0.0
        for Xb, mb, yb in dl:
            opt.zero_grad()
            pred = model(Xb, mb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * Xb.size(0)
        epoch_loss /= len(ds)

        model.eval()
        with torch.no_grad():
            pred_val_std = model(Xv, mv).numpy()
        pred_val = pred_val_std * y_std + y_mean
        val_r2 = _r2(y_val, pred_val)
        val_mean = float(np.mean(val_r2))

        if val_mean > best_val_r2_mean:
            best_val_r2_mean = val_mean
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % log_every == 0 or epoch == hp["epochs"] - 1):
            print(f"  fold {fold} epoch {epoch:3d}: train_loss={epoch_loss:.4f}  "
                  f"val_R²={val_r2.tolist()} (mean={val_mean:.4f})")
        if no_improve >= patience:
            if verbose:
                print(f"  fold {fold}: early stop @ epoch {epoch}, "
                      f"best epoch {best_epoch}, best val R² mean {best_val_r2_mean:.4f}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_val_std = model(Xv, mv).numpy()
    pred_val = pred_val_std * y_std + y_mean
    val_r2 = _r2(y_val, pred_val)

    ck = {
        "state_dict": best_state,
        "hp": hp,
        "y_mean": y_mean,
        "y_std": y_std,
        "val_r2_per_target": val_r2.tolist(),
        "best_epoch": best_epoch,
        "n_train": int((~is_val).sum()),
        "n_val": int(is_val.sum()),
    }
    return ck, pred_val, y_val, is_val


def build_model_from_checkpoint(ck, n_features, window):
    hp = ck["hp"]
    model = UniversalTransformer(
        n_features=n_features, window=window,
        d_model=hp["d_model"], nhead=hp["nhead"], num_layers=hp["num_layers"],
        ff_mult=hp["ff_mult"], n_outputs=3, dropout=hp["dropout"],
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def run_5fold(mode="joint", hp=None, save=True, verbose=True,
              checkpoint_prefix="universal"):
    hp = hp or dict(DEFAULT_HP)
    panel = load_panel(mode=mode, verbose=verbose)
    n_folds = 5
    fold_r2 = []
    oof_pred = np.zeros_like(panel["y"])
    oof_seen = np.zeros(panel["y"].shape[0], dtype=bool)
    t0 = time.time()
    for fold in range(n_folds):
        ck, val_pred, val_true, is_val = train_one_fold(
            panel, fold, hp, n_folds=n_folds, verbose=verbose)
        fold_r2.append(ck["val_r2_per_target"])
        oof_pred[is_val] = val_pred
        oof_seen |= is_val
        if save:
            path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_prefix}_fold{fold}.pt")
            torch.save(ck, path)
            if verbose:
                print(f"  saved {path}")
    elapsed = time.time() - t0
    fold_r2 = np.array(fold_r2)
    overall_r2 = _r2(panel["y"][oof_seen], oof_pred[oof_seen])

    summary = {
        "mode": mode,
        "architecture": "UniversalTransformer (no country embedding)",
        "n_folds": n_folds,
        "elapsed_sec": elapsed,
        "fold_r2_per_target": fold_r2.tolist(),
        "fold_r2_mean": fold_r2.mean(axis=0).tolist(),
        "fold_r2_std": fold_r2.std(axis=0).tolist(),
        "overall_r2_oof": overall_r2.tolist(),
        "targets": ["LE", "TFR", "U5MR"],
        "n_samples": int(oof_seen.sum()),
    }
    if save:
        with open(os.path.join(CHECKPOINT_DIR, f"{checkpoint_prefix}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=float)
    if verbose:
        print(f"\n=== {checkpoint_prefix} summary ===")
        for i, t in enumerate(summary["targets"]):
            print(f"  {t}: mean fold R²={summary['fold_r2_mean'][i]:.4f} "
                  f"±{summary['fold_r2_std'][i]:.4f}  "
                  f"overall oof R²={summary['overall_r2_oof'][i]:.4f}")
        print(f"  elapsed: {elapsed:.1f}s")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="joint",
                    choices=["joint", "wcde_only", "bl_only", "no_education",
                             "education_only", "gdp_only"])
    ap.add_argument("--epochs", type=int, default=DEFAULT_HP["epochs"])
    args = ap.parse_args()

    hp = dict(DEFAULT_HP)
    hp["epochs"] = args.epochs
    run_5fold(mode=args.mode, hp=hp)
