# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/run_universal_evidence_parent_lag.py
# Paper:   "The Long Childhood: On the Convergence of Humanity"
#
# Produces:
#   Parent-vantage variant of the universal-biology evidence suite.
#   Trains three SEPARATE single-target UniversalTransformer models,
#   each predicting one outcome at its biological horizon:
#     - TFR  at T + LAG_TFR             = T+5
#     - U5MR at T + LAG_CHILDREARING    = T+12
#     - LE   at T + LAG_LE              = T+12
#
#   Lightweight gate suite (B0/B1/B2 per outcome):
#     B0 baseline       5-fold OOF R² with all features.
#     B1 G1 education   Zero education features at inference; R² drop.
#     B2 G2 GDP         Zero GDP at inference; R² drop.
#
#   This is the per-outcome counterpart to run_universal_evidence.py's
#   T+25 multi-target run, with the architecture and feature pipeline
#   held fixed so the comparison varies only the target horizon.
#
# Inputs:
#   scripts/ml/data_loader_parent_lag.load_parent_lag_panels(mode=...)
#
# Outputs:
#   scripts/ml/checkpoints/universal_parent_lag_<outcome>_fold{k}.pt
#   scripts/ml/checkin/universal_evidence_parent_lag.json
# =============================================================================
"""
run_universal_evidence_parent_lag.py

Three single-target UniversalTransformer models, each trained at its
own biological lag. Reports OOF R² and education / GDP ablation drops
per outcome. Output JSON mirrors universal_evidence.json's key
structure so the two suites are directly comparable.
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

from _shared import write_checkin, ML_CHECKIN
from data_loader import country_holdout_mask, feature_ablation
from data_loader_parent_lag import load_parent_lag_panels, PARENT_LAG_HORIZONS
from universal_transformer import (
    UniversalTransformer, DEFAULT_HP, CHECKPOINT_DIR,
    _standardize_y, _r2,
)

TARGETS = ["LE", "TFR", "U5MR"]
N_FOLDS = 5


def _r2_per_target(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def _train_one_fold_single(panel, fold, hp, n_folds, verbose=True, log_every=40):
    """Train one fold for a single-target panel (y shape [N, 1]).

    Mirrors universal_transformer.train_one_fold but constructs the
    model with n_outputs=1 instead of 3. All hyperparameters and the
    fold-construction seed are inherited from the original module.
    """
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
        "n_outputs": int(y.shape[1]),
    }
    return ck, pred_val, y_val, is_val


def _build_model_from_ck_single(ck, n_features, window):
    """Rebuild a single-target UniversalTransformer from a checkpoint."""
    hp = ck["hp"]
    n_out = int(ck.get("n_outputs", 1))
    model = UniversalTransformer(
        n_features=n_features, window=window,
        d_model=hp["d_model"], nhead=hp["nhead"],
        num_layers=hp["num_layers"], ff_mult=hp["ff_mult"],
        n_outputs=n_out, dropout=hp["dropout"],
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def _run_5fold_single(panel, hp, ckpt_prefix, verbose=True):
    """5-fold OOF for a single-target panel; saves per-fold checkpoints."""
    fold_r2 = []
    oof_pred = np.zeros_like(panel["y"])
    oof_seen = np.zeros(panel["y"].shape[0], dtype=bool)
    t0 = time.time()
    for fold in range(N_FOLDS):
        ck, val_pred, val_true, is_val = _train_one_fold_single(
            panel, fold, hp, n_folds=N_FOLDS, verbose=verbose)
        fold_r2.append(ck["val_r2_per_target"])
        oof_pred[is_val] = val_pred
        oof_seen |= is_val
        path = os.path.join(CHECKPOINT_DIR, f"{ckpt_prefix}_fold{fold}.pt")
        torch.save(ck, path)
        if verbose:
            print(f"  saved {path}")
    elapsed = time.time() - t0
    fold_r2 = np.array(fold_r2)
    overall_r2 = _r2(panel["y"][oof_seen], oof_pred[oof_seen])
    return {
        "fold_r2_per_target": fold_r2.tolist(),
        "fold_r2_mean": fold_r2.mean(axis=0).tolist(),
        "fold_r2_std": fold_r2.std(axis=0).tolist(),
        "overall_r2_oof": overall_r2.tolist(),
        "n_samples": int(oof_seen.sum()),
        "elapsed_sec": elapsed,
    }


def _oof_predict_single(panel, ckpt_prefix):
    """Reload per-fold checkpoints and produce OOF predictions on a
    (possibly-ablated) panel that shares the same fold construction."""
    Xt = torch.tensor(panel["X"], dtype=torch.float32)
    mt = torch.tensor(panel["mask"], dtype=torch.float32)
    n_features = panel["X"].shape[-1]
    window = panel["window"]
    pred_oof = np.zeros_like(panel["y"])
    seen = np.zeros(panel["y"].shape[0], dtype=bool)
    for fold in range(N_FOLDS):
        path = os.path.join(CHECKPOINT_DIR, f"{ckpt_prefix}_fold{fold}.pt")
        ck = torch.load(path, weights_only=False)
        model = _build_model_from_ck_single(ck, n_features, window)
        is_val = country_holdout_mask(panel, n_folds=N_FOLDS, fold=fold,
                                       seed=ck["hp"]["seed"])
        with torch.no_grad():
            pred_std = model(Xt[is_val], mt[is_val]).numpy()
        pred_oof[is_val] = pred_std * ck["y_std"] + ck["y_mean"]
        seen |= is_val
    return pred_oof, seen


def _run_gates_single(panel, ckpt_prefix, target_name):
    """B1 + B2 for a single-target trained model."""
    baseline_pred, seen = _oof_predict_single(panel, ckpt_prefix)
    baseline_r2 = _r2_per_target(panel["y"][seen], baseline_pred[seen])

    p_no_edu = feature_ablation(panel,
        drop_groups=["wcde_education", "bl_education", "derived"])
    pred_no_edu, _ = _oof_predict_single(p_no_edu, ckpt_prefix)
    r2_no_edu = _r2_per_target(panel["y"][seen], pred_no_edu[seen])
    g1_drop = (baseline_r2 - r2_no_edu) / np.maximum(np.abs(baseline_r2), 1e-12)

    p_no_gdp = feature_ablation(panel, drop_groups=["gdp"])
    pred_no_gdp, _ = _oof_predict_single(p_no_gdp, ckpt_prefix)
    r2_no_gdp = _r2_per_target(panel["y"][seen], pred_no_gdp[seen])
    g2_drop = (baseline_r2 - r2_no_gdp) / np.maximum(np.abs(baseline_r2), 1e-12)

    return {
        "n_evaluated": int(seen.sum()),
        "baseline_r2": {target_name: float(baseline_r2[0])},
        "g1_education_ablation": {
            "ablated_r2": {target_name: float(r2_no_edu[0])},
            "r2_drop_fraction": {target_name: float(g1_drop[0])},
            "max_drop": float(g1_drop[0]),
        },
        "g2_gdp_ablation": {
            "ablated_r2": {target_name: float(r2_no_gdp[0])},
            "r2_drop_fraction": {target_name: float(g2_drop[0])},
            "max_drop": float(g2_drop[0]),
        },
    }


def main():
    hp = dict(DEFAULT_HP)
    t_total = time.time()

    print("=" * 72)
    print("PARENT-VANTAGE UNIVERSAL EVIDENCE SUITE")
    print("=" * 72)
    print(f"Hyperparameters: epochs={hp['epochs']}, d_model={hp['d_model']}, "
          f"lr={hp['lr']}, batch={hp['batch_size']}, n_folds={N_FOLDS}")
    print(f"Per-outcome horizons: {PARENT_LAG_HORIZONS}")

    panels = load_parent_lag_panels(mode="joint", verbose=True)

    results_per_outcome = {}
    for tname in ("LE", "TFR", "U5MR"):
        horizon = PARENT_LAG_HORIZONS[tname]
        ckpt_prefix = f"universal_parent_lag_{tname.lower()}"
        print(f"\n=== {tname} @ T+{horizon}: train 5-fold ===")
        panel = panels[tname]
        b0 = _run_5fold_single(panel, hp, ckpt_prefix=ckpt_prefix, verbose=True)

        print(f"\n=== {tname} @ T+{horizon}: B1 + B2 gates ===")
        gates = _run_gates_single(panel, ckpt_prefix, target_name=tname)

        results_per_outcome[tname] = {
            "horizon": horizon,
            "B0_joint_5fold": {
                "fold_r2_mean": b0["fold_r2_mean"],
                "fold_r2_std": b0["fold_r2_std"],
                "overall_r2_oof": b0["overall_r2_oof"],
                "n_samples": b0["n_samples"],
                "elapsed_sec": b0["elapsed_sec"],
            },
            "B1_B2_gates_on_joint": gates,
        }

    out = {
        "method": (
            "Parent-vantage UniversalTransformer (no country embedding). "
            "Three SEPARATE single-target models, one per outcome at its "
            "biological horizon: TFR at T+5, U5MR at T+12, "
            "LE at T+12 (childrearing window). "
            "Same encoder architecture, same feature pipeline, same fold "
            "construction as the T+25 multi-target build (scripts/ml/"
            "run_universal_evidence.py); only the target horizon varies."
        ),
        "horizons": {
            "LE": PARENT_LAG_HORIZONS["LE"],
            "TFR": PARENT_LAG_HORIZONS["TFR"],
            "U5MR": PARENT_LAG_HORIZONS["U5MR"],
        },
        "hyperparameters": {
            "d_model": hp["d_model"], "nhead": hp["nhead"],
            "num_layers": hp["num_layers"], "ff_mult": hp["ff_mult"],
            "dropout": hp["dropout"], "lr": hp["lr"],
            "weight_decay": hp["weight_decay"],
            "batch_size": hp["batch_size"], "epochs": hp["epochs"],
            "patience": hp["patience"], "seed": hp["seed"],
            "n_folds": N_FOLDS,
        },
        "per_outcome": results_per_outcome,
        "total_elapsed_sec": time.time() - t_total,
    }

    write_checkin("universal_evidence_parent_lag.json", out,
                  base_dir=ML_CHECKIN,
                  script_path="scripts/ml/run_universal_evidence_parent_lag.py")

    print("\n" + "=" * 72)
    print("PARENT-VANTAGE EVIDENCE SUMMARY")
    print("=" * 72)
    for tname in ("LE", "TFR", "U5MR"):
        r = results_per_outcome[tname]
        b0 = r["B0_joint_5fold"]
        g = r["B1_B2_gates_on_joint"]
        print(f"\n{tname} @ T+{r['horizon']}:")
        print(f"  5-fold mean R² = {b0['fold_r2_mean'][0]:.4f}  "
              f"(std {b0['fold_r2_std'][0]:.4f})")
        print(f"  OOF R²         = {b0['overall_r2_oof'][0]:.4f}  "
              f"(n={b0['n_samples']})")
        print(f"  G1 edu drop    = {g['g1_education_ablation']['max_drop']:.4f}")
        print(f"  G2 GDP drop    = {g['g2_gdp_ablation']['max_drop']:.4f}")

    print(f"\nTotal elapsed: {out['total_elapsed_sec'] / 60:.1f} min")


if __name__ == "__main__":
    main()
