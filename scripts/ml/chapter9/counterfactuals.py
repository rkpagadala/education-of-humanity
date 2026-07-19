# =============================================================================
# scripts/ml/chapter9/counterfactuals.py
#
# Country-counterfactual runner. Uses the pretrained parent-vantage
# transformer checkpoints (three single-target models, each at its
# biological horizon) to predict outcomes for swapped education
# trajectories. Companion to the country-histories chapter.
#
# For each of N country pairs (e.g., Korea/Philippines, Cuba/DR) and each
# outcome read at its biological horizon (LE @ T+12, TFR @ T+5, U5MR @ T+12):
#   1. Predict country A's outcome given country A's actual features.
#   2. Predict country A's outcome given country B's education features
#      (everything else from country A held constant).
#   3. The difference is what the model says A's education trajectory
#      "bought" — the counterfactual education gap × the model's marginal
#      response, at the horizon at which that outcome actually moves.
#
# Each country is scored out-of-fold: A is read by the fold-k model in
# which A was held out, so A is never in that model's training set.
#
# Outputs:
#   scripts/ml/checkin/chapter9_counterfactuals.json
# =============================================================================
"""
counterfactuals.py

Country-swapping counterfactual evaluations on the parent-vantage
universal transformer checkpoints (one single-target model per outcome,
each at its biological horizon).

Requires the universal_parent_lag_{le,tfr,u5mr} fold checkpoints to exist
locally (produced by run_universal_evidence_parent_lag.py).
"""

import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(ML_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, ML_DIR)

from _shared import write_checkin, ML_CHECKIN
from data_loader import country_holdout_mask
from data_loader_parent_lag import load_parent_lag_panels, PARENT_LAG_HORIZONS
from universal_transformer import UniversalTransformer, CHECKPOINT_DIR

TARGETS = ["LE", "TFR", "U5MR"]

# Country pairs to test. Each pair: (country_to_predict, country_whose_edu_to_swap_in).
# Drawn from the country-histories chapter's natural-experiment cases.
DEFAULT_PAIRS = [
    ("south korea", "philippines"),
    ("taiwan", "philippines"),
    ("cuba", "dominican republic"),
    ("sri lanka", "pakistan"),
    ("bangladesh", "pakistan"),
    ("china", "india"),
    ("vietnam", "myanmar"),
    ("botswana", "zimbabwe"),
    ("rwanda", "burundi"),
    ("kenya", "tanzania"),
]

DEFAULT_T_OBSERVATION = 1990   # year whose feature window is the swap input


def _swap_education(panel, idx_target, idx_source):
    """Build a copy of the [W, F] feature window for `idx_target` with
    education features overwritten by those of `idx_source`."""
    edu_groups = ("wcde_education", "bl_education", "derived")
    edu_idx = []
    for g in edu_groups:
        edu_idx.extend(panel["feature_groups"].get(g, []))
    edu_idx = sorted(set(edu_idx))

    X_target = panel["X"][idx_target].copy()
    X_source = panel["X"][idx_source]
    for i in edu_idx:
        X_target[:, i] = X_source[:, i]
    return X_target, panel["mask"][idx_target]


def _sample_index(panel, country_name, T):
    """Return panel row index for (country, T), or None."""
    if country_name not in panel["countries"]:
        return None
    cid = panel["countries"].index(country_name)
    matches = np.where(
        (panel["country_ids"] == cid) & (panel["years"] == T)
    )[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


def _build_single(ck, n_features, window):
    """Rebuild a single-target UniversalTransformer from a checkpoint.

    universal_transformer.build_model_from_checkpoint hardcodes n_outputs=3;
    the parent-lag checkpoints are single-target, so the head size is read
    from the checkpoint here."""
    hp = ck["hp"]
    model = UniversalTransformer(
        n_features=n_features, window=window,
        d_model=hp["d_model"], nhead=hp["nhead"], num_layers=hp["num_layers"],
        ff_mult=hp["ff_mult"], n_outputs=int(ck.get("n_outputs", 1)),
        dropout=hp["dropout"],
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def _predict_single(model, X, mask, y_mean, y_std):
    """Single-target prediction in original outcome units (scalar)."""
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    Mt = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_std = model(Xt, Mt).numpy()[0]
    return float((pred_std * y_std + y_mean)[0])


def _load_outcome_folds(panel, ckpt_prefix, n_folds):
    """Load the n_folds single-target checkpoints for one outcome, with the
    country-holdout validation mask each fold was scored on."""
    n_features = panel["X"].shape[-1]
    window = panel["window"]
    folds = []
    for fold in range(n_folds):
        path = os.path.join(CHECKPOINT_DIR, f"{ckpt_prefix}_fold{fold}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        ck = torch.load(path, weights_only=False)
        model = _build_single(ck, n_features, window)
        is_val = country_holdout_mask(panel, n_folds=n_folds, fold=fold,
                                      seed=ck["hp"]["seed"])
        folds.append({
            "model": model, "y_mean": ck["y_mean"], "y_std": ck["y_std"],
            "is_val": is_val,
        })
    return folds


def run_counterfactuals_oof(pairs=None, T=DEFAULT_T_OBSERVATION,
                            n_folds=5, ckpt_prefix_fmt="universal_parent_lag_{target}",
                            verbose=True):
    """Out-of-fold country-swap counterfactuals on the parent-vantage models.

    One single-target model per outcome, each read at its biological horizon
    (LE @ T+12, TFR @ T+5, U5MR @ T+12). For each country A, the prediction
    uses the fold-k checkpoint in which A was held out, so A is never in that
    model's training set. B's education features are swapped into A's feature
    window; everything else stays A's. The reported gap is what the model
    attributes to A's own education trajectory relative to B's, at the horizon
    at which that outcome moves.

    ckpt_prefix_fmt selects which fold checkpoints to read; "{target}" is
    filled with the lower-cased outcome name. The default reads the canonical
    single-seed checkpoints the residual atlas trains; the cloud big-run reads
    its per-seed checkpoints by passing e.g. "trf_parent_{target}_s42".
    """
    pairs = pairs or DEFAULT_PAIRS
    panels = load_parent_lag_panels(mode="joint", verbose=verbose)

    # Per-outcome fold models + validation masks.
    outcome_folds = {
        t: _load_outcome_folds(panels[t], ckpt_prefix_fmt.format(target=t.lower()),
                               n_folds)
        for t in TARGETS
    }

    def fold_for_sample(folds, idx):
        for fold in range(n_folds):
            if folds[fold]["is_val"][idx]:
                return fold
        return None

    results = []
    for country_a, country_b in pairs:
        rec = {"country_a": country_a, "country_b": country_b, "T": T}
        gap, pred_own, pred_swap, oyear, ofold = {}, {}, {}, {}, {}
        missing = []
        for t in TARGETS:
            panel = panels[t]
            ia = _sample_index(panel, country_a, T)
            ib = _sample_index(panel, country_b, T)
            if ia is None or ib is None:
                missing.append(t)
                continue
            fold = fold_for_sample(outcome_folds[t], ia)
            f = outcome_folds[t][fold]
            p_own = _predict_single(f["model"], panel["X"][ia],
                                    panel["mask"][ia], f["y_mean"], f["y_std"])
            X_swap, M_swap = _swap_education(panel, ia, ib)
            p_swap = _predict_single(f["model"], X_swap, M_swap,
                                     f["y_mean"], f["y_std"])
            gap[t] = p_own - p_swap
            pred_own[t] = p_own
            pred_swap[t] = p_swap
            oyear[t] = T + panel["pred_horizon"]
            ofold[t] = fold

        if missing and len(missing) == len(TARGETS):
            rec["status"] = f"missing_sample (all outcomes): a/b absent for {country_a}/{country_b}"
            results.append(rec)
            if verbose:
                print(f"  {country_a:<20} swap-with {country_b:<20} — missing, skipped")
            continue

        rec.update({
            "horizons": {t: PARENT_LAG_HORIZONS[t] for t in gap},
            "outcome_year": oyear,
            "oof_fold": ofold,
            "pred_a_with_own_edu": pred_own,
            "pred_a_with_b_edu": pred_swap,
            "education_attributable_gap": gap,
        })
        if missing:
            rec["missing_outcomes"] = missing
        results.append(rec)
        if verbose:
            print(f"  {country_a:<20} swap-with {country_b:<20}")
            for t in TARGETS:
                if t in gap:
                    print(f"    {t:<4} @T+{PARENT_LAG_HORIZONS[t]:<2} ({oyear[t]}, oof fold {ofold[t]}): "
                          f"pred(A own)={pred_own[t]:>7.2f}  "
                          f"pred(A with B edu)={pred_swap[t]:>7.2f}  "
                          f"gap={gap[t]:>+7.2f}")

    # Stable dict keyed by "A__B" for traceable paper-number lookups.
    gaps_by_pair = {
        f"{r['country_a']}__{r['country_b']}": r["education_attributable_gap"]
        for r in results if "education_attributable_gap" in r
    }
    return {
        "checkpoint": ckpt_prefix_fmt.format(target="<outcome>"),
        "mode": "out_of_fold_parent_lag",
        "n_folds": n_folds,
        "horizons": dict(PARENT_LAG_HORIZONS),
        "T_observation": T,
        "results": results,
        "gaps_by_pair": gaps_by_pair,
    }


def main():
    out = run_counterfactuals_oof()
    write_checkin("chapter9_counterfactuals.json", out,
                  base_dir=ML_CHECKIN,
                  script_path="scripts/ml/chapter9/counterfactuals.py")
    ok = [r for r in out["results"] if "education_attributable_gap" in r]
    print(f"\nWrote: chapter9_counterfactuals.json with "
          f"{len(ok)}/{len(out['results'])} pairs (out-of-fold, parent-vantage)")


if __name__ == "__main__":
    main()
