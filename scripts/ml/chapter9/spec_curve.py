# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/chapter9/spec_curve.py
# Paper:   "The Long Childhood: On the Convergence of Humanity"
# Section: Chapter 9 — The Panel
#
# Produces:
#   The specification curve. Same panel, same country-clustered folds,
#   same education-ablation, run through four model classes plus the
#   transformer (added at aggregation):
#     - Ridge (L2-regularized linear)
#     - Lasso (L1-regularized linear, high-dim variable selection)
#     - Random Forest (non-linear, no architectural priors)
#     - Gradient Boosting (sklearn implementation; econ's preferred ML)
#   Each architecture reports OOF R² and education R² contribution
#   (R² drop when education features are zeroed at inference).
#
#   NOTE on linear models. Every row here is scored by country-clustered
#   cross-validation — whole countries are held out of training — so each
#   row must be a valid OUT-OF-SAMPLE cross-country predictor. Two linear
#   estimators are deliberately NOT in the curve:
#     • Country fixed effects: a held-out country's fixed effect is never
#       seen in training, is unidentified, and the out-of-fold prediction
#       diverges (R² → large negative). FE is an in-sample within-country
#       estimator.
#     • Unregularized pooled OLS: with ~500 flattened features and whole-
#       country holdout it overfits badly; ablating education sends the
#       ablated R² strongly negative, so the education "drop" exceeds 1.0
#       and is uninterpretable.
#   The linear model class is therefore represented by its REGULARIZED
#   members (ridge, lasso), which are well-posed OOS predictors. The
#   in-sample within-FE β belongs to — and is reported in — the headline
#   panel table with cluster-robust SEs, not in this OOS curve.
#   (`ols_pooled` remains available in _make_model for diagnostics but is
#   not part of the default curve.)
#
#   The transformer row is produced by run_universal_evidence.py and
#   the parent-vantage variant; this script's outputs are merged with
#   those into the final Chapter 9 table.
#
# Outputs:
#   scripts/ml/checkin/spec_curve.json
#
# Compute:
#   CPU-only. ~10-30 minutes on a modern laptop; ~5 minutes on n2-standard-8.
# =============================================================================
"""
spec_curve.py

Multi-method specification curve. Tests whether the panel's
education-load-bearing finding is a transformer artifact or
a finding that survives every reasonable model class.

The discipline: each method uses the SAME flattened panel, SAME
country-clustered 5-fold splits, SAME education-ablation API.
If education's R² contribution is 22-31% across all five methods,
the result is method-agnostic.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(ML_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, ML_DIR)

from _shared import write_checkin, ML_CHECKIN
from data_loader import load_panel, country_holdout_mask

warnings.filterwarnings("ignore")

TARGETS = ["LE", "TFR", "U5MR"]
N_FOLDS = 5
DEFAULT_SEED = 42


def flatten_panel(panel):
    """Convert [B, W, F] feature tensor to [B, W*F] tabular form.

    Also constructs masked-aware features: each timestep's contribution
    is zeroed where the validity mask says the timestep is missing.
    """
    X = panel["X"]  # [B, W, F]
    mask = panel["mask"]  # [B, W]
    B, W, F = X.shape
    X_masked = X * mask[:, :, None]
    return X_masked.reshape(B, W * F)


def flatten_feature_groups(panel):
    """Map feature-group index lists from [F] space to [W*F] flat space."""
    W = panel["window"]
    F = panel["X"].shape[-1]
    flat = {}
    for gname, indices in panel["feature_groups"].items():
        flat[gname] = [w * F + i for w in range(W) for i in indices]
    return flat


def ablate_education_flat(X_flat, panel_flat_groups):
    """Return X_flat copy with all education-related features zeroed."""
    edu_indices = []
    for gname in ("wcde_education", "bl_education", "derived"):
        edu_indices.extend(panel_flat_groups.get(gname, []))
    X_ab = X_flat.copy()
    X_ab[:, edu_indices] = 0.0
    return X_ab


def _r2(y_true, y_pred):
    """Per-target R². Inputs: [N, K]. Output: [K]."""
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def _make_model(name):
    """Factory: returns a fresh sklearn estimator per method name.

    Linear methods use intercept=True since features are zero-mean but
    targets are not (LE ~70, TFR ~3.5, U5MR ~50). RidgeCV / LassoCV
    auto-tune the regularization on each fold.
    """
    if name == "ols_pooled":
        return LinearRegression(fit_intercept=True)    # unregularized linear, no country dummies
    if name == "ridge":
        return RidgeCV(alphas=np.logspace(-2, 3, 12), fit_intercept=True)
    if name == "lasso":
        return LassoCV(alphas=np.logspace(-3, 1, 12), fit_intercept=True,
                       max_iter=20_000, cv=3, random_state=DEFAULT_SEED)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=3,
            n_jobs=-1, random_state=DEFAULT_SEED,
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            min_samples_leaf=5, random_state=DEFAULT_SEED,
        )
    raise ValueError(f"unknown method: {name}")


def _fit_predict_one_target(X_train, y_train, X_val, X_val_ablated, model_name):
    """Train one model for a single target column; return (val_pred, val_pred_ablated).

    sklearn's tree-based regressors don't natively do multi-output well
    on small data; we train per-target to keep each model focused.
    """
    model = _make_model(model_name)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    pred_ab = model.predict(X_val_ablated)
    return pred, pred_ab


def run_one_method(panel, panel_flat_groups, method, verbose=True):
    """Run 5-fold OOF for one architecture, with education ablation.

    Returns dict with per-target R² (baseline and ablated) and R² drop.
    """
    X_flat = flatten_panel(panel)
    X_flat_ablated = ablate_education_flat(X_flat, panel_flat_groups)
    y = panel["y"]
    country_ids = panel["country_ids"]
    n_countries = len(panel["countries"])

    # All methods are cross-country generalizers scored by country-holdout
    # CV; none get country dummies (see header note — FE is unidentified for
    # held-out countries and belongs in the in-sample headline table instead).
    X_design = X_flat
    X_design_ablated = X_flat_ablated

    oof_pred = np.zeros_like(y, dtype=np.float64)
    oof_pred_ablated = np.zeros_like(y, dtype=np.float64)
    seen = np.zeros(y.shape[0], dtype=bool)

    t0 = time.time()
    for fold in range(N_FOLDS):
        is_val = country_holdout_mask(panel, n_folds=N_FOLDS, fold=fold,
                                       seed=DEFAULT_SEED)
        X_tr = X_design[~is_val]
        X_va = X_design[is_val]
        X_va_ab = X_design_ablated[is_val]
        y_tr = y[~is_val]

        for k in range(y.shape[1]):
            pred, pred_ab = _fit_predict_one_target(
                X_tr, y_tr[:, k], X_va, X_va_ab, method)
            oof_pred[is_val, k] = pred
            oof_pred_ablated[is_val, k] = pred_ab
        seen |= is_val
        if verbose:
            print(f"  {method} fold {fold}/{N_FOLDS - 1} done "
                  f"(n_val={int(is_val.sum())})")
    elapsed = time.time() - t0

    r2_base = _r2(y[seen], oof_pred[seen])
    r2_ablated = _r2(y[seen], oof_pred_ablated[seen])
    # Absolute R² drop (additive, not fractional — closer to OLS partial R²)
    r2_drop_abs = r2_base - r2_ablated
    r2_drop_frac = (r2_base - r2_ablated) / np.maximum(np.abs(r2_base), 1e-12)

    return {
        "method": method,
        "elapsed_sec": elapsed,
        "n_evaluated": int(seen.sum()),
        "n_features_input": int(X_design.shape[1]),
        "r2_baseline":      {t: float(v) for t, v in zip(TARGETS, r2_base)},
        "r2_ablated_edu":   {t: float(v) for t, v in zip(TARGETS, r2_ablated)},
        "r2_drop_absolute": {t: float(v) for t, v in zip(TARGETS, r2_drop_abs)},
        "r2_drop_fraction": {t: float(v) for t, v in zip(TARGETS, r2_drop_frac)},
        "r2_drop_abs_max":  float(np.max(r2_drop_abs)),
        "r2_drop_abs_mean": float(np.mean(r2_drop_abs)),
    }


def run_spec_curve(methods=None, verbose=True):
    """Main entry: run all methods on the joint panel; return aggregated dict."""
    methods = methods or [
        "ridge", "lasso",
        "random_forest", "gradient_boosting",
    ]
    panel = load_panel(mode="joint", verbose=verbose)
    panel_flat_groups = flatten_feature_groups(panel)
    if verbose:
        print(f"Panel loaded: n={panel['X'].shape[0]}, "
              f"window={panel['window']}, F={panel['X'].shape[-1]}, "
              f"flat_F={panel['X'].shape[-1] * panel['window']}")
        edu_count = sum(len(panel_flat_groups.get(g, []))
                        for g in ("wcde_education", "bl_education", "derived"))
        print(f"Flat education feature count: {edu_count}")

    results = {}
    t_total = time.time()
    for m in methods:
        if verbose:
            print(f"\n--- {m} ---")
        results[m] = run_one_method(panel, panel_flat_groups, m, verbose=verbose)
        if verbose:
            r = results[m]
            print(f"  baseline R²: " +
                  ", ".join(f"{t}={r['r2_baseline'][t]:.3f}" for t in TARGETS))
            print(f"  edu R² drop (absolute): " +
                  ", ".join(f"{t}={r['r2_drop_absolute'][t]:+.3f}" for t in TARGETS))

    out = {
        "method": (
            "Specification curve across five model classes on the same "
            "panel, same country-clustered 5-fold splits, same education-"
            "ablation API. Each method's row reports baseline R² and the "
            "R² drop when education features (wcde_education, bl_education, "
            "derived) are zeroed at inference. If education is load-bearing, "
            "the drop is large and consistent across methods."
        ),
        "n_folds": N_FOLDS,
        "seed": DEFAULT_SEED,
        "targets": TARGETS,
        "methods": list(results.keys()),
        "results": results,
        "total_elapsed_sec": time.time() - t_total,
    }
    return out


def run_spec_curve_parent(methods=None, verbose=True):
    """Parent-vantage spec curve: the four sklearn model classes on each
    single-target panel at its biological horizon (LE@12 via LAG_LE, TFR@5,
    U5MR@12). Same country-clustered 5-fold education-ablation API as the
    joint curve. The transformer row is produced separately by the parent-
    vantage transformer run (universal_evidence_parent_lag.json)."""
    from data_loader_parent_lag import load_parent_lag_panels, PARENT_LAG_HORIZONS
    methods = methods or [
        "ridge", "lasso", "random_forest", "gradient_boosting",
    ]
    panels = load_parent_lag_panels(mode="joint", verbose=verbose)
    results = {m: {"method": m, "r2_baseline": {}, "r2_ablated_edu": {},
                   "r2_drop_absolute": {}} for m in methods}
    t_total = time.time()
    for tgt in ("LE", "TFR", "U5MR"):
        panel = panels[tgt]
        groups = flatten_feature_groups(panel)
        if verbose:
            print(f"\n=== {tgt} @ T+{PARENT_LAG_HORIZONS[tgt]} "
                  f"(n={panel['X'].shape[0]}) ===")
        for m in methods:
            r = run_one_method(panel, groups, m, verbose=verbose)
            # single-target panel -> one column; key is TARGETS[0] regardless
            only = next(iter(r["r2_drop_absolute"]))
            results[m]["r2_baseline"][tgt] = r["r2_baseline"][only]
            results[m]["r2_ablated_edu"][tgt] = r["r2_ablated_edu"][only]
            results[m]["r2_drop_absolute"][tgt] = r["r2_drop_absolute"][only]
    for m in methods:
        drops = list(results[m]["r2_drop_absolute"].values())
        results[m]["r2_drop_abs_mean"] = float(np.mean(drops))
    out = {
        "method": (
            "Parent-vantage specification curve: four sklearn model classes, "
            "country-clustered 5-fold, education-ablation, on single-target "
            "panels at per-outcome biological horizons (LE at T+12 "
            "(childrearing window) via LAG_LE, "
            "TFR@5 LAG_TFR, U5MR@12 LAG_CHILDREARING). Transformer row from "
            "universal_evidence_parent_lag.json."
        ),
        "horizons": {t: PARENT_LAG_HORIZONS[t] for t in ("LE", "TFR", "U5MR")},
        "n_folds": N_FOLDS,
        "seed": DEFAULT_SEED,
        "targets": TARGETS,
        "methods": list(results.keys()),
        "results": results,
        "total_elapsed_sec": time.time() - t_total,
    }
    return out


def _print_spec_table(out, title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"{'method':<22} {'LE drop':>10} {'TFR drop':>10} {'U5MR drop':>10} "
          f"{'mean drop':>10}")
    for m, r in out["results"].items():
        drops = r["r2_drop_absolute"]
        print(f"{m:<22} {drops['LE']:>+10.3f} {drops['TFR']:>+10.3f} "
              f"{drops['U5MR']:>+10.3f} {r['r2_drop_abs_mean']:>+10.3f}")
    print(f"\nTotal elapsed: {out['total_elapsed_sec'] / 60:.1f} min")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", action="store_true",
                    help="Parent-vantage spec curve (per-outcome biological "
                         "horizons) -> spec_curve_parent.json")
    args = ap.parse_args()

    if args.parent:
        out = run_spec_curve_parent()
        write_checkin("spec_curve_parent.json", out, base_dir=ML_CHECKIN,
                      script_path="scripts/ml/chapter9/spec_curve.py")
        _print_spec_table(out, "PARENT-VANTAGE SPEC CURVE — education R² "
                               "contribution by method")
        return

    out = run_spec_curve()
    write_checkin("spec_curve.json", out, base_dir=ML_CHECKIN,
                  script_path="scripts/ml/chapter9/spec_curve.py")
    _print_spec_table(out, "SPECIFICATION CURVE — education R² "
                           "contribution by method")


if __name__ == "__main__":
    main()
