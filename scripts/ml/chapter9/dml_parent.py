"""
dml_parent.py
=============

Parent-vantage double / debiased ML (Chernozhukov et al. 2018), the
per-outcome-biological-horizon counterpart of the joint T+25 DML in
run_job.run_dml. One single-target panel per outcome at its own lag
(LE@12 via LAG_LE, TFR@5, U5MR@12 via data_loader_parent_lag).

For each outcome, cross-fit two nuisances on the non-education features:
  g(X) = E[Y | X_other],  m(X) = E[edu | X_other]
then the orthogonal score gives the doubly-robust education effect theta.

Writes: scripts/ml/checkin/dml_parent.json
Run:    python scripts/ml/chapter9/dml_parent.py
"""

import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(ML_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, ML_DIR)

from _shared import write_checkin, ML_CHECKIN          # noqa: E402
from data_loader_parent_lag import (                    # noqa: E402
    load_parent_lag_panels, PARENT_LAG_HORIZONS)
from spec_curve import flatten_panel, flatten_feature_groups  # noqa: E402

TARGETS = ["LE", "TFR", "U5MR"]
N_FOLDS = 5
SEEDS = [42, 43, 44, 45, 46]


def _dml_one_target(panel, seeds=SEEDS, n_folds=N_FOLDS):
    """Orthogonal-score DML for a single-target panel; returns per-seed thetas."""
    flat_groups = flatten_feature_groups(panel)
    X = flatten_panel(panel)
    y_t = panel["y"][:, 0]

    edu_idx = []
    for g in ("wcde_education", "bl_education", "derived"):
        edu_idx.extend(flat_groups.get(g, []))
    edu_idx = sorted(set(edu_idx))
    non_edu_idx = [i for i in range(X.shape[1]) if i not in set(edu_idx)]

    X_edu = X[:, edu_idx].mean(axis=1)
    X_other = X[:, non_edu_idx]

    seed_means = []
    for seed in seeds:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        theta_folds = []
        for tr_idx, te_idx in kf.split(X_other):
            g_model = GradientBoostingRegressor(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                random_state=seed)
            g_model.fit(X_other[tr_idx], y_t[tr_idx])
            resid_y = y_t[te_idx] - g_model.predict(X_other[te_idx])
            m_model = GradientBoostingRegressor(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                random_state=seed)
            m_model.fit(X_other[tr_idx], X_edu[tr_idx])
            resid_edu = X_edu[te_idx] - m_model.predict(X_other[te_idx])
            num = (resid_edu * resid_y).sum()
            den = (resid_edu * resid_edu).sum()
            theta_folds.append(float(num / max(den, 1e-12)))
        seed_means.append(float(np.mean(theta_folds)))
    return seed_means


def main():
    panels = load_parent_lag_panels(mode="joint", verbose=True)
    by_target = {}
    for t in TARGETS:
        print(f"\n=== DML {t} @ T+{PARENT_LAG_HORIZONS[t]} ===")
        seed_means = _dml_one_target(panels[t])
        by_target[t] = {
            "horizon": PARENT_LAG_HORIZONS[t],
            "theta_seed_estimates": seed_means,
            "theta_median": float(np.median(seed_means)),
            "theta_ci95": [float(np.percentile(seed_means, 2.5)),
                           float(np.percentile(seed_means, 97.5))],
            "n_samples": int(panels[t]["y"].shape[0]),
        }
        print(f"  theta_median = {by_target[t]['theta_median']:.4f}  "
              f"ci95 = {by_target[t]['theta_ci95']}")

    out = {
        "method": (
            "Parent-vantage double/debiased ML (orthogonal score, GBM "
            "nuisances, 5 seeds x 5-fold). Per-outcome single-target panels "
            "at biological horizons: LE at T+12 (childrearing window) (LAG_LE), "
            "TFR@5, U5MR@12. theta is "
            "the doubly-robust effect of the aggregate education signal on "
            "each outcome at its own lag."
        ),
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "by_target": by_target,
    }
    write_checkin("dml_parent.json", out, base_dir=ML_CHECKIN,
                  script_path="scripts/ml/chapter9/dml_parent.py")


if __name__ == "__main__":
    main()
