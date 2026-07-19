# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/longrun/panel_fe.py
# Paper:   "The Long Childhood: On the Convergence of Humanity"
#
# Panel-FE OLS + GBM baseline runners for the longrun (pre-medicine) test.
#
# These are the simple-spec companions to the transformer. They mirror
# the paper's primary spec (country fixed effects, education at T,
# outcome at T+25, country-clustered SE) on the smaller pre-1950 panel.
# =============================================================================
"""
Panel-FE OLS + GBM baseline runners for the longrun (pre-medicine) test.

These are the simple-spec companions to the transformer. They mirror
the paper's primary spec (country fixed effects, education at T,
outcome at T+25, country-clustered SE) on the smaller pre-1950 panel.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold


TARGETS = ["LE", "U5MR", "TFR"]


def _target_col(panel, target_name):
    """Column index of target_name in panel['y'].

    The longrun panel's y columns are mode-dependent (within = [LE, U5MR];
    forward = [LE, U5MR, TFR]), so resolve against panel['target_names']
    when present, falling back to the global 3-target order otherwise.
    """
    names = panel.get("target_names", TARGETS)
    return names.index(target_name)


def _flatten_window(panel):
    """Average each feature across the WINDOW timesteps where mask=1.
    Returns (N, F) array. NaN-aware via the mask channel."""
    X = panel["X"]   # (N, W, F)
    mask = panel["mask"]   # (N, W)
    weighted = (X * mask[:, :, None]).sum(axis=1)
    denom = mask.sum(axis=1)[:, None].clip(min=1.0)
    return weighted / denom


def _sum_edu_features(X_flat, panel):
    """Sum the WCDE-education features into one 'education_t' regressor."""
    edu_cols = panel["feature_groups"]["wcde_education"]
    return X_flat[:, edu_cols].sum(axis=1)


def run_country_fe_ols(panel, target_name, seed=42):
    """OLS with country fixed effects, education-summed regressor,
    country-clustered standard errors, 5-fold country-grouped OOF R².
    """
    ti = _target_col(panel, target_name)
    y = panel["y"][:, ti]
    cids = panel["country_ids"]
    X_flat = _flatten_window(panel)
    edu = _sum_edu_features(X_flat, panel)

    # OLS with country FE: regress (y - country_mean(y)) on (edu - country_mean(edu))
    df = pd.DataFrame({"y": y, "edu": edu, "cid": cids})
    df["y_demean"] = df["y"] - df.groupby("cid")["y"].transform("mean")
    df["edu_demean"] = df["edu"] - df.groupby("cid")["edu"].transform("mean")
    num = (df["y_demean"] * df["edu_demean"]).sum()
    den = (df["edu_demean"] ** 2).sum()
    beta = float(num / max(den, 1e-12))

    # Cluster-robust SE (CR1)
    residual = df["y_demean"] - beta * df["edu_demean"]
    df["uX"] = residual * df["edu_demean"]
    cluster_sums = df.groupby("cid")["uX"].sum()
    meat = (cluster_sums ** 2).sum()
    bread = max(den, 1e-12)
    n_clusters = df["cid"].nunique()
    se = float(np.sqrt(meat / bread**2) * np.sqrt(n_clusters / max(n_clusters - 1, 1)))

    # Country-clustered OOF R²
    n_folds = min(5, n_clusters)
    gkf = GroupKFold(n_splits=n_folds)
    oof_pred = np.zeros_like(y)
    for tr, te in gkf.split(X_flat, y, groups=cids):
        country_means_tr = pd.Series(y[tr], index=cids[tr]).groupby(level=0).mean().to_dict()
        edu_means_tr    = pd.Series(edu[tr], index=cids[tr]).groupby(level=0).mean().to_dict()
        # Compute β on train fold using within transformation
        y_dt = y[tr]   - np.array([country_means_tr[c] for c in cids[tr]])
        e_dt = edu[tr] - np.array([edu_means_tr[c]    for c in cids[tr]])
        b = (y_dt * e_dt).sum() / max((e_dt ** 2).sum(), 1e-12)
        # Predict on test using train country means (or train-set mean for new countries)
        global_y_mean = float(y[tr].mean())
        global_e_mean = float(edu[tr].mean())
        for i_te in te:
            c = int(cids[i_te])
            c_mean = country_means_tr.get(c, global_y_mean)
            e_mean = edu_means_tr.get(c,    global_e_mean)
            oof_pred[i_te] = c_mean + b * (edu[i_te] - e_mean)
    ss_res = ((y - oof_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_oof = float(1.0 - ss_res / max(ss_tot, 1e-12))

    return {
        "beta_edu": beta,
        "cluster_se": se,
        "r2_oof": r2_oof,
        "n": int(len(y)),
        "n_countries": int(n_clusters),
        "target": target_name,
    }


def run_gbm_with_edu_ablation(panel, target_name, seed=42):
    """GBM trained on flattened features, country-grouped 5-fold OOF.
    Ablates education at inference to measure R² drop.
    """
    ti = _target_col(panel, target_name)
    y = panel["y"][:, ti]
    cids = panel["country_ids"]
    X_flat = _flatten_window(panel)
    edu_cols = panel["feature_groups"]["wcde_education"]

    n_folds = min(5, int(np.unique(cids).size))
    gkf = GroupKFold(n_splits=n_folds)
    oof_pred = np.zeros_like(y)
    oof_pred_ab = np.zeros_like(y)
    for tr, te in gkf.split(X_flat, y, groups=cids):
        gbm = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=seed,
        )
        gbm.fit(X_flat[tr], y[tr])
        oof_pred[te] = gbm.predict(X_flat[te])
        X_te_ab = X_flat[te].copy()
        X_te_ab[:, edu_cols] = 0.0
        oof_pred_ab[te] = gbm.predict(X_te_ab)

    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_base = 1.0 - ((y - oof_pred) ** 2).sum() / max(ss_tot, 1e-12)
    r2_ab = 1.0 - ((y - oof_pred_ab) ** 2).sum() / max(ss_tot, 1e-12)
    return {
        "r2_baseline": float(r2_base),
        "r2_ablated_edu": float(r2_ab),
        "r2_drop": float(r2_base - r2_ab),
        "n": int(len(y)),
        "n_countries": int(np.unique(cids).size),
        "target": target_name,
    }
