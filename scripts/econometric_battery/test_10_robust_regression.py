"""
test_10_robust_regression.py — Outlier-robust estimators.

If the linear OLS β is driven by influential outliers, robust estimators
should give substantially different point estimates. Stable β across
estimators is evidence that the result is not driven by a few extreme
points.

Estimators:
  1. OLS (baseline)
  2. Huber M-estimator (k=1.345)
  3. Tukey biweight M-estimator (c=4.685)
  4. Least Absolute Deviations (LAD) / quantile-median
  5. Quantile regression at τ = 0.10, 0.25, 0.75, 0.90
  6. Theil-Sen estimator (median of pairwise slopes)
  7. RLM iteratively reweighted least squares
  8. Trimmed-mean-residual OLS (drop top/bottom 5%)
  9. Winsorize-then-OLS (5% / 95%)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.robust as smrob
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _battery import (  # noqa: E402
    panel, demean, banner, section, pass_fail, write_battery_checkin,
)


def prep(df):
    d = demean(df.dropna(subset=["lower_sec", "parent_edu"]),
               ["lower_sec", "parent_edu"])
    X = sm.add_constant(d[["parent_edu_d"]], has_constant="add").values
    y = d["lower_sec_d"].values
    return d, X, y


def fit_ols(X, y):
    m = sm.OLS(y, X).fit()
    return float(m.params[1]), float(m.bse[1])


def fit_huber(X, y):
    m = sm.RLM(y, X, M=smrob.norms.HuberT(t=1.345)).fit()
    return float(m.params[1]), float(m.bse[1])


def fit_tukey(X, y):
    m = sm.RLM(y, X, M=smrob.norms.TukeyBiweight(c=4.685)).fit()
    return float(m.params[1]), float(m.bse[1])


def fit_quantile(X, y, q=0.5):
    Xdf = pd.DataFrame(X, columns=["const", "x"])
    res = sm.QuantReg(y, Xdf).fit(q=q)
    return float(res.params["x"]), float(res.bse["x"])


def fit_theil_sen(X, y):
    """Median of pairwise slopes (slow O(n²); subsample if n too large)."""
    x = X[:, 1]
    n = len(x)
    if n > 3000:
        rng = np.random.default_rng(2026)
        idx = rng.choice(n, size=3000, replace=False)
        x, y = x[idx], y[idx]
        n = 3000
    # Pairs with x_i != x_j
    slopes = []
    rng = np.random.default_rng(2026)
    # Random sample of pairs (full O(n²) at n=1665 gives 1.4M; manageable)
    if n <= 2000:
        i, j = np.triu_indices(n, k=1)
        dx = x[j] - x[i]
        mask = np.abs(dx) > 1e-9
        slopes = (y[j[mask]] - y[i[mask]]) / dx[mask]
    else:
        K = 200000
        ii = rng.integers(0, n, size=K)
        jj = rng.integers(0, n, size=K)
        keep = (ii != jj) & (np.abs(x[jj] - x[ii]) > 1e-9)
        slopes = (y[jj[keep]] - y[ii[keep]]) / (x[jj[keep]] - x[ii[keep]])
    return float(np.median(slopes)), float(stats.median_abs_deviation(slopes) / np.sqrt(n))


def fit_trimmed(X, y, alpha=0.05):
    """Iterate: OLS → drop largest |resid| 2·alpha → refit."""
    m = sm.OLS(y, X).fit()
    res = m.resid
    lo = np.quantile(res, alpha)
    hi = np.quantile(res, 1 - alpha)
    mask = (res > lo) & (res < hi)
    m2 = sm.OLS(y[mask], X[mask]).fit()
    return float(m2.params[1]), float(m2.bse[1])


def fit_winsorized(X, y, alpha=0.05):
    yw = y.copy()
    xw = X[:, 1].copy()
    for arr in (yw, xw):
        lo = np.quantile(arr, alpha)
        hi = np.quantile(arr, 1 - alpha)
        np.clip(arr, lo, hi, out=arr)
    Xw = np.column_stack([np.ones_like(xw), xw])
    m = sm.OLS(yw, Xw).fit()
    return float(m.params[1]), float(m.bse[1])


def main():
    banner("ECONOMETRIC BATTERY 10 — Robust regression")
    df = panel(include_outcomes=False, include_gdp=False)
    d, X, y = prep(df)
    print(f"Panel: {len(d)} obs · {d['country'].nunique()} countries")

    rows = []

    def add(label, fn):
        b, s = fn()
        t = b / s if s and s == s else float("nan")
        rows.append((label, b, s, t))
        print(f"  {label:<40} β={b:+.4f}  SE={s:.4f}  t={t:+.2f}")

    section("Robust point estimators")
    add("OLS (baseline)",          lambda: fit_ols(X, y))
    add("Huber M (k=1.345)",       lambda: fit_huber(X, y))
    add("Tukey biweight (c=4.685)", lambda: fit_tukey(X, y))
    add("Quantile regression τ=0.50 (LAD)", lambda: fit_quantile(X, y, 0.50))
    add("Theil-Sen (median of pairwise slopes)", lambda: fit_theil_sen(X, y))
    add("Trimmed 5% residuals",   lambda: fit_trimmed(X, y, alpha=0.05))
    add("Winsorized 5/95",         lambda: fit_winsorized(X, y, alpha=0.05))

    section("Quantile regression sweep")
    for q in [0.10, 0.25, 0.75, 0.90]:
        b, s = fit_quantile(X, y, q)
        t = b / s if s else float("nan")
        print(f"  τ={q:.2f}    β={b:+.4f}  SE={s:.4f}  t={t:+.2f}")

    section("Summary")
    bs = [r[1] for r in rows]
    print(f"  β range across estimators: [{min(bs):.4f}, {max(bs):.4f}]")
    print(f"  Spread:                    {max(bs) - min(bs):.4f}")
    print(f"  Max |β - OLS| / OLS:       {max(abs(b - rows[0][1]) for b in bs) / rows[0][1]:.3f}")

    write_battery_checkin("10_robust_regression", {
        "panel": {"n_obs": int(len(d)), "n_countries": int(d["country"].nunique())},
        "results": [{"estimator": l, "beta": b, "se": s, "t": t} for l, b, s, t in rows],
    }, script_path=__file__)

    # Pass: all positive and all within 30% of OLS
    ols = rows[0][1]
    stable = all(b > 0 and abs(b - ols) / ols < 0.30 for b in bs)
    print()
    print(pass_fail(stable, "β stable across robust estimators (within 30% of OLS, all positive)"))


if __name__ == "__main__":
    main()
