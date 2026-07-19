"""
test_16_functional_form.py — Variable transformations and link functions.

While test 07 covers polynomial/spline/threshold deviations from linearity,
this script tests transformations of the variables themselves:
log, arcsine-sqrt (Anscombe), Box-Cox, logit, identity, square-root.

The headline education variable is a bounded percentage [0, 100]. Bounded
outcomes can be better modeled in transformed space; the question is
whether β remains sign-stable and substantively similar.

Transformations tested:
  1. Identity (baseline)
  2. Log(y + 1)
  3. Arcsine-sqrt of y/100 (Anscombe variance-stabilising for proportions)
  4. Logit of y/100 (with small-number adjustment)
  5. Square-root
  6. Box-Cox optimal λ
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _battery import (  # noqa: E402
    panel, demean, banner, section, pass_fail, write_battery_checkin,
)


def transform(y, mode):
    if mode == "identity":
        return y
    if mode == "log":
        return np.log(y + 1)
    if mode == "sqrt":
        return np.sqrt(np.clip(y, 0, None))
    if mode == "arcsine":
        p = np.clip(y / 100.0, 0.001, 0.999)
        return np.arcsin(np.sqrt(p))
    if mode == "logit":
        p = np.clip(y / 100.0, 0.005, 0.995)
        return np.log(p / (1 - p))
    raise ValueError(f"unknown mode {mode}")


def fit_with_transform(df, ymode, xmode) -> dict:
    d = df.dropna(subset=["lower_sec", "parent_edu"]).copy()
    d["y_t"] = transform(d["lower_sec"].values, ymode)
    d["x_t"] = transform(d["parent_edu"].values, xmode)
    d = demean(d, ["y_t", "x_t"])
    X = sm.add_constant(d[["x_t_d"]], has_constant="add")
    m = sm.OLS(d["y_t_d"], X).fit(cov_type="cluster",
                                   cov_kwds={"groups": d["country"]})
    return {"ymode": ymode, "xmode": xmode,
            "beta": float(m.params["x_t_d"]),
            "se": float(m.bse["x_t_d"]),
            "t": float(m.tvalues["x_t_d"]),
            "r2": float(m.rsquared)}


def box_cox_search(df):
    """Search over λ for Box-Cox on parent_edu (positive series)."""
    d = df.dropna(subset=["lower_sec", "parent_edu"]).copy()
    d = d[d["parent_edu"] > 0]
    bestlam, bestll = None, -np.inf
    for lam in np.linspace(-1.0, 2.0, 31):
        if lam == 0:
            xt = np.log(d["parent_edu"])
        else:
            xt = (d["parent_edu"] ** lam - 1) / lam
        d2 = d.copy()
        d2["xt"] = xt
        d2 = demean(d2, ["lower_sec", "xt"])
        X = sm.add_constant(d2[["xt_d"]], has_constant="add")
        m = sm.OLS(d2["lower_sec_d"], X).fit()
        ll = float(m.llf) + (lam - 1) * float(np.log(d["parent_edu"]).sum())
        if ll > bestll:
            bestll = ll
            bestlam = lam
    # Refit at best λ
    if bestlam == 0:
        xt = np.log(d["parent_edu"])
    else:
        xt = (d["parent_edu"] ** bestlam - 1) / bestlam
    d["xt"] = xt
    d = demean(d, ["lower_sec", "xt"])
    X = sm.add_constant(d[["xt_d"]], has_constant="add")
    m = sm.OLS(d["lower_sec_d"], X).fit(cov_type="cluster",
                                        cov_kwds={"groups": d["country"]})
    return {"best_lambda": float(bestlam),
            "beta": float(m.params["xt_d"]),
            "t": float(m.tvalues["xt_d"]),
            "r2": float(m.rsquared)}


def main():
    banner("ECONOMETRIC BATTERY 16 — Functional form / transformations")
    df = panel(include_outcomes=False, include_gdp=False)
    print(f"Panel: {len(df)} obs · {df['country'].nunique()} countries")

    section("Symmetric transformations (same on x and y)")
    rows = []
    for mode in ["identity", "log", "sqrt", "arcsine", "logit"]:
        r = fit_with_transform(df, mode, mode)
        rows.append(r)
        print(f"  {mode:<10}  β={r['beta']:+.4f}  SE={r['se']:.4f}  "
              f"t={r['t']:+.2f}  R²={r['r2']:.4f}")

    section("Box-Cox search on parent_edu")
    bc = box_cox_search(df)
    print(f"  best λ:   {bc['best_lambda']:+.2f}")
    print(f"  β at best λ: {bc['beta']:+.4f}  t={bc['t']:+.2f}  R²={bc['r2']:.4f}")

    section("Cross transformations (y identity, x varies)")
    cross = []
    for xmode in ["identity", "log", "sqrt", "arcsine", "logit"]:
        r = fit_with_transform(df, "identity", xmode)
        cross.append(r)
        print(f"  y=identity, x={xmode:<10}  β={r['beta']:+.4f}  t={r['t']:+.2f}  R²={r['r2']:.4f}")

    section("Verdict")
    all_pos = all(r["beta"] > 0 for r in rows + cross)
    all_sig = all(abs(r["t"]) > 5 for r in rows + cross)
    print(f"  Sign-stable across transforms:  {all_pos}")
    print(f"  All |t| > 5:                    {all_sig}")

    write_battery_checkin("16_functional_form", {
        "symmetric": rows,
        "cross": cross,
        "box_cox": bc,
    }, script_path=__file__)

    print()
    print(pass_fail(all_pos and all_sig, "β positive and highly significant under every transform"))


if __name__ == "__main__":
    main()
