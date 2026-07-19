"""
test_09_influence.py — Influence diagnostics battery.

Tests whether the headline β is driven by a small number of influential
observations or countries.

Diagnostics:
  1. Cook's distance per country (group-level)
  2. DFBETAs per country
  3. Leave-one-country-out β distribution
  4. Hat-matrix leverage per country (group leverage)
  5. Studentized residuals
  6. Maximum-influence countries by each criterion
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _battery import (  # noqa: E402
    panel, demean, banner, section, pass_fail, write_battery_checkin,
)


def baseline_fit(df):
    d = demean(df.dropna(subset=["lower_sec", "parent_edu"]), ["lower_sec", "parent_edu"])
    X = sm.add_constant(d[["parent_edu_d"]], has_constant="add")
    m = sm.OLS(d["lower_sec_d"], X).fit(cov_type="cluster",
                                        cov_kwds={"groups": d["country"]})
    return d, X, m


def cooks_d_per_country(df) -> dict:
    """Group Cook's D: refit dropping each country; D_c = (β̂ - β̂_(-c))²·(X'X)_11 / (k·σ̂²)."""
    d, X, m = baseline_fit(df)
    beta_full = float(m.params["parent_edu_d"])
    sigma2 = float((m.resid ** 2).sum() / max(int(m.nobs) - X.shape[1], 1))
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Ds = []
    for c in d["country"].unique():
        sub = d[d["country"] != c]
        Xs = sm.add_constant(sub[["parent_edu_d"]], has_constant="add")
        ms = sm.OLS(sub["lower_sec_d"], Xs).fit()
        b = float(ms.params["parent_edu_d"])
        D = ((beta_full - b) ** 2 / max(XtX_inv[1, 1] * sigma2, 1e-12)) / 2  # k=2
        Ds.append((c, D, b))
    Ds.sort(key=lambda x: -x[1])
    return {"top10_cooks_d": [{"country": c, "D": float(D), "beta_without": float(b)}
                              for c, D, b in Ds[:10]],
            "max_D": float(Ds[0][1]),
            "rule_of_thumb_4_over_n": 4.0 / d["country"].nunique()}


def dfbetas_per_country(df) -> dict:
    d, X, m = baseline_fit(df)
    beta_full = float(m.params["parent_edu_d"])
    se_full = float(m.bse["parent_edu_d"])
    df_betas = []
    for c in d["country"].unique():
        sub = d[d["country"] != c]
        Xs = sm.add_constant(sub[["parent_edu_d"]], has_constant="add")
        ms = sm.OLS(sub["lower_sec_d"], Xs).fit()
        b = float(ms.params["parent_edu_d"])
        df_betas.append((c, (beta_full - b) / se_full, b))
    df_betas.sort(key=lambda x: -abs(x[1]))
    return {"top10_abs_dfbeta": [{"country": c, "dfbeta": float(d), "beta_without": float(b)}
                                  for c, d, b in df_betas[:10]],
            "rule_of_thumb_2_over_sqrtN": 2.0 / np.sqrt(d["country"].nunique())}


def leave_one_country_out(df) -> dict:
    d, X, m = baseline_fit(df)
    beta_full = float(m.params["parent_edu_d"])
    betas = []
    for c in d["country"].unique():
        sub = d[d["country"] != c]
        Xs = sm.add_constant(sub[["parent_edu_d"]], has_constant="add")
        ms = sm.OLS(sub["lower_sec_d"], Xs).fit()
        betas.append((c, float(ms.params["parent_edu_d"])))
    betas.sort(key=lambda x: x[1])
    bvals = [b for _, b in betas]
    return {
        "n_countries": int(len(bvals)),
        "full_beta": beta_full,
        "min_beta": float(min(bvals)),
        "max_beta": float(max(bvals)),
        "median_beta": float(np.median(bvals)),
        "spread": float(max(bvals) - min(bvals)),
        "5_smallest_when_removed": [{"country": c, "beta": float(b)}
                                    for c, b in betas[:5]],
        "5_largest_when_removed": [{"country": c, "beta": float(b)}
                                   for c, b in betas[-5:]],
    }


def leverage_per_country(df) -> dict:
    d, X, m = baseline_fit(df)
    Xv = X.values
    XtX_inv = np.linalg.inv(Xv.T @ Xv)
    h = np.einsum("ij,jk,ik->i", Xv, XtX_inv, Xv)
    d2 = d.copy()
    d2["h"] = h
    g = d2.groupby("country")["h"].sum().sort_values(ascending=False)
    return {
        "top10_leverage": [{"country": c, "sum_h": float(v)}
                          for c, v in g.head(10).items()],
        "rule_of_thumb_avg_x_3": float(3 * X.shape[1] / int(m.nobs)),
    }


def studentized_residuals(df) -> dict:
    d, X, m = baseline_fit(df)
    Xv = X.values
    XtX_inv = np.linalg.inv(Xv.T @ Xv)
    h = np.einsum("ij,jk,ik->i", Xv, XtX_inv, Xv)
    sigma2 = float((m.resid ** 2).sum() / max(int(m.nobs) - X.shape[1], 1))
    r = m.resid.values / np.sqrt(sigma2 * (1 - h))
    d2 = d.copy()
    d2["studres"] = r
    out = d2.assign(abs_r=lambda x: np.abs(x["studres"])).sort_values("abs_r",
                                                                      ascending=False)
    return {"top10_outlying_obs": [{"country": row["country"],
                                    "year": int(row["year"]),
                                    "studres": float(row["studres"])}
                                   for _, row in out.head(10).iterrows()],
            "share_abs_above_3": float((np.abs(r) > 3).mean()),
            "share_abs_above_4": float((np.abs(r) > 4).mean())}


def main():
    banner("ECONOMETRIC BATTERY 09 — Influence diagnostics")
    df = panel(include_outcomes=False, include_gdp=False)
    print(f"Panel: {len(df)} obs · {df['country'].nunique()} countries")

    section("Cook's D per country (LOO)")
    cd = cooks_d_per_country(df)
    print(f"  Rule of thumb threshold 4/N:  {cd['rule_of_thumb_4_over_n']:.4f}")
    print(f"  Max D:                         {cd['max_D']:.4f}")
    for r in cd["top10_cooks_d"][:5]:
        print(f"    {r['country']:<30}  D={r['D']:.4f}  β_-c={r['beta_without']:+.4f}")

    section("DFBETAs per country")
    dfb = dfbetas_per_country(df)
    print(f"  Rule of thumb 2/√N:  {dfb['rule_of_thumb_2_over_sqrtN']:.4f}")
    for r in dfb["top10_abs_dfbeta"][:5]:
        print(f"    {r['country']:<30}  DFBETA={r['dfbeta']:+.4f}  β_-c={r['beta_without']:+.4f}")

    section("Leave-one-country-out β distribution")
    loo = leave_one_country_out(df)
    print(f"  Full β:        {loo['full_beta']:+.4f}")
    print(f"  LOO β range:   [{loo['min_beta']:.4f}, {loo['max_beta']:.4f}]")
    print(f"  LOO median:    {loo['median_beta']:+.4f}")
    print(f"  Spread:        {loo['spread']:.4f}")
    print(f"  5 countries whose REMOVAL shrinks β most:")
    for r in loo["5_smallest_when_removed"]:
        print(f"    without {r['country']:<25}  β = {r['beta']:+.4f}")
    print(f"  5 countries whose REMOVAL grows β most:")
    for r in loo["5_largest_when_removed"]:
        print(f"    without {r['country']:<25}  β = {r['beta']:+.4f}")

    section("Group leverage")
    lev = leverage_per_country(df)
    print(f"  3·k/N threshold:  {lev['rule_of_thumb_avg_x_3']:.4f}")
    for r in lev["top10_leverage"][:5]:
        print(f"    {r['country']:<30}  Σh={r['sum_h']:.4f}")

    section("Studentized residuals")
    sr = studentized_residuals(df)
    print(f"  Share |r*| > 3:   {sr['share_abs_above_3']:.4f}")
    print(f"  Share |r*| > 4:   {sr['share_abs_above_4']:.4f}")
    print("  Worst observations:")
    for r in sr["top10_outlying_obs"][:5]:
        print(f"    {r['country']:<25}  yr={r['year']}  r*={r['studres']:+.3f}")

    section("Verdict")
    no_pivot = loo["min_beta"] > 0.40 and loo["max_beta"] < 0.55
    print(f"  All LOO β > 0.40:  {loo['min_beta'] > 0.40}")
    print(f"  All LOO β < 0.55:  {loo['max_beta'] < 0.55}")
    print(f"  No single country pivots the result:  {no_pivot}")

    write_battery_checkin("09_influence", {
        "panel": {"n_obs": int(len(df)),
                  "n_countries": int(df["country"].nunique())},
        "cooks_d": cd, "dfbetas": dfb, "loo": loo,
        "leverage": lev, "studentized_residuals": sr,
    }, script_path=__file__)

    print()
    print(pass_fail(no_pivot, "no single country pivots the headline β"))


if __name__ == "__main__":
    main()
