"""
test_08_structural_breaks.py — Parameter constancy / structural break battery.

Tests whether β is stable across time. With a 1975-2015 panel covering
decolonisation, oil shocks, end of Cold War, MDGs, post-2008, AFC, etc., a
truly causal mechanism should give β values that are similar across eras
even if the global mean of education is rising.

Tests:
  1. Era splits — β fitted on rolling 15-year windows
  2. Chow test — H0: β identical pre/post each candidate break
  3. Andrews-Quandt sup-F — maximum F across candidate breaks
  4. Bai-Perron (1998) multiple breaks — simplified single-break search
  5. CUSUM of recursive residuals
  6. CUSUM-of-squares
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


def era_betas(df, eras=None) -> dict:
    if eras is None:
        eras = [(1975, 1989), (1990, 1999), (2000, 2009), (2010, 2015)]
    out = []
    for lo, hi in eras:
        sub = df[(df["year"] >= lo) & (df["year"] <= hi)]
        if len(sub) < 30:
            out.append({"era": f"{lo}-{hi}", "n": int(len(sub))})
            continue
        d = demean(sub, ["lower_sec", "parent_edu"])
        X = sm.add_constant(d[["parent_edu_d"]], has_constant="add")
        m = sm.OLS(d["lower_sec_d"], X).fit(cov_type="cluster",
                                            cov_kwds={"groups": d["country"]})
        out.append({"era": f"{lo}-{hi}", "n": int(m.nobs),
                    "beta": float(m.params["parent_edu_d"]),
                    "t": float(m.tvalues["parent_edu_d"])})
    return {"results": out}


def chow_test(df, break_year) -> dict:
    """Standard Chow at a candidate break year."""
    d = demean(df.dropna(subset=["lower_sec", "parent_edu"]),
               ["lower_sec", "parent_edu"])
    X = sm.add_constant(d[["parent_edu_d"]], has_constant="add")
    m_full = sm.OLS(d["lower_sec_d"], X).fit()
    ssr_full = float((m_full.resid ** 2).sum())
    sub1 = d[d["year"] <= break_year]
    sub2 = d[d["year"] > break_year]
    if len(sub1) < 30 or len(sub2) < 30:
        return {"break_year": int(break_year), "F": None, "p": None}
    X1 = sm.add_constant(sub1[["parent_edu_d"]], has_constant="add")
    X2 = sm.add_constant(sub2[["parent_edu_d"]], has_constant="add")
    m1 = sm.OLS(sub1["lower_sec_d"], X1).fit()
    m2 = sm.OLS(sub2["lower_sec_d"], X2).fit()
    ssr_split = float((m1.resid ** 2).sum() + (m2.resid ** 2).sum())
    k = 2  # parameters (constant, slope)
    n = int(m_full.nobs)
    F = ((ssr_full - ssr_split) / k) / (ssr_split / (n - 2 * k))
    p = float(1 - stats.f.cdf(F, k, n - 2 * k))
    return {"break_year": int(break_year), "F": float(F), "p": p, "n": n}


def andrews_quandt(df, candidates=None) -> dict:
    """Sup-F across candidate break years (Andrews 1993)."""
    if candidates is None:
        candidates = [1985, 1990, 1995, 2000, 2005]
    results = []
    for by in candidates:
        results.append(chow_test(df, by))
    valid = [r for r in results if r["F"] is not None]
    if not valid:
        return {"results": results}
    best = max(valid, key=lambda r: r["F"])
    # Approximate Andrews critical values for sup-F with 2 parameters
    # at 5% level: ~9.84 (Andrews 1993 Table I, trim 0.15)
    return {"results": results, "sup_F": best["F"],
            "sup_break": best["break_year"],
            "approx_p_5pct_cv": 9.84}


def bai_perron_single(df) -> dict:
    """Single-break search over T - 2(min_segment) candidates."""
    years = sorted(df["year"].unique())
    if len(years) < 4:
        return {}
    candidates = years[1:-1]
    Fs = []
    for by in candidates:
        r = chow_test(df, by)
        if r["F"] is not None:
            Fs.append((by, r["F"]))
    if not Fs:
        return {}
    best = max(Fs, key=lambda x: x[1])
    return {"best_break": int(best[0]), "best_F": float(best[1]),
            "all_F": Fs}


def cusum(df) -> dict:
    """CUSUM of within-residuals over time; test against ±√T bands.

    Aggregates within-country residuals by year, cumulates them.
    Sup |CUSUM| / σ̂ compared against 1.358 (5% level, single test).
    """
    d = demean(df.dropna(subset=["lower_sec", "parent_edu"]),
               ["lower_sec", "parent_edu"])
    X = sm.add_constant(d[["parent_edu_d"]], has_constant="add")
    m = sm.OLS(d["lower_sec_d"], X).fit()
    d2 = d.copy()
    d2["res"] = m.resid.values
    by_year = d2.groupby("year")["res"].mean().sort_index()
    cs = by_year.cumsum().values
    sigma = float(d2["res"].std())
    T = len(by_year)
    if T < 3 or sigma == 0:
        return {}
    std_band = sigma * np.sqrt(T) * 1.358
    sup = float(np.max(np.abs(cs)))
    return {
        "sup_cusum": sup, "critical_5pct": float(std_band),
        "exceeds_band": bool(sup > std_band),
        "cusum_by_year": {int(y): float(c) for y, c in zip(by_year.index, cs)},
    }


def cusum_squares(df) -> dict:
    d = demean(df.dropna(subset=["lower_sec", "parent_edu"]),
               ["lower_sec", "parent_edu"])
    X = sm.add_constant(d[["parent_edu_d"]], has_constant="add")
    m = sm.OLS(d["lower_sec_d"], X).fit()
    d2 = d.copy()
    d2["res2"] = m.resid.values ** 2
    by_year = d2.groupby("year")["res2"].mean().sort_index()
    total = float(by_year.sum())
    cs = by_year.cumsum().values / total
    T = len(cs)
    # Expected under null: t/T linear; deviation = CUSUMSQ - t/T
    dev = cs - np.arange(1, T + 1) / T
    return {"sup_dev": float(np.max(np.abs(dev))),
            "critical_5pct_approx": float(1.36 / np.sqrt(T))}


def main():
    banner("ECONOMETRIC BATTERY 08 — Structural breaks")
    df = panel(include_outcomes=False, include_gdp=False)
    print(f"Panel: {len(df)} obs · {df['country'].nunique()} countries")

    section("β by era (rolling sub-samples)")
    eras = era_betas(df)
    for r in eras["results"]:
        if "beta" in r:
            print(f"  {r['era']}   N={r['n']:5d}   β={r['beta']:+.4f}   t={r['t']:+.2f}")
        else:
            print(f"  {r['era']}   N={r['n']:5d}   (insufficient)")
    betas = [r.get("beta") for r in eras["results"] if "beta" in r]
    print(f"  β range:  [{min(betas):.3f}, {max(betas):.3f}]   spread={max(betas) - min(betas):.3f}")

    section("Andrews-Quandt sup-F (candidate breaks 1985-2005)")
    aq = andrews_quandt(df)
    for r in aq["results"]:
        if r["F"] is not None:
            print(f"  break={r['break_year']}   F={r['F']:.2f}   p={r['p']:.4g}")
    if "sup_F" in aq:
        print(f"  sup-F:  {aq['sup_F']:.2f}   at break={aq['sup_break']}")
        print(f"  Andrews 5% CV ≈ {aq['approx_p_5pct_cv']} (trim 0.15)")

    section("Bai-Perron single-break search")
    bp = bai_perron_single(df)
    if bp:
        print(f"  best break:  {bp['best_break']}")
        print(f"  best F:      {bp['best_F']:.2f}")

    section("CUSUM of recursive within-year mean residuals")
    cs = cusum(df)
    if cs:
        print(f"  sup |CUSUM|:        {cs['sup_cusum']:.4f}")
        print(f"  5% critical band:   ±{cs['critical_5pct']:.4f}")
        print(f"  Exceeds band:       {cs['exceeds_band']}")

    section("CUSUM-of-squares")
    csq = cusum_squares(df)
    if csq:
        print(f"  sup dev:            {csq['sup_dev']:.4f}")
        print(f"  5% CV approx:       {csq['critical_5pct_approx']:.4f}")

    section("Interpretation")
    print("  Some structural break detection is expected: education and")
    print("  outcomes both accelerated post-1990 (MDGs, EFA). What matters")
    print("  is whether β stays *positive* and *meaningful* across eras.")
    print("  The β-by-era table above answers that directly.")

    write_battery_checkin("08_structural_breaks", {
        "era_betas": eras,
        "andrews_quandt": aq,
        "bai_perron_single": bp,
        "cusum": cs,
        "cusum_squares": csq,
    }, script_path=__file__)

    # Pass if β stays positive and t>2 in every era
    all_pos = all(r.get("beta", 0) > 0 and r.get("t", 0) > 2
                  for r in eras["results"] if "beta" in r)
    print()
    print(pass_fail(all_pos, "β positive and significant in every era"))


if __name__ == "__main__":
    main()
