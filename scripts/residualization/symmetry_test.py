"""
residualization/symmetry_test.py
================================
Empirical test of the "Frisch-Waugh asymmetry" objection to the paper's
residualization strategy.

OBJECTION
---------
The objection: "Strip education's contribution from income, then test
residual income" advantages the protected variable (education) over the
residualized variable (GDP). Reverse the order — strip income's
contribution from education, then test residual education — and
education would explain far less.

TEST
----
For each (outcome, education-level) cell, compute four country-FE R²:

  1. Edu(T)        →  Y(T+25)        raw education
  2. log_GDP(T)    →  Y(T+25)        raw GDP
  3. GDP_resid(T)  →  Y(T+25)        GDP after stripping education's effect on GDP
  4. Edu_resid(T)  →  Y(T+25)        Edu after stripping GDP's effect on education

The objection predicts (4) collapses the same way (3) does. If (4) stays
near (1) while (3) stays near 0, the asymmetry is a fact about which
variable carries unique information — not a specification choice.

Method: country fixed effects (within-country variation only).
Entry-cohort design: each country enters at the year it crosses 10%
education and exits when it exceeds 60% (ceiling). T = 1960-1990 (5yr),
lag = 25 years.

Outcomes: life expectancy, TFR (under-five mortality + GDP omitted —
the objection doesn't change with the dependent variable, only the
direction of partial-ing).
"""

import os
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(1, os.path.join(THIS_DIR, ".."))

from _shared import (  # noqa: E402
    load_education, load_wb, interpolate_to_annual,
    precompute_entry_years, build_panel, fmt_r2,
    fe_r2, _demean_and_filter, write_checkin,
)


def fe_residualize(data, x_col, controls):
    """Residualize x_col against controls (with country FE).

    Returns (sub_with_residual, control_x_r2) or None.
    """
    cols = [x_col] + list(controls)
    result = _demean_and_filter(data, cols)
    if result is None:
        return None
    sub, dm, _ = result
    y = dm[x_col].to_numpy()
    X = np.column_stack([dm[c].to_numpy() for c in controls])
    ok = ~np.isnan(y)
    for j in range(X.shape[1]):
        ok &= ~np.isnan(X[:, j])
    if ok.sum() < 10:
        return None
    Xo, yo = X[ok], y[ok]
    XtX = Xo.T @ Xo
    Xty = Xo.T @ yo
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return None
    yhat = X @ beta
    syy = float(np.dot(yo, yo))
    ssr = float(np.dot(yo - yhat[ok], yo - yhat[ok]))
    r2 = 1.0 - ssr / syy if syy > 0.0 else np.nan
    sub[f"{x_col}_resid"] = y - yhat
    return sub, r2


def run_cell(panel, entry_years, outcome_col, threshold=10, ceiling=60):
    cohort = entry_years.get(threshold, {})
    if len(cohort) < 3:
        return None
    entry_series = panel["country"].map(cohort)
    mask = (
        entry_series.notna()
        & (panel["t"] >= entry_series)
        & (panel["edu_t"] <= ceiling)
    )
    sub = panel[mask].copy()
    if len(sub) < 10:
        return None

    r2_e, n, ctry = fe_r2("edu_t", outcome_col, sub)
    r2_g, _, _ = fe_r2("log_gdp_t", outcome_col, sub)

    # Direction 1: residualize GDP on education (current paper procedure)
    res_gdp = fe_residualize(sub, "log_gdp_t", ["edu_t"])
    if res_gdp is None:
        return None
    sub_g, edu_explains_gdp_r2 = res_gdp
    r2_gdp_resid, _, _ = fe_r2("log_gdp_t_resid", outcome_col, sub_g)

    # Direction 2: residualize education on GDP (the symmetric test)
    res_edu = fe_residualize(sub, "edu_t", ["log_gdp_t"])
    if res_edu is None:
        return None
    sub_e, gdp_explains_edu_r2 = res_edu
    r2_edu_resid, _, _ = fe_r2("edu_t_resid", outcome_col, sub_e)

    return {
        "edu_r2": float(r2_e),
        "gdp_r2": float(r2_g),
        "gdp_resid_r2": float(r2_gdp_resid),
        "edu_resid_r2": float(r2_edu_resid),
        "edu_explains_gdp_r2": float(edu_explains_gdp_r2),
        "gdp_explains_edu_r2": float(gdp_explains_edu_r2),
        "edu_resid_retention": (
            float(r2_edu_resid / r2_e) if r2_e > 0 else None
        ),
        "gdp_resid_retention": (
            float(r2_gdp_resid / r2_g) if r2_g > 0 else None
        ),
        "n": int(n),
        "countries": int(ctry),
    }


EDU_LEVELS = {
    "primary": "primary",
    "lower_secondary": "lower_sec",
    "upper_secondary": "upper_sec",
}

OUTCOMES = {
    "life_expectancy": ("life_expectancy_years.csv", "le_tp25"),
    "tfr": ("children_per_woman_total_fertility.csv", "tfr_tp25"),
}

T_YEARS = list(range(1960, 1995, 5))
LAG = 25
THRESHOLD = 10
CEILING = 60

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")

results = {}

for outcome_name, (csv, ycol) in OUTCOMES.items():
    print(f"\n{'#' * 90}")
    print(f"# OUTCOME: {outcome_name.upper()}  (Y = {ycol})")
    print(f"{'#' * 90}")
    out_raw = load_wb(csv)

    results[outcome_name] = {}
    print(
        f"\n{'level':<18} {'edu→Y':>8} {'GDP→Y':>8} "
        f"{'GDP_r→Y':>9} {'edu_r→Y':>9} "
        f"{'edu→GDP':>9} {'GDP→edu':>9} "
        f"{'edu_keep':>9} {'GDP_keep':>9} {'n':>5} {'ctry':>5}"
    )
    print("-" * 110)

    for level_name, col_name in EDU_LEVELS.items():
        edu_annual = interpolate_to_annual(edu_raw, col_name)
        panel = build_panel(edu_annual, out_raw, gdp_raw, T_YEARS, LAG, ycol)
        entry_years = precompute_entry_years(edu_annual)
        r = run_cell(panel, entry_years, ycol, THRESHOLD, CEILING)
        if r is None:
            print(f"{level_name:<18}  (insufficient data)")
            continue
        results[outcome_name][level_name] = r
        print(
            f"{level_name:<18} "
            f"{fmt_r2(r['edu_r2']):>8} {fmt_r2(r['gdp_r2']):>8} "
            f"{fmt_r2(r['gdp_resid_r2']):>9} {fmt_r2(r['edu_resid_r2']):>9} "
            f"{fmt_r2(r['edu_explains_gdp_r2']):>9} "
            f"{fmt_r2(r['gdp_explains_edu_r2']):>9} "
            f"{(r['edu_resid_retention'] or 0):>9.2f} "
            f"{(r['gdp_resid_retention'] or 0):>9.2f} "
            f"{r['n']:>5} {r['countries']:>5}"
        )

print("\n\n" + "=" * 90)
print("VERDICT")
print("=" * 90)
print(
    "edu_keep = edu_resid_R² / edu_R² (share of education's predictive power\n"
    "           that survives partialling out GDP)"
)
print(
    "GDP_keep = GDP_resid_R² / GDP_R² (share of GDP's predictive power that\n"
    "           survives partialling out education)"
)
print()
print(
    "Critic's claim: reversing the residualization direction would make\n"
    "education collapse the way GDP does. If true, edu_keep ≈ GDP_keep ≈ 0.\n"
)

for outcome_name, lev in results.items():
    print(f"\n{outcome_name.upper()}:")
    for level_name, r in lev.items():
        print(
            f"  {level_name:<18}  edu_keep={r['edu_resid_retention']:.2f}   "
            f"GDP_keep={(r['gdp_resid_retention'] or 0):.2f}"
        )

write_checkin(
    "residualization_symmetry_test.json",
    {
        "method": (
            "Symmetric Frisch-Waugh test. For each (outcome, edu-level) "
            "cell with country FE and entry-cohort 10/60: report edu→Y, "
            "GDP→Y, GDP_resid→Y (current paper), and edu_resid→Y (symmetric). "
            "edu_keep and GDP_keep are retention ratios. T=1960-1990 (5yr), "
            "lag=25."
        ),
        "threshold": THRESHOLD,
        "ceiling": CEILING,
        "results": results,
    },
    "scripts/residualization/symmetry_test.py",
)
