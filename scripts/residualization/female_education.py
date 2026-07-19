"""
residualization/female_education.py
=================================
Reruns the residualised analysis for LE, TFR, U5MR, child education
using female-only completion rates from WCDE. Compares to both-sexes.

Script output is NOT cited in the paper. The country panel cannot
decompose by sex: within-country male and female lower-secondary
completion are correlated at >0.9 (the home niche moves as a unit),
so the country panel cannot statistically distinguish the
female-specific channel. The mother-weighting claim in the paper is
sourced from the cross-sectional microdata literature (Caldwell 1979;
Schultz 2002; Jensen 2010; Duflo 2012), which identifies the channel
at the within-country individual level. See paper §9.8 lines
3204–3212.

This script is retained for transparency — it documents what a panel
horse race between female-only and both-sexes completion produces.
It is not a valid sex-decomposition because the regressors are
near-collinear within country. Per-outcome R² comparisons in the
JSON output should not be read as evidence that female education is
"stronger" than the aggregate.
"""

import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _shared import (
    load_education, load_wb, interpolate_to_annual,
    precompute_entry_years, build_panel, build_child_edu_panel,
    filter_panel, fe_r2, fe_residualize_gdp, write_checkin, fmt_r2,
    LAG_TFR, LAG_GENERATION, LAG_LE, LAG_CHILDREARING,
)

T_YEARS = list(range(1960, 1995, 5))
CEILINGS = [60, 90]
COL_NAME = "lower_sec"

# Outcome → lag. TFR uses LAG_TFR=5 (biological parent-of-newborn timing).
# LE and U5MR use the childrearing window (LE is mortality-dominated); child
# education (EDU) keeps LAG_GENERATION=28 (edu→edu transmission).
OUTCOME_LAG = {
    "LE":       LAG_LE,
    "TFR":      LAG_TFR,
    "U5MR":     LAG_CHILDREARING,
    "ChildEdu": LAG_GENERATION,
}

print("Loading data...")
edu_both = load_education("completion_both_long.csv")
edu_female = load_education("completion_female_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
le_df = load_wb("life_expectancy_years.csv")
tfr_df = load_wb("children_per_woman_total_fertility.csv")
u5mr_df = load_wb("child_mortality_u5.csv")

all_results = {}

for sex_label, edu_raw in [("both", edu_both), ("female", edu_female)]:
    print(f"\n\n{'*' * 90}")
    print(f"* SEX: {sex_label.upper()}")
    print(f"{'*' * 90}")

    edu_annual = interpolate_to_annual(edu_raw, COL_NAME)
    entry_years = precompute_entry_years(edu_annual)

    sex_results = {}

    # WB outcomes
    for outcome_label, outcome_col, outcome_df in [
        ("LE", "le", le_df),
        ("TFR", "tfr", tfr_df),
        ("U5MR", "u5mr", u5mr_df),
    ]:
        panel = build_panel(edu_annual, outcome_df, gdp_df, T_YEARS,
                            OUTCOME_LAG[outcome_label], outcome_col)
        print(f"\n  {outcome_label} (n={len(panel)}, countries={panel['country'].nunique()})")

        outcome_results = {}
        for ceiling in CEILINGS:
            cohort = entry_years.get(10, {})
            sub = filter_panel(panel, cohort, ceiling)
            if len(sub) < 10:
                continue
            r2_e, n_e, c_e = fe_r2("edu_t", outcome_col, sub)
            r2_g, _, _ = fe_r2("log_gdp_t", outcome_col, sub)
            resid = fe_residualize_gdp(sub)
            r2_resid = np.nan
            if resid is not None:
                sub_r, _ = resid
                r2_resid, _, _ = fe_r2("gdp_resid", outcome_col, sub_r)

            print(f"    ceil={ceiling}%: Edu={fmt_r2(r2_e)}  GDP={fmt_r2(r2_g)}  Resid={fmt_r2(r2_resid)}  n={n_e} ctry={c_e}")
            outcome_results[str(ceiling)] = {
                "edu_r2": round(r2_e, 3) if not np.isnan(r2_e) else None,
                "raw_gdp_r2": round(r2_g, 3) if not np.isnan(r2_g) else None,
                "resid_gdp_r2": round(r2_resid, 3) if not np.isnan(r2_resid) else None,
                "n": n_e, "countries": c_e,
            }
        sex_results[outcome_label] = outcome_results

    # Child education (parent→child)
    panel_ce = build_child_edu_panel(edu_annual, gdp_df, T_YEARS,
                                     OUTCOME_LAG["ChildEdu"])
    print(f"\n  ChildEdu (n={len(panel_ce)}, countries={panel_ce['country'].nunique()})")
    ce_results = {}
    for ceiling in CEILINGS:
        cohort = entry_years.get(10, {})
        sub = filter_panel(panel_ce, cohort, ceiling)
        if len(sub) < 10:
            continue
        r2_e, n_e, c_e = fe_r2("edu_t", "child_edu", sub)
        r2_g, _, _ = fe_r2("log_gdp_t", "child_edu", sub)
        resid = fe_residualize_gdp(sub)
        r2_resid = np.nan
        if resid is not None:
            sub_r, _ = resid
            r2_resid, _, _ = fe_r2("gdp_resid", "child_edu", sub_r)
        print(f"    ceil={ceiling}%: Edu={fmt_r2(r2_e)}  GDP={fmt_r2(r2_g)}  Resid={fmt_r2(r2_resid)}  n={n_e} ctry={c_e}")
        ce_results[str(ceiling)] = {
            "edu_r2": round(r2_e, 3) if not np.isnan(r2_e) else None,
            "raw_gdp_r2": round(r2_g, 3) if not np.isnan(r2_g) else None,
            "resid_gdp_r2": round(r2_resid, 3) if not np.isnan(r2_resid) else None,
            "n": n_e, "countries": c_e,
        }
    sex_results["ChildEdu"] = ce_results
    all_results[sex_label] = sex_results

# Comparison
print(f"\n\n{'=' * 90}")
print(f"FEMALE vs BOTH-SEXES: lower secondary, entry=10%, ceiling=90%")
print(f"{'=' * 90}")
print(f"{'Outcome':<12} {'Both Edu':>9} {'Female Edu':>11} {'Both Resid':>11} {'Female Resid':>13}")
print("-" * 60)
for outcome in ["LE", "TFR", "U5MR", "ChildEdu"]:
    rb = all_results["both"].get(outcome, {}).get("90", {})
    rf = all_results["female"].get(outcome, {}).get("90", {})
    print(f"{outcome:<12} {fmt_r2(rb.get('edu_r2')):>9} {fmt_r2(rf.get('edu_r2')):>11} "
          f"{fmt_r2(rb.get('resid_gdp_r2')):>11} {fmt_r2(rf.get('resid_gdp_r2')):>13}")

write_checkin("female_education_residualized.json", {
    "method": (
        "Country FE, residualized GDP. Female vs both-sexes lower secondary. "
        "Entry=10%, ceilings 60/90. T=1960-1990. Outcome-specific lags: TFR at "
        f"T+LAG_TFR={LAG_TFR} (biological parent-of-newborn timing); LE, U5MR, "
        f"and child education at T+LAG_GENERATION={LAG_GENERATION} (time-to-"
        "agency / edu→edu transmission)."
    ),
    "outcome_lags": OUTCOME_LAG,
    "results": all_results,
}, script_path="scripts/residualization/female_education.py")
