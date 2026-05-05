"""
robustness/lag_sensitivity.py
===================
Does the zero GDP result hold at different lags?

Runs the residualized analysis at lags 15, 20, 25, 30 for all four outcomes
(LE, TFR, U5MR, child education). Lower secondary, entry=10%.

If the result holds across lags, it's not an artifact of the 25-year choice.
"""

import os, sys
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _shared import load_education, load_wb, interpolate_to_annual, write_checkin, fmt_r2
from residualization._shared import (
    build_panel, build_child_edu_panel, precompute_entry_years,
    filter_panel, compare_predictors,
)

T_YEARS_BASE = list(range(1960, 2000, 5))
LAGS = [15, 20, 25, 30]
CEILINGS = [60, 90]
COL_NAME = "lower_sec"

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
le_df = load_wb("life_expectancy_years.csv")
tfr_df = load_wb("children_per_woman_total_fertility.csv")
u5mr_df = load_wb("child_mortality_u5.csv")

edu_annual = interpolate_to_annual(edu_raw, COL_NAME)
entry_years = precompute_entry_years(edu_annual)

all_results = {}

for lag in LAGS:
    print(f"\n\n{'*' * 90}")
    print(f"* LAG = {lag} years")
    print(f"{'*' * 90}")

    # Adjust T_YEARS so outcomes don't exceed data
    # LE/TFR go to 2022, U5MR to 2015
    max_t_le = min(2022 - lag, 1990)
    max_t_u5 = min(2015 - lag, 1990)
    t_years_le = [t for t in T_YEARS_BASE if t <= max_t_le]
    t_years_u5 = [t for t in T_YEARS_BASE if t <= max_t_u5]

    lag_results = {}

    # WB outcomes
    for outcome_label, outcome_col, outcome_df, t_years in [
        ("LE", "le", le_df, t_years_le),
        ("TFR", "tfr", tfr_df, t_years_le),
        ("U5MR", "u5mr", u5mr_df, t_years_u5),
    ]:
        panel = build_panel(edu_annual, outcome_df, gdp_df, t_years, lag, outcome_col)
        if len(panel) < 10:
            print(f"  {outcome_label}: insufficient data at lag={lag}")
            continue

        print(f"\n  {outcome_label} (n={len(panel)}, countries={panel['country'].nunique()})")
        for ceiling in CEILINGS:
            cohort = entry_years.get(10, {})
            sub = filter_panel(panel, cohort, ceiling)
            if len(sub) < 10:
                continue
            cp = compare_predictors(sub, outcome_col)
            print(f"    ceil={ceiling}%: Edu={fmt_r2(cp['edu_r2'])}  GDP={fmt_r2(cp['gdp_r2'])}  "
                  f"Resid={fmt_r2(cp['resid_gdp_r2'])}  n={cp['n']} ctry={cp['countries']}")

            lag_results[f"{outcome_label}_ceil{ceiling}"] = {
                "edu_r2": round(cp["edu_r2"], 3) if not np.isnan(cp["edu_r2"]) else None,
                "raw_gdp_r2": round(cp["gdp_r2"], 3) if not np.isnan(cp["gdp_r2"]) else None,
                "resid_gdp_r2": round(cp["resid_gdp_r2"], 3) if not np.isnan(cp["resid_gdp_r2"]) else None,
                "n": cp["n"], "countries": cp["countries"],
            }

    # Child education
    t_years_edu = [t for t in T_YEARS_BASE if t <= 1990]
    panel_ce = build_child_edu_panel(edu_annual, gdp_df, t_years_edu, lag)
    if len(panel_ce) >= 10:
        print(f"\n  ChildEdu (n={len(panel_ce)}, countries={panel_ce['country'].nunique()})")
        for ceiling in CEILINGS:
            cohort = entry_years.get(10, {})
            sub = filter_panel(panel_ce, cohort, ceiling)
            if len(sub) < 10:
                continue
            cp = compare_predictors(sub, "child_edu")
            print(f"    ceil={ceiling}%: Edu={fmt_r2(cp['edu_r2'])}  GDP={fmt_r2(cp['gdp_r2'])}  "
                  f"Resid={fmt_r2(cp['resid_gdp_r2'])}  n={cp['n']} ctry={cp['countries']}")

            lag_results[f"ChildEdu_ceil{ceiling}"] = {
                "edu_r2": round(cp["edu_r2"], 3) if not np.isnan(cp["edu_r2"]) else None,
                "raw_gdp_r2": round(cp["gdp_r2"], 3) if not np.isnan(cp["gdp_r2"]) else None,
                "resid_gdp_r2": round(cp["resid_gdp_r2"], 3) if not np.isnan(cp["resid_gdp_r2"]) else None,
                "n": cp["n"], "countries": cp["countries"],
            }

    all_results[str(lag)] = lag_results

# Summary
print(f"\n\n{'=' * 90}")
print(f"SUMMARY: lower secondary, entry=10%, ceiling=90%")
print(f"{'=' * 90}")
print(f"{'Lag':<6} {'LE Edu':>7} {'LE Res':>7} | {'TFR Edu':>8} {'TFR Res':>8} | {'U5 Edu':>7} {'U5 Res':>7} | {'CE Edu':>7} {'CE Res':>7}")
print("-" * 85)
def _get(results_dict, key, field):
    v = results_dict.get(key, {}).get(field)
    return f"{v:.3f}" if v is not None else "n/a"

for lag in LAGS:
    lr = all_results[str(lag)]
    print(f"  {lag}yr {_get(lr,'LE_ceil90','edu_r2'):>7} {_get(lr,'LE_ceil90','resid_gdp_r2'):>7} | "
          f"{_get(lr,'TFR_ceil90','edu_r2'):>8} {_get(lr,'TFR_ceil90','resid_gdp_r2'):>8} | "
          f"{_get(lr,'U5MR_ceil90','edu_r2'):>7} {_get(lr,'U5MR_ceil90','resid_gdp_r2'):>7} | "
          f"{_get(lr,'ChildEdu_ceil90','edu_r2'):>7} {_get(lr,'ChildEdu_ceil90','resid_gdp_r2'):>7}")

write_checkin("lag_sensitivity.json", {
    "method": "Country FE, residualized GDP. Lags 15/20/25/30. Lower secondary, entry=10%. Four outcomes.",
    "results": all_results,
}, script_path="scripts/robustness/lag_sensitivity.py")
