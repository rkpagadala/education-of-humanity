"""
robustness/twfe_all_outcomes.py
==========================
Two-way fixed effects (country + time) residualization for all four outcomes:
  - Life expectancy (T+25)
  - TFR (T+25)
  - Child education (T+25)
  - Child mortality U5 (T+25)

Adding time FE rules out global trends (medical technology, green revolution)
as confounders. If the zero GDP result survives two-way FE, it's bulletproof.
"""

import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from residualization._shared import (
    build_panel, build_child_edu_panel, precompute_entry_years,
    run_residualized_sweep, print_summary,
    fe_twoway_r2, fe_residualize_gdp_twoway,
)
from _shared import load_education, load_wb, interpolate_to_annual, write_checkin, fmt_r2

EDU_LEVELS = {"primary": "primary", "lower_secondary": "lower_sec", "upper_secondary": "upper_sec"}
T_YEARS = list(range(1960, 1995, 5))
LAG = 25
CEILINGS = [50, 60, 70, 80, 90]

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
le_df = load_wb("life_expectancy_years.csv")
tfr_df = load_wb("children_per_woman_total_fertility.csv")
u5mr_df = load_wb("child_mortality_u5.csv")

OUTCOMES = {
    "LE": ("le_tp25", le_df),
    "TFR": ("tfr_tp25", tfr_df),
    "U5MR": ("u5mr_tp25", u5mr_df),
}

# Pre-compute interpolation and entry years (same for all outcomes)
edu_annual_cache = {}
entry_years_cache = {}
for level_name, col_name in EDU_LEVELS.items():
    edu_annual_cache[level_name] = interpolate_to_annual(edu_raw, col_name)
    entry_years_cache[level_name] = precompute_entry_years(edu_annual_cache[level_name])

all_results = {}

for outcome_label, (outcome_col, outcome_df) in OUTCOMES.items():
    print(f"\n\n{'*' * 90}")
    print(f"* OUTCOME: {outcome_label}  (two-way FE: country + time)")
    print(f"{'*' * 90}")

    outcome_results = {}

    for level_name, col_name in EDU_LEVELS.items():
        print(f"\n{'#' * 90}")
        print(f"# {level_name.upper().replace('_', ' ')}  →  {outcome_label}(T+25)")
        print(f"{'#' * 90}")

        edu_annual = edu_annual_cache[level_name]
        panel = build_panel(edu_annual, outcome_df, gdp_df, T_YEARS, LAG, outcome_col)
        print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries")

        entry_years = entry_years_cache[level_name]
        results = run_residualized_sweep(
            panel, entry_years, outcome_col, CEILINGS,
            fe_func=fe_twoway_r2, resid_func=fe_residualize_gdp_twoway,
            label=outcome_label, print_every=10
        )
        print_summary(results, CEILINGS, label=outcome_label)
        outcome_results[level_name] = results

    # Cross-level for this outcome
    for ceiling in [60, 90]:
        print(f"\n{'=' * 90}")
        print(f"CROSS-LEVEL (two-way FE): entry=10%, ceiling={ceiling}%  →  {outcome_label}")
        print(f"{'=' * 90}")
        print(f"{'Level':<20} {'Edu':>7} {'GDP':>7} {'Resid':>9} {'Edu→GDP':>8}")
        print("-" * 55)
        for level_name in EDU_LEVELS:
            r = outcome_results[level_name].get(str(ceiling), {}).get("10", {})
            if r and r.get("edu_r2") is not None:
                print(f"{level_name:<20} {fmt_r2(r.get('edu_r2')):>7} {fmt_r2(r.get('raw_gdp_r2')):>7} "
                      f"{fmt_r2(r.get('resid_gdp_r2')):>9} {fmt_r2(r.get('edu_gdp_r2')):>8}")

    all_results[outcome_label] = outcome_results

# Also add child education (parent→child, not WB data)
print(f"\n\n{'*' * 90}")
print(f"* OUTCOME: CHILD EDUCATION  (two-way FE: country + time)")
print(f"{'*' * 90}")

child_edu_results = {}
for level_name, col_name in EDU_LEVELS.items():
    print(f"\n# {level_name.upper().replace('_', ' ')}  →  Child Edu(T+25)")
    edu_annual = edu_annual_cache[level_name]
    panel = build_child_edu_panel(edu_annual, gdp_df, T_YEARS, LAG)
    print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries")
    entry_years = entry_years_cache[level_name]
    results = run_residualized_sweep(
        panel, entry_years, "child_edu", CEILINGS,
        fe_func=fe_twoway_r2, resid_func=fe_residualize_gdp_twoway,
        label="ChildEdu", print_every=10
    )
    print_summary(results, CEILINGS, label="ChildEdu")
    child_edu_results[level_name] = results

all_results["ChildEdu"] = child_edu_results

# Final cross-outcome summary
print(f"\n\n{'=' * 90}")
print(f"GRAND SUMMARY: Two-way FE, entry=10%, ceiling=90%, lower_secondary")
print(f"{'=' * 90}")
print(f"{'Outcome':<15} {'Edu':>7} {'GDP':>7} {'Resid':>9} {'Edu→GDP':>8}")
print("-" * 50)
for outcome_label in ["LE", "TFR", "U5MR", "ChildEdu"]:
    r = all_results[outcome_label].get("lower_secondary", {}).get("90", {}).get("10", {})
    if r and r.get("edu_r2") is not None:
        print(f"{outcome_label:<15} {fmt_r2(r.get('edu_r2')):>7} {fmt_r2(r.get('raw_gdp_r2')):>7} "
              f"{fmt_r2(r.get('resid_gdp_r2')):>9} {fmt_r2(r.get('edu_gdp_r2')):>8}")

write_checkin("twfe_all_outcomes.json", {
    "method": "Two-way FE (country + time). Residualized GDP. Entry-cohort + ceiling. T=1960-1990, lag=25. Four outcomes × three education levels.",
    "results": {k: {lev: v for lev, v in out.items()} for k, out in all_results.items()},
}, script_path="scripts/robustness/twfe_all_outcomes.py")
