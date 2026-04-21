"""
residualization/child_education.py
=====================================
Does GDP independently predict intergenerational education transmission?

Parent education(T) → Child education(T+25): the PTE mechanism.
GDP(T) → Child education(T+25): the income-led hypothesis.
Residualized GDP(T) → Child education(T+25): GDP's independent effect.

If residualized GDP ≈ 0, education transmits itself without income.
Completes the triangle: LE (done), TFR (done), child education.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _shared import (
    load_education, load_wb, interpolate_to_annual,
    build_child_edu_panel, precompute_entry_years, run_residualized_sweep,
    print_summary, write_checkin, fmt_r2,
)

EDU_LEVELS = {"primary": "primary", "lower_secondary": "lower_sec", "upper_secondary": "upper_sec"}
T_YEARS = list(range(1960, 1995, 5))
LAG = 25
CEILINGS = [50, 60, 70, 80, 90]

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")

all_results = {}

for level_name, col_name in EDU_LEVELS.items():
    print(f"\n{'#' * 90}")
    print(f"# {level_name.upper().replace('_', ' ')}  →  CHILD EDUCATION(T+25)")
    print(f"{'#' * 90}")

    edu_annual = interpolate_to_annual(edu_raw, col_name)

    panel = build_child_edu_panel(edu_annual, gdp_df, T_YEARS, LAG)
    print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries")

    entry_years = precompute_entry_years(edu_annual)
    results = run_residualized_sweep(panel, entry_years, "child_edu", CEILINGS, label="ChildEdu")
    print_summary(results, CEILINGS, label="ChildEdu")
    all_results[level_name] = results

# Cross-level
for ceiling in [60, 90]:
    print(f"\n{'=' * 90}")
    print(f"CROSS-LEVEL: entry=10%, ceiling={ceiling}%  →  Child Education(T+25)")
    print(f"{'=' * 90}")
    print(f"{'Level':<20} {'Edu→CE':>7} {'GDP→CE':>7} {'Resid→CE':>9} {'Edu→GDP':>8}")
    print("-" * 55)
    for level_name in EDU_LEVELS:
        r = all_results[level_name].get(str(ceiling), {}).get("10", {})
        if r and r.get("edu_r2") is not None:
            print(f"{level_name:<20} {fmt_r2(r.get('edu_r2')):>7} {fmt_r2(r.get('raw_gdp_r2')):>7} "
                  f"{fmt_r2(r.get('resid_gdp_r2')):>9} {fmt_r2(r.get('edu_gdp_r2')):>8}")

write_checkin("edu_vs_gdp_child_edu_residualized.json", {
    "method": "Country FE, residualized GDP. Parent edu(T) → child edu(T+25). Entry-cohort + ceiling. T=1960-1990, lag=25.",
    "levels": all_results,
}, script_path="scripts/residualization/child_education.py")
