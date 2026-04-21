"""
residualization/education_vs_gdp.py
===========================
Does GDP predict life expectancy *independently* of education?

PROBLEM
-------
At higher education levels, GDP correlates with LE partly because
education drives GDP growth. Countries develop serially: education
first, then GDP rises, then health improves. Raw GDP R² at upper
secondary levels overstates GDP's independent contribution.

METHOD
------
Two-step residualization with country fixed effects:

Step 1: Regress log_GDP(T) on education(T) with country FE.
        Residuals = "GDP not explained by education"
        (orthogonal to education by construction)

Step 2: Regress LE(T+25) on these residuals with country FE.
        R² = GDP's independent predictive power for future LE,
        after removing education's contribution to GDP.

Compare:
  - Edu R²:          education → LE(T+25)
  - Raw GDP R²:      log_GDP → LE(T+25)
  - Residual GDP R²: GDP_residual → LE(T+25)

If residual GDP R² ≈ 0, GDP was just a proxy for education.

Runs for primary, lower secondary, upper secondary.
Entry-cohort design with ceilings.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _shared import (
    load_education, load_wb, interpolate_to_annual,
    precompute_entry_years, build_panel, run_residualized_sweep,
    print_summary, write_checkin, CHECKIN, fmt_r2,
)

# ── Load data ────────────────────────────────────────────────────────

print("Loading data...")

edu_raw = load_education("completion_both_long.csv")
le_raw = load_wb("life_expectancy_years.csv")
gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")

# ── Education levels ─────────────────────────────────────────────────

EDU_LEVELS = {
    "primary": "primary",
    "lower_secondary": "lower_sec",
    "upper_secondary": "upper_sec",
}

T_YEARS = list(range(1960, 1995, 5))
LAG = 25
CEILINGS = [50, 60, 70, 80, 90]

all_results = {}

for level_name, col_name in EDU_LEVELS.items():
    print(f"\n{'#' * 90}")
    print(f"# {level_name.upper().replace('_', ' ')}")
    print(f"{'#' * 90}")

    # Interpolate to annual
    edu_annual = interpolate_to_annual(edu_raw, col_name)

    # Build panel
    panel = build_panel(edu_annual, le_raw, gdp_raw, T_YEARS, LAG, "le_tp25")

    # Precompute entry years
    entry_years = precompute_entry_years(edu_annual)

    # Run ceiling × threshold sweep with residualization
    level_results = run_residualized_sweep(
        panel, entry_years, "le_tp25", CEILINGS, label="LE")

    all_results[level_name] = level_results

    # Summary
    print_summary(level_results, CEILINGS, "LE")


# ── Cross-level comparison ───────────────────────────────────────────

print(f"\n\n{'=' * 90}")
print("CROSS-LEVEL COMPARISON: entry=10%, ceiling=60%")
print("Edu→LE:    education predicts future life expectancy")
print("GDP→LE:    raw GDP predicts future life expectancy")
print("Resid→LE:  GDP AFTER removing education's effect on GDP → LE")
print("Edu→GDP:   how much of GDP is explained by education")
print(f"{'=' * 90}")
print(f"{'Level':<20} {'Edu→LE':>7} {'GDP→LE':>7} {'Resid→LE':>9} {'Edu→GDP':>8}")
print("-" * 55)
for level_name in EDU_LEVELS:
    r = all_results[level_name].get("60", {}).get("10", {})
    if r and r.get("edu_r2") is not None:
        print(f"{level_name:<20} {fmt_r2(r.get('edu_r2')):>7} {fmt_r2(r.get('raw_gdp_r2')):>7} "
              f"{fmt_r2(r.get('resid_gdp_r2')):>9} {fmt_r2(r.get('edu_gdp_r2')):>8}")

print(f"\n{'=' * 90}")
print("CROSS-LEVEL COMPARISON: entry=10%, ceiling=90%")
print(f"{'=' * 90}")
print(f"{'Level':<20} {'Edu→LE':>7} {'GDP→LE':>7} {'Resid→LE':>9} {'Edu→GDP':>8}")
print("-" * 55)
for level_name in EDU_LEVELS:
    r = all_results[level_name].get("90", {}).get("10", {})
    if r and r.get("edu_r2") is not None:
        print(f"{level_name:<20} {fmt_r2(r.get('edu_r2')):>7} {fmt_r2(r.get('raw_gdp_r2')):>7} "
              f"{fmt_r2(r.get('resid_gdp_r2')):>9} {fmt_r2(r.get('edu_gdp_r2')):>8}")

# ── Checkin ──────────────────────────────────────────────────────────

checkin = {
    "method": (
        "Country FE. Two-step residualization: (1) regress log_GDP on "
        "education with country FE to get GDP residuals (GDP not explained "
        "by education), (2) use residuals to predict LE(T+25). Compares "
        "education R², raw GDP R², and residualized GDP R². "
        "Entry-cohort with ceiling. T=1960-1990, lag=25. "
        "Three education levels: primary, lower secondary, upper secondary."
    ),
    "levels": all_results,
}

write_checkin("edu_vs_gdp_residualized.json", checkin,
              "scripts/residualization/education_vs_gdp.py")
