"""
residualization/education_vs_tfr.py
================================
Does education or GDP predict future fertility (TFR)?
And does GDP have any independent effect after controlling for
education's contribution to GDP?

Same design as residualization/education_vs_gdp.py but with TFR(T+25) as outcome.

METHOD
------
Country fixed effects, entry-cohort with ceiling.
  - Edu R²:     education(T) → TFR(T+25)
  - GDP R²:     log_GDP(T) → TFR(T+25)
  - Resid R²:   GDP_residual(T) → TFR(T+25)
  - Edu→GDP:    how much of GDP is explained by education

Three education levels: primary, lower secondary, upper secondary.
T = 1960-1990 (5yr), lag = 25 years.
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

tfr_raw = load_wb("children_per_woman_total_fertility.csv")
print(f"  TFR: {len(tfr_raw)} countries")

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
    panel = build_panel(edu_annual, tfr_raw, gdp_raw, T_YEARS, LAG, "tfr_tp25")

    # Precompute entry years
    entry_years = precompute_entry_years(edu_annual)

    # Run ceiling × threshold sweep with residualization
    level_results = run_residualized_sweep(
        panel, entry_years, "tfr_tp25", CEILINGS, label="TFR")

    all_results[level_name] = level_results

    # Summary
    print_summary(level_results, CEILINGS, "TFR")


# ── Cross-level comparison ───────────────────────────────────────────

print(f"\n\n{'=' * 90}")
print("CROSS-LEVEL COMPARISON: entry=10%, ceiling=60%")
print("Edu→TFR:    education predicts future fertility")
print("GDP→TFR:    raw GDP predicts future fertility")
print("Resid→TFR:  GDP AFTER removing education's effect on GDP → LE")
print("Edu→GDP:   how much of GDP is explained by education")
print(f"{'=' * 90}")
print(f"{'Level':<20} {'Edu→TFR':>7} {'GDP→TFR':>7} {'Resid→TFR':>9} {'Edu→GDP':>8}")
print("-" * 55)
for level_name in EDU_LEVELS:
    r = all_results[level_name].get("60", {}).get("10", {})
    if r and r.get("edu_r2") is not None:
        print(f"{level_name:<20} {fmt_r2(r.get('edu_r2')):>7} {fmt_r2(r.get('raw_gdp_r2')):>7} "
              f"{fmt_r2(r.get('resid_gdp_r2')):>9} {fmt_r2(r.get('edu_gdp_r2')):>8}")

print(f"\n{'=' * 90}")
print("CROSS-LEVEL COMPARISON: entry=10%, ceiling=90%")
print(f"{'=' * 90}")
print(f"{'Level':<20} {'Edu→TFR':>7} {'GDP→TFR':>7} {'Resid→TFR':>9} {'Edu→GDP':>8}")
print("-" * 55)
for level_name in EDU_LEVELS:
    r = all_results[level_name].get("90", {}).get("10", {})
    if r and r.get("edu_r2") is not None:
        print(f"{level_name:<20} {fmt_r2(r.get('edu_r2')):>7} {fmt_r2(r.get('raw_gdp_r2')):>7} "
              f"{fmt_r2(r.get('resid_gdp_r2')):>9} {fmt_r2(r.get('edu_gdp_r2')):>8}")

# ── Checkin ──────────────────────────────────────────────────────────

# Extract Fert-primary-R2: primary edu→TFR R² at entry=10%, ceiling=90%
_fert_primary_r2 = None
try:
    _fert_primary_r2 = all_results["primary"]["90"]["10"]["edu_r2"]
except (KeyError, TypeError):
    pass

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
    "numbers": {
        "Fert-primary-R2": _fert_primary_r2,
    },
}

write_checkin("edu_vs_gdp_tfr_residualized.json", checkin,
              "scripts/residualization/education_vs_tfr.py")
