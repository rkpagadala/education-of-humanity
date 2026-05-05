"""
residualization/by_education_level.py
=======================
Entry-cohort + ceiling analysis for three education levels:
  - Primary completion
  - Lower secondary completion
  - Upper secondary completion

For each level, runs the same design as residualization/by_entry_ceiling.py:
  1. Interpolate education to annual values
  2. For each (entry_threshold, ceiling) pair, find countries that crossed
     entry, include observations while education <= ceiling
  3. Country fixed-effects: education(T) → LE(T+25) vs GDP(T) → LE(T+25)

Entry thresholds: 10% to 90% (1% steps)
Ceilings: 50%, 60%, 70%, 80%, 90%
"""

import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _shared import (
    load_education, load_wb, interpolate_to_annual,
    precompute_entry_years, filter_panel, build_panel,
    fe_r2, write_checkin, CHECKIN,
)

# ── Load data ────────────────────────────────────────────────────────

print("Loading data...")

edu_raw = load_education("completion_both_long.csv")
le_raw = load_wb("life_expectancy_years.csv")
gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")

# ── Education levels to test ─────────────────────────────────────────

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
    print(f"# EDUCATION LEVEL: {level_name.upper().replace('_', ' ')}")
    print(f"# Column: {col_name}")
    print(f"{'#' * 90}")

    # Interpolate to annual
    edu_annual = interpolate_to_annual(edu_raw, col_name)

    # Build panel
    panel = build_panel(edu_annual, le_raw, gdp_raw, T_YEARS, LAG, "le_tp25")
    print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries")

    # Precompute entry years
    entry_years = precompute_entry_years(edu_annual)

    # Run ceiling sweep
    level_results = {}

    for ceiling in CEILINGS:
        print(f"\n  Ceiling = {ceiling}%")
        print(f"  {'Entry':<10} {'Edu R²':>8} {'n':>6} {'Ctry':>6}   "
              f"{'GDP R²':>8} {'n':>6} {'Ctry':>6}   {'Ratio':>8}")
        print(f"  {'-' * 80}")

        ceil_results = {}

        for threshold in range(10, 91):
            if threshold > ceiling:
                break

            cohort = entry_years.get(threshold, {})
            if len(cohort) < 3:
                continue

            sub = filter_panel(panel, cohort, ceiling)
            if len(sub) < 10:
                continue

            r2_e, n_e, c_e = fe_r2("edu_t", "le_tp25", sub)
            r2_g, n_g, c_g = fe_r2("log_gdp_t", "le_tp25", sub)

            if not np.isnan(r2_g) and r2_g > 0.001:
                ratio_s = f"{r2_e / r2_g:.1f}x"
            elif not np.isnan(r2_e):
                ratio_s = "GDP≈0"
            else:
                ratio_s = "n/a"

            r2_e_s = f"{r2_e:.3f}" if not np.isnan(r2_e) else "n/a"
            r2_g_s = f"{r2_g:.3f}" if not np.isnan(r2_g) else "n/a"

            if threshold % 10 == 0 or threshold == 10:
                print(f"  >= {threshold}%{'':<4} {r2_e_s:>8} {n_e:>6} {c_e:>6}   "
                      f"{r2_g_s:>8} {n_g:>6} {c_g:>6}   {ratio_s:>8}")

            ceil_results[str(threshold)] = {
                "edu_r2": round(r2_e, 3) if not np.isnan(r2_e) else None,
                "edu_n": n_e, "edu_countries": c_e,
                "gdp_r2": round(r2_g, 3) if not np.isnan(r2_g) else None,
                "gdp_n": n_g, "gdp_countries": c_g,
            }

        level_results[str(ceiling)] = ceil_results

    all_results[level_name] = level_results

    # Summary for this level
    print(f"\n  SUMMARY ({level_name}): entry=10%, varying ceiling")
    print(f"  {'Ceiling':<10} {'Edu R²':>8} {'GDP R²':>8} {'Ratio':>8} {'n':>6} {'Ctry':>6}")
    print(f"  {'-' * 50}")
    for ceiling in CEILINGS:
        r = level_results[str(ceiling)].get("10", {})
        if r and r.get("edu_r2") is not None:
            ratio = f"{r['edu_r2']/r['gdp_r2']:.1f}x" if r.get("gdp_r2") and r["gdp_r2"] > 0.001 else "GDP≈0"
            print(f"  <= {ceiling}%{'':<4} {r['edu_r2']:>8.3f} {r.get('gdp_r2', 0):>8.3f} "
                  f"{ratio:>8} {r['edu_n']:>6} {r['edu_countries']:>6}")


# ── Cross-level comparison ───────────────────────────────────────────

print(f"\n\n{'=' * 90}")
print("CROSS-LEVEL COMPARISON: entry=10%, ceiling=60%")
print(f"{'=' * 90}")
print(f"{'Level':<20} {'Edu R²':>8} {'GDP R²':>8} {'Ratio':>8} {'n':>6} {'Ctry':>6}")
print("-" * 60)
for level_name in EDU_LEVELS:
    r = all_results[level_name].get("60", {}).get("10", {})
    if r and r.get("edu_r2") is not None:
        ratio = f"{r['edu_r2']/r['gdp_r2']:.1f}x" if r.get("gdp_r2") and r["gdp_r2"] > 0.001 else "GDP≈0"
        print(f"{level_name:<20} {r['edu_r2']:>8.3f} {r.get('gdp_r2', 0):>8.3f} "
              f"{ratio:>8} {r['edu_n']:>6} {r['edu_countries']:>6}")

# ── Checkin ──────────────────────────────────────────────────────────

checkin = {
    "method": (
        "Country FE (demeaned). Entry-cohort with ceiling, tested at three "
        "education levels: primary, lower secondary, upper secondary. "
        "T = 1960-1990 (5yr), outcome = LE(T+25). "
        "Education interpolated from WCDE 5-year to annual (linear)."
    ),
    "levels": all_results,
}

write_checkin("edu_vs_gdp_by_level.json", checkin,
              "scripts/residualization/by_education_level.py")
