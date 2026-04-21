"""
residualization/by_entry_ceiling.py
============================
Entry-cohort design with education ceiling.

For each (entry_threshold, ceiling) pair:
1. Find countries that first crossed entry_threshold%.
2. Include observations from entry onward, BUT only while education <= ceiling%.
3. Run country fixed-effects: education(T) → LE(T+25) vs GDP(T) → LE(T+25).

This prevents high-education observations from dominating the sample.

Entry thresholds: 10% to 90% (1% steps)
Ceilings: 50%, 55%, 60%, 65%, 70%, 75%, 80%, 85%, 90%
"""

import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _shared import (
    load_education, load_wb, interpolate_to_annual, fmt_r2,
    precompute_entry_years, filter_panel, build_panel,
    fe_r2, write_checkin, CHECKIN,
)

# ── Load data ────────────────────────────────────────────────────────

print("Loading data...")

edu = load_education("completion_both_long.csv")
le_raw = load_wb("life_expectancy_years.csv")
gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")

# ── Interpolate education to annual ──────────────────────────────────

print("Interpolating education to annual...")

edu_annual = interpolate_to_annual(edu, "lower_sec")

# ── Build full panel ─────────────────────────────────────────────────

T_YEARS = list(range(1960, 1995, 5))
LAG = 25

print(f"Building panel (T={T_YEARS[0]}-{T_YEARS[-1]}, lag={LAG})...")

panel = build_panel(edu_annual, le_raw, gdp_raw, T_YEARS, LAG, "le_tp25")
print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries")


# ── Precompute entry years for all thresholds ────────────────────────

print("Precomputing entry years...")
entry_years = precompute_entry_years(edu_annual)


# ── Run 2D sweep: entry threshold × ceiling ──────────────────────────

CEILINGS = list(range(50, 95, 5))  # 50, 55, 60, 65, 70, 75, 80, 85, 90

results = {}

for ceiling in CEILINGS:
    print(f"\n{'=' * 90}")
    print(f"CEILING = {ceiling}%  (only observations where education <= {ceiling}%)")
    print(f"{'=' * 90}")
    print(f"{'Entry':<10} {'Edu R²':>8} {'n':>6} {'Ctry':>6}   "
          f"{'GDP R²':>8} {'n':>6} {'Ctry':>6}   {'Ratio':>8}")
    print("-" * 90)

    ceil_results = {}

    for threshold in range(10, 91):
        if threshold > ceiling:
            break  # can't enter above the ceiling

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

        if threshold % 10 == 0 or threshold == 10:
            print(f"  >= {threshold}%{'':<4} {fmt_r2(r2_e):>8} {n_e:>6} {c_e:>6}   "
                  f"{fmt_r2(r2_g):>8} {n_g:>6} {c_g:>6}   {ratio_s:>8}")

        ceil_results[str(threshold)] = {
            "edu_r2": round(r2_e, 3) if not np.isnan(r2_e) else None,
            "edu_n": n_e, "edu_countries": c_e,
            "gdp_r2": round(r2_g, 3) if not np.isnan(r2_g) else None,
            "gdp_n": n_g, "gdp_countries": c_g,
        }

    results[str(ceiling)] = ceil_results

# ── Summary table ────────────────────────────────────────────────────

print(f"\n\n{'=' * 100}")
print("SUMMARY: Edu R² / GDP R² at entry=10% for each ceiling")
print(f"{'=' * 100}")
print(f"{'Ceiling':<10} {'Edu R²':>8} {'GDP R²':>8} {'Ratio':>8} {'Edu n':>8} {'Ctry':>6}")
print("-" * 50)
for ceiling in CEILINGS:
    r = results[str(ceiling)].get("10", {})
    if r and r.get("edu_r2") is not None:
        ratio = f"{r['edu_r2']/r['gdp_r2']:.1f}x" if r.get("gdp_r2") and r["gdp_r2"] > 0.001 else "GDP≈0"
        print(f"  <= {ceiling}%{'':<4} {r['edu_r2']:>8.3f} {r['gdp_r2']:>8.3f} {ratio:>8} {r['edu_n']:>8} {r['edu_countries']:>6}")

# ── Checkin ──────────────────────────────────────────────────────────

checkin = {
    "method": (
        "Country FE (demeaned). Entry-cohort with ceiling: enter when "
        "country first crosses entry threshold, include observations "
        "while education <= ceiling. T = 1960-1990 (5yr), outcome = LE(T+25). "
        "Education interpolated from WCDE 5-year to annual (linear)."
    ),
    "ceilings": {str(c): results[str(c)] for c in CEILINGS},
}

write_checkin("edu_vs_gdp_entry_ceiling.json", checkin,
              "scripts/residualization/by_entry_ceiling.py")
