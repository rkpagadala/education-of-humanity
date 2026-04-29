"""
residualization/by_entry_threshold.py
==============================
Does education or GDP predict future life expectancy?

Same question as residualization/education_predicts_le.py but with a better sample
definition: instead of "observations where education < X%", we define
entry cohorts — countries that first crossed a given education threshold,
tracked from that point onward through their full development trajectory.

METHOD
------
1. Interpolate WCDE 5-year education data to annual values (linear).
2. For each entry threshold (10%, 11%, ..., 90%), find the first year
   each country crosses that threshold.
3. Build a panel: for each country in the cohort, include all T-year
   observations from the entry year onward (T = 1960..1990, 5-year steps).
4. Run country fixed-effects regression:
   - education(T) → life expectancy(T+25)
   - log GDP(T) → life expectancy(T+25)
5. Report within-R² at each threshold.

This gives more observations per country (full trajectory after entry)
and a cleaner sample definition ("countries that started from X% or below").

DATA
----
Same as residualization/education_predicts_le.py:
- Education: WCDE v3, lower secondary completion, both sexes, age 20-24
- Life expectancy: World Bank WDI (SP.DYN.LE00.IN)
- GDP: World Bank WDI, constant 2015 USD (NY.GDP.PCAP.KD), log-transformed

OUTPUT
------
Table of R² values at each 1% threshold, plus JSON checkin file.
"""

import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _shared import (
    load_education, load_wb, interpolate_to_annual,
    precompute_entry_years, filter_panel, build_panel, fe_r2, write_checkin,
)

# ── Load data ────────────────────────────────────────────────────────

print("Loading data...")

# Education: WCDE v3, lower secondary completion, both sexes
edu = load_education("completion_both_long.csv")
print(f"  Education: {edu['country'].nunique()} countries, "
      f"years {edu['year'].min()}-{edu['year'].max()}")

# Life expectancy: World Bank WDI
le_raw = load_wb("life_expectancy_years.csv")
print(f"  Life expectancy: {len(le_raw)} countries")

# GDP per capita: World Bank WDI, constant 2015 USD
gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")
print(f"  GDP: {len(gdp_raw)} countries")


# ── Interpolate education to annual values ───────────────────────────
# WCDE is at 5-year intervals. Interpolate linearly to get annual
# values so we can find the precise year each country crosses a threshold.

print("\nInterpolating education to annual values...")

edu_annual = interpolate_to_annual(edu, "lower_sec")
print(f"  Interpolated {len(edu_annual)} countries")


# ── Build the full panel (all T-years, all countries) ────────────────
# We build once, then filter per threshold.

T_YEARS = list(range(1960, 1995, 5))  # 1960, 1965, 1970, 1975, 1980, 1985, 1990
LAG = 25

print(f"\nBuilding full panel (T={T_YEARS[0]}-{T_YEARS[-1]}, lag={LAG})...")

panel = build_panel(edu_annual, le_raw, gdp_raw, T_YEARS, LAG, "le_tp25")
print(f"  Full panel: {len(panel)} obs, {panel['country'].nunique()} countries")

entry_years = precompute_entry_years(edu_annual)

# ── Run at every 1% threshold from 10% to 90% ───────────────────────

print("\n" + "=" * 90)
print("Entry-Cohort Analysis: Education(T) vs GDP(T) → Life Expectancy(T+25)")
print("Country fixed effects | T = 1960-1990 (5yr) | Lag = 25 years")
print("Entry = first year country crosses threshold; all observations from then on")
print("=" * 90)
print(f"{'Threshold':<12} {'Edu R²':>8} {'n':>6} {'Ctry':>6}   "
      f"{'GDP R²':>8} {'n':>6} {'Ctry':>6}   {'Ratio':>8}")
print("-" * 90)

results = {}

for threshold in range(10, 91):
    cohort = entry_years.get(threshold, {})

    if len(cohort) < 3:
        continue

    sub = filter_panel(panel, cohort, ceiling=100)

    if len(sub) < 10:
        continue

    r2_e, n_e, c_e = fe_r2("edu_t", "le_tp25", sub)
    r2_g, n_g, c_g = fe_r2("log_gdp_t", "le_tp25", sub)

    if not np.isnan(r2_g) and r2_g > 0.001:
        ratio = f"{r2_e / r2_g:.1f}x"
    elif not np.isnan(r2_e):
        ratio = "GDP≈0"
    else:
        ratio = "n/a"

    r2_e_s = f"{r2_e:.3f}" if not np.isnan(r2_e) else "n/a"
    r2_g_s = f"{r2_g:.3f}" if not np.isnan(r2_g) else "n/a"

    # Print every 5% to keep output readable, but store all
    if threshold % 5 == 0 or threshold == 10:
        print(f"  >= {threshold}%{'':<6} {r2_e_s:>8} {n_e:>6} {c_e:>6}   "
              f"{r2_g_s:>8} {n_g:>6} {c_g:>6}   {ratio:>8}")

    results[str(threshold)] = {
        "edu_r2": round(r2_e, 3) if not np.isnan(r2_e) else None,
        "edu_n": n_e,
        "edu_countries": c_e,
        "gdp_r2": round(r2_g, 3) if not np.isnan(r2_g) else None,
        "gdp_n": n_g,
        "gdp_countries": c_g,
    }

# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

# Find the range of edu R² and GDP R²
edu_r2s = [v["edu_r2"] for v in results.values() if v["edu_r2"] is not None]
gdp_r2s = [v["gdp_r2"] for v in results.values() if v["gdp_r2"] is not None]

if edu_r2s and gdp_r2s:
    print(f"Education R² range: {min(edu_r2s):.3f} - {max(edu_r2s):.3f}")
    print(f"GDP R² range:       {min(gdp_r2s):.3f} - {max(gdp_r2s):.3f}")
    print(f"Education consistently explains {min(edu_r2s)*100:.0f}-{max(edu_r2s)*100:.0f}% "
          f"of within-country variation in future life expectancy.")
    print(f"GDP consistently explains {min(gdp_r2s)*100:.0f}-{max(gdp_r2s)*100:.0f}%.")

# ── Write checkin file ───────────────────────────────────────────────

checkin = {
    "method": (
        "Country FE (demeaned). Entry-cohort design: for each threshold "
        "(10-90%), find first year each country crosses it, then include "
        "all T-year observations from entry onward. "
        "T = 1960/1965/.../1990, outcome = LE(T+25). "
        "Education interpolated from WCDE 5-year to annual (linear)."
    ),
    "education_variable": "lower secondary completion, both sexes, age 20-24 (WCDE v3)",
    "gdp_variable": "log GDP per capita, constant 2015 USD (World Bank WDI)",
    "le_variable": "life expectancy at birth (World Bank WDI)",
    "thresholds": results,
}

write_checkin("edu_vs_gdp_entry_threshold.json", checkin,
              "scripts/residualization/by_entry_threshold.py")
