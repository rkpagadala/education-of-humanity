"""
residualization/cutoff_all_outcomes.py
======================================
Generalize the education-cutoff R² comparison from life expectancy to all
four outcomes:
  1. Life expectancy at T+25 (LE)
  2. Total fertility rate at T+25 (TFR)
  3. Log under-5 mortality at T+25 (log U5MR)
  4. Child education at T+25 (same country, intergenerational)

For each outcome × cutoff (<10, <30, <50, <70, <90, full panel), compute
within-country FE R² for education(T) vs log_GDP(T) as predictors.

The paper currently features LE at <30%: education 31%, GDP 2%, 15× gap.
This script produces the analogous table across all four outcomes.

METHODOLOGY
-----------
Matches scripts/residualization/education_predicts_le.py:
  - Country fixed effects (demeaned by country; ≥2 obs per country)
  - T years: 1975, 1980, 1985, 1990 (WCDE 5-year intervals)
  - Lag: 25 years → outcomes at 2000, 2005, 2010, 2015
  - Education: WCDE v3, lower secondary completion, both sexes, age 20-24
  - GDP: log GDP per capita, constant 2017 USD (WDI)
  - U5MR: log-transformed (standard for mortality; compresses skew)
  - Child education: lower secondary completion at T+25 in the same country

The cutoff operates on education(T) — "among observations where the country
had less than C% lower secondary completion at T, how well does education at
T predict the outcome at T+25 once we remove country fixed effects?"

OUTPUT
------
- Console table: 6 cutoffs × 4 outcomes × {edu R², GDP R², ratio}
- checkin/cutoff_all_outcomes.json
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from _shared import (
    load_education, load_wb, get_wb_val, fmt_r2, write_checkin,
)
from residualization._shared import fe_r2


# ── Config ───────────────────────────────────────────────────────────
T_YEARS = [1975, 1980, 1985, 1990]
LAG = 25
CUTOFFS = [10, 30, 50, 70, 90, None]  # None = full panel
OUTCOMES = ["le", "tfr", "log_u5mr", "child_edu"]
OUTCOME_LABELS = {
    "le": "Life expectancy (T+25)",
    "tfr": "Fertility rate (T+25)",
    "log_u5mr": "log Under-5 mortality (T+25)",
    "child_edu": "Child education (T+25)",
}


# ── Load data ────────────────────────────────────────────────────────
print("Loading data...")

edu = load_education("completion_both_long.csv")
print(f"  Education: {edu['country'].nunique()} countries")

le_raw = load_wb("life_expectancy_years.csv")
tfr_raw = load_wb("children_per_woman_total_fertility.csv")
u5mr_raw = load_wb("child_mortality_u5.csv")
gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")
print(f"  LE: {len(le_raw)}, TFR: {len(tfr_raw)}, U5MR: {len(u5mr_raw)}, "
      f"GDP: {len(gdp_raw)}")


# ── Build panel ──────────────────────────────────────────────────────
# For each country × T, collect education(T), log_GDP(T), and each outcome
# at T+25. Child-education outcome uses WCDE lower-sec at T+25 in the same
# country (the intergenerational measure).

print(f"\nBuilding panel (T={T_YEARS}, lag={LAG})...")

# Precompute a country-indexed education series for fast T+25 lookup
edu_lookup = {}
for c, grp in edu.groupby("country"):
    edu_lookup[c] = grp.set_index("year")["lower_sec"]

rows = []
for c in sorted(edu["country"].unique()):
    edu_c = edu_lookup[c]
    for t in T_YEARS:
        if t not in edu_c.index:
            continue
        edu_t = edu_c.loc[t]
        if np.isnan(edu_t):
            continue
        tp25 = t + LAG

        gdp_t = get_wb_val(gdp_raw, c, t)
        log_gdp = np.log(gdp_t) if (not np.isnan(gdp_t) and gdp_t > 0) else np.nan

        le_tp25 = get_wb_val(le_raw, c, tp25)
        tfr_tp25 = get_wb_val(tfr_raw, c, tp25)
        u5_tp25 = get_wb_val(u5mr_raw, c, tp25)
        log_u5 = np.log(u5_tp25) if (not np.isnan(u5_tp25) and u5_tp25 > 0) else np.nan

        child_edu = (edu_c.loc[tp25] if tp25 in edu_c.index else np.nan)

        rows.append({
            "country": c,
            "t": t,
            "edu_t": edu_t,
            "log_gdp_t": log_gdp,
            "le": le_tp25,
            "tfr": tfr_tp25,
            "log_u5mr": log_u5,
            "child_edu": child_edu,
        })

panel = pd.DataFrame(rows)
print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries")
for o in OUTCOMES:
    print(f"    {o}: {panel[o].notna().sum()} non-null obs")


# ── Run all cutoffs × all outcomes ───────────────────────────────────
print("\n" + "=" * 95)
print("Education(T) vs log_GDP(T) predicting outcomes(T+25) — Country FE (within R²)")
print(f"T = {T_YEARS} | Lag = {LAG} years")
print("=" * 95)

header = f"{'Outcome':<26} {'Cutoff':<8} {'Edu R²':>8} {'GDP R²':>8} {'Ratio':>8} " \
         f"{'n':>5} {'Ctry':>5}"
print(header)
print("-" * 95)

results = {o: {} for o in OUTCOMES}

for outcome in OUTCOMES:
    for cutoff in CUTOFFS:
        label = f"< {cutoff}%" if cutoff else "All"
        sub = panel[panel["edu_t"] < cutoff] if cutoff else panel

        r2_e, n_e, c_e = fe_r2("edu_t", outcome, sub)
        r2_g, n_g, c_g = fe_r2("log_gdp_t", outcome, sub)

        if not np.isnan(r2_g) and r2_g > 0.001 and not np.isnan(r2_e):
            ratio_val = r2_e / r2_g
            ratio_str = f"{ratio_val:.1f}x"
        elif not np.isnan(r2_e):
            ratio_val = None
            ratio_str = "GDP~0"
        else:
            ratio_val = None
            ratio_str = "n/a"

        print(f"{OUTCOME_LABELS[outcome]:<26} {label:<8} "
              f"{fmt_r2(r2_e):>8} {fmt_r2(r2_g):>8} {ratio_str:>8} "
              f"{n_e:>5} {c_e:>5}")

        key = f"lt{cutoff}" if cutoff else "all"
        results[outcome][key] = {
            "cutoff": cutoff,
            "edu_r2": round(r2_e, 3) if not np.isnan(r2_e) else None,
            "gdp_r2": round(r2_g, 3) if not np.isnan(r2_g) else None,
            "ratio": round(ratio_val, 2) if ratio_val is not None else None,
            "edu_n": int(n_e),
            "edu_countries": int(c_e),
            "gdp_n": int(n_g),
            "gdp_countries": int(c_g),
        }
    print("-" * 95)


# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 95)
print("SUMMARY — ratio (education R² / GDP R²) by cutoff")
print("=" * 95)
print(f"{'Outcome':<26} " + " ".join(f"{('<'+str(c)+'%') if c else 'All':>10}"
                                      for c in CUTOFFS))
print("-" * 95)
for outcome in OUTCOMES:
    row = [OUTCOME_LABELS[outcome]]
    for cutoff in CUTOFFS:
        key = f"lt{cutoff}" if cutoff else "all"
        r = results[outcome].get(key, {})
        ratio = r.get("ratio")
        row.append(f"{ratio:.1f}x" if ratio is not None else "  n/a")
    print(f"{row[0]:<26} " + " ".join(f"{v:>10}" for v in row[1:]))


# ── Write checkin ────────────────────────────────────────────────────
write_checkin("cutoff_all_outcomes.json", {
    "method": (
        "Country FE (demeaned). Cutoff applied to education(T). "
        f"T = {T_YEARS}, lag = {LAG}. "
        "Outcomes: LE, TFR, log U5MR, child education (lower-sec at T+25)."
    ),
    "education_variable": "lower secondary completion, both sexes, age 20-24 (WCDE v3)",
    "gdp_variable": "log GDP per capita, constant 2017 USD (World Bank WDI)",
    "outcome_variables": {
        "le": "life expectancy at birth (WDI SP.DYN.LE00.IN)",
        "tfr": "total fertility rate (WDI SP.DYN.TFRT.IN)",
        "log_u5mr": "log under-5 mortality per 1000 (WDI)",
        "child_edu": "lower secondary completion at T+25, same country (WCDE v3)",
    },
    "cutoffs": [str(c) if c else "all" for c in CUTOFFS],
    "panel_size": {
        "total_obs": int(len(panel)),
        "total_countries": int(panel["country"].nunique()),
    },
    "results": results,
}, script_path="scripts/residualization/cutoff_all_outcomes.py")
