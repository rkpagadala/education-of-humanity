"""
residualization/by_level_lag0.py
=================================
By-level entry-cohort decomposition at LAG 0 (contemporaneous).

Parallels by_education_level.py (LE at lag 28) and education_vs_tfr.py
(TFR at lag 28), but at the §Three Timescales contemporaneous timescale.

Predictor cohort: 20-24 cohort completion (WCDE v3) at year T,
  one row per (level, country, T). T = 1960-2015, 5-yr stops.
Outcomes (lag 0): log(TFR), log(LE), log(GDP_pc).
  §Three Timescales mechanism: the 20-24 cohort is already in its
  reproductive/productive window at T; literacy acts in real time.
Sample: entry-cohort (country enters when its primary/lower-sec/upper-sec
  crosses the entry threshold; observations kept while education ≤ ceiling).
  Each education level uses its OWN threshold sample — matching the design
  the paper's T+28 §"Composition by education level" numbers come from.
Estimation: country FE (demean within country), within-R² and clustered SE.

This script closes the gap noted in the 7164e0d session: §Three Timescales
introduced the contemporaneous timescale and rewrote the §Mother routing
prose around "primary literacy at lag 0," but did NOT empirically test
primary vs lower-sec vs upper-sec at lag 0 in the entry-cohort design.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _shared import (
    load_education, load_wb, interpolate_to_annual,
    precompute_entry_years, filter_panel, build_panel, fe_r2,
    fe_residualize_gdp, write_checkin, fmt_r2,
)

# ── Load data ────────────────────────────────────────────────────────

print("Loading data...")

edu_raw = load_education("completion_both_long.csv")
tfr_raw = load_wb("children_per_woman_total_fertility.csv")
le_raw  = load_wb("life_expectancy_years.csv")
gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")

EDU_LEVELS = {
    "primary":         "primary",
    "lower_secondary": "lower_sec",
    "upper_secondary": "upper_sec",
}

# (label, outcome_df, log-transform?)
OUTCOMES = [
    ("log_tfr", tfr_raw, True),
    ("log_le",  le_raw,  True),
    ("log_gdp", gdp_raw, True),
]

T_YEARS  = list(range(1960, 2016, 5))   # lag 0 — full panel window
LAG      = 0
CEILINGS = [50, 60, 70, 80, 90]

all_results = {}

for level_name, col_name in EDU_LEVELS.items():
    print(f"\n{'#' * 90}")
    print(f"# LEVEL: {level_name.upper().replace('_', ' ')}  (WCDE column: {col_name})")
    print(f"{'#' * 90}")

    edu_annual  = interpolate_to_annual(edu_raw, col_name)
    entry_years = precompute_entry_years(edu_annual)

    level_results = {}

    for out_label, out_raw, log_it in OUTCOMES:
        # Build panel: edu at T, outcome at T+0 (i.e. T)
        panel = build_panel(edu_annual, out_raw, gdp_raw, T_YEARS, LAG, "outcome_raw")
        if log_it:
            panel["y"] = np.where(panel["outcome_raw"] > 0,
                                   np.log(panel["outcome_raw"]), np.nan)
        else:
            panel["y"] = panel["outcome_raw"]

        # The fe_residualize_gdp helper expects log_gdp_t to be its own column.
        # Already present from build_panel.

        outcome_results = {}
        for ceiling in CEILINGS:
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

                r2_e, n_e, c_e = fe_r2("edu_t",     "y", sub)
                r2_g, n_g, c_g = fe_r2("log_gdp_t", "y", sub)

                resid = fe_residualize_gdp(sub)
                if resid is not None:
                    sub_r, edu_gdp_r2 = resid
                    r2_resid, _, _ = fe_r2("gdp_resid", "y", sub_r)
                else:
                    r2_resid, edu_gdp_r2 = np.nan, np.nan

                ceil_results[str(threshold)] = {
                    "edu_r2":      round(r2_e, 3)     if not np.isnan(r2_e)     else None,
                    "raw_gdp_r2":  round(r2_g, 3)     if not np.isnan(r2_g)     else None,
                    "resid_gdp_r2":round(r2_resid, 3) if not np.isnan(r2_resid) else None,
                    "edu_gdp_r2":  round(edu_gdp_r2, 3) if not np.isnan(edu_gdp_r2) else None,
                    "edu_n":       n_e,
                    "edu_countries": c_e,
                }

            outcome_results[str(ceiling)] = ceil_results

        level_results[out_label] = outcome_results

        # Headline summary: entry=10, ceiling=90
        r = level_results[out_label].get("90", {}).get("10", {})
        if r and r.get("edu_r2") is not None:
            print(f"  {out_label:<10} entry=10 ceiling=90:  "
                  f"edu R²={r['edu_r2']:.3f}  raw GDP R²={fmt_r2(r['raw_gdp_r2'])}  "
                  f"resid GDP R²={fmt_r2(r['resid_gdp_r2'])}  "
                  f"n={r['edu_n']}  countries={r['edu_countries']}")

    all_results[level_name] = level_results


# ── Cross-level headline ────────────────────────────────────────────

print(f"\n\n{'=' * 90}")
print("HEADLINE: lag-0 by-level R² at entry=10, ceiling=90  (within-country FE)")
print(f"{'=' * 90}")
print(f"{'Outcome':<10}  {'Primary':>22}  {'Lower-sec':>22}  {'Upper-sec':>22}")
print("-" * 90)
for out_label, _, _ in OUTCOMES:
    cells = []
    for lvl in ["primary", "lower_secondary", "upper_secondary"]:
        r = all_results[lvl][out_label].get("90", {}).get("10", {})
        if r and r.get("edu_r2") is not None:
            cells.append(f"R²={r['edu_r2']:.3f}  n={r['edu_n']:4d}  C={r['edu_countries']:3d}".rjust(22))
        else:
            cells.append(f"{'--':>22}")
    print(f"{out_label:<10}  " + "  ".join(cells))

print(f"\n{'=' * 90}")
print("Winner per outcome at lag 0 (entry=10, ceiling=90)")
print(f"{'=' * 90}")
for out_label, _, _ in OUTCOMES:
    best_lvl, best_r2 = None, -1.0
    for lvl in ["primary", "lower_secondary", "upper_secondary"]:
        r = all_results[lvl][out_label].get("90", {}).get("10", {})
        if r and r.get("edu_r2") is not None and r["edu_r2"] > best_r2:
            best_lvl, best_r2 = lvl, r["edu_r2"]
    print(f"  {out_label:<10} → {best_lvl:<18}  R²={best_r2:.3f}")


# ── Checkin ──────────────────────────────────────────────────────────

# Pull the entry=10, ceiling=90 cells into a flat headline for the verify
# registry to address by short keys (parallel to the lag-28 by-level scripts).
headline = {}
for lvl in EDU_LEVELS:
    short = {"primary": "pri", "lower_secondary": "low", "upper_secondary": "upp"}[lvl]
    for out_label, _, _ in OUTCOMES:
        r = all_results[lvl][out_label].get("90", {}).get("10", {})
        if r is None:
            continue
        for k in ("edu_r2", "raw_gdp_r2", "resid_gdp_r2", "edu_gdp_r2"):
            v = r.get(k)
            if v is not None:
                headline[f"{out_label}.{short}.{k}"] = v
        headline[f"{out_label}.{short}.n"]         = r["edu_n"]
        headline[f"{out_label}.{short}.countries"] = r["edu_countries"]

checkin = {
    "method": (
        "Entry-cohort × ceiling by education level at LAG 0 (contemporaneous). "
        "Three levels: primary, lower secondary, upper secondary; each uses "
        "its own threshold sample. Outcomes: log(TFR), log(LE), log(GDP_pc). "
        "Country FE; within-R²; T = 1960-2015 (5yr stops); lag = 0 years. "
        "Mechanism (§Three Timescales): the 20-24 cohort is already in its "
        "reproductive/productive window at T; literacy acts in real time. "
        "This closes the gap left by the 7164e0d session, which introduced "
        "the contemporaneous timescale but only ran single-level Table-7-style "
        "stepwise (table_contemporaneous.py)."
    ),
    "levels":   all_results,
    "numbers":  headline,
}

write_checkin("by_level_lag0.json", checkin,
              "scripts/residualization/by_level_lag0.py")
