"""
residualization/by_level_lag15.py
==================================
By-level entry-cohort decomposition at LAG 15 (childrearing window).

Parallels by_level_lag0.py and the lag-25 by-level scripts, at the
§Three Timescales childrearing-window timescale.

Predictor cohort: 20-24 cohort completion (WCDE v3) at year T.
Outcome (lag 15): log(U5MR).
  §Three Timescales mechanism: the cohort's own children must be born
  and survive (or not) to age 5 before the survival outcome registers;
  outcomes_r2_by_lag.json places the U5MR R² peak at lag 10-15 (R²=0.716).
Sample: entry-cohort with ceiling, each level uses its own threshold sample.
Estimation: country FE; within-R².
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

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
u5_raw  = load_wb("child_mortality_u5.csv")
gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")

EDU_LEVELS = {
    "primary":         "primary",
    "lower_secondary": "lower_sec",
    "upper_secondary": "upper_sec",
}

T_YEARS  = list(range(1960, 2001, 5))   # outcome at T+15 → T_max = 2000
LAG      = 15
CEILINGS = [50, 60, 70, 80, 90]

all_results = {}

for level_name, col_name in EDU_LEVELS.items():
    print(f"\n{'#' * 90}")
    print(f"# LEVEL: {level_name.upper().replace('_', ' ')}  (WCDE column: {col_name})")
    print(f"{'#' * 90}")

    edu_annual  = interpolate_to_annual(edu_raw, col_name)
    entry_years = precompute_entry_years(edu_annual)

    # Build panel: edu at T, U5MR at T+15
    panel = build_panel(edu_annual, u5_raw, gdp_raw, T_YEARS, LAG, "u5_raw")
    panel["y"] = np.where(panel["u5_raw"] > 0, np.log(panel["u5_raw"]), np.nan)

    level_results = {}
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
                "edu_r2":       round(r2_e, 3)     if not np.isnan(r2_e)     else None,
                "raw_gdp_r2":   round(r2_g, 3)     if not np.isnan(r2_g)     else None,
                "resid_gdp_r2": round(r2_resid, 3) if not np.isnan(r2_resid) else None,
                "edu_gdp_r2":   round(edu_gdp_r2, 3) if not np.isnan(edu_gdp_r2) else None,
                "edu_n":        n_e,
                "edu_countries": c_e,
            }

        level_results[str(ceiling)] = ceil_results

    all_results[level_name] = level_results

    # Headline (entry=10, ceiling=90)
    r = level_results.get("90", {}).get("10", {})
    if r and r.get("edu_r2") is not None:
        print(f"  log_u5 entry=10 ceiling=90:  edu R²={r['edu_r2']:.3f}  "
              f"raw GDP R²={fmt_r2(r['raw_gdp_r2'])}  "
              f"resid GDP R²={fmt_r2(r['resid_gdp_r2'])}  "
              f"n={r['edu_n']}  countries={r['edu_countries']}")


# ── Cross-level headline ────────────────────────────────────────────

print(f"\n\n{'=' * 90}")
print("HEADLINE: lag-15 log(U5MR) by-level R² at entry=10, ceiling=90")
print(f"{'=' * 90}")
print(f"{'':<10}  {'Primary':>22}  {'Lower-sec':>22}  {'Upper-sec':>22}")
print("-" * 90)
cells = []
for lvl in ["primary", "lower_secondary", "upper_secondary"]:
    r = all_results[lvl].get("90", {}).get("10", {})
    if r and r.get("edu_r2") is not None:
        cells.append(f"R²={r['edu_r2']:.3f}  n={r['edu_n']:4d}  C={r['edu_countries']:3d}".rjust(22))
    else:
        cells.append(f"{'--':>22}")
print(f"{'log_u5':<10}  " + "  ".join(cells))

print(f"\n{'=' * 90}")
print("Winner at lag 15 (entry=10, ceiling=90)")
print(f"{'=' * 90}")
best_lvl, best_r2 = None, -1.0
for lvl in ["primary", "lower_secondary", "upper_secondary"]:
    r = all_results[lvl].get("90", {}).get("10", {})
    if r and r.get("edu_r2") is not None and r["edu_r2"] > best_r2:
        best_lvl, best_r2 = lvl, r["edu_r2"]
print(f"  log_u5 → {best_lvl}  R²={best_r2:.3f}")


# ── Checkin ──────────────────────────────────────────────────────────

headline = {}
for lvl in EDU_LEVELS:
    short = {"primary": "pri", "lower_secondary": "low", "upper_secondary": "upp"}[lvl]
    r = all_results[lvl].get("90", {}).get("10", {})
    if r is None:
        continue
    for k in ("edu_r2", "raw_gdp_r2", "resid_gdp_r2", "edu_gdp_r2"):
        v = r.get(k)
        if v is not None:
            headline[f"log_u5.{short}.{k}"] = v
    headline[f"log_u5.{short}.n"]         = r["edu_n"]
    headline[f"log_u5.{short}.countries"] = r["edu_countries"]

checkin = {
    "method": (
        "Entry-cohort × ceiling by education level at LAG 15 (childrearing "
        "window). Three levels: primary, lower secondary, upper secondary; "
        "each uses its own threshold sample. Outcome: log(U5MR). Country FE; "
        "within-R²; T = 1960-2000 (5yr stops); lag = 15 years. "
        "Mechanism (§Three Timescales): the 20-24 cohort's own children must "
        "be born and survive (or not) to age 5 before the survival outcome "
        "registers; outcomes_r2_by_lag.json places U5MR R² peak at lag 10-15."
    ),
    "levels":  all_results,
    "numbers": headline,
}

write_checkin("by_level_lag15.json", checkin,
              "scripts/residualization/by_level_lag15.py")
