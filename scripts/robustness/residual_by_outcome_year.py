"""
robustness/residual_by_outcome_year.py
================================
Sweep residualized GDP R² by individual outcome year for ALL four outcomes:
  - Life expectancy
  - Total fertility rate
  - Child education (lower secondary, T+25)
  - Under-5 mortality

For each outcome year from 1985 to 2020, restrict to observations where
T+lag <= that year, then report education R² and residualized GDP R².

This reveals whether GDP's apparent signal in U5MR is a post-MDG artifact
and confirms that other outcomes show no such pattern.

Entry-cohort design (entry >= 10%, ceiling <= 90%), country FE,
lower secondary completion, T=1960-1990, lag=25.
Clustered standard errors by country.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from residualization._shared import (
    load_education, load_wb, get_wb_val, interpolate_to_annual,
    precompute_entry_years, build_panel, filter_panel,
    fe_residualize_gdp, clustered_fe,
)
from _shared import write_checkin, fmt_r2

# ── Parameters ────────────────────────────────────────────────────

T_YEARS = list(range(1960, 1995, 5))
LAG = 25
COL_NAME = "lower_sec"
CEILING = 90
ENTRY_THRESHOLD = 10


# ── Load data ─────────────────────────────────────────────────────

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
le_df = load_wb("life_expectancy_years.csv")
tfr_df = load_wb("children_per_woman_total_fertility.csv")
u5mr_df = load_wb("child_mortality_u5.csv")

edu_annual = interpolate_to_annual(edu_raw, COL_NAME)
entry_years = precompute_entry_years(edu_annual)
cohort = entry_years[ENTRY_THRESHOLD]

# ── Build panels ──────────────────────────────────────────────────

outcomes = {
    "Life expectancy": ("le", le_df),
    "TFR":             ("tfr", tfr_df),
    "Child education": ("child_edu", None),  # special case
    "U5MR":            ("u5mr", u5mr_df),
}

# Child education panel: education at T+25 predicted by education at T
child_edu_rows = []
for c in sorted(edu_annual.keys()):
    s = edu_annual[c]
    for t in T_YEARS:
        if t not in s.index or (t + LAG) not in s.index:
            continue
        edu_val = s[t]
        child_val = s[t + LAG]
        gdp_t_raw = get_wb_val(gdp_df, c, t)
        if np.isnan(edu_val) or np.isnan(child_val):
            continue
        child_edu_rows.append({
            "country": c, "t": t, "edu_t": edu_val,
            "log_gdp_t": np.log(gdp_t_raw) if not np.isnan(gdp_t_raw) and gdp_t_raw > 0 else np.nan,
            "child_edu": child_val,
        })
child_edu_panel = pd.DataFrame(child_edu_rows)

panels = {}
for name, (col, df) in outcomes.items():
    if col == "child_edu":
        panels[name] = filter_panel(child_edu_panel, cohort, CEILING)
    else:
        p = build_panel(edu_annual, df, gdp_df, T_YEARS, LAG, col)
        panels[name] = filter_panel(p, cohort, CEILING)

# ── Sweep by individual outcome year ──────────────────────────────

YEARS = list(range(1985, 2021))

print("\n" + "=" * 100)
print("EDUCATION R² AND RESIDUALIZED GDP R² BY OUTCOME YEAR — ALL FOUR OUTCOMES")
print("=" * 100)
print(f"Entry >= {ENTRY_THRESHOLD}%, ceiling <= {CEILING}%, T=1960-1990, lag={LAG}")
print(f"Clustered SEs by country\n")

# Header
print(f"{'Year':>6}  ", end="")
for name in outcomes:
    print(f"{'Edu R²':>7} {'Res R²':>7} {'p':>6} {'n':>5}  ", end="")
print()
print("-" * 120)

all_results = {name: [] for name in outcomes}

for year in YEARS:
    print(f"  {year}  ", end="")

    for name, (col, _) in outcomes.items():
        panel = panels[name]
        mask = (panel["t"] + LAG) <= year
        sub = panel[mask].copy()

        if len(sub) < 10 or sub["country"].nunique() < 3:
            print(f"{'---':>7} {'---':>7} {'---':>6} {'---':>5}  ", end="")
            all_results[name].append({
                "year": year, "edu_r2": None, "resid_r2": None, "resid_p": None, "n": 0
            })
            continue

        # Education R²
        res_e = clustered_fe("edu_t", col, sub)

        # Residualized GDP
        resid = fe_residualize_gdp(sub)
        res_r = None
        if resid is not None:
            sub_r, _ = resid
            res_r = clustered_fe("gdp_resid", col, sub_r)

        r2_e = res_e["r2"] if res_e else np.nan
        r2_r = res_r["r2"] if res_r else np.nan
        p_r = res_r["pval"] if res_r else np.nan
        n = res_e["n"] if res_e else 0

        def fmtp(v):
            return f"{v:.2f}" if not np.isnan(v) else "---"

        print(f"{fmt_r2(r2_e):>7} {fmt_r2(r2_r):>7} {fmtp(p_r):>6} {n:>5}  ", end="")

        all_results[name].append({
            "year": year,
            "edu_r2": round(r2_e, 4) if not np.isnan(r2_e) else None,
            "resid_r2": round(r2_r, 4) if not np.isnan(r2_r) else None,
            "resid_p": round(p_r, 4) if not np.isnan(p_r) else None,
            "n": n,
        })

    print()

# ── Summary ───────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("SUMMARY: WHEN DOES RESIDUALIZED GDP BECOME SIGNIFICANT (p < 0.05)?")
print("=" * 100)

for name in outcomes:
    significant = [r for r in all_results[name]
                   if r["resid_p"] is not None and r["resid_p"] < 0.05]
    if significant:
        first = significant[0]["year"]
        print(f"  {name:<20} First significant at outcome year {first}  "
              f"(R²={significant[0]['resid_r2']:.3f}, p={significant[0]['resid_p']:.3f})")
    else:
        print(f"  {name:<20} NEVER significant (p < 0.05)")

# ── Checkin ────────────────────────────────────────────────────────

write_checkin("residual_by_outcome_year_all.json", {
    "method": (
        "Residualized GDP R² swept by individual outcome year for all four outcomes. "
        f"Entry >= {ENTRY_THRESHOLD}%, ceiling <= {CEILING}%, T=1960-1990, lag={LAG}. "
        "Clustered SEs by country."
    ),
    "results": all_results,
}, script_path="scripts/robustness/residual_by_outcome_year.py")
