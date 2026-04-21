"""
robustness/completion_vs_test_scores.py
=======================================
Horse race: does completion or test scores better predict
life expectancy, TFR, and child mortality?

CONTEXT
-------
Hanushek & Woessmann (2008, 2015) argue that cognitive skills
(test scores), not years of schooling or completion, drive
economic growth. Their outcome is GDP growth. This paper's
outcomes are human development: LE, TFR, U5MR.

This script tests whether Hanushek's preferred measure (test
scores) or ours (lower secondary completion) better predicts
development outcomes on the OVERLAP SAMPLE — countries where
both measures exist.

DATA
----
Test scores: World Bank Harmonized Learning Outcomes (HLO)
database (Angrist et al. 2021). Secondary level, nationally
representative, averaged across subjects per country-year.

Coverage gap is itself a finding:
  - Completion: 185 countries, 1950-2015
  - Test scores: ~100 countries, 2000-2015

METHOD
------
1. Cross-sectional comparison (no lag): test score at T vs
   completion at T, predicting outcome at T. Gives test scores
   their best shot — no lag requirement.

2. Short-lag panel (lag=10, 15): test score at T → outcome at
   T+lag. The 25-year specification cannot be run because test
   data only goes back to 2000.

3. On the overlap sample, run the full horse race: education
   alone, test scores alone, both together.

Country FE throughout. Clustered SEs.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from residualization._shared import (
    load_education, load_wb, interpolate_to_annual,
    build_panel, fe_r2, clustered_fe, get_wb_val,
    write_checkin, DATA, NAME_MAP,
)
from _shared import fmt_r2

# ── HLO country name → WB lowercase name ───────────────────────────

HLO_NAME_MAP = {
    "czech republic": "czechia",
    "egypt": "egypt, arab rep.",
    "hong kong, sar china": "hong kong sar, china",
    "iran, islamic republic of": "iran, islamic rep.",
    "korea\u00ac\u2020(south)": "korea, rep.",
    "kyrgyzstan": "kyrgyz republic",
    "macao, sar china": "macao sar, china",
    "macedonia, republic of": "north macedonia",
    "serbia and montenegro": "serbia",
    "slovakia": "slovak republic",
    "syrian arab republic\u00ac\u2020(syria)": "syrian arab republic",
    "taiwan, republic of china": "taiwan",
    "turkey": "turkiye",
    "united states of america": "united states",
}


def load_hlo():
    """
    Load HLO test scores: secondary level, nationally representative,
    averaged across subjects per country-year.
    Returns DataFrame in WB wide format (lowercase country index, year columns).
    """
    raw = pd.read_csv(os.path.join(DATA, "hlo_raw.csv"))
    sec = raw[(raw["level"] == "sec") & (raw["n_res"] == 1)].copy()

    # Normalise country names to WB lowercase
    sec["country"] = sec["country"].str.lower()
    sec["country"] = sec["country"].replace(HLO_NAME_MAP)

    # Average across subjects per country-year
    avg = sec.groupby(["country", "year"])["hlo"].mean().reset_index()

    # Pivot to wide format
    wide = avg.pivot(index="country", columns="year", values="hlo")
    wide.columns = [str(int(c)) for c in wide.columns]
    for c in wide.columns:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")
    return wide


# ── Load data ───────────────────────────────────────────────────────

print("Loading data...")

edu_raw = load_education("completion_both_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
le_df = load_wb("life_expectancy_years.csv")
tfr_df = load_wb("children_per_woman_total_fertility.csv")
u5mr_df = load_wb("child_mortality_u5.csv")
hlo_df = load_hlo()

edu_annual = interpolate_to_annual(edu_raw, "lower_sec")

print(f"HLO test scores: {len(hlo_df)} countries, years {list(hlo_df.columns)}")

OUTCOMES = {
    "le": ("Life expectancy", le_df),
    "tfr": ("TFR", tfr_df),
    "u5mr": ("U5MR", u5mr_df),
}


# ── Helper: add test scores to panel ────────────────────────────────

def add_test_scores(panel, hlo_wide):
    """Add test_score_t column to panel by matching country+year."""
    scores = []
    for _, row in panel.iterrows():
        scores.append(get_wb_val(hlo_wide, row["country"], row["t"]))
    panel = panel.copy()
    panel["test_score_t"] = scores
    return panel


def overlap_panel(panel):
    """Restrict panel to observations with both completion and test scores."""
    return panel.dropna(subset=["edu_t", "test_score_t"]).copy()


# ── Analysis 1: Coverage comparison ─────────────────────────────────

print(f"\n{'=' * 80}")
print("COVERAGE COMPARISON")
print(f"{'=' * 80}")

# Full completion panel (same years as HLO)
T_YEARS_HLO = [2000, 2003, 2006, 2007, 2009, 2011, 2012, 2015]

panel_le_full = build_panel(edu_annual, le_df, gdp_df, T_YEARS_HLO, 0, "outcome")
panel_le_full = add_test_scores(panel_le_full, hlo_df)
panel_le_overlap = overlap_panel(panel_le_full)

n_edu_countries = panel_le_full["country"].nunique()
n_overlap_countries = panel_le_overlap["country"].nunique()
n_edu_obs = len(panel_le_full)
n_overlap_obs = len(panel_le_overlap)

print(f"  Completion data:  {n_edu_countries} countries, {n_edu_obs} obs")
print(f"  Overlap sample:   {n_overlap_countries} countries, {n_overlap_obs} obs")
print(f"  Lost:             {n_edu_countries - n_overlap_countries} countries, "
      f"{n_edu_obs - n_overlap_obs} obs")

coverage = {
    "completion_countries": n_edu_countries,
    "completion_obs": n_edu_obs,
    "overlap_countries": n_overlap_countries,
    "overlap_obs": n_overlap_obs,
}

# ── Analysis 2: Cross-sectional horse race (lag=0) ──────────────────

print(f"\n{'=' * 80}")
print("CROSS-SECTIONAL HORSE RACE (lag=0, contemporary outcomes)")
print("Gives test scores their best shot — no lag, no data loss")
print(f"{'=' * 80}")

print(f"\n  {'Outcome':<8} {'Edu R²':>7} {'Test R²':>7} {'Edu β':>8} {'Test β':>8} "
      f"{'Edu p':>7} {'Test p':>7} {'n':>5} {'Ctry':>5}")
print("  " + "-" * 75)

cross_sectional = {}

for key, (label, outcome_df) in OUTCOMES.items():
    panel = build_panel(edu_annual, outcome_df, gdp_df, T_YEARS_HLO, 0, "outcome")
    panel = add_test_scores(panel, hlo_df)
    ol = overlap_panel(panel)

    if len(ol) < 10:
        print(f"  {label:<8} insufficient data")
        continue

    res_edu = clustered_fe("edu_t", "outcome", ol)
    res_test = clustered_fe("test_score_t", "outcome", ol)

    def fmtp(v): return f"{v:.4f}" if v is not None else "n/a"

    print(f"  {label:<8} "
          f"{fmt_r2(res_edu['r2']) if res_edu else 'n/a':>7} "
          f"{fmt_r2(res_test['r2']) if res_test else 'n/a':>7} "
          f"{fmt_r2(res_edu['beta']) if res_edu else 'n/a':>8} "
          f"{fmt_r2(res_test['beta']) if res_test else 'n/a':>8} "
          f"{fmtp(res_edu['pval']) if res_edu else 'n/a':>7} "
          f"{fmtp(res_test['pval']) if res_test else 'n/a':>7} "
          f"{res_edu['n'] if res_edu else 0:>5} "
          f"{res_edu['countries'] if res_edu else 0:>5}")

    cross_sectional[key] = {
        "edu": {k: round(v, 4) if isinstance(v, float) else v
                for k, v in res_edu.items()} if res_edu else None,
        "test": {k: round(v, 4) if isinstance(v, float) else v
                 for k, v in res_test.items()} if res_test else None,
    }


# ── Analysis 3: Short-lag panel horse race ───────────────────────────

short_lag_results = {}

for lag in [10, 15]:
    print(f"\n{'=' * 80}")
    print(f"PANEL HORSE RACE (lag={lag} years, country FE)")
    print(f"Education or test scores at T → outcome at T+{lag}")
    print(f"{'=' * 80}")

    print(f"\n  {'Outcome':<8} {'Edu R²':>7} {'Test R²':>7} {'Edu β':>8} {'Test β':>8} "
          f"{'Edu p':>7} {'Test p':>7} {'n':>5} {'Ctry':>5}")
    print("  " + "-" * 75)

    lag_results = {}

    for key, (label, outcome_df) in OUTCOMES.items():
        panel = build_panel(edu_annual, outcome_df, gdp_df, T_YEARS_HLO, lag, "outcome")
        panel = add_test_scores(panel, hlo_df)
        ol = overlap_panel(panel)

        if len(ol) < 10 or ol["country"].nunique() < 3:
            print(f"  {label:<8} insufficient data (n={len(ol)}, "
                  f"countries={ol['country'].nunique() if len(ol) > 0 else 0})")
            lag_results[key] = None
            continue

        res_edu = clustered_fe("edu_t", "outcome", ol)
        res_test = clustered_fe("test_score_t", "outcome", ol)

        def fmtp(v): return f"{v:.4f}" if v is not None else "n/a"

        print(f"  {label:<8} "
              f"{fmt_r2(res_edu['r2']) if res_edu else 'n/a':>7} "
              f"{fmt_r2(res_test['r2']) if res_test else 'n/a':>7} "
              f"{fmt_r2(res_edu['beta']) if res_edu else 'n/a':>8} "
              f"{fmt_r2(res_test['beta']) if res_test else 'n/a':>8} "
              f"{fmtp(res_edu['pval']) if res_edu else 'n/a':>7} "
              f"{fmtp(res_test['pval']) if res_test else 'n/a':>7} "
              f"{res_edu['n'] if res_edu else 0:>5} "
              f"{res_edu['countries'] if res_edu else 0:>5}")

        lag_results[key] = {
            "edu": {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in res_edu.items()} if res_edu else None,
            "test": {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in res_test.items()} if res_test else None,
        }

    short_lag_results[str(lag)] = lag_results


# ── Analysis 4: Full-sample comparison ───────────────────────────────
# Show what happens when we run the same spec on all 185 countries
# vs only the ~100 with test scores

print(f"\n{'=' * 80}")
print("FULL SAMPLE vs OVERLAP: education-only R² (lag=0)")
print("Does restricting to HLO-available countries change the education result?")
print(f"{'=' * 80}")

print(f"\n  {'Outcome':<8} {'Full R²':>8} {'Full n':>6} {'Overlap R²':>10} {'Overlap n':>9}")
print("  " + "-" * 50)

sample_comparison = {}

for key, (label, outcome_df) in OUTCOMES.items():
    panel = build_panel(edu_annual, outcome_df, gdp_df, T_YEARS_HLO, 0, "outcome")
    panel_ts = add_test_scores(panel, hlo_df)
    ol = overlap_panel(panel_ts)

    r2_full, n_full, _ = fe_r2("edu_t", "outcome", panel)
    r2_ol, n_ol, _ = fe_r2("edu_t", "outcome", ol)

    print(f"  {label:<8} {fmt_r2(r2_full):>8} {n_full:>6} {fmt_r2(r2_ol):>10} {n_ol:>9}")

    sample_comparison[key] = {
        "full_r2": round(r2_full, 4) if not np.isnan(r2_full) else None,
        "full_n": n_full,
        "overlap_r2": round(r2_ol, 4) if not np.isnan(r2_ol) else None,
        "overlap_n": n_ol,
    }


# ── Checkin ──────────────────────────────────────────────────────────

checkin = {
    "method": (
        "Horse race between lower secondary completion (WCDE v3) and "
        "harmonized test scores (HLO, Angrist et al. 2021) as predictors "
        "of LE, TFR, U5MR. Country FE, clustered SEs. Cross-sectional "
        "(lag=0) and short-lag (10, 15yr) specifications. Overlap sample "
        "only (countries with both measures). 25-year lag impossible: "
        "test data begins 2000."
    ),
    "coverage": coverage,
    "cross_sectional": cross_sectional,
    "short_lag": short_lag_results,
    "sample_comparison": sample_comparison,
}

write_checkin("completion_vs_test_scores.json", checkin,
              "scripts/robustness/completion_vs_test_scores.py")
