"""
robustness/u5mr_by_edu_level.py
=====================
Does the post-2000 residualized GDP signal for child mortality vary
by education level?

Hypothesis: in more educated societies, the communication environment
itself carries health information (TV content, social norms, health
worker quality). The post-2000 GDP signal should be STRONGEST at the
very lowest education levels (where external tech delivery matters most)
and WEAKEST as education rises (where education already does the work).

Alternatively, if the signal is education-mediated communication, it
should appear at moderate education levels where the information
environment has shifted but individual completion hasn't yet peaked.

Split by parental education bands: 0-10%, 10-20%, 20-30%, 30-50%, 50%+
Test for both U5MR and U1MR.

Entry-cohort design, country FE, lower secondary completion,
T=1960-1990, lag=25. Clustered SEs.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from residualization._shared import (
    load_education, load_wb, interpolate_to_annual, precompute_entry_years,
    build_panel, filter_panel, fe_residualize_gdp, clustered_fe,
    write_checkin,
)

T_YEARS = list(range(1960, 1995, 5))
LAG = 25
COL_NAME = "lower_sec"


def run_analysis(panel, outcome_col, label):
    """Run the full analysis for one outcome. Returns dict of results."""

    print(f"\n{'=' * 95}")
    print(f"  {label}")
    print(f"{'=' * 95}")

    # Education bands: select OBSERVATIONS (not countries) in each band
    bands = [
        ("0-10%",  0, 10),
        ("10-20%", 10, 20),
        ("20-30%", 20, 30),
        ("30-50%", 30, 50),
        ("0-20%",  0, 20),
        ("0-30%",  0, 30),
        ("0-50%",  0, 50),
    ]

    print(f"\n  {'Band':<10} {'Period':<15} {'Edu R²':>7} {'Res R²':>7} {'p':>7} {'n':>5} {'Ctry':>5}")
    print("  " + "-" * 65)

    results = {}

    for band_label, lo, hi in bands:
        # Select observations where parental education is in this band
        mask_band = (panel["edu_t"] >= lo) & (panel["edu_t"] < hi)
        sub_band = panel[mask_band].copy()

        if len(sub_band) < 10:
            continue

        for period_label, period_mask in [
            ("All years",   pd.Series(True, index=sub_band.index)),
            ("Before 2000", (sub_band["t"] + LAG) < 2000),
            ("After 2000",  (sub_band["t"] + LAG) >= 2000),
        ]:
            sub = sub_band[period_mask].copy()

            if len(sub) < 10 or sub["country"].nunique() < 3:
                print(f"  {band_label:<10} {period_label:<15} {'---':>7} {'---':>7} {'---':>7} {'---':>5} {'---':>5}")
                continue

            res_e = clustered_fe("edu_t", outcome_col, sub)

            resid = fe_residualize_gdp(sub)
            res_r = None
            if resid is not None:
                sub_r, _ = resid
                res_r = clustered_fe("gdp_resid", outcome_col, sub_r)

            r2_e = f"{res_e['r2']:.3f}" if res_e else "---"
            r2_r = f"{res_r['r2']:.3f}" if res_r else "---"
            p_r = f"{res_r['pval']:.4f}" if res_r else "---"
            n = res_e['n'] if res_e else 0
            ctry = res_e['countries'] if res_e else 0

            print(f"  {band_label:<10} {period_label:<15} {r2_e:>7} {r2_r:>7} {p_r:>7} {n:>5} {ctry:>5}")

            key = f"{band_label}|{period_label}"
            results[key] = {
                "edu_r2": res_e["r2"] if res_e else None,
                "resid_r2": res_r["r2"] if res_r else None,
                "resid_p": res_r["pval"] if res_r else None,
                "n": n,
                "countries": ctry,
            }

        print()

    return results


# ── Load data ─────────────────────────────────────────────────────

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
u5mr_df = load_wb("child_mortality_u5.csv")
u1mr_df = load_wb("infant_mortality_u1.csv")

edu_annual = interpolate_to_annual(edu_raw, COL_NAME)

# Build panels WITHOUT entry-cohort filtering (we'll filter by band instead)
panel_u5 = build_panel(edu_annual, u5mr_df, gdp_df, T_YEARS, LAG, "mortality")
panel_u1 = build_panel(edu_annual, u1mr_df, gdp_df, T_YEARS, LAG, "mortality")

# Drop missing GDP (needed for residualization)
panel_u5 = panel_u5.dropna(subset=["log_gdp_t"])
panel_u1 = panel_u1.dropna(subset=["log_gdp_t"])

print(f"U5MR panel: {len(panel_u5)} obs, {panel_u5['country'].nunique()} countries")
print(f"U1MR panel: {len(panel_u1)} obs, {panel_u1['country'].nunique()} countries")

# ── Distribution of education levels ──────────────────────────────

print(f"\n  Education level distribution (U5MR panel):")
for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100)]:
    n = ((panel_u5["edu_t"] >= lo) & (panel_u5["edu_t"] < hi)).sum()
    ctry = panel_u5[(panel_u5["edu_t"] >= lo) & (panel_u5["edu_t"] < hi)]["country"].nunique()
    print(f"    {lo}-{hi}%: {n} obs, {ctry} countries")

# ── Run analyses ──────────────────────────────────────────────────

results_u5 = run_analysis(panel_u5, "mortality", "U5MR (UNDER-5 MORTALITY) BY EDUCATION BAND")
results_u1 = run_analysis(panel_u1, "mortality", "U1MR (INFANT MORTALITY) BY EDUCATION BAND")

# ── Checkin ────────────────────────────────────────────────────────

write_checkin("u5mr_by_edu_level.json", {
    "u5mr": results_u5,
    "u1mr": results_u1,
}, script_path="scripts/robustness/u5mr_by_edu_level.py")
