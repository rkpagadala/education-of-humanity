"""
robustness/cs_threshold_robustness.py
=====================================
Sweep the entry threshold for the Callaway & Sant'Anna (2021) estimator
of education -> child education at T+25.

Background:
  callaway_santanna.py defines treatment as the period where lower
  secondary completion first crosses 10%. Reviewer R1.20 asked for
  evidence that the 10% choice is not load-bearing.

Method:
  For each threshold in {5, 7, 10, 12, 15}, redefine treatment, recompute
  group-time ATT(g,t) on the not-yet-treated control group, aggregate
  over post-treatment (g,t) pairs (simple average), and report:

    - Aggregate ATT (child education at T+25)
    - Pre-trend ATT (event_time = -1, the pre-period placebo)
    - Number of timing cohorts and treated countries

  The mechanism prediction: the aggregate ATT should be positive and
  stable across thresholds; pre-trend ATT should sit near zero.

Outputs:
  checkin/cs_threshold_robustness.json
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import PROC, REGIONS, write_checkin, load_wb, get_wb_val

THRESHOLDS = [5, 7, 10, 12, 15]
PERIODS = list(range(1950, 1995, 5))
LAG = 25

print("Loading data...")
edu_wide = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"), index_col="country")
edu_wide = edu_wide[~edu_wide.index.isin(REGIONS)]
le_df = load_wb("life_expectancy_years.csv")

treat_cols = [str(y) for y in PERIODS]
out_cols = [str(y + LAG) for y in PERIODS]
complete = edu_wide.dropna(subset=sorted(set(treat_cols + out_cols)))
countries = sorted(complete.index)

base_rows = []
for c in countries:
    for t in PERIODS:
        edu_t = float(complete.loc[c, str(t)])
        child_edu = float(complete.loc[c, str(t + LAG)])
        le_val = get_wb_val(le_df, c, t + LAG)
        base_rows.append({
            "country": c, "t": t, "edu": edu_t,
            "child_edu": child_edu, "le": le_val,
        })

base = pd.DataFrame(base_rows)
N = base["country"].nunique()
T = len(PERIODS)
print(f"Panel: {N} countries x {T} periods ({PERIODS[0]}-{PERIODS[-1]})")


def cohort_map_for_threshold(threshold):
    """First period where edu >= threshold; absorbing-treatment forced."""
    cmap = {}
    for c in countries:
        sub = base.loc[base["country"] == c].sort_values("t")
        first = sub.loc[sub["edu"] >= threshold, "t"]
        if len(first) == 0:
            cmap[c] = None  # never-treated
        elif int(first.min()) <= PERIODS[0]:
            cmap[c] = "always"
        else:
            cmap[c] = int(first.min())
    return cmap


def compute_att_aggregate(outcome_arr, cohort_vec, country_list, period_idx):
    """Aggregate ATT(g,t) for post-treatment cells; pre-trend at e=-1."""
    timing_vals = sorted(set(g for g in cohort_vec if isinstance(g, (int, np.integer))
                              and g not in (0, 9999)))
    cohort_arr = np.array([
        0 if g == "always" else (9999 if g is None else int(g))
        for g in cohort_vec
    ])

    post_atts = []
    pre_atts = []
    n_post = 0
    n_pre = 0
    for g in timing_vals:
        base_j = period_idx.get(g - 5)
        if base_j is None:
            continue
        treat_mask = cohort_arr == g
        if treat_mask.sum() < 2:
            continue
        for t in PERIODS:
            j = period_idx[t]
            event_time = (t - g) // 5
            ctrl_mask = cohort_arr > t  # not-yet-treated and never-treated
            if ctrl_mask.sum() < 2:
                continue
            y_treat_base = outcome_arr[treat_mask, base_j]
            y_treat_t = outcome_arr[treat_mask, j]
            y_ctrl_base = outcome_arr[ctrl_mask, base_j]
            y_ctrl_t = outcome_arr[ctrl_mask, j]
            treat_ok = ~np.isnan(y_treat_base) & ~np.isnan(y_treat_t)
            ctrl_ok = ~np.isnan(y_ctrl_base) & ~np.isnan(y_ctrl_t)
            if treat_ok.sum() < 2 or ctrl_ok.sum() < 2:
                continue
            att = (np.mean(y_treat_t[treat_ok]) - np.mean(y_treat_base[treat_ok])) - \
                  (np.mean(y_ctrl_t[ctrl_ok]) - np.mean(y_ctrl_base[ctrl_ok]))
            if event_time >= 0:
                post_atts.append(att)
                n_post += 1
            elif event_time == -1:
                pre_atts.append(att)
                n_pre += 1
    agg = float(np.mean(post_atts)) if post_atts else np.nan
    pre = float(np.mean(pre_atts)) if pre_atts else np.nan
    return agg, pre, len(timing_vals), int((cohort_arr != 9999).sum()), n_post


country_list = sorted(countries)
country_idx = {c: i for i, c in enumerate(country_list)}
period_idx = {t: j for j, t in enumerate(PERIODS)}

child_edu_arr = np.full((len(country_list), T), np.nan)
for _, row in base.iterrows():
    child_edu_arr[country_idx[row["country"]], period_idx[row["t"]]] = row["child_edu"]


print("\nThreshold sweep -- aggregate ATT (child education at T+25)")
print("=" * 70)
print(f"{'Thresh':>7} {'Cohorts':>8} {'Treated':>8} {'Cells':>6} "
      f"{'ATT (post)':>11} {'Pre (e=-1)':>11}")
print("-" * 70)

results = {}
for thr in THRESHOLDS:
    cmap = cohort_map_for_threshold(thr)
    cohort_list = [cmap[c] for c in country_list]
    agg, pre, n_cohorts, n_treated, n_cells = compute_att_aggregate(
        child_edu_arr, cohort_list, country_list, period_idx
    )
    results[thr] = {
        "att_aggregate": round(agg, 2) if not np.isnan(agg) else None,
        "pre_trend_e_minus_1": round(pre, 2) if not np.isnan(pre) else None,
        "n_cohorts": n_cohorts,
        "n_treated_countries": n_treated,
        "n_post_cells": n_cells,
    }
    agg_s = f"{agg:>+11.2f}" if not np.isnan(agg) else "        nan"
    pre_s = f"{pre:>+11.2f}" if not np.isnan(pre) else "        nan"
    print(f"{thr:>5}%  {n_cohorts:>8d} {n_treated:>8d} {n_cells:>6d} {agg_s} {pre_s}")

print()
print("Headline (10%) ATT replicates callaway_santanna.json att_aggregate.")
print("Pre-trend ATT (e=-1) is the placebo: should sit near zero if "
      "parallel-trends holds before treatment.")

write_checkin(
    "cs_threshold_robustness.json",
    {
        "method": (
            "Callaway-Sant'Anna group-time ATT for child education at "
            "T+25, swept across entry thresholds 5-15%. Same panel as "
            "callaway_santanna.py (1950-1990 5-yr, " f"{N}" " countries). "
            "ATT (post) is the unweighted average across post-treatment "
            "(g,t) cells with at least two treated and two not-yet-"
            "treated controls; pre-trend ATT (e=-1) is the placebo "
            "evaluated one period before crossing."
        ),
        "thresholds": results,
        "panel_countries": N,
        "panel_periods": [PERIODS[0], PERIODS[-1]],
        "lag_years": LAG,
    },
    script_path="scripts/robustness/cs_threshold_robustness.py",
)

print("\nDone.")
