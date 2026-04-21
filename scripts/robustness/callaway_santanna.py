"""
robustness/callaway_santanna.py
===============================
Callaway & Sant'Anna (2021) heterogeneity-robust estimator for the
education → development effect under staggered adoption.

The standard TWFE estimator collapses (β = 0.08, p = 0.08) because it
uses already-transitioned countries as counterfactuals for newly-
expanding ones (see goodman_bacon_decomposition.py). This script
implements the Callaway-Sant'Anna group-time ATT estimator, which
avoids the negative-weighting pathology by restricting control groups
to NOT-YET-TREATED units.

Method:
  1. Define treatment cohorts: group g = first 5-year period where
     country's lower secondary completion ≥ 10% (absorbing treatment).
  2. For each (cohort g, period t) with t ≥ g:
       ATT(g,t) = [Ȳ_t(g) - Ȳ_{g-5}(g)] - [Ȳ_t(C) - Ȳ_{g-5}(C)]
     where C = countries not yet treated at t (g_i > t or never-treated).
  3. Aggregate:
       - Simple ATT: unweighted average across post-treatment (g,t) pairs
       - Dynamic ATT(e): average by event time e = t - g (event study)
  4. Inference: clustered bootstrap by country (500 replications).

Outcomes:
  - Primary: child education at T+25 (matches Table A1)
  - Secondary: life expectancy at T+25

References:
  Callaway, B. & Sant'Anna, P. (2021). Difference-in-Differences with
    Multiple Time Periods. Journal of Econometrics, 225(2), 200–230.

Data:
  Education: WCDE v3, lower secondary completion, both sexes, age 20–24
  Life expectancy: World Bank WDI (SP.DYN.LE00.IN)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import PROC, DATA, REGIONS, write_checkin, load_wb, get_wb_val, NAME_MAP

# ── Constants ────────────────────────────────────────────────────────
THRESHOLD = 10
PERIODS = list(range(1950, 1995, 5))
LAG = 25
N_BOOT = 500
RNG_SEED = 42
FIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "figures")

# ── Load data ────────────────────────────────────────────────────────
print("Loading data...")
edu_wide = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"), index_col="country")
edu_wide = edu_wide[~edu_wide.index.isin(REGIONS)]
le_df = load_wb("life_expectancy_years.csv")

treat_cols = [str(y) for y in PERIODS]
out_cols = [str(y + LAG) for y in PERIODS]
complete = edu_wide.dropna(subset=sorted(set(treat_cols + out_cols)))
countries = sorted(complete.index)

rows = []
for c in countries:
    for t in PERIODS:
        edu_t = float(complete.loc[c, str(t)])
        child_edu = float(complete.loc[c, str(t + LAG)])
        le_val = get_wb_val(le_df, c, t + LAG)
        rows.append({
            "country": c, "t": t, "edu": edu_t,
            "d": int(edu_t >= THRESHOLD),
            "child_edu": child_edu, "le": le_val,
        })

panel = pd.DataFrame(rows)
N = panel["country"].nunique()
T = len(PERIODS)

# Force absorbing treatment
for c in countries:
    d = panel.loc[panel["country"] == c, "d"].values.copy()
    if np.any(np.diff(d) < 0):
        first_on = np.argmax(d == 1)
        if d[first_on] == 1:
            d[first_on:] = 1
            panel.loc[panel["country"] == c, "d"] = d

print(f"Panel: {N} countries × {T} periods ({PERIODS[0]}–{PERIODS[-1]})")

# ── Treatment cohorts ────────────────────────────────────────────────
cohort_map = {}
for c in countries:
    dv = panel.loc[panel["country"] == c].sort_values("t")
    tr = dv[dv["d"] == 1]["t"]
    if len(tr) == 0:
        cohort_map[c] = None
    elif len(tr) == T:
        cohort_map[c] = "always"
    else:
        cohort_map[c] = int(tr.min())

panel["g"] = panel["country"].map(cohort_map)

timing_vals = sorted(set(g for g in cohort_map.values() if g not in (None, "always")))
always_ids = [c for c, g in cohort_map.items() if g == "always"]
never_ids = [c for c, g in cohort_map.items() if g is None]
timing_ids = [c for c, g in cohort_map.items() if g not in (None, "always")]

print(f"  Always treated: {len(always_ids)} | Timing: {len(timing_ids)} | Never treated: {len(never_ids)}")
for g in timing_vals:
    n_g = sum(1 for _, gg in cohort_map.items() if gg == g)
    print(f"    Cohort g={g}: {n_g} countries")


# ── Precompute country-level outcome means per period ────────────────
# For fast bootstrap: store country × period arrays, then resample rows
country_list = sorted(countries)
country_idx = {c: i for i, c in enumerate(country_list)}
period_idx = {t: j for j, t in enumerate(PERIODS)}
cohort_arr = np.array([cohort_map[c] if isinstance(cohort_map[c], int)
                        else (0 if cohort_map[c] == "always" else 9999)
                        for c in country_list])
# 0 = always, 9999 = never, else = timing year

# Outcome arrays: shape (N_countries, T_periods)
child_edu_arr = np.full((len(country_list), T), np.nan)
le_arr = np.full((len(country_list), T), np.nan)
for _, row in panel.iterrows():
    i = country_idx[row["country"]]
    j = period_idx[row["t"]]
    child_edu_arr[i, j] = row["child_edu"]
    le_arr[i, j] = row["le"]


# ── Vectorized ATT(g,t) computation ─────────────────────────────────
def compute_all_att_gt_fast(outcome_arr, cohort_vec, periods=PERIODS):
    """
    Compute ATT(g,t) for all valid (g,t) pairs using vectorized operations.

    Parameters:
      outcome_arr: (N_countries, T_periods) array of outcomes
      cohort_vec: (N_countries,) array of cohort timing (0=always, 9999=never)

    Returns list of dicts with g, t, event_time, att, n_treat, n_ctrl.
    """
    results = []
    for g in timing_vals:
        base_j = period_idx.get(g - 5)
        if base_j is None:
            continue

        # Treatment group mask
        treat_mask = cohort_vec == g
        n_treat = treat_mask.sum()
        if n_treat < 2:
            continue

        for t in periods:
            j = period_idx[t]
            event_time = (t - g) // 5

            # Control: not-yet-treated at t (cohort > t) or never-treated (9999)
            ctrl_mask = (cohort_vec > t)  # includes 9999 (never)
            n_ctrl = ctrl_mask.sum()
            if n_ctrl < 2:
                continue

            # Treatment group: outcome change from base to t
            y_treat_base = outcome_arr[treat_mask, base_j]
            y_treat_t = outcome_arr[treat_mask, j]

            # Control: outcome change from base to t
            y_ctrl_base = outcome_arr[ctrl_mask, base_j]
            y_ctrl_t = outcome_arr[ctrl_mask, j]

            # Drop NaN
            treat_ok = ~np.isnan(y_treat_base) & ~np.isnan(y_treat_t)
            ctrl_ok = ~np.isnan(y_ctrl_base) & ~np.isnan(y_ctrl_t)

            if treat_ok.sum() < 2 or ctrl_ok.sum() < 2:
                continue

            treat_change = np.mean(y_treat_t[treat_ok]) - np.mean(y_treat_base[treat_ok])
            ctrl_change = np.mean(y_ctrl_t[ctrl_ok]) - np.mean(y_ctrl_base[ctrl_ok])
            att = treat_change - ctrl_change

            results.append({
                "g": g, "t": t, "event_time": event_time,
                "att": att, "n_treat": int(treat_ok.sum()), "n_ctrl": int(ctrl_ok.sum()),
            })

    return results


def aggregate_results(att_list):
    """Aggregate ATT results: overall and by event time."""
    if not att_list:
        return np.nan, {}

    df = pd.DataFrame(att_list)
    post = df[df["event_time"] >= 0]
    att_agg = post["att"].mean() if len(post) > 0 else np.nan

    dynamic = {}
    for e, grp in df.groupby("event_time"):
        dynamic[int(e)] = {"att": grp["att"].mean(), "n_cells": len(grp)}

    return att_agg, dynamic


def bootstrap_cs(outcome_arr, cohort_vec, n_boot=N_BOOT, seed=RNG_SEED):
    """
    Clustered bootstrap: resample countries (with replacement),
    recompute all ATT(g,t) and aggregates.
    """
    rng = np.random.RandomState(seed)
    n_countries = len(cohort_vec)

    # Point estimates
    att_list = compute_all_att_gt_fast(outcome_arr, cohort_vec)
    att_agg, att_dyn = aggregate_results(att_list)

    # Bootstrap
    boot_aggs = []
    boot_dyns = {e: [] for e in att_dyn}

    for b in range(n_boot):
        # Resample country indices
        boot_idx = rng.choice(n_countries, size=n_countries, replace=True)
        boot_outcome = outcome_arr[boot_idx]
        boot_cohort = cohort_vec[boot_idx]

        boot_att = compute_all_att_gt_fast(boot_outcome, boot_cohort)
        boot_att_agg, boot_att_dyn = aggregate_results(boot_att)

        if not np.isnan(boot_att_agg):
            boot_aggs.append(boot_att_agg)

        for e in boot_dyns:
            if e in boot_att_dyn:
                boot_dyns[e].append(boot_att_dyn[e]["att"])

    # Confidence intervals
    se = np.std(boot_aggs) if boot_aggs else np.nan
    ci_lo = np.percentile(boot_aggs, 2.5) if len(boot_aggs) >= 20 else np.nan
    ci_hi = np.percentile(boot_aggs, 97.5) if len(boot_aggs) >= 20 else np.nan

    dyn_ci = {}
    for e in sorted(att_dyn):
        vals = boot_dyns.get(e, [])
        vals = [v for v in vals if not np.isnan(v)]
        dyn_ci[e] = {
            "att": att_dyn[e]["att"],
            "n_cells": att_dyn[e]["n_cells"],
            "se": np.std(vals) if len(vals) >= 20 else np.nan,
            "ci_lo": np.percentile(vals, 2.5) if len(vals) >= 20 else np.nan,
            "ci_hi": np.percentile(vals, 97.5) if len(vals) >= 20 else np.nan,
        }

    return {
        "att_gt": pd.DataFrame(att_list),
        "att_agg": att_agg,
        "att_se": se,
        "att_ci": (ci_lo, ci_hi),
        "att_dynamic": dyn_ci,
    }


# ── Run for child education ─────────────────────────────────────────
print(f"\n{'=' * 70}")
print("CALLAWAY-SANT'ANNA ESTIMATOR: Child Education (T+25)")
print(f"{'=' * 70}")
print(f"Treatment: education ≥ {THRESHOLD}% | Control: not-yet-treated")
print(f"Bootstrap: {N_BOOT} replications, clustered by country\n")

result_edu = bootstrap_cs(child_edu_arr, cohort_arr)

att_gt_edu = result_edu["att_gt"]
print(f"Group-time cells: {len(att_gt_edu)} (g,t) pairs")
print(f"  Post-treatment: {(att_gt_edu['event_time'] >= 0).sum()}")
print(f"  Pre-treatment:  {(att_gt_edu['event_time'] < 0).sum()}")

print(f"\n  AGGREGATE ATT = {result_edu['att_agg']:.2f}")
print(f"  SE = {result_edu['att_se']:.2f}")
print(f"  95% CI = [{result_edu['att_ci'][0]:.2f}, {result_edu['att_ci'][1]:.2f}]")

# Compare to TWFE
panel["d_dm1"] = panel["d"] - panel.groupby("country")["d"].transform("mean")
panel["y_dm1"] = panel["child_edu"] - panel.groupby("country")["child_edu"].transform("mean")
panel["d_dm2"] = panel["d_dm1"] - panel.groupby("t")["d_dm1"].transform("mean")
panel["y_dm2"] = panel["y_dm1"] - panel.groupby("t")["y_dm1"].transform("mean")
beta_twfe = (panel["d_dm2"] * panel["y_dm2"]).sum() / (panel["d_dm2"] ** 2).sum()

print(f"\n  Comparison:")
print(f"    TWFE β (binary):         {beta_twfe:.2f}")
print(f"    Callaway-Sant'Anna ATT:  {result_edu['att_agg']:.2f}")
if beta_twfe > 0:
    print(f"    Ratio (CS / TWFE):       {result_edu['att_agg'] / beta_twfe:.2f}×")

# Event study
print(f"\n  Event Study (child education):")
print(f"  {'Event':>6} {'Years':>6} {'ATT':>8} {'SE':>7} {'95% CI':>22} {'Cells':>6}")
print(f"  {'-' * 60}")
for e in sorted(result_edu["att_dynamic"]):
    d = result_edu["att_dynamic"][e]
    sig = " *" if not np.isnan(d["ci_lo"]) and (d["ci_lo"] > 0 or d["ci_hi"] < 0) else ""
    print(f"  {e:>6} {e*5:>+5}yr {d['att']:>8.2f} {d['se']:>7.2f}"
          f"  [{d['ci_lo']:>8.2f}, {d['ci_hi']:>8.2f}] {d['n_cells']:>6}{sig}")


# ── Run for life expectancy ──────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"CALLAWAY-SANT'ANNA ESTIMATOR: Life Expectancy (T+25)")
print(f"{'=' * 70}")

result_le = bootstrap_cs(le_arr, cohort_arr)

att_gt_le = result_le["att_gt"]
n_le_cells = len(att_gt_le)
print(f"Group-time cells: {n_le_cells} (g,t) pairs")

print(f"\n  AGGREGATE ATT = {result_le['att_agg']:.2f} years")
print(f"  SE = {result_le['att_se']:.2f}")
print(f"  95% CI = [{result_le['att_ci'][0]:.2f}, {result_le['att_ci'][1]:.2f}]")

# Event study
print(f"\n  Event Study (life expectancy):")
print(f"  {'Event':>6} {'Years':>6} {'ATT':>8} {'SE':>7} {'95% CI':>22} {'Cells':>6}")
print(f"  {'-' * 60}")
for e in sorted(result_le["att_dynamic"]):
    d = result_le["att_dynamic"][e]
    sig = " *" if not np.isnan(d["ci_lo"]) and (d["ci_lo"] > 0 or d["ci_hi"] < 0) else ""
    print(f"  {e:>6} {e*5:>+5}yr {d['att']:>8.2f} {d['se']:>7.2f}"
          f"  [{d['ci_lo']:>8.2f}, {d['ci_hi']:>8.2f}] {d['n_cells']:>6}{sig}")


# ── Figure: Event study ──────────────────────────────────────────────
os.makedirs(FIG_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, result, title, ylabel in [
    (axes[0], result_edu, "Child Education (T+25)", "ATT (percentage points)"),
    (axes[1], result_le, "Life Expectancy (T+25)", "ATT (years)"),
]:
    dyn = result["att_dynamic"]
    events = sorted(dyn.keys())
    atts = [dyn[e]["att"] for e in events]
    ci_lo = [dyn[e]["ci_lo"] for e in events]
    ci_hi = [dyn[e]["ci_hi"] for e in events]
    event_years = [e * 5 for e in events]

    ax.fill_between(event_years, ci_lo, ci_hi, alpha=0.2, color="#2166ac")
    ax.plot(event_years, atts, "o-", color="#2166ac", markersize=5, linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.axvline(-2.5, color="black", linestyle="--", linewidth=0.8,
               label="Treatment onset")

    post_atts = [dyn[e]["att"] for e in events if e >= 0]
    if post_atts:
        agg = result["att_agg"]
        ax.axhline(agg, color="#b2182b", linestyle=":", linewidth=1,
                   label=f"Aggregate ATT = {agg:.1f}")

    ax.set_xlabel("Years relative to treatment", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Callaway & Sant'Anna (2021): Education ≥ 10% → Development Outcomes\n"
             f"Control: not-yet-treated | {N_BOOT} bootstrap | {N} countries, "
             f"{PERIODS[0]}–{PERIODS[-1]}",
             fontsize=11, y=1.02)
fig.tight_layout()
for ext in ("pdf", "png"):
    path = os.path.join(FIG_DIR, f"callaway_santanna_event_study.{ext}")
    fig.savefig(path, dpi=200, bbox_inches="tight")
print(f"\n  Figure saved to figures/callaway_santanna_event_study.{{pdf,png}}")
plt.close(fig)


# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"  TWFE β (binary):                {beta_twfe:.1f}")
print(f"  Callaway-Sant'Anna ATT (edu):   {result_edu['att_agg']:.1f}"
      f"  (95% CI: [{result_edu['att_ci'][0]:.1f}, {result_edu['att_ci'][1]:.1f}])")
print(f"  Callaway-Sant'Anna ATT (LE):    {result_le['att_agg']:.1f} years"
      f"  (95% CI: [{result_le['att_ci'][0]:.1f}, {result_le['att_ci'][1]:.1f}])")
if beta_twfe > 0:
    print(f"\n  The CS estimator is {result_edu['att_agg']/beta_twfe:.1f}× larger than TWFE")
print(f"  because it uses only not-yet-treated countries as controls,")
print(f"  avoiding the attenuation from already-transitioning comparisons")
print(f"  identified by the Goodman-Bacon decomposition.")


# ── Checkin JSON ─────────────────────────────────────────────────────
write_checkin("callaway_santanna.json", {
    "method": (
        "Callaway & Sant'Anna (2021) group-time ATT. "
        f"Treatment: lower secondary ≥ {THRESHOLD}%. "
        f"Control: not-yet-treated at each (g,t). "
        f"Panel: {N} countries, {PERIODS[0]}–{PERIODS[-1]}, 5yr. "
        f"Bootstrap: {N_BOOT} reps, clustered by country."
    ),
    "child_education": {
        "att_aggregate": round(result_edu["att_agg"], 2),
        "att_se": round(result_edu["att_se"], 2),
        "att_ci_lo": round(result_edu["att_ci"][0], 2),
        "att_ci_hi": round(result_edu["att_ci"][1], 2),
        "twfe_beta": round(beta_twfe, 2),
        "cs_over_twfe": round(result_edu["att_agg"] / beta_twfe, 2) if beta_twfe > 0 else None,
        "n_gt_cells": len(att_gt_edu),
        "event_study": {
            str(e): {"att": round(d["att"], 2), "se": round(d["se"], 2),
                      "ci_lo": round(d["ci_lo"], 2), "ci_hi": round(d["ci_hi"], 2)}
            for e, d in result_edu["att_dynamic"].items()
        },
    },
    "life_expectancy": {
        "att_aggregate": round(result_le["att_agg"], 2),
        "att_se": round(result_le["att_se"], 2),
        "att_ci_lo": round(result_le["att_ci"][0], 2),
        "att_ci_hi": round(result_le["att_ci"][1], 2),
        "n_gt_cells": n_le_cells,
        "event_study": {
            str(e): {"att": round(d["att"], 2), "se": round(d["se"], 2),
                      "ci_lo": round(d["ci_lo"], 2), "ci_hi": round(d["ci_hi"], 2)}
            for e, d in result_le["att_dynamic"].items()
        },
    },
    "reference": "Callaway & Sant'Anna (2021), J. Econometrics 225(2): 200–230",
}, script_path="scripts/robustness/callaway_santanna.py")
