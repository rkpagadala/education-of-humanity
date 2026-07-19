"""
grandparent_effect.py
=====================
Life-history label (§the-generational-lag):
  Predictors: GRANDPARENT and PARENT 20-24 cohort completion (jointly),
  at per-outcome biological lags (TFR parent T-5 / grandparent T-33;
  LE and child education parent T-28 / grandparent T-56).
  Outcome: TFR/LE/U5MR/child edu at year T.
  Generational relation: two-step vs one-step cross-generation.
  Tests whether the grandparent's education adds INDEPENDENT signal
  on top of the parent's. Critical result: at low parental baselines
  (<50% completion), grandparent's primary β=-0.059 — nearly 2× the
  parent's β=-0.033 — on TFR. This is the empirical source of the
  "grandmother independent channel" cross-referenced from §1546
  and §1747 (primary→TFR composition-by-level).
  Use: identification of the kin-radius transmission channel.

Test whether grandparent-generation education independently predicts
fertility and other outcomes, controlling for parent-generation education
(per-outcome biological lags; see the method string below).

If the home-niche mechanism transmits across two generations simultaneously,
grandparent education should have a residual effect on:
  (a) grandchild's fertility (TFR at T)
  (b) grandchild's education (child edu at T)
  (c) life expectancy (LE at T)

This effect will be subtle — parent education absorbs most of the
grandparent's influence. The test asks whether the grandparent generation
adds anything beyond what the parent generation already transmits.

(A grandmother/grandfather sex split was previously computed here but
removed: at the biological TFR lag it is confounded by the historical
female-schooling gap — grandmother education near the floor, coefficient
unidentified — and so cannot establish a gendered mechanism.)

Panel structure (per-outcome biological lags):
  - TFR:                    parent E_{i,t-5},  grandparent E_{i,t-33}
  - LE / child education:   parent E_{i,t-28}, grandparent E_{i,t-56}
    (LE-at-birth is mortality-dominated; child education is the next
     generation, so both genuinely sit a generation/two back)
  (WCDE lower sec, both sexes, 20-24; outcomes at year T)

Country fixed effects throughout.

Sources:
  - Education: WCDE v3, lower secondary completion, age 20-24
    (both sexes, female-only, and male-only variants)
  - TFR: World Bank WDI (SP.DYN.TFRT.IN)
  - LE: World Bank WDI (SP.DYN.LE00.IN)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from _shared import (  # noqa: E402
    CHECKIN,
    load_education, load_wb, get_wb_val,
    LAG_TFR, LAG_GENERATION, LAG_TWO_GENERATIONS,
)

# ── Load data via canonical loaders ────────────────────────────────────
# load_education() filters out regional aggregates; load_wb() expands every
# known country-name alias so the WCDE → WDI join resolves by raw lookup.
edu = load_education("lower_sec_both.csv")
le = load_wb("life_expectancy_years.csv")
tfr = load_wb("children_per_woman_total_fertility.csv")

# Off-grid lags (grandparent T-28, child education T+28 under the 28-year
# generation) fall between the 5-year WCDE cohort columns; interpolate the
# year columns to annual so lagged lookups resolve. Parent-anchor iteration
# in the builders stays on the 5-year grid (int(c) % 5 == 0), so cohort
# spacing and sample density are unchanged — only the off-grid lookups are
# interpolated. (Mirrors grandparent_effect_all_outcomes.py.)
_yc = sorted([c for c in edu.columns if c.isdigit()], key=int)
_w = edu.set_index("country")[_yc].copy()
_w.columns = [int(c) for c in _w.columns]
_w = _w.reindex(columns=range(_w.columns.min(), _w.columns.max() + 1)).interpolate(
    axis=1, limit_area="inside")
_w.columns = [str(c) for c in _w.columns]
edu = edu[["country"]].join(_w, on="country")


def build_panel(edu_df, outcome_df, outcome_name, lag_gm, lag_m,
                log_outcome=False):
    """Build three-generation panel.

    For each country-year observation, anchoring on the parent's education
    year t_edu_m (the year the parent cohort was age 20-24):
      - grandparent_edu = education at t_edu_m - (lag_gm - lag_m)
      - parent_edu      = education at t_edu_m
      - outcome         = outcome variable at t_edu_m + lag_m
        (optionally log-transformed)

    Lags MUST be passed explicitly — there is no canonical default for
    a three-generation panel because outcome timing is biologically
    outcome-specific (LAG_TFR=5, LAG_GENERATION=28 for edu→LE/edu).

    Education is at 5-year intervals; outcomes are annual. Outcome
    lookup uses get_wb_val() so any known country-name alias resolves.
    """
    rows = []
    edu_years = sorted([int(c) for c in edu_df.columns if c.isdigit() and int(c) % 5 == 0])

    for _, row in edu_df.iterrows():
        country = row["country"]

        for t_edu_m in edu_years:
            t_edu_gm = t_edu_m - (lag_gm - lag_m)  # grandparent's education year
            t_outcome = t_edu_m + lag_m            # outcome year

            if str(t_edu_gm) not in edu_df.columns:
                continue

            gm_edu = row[str(t_edu_gm)]
            m_edu = row[str(t_edu_m)]

            if pd.isna(gm_edu) or pd.isna(m_edu):
                continue

            outcome_val = get_wb_val(outcome_df, country, t_outcome)
            if pd.isna(outcome_val):
                continue
            if log_outcome:
                if outcome_val <= 0:
                    continue
                outcome_val = float(np.log(outcome_val))

            rows.append({
                "country": country,
                "t_outcome": t_outcome,
                "grandparent_edu": gm_edu,
                "parent_edu": m_edu,
                outcome_name: outcome_val,
            })

    return pd.DataFrame(rows)


def build_edu_panel(edu_df, lag_gm, lag_m, lag_child):
    """Build three-generation panel for child education outcome.

    Anchoring on parent's age-20-24 year t_m:
      - grandparent_edu at t_m - (lag_gm - lag_m)
      - parent_edu      at t_m
      - child_edu       at t_m + lag_child

    edu→edu transmission is one biological generation per step
    (LAG_GENERATION=28) — pass that explicitly at the call site.
    """
    rows = []
    edu_years = sorted([int(c) for c in edu_df.columns if c.isdigit() and int(c) % 5 == 0])

    for _, row in edu_df.iterrows():
        for t_m in edu_years:
            t_gm = t_m - (lag_gm - lag_m)
            t_child = t_m + lag_m  # child edu one generation (28 yr) after mother

            if str(t_gm) not in edu_df.columns or str(t_child) not in edu_df.columns:
                continue

            gm_edu = row[str(t_gm)]
            m_edu = row[str(t_m)]
            child_edu = row[str(t_child)]

            if pd.isna(gm_edu) or pd.isna(m_edu) or pd.isna(child_edu):
                continue

            rows.append({
                "country": row["country"],
                "t_child": t_child,
                "grandparent_edu": gm_edu,
                "parent_edu": m_edu,
                "child_edu": child_edu,
            })

    return pd.DataFrame(rows)


def run_fe_regression(df, outcome_col, predictors, country_col="country"):
    """Run country fixed effects regression with clustered SEs."""
    df_clean = df.dropna(subset=[outcome_col] + predictors + [country_col])
    if len(df_clean) < 20:
        return None

    # Country dummies
    dummies = pd.get_dummies(df_clean[country_col], drop_first=True, dtype=float)
    X = pd.concat([df_clean[predictors].reset_index(drop=True),
                    dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = df_clean[outcome_col].reset_index(drop=True)

    try:
        # Clustered standard errors
        groups = df_clean[country_col].reset_index(drop=True)
        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})

        # Within-R² (from demeaned regression)
        df_dm = df_clean.copy()
        means = df_dm.groupby(country_col)[[outcome_col] + predictors].transform("mean")
        y_dm = df_dm[outcome_col].values - means[outcome_col].values
        X_dm = df_dm[predictors].values - means[predictors].values
        if X_dm.shape[0] > X_dm.shape[1]:
            from numpy.linalg import lstsq
            beta, _, _, _ = lstsq(X_dm, y_dm, rcond=None)
            y_hat = X_dm @ beta
            ss_res = np.sum((y_dm - y_hat) ** 2)
            ss_tot = np.sum(y_dm ** 2)
            within_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        else:
            within_r2 = 0

        result = {
            "n": int(len(df_clean)),
            "n_countries": int(df_clean[country_col].nunique()),
            "within_r2": round(within_r2, 4),
        }
        for p in predictors:
            result[f"beta_{p}"] = round(float(model.params[p]), 4)
            result[f"pval_{p}"] = round(float(model.pvalues[p]), 4)
            result[f"se_{p}"] = round(float(model.bse[p]), 4)
        return result
    except Exception as e:
        print(f"  Regression failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# RUN TESTS
# ══════════════════════════════════════════════════════════════════════
results = {}

print("=" * 70)
print("GRANDPARENT EFFECT TEST")
print("=" * 70)

# ── Test 1: TFR ──────────────────────────────────────────────────────
# Biological anchoring: TFR at year T is driven by the cohort currently age
# 25-29 (parent of newborns at T). That cohort completed schooling LAG_TFR=5
# years earlier. Their parents (grandparents of the newborns) completed
# schooling one generation before them: LAG_TFR + LAG_GENERATION = 33.
print("\n--- TFR (fertility) ---")
panel_tfr = build_panel(edu, tfr, "tfr",
                        lag_m=LAG_TFR,
                        lag_gm=LAG_TFR + LAG_GENERATION)
print(f"  Panel: {len(panel_tfr)} obs, {panel_tfr['country'].nunique()} countries")

# Model 1: TFR ~ parent_edu only
m1 = run_fe_regression(panel_tfr, "tfr", ["parent_edu"])
print(f"  Model 1 (parent only):       β_p={m1['beta_parent_edu']:+.4f}  R²={m1['within_r2']:.4f}  n={m1['n']}")

# Model 2: TFR ~ parent_edu + grandparent_edu
m2 = run_fe_regression(panel_tfr, "tfr", ["parent_edu", "grandparent_edu"])
print(f"  Model 2 (parent + GP):       β_p={m2['beta_parent_edu']:+.4f}  β_gp={m2['beta_grandparent_edu']:+.4f}  (p={m2['pval_grandparent_edu']:.4f})  R²={m2['within_r2']:.4f}")

# Model 3: TFR ~ grandparent_edu only (for comparison)
m3 = run_fe_regression(panel_tfr, "tfr", ["grandparent_edu"])
print(f"  Model 3 (GP only):           β_gp={m3['beta_grandparent_edu']:+.4f}  R²={m3['within_r2']:.4f}")

r2_gain_tfr = m2["within_r2"] - m1["within_r2"]
print(f"  R² gain from adding GP:      {r2_gain_tfr:.4f}")
print(f"  GP β / Parent β ratio:       {abs(m2['beta_grandparent_edu'] / m2['beta_parent_edu']):.3f}")

results["tfr"] = {
    "parent_only": m1, "parent_gp": m2, "gp_only": m3,
    "r2_gain": round(r2_gain_tfr, 4),
}

# ── Test 2: Life expectancy ──────────────────────────────────────────
# LE uses the one-generation lag (LAG_GENERATION=28); the cohort that's
# currently exercising adult household decisions completed schooling 25
# years earlier. (Biological re-anchoring of LE not requested.)
print("\n--- Life expectancy ---")
panel_le = build_panel(edu, le, "le",
                       lag_m=LAG_GENERATION,
                       lag_gm=LAG_TWO_GENERATIONS)
print(f"  Panel: {len(panel_le)} obs, {panel_le['country'].nunique()} countries")

m1_le = run_fe_regression(panel_le, "le", ["parent_edu"])
m2_le = run_fe_regression(panel_le, "le", ["parent_edu", "grandparent_edu"])
m3_le = run_fe_regression(panel_le, "le", ["grandparent_edu"])

print(f"  Model 1 (parent only):       β_p={m1_le['beta_parent_edu']:+.4f}  R²={m1_le['within_r2']:.4f}")
print(f"  Model 2 (parent + GP):       β_p={m2_le['beta_parent_edu']:+.4f}  β_gp={m2_le['beta_grandparent_edu']:+.4f}  (p={m2_le['pval_grandparent_edu']:.4f})  R²={m2_le['within_r2']:.4f}")
print(f"  Model 3 (GP only):           β_gp={m3_le['beta_grandparent_edu']:+.4f}  R²={m3_le['within_r2']:.4f}")

r2_gain_le = m2_le["within_r2"] - m1_le["within_r2"]
print(f"  R² gain from adding GP:      {r2_gain_le:.4f}")

results["le"] = {
    "parent_only": m1_le, "parent_gp": m2_le, "gp_only": m3_le,
    "r2_gain": round(r2_gain_le, 4),
}

# ── Test 3: Child education ──────────────────────────────────────────
print("\n--- Child education (grandchild generation) ---")
panel_edu = build_edu_panel(edu,
                            lag_m=LAG_GENERATION,
                            lag_gm=LAG_TWO_GENERATIONS,
                            lag_child=LAG_GENERATION)
print(f"  Panel: {len(panel_edu)} obs, {panel_edu['country'].nunique()} countries")

m1_edu = run_fe_regression(panel_edu, "child_edu", ["parent_edu"])
m2_edu = run_fe_regression(panel_edu, "child_edu", ["parent_edu", "grandparent_edu"])
m3_edu = run_fe_regression(panel_edu, "child_edu", ["grandparent_edu"])

print(f"  Model 1 (parent only):       β_p={m1_edu['beta_parent_edu']:+.4f}  R²={m1_edu['within_r2']:.4f}")
print(f"  Model 2 (parent + GP):       β_p={m2_edu['beta_parent_edu']:+.4f}  β_gp={m2_edu['beta_grandparent_edu']:+.4f}  (p={m2_edu['pval_grandparent_edu']:.4f})  R²={m2_edu['within_r2']:.4f}")
print(f"  Model 3 (GP only):           β_gp={m3_edu['beta_grandparent_edu']:+.4f}  R²={m3_edu['within_r2']:.4f}")

r2_gain_edu = m2_edu["within_r2"] - m1_edu["within_r2"]
print(f"  R² gain from adding GP:      {r2_gain_edu:.4f}")

results["child_edu"] = {
    "parent_only": m1_edu, "parent_gp": m2_edu, "gp_only": m3_edu,
    "r2_gain": round(r2_gain_edu, 4),
}

# ── Test 4: Restrict to low-education countries (where effect should be strongest)
print("\n--- TFR, countries with parent edu < 50% ---")
panel_tfr_low = panel_tfr[panel_tfr["parent_edu"] < 50].copy()  # low-education subsample
print(f"  Panel: {len(panel_tfr_low)} obs, {panel_tfr_low['country'].nunique()} countries")

if len(panel_tfr_low) > 30 and panel_tfr_low["country"].nunique() > 5:
    m1_low = run_fe_regression(panel_tfr_low, "tfr", ["parent_edu"])
    m2_low = run_fe_regression(panel_tfr_low, "tfr", ["parent_edu", "grandparent_edu"])
    if m1_low and m2_low:
        print(f"  Model 1 (parent only):       β_p={m1_low['beta_parent_edu']:+.4f}  R²={m1_low['within_r2']:.4f}")
        print(f"  Model 2 (parent + GP):       β_p={m2_low['beta_parent_edu']:+.4f}  β_gp={m2_low['beta_grandparent_edu']:+.4f}  (p={m2_low['pval_grandparent_edu']:.4f})  R²={m2_low['within_r2']:.4f}")
        r2_gain_low = m2_low["within_r2"] - m1_low["within_r2"]
        print(f"  R² gain from adding GP:      {r2_gain_low:.4f}")
        results["tfr_low_edu"] = {
            "parent_only": m1_low, "parent_gp": m2_low,
            "r2_gain": round(r2_gain_low, 4),
        }

# ── Test 4b: Same as Test 4 but with log(TFR) — matches tab:edu-outcomes ──
# Adds a log-spec companion to the level-TFR low-edu result so the
# paragraph can cite either scale. The level result above is preserved
# for backward-compatible verify regs.
print("\n--- log(TFR), countries with parent edu < 50% ---")
panel_tfr_log = build_panel(edu, tfr, "tfr_log",
                            lag_m=LAG_TFR,
                            lag_gm=LAG_TFR + LAG_GENERATION,
                            log_outcome=True)
panel_tfr_log_low = panel_tfr_log[panel_tfr_log["parent_edu"] < 50].copy()
print(f"  Panel: {len(panel_tfr_log_low)} obs, "
      f"{panel_tfr_log_low['country'].nunique()} countries")

if len(panel_tfr_log_low) > 30 and panel_tfr_log_low["country"].nunique() > 5:
    m1_low_log = run_fe_regression(panel_tfr_log_low, "tfr_log", ["parent_edu"])
    m2_low_log = run_fe_regression(panel_tfr_log_low, "tfr_log",
                                   ["parent_edu", "grandparent_edu"])
    if m1_low_log and m2_low_log:
        print(f"  Model 1 (parent only):       "
              f"β_p={m1_low_log['beta_parent_edu']:+.4f}  "
              f"R²={m1_low_log['within_r2']:.4f}")
        print(f"  Model 2 (parent + GP):       "
              f"β_p={m2_low_log['beta_parent_edu']:+.4f}  "
              f"β_gp={m2_low_log['beta_grandparent_edu']:+.4f}  "
              f"(p={m2_low_log['pval_grandparent_edu']:.4f})  "
              f"R²={m2_low_log['within_r2']:.4f}")
        r2_gain_low_log = m2_low_log["within_r2"] - m1_low_log["within_r2"]
        print(f"  R² gain from adding GP:      {r2_gain_low_log:.4f}")
        results["tfr_log_low_edu"] = {
            "parent_only": m1_low_log, "parent_gp": m2_low_log,
            "r2_gain": round(r2_gain_low_log, 4),
        }

# ── Test 5: Does GM education predict mother's education? ────────────
# (This is really the parent→child test at the previous generation)
print("\n--- Parent education ~ Grandparent education (one-gen transmission) ---")
panel_m_gm = panel_edu[["country", "grandparent_edu", "parent_edu"]].drop_duplicates()
m_gm = run_fe_regression(panel_m_gm, "parent_edu", ["grandparent_edu"])
if m_gm:
    print(f"  β_gp={m_gm['beta_grandparent_edu']:+.4f}  R²={m_gm['within_r2']:.4f}  n={m_gm['n']}")
    results["gp_to_parent"] = m_gm

# ── Test 6: Decomposition — how much of GM's effect runs through mother?
print("\n" + "=" * 70)
print("DECOMPOSITION")
print("=" * 70)

# For TFR:
gp_total = m3["beta_grandparent_edu"]  # GM alone
gp_direct = m2["beta_grandparent_edu"]  # GM controlling for mother
gp_indirect = gp_total - gp_direct     # part running through mother
pct_direct = abs(gp_direct / gp_total) * 100 if gp_total != 0 else 0
pct_indirect = abs(gp_indirect / gp_total) * 100 if gp_total != 0 else 0

print(f"\n  TFR:")
print(f"    GP total effect (alone):     {gp_total:+.4f}")
print(f"    GP direct effect (| parent): {gp_direct:+.4f}  ({pct_direct:.1f}%)")
print(f"    GP indirect (via parent):    {gp_indirect:+.4f}  ({pct_indirect:.1f}%)")

results["decomposition_tfr"] = {
    "gp_total": round(gp_total, 4),
    "gp_direct": round(gp_direct, 4),
    "gp_indirect": round(gp_indirect, 4),
    "pct_direct": round(pct_direct, 1),
    "pct_indirect": round(pct_indirect, 1),
}

# For LE:
gp_total_le = m3_le["beta_grandparent_edu"]
gp_direct_le = m2_le["beta_grandparent_edu"]
gp_indirect_le = gp_total_le - gp_direct_le
pct_direct_le = abs(gp_direct_le / gp_total_le) * 100 if gp_total_le != 0 else 0
pct_indirect_le = abs(gp_indirect_le / gp_total_le) * 100 if gp_total_le != 0 else 0

print(f"\n  Life expectancy:")
print(f"    GP total effect (alone):     {gp_total_le:+.4f}")
print(f"    GP direct effect (| parent): {gp_direct_le:+.4f}  ({pct_direct_le:.1f}%)")
print(f"    GP indirect (via parent):    {gp_indirect_le:+.4f}  ({pct_indirect_le:.1f}%)")

results["decomposition_le"] = {
    "gp_total": round(gp_total_le, 4),
    "gp_direct": round(gp_direct_le, 4),
    "gp_indirect": round(gp_indirect_le, 4),
    "pct_direct": round(pct_direct_le, 1),
    "pct_indirect": round(pct_indirect_le, 1),
}

# (Sex comparison removed 2026-05-31: the grandmother/grandfather split is
# not surfaced anywhere — at the biological TFR lag it is confounded by the
# historical female-schooling gap and cannot establish a gendered mechanism.)


# ── Save checkin ──────────────────────────────────────────────────────
output = {
    "method": ("Grandparent effect test: does grandparent-generation education "
               "predict outcomes controlling for parent-generation education? "
               "Per-outcome biological lags: TFR parent T-5 / grandparent T-33; "
               "LE and child education parent T-28 / grandparent T-56 (LE-at-birth "
               "is mortality-dominated and child education is the next generation). "
               "Country FE, clustered SEs. WCDE v3 + World Bank WDI, both sexes."),
    "results": results,
}

outpath = os.path.join(CHECKIN, "grandparent_effect.json")
with open(outpath, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {outpath}")
