"""
grandparent_effect.py
=====================
Test whether grandparent-generation education (T-50) independently predicts
fertility and other outcomes, controlling for parent-generation education (T-25).

If the PT mechanism transmits across two generations simultaneously,
grandparent education should have a residual effect on:
  (a) grandchild's fertility (TFR at T)
  (b) grandchild's education (child edu at T)
  (c) life expectancy (LE at T)

This effect will be subtle — parent education absorbs most of the
grandparent's influence. The test asks whether the grandparent generation
adds anything beyond what the parent generation already transmits.

Also runs a sex comparison: grandmother (female-only) vs grandfather
(male-only) vs grandparent (both sexes) to test whether the effect
is gendered. In low-education settings, both educated grandfathers
and grandmothers serve as visible exemplars — oblique transmission
to the wider community, not just vertical transmission to own kin.

Panel structure:
  - Grandparent education: E_{i, t-50}  (WCDE lower sec, both sexes, 20-24)
  - Parent education:      E_{i, t-25}  (WCDE lower sec, both sexes, 20-24)
  - Outcomes at time T:    TFR, LE, child education (WCDE at T+25)

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
from _shared import PROC, DATA, CHECKIN, REGIONS, NAME_MAP, WB_REGIONS

# ── Load data ─────────────────────────────────────────────────────────
edu_raw = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"))
le_raw = pd.read_csv(os.path.join(DATA, "life_expectancy_years.csv"))
tfr_raw = pd.read_csv(os.path.join(DATA, "children_per_woman_total_fertility.csv"))

# ── Filter to sovereign countries ─────────────────────────────────────
edu = edu_raw[~edu_raw["country"].isin(REGIONS)].copy()

# Map WCDE names to lowercase for WDI join
edu["country_lc"] = edu["country"].map(NAME_MAP).fillna(edu["country"].str.lower())

# WDI: lowercase country names
le_raw["Country"] = le_raw["Country"].str.lower()
tfr_raw["Country"] = tfr_raw["Country"].str.lower()

# Filter WDI regions
le = le_raw[~le_raw["Country"].isin(WB_REGIONS)].copy()
tfr = tfr_raw[~tfr_raw["Country"].isin(WB_REGIONS)].copy()


def build_panel(edu_df, outcome_df, outcome_name, lag_gm=50, lag_m=25):
    """Build three-generation panel.

    For each country-year observation:
      - grandparent_edu = education at T - lag_gm
      - parent_edu      = education at T - lag_m
      - outcome         = outcome variable at T

    Education is at 5-year intervals; outcomes are annual.
    We match education years exactly (5-year multiples) and use annual outcomes.
    """
    rows = []
    edu_years = sorted([int(c) for c in edu_df.columns if c.isdigit()])

    for _, row in edu_df.iterrows():
        country_wdi = row["country_lc"]
        outcome_row = outcome_df[outcome_df["Country"] == country_wdi]
        if outcome_row.empty:
            continue

        for t_edu_m in edu_years:
            t_edu_gm = t_edu_m - (lag_gm - lag_m)  # grandmother's education year
            t_outcome = t_edu_m + lag_m              # outcome year

            if str(t_edu_gm) not in edu_df.columns:
                continue

            gm_edu = row[str(t_edu_gm)]
            m_edu = row[str(t_edu_m)]

            if pd.isna(gm_edu) or pd.isna(m_edu):
                continue

            # Get outcome at t_outcome (annual)
            t_out_str = str(t_outcome)
            if t_out_str not in outcome_row.columns:
                continue
            outcome_val = outcome_row[t_out_str].values[0]
            if pd.isna(outcome_val):
                continue

            rows.append({
                "country": row["country"],
                "t_outcome": t_outcome,
                "grandparent_edu": gm_edu,
                "parent_edu": m_edu,
                outcome_name: outcome_val,
            })

    return pd.DataFrame(rows)


def build_edu_panel(edu_df, lag_gm=50, lag_m=25, lag_child=25):
    """Build three-generation panel for child education outcome.

    grandparent_edu at T-50, parent_edu at T-25, child_edu at T+25.
    Relative to mother's year T: GM at T-25, child at T+25.
    """
    rows = []
    edu_years = sorted([int(c) for c in edu_df.columns if c.isdigit()])

    for _, row in edu_df.iterrows():
        for t_m in edu_years:
            t_gm = t_m - (lag_gm - lag_m)
            t_child = t_m + lag_m  # child edu 25 years after mother

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
print("\n--- TFR (fertility) ---")
panel_tfr = build_panel(edu, tfr, "tfr")
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
print("\n--- Life expectancy ---")
panel_le = build_panel(edu, le, "le")
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
panel_edu = build_edu_panel(edu)
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

# ══════════════════════════════════════════════════════════════════════
# SEX COMPARISON: Grandmother vs Grandfather vs Both
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SEX COMPARISON: grandmother (F) vs grandfather (M) vs grandparent (both)")
print("=" * 70)

edu_female = pd.read_csv(os.path.join(PROC, "lower_sec_female.csv"))
edu_male = pd.read_csv(os.path.join(PROC, "lower_sec_male.csv"))

edu_f = edu_female[~edu_female["country"].isin(REGIONS)].copy()
edu_m_sex = edu_male[~edu_male["country"].isin(REGIONS)].copy()

edu_f["country_lc"] = edu_f["country"].map(NAME_MAP).fillna(edu_f["country"].str.lower())
edu_m_sex["country_lc"] = edu_m_sex["country"].map(NAME_MAP).fillna(edu_m_sex["country"].str.lower())

sex_results = {}

for sex_label, edu_sex in [("female (grandmother)", edu_f),
                            ("male (grandfather)", edu_m_sex),
                            ("both (grandparent)", edu)]:
    print(f"\n  --- {sex_label} ---")
    for outcome_name, outcome_df, out_label in [("tfr", tfr, "TFR"), ("le", le, "LE")]:
        panel = build_panel(edu_sex, outcome_df, outcome_name)
        if len(panel) < 30:
            print(f"    {out_label}: insufficient data ({len(panel)} obs)")
            continue

        # Parent only
        m1_sex = run_fe_regression(panel, outcome_name, ["parent_edu"])
        # Parent + grandparent
        m2_sex = run_fe_regression(panel, outcome_name, ["parent_edu", "grandparent_edu"])
        # Grandparent only
        m3_sex = run_fe_regression(panel, outcome_name, ["grandparent_edu"])

        if m2_sex and m1_sex:
            r2_gain = m2_sex["within_r2"] - m1_sex["within_r2"]
            print(f"    {out_label}: β_gp={m2_sex['beta_grandparent_edu']:+.4f} "
                  f"(p={m2_sex['pval_grandparent_edu']:.4f})  "
                  f"R² gain={r2_gain:.4f}  n={m2_sex['n']}")
            sex_results[f"{sex_label}_{outcome_name}"] = {
                "parent_only_r2": m1_sex["within_r2"],
                "parent_gp_r2": m2_sex["within_r2"],
                "r2_gain": round(r2_gain, 4),
                "beta_gp": m2_sex["beta_grandparent_edu"],
                "pval_gp": m2_sex["pval_grandparent_edu"],
                "n": m2_sex["n"],
            }

    # Low-education subsample (TFR only)
    panel_tfr_sex = build_panel(edu_sex, tfr, "tfr")
    panel_low = panel_tfr_sex[panel_tfr_sex["parent_edu"] < 50].copy()
    if len(panel_low) > 30 and panel_low["country"].nunique() > 5:
        m1_l = run_fe_regression(panel_low, "tfr", ["parent_edu"])
        m2_l = run_fe_regression(panel_low, "tfr", ["parent_edu", "grandparent_edu"])
        if m1_l and m2_l:
            r2_g = m2_l["within_r2"] - m1_l["within_r2"]
            print(f"    TFR (low edu): β_gp={m2_l['beta_grandparent_edu']:+.4f} "
                  f"(p={m2_l['pval_grandparent_edu']:.4f})  "
                  f"R² gain={r2_g:.4f}  n={m2_l['n']}")
            sex_results[f"{sex_label}_tfr_low"] = {
                "r2_gain": round(r2_g, 4),
                "beta_gp": m2_l["beta_grandparent_edu"],
                "pval_gp": m2_l["pval_grandparent_edu"],
                "n": m2_l["n"],
            }

results["sex_comparison"] = sex_results

# Summary table
print(f"\n{'SEX COMPARISON SUMMARY':<45}")
print(f"{'Sex':<30} {'β_gp(TFR)':>10} {'β_gp(LE)':>10} {'R²+TFR':>8} {'R²+LE':>8}")
print("-" * 66)
for sex_label in ["female (grandmother)", "male (grandfather)", "both (grandparent)"]:
    tfr_r = sex_results.get(f"{sex_label}_tfr", {})
    le_r = sex_results.get(f"{sex_label}_le", {})
    b_tfr = f"{tfr_r['beta_gp']:+.4f}" if tfr_r else "n/a"
    b_le = f"{le_r['beta_gp']:+.4f}" if le_r else "n/a"
    rg_tfr = f"{tfr_r['r2_gain']:.4f}" if tfr_r else "n/a"
    rg_le = f"{le_r['r2_gain']:.4f}" if le_r else "n/a"
    print(f"  {sex_label:<28} {b_tfr:>10} {b_le:>10} {rg_tfr:>8} {rg_le:>8}")


# ── Save checkin ──────────────────────────────────────────────────────
output = {
    "method": ("Grandparent effect test: does grandparent education (T-50) predict "
               "outcomes controlling for parent education (T-25)? "
               "Country FE, clustered SEs. WCDE v3 + World Bank WDI. "
               "Both sexes for main test; female/male comparison in sex_comparison."),
    "results": results,
}

outpath = os.path.join(CHECKIN, "grandparent_effect.json")
with open(outpath, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {outpath}")
