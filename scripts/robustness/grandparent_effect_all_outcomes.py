"""
grandparent_effect_all_outcomes.py
==================================
Generalization of grandparent_effect.py to all four development outcomes.

Test whether grandparent-generation education (T-50) independently predicts
each development outcome, controlling for parent-generation education (T-25).

Outcomes covered:
  - Life expectancy (LE)            -- WDI
  - Total fertility rate (TFR)      -- WDI
  - Under-5 mortality, log (U5M)    -- WDI (log-transformed)
  - Child education (child_edu)     -- WCDE (education at T+25)

Methodology mirrors grandparent_effect.py:
  - Grandparent education: E_{i, t-50}   (WCDE lower sec, both sexes, 20-24)
  - Parent education:      E_{i, t-25}   (WCDE lower sec, both sexes, 20-24)
  - Outcome at T (for LE/TFR/U5M), or grandchild education at T+25.
  - Country fixed effects with cluster-robust SEs.
  - Low-education subsample: parent_edu < 50%.

For each outcome we report:
  - Model 1: outcome ~ parent_edu only
  - Model 2: outcome ~ parent_edu + grandparent_edu   (the independence test)
  - Model 3: outcome ~ grandparent_edu only           (for decomposition)
  - R² gain from adding grandparent to parent-only model.

Output: checkin/grandparent_effect_all_outcomes.json, plus a ranking of
outcomes by the strength of the grandparent channel.

Sources:
  - Education: WCDE v3, lower secondary completion, both sexes, age 20-24
  - LE:  WDI SP.DYN.LE00.IN
  - TFR: WDI SP.DYN.TFRT.IN
  - U5M: WDI SH.DYN.MORT (child_mortality_u5.csv)
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
u5_raw = pd.read_csv(os.path.join(DATA, "child_mortality_u5.csv"))

# ── Filter to sovereign countries ─────────────────────────────────────
edu = edu_raw[~edu_raw["country"].isin(REGIONS)].copy()
edu["country_lc"] = edu["country"].map(NAME_MAP).fillna(edu["country"].str.lower())

for df in (le_raw, tfr_raw, u5_raw):
    df["Country"] = df["Country"].str.lower()

le = le_raw[~le_raw["Country"].isin(WB_REGIONS)].copy()
tfr = tfr_raw[~tfr_raw["Country"].isin(WB_REGIONS)].copy()
u5 = u5_raw[~u5_raw["Country"].isin(WB_REGIONS)].copy()


# ── Panel builders ────────────────────────────────────────────────────
def build_panel(edu_df, outcome_df, outcome_name, lag_gm=50, lag_m=25,
                log_outcome=False):
    """Build three-generation panel.

    For each country-year observation:
      - grandparent_edu = education at T - lag_gm
      - parent_edu      = education at T - lag_m
      - outcome         = outcome variable at T (optionally log-transformed)
    """
    rows = []
    edu_years = sorted([int(c) for c in edu_df.columns if c.isdigit()])

    for _, row in edu_df.iterrows():
        country_wdi = row["country_lc"]
        outcome_row = outcome_df[outcome_df["Country"] == country_wdi]
        if outcome_row.empty:
            continue

        for t_edu_m in edu_years:
            t_edu_gm = t_edu_m - (lag_gm - lag_m)
            t_outcome = t_edu_m + lag_m

            if str(t_edu_gm) not in edu_df.columns:
                continue
            gm_edu = row[str(t_edu_gm)]
            m_edu = row[str(t_edu_m)]
            if pd.isna(gm_edu) or pd.isna(m_edu):
                continue

            t_out_str = str(t_outcome)
            if t_out_str not in outcome_row.columns:
                continue
            outcome_val = outcome_row[t_out_str].values[0]
            if pd.isna(outcome_val):
                continue
            if log_outcome:
                if outcome_val <= 0:
                    continue
                outcome_val = float(np.log(outcome_val))

            rows.append({
                "country": row["country"],
                "t_outcome": t_outcome,
                "grandparent_edu": gm_edu,
                "parent_edu": m_edu,
                outcome_name: outcome_val,
            })

    return pd.DataFrame(rows)


def build_edu_panel(edu_df, lag_gm=50, lag_m=25):
    """Build three-generation panel for child education outcome.

    grandparent_edu at T-50, parent_edu at T-25, child_edu at T+25
    (relative to mother year T: GM at T-25, child at T+25 → all from one CSV).
    """
    rows = []
    edu_years = sorted([int(c) for c in edu_df.columns if c.isdigit()])

    for _, row in edu_df.iterrows():
        for t_m in edu_years:
            t_gm = t_m - (lag_gm - lag_m)
            t_child = t_m + lag_m
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


# ── Regression utility (country FE + within-R²) ──────────────────────
def run_fe_regression(df, outcome_col, predictors, country_col="country"):
    df_clean = df.dropna(subset=[outcome_col] + predictors + [country_col])
    if len(df_clean) < 20:
        return None

    dummies = pd.get_dummies(df_clean[country_col], drop_first=True, dtype=float)
    X = pd.concat([df_clean[predictors].reset_index(drop=True),
                   dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = df_clean[outcome_col].reset_index(drop=True)

    try:
        groups = df_clean[country_col].reset_index(drop=True)
        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})

        # Within-R² via demeaned regression
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
            result[f"beta_{p}"] = round(float(model.params[p]), 6)
            result[f"pval_{p}"] = round(float(model.pvalues[p]), 6)
            result[f"se_{p}"] = round(float(model.bse[p]), 6)
        return result
    except Exception as e:
        print(f"  Regression failed: {e}")
        return None


# ── Run all four outcomes ─────────────────────────────────────────────
def run_outcome(panel, outcome_col, label, low_edu_cutoff=50.0):
    """Run the three-model sequence on the full panel and the low-edu subsample."""
    block = {"outcome": label, "outcome_col": outcome_col}

    print(f"\n--- {label} ---")
    print(f"  Full panel: {len(panel)} obs, {panel['country'].nunique()} countries")

    m1 = run_fe_regression(panel, outcome_col, ["parent_edu"])
    m2 = run_fe_regression(panel, outcome_col, ["parent_edu", "grandparent_edu"])
    m3 = run_fe_regression(panel, outcome_col, ["grandparent_edu"])

    if m1 and m2:
        r2_gain = m2["within_r2"] - m1["within_r2"]
        ratio = (abs(m2["beta_grandparent_edu"] / m2["beta_parent_edu"])
                 if m2["beta_parent_edu"] else float("nan"))
        print(f"  Full | M1 parent-only: beta_p={m1['beta_parent_edu']:+.4f}  "
              f"R²={m1['within_r2']:.4f}  n={m1['n']}")
        print(f"  Full | M2 parent+GP:  beta_p={m2['beta_parent_edu']:+.4f}  "
              f"beta_gp={m2['beta_grandparent_edu']:+.4f}  "
              f"(p={m2['pval_grandparent_edu']:.4f})  "
              f"R²={m2['within_r2']:.4f}")
        print(f"  Full | M3 GP-only:    beta_gp={m3['beta_grandparent_edu']:+.4f}  "
              f"R²={m3['within_r2']:.4f}")
        print(f"  Full | R² gain from adding GP: {r2_gain:.4f}   "
              f"|β_gp/β_p|: {ratio:.3f}")
        block["full"] = {
            "parent_only": m1, "parent_gp": m2, "gp_only": m3,
            "r2_gain": round(r2_gain, 4),
            "beta_ratio_gp_over_p": round(ratio, 4) if not np.isnan(ratio) else None,
        }

    # Low-education subsample
    panel_low = panel[panel["parent_edu"] < low_edu_cutoff].copy()
    print(f"  Low-edu (parent<{low_edu_cutoff:.0f}%): {len(panel_low)} obs, "
          f"{panel_low['country'].nunique()} countries")
    if len(panel_low) > 30 and panel_low["country"].nunique() > 5:
        m1_lo = run_fe_regression(panel_low, outcome_col, ["parent_edu"])
        m2_lo = run_fe_regression(panel_low, outcome_col, ["parent_edu", "grandparent_edu"])
        m3_lo = run_fe_regression(panel_low, outcome_col, ["grandparent_edu"])
        if m1_lo and m2_lo:
            r2_gain_lo = m2_lo["within_r2"] - m1_lo["within_r2"]
            ratio_lo = (abs(m2_lo["beta_grandparent_edu"] / m2_lo["beta_parent_edu"])
                        if m2_lo["beta_parent_edu"] else float("nan"))
            print(f"  Low  | M1 parent-only: beta_p={m1_lo['beta_parent_edu']:+.4f}  "
                  f"R²={m1_lo['within_r2']:.4f}  n={m1_lo['n']}")
            print(f"  Low  | M2 parent+GP:  beta_p={m2_lo['beta_parent_edu']:+.4f}  "
                  f"beta_gp={m2_lo['beta_grandparent_edu']:+.4f}  "
                  f"(p={m2_lo['pval_grandparent_edu']:.4f})  "
                  f"R²={m2_lo['within_r2']:.4f}")
            print(f"  Low  | M3 GP-only:    beta_gp={m3_lo['beta_grandparent_edu']:+.4f}  "
                  f"R²={m3_lo['within_r2']:.4f}")
            print(f"  Low  | R² gain from adding GP: {r2_gain_lo:.4f}   "
                  f"|β_gp/β_p|: {ratio_lo:.3f}")
            block["low_edu"] = {
                "cutoff": low_edu_cutoff,
                "parent_only": m1_lo, "parent_gp": m2_lo, "gp_only": m3_lo,
                "r2_gain": round(r2_gain_lo, 4),
                "beta_ratio_gp_over_p": round(ratio_lo, 4) if not np.isnan(ratio_lo) else None,
            }
    return block


# ══════════════════════════════════════════════════════════════════════
# BUILD PANELS
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("GRANDPARENT EFFECT — ALL FOUR OUTCOMES")
print("=" * 70)

panel_le = build_panel(edu, le, "le")
panel_tfr = build_panel(edu, tfr, "tfr")
panel_u5 = build_panel(edu, u5, "u5_log", log_outcome=True)
panel_edu = build_edu_panel(edu)

# ══════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════
all_results = {
    "le":        run_outcome(panel_le,  "le",        "Life expectancy (LE)"),
    "tfr":       run_outcome(panel_tfr, "tfr",       "Total fertility rate (TFR)"),
    "u5_log":    run_outcome(panel_u5,  "u5_log",    "Under-5 mortality (log)"),
    "child_edu": run_outcome(panel_edu, "child_edu", "Child education (T+25)"),
}


# ══════════════════════════════════════════════════════════════════════
# RANKING
# ══════════════════════════════════════════════════════════════════════
def rank_outcomes(results, scope):
    """Rank outcomes by grandparent-channel strength.

    Primary metric: R² gain from adding grandparent to parent-only model.
    Tiebreaker: |β_gp / β_p|.
    """
    rows = []
    for key, block in results.items():
        if scope not in block:
            continue
        b = block[scope]
        rows.append({
            "key": key,
            "label": block["outcome"],
            "r2_gain": b["r2_gain"],
            "beta_ratio": b.get("beta_ratio_gp_over_p"),
            "beta_gp": b["parent_gp"]["beta_grandparent_edu"],
            "beta_p":  b["parent_gp"]["beta_parent_edu"],
            "pval_gp": b["parent_gp"]["pval_grandparent_edu"],
            "n":       b["parent_gp"]["n"],
        })
    rows.sort(key=lambda r: (r["r2_gain"], r["beta_ratio"] or 0), reverse=True)
    return rows


ranking_full = rank_outcomes(all_results, "full")
ranking_low = rank_outcomes(all_results, "low_edu")

print("\n" + "=" * 70)
print("RANKING — Full panel (by ΔR² from adding grandparent)")
print("=" * 70)
for i, r in enumerate(ranking_full, 1):
    print(f"  {i}. {r['label']:<30}  ΔR²={r['r2_gain']:+.4f}   "
          f"|β_gp/β_p|={r['beta_ratio']:.3f}   "
          f"β_gp={r['beta_gp']:+.4f}  p={r['pval_gp']:.4f}  n={r['n']}")

print("\n" + "=" * 70)
print("RANKING — Low-education subsample (parent_edu < 50%)")
print("=" * 70)
for i, r in enumerate(ranking_low, 1):
    print(f"  {i}. {r['label']:<30}  ΔR²={r['r2_gain']:+.4f}   "
          f"|β_gp/β_p|={r['beta_ratio']:.3f}   "
          f"β_gp={r['beta_gp']:+.4f}  p={r['pval_gp']:.4f}  n={r['n']}")


# ══════════════════════════════════════════════════════════════════════
# SAVE CHECKIN
# ══════════════════════════════════════════════════════════════════════
output = {
    "method": (
        "Grandparent-channel independence test across all four development "
        "outcomes. For each outcome, fit a country-FE panel with cluster-robust "
        "SEs: (1) outcome ~ parent_edu, (2) outcome ~ parent_edu + grandparent_edu, "
        "(3) outcome ~ grandparent_edu. Grandparent education is at T-50, parent "
        "at T-25 (WCDE v3 lower secondary completion, both sexes, ages 20-24). "
        "Under-5 mortality is log-transformed. Low-education subsample: "
        "parent_edu < 50%. Ranking by ΔR² (R²(M2) − R²(M1))."
    ),
    "outcomes": all_results,
    "ranking_full": ranking_full,
    "ranking_low_edu": ranking_low,
    "produced": str(pd.Timestamp.now().date()),
    "script": "scripts/robustness/grandparent_effect_all_outcomes.py",
}

outpath = os.path.join(CHECKIN, "grandparent_effect_all_outcomes.json")
with open(outpath, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {outpath}")
