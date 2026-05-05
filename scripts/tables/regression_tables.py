"""
tables/regression_tables.py
=====================
Formal regression output with clustered standard errors.

For each outcome (LE, TFR, U5MR, child education), reports:
  - β coefficient for education
  - β coefficient for GDP
  - β coefficient for residualized GDP
  - Standard errors clustered by country
  - p-values
  - Within-R²

Uses statsmodels PanelOLS (if available) or manual clustering.
Lower secondary, entry=10%, ceilings 60/90.
"""

import os, sys
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from residualization._shared import *

try:
    from linearmodels.panel import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    print("WARNING: linearmodels not installed. Using manual clustered SEs.")
    print("  pip install linearmodels")

T_YEARS = list(range(1960, 1995, 5))
LAG = 25
CEILINGS = [60, 90]
COL_NAME = "lower_sec"

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
le_df = load_wb("life_expectancy_years.csv")
tfr_df = load_wb("children_per_woman_total_fertility.csv")
u5mr_df = load_wb("child_mortality_u5.csv")

edu_annual = interpolate_to_annual(edu_raw, COL_NAME)
entry_years = precompute_entry_years(edu_annual)


def panel_ols_result(x_col, y_col, data):
    """Use linearmodels PanelOLS if available."""
    if not HAS_LINEARMODELS:
        return clustered_fe(x_col, y_col, data)

    sub = data.dropna(subset=[x_col, y_col]).copy()
    counts = sub.groupby("country").size()
    sub = sub[sub["country"].isin(counts[counts >= 2].index)]
    if sub["country"].nunique() < 3 or len(sub) < 10:
        return None

    sub = sub.set_index(["country", "t"])
    try:
        mod = PanelOLS(sub[y_col], sub[[x_col]], entity_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        return {
            "beta": float(res.params[x_col]),
            "se": float(res.std_errors[x_col]),
            "pval": float(res.pvalues[x_col]),
            "r2": float(res.rsquared_within),
            "n": int(res.nobs),
            "countries": int(sub.index.get_level_values(0).nunique()),
        }
    except Exception as e:
        print(f"    PanelOLS failed: {e}, falling back to manual")
        sub = sub.reset_index()
        return clustered_fe(x_col, y_col, sub)


# ── Build panels ─────────────────────────────────────────────────────

all_results = {}

# WB outcomes
for outcome_label, outcome_col, outcome_df in [
    ("LE", "le", le_df),
    ("TFR", "tfr", tfr_df),
    ("U5MR", "u5mr", u5mr_df),
]:
    panel = build_panel(edu_annual, outcome_df, gdp_df, T_YEARS, LAG, outcome_col)
    outcome_results = {}

    for ceiling in CEILINGS:
        print(f"\n{'=' * 80}")
        print(f"{outcome_label}, ceiling={ceiling}%, entry=10%")
        print(f"{'=' * 80}")

        cohort = entry_years.get(10, {})
        sub = filter_panel(panel, cohort, ceiling)

        print(f"{'Predictor':<25} {'β':>10} {'SE':>10} {'p':>10} {'R²':>8} {'n':>6} {'Ctry':>5}")
        print("-" * 80)

        ceil_results = {}
        for pred_label, x_col, use_data in [
            ("Education", "edu_t", sub),
            ("GDP (raw)", "log_gdp_t", sub),
        ]:
            res = panel_ols_result(x_col, outcome_col, use_data)
            if res:
                print(f"{pred_label:<25} {res['beta']:>10.4f} {res['se']:>10.4f} "
                      f"{res['pval']:>10.4f} {res['r2']:>8.3f} {res['n']:>6} {res['countries']:>5}")
                ceil_results[pred_label] = res

        # Residualized GDP
        resid = fe_residualize_gdp(sub)
        if resid is not None:
            sub_r, edu_gdp_r2 = resid
            res = panel_ols_result("gdp_resid", outcome_col, sub_r)
            if res:
                print(f"{'GDP (residualized)':<25} {res['beta']:>10.4f} {res['se']:>10.4f} "
                      f"{res['pval']:>10.4f} {res['r2']:>8.3f} {res['n']:>6} {res['countries']:>5}")
                ceil_results["GDP (residualized)"] = res

        outcome_results[str(ceiling)] = ceil_results

    all_results[outcome_label] = outcome_results

# Child education
panel_ce = build_child_edu_panel(edu_annual, gdp_df, T_YEARS, LAG)

ce_results = {}
for ceiling in CEILINGS:
    print(f"\n{'=' * 80}")
    print(f"Child Education, ceiling={ceiling}%, entry=10%")
    print(f"{'=' * 80}")

    cohort = entry_years.get(10, {})
    sub = filter_panel(panel_ce, cohort, ceiling)

    print(f"{'Predictor':<25} {'β':>10} {'SE':>10} {'p':>10} {'R²':>8} {'n':>6} {'Ctry':>5}")
    print("-" * 80)

    ceil_res = {}
    for pred_label, x_col, use_data in [
        ("Parent Education", "edu_t", sub),
        ("GDP (raw)", "log_gdp_t", sub),
    ]:
        res = panel_ols_result(x_col, "child_edu", use_data)
        if res:
            print(f"{pred_label:<25} {res['beta']:>10.4f} {res['se']:>10.4f} "
                  f"{res['pval']:>10.4f} {res['r2']:>8.3f} {res['n']:>6} {res['countries']:>5}")
            ceil_res[pred_label] = res

    resid = fe_residualize_gdp(sub)
    if resid is not None:
        sub_r, _ = resid
        res = panel_ols_result("gdp_resid", "child_edu", sub_r)
        if res:
            print(f"{'GDP (residualized)':<25} {res['beta']:>10.4f} {res['se']:>10.4f} "
                  f"{res['pval']:>10.4f} {res['r2']:>8.3f} {res['n']:>6} {res['countries']:>5}")
            ceil_res["GDP (residualized)"] = res

    ce_results[str(ceiling)] = ceil_res

all_results["ChildEdu"] = ce_results

# ── Country-level FE residuals for ChildEdu (policy over-performers) ─────
# These are the "policy over-performer" residuals for Table 4 in the paper.
# Uses the FULL child education panel (no entry/ceiling filter) to compute
# within-country FE residuals at the latest time period (T=1990, child at 2015).
# The residual measures how far a country exceeded its own historical trend.

T3_COUNTRIES = {
    "Maldives": "T3-Maldives-resid",
    "Cape Verde": "T3-CapeVerde-resid",
    "Bhutan": "T3-Bhutan-resid",
    "Tunisia": "T3-Tunisia-resid",
    "Nepal": "T3-Nepal-resid",
    "Viet Nam": "T3-Vietnam-resid",
    "Bangladesh": "T3-Bangladesh-resid",
    "India": "T3-India-resid",
    "Qatar": "T3-Qatar-resid",
}

country_residuals = {}

# Build unfiltered child education panel (all countries, all periods)
_resid_sub = panel_ce.dropna(subset=["edu_t", "child_edu"]).copy()
_counts = _resid_sub.groupby("country").size()
_resid_sub = _resid_sub[_resid_sub["country"].isin(_counts[_counts >= 2].index)]

# Country FE via demeaning
_resid_sub["edu_dm"] = _resid_sub["edu_t"] - _resid_sub.groupby("country")["edu_t"].transform("mean")
_resid_sub["ce_dm"] = _resid_sub["child_edu"] - _resid_sub.groupby("country")["child_edu"].transform("mean")

# Regress demeaned child_edu on demeaned parent_edu
_model_op = sm.OLS(_resid_sub["ce_dm"].values, _resid_sub["edu_dm"].values).fit()
_beta_op = float(_model_op.params[0])
print(f"\nOver-performer FE: beta={_beta_op:.3f}, n={len(_resid_sub)}, countries={_resid_sub['country'].nunique()}")

# Within-country residual at latest T for each country
_resid_sub["resid"] = _resid_sub["ce_dm"] - _beta_op * _resid_sub["edu_dm"]
_latest = _resid_sub.sort_values("t").groupby("country").last().reset_index()
_latest_resid = _latest.set_index("country")["resid"]

for wcde_name, label in T3_COUNTRIES.items():
    if wcde_name in _latest_resid.index:
        country_residuals[label] = round(float(_latest_resid[wcde_name]), 1)
    else:
        print(f"  WARNING: {wcde_name} not found in ChildEdu panel")

print(f"\nTable 3 country residuals (ChildEdu, ceiling=90):")
for label, val in sorted(country_residuals.items(), key=lambda x: -abs(x[1])):
    print(f"  {label}: {val:+.1f}")

write_checkin("regression_tables.json", {
    "method": "Country FE with clustered SEs. β, SE, p-value for education, raw GDP, residualized GDP. Lower secondary, entry=10%.",
    "results": {k: {c: {p: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in pred.items()} for p, pred in ceil.items()} for c, ceil in out.items()} for k, out in all_results.items()},
    "country_residuals": country_residuals,
}, script_path="scripts/tables/regression_tables.py")
