"""
tables/regression_tables.py
=====================

PAPER ANCHOR — Chapter 9 §Universality Across Subsamples (over-performers)
  Question: which countries' children are completing lower-sec far beyond
            what their parents' generation would predict?
  Produces: Table~\\ref{tab:over-performers} — eight over-performers at
            T+25=2015 (Maldives, Cape Verde, Bhutan, Tunisia, Nepal, Viet
            Nam, Bangladesh, India). Uniformly poor (GDP per capita
            $876-$9,645), refuting the income-prerequisite claim.
  Checkin:  checkin/regression_tables.json (and per-outcome companion JSONs).

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
# Outcome-specific lag, biologically anchored. TFR uses biological parent
# timing (LAG_TFR=5; cohort age 20-24 at T is at the reproductive peak by T+5).
# U5MR uses the childrearing window (LAG_CHILDREARING=12; the cohort's own
# children are at risk of under-5 mortality through ~T+10-15). LE uses
# LAG_LE=25, one generation forward (the cohort's own longevity realised across
# its adult life; also disciplines against reverse causality). Child education
# keeps the cross-generation lag (LAG_GENERATION=28, edu→edu one-step).
OUTCOME_LAG = {
    "le":        LAG_LE,
    "tfr":       LAG_TFR,
    "u5mr":      LAG_CHILDREARING,
    "child_edu": LAG_GENERATION,
}
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
    panel = build_panel(
        edu_annual, outcome_df, gdp_df, T_YEARS,
        OUTCOME_LAG[outcome_col], outcome_col,
    )
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

# Child education (cohort-to-cohort, LAG_GENERATION).
panel_ce = build_child_edu_panel(
    edu_annual, gdp_df, T_YEARS, OUTCOME_LAG["child_edu"],
)

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

# Within-country residual at T=1990 (child cohort observed at T+25=2015).
# Paper Table 8 notes label these as "2015 FE residuals" and the table
# notes promise "each country's residual at T=1990"; enforce that here
# rather than taking the last available observation per country.
_resid_sub["resid"] = _resid_sub["ce_dm"] - _beta_op * _resid_sub["edu_dm"]
_t_ref = 1990
_at_ref = _resid_sub[_resid_sub["t"] == _t_ref]
_latest_resid = _at_ref.set_index("country")["resid"]

for wcde_name, label in T3_COUNTRIES.items():
    if wcde_name in _latest_resid.index:
        country_residuals[label] = round(float(_latest_resid[wcde_name]), 1)
    else:
        print(f"  WARNING: {wcde_name} not found in ChildEdu panel")

print(f"\nTable 3 country residuals (ChildEdu, ceiling=90):")
for label, val in sorted(country_residuals.items(), key=lambda x: -abs(x[1])):
    print(f"  {label}: {val:+.1f}")


# ── Common-sample analysis (intersection of all four outcomes) ──────
# The headline `tab:residualisation` reports each outcome on its own
# max-sample (LE/TFR n=822, child edu n=856, U5MR n=787). Coefficients
# across rows are therefore not directly comparable. Build a common
# sample = intersection of country-t rows where edu, log_gdp, and all
# four outcomes are non-null at ceiling=90, then re-run the residualised
# GDP regression. Same exercise with year FE added as a robustness
# variant.

print(f"\n{'=' * 80}")
print("COMMON-SAMPLE ANALYSIS (intersection of all four outcomes)")
print(f"{'=' * 80}")


def _panel_ols_year_fe(x_col, y_col, data):
    """PanelOLS with country AND year fixed effects, clustered SE."""
    if not HAS_LINEARMODELS:
        return None
    sub = data.dropna(subset=[x_col, y_col]).copy()
    counts = sub.groupby("country").size()
    sub = sub[sub["country"].isin(counts[counts >= 2].index)]
    if sub["country"].nunique() < 3 or len(sub) < 10:
        return None
    sub = sub.set_index(["country", "t"])
    try:
        mod = PanelOLS(sub[y_col], sub[[x_col]],
                       entity_effects=True, time_effects=True,
                       drop_absorbed=True)
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
        print(f"    PanelOLS year-FE failed: {e}")
        return None


# Build per-outcome panels (filtered to ceiling=90, entry=10%).
_cohort = entry_years.get(10, {})
_panels_for_intersection = {}
for outcome_label, outcome_col, outcome_df in [
    ("LE", "le", le_df),
    ("TFR", "tfr", tfr_df),
    ("U5MR", "u5mr", u5mr_df),
]:
    p = build_panel(
        edu_annual, outcome_df, gdp_df, T_YEARS,
        OUTCOME_LAG[outcome_col], outcome_col,
    )
    sub = filter_panel(p, _cohort, 90).dropna(
        subset=["edu_t", "log_gdp_t", outcome_col]).copy()
    _panels_for_intersection[outcome_label] = sub

# Child edu intersection key: needs both parent_edu (edu_t) and child_edu.
_ce_filtered = filter_panel(panel_ce, _cohort, 90).dropna(
    subset=["edu_t", "log_gdp_t", "child_edu"]).copy()
_panels_for_intersection["ChildEdu"] = _ce_filtered

# Intersection: (country, t) cells present in ALL four panels.
_common_keys = None
for label, p in _panels_for_intersection.items():
    keys = set(zip(p["country"], p["t"]))
    _common_keys = keys if _common_keys is None else _common_keys & keys

print(f"Common (country, t) cells across all four outcomes: {len(_common_keys)}")

# Filter each panel to the common keys, then merge.
_common_long = None
for label, p in _panels_for_intersection.items():
    keep = p[[(c, t) in _common_keys for c, t in zip(p["country"], p["t"])]].copy()
    rename_col = "child_edu" if label == "ChildEdu" else _panels_for_intersection[label].columns.intersection(["le", "tfr", "u5mr"]).tolist()[0]
    cols_to_keep = ["country", "t", "edu_t", "log_gdp_t", rename_col]
    keep = keep[cols_to_keep].rename(columns={rename_col: rename_col})
    if _common_long is None:
        _common_long = keep
    else:
        # Outer-merge on (country, t, edu_t, log_gdp_t) — these should be
        # identical across panels at the common keys.
        _common_long = _common_long.merge(
            keep[["country", "t", rename_col]],
            on=["country", "t"],
            how="inner",
        )

print(f"Common-sample merged panel: n={len(_common_long)}, "
      f"countries={_common_long['country'].nunique()}")

# Re-run residualised GDP for each outcome on the common sample.
_common_results = {}
for outcome_label, outcome_col in [
    ("LE", "le"), ("TFR", "tfr"), ("U5MR", "u5mr"), ("ChildEdu", "child_edu"),
]:
    if outcome_col not in _common_long.columns:
        continue
    sub_o = _common_long[["country", "t", "edu_t", "log_gdp_t", outcome_col]].copy()
    resid = fe_residualize_gdp(sub_o)

    out_block = {}
    # Country-FE only
    edu_res = panel_ols_result("edu_t", outcome_col, sub_o)
    raw_res = panel_ols_result("log_gdp_t", outcome_col, sub_o)
    if resid is not None:
        sub_r, _ = resid
        resid_res = panel_ols_result("gdp_resid", outcome_col, sub_r)
    else:
        resid_res = None
        sub_r = sub_o

    if edu_res:
        out_block["Education_FE"] = edu_res
    if raw_res:
        out_block["GDP_raw_FE"] = raw_res
    if resid_res:
        out_block["GDP_resid_FE"] = resid_res

    # Country + Year FE (robustness; spec-specific, doesn't generalise to headline)
    edu_res_yfe = _panel_ols_year_fe("edu_t", outcome_col, sub_o)
    if resid is not None:
        resid_res_yfe = _panel_ols_year_fe("gdp_resid", outcome_col, sub_r)
    else:
        resid_res_yfe = None
    if edu_res_yfe:
        out_block["Education_2WFE"] = edu_res_yfe
    if resid_res_yfe:
        out_block["GDP_resid_2WFE"] = resid_res_yfe

    print(f"\n  {outcome_label}:")
    for spec_name, res in out_block.items():
        if res:
            print(f"    {spec_name:<22} β={res['beta']:+.4f}  SE={res['se']:.4f}  "
                  f"p={res['pval']:.4f}  R²={res['r2']:.4f}  n={res['n']}  "
                  f"countries={res['countries']}")
    _common_results[outcome_label] = out_block


write_checkin("regression_tables.json", {
    "method": (
        "Country FE with clustered SEs. β, SE, p-value for education, raw "
        "GDP, residualized GDP. Lower secondary, entry=10%. Outcome-specific "
        f"lag: TFR uses LAG_TFR={LAG_TFR} (biological parent timing); "
        f"U5MR uses LAG_CHILDREARING={LAG_CHILDREARING} (childrearing window); "
        f"LE, child education use LAG_GENERATION={LAG_GENERATION} "
        "(time-to-agency / edu→edu one-step)."
    ),
    "outcome_lag": OUTCOME_LAG,
    "results": {k: {c: {p: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in pred.items()} for p, pred in ceil.items()} for c, ceil in out.items()} for k, out in all_results.items()},
    "country_residuals": country_residuals,
    "common_sample": {
        "n_obs": int(len(_common_long)),
        "n_countries": int(_common_long["country"].nunique()),
        "description": (
            "Common sample = intersection of (country, t) cells where all "
            "four outcomes (LE, TFR, U5MR, child education) AND education "
            "AND log GDP are non-null at ceiling=90, entry=10%. "
            "Specs: Education_FE / GDP_raw_FE / GDP_resid_FE use country FE "
            "only; Education_2WFE / GDP_resid_2WFE add year FE. The 2WFE "
            "year-FE variant is a spec-specific robustness check for the "
            "residualised regression where education has been partialled "
            "out of the regressor; it does not generalise to the headline "
            "spec, which the paper rejects on substantive grounds "
            "(year FE absorb the species-level mechanism)."
        ),
        "outcomes": {
            k: {spec: {kk: round(vv, 4) if isinstance(vv, float) else vv
                       for kk, vv in res.items()}
                for spec, res in block.items()}
            for k, block in _common_results.items()
        },
    },
}, script_path="scripts/tables/regression_tables.py")
