# =============================================================================
# PAPER REFERENCE
# Script:  scripts/robustness/log_outcomes.py
# Paper:   "Education of Humanity"
#
# Produces:
#   Log-linear robustness for the Table 2 forward-prediction specification.
#   For LE, TFR, and U5MR at T+25, runs the headline regression
#
#     outcome_{t+25} = β · parent_edu_t + γ · log_gdp_t + α_i + u_t
#
#   with the outcome in logs (elasticity/semi-elasticity form) alongside
#   the levels comparator. Education remains in percentage points (0-100),
#   so β on parent is a semi-elasticity: "% change in outcome per pp rise
#   in parent completion."
#
#   This addresses the reviewer's point that log-transforming continuous
#   non-percentage outcomes gives elasticity interpretation and improves
#   consistency across the tables.
#
# Output: checkin/log_outcomes.json
# =============================================================================
"""Log(outcome) robustness for LE, TFR, U5MR with country FE."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from _shared import (  # noqa: E402
    load_wb, load_education, interpolate_to_annual, write_checkin,
    get_wb_val,
)

PARENTAL_LAG = 25
T_YEARS = list(range(1975, 1991, 5))


def build_panel(edu_annual, outcome_df, gdp_df, outcome_name):
    rows = []
    for c in sorted(edu_annual.keys()):
        s = edu_annual[c]
        for t in T_YEARS:
            if t not in s.index:
                continue
            parent = float(s[t])
            gdp = get_wb_val(gdp_df, c, t)
            outcome = get_wb_val(outcome_df, c, t + PARENTAL_LAG)
            if any(pd.isna(v) for v in (parent, outcome)):
                continue
            if np.isnan(gdp) or gdp <= 0 or outcome <= 0:
                continue
            rows.append({
                "country": c,
                "year":    t,
                "parent":  parent,
                "log_gdp": np.log(gdp),
                outcome_name: outcome,
                f"log_{outcome_name}": np.log(outcome),
            })
    return pd.DataFrame(rows)


def fit(d, x_cols, y_col):
    panel = d.set_index(["country", "year"])
    mod = PanelOLS(panel[y_col], panel[x_cols],
                   entity_effects=True, drop_absorbed=True, check_rank=False)
    return mod.fit(cov_type="clustered", cluster_entity=True)


def report(df, outcome_name):
    lvl = fit(df, ["parent", "log_gdp"], outcome_name)
    log = fit(df, ["parent", "log_gdp"], f"log_{outcome_name}")
    return {
        "levels": {
            "parent_beta": round(float(lvl.params["parent"]), 4),
            "parent_se":   round(float(lvl.std_errors["parent"]), 4),
            "parent_p":    round(float(lvl.pvalues["parent"]), 4),
            "log_gdp_beta": round(float(lvl.params["log_gdp"]), 4),
            "log_gdp_se":   round(float(lvl.std_errors["log_gdp"]), 4),
            "log_gdp_p":    round(float(lvl.pvalues["log_gdp"]), 4),
            "r2_within":    round(float(lvl.rsquared_within), 4),
            "n": int(lvl.nobs),
            "countries": int(lvl.entity_info.total),
        },
        "log": {
            "parent_beta": round(float(log.params["parent"]), 5),
            "parent_se":   round(float(log.std_errors["parent"]), 5),
            "parent_p":    round(float(log.pvalues["parent"]), 4),
            "parent_semi_elast_pct_per_pp": round(
                100 * float(log.params["parent"]), 3),
            "parent_semi_elast_se_pct": round(
                100 * float(log.std_errors["parent"]), 3),
            "log_gdp_beta": round(float(log.params["log_gdp"]), 4),
            "log_gdp_se":   round(float(log.std_errors["log_gdp"]), 4),
            "log_gdp_p":    round(float(log.pvalues["log_gdp"]), 4),
            "r2_within":    round(float(log.rsquared_within), 4),
            "n": int(log.nobs),
            "countries": int(log.entity_info.total),
        },
    }


def main():
    print("Loading data...")
    edu_raw = load_education("completion_both_long.csv")
    edu_annual = interpolate_to_annual(edu_raw, "lower_sec")
    gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
    le_df = load_wb("life_expectancy_years.csv")
    tfr_df = load_wb("children_per_woman_total_fertility.csv")
    u5mr_df = load_wb("child_mortality_u5.csv")

    le_panel = build_panel(edu_annual, le_df, gdp_df, "le")
    tfr_panel = build_panel(edu_annual, tfr_df, gdp_df, "tfr")
    u5mr_panel = build_panel(edu_annual, u5mr_df, gdp_df, "u5mr")

    print(f"LE panel:   {len(le_panel):4d} obs, "
          f"{le_panel['country'].nunique()} countries")
    print(f"TFR panel:  {len(tfr_panel):4d} obs, "
          f"{tfr_panel['country'].nunique()} countries")
    print(f"U5MR panel: {len(u5mr_panel):4d} obs, "
          f"{u5mr_panel['country'].nunique()} countries")

    le = report(le_panel, "le")
    tfr = report(tfr_panel, "tfr")
    u5mr = report(u5mr_panel, "u5mr")

    fmt = ("  {:12s}  β_lvl={:+9.4f} (SE {:.4f})   β_log={:+9.5f} "
           "(SE {:.5f})   ≈{:+6.3f}% per pp")
    print("\nResults (β on parental completion, country FE + log GDP control):")
    for name, r in [("LE", le), ("TFR", tfr), ("U5MR", u5mr)]:
        print(fmt.format(
            name,
            r["levels"]["parent_beta"], r["levels"]["parent_se"],
            r["log"]["parent_beta"],    r["log"]["parent_se"],
            r["log"]["parent_semi_elast_pct_per_pp"]))

    write_checkin("log_outcomes.json", {
        "notes": (
            "Log(outcome) robustness for forward-prediction regressions. "
            "outcome_{t+25} on parent_edu_t + log_gdp_t + country FE, "
            "country-clustered SEs. β_log interpreted as semi-elasticity: "
            "percent change in outcome per 1 pp rise in parental completion. "
            "Education remains in percentage points (0-100). "
            "Sample: 1975-1990 base years (2000-2015 outcome years)."
        ),
        "results": {"le": le, "tfr": tfr, "u5mr": u5mr},
        "numbers": {
            "le_levels_parent_beta": le["levels"]["parent_beta"],
            "le_log_parent_beta":    le["log"]["parent_beta"],
            "le_log_semi_elast_pct": le["log"]["parent_semi_elast_pct_per_pp"],
            "le_log_semi_elast_se_pct": le["log"]["parent_semi_elast_se_pct"],
            "le_n":         le["levels"]["n"],
            "le_countries": le["levels"]["countries"],
            "tfr_levels_parent_beta": tfr["levels"]["parent_beta"],
            "tfr_log_parent_beta":    tfr["log"]["parent_beta"],
            "tfr_log_semi_elast_pct": tfr["log"]["parent_semi_elast_pct_per_pp"],
            "tfr_log_semi_elast_se_pct": tfr["log"]["parent_semi_elast_se_pct"],
            "tfr_n":         tfr["levels"]["n"],
            "tfr_countries": tfr["levels"]["countries"],
            "u5mr_levels_parent_beta": u5mr["levels"]["parent_beta"],
            "u5mr_log_parent_beta":    u5mr["log"]["parent_beta"],
            "u5mr_log_semi_elast_pct": u5mr["log"]["parent_semi_elast_pct_per_pp"],
            "u5mr_log_semi_elast_se_pct": u5mr["log"]["parent_semi_elast_se_pct"],
            "u5mr_n":         u5mr["levels"]["n"],
            "u5mr_countries": u5mr["levels"]["countries"],
        },
    }, script_path="scripts/robustness/log_outcomes.py")
    print("\nDone.")


if __name__ == "__main__":
    main()
