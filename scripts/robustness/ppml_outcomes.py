# =============================================================================
# PAPER REFERENCE
# Script:  scripts/robustness/ppml_outcomes.py
# Paper:   "Education of Humanity"
#
# Produces:
#   Poisson Pseudo-Maximum Likelihood (Santos Silva & Tenreyro 2006, REStat)
#   robustness check for the two right-skewed, non-negative outcomes:
#     - Total fertility rate at T+25
#     - Under-5 mortality at T+25  (per 1,000 live births)
#
#   Specification (multiplicative form):
#     E[y_{i,t+25} | x] = exp(α_i + β_edu · parent_edu_t + β_gdp · log_gdp_t)
#
#   Reports β, SE, p-value, and n/countries. Country-clustered sandwich SEs.
#   Country fixed effects implemented via country dummies (absorbed by GLM).
#   The OLS log-linear comparator appears in log_outcomes.json.
#
# Why PPML: the standard log-OLS specification drops zeros and is sensitive
# to heteroscedasticity in E[log y] (Jensen gap); PPML is consistent under
# arbitrary heteroscedasticity provided the conditional mean is correctly
# specified. For outcomes bounded at zero and positively skewed, this is
# the standard robustness.
#
# Output: checkin/ppml_outcomes.json
# =============================================================================
"""PPML robustness for TFR and U5MR (country FE, clustered SE)."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from _shared import (  # noqa: E402
    PROC, load_wb, load_education, interpolate_to_annual, write_checkin,
    get_wb_val,
)

PARENTAL_LAG = 25
T_YEARS = list(range(1975, 1991, 5))  # outcome year = t+25 ∈ 2000..2015


def build_panel(edu_annual, outcome_df, gdp_df):
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
            if np.isnan(gdp) or gdp <= 0:
                continue
            rows.append({
                "country": c,
                "year":    t,
                "parent":  parent,
                "log_gdp": np.log(gdp),
                "outcome": outcome,
            })
    return pd.DataFrame(rows)


def ppml(df, label):
    """Poisson GLM with country dummies; sandwich SEs clustered by country."""
    d = df.copy()
    # Drop non-positive outcomes (PPML requires y >= 0; the data have
    # no exact zeros for TFR/U5MR in this window, but be defensive).
    d = d[d["outcome"] >= 0].copy()
    X = pd.get_dummies(d["country"], prefix="c", drop_first=True).astype(float)
    X["parent"]  = d["parent"].values
    X["log_gdp"] = d["log_gdp"].values
    X = sm.add_constant(X, has_constant="add")

    model = sm.GLM(d["outcome"].values, X.values,
                   family=sm.families.Poisson())
    res = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": d["country"].values},
    )
    # extract coefficients by column name
    cols = list(X.columns)
    idx_p   = cols.index("parent")
    idx_g   = cols.index("log_gdp")

    beta_p  = float(res.params[idx_p])
    se_p    = float(res.bse[idx_p])
    p_p     = float(res.pvalues[idx_p])
    beta_g  = float(res.params[idx_g])
    se_g    = float(res.bse[idx_g])
    p_g     = float(res.pvalues[idx_g])

    n = len(d)
    n_ctry = d["country"].nunique()
    print(f"\n{label}: n={n}, countries={n_ctry}")
    print(f"  β(parent)   = {beta_p:+.5f}   SE={se_p:.5f}   p={p_p:.4f}")
    print(f"                (semi-elasticity: 1 pp rise in parent edu → "
          f"{100*(np.exp(beta_p)-1):+.3f}% change in {label})")
    print(f"  β(log_gdp)  = {beta_g:+.4f}    SE={se_g:.4f}   p={p_g:.4f}")
    return {
        "n": n,
        "countries": n_ctry,
        "parent_beta": round(beta_p, 5),
        "parent_se":   round(se_p, 5),
        "parent_p":    round(p_p, 4),
        "parent_semi_elast_pct": round(100 * (np.exp(beta_p) - 1), 4),
        "parent_semi_elast_se_pct": round(100 * np.exp(beta_p) * se_p, 4),
        "log_gdp_beta": round(beta_g, 4),
        "log_gdp_se":   round(se_g, 4),
        "log_gdp_p":    round(p_g, 4),
    }


def main():
    print("Loading data...")
    edu_raw = load_education("completion_both_long.csv")
    edu_annual = interpolate_to_annual(edu_raw, "lower_sec")
    gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
    tfr_df = load_wb("children_per_woman_total_fertility.csv")
    u5mr_df = load_wb("child_mortality_u5.csv")

    tfr_panel = build_panel(edu_annual, tfr_df, gdp_df)
    u5mr_panel = build_panel(edu_annual, u5mr_df, gdp_df)
    print(f"TFR panel:  {len(tfr_panel):4d} obs, "
          f"{tfr_panel['country'].nunique()} countries")
    print(f"U5MR panel: {len(u5mr_panel):4d} obs, "
          f"{u5mr_panel['country'].nunique()} countries")

    tfr = ppml(tfr_panel, "TFR(T+25)")
    u5mr = ppml(u5mr_panel, "U5MR(T+25)")

    write_checkin("ppml_outcomes.json", {
        "notes": (
            "Poisson Pseudo-Maximum Likelihood (Santos Silva & Tenreyro 2006) "
            "with country dummies and country-clustered sandwich SEs. "
            "Outcomes at T+25: total fertility rate, under-5 mortality. "
            "Sample: 1975-1990 base years (2000-2015 outcome years); "
            "WCDE v3 lower secondary completion + WB WDI."
        ),
        "results": {"tfr": tfr, "u5mr": u5mr},
        "numbers": {
            "tfr_parent_beta": tfr["parent_beta"],
            "tfr_parent_se":   tfr["parent_se"],
            "tfr_parent_p":    tfr["parent_p"],
            "tfr_parent_semi_elast_pct": tfr["parent_semi_elast_pct"],
            "tfr_parent_semi_elast_se_pct": tfr["parent_semi_elast_se_pct"],
            "tfr_log_gdp_beta": tfr["log_gdp_beta"],
            "tfr_n":         tfr["n"],
            "tfr_countries": tfr["countries"],
            "u5mr_parent_beta": u5mr["parent_beta"],
            "u5mr_parent_se":   u5mr["parent_se"],
            "u5mr_parent_p":    u5mr["parent_p"],
            "u5mr_parent_semi_elast_pct": u5mr["parent_semi_elast_pct"],
            "u5mr_parent_semi_elast_se_pct": u5mr["parent_semi_elast_se_pct"],
            "u5mr_log_gdp_beta": u5mr["log_gdp_beta"],
            "u5mr_n":         u5mr["n"],
            "u5mr_countries": u5mr["countries"],
        },
    }, script_path="scripts/robustness/ppml_outcomes.py")
    print("\nDone.")


if __name__ == "__main__":
    main()
