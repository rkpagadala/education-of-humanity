"""
residualization/residualization_common_sample.py
================================================

PAPER ANCHOR — Chapter 9 §Education vs GDP at Population Scale (Notes block)
  Question: do the residualised-GDP nulls in Table~\\ref{tab:residualisation}
            survive restriction to the single common sample (n=645, 123
            countries) where every country-year carries all four outcomes?
  Produces: paper Notes claim that on the common sample, residualised-GDP
            within-R^2 <= 0.005 on every outcome and p >= 0.37 on every
            coefficient.
  Checkin:  checkin/residualization_common_sample.json

Common-sample robustness for Table~\\ref{tab:residualisation}.

The headline residualisation table reports four outcomes (LE, TFR,
child education, log U5MR) on outcome-specific panels: n=822 for LE
and TFR, n=856 for child education, n=787 for under-5 mortality.
Reviewer ask (E4): does the conclusion (residualised GDP $R^2 \\le
0.019$, all $p \\ge 0.11$) hold on a single common sample where every
country-year carries all four outcomes?

Method (matches education_vs_gdp.py except for the common-sample
restriction):
  - Build per-outcome panels (T = 1960..1990 step 5). Outcome-specific
    lags: TFR at T+LAG_TFR=5 (biological parent-of-newborn timing); log
    U5MR at T+LAG_CHILDREARING=12 (childrearing window); LE, child
    education at T+LAG_GENERATION=28 (time-to-agency / edu→edu
    transmission).
  - Inner-join on (country, t) over rows where edu(T), log GDP(T),
    LE(T+28), TFR(T+5), child edu(T+28), and U5MR(T+12) are *all*
    observed at their respective lags.
  - Apply the entry-cohort filter (parental completion >= 10%,
    ceiling 90%) used in the headline table.
  - For each of the four outcomes: country-FE residualise log GDP on
    education, then country-FE regress outcome on the GDP residual.

Output: checkin/residualization_common_sample.json with the four-row
table on the common sample plus the per-outcome (headline) sample
sizes for comparison.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from _shared import (
    load_education, load_wb, interpolate_to_annual,
    build_panel, build_child_edu_panel,
    precompute_entry_years, filter_panel,
    fe_r2, fe_residualize_gdp, write_checkin,
    LAG_TFR, LAG_CHILDREARING, LAG_GENERATION,
)


# ── Parameters (match education_vs_gdp.py headline) ──────────────────

T_YEARS = list(range(1960, 1995, 5))
ENTRY = 10       # parental completion >= 10%
CEILING = 90     # parental completion <= 90%
EDU_COL = "lower_sec"   # lower-secondary completion, both sexes

# Outcome → lag. TFR uses LAG_TFR=5 (biological parent-of-newborn timing).
# U5MR (level and log) uses LAG_CHILDREARING=12 (childrearing window). LE and
# child education use LAG_GENERATION=28 (time-to-agency / edu→edu transmission).
OUTCOME_LAG = {
    "le_tp28":       LAG_GENERATION,
    "tfr_tp5":      LAG_TFR,
    "u5mr_tp12":     LAG_CHILDREARING,
    "log_u5mr_tp12": LAG_CHILDREARING,
    "ced_tp28":      LAG_GENERATION,
}


# ── Load data ────────────────────────────────────────────────────────

print("Loading data...")
edu_raw  = load_education("completion_both_long.csv")
le_raw   = load_wb("life_expectancy_years.csv")
tfr_raw  = load_wb("children_per_woman_total_fertility.csv")
u5mr_raw = load_wb("child_mortality_u5.csv")
gdp_raw  = load_wb("gdppercapita_us_inflation_adjusted.csv")

edu_annual = interpolate_to_annual(edu_raw, EDU_COL)

# Per-outcome panels (matches headline columns). Column suffixes name the
# outcome's biological horizon: tfr_tp5 (T+5), u5mr_tp12 (T+12), le_tp28 and
# ced_tp28 (T+28, one 28-year generation).
panel_le    = build_panel(edu_annual, le_raw,   gdp_raw, T_YEARS,
                          OUTCOME_LAG["le_tp28"],   "le_tp28")
panel_tfr   = build_panel(edu_annual, tfr_raw,  gdp_raw, T_YEARS,
                          OUTCOME_LAG["tfr_tp5"],  "tfr_tp5")
panel_u5mr  = build_panel(edu_annual, u5mr_raw, gdp_raw, T_YEARS,
                          OUTCOME_LAG["u5mr_tp12"], "u5mr_tp12")
panel_ced   = build_child_edu_panel(edu_annual, gdp_raw, T_YEARS,
                                     OUTCOME_LAG["ced_tp28"])
panel_ced   = panel_ced.rename(columns={"child_edu": "ced_tp28"})

# Apply log transforms to match Table 13's transformation choices
panel_u5mr["log_u5mr_tp12"] = np.log(panel_u5mr["u5mr_tp12"].where(
    panel_u5mr["u5mr_tp12"] > 0))

# ── Common-sample inner join ─────────────────────────────────────────

key = ["country", "t"]
merged = (
    panel_le[key + ["edu_t", "log_gdp_t", "le_tp28"]]
    .merge(panel_tfr[key + ["tfr_tp5"]], on=key, how="inner")
    .merge(panel_u5mr[key + ["u5mr_tp12", "log_u5mr_tp12"]], on=key, how="inner")
    .merge(panel_ced[key + ["ced_tp28"]], on=key, how="inner")
)
print(f"\nMerged panel before entry-cohort filter: "
      f"n={len(merged)}, countries={merged['country'].nunique()}")

# Entry-cohort filter at threshold 10, ceiling 90 (headline filter)
entry_years = precompute_entry_years(edu_annual)
cohort = entry_years[ENTRY]
common = filter_panel(merged, cohort, CEILING)
common = common.dropna(subset=["edu_t", "log_gdp_t", "le_tp28",
                                "tfr_tp5", "log_u5mr_tp12", "ced_tp28"])
print(f"Common sample (entry={ENTRY}%, ceiling={CEILING}%): "
      f"n={len(common)}, countries={common['country'].nunique()}")


# ── Residualisation on common sample ─────────────────────────────────

def resid_and_fit(panel, outcome_col):
    """Country-FE residualise log GDP on education, then country-FE
    regress outcome on the GDP residual. Returns (edu_r2, raw_gdp_r2,
    resid_gdp_r2, edu_gdp_r2, p_resid, n, n_countries)."""
    edu_r2, n_obs, n_c = fe_r2("edu_t", outcome_col, panel)
    raw_r2, _, _       = fe_r2("log_gdp_t", outcome_col, panel)
    out = fe_residualize_gdp(panel)
    if out is None:
        return dict(edu_r2=edu_r2, raw_gdp_r2=raw_r2,
                    resid_gdp_r2=np.nan, edu_gdp_r2=np.nan,
                    n=n_obs, n_countries=n_c)
    sub, edu_gdp_r2 = out
    sub = sub.dropna(subset=["gdp_resid", outcome_col]).copy()
    resid_r2, n_r, n_cr = fe_r2("gdp_resid", outcome_col, sub)

    # Compute p-value for residual GDP coefficient (country-clustered)
    # Wald-style via demeaned OLS, clustered SE by country
    def cluster_p(panel, x_col, y_col):
        from _shared import _demean_and_filter
        result = _demean_and_filter(panel, [x_col, y_col])
        if result is None:
            return np.nan
        s, dm, _ = result
        X = dm[x_col].to_numpy()
        Y = dm[y_col].to_numpy()
        ok = ~np.isnan(X) & ~np.isnan(Y)
        X = X[ok]; Y = Y[ok]
        if len(X) < 10:
            return np.nan
        beta = float(np.dot(X, Y) / np.dot(X, X))
        resid = Y - beta * X
        # cluster SE
        codes, _ = pd.factorize(s.iloc[ok]["country"], sort=False)
        xe = X * resid
        sum_g = np.bincount(codes, weights=xe)
        meat = float(np.dot(sum_g, sum_g))
        bread = float(np.dot(X, X)) ** 2
        var = meat / bread
        se = np.sqrt(var) if var > 0 else np.nan
        from scipy import stats
        if not np.isnan(se) and se > 0:
            return float(2 * (1 - stats.norm.cdf(abs(beta) / se)))
        return np.nan

    p_resid = cluster_p(sub, "gdp_resid", outcome_col)
    return dict(edu_r2=edu_r2, raw_gdp_r2=raw_r2,
                resid_gdp_r2=resid_r2, edu_gdp_r2=edu_gdp_r2,
                p_resid=p_resid, n=n_r, n_countries=n_cr)


outcomes = [
    ("life_expectancy",   "le_tp28"),
    ("tfr",               "tfr_tp5"),
    ("child_education",   "ced_tp28"),
    ("log_u5mr",          "log_u5mr_tp12"),
]

results = {}
print("\nCommon-sample residualisation results:")
print(f"{'Outcome':<22} {'Edu R²':>8} {'Raw GDP R²':>11} "
      f"{'Resid R²':>10} {'p(resid)':>10} {'n':>5} {'cn':>4}")
for name, col in outcomes:
    r = resid_and_fit(common, col)
    results[name] = r
    print(f"{name:<22} {r['edu_r2']:>8.3f} {r['raw_gdp_r2']:>11.3f} "
          f"{r['resid_gdp_r2']:>10.3f} {r.get('p_resid', float('nan')):>10.3f} "
          f"{r['n']:>5d} {r['n_countries']:>4d}")


# ── Headline sample sizes (for comparison) ───────────────────────────

print("\nPer-outcome headline sample sizes (from existing checkins):")
headline_n = {
    "life_expectancy":  {"n": 822, "n_countries": 152},
    "tfr":              {"n": 822, "n_countries": 152},
    "child_education":  {"n": 856, "n_countries": 157},
    "log_u5mr":         {"n": 787, "n_countries": 147},
}
for k, v in headline_n.items():
    print(f"  {k:<22} n={v['n']}, countries={v['n_countries']}")

# ── Checkin ──────────────────────────────────────────────────────────

passes = all(
    (r["resid_gdp_r2"] <= 0.02 and (np.isnan(r.get("p_resid", np.nan))
                                    or r.get("p_resid", np.nan) >= 0.10))
    for r in results.values()
)
print(f"\nConclusion (Table residualisation) holds on common sample: {passes}")

# Use the effective FE-regression sample (after singleton drops) as
# the headline common-sample size, since that's what the regressions
# actually run on. All four outcomes converge on the same effective n
# because they share the same edu/log-GDP/outcome NaN pattern after
# the inner-join.
effective_n = next(iter(results.values()))["n"]
effective_c = next(iter(results.values()))["n_countries"]

checkin = {
    "method": (
        "Common-sample robustness for the Residualised-GDP table. "
        "Inner-join on (country, t) over rows where LE (T+28), TFR (T+5), "
        "child education (T+28), and U5MR (T+12) are ALL observed, given "
        "edu(T) and log GDP(T). Outcome-specific lags: TFR uses LAG_TFR=5 "
        "(biological parent-of-newborn timing); U5MR uses LAG_CHILDREARING=12 "
        "(childrearing window); LE and child education use LAG_GENERATION=28 "
        "(time-to-agency / edu→edu transmission). Entry-cohort filter "
        "(entry=10, ceiling=90) matches headline. Country-FE Frisch-Waugh-"
        "Lovell residualisation, country-clustered standard errors for the "
        "residual-GDP p-value."
    ),
    "outcome_lags": OUTCOME_LAG,
    "common_sample": {
        "n":           int(effective_n),
        "n_countries": int(effective_c),
        "n_pre_fe_filter":           int(common.shape[0]),
        "n_countries_pre_fe_filter": int(common["country"].nunique()),
    },
    "headline_sample_sizes": headline_n,
    "results": results,
    "conclusion_holds": bool(passes),
}

write_checkin(
    "residualization_common_sample.json", checkin,
    "scripts/residualization/residualization_common_sample.py",
)
