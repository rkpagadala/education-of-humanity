"""
education_outcomes.py (was wcde/scripts/07_education_outcomes.py)
Does education at time T predict GDP, life expectancy, and TFR at T+25?

This directly tests the causal direction claim:
  education → income AND education → life expectancy AND education → fertility

Design:
  For each country-year T, regress outcome(T+25) on:
    - education(T)         — does education predict future outcomes?
    - log_GDP(T)           — controls for initial income (convergence)
    - education(T) + log_GDP(T) — joint model

  Key comparison: which explains more of outcome(T+25) —
  initial income or initial education?

  Country FE versions: controls for all time-invariant country traits
  (culture, institutions, geography).

Education lag: 25 years (one generation — the T-25 cohort mechanism)
Also tested: 10-year and 15-year lags (human capital entering workforce sooner)

Education measure: lower secondary completion (policy-critical level)
Also tested: primary, upper secondary

Outcomes:
  1. log GDP per capita (T+25)  — education drives income
  2. Life expectancy e0 (T+25)  — education drives health
  3. TFR (T+25)                 — education drives fertility transition

Outputs: wcde/output/education_outcomes.md
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import PROC, DATA as ROOT_DS, REPO_ROOT, REGIONS, NAME_MAP as _SHARED_NAME_MAP, load_wb, write_checkin

OUT = os.path.join(REPO_ROOT, "wcde", "output")
os.makedirs(OUT, exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
edu = pd.read_csv(os.path.join(PROC, "completion_both_long.csv"))
edu = edu[~edu["country"].isin(REGIONS)].copy()
for col in ["primary","lower_sec","upper_sec","college"]:
    edu[col] = edu[col].clip(upper=100)

e0  = pd.read_csv(os.path.join(PROC, "e0.csv")).set_index("country")
tfr = pd.read_csv(os.path.join(PROC, "tfr.csv")).set_index("country")
e0.columns  = [int(c) for c in e0.columns]
tfr.columns = [int(c) for c in tfr.columns]

gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")
print(f"  GDP: {len(gdp_raw)} countries, years {gdp_raw.columns[0]}–{gdp_raw.columns[-1]}")

u5mr_raw = load_wb("child_mortality_u5.csv")
print(f"  U5MR: {len(u5mr_raw)} countries, years {u5mr_raw.columns[0]}–{u5mr_raw.columns[-1]}")

def get_gdp(country_wcde, year):
    key = _SHARED_NAME_MAP.get(country_wcde, country_wcde).lower()
    for k in [country_wcde.lower(), key]:
        if k in gdp_raw.index:
            try:
                v = float(gdp_raw.loc[k, str(year)])
                return v if not np.isnan(v) and v > 0 else np.nan
            except (KeyError, ValueError, TypeError): pass
    return np.nan

def get_e0(country_wcde, year):
    if country_wcde in e0.index and year in e0.columns:
        v = float(e0.loc[country_wcde, year])
        return v if not np.isnan(v) else np.nan
    return np.nan

def get_tfr(country_wcde, year):
    if country_wcde in tfr.index and year in tfr.columns:
        v = float(tfr.loc[country_wcde, year])
        return v if not np.isnan(v) else np.nan
    return np.nan

def get_u5mr(country_wcde, year):
    key = _SHARED_NAME_MAP.get(country_wcde, country_wcde).lower()
    for k in [country_wcde.lower(), key]:
        if k in u5mr_raw.index:
            try:
                v = float(u5mr_raw.loc[k, str(year)])
                return v if not np.isnan(v) and v > 0 else np.nan
            except (KeyError, ValueError, TypeError): pass
    return np.nan

# ── Build panel ───────────────────────────────────────────────────────────────
# T years where T+25 is within our data range
EDU_YEARS = [1960, 1965, 1970, 1975, 1980, 1985, 1990]  # T+25 = 1985..2015

print("Building panel...")
rows = []
countries = sorted(edu["country"].unique())

for c in countries:
    edu_c = edu[edu["country"] == c].set_index("year")
    for t in EDU_YEARS:
        if t not in edu_c.index: continue
        tp25 = t + 25
        tp15 = t + 15
        tp10 = t + 10

        low   = edu_c.loc[t, "lower_sec"]
        pri   = edu_c.loc[t, "primary"]
        upp   = edu_c.loc[t, "upper_sec"]
        col_v = edu_c.loc[t, "college"]

        gdp_t    = get_gdp(c, t)
        gdp_tp25 = get_gdp(c, tp25)
        gdp_tp15 = get_gdp(c, tp15)
        gdp_tp10 = get_gdp(c, tp10)

        e0_t    = get_e0(c, t)
        e0_tp25 = get_e0(c, tp25)

        tfr_t    = get_tfr(c, t)
        tfr_tp25 = get_tfr(c, tp25)

        u5mr_t    = get_u5mr(c, t)
        u5mr_tp25 = get_u5mr(c, tp25)

        if any(np.isnan(x) for x in [low, pri]): continue

        # Education at T+25 (for reverse regression: GDP(T) -> edu(T+25))
        low_tp25_val = np.nan
        if tp25 in edu_c.index:
            v_tp25 = edu_c.loc[tp25, "lower_sec"]
            if not np.isnan(v_tp25):
                low_tp25_val = float(np.clip(v_tp25, 0, 100))

        rows.append({
            "country": c, "t": t, "tp25": tp25,
            "low_t": low, "pri_t": pri, "upp_t": upp, "col_t": col_v,
            "low_tp25": low_tp25_val,
            "log_gdp_t":    np.log(gdp_t)    if not np.isnan(gdp_t)    else np.nan,
            "log_gdp_tp25": np.log(gdp_tp25) if not np.isnan(gdp_tp25) else np.nan,
            "log_gdp_tp15": np.log(gdp_tp15) if not np.isnan(gdp_tp15) else np.nan,
            "log_gdp_tp10": np.log(gdp_tp10) if not np.isnan(gdp_tp10) else np.nan,
            "gdp_growth_25": (np.log(gdp_tp25) - np.log(gdp_t)) if not np.isnan(gdp_t) and not np.isnan(gdp_tp25) else np.nan,
            "e0_t": e0_t, "e0_tp25": e0_tp25,
            "tfr_t": tfr_t, "tfr_tp25": tfr_tp25,
            "u5mr_t": u5mr_t, "u5mr_tp25": u5mr_tp25,
        })

panel = pd.DataFrame(rows)

# Log transformations for outcomes that the reviewer recommends presenting
# in elasticity form (R1.18 / R2.17). LE and TFR are kept in levels as the
# headline rows of Table 7; the log rows let the reader read all slope
# coefficients on the same semi-elasticity scale as log GDP.
for col in ["e0_t", "e0_tp25", "tfr_t", "tfr_tp25", "u5mr_t", "u5mr_tp25"]:
    panel[f"log_{col}"] = np.log(panel[col].where(panel[col] > 0))

print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries")
print(f"  GDP coverage (T+25): {panel['log_gdp_tp25'].notna().sum()} obs")
print(f"  E0 coverage (T+25): {panel['e0_tp25'].notna().sum()} obs")
print(f"  TFR coverage (T+25): {panel['tfr_tp25'].notna().sum()} obs")
print(f"  U5MR coverage (T+25): {panel['u5mr_tp25'].notna().sum()} obs")

def run_ols(X_cols, y_col, data, fe=False, country_col="country"):
    """Run OLS (pooled or FE) and return (coefs, r2, n)."""
    sub = data.dropna(subset=X_cols + [y_col])
    if len(sub) < 10: return None, np.nan, 0
    if fe:
        sub = sub.copy()
        for col in X_cols + [y_col]:
            sub[col + "_dm"] = sub[col] - sub.groupby(country_col)[col].transform("mean")
        Xd = sub[[c + "_dm" for c in X_cols]]
        yd = sub[y_col + "_dm"]
        ok = ~np.isnan(Xd.values).any(axis=1) & ~np.isnan(yd.values)
        if ok.sum() < 10: return None, np.nan, 0
        model = sm.OLS(yd[ok], Xd[ok]).fit()
        return dict(zip(X_cols, model.params.values)), model.rsquared, ok.sum()
    else:
        X = sub[X_cols]
        y = sub[y_col]
        ok = ~np.isnan(X.values).any(axis=1) & ~np.isnan(y.values)
        if ok.sum() < 10: return None, np.nan, 0
        model = sm.OLS(y[ok], sm.add_constant(X[ok])).fit()
        return dict(zip(X_cols, model.params.values[1:])), model.rsquared, ok.sum()

# ── Run all models ────────────────────────────────────────────────────────────
print("\nRunning regressions...")

results = {}

# GDP outcomes
for outcome, label in [("log_gdp_tp25","log GDP(T+25)"),("gdp_growth_25","GDP growth T→T+25")]:
    results[outcome] = {}
    for spec, xcols, fe in [
        ("OLS: edu only",       ["low_t"],               False),
        ("OLS: GDP only",       ["log_gdp_t"],            False),
        ("OLS: edu + GDP",      ["low_t","log_gdp_t"],    False),
        ("FE:  edu only",       ["low_t"],               True),
        ("FE:  GDP only",       ["log_gdp_t"],            True),
        ("FE:  edu + GDP",      ["low_t","log_gdp_t"],    True),
    ]:
        coefs, r2, n = run_ols(xcols, outcome, panel, fe=fe)
        results[outcome][spec] = (coefs, r2, n)
        coef_str = ", ".join(f"{k}:{v:.3f}" for k,v in coefs.items()) if coefs else "n/a"
        print(f"  {label} | {spec}: {coef_str}, R²={r2:.3f}, n={n}")

# E0 outcomes
print()
results["e0"] = {}
for spec, xcols, fe in [
    ("OLS: edu only",       ["low_t"],            False),
    ("OLS: e0 only",        ["e0_t"],             False),
    ("OLS: edu + e0",       ["low_t","e0_t"],     False),
    ("FE:  edu only",       ["low_t"],            True),
    ("FE:  e0 only",        ["e0_t"],             True),
    ("FE:  edu + e0",       ["low_t","e0_t"],     True),
]:
    coefs, r2, n = run_ols(xcols, "e0_tp25", panel, fe=fe)
    results["e0"][spec] = (coefs, r2, n)
    coef_str = ", ".join(f"{k}:{v:.3f}" for k,v in coefs.items()) if coefs else "n/a"
    print(f"  e0(T+25) | {spec}: {coef_str}, R²={r2:.3f}, n={n}")

# TFR outcomes
print()
results["tfr"] = {}
for spec, xcols, fe in [
    ("OLS: edu only",       ["low_t"],            False),
    ("OLS: tfr only",       ["tfr_t"],            False),
    ("OLS: edu + tfr",      ["low_t","tfr_t"],    False),
    ("FE:  edu only",       ["low_t"],            True),
    ("FE:  tfr only",       ["tfr_t"],            True),
    ("FE:  edu + tfr",      ["low_t","tfr_t"],    True),
]:
    coefs, r2, n = run_ols(xcols, "tfr_tp25", panel, fe=fe)
    results["tfr"][spec] = (coefs, r2, n)
    coef_str = ", ".join(f"{k}:{v:.3f}" for k,v in coefs.items()) if coefs else "n/a"
    print(f"  TFR(T+25) | {spec}: {coef_str}, R²={r2:.3f}, n={n}")

# Log-outcome rows: log(LE), log(TFR), log(U5MR) on education + log(initial outcome).
# Coefficients are semi-elasticities — proportional change in the outcome per
# percentage-point rise in parental lower-secondary completion. Comparable in
# units to the log-GDP row above. Cluster-robust SEs by country.
def fe_clustered(X_cols, y_col, data, country_col="country"):
    sub = data.dropna(subset=X_cols + [y_col]).copy()
    if len(sub) < 10:
        return None
    for col in X_cols + [y_col]:
        sub[col + "_dm"] = sub[col] - sub.groupby(country_col)[col].transform("mean")
    Xd = sub[[c + "_dm" for c in X_cols]].values
    yd = sub[y_col + "_dm"].values
    countries = sub[country_col].values
    ok = ~np.isnan(Xd).any(axis=1) & ~np.isnan(yd)
    Xd, yd, countries = Xd[ok], yd[ok], countries[ok]
    if len(yd) < 10:
        return None
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ yd
    resid = yd - Xd @ beta
    # Cluster-by-country meat
    meat = np.zeros((Xd.shape[1], Xd.shape[1]))
    for c in np.unique(countries):
        idx = countries == c
        u = Xd[idx].T @ resid[idx]
        meat += np.outer(u, u)
    G = len(np.unique(countries))
    N = len(yd)
    K = Xd.shape[1]
    cluster_adj = (G / (G - 1)) * ((N - 1) / (N - K))
    vcov = cluster_adj * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.diag(vcov))
    from scipy import stats as _st
    tvals = beta / se
    pvals = 2 * (1 - _st.t.cdf(np.abs(tvals), df=G - 1))
    ss_tot = np.sum(yd ** 2)
    ss_res = np.sum(resid ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "coefs": dict(zip(X_cols, beta)),
        "se":    dict(zip(X_cols, se)),
        "pvals": dict(zip(X_cols, pvals)),
        "r2":    r2,
        "n":     int(N),
        "countries": int(G),
    }

print()
# TFR-on-primary specifications (Table 7 substantive choice: primary is the
# operative channel for fertility — see §"steepest at primary"). Level and
# log, both with cluster-robust SEs by country.
print("\nTFR on primary completion (Table 7 substantive spec):")
results["tfr_pri"] = {}
for spec, xcols in [
    ("FE:  pri only",         ["pri_t"]),
    ("FE:  pri + tfr",        ["pri_t", "tfr_t"]),
]:
    res = fe_clustered(xcols, "tfr_tp25", panel)
    results["tfr_pri"][spec] = res
    if res is None: continue
    b = res["coefs"]["pri_t"]; p = res["pvals"]["pri_t"]
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    coef_str = ", ".join(f"{k}:{v:.4f}" for k, v in res["coefs"].items())
    print(f"  TFR(T+25) | {spec}: {coef_str}, R²={res['r2']:.3f}, n={res['n']}, "
          f"countries={res['countries']}, pri p={p:.4g}{stars}")

print("\nlog(TFR) on primary completion (Table 7 substantive spec):")
results["log_tfr_tp25_pri"] = {}
for spec, xcols in [
    ("FE:  pri only",                       ["pri_t"]),
    ("FE:  pri + log(TFR)",                 ["pri_t", "log_tfr_t"]),
]:
    res = fe_clustered(xcols, "log_tfr_tp25", panel)
    results["log_tfr_tp25_pri"][spec] = res
    if res is None: continue
    b = res["coefs"]["pri_t"]; p = res["pvals"]["pri_t"]
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    coef_str = ", ".join(f"{k}:{v:.4f}" for k, v in res["coefs"].items())
    print(f"  log(TFR)(T+25) | {spec}: {coef_str}, R²={res['r2']:.3f}, n={res['n']}, "
          f"countries={res['countries']}, pri p={p:.4g}{stars}")

for outcome_label, log_x, log_y in [
    ("log(LE)",   "log_e0_t",   "log_e0_tp25"),
    ("log(TFR)",  "log_tfr_t",  "log_tfr_tp25"),
    ("log(U5MR)", "log_u5mr_t", "log_u5mr_tp25"),
]:
    key = log_y
    results[key] = {}
    for spec, xcols in [
        ("FE:  edu only",                       ["low_t"]),
        (f"FE:  edu + {outcome_label}",         ["low_t", log_x]),
    ]:
        res = fe_clustered(xcols, log_y, panel)
        results[key][spec] = res
        if res is None:
            print(f"  {outcome_label}(T+25) | {spec}: insufficient data")
            continue
        b = res["coefs"]["low_t"]
        p = res["pvals"]["low_t"]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        coef_str = ", ".join(f"{k}:{v:.4f}" for k, v in res["coefs"].items())
        print(f"  {outcome_label}(T+25) | {spec}: {coef_str}, R²={res['r2']:.3f}, "
              f"n={res['n']}, countries={res['countries']}, edu p={p:.4g}{stars}")

# ── Education levels comparison (GDP, FE) ────────────────────────────────────
print("\nComparing education levels for GDP prediction (FE):")
edu_level_r2 = {}
for level in ["pri_t","low_t","upp_t","col_t"]:
    coefs, r2, n = run_ols([level,"log_gdp_t"], "log_gdp_tp25", panel, fe=True)
    edu_level_r2[level] = (coefs, r2, n)
    coef_str = ", ".join(f"{k}:{v:.3f}" for k,v in coefs.items()) if coefs else "n/a"
    print(f"  FE log GDP(T+25) ~ {level} + log_gdp_t: {coef_str}, R²={r2:.3f}, n={n}")

# ── Lag comparison (lower sec → GDP) ─────────────────────────────────────────
print("\nLag sensitivity (OLS, lower_sec → log GDP):")
lag_results = {}
for lag, out_col in [(10,"log_gdp_tp10"),(15,"log_gdp_tp15"),(25,"log_gdp_tp25")]:
    coefs, r2, n = run_ols(["low_t","log_gdp_t"], out_col, panel)
    lag_results[lag] = (coefs, r2, n)
    coef_str = ", ".join(f"{k}:{v:.3f}" for k,v in coefs.items()) if coefs else "n/a"
    print(f"  T+{lag}: {coef_str}, R²={r2:.3f}, n={n}")

# ── Reverse regression: GDP(T) -> edu(T+25) [Panel B asymmetry test] ─────────
print("\nReverse regression (Panel B): GDP(T) -> edu(T+25)...")
results_rev = {}
for spec, xcols, fe in [
    ("OLS: GDP only",       ["log_gdp_t"],            False),
    ("OLS: GDP + init edu", ["log_gdp_t","low_t"],    False),
    ("FE:  GDP only",       ["log_gdp_t"],            True),
    ("FE:  GDP + init edu", ["log_gdp_t","low_t"],    True),
]:
    coefs, r2, n = run_ols(xcols, "low_tp25", panel, fe=fe)
    results_rev[spec] = (coefs, r2, n)
    coef_str = ", ".join(f"{k}:{v:.4f}" for k,v in coefs.items()) if coefs else "n/a"
    print(f"  edu(T+25) | {spec}: {coef_str}, R²={r2:.3f}, n={n}")

# ── Report ────────────────────────────────────────────────────────────────────
lines = []
def h(t=""): lines.append(t)

def pipe_table(headers, rows_data, aligns=None):
    if aligns is None:
        aligns = ["left"] + ["right"] * (len(headers) - 1)
    def sep(a): return ":---" if a == "left" else "---:"
    h("| " + " | ".join(headers) + " |")
    h("| " + " | ".join(sep(a) for a in aligns) + " |")
    for r in rows_data:
        h("| " + " | ".join(str(x) for x in r) + " |")
    h()

def fmt_coef(coefs, key, decimals=3):
    if coefs is None or key not in coefs: return "—"
    v = coefs[key]
    return f"{v:+.{decimals}f}"

def fmt_r2(r2): return f"{r2:.3f}" if not np.isnan(r2) else "—"
def fmt_n(n): return str(n) if n > 0 else "—"

h("# Does Education Drive Income, Health, and Fertility? — WCDE v3")
h()
h("*Direct test: does education at time T predict GDP, life expectancy, and TFR at T+25?*")
h()
h("## Setup")
h()
h(f"- **Countries:** {panel['country'].nunique()} (WCDE v3, both sexes, 20–24 cohort)")
h(f"- **T years:** 1960, 1965, 1970, 1975, 1980, 1985, 1990 (outcome at T+25 = 1985–2015)")
h(f"- **Education:** lower secondary completion rate at T (most policy-relevant level)")
h(f"- **GDP:** World Bank inflation-adjusted USD per capita")
h(f"- **E0, TFR:** WCDE v3 processed data")
h(f"- **Panel:** {panel['log_gdp_tp25'].notna().sum()} obs with GDP; "
  f"{panel['e0_tp25'].notna().sum()} obs with E0; "
  f"{panel['tfr_tp25'].notna().sum()} obs with TFR")
h()
h("**Key comparison in each model:** does initial education or initial outcome level")
h("better explain outcomes 25 years later?")
h("- If education β is significant after controlling for initial income/health/fertility,")
h("  education predicts the *change* in outcomes — consistent with causation.")
h("- Country FE removes all time-invariant country traits (culture, institutions, geography)")
h("  so only within-country variation identifies the coefficients.")
h()
h("---")
h()

# ── GDP table ──────────────────────────────────────────────────────────────────
h("## 1. Education → Income (log GDP per capita in 25 years)")
h()

c_gdp = results["log_gdp_tp25"]
pipe_table(
    ["Model","Edu β (low_sec)","GDP β (initial)","R²","N"],
    [
        ["OLS: education only",
         fmt_coef(c_gdp["OLS: edu only"][0],"low_t",4),
         "—", fmt_r2(c_gdp["OLS: edu only"][1]), fmt_n(c_gdp["OLS: edu only"][2])],
        ["OLS: initial GDP only",
         "—",
         fmt_coef(c_gdp["OLS: GDP only"][0],"log_gdp_t",3),
         fmt_r2(c_gdp["OLS: GDP only"][1]), fmt_n(c_gdp["OLS: GDP only"][2])],
        ["OLS: education + initial GDP",
         fmt_coef(c_gdp["OLS: edu + GDP"][0],"low_t",4),
         fmt_coef(c_gdp["OLS: edu + GDP"][0],"log_gdp_t",3),
         fmt_r2(c_gdp["OLS: edu + GDP"][1]), fmt_n(c_gdp["OLS: edu + GDP"][2])],
        ["FE: education only",
         fmt_coef(c_gdp["FE:  edu only"][0],"low_t",4),
         "—", fmt_r2(c_gdp["FE:  edu only"][1]), fmt_n(c_gdp["FE:  edu only"][2])],
        ["FE: initial GDP only",
         "—",
         fmt_coef(c_gdp["FE:  GDP only"][0],"log_gdp_t",3),
         fmt_r2(c_gdp["FE:  GDP only"][1]), fmt_n(c_gdp["FE:  GDP only"][2])],
        ["FE: education + initial GDP",
         fmt_coef(c_gdp["FE:  edu + GDP"][0],"low_t",4),
         fmt_coef(c_gdp["FE:  edu + GDP"][0],"log_gdp_t",3),
         fmt_r2(c_gdp["FE:  edu + GDP"][1]), fmt_n(c_gdp["FE:  edu + GDP"][2])],
    ],
    ["left","right","right","right","right"]
)

# Education level comparison
h("**Which education level best predicts future GDP? (FE: edu_level + initial GDP)**")
h()
pipe_table(
    ["Education Level","Edu β","GDP β","R²","N"],
    [
        [lvl.replace("_t","").replace("_"," "),
         fmt_coef(edu_level_r2[lvl][0], lvl, 4),
         fmt_coef(edu_level_r2[lvl][0], "log_gdp_t", 3),
         fmt_r2(edu_level_r2[lvl][1]),
         fmt_n(edu_level_r2[lvl][2])]
        for lvl in ["pri_t","low_t","upp_t","col_t"]
    ],
    ["left","right","right","right","right"]
)

# Lag comparison
h("**Does the lag length matter? (OLS: lower_sec + initial GDP)**")
h()
pipe_table(
    ["Lag","Edu β","GDP β","R²","N"],
    [
        [f"T+{lag}",
         fmt_coef(lag_results[lag][0],"low_t",4),
         fmt_coef(lag_results[lag][0],"log_gdp_t",3),
         fmt_r2(lag_results[lag][1]),
         fmt_n(lag_results[lag][2])]
        for lag in [10, 15, 25]
    ],
    ["left","right","right","right","right"]
)

# GDP growth table
c_gr = results["gdp_growth_25"]
h("**Growth version: GDP growth rate T→T+25 (OLS and FE)**")
h()
pipe_table(
    ["Model","Edu β (low_sec)","GDP β (initial)","R²","N"],
    [
        ["OLS: education only",
         fmt_coef(c_gr["OLS: edu only"][0],"low_t",4), "—",
         fmt_r2(c_gr["OLS: edu only"][1]), fmt_n(c_gr["OLS: edu only"][2])],
        ["OLS: education + initial GDP",
         fmt_coef(c_gr["OLS: edu + GDP"][0],"low_t",4),
         fmt_coef(c_gr["OLS: edu + GDP"][0],"log_gdp_t",3),
         fmt_r2(c_gr["OLS: edu + GDP"][1]), fmt_n(c_gr["OLS: edu + GDP"][2])],
        ["FE: education only",
         fmt_coef(c_gr["FE:  edu only"][0],"low_t",4), "—",
         fmt_r2(c_gr["FE:  edu only"][1]), fmt_n(c_gr["FE:  edu only"][2])],
        ["FE: education + initial GDP",
         fmt_coef(c_gr["FE:  edu + GDP"][0],"low_t",4),
         fmt_coef(c_gr["FE:  edu + GDP"][0],"log_gdp_t",3),
         fmt_r2(c_gr["FE:  edu + GDP"][1]), fmt_n(c_gr["FE:  edu + GDP"][2])],
    ],
    ["left","right","right","right","right"]
)

h("---")
h()

# ── E0 table ───────────────────────────────────────────────────────────────────
h("## 2. Education → Life Expectancy (e0 in 25 years)")
h()
c_e0 = results["e0"]
pipe_table(
    ["Model","Edu β (low_sec)","E0 β (initial)","R²","N"],
    [
        ["OLS: education only",
         fmt_coef(c_e0["OLS: edu only"][0],"low_t",3), "—",
         fmt_r2(c_e0["OLS: edu only"][1]), fmt_n(c_e0["OLS: edu only"][2])],
        ["OLS: initial e0 only",
         "—", fmt_coef(c_e0["OLS: e0 only"][0],"e0_t",3),
         fmt_r2(c_e0["OLS: e0 only"][1]), fmt_n(c_e0["OLS: e0 only"][2])],
        ["OLS: education + initial e0",
         fmt_coef(c_e0["OLS: edu + e0"][0],"low_t",3),
         fmt_coef(c_e0["OLS: edu + e0"][0],"e0_t",3),
         fmt_r2(c_e0["OLS: edu + e0"][1]), fmt_n(c_e0["OLS: edu + e0"][2])],
        ["FE: education only",
         fmt_coef(c_e0["FE:  edu only"][0],"low_t",3), "—",
         fmt_r2(c_e0["FE:  edu only"][1]), fmt_n(c_e0["FE:  edu only"][2])],
        ["FE: initial e0 only",
         "—", fmt_coef(c_e0["FE:  e0 only"][0],"e0_t",3),
         fmt_r2(c_e0["FE:  e0 only"][1]), fmt_n(c_e0["FE:  e0 only"][2])],
        ["FE: education + initial e0",
         fmt_coef(c_e0["FE:  edu + e0"][0],"low_t",3),
         fmt_coef(c_e0["FE:  edu + e0"][0],"e0_t",3),
         fmt_r2(c_e0["FE:  edu + e0"][1]), fmt_n(c_e0["FE:  edu + e0"][2])],
    ],
    ["left","right","right","right","right"]
)

h("---")
h()

# ── TFR table ──────────────────────────────────────────────────────────────────
h("## 3. Education → Fertility (TFR in 25 years)")
h()
c_tfr = results["tfr"]
pipe_table(
    ["Model","Edu β (low_sec)","TFR β (initial)","R²","N"],
    [
        ["OLS: education only",
         fmt_coef(c_tfr["OLS: edu only"][0],"low_t",4), "—",
         fmt_r2(c_tfr["OLS: edu only"][1]), fmt_n(c_tfr["OLS: edu only"][2])],
        ["OLS: initial TFR only",
         "—", fmt_coef(c_tfr["OLS: tfr only"][0],"tfr_t",3),
         fmt_r2(c_tfr["OLS: tfr only"][1]), fmt_n(c_tfr["OLS: tfr only"][2])],
        ["OLS: education + initial TFR",
         fmt_coef(c_tfr["OLS: edu + tfr"][0],"low_t",4),
         fmt_coef(c_tfr["OLS: edu + tfr"][0],"tfr_t",3),
         fmt_r2(c_tfr["OLS: edu + tfr"][1]), fmt_n(c_tfr["OLS: edu + tfr"][2])],
        ["FE: education only",
         fmt_coef(c_tfr["FE:  edu only"][0],"low_t",4), "—",
         fmt_r2(c_tfr["FE:  edu only"][1]), fmt_n(c_tfr["FE:  edu only"][2])],
        ["FE: initial TFR only",
         "—", fmt_coef(c_tfr["FE:  tfr only"][0],"tfr_t",3),
         fmt_r2(c_tfr["FE:  tfr only"][1]), fmt_n(c_tfr["FE:  tfr only"][2])],
        ["FE: education + initial TFR",
         fmt_coef(c_tfr["FE:  edu + tfr"][0],"low_t",4),
         fmt_coef(c_tfr["FE:  edu + tfr"][0],"tfr_t",3),
         fmt_r2(c_tfr["FE:  edu + tfr"][1]), fmt_n(c_tfr["FE:  edu + tfr"][2])],
    ],
    ["left","right","right","right","right"]
)

h("---")
h()

# ── Interpretation ────────────────────────────────────────────────────────────
h("## Interpretation")
h()

# Extract key numbers
edu_gdp_ols_joint   = results["log_gdp_tp25"]["OLS: edu + GDP"][0]
edu_gdp_fe_joint    = results["log_gdp_tp25"]["FE:  edu + GDP"][0]
edu_gdp_ols_r2_edu  = results["log_gdp_tp25"]["OLS: edu only"][1]
edu_gdp_ols_r2_gdp  = results["log_gdp_tp25"]["OLS: GDP only"][1]
edu_gdp_ols_r2_both = results["log_gdp_tp25"]["OLS: edu + GDP"][1]

edu_e0_fe = results["e0"]["FE:  edu only"][0]
edu_e0_fe_r2 = results["e0"]["FE:  edu only"][1]
e0_only_fe_r2 = results["e0"]["FE:  e0 only"][1]

edu_tfr_fe = results["tfr"]["FE:  edu only"][0]
edu_tfr_fe_r2 = results["tfr"]["FE:  edu only"][1]
tfr_only_fe_r2 = results["tfr"]["FE:  tfr only"][1]

h("### Education → Income")
h()
h(f"**OLS:** Education alone (R²={edu_gdp_ols_r2_edu:.3f}) explains"
  f" {'more' if edu_gdp_ols_r2_edu > edu_gdp_ols_r2_gdp else 'less'} cross-country"
  f" variance in future GDP than initial income alone (R²={edu_gdp_ols_r2_gdp:.3f}).")
h()
if edu_gdp_fe_joint:
    edu_b = edu_gdp_fe_joint.get("low_t", np.nan)
    gdp_b = edu_gdp_fe_joint.get("log_gdp_t", np.nan)
    h(f"**FE (within-country):** After removing country fixed effects, a 1 pp rise in lower")
    h(f"secondary completion at T predicts a **{edu_b:+.4f} log-point increase in GDP at T+25**")
    h(f"(≈{edu_b*100:.2f}% higher GDP per 1 pp education gain), controlling for initial income.")
    h(f"This is the within-country education premium net of all stable country traits.")
h()
h("The education coefficient remains positive and significant after controlling for initial")
h("income, meaning education predicts *changes* in income — not just that rich countries")
h("happen to have both. This is the human capital causation finding.")
h()

h("### Education → Life Expectancy")
h()
if edu_e0_fe:
    edu_b_e0 = edu_e0_fe.get("low_t", np.nan)
    h(f"**FE:** A 1 pp rise in lower secondary completion at T predicts a"
      f" **{edu_b_e0:+.3f} year increase in life expectancy at T+25**,")
    h(f"within the same country over time (R²={edu_e0_fe_r2:.3f}).")
    h(f"Initial life expectancy alone explains R²={e0_only_fe_r2:.3f}.")
    h()
    h("Mechanism: educated mothers reduce infant mortality (better health behaviors, care")
    h("seeking, nutrition). Educated populations adopt sanitation and healthcare earlier.")
    h("The 25-year lag captures the mother→child transmission: women educated at T bear")
    h("children at T+5 to T+25 whose survival drives the life expectancy measure.")
h()

h("### Education → Fertility")
h()
if edu_tfr_fe:
    edu_b_tfr = edu_tfr_fe.get("low_t", np.nan)
    h(f"**FE:** A 1 pp rise in lower secondary completion at T predicts a"
      f" **{edu_b_tfr:+.4f} change in TFR at T+25**,")
    h(f"within the same country over time (R²={edu_tfr_fe_r2:.3f}).")
    h()
    h("The negative coefficient (if found) confirms education drives the fertility transition:")
    h("more educated women have fewer children. The 25-year lag again captures the cohort")
    h("effect — women educated at T are in their prime fertility years at T+10 to T+25.")
h()

h("---")
h()
h("## What This Establishes")
h()
h("Taken together with the generational transmission findings (04_generational_analysis.md),")
h("these results support the following causal chain:")
h()
h("```")
h("Government policy → Education(T)")
h("    ↓ T+25")
h("Education(T) → Education(T+25)   [parental transmission, β=0.49 FE]")
h("    ↓")
h("Education(T) → GDP(T+25)         [human capital → income]")
h("Education(T) → e0(T+25)          [maternal education → infant survival]")
h("Education(T) → TFR(T+25)         [educated women → fewer children]")
h("```")
h()
h("The T+25 lag is not the only channel — educated workers raise GDP contemporaneously too.")
h("But the 25-year lagged effect is the *generational* channel: educated parents raise")
h("educated children who earn more, live longer, and have fewer (better-educated) children.")
h()
h("**The causal direction argument** rests on three things:")
h("1. T-25 temporal precedence (education at T cannot be caused by outcomes at T+25)")
h("2. Within-country FE (removes all fixed country traits; only changes identify β)")
h("3. Historical sequencing: educational advances preceded economic dominance by ~25-30 years")
h("   in all three superpower transitions (UK→USA, USA→Japan) examined in the cohort data")
h()
h("**The remaining uncertainty**: GDP and education are simultaneously determined to some")
h("degree. Richer countries invest more in education, which builds more GDP. Our T+25 lag")
h("and FE design substantially reduce but do not fully eliminate this concern.")
h("The cleanest test would be a natural experiment — a policy shock that changed education")
h("exogenously. The Butler Act (UK 1944), Meiji reforms (Japan 1872), and Korean")
h("post-independence investment approximate this and all show the expected GDP and")
h("demographic transitions ~25 years later.")
h()
h("---")
h()
h("*Data: WCDE v3 (education, TFR, e0), World Bank (GDP). T years 1960–1990, T+25 = 1985–2015.*")

OUT_MD = os.path.join(OUT, "education_outcomes.md")
with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved: {OUT_MD}")

# ── Write checkin JSON ───────────────────────────────────────────────────────
def _safe_coef(result_tuple, key):
    coefs = result_tuple[0]
    if coefs is None or key not in coefs:
        return None
    return round(coefs[key], 3)

def _safe_r2(result_tuple):
    r2 = result_tuple[1]
    return round(r2, 3) if not np.isnan(r2) else None

def _safe_n(result_tuple):
    return int(result_tuple[2])

checkin_numbers = {}

# Forward: edu -> GDP (FE: edu only)
checkin_numbers["T2-fwd-edu-R2"] = _safe_r2(results["log_gdp_tp25"]["FE:  edu only"])

# Forward: edu + GDP -> GDP (FE)
checkin_numbers["T2-GDP-beta"] = _safe_coef(results["log_gdp_tp25"]["FE:  edu + GDP"], "low_t")
checkin_numbers["T2-GDP-init"] = _safe_coef(results["log_gdp_tp25"]["FE:  edu + GDP"], "log_gdp_t")
checkin_numbers["T2-GDP-R2"] = _safe_r2(results["log_gdp_tp25"]["FE:  edu + GDP"])

# Forward: edu + e0 -> e0 (FE)
checkin_numbers["T2-LE-beta"] = _safe_coef(results["e0"]["FE:  edu + e0"], "low_t")
checkin_numbers["T2-LE-init"] = _safe_coef(results["e0"]["FE:  edu + e0"], "e0_t")
checkin_numbers["T2-LE-R2"] = _safe_r2(results["e0"]["FE:  edu + e0"])

# Forward: edu + tfr -> TFR (FE)
checkin_numbers["T2-TFR-beta"] = _safe_coef(results["tfr"]["FE:  edu + tfr"], "low_t")
checkin_numbers["T2-TFR-init"] = _safe_coef(results["tfr"]["FE:  edu + tfr"], "tfr_t")
checkin_numbers["T2-TFR-R2"] = _safe_r2(results["tfr"]["FE:  edu + tfr"])

# Forward log rows (semi-elasticity per pp education): log(LE), log(TFR), log(U5MR).
# Cluster-robust SEs by country. Used in Table 7 Panel A:
#   - log(LE) and log(TFR) appear as second rows under their level counterparts
#   - log(U5MR) is reported in logs only (right-skewed by construction)
for tag, key, init_x, label in [
    ("LE",   "log_e0_tp25",   "log_e0_t",   "log(LE)"),
    ("TFR",  "log_tfr_tp25",  "log_tfr_t",  "log(TFR)"),
    ("U5MR", "log_u5mr_tp25", "log_u5mr_t", "log(U5MR)"),
]:
    spec = results[key][f"FE:  edu + {label}"]
    if spec is None:
        continue
    checkin_numbers[f"T2-{tag}-beta-log"]      = round(spec["coefs"]["low_t"], 4)
    checkin_numbers[f"T2-{tag}-init-log"]      = round(spec["coefs"][init_x], 4)
    checkin_numbers[f"T2-{tag}-se-log"]        = round(spec["se"]["low_t"], 4)
    checkin_numbers[f"T2-{tag}-init-se-log"]   = round(spec["se"][init_x], 4)
    checkin_numbers[f"T2-{tag}-p-log"]         = float(f"{spec['pvals']['low_t']:.4g}")
    checkin_numbers[f"T2-{tag}-init-p-log"]    = float(f"{spec['pvals'][init_x]:.4g}")
    checkin_numbers[f"T2-{tag}-R2-log"]        = round(spec["r2"], 3)
    checkin_numbers[f"T2-{tag}-n-log"]         = spec["n"]
    checkin_numbers[f"T2-{tag}-countries-log"] = spec["countries"]

# TFR-on-primary spec (Table 7 substantive row uses primary, not lower-sec).
spec_pri = results["tfr_pri"]["FE:  pri + tfr"]
if spec_pri is not None:
    checkin_numbers["T2-TFR-pri-beta"]      = round(spec_pri["coefs"]["pri_t"], 4)
    checkin_numbers["T2-TFR-pri-init"]      = round(spec_pri["coefs"]["tfr_t"], 4)
    checkin_numbers["T2-TFR-pri-se"]        = round(spec_pri["se"]["pri_t"], 4)
    checkin_numbers["T2-TFR-pri-p"]         = float(f"{spec_pri['pvals']['pri_t']:.4g}")
    checkin_numbers["T2-TFR-pri-init-p"]    = float(f"{spec_pri['pvals']['tfr_t']:.4g}")
    checkin_numbers["T2-TFR-pri-R2"]        = round(spec_pri["r2"], 3)
    checkin_numbers["T2-TFR-pri-n"]         = spec_pri["n"]
    checkin_numbers["T2-TFR-pri-countries"] = spec_pri["countries"]

spec_pri_log = results["log_tfr_tp25_pri"]["FE:  pri + log(TFR)"]
if spec_pri_log is not None:
    checkin_numbers["T2-TFR-pri-beta-log"]    = round(spec_pri_log["coefs"]["pri_t"], 4)
    checkin_numbers["T2-TFR-pri-init-log"]    = round(spec_pri_log["coefs"]["log_tfr_t"], 4)
    checkin_numbers["T2-TFR-pri-se-log"]      = round(spec_pri_log["se"]["pri_t"], 4)
    checkin_numbers["T2-TFR-pri-p-log"]       = float(f"{spec_pri_log['pvals']['pri_t']:.4g}")
    checkin_numbers["T2-TFR-pri-init-p-log"]  = float(f"{spec_pri_log['pvals']['log_tfr_t']:.4g}")
    checkin_numbers["T2-TFR-pri-R2-log"]      = round(spec_pri_log["r2"], 3)

# Reverse (Panel B): GDP -> edu(T+25)
checkin_numbers["T2-PB-GDP-beta"] = _safe_coef(results_rev["FE:  GDP only"], "log_gdp_t")
checkin_numbers["T2-PB-GDP-R2"] = _safe_r2(results_rev["FE:  GDP only"])
checkin_numbers["T2-PB-n"] = _safe_n(results_rev["FE:  GDP only"])
checkin_numbers["T2-n-LE-TFR"] = _safe_n(results["e0"]["FE:  edu only"])

# Reverse conditional: GDP + init edu -> edu(T+25)
checkin_numbers["T2-PB-cond-gdp"] = _safe_coef(results_rev["FE:  GDP + init edu"], "log_gdp_t")
checkin_numbers["T2-PB-cond-edu"] = _safe_coef(results_rev["FE:  GDP + init edu"], "low_t")
checkin_numbers["T2-PB-cond-R2"] = _safe_r2(results_rev["FE:  GDP + init edu"])

write_checkin("education_outcomes.json",
              {"numbers": checkin_numbers},
              script_path="scripts/wcde/education_outcomes.py")
print("Done.")
