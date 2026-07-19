"""
education_outcomes_parent_lag.py
================================
Parent-effect sibling of education_outcomes.py.

The original education_outcomes.py runs every outcome at T+28. For TFR and
U5MR, that vantage is biologically the **grandparent** generation: the
cohort that completed schooling at T (age 20-24) is age 48-52 at T+28 —
past childbearing, and grandparent territory relative to newborns whose
fertility/survival is measured at T+28.

This script holds the **parent** vantage by using per-outcome biological
lags from scripts/_shared.py:

  - TFR  at T + LAG_TFR             (=5)   mother-of-newborn timing
  - U5MR at T + LAG_CHILDREARING    (=12)  cohort's own children's survival
  - LE   at T + LAG_LE              (=25)  society-wide longevity vantage
                                           (LE is dominated by older adults)
  - GDP  at T + LAG_CONTEMPORANEOUS (=0)   cohort's own productivity

Initial-outcome AR control handling:
  - For TFR (T+5) and U5MR (T+12) we keep an "initial outcome at T" control.
    The AR window is 5 / 12 years rather than 25 — meaningfully shorter
    than the original, but still a valid baseline.
  - For LE and GDP (T+0) the initial-outcome control is degenerate (it
    IS the outcome). We drop the AR control there and run "edu only" and
    "edu + log GDP" (income control, not AR).

EDU_YEARS window:
  Single common window range(1960, 2011, 5) — i.e., 1960-2010 in 5-year
  steps. Per-outcome coverage will differ at the tail (TFR at T+5 reaches
  2015; U5MR at T+12 reaches 2022 — clipped by data availability;
  LE/GDP at T+0 reach 2010).

Output:
  - checkin/education_outcomes_parent_lag.json
  - wcde/output/education_outcomes_parent_lag.md

NOT registered in verify_the_long_childhood.py — this is an exploratory
comparison artifact; the user will decide whether to bring any of these
numbers into the paper.
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
from _shared import (
    PROC,
    DATA as ROOT_DS,
    REPO_ROOT,
    REGIONS,
    NAME_MAP as _SHARED_NAME_MAP,
    load_wb,
    write_checkin,
    LAG_TFR,
    LAG_CHILDREARING,
    LAG_CONTEMPORANEOUS,
    LAG_LE,
)

OUT = os.path.join(REPO_ROOT, "wcde", "output")
os.makedirs(OUT, exist_ok=True)

# Per-outcome biological lags (years after T).
OUTCOME_LAG = {
    "tfr":  LAG_TFR,             # = 5   mother-of-newborn timing
    "u5mr": LAG_CHILDREARING,    # = 12  cohort's own children's U5MR
    "e0":   LAG_LE,              # = 25  society-wide longevity vantage
    "gdp":  LAG_CONTEMPORANEOUS, # = 0   cohort's own productivity
}


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
edu = pd.read_csv(os.path.join(PROC, "completion_both_long.csv"))
edu = edu[~edu["country"].isin(REGIONS)].copy()
for col in ["primary", "lower_sec", "upper_sec", "college"]:
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
# Common T-year window. With biological lags this expands from the original
# 1960–1990 (limited by T+25 needing to stay within data) to 1960–2010.
EDU_YEARS = list(range(1960, 2011, 5))

print("Building panel...")
rows = []
countries = sorted(edu["country"].unique())

for c in countries:
    edu_c = edu[edu["country"] == c].set_index("year")
    for t in EDU_YEARS:
        if t not in edu_c.index: continue

        # Outcome years at biological lags.
        t_tfr  = t + OUTCOME_LAG["tfr"]   # T+5
        t_u5mr = t + OUTCOME_LAG["u5mr"]  # T+12
        t_e0   = t + OUTCOME_LAG["e0"]    # T+0
        t_gdp  = t + OUTCOME_LAG["gdp"]   # T+0

        low   = edu_c.loc[t, "lower_sec"]
        pri   = edu_c.loc[t, "primary"]
        upp   = edu_c.loc[t, "upper_sec"]
        col_v = edu_c.loc[t, "college"]

        # Initial-outcome controls. "_t" is at T; outcome columns are at the
        # biological lag and named accordingly.
        gdp_t      = get_gdp(c, t)
        log_gdp_t  = np.log(gdp_t) if not np.isnan(gdp_t) else np.nan

        # TFR at T+5; initial TFR at T (5-year AR window)
        tfr_t      = get_tfr(c, t)
        tfr_tp5    = get_tfr(c, t_tfr)

        # U5MR at T+12; initial U5MR at T (12-year AR window)
        u5mr_t     = get_u5mr(c, t)
        u5mr_tp12  = get_u5mr(c, t_u5mr)

        # LE at T+0 — same as initial; no AR control. We still store e0_t
        # for symmetry (degenerate); the regressions below do not use it as
        # both LHS and RHS.
        e0_t       = get_e0(c, t_e0)

        if any(np.isnan(x) for x in [low, pri]): continue

        rows.append({
            "country": c, "t": t,
            "low_t": low, "pri_t": pri, "upp_t": upp, "col_t": col_v,
            "log_gdp_t":   log_gdp_t,
            "tfr_t":       tfr_t,
            "tfr_tp5":     tfr_tp5,
            "u5mr_t":      u5mr_t,
            "u5mr_tp12":   u5mr_tp12,
            "e0_t":        e0_t,
        })

panel = pd.DataFrame(rows)

# Log transformations of outcomes and lagged initial values.
for col in ["e0_t", "tfr_t", "tfr_tp5", "u5mr_t", "u5mr_tp12"]:
    panel[f"log_{col}"] = np.log(panel[col].where(panel[col] > 0))

print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries")
print(f"  GDP coverage (T+0):    {panel['log_gdp_t'].notna().sum()} obs")
print(f"  E0 coverage  (T+0):    {panel['e0_t'].notna().sum()} obs")
print(f"  TFR coverage (T+5):    {panel['tfr_tp5'].notna().sum()} obs")
print(f"  U5MR coverage (T+12):  {panel['u5mr_tp12'].notna().sum()} obs")


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


def fe_clustered(X_cols, y_col, data, country_col="country"):
    """FE via demeaning with cluster-robust SEs by country (matches original)."""
    sub = data.dropna(subset=X_cols + [y_col]).copy()
    if len(sub) < 10:
        return None
    for col in X_cols + [y_col]:
        sub[col + "_dm"] = sub[col] - sub.groupby(country_col)[col].transform("mean")
    Xd = sub[[c + "_dm" for c in X_cols]].values
    yd = sub[y_col + "_dm"].values
    countries_arr = sub[country_col].values
    ok = ~np.isnan(Xd).any(axis=1) & ~np.isnan(yd)
    Xd, yd, countries_arr = Xd[ok], yd[ok], countries_arr[ok]
    if len(yd) < 10:
        return None
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ yd
    resid = yd - Xd @ beta
    meat = np.zeros((Xd.shape[1], Xd.shape[1]))
    for c in np.unique(countries_arr):
        idx = countries_arr == c
        u = Xd[idx].T @ resid[idx]
        meat += np.outer(u, u)
    G = len(np.unique(countries_arr))
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

# ── Run regressions ───────────────────────────────────────────────────────────
print("\nRunning regressions at biological per-outcome lags...")

results = {}

# ── GDP at T+0 ───────────────────────────────────────────────────────────────
# Outcome is log_gdp_t itself; no AR control. Pure within-country test of
# whether edu(T) co-moves with contemporaneous log GDP.
print()
results["log_gdp_t"] = {}
for spec, xcols, fe in [
    ("OLS: edu only",   ["low_t"], False),
    ("FE:  edu only",   ["low_t"], True),
]:
    coefs, r2, n = run_ols(xcols, "log_gdp_t", panel, fe=fe)
    results["log_gdp_t"][spec] = (coefs, r2, n)
    coef_str = ", ".join(f"{k}:{v:.3f}" for k,v in coefs.items()) if coefs else "n/a"
    print(f"  log GDP(T) | {spec}: {coef_str}, R²={r2:.3f}, n={n}")

# ── LE at T+0 ────────────────────────────────────────────────────────────────
# Outcome is e0_t itself; no AR control. Add log GDP as an income control
# (not an AR control) in the joint spec.
print()
results["e0_t"] = {}
for spec, xcols, fe in [
    ("OLS: edu only",   ["low_t"],              False),
    ("OLS: edu + GDP",  ["low_t", "log_gdp_t"], False),
    ("FE:  edu only",   ["low_t"],              True),
    ("FE:  edu + GDP",  ["low_t", "log_gdp_t"], True),
]:
    coefs, r2, n = run_ols(xcols, "e0_t", panel, fe=fe)
    results["e0_t"][spec] = (coefs, r2, n)
    coef_str = ", ".join(f"{k}:{v:.3f}" for k,v in coefs.items()) if coefs else "n/a"
    print(f"  e0(T) | {spec}: {coef_str}, R²={r2:.3f}, n={n}")

# ── TFR at T+5 ───────────────────────────────────────────────────────────────
# 5-year AR control: initial TFR at T predicts TFR at T+5.
print()
results["tfr_tp5"] = {}
for spec, xcols, fe in [
    ("OLS: edu only",       ["low_t"],            False),
    ("OLS: tfr only",       ["tfr_t"],            False),
    ("OLS: edu + tfr",      ["low_t","tfr_t"],    False),
    ("FE:  edu only",       ["low_t"],            True),
    ("FE:  tfr only",       ["tfr_t"],            True),
    ("FE:  edu + tfr",      ["low_t","tfr_t"],    True),
]:
    coefs, r2, n = run_ols(xcols, "tfr_tp5", panel, fe=fe)
    results["tfr_tp5"][spec] = (coefs, r2, n)
    coef_str = ", ".join(f"{k}:{v:.3f}" for k,v in coefs.items()) if coefs else "n/a"
    print(f"  TFR(T+5) | {spec}: {coef_str}, R²={r2:.3f}, n={n}")

# ── U5MR at T+12 ─────────────────────────────────────────────────────────────
# 12-year AR control.
print()
results["u5mr_tp12"] = {}
for spec, xcols, fe in [
    ("OLS: edu only",       ["low_t"],              False),
    ("OLS: u5mr only",      ["u5mr_t"],             False),
    ("OLS: edu + u5mr",     ["low_t","u5mr_t"],     False),
    ("FE:  edu only",       ["low_t"],              True),
    ("FE:  u5mr only",      ["u5mr_t"],             True),
    ("FE:  edu + u5mr",     ["low_t","u5mr_t"],     True),
]:
    coefs, r2, n = run_ols(xcols, "u5mr_tp12", panel, fe=fe)
    results["u5mr_tp12"][spec] = (coefs, r2, n)
    coef_str = ", ".join(f"{k}:{v:.3f}" for k,v in coefs.items()) if coefs else "n/a"
    print(f"  U5MR(T+12) | {spec}: {coef_str}, R²={r2:.3f}, n={n}")

# ── Log-outcome rows with cluster-robust SEs ─────────────────────────────────
print("\nLog-outcome FE rows (cluster-robust by country):")

# log(LE) at T+0
results["log_e0_t"] = {}
for spec, xcols in [
    ("FE:  edu only", ["low_t"]),
]:
    res = fe_clustered(xcols, "log_e0_t", panel)
    results["log_e0_t"][spec] = res
    if res is None:
        print(f"  log(LE)(T) | {spec}: insufficient data"); continue
    b = res["coefs"]["low_t"]; p = res["pvals"]["low_t"]
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    coef_str = ", ".join(f"{k}:{v:.4f}" for k, v in res["coefs"].items())
    print(f"  log(LE)(T) | {spec}: {coef_str}, R²={res['r2']:.3f}, "
          f"n={res['n']}, countries={res['countries']}, edu p={p:.4g}{stars}")

# log(TFR) at T+5 with log(TFR_t) AR control
results["log_tfr_tp5"] = {}
for spec, xcols in [
    ("FE:  edu only",            ["low_t"]),
    ("FE:  edu + log(TFR)",      ["low_t", "log_tfr_t"]),
]:
    res = fe_clustered(xcols, "log_tfr_tp5", panel)
    results["log_tfr_tp5"][spec] = res
    if res is None:
        print(f"  log(TFR)(T+5) | {spec}: insufficient data"); continue
    b = res["coefs"]["low_t"]; p = res["pvals"]["low_t"]
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    coef_str = ", ".join(f"{k}:{v:.4f}" for k, v in res["coefs"].items())
    print(f"  log(TFR)(T+5) | {spec}: {coef_str}, R²={res['r2']:.3f}, "
          f"n={res['n']}, countries={res['countries']}, edu p={p:.4g}{stars}")

# log(U5MR) at T+12 with log(U5MR_t) AR control
results["log_u5mr_tp12"] = {}
for spec, xcols in [
    ("FE:  edu only",             ["low_t"]),
    ("FE:  edu + log(U5MR)",      ["low_t", "log_u5mr_t"]),
]:
    res = fe_clustered(xcols, "log_u5mr_tp12", panel)
    results["log_u5mr_tp12"][spec] = res
    if res is None:
        print(f"  log(U5MR)(T+12) | {spec}: insufficient data"); continue
    b = res["coefs"]["low_t"]; p = res["pvals"]["low_t"]
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    coef_str = ", ".join(f"{k}:{v:.4f}" for k, v in res["coefs"].items())
    print(f"  log(U5MR)(T+12) | {spec}: {coef_str}, R²={res['r2']:.3f}, "
          f"n={res['n']}, countries={res['countries']}, edu p={p:.4g}{stars}")

# TFR-on-primary specs at T+5 (the substantive Table 7 row, parent vantage).
print("\nTFR on primary completion at T+5 (parent vantage):")
results["tfr_pri_tp5"] = {}
for spec, xcols in [
    ("FE:  pri only",   ["pri_t"]),
    ("FE:  pri + tfr",  ["pri_t", "tfr_t"]),
]:
    res = fe_clustered(xcols, "tfr_tp5", panel)
    results["tfr_pri_tp5"][spec] = res
    if res is None: continue
    b = res["coefs"]["pri_t"]; p = res["pvals"]["pri_t"]
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    coef_str = ", ".join(f"{k}:{v:.4f}" for k, v in res["coefs"].items())
    print(f"  TFR(T+5) | {spec}: {coef_str}, R²={res['r2']:.3f}, n={res['n']}, "
          f"countries={res['countries']}, pri p={p:.4g}{stars}")

print("\nlog(TFR) on primary completion at T+5 (parent vantage):")
results["log_tfr_tp5_pri"] = {}
for spec, xcols in [
    ("FE:  pri only",            ["pri_t"]),
    ("FE:  pri + log(TFR)",      ["pri_t", "log_tfr_t"]),
]:
    res = fe_clustered(xcols, "log_tfr_tp5", panel)
    results["log_tfr_tp5_pri"][spec] = res
    if res is None: continue
    b = res["coefs"]["pri_t"]; p = res["pvals"]["pri_t"]
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    coef_str = ", ".join(f"{k}:{v:.4f}" for k, v in res["coefs"].items())
    print(f"  log(TFR)(T+5) | {spec}: {coef_str}, R²={res['r2']:.3f}, n={res['n']}, "
          f"countries={res['countries']}, pri p={p:.4g}{stars}")

# Education level comparison for the contemporaneous GDP outcome.
print("\nComparing education levels for GDP(T) (FE):")
edu_level_r2 = {}
for level in ["pri_t", "low_t", "upp_t", "col_t"]:
    coefs, r2, n = run_ols([level], "log_gdp_t", panel, fe=True)
    edu_level_r2[level] = (coefs, r2, n)
    coef_str = ", ".join(f"{k}:{v:.3f}" for k,v in coefs.items()) if coefs else "n/a"
    print(f"  FE log GDP(T) ~ {level}: {coef_str}, R²={r2:.3f}, n={n}")


# ── Report (markdown) ─────────────────────────────────────────────────────────
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

h("# Education → Outcomes at Biological Per-Outcome Lags — WCDE v3")
h()
h("*Parent-effect sibling of `education_outcomes.md` (which runs all outcomes at T+25,"
  " a grandparent vantage for TFR/U5MR).*")
h()
h("**Design.** Per-outcome biological lags from `scripts/_shared.py`:")
h()
h("- TFR at T + LAG_TFR (=5) — cohort age 20-24 at T is age 25-29 at T+5, reproductive peak")
h("- U5MR at T + LAG_CHILDREARING (=12) — cohort's own children's under-5 mortality")
h("- LE at T + LAG_LE (=25) — society-wide longevity vantage (dominated by older adults)")
h("- log GDP at T + LAG_CONTEMPORANEOUS (=0) — cohort's own productivity")
h()
h("For LE and GDP the outcome is at T, so the original 'initial outcome at T' AR")
h("control is degenerate. We drop the AR control for LE and GDP and use log GDP as")
h("an income control only. For TFR and U5MR we keep an AR baseline at T (5-year and")
h("12-year windows respectively).")
h()
h("## Setup")
h()
h(f"- **Countries:** {panel['country'].nunique()} (WCDE v3, both sexes, 20–24 cohort)")
h(f"- **T years:** {EDU_YEARS[0]}–{EDU_YEARS[-1]} (5-year steps)")
h(f"- **Education:** lower secondary completion rate at T (primary also reported)")
h(f"- **Panel:** {panel['log_gdp_t'].notna().sum()} obs with GDP(T); "
  f"{panel['e0_t'].notna().sum()} obs with LE(T); "
  f"{panel['tfr_tp5'].notna().sum()} obs with TFR(T+5); "
  f"{panel['u5mr_tp12'].notna().sum()} obs with U5MR(T+12)")
h()
h("---")
h()

# ── GDP table ─────────────────────────────────────────────────────────────────
h("## 1. Education → log GDP per capita at T+0")
h()
c_gdp = results["log_gdp_t"]
pipe_table(
    ["Model", "Edu β (low_sec)", "R²", "N"],
    [
        ["OLS: education only",
         fmt_coef(c_gdp["OLS: edu only"][0], "low_t", 4),
         fmt_r2(c_gdp["OLS: edu only"][1]), fmt_n(c_gdp["OLS: edu only"][2])],
        ["FE: education only",
         fmt_coef(c_gdp["FE:  edu only"][0], "low_t", 4),
         fmt_r2(c_gdp["FE:  edu only"][1]), fmt_n(c_gdp["FE:  edu only"][2])],
    ],
    ["left", "right", "right", "right"]
)

h("**Which education level best predicts contemporaneous GDP? (FE)**")
h()
pipe_table(
    ["Education Level", "Edu β", "R²", "N"],
    [
        [lvl.replace("_t","").replace("_"," "),
         fmt_coef(edu_level_r2[lvl][0], lvl, 4),
         fmt_r2(edu_level_r2[lvl][1]),
         fmt_n(edu_level_r2[lvl][2])]
        for lvl in ["pri_t","low_t","upp_t","col_t"]
    ],
    ["left","right","right","right"]
)

h("---")
h()

# ── LE table ──────────────────────────────────────────────────────────────────
h("## 2. Education → Life Expectancy at T+0")
h()
c_e0 = results["e0_t"]
pipe_table(
    ["Model","Edu β (low_sec)","log GDP β","R²","N"],
    [
        ["OLS: education only",
         fmt_coef(c_e0["OLS: edu only"][0], "low_t", 3), "—",
         fmt_r2(c_e0["OLS: edu only"][1]), fmt_n(c_e0["OLS: edu only"][2])],
        ["OLS: education + log GDP",
         fmt_coef(c_e0["OLS: edu + GDP"][0], "low_t", 3),
         fmt_coef(c_e0["OLS: edu + GDP"][0], "log_gdp_t", 3),
         fmt_r2(c_e0["OLS: edu + GDP"][1]), fmt_n(c_e0["OLS: edu + GDP"][2])],
        ["FE: education only",
         fmt_coef(c_e0["FE:  edu only"][0], "low_t", 3), "—",
         fmt_r2(c_e0["FE:  edu only"][1]), fmt_n(c_e0["FE:  edu only"][2])],
        ["FE: education + log GDP",
         fmt_coef(c_e0["FE:  edu + GDP"][0], "low_t", 3),
         fmt_coef(c_e0["FE:  edu + GDP"][0], "log_gdp_t", 3),
         fmt_r2(c_e0["FE:  edu + GDP"][1]), fmt_n(c_e0["FE:  edu + GDP"][2])],
    ],
    ["left","right","right","right","right"]
)

h("---")
h()

# ── TFR table ─────────────────────────────────────────────────────────────────
h("## 3. Education → TFR at T+5 (parent vantage)")
h()
c_tfr = results["tfr_tp5"]
pipe_table(
    ["Model","Edu β (low_sec)","TFR β (initial)","R²","N"],
    [
        ["OLS: education only",
         fmt_coef(c_tfr["OLS: edu only"][0], "low_t", 4), "—",
         fmt_r2(c_tfr["OLS: edu only"][1]), fmt_n(c_tfr["OLS: edu only"][2])],
        ["OLS: initial TFR only",
         "—", fmt_coef(c_tfr["OLS: tfr only"][0], "tfr_t", 3),
         fmt_r2(c_tfr["OLS: tfr only"][1]), fmt_n(c_tfr["OLS: tfr only"][2])],
        ["OLS: education + initial TFR",
         fmt_coef(c_tfr["OLS: edu + tfr"][0], "low_t", 4),
         fmt_coef(c_tfr["OLS: edu + tfr"][0], "tfr_t", 3),
         fmt_r2(c_tfr["OLS: edu + tfr"][1]), fmt_n(c_tfr["OLS: edu + tfr"][2])],
        ["FE: education only",
         fmt_coef(c_tfr["FE:  edu only"][0], "low_t", 4), "—",
         fmt_r2(c_tfr["FE:  edu only"][1]), fmt_n(c_tfr["FE:  edu only"][2])],
        ["FE: initial TFR only",
         "—", fmt_coef(c_tfr["FE:  tfr only"][0], "tfr_t", 3),
         fmt_r2(c_tfr["FE:  tfr only"][1]), fmt_n(c_tfr["FE:  tfr only"][2])],
        ["FE: education + initial TFR",
         fmt_coef(c_tfr["FE:  edu + tfr"][0], "low_t", 4),
         fmt_coef(c_tfr["FE:  edu + tfr"][0], "tfr_t", 3),
         fmt_r2(c_tfr["FE:  edu + tfr"][1]), fmt_n(c_tfr["FE:  edu + tfr"][2])],
    ],
    ["left","right","right","right","right"]
)

h("---")
h()

# ── U5MR table ────────────────────────────────────────────────────────────────
h("## 4. Education → U5MR at T+12 (parent vantage)")
h()
c_u5 = results["u5mr_tp12"]
pipe_table(
    ["Model","Edu β (low_sec)","U5MR β (initial)","R²","N"],
    [
        ["OLS: education only",
         fmt_coef(c_u5["OLS: edu only"][0], "low_t", 3), "—",
         fmt_r2(c_u5["OLS: edu only"][1]), fmt_n(c_u5["OLS: edu only"][2])],
        ["OLS: initial U5MR only",
         "—", fmt_coef(c_u5["OLS: u5mr only"][0], "u5mr_t", 3),
         fmt_r2(c_u5["OLS: u5mr only"][1]), fmt_n(c_u5["OLS: u5mr only"][2])],
        ["OLS: education + initial U5MR",
         fmt_coef(c_u5["OLS: edu + u5mr"][0], "low_t", 3),
         fmt_coef(c_u5["OLS: edu + u5mr"][0], "u5mr_t", 3),
         fmt_r2(c_u5["OLS: edu + u5mr"][1]), fmt_n(c_u5["OLS: edu + u5mr"][2])],
        ["FE: education only",
         fmt_coef(c_u5["FE:  edu only"][0], "low_t", 3), "—",
         fmt_r2(c_u5["FE:  edu only"][1]), fmt_n(c_u5["FE:  edu only"][2])],
        ["FE: initial U5MR only",
         "—", fmt_coef(c_u5["FE:  u5mr only"][0], "u5mr_t", 3),
         fmt_r2(c_u5["FE:  u5mr only"][1]), fmt_n(c_u5["FE:  u5mr only"][2])],
        ["FE: education + initial U5MR",
         fmt_coef(c_u5["FE:  edu + u5mr"][0], "low_t", 3),
         fmt_coef(c_u5["FE:  edu + u5mr"][0], "u5mr_t", 3),
         fmt_r2(c_u5["FE:  edu + u5mr"][1]), fmt_n(c_u5["FE:  edu + u5mr"][2])],
    ],
    ["left","right","right","right","right"]
)

h("---")
h()
h("*Data: WCDE v3 (education, TFR, e0), World Bank (GDP, U5MR). T years "
  f"{EDU_YEARS[0]}–{EDU_YEARS[-1]} in 5-year steps. Biological per-outcome lags.*")

OUT_MD = os.path.join(OUT, "education_outcomes_parent_lag.md")
with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved: {OUT_MD}")


# ── Write checkin JSON ────────────────────────────────────────────────────────
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

# GDP at T+0 (FE: edu only) — keep same key tag style as original.
checkin_numbers["T2-fwd-edu-R2"]   = _safe_r2(results["log_gdp_t"]["FE:  edu only"])
checkin_numbers["T2-GDP-beta"]     = _safe_coef(results["log_gdp_t"]["FE:  edu only"], "low_t")
checkin_numbers["T2-GDP-R2"]       = _safe_r2(results["log_gdp_t"]["FE:  edu only"])
checkin_numbers["T2-GDP-n"]        = _safe_n(results["log_gdp_t"]["FE:  edu only"])

# LE at T+0 (FE: edu + log GDP)
checkin_numbers["T2-LE-beta"]      = _safe_coef(results["e0_t"]["FE:  edu + GDP"], "low_t")
checkin_numbers["T2-LE-gdp"]       = _safe_coef(results["e0_t"]["FE:  edu + GDP"], "log_gdp_t")
checkin_numbers["T2-LE-R2"]        = _safe_r2(results["e0_t"]["FE:  edu + GDP"])
checkin_numbers["T2-LE-n"]         = _safe_n(results["e0_t"]["FE:  edu + GDP"])
checkin_numbers["T2-LE-edu-only-beta"] = _safe_coef(results["e0_t"]["FE:  edu only"], "low_t")
checkin_numbers["T2-LE-edu-only-R2"]   = _safe_r2(results["e0_t"]["FE:  edu only"])

# TFR at T+5 (FE: edu + initial TFR)
checkin_numbers["T2-TFR-beta"]     = _safe_coef(results["tfr_tp5"]["FE:  edu + tfr"], "low_t")
checkin_numbers["T2-TFR-init"]     = _safe_coef(results["tfr_tp5"]["FE:  edu + tfr"], "tfr_t")
checkin_numbers["T2-TFR-R2"]       = _safe_r2(results["tfr_tp5"]["FE:  edu + tfr"])
checkin_numbers["T2-TFR-n"]        = _safe_n(results["tfr_tp5"]["FE:  edu + tfr"])
checkin_numbers["T2-TFR-edu-only-beta"] = _safe_coef(results["tfr_tp5"]["FE:  edu only"], "low_t")
checkin_numbers["T2-TFR-edu-only-R2"]   = _safe_r2(results["tfr_tp5"]["FE:  edu only"])

# U5MR at T+12 (FE: edu + initial U5MR)
checkin_numbers["T2-U5MR-beta"]    = _safe_coef(results["u5mr_tp12"]["FE:  edu + u5mr"], "low_t")
checkin_numbers["T2-U5MR-init"]    = _safe_coef(results["u5mr_tp12"]["FE:  edu + u5mr"], "u5mr_t")
checkin_numbers["T2-U5MR-R2"]      = _safe_r2(results["u5mr_tp12"]["FE:  edu + u5mr"])
checkin_numbers["T2-U5MR-n"]       = _safe_n(results["u5mr_tp12"]["FE:  edu + u5mr"])
checkin_numbers["T2-U5MR-edu-only-beta"] = _safe_coef(results["u5mr_tp12"]["FE:  edu only"], "low_t")
checkin_numbers["T2-U5MR-edu-only-R2"]   = _safe_r2(results["u5mr_tp12"]["FE:  edu only"])

# Log-outcome rows (cluster-robust).
spec_le_log = results["log_e0_t"]["FE:  edu only"]
if spec_le_log is not None:
    checkin_numbers["T2-LE-beta-log"]      = round(spec_le_log["coefs"]["low_t"], 4)
    checkin_numbers["T2-LE-se-log"]        = round(spec_le_log["se"]["low_t"], 4)
    checkin_numbers["T2-LE-p-log"]         = float(f"{spec_le_log['pvals']['low_t']:.4g}")
    checkin_numbers["T2-LE-R2-log"]        = round(spec_le_log["r2"], 3)
    checkin_numbers["T2-LE-n-log"]         = spec_le_log["n"]
    checkin_numbers["T2-LE-countries-log"] = spec_le_log["countries"]

spec_tfr_log = results["log_tfr_tp5"]["FE:  edu + log(TFR)"]
if spec_tfr_log is not None:
    checkin_numbers["T2-TFR-beta-log"]      = round(spec_tfr_log["coefs"]["low_t"], 4)
    checkin_numbers["T2-TFR-init-log"]      = round(spec_tfr_log["coefs"]["log_tfr_t"], 4)
    checkin_numbers["T2-TFR-se-log"]        = round(spec_tfr_log["se"]["low_t"], 4)
    checkin_numbers["T2-TFR-init-se-log"]   = round(spec_tfr_log["se"]["log_tfr_t"], 4)
    checkin_numbers["T2-TFR-p-log"]         = float(f"{spec_tfr_log['pvals']['low_t']:.4g}")
    checkin_numbers["T2-TFR-init-p-log"]    = float(f"{spec_tfr_log['pvals']['log_tfr_t']:.4g}")
    checkin_numbers["T2-TFR-R2-log"]        = round(spec_tfr_log["r2"], 3)
    checkin_numbers["T2-TFR-n-log"]         = spec_tfr_log["n"]
    checkin_numbers["T2-TFR-countries-log"] = spec_tfr_log["countries"]

spec_u5_log = results["log_u5mr_tp12"]["FE:  edu + log(U5MR)"]
if spec_u5_log is not None:
    checkin_numbers["T2-U5MR-beta-log"]      = round(spec_u5_log["coefs"]["low_t"], 4)
    checkin_numbers["T2-U5MR-init-log"]      = round(spec_u5_log["coefs"]["log_u5mr_t"], 4)
    checkin_numbers["T2-U5MR-se-log"]        = round(spec_u5_log["se"]["low_t"], 4)
    checkin_numbers["T2-U5MR-init-se-log"]   = round(spec_u5_log["se"]["log_u5mr_t"], 4)
    checkin_numbers["T2-U5MR-p-log"]         = float(f"{spec_u5_log['pvals']['low_t']:.4g}")
    checkin_numbers["T2-U5MR-init-p-log"]    = float(f"{spec_u5_log['pvals']['log_u5mr_t']:.4g}")
    checkin_numbers["T2-U5MR-R2-log"]        = round(spec_u5_log["r2"], 3)
    checkin_numbers["T2-U5MR-n-log"]         = spec_u5_log["n"]
    checkin_numbers["T2-U5MR-countries-log"] = spec_u5_log["countries"]

# TFR-on-primary at T+5
spec_pri = results["tfr_pri_tp5"]["FE:  pri + tfr"]
if spec_pri is not None:
    checkin_numbers["T2-TFR-pri-beta"]      = round(spec_pri["coefs"]["pri_t"], 4)
    checkin_numbers["T2-TFR-pri-init"]      = round(spec_pri["coefs"]["tfr_t"], 4)
    checkin_numbers["T2-TFR-pri-se"]        = round(spec_pri["se"]["pri_t"], 4)
    checkin_numbers["T2-TFR-pri-p"]         = float(f"{spec_pri['pvals']['pri_t']:.4g}")
    checkin_numbers["T2-TFR-pri-init-p"]    = float(f"{spec_pri['pvals']['tfr_t']:.4g}")
    checkin_numbers["T2-TFR-pri-R2"]        = round(spec_pri["r2"], 3)
    checkin_numbers["T2-TFR-pri-n"]         = spec_pri["n"]
    checkin_numbers["T2-TFR-pri-countries"] = spec_pri["countries"]

spec_pri_log = results["log_tfr_tp5_pri"]["FE:  pri + log(TFR)"]
if spec_pri_log is not None:
    checkin_numbers["T2-TFR-pri-beta-log"]    = round(spec_pri_log["coefs"]["pri_t"], 4)
    checkin_numbers["T2-TFR-pri-init-log"]    = round(spec_pri_log["coefs"]["log_tfr_t"], 4)
    checkin_numbers["T2-TFR-pri-se-log"]      = round(spec_pri_log["se"]["pri_t"], 4)
    checkin_numbers["T2-TFR-pri-p-log"]       = float(f"{spec_pri_log['pvals']['pri_t']:.4g}")
    checkin_numbers["T2-TFR-pri-init-p-log"]  = float(f"{spec_pri_log['pvals']['log_tfr_t']:.4g}")
    checkin_numbers["T2-TFR-pri-R2-log"]      = round(spec_pri_log["r2"], 3)

# Lag tags recorded for traceability.
checkin_numbers["lag_tfr"]  = LAG_TFR
checkin_numbers["lag_u5mr"] = LAG_CHILDREARING
checkin_numbers["lag_le"]   = LAG_LE
checkin_numbers["lag_gdp"]  = LAG_CONTEMPORANEOUS

write_checkin("education_outcomes_parent_lag.json",
              {"numbers": checkin_numbers},
              script_path="scripts/wcde/education_outcomes_parent_lag.py")
print("Done.")
