"""
table_short_15yr.py
====================
Companion to table_contemporaneous.py and table7_stepwise.py: produces
the 15-year-lag panel coefficients corresponding to the "leader's
timeline" framing in §the-generational-lag. Today's primary entrants
(age 6) reach age 21 in 15 years; today's lower-sec students (age
12-15) hit prime fertility T+5 to T+10. The 15-year lag is also
where U5MR within-country R² peaks (childrearing window).

Panel — outcome at T+15 on education at T:
  (1) edu only,                       country FE
  (2) edu + log GDP at T,             country FE
  (3) edu,                            country & year FE
  (4) edu + log GDP at T,             country & year FE

Outcomes:
  - log GDP per capita     (lower-secondary as predictor)
  - log(LE)                (lower-secondary as predictor)
  - log(TFR)               (primary as predictor)
  - log(U5MR)              (lower-secondary as predictor) — this is where
                            U5MR's lag-sweep R² peaks at 0.716

Empirical headline: U5MR R²(lag 15) = 0.716, the highest in the lag
sweep across all outcomes. TFR R²(lag 15) = 0.580 (still strong but
below lag-0 peak of 0.668). LE R²(lag 15) = 0.443. log GDP
R²(lag 15) = 0.362.

Output: checkin/table_short_15yr.json
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import (PROC, REGIONS, write_checkin, load_wb,
                     NAME_MAP as _SHARED_NAME_MAP)

EDU_YEARS = list(range(1960, 2001, 5))  # need T+15 to land in WDI range
LAG = 15


edu_long = pd.read_csv(os.path.join(PROC, "cohort_completion_both_long.csv"))
edu_long = edu_long[~edu_long["country"].isin(REGIONS)]
edu_long = edu_long.rename(columns={"cohort_year": "year"})
edu_long["country"] = edu_long["country"].str.lower().str.strip()

gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
le_df  = load_wb("life_expectancy_years.csv")
tfr_df = load_wb("children_per_woman_total_fertility.csv")
u5_df  = load_wb("child_mortality_u5.csv")


def _name(df, c):
    if c in df.index:
        return c
    return _SHARED_NAME_MAP.get(c, c) if _SHARED_NAME_MAP.get(c, c) in df.index else None


def _val(df, c, year):
    nm = _name(df, c)
    if nm is None or str(year) not in df.columns:
        return np.nan
    v = df.loc[nm, str(year)]
    return float(v) if pd.notna(v) else np.nan


print(f"Building T+{LAG} panel...")
rows = []
for c in sorted(edu_long["country"].unique()):
    sub = edu_long[edu_long["country"] == c].set_index("year")
    for t in EDU_YEARS:
        if t not in sub.index:
            continue
        tp = t + LAG
        low_t = sub.loc[t, "lower_sec"]
        pri_t = sub.loc[t, "primary"]
        if np.isnan(low_t):
            continue
        gdp_t  = _val(gdp_df, c, t)
        gdp_tp = _val(gdp_df, c, tp)
        le_tp  = _val(le_df,  c, tp)
        tfr_tp = _val(tfr_df, c, tp)
        u5_tp  = _val(u5_df,  c, tp)
        rows.append({
            "country": c, "t": t,
            "low_t": low_t, "pri_t": pri_t,
            "log_gdp_t":  np.log(gdp_t)  if pd.notna(gdp_t)  and gdp_t  > 0 else np.nan,
            "log_gdp_tp": np.log(gdp_tp) if pd.notna(gdp_tp) and gdp_tp > 0 else np.nan,
            "log_le_tp":  np.log(le_tp)  if pd.notna(le_tp)  and le_tp  > 0 else np.nan,
            "log_tfr_tp": np.log(tfr_tp) if pd.notna(tfr_tp) and tfr_tp > 0 else np.nan,
            "log_u5_tp":  np.log(u5_tp)  if pd.notna(u5_tp)  and u5_tp  > 0 else np.nan,
        })

panel = pd.DataFrame(rows)
print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries")


def fe_clustered(X_cols, y_col, data, *, year_fe=False, country_col="country"):
    sub = data.dropna(subset=X_cols + [y_col]).copy()
    if year_fe:
        sub = sub.dropna(subset=["t"])
        for col in X_cols + [y_col]:
            for _ in range(20):
                sub[col] = sub[col] - sub.groupby(country_col)[col].transform("mean")
                sub[col] = sub[col] - sub.groupby("t")[col].transform("mean")
    else:
        for col in X_cols + [y_col]:
            sub[col] = sub[col] - sub.groupby(country_col)[col].transform("mean")

    Xd = sub[X_cols].to_numpy(dtype=float)
    yd = sub[y_col].to_numpy(dtype=float)
    countries = sub[country_col].to_numpy()
    ok = ~np.isnan(Xd).any(axis=1) & ~np.isnan(yd)
    Xd, yd, countries = Xd[ok], yd[ok], countries[ok]
    if len(yd) < 10:
        return None
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ yd
    resid = yd - Xd @ beta
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
    ss_tot = float(np.sum(yd ** 2))
    ss_res = float(np.sum(resid ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "coefs": dict(zip(X_cols, beta.tolist())),
        "se":    dict(zip(X_cols, se.tolist())),
        "p":     dict(zip(X_cols, pvals.tolist())),
        "r2":    float(r2),
        "n":     int(N),
        "countries": int(G),
    }


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


PANEL_OUTCOMES = [
    ("log_gdp", "low_t", "log_gdp_tp", "log GDP"),
    ("log_le",  "low_t", "log_le_tp",  "log(LE)"),
    ("log_tfr", "pri_t", "log_tfr_tp", "log(TFR) [primary]"),
    ("log_u5",  "low_t", "log_u5_tp",  "log(U5MR)"),
]

SPECS = [
    ("c1_edu",         ["{pred}"],                          False),
    ("c2_edu_gdp",     ["{pred}", "log_gdp_t"],             False),
    ("c3_edu_yfe",     ["{pred}"],                          True),
    ("c4_edu_gdp_yfe", ["{pred}", "log_gdp_t"],             True),
]


def run_outcome(panel_df, predictor, outcome, label):
    out = {}
    for spec_id, xcols_tpl, year_fe in SPECS:
        xcols = [c.replace("{pred}", predictor) for c in xcols_tpl]
        xcols = list(dict.fromkeys(xcols))
        res = fe_clustered(xcols, outcome, panel_df, year_fe=year_fe)
        out[spec_id] = res
        if res is None:
            print(f"  {label} | {spec_id}: insufficient data")
            continue
        b = res["coefs"][predictor]
        s = res["se"][predictor]
        p = res["p"][predictor]
        print(f"  {label:<22} | {spec_id:<18} "
              f"β={b:>+8.4f}  SE={s:.4f}  p={p:.3g}{stars(p)}  "
              f"n={res['n']:>5d}  C={res['countries']:>3d}  R²={res['r2']:.3f}")
    return out


print(f"\nT+{LAG} panel — outcome(T+{LAG}) on edu(T):")
results = {}
for tag, predictor, outcome, label in PANEL_OUTCOMES:
    results[tag] = run_outcome(panel, predictor, outcome, label)


def _pack(results_dict, label):
    flat = {}
    for tag, specs in results_dict.items():
        for spec_id, res in specs.items():
            if res is None:
                continue
            for var, b in res["coefs"].items():
                flat[f"{label}.{tag}.{spec_id}.{var}.beta"] = round(b, 4)
                flat[f"{label}.{tag}.{spec_id}.{var}.se"]   = round(res["se"][var], 4)
                flat[f"{label}.{tag}.{spec_id}.{var}.p"]    = float(f"{res['p'][var]:.4g}")
            flat[f"{label}.{tag}.{spec_id}.r2"]        = round(res["r2"], 3)
            flat[f"{label}.{tag}.{spec_id}.n"]         = int(res["n"])
            flat[f"{label}.{tag}.{spec_id}.countries"] = int(res["countries"])
    return flat


write_checkin(
    "table_short_15yr.json",
    {
        "method": (
            f"Leader's-timeline (T+{LAG}) version of Table 7 Panel A. "
            "Regresses each outcome at T+15 on lower-secondary completion "
            "at T (primary for log(TFR)). Country FE; country-clustered "
            "SEs. Four columns: c1 edu only, c2 + log GDP at T, c3 + year "
            "FE, c4 + log GDP + year FE. Education years 1960-2000 step 5. "
            "Empirical context: this is the U5MR peak lag (childrearing "
            "window: 20-24 cohort at T has children born and surviving to "
            "age 5 within T+10 to T+15). Forward-projection significance "
            "for policy: today's primary entrants reach age 21 in 15 years."
        ),
        "numbers": _pack(results, "panel"),
    },
    script_path="scripts/wcde/table_short_15yr.py",
)
print("\nDone.")
