"""
table_contemporaneous.py
=========================
Panel A: each outcome on lower-secondary completion at its canonical
per-outcome lag per _shared.py. The lags are NOT uniform — each outcome
is measured at the lag that matches the cohort's biological vantage on
that outcome:

  - log GDP per capita    lag 0    LAG_CONTEMPORANEOUS — cohort's own
                                   productivity, same year (lower-sec)
  - log(LE)               lag 25   LAG_LE — society-wide average longevity
                                   is dominated by older adults, so it
                                   reflects the schooling of the ~45-49
                                   cohort measured ~25 years earlier; we
                                   read it one generation forward (lower-sec)
  - log(TFR)              lag 5    LAG_TFR — biological parent timing, the
                                   cohort aged 20-24 at T is at reproductive
                                   peak at T+5 (lower-sec)
  - log(U5MR)             lag 12   LAG_CHILDREARING — the cohort's own
                                   children must be born and survive to 5
                                   (lower-sec)

Note: the GDP-at-T column used as a *control* in c2 is always lag 0
regardless of the outcome lag — it conditions on income at the predictor
year, not at the outcome year.

Two-column stepwise per outcome: (1) edu only; (2) + log GDP at T as a
control. No year-fixed-effects specifications — year FE over-control by
absorbing the global education-driven time trend.

The cross-generation (T+28) variant for all outcomes lives in
table7_stepwise.py and is retained as a forward-prediction identification
robustness.

Output: checkin/table_contemporaneous.json
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import (PROC, REGIONS, write_checkin, load_wb,
                     NAME_MAP as _SHARED_NAME_MAP,
                     LAG_CONTEMPORANEOUS, LAG_LE, LAG_TFR, LAG_CHILDREARING)

EDU_YEARS = list(range(1960, 2016, 5))
# Per-outcome canonical lag (NOT uniform — see module docstring).
OUTCOME_LAG = {
    "log_gdp": LAG_CONTEMPORANEOUS,  # lag 0  — cohort's own productivity
    "log_le":  LAG_LE,               # lag 25 — society-wide longevity vantage
    "log_tfr": LAG_TFR,              # lag 5  — biological parent timing
    "log_u5":  LAG_CHILDREARING,     # lag 12 — childrearing window
}


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


print("Building canonical-lag panel...")
rows = []
for c in sorted(edu_long["country"].unique()):
    sub = edu_long[edu_long["country"] == c].set_index("year")
    for t in EDU_YEARS:
        if t not in sub.index:
            continue
        low_t = sub.loc[t, "lower_sec"]
        pri_t = sub.loc[t, "primary"]
        if np.isnan(low_t):
            continue
        # Outcomes at canonical lag per _shared.py
        gdp_t   = _val(gdp_df, c, t + OUTCOME_LAG["log_gdp"])
        le_t    = _val(le_df,  c, t + OUTCOME_LAG["log_le"])
        tfr_t   = _val(tfr_df, c, t + OUTCOME_LAG["log_tfr"])
        u5_tp12 = _val(u5_df,  c, t + OUTCOME_LAG["log_u5"])
        # GDP control at predictor year T (always lag 0 for the control)
        gdp_ctrl_t = _val(gdp_df, c, t)
        rows.append({
            "country": c, "t": t,
            "low_t": low_t, "pri_t": pri_t,
            "log_gdp_t":  np.log(gdp_t)   if pd.notna(gdp_t)   and gdp_t   > 0 else np.nan,
            "log_le_t":   np.log(le_t)    if pd.notna(le_t)    and le_t    > 0 else np.nan,
            "log_tfr_t":  np.log(tfr_t)   if pd.notna(tfr_t)   and tfr_t   > 0 else np.nan,
            "log_u5_tp12":np.log(u5_tp12) if pd.notna(u5_tp12) and u5_tp12 > 0 else np.nan,
            # Separate GDP-at-T column for use as a control (always lag 0).
            "log_gdp_ctrl_t": np.log(gdp_ctrl_t) if pd.notna(gdp_ctrl_t) and gdp_ctrl_t > 0 else np.nan,
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
    # (tag, predictor, outcome, label)
    ("log_gdp", "low_t", "log_gdp_t",   "log GDP        [lag 0]"),
    ("log_le",  "low_t", "log_le_t",    "log(LE)        [lag 25]"),
    ("log_tfr", "low_t", "log_tfr_t",   "log(TFR)       [lag 5]"),
    ("log_u5",  "low_t", "log_u5_tp12", "log(U5MR)      [lag 12]"),
]

SPECS = [
    # (spec_id, predictors, year_fe)
    # Two-column stepwise only: edu, +log GDP. No year-FE specs — year FE
    # over-control by absorbing the global education-driven time trend.
    ("c1_edu",         ["{pred}"],                              False),
    ("c2_edu_gdp",     ["{pred}", "log_gdp_ctrl_t"],            False),
]


def run_outcome(panel_df, predictor, outcome, label):
    out = {}
    for spec_id, xcols_tpl, year_fe in SPECS:
        xcols = [c.replace("{pred}", predictor) for c in xcols_tpl]
        # Skip GDP-as-control when outcome itself is log_gdp_t
        # (would be a tautology — GDP at T on edu at T already implicitly
        # controls for GDP at T).
        if outcome == "log_gdp_t" and "log_gdp_ctrl_t" in xcols and predictor != "log_gdp_ctrl_t":
            xcols = [c for c in xcols if c != "log_gdp_ctrl_t"]
        # Dedupe (in case predictor already is log_gdp_t)
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


print("\nCanonical-lag panel — outcome at canonical lag on lower-sec at T:")
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
    "table_contemporaneous.json",
    {
        "method": (
            "Panel A at per-outcome canonical lag (per _shared.py). "
            "Each outcome regressed on lower-secondary completion at T: "
            "GDP at lag 0 (contemporaneous, LAG_CONTEMPORANEOUS — cohort's "
            "own productivity); LE at lag 25 (LAG_LE — society-wide average "
            "longevity is dominated by older adults, so it reflects the "
            "schooling of the ~45-49 cohort measured ~25 years earlier); "
            "TFR at lag 5 (biological parent timing, LAG_TFR — the cohort "
            "aged 20-24 at T is at reproductive peak at T+5); U5MR at lag "
            "12 (childrearing window, LAG_CHILDREARING — the cohort's own "
            "children must be born and survive to 5). The GDP-at-T control "
            "in c2 is always lag 0. Country FE "
            "throughout; country-clustered SEs. Two columns: c1 edu only, "
            "c2 + log GDP at T (control). No year-fixed-effects "
            "specifications — year FE over-control by absorbing the global "
            "education-driven time trend. Education years 1960-2015 step 5. "
            "Cross-generation "
            "(T+28) variant lives in table7_stepwise.json as forward-"
            "prediction identification robustness."
        ),
        "numbers": _pack(results, "panel"),
    },
    script_path="scripts/wcde/table_contemporaneous.py",
)
print("\nDone.")
