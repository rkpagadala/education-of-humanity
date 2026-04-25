"""
table7_stepwise.py
==================
R2.15 / R2.16 / R2.19: produce the four-column stepwise structure for
Table 7 Panels A and B.

Panel A — outcomes at T+25 on lower-secondary (or primary, for TFR) at T:
  (1) edu only,                country FE
  (2) edu + initial outcome,   country FE      [previous Table 7 spec]
  (3) edu + initial outcome + log GDP per capita, country FE
  (4) col 3 + year FE

Panel B — lower-secondary at T+25 on log GDP per capita at T:
  (1) GDP only,                country FE      [previous spec]
  (2) GDP + initial education, country FE      [previous spec]
  (3) GDP only,                country & year FE
  (4) GDP + initial education, country & year FE

Two sample versions per panel:
  - "max":    drop NA only on the regressors actually used by the spec;
              this is the GMM estimate on every observation available
              (same as the previous Table 7 headline).
  - "common": restrict to the GDP-merged sample (n=927, 179 countries
              for Panel A; same sample for Panel B). Reviewer's preferred
              "common" structure for direct cross-spec comparability —
              shipped as the appendix robustness check.

Outcomes (Panel A):
  - log GDP per capita  (lower-secondary as predictor)
  - log(LE)             (lower-secondary as predictor)
  - log(TFR)            (primary as predictor — the operative channel)
  - log(U5MR)           (lower-secondary as predictor)

Output: checkin/table7_stepwise.json
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import (PROC, REGIONS, write_checkin, load_wb,
                     NAME_MAP as _SHARED_NAME_MAP)

# ── Panel construction (lifted from education_outcomes.py) ───────────────────
edu_long = pd.read_csv(os.path.join(PROC, "cohort_completion_both_long.csv"))
edu_long = edu_long[~edu_long["country"].isin(REGIONS)]
edu_long = edu_long.rename(columns={
    "cohort_year": "year",
    "lower_sec": "lower_sec",
    "primary":   "primary",
})
edu_long["country"] = edu_long["country"].str.lower().str.strip()

gdp_df  = load_wb("gdppercapita_us_inflation_adjusted.csv")
le_df   = load_wb("life_expectancy_years.csv")
tfr_df  = load_wb("children_per_woman_total_fertility.csv")
u5_df   = load_wb("child_mortality_u5.csv")


def _name(df, c):
    """Map a WCDE country name to the WB index by NAME_MAP / fallback."""
    if c in df.index:
        return c
    mapped = _SHARED_NAME_MAP.get(c, c)
    if mapped in df.index:
        return mapped
    return None


def _val(df, c, year):
    nm = _name(df, c)
    if nm is None or str(year) not in df.columns:
        return np.nan
    v = df.loc[nm, str(year)]
    return float(v) if pd.notna(v) else np.nan


EDU_YEARS = [1960, 1965, 1970, 1975, 1980, 1985, 1990]
LAG = 25

print("Building panel...")
rows = []
for c in sorted(edu_long["country"].unique()):
    sub = edu_long[edu_long["country"] == c].set_index("year")
    for t in EDU_YEARS:
        if t not in sub.index:
            continue
        tp = t + LAG
        low_t  = sub.loc[t, "lower_sec"]
        pri_t  = sub.loc[t, "primary"]
        if np.isnan(low_t):
            continue
        gdp_t  = _val(gdp_df, c, t)
        gdp_tp = _val(gdp_df, c, tp)
        le_t   = _val(le_df,  c, t)
        le_tp  = _val(le_df,  c, tp)
        tfr_t  = _val(tfr_df, c, t)
        tfr_tp = _val(tfr_df, c, tp)
        u5_t   = _val(u5_df,  c, t)
        u5_tp  = _val(u5_df,  c, tp)
        rows.append({
            "country": c, "t": t,
            "low_t": low_t, "pri_t": pri_t,
            "log_gdp_t":   np.log(gdp_t)  if pd.notna(gdp_t)  and gdp_t  > 0 else np.nan,
            "log_gdp_tp":  np.log(gdp_tp) if pd.notna(gdp_tp) and gdp_tp > 0 else np.nan,
            "le_t": le_t, "le_tp": le_tp,
            "tfr_t": tfr_t, "tfr_tp": tfr_tp,
            "u5_t":  u5_t,  "u5_tp":  u5_tp,
        })
panel = pd.DataFrame(rows)
for col in ["le_t", "le_tp", "tfr_t", "tfr_tp", "u5_t", "u5_tp"]:
    panel[f"log_{col}"] = np.log(panel[col].where(panel[col] > 0))

# Common-sample mask: every cell needed by the most demanding spec
# (col 4: edu + initial outcome + log GDP, country & year FE) is non-missing.
# Uses log_gdp_t, log_gdp_tp (or whichever outcome row uses), edu, initial.
print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries")


# ── Estimator: country FE (+ optional year FE), country-clustered SE ─────────
def fe_clustered(X_cols, y_col, data, *, year_fe=False, country_col="country"):
    sub = data.dropna(subset=X_cols + [y_col]).copy()
    if year_fe:
        sub = sub.dropna(subset=["t"])
        # double-demeaning trick: subtract country mean, then subtract year mean
        # (one-step Wansbeek-Kapteyn projection works for balanced; for our
        # unbalanced case use iterated demeaning to convergence).
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


# ── Panel A specs ────────────────────────────────────────────────────────────
# Each row corresponds to a (predictor, outcome, initial-outcome col) triple.
PANEL_A_OUTCOMES = [
    # (tag,         predictor, outcome,         initial-outcome col, label)
    ("log_gdp",     "low_t",   "log_gdp_tp",    "log_gdp_t",  "log GDP"),
    ("log_le",      "low_t",   "log_le_tp",     "log_le_t",   "log(LE)"),
    ("log_tfr",     "pri_t",   "log_tfr_tp",    "log_tfr_t",  "log(TFR) [primary]"),
    ("log_u5",      "low_t",   "log_u5_tp",     "log_u5_t",   "log(U5MR)"),
]

# Common sample: every observation must have lower-sec + log_gdp_t + log_gdp_tp
# + log_le_t/tp + log_tfr_t/tp + log_u5_t/tp + primary at T.
COMMON_VARS = ["low_t", "pri_t",
               "log_gdp_t", "log_gdp_tp",
               "log_le_t",  "log_le_tp",
               "log_tfr_t", "log_tfr_tp",
               "log_u5_t",  "log_u5_tp"]
common_panel = panel.dropna(subset=COMMON_VARS).copy()
print(f"  Common (GDP+LE+TFR+U5) sample: {len(common_panel)} obs, "
      f"{common_panel['country'].nunique()} countries")


def panel_a_specs(predictor, init_col):
    """Build the 4 specs; deduplicate when init_col is itself log_gdp_t."""
    base_c3 = [predictor, init_col]
    if init_col != "log_gdp_t":
        base_c3 = base_c3 + ["log_gdp_t"]
    return [
        ("c1_edu_only",          [predictor],                False),
        ("c2_edu_init",          [predictor, init_col],      False),
        ("c3_edu_init_gdp",      base_c3,                    False),
        ("c4_edu_init_gdp_yfe",  base_c3,                    True),
    ]


def run_outcome_specs(df, predictor, outcome, init_col, label):
    out = {}
    for spec_id, xcols, year_fe in panel_a_specs(predictor, init_col):
        # Skip GDP-as-control specs when the OUTCOME is log_gdp_tp
        # — the regression would put log_gdp_t as both control and on the
        # causal path; reviewer's recommendation already accounts for this
        # by listing GDP as the outcome in its own row. We still report c3/c4
        # for log_gdp by using the *initial* GDP as a convergence control,
        # i.e. the same "log_gdp_t" already appears as the initial-outcome
        # column — so c2 == c3 in that case. We let it run and report identical.
        res = fe_clustered(xcols, outcome, df, year_fe=year_fe)
        out[spec_id] = res
        if res is None:
            print(f"  {label} | {spec_id}: insufficient data")
            continue
        b = res["coefs"][predictor]
        s = res["se"][predictor]
        p = res["p"][predictor]
        print(f"  {label:<22} | {spec_id:<22} "
              f"β={b:>+8.4f}  SE={s:.4f}  p={p:.3g}{stars(p)}  "
              f"n={res['n']:>5d}  C={res['countries']:>3d}  R²={res['r2']:.3f}")
    return out


print("\nPanel A (max sample per outcome):")
results_A_max = {}
for tag, predictor, outcome, init_col, label in PANEL_A_OUTCOMES:
    results_A_max[tag] = run_outcome_specs(panel, predictor, outcome, init_col, label)

print("\nPanel A (common 927/179 sample):")
results_A_common = {}
for tag, predictor, outcome, init_col, label in PANEL_A_OUTCOMES:
    results_A_common[tag] = run_outcome_specs(common_panel, predictor, outcome,
                                                init_col, label)


# ── Panel B specs ────────────────────────────────────────────────────────────
# Outcome: lower-secondary at T+25; predictor: log GDP at T (and initial edu).
# We need lower-sec at T+25 on the panel — the existing producer pulls this
# from low_tp25 which we don't have here. Let's add it.
print("\nAdding lower-sec at T+25 column...")
extra_rows = []
edu_long_idx = edu_long.set_index(["country", "year"])
for _, row in panel.iterrows():
    tp = row["t"] + LAG
    try:
        low_tp = edu_long_idx.loc[(row["country"], tp), "lower_sec"]
        extra_rows.append(float(low_tp) if pd.notna(low_tp) else np.nan)
    except KeyError:
        extra_rows.append(np.nan)
panel["low_tp"] = extra_rows
common_panel["low_tp"] = panel.loc[common_panel.index, "low_tp"]

PANEL_B_SPECS = [
    ("b1_gdp_only",            ["log_gdp_t"],                       False),
    ("b2_gdp_init_edu",        ["log_gdp_t", "low_t"],               False),
    ("b3_gdp_only_yfe",        ["log_gdp_t"],                       True),
    ("b4_gdp_init_edu_yfe",    ["log_gdp_t", "low_t"],               True),
]


def run_panel_b(df, label):
    out = {}
    for spec_id, xcols, year_fe in PANEL_B_SPECS:
        res = fe_clustered(xcols, "low_tp", df, year_fe=year_fe)
        out[spec_id] = res
        if res is None:
            print(f"  {label} | {spec_id}: insufficient data")
            continue
        b = res["coefs"]["log_gdp_t"]
        s = res["se"]["log_gdp_t"]
        p = res["p"]["log_gdp_t"]
        print(f"  {label:<14} | {spec_id:<22} "
              f"β_GDP={b:>+8.4f}  SE={s:.4f}  p={p:.3g}{stars(p)}  "
              f"n={res['n']:>5d}  C={res['countries']:>3d}  R²={res['r2']:.3f}")
    return out


print("\nPanel B (max sample):")
results_B_max = run_panel_b(panel, "Panel B max")
print("\nPanel B (common 927/179 sample):")
results_B_common = run_panel_b(common_panel, "Panel B common")


# ── Pack results into checkin JSON ───────────────────────────────────────────
def _pack_panel_a(results_dict, label):
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


def _pack_panel_b(results_dict, label):
    flat = {}
    for spec_id, res in results_dict.items():
        if res is None:
            continue
        for var, b in res["coefs"].items():
            flat[f"{label}.{spec_id}.{var}.beta"] = round(b, 4)
            flat[f"{label}.{spec_id}.{var}.se"]   = round(res["se"][var], 4)
            flat[f"{label}.{spec_id}.{var}.p"]    = float(f"{res['p'][var]:.4g}")
        flat[f"{label}.{spec_id}.r2"]        = round(res["r2"], 3)
        flat[f"{label}.{spec_id}.n"]         = int(res["n"])
        flat[f"{label}.{spec_id}.countries"] = int(res["countries"])
    return flat


numbers = {}
numbers.update(_pack_panel_a(results_A_max,    "panelA_max"))
numbers.update(_pack_panel_a(results_A_common, "panelA_common"))
numbers.update(_pack_panel_b(results_B_max,    "panelB_max"))
numbers.update(_pack_panel_b(results_B_common, "panelB_common"))

write_checkin(
    "table7_stepwise.json",
    {
        "method": (
            "Table 7 four-column stepwise. Panel A regresses each outcome "
            "at T+25 on lower-secondary completion at T (primary for log(TFR)) "
            "with increasing controls: c1 edu only, c2 + initial outcome, "
            "c3 + log GDP, c4 + year FE. Panel B regresses lower-secondary "
            "at T+25 on log GDP at T: b1 GDP only, b2 + initial edu, b3 GDP "
            "only + year FE, b4 + initial edu + year FE. Country fixed "
            "effects throughout; country-clustered standard errors. Two "
            "sample versions: max (drop NAs only on regressors used) and "
            "common (drop NAs on the union of vars across the most "
            "demanding spec; same sample as the GDP row of the previous "
            "Table 7 = 927/179)."
        ),
        "numbers": numbers,
    },
    script_path="scripts/wcde/table7_stepwise.py",
)

print("\nDone.")
