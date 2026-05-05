# =============================================================================
# PAPER REFERENCE
# Script:  scripts/robustness/barro_lee_replication.py
# Paper:   "Education of Nations"
#
# Produces:
#   Replication of core forward-prediction regressions using Barro-Lee
#   education data instead of WCDE v3, to show results are not an artifact
#   of WCDE's reconstruction methodology.
#
#   Four tests:
#     1. Barro-Lee education (T) → TFR, LE at T+25 (full panel, FE)
#     2. WCDE restricted to post-1970 only (where data quality is highest)
#     3. Side-by-side R² comparison: WCDE full vs BL vs WCDE post-1970
#     4. FWL residualization on Barro-Lee: GDP stripped of education predicts nothing
#
# Inputs:
#   data/barro_lee.csv (extracted from World Bank EdStats)
#   wcde/data/processed/completion_both_long.csv
#   data/life_expectancy_years.csv
#   data/children_per_woman_total_fertility.csv
#   data/gdppercapita_us_inflation_adjusted.csv
#
# Outputs:
#   checkin/barro_lee_replication.json
# =============================================================================
"""
barro_lee_replication.py

Replicate the forward-prediction regression (education at T → outcomes
at T+25) using Barro-Lee data. Show the result holds on an independent
education dataset with different methodology, and on a restricted post-1970
WCDE panel where reconstruction quality is highest. Also replicate the
FWL residualization: GDP stripped of education's contribution has near-zero
predictive power for outcomes — on Barro-Lee data, not just WCDE.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from _shared import PROC, DATA, CHECKIN, REGIONS, NAME_MAP, write_checkin

# ── Load outcome data (World Bank WDI) ───────────────────────────────────
le_raw = pd.read_csv(os.path.join(DATA, "life_expectancy_years.csv"))
tfr_raw = pd.read_csv(os.path.join(DATA, "children_per_woman_total_fertility.csv"))

def load_wb(df_raw):
    """Load WB CSV into country×year lookup."""
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]
    country_col = df.columns[0]
    df = df.set_index(country_col)
    df.index = df.index.str.strip().str.lower()
    # Keep only numeric year columns
    year_cols = [c for c in df.columns if c.isdigit()]
    df = df[year_cols]
    for c in year_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

gdp_raw = pd.read_csv(os.path.join(DATA, "gdppercapita_us_inflation_adjusted.csv"))

le_wb = load_wb(le_raw)
tfr_wb = load_wb(tfr_raw)
gdp_wb = load_wb(gdp_raw)

# Barro-Lee country name → WB name mapping
BL_TO_WB = {
    "Korea, Rep.": "korea, rep.",
    "Iran, Islamic Rep.": "iran, islamic rep.",
    "Egypt, Arab Rep.": "egypt, arab rep.",
    "Venezuela, RB": "venezuela, rb",
    "Congo, Dem. Rep.": "congo, dem. rep.",
    "Congo, Rep.": "congo, rep.",
    "Gambia, The": "gambia, the",
    "Lao PDR": "lao pdr",
    "Syrian Arab Republic": "syrian arab republic",
    "Yemen, Rep.": "yemen, rep.",
    "Kyrgyz Republic": "kyrgyz republic",
}

def get_wb_val(wb_df, country_bl, year):
    """Look up WB value for a Barro-Lee country name."""
    key = BL_TO_WB.get(country_bl, country_bl).lower()
    if key in wb_df.index:
        yr_str = str(year)
        if yr_str in wb_df.columns:
            v = wb_df.loc[key, yr_str]
            if pd.notna(v):
                return float(v)
    return np.nan


def fe_regression(panel, x_col, y_col, country_col="country"):
    """Run country-FE regression: demean then OLS."""
    sub = panel.dropna(subset=[x_col, y_col, country_col]).copy()
    if len(sub) < 20:
        return None
    for col in [x_col, y_col]:
        sub[col + "_dm"] = sub[col] - sub.groupby(country_col)[col].transform("mean")
    X = sub[[x_col + "_dm"]]
    y = sub[y_col + "_dm"]
    model = sm.OLS(y, X).fit(cov_type='cluster',
                              cov_kwds={'groups': sub[country_col]})
    return {
        "beta": float(model.params.iloc[0]),
        "se": float(model.bse.iloc[0]),
        "t": float(model.tvalues.iloc[0]),
        "p": float(model.pvalues.iloc[0]),
        "r2_within": float(model.rsquared),
        "n": int(len(sub)),
        "n_countries": int(sub[country_col].nunique()),
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: Barro-Lee education → outcomes at T+25
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: BARRO-LEE FORWARD PREDICTION (education T → outcomes T+25)")
print("=" * 70)

bl = pd.read_csv(os.path.join(DATA, "barro_lee.csv"))

# Source: Barro-Lee v3.0 (Sep 2021), from GitHub:
#   https://raw.githubusercontent.com/barrolee/BarroLeeDataSet/master/BLData/BL_v3_MF.csv
# Filtered to age 15-24, both sexes. Columns:
#   atleast_some_sec = ls + lh (% with at least some secondary education)
#   yr_sch = mean years of total schooling
# "atleast_some_sec" is the closest BL analogue to WCDE lower sec completion.
#
# BL data: 146 countries, 5-year intervals 1950-2015.
# For T+25: edu at T predicts outcome at T+25.
# Using T = {1960..1990} → outcomes at {1985..2015}.

# BL uses its own country names; map to WB names for outcome lookup
BL_TO_WB_NAME = {
    "Republic of Korea": "korea, rep.",
    "Iran, Islamic Republic of": "iran, islamic rep.",
    "Egypt": "egypt, arab rep.",
    "Venezuela": "venezuela, rb",
    "Democratic Republic of the Congo": "congo, dem. rep.",
    "Congo": "congo, rep.",
    "Gambia": "gambia, the",
    "Lao People's Democratic Republic": "lao pdr",
    "Syrian Arab Republic": "syrian arab republic",
    "Yemen": "yemen, rep.",
    "Kyrgyzstan": "kyrgyz republic",
    "Viet Nam": "vietnam",
    "United Republic of Tanzania": "tanzania",
    "Bolivia (Plurinational State of)": "bolivia",
    "United States of America": "united states",
    "United Kingdom of Great Britain and Northern Ireland": "united kingdom",
    "Russian Federation": "russia",
    "Republic of Moldova": "moldova",
    "Eswatini": "eswatini",
    "Cabo Verde": "cabo verde",
    "Czechia": "czech republic",
    "North Macedonia": "north macedonia",
    "Côte d'Ivoire": "cote d'ivoire",
    "Democratic People's Republic of Korea": "korea, dem. people's rep.",
    "China, Hong Kong Special Administrative Region": "hong kong sar, china",
    "China, Macao Special Administrative Region": "macao sar, china",
}

T_YEARS_BL = [1960, 1965, 1970, 1975, 1980, 1985, 1990]  # T+25 = 1985..2015
LAG = 25

rows_bl = []
for _, row in bl.iterrows():
    t = int(row["year"])
    if t not in T_YEARS_BL:
        continue
    country = row["country"]
    edu_sec = row.get("atleast_some_sec", np.nan)
    edu_mys = row.get("yr_sch", np.nan)

    wb_name = BL_TO_WB_NAME.get(country, country).lower()
    le_val = np.nan
    tfr_val = np.nan
    yr_str = str(t + LAG)
    if wb_name in le_wb.index and yr_str in le_wb.columns:
        le_val = le_wb.loc[wb_name, yr_str]
        le_val = float(le_val) if pd.notna(le_val) else np.nan
    if wb_name in tfr_wb.index and yr_str in tfr_wb.columns:
        tfr_val = tfr_wb.loc[wb_name, yr_str]
        tfr_val = float(tfr_val) if pd.notna(tfr_val) else np.nan

    # GDP at time T (for residualization)
    gdp_val = np.nan
    t_str = str(t)
    if wb_name in gdp_wb.index and t_str in gdp_wb.columns:
        gv = gdp_wb.loc[wb_name, t_str]
        if pd.notna(gv) and float(gv) > 0:
            gdp_val = float(gv)

    rows_bl.append({
        "country": country, "t": t,
        "edu_sec": edu_sec, "edu_mys": edu_mys,
        "le_t25": le_val, "tfr_t25": tfr_val,
        "log_gdp_t": np.log(gdp_val) if not np.isnan(gdp_val) else np.nan,
    })

bl_panel = pd.DataFrame(rows_bl)
print(f"BL panel: {len(bl_panel)} obs, {bl_panel['country'].nunique()} countries")
print(f"  LE coverage: {bl_panel['le_t25'].notna().sum()}")
print(f"  TFR coverage: {bl_panel['tfr_t25'].notna().sum()}")

results_bl = {}
for edu_var, edu_label in [("edu_sec", "At-least-some-secondary (%)"),
                            ("edu_mys", "Mean years of schooling")]:
    for outcome, out_label in [("le_t25", "Life expectancy"),
                                ("tfr_t25", "TFR")]:
        key = f"BL_{edu_var}→{outcome}"
        res = fe_regression(bl_panel, edu_var, outcome)
        results_bl[key] = res
        if res:
            print(f"\n  {edu_label} → {out_label} (T+25), FE:")
            print(f"    β = {res['beta']:.4f}, t = {res['t']:.2f}, p = {res['p']:.4f}")
            print(f"    R² (within) = {res['r2_within']:.3f}")
            print(f"    N = {res['n']}, countries = {res['n_countries']}")


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: WCDE FULL PANEL (baseline comparison)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: WCDE FULL PANEL (education T → outcomes T+25)")
print("=" * 70)

edu_wcde = pd.read_csv(os.path.join(PROC, "completion_both_long.csv"))
edu_wcde = edu_wcde[~edu_wcde["country"].isin(REGIONS)].copy()

T_YEARS_WCDE = [1960, 1965, 1970, 1975, 1980, 1985, 1990]

# Need to map WCDE names to WB names
rows_wcde = []
for _, row in edu_wcde.iterrows():
    t = int(row["year"])
    if t not in T_YEARS_WCDE:
        continue
    country_wcde = row["country"]
    edu_val = row["lower_sec"]
    if pd.isna(edu_val):
        continue

    # Map WCDE name to WB name for outcome lookup
    wb_name = NAME_MAP.get(country_wcde, country_wcde).lower()
    le_val = np.nan
    tfr_val = np.nan
    yr_str = str(t + LAG)
    if wb_name in le_wb.index and yr_str in le_wb.columns:
        le_val = le_wb.loc[wb_name, yr_str]
    if wb_name in tfr_wb.index and yr_str in tfr_wb.columns:
        tfr_val = tfr_wb.loc[wb_name, yr_str]

    # GDP at time T (for residualization comparison)
    gdp_val = np.nan
    t_str = str(t)
    if wb_name in gdp_wb.index and t_str in gdp_wb.columns:
        gv = gdp_wb.loc[wb_name, t_str]
        if pd.notna(gv) and float(gv) > 0:
            gdp_val = float(gv)

    rows_wcde.append({
        "country": country_wcde, "t": t,
        "edu_ls": edu_val,
        "le_t25": float(le_val) if pd.notna(le_val) else np.nan,
        "tfr_t25": float(tfr_val) if pd.notna(tfr_val) else np.nan,
        "log_gdp_t": np.log(gdp_val) if not np.isnan(gdp_val) else np.nan,
    })

wcde_panel = pd.DataFrame(rows_wcde)
print(f"WCDE full panel: {len(wcde_panel)} obs, {wcde_panel['country'].nunique()} countries")

results_wcde_full = {}
for outcome, out_label in [("le_t25", "Life expectancy"), ("tfr_t25", "TFR")]:
    key = f"WCDE_full→{outcome}"
    res = fe_regression(wcde_panel, "edu_ls", outcome)
    results_wcde_full[key] = res
    if res:
        print(f"\n  Lower sec completion → {out_label} (T+25), FE:")
        print(f"    β = {res['beta']:.4f}, t = {res['t']:.2f}, p = {res['p']:.4f}")
        print(f"    R² (within) = {res['r2_within']:.3f}")
        print(f"    N = {res['n']}, countries = {res['n_countries']}")


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: WCDE POST-1970 ONLY (restricted panel)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: WCDE POST-1970 PANEL (education T → outcomes T+25)")
print("=" * 70)

wcde_post70 = wcde_panel[wcde_panel["t"] >= 1970].copy()
print(f"WCDE post-1970 panel: {len(wcde_post70)} obs, {wcde_post70['country'].nunique()} countries")

results_wcde_post70 = {}
for outcome, out_label in [("le_t25", "Life expectancy"), ("tfr_t25", "TFR")]:
    key = f"WCDE_post70→{outcome}"
    res = fe_regression(wcde_post70, "edu_ls", outcome)
    results_wcde_post70[key] = res
    if res:
        print(f"\n  Lower sec completion → {out_label} (T+25), FE:")
        print(f"    β = {res['beta']:.4f}, t = {res['t']:.2f}, p = {res['p']:.4f}")
        print(f"    R² (within) = {res['r2_within']:.3f}")
        print(f"    N = {res['n']}, countries = {res['n_countries']}")


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY: R² (within) COMPARISON")
print("=" * 70)
print(f"\n{'Specification':<45} {'→ LE':>10} {'→ TFR':>10}")
print("-" * 65)

def get_r2(results, key):
    r = results.get(key, {})
    return f"{r['r2_within']:.3f}" if r and r.get('r2_within') else "n/a"

def get_p(results, key):
    r = results.get(key, {})
    return r.get('p', 999) if r else 999

print(f"{'WCDE full (1960-1990)':<45} {get_r2(results_wcde_full, 'WCDE_full→le_t25'):>10} "
      f"{get_r2(results_wcde_full, 'WCDE_full→tfr_t25'):>10}")
print(f"{'WCDE post-1970 only':<45} {get_r2(results_wcde_post70, 'WCDE_post70→le_t25'):>10} "
      f"{get_r2(results_wcde_post70, 'WCDE_post70→tfr_t25'):>10}")
print(f"{'Barro-Lee: at-least-some-sec (%)':<45} {get_r2(results_bl, 'BL_edu_sec→le_t25'):>10} "
      f"{get_r2(results_bl, 'BL_edu_sec→tfr_t25'):>10}")
print(f"{'Barro-Lee: mean years of schooling':<45} {get_r2(results_bl, 'BL_edu_mys→le_t25'):>10} "
      f"{get_r2(results_bl, 'BL_edu_mys→tfr_t25'):>10}")

print(f"\n{'Significance (p-values)':<45} {'→ LE':>10} {'→ TFR':>10}")
print("-" * 65)
for label, results, le_key, tfr_key in [
    ("WCDE full", results_wcde_full, "WCDE_full→le_t25", "WCDE_full→tfr_t25"),
    ("WCDE post-1970", results_wcde_post70, "WCDE_post70→le_t25", "WCDE_post70→tfr_t25"),
    ("BL: sec %", results_bl, "BL_edu_sec→le_t25", "BL_edu_sec→tfr_t25"),
    ("BL: mean yrs", results_bl, "BL_edu_mys→le_t25", "BL_edu_mys→tfr_t25"),
]:
    le_p = get_p(results, le_key)
    tfr_p = get_p(results, tfr_key)
    le_str = f"{le_p:.4f}" if le_p < 999 else "n/a"
    tfr_str = f"{tfr_p:.4f}" if tfr_p < 999 else "n/a"
    print(f"{label:<45} {le_str:>10} {tfr_str:>10}")


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: FWL RESIDUALIZATION ON BARRO-LEE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: FWL RESIDUALIZATION (GDP stripped of education → outcomes)")
print("=" * 70)
print("Frisch-Waugh-Lovell: regress log GDP on education (country FE),")
print("take residuals, regress outcomes on residuals. If education is the")
print("active ingredient, residualized GDP should have near-zero R².\n")


def fe_residualize(panel, edu_col, outcome_col, country_col="country"):
    """FWL residualization with country FE.

    1. Demean edu, log_gdp, outcome by country
    2. Regress demeaned log_gdp on demeaned edu → residual
    3. Regress demeaned outcome on residual → R² of GDP-net-of-education
    Also returns raw GDP R² and education R² for comparison.
    """
    cols = [edu_col, "log_gdp_t", outcome_col]
    sub = panel.dropna(subset=cols + [country_col]).copy()
    if len(sub) < 20:
        return None

    # Demean by country
    for col in cols:
        sub[col + "_dm"] = sub[col] - sub.groupby(country_col)[col].transform("mean")

    # Education → outcome (baseline)
    m_edu = sm.OLS(sub[outcome_col + "_dm"],
                   sub[edu_col + "_dm"]).fit(
        cov_type='cluster', cov_kwds={'groups': sub[country_col]})

    # Raw GDP → outcome
    m_gdp = sm.OLS(sub[outcome_col + "_dm"],
                   sub["log_gdp_t_dm"]).fit(
        cov_type='cluster', cov_kwds={'groups': sub[country_col]})

    # FWL: regress GDP on education, get residual
    m_gdp_edu = sm.OLS(sub["log_gdp_t_dm"],
                       sub[edu_col + "_dm"]).fit()
    sub["gdp_resid"] = sub["log_gdp_t_dm"] - m_gdp_edu.predict(sub[edu_col + "_dm"])

    # Residualized GDP → outcome
    m_resid = sm.OLS(sub[outcome_col + "_dm"],
                     sub["gdp_resid"]).fit(
        cov_type='cluster', cov_kwds={'groups': sub[country_col]})

    return {
        "edu_r2": float(m_edu.rsquared),
        "edu_p": float(m_edu.pvalues.iloc[0]),
        "raw_gdp_r2": float(m_gdp.rsquared),
        "raw_gdp_p": float(m_gdp.pvalues.iloc[0]),
        "resid_gdp_r2": float(m_resid.rsquared),
        "resid_gdp_p": float(m_resid.pvalues.iloc[0]),
        "edu_gdp_r2": float(m_gdp_edu.rsquared),
        "n": int(len(sub)),
        "n_countries": int(sub[country_col].nunique()),
    }


results_resid = {}

# Barro-Lee: at-least-some-secondary
for edu_var, edu_label in [("edu_sec", "BL at-least-some-sec"),
                            ("edu_mys", "BL mean years of schooling")]:
    for outcome, out_label in [("le_t25", "LE"), ("tfr_t25", "TFR")]:
        key = f"BL_{edu_var}_resid→{outcome}"
        res = fe_residualize(bl_panel, edu_var, outcome)
        results_resid[key] = res
        if res:
            print(f"  {edu_label} → {out_label}:")
            print(f"    Edu R² = {res['edu_r2']:.3f}  |  Raw GDP R² = {res['raw_gdp_r2']:.3f}"
                  f"  |  Resid GDP R² = {res['resid_gdp_r2']:.3f}")
            print(f"    Edu→GDP R² = {res['edu_gdp_r2']:.3f}  |  N = {res['n']}, countries = {res['n_countries']}")

# WCDE full panel
print()
for outcome, out_label in [("le_t25", "LE"), ("tfr_t25", "TFR")]:
    key = f"WCDE_full_resid→{outcome}"
    res = fe_residualize(wcde_panel, "edu_ls", outcome)
    results_resid[key] = res
    if res:
        print(f"  WCDE lower sec → {out_label}:")
        print(f"    Edu R² = {res['edu_r2']:.3f}  |  Raw GDP R² = {res['raw_gdp_r2']:.3f}"
              f"  |  Resid GDP R² = {res['resid_gdp_r2']:.3f}")
        print(f"    Edu→GDP R² = {res['edu_gdp_r2']:.3f}  |  N = {res['n']}, countries = {res['n_countries']}")

# Summary
print(f"\n{'RESIDUALIZATION SUMMARY':<45}")
print(f"{'Dataset / edu measure':<40} {'Resid→LE':>10} {'Resid→TFR':>10}")
print("-" * 60)
for label, le_key, tfr_key in [
    ("WCDE lower sec", "WCDE_full_resid→le_t25", "WCDE_full_resid→tfr_t25"),
    ("BL at-least-some-sec", "BL_edu_sec_resid→le_t25", "BL_edu_sec_resid→tfr_t25"),
    ("BL mean years", "BL_edu_mys_resid→le_t25", "BL_edu_mys_resid→tfr_t25"),
]:
    le_r = results_resid.get(le_key, {})
    tfr_r = results_resid.get(tfr_key, {})
    le_str = f"{le_r['resid_gdp_r2']:.3f}" if le_r and le_r.get('resid_gdp_r2') is not None else "n/a"
    tfr_str = f"{tfr_r['resid_gdp_r2']:.3f}" if tfr_r and tfr_r.get('resid_gdp_r2') is not None else "n/a"
    print(f"  {label:<38} {le_str:>10} {tfr_str:>10}")


# ── Checkin ──────────────────────────────────────────────────────────────
checkin = {"test": "Barro-Lee replication of forward-prediction and residualization"}

for label, results in [("bl", results_bl),
                        ("wcde_full", results_wcde_full),
                        ("wcde_post70", results_wcde_post70)]:
    for key, res in results.items():
        if res:
            safe_key = key.replace("→", "_to_")
            checkin[f"{label}_{safe_key}_r2"] = round(res["r2_within"], 4)
            checkin[f"{label}_{safe_key}_p"] = round(res["p"], 6)
            checkin[f"{label}_{safe_key}_beta"] = round(res["beta"], 4)
            checkin[f"{label}_{safe_key}_n"] = res["n"]

# Residualization results
for key, res in results_resid.items():
    if res:
        safe_key = key.replace("→", "_to_")
        checkin[f"resid_{safe_key}_edu_r2"] = round(res["edu_r2"], 4)
        checkin[f"resid_{safe_key}_raw_gdp_r2"] = round(res["raw_gdp_r2"], 4)
        checkin[f"resid_{safe_key}_resid_gdp_r2"] = round(res["resid_gdp_r2"], 4)
        checkin[f"resid_{safe_key}_resid_gdp_p"] = round(res["resid_gdp_p"], 6)
        checkin[f"resid_{safe_key}_n"] = res["n"]

write_checkin("barro_lee_replication.json", checkin,
              script_path="scripts/robustness/barro_lee_replication.py")
