"""
robustness/ajr_n61_deep_dive.py
================================
Deep-dive analysis of the AJR (2001) base sample (~64 former colonies)
through the paper's framework: education -> demographic convergence
(TFR<3.65 ∧ LE>69.8) via 25-year lag, home-niche mechanism,
Protestant-vs-Catholic schooling tradition.

Outputs:
  data/ajr2001/ajr_n61_country_table.csv  -- one row per country
  checkin/ajr_n61_deep_dive.json          -- statistical findings

Hypothesis battery:
  T1.1 Horse race: outcome ~ z(edu_1950) + z(avexpr) + z(logem4)
  T1.2 Per-country β_g pooled by edu_1950 baseline bucket
  T1.3 Crossing-year prediction from edu_1950 reaching empirical floor
  T1.4 Forward mediation: avexpr -> edu_1950 -> outcome
  T1.5 Correlation matrix on key colonial-era and modern variables
  T3.1 Repeat horse race on AJR's own outcome (loggdppc 1995)
  T3.2 Reverse mediation: edu_1900_cohort -> avexpr -> outcome
  T3.3 avexpr as IV for edu_1950
  M1   25-year first-difference: Δedu_1925→1950 → Δoutcome_1950→1975
  M2   Female-edu differential: edu_female_1950 vs edu_both_1950
  M5   Hollow-edu check: Cuba/Vietnam years-of-schooling vs completion
  M7   Settler-mortality vs disease ecology (latitude proxy)
  M10  avexpr temporal mismatch (measured 1985-95, post-independence)

Pre-registration: T1.1-T1.5 confirmatory (Holm-Bonferroni).
T3 and M-set exploratory (no correction).
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _shared import (PROC, DATA, REGIONS, load_wb, NAME_MAP, write_checkin,
                     TFR_THRESHOLD, LE_THRESHOLD)


# ── Load colony classifications ───────────────────────────────────────
COLONIES_CODE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "colonial_vs_institutions.py")).read()
start = COLONIES_CODE.index("COLONIES = {")
depth = 0
for i, ch in enumerate(COLONIES_CODE[start:]):
    if ch == '{':
        depth += 1
    elif ch == '}':
        depth -= 1
        if depth == 0:
            end = start + i + 1
            break
exec(COLONIES_CODE[start:end])

COLONIES_AJR_EXT = {
    "Bahamas":          ("Britain", "protestant"),
    "Gabon":            ("France",  "catholic"),
    "Hong Kong, China": ("Britain", "protestant"),
}
COLONIES.update(COLONIES_AJR_EXT)

ISO3_TO_WCDE = {
    "AGO": "Angola", "ARG": "Argentina", "AUS": "Australia",
    "BFA": "Burkina Faso", "BGD": "Bangladesh", "BHS": "Bahamas",
    "BOL": "Bolivia (Plurinational State of)", "BRA": "Brazil",
    "CAN": "Canada", "CHL": "Chile", "CIV": "Côte d'Ivoire",
    "CMR": "Cameroon", "COG": "Congo", "COL": "Colombia",
    "CRI": "Costa Rica", "DOM": "Dominican Republic", "DZA": "Algeria",
    "ECU": "Ecuador", "EGY": "Egypt", "ETH": "Ethiopia",
    "GAB": "Gabon", "GHA": "Ghana", "GIN": "Guinea", "GMB": "Gambia",
    "GTM": "Guatemala", "GUY": "Guyana", "HKG": "Hong Kong, China",
    "HND": "Honduras", "HTI": "Haiti", "IDN": "Indonesia",
    "IND": "India", "JAM": "Jamaica", "KEN": "Kenya", "LKA": "Sri Lanka",
    "MAR": "Morocco", "MDG": "Madagascar", "MEX": "Mexico", "MLI": "Mali",
    "MLT": "Malta", "MYS": "Malaysia", "NER": "Niger", "NGA": "Nigeria",
    "NIC": "Nicaragua", "NZL": "New Zealand", "PAK": "Pakistan",
    "PAN": "Panama", "PER": "Peru", "PRY": "Paraguay", "SDN": "Sudan",
    "SEN": "Senegal", "SGP": "Singapore", "SLE": "Sierra Leone",
    "SLV": "El Salvador", "TGO": "Togo", "TTO": "Trinidad and Tobago",
    "TUN": "Tunisia", "TZA": "United Republic of Tanzania",
    "UGA": "Uganda", "URY": "Uruguay", "USA": "United States of America",
    "VEN": "Venezuela (Bolivarian Republic of)", "VNM": "Viet Nam",
    "ZAF": "South Africa", "ZAR": "Democratic Republic of the Congo",
}

POLITY_MAP = {
    "Republic of Korea": "Korea South", "Viet Nam": "Vietnam",
    "Taiwan Province of China": "Taiwan", "Iran (Islamic Republic of)": "Iran",
    "Russian Federation": "Russia", "United States of America": "United States",
    "United Republic of Tanzania": "Tanzania",
    "Democratic Republic of the Congo": "Congo Kinshasa",
    "Congo": "Congo Brazzaville",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Republic of Moldova": "Moldova", "Syrian Arab Republic": "Syria",
    "Lao People's Democratic Republic": "Laos", "Eswatini": "Swaziland",
    "Cabo Verde": "Cape Verde", "Czechia": "Czech Republic",
    "Myanmar": "Myanmar (Burma)", "Côte d'Ivoire": "Ivory Coast",
    "Hong Kong, China": "Hong Kong",
}

# Approximate latitude (degrees, signed) for M7 disease-ecology test
COUNTRY_LAT = {
    "Angola": -12.5, "Argentina": -38.4, "Australia": -25.3,
    "Burkina Faso": 12.2, "Bangladesh": 23.7, "Bahamas": 25.0,
    "Bolivia (Plurinational State of)": -16.3, "Brazil": -10.0,
    "Canada": 56.1, "Chile": -35.7, "Côte d'Ivoire": 7.5,
    "Cameroon": 7.4, "Congo": -0.2, "Colombia": 4.6,
    "Costa Rica": 9.9, "Dominican Republic": 18.7, "Algeria": 28.0,
    "Ecuador": -1.8, "Egypt": 26.8, "Gabon": -0.8, "Ghana": 7.9,
    "Guinea": 9.9, "Gambia": 13.4, "Guatemala": 15.8, "Guyana": 4.9,
    "Hong Kong, China": 22.3, "Honduras": 15.2, "Haiti": 18.9,
    "Indonesia": -0.8, "India": 20.6, "Jamaica": 18.1, "Kenya": -0.0,
    "Sri Lanka": 7.9, "Morocco": 31.8, "Madagascar": -18.8,
    "Mexico": 23.6, "Mali": 17.6, "Malta": 35.9, "Malaysia": 4.2,
    "Niger": 17.6, "Nigeria": 9.1, "Nicaragua": 12.9,
    "New Zealand": -40.9, "Pakistan": 30.4, "Panama": 8.5, "Peru": -9.2,
    "Paraguay": -23.4, "Sudan": 12.9, "Senegal": 14.5, "Singapore": 1.4,
    "Sierra Leone": 8.5, "El Salvador": 13.8, "Togo": 8.6,
    "Trinidad and Tobago": 10.7, "Tunisia": 33.9,
    "United Republic of Tanzania": -6.4, "Uganda": 1.4, "Uruguay": -32.5,
    "United States of America": 37.1,
    "Venezuela (Bolivarian Republic of)": 6.4, "Viet Nam": 14.1,
    "South Africa": -30.6, "Democratic Republic of the Congo": -4.0,
}

# Approximate independence year for M10 temporal-mismatch context
INDEPENDENCE_YEAR = {
    "Argentina": 1816, "Mexico": 1821, "Chile": 1818, "Peru": 1821,
    "Colombia": 1819, "Venezuela (Bolivarian Republic of)": 1821,
    "Ecuador": 1822, "Bolivia (Plurinational State of)": 1825,
    "Uruguay": 1828, "Paraguay": 1811, "Guatemala": 1821,
    "Honduras": 1821, "Nicaragua": 1821, "Costa Rica": 1821,
    "El Salvador": 1821, "Panama": 1903, "Cuba": 1902, "Brazil": 1822,
    "Dominican Republic": 1844, "Haiti": 1804,
    "United States of America": 1776, "Canada": 1867, "Australia": 1901,
    "New Zealand": 1907, "Egypt": 1922, "Iraq": 1932,
    "India": 1947, "Pakistan": 1947, "Bangladesh": 1971,
    "Sri Lanka": 1948, "Myanmar": 1948, "Indonesia": 1945,
    "Philippines": 1946, "Viet Nam": 1945, "Cambodia": 1953,
    "Laos": 1953, "Malaysia": 1957, "Singapore": 1965, "Brunei": 1984,
    "Hong Kong, China": 1997, "Bahamas": 1973, "Jamaica": 1962,
    "Trinidad and Tobago": 1962, "Guyana": 1966, "Belize": 1981,
    "Barbados": 1966, "Cyprus": 1960, "Malta": 1964,
    "South Africa": 1910, "Zimbabwe": 1980, "Namibia": 1990,
    "Ghana": 1957, "Nigeria": 1960, "Sierra Leone": 1961,
    "Gambia": 1965, "Kenya": 1963, "Uganda": 1962,
    "United Republic of Tanzania": 1961, "Zambia": 1964, "Malawi": 1964,
    "Botswana": 1966, "Lesotho": 1966, "Eswatini": 1968,
    "Sudan": 1956, "Algeria": 1962, "Tunisia": 1956, "Morocco": 1956,
    "Senegal": 1960, "Mali": 1960, "Niger": 1960, "Burkina Faso": 1960,
    "Côte d'Ivoire": 1960, "Guinea": 1958, "Togo": 1960,
    "Benin": 1960, "Cameroon": 1960, "Madagascar": 1960,
    "Gabon": 1960, "Congo": 1960,
    "Democratic Republic of the Congo": 1960, "Rwanda": 1962,
    "Burundi": 1962, "Angola": 1975, "Mozambique": 1975,
    "Guinea-Bissau": 1974, "Cabo Verde": 1975,
    "Ethiopia": 1941,  # Italian occupation ended
}


# ── Load AJR base sample ──────────────────────────────────────────────

ajr_full = pd.read_stata(os.path.join(DATA, "ajr2001", "maketable1.dta"))
ajr = ajr_full[ajr_full["baseco"] == 1][
    ["shortnam", "avexpr", "logem4", "logpgp95", "euro1900"]].copy()
ajr["country"] = ajr["shortnam"].map(ISO3_TO_WCDE)
unmapped = ajr[ajr["country"].isna()]["shortnam"].tolist()
if unmapped:
    raise SystemExit(f"Unmapped ISO3 codes: {unmapped}")
ajr = ajr.set_index("country")


# ── Load WCDE education panels ────────────────────────────────────────

def load_wcde_panel(filename):
    df = pd.read_csv(os.path.join(PROC, filename))
    df = df[~df["country"].isin(REGIONS)].copy()
    return df.set_index("country")


lsec_both = load_wcde_panel("lower_sec_both.csv")
lsec_female = load_wcde_panel("lower_sec_female.csv")
primary_both = load_wcde_panel("primary_both.csv")

# Cohort data for tracing colonial-era schooling
cohort = pd.read_csv(os.path.join(PROC, "cohort_completion_both_long.csv"))
cohort = cohort[~cohort["country"].isin(REGIONS)]


def cohort_lsec(country, cohort_year):
    """Lower-sec completion of a given cohort, as observed in 1950."""
    sub = cohort[(cohort["country"] == country)
                 & (cohort["cohort_year"] == cohort_year)]
    if len(sub) == 0:
        return np.nan
    return float(sub.iloc[0]["lower_sec"])


def cohort_primary(country, cohort_year):
    sub = cohort[(cohort["country"] == country)
                 & (cohort["cohort_year"] == cohort_year)]
    if len(sub) == 0:
        return np.nan
    return float(sub.iloc[0]["primary"])


# ── Load WB outcome series ────────────────────────────────────────────

gdp = load_wb("gdppercapita_us_inflation_adjusted.csv")
le = load_wb("life_expectancy_years.csv")
tfr = load_wb("children_per_woman_total_fertility.csv")
u5mr = load_wb("child_mortality_u5.csv")


def get_wb(df, country, year):
    key = NAME_MAP.get(country, country).lower()
    for k in [country.lower(), key]:
        if k in df.index:
            try:
                v = float(df.loc[k, str(year)])
                return v if not np.isnan(v) else np.nan
            except (KeyError, ValueError):
                pass
    return np.nan


# ── Polity5 ──────────────────────────────────────────────────────────

polity_df = pd.read_excel(os.path.join(DATA, "p5v2018.xls"))
polity_2015 = polity_df[polity_df.year == 2015][["country", "polity2"]].copy()
polity_2015 = polity_2015.set_index("country")


def get_polity(country, year=2015):
    if year == 2015:
        pname = POLITY_MAP.get(country, country)
        if pname in polity_2015.index:
            v = polity_2015.loc[pname, "polity2"]
            return float(v) if not np.isnan(v) else np.nan
    return np.nan


# ── Build country-level table ────────────────────────────────────────

def edu_at(panel, country, year):
    if country not in panel.index:
        return np.nan
    val = panel.loc[country, str(year)]
    try:
        v = float(val)
        return v if not np.isnan(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def first_crossing_year(country):
    """Year country first crosses TFR<3.65 ∧ LE>69.8 (annual WB data)."""
    for yr in range(1960, 2023):
        t = get_wb(tfr, country, yr)
        l = get_wb(le, country, yr)
        if not np.isnan(t) and not np.isnan(l):
            if t < TFR_THRESHOLD and l > LE_THRESHOLD:
                return yr
    return np.nan


print("=" * 78)
print("AJR n=61 DEEP DIVE: CONSTRUCTING COUNTRY TABLE")
print("=" * 78)
print(f"\nAJR base sample: {len(ajr)} colonies")

rows = []
drop_log = []
for country in ajr.index:
    in_colonies = country in COLONIES
    colonizer, religion = (COLONIES.get(country, (None, "none")))
    never_colonized = colonizer is None
    if never_colonized:
        drop_log.append((country, "never colonized in COLONIES dict"))
        continue
    edu_1950_v = edu_at(lsec_both, country, 1950)
    if np.isnan(edu_1950_v):
        drop_log.append((country, "no edu_1950"))
        continue
    rows.append({
        "country": country,
        "shortnam": ajr.loc[country, "shortnam"],
        "colonizer": colonizer,
        "religion": religion,
        "protestant": 1 if religion == "protestant" else 0,
        "independence_year": INDEPENDENCE_YEAR.get(country, np.nan),
        "latitude": COUNTRY_LAT.get(country, np.nan),
        "avexpr": float(ajr.loc[country, "avexpr"]),
        "logem4": float(ajr.loc[country, "logem4"]),
        "logpgp95_ajr": float(ajr.loc[country, "logpgp95"]),
        "euro1900": float(ajr.loc[country, "euro1900"]),
        "polity2_2015": get_polity(country),
        "edu_1900_cohort_lsec": cohort_lsec(country, 1900),
        "edu_1925_cohort_lsec": cohort_lsec(country, 1925),
        "edu_1900_cohort_primary": cohort_primary(country, 1900),
        "edu_1925_cohort_primary": cohort_primary(country, 1925),
        "edu_1950": edu_1950_v,
        "edu_1975": edu_at(lsec_both, country, 1975),
        "edu_2000": edu_at(lsec_both, country, 2000),
        "edu_2020": edu_at(lsec_both, country, 2020),
        "edu_female_1950": edu_at(lsec_female, country, 1950),
        "edu_female_1975": edu_at(lsec_female, country, 1975),
        "primary_1950": edu_at(primary_both, country, 1950),
        "log_gdp_1960": (np.log(get_wb(gdp, country, 1960))
                          if not np.isnan(get_wb(gdp, country, 1960))
                          and get_wb(gdp, country, 1960) > 0 else np.nan),
        "log_gdp_2020": (np.log(get_wb(gdp, country, 2020))
                          if not np.isnan(get_wb(gdp, country, 2020))
                          and get_wb(gdp, country, 2020) > 0 else np.nan),
        "le_1960": get_wb(le, country, 1960),
        "le_1975": get_wb(le, country, 1975),
        "le_2020": get_wb(le, country, 2020),
        "tfr_1960": get_wb(tfr, country, 1960),
        "tfr_1975": get_wb(tfr, country, 1975),
        "tfr_2020": get_wb(tfr, country, 2020),
        "u5mr_1960": get_wb(u5mr, country, 1960),
        "u5mr_2020": get_wb(u5mr, country, 2020),
        "crossing_year": first_crossing_year(country),
    })

df = pd.DataFrame(rows).sort_values("country").reset_index(drop=True)
# edu (20-24 lower-sec, both sexes) at the crossing year, interpolated.
def _edu_at_year_interp_pre(country, year):
    if np.isnan(year):
        return np.nan
    yr_lo = int(year // 5) * 5
    yr_hi = yr_lo + 5
    v_lo = edu_at(lsec_both, country, yr_lo)
    v_hi = edu_at(lsec_both, country, yr_hi)
    if np.isnan(v_lo) and np.isnan(v_hi):
        return np.nan
    if np.isnan(v_lo):
        return v_hi
    if np.isnan(v_hi):
        return v_lo
    w = (year - yr_lo) / 5.0
    return v_lo * (1 - w) + v_hi * w
df["edu_at_cross"] = df.apply(
    lambda r: _edu_at_year_interp_pre(r["country"], r["crossing_year"]), axis=1)
print(f"After merge: n = {len(df)}")
print(f"Drops: {len(drop_log)}")
for c, why in drop_log:
    print(f"  - {c}: {why}")
print(f"\n  Protestant: {(df.protestant == 1).sum()}")
print(f"  Catholic:   {(df.protestant == 0).sum()}")
print(f"  Crossed by 2022: {df['crossing_year'].notna().sum()}")
print(f"    of which Protestant: {((df.protestant == 1) & df['crossing_year'].notna()).sum()}")
print(f"    of which Catholic:   {((df.protestant == 0) & df['crossing_year'].notna()).sum()}")

# Save the table immediately so downstream tasks can read it.
csv_path = os.path.join(DATA, "ajr2001", "ajr_n61_country_table.csv")
df.to_csv(csv_path, index=False)
print(f"\nCountry table written: {csv_path}")


# ── T1.5 Correlation Matrix ──────────────────────────────────────────

print(f"\n{'═' * 78}")
print("T1.5 CORRELATION MATRIX")
print(f"{'═' * 78}")
corr_cols = ["edu_1900_cohort_lsec", "edu_1925_cohort_lsec", "edu_1950",
             "logem4", "avexpr", "polity2_2015", "protestant",
             "log_gdp_2020", "le_2020", "tfr_2020"]
corr_mat = df[corr_cols].corr()
print(corr_mat.round(2).to_string())

# Key correlations the paper predicts:
key_corrs = {
    "corr_edu1900_avexpr": float(corr_mat.loc["edu_1900_cohort_lsec", "avexpr"]),
    "corr_edu1900_logem4": float(corr_mat.loc["edu_1900_cohort_lsec", "logem4"]),
    "corr_edu1900_polity2": float(corr_mat.loc["edu_1900_cohort_lsec", "polity2_2015"]),
    "corr_edu1950_avexpr": float(corr_mat.loc["edu_1950", "avexpr"]),
    "corr_edu1950_loggdp2020": float(corr_mat.loc["edu_1950", "log_gdp_2020"]),
    "corr_avexpr_loggdp2020": float(corr_mat.loc["avexpr", "log_gdp_2020"]),
    "corr_protestant_edu1950": float(corr_mat.loc["protestant", "edu_1950"]),
}
print("\nKey correlations (paper predictions):")
for k, v in key_corrs.items():
    print(f"  {k:<35}  {v:+.3f}")


# ── T1.1 + T3.1 Horse Race ───────────────────────────────────────────

print(f"\n{'═' * 78}")
print("T1.1 HORSE RACE: outcome ~ z(edu_1950) + z(avexpr) + z(logem4)")
print(f"  + T3.1: same model with AJR's loggdppc_1995")
print(f"{'═' * 78}")


def zscore(s):
    s = pd.Series(s)
    return (s - s.mean()) / s.std(ddof=0)


def vif(X):
    """Variance inflation factor for each column of design matrix (incl. const)."""
    out = {}
    cols = list(X.columns)
    for c in cols:
        if c == "const":
            continue
        y = X[c].values
        Xo = X.drop(columns=[c]).values
        try:
            r2 = sm.OLS(y, Xo).fit().rsquared
            out[c] = 1.0 / (1.0 - r2) if r2 < 0.9999 else float("inf")
        except Exception:
            out[c] = float("nan")
    return out


horse_race = {}
for outcome in ["log_gdp_2020", "le_2020", "tfr_2020", "logpgp95_ajr"]:
    sub = df.dropna(subset=[outcome, "edu_1950", "avexpr", "logem4"]).copy()
    n = len(sub)
    X = pd.DataFrame({
        "const": 1.0,
        "z_edu_1950": zscore(sub["edu_1950"]),
        "z_avexpr": zscore(sub["avexpr"]),
        "z_logem4": zscore(sub["logem4"]),
    })
    Y = sub[outcome].values
    fit = sm.OLS(Y, X).fit()
    vifs = vif(X)
    print(f"\n  {outcome}  (n = {n})")
    print(f"  {'─' * 60}")
    for var in ["z_edu_1950", "z_avexpr", "z_logem4"]:
        coef = fit.params[var]; se = fit.bse[var]; t = fit.tvalues[var]; p = fit.pvalues[var]
        print(f"  {var:<14}  coef={coef:+.4f}  se={se:.4f}  t={t:+.2f}  p={p:.4f}  VIF={vifs[var]:.1f}")
    print(f"  R²={fit.rsquared:.3f}")
    horse_race[outcome] = {
        "n": int(n),
        "edu_coef": round(float(fit.params["z_edu_1950"]), 4),
        "edu_t": round(float(fit.tvalues["z_edu_1950"]), 2),
        "edu_p": round(float(fit.pvalues["z_edu_1950"]), 4),
        "avexpr_coef": round(float(fit.params["z_avexpr"]), 4),
        "avexpr_t": round(float(fit.tvalues["z_avexpr"]), 2),
        "avexpr_p": round(float(fit.pvalues["z_avexpr"]), 4),
        "logem4_coef": round(float(fit.params["z_logem4"]), 4),
        "logem4_t": round(float(fit.tvalues["z_logem4"]), 2),
        "logem4_p": round(float(fit.pvalues["z_logem4"]), 4),
        "r2": round(float(fit.rsquared), 3),
        "vif_edu": round(vifs["z_edu_1950"], 2),
        "vif_avexpr": round(vifs["z_avexpr"], 2),
        "vif_logem4": round(vifs["z_logem4"], 2),
    }

# Holm-Bonferroni on the 4 outcomes for the edu_1950 coefficient (Tier 1)
edu_pvals = sorted([(o, horse_race[o]["edu_p"]) for o in horse_race], key=lambda x: x[1])
holm = {}
m = len(edu_pvals)
for rank, (o, p) in enumerate(edu_pvals):
    adj = p * (m - rank)
    holm[o] = round(min(adj, 1.0), 4)
print(f"\n  Holm-Bonferroni adjusted p (edu_1950 coef across 4 outcomes):")
for o, p in holm.items():
    print(f"    {o:<18}  p_adj = {p}")


# ── T1.4 + T3.2 Mediation ────────────────────────────────────────────

print(f"\n{'═' * 78}")
print("T1.4 MEDIATION: avexpr -> edu_1950 -> outcome  (and reverse for T3.2)")
print(f"{'═' * 78}")


def baron_kenny(d, treat, mediator, outcome, n_boot=2000, seed=42):
    """Baron-Kenny mediation with Sobel z and bootstrap CI for indirect effect."""
    sub = d.dropna(subset=[treat, mediator, outcome]).copy()
    n = len(sub)
    Y = sub[outcome].values
    M = sub[mediator].values
    T = sub[treat].values
    # Path a: M ~ T
    a_fit = sm.OLS(M, sm.add_constant(T)).fit()
    a = a_fit.params[1]; sa = a_fit.bse[1]
    # Path b: Y ~ M + T  (b is on M, c' is on T)
    b_fit = sm.OLS(Y, sm.add_constant(np.column_stack([M, T]))).fit()
    b = b_fit.params[1]; sb = b_fit.bse[1]
    cprime = b_fit.params[2]
    # Total effect c: Y ~ T
    c_fit = sm.OLS(Y, sm.add_constant(T)).fit()
    c = c_fit.params[1]
    indirect = a * b
    # Sobel z
    sobel_se = np.sqrt(b**2 * sa**2 + a**2 * sb**2)
    sobel_z = indirect / sobel_se if sobel_se > 0 else np.nan
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z))) if not np.isnan(sobel_z) else np.nan
    # Bootstrap CI on indirect
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    boot = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        try:
            af = sm.OLS(M[s], sm.add_constant(T[s])).fit().params[1]
            bf = sm.OLS(Y[s], sm.add_constant(np.column_stack([M[s], T[s]]))).fit().params[1]
            boot.append(af * bf)
        except Exception:
            continue
    if boot:
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    else:
        ci_lo = ci_hi = np.nan
    prop_mediated = (indirect / c) if c != 0 else np.nan
    return {
        "n": int(n),
        "total_c": round(float(c), 4),
        "direct_cprime": round(float(cprime), 4),
        "indirect_ab": round(float(indirect), 4),
        "indirect_ci_lo": round(float(ci_lo), 4),
        "indirect_ci_hi": round(float(ci_hi), 4),
        "sobel_z": round(float(sobel_z), 2),
        "sobel_p": round(float(sobel_p), 4),
        "prop_mediated": round(float(prop_mediated), 3),
    }


mediation = {}
for outcome in ["log_gdp_2020", "le_2020", "tfr_2020"]:
    fwd = baron_kenny(df, "avexpr", "edu_1950", outcome)
    rev = baron_kenny(df, "edu_1900_cohort_lsec", "avexpr", outcome)
    print(f"\n  {outcome}")
    print(f"    Forward (avexpr -> edu_1950 -> outcome):")
    print(f"      total c = {fwd['total_c']:+.4f}, indirect ab = {fwd['indirect_ab']:+.4f}")
    print(f"      Sobel z = {fwd['sobel_z']:+.2f}, p = {fwd['sobel_p']:.4f}")
    print(f"      bootstrap 95% CI: [{fwd['indirect_ci_lo']:+.4f}, {fwd['indirect_ci_hi']:+.4f}]")
    print(f"      proportion mediated: {fwd['prop_mediated']:.2f}")
    print(f"    Reverse (edu_1900 -> avexpr -> outcome):")
    print(f"      total c = {rev['total_c']:+.4f}, indirect ab = {rev['indirect_ab']:+.4f}")
    print(f"      Sobel z = {rev['sobel_z']:+.2f}, p = {rev['sobel_p']:.4f}")
    print(f"      bootstrap 95% CI: [{rev['indirect_ci_lo']:+.4f}, {rev['indirect_ci_hi']:+.4f}]")
    print(f"      proportion mediated: {rev['prop_mediated']:.2f}")
    mediation[outcome] = {"forward": fwd, "reverse": rev}


# ── T1.3 + M1 Crossing-Year Prediction & First-Difference ────────────

print(f"\n{'═' * 78}")
print("T1.3 CROSSING-YEAR PREDICTION + M1 FIRST-DIFFERENCE")
print(f"{'═' * 78}")

# Empirical pattern at LE crossing across the global panel (paper §literate-ct):
#   median lower-sec completion at LE crossing = 65 (CCT-LE-lsec-median)
#   p10 lower-sec at LE crossing               = 42 (CCT-LE-lsec-p10)
# The floor is CONTEMPORANEOUS with crossing, not 25 years prior — so we test
# whether edu-at-crossing for AJR-sample crossers matches the global pattern.

GLOBAL_LSEC_AT_LE_CROSS_MEDIAN = 65.0
GLOBAL_LSEC_AT_LE_CROSS_P10 = 42.0


def edu_at_year_interp(country, panel, year):
    """Linearly interpolate 20-24 lower-sec completion for the year between 5-year points."""
    if np.isnan(year):
        return np.nan
    yr_lo = int(year // 5) * 5
    yr_hi = yr_lo + 5
    v_lo = edu_at(panel, country, yr_lo)
    v_hi = edu_at(panel, country, yr_hi)
    if np.isnan(v_lo) and np.isnan(v_hi):
        return np.nan
    if np.isnan(v_lo):
        return v_hi
    if np.isnan(v_hi):
        return v_lo
    w = (year - yr_lo) / 5.0
    return v_lo * (1 - w) + v_hi * w


crossing_pred = []
for _, row in df.iterrows():
    country = row["country"]
    actual = row["crossing_year"]
    edu_at_cross = edu_at_year_interp(country, lsec_both, actual) if not np.isnan(actual) else np.nan
    crossing_pred.append({
        "country": country,
        "actual_cross": actual,
        "edu_at_cross": edu_at_cross,
    })

cp_df = pd.DataFrame(crossing_pred)
crossers = cp_df.dropna(subset=["edu_at_cross", "actual_cross"])
print(f"\n  Crossers with edu-at-cross measurable: n = {len(crossers)}")
if len(crossers) > 0:
    eac = crossers["edu_at_cross"]
    print(f"  Lower-sec completion AT crossing (20-24 cohort, interp.):")
    print(f"    median = {eac.median():.1f}%   (global panel median: {GLOBAL_LSEC_AT_LE_CROSS_MEDIAN}%)")
    print(f"    mean   = {eac.mean():.1f}%")
    print(f"    p10    = {eac.quantile(0.1):.1f}%   (global panel p10: {GLOBAL_LSEC_AT_LE_CROSS_P10}%)")
    print(f"    p25    = {eac.quantile(0.25):.1f}%")
    print(f"    p75    = {eac.quantile(0.75):.1f}%")
    print(f"  AJR-sample crossers below global p10 floor (42%): "
          f"{(eac < GLOBAL_LSEC_AT_LE_CROSS_P10).sum()} of {len(crossers)}")
    print(f"  AJR-sample crossers below global median floor (65%): "
          f"{(eac < GLOBAL_LSEC_AT_LE_CROSS_MEDIAN).sum()} of {len(crossers)}")
    cross_pred_summary = {
        "n_crossers_with_edu": int(len(crossers)),
        "edu_at_cross_median_pct": round(float(eac.median()), 1),
        "edu_at_cross_mean_pct": round(float(eac.mean()), 1),
        "edu_at_cross_p10_pct": round(float(eac.quantile(0.1)), 1),
        "edu_at_cross_p25_pct": round(float(eac.quantile(0.25)), 1),
        "n_below_global_p10": int((eac < GLOBAL_LSEC_AT_LE_CROSS_P10).sum()),
        "n_below_global_median": int((eac < GLOBAL_LSEC_AT_LE_CROSS_MEDIAN).sum()),
        "global_panel_median_lsec_at_le_cross": GLOBAL_LSEC_AT_LE_CROSS_MEDIAN,
        "global_panel_p10_lsec_at_le_cross": GLOBAL_LSEC_AT_LE_CROSS_P10,
    }
else:
    cross_pred_summary = {"n_crossers_with_edu": 0}

# M1 first-difference
print(f"\n  M1 first-difference: Δedu_1925→1950 → Δoutcome_1950→1975")
fd_results = {}
for outcome_col, base_yr in [("le_1975", 1960), ("tfr_1975", 1960)]:
    base_outcome = outcome_col.replace("1975", str(base_yr))
    sub = df.dropna(subset=["edu_1925_cohort_lsec", "edu_1950",
                              outcome_col, base_outcome]).copy()
    if len(sub) < 10:
        print(f"    {outcome_col}: insufficient data (n={len(sub)})")
        continue
    sub["d_edu"] = sub["edu_1950"] - sub["edu_1925_cohort_lsec"]
    sub["d_out"] = sub[outcome_col] - sub[base_outcome]
    X = sm.add_constant(sub["d_edu"].values)
    fit = sm.OLS(sub["d_out"].values, X).fit()
    print(f"    Δ{outcome_col[:-5]}: n={len(sub)}, "
          f"β(Δedu)={fit.params[1]:+.4f}, t={fit.tvalues[1]:+.2f}, "
          f"p={fit.pvalues[1]:.4f}, R²={fit.rsquared:.3f}")
    fd_results[outcome_col] = {
        "n": int(len(sub)), "beta": round(float(fit.params[1]), 4),
        "t": round(float(fit.tvalues[1]), 2),
        "p": round(float(fit.pvalues[1]), 4),
        "r2": round(float(fit.rsquared), 3),
    }


# ── T1.2 β_g pooled by edu_1950 baseline bucket ──────────────────────

print(f"\n{'═' * 78}")
print("T1.2 β_g POOLED BY edu_1950 BASELINE BUCKET")
print(f"  child_edu(T+25) ~ parent_edu(T)  per bucket")
print(f"{'═' * 78}")

# For each country, parent_edu = edu_1950, child_edu = edu_1975 (25 years later)
# Pool by edu_1950 quartiles.
beta_g = df.dropna(subset=["edu_1950", "edu_1975"]).copy()
beta_g["bucket"] = pd.qcut(beta_g["edu_1950"], q=4,
                            labels=["Q1_low", "Q2", "Q3", "Q4_high"],
                            duplicates="drop")
bg_results = {}
print(f"\n  {'Bucket':<10} {'n':>3} {'edu_1950 mean':>14} {'β_g (slope)':>12} {'t':>6} {'R²':>6}")
print(f"  {'-' * 58}")
for bucket in beta_g["bucket"].cat.categories:
    sub = beta_g[beta_g["bucket"] == bucket]
    if len(sub) < 4:
        continue
    X = sm.add_constant(sub["edu_1950"].values)
    fit = sm.OLS(sub["edu_1975"].values, X).fit()
    beta = fit.params[1]
    t = fit.tvalues[1]
    print(f"  {bucket:<10} {len(sub):>3} {sub['edu_1950'].mean():>14.1f} "
          f"{beta:>+12.3f} {t:>+6.2f} {fit.rsquared:>6.3f}")
    bg_results[str(bucket)] = {
        "n": int(len(sub)),
        "edu_1950_mean": round(float(sub['edu_1950'].mean()), 1),
        "beta_g": round(float(beta), 3),
        "t": round(float(t), 2),
        "r2": round(float(fit.rsquared), 3),
    }


# ── M2, M5, M7, M10 ──────────────────────────────────────────────────

print(f"\n{'═' * 78}")
print("M-set NOVEL ANGLES")
print(f"{'═' * 78}")

# M2 Female differential
print(f"\n  M2: Female edu_1950 vs both-sexes edu_1950 as predictor of LE_2020")
m2_results = {}
for predictor in ["edu_1950", "edu_female_1950"]:
    sub = df.dropna(subset=[predictor, "le_2020"])
    X = sm.add_constant(sub[predictor].values)
    fit = sm.OLS(sub["le_2020"].values, X).fit()
    print(f"    {predictor:<20}  n={len(sub)}  "
          f"β={fit.params[1]:+.4f}  t={fit.tvalues[1]:+.2f}  R²={fit.rsquared:.3f}")
    m2_results[predictor] = {
        "n": int(len(sub)), "beta": round(float(fit.params[1]), 4),
        "t": round(float(fit.tvalues[1]), 2), "r2": round(float(fit.rsquared), 3),
    }

# M5 Hollow-edu check on Cuba and Vietnam (both in AJR base)
print(f"\n  M5: Hollow-education check — within-AJR-sample candidates")
m5_results = {}
# Cuba is NOT in the AJR-64 base sample. Vietnam (VNM) is. Only Vietnam is
# the within-AJR-sample socialist-credentialing test.
for c in ["Viet Nam"]:
    if c not in df["country"].values:
        print(f"    {c}: not in n=61 sample")
        m5_results[c] = "not_in_sample"
        continue
    row = df[df["country"] == c].iloc[0]
    edu_2000 = row["edu_2000"]
    le_2020 = row["le_2020"]
    tfr_2020 = row["tfr_2020"]
    # Compare against global non-AJR fit (here: AJR sample fit) — flag if outlier
    fit_sub = df.dropna(subset=["edu_2000", "le_2020"])
    fit_le = sm.OLS(fit_sub["le_2020"].values,
                     sm.add_constant(fit_sub["edu_2000"].values)).fit()
    pred_le = fit_le.params[0] + fit_le.params[1] * edu_2000
    le_resid = le_2020 - pred_le
    print(f"    {c}: edu_2000={edu_2000:.1f}%, LE_2020={le_2020:.1f}, "
          f"predicted={pred_le:.1f}, residual={le_resid:+.1f}")
    m5_results[c] = {
        "edu_2000": round(float(edu_2000), 1),
        "le_2020": round(float(le_2020), 1),
        "le_predicted_from_edu": round(float(pred_le), 1),
        "le_residual": round(float(le_resid), 2),
    }

# M7 logem4 vs latitude (disease ecology proxy)
print(f"\n  M7: Settler mortality vs latitude (|lat| as tropical-disease proxy)")
sub = df.dropna(subset=["logem4", "latitude"]).copy()
sub["abs_lat"] = sub["latitude"].abs()
X = sm.add_constant(sub["abs_lat"].values)
fit = sm.OLS(sub["logem4"].values, X).fit()
m7_logem4_lat_r2 = float(fit.rsquared)
m7_corr = float(sub[["logem4", "abs_lat"]].corr().iloc[0, 1])
print(f"    n={len(sub)}, corr(logem4, |lat|) = {m7_corr:+.3f}, "
      f"R² regressing logem4 on |lat| = {m7_logem4_lat_r2:.3f}")
print(f"    Reading: high R² => settler mortality largely tracks tropical "
      f"latitude; AJR's instrument is partly 'tropics' which is also where "
      f"colonizers didn't settle and didn't build mass schools.")
m7_results = {
    "n": int(len(sub)),
    "corr_logem4_abs_lat": round(m7_corr, 3),
    "r2_logem4_on_abs_lat": round(m7_logem4_lat_r2, 3),
}

# M10 avexpr temporal mismatch
print(f"\n  M10: avexpr (1985-95) is post-independence for nearly the whole sample")
sub = df.dropna(subset=["independence_year"]).copy()
sub["years_post_indep_at_avexpr_window"] = 1990 - sub["independence_year"]
print(f"    Years between independence and avexpr midpoint (1990):")
print(f"      median = {sub['years_post_indep_at_avexpr_window'].median():.0f}")
print(f"      min = {sub['years_post_indep_at_avexpr_window'].min():.0f}, "
      f"max = {sub['years_post_indep_at_avexpr_window'].max():.0f}")
print(f"      n countries with avexpr measured >25 years post-independence: "
      f"{(sub['years_post_indep_at_avexpr_window'] > 25).sum()} of {len(sub)}")
m10_results = {
    "n_with_independence": int(len(sub)),
    "median_years_post_indep": int(sub['years_post_indep_at_avexpr_window'].median()),
    "n_avexpr_25plus_years_post_indep": int((sub['years_post_indep_at_avexpr_window'] > 25).sum()),
}


# ── T3.3 avexpr as IV for edu_1950 ──────────────────────────────────

print(f"\n{'═' * 78}")
print("T3.3 avexpr AS IV FOR edu_1950 (predicting log_gdp_2020)")
print(f"  If avexpr ALSO works as instrument for education, the channels")
print(f"  cannot be separated by IV identification.")
print(f"{'═' * 78}")

sub = df.dropna(subset=["avexpr", "edu_1950", "log_gdp_2020"]).copy()
n_iv = len(sub)
Z = sm.add_constant(sub[["avexpr"]].values)
fs = sm.OLS(sub["edu_1950"].values, Z).fit()
fs_F = float(fs.fvalue)
edu_hat = fs.fittedvalues
ss = sm.OLS(sub["log_gdp_2020"].values,
             sm.add_constant(edu_hat)).fit()
ss_coef = float(ss.params[1])
# correct se
resid = sub["log_gdp_2020"].values - (ss.params[0] + ss_coef * sub["edu_1950"].values)
sigma2 = np.sum(resid**2) / (n_iv - 2)
X2 = sm.add_constant(edu_hat)
ss_se = float(np.sqrt(sigma2 * np.linalg.inv(X2.T @ X2)[1, 1]))
ss_t = ss_coef / ss_se
print(f"  n = {n_iv}")
print(f"  First stage F (avexpr -> edu_1950):  {fs_F:.2f}  "
      f"({'STRONG' if fs_F > 10 else 'WEAK'})")
print(f"  Second stage edu coef on log GDP:  {ss_coef:+.4f}  "
      f"(se={ss_se:.4f}, t={ss_t:+.2f})")
print(f"  Reading: AJR's own variable is a {'strong' if fs_F > 10 else 'weak'} "
      f"instrument for education. The instrument cannot distinguish channels.")
t33_results = {
    "n": int(n_iv),
    "first_stage_F": round(fs_F, 2),
    "second_stage_edu_coef": round(ss_coef, 4),
    "second_stage_t": round(float(ss_t), 2),
}


# ── Save JSON checkin ────────────────────────────────────────────────

# Univariate R² for the §The Colonial Test variance paragraph (replaces
# Polity5 R² with avexpr R² on the AJR-64 sample).
print(f"\n{'═' * 78}")
print("UNIVARIATE R² ON AJR-64 (for §The Colonial Test variance paragraph)")
print(f"{'═' * 78}")

univar_r2 = {}
for p in ["edu_1950", "avexpr", "logem4", "edu_1900_cohort_lsec", "protestant"]:
    sub = df.dropna(subset=[p, "log_gdp_2020"])
    X = sm.add_constant(sub[p].values)
    fit = sm.OLS(sub["log_gdp_2020"].values, X).fit()
    r2 = round(float(fit.rsquared), 3)
    univar_r2[p] = {"r2": r2, "n": int(len(sub))}
    print(f"  {p:<28} R²={r2:.3f}  (n={len(sub)})")

# Edu + religion together (does religion add anything beyond edu?)
sub_er = df.dropna(subset=["edu_1950", "protestant", "log_gdp_2020"])
X_er = sm.add_constant(sub_er[["edu_1950", "protestant"]].values)
fit_er = sm.OLS(sub_er["log_gdp_2020"].values, X_er).fit()
univar_r2["edu_plus_religion"] = {
    "r2": round(float(fit_er.rsquared), 3), "n": int(len(sub_er))}
print(f"  edu+religion                R²={univar_r2['edu_plus_religion']['r2']:.3f}")
print(f"  (vs edu alone R²={univar_r2['edu_1950']['r2']:.3f}: religion adds ~"
      f"{univar_r2['edu_plus_religion']['r2'] - univar_r2['edu_1950']['r2']:+.3f})")

checkin = {
    "n_ajr_base": int(len(ajr)),
    "n_n61_sample": int(len(df)),
    "univar_r2_log_gdp_2020": univar_r2,
    "n_protestant": int((df.protestant == 1).sum()),
    "n_catholic": int((df.protestant == 0).sum()),
    "n_crossed_by_2022": int(df["crossing_year"].notna().sum()),
    "n_protestant_crossed": int(((df.protestant == 1) & df["crossing_year"].notna()).sum()),
    "n_catholic_crossed": int(((df.protestant == 0) & df["crossing_year"].notna()).sum()),
    "drops": [{"country": c, "reason": w} for c, w in drop_log],
    # T1.5
    "correlations": {k: round(v, 3) for k, v in key_corrs.items()},
    # T1.1 + T3.1
    "horse_race": horse_race,
    "horse_race_holm_p": holm,
    # T1.4 + T3.2
    "mediation": mediation,
    # T1.3 + M1
    "crossing_year_prediction": cross_pred_summary,
    "first_difference_M1": fd_results,
    # T1.2
    "beta_g_buckets": bg_results,
    # M-set
    "m2_female_differential": m2_results,
    "m5_hollow_edu_check": m5_results,
    "m7_settler_mortality_vs_latitude": m7_results,
    "m10_avexpr_temporal_mismatch": m10_results,
    # T3.3
    "t33_avexpr_iv_for_education": t33_results,
}

write_checkin("ajr_n61_deep_dive.json", checkin,
              script_path="scripts/robustness/ajr_n61_deep_dive.py")
print("\nDone.")
