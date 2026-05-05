"""
robustness/colonial_global.py
=================================
Comprehensive AJR vs. paper contest on the GLOBAL country panel,
not restricted to AJR's 64-colony selected sample.

Sample selection rule (one rule, applied uniformly):
  Country must be in WCDE v3 country list (excludes regional aggregates),
  must have lower-secondary completion at edu_1950,
  must have at least one of {log_gdp_2020, le_2020, tfr_2020, u5mr_2015}.
No other exclusions. Microstates, conflict states, settler colonies, never-
colonized, and post-Soviet states all stay in. Every drop is logged.

AJR's identification (settler-mortality IV) cannot be tested globally
because the instrument exists only for AJR-64. So this script tests the
broader institutions hypothesis using polity2 (Polity5) as the cross-
country institutional proxy. polity2 is the standard global institutional
measure used in AJR (2005, 2008) and most subsequent literature.

Tests:
  B1  Global horse race: outcome ~ z(edu_1950) + z(polity2) + z(log_gdp_1960)
  B2  In-AJR vs out-of-AJR vs full-panel comparison of B1
  B3  Falsifier search: authoritarian+high-edu vs democratic+low-edu cases
  B4  Crossing-year empirical floor reproduction on global sample
  B5  β_g amplification by edu_1950 quartile globally
  B6  Within-region tests across 7 World Bank regions

Outputs:
  data/colonial_global/global_country_table.csv
  checkin/colonial_global.json
  checkin/colonial_global_report.md
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
                     CHECKIN, TFR_THRESHOLD, LE_THRESHOLD)


# ── Load existing colony classifications + extend ───────────────────

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

# Extension: never-colonized, post-Soviet, Eastern European, European
# metropoles, and other countries needed for the global panel. The
# (colonizer, religion) tuple here is informational; for never-colonized,
# religion follows the dominant indigenous tradition where "tradition"
# corresponds to whether the historical schooling regime was Catholic-
# Counter-Reformation-influenced (rare here), or set independently.
COLONIES_GLOBAL_EXT = {
    # Never-colonized (or self-determined education)
    "Bahamas": ("Britain", "protestant"),
    "Gabon": ("France", "catholic"),
    "Hong Kong, China": ("Britain", "protestant"),
    # European metropoles — never colonized in the AJR sense
    "United Kingdom of Great Britain and Northern Ireland": (None, "protestant"),
    "France": (None, "catholic"),
    "Spain": (None, "catholic"),
    "Portugal": (None, "catholic"),
    "Netherlands": (None, "protestant"),
    "Belgium": (None, "catholic"),
    "Italy": (None, "catholic"),
    "Germany": (None, "protestant"),
    "Austria": (None, "catholic"),
    "Switzerland": (None, "protestant"),
    "Ireland": (None, "catholic"),
    "Sweden": (None, "protestant"),
    "Norway": (None, "protestant"),
    "Denmark": (None, "protestant"),
    "Finland": (None, "protestant"),
    "Iceland": (None, "protestant"),
    "Greece": (None, "orthodox"),
    "Luxembourg": (None, "catholic"),
    # Eastern Europe (formerly under Soviet sphere but not Soviet republics)
    "Poland": (None, "catholic"),
    "Czechia": (None, "catholic"),
    "Slovakia": (None, "catholic"),
    "Hungary": (None, "catholic"),
    "Romania": (None, "orthodox"),
    "Bulgaria": (None, "orthodox"),
    "Albania": (None, "muslim"),
    "Serbia": (None, "orthodox"),
    "Croatia": (None, "catholic"),
    "Slovenia": (None, "catholic"),
    "Bosnia and Herzegovina": (None, "muslim"),
    "North Macedonia": (None, "orthodox"),
    "Montenegro": (None, "orthodox"),
    # Former Soviet republics (treated as USSR-administered for the religion
    # variable — not a colonial inheritance the paper's framework uses)
    "Russian Federation": (None, "orthodox"),
    "Ukraine": (None, "orthodox"),
    "Belarus": (None, "orthodox"),
    "Republic of Moldova": (None, "orthodox"),
    "Estonia": (None, "protestant"),
    "Latvia": (None, "protestant"),
    "Lithuania": (None, "catholic"),
    "Armenia": (None, "orthodox"),
    "Azerbaijan": (None, "muslim"),
    "Georgia": (None, "orthodox"),
    "Kazakhstan": (None, "muslim"),
    "Kyrgyzstan": (None, "muslim"),
    "Uzbekistan": (None, "muslim"),
    "Tajikistan": (None, "muslim"),
    "Turkmenistan": (None, "muslim"),
    # East and Southeast Asia
    "China": (None, "none"),
    "Japan": (None, "none"),
    "Republic of Korea": ("Japan", "none"),
    "Democratic People's Republic of Korea": ("Japan", "none"),
    "Taiwan Province of China": ("Japan", "none"),
    "Mongolia": (None, "buddhist"),
    "Thailand": (None, "buddhist"),
    "Philippines": ("Spain", "catholic"),  # Spain then USA
    # Already in COLONIES dict via colonial_vs_institutions.py:
    # "Viet Nam", "Cambodia", "Lao People's Democratic Republic"
    # Middle East / MENA
    "Iran (Islamic Republic of)": (None, "muslim"),
    "Israel": ("Britain", "jewish"),  # British Mandate Palestine
    "Lebanon": ("France", "catholic"),
    "Saudi Arabia": (None, "muslim"),
    "Iraq": ("Britain", "muslim"),
    "Kuwait": ("Britain", "muslim"),
    "Yemen": ("Britain", "muslim"),
    "Oman": ("Britain", "muslim"),
    "Qatar": ("Britain", "muslim"),
    "United Arab Emirates": ("Britain", "muslim"),
    "Bahrain": ("Britain", "muslim"),
    "Türkiye": (None, "muslim"),
    # Africa not in COLONIES yet
    "Liberia": (None, "none"),  # Founded by freed slaves
    "Mauritius": ("Britain", "protestant"),
    "Seychelles": ("Britain", "protestant"),
    "Equatorial Guinea": ("Spain", "catholic"),
    "Sao Tome and Principe": ("Portugal", "catholic"),
    "Mozambique": ("Portugal", "catholic"),
    "Guinea-Bissau": ("Portugal", "catholic"),
    "Cabo Verde": ("Portugal", "catholic"),
    "Comoros": ("France", "muslim"),
    "Djibouti": ("France", "muslim"),
    # South Asia
    "Maldives": ("Britain", "muslim"),
    "Bhutan": (None, "buddhist"),
    "Nepal": (None, "hindu"),
    "Afghanistan": (None, "muslim"),
    # Caribbean / Pacific
    "Belize": ("Britain", "protestant"),
    "Barbados": ("Britain", "protestant"),
    "Suriname": ("Netherlands", "protestant"),
    "Fiji": ("Britain", "protestant"),
    "Papua New Guinea": ("Britain", "protestant"),
    "Solomon Islands": ("Britain", "protestant"),
    "Vanuatu": ("Britain", "protestant"),
    "Samoa": ("Britain", "protestant"),
    "Tonga": ("Britain", "protestant"),
    "Micronesia (Federated States of)": (None, "none"),
    "Kiribati": ("Britain", "protestant"),
    "Brunei Darussalam": ("Britain", "muslim"),
    "Timor-Leste": ("Portugal", "catholic"),
    # Catch a few WB-style names that may differ
    "Republic of Korea": ("Japan", "none"),
    "Lao People's Democratic Republic": ("France", "catholic"),
    "Syrian Arab Republic": ("France", "catholic"),
}
COLONIES.update(COLONIES_GLOBAL_EXT)


# ── Region map (WB-style 7 regions) ──────────────────────────────────

REGION = {}
EAP = ["Australia", "Brunei Darussalam", "Cambodia", "China", "Fiji",
       "Indonesia", "Japan", "Kiribati", "Lao People's Democratic Republic",
       "Malaysia", "Micronesia (Federated States of)", "Mongolia", "Myanmar",
       "New Zealand", "Papua New Guinea", "Philippines", "Republic of Korea",
       "Democratic People's Republic of Korea", "Samoa", "Singapore",
       "Solomon Islands", "Taiwan Province of China", "Thailand",
       "Timor-Leste", "Tonga", "Vanuatu", "Viet Nam", "Hong Kong, China"]
for c in EAP:
    REGION[c] = "East Asia & Pacific"

ECA = ["Albania", "Armenia", "Austria", "Azerbaijan", "Belarus",
       "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
       "Czechia", "Denmark", "Estonia", "Finland", "France", "Georgia",
       "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy",
       "Kazakhstan", "Kyrgyzstan", "Latvia", "Lithuania", "Luxembourg",
       "Malta", "Montenegro", "Netherlands", "North Macedonia", "Norway",
       "Poland", "Portugal", "Republic of Moldova", "Romania",
       "Russian Federation", "Serbia", "Slovakia", "Slovenia", "Spain",
       "Sweden", "Switzerland", "Tajikistan", "Türkiye", "Turkmenistan",
       "Ukraine", "United Kingdom of Great Britain and Northern Ireland",
       "Uzbekistan"]
for c in ECA:
    REGION[c] = "Europe & Central Asia"

LAC = ["Antigua and Barbuda", "Argentina", "Bahamas", "Barbados", "Belize",
       "Bolivia (Plurinational State of)", "Brazil", "Chile", "Colombia",
       "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "Ecuador",
       "El Salvador", "Grenada", "Guatemala", "Guyana", "Haiti", "Honduras",
       "Jamaica", "Mexico", "Nicaragua", "Panama", "Paraguay", "Peru",
       "Saint Kitts and Nevis", "Saint Lucia",
       "Saint Vincent and the Grenadines", "Suriname",
       "Trinidad and Tobago", "Uruguay",
       "Venezuela (Bolivarian Republic of)"]
for c in LAC:
    REGION[c] = "Latin America & Caribbean"

MENA = ["Algeria", "Bahrain", "Egypt", "Iran (Islamic Republic of)",
        "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Libya",
        "Morocco", "Oman", "Qatar", "Saudi Arabia", "Syrian Arab Republic",
        "Tunisia", "United Arab Emirates", "Yemen"]
for c in MENA:
    REGION[c] = "Middle East & North Africa"

NA = ["Canada", "United States of America"]
for c in NA:
    REGION[c] = "North America"

SA = ["Afghanistan", "Bangladesh", "Bhutan", "India", "Maldives", "Nepal",
      "Pakistan", "Sri Lanka"]
for c in SA:
    REGION[c] = "South Asia"

SSA = ["Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
       "Cabo Verde", "Cameroon", "Central African Republic", "Chad",
       "Comoros", "Congo", "Côte d'Ivoire",
       "Democratic Republic of the Congo", "Djibouti", "Equatorial Guinea",
       "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana",
       "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia",
       "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius",
       "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
       "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone",
       "Somalia", "South Africa", "South Sudan", "Sudan",
       "United Republic of Tanzania", "Togo", "Uganda", "Zambia",
       "Zimbabwe"]
for c in SSA:
    REGION[c] = "Sub-Saharan Africa"


# ── Load AJR-64 base sample for the in_ajr64 flag ────────────────────

ajr_full = pd.read_stata(os.path.join(DATA, "ajr2001", "maketable1.dta"))
ajr_base = ajr_full[ajr_full["baseco"] == 1][["shortnam", "avexpr", "logem4", "logpgp95"]]

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
AJR_64_SET = {ISO3_TO_WCDE[c] for c in ajr_base["shortnam"].dropna()
              if c in ISO3_TO_WCDE}


# ── Polity name map ─────────────────────────────────────────────────

POLITY_MAP = {
    "Republic of Korea": "Korea South",
    "Democratic People's Republic of Korea": "Korea North",
    "Viet Nam": "Vietnam",
    "Taiwan Province of China": "Taiwan",
    "Iran (Islamic Republic of)": "Iran",
    "Russian Federation": "Russia",
    "United States of America": "United States",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "United Republic of Tanzania": "Tanzania",
    "Democratic Republic of the Congo": "Congo Kinshasa",
    "Congo": "Congo Brazzaville",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Republic of Moldova": "Moldova",
    "Syrian Arab Republic": "Syria",
    "Lao People's Democratic Republic": "Laos",
    "Eswatini": "Swaziland",
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "Myanmar": "Myanmar (Burma)",
    "Côte d'Ivoire": "Ivory Coast",
    "Hong Kong, China": "Hong Kong",
    "Türkiye": "Turkey",
    "Brunei Darussalam": "Brunei",
}


# ── Load data ───────────────────────────────────────────────────────

def load_wcde_panel(filename):
    df = pd.read_csv(os.path.join(PROC, filename))
    df = df[~df["country"].isin(REGIONS)].copy()
    return df.set_index("country")


lsec_both = load_wcde_panel("lower_sec_both.csv")
lsec_female = load_wcde_panel("lower_sec_female.csv")
primary_both = load_wcde_panel("primary_both.csv")

cohort = pd.read_csv(os.path.join(PROC, "cohort_completion_both_long.csv"))
cohort = cohort[~cohort["country"].isin(REGIONS)]


def cohort_lsec(country, cohort_year):
    sub = cohort[(cohort["country"] == country) & (cohort["cohort_year"] == cohort_year)]
    if len(sub) == 0:
        return np.nan
    return float(sub.iloc[0]["lower_sec"])


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


polity_df = pd.read_excel(os.path.join(DATA, "p5v2018.xls"))


def polity_avg(country, start_yr, end_yr):
    pname = POLITY_MAP.get(country, country)
    sub = polity_df[(polity_df["country"] == pname)
                    & (polity_df["year"] >= start_yr)
                    & (polity_df["year"] <= end_yr)]["polity2"]
    sub = sub.dropna()
    return float(sub.mean()) if len(sub) > 0 else np.nan


def edu_at(panel, country, year):
    if country not in panel.index:
        return np.nan
    val = panel.loc[country, str(year)]
    try:
        v = float(val); return v if not np.isnan(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def first_crossing_year(country):
    for yr in range(1960, 2023):
        t = get_wb(tfr, country, yr); l = get_wb(le, country, yr)
        if not np.isnan(t) and not np.isnan(l):
            if t < TFR_THRESHOLD and l > LE_THRESHOLD:
                return yr
    return np.nan


# ── Build global country table ──────────────────────────────────────

print("=" * 78)
print("AJR-BUSTING GLOBAL: building country panel")
print("=" * 78)

candidate_countries = sorted(set(lsec_both.index) - set(REGIONS))
print(f"\nWCDE candidate countries (after region filter): {len(candidate_countries)}")

rows = []
drop_log = []
for country in candidate_countries:
    edu_1950 = edu_at(lsec_both, country, 1950)
    if np.isnan(edu_1950):
        drop_log.append((country, "no edu_1950"))
        continue
    le_2020 = get_wb(le, country, 2020)
    tfr_2020 = get_wb(tfr, country, 2020)
    gdp_2020 = get_wb(gdp, country, 2020)
    u5_2015 = get_wb(u5mr, country, 2015)
    if all(np.isnan(v) for v in [le_2020, tfr_2020, gdp_2020, u5_2015]):
        drop_log.append((country, "no 2020 outcome"))
        continue
    colonizer, religion = COLONIES.get(country, (None, "unknown"))
    rows.append({
        "country": country,
        "region": REGION.get(country, "Other"),
        "in_ajr64": country in AJR_64_SET,
        "colonizer": colonizer,
        "religion": religion,
        "ever_colonized": colonizer is not None,
        "protestant": 1 if religion == "protestant" else 0,
        "catholic": 1 if religion == "catholic" else 0,
        "edu_1900_cohort_lsec": cohort_lsec(country, 1900),
        "edu_1925_cohort_lsec": cohort_lsec(country, 1925),
        "edu_1950": edu_1950,
        "edu_1975": edu_at(lsec_both, country, 1975),
        "edu_2000": edu_at(lsec_both, country, 2000),
        "edu_2020": edu_at(lsec_both, country, 2020),
        "edu_female_1950": edu_at(lsec_female, country, 1950),
        "primary_1950": edu_at(primary_both, country, 1950),
        "log_gdp_1960": (np.log(get_wb(gdp, country, 1960))
                          if not np.isnan(get_wb(gdp, country, 1960))
                          and get_wb(gdp, country, 1960) > 0 else np.nan),
        "log_gdp_2020": (np.log(gdp_2020) if not np.isnan(gdp_2020)
                          and gdp_2020 > 0 else np.nan),
        "le_1960": get_wb(le, country, 1960),
        "le_2020": le_2020,
        "tfr_1960": get_wb(tfr, country, 1960),
        "tfr_2020": tfr_2020,
        "u5mr_1960": get_wb(u5mr, country, 1960),
        "u5mr_2015": u5_2015,
        "polity2_avg_1980_2010": polity_avg(country, 1980, 2010),
        "polity2_avg_1995_2010": polity_avg(country, 1995, 2010),
        "polity2_2015": polity_avg(country, 2015, 2015),
        "crossing_year": first_crossing_year(country),
    })

df = pd.DataFrame(rows).sort_values("country").reset_index(drop=True)
n_full = len(df)
print(f"Global panel: n = {n_full}")
print(f"Drops: {len(drop_log)}")
n_drop_no_edu = sum(1 for _, why in drop_log if why == "no edu_1950")
n_drop_no_outcome = sum(1 for _, why in drop_log if why == "no 2020 outcome")
print(f"  no edu_1950: {n_drop_no_edu}")
print(f"  no 2020 outcome: {n_drop_no_outcome}")
print(f"\nIn AJR-64: {df['in_ajr64'].sum()}")
print(f"Out of AJR-64: {(~df['in_ajr64']).sum()}")
print(f"Ever colonized: {df['ever_colonized'].sum()}")
print(f"Never colonized: {(~df['ever_colonized']).sum()}")
print(f"Crossed by 2022: {df['crossing_year'].notna().sum()}")

# Edu at crossing — interpolated
def edu_at_year_interp(country, year):
    if pd.isna(year):
        return np.nan
    yr_lo = int(year // 5) * 5
    yr_hi = yr_lo + 5
    v_lo = edu_at(lsec_both, country, yr_lo)
    v_hi = edu_at(lsec_both, country, yr_hi)
    if np.isnan(v_lo) and np.isnan(v_hi):
        return np.nan
    if np.isnan(v_lo): return v_hi
    if np.isnan(v_hi): return v_lo
    w = (year - yr_lo) / 5.0
    return v_lo * (1 - w) + v_hi * w

df["edu_at_cross"] = df.apply(
    lambda r: edu_at_year_interp(r["country"], r["crossing_year"]), axis=1)

# Save the table
out_dir = os.path.join(DATA, "colonial_global")
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "global_country_table.csv")
df.to_csv(csv_path, index=False)
print(f"\nGlobal country table: {csv_path}")


# ── Helpers ─────────────────────────────────────────────────────────

def zscore(s):
    s = pd.Series(s)
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - s.mean()) / sd


def horse_race(d, outcome, predictors, label="", print_out=True):
    sub = d.dropna(subset=[outcome] + predictors).copy()
    n = len(sub)
    if n < 10:
        return {"n": n, "label": label, "insufficient": True}
    X_data = {"const": np.ones(n)}
    for p in predictors:
        X_data[f"z_{p}"] = zscore(sub[p]).values
    X = pd.DataFrame(X_data, index=sub.index)
    Y = sub[outcome].values
    fit = sm.OLS(Y, X.values).fit()
    cols = list(X.columns)
    out = {"n": int(n), "label": label, "r2": round(float(fit.rsquared), 3)}
    for i, name in enumerate(cols):
        if name == "const": continue
        out[f"{name}_coef"] = round(float(fit.params[i]), 4)
        out[f"{name}_t"] = round(float(fit.tvalues[i]), 2)
        out[f"{name}_p"] = round(float(fit.pvalues[i]), 4)
    if print_out:
        print(f"\n  {label} (n={n}, R²={out['r2']})")
        for p in predictors:
            zk = f"z_{p}"
            print(f"    {p:<25} coef={out[zk+'_coef']:+.4f} "
                  f"t={out[zk+'_t']:+.2f} p={out[zk+'_p']:.4f}")
    return out


# ── Build derived variables for rate-of-expansion tests ─────────────

df["delta_edu_1950_2000"] = df["edu_2000"] - df["edu_1950"]
df["delta_edu_1950_1975"] = df["edu_1975"] - df["edu_1950"]
df["delta_edu_1925_1950"] = df["edu_1950"] - df["edu_1925_cohort_lsec"]


# ── B1 + B2: Horse race across sample tiers and proxies ─────────────

print(f"\n{'═' * 78}")
print("B1 + B2: GLOBAL HORSE RACE WITH SAMPLE-TIER COMPARISON")
print(f"{'═' * 78}")

OUTCOMES = ["log_gdp_2020", "le_2020", "tfr_2020", "u5mr_2015"]
INSTITUTION_PROXIES = ["polity2_avg_1980_2010", "polity2_avg_1995_2010",
                        "polity2_2015"]
SAMPLE_TIERS = {
    "full_panel": df,
    "ajr_64_only": df[df["in_ajr64"]],
    "non_ajr_64": df[~df["in_ajr64"]],
    "ever_colonized": df[df["ever_colonized"]],
    "never_colonized": df[~df["ever_colonized"]],
    # Entry-cohort design (paper §the-panel): track countries from a low
    # threshold forward while ceiling <= 90%. The lower bound is varied to
    # test sensitivity — different bounds isolate different segments of
    # the schooling expansion curve. n=8 countries are at >= 90% ceiling
    # in 1950 (Iceland, Denmark, Slovakia, Norway, Japan, UK, Czech, Germany).
    "entry_cohort_10_90": df[(df["edu_1950"] > 10) & (df["edu_1950"] < 90)],
    "entry_cohort_20_90": df[(df["edu_1950"] > 20) & (df["edu_1950"] < 90)],
    "entry_cohort_30_90": df[(df["edu_1950"] > 30) & (df["edu_1950"] < 90)],
    "entry_cohort_50_90": df[(df["edu_1950"] > 50) & (df["edu_1950"] < 90)],
    "entry_cohort_70_90": df[(df["edu_1950"] > 70) & (df["edu_1950"] < 90)],
    "below_90": df[df["edu_1950"] < 90],
    # AJR-64 with entry-cohort restriction — apply the same lens to AJR's
    # own sample. Tests whether the rate-of-expansion specification still
    # beats institutions on AJR's home ground.
    "ajr_64_entry_10_90": df[df["in_ajr64"] & (df["edu_1950"] > 10) & (df["edu_1950"] < 90)],
    "ajr_64_entry_20_90": df[df["in_ajr64"] & (df["edu_1950"] > 20) & (df["edu_1950"] < 90)],
    "ajr_64_below_90": df[df["in_ajr64"] & (df["edu_1950"] < 90)],
}

b1_results = {}
for tier_name, tier_df in SAMPLE_TIERS.items():
    if len(tier_df) < 10:
        continue
    print(f"\n--- TIER: {tier_name} (n={len(tier_df)}) ---")
    b1_results[tier_name] = {}
    for institution in INSTITUTION_PROXIES:
        b1_results[tier_name][institution] = {}
        for outcome in OUTCOMES:
            label = f"{outcome} ~ z(edu_1950) + z({institution}) + z(log_gdp_1960)"
            r = horse_race(tier_df, outcome,
                           ["edu_1950", institution, "log_gdp_1960"],
                           label=f"{tier_name} | {label}",
                           print_out=False)
            b1_results[tier_name][institution][outcome] = r

# Print compact summary for the most important tier+proxy combination
print(f"\n{'═' * 78}")
print("B1 SUMMARY: full_panel × polity2_avg_1980_2010 (the headline)")
print(f"{'═' * 78}")
hr = b1_results["full_panel"]["polity2_avg_1980_2010"]
print(f"\n  {'outcome':<14} {'n':>4} {'edu_t':>8} {'edu_p':>8} {'pol_t':>8} {'pol_p':>8} {'gdp60_t':>9} {'gdp60_p':>9} {'R²':>5}")
for o in OUTCOMES:
    r = hr[o]
    if r.get("insufficient"):
        print(f"  {o:<14} {r['n']:>4} (insufficient)")
        continue
    print(f"  {o:<14} {r['n']:>4} {r['z_edu_1950_t']:>+8.2f} "
          f"{r['z_edu_1950_p']:>8.4f} "
          f"{r['z_polity2_avg_1980_2010_t']:>+8.2f} "
          f"{r['z_polity2_avg_1980_2010_p']:>8.4f} "
          f"{r['z_log_gdp_1960_t']:>+9.2f} "
          f"{r['z_log_gdp_1960_p']:>9.4f} "
          f"{r['r2']:>.3f}")


# ── Cross-tier comparison table ─────────────────────────────────────

print(f"\n{'═' * 78}")
print("B2: edu_1950 t-stat across tiers (institution=polity2_avg_1980_2010)")
print(f"{'═' * 78}")
print(f"\n  {'tier':<20} {'n':>4} {'gdp_t':>8} {'le_t':>8} {'tfr_t':>8} {'u5_t':>8}")
for tier in SAMPLE_TIERS:
    if tier not in b1_results:
        continue
    n_tier = len(SAMPLE_TIERS[tier])
    cells = [tier, str(n_tier)]
    for o in OUTCOMES:
        r = b1_results[tier]["polity2_avg_1980_2010"][o]
        if r.get("insufficient"):
            cells.append("--")
        else:
            cells.append(f"{r['z_edu_1950_t']:+.2f}")
    print(f"  {cells[0]:<20} {cells[1]:>4} "
          f"{cells[2]:>8} {cells[3]:>8} {cells[4]:>8} {cells[5]:>8}")

print(f"\n  {'tier':<20} {'n':>4} {'pol_gdp_t':>10} {'pol_le_t':>10} {'pol_tfr_t':>10} {'pol_u5_t':>10}")
for tier in SAMPLE_TIERS:
    if tier not in b1_results:
        continue
    n_tier = len(SAMPLE_TIERS[tier])
    cells = [tier, str(n_tier)]
    for o in OUTCOMES:
        r = b1_results[tier]["polity2_avg_1980_2010"][o]
        if r.get("insufficient"):
            cells.append("--")
        else:
            cells.append(f"{r['z_polity2_avg_1980_2010_t']:+.2f}")
    print(f"  {cells[0]:<20} {cells[1]:>4} "
          f"{cells[2]:>10} {cells[3]:>10} {cells[4]:>10} {cells[5]:>10}")


# ── B1b: Rate-of-expansion (Δedu) horse race ────────────────────────
#
# The paper's mechanism is about schooling EXPANSION (rate), not the 1950
# level. Run the same horse race with Δedu_1950→2000 as the education
# variable. This is robust to the entry-cohort restriction concern: rate
# captures whether a country expanded its schooling, not where it started.

print(f"\n{'═' * 78}")
print("B1b: RATE-OF-EXPANSION HORSE RACE (Δedu_1950→2000 instead of edu_1950)")
print(f"{'═' * 78}")

b1b_results = {}
for tier_name in ["full_panel", "ajr_64_only", "non_ajr_64",
                   "entry_cohort_10_90", "entry_cohort_20_90",
                   "entry_cohort_30_90", "below_90",
                   "ajr_64_entry_10_90", "ajr_64_entry_20_90",
                   "ajr_64_below_90"]:
    tier_df = SAMPLE_TIERS[tier_name]
    if len(tier_df) < 10: continue
    print(f"\n--- TIER: {tier_name} (n={len(tier_df)}) ---")
    b1b_results[tier_name] = {}
    for outcome in OUTCOMES:
        r = horse_race(tier_df, outcome,
                       ["delta_edu_1950_2000", "polity2_avg_1980_2010",
                        "log_gdp_1960"],
                       label=f"{tier_name} | {outcome}",
                       print_out=False)
        b1b_results[tier_name][outcome] = r

print(f"\nB1b SUMMARY: full_panel — Δedu_1950→2000 vs polity2 vs log_gdp_1960")
print(f"  {'outcome':<14} {'n':>4} {'Δedu_t':>8} {'Δedu_p':>8} {'pol_t':>8} {'pol_p':>8} {'gdp60_t':>8} {'R²':>6}")
for o in OUTCOMES:
    r = b1b_results["full_panel"][o]
    if r.get("insufficient"): continue
    print(f"  {o:<14} {r['n']:>4} "
          f"{r['z_delta_edu_1950_2000_t']:>+8.2f} {r['z_delta_edu_1950_2000_p']:>8.4f} "
          f"{r['z_polity2_avg_1980_2010_t']:>+8.2f} {r['z_polity2_avg_1980_2010_p']:>8.4f} "
          f"{r['z_log_gdp_1960_t']:>+8.2f} {r['r2']:>6.3f}")

print(f"\n  Cross-tier (Δedu_1950→2000 t, log_gdp_2020 outcome):")
print(f"  {'tier':<24} {'n':>4} {'Δedu_t':>8} {'pol_t':>8} {'gdp60_t':>9}")
for tier in ["full_panel", "ajr_64_only", "non_ajr_64",
              "entry_cohort_10_90", "entry_cohort_20_90",
              "entry_cohort_30_90", "below_90",
              "ajr_64_entry_10_90", "ajr_64_entry_20_90", "ajr_64_below_90"]:
    if tier in b1b_results:
        r = b1b_results[tier]["log_gdp_2020"]
        if r.get("insufficient"):
            print(f"  {tier:<24} {r['n']:>4} (insufficient)")
            continue
        print(f"  {tier:<24} {r['n']:>4} "
              f"{r['z_delta_edu_1950_2000_t']:>+8.2f} "
              f"{r['z_polity2_avg_1980_2010_t']:>+8.2f} "
              f"{r['z_log_gdp_1960_t']:>+9.2f}")

# Same cross-tier for tfr_2020 and le_2020 — these are the demographic
# crossing variables, more directly tied to the paper's mechanism than GDP.
for outcome_name in ["le_2020", "tfr_2020", "u5mr_2015"]:
    print(f"\n  Cross-tier (Δedu_1950→2000 t, {outcome_name} outcome):")
    print(f"  {'tier':<24} {'n':>4} {'Δedu_t':>8} {'pol_t':>8} {'gdp60_t':>9}")
    for tier in ["full_panel", "ajr_64_only", "non_ajr_64",
                  "entry_cohort_10_90", "entry_cohort_20_90",
                  "entry_cohort_30_90", "below_90",
                  "ajr_64_entry_10_90", "ajr_64_entry_20_90",
                  "ajr_64_below_90"]:
        if tier in b1b_results:
            r = b1b_results[tier].get(outcome_name, {})
            if not r or r.get("insufficient"):
                print(f"  {tier:<24} {r.get('n', 0):>4} (insufficient)")
                continue
            print(f"  {tier:<24} {r['n']:>4} "
                  f"{r['z_delta_edu_1950_2000_t']:>+8.2f} "
                  f"{r['z_polity2_avg_1980_2010_t']:>+8.2f} "
                  f"{r['z_log_gdp_1960_t']:>+9.2f}")


# ── B3: Falsifier search ────────────────────────────────────────────

print(f"\n{'═' * 78}")
print("B3: FALSIFIER SEARCH")
print(f"{'═' * 78}")

# Authoritarian + high-edu + crossed: paper predicts crossing, AJR predicts no
# polity2 < 0 = authoritarian; edu > median; crossed
auth_cutoff = 0.0
edu_med = df["edu_1950"].median()
auth_high_edu = df[(df["polity2_avg_1980_2010"] < auth_cutoff)
                    & (df["edu_1950"] > edu_med)
                    & df["crossing_year"].notna()].copy()
auth_high_edu = auth_high_edu.sort_values("crossing_year")
print(f"\nAUTHORITARIAN (polity2<0) + HIGH edu_1950 (>median {edu_med:.1f}%) + CROSSED:")
print(f"  Paper: predicts crossing. AJR: predicts no crossing.")
print(f"  n = {len(auth_high_edu)}")
print(f"  {'country':<28} {'pol_avg':>8} {'edu_50':>7} {'cross_yr':>9}")
for _, r in auth_high_edu.iterrows():
    print(f"  {r['country'][:28]:<28} {r['polity2_avg_1980_2010']:>+8.2f} "
          f"{r['edu_1950']:>7.1f} {int(r['crossing_year']):>9}")

# Democratic + low-edu + not crossed: AJR predicts crossing, paper predicts no
dem_low_edu = df[(df["polity2_avg_1980_2010"] >= 5)
                  & (df["edu_1950"] < edu_med)
                  & df["crossing_year"].isna()].copy()
dem_low_edu = dem_low_edu.sort_values("polity2_avg_1980_2010", ascending=False)
print(f"\nDEMOCRATIC (polity2>=5) + LOW edu_1950 (<median {edu_med:.1f}%) + NOT CROSSED:")
print(f"  Paper: predicts non-crossing. AJR: predicts crossing.")
print(f"  n = {len(dem_low_edu)}")
print(f"  {'country':<28} {'pol_avg':>8} {'edu_50':>7} {'le_2020':>9} {'tfr_2020':>9}")
for _, r in dem_low_edu.iterrows():
    le = r['le_2020']; tf = r['tfr_2020']
    print(f"  {r['country'][:28]:<28} {r['polity2_avg_1980_2010']:>+8.2f} "
          f"{r['edu_1950']:>7.1f} "
          f"{(f'{le:.1f}' if not pd.isna(le) else '   --'):>9} "
          f"{(f'{tf:.2f}' if not pd.isna(tf) else '   --'):>9}")


# ── B4: Crossing-year empirical floor reproduction ──────────────────

print(f"\n{'═' * 78}")
print("B4: CROSSING-YEAR EMPIRICAL FLOOR ON GLOBAL SAMPLE")
print(f"{'═' * 78}")

GLOBAL_LSEC_AT_LE_CROSS_MEDIAN = 65.0
GLOBAL_LSEC_AT_LE_CROSS_P10 = 42.0

crossers = df.dropna(subset=["edu_at_cross", "crossing_year"]).copy()
print(f"\nGlobal-panel crossers with edu-at-cross measurable: n = {len(crossers)}")
eac = crossers["edu_at_cross"]
print(f"  Lower-sec completion AT crossing (20-24 cohort, interp.):")
print(f"    median = {eac.median():.1f}%   (paper's global median: {GLOBAL_LSEC_AT_LE_CROSS_MEDIAN}%)")
print(f"    mean   = {eac.mean():.1f}%")
print(f"    p10    = {eac.quantile(0.1):.1f}%   (paper's global p10:    {GLOBAL_LSEC_AT_LE_CROSS_P10}%)")
print(f"    p25    = {eac.quantile(0.25):.1f}%")
print(f"    p75    = {eac.quantile(0.75):.1f}%")

# By tier
b4_by_tier = {}
print(f"\n  By tier:")
print(f"  {'tier':<20} {'n_crossers':>10} {'median':>8} {'p10':>8}")
for tier_name, tier_df in SAMPLE_TIERS.items():
    sub = tier_df.dropna(subset=["edu_at_cross", "crossing_year"])
    if len(sub) >= 5:
        m = sub["edu_at_cross"].median()
        p10 = sub["edu_at_cross"].quantile(0.1)
        b4_by_tier[tier_name] = {
            "n_crossers": int(len(sub)),
            "edu_at_cross_median": round(float(m), 1),
            "edu_at_cross_p10": round(float(p10), 1),
        }
        print(f"  {tier_name:<20} {len(sub):>10} {m:>8.1f} {p10:>8.1f}")
    else:
        b4_by_tier[tier_name] = {"n_crossers": int(len(sub)), "insufficient": True}


# ── B5: β_g amplification globally ──────────────────────────────────

print(f"\n{'═' * 78}")
print("B5: β_g AMPLIFICATION BY edu_1950 QUARTILE (global panel)")
print(f"{'═' * 78}")

bg_data = df.dropna(subset=["edu_1950", "edu_1975"]).copy()
bg_data["bucket"] = pd.qcut(bg_data["edu_1950"], q=4,
                              labels=["Q1_low", "Q2", "Q3", "Q4_high"],
                              duplicates="drop")
b5_results = {}
print(f"\n  {'Bucket':<10} {'n':>3} {'edu_1950 mean':>14} {'β_g (slope)':>12} {'t':>6} {'R²':>6}")
print(f"  {'-' * 58}")
for bucket in bg_data["bucket"].cat.categories:
    sub = bg_data[bg_data["bucket"] == bucket]
    if len(sub) < 4: continue
    X = sm.add_constant(sub["edu_1950"].values)
    fit = sm.OLS(sub["edu_1975"].values, X).fit()
    print(f"  {bucket:<10} {len(sub):>3} {sub['edu_1950'].mean():>14.1f} "
          f"{fit.params[1]:>+12.3f} {fit.tvalues[1]:>+6.2f} {fit.rsquared:>6.3f}")
    b5_results[str(bucket)] = {
        "n": int(len(sub)),
        "edu_1950_mean": round(float(sub["edu_1950"].mean()), 1),
        "beta_g": round(float(fit.params[1]), 3),
        "t": round(float(fit.tvalues[1]), 2),
        "r2": round(float(fit.rsquared), 3),
    }


# ── B6: Within-region tests ─────────────────────────────────────────

print(f"\n{'═' * 78}")
print("B6: WITHIN-REGION HORSE RACE")
print(f"  outcome=log_gdp_2020, predictors=z(edu_1950) + z(polity2_avg_1980_2010)")
print(f"{'═' * 78}")

REGION_LIST = sorted(df["region"].dropna().unique())
b6_results = {}
print(f"\n  {'region':<28} {'n':>3} {'edu_t':>8} {'edu_p':>8} {'pol_t':>8} {'pol_p':>8} {'R²':>6}")
for region in REGION_LIST:
    sub = df[df["region"] == region]
    if len(sub) < 6: continue
    r = horse_race(sub, "log_gdp_2020",
                    ["edu_1950", "polity2_avg_1980_2010"],
                    label=region, print_out=False)
    if r.get("insufficient"):
        continue
    b6_results[region] = r
    print(f"  {region[:28]:<28} {r['n']:>3} "
          f"{r['z_edu_1950_t']:>+8.2f} {r['z_edu_1950_p']:>8.4f} "
          f"{r['z_polity2_avg_1980_2010_t']:>+8.2f} "
          f"{r['z_polity2_avg_1980_2010_p']:>8.4f} {r['r2']:>6.3f}")


# ── Save JSON checkin ───────────────────────────────────────────────

checkin = {
    "n_full_panel": int(n_full),
    "drops": [{"country": c, "reason": w} for c, w in drop_log],
    "n_in_ajr64": int(df["in_ajr64"].sum()),
    "n_out_ajr64": int((~df["in_ajr64"]).sum()),
    "n_ever_colonized": int(df["ever_colonized"].sum()),
    "n_never_colonized": int((~df["ever_colonized"]).sum()),
    "n_crossed_by_2022": int(df["crossing_year"].notna().sum()),
    "tier_sizes": {t: int(len(d)) for t, d in SAMPLE_TIERS.items()},
    "b1_horse_race": b1_results,
    "b1b_delta_edu_horse_race": b1b_results,
    "b3_authoritarian_high_edu_crossers": [
        {"country": r["country"],
         "polity2_avg": round(float(r["polity2_avg_1980_2010"]), 2),
         "edu_1950": round(float(r["edu_1950"]), 1),
         "crossing_year": int(r["crossing_year"])}
        for _, r in auth_high_edu.iterrows()
    ],
    "b3_democratic_low_edu_non_crossers": [
        {"country": r["country"],
         "polity2_avg": round(float(r["polity2_avg_1980_2010"]), 2),
         "edu_1950": round(float(r["edu_1950"]), 1),
         "le_2020": (round(float(r["le_2020"]), 1) if not pd.isna(r["le_2020"]) else None),
         "tfr_2020": (round(float(r["tfr_2020"]), 2) if not pd.isna(r["tfr_2020"]) else None)}
        for _, r in dem_low_edu.iterrows()
    ],
    "b4_crossing_floor_global": {
        "n": int(len(crossers)),
        "median_pct": round(float(eac.median()), 1),
        "p10_pct": round(float(eac.quantile(0.1)), 1),
        "by_tier": b4_by_tier,
    },
    "b5_beta_g_buckets_global": b5_results,
    "b6_within_region": b6_results,
    "edu_1950_median_global": round(float(edu_med), 1),
}
write_checkin("colonial_global.json", checkin,
              script_path="scripts/robustness/colonial_global.py")
print("\nDone.")
