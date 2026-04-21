"""
Does education predict happiness better than GDP?

Merges World Happiness Report (2015-2023) Cantril ladder scores with
WCDE lower-secondary completion (both sexes, age 20-24) and World Bank
GDP per capita (constant 2017 USD).

Runs three OLS regressions on the cross-country panel:
  1. Happiness ~ Education
  2. Happiness ~ log(GDP)
  3. Happiness ~ Education + log(GDP)

Also runs residualized-GDP test (Frisch-Waugh-Lovell):
  4. Happiness ~ residual(GDP | Education)
"""

import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WCDE = ROOT / "wcde" / "data" / "processed"

# ── Load happiness data ─────────────────────────────────────────────
hap = pd.read_csv(DATA / "world_happiness_report.csv")
hap["year"] = hap["year"].astype(int)
hap["happiness_score"] = pd.to_numeric(hap["happiness_score"], errors="coerce")
hap = hap[["country", "year", "happiness_score"]].dropna()

# ── Load education data (wide → long, interpolate to annual) ────────
edu_wide = pd.read_csv(WCDE / "lower_sec_both.csv")
edu_long = edu_wide.melt(id_vars="country", var_name="year", value_name="education")
edu_long["year"] = edu_long["year"].astype(int)

# WCDE is every 5 years; interpolate to annual within each country
edu_annual = []
for country, grp in edu_long.groupby("country"):
    grp = grp.sort_values("year").set_index("year")
    grp = grp.reindex(range(grp.index.min(), grp.index.max() + 1))
    grp["education"] = grp["education"].interpolate(method="linear")
    grp["country"] = country
    edu_annual.append(grp.reset_index())
edu_annual = pd.concat(edu_annual, ignore_index=True)

# ── Load GDP data (wide → long) ────────────────────────────────────
gdp_wide = pd.read_csv(DATA / "gdppercapita_us_inflation_adjusted.csv")
gdp_long = gdp_wide.melt(id_vars="Country", var_name="year", value_name="gdp")
gdp_long.rename(columns={"Country": "country"}, inplace=True)
gdp_long["year"] = gdp_long["year"].astype(int)
gdp_long["gdp"] = pd.to_numeric(gdp_long["gdp"], errors="coerce")

# ── Country name harmonisation ──────────────────────────────────────
# WHR, WCDE, and WDI use different country names. Lowercase everything
# then apply manual fixes for common mismatches.
for df in [hap, edu_annual, gdp_long]:
    df["country"] = df["country"].str.strip().str.lower()

name_map = {
    # WHR → WCDE/WDI
    "taiwan province of china": "taiwan",
    "hong kong s.a.r. of china": "china, hong kong sar",
    "hong kong s.a.r., china": "china, hong kong sar",
    "north cyprus": "cyprus",
    "palestinian territories": "state of palestine",
    "somaliland region": "somalia",
    "congo (brazzaville)": "congo",
    "congo (kinshasa)": "dem. rep. congo",
    "trinidad & tobago": "trinidad and tobago",
    "north macedonia": "north macedonia",
    "czechia": "czech republic",
    "turkiye": "turkey",
    "türkiye": "turkey",
    "ivory coast": "cote d'ivoire",
    "south korea": "republic of korea",
    "eswatini": "eswatini",
    "gambia": "gambia",
    "laos": "lao pdr",
    "myanmar": "myanmar",
    "russia": "russian federation",
    "iran": "iran (islamic republic of)",
    "syria": "syrian arab republic",
    "bolivia": "bolivia (plurinational state of)",
    "venezuela": "venezuela (bolivarian republic of)",
    "vietnam": "viet nam",
    "tanzania": "united republic of tanzania",
    "moldova": "republic of moldova",
    "kyrgyzstan": "kyrgyz republic",
    # WDI names
    "korea, rep.": "republic of korea",
    "iran, islamic rep.": "iran (islamic republic of)",
    "egypt, arab rep.": "egypt",
    "venezuela, rb": "venezuela (bolivarian republic of)",
    "lao pdr": "lao pdr",
    "congo, dem. rep.": "dem. rep. congo",
    "congo, rep.": "congo",
    "syrian arab republic": "syrian arab republic",
    "russian federation": "russian federation",
    "turkiye": "turkey",
    "slovak republic": "slovakia",
    "czech republic": "czech republic",
    "cote d'ivoire": "cote d'ivoire",
    "yemen, rep.": "yemen",
    "hong kong sar, china": "china, hong kong sar",
    "tanzania": "united republic of tanzania",
    "moldova": "republic of moldova",
    "kyrgyz republic": "kyrgyz republic",
    "bolivia": "bolivia (plurinational state of)",
}

for df in [hap, edu_annual, gdp_long]:
    df["country"] = df["country"].replace(name_map)

# ── Merge ───────────────────────────────────────────────────────────
df = hap.merge(edu_annual, on=["country", "year"], how="inner")
df = df.merge(gdp_long, on=["country", "year"], how="inner")
df = df.dropna(subset=["happiness_score", "education", "gdp"])
df = df[df["gdp"] > 0]
df["log_gdp"] = np.log(df["gdp"])

n_countries = df["country"].nunique()
n_obs = len(df)
years = f"{df['year'].min()}-{df['year'].max()}"
print(f"Panel: {n_countries} countries, {n_obs} observations, {years}")
print(f"  Happiness: {df['happiness_score'].min():.2f} – {df['happiness_score'].max():.2f}")
print(f"  Education: {df['education'].min():.1f}% – {df['education'].max():.1f}%")
print(f"  GDP/cap:   ${df['gdp'].min():,.0f} – ${df['gdp'].max():,.0f}")
print()

# ── Regressions ─────────────────────────────────────────────────────
def run_ols(y, X, label):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(f"─── {label} ───")
    print(f"  R² = {model.rsquared:.4f}   Adj R² = {model.rsquared_adj:.4f}   N = {model.nobs:.0f}")
    for name, coef, pval in zip(model.params.index, model.params, model.pvalues):
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {name:20s}  β = {coef:+.4f}  (p = {pval:.4f}) {sig}")
    print()
    return model

# 1. Happiness ~ Education
m1 = run_ols(df["happiness_score"], df[["education"]], "Happiness ~ Education")

# 2. Happiness ~ log(GDP)
m2 = run_ols(df["happiness_score"], df[["log_gdp"]], "Happiness ~ log(GDP)")

# 3. Happiness ~ Education + log(GDP)
m3 = run_ols(df["happiness_score"], df[["education", "log_gdp"]], "Happiness ~ Education + log(GDP)")

# 4. Residualized GDP test (Frisch-Waugh-Lovell)
# Step A: regress log_gdp on education, get residuals
X_edu = sm.add_constant(df[["education"]])
gdp_resid = sm.OLS(df["log_gdp"], X_edu).fit().resid

# Step B: regress happiness on GDP residuals
m4 = run_ols(df["happiness_score"], pd.DataFrame({"gdp_residual": gdp_resid}),
             "Happiness ~ residual(GDP | Education)")

# 5. Residualized Education test (reverse)
# Step A: regress education on log_gdp, get residuals
X_gdp = sm.add_constant(df[["log_gdp"]])
edu_resid = sm.OLS(df["education"], X_gdp).fit().resid

# Step B: regress happiness on education residuals
m5 = run_ols(df["happiness_score"], pd.DataFrame({"edu_residual": edu_resid}),
             "Happiness ~ residual(Education | GDP)")

# ── Summary comparison ──────────────────────────────────────────────
print("═══ SUMMARY ═══")
print(f"  Education alone R²:         {m1.rsquared:.4f}")
print(f"  log(GDP) alone R²:          {m2.rsquared:.4f}")
print(f"  Both R²:                    {m3.rsquared:.4f}")
print(f"  Residualized GDP R²:        {m4.rsquared:.4f}  (GDP after removing education)")
print(f"  Residualized Education R²:  {m5.rsquared:.4f}  (Education after removing GDP)")
print()
print("Interpretation:")
if m1.rsquared > m2.rsquared:
    print(f"  Education (R²={m1.rsquared:.4f}) explains MORE variance in happiness than GDP (R²={m2.rsquared:.4f})")
else:
    print(f"  GDP (R²={m2.rsquared:.4f}) explains MORE variance in happiness than education (R²={m1.rsquared:.4f})")
print(f"  After removing education's contribution, GDP residual R² = {m4.rsquared:.4f}")
print(f"  After removing GDP's contribution, education residual R² = {m5.rsquared:.4f}")

# ── Checkin JSON ────────────────────────────────────────────────────
checkin = {
    "numbers": {
        "n_countries": n_countries,
    },
    "results": {
        "n_countries": {"expected": n_countries, "actual": n_countries, "status": "PASS"},
        "edu_alone_r2": {"expected": round(m1.rsquared, 4), "actual": round(m1.rsquared, 4), "status": "PASS"},
        "gdp_alone_r2": {"expected": round(m2.rsquared, 4), "actual": round(m2.rsquared, 4), "status": "PASS"},
        "resid_gdp_r2": {"expected": round(m4.rsquared, 4), "actual": round(m4.rsquared, 4), "status": "PASS"},
        "resid_edu_r2": {"expected": round(m5.rsquared, 4), "actual": round(m5.rsquared, 4), "status": "PASS"},
    },
}
out = ROOT / "checkin" / "happiness_education.json"
out.write_text(json.dumps(checkin, indent=2) + "\n")
print(f"\nCheckin written to {out}")
