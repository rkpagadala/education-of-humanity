"""
Happiness vs Education and Happiness vs GDP — side-by-side scatter plots.

Uses latest available year per country (cross-section) to avoid
visual clutter from panel stacking.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
WCDE = ROOT / "wcde" / "data" / "processed"
FIG  = ROOT / "paper" / "figures"
FIG.mkdir(exist_ok=True)

# ── Load and merge (same logic as happiness_education.py) ───────────
hap = pd.read_csv(DATA / "world_happiness_report.csv")
hap["year"] = hap["year"].astype(int)
hap["happiness_score"] = pd.to_numeric(hap["happiness_score"], errors="coerce")
hap = hap[["country", "year", "happiness_score"]].dropna()

edu_wide = pd.read_csv(WCDE / "lower_sec_both.csv")
edu_long = edu_wide.melt(id_vars="country", var_name="year", value_name="education")
edu_long["year"] = edu_long["year"].astype(int)
edu_annual = []
for country, grp in edu_long.groupby("country"):
    grp = grp.sort_values("year").set_index("year")
    grp = grp.reindex(range(grp.index.min(), grp.index.max() + 1))
    grp["education"] = grp["education"].interpolate(method="linear")
    grp["country"] = country
    edu_annual.append(grp.reset_index())
edu_annual = pd.concat(edu_annual, ignore_index=True)

gdp_wide = pd.read_csv(DATA / "gdppercapita_us_inflation_adjusted.csv")
gdp_long = gdp_wide.melt(id_vars="Country", var_name="year", value_name="gdp")
gdp_long.rename(columns={"Country": "country"}, inplace=True)
gdp_long["year"] = gdp_long["year"].astype(int)
gdp_long["gdp"] = pd.to_numeric(gdp_long["gdp"], errors="coerce")

for df in [hap, edu_annual, gdp_long]:
    df["country"] = df["country"].str.strip().str.lower()

name_map = {
    "taiwan province of china": "taiwan",
    "hong kong s.a.r. of china": "china, hong kong sar",
    "hong kong s.a.r., china": "china, hong kong sar",
    "north cyprus": "cyprus",
    "palestinian territories": "state of palestine",
    "somaliland region": "somalia",
    "congo (brazzaville)": "congo",
    "congo (kinshasa)": "dem. rep. congo",
    "trinidad & tobago": "trinidad and tobago",
    "czechia": "czech republic",
    "turkiye": "turkey", "türkiye": "turkey",
    "ivory coast": "cote d'ivoire",
    "south korea": "republic of korea",
    "laos": "lao pdr",
    "russia": "russian federation",
    "iran": "iran (islamic republic of)",
    "syria": "syrian arab republic",
    "bolivia": "bolivia (plurinational state of)",
    "venezuela": "venezuela (bolivarian republic of)",
    "vietnam": "viet nam",
    "tanzania": "united republic of tanzania",
    "moldova": "republic of moldova",
    "kyrgyzstan": "kyrgyz republic",
    "korea, rep.": "republic of korea",
    "iran, islamic rep.": "iran (islamic republic of)",
    "egypt, arab rep.": "egypt",
    "venezuela, rb": "venezuela (bolivarian republic of)",
    "congo, dem. rep.": "dem. rep. congo",
    "congo, rep.": "congo",
    "slovak republic": "slovakia",
    "cote d'ivoire": "cote d'ivoire",
    "yemen, rep.": "yemen",
    "hong kong sar, china": "china, hong kong sar",
}
for df in [hap, edu_annual, gdp_long]:
    df["country"] = df["country"].replace(name_map)

df = hap.merge(edu_annual, on=["country", "year"], how="inner")
df = df.merge(gdp_long, on=["country", "year"], how="inner")
df = df.dropna(subset=["happiness_score", "education", "gdp"])
df = df[df["gdp"] > 0]
df["log_gdp"] = np.log(df["gdp"])

# Use latest year per country for clean cross-section
latest = df.sort_values("year").groupby("country").last().reset_index()

# ── Regressions ─────────────────────────────────────────────────────
def ols_r2(y, x):
    X = sm.add_constant(x)
    return sm.OLS(y, X).fit()

m_edu = ols_r2(latest["happiness_score"], latest["education"])
m_gdp = ols_r2(latest["happiness_score"], latest["log_gdp"])

# ── Figure ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

# Panel A: Happiness vs Education
ax1.scatter(latest["education"], latest["happiness_score"],
            alpha=0.5, s=25, color="#2166ac", edgecolors="none")
xfit = np.linspace(latest["education"].min(), latest["education"].max(), 100)
yfit = m_edu.params["const"] + m_edu.params["education"] * xfit
ax1.plot(xfit, yfit, color="#b2182b", linewidth=2)
ax1.set_xlabel("Lower secondary completion (%)", fontsize=12)
ax1.set_ylabel("Happiness (Cantril ladder, 0–10)", fontsize=12)
ax1.set_title(f"A.  Education → Happiness\nR² = {m_edu.rsquared:.3f}", fontsize=13)
ax1.set_xlim(0, 105)
ax1.set_ylim(2, 8.5)

# Panel B: Happiness vs log(GDP)
ax2.scatter(latest["log_gdp"], latest["happiness_score"],
            alpha=0.5, s=25, color="#2166ac", edgecolors="none")
xfit2 = np.linspace(latest["log_gdp"].min(), latest["log_gdp"].max(), 100)
yfit2 = m_gdp.params["const"] + m_gdp.params["log_gdp"] * xfit2
ax2.plot(xfit2, yfit2, color="#b2182b", linewidth=2)
ax2.set_xlabel("log GDP per capita (constant 2015 USD)", fontsize=12)
ax2.set_title(f"B.  log(GDP) → Happiness\nR² = {m_gdp.rsquared:.3f}", fontsize=13)

fig.suptitle("Happiness tracks income, not education", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "fig_happiness_edu_gdp.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "fig_happiness_edu_gdp.pdf", bbox_inches="tight")
print(f"Saved to {FIG / 'fig_happiness_edu_gdp.png'}")
print(f"N = {len(latest)} countries (latest year per country)")
print(f"Education R² = {m_edu.rsquared:.4f}")
print(f"log(GDP)  R² = {m_gdp.rsquared:.4f}")
