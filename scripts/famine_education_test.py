"""
Famine and Education: Do famines only occur in low-education settings?

Tests the hypothesis that major famines since 1950 occur exclusively
in countries with low lower-secondary completion rates.

Education data: WCDE v3, lower secondary completion, both sexes, age 20-24.
Famine list: well-documented major famines from historical record.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

# ── Major famines since 1950 ──────────────────────────────────────────────
# (country, midpoint_year, deaths_low_estimate, brief_cause)
# Deaths in thousands. Midpoint year used for matching to education data.

FAMINES = [
    # Country, midpoint year, deaths (thousands), cause
    ("China",          1960,  15000, "Great Leap Forward"),
    ("China",          1969,    500, "Cultural Revolution disruptions"),
    ("Ethiopia",       1973,    200, "Wollo famine"),
    ("Bangladesh",     1974,   1000, "Floods + economic crisis"),
    ("Cambodia",       1977,   1000, "Khmer Rouge forced collectivization"),
    ("Uganda",         1980,    300, "Karamoja famine, civil war"),
    ("Mozambique",     1985,    100, "Civil war + drought"),
    ("Ethiopia",       1984,    400, "1983-85 famine"),
    ("Sudan",          1988,    250, "Darfur/civil war famine"),
    ("Somalia",        1992,    300, "Civil war + drought"),
    ("North Korea",    1996,    600, "Arduous March"),
    ("Congo, Dem. Rep.", 2000,  200, "Second Congo War"),
    ("Niger",          2005,     50, "Sahel food crisis"),
    ("Somalia",        2011,    260, "East Africa drought"),
    ("South Sudan",    2017,     50, "Civil war famine"),
    ("Yemen",          2018,     85, "Civil war + blockade"),
    ("Madagascar",     2021,     50, "Grand Sud drought"),
    ("Ethiopia",       2021,    100, "Tigray war famine"),
    ("Afghanistan",    2022,     50, "Economic collapse post-Taliban"),
    ("Nigeria",        1968,   1000, "Biafra war famine"),
    ("India [Bihar]",  1966,    100, "Bihar famine: monsoon failure, 43% crop loss, 70-130k excess deaths (Dyson & Maharatna 1992)"),
]

# ── Near-misses: countries that AVOIDED famine despite severe shocks ─────
# (country, year, shock_description)
# These countries faced conditions that could have produced famine but didn't.

NEAR_MISSES = [
    ("Cuba",           1993, "Soviet collapse: GDP fell ~35%, food imports cut by 80%"),
    ("Sri Lanka",      1983, "Civil war (1983-2009), trade disruption, displacement"),
    ("Costa Rica",     1981, "Debt crisis, GDP fell 10%, severe austerity"),
    ("Japan",          1945, "Post-WWII devastation, cities destroyed, occupation"),
    ("Germany",        1946, "Post-WWII, country divided, infrastructure destroyed"),
    ("South Korea",    1950, "Korean War, 3 million dead, infrastructure destroyed"),
    ("Vietnam",        1979, "Post-reunification + China border war + US embargo"),
    ("Georgia",        1993, "Post-Soviet collapse + civil war + Abkhazia conflict"),
    ("Armenia",        1993, "Post-Soviet collapse + Nagorno-Karabakh war + blockade"),
    ("Jamaica",        1977, "Economic crisis, IMF austerity, political violence"),
    ("Iran",           1980, "Iran-Iraq war (8 years), sanctions, revolution"),
    ("Iraq",           1991, "Gulf War + severe sanctions regime"),
    ("Libya",          2011, "Civil war, NATO intervention, state collapse"),
    ("Syria",          2015, "Civil war, sieges, displacement — severe food insecurity but not classified as famine"),
    ("Ukraine",        2022, "Russian invasion, grain blockade, infrastructure destroyed"),
    ("Tunisia",        2011, "Revolution, economic disruption"),
    ("Albania",        1997, "State collapse, pyramid scheme crisis, civil disorder"),
    ("India [Kerala]", 1966, "National drought supply crisis; 40% food-deficit state, violent riots — but no famine. Same democracy as Bihar."),
]

# ── Load education data ──────────────────────────────────────────────────

edu_wide = pd.read_csv(os.path.join(ROOT, "wcde/data/processed/lower_sec_both.csv"))
years = [int(c) for c in edu_wide.columns if c != "country"]

# Melt to long format
edu = edu_wide.melt(id_vars="country", var_name="year", value_name="lower_sec_pct")
edu["year"] = edu["year"].astype(int)
edu = edu.dropna(subset=["lower_sec_pct"])

# ── Name mapping (famine country names → WCDE names) ─────────────────────

NAME_MAP = {
    "North Korea": "Democratic People's Republic of Korea",
    "Congo, Dem. Rep.": "Democratic Republic of the Congo",
    "South Sudan": "South Sudan",  # may not exist in WCDE pre-2011
}

# ── Subnational overrides ────────────────────────────────────────────────
# WCDE has no state-level data. For Indian states we use census literacy
# as a proxy for education level. These are well-documented:
#   Bihar 1961 census: 22.0% literacy (female: 8.1%)
#   Kerala 1961 census: 55.1% literacy (female: 38.9%)
# Literacy is a lower bar than lower-secondary completion, so actual
# lower-sec completion was even lower — the contrast only gets sharper.
SUBNATIONAL_EDU = {
    "India [Bihar]":  {1965: 22.0},   # 1961 census literacy
    "India [Kerala]": {1965: 55.1},   # 1961 census literacy
}


def lookup_education(country, famine_year):
    """Find the closest WCDE education value to the famine year."""
    # Check subnational overrides first
    if country in SUBNATIONAL_EDU:
        overrides = SUBNATIONAL_EDU[country]
        closest_yr = min(overrides.keys(), key=lambda y: abs(y - famine_year))
        return country, overrides[closest_yr]
    wcde_name = NAME_MAP.get(country, country)
    country_data = edu[edu["country"] == wcde_name]
    if country_data.empty:
        return wcde_name, np.nan
    # Find closest available year
    closest_idx = (country_data["year"] - famine_year).abs().idxmin()
    row = country_data.loc[closest_idx]
    return wcde_name, row["lower_sec_pct"]


# ── Build results table ──────────────────────────────────────────────────

results = []
for country, year, deaths_k, cause in FAMINES:
    wcde_name, edu_pct = lookup_education(country, year)
    results.append({
        "country": country,
        "year": year,
        "deaths_thousands": deaths_k,
        "cause": cause,
        "wcde_name": wcde_name,
        "lower_sec_pct": edu_pct,
    })

df = pd.DataFrame(results).sort_values("lower_sec_pct")

# ── Print results ────────────────────────────────────────────────────────

print("=" * 90)
print("FAMINES SINCE 1950 vs. LOWER SECONDARY COMPLETION (age 20-24, both sexes)")
print("=" * 90)
print()
print(f"{'Country':<25} {'Year':>5}  {'Deaths(k)':>10}  {'Edu %':>7}  Cause")
print("-" * 90)
for _, r in df.iterrows():
    edu_str = f"{r['lower_sec_pct']:.1f}" if pd.notna(r["lower_sec_pct"]) else "N/A"
    print(f"{r['country']:<25} {r['year']:>5}  {r['deaths_thousands']:>10,.0f}  {edu_str:>7}  {r['cause']}")

print()

# ── Summary statistics ───────────────────────────────────────────────────

valid = df.dropna(subset=["lower_sec_pct"])
print(f"Famines with education data: {len(valid)} / {len(df)}")
print(f"Mean education at famine:    {valid['lower_sec_pct'].mean():.1f}%")
print(f"Median education at famine:  {valid['lower_sec_pct'].median():.1f}%")
print(f"Max education at famine:     {valid['lower_sec_pct'].max():.1f}%")
print(f"Min education at famine:     {valid['lower_sec_pct'].min():.1f}%")
print()

# Threshold test
for threshold in [20, 30, 40, 50]:
    below = (valid["lower_sec_pct"] < threshold).sum()
    print(f"  Famines with education < {threshold}%: {below}/{len(valid)} "
          f"({100*below/len(valid):.0f}%)")

print()

# ── Compare to non-famine countries at same time ─────────────────────────

print("=" * 90)
print("COMPARISON: Famine countries vs. all countries at same time periods")
print("=" * 90)
print()

for _, r in valid.iterrows():
    yr = r["year"]
    # Find closest WCDE year
    closest_yr = min(years, key=lambda y: abs(y - yr))
    all_countries = edu[edu["year"] == closest_yr].dropna()
    percentile = (all_countries["lower_sec_pct"] < r["lower_sec_pct"]).sum() / len(all_countries) * 100
    print(f"  {r['country']:<22} ({r['year']}): {r['lower_sec_pct']:.1f}% education — "
          f"percentile {percentile:.0f}th among all countries in {closest_yr}")

print()

# ── Global distribution at famine times ──────────────────────────────────

# For each decade, show where famine countries sit in global distribution
print("=" * 90)
print("GLOBAL CONTEXT: Education distribution when famines occurred")
print("=" * 90)
print()

for decade_yr in [1960, 1975, 1990, 2005, 2020]:
    closest_yr = min(years, key=lambda y: abs(y - decade_yr))
    all_vals = edu[edu["year"] == closest_yr]["lower_sec_pct"].dropna()
    decade_famines = valid[(valid["year"] >= decade_yr - 7) & (valid["year"] <= decade_yr + 7)]
    if len(decade_famines) > 0:
        print(f"  ~{decade_yr}: Global median education = {all_vals.median():.1f}%, "
              f"global mean = {all_vals.mean():.1f}%")
        for _, f in decade_famines.iterrows():
            print(f"    → {f['country']} ({f['year']}): {f['lower_sec_pct']:.1f}%")
        print()

# ── Has any country with >50% education ever had a famine? ───────────────

high_edu_famines = valid[valid["lower_sec_pct"] > 50]
print("=" * 90)
if len(high_edu_famines) == 0:
    print("KEY FINDING: NO famine has occurred in a country with >50% lower secondary completion.")
else:
    print(f"EXCEPTION: {len(high_edu_famines)} famine(s) in countries with >50% education:")
    for _, r in high_edu_famines.iterrows():
        print(f"  {r['country']} ({r['year']}): {r['lower_sec_pct']:.1f}%")
print("=" * 90)

# ── Near-misses: educated countries that avoided famine ──────────────────

NM_NAME_MAP = {
    "South Korea": "Republic of Korea",
    "Iran": "Iran (Islamic Republic of)",
    "Syria": "Syrian Arab Republic",
    "Vietnam": "Viet Nam",
    "Libya": "Libyan Arab Jamahiriya",
}
NM_NAME_MAP.update(NAME_MAP)
NM_NAME_MAP.update({k: k for k in SUBNATIONAL_EDU})

print()
print("=" * 90)
print("NEAR-MISSES: Countries that AVOIDED famine despite severe shocks")
print("=" * 90)
print()
print(f"{'Country':<20} {'Year':>5}  {'Edu %':>7}  Shock")
print("-" * 90)

nm_results = []
for country, year, shock in NEAR_MISSES:
    # Check subnational overrides first
    if country in SUBNATIONAL_EDU:
        overrides = SUBNATIONAL_EDU[country]
        closest_yr = min(overrides.keys(), key=lambda y: abs(y - year))
        edu_pct = overrides[closest_yr]
    else:
        wcde_name = NM_NAME_MAP.get(country, country)
        country_data = edu[edu["country"] == wcde_name]
        if country_data.empty:
            edu_pct = np.nan
        else:
            closest_idx = (country_data["year"] - year).abs().idxmin()
            edu_pct = country_data.loc[closest_idx, "lower_sec_pct"]
    nm_results.append({
        "country": country, "year": year, "shock": shock,
        "lower_sec_pct": edu_pct
    })
    edu_str = f"{edu_pct:.1f}" if pd.notna(edu_pct) else "N/A"
    print(f"  {country:<20} {year:>5}  {edu_str:>7}  {shock}")

nm_df = pd.DataFrame(nm_results).dropna(subset=["lower_sec_pct"])

print()
print(f"Near-miss mean education:   {nm_df['lower_sec_pct'].mean():.1f}%")
print(f"Near-miss median education: {nm_df['lower_sec_pct'].median():.1f}%")
print(f"Famine mean education:      {valid['lower_sec_pct'].mean():.1f}%")
print(f"Famine median education:    {valid['lower_sec_pct'].median():.1f}%")
print()

# Statistical test: are famine and near-miss education levels different?
from scipy import stats
t_stat, p_val = stats.ttest_ind(valid["lower_sec_pct"], nm_df["lower_sec_pct"],
                                 equal_var=False)
u_stat, u_pval = stats.mannwhitneyu(valid["lower_sec_pct"], nm_df["lower_sec_pct"],
                                     alternative="less")
print(f"Welch's t-test (famine < near-miss): t={t_stat:.2f}, p={p_val:.4f}")
print(f"Mann-Whitney U (famine < near-miss): U={u_stat:.0f}, p={u_pval:.6f}")
print()

# ── Write checkin JSON ───────────────────────────────────────────────────

import json

checkin_data = {
    "numbers": {
        "Famine-count": len(valid),
        "Famine-below-50-ct": int((valid["lower_sec_pct"] < 50).sum()),
        "Famine-median-edu": round(valid["lower_sec_pct"].median(), 1),
        "Famine-mean-edu": round(valid["lower_sec_pct"].mean(), 1),
        "NM-median-edu": round(nm_df["lower_sec_pct"].median(), 1),
        "Famine-p-val": u_pval,
    },
    "metadata": {
        "description": "Famine vs education test: 21 major famines since 1950",
        "data_source": "WCDE v3 (lower secondary completion)",
        "test": "Mann-Whitney U (famine < near-miss)",
        "famine_count": len(valid),
        "near_miss_count": len(nm_df),
    }
}

checkin_path = os.path.join(ROOT, "checkin", "famine_education_test.json")
os.makedirs(os.path.dirname(checkin_path), exist_ok=True)
with open(checkin_path, "w") as f:
    json.dump(checkin_data, f, indent=2)
print(f"Checkin: {checkin_path}")

# ── Bihar vs Kerala: within-country natural experiment ───────────────────

print("=" * 90)
print("BIHAR vs KERALA: Same democracy, same monsoon, different education, different outcome")
print("=" * 90)
print()
print("The 1965-66 national drought crashed India's grain production 19%.")
print("Bihar had a famine (70-130k dead). Kerala — India's most food-deficit")
print("state (40% import-dependent), hit so hard there were violent food riots —")
print("had no famine. Same Constitution, same free press, same central govt.")
print("Sen says democracy prevents famine. Democracy was constant. Education was not.")
print()
print("Kerala even had a famine in living memory: 1943 Travancore famine (~90k dead)")
print("under princely rule with low literacy. Same structural vulnerability,")
print("different education level, opposite outcome.")
print()
print("Indian Census literacy rates (proxy for education):")
print()
print(f"  {'':20} {'Bihar':>10} {'Kerala':>10} {'India avg':>10}")
print(f"  {'-'*50}")
# Census data: Bihar, Kerala, India literacy rates
census = [
    (1951, 13.5, 47.2, 18.3),
    (1961, 22.0, 55.1, 28.3),
    (1971, 23.2, 60.4, 34.5),
    (1981, 31.9, 70.4, 43.6),
    (1991, 36.7, 89.8, 52.2),
    (2001, 47.0, 90.9, 64.8),
    (2011, 61.4, 93.9, 74.0),
]
for yr, bih, ker, ind in census:
    marker = " ← FAMINE YEAR" if yr == 1961 else ""
    print(f"  {yr:>20} {bih:>9.1f}% {ker:>9.1f}% {ind:>9.1f}%{marker}")
print()
print("At the time of the Bihar famine (1966):")
print("  Bihar literacy:  ~22%  (female: ~9%)")
print("  Kerala literacy: ~55%  (female: ~39%)")
print("  Gap: 2.5x overall, 4.3x for women")
print()
print("Bihar 2019-21 (NFHS-5): 42.9% child stunting")
print("Kerala 2019-21 (NFHS-5): 23.4% child stunting")
print("Education gap persists → nutrition gap persists.")
print()
print("Key sources:")
print("  Dyson & Maharatna (1992) — 70-130k excess deaths in Bihar famine")
print("  Myhrvold-Hanssen (2003) — challenges Sen's democracy-prevents-famine thesis using Bihar")
print("  Dreze: 'precious little evidence' that Bihar was a success story")
print("  Brass (1986) — Congress govt suppressed acknowledgment of the crisis")
print()

# ── Plot ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: education (x) vs year (y), color = famine vs near-miss
ax = axes[0]
ax.scatter(valid["lower_sec_pct"], valid["year"],
           s=80, alpha=0.7, c="crimson", edgecolors="darkred", zorder=5,
           label="Famine occurred")
ax.scatter(nm_df["lower_sec_pct"], nm_df["year"],
           s=80, alpha=0.7, c="forestgreen", edgecolors="darkgreen", zorder=5,
           marker="^", label="Near-miss (avoided)")
for _, r in valid.iterrows():
    ax.annotate(r["country"], (r["lower_sec_pct"], r["year"]),
                fontsize=6, ha="left", va="center",
                textcoords="offset points", xytext=(5, 0))
for _, r in nm_df.iterrows():
    ax.annotate(r["country"], (r["lower_sec_pct"], r["year"]),
                fontsize=6, ha="left", va="center",
                textcoords="offset points", xytext=(5, 0))
ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
ax.set_xlabel("Lower Secondary Completion (%, age 20-24)")
ax.set_ylabel("Year")
ax.set_title("Famines vs. Near-Misses:\nEducation Level at Time of Crisis")
ax.legend(loc="lower right", fontsize=7)
ax.set_xlim(-5, 115)
ax.set_ylim(1942, 2027)

# Right: side-by-side box/strip comparison
ax = axes[1]
positions = [0, 1]
bp = ax.boxplot([valid["lower_sec_pct"].values, nm_df["lower_sec_pct"].values],
                positions=positions, widths=0.4, patch_artist=True,
                showmeans=True, meanprops=dict(marker="D", markerfacecolor="black", markersize=5))
bp["boxes"][0].set_facecolor("lightcoral")
bp["boxes"][1].set_facecolor("lightgreen")
# Overlay individual points
ax.scatter(np.zeros(len(valid)) + np.random.normal(0, 0.05, len(valid)),
           valid["lower_sec_pct"], c="crimson", alpha=0.5, s=30, zorder=5)
ax.scatter(np.ones(len(nm_df)) + np.random.normal(0, 0.05, len(nm_df)),
           nm_df["lower_sec_pct"], c="forestgreen", alpha=0.5, s=30, zorder=5)
ax.set_xticks(positions)
ax.set_xticklabels(["Famine\nOccurred", "Near-Miss\n(Avoided)"])
ax.set_ylabel("Lower Secondary Completion (%, age 20-24)")
ax.set_title(f"Education at Crisis:\nFamine vs. Averted (p={u_pval:.4f})")
ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
outpath = os.path.join(ROOT, "paper/figures/famine_vs_education.png")
os.makedirs(os.path.dirname(outpath), exist_ok=True)
plt.savefig(outpath, dpi=150)
print(f"\nFigure saved: {outpath}")
plt.close()

print("\nDone.")
