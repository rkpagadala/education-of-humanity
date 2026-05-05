"""
Verify Russia and South Africa numbers for the paper.
Sources:
  - Education: WCDE v3, lower secondary completion, both sexes, age 20-24
  - Life expectancy: World Bank WDI (SP.DYN.LE00.IN)
  - TFR: World Bank WDI (SP.DYN.TFRT.IN)
"""

import os
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Load data ──────────────────────────────────────────────────────────
edu = pd.read_csv(f"{BASE}/wcde/data/processed/lower_sec_both.csv")
le  = pd.read_csv(f"{BASE}/data/life_expectancy_years.csv")
tfr = pd.read_csv(f"{BASE}/data/children_per_woman_total_fertility.csv")

# Education: columns are country + year strings (5-year intervals)
# WDI: columns are Country + annual year strings

def get_edu(df, country, years):
    row = df[df["country"] == country]
    if row.empty:
        print(f"  !! Country '{country}' not found in education data")
        return
    for y in years:
        val = row[str(y)].values[0]
        print(f"  {y}: {val:.2f}%")

def get_wdi(df, country, years, label):
    row = df[df["Country"] == country]
    if row.empty:
        print(f"  !! Country '{country}' not found in {label} data")
        return
    for y in years:
        col = str(y)
        if col in row.columns:
            val = row[col].values[0]
            print(f"  {y}: {val:.2f}" if pd.notna(val) else f"  {y}: N/A")
        else:
            print(f"  {y}: column not found")

# ── RUSSIA ─────────────────────────────────────────────────────────────
print("=" * 60)
print("RUSSIA (Russian Federation)")
print("=" * 60)

print("\nLower secondary completion (both sexes, 20-24), WCDE v3:")
get_edu(edu, "Russian Federation", [1990, 2015])

print("\nLife expectancy (World Bank WDI):")
get_wdi(le, "Russian Federation",
        [1988, 1990, 1994, 2000, 2005, 2009, 2015, 2019],
        "life expectancy")

print("\nTFR (World Bank WDI):")
get_wdi(tfr, "Russian Federation",
        [1990, 1995, 2000, 2005, 2010, 2015, 2019],
        "TFR")

# ── SOUTH AFRICA ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SOUTH AFRICA")
print("=" * 60)

print("\nLower secondary completion (both sexes, 20-24), WCDE v3:")
get_edu(edu, "South Africa", [1960, 1980, 1990, 2000, 2010, 2015])

print("\nLife expectancy (World Bank WDI):")
get_wdi(le, "South Africa",
        [1990, 1995, 2000, 2005, 2010, 2015, 2019],
        "life expectancy")

print("\nTFR (World Bank WDI):")
get_wdi(tfr, "South Africa",
        [1990, 1995, 2000, 2005, 2010, 2015, 2019],
        "TFR")
