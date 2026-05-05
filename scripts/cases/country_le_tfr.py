# =============================================================================
# PAPER REFERENCE
# Script:  scripts/cases/country_le_tfr.py
# Paper:   "Education of Nations"
#
# Produces:
#   Verification of all country-specific life expectancy (LE) and total
#   fertility rate (TFR) values cited in the paper. Loads World Bank
#   annual data and WCDE 5-year data, checks each claim, flags
#   mismatches, and writes checkin/country_le_tfr.json.
#
# Inputs:
#   data/life_expectancy_years.csv          (World Bank, annual)
#   data/children_per_woman_total_fertility.csv  (World Bank, annual)
#   wcde/data/processed/e0.csv              (WCDE, 5-year)
#   wcde/data/processed/tfr.csv             (WCDE, 5-year)
#
# Cross-check for all sections that cite LE or TFR.
# =============================================================================
"""
cases/country_le_tfr.py

Load World Bank and WCDE life expectancy / TFR data and verify every
country-specific LE and TFR value cited in the paper.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _shared import DATA, PROC, load_wb, write_checkin


# ── Load World Bank data (annual, long format) ──────────────────────────────
def _wb_to_long(filename):
    wide = load_wb(filename)
    wide = wide.reset_index().rename(columns={"Country": "country_lc"})
    long = wide.melt(id_vars="country_lc", var_name="year", value_name="value")
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["Country"] = long["country_lc"]
    return long

le_wb = _wb_to_long("life_expectancy_years.csv")
tfr_wb = _wb_to_long("children_per_woman_total_fertility.csv")

# ── Load WCDE data (5-year) ─────────────────────────────────────────────────
def load_wcde_wide(filename):
    """Load a WCDE wide CSV: country, then 5-year columns."""
    raw = pd.read_csv(os.path.join(PROC, filename))
    raw.rename(columns={raw.columns[0]: "country"}, inplace=True)
    year_cols = [c for c in raw.columns if c != "country"]
    long = raw.melt(id_vars="country", var_name="year", value_name="value")
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long

le_wcde = load_wcde_wide("e0.csv")
tfr_wcde = load_wcde_wide("tfr.csv")

# NAME_MAP imported from _shared (WCDE → WB lowercase mapping)

def get_wb(df, country_lc, year):
    """Get WB value, trying exact match on lowercase country name."""
    row = df[(df["country_lc"] == country_lc) & (df["year"] == year)]
    if row.empty or pd.isna(row["value"].iloc[0]):
        return None
    return round(float(row["value"].iloc[0]), 1)

def get_wcde(df, country, year):
    """Get WCDE value by country name and year."""
    row = df[(df["country"] == country) & (df["year"] == year)]
    if row.empty or pd.isna(row["value"].iloc[0]):
        return None
    return round(float(row["value"].iloc[0]), 1)

# ── Define claims ────────────────────────────────────────────────────────────
# (label, source, dataset_key, country_key, year, cited_value, tolerance)
# source: "wb_le", "wb_tfr", "wcde_le", "wcde_tfr"
CLAIMS = [
    # USA 1960
    ("USA 1960 TFR=3.65",  "wb_tfr", "united states", 1960, 3.65, 0.15),
    ("USA 1960 LE=69.8",   "wb_le",  "united states", 1960, 69.8, 1.0),
    # Japan 1960
    ("Japan 1960 TFR=2.0", "wb_tfr", "japan", 1960, 2.0, 0.15),
    ("Japan 1960 LE=67.7", "wb_le",  "japan", 1960, 67.7, 1.0),
    # Sri Lanka LE
    ("Sri Lanka 1988 LE=69.0", "wb_le", "sri lanka", 1988, 69.0, 1.0),
    ("Sri Lanka 1989 LE=67.3", "wb_le", "sri lanka", 1989, 67.3, 1.5),
    ("Sri Lanka 1993 LE=69.8", "wb_le", "sri lanka", 1993, 69.8, 1.0),
    # Myanmar
    ("Myanmar 1960 TFR=5.9", "wb_tfr", "myanmar", 1960, 5.9, 0.2),
    ("Myanmar 2015 TFR=2.3", "wb_tfr", "myanmar", 2015, 2.3, 0.2),
    ("Myanmar 1960 LE=44.1", "wb_le",  "myanmar", 1960, 44.1, 1.5),
    ("Myanmar 2015 LE=65.3", "wb_le",  "myanmar", 2015, 65.3, 1.5),
    # Uganda / India 1960
    ("Uganda 1960 LE=45.6", "wb_le", "uganda", 1960, 45.6, 1.5),
    ("India 1960 LE=45.6",  "wb_le", "india",  1960, 45.6, 3.0),
    ("Uganda 1980 LE=43.5", "wb_le", "uganda", 1980, 43.5, 1.5),
    # Cuba
    ("Cuba 1960 LE=63.3",  "wb_le", "cuba", 1960, 63.3, 1.5),
    ("Cuba 1974 LE=69.9",  "wb_le", "cuba", 1974, 69.9, 1.5),
    # Korea
    ("Korea 1965 LE=55.9", "wb_le", "korea, rep.", 1965, 55.9, 1.5),
    # China ~1960
    ("China ~1960 LE~53",  "wb_le", "china", 1960, 53.0, 5.0),
    # Bangladesh 2014
    ("Bangladesh 2014 LE=70.0", "wb_le",  "bangladesh", 2014, 70.0, 1.5),
    ("Bangladesh 2014 TFR=2.23","wb_tfr", "bangladesh", 2014, 2.23, 0.15),
    # Uganda latest (Table 4) — check 2020 or 2022
    ("Uganda Table4 TFR=4.39", "wb_tfr", "uganda", 2022, 4.39, 0.2),
    ("Uganda Table4 LE=67.7",  "wb_le",  "uganda", 2022, 67.7, 1.0),
]

# ── Run checks ───────────────────────────────────────────────────────────────
results = []
print("=" * 80)
print("COUNTRY LE / TFR VERIFICATION")
print("=" * 80)
print(f"{'Claim':<35} {'Cited':>8} {'Actual':>8} {'Diff':>8} {'OK?':>5}")
print("-" * 80)

for label, source, country_key, year, cited, tol in CLAIMS:
    if source == "wb_le":
        actual = get_wb(le_wb, country_key, year)
    elif source == "wb_tfr":
        actual = get_wb(tfr_wb, country_key, year)
    elif source == "wcde_le":
        actual = get_wcde(le_wcde, country_key, year)
    elif source == "wcde_tfr":
        actual = get_wcde(tfr_wcde, country_key, year)
    else:
        actual = None

    if actual is None:
        diff = None
        ok = False
        actual_str = "N/A"
        diff_str = "N/A"
    else:
        diff = round(actual - cited, 2)
        ok = bool(abs(diff) <= tol)
        actual_str = f"{actual:.1f}"
        diff_str = f"{diff:+.1f}"

    status = "PASS" if ok else "FAIL"
    print(f"{label:<35} {cited:>8.2f} {actual_str:>8} {diff_str:>8} {status:>5}")
    results.append({
        "claim": label,
        "cited": cited,
        "actual": actual,
        "diff": diff,
        "pass": ok,
    })

# ── Sri Lanka crossing check ────────────────────────────────────────────────
print("\nSri Lanka LE around 1988-1995 (crossed 69.8?):")
for yr in range(1986, 1996):
    v = get_wb(le_wb, "sri lanka", yr)
    if v:
        marker = " <-- crossed 69.8" if yr >= 1993 and v >= 69.8 else ""
        print(f"  {yr}: {v:.1f}{marker}")

# ── Uganda LE/TFR recent years ───────────────────────────────────────────────
print("\nUganda recent LE and TFR (for Table 4):")
for yr in [2018, 2019, 2020, 2021, 2022]:
    le_v = get_wb(le_wb, "uganda", yr)
    tfr_v = get_wb(tfr_wb, "uganda", yr)
    le_str = f"{le_v:.1f}" if le_v else "N/A"
    tfr_str = f"{tfr_v:.2f}" if tfr_v else "N/A"
    print(f"  {yr}: LE={le_str}, TFR={tfr_str}")

# ── Summary ──────────────────────────────────────────────────────────────────
n_pass = sum(1 for r in results if r["pass"])
n_total = len(results)
print(f"\n{'=' * 80}")
print(f"SUMMARY: {n_pass}/{n_total} LE/TFR claims passed")
print(f"{'=' * 80}")

# ── Write checkin ────────────────────────────────────────────────────────────
write_checkin("country_le_tfr.json", {
    "sources": [
        "data/life_expectancy_years.csv",
        "data/children_per_woman_total_fertility.csv",
        "wcde/data/processed/e0.csv",
        "wcde/data/processed/tfr.csv",
    ],
    "claims": results,
    "summary": f"{n_pass}/{n_total} LE/TFR claims passed",
}, script_path="scripts/cases/country_le_tfr.py")
