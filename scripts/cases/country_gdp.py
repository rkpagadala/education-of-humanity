# =============================================================================
# PAPER REFERENCE
# Script:  scripts/cases/country_gdp.py
# Paper:   "Education of Nations"
#
# Produces:
#   Verification of all country-specific GDP per capita figures cited in
#   the paper. Loads World Bank constant-2017-USD GDP data, checks each
#   claim, flags mismatches, and writes checkin/country_gdp.json.
#
# Inputs:
#   data/gdppercapita_us_inflation_adjusted.csv
#
# Cross-check for all sections that cite GDP per capita.
# =============================================================================
"""
cases/country_gdp.py

Load World Bank GDP per capita data (constant 2015 USD) and verify every
country-specific GDP figure cited in the paper.
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _shared import load_wb, write_checkin

# ── Load data ────────────────────────────────────────────────────────────────
gdp = load_wb("gdppercapita_us_inflation_adjusted.csv")

def get_gdp(country_lc, year):
    """Get GDP for a country/year using lowercase matching."""
    if country_lc not in gdp.index:
        return None
    try:
        val = float(gdp.loc[country_lc, str(year)])
        return round(val, 0) if not np.isnan(val) else None
    except (KeyError, ValueError):
        return None

# ── Define claims ────────────────────────────────────────────────────────────
# (label, wb_country_lowercase, year, cited_value, tolerance_pct)
# tolerance_pct = fractional tolerance (e.g. 0.15 = 15%)
CLAIMS = [
    ("Korea 1960 GDP", "korea, rep.", 1960, 1038, 0.15),
    ("Costa Rica 1960 GDP", "costa rica", 1960, 3609, 0.15),
    ("Costa Rica 1990 GDP", "costa rica", 1990, 6037, 0.15),
    ("Korea 1990 GDP", "korea, rep.", 1990, 9673, 0.15),
    ("Bangladesh 2014 GDP", "bangladesh", 2014, 1159, 0.15),
    ("Nepal ~1990 GDP", "nepal", 1990, 423, 0.20),
    ("Myanmar 2015 GDP (<$1200)", "myanmar", 2015, 1200, 0.25),
    ("Qatar 2015 GDP (~$69k)", "qatar", 2015, 69000, 0.20),
]

# ── Run checks ───────────────────────────────────────────────────────────────
results = []
print("=" * 80)
print("COUNTRY GDP VERIFICATION (constant 2015 USD)")
print("=" * 80)
print(f"{'Claim':<30} {'Cited':>10} {'Actual':>10} {'Diff%':>8} {'OK?':>5}")
print("-" * 80)

for label, country_lc, year, cited, tol in CLAIMS:
    actual = get_gdp(country_lc, year)
    if actual is None:
        diff_pct = None
        ok = False
        actual_str = "N/A"
        diff_str = "N/A"
    else:
        diff_pct = (actual - cited) / cited
        ok = bool(abs(diff_pct) <= tol)
        actual_str = f"${actual:,.0f}"
        diff_str = f"{diff_pct:+.1%}"

    # Special case: Myanmar check is "below $1,200"
    if "Myanmar" in label and actual is not None:
        ok = bool(actual < 1200)
        diff_str = f"{'<' if actual < 1200 else '>='} $1200"

    status = "PASS" if ok else "FAIL"
    cited_str = f"${cited:,}"
    print(f"{label:<30} {cited_str:>10} {actual_str:>10} {diff_str:>8} {status:>5}")
    results.append({
        "claim": label,
        "cited": cited,
        "actual": actual,
        "diff_pct": round(diff_pct, 4) if diff_pct is not None else None,
        "pass": ok,
    })

# ── Also print nearby years for Nepal ────────────────────────────────────────
print("\nNepal GDP around 1990s:")
for yr in range(1988, 2000):
    v = get_gdp("nepal", yr)
    if v:
        print(f"  {yr}: ${v:,.0f}")

# ── Summary ──────────────────────────────────────────────────────────────────
n_pass = sum(1 for r in results if r["pass"])
n_total = len(results)
print(f"\n{'=' * 80}")
print(f"SUMMARY: {n_pass}/{n_total} GDP claims passed")
print(f"{'=' * 80}")

# ── Write checkin ────────────────────────────────────────────────────────────
write_checkin("country_gdp.json", {
    "source": "data/gdppercapita_us_inflation_adjusted.csv",
    "claims": results,
    "summary": f"{n_pass}/{n_total} GDP claims passed",
}, script_path="scripts/cases/country_gdp.py")
