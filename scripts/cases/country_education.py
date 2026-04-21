# =============================================================================
# PAPER REFERENCE
# Script:  scripts/cases/country_education.py
# Paper:   "Education of Nations"
#
# Produces:
#   Verification of all country-specific education completion percentages
#   cited in the paper. Loads WCDE completion_both_long.csv, checks each
#   claim, flags mismatches, and writes checkin/country_education.json.
#
# Inputs:
#   wcde/data/processed/completion_both_long.csv
#
# Cross-check for all sections that cite country education percentages.
# =============================================================================
"""
cases/country_education.py

Load WCDE lower-secondary completion data and verify every country-specific
education percentage cited in the paper.
"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _shared import PROC, write_checkin

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(PROC, "completion_both_long.csv"))

def get_val(country, year, col="lower_sec"):
    """Get education completion value for a country/year."""
    row = df[(df["country"] == country) & (df["year"] == year)]
    if row.empty:
        return None
    return round(float(row[col].iloc[0]), 2)

# ── Define claims ────────────────────────────────────────────────────────────
# Each claim: (label, country_in_wcde, year, column, claimed_value, tolerance)
CLAIMS = [
    # Korea
    ("Korea 1950 lower_sec", "Republic of Korea", 1950, "lower_sec", 25.0, 3.0),
    ("Korea 1955 lower_sec", "Republic of Korea", 1955, "lower_sec", 25.0, 3.0),
    ("Korea 1985 lower_sec", "Republic of Korea", 1985, "lower_sec", 94.0, 3.0),
    ("Korea 1990 lower_sec", "Republic of Korea", 1990, "lower_sec", 94.0, 3.0),
    # Philippines
    ("Philippines 1950 lower_sec", "Philippines", 1950, "lower_sec", 22.0, 3.0),
    # Cuba
    ("Cuba 1960 lower_sec", "Cuba", 1960, "lower_sec", 40.3, 2.0),
    # Taiwan
    ("Taiwan 1950 lower_sec", "Taiwan Province of China", 1950, "lower_sec", 18.0, 3.0),
    # Cambodia
    ("Cambodia 1975 lower_sec", "Cambodia", 1975, "lower_sec", 10.1, 2.0),
    ("Cambodia 1980 lower_sec", "Cambodia", 1980, "lower_sec", 9.4, 2.0),
    ("Cambodia 1985 lower_sec", "Cambodia", 1985, "lower_sec", 9.5, 2.0),
    ("Cambodia 1995 lower_sec", "Cambodia", 1995, "lower_sec", 35.1, 3.0),
    # Vietnam
    ("Vietnam 1960 lower_sec", "Viet Nam", 1960, "lower_sec", 20.0, 2.0),
    ("Vietnam 2015 lower_sec", "Viet Nam", 2015, "lower_sec", 80.8, 3.0),
    # Bangladesh
    ("Bangladesh 1960 lower_sec", "Bangladesh", 1960, "lower_sec", 11.4, 2.0),
    # Nepal
    ("Nepal 1990 lower_sec ~27%", "Nepal", 1990, "lower_sec", 27.0, 3.0),
    # China
    ("China 1965 lower_sec", "China", 1965, "lower_sec", 30.9, 3.0),
    ("China 1980 lower_sec", "China", 1980, "lower_sec", 62.0, 3.0),
    ("China 1990 lower_sec", "China", 1990, "lower_sec", 75.0, 3.0),
    # Singapore
    ("Singapore 1950 lower_sec", "Singapore", 1950, "lower_sec", 13.0, 3.0),
    ("Singapore 1995 lower_sec", "Singapore", 1995, "lower_sec", 94.0, 3.0),
]

# ── Run checks ───────────────────────────────────────────────────────────────
results = []
print("=" * 80)
print("COUNTRY EDUCATION VERIFICATION")
print("=" * 80)
print(f"{'Claim':<40} {'Cited':>8} {'Actual':>8} {'Diff':>8} {'OK?':>5}")
print("-" * 80)

for label, country, year, col, cited, tol in CLAIMS:
    actual = get_val(country, year, col)
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
    print(f"{label:<40} {cited:>8.1f} {actual_str:>8} {diff_str:>8} {status:>5}")
    results.append({
        "claim": label,
        "cited": cited,
        "actual": actual,
        "diff": diff,
        "pass": ok,
    })

# ── Nepal special check (~27%) ──────────────────────────────────────────────
nepal_val = get_val("Nepal", 1990, "lower_sec")
print(f"\nNepal 1990 lower_sec = {nepal_val:.1f}%  (paper says 'just 27%': {'PASS' if 24 <= nepal_val <= 30 else 'FAIL'})")

# ── Korea expansion rate per 5 years ────────────────────────────────────────
print("\n" + "=" * 80)
print("KOREA: EXPANSION RATE PER 5-YEAR PERIOD")
print("=" * 80)
korea = df[df["country"] == "Republic of Korea"].sort_values("year")
korea_vals = korea[["year", "lower_sec"]].set_index("year")["lower_sec"]
print(f"{'Period':<20} {'Start':>8} {'End':>8} {'Gain (pp)':>10} {'pp/yr':>8}")
print("-" * 60)
korea_expansion = []
for y in range(1950, 1995, 5):
    if y in korea_vals.index and (y + 5) in korea_vals.index:
        start = korea_vals[y]
        end = korea_vals[y + 5]
        gain = end - start
        rate = gain / 5
        in_range = 10 <= gain <= 14
        print(f"{y}-{y+5:<15} {start:>8.1f} {end:>8.1f} {gain:>10.1f} {rate:>8.2f}  {'10-14pp' if in_range else ''}")
        korea_expansion.append({
            "period": f"{y}-{y+5}",
            "start": round(start, 1),
            "end": round(end, 1),
            "gain_pp": round(gain, 1),
            "pp_per_yr": round(rate, 2),
        })

# ── Expansion rates: Korea, Singapore, India, Bangladesh ─────────────────────
print("\n" + "=" * 80)
print("EXPANSION RATES (pp/yr)")
print("=" * 80)

# (label, wcde_name, start_year, end_year, cited_rate)
RATE_CLAIMS = [
    ("Korea", "Republic of Korea", 1955, 1985, 2.13),
    ("Singapore", "Singapore", 1950, 1995, 1.80),
    ("India", "India", 1950, 2020, 0.87),
    ("Bangladesh", "Bangladesh", 1990, 2020, 1.30),
]

rate_results = []
print(f"{'Country':<20} {'Start Yr':>8} {'Start %':>8} {'End Yr':>8} {'End %':>8} {'pp/yr':>8} {'Cited':>8} {'OK?':>5}")
print("-" * 90)
for label, country, yr0, yr1, cited_rate in RATE_CLAIMS:
    cdata = df[df["country"] == country].sort_values("year")
    cvals = cdata.set_index("year")["lower_sec"]
    if yr0 not in cvals.index or yr1 not in cvals.index:
        print(f"{label:<20} MISSING DATA for {yr0} or {yr1}")
        continue
    v0 = cvals[yr0]
    v1 = cvals[yr1]
    yrs = yr1 - yr0
    rate = (v1 - v0) / yrs
    diff = abs(rate - cited_rate)
    ok = bool(diff < 0.3)
    status = "PASS" if ok else "FAIL"
    print(f"{label:<20} {yr0:>8} {v0:>8.1f} {yr1:>8} {v1:>8.1f} {rate:>8.2f} {cited_rate:>8.2f} {status:>5}")
    rate_results.append({
        "country": label,
        "start_year": yr0,
        "end_year": yr1,
        "actual_rate": round(rate, 2),
        "cited_rate": cited_rate,
        "pass": ok,
    })

# ── Summary ──────────────────────────────────────────────────────────────────
n_pass = sum(1 for r in results if r["pass"])
n_total = len(results)
print(f"\n{'=' * 80}")
print(f"SUMMARY: {n_pass}/{n_total} point claims passed")
print(f"{'=' * 80}")

# ── Write checkin ────────────────────────────────────────────────────────────
write_checkin("country_education.json", {
    "source": "wcde/data/processed/completion_both_long.csv",
    "point_claims": results,
    "korea_expansion": korea_expansion,
    "expansion_rates": rate_results,
    "summary": f"{n_pass}/{n_total} point claims passed",
}, script_path="scripts/cases/country_education.py")
