"""
cases/college_le_gradient.py

Verify Section 3.2 college-completion / life-expectancy gradient.

Expected numbers:
  - Among 70 countries with >85% lower-secondary completion in 2010
  - College completion quartiles correlate with LE at r=0.44
  - LE ranges from 73.5 (lowest college quartile) to 79.0 (highest)
  - 5.5-year gradient

Usage:
    python scripts/cases/college_le_gradient.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _shared import PROC, DATA, REGIONS, write_checkin

COMP_PATH = os.path.join(PROC, "completion_both_long.csv")
E0_PATH = os.path.join(PROC, "e0.csv")
WDI_LE_PATH = os.path.join(DATA, "life_expectancy_years.csv")

# Territories / dependencies / sub-national entities to exclude beyond REGIONS
AGGREGATES = REGIONS | {
    "Less developed regions, excluding least developed countries",
    "Less developed regions, excluding China",
    "Aruba", "Curaçao",
    "French Polynesia", "Guadeloupe", "Guam",
    "Hong Kong Special Administrative Region of China",
    "Macao Special Administrative Region of China",
    "Martinique", "Mayotte", "New Caledonia",
    "Occupied Palestinian Territory",
    "Puerto Rico", "Reunion",
    "United States Virgin Islands",
    "Taiwan Province of China",
}


def main():
    # ── Load WCDE completion data for 2010 ───────────────────────────────
    comp = pd.read_csv(COMP_PATH)
    comp_2010 = comp[comp["year"] == 2010][["country", "lower_sec", "college"]].copy()
    comp_2010 = comp_2010[~comp_2010["country"].isin(AGGREGATES)]
    comp_2010 = comp_2010.dropna(subset=["lower_sec", "college"])

    # Filter to countries with >85% lower-secondary completion
    high_ls = comp_2010[comp_2010["lower_sec"] > 85].copy()

    print("=" * 70)
    print("COLLEGE-LE GRADIENT (Section 3.2)")
    print("=" * 70)
    print(f"\nCountries with >85% lower-sec in 2010: {len(high_ls)}")

    # ── Load LE for 2010 ─────────────────────────────────────────────────
    # Try WCDE e0 first (matched country names)
    e0 = pd.read_csv(E0_PATH)
    e0 = e0.set_index("country")
    e0.columns = [int(c) if str(c).isdigit() else c for c in e0.columns]

    # Also load WDI LE for fallback
    wdi_le = pd.read_csv(WDI_LE_PATH).set_index("Country")
    wdi_le.columns = [int(c) if str(c).isdigit() else c for c in wdi_le.columns]

    # Map LE from WCDE e0 (2010 column)
    high_ls["le_wcde"] = high_ls["country"].map(
        lambda c: float(e0.loc[c, 2010]) if c in e0.index and 2010 in e0.columns else np.nan
    )

    # Use WCDE LE as primary
    high_ls["le"] = high_ls["le_wcde"]

    high_ls = high_ls.dropna(subset=["le"])
    print(f"Countries with LE data: {len(high_ls)}")

    if len(high_ls) < 10:
        print("ERROR: too few countries for analysis")
        sys.exit(1)

    # ── College quartiles ────────────────────────────────────────────────
    high_ls["college_quartile"] = pd.qcut(high_ls["college"], 4, labels=["Q1", "Q2", "Q3", "Q4"])

    quartile_means = high_ls.groupby("college_quartile", observed=True)["le"].mean()
    quartile_college = high_ls.groupby("college_quartile", observed=True)["college"].mean()
    quartile_n = high_ls.groupby("college_quartile", observed=True).size()

    print("\n--- College quartiles ---")
    print(f"  {'Quartile':10s} {'N':>4s} {'College%':>10s} {'Mean LE':>10s}")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        if q in quartile_means.index:
            print(f"  {q:10s} {quartile_n[q]:4d} {quartile_college[q]:10.1f} {quartile_means[q]:10.1f}")

    # ── Correlation ──────────────────────────────────────────────────────
    corr = high_ls["college"].corr(high_ls["le"])
    print(f"\n  Pearson r (college vs LE): {corr:.2f}")

    # ── Gradient ─────────────────────────────────────────────────────────
    q1_le = float(quartile_means.get("Q1", np.nan))
    q4_le = float(quartile_means.get("Q4", np.nan))
    gradient = q4_le - q1_le

    print(f"  Q1 mean LE: {q1_le:.1f}")
    print(f"  Q4 mean LE: {q4_le:.1f}")
    print(f"  Gradient (Q4 - Q1): {gradient:.1f} years")

    # ── Check against expected values ────────────────────────────────────
    print("\n--- Verification ---")
    results = {}

    # N countries
    n_countries = len(high_ls)
    n_diff = abs(n_countries - 70)
    n_status = "PASS" if n_diff <= 5 else "CHECK"
    print(f"  N countries: expected=~70, actual={n_countries}, diff={n_diff} -> {n_status}")
    results["n_countries"] = {"expected": 70, "actual": n_countries, "status": n_status}

    # Correlation
    corr_diff = abs(corr - 0.44)
    corr_status = "PASS" if corr_diff <= 0.10 else "CHECK"
    print(f"  Correlation: expected=0.44, actual={corr:.2f}, diff={corr_diff:.2f} -> {corr_status}")
    results["correlation"] = {"expected": 0.44, "actual": round(corr, 3), "status": corr_status}

    # Q1 LE
    q1_diff = abs(q1_le - 73.5)
    q1_status = "PASS" if q1_diff <= 2.0 else "CHECK"
    print(f"  Q1 LE: expected=73.5, actual={q1_le:.1f}, diff={q1_diff:.1f} -> {q1_status}")
    results["q1_le"] = {"expected": 73.5, "actual": round(q1_le, 2), "status": q1_status}

    # Q4 LE
    q4_diff = abs(q4_le - 79.0)
    q4_status = "PASS" if q4_diff <= 2.0 else "CHECK"
    print(f"  Q4 LE: expected=79.0, actual={q4_le:.1f}, diff={q4_diff:.1f} -> {q4_status}")
    results["q4_le"] = {"expected": 79.0, "actual": round(q4_le, 2), "status": q4_status}

    # Gradient
    grad_diff = abs(gradient - 5.5)
    grad_status = "PASS" if grad_diff <= 1.5 else "CHECK"
    print(f"  Gradient: expected=5.5, actual={gradient:.1f}, diff={grad_diff:.1f} -> {grad_status}")
    results["gradient"] = {"expected": 5.5, "actual": round(gradient, 2), "status": grad_status}

    # ── List countries by quartile (for reference) ───────────────────────
    print("\n--- Countries by quartile ---")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        subset = high_ls[high_ls["college_quartile"] == q].sort_values("college")
        countries = [f"{r['country']} ({r['college']:.1f}%)" for _, r in subset.iterrows()]
        print(f"  {q}: {', '.join(countries)}")

    # ── Overall ──────────────────────────────────────────────────────────
    all_pass = all(r["status"] == "PASS" for r in results.values())
    overall = "PASS" if all_pass else "SOME_NOTES"

    print(f"\nOverall: {overall}")

    # ── Write checkin JSON ───────────────────────────────────────────────
    write_checkin("college_le_gradient.json", {
        "results": results,
        "overall": overall,
    }, script_path="scripts/cases/college_le_gradient.py")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
