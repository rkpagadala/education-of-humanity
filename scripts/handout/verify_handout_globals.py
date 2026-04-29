"""
handout/verify_handout_globals.py
=================================
Verify the global education and GDP averages flagged "needs verification"
in 2026-04-28_Handout_Documentation (Isabella Reczkowski, Stanford King
Center handout, 5 May 2026).

Numbers under test:
  Global lower-sec completion: 1965=28%, 1990=45%, 2015=65%
  Global GDP per capita:       1965≈$3,000, 1990≈$5,000, 2015≈$10,500

Sources:
  Education: wcde/data/processed/cohort_lower_sec_both.csv (World row)
  GDP:       data/gdppercapita_us_inflation_adjusted.csv (World row;
             NY.GDP.PCAP.KD, constant 2015 USD per data/rebuild_datasets.py)

Also re-checks Bangladesh own-country values (1965 = 14% in handout) against
the same source.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, ".."))

from _shared import REPO_ROOT, CHECKIN, write_checkin  # noqa: E402

YEARS = [1965, 1990, 2015]


def load_wide(path, key_col="country"):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.lower() == key_col.lower()]
    if not cols:
        raise RuntimeError(f"no country column in {path}")
    df.columns = [c if c != cols[0] else "country" for c in df.columns]
    df["country"] = df["country"].str.lower()
    return df.set_index("country")


def lookup(df, key, year):
    key = key.lower()
    if key not in df.index:
        return np.nan
    val = df.loc[key, str(year)]
    try:
        return float(val) if not pd.isna(val) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ── Education (WCDE v3, cohort lower-sec, both sexes) ────────────────
edu_path = os.path.join(REPO_ROOT, "wcde", "data", "processed",
                        "cohort_lower_sec_both.csv")
edu = load_wide(edu_path)

# ── GDP (World Bank WDI, NY.GDP.PCAP.KD) ─────────────────────────────
gdp_path = os.path.join(REPO_ROOT, "data",
                        "gdppercapita_us_inflation_adjusted.csv")
gdp_raw = pd.read_csv(gdp_path)
gdp_raw["country"] = gdp_raw["Country"].str.lower()
gdp = gdp_raw.set_index("country").drop(columns=["Country"])

# ── Handout claims ───────────────────────────────────────────────────
HANDOUT = {
    "global_edu_pct": {1965: 28.0, 1990: 45.0, 2015: 65.0},
    "global_gdp_usd": {1965: 3000.0, 1990: 5000.0, 2015: 10500.0},
    "bangladesh_edu_pct": {1965: 14.0, 1990: 23.0, 2015: 53.0},
    "bangladesh_gdp_usd": {1990: 279.0, 2015: 1086.0},
}

print("=" * 78)
print("HANDOUT GLOBAL VERIFICATION")
print("Source: WCDE v3 cohort_lower_sec_both.csv (World) + WDI NY.GDP.PCAP.KD")
print("=" * 78)

results = {}


def fmt(v, kind):
    if pd.isna(v):
        return "  N/A "
    if kind == "pct":
        return f"{v:6.2f}%"
    return f"${v:>9,.0f}"


def check(label, claim, actual, kind, tol_pct=2.0):
    diff = abs(claim - actual) if not pd.isna(actual) else np.nan
    if pd.isna(diff):
        verdict = "MISSING"
    elif kind == "pct":
        verdict = "OK" if diff <= tol_pct else "MISMATCH"
    else:
        rel = diff / claim if claim else 0
        verdict = "OK" if rel <= 0.05 else "MISMATCH"
    print(f"  {label:<40} handout={fmt(claim, kind)}  "
          f"actual={fmt(actual, kind)}  →  {verdict}")
    return {
        "claim": claim,
        "actual": None if pd.isna(actual) else round(actual, 2),
        "verdict": verdict,
    }


# Global education
print("\nGlobal lower-secondary completion (World aggregate)")
print("-" * 78)
results["global_edu_pct"] = {}
for y in YEARS:
    actual = lookup(edu, "world", y)
    results["global_edu_pct"][y] = check(
        f"Global lower-sec {y}", HANDOUT["global_edu_pct"][y], actual, "pct"
    )

# Global GDP
print("\nGlobal GDP per capita (World, constant 2015 USD per rebuild_datasets.py)")
print("-" * 78)
results["global_gdp_usd"] = {}
for y in YEARS:
    actual = lookup(gdp, "world", y)
    results["global_gdp_usd"][y] = check(
        f"Global GDP {y}", HANDOUT["global_gdp_usd"][y], actual, "gdp"
    )

# Bangladesh edu
print("\nBangladesh own-country lower-secondary completion")
print("-" * 78)
results["bangladesh_edu_pct"] = {}
for y in YEARS:
    actual = lookup(edu, "bangladesh", y)
    results["bangladesh_edu_pct"][y] = check(
        f"Bangladesh edu {y}", HANDOUT["bangladesh_edu_pct"][y], actual, "pct"
    )

# Bangladesh GDP
print("\nBangladesh GDP per capita")
print("-" * 78)
results["bangladesh_gdp_usd"] = {}
for y in [1990, 2015]:
    actual = lookup(gdp, "bangladesh", y)
    results["bangladesh_gdp_usd"][y] = check(
        f"Bangladesh GDP {y}", HANDOUT["bangladesh_gdp_usd"][y], actual, "gdp"
    )

# ── Summary ──────────────────────────────────────────────────────────
all_verdicts = []
for k, v in results.items():
    for y, r in v.items():
        all_verdicts.append((k, y, r["verdict"]))

n_ok = sum(1 for _, _, v in all_verdicts if v == "OK")
n_mis = sum(1 for _, _, v in all_verdicts if v == "MISMATCH")
n_missing = sum(1 for _, _, v in all_verdicts if v == "MISSING")

print("\n" + "=" * 78)
print(f"SUMMARY: {n_ok} OK, {n_mis} MISMATCH, {n_missing} MISSING "
      f"out of {len(all_verdicts)} checks")
print("=" * 78)

if n_mis:
    print("\nMISMATCHES (handout vs source):")
    for k, y, v in all_verdicts:
        if v == "MISMATCH":
            r = results[k][y]
            print(f"  {k} {y}: handout={r['claim']}  source={r['actual']}")

write_checkin(
    "handout_globals_verification.json",
    {
        "method": (
            "Direct CSV lookup. WCDE world row from cohort_lower_sec_both.csv; "
            "WDI world row from gdppercapita_us_inflation_adjusted.csv. "
            "Tolerance: ±2pp for education, ±5% for GDP."
        ),
        "handout": HANDOUT,
        "results": results,
        "summary": {"ok": n_ok, "mismatch": n_mis, "missing": n_missing},
    },
    "scripts/handout/verify_handout_globals.py",
)
