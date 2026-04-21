"""
cases/threshold_crossings.py

Verify Table 4 development crossing dates.
For each country, find when BOTH thresholds were crossed:
  TFR < 3.65  AND  LE > 69.8

Expected crossing years:
  Taiwan ~1970, South Korea ~1987, Cuba ~1974,
  Bangladesh ~2014, Sri Lanka ~1993, China ~1994,
  Kerala ~1982 (may not be in WB data), Uganda: check current

Usage:
    python scripts/cases/threshold_crossings.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _shared import DATA, PROC, write_checkin

TFR_THRESHOLD = 3.65
LE_THRESHOLD = 69.8

# ── WDI data (annual, 1960–) ────────────────────────────────────────────
# Country column is title-case with WDI conventions
WDI_LE_PATH = os.path.join(DATA, "life_expectancy_years.csv")
WDI_TFR_PATH = os.path.join(DATA, "children_per_woman_total_fertility.csv")

# ── WCDE data (5-year bins, 1950–) ──────────────────────────────────────
WCDE_E0_PATH = os.path.join(PROC, "e0.csv")
WCDE_TFR_PATH = os.path.join(PROC, "tfr.csv")

# Country name mapping: common name -> (WDI name, WCDE name)
COUNTRIES = {
    "Taiwan":      (None,              "Taiwan Province of China"),
    "South Korea": ("Korea, Rep.",     "Republic of Korea"),
    "Cuba":        ("Cuba",            "Cuba"),
    "Bangladesh":  ("Bangladesh",      "Bangladesh"),
    "Sri Lanka":   ("Sri Lanka",       "Sri Lanka"),
    "China":       ("China",           "China"),
    "Kerala":      (None,              None),   # sub-national, may not exist
    "Uganda":      ("Uganda",          "Uganda"),
}

EXPECTED_BOTH = {
    "Taiwan":      1970,
    "South Korea": 1987,
    "Cuba":        1974,
    "Bangladesh":  2014,
    "Sri Lanka":   1993,
    "China":       1994,
    "Kerala":      1982,
    "Uganda":      None,  # not expected to have crossed yet
}


def load_wdi_wide(path):
    """Load WDI CSV with Country as index, year columns as int."""
    df = pd.read_csv(path)
    df = df.set_index("Country")
    df.columns = [int(c) if c.isdigit() else c for c in df.columns]
    year_cols = [c for c in df.columns if isinstance(c, int)]
    return df[year_cols].apply(pd.to_numeric, errors="coerce")


def load_wcde_wide(path):
    """Load WCDE wide CSV with country as index, year columns as int."""
    df = pd.read_csv(path)
    df = df.set_index("country")
    df.columns = [int(c) if c.isdigit() else c for c in df.columns]
    year_cols = [c for c in df.columns if isinstance(c, int)]
    return df[year_cols].apply(pd.to_numeric, errors="coerce")


def first_crossing_below(series, threshold):
    """Return the first year where value drops below threshold."""
    mask = series.dropna() < threshold
    if mask.any():
        return int(mask.idxmax())
    return None


def first_crossing_above(series, threshold):
    """Return the first year where value rises above threshold."""
    mask = series.dropna() > threshold
    if mask.any():
        return int(mask.idxmax())
    return None


def main():
    # Load all four data sources
    wdi_le = load_wdi_wide(WDI_LE_PATH)
    wdi_tfr = load_wdi_wide(WDI_TFR_PATH)
    wcde_e0 = load_wcde_wide(WCDE_E0_PATH)
    wcde_tfr = load_wcde_wide(WCDE_TFR_PATH)

    results = {}
    all_pass = True

    print("=" * 78)
    print("TABLE 4: DEVELOPMENT CROSSING DATES")
    print(f"Thresholds: TFR < {TFR_THRESHOLD}, LE > {LE_THRESHOLD}")
    print("=" * 78)

    for common_name, (wdi_name, wcde_name) in COUNTRIES.items():
        print(f"\n--- {common_name} ---")
        entry = {"expected_both": EXPECTED_BOTH.get(common_name)}

        # Collect TFR crossing year from best available source
        tfr_year_wdi = None
        tfr_year_wcde = None

        if wdi_name and wdi_name in wdi_tfr.index:
            tfr_year_wdi = first_crossing_below(wdi_tfr.loc[wdi_name], TFR_THRESHOLD)
            print(f"  WDI TFR < {TFR_THRESHOLD}: {tfr_year_wdi}")
        else:
            print(f"  WDI TFR: not available ({wdi_name})")

        if wcde_name and wcde_name in wcde_tfr.index:
            tfr_year_wcde = first_crossing_below(wcde_tfr.loc[wcde_name], TFR_THRESHOLD)
            print(f"  WCDE TFR < {TFR_THRESHOLD}: {tfr_year_wcde}")
        else:
            print(f"  WCDE TFR: not available ({wcde_name})")

        # Best TFR year: prefer WDI (annual resolution), fall back to WCDE
        tfr_year = tfr_year_wdi if tfr_year_wdi is not None else tfr_year_wcde

        # Collect LE crossing year from best available source
        le_year_wdi = None
        le_year_wcde = None

        if wdi_name and wdi_name in wdi_le.index:
            le_year_wdi = first_crossing_above(wdi_le.loc[wdi_name], LE_THRESHOLD)
            print(f"  WDI LE > {LE_THRESHOLD}: {le_year_wdi}")
        else:
            print(f"  WDI LE: not available ({wdi_name})")

        if wcde_name and wcde_name in wcde_e0.index:
            le_year_wcde = first_crossing_above(wcde_e0.loc[wcde_name], LE_THRESHOLD)
            print(f"  WCDE LE > {LE_THRESHOLD}: {le_year_wcde}")
        else:
            print(f"  WCDE LE: not available ({wcde_name})")

        # Best LE year: prefer WDI (annual), fall back to WCDE
        le_year = le_year_wdi if le_year_wdi is not None else le_year_wcde

        # Both-crossed year = max of the two (later threshold to be met)
        if tfr_year is not None and le_year is not None:
            both_year = max(tfr_year, le_year)
        else:
            both_year = None

        print(f"  TFR crossing: {tfr_year}, LE crossing: {le_year}")
        print(f"  BOTH crossed: {both_year}")

        expected = EXPECTED_BOTH.get(common_name)
        if expected is not None and both_year is not None:
            diff = abs(both_year - expected)
            status = "PASS" if diff <= 2 else "MISMATCH"
            if status == "MISMATCH":
                all_pass = False
            print(f"  Expected: {expected}, Diff: {diff} -> {status}")
        elif expected is None and common_name == "Uganda":
            # Uganda: just report current values
            latest_tfr = None
            latest_le = None
            if wdi_name and wdi_name in wdi_tfr.index:
                s = wdi_tfr.loc[wdi_name].dropna()
                if len(s) > 0:
                    latest_tfr = float(s.iloc[-1])
                    print(f"  Uganda latest TFR: {latest_tfr:.2f} (year {int(s.index[-1])})")
            if wdi_name and wdi_name in wdi_le.index:
                s = wdi_le.loc[wdi_name].dropna()
                if len(s) > 0:
                    latest_le = float(s.iloc[-1])
                    print(f"  Uganda latest LE: {latest_le:.1f} (year {int(s.index[-1])})")
            entry["latest_tfr"] = latest_tfr
            entry["latest_le"] = latest_le
            status = "INFO"
        elif expected is not None and both_year is None:
            print(f"  Expected: {expected}, but could not compute -> NOTE")
            status = "NOTE"
        else:
            status = "NOTE"

        entry.update({
            "tfr_crossing_wdi": tfr_year_wdi,
            "tfr_crossing_wcde": tfr_year_wcde,
            "le_crossing_wdi": le_year_wdi,
            "le_crossing_wcde": le_year_wcde,
            "tfr_crossing_best": tfr_year,
            "le_crossing_best": le_year,
            "both_crossed": both_year,
            "status": status,
        })
        results[common_name] = entry

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, r in results.items():
        exp = r["expected_both"]
        got = r["both_crossed"]
        print(f"  {name:15s}  expected={exp}  got={got}  [{r['status']}]")

    overall = "PASS" if all_pass else "SOME_MISMATCHES"
    print(f"\nOverall: {overall}")

    # ── Write checkin JSON ───────────────────────────────────────────────
    write_checkin("table4_crossings.json", {
        "thresholds": {"TFR": TFR_THRESHOLD, "LE": LE_THRESHOLD},
        "results": results,
        "overall": overall,
    }, script_path="scripts/cases/threshold_crossings.py")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
