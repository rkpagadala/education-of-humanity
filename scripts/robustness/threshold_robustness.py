"""
robustness/threshold_robustness.py

Table A4: Development threshold robustness — crossing dates under three
specifications (World Bank WDI data; Taiwan from WCDE).

Specs:
  Loose:   TFR < 4.5,  LE > 65
  Main:    TFR < 3.65, LE > 69.8  (1960 USA)
  Strict:  TFR < 2.5,  LE > 72.6  (USA 1975)

For each country, finds the first year when BOTH thresholds are met.
Shift = strict year − loose year.

Usage:
    python scripts/robustness/threshold_robustness.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _shared import DATA, PROC, write_checkin

# ── Three threshold specifications ────────────────────────────────────

SPECS = {
    "loose":  {"tfr": 4.5,  "le": 65.0},
    "main":   {"tfr": 3.65, "le": 69.8},
    "strict": {"tfr": 2.5,  "le": 72.6},
}

# ── Countries in Table A4 ─────────────────────────────────────────────
# (common name, WDI name, WCDE name)

TABLE_A4_COUNTRIES = [
    ("Cuba",        "Cuba",         "Cuba"),
    ("South Korea", "Korea, Rep.",  "Republic of Korea"),
    ("Sri Lanka",   "Sri Lanka",    "Sri Lanka"),
    ("China",       "China",        "China"),
    ("Bangladesh",  "Bangladesh",   "Bangladesh"),
    ("Taiwan",      None,           "Taiwan Province of China"),
]


def load_wdi_wide(path):
    df = pd.read_csv(path).set_index("Country")
    df.columns = [int(c) if c.isdigit() else c for c in df.columns]
    year_cols = [c for c in df.columns if isinstance(c, int)]
    return df[year_cols].apply(pd.to_numeric, errors="coerce")


def load_wcde_wide(path):
    df = pd.read_csv(path).set_index("country")
    df.columns = [int(c) if c.isdigit() else c for c in df.columns]
    year_cols = [c for c in df.columns if isinstance(c, int)]
    return df[year_cols].apply(pd.to_numeric, errors="coerce")


def first_below(series, threshold):
    mask = series.dropna() < threshold
    return int(mask.idxmax()) if mask.any() else None


def first_above(series, threshold):
    mask = series.dropna() > threshold
    return int(mask.idxmax()) if mask.any() else None


def crossing_year(wdi_tfr, wdi_le, wcde_tfr, wcde_e0,
                  wdi_name, wcde_name, tfr_thresh, le_thresh):
    """Find first year BOTH thresholds are met. Prefer WDI (annual)."""
    tfr_yr = None
    le_yr = None

    if wdi_name and wdi_name in wdi_tfr.index:
        tfr_yr = first_below(wdi_tfr.loc[wdi_name], tfr_thresh)
    if tfr_yr is None and wcde_name and wcde_name in wcde_tfr.index:
        tfr_yr = first_below(wcde_tfr.loc[wcde_name], tfr_thresh)

    if wdi_name and wdi_name in wdi_le.index:
        le_yr = first_above(wdi_le.loc[wdi_name], le_thresh)
    if le_yr is None and wcde_name and wcde_name in wcde_e0.index:
        le_yr = first_above(wcde_e0.loc[wcde_name], le_thresh)

    if tfr_yr is not None and le_yr is not None:
        return max(tfr_yr, le_yr)
    return None


def main():
    wdi_le = load_wdi_wide(os.path.join(DATA, "life_expectancy_years.csv"))
    wdi_tfr = load_wdi_wide(os.path.join(DATA, "children_per_woman_total_fertility.csv"))
    wcde_e0 = load_wcde_wide(os.path.join(PROC, "e0.csv"))
    wcde_tfr = load_wcde_wide(os.path.join(PROC, "tfr.csv"))

    print("=" * 78)
    print("TABLE A4: THRESHOLD ROBUSTNESS — CROSSING DATES UNDER THREE SPECS")
    print("=" * 78)
    print(f"  Loose:  TFR < {SPECS['loose']['tfr']},  LE > {SPECS['loose']['le']}")
    print(f"  Main:   TFR < {SPECS['main']['tfr']}, LE > {SPECS['main']['le']}")
    print(f"  Strict: TFR < {SPECS['strict']['tfr']},  LE > {SPECS['strict']['le']}")

    results = {}

    print(f"\n  {'Country':<15} {'Loose':>6} {'Main':>6} {'Strict':>8} {'Shift':>6}")
    print(f"  {'-'*45}")

    for common, wdi_name, wcde_name in TABLE_A4_COUNTRIES:
        dates = {}
        for spec_name, thresholds in SPECS.items():
            yr = crossing_year(wdi_tfr, wdi_le, wcde_tfr, wcde_e0,
                               wdi_name, wcde_name,
                               thresholds["tfr"], thresholds["le"])
            dates[spec_name] = yr

        loose = dates["loose"]
        strict = dates["strict"]
        shift = (strict - loose) if (strict and loose) else None
        shift_str = f"{shift} yrs" if shift is not None else "---"
        strict_str = str(strict) if strict else "not yet"

        print(f"  {common:<15} {loose or 'n/a':>6} {dates['main'] or 'n/a':>6} "
              f"{strict_str:>8} {shift_str:>6}")

        results[common] = {
            "loose": loose,
            "main": dates["main"],
            "strict": strict,
            "shift": shift,
        }

    # ── Expansion rates (pp/yr) for footnote ──────────────────────────

    print(f"\n  Expansion rates (education pp/yr):")
    edu = load_wcde_wide(os.path.join(PROC, "lower_sec_both.csv"))

    rates = {}
    for common, _, wcde_name in TABLE_A4_COUNTRIES:
        if wcde_name not in edu.index:
            continue
        s = edu.loc[wcde_name].dropna().sort_index()
        years = sorted(s.index)
        if len(years) < 2:
            continue
        start_yr = years[0]
        end_yr = years[-1]
        rate = (s[end_yr] - s[start_yr]) / (end_yr - start_yr)
        rates[common] = round(rate, 2)
        print(f"    {common:<15} {s[start_yr]:5.1f}% ({start_yr}) → "
              f"{s[end_yr]:5.1f}% ({end_yr}) = {rate:.2f} pp/yr")

    # ── Write checkin ─────────────────────────────────────────────────

    write_checkin("threshold_robustness.json", {
        "specs": SPECS,
        "results": results,
        "expansion_rates_ppyr": rates,
    }, script_path="scripts/robustness/threshold_robustness.py")


if __name__ == "__main__":
    main()
