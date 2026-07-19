"""
Entry-cohort decadal expansion rates for the institutional-challenge test.

Question: did the global mean rate of lower-secondary expansion rise as
institutions matured? The institutional-challenge claim is no.

Method: for each decade D in {1950, 1960, ..., 2010}:
  - Restrict to countries whose lower-secondary completion lies in the
    active-transition window [10, 90] at the decade start (entry-cohort).
  - For each such country, compute pp/yr = (level(D+10) - level(D)) / 10.
  - Take the mean across qualifying countries.

Source: WCDE v3, lower secondary completion, both sexes, age 20-24.
Output: checkin/institution_test_entry_cohort_rates.json
"""

import json
from pathlib import Path

import pandas as pd

from _shared import REGIONS

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "wcde" / "data" / "processed" / "lower_sec_both.csv"
OUT = ROOT / "checkin" / "institution_test_entry_cohort_rates.json"

WINDOW_LO, WINDOW_HI = 10.0, 90.0
DECADE_STARTS = [1950, 1960, 1970, 1980, 1990, 2000, 2010]


def main() -> None:
    df = pd.read_csv(SRC)
    df = df[~df["country"].isin(REGIONS)].copy()

    decadal = {}
    for d in DECADE_STARTS:
        start_col, end_col = str(d), str(d + 10)
        if end_col not in df.columns:
            continue
        sub = df[[ "country", start_col, end_col ]].dropna()
        sub = sub[(sub[start_col] >= WINDOW_LO) & (sub[start_col] <= WINDOW_HI)]
        rates = (sub[end_col] - sub[start_col]) / 10.0
        decadal[f"{d}-{d+10}"] = {
            "n_countries": int(len(sub)),
            "mean_pp_per_year": round(float(rates.mean()), 3),
            "median_pp_per_year": round(float(rates.median()), 3),
            "sd_pp_per_year": round(float(rates.std()), 3),
        }

    # Also compute the legacy 25-year-era split to confirm reproducibility
    # of the existing paper numbers (1.06 / 0.82 / 0.95).
    legacy = {}
    for d0, d1 in [(1950, 1975), (1975, 2000), (2000, 2015)]:
        start_col, end_col = str(d0), str(d1)
        sub = df[["country", start_col, end_col]].dropna()
        sub = sub[(sub[start_col] >= WINDOW_LO) & (sub[start_col] <= WINDOW_HI)]
        years = d1 - d0
        rates = (sub[end_col] - sub[start_col]) / years
        legacy[f"{d0}-{d1}"] = {
            "n_countries": int(len(sub)),
            "mean_pp_per_year": round(float(rates.mean()), 3),
        }

    result = {
        "source": "WCDE v3 lower-sec both 20-24, entry-cohort: lower-sec at decade start in [10, 90]",
        "decadal_entry_cohort": decadal,
        "legacy_25yr_era_for_comparison": legacy,
    }

    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
