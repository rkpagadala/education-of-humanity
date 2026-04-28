"""
soviet_education_inflation_test.py

Test whether reported Soviet-era education attainment in Central Asia
was inflated relative to the rest of the USSR and to comparable
non-Soviet neighbors with similar 1917 starting conditions.

Three diagnostics (1970 and 1980 snapshots):
  (1) Female/male gap in lower-sec completion.
      Large real gap is historically normal in rural Muslim populations.
      Unusually small gap suggests uniform credential reporting.
  (2) Primary completion vs lower-sec completion.
      Primary must exceed lower-sec (you can't complete the latter
      without the former). A reversal is a direct inflation signal.
  (3) Soviet Central Asia vs contemporaneous non-Soviet neighbors
      (Iran, Turkey, Afghanistan, Pakistan) with similar pre-1917
      educational baseline.

Data source: WCDE v3 (IIASA reconstruction). The reconstruction
relies on Soviet-era census reporting for early years, so if
inflation was in the source, it may still be in the series.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import PROC, write_checkin

CENTRAL_ASIA = [
    "Kazakhstan", "Uzbekistan", "Turkmenistan",
    "Kyrgyzstan", "Tajikistan",
]
CAUCASUS = ["Azerbaijan", "Armenia", "Georgia"]
BALTICS = ["Latvia", "Estonia", "Lithuania"]
SLAVIC_WEST = ["Russian Federation", "Ukraine", "Belarus"]
# Non-Soviet neighbors: similar pre-1917 starting conditions,
# similar religion/geography for some of Central Asia.
NEIGHBORS = ["Iran (Islamic Republic of)", "Turkey", "Afghanistan",
             "Pakistan"]

YEARS = [1960, 1970, 1980, 1990]


def load_wide(name):
    df = pd.read_csv(os.path.join(PROC, f"{name}.csv"), index_col="country")
    df.columns = df.columns.astype(int)
    return df


def fetch(df, country, years):
    if country not in df.index:
        return [None] * len(years)
    return [df.loc[country, y] if y in df.columns else None for y in years]


def main():
    lsec_b = load_wide("lower_sec_both")
    lsec_f = load_wide("lower_sec_female")
    lsec_m = load_wide("lower_sec_male")
    prim_b = load_wide("primary_both")

    groups = [
        ("Central Asia (USSR)", CENTRAL_ASIA),
        ("Caucasus (USSR)", CAUCASUS),
        ("Baltics (USSR)", BALTICS),
        ("Slavic west (USSR)", SLAVIC_WEST),
        ("Non-Soviet neighbors", NEIGHBORS),
    ]

    # ── Diagnostic 1: Female/male gap in lower-sec ───────────────────
    print("=" * 78)
    print("DIAGNOSTIC 1 — Female minus male lower-sec completion (pp)")
    print("=" * 78)
    print(f"{'Country':<38}  {'1960':>6}  {'1970':>6}  {'1980':>6}  "
          f"{'1990':>6}")
    for group_name, countries in groups:
        print(f"\n  [{group_name}]")
        for c in countries:
            fvals = fetch(lsec_f, c, YEARS)
            mvals = fetch(lsec_m, c, YEARS)
            gaps = []
            for f, m in zip(fvals, mvals):
                if f is None or m is None:
                    gaps.append("    --")
                else:
                    gaps.append(f"{f - m:+6.1f}")
            print(f"  {c:<38}  " + "  ".join(gaps))

    # ── Diagnostic 2: Primary minus lower-sec (both sexes) ─────────
    # Primary >= lower-sec should hold by definition. Negative = impossible.
    print()
    print("=" * 78)
    print("DIAGNOSTIC 2 — Primary minus lower-sec, both sexes (pp)")
    print("  Should be >=0 always. Negative = reported lower-sec exceeds")
    print("  primary, which is mathematically impossible.")
    print("=" * 78)
    print(f"{'Country':<38}  {'1960':>6}  {'1970':>6}  {'1980':>6}  "
          f"{'1990':>6}")
    for group_name, countries in groups:
        print(f"\n  [{group_name}]")
        for c in countries:
            pvals = fetch(prim_b, c, YEARS)
            lvals = fetch(lsec_b, c, YEARS)
            deltas = []
            for p, l in zip(pvals, lvals):
                if p is None or l is None:
                    deltas.append("    --")
                else:
                    d = p - l
                    marker = "" if d >= 0 else " *"
                    deltas.append(f"{d:+6.1f}{marker}")
            print(f"  {c:<38}  " + "  ".join(deltas))

    # ── Diagnostic 3: Raw lower-sec levels ─────────────────────────
    print()
    print("=" * 78)
    print("DIAGNOSTIC 3 — Lower-sec completion, both sexes (%)")
    print("=" * 78)
    print(f"{'Country':<38}  {'1960':>6}  {'1970':>6}  {'1980':>6}  "
          f"{'1990':>6}")
    for group_name, countries in groups:
        print(f"\n  [{group_name}]")
        for c in countries:
            vals = fetch(lsec_b, c, YEARS)
            cells = []
            for v in vals:
                cells.append("    --" if v is None else f"{v:6.1f}")
            print(f"  {c:<38}  " + "  ".join(cells))

    # ── JSON emission ──────────────────────────────────────────────
    def _group_mean(df, countries, year):
        vals = [df.loc[c, year] for c in countries
                if c in df.index and year in df.columns
                and not pd.isna(df.loc[c, year])]
        return round(float(np.mean(vals)), 1) if vals else None

    numbers = {}
    # Female-male lsec gap (pp) by subgroup × year
    fm_gap = lsec_f - lsec_m
    for group_name, countries in groups:
        tag = group_name.split(" (")[0].lower().replace(" ", "_")
        for y in YEARS:
            numbers[f"fm_gap_{tag}_{y}"] = _group_mean(fm_gap, countries, y)
    # Primary − lsec dropoff (pp)
    p_minus_l = prim_b - lsec_b
    for group_name, countries in groups:
        tag = group_name.split(" (")[0].lower().replace(" ", "_")
        for y in YEARS:
            numbers[f"prim_minus_lsec_{tag}_{y}"] = _group_mean(
                p_minus_l, countries, y)
    # Raw lsec completion (%)
    for group_name, countries in groups:
        tag = group_name.split(" (")[0].lower().replace(" ", "_")
        for y in YEARS:
            numbers[f"lsec_level_{tag}_{y}"] = _group_mean(
                lsec_b, countries, y)
    # Specific country-year values the paper quotes
    for c, label in [("Kazakhstan", "kazakhstan"),
                     ("Turkmenistan", "turkmenistan"),
                     ("Iran (Islamic Republic of)", "iran"),
                     ("Afghanistan", "afghanistan"),
                     ("Pakistan", "pakistan"),
                     ("Turkey", "turkey")]:
        for y in (1970, 1980):
            if c in lsec_b.index and y in lsec_b.columns:
                v = lsec_b.loc[c, y]
                numbers[f"lsec_{label}_{y}"] = (
                    round(float(v), 1) if not pd.isna(v) else None)

    write_checkin("soviet_inflation.json", {
        "numbers": numbers,
    }, script_path="scripts/soviet_education_inflation_test.py")


if __name__ == "__main__":
    main()
