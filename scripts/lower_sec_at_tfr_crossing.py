"""
lower_sec_at_tfr_crossing.py

Distribution of lower-secondary completion (both sexes, age 20-24,
WCDE v3) at the year each country's TFR first falls below 3.65 (the
1960 US value).

Lower-sec analogue of primary_at_tfr_crossing.py. Created 2026-05-26
when the paper switched its TFR predictor from primary to lower-sec
(canonical lag 0 per _shared.py; lower-sec dominates primary in
within-country panel R^2 for TFR per by_level_lag0.json).
"""
import json
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import (
    PROC, CHECKIN, TFR_THRESHOLD, LAG_TWO_GENERATIONS,
    load_wide_indicator, completion_at_year, REGIONS, WB_TO_WCDE,
)

START_YEAR = 1960
END_YEAR = 2022

USSR = {
    "russian federation", "ukraine", "belarus",
    "estonia", "latvia", "lithuania",
    "kazakhstan", "uzbekistan", "turkmenistan",
    "kyrgyz republic", "tajikistan",
    "azerbaijan", "armenia", "georgia",
    "moldova",
}

DISRUPTION = {"cambodia"}

EUROPE = {
    "portugal", "spain", "italy", "greece", "malta", "cyprus",
    "france", "germany", "united kingdom", "ireland",
    "netherlands", "belgium", "luxembourg",
    "austria", "switzerland",
    "denmark", "norway", "sweden", "finland", "iceland",
    "andorra", "monaco", "san marino", "liechtenstein",
}


GP_LAG = LAG_TWO_GENERATIONS  # years — two generations (~28yr each), grandparent cohort


def main():
    tfr = load_wide_indicator("children_per_woman_total_fertility.csv")
    lsec = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"),
                       index_col="country")
    lsec.columns = lsec.columns.astype(int)
    lsec.index = [s.lower() for s in lsec.index]

    # Grandparent (~50yr prior) lower-sec — uses cohort series, which
    # extends back to 1870 (lower_sec_both.csv only covers 1950+).
    gp = pd.read_csv(os.path.join(PROC, "cohort_lower_sec_both.csv"),
                     index_col="country")
    gp.columns = gp.columns.astype(int)
    gp.index = [s.lower() for s in gp.index]

    # First year TFR < 3.65, per country
    first_cross = {}
    for yr in range(START_YEAR, END_YEAR + 1):
        yr_str = str(yr)
        if yr_str not in tfr.columns:
            continue
        tfr_y = tfr[yr_str].dropna()
        crossed = set(tfr_y[tfr_y < TFR_THRESHOLD].index)
        for c in crossed:
            if c not in first_cross:
                first_cross[c] = yr

    recs = []
    for wdi_lc, cross_y in first_cross.items():
        wcde_lc = WB_TO_WCDE.get(wdi_lc, wdi_lc)
        if wcde_lc in REGIONS or wcde_lc not in lsec.index:
            continue
        l_at = completion_at_year(lsec, wcde_lc, cross_y)
        if pd.isna(l_at):
            continue
        gp_at = completion_at_year(gp, wcde_lc, cross_y - GP_LAG)
        recs.append({
            "country": wdi_lc,
            "crossing_year": cross_y,
            "lower_sec_at_cross": l_at,
            "gp_lower_sec": gp_at,
            "ussr": wdi_lc in USSR,
            "europe": wdi_lc in EUROPE,
            "disruption": wdi_lc in DISRUPTION,
            "left_censored": cross_y == 1960,
        })

    df = pd.DataFrame(recs)
    clean = df[~df["ussr"] & ~df["europe"]
               & ~df["disruption"] & ~df["left_censored"]].copy()
    clean = clean.sort_values("lower_sec_at_cross")

    print(f"All countries with a lower-sec-at-crossing (TFR<3.65) value: "
          f"{len(df)}")
    print(f"  USSR (excluded):                 {df['ussr'].sum()}")
    print(f"  Europe (excluded):               {df['europe'].sum()}")
    print(f"  Khmer Rouge disruption (excl.):  {df['disruption'].sum()}")
    print(f"  left-censored at 1960 (excluded):{df['left_censored'].sum()}")
    print(f"Clean set:                         {len(clean)}")
    print()

    s = clean["lower_sec_at_cross"]
    print("LOWER-SEC COMPLETION AT YEAR TFR<3.65 — clean set")
    print(f"  min    = {s.min():.1f}%  ({clean.iloc[0]['country']}, "
          f"{int(clean.iloc[0]['crossing_year'])})")
    for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90):
        lab = "median" if q == 0.50 else f"p{int(q*100)}"
        print(f"  {lab:6s} = {s.quantile(q):.1f}%")
    print(f"  mean   = {s.mean():.1f}%")
    print(f"  max    = {s.max():.1f}%  ({clean.iloc[-1]['country']}, "
          f"{int(clean.iloc[-1]['crossing_year'])})")
    print()

    print("LOWEST 20 by lower-sec-at-cross (with GP lower-sec, T-50):")
    print(clean.head(20)[["country", "crossing_year",
                          "lower_sec_at_cross", "gp_lower_sec"]]
          .to_string(index=False, float_format=lambda x: f"{x:6.1f}",
                     na_rep="   n/a"))
    print()

    print("HISTOGRAM of lower-sec-at-cross (10-pp bins, clean set):")
    bins = list(range(0, 110, 10))
    hist, edges = np.histogram(s, bins=bins)
    for lo, hi, n in zip(edges[:-1], edges[1:], hist):
        bar = "#" * n
        print(f"  {int(lo):3d}-{int(hi):3d}% : {n:3d} {bar}")
    print()

    gp_s = clean["gp_lower_sec"].dropna()
    print(f"GRANDPARENT LOWER-SEC (T-{GP_LAG}) — clean set, "
          f"n={len(gp_s)} of {len(clean)}")
    print(f"  min    = {gp_s.min():.1f}%")
    for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90):
        lab = "median" if q == 0.50 else f"p{int(q*100)}"
        print(f"  {lab:6s} = {gp_s.quantile(q):.1f}%")
    print(f"  mean   = {gp_s.mean():.1f}%")
    print(f"  max    = {gp_s.max():.1f}%")
    print()

    def _quantiles(series):
        return {
            "p10": round(float(series.quantile(0.10)), 1),
            "p25": round(float(series.quantile(0.25)), 1),
            "median": round(float(series.median()), 1),
            "p75": round(float(series.quantile(0.75)), 1),
            "p90": round(float(series.quantile(0.90)), 1),
        }

    checkin = {
        "n_clean": int(len(clean)),
        "lower_sec_at_cross": _quantiles(clean["lower_sec_at_cross"]),
        "gp_lower_sec": _quantiles(clean["gp_lower_sec"].dropna()),
    }
    checkin_path = os.path.join(CHECKIN, "lower_sec_at_tfr_crossing.json")
    with open(checkin_path, "w") as f:
        json.dump(checkin, f, indent=2)
    print(f"\nCheckin written to {checkin_path}")


if __name__ == "__main__":
    main()
