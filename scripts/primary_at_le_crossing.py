"""
primary_at_le_crossing.py

Distribution of primary completion (both sexes, age 20-24, WCDE v3) at
the year each country's life expectancy first exceeds 69.8 (1960 US).

Companion to primary_at_tfr_crossing.py. Same exclusion policy:
USSR successor states (reported education is suspect) and left-censored
countries (already above threshold in 1960). Europe excluded to test
that the floor holds outside the post-war European accelerants.
"""
import json
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import (
    PROC, CHECKIN, LE_THRESHOLD,
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

# Mass-mortality / Khmer Rouge disruption — LE/TFR movements here are
# artifacts of the shock, not an education-led phenotype shift.
DISRUPTION = {"cambodia"}

EUROPE = {
    "portugal", "spain", "italy", "greece", "malta", "cyprus",
    "france", "germany", "united kingdom", "ireland",
    "netherlands", "belgium", "luxembourg",
    "austria", "switzerland",
    "denmark", "norway", "sweden", "finland", "iceland",
    "andorra", "monaco", "san marino", "liechtenstein",
}

GP_LAG = 50  # years — two generations, grandparent cohort


def main():
    le = load_wide_indicator("life_expectancy_years.csv")
    prim = pd.read_csv(os.path.join(PROC, "primary_both.csv"),
                       index_col="country")
    prim.columns = prim.columns.astype(int)
    prim.index = [s.lower() for s in prim.index]

    gp = pd.read_csv(os.path.join(PROC, "cohort_primary_both.csv"),
                     index_col="country")
    gp.columns = gp.columns.astype(int)
    gp.index = [s.lower() for s in gp.index]

    lsec = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"),
                       index_col="country")
    lsec.columns = lsec.columns.astype(int)
    lsec.index = [s.lower() for s in lsec.index]

    gp_lsec = pd.read_csv(os.path.join(PROC, "cohort_lower_sec_both.csv"),
                          index_col="country")
    gp_lsec.columns = gp_lsec.columns.astype(int)
    gp_lsec.index = [s.lower() for s in gp_lsec.index]

    # First year LE > 69.8, per country
    first_cross = {}
    for yr in range(START_YEAR, END_YEAR + 1):
        yr_str = str(yr)
        if yr_str not in le.columns:
            continue
        le_y = le[yr_str].dropna()
        crossed = set(le_y[le_y > LE_THRESHOLD].index)
        for c in crossed:
            if c not in first_cross:
                first_cross[c] = yr

    recs = []
    for wdi_lc, cross_y in first_cross.items():
        wcde_lc = WB_TO_WCDE.get(wdi_lc, wdi_lc)
        if wcde_lc in REGIONS or wcde_lc not in prim.index:
            continue
        p_at = completion_at_year(prim, wcde_lc, cross_y)
        if pd.isna(p_at):
            continue
        gp_at = completion_at_year(gp, wcde_lc, cross_y - GP_LAG)
        ls_at = completion_at_year(lsec, wcde_lc, cross_y)
        gp_ls_at = completion_at_year(gp_lsec, wcde_lc, cross_y - GP_LAG)
        recs.append({
            "country": wdi_lc,
            "crossing_year": cross_y,
            "primary_at_cross": p_at,
            "gp_primary": gp_at,
            "lsec_at_cross": ls_at,
            "gp_lsec": gp_ls_at,
            "ussr": wdi_lc in USSR,
            "europe": wdi_lc in EUROPE,
            "disruption": wdi_lc in DISRUPTION,
            "left_censored": cross_y == 1960,
        })

    df = pd.DataFrame(recs)
    clean = df[~df["ussr"] & ~df["europe"]
               & ~df["disruption"] & ~df["left_censored"]].copy()
    clean = clean.sort_values("primary_at_cross")

    print(f"All countries with a primary-at-crossing (LE>69.8) value: "
          f"{len(df)}")
    print(f"  USSR (excluded):                 {df['ussr'].sum()}")
    print(f"  Europe (excluded):               {df['europe'].sum()}")
    print(f"  Khmer Rouge disruption (excl.):  {df['disruption'].sum()}")
    print(f"  left-censored at 1960 (excluded):{df['left_censored'].sum()}")
    print(f"Clean set:                         {len(clean)}")
    print()

    s = clean["primary_at_cross"]
    print("PRIMARY COMPLETION AT YEAR LE>69.8 — clean set")
    print(f"  min    = {s.min():.1f}%  ({clean.iloc[0]['country']}, "
          f"{int(clean.iloc[0]['crossing_year'])})")
    for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90):
        lab = "median" if q == 0.50 else f"p{int(q*100)}"
        print(f"  {lab:6s} = {s.quantile(q):.1f}%")
    print(f"  mean   = {s.mean():.1f}%")
    print(f"  max    = {s.max():.1f}%  ({clean.iloc[-1]['country']}, "
          f"{int(clean.iloc[-1]['crossing_year'])})")
    print()

    print("LOWEST 20 by primary-at-cross (with GP primary, T-50):")
    print(clean.head(20)[["country", "crossing_year",
                          "primary_at_cross", "gp_primary"]]
          .to_string(index=False, float_format=lambda x: f"{x:6.1f}",
                     na_rep="   n/a"))
    print()

    print("HISTOGRAM of primary-at-cross (10-pp bins, clean set):")
    bins = list(range(0, 110, 10))
    hist, edges = np.histogram(s, bins=bins)
    for lo, hi, n in zip(edges[:-1], edges[1:], hist):
        bar = "#" * n
        print(f"  {int(lo):3d}-{int(hi):3d}% : {n:3d} {bar}")
    print()

    gp_s = clean["gp_primary"].dropna()
    print(f"GRANDPARENT PRIMARY (T-{GP_LAG}) — clean set, "
          f"n={len(gp_s)} of {len(clean)}")
    print(f"  min    = {gp_s.min():.1f}%")
    for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90):
        lab = "median" if q == 0.50 else f"p{int(q*100)}"
        print(f"  {lab:6s} = {gp_s.quantile(q):.1f}%")
    print(f"  mean   = {gp_s.mean():.1f}%")
    print(f"  max    = {gp_s.max():.1f}%")
    print()

    print(f"HISTOGRAM of GP primary (T-{GP_LAG}, 10-pp bins):")
    hist, edges = np.histogram(gp_s, bins=bins)
    for lo, hi, n in zip(edges[:-1], edges[1:], hist):
        bar = "#" * n
        print(f"  {int(lo):3d}-{int(hi):3d}% : {n:3d} {bar}")
    print()

    # ── Lower-secondary rows (parent + grandparent) ─────────────────
    ls_s = clean["lsec_at_cross"].dropna()
    gp_ls_s = clean["gp_lsec"].dropna()
    print("LOWER-SECONDARY AT LE CROSSING — clean set")
    print(f"{'Stat':<8}  {'lsec@cross':>10}  {'gp lsec (T-50)':>15}")
    for q_label, q in [("min", None), ("p10", 0.10), ("p25", 0.25),
                        ("median", 0.50), ("p75", 0.75), ("p90", 0.90),
                        ("max", None), ("mean", None)]:
        if q is None and q_label == "min":
            a, b = ls_s.min(), gp_ls_s.min()
        elif q is None and q_label == "max":
            a, b = ls_s.max(), gp_ls_s.max()
        elif q is None and q_label == "mean":
            a, b = ls_s.mean(), gp_ls_s.mean()
        else:
            a, b = ls_s.quantile(q), gp_ls_s.quantile(q)
        print(f"{q_label:<8}  {a:>9.1f}%  {b:>14.1f}%")
    print(f"n={len(ls_s)} lsec, n={len(gp_ls_s)} gp_lsec")
    print()

    print("COMPARISON:")
    print(f"{'Subset':<50}  {'n':>3}  {'min':>6}  {'p25':>6}  "
          f"{'median':>7}  {'p75':>6}")
    for label, sub in [
        ("All crossed (raw)", df),
        ("USSR only", df[df['ussr']]),
        ("Europe only (non-USSR)",
         df[df['europe'] & ~df['ussr']]),
        ("Non-USSR (left-censored excluded)",
         df[~df['ussr'] & ~df['left_censored']]),
        ("Non-European non-USSR (CLEAN)", clean),
    ]:
        if len(sub) == 0:
            continue
        v = sub["primary_at_cross"]
        print(f"{label:<50}  {len(sub):>3}  "
              f"{v.min():>6.1f}  {v.quantile(0.25):>6.1f}  "
              f"{v.median():>7.1f}  {v.quantile(0.75):>6.1f}")

    # ── Write checkin JSON ──────────────────────────────────────────
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
        "primary_at_cross": _quantiles(clean["primary_at_cross"]),
        "lsec_at_cross": _quantiles(clean["lsec_at_cross"].dropna()),
        "gp_primary": _quantiles(clean["gp_primary"].dropna()),
        "gp_lsec": _quantiles(clean["gp_lsec"].dropna()),
    }
    checkin_path = os.path.join(CHECKIN, "primary_at_le_crossing.json")
    with open(checkin_path, "w") as f:
        json.dump(checkin, f, indent=2)
    print(f"\nCheckin written to {checkin_path}")


if __name__ == "__main__":
    main()
