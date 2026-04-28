"""
lsec_threshold_forward_test.py

Forward-lag test of the PT mechanism:
  For each candidate lsec threshold x ∈ {25%, 35%, 50%, 75%}, find
  each country's year T_x when lsec completion (age 20-24) first
  reached x%. Check whether the country had crossed the developmental
  phenotype (TFR < 3.65 AND LE > 69.8) by T_x + 25.

Under the paper's claim ("every country that reaches the educational
threshold crosses within one human generation"), the hit rate at the
true sufficiency threshold should be close to 100%. Below the true
threshold, the hit rate should drop sharply.

Sample: non-European, non-post-socialist countries with both WCDE
and WDI coverage through the relevant window. Post-socialist countries
excluded because their reported lsec fails the phenotype-consistency
test (scripts/which_edu_measure_is_correct.py).
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import (
    PROC, TFR_THRESHOLD, LE_THRESHOLD,
    load_wide_indicator, REGIONS, WB_TO_WCDE, write_checkin,
)

LAG = 25
START_YEAR = 1960
END_YEAR = 2022
OUTCOME_END = 2022

POST_SOCIALIST = {
    # USSR proper (15 republics). Warsaw Pact and Yugoslav satellites
    # had their own national statistical offices and their reported
    # education passes the phenotype-consistency test under
    # Barro-Lee (see scripts/which_edu_measure_is_correct.py).
    # Only the Soviet-Goskomstat set fails that test.
    "russian federation", "ukraine", "belarus", "estonia", "latvia",
    "lithuania", "kazakhstan", "uzbekistan", "turkmenistan",
    "kyrgyz republic", "tajikistan", "azerbaijan", "armenia",
    "georgia", "moldova",
}
EUROPE = {
    "portugal", "spain", "italy", "greece", "malta", "cyprus",
    "france", "germany", "united kingdom", "ireland", "netherlands",
    "belgium", "luxembourg", "austria", "switzerland", "denmark",
    "norway", "sweden", "finland", "iceland", "andorra", "monaco",
    "san marino", "liechtenstein",
}


def first_year_reaches(row, pct):
    """First year in the wide series where value >= pct.
    Interpolates linearly between 5-year bins."""
    years = [int(y) for y in row.index]
    vals = row.values.astype(float)
    # Sort
    order = np.argsort(years)
    years = np.array(years)[order]
    vals = vals[order]
    # Linear scan for first crossing via interpolation
    for i in range(1, len(years)):
        v0, v1 = vals[i - 1], vals[i]
        if np.isnan(v0) or np.isnan(v1):
            continue
        if v0 >= pct:
            return int(years[i - 1])
        if v1 >= pct:
            # Interpolate between (years[i-1], v0) and (years[i], v1)
            frac = (pct - v0) / (v1 - v0) if v1 != v0 else 0
            return int(round(years[i - 1] + frac * (years[i] - years[i - 1])))
    return None


def crossed_by(tfr_row, le_row, year):
    """Did the country cross both thresholds on or before `year`?"""
    if year > OUTCOME_END:
        return None  # unknown — outcome year not yet observable
    for y in range(START_YEAR, year + 1):
        ys = str(y)
        if ys not in tfr_row.index or ys not in le_row.index:
            continue
        tv = tfr_row[ys]
        lv = le_row[ys]
        if pd.notna(tv) and pd.notna(lv) and tv < TFR_THRESHOLD \
                and lv > LE_THRESHOLD:
            return True
    return False


def main():
    tfr = load_wide_indicator("children_per_woman_total_fertility.csv")
    le = load_wide_indicator("life_expectancy_years.csv")
    lsec = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"),
                       index_col="country")
    lsec.columns = lsec.columns.astype(int)
    lsec_lc = lsec.copy()
    lsec_lc.index = [s.lower() for s in lsec_lc.index]

    thresholds = [25, 35, 50, 65, 75, 80, 85, 90, 95]

    # Build base country list: non-European, non-post-socialist,
    # non-aggregate
    countries = []
    for wcde_lc in lsec_lc.index:
        if wcde_lc in [r.lower() for r in REGIONS]:
            continue
        wdi_lc = None
        for k, v in WB_TO_WCDE.items():
            if v == wcde_lc:
                wdi_lc = k
                break
        if wdi_lc is None:
            wdi_lc = wcde_lc
        if wdi_lc in POST_SOCIALIST or wdi_lc in EUROPE:
            continue
        if wdi_lc not in tfr.index or wdi_lc not in le.index:
            continue
        countries.append((wcde_lc, wdi_lc))

    print(f"Test sample: {len(countries)} non-European, "
          f"non-post-socialist countries with WCDE + WDI coverage")
    print()

    print(f"Forward test: lsec ≥ x% at T  ⇒  crossed by T+{LAG}?")
    print()
    print(f"{'Threshold x':<12}  {'n_reached':>10}  {'crossed by T+25':>18}  "
          f"{'n_unknown':>10}  {'hit rate':>10}")

    results = {}  # threshold -> dict
    for x in thresholds:
        n_reached = 0
        n_hit = 0
        n_unknown = 0
        examples_miss = []
        for wcde_lc, wdi_lc in countries:
            row = lsec_lc.loc[wcde_lc]
            T = first_year_reaches(row, x)
            if T is None:
                continue
            n_reached += 1
            outcome_year = T + LAG
            hit = crossed_by(tfr.loc[wdi_lc], le.loc[wdi_lc], outcome_year)
            if hit is None:
                n_unknown += 1
            elif hit:
                n_hit += 1
            else:
                examples_miss.append((wdi_lc, T, outcome_year))
        n_determined = n_reached - n_unknown
        rate = 100 * n_hit / n_determined if n_determined > 0 else 0
        print(f"{x:>3}%         {n_reached:>10}  "
              f"{n_hit:>18}  {n_unknown:>10}  "
              f"{rate:>9.1f}%")
        results[x] = {
            "n_reached": n_reached,
            "n_hit": n_hit,
            "n_unknown": n_unknown,
            "n_determined": n_determined,
            "hit_rate": round(rate, 1),
            "misses": [
                {"country": name, "T": T, "outcome_year": oy}
                for name, T, oy in sorted(examples_miss,
                                          key=lambda r: r[1])[:6]],
        }

        if examples_miss and x in [50, 75, 85]:
            print(f"    misses (lsec reached {x}% at T, "
                  f"not crossed by T+{LAG}):")
            for name, T, oy in sorted(examples_miss,
                                       key=lambda r: r[1])[:15]:
                print(f"      {name:<30}  T={T}  T+25={oy}")

    print()
    print("INTERPRETATION:")
    print("  Hit rate rising with threshold confirms the mechanism is")
    print("  monotonic in educational attainment. A sufficiency")
    print("  threshold exists at the completion level where hit rate")
    print("  approaches 100%.")

    # ── JSON ──────────────────────────────────────────────────────
    numbers = {"sample_n": len(countries)}
    for x, r in results.items():
        numbers[f"thresh_{x}_n_reached"] = r["n_reached"]
        numbers[f"thresh_{x}_n_hit"] = r["n_hit"]
        numbers[f"thresh_{x}_hit_rate"] = r["hit_rate"]
    # Sufficiency summary: where does hit rate cross 94% and 100%?
    sorted_thr = sorted(results.keys())
    numbers["sufficiency_94pct_threshold"] = next(
        (x for x in sorted_thr if results[x]["hit_rate"] >= 94), None)
    numbers["sufficiency_100pct_threshold"] = next(
        (x for x in sorted_thr if results[x]["hit_rate"] >= 100
         and results[x]["n_determined"] > 0), None)

    write_checkin("lsec_thresholds.json", {
        "numbers": numbers,
    }, script_path="scripts/lsec_threshold_forward_test.py")


if __name__ == "__main__":
    main()
