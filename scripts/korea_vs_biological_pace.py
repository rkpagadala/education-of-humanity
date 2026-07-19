"""
Korea's WCDE-record LS expansion rate vs the biological pace ceiling.

Two clocks are in play and easy to confuse:
  * calendar year T  -- the year of WCDE measurement (20-24 cohort)
  * cohort year      -- the calendar year in which the measured cohort
                        completed lower-secondary (age 15)

WCDE measures LS completion among 20-24 year olds. Mid-age 22 in year T
means the cohort completed LS at age 15 in year T - 7.

Biology: a child entering school at age 6 needs 9 years to complete LS.
So if a state flips delivery in calendar year T_policy, the first cohort
to complete under the new regime is the one born in T_policy - 6, which
completes LS in T_policy + 9 (cohort year T_policy + 9).

This script computes Korea's actual transition time and rate, and the
biological minimum time and rate from the same starting position.
"""
from pathlib import Path
import json
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WCDE = REPO / "wcde" / "data" / "raw" / "prop_both.csv"
CHECKIN = REPO / "checkin" / "korea_vs_biological_pace.json"

LS_PLUS = ["Lower Secondary", "Upper Secondary", "Post Secondary"]
START_AGE = 6                 # school start
LS_COMPLETION_AGE = 15        # age at LS completion
COHORT_LOAD_YEARS = LS_COMPLETION_AGE - START_AGE   # = 9
MEASUREMENT_AGE_MID = 22      # mid of 20-24 window

# years used to define the transition
YEAR_START = 1955
YEAR_END = 1985


def load_korea_ls_plus():
    df = pd.read_csv(WCDE)
    korea = df[
        (df["name"] == "Republic of Korea")
        & (df["scenario"] == 2)
        & (df["sex"] == "Both")
        & (df["age"] == "20--24")
        & (df["education"].isin(LS_PLUS))
    ]
    return korea.groupby("year")["prop"].sum().reset_index()


def main():
    series = load_korea_ls_plus()

    pct_start = float(series.loc[series["year"] == YEAR_START, "prop"].iloc[0])
    pct_end   = float(series.loc[series["year"] == YEAR_END,   "prop"].iloc[0])

    pp_gain = pct_end - pct_start
    actual_years = YEAR_END - YEAR_START
    actual_rate = pp_gain / actual_years

    # cohort-year translation: 20-24 measurement in calendar year Y
    # corresponds to the cohort that completed LS in Y - (22 - 15) = Y - 7.
    cohort_offset = MEASUREMENT_AGE_MID - LS_COMPLETION_AGE   # = 7
    cohort_start = YEAR_START - cohort_offset
    cohort_end   = YEAR_END   - cohort_offset

    # Biological minimum, anchored to the same starting position:
    # if Korea had flipped delivery in calendar year YEAR_START
    # (when 30% was measured), the earliest cohort that could complete
    # LS at 100% is the one whose 6-year-olds entered school that year:
    # they complete LS in YEAR_START + COHORT_LOAD_YEARS.
    bio_first_full_cohort = YEAR_START + COHORT_LOAD_YEARS
    bio_cohort_span = bio_first_full_cohort - cohort_start
    pp_to_universal = 100.0 - pct_start
    bio_rate = pp_to_universal / bio_cohort_span

    korea_share_of_bio = actual_rate / bio_rate

    out = {
        "korea_ls_pct_at_year_start": round(pct_start, 2),
        "korea_ls_pct_at_year_end": round(pct_end, 2),
        "year_start": YEAR_START,
        "year_end": YEAR_END,
        "korea_pp_gain": round(pp_gain, 2),
        "korea_actual_years": actual_years,
        "korea_actual_rate_pp_per_yr": round(actual_rate, 2),
        "cohort_offset_years": cohort_offset,
        "korea_starting_cohort_ls_completion_year": cohort_start,
        "korea_ending_cohort_ls_completion_year": cohort_end,
        "biological_load_years_per_cohort": COHORT_LOAD_YEARS,
        "biological_first_full_cohort_year": bio_first_full_cohort,
        "biological_cohort_span_to_universal": bio_cohort_span,
        "biological_max_rate_pp_per_yr": round(bio_rate, 2),
        "korea_share_of_biological_pace": round(korea_share_of_bio, 2),
    }

    print(json.dumps(out, indent=2))
    CHECKIN.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
