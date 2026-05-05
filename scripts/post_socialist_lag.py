"""
post_socialist_lag.py
=====================
Post-socialist crosser counts and the lag from reported 90% lower-secondary
completion to the developmental-phenotype crossing (TFR < 3.65 AND LE > 69.8).

The paper (§goskomstat-anomaly) argues the IIASA reconstruction inherited
Goskomstat's inflated education numbers for the 15 USSR republics. One
diagnostic: market-economy crossers reach 90% lower-sec ~18 years before
the phenotype crossing; post-socialist crossers reach the two
simultaneously — or, in Turkmenistan's case, 52 years late.

Outputs:
  - n_postsoc_crossers: count of post-socialist countries that crossed
  - median_lag_market: median (year_90pct − crossing_year) for market crossers
                       (negative because 90% comes first)
  - median_lag_postsoc: same for post-socialist
  - turkmenistan_lag: Turkmenistan's specific gap

Data: WCDE v3 lower_sec_both, WB WDI life expectancy + TFR.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import (
    PROC, TFR_THRESHOLD, LE_THRESHOLD,
    load_wide_indicator, REGIONS, write_checkin,
)

# ── Country sets ──────────────────────────────────────────────────────
USSR = {
    "russian federation", "ukraine", "belarus", "estonia", "latvia",
    "lithuania", "kazakhstan", "uzbekistan", "turkmenistan",
    "kyrgyzstan", "tajikistan", "azerbaijan", "armenia",
    "georgia", "moldova",
}
WARSAW_AND_YUGOSLAV = {
    # Warsaw Pact (excluding USSR proper): East Germany merges into
    # united Germany 1990, omitted; remaining six.
    "poland", "czech republic", "slovakia", "hungary", "romania",
    "bulgaria", "albania",
    # Yugoslav successor states
    "slovenia", "croatia", "bosnia and herzegovina", "serbia",
    "montenegro", "north macedonia", "the former yugoslav republic of macedonia",
}
POST_SOCIALIST = USSR | WARSAW_AND_YUGOSLAV


def first_year_at_or_above(series, threshold):
    """Linearly interpolate to find first year series >= threshold."""
    series = series.dropna().sort_index()
    if len(series) < 2:
        return None
    if series.iloc[0] >= threshold:
        return float(series.index[0])
    above = series[series >= threshold]
    if len(above) == 0:
        return None
    first_above = above.index[0]
    prior = series[series.index < first_above]
    if len(prior) == 0:
        return float(first_above)
    last_below_year = prior.index[-1]
    last_below_val = prior.iloc[-1]
    above_val = series.loc[first_above]
    if above_val == last_below_val:
        return float(first_above)
    frac = (threshold - last_below_val) / (above_val - last_below_val)
    return float(last_below_year + frac * (first_above - last_below_year))


def first_year_at_or_below(series, threshold):
    """Linearly interpolate to find first year series <= threshold."""
    series = series.dropna().sort_index()
    if len(series) < 2:
        return None
    if series.iloc[0] <= threshold:
        return float(series.index[0])
    below = series[series <= threshold]
    if len(below) == 0:
        return None
    first_below = below.index[0]
    prior = series[series.index < first_below]
    if len(prior) == 0:
        return float(first_below)
    last_above_year = prior.index[-1]
    last_above_val = prior.iloc[-1]
    below_val = series.loc[first_below]
    if below_val == last_above_val:
        return float(first_below)
    frac = (last_above_val - threshold) / (last_above_val - below_val)
    return float(last_above_year + frac * (first_below - last_above_year))


def main():
    edu = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"),
                      index_col="country")
    edu = edu[~edu.index.isin(REGIONS)]
    edu.columns = edu.columns.astype(int)

    tfr = load_wide_indicator("children_per_woman_total_fertility.csv")
    le = load_wide_indicator("life_expectancy_years.csv")
    tfr.columns = tfr.columns.astype(int)
    le.columns = le.columns.astype(int)

    rows = []
    for country in edu.index:
        wdi = country.lower()
        # Year 90% lower-sec reached (interpolated)
        year_90 = first_year_at_or_above(edu.loc[country], 90)
        if year_90 is None:
            continue
        # Year of phenotype crossing: BOTH TFR<3.65 AND LE>69.8
        if wdi not in tfr.index or wdi not in le.index:
            # Try a few common name fixes
            fixes = {
                "russian federation": "russian federation",
                "republic of korea": "korea, rep.",
                "viet nam": "vietnam",
                "the former yugoslav republic of macedonia":
                    "north macedonia",
                "republic of moldova": "moldova",
            }
            wdi = fixes.get(wdi, wdi)
            if wdi not in tfr.index or wdi not in le.index:
                continue
        year_tfr = first_year_at_or_below(tfr.loc[wdi], TFR_THRESHOLD)
        year_le = first_year_at_or_above(le.loc[wdi], LE_THRESHOLD)
        if year_tfr is None or year_le is None:
            continue
        crossing_year = max(year_tfr, year_le)
        lag = year_90 - crossing_year  # negative if 90% came first
        rows.append({
            "country": country,
            "year_90pct_lsec": round(year_90, 1),
            "crossing_year": round(crossing_year, 1),
            "lag_yrs": round(lag, 1),
            "post_socialist": country.lower() in POST_SOCIALIST,
        })

    df = pd.DataFrame(rows)
    market = df[~df["post_socialist"]]
    postsoc = df[df["post_socialist"]]

    n_postsoc_crossers = int(len(postsoc))
    median_lag_market = float(market["lag_yrs"].median())
    median_lag_postsoc = float(postsoc["lag_yrs"].median())
    turkm_row = df[df["country"].str.lower() == "turkmenistan"]
    turkmenistan_lag = (float(turkm_row["lag_yrs"].iloc[0])
                        if len(turkm_row) == 1 else None)

    print(f"Total crossers (90% lsec + phenotype): {len(df)}")
    print(f"Post-socialist crossers: {n_postsoc_crossers}")
    print(f"  USSR: {sum(c.lower() in USSR for c in postsoc['country'])}")
    print(f"  Warsaw/Yugoslav: "
          f"{sum(c.lower() in WARSAW_AND_YUGOSLAV for c in postsoc['country'])}")
    print(f"Median lag (90% before phenotype crossing):")
    print(f"  Market: {median_lag_market:+.1f} years")
    print(f"  Post-socialist: {median_lag_postsoc:+.1f} years")
    print(f"  Turkmenistan: {turkmenistan_lag} years")

    write_checkin("post_socialist_lag.json", {
        "method": "First-crossing interpolation: year of 90% lower-sec "
                  "completion (WCDE v3) minus year of phenotype crossing "
                  "(TFR<3.65 AND LE>69.8, WDI). Negative = 90% came first.",
        "numbers": {
            "n_total_crossers": int(len(df)),
            "n_postsoc_crossers": n_postsoc_crossers,
            "median_lag_market_yrs": round(median_lag_market, 1),
            "median_lag_postsoc_yrs": round(median_lag_postsoc, 1),
            "turkmenistan_lag_yrs": (round(turkmenistan_lag, 1)
                                     if turkmenistan_lag is not None
                                     else None),
            # Magnitude versions (paper quotes "18 years before")
            "median_lag_market_yrs_abs": round(abs(median_lag_market)),
            "turkmenistan_lag_yrs_abs": (round(abs(turkmenistan_lag))
                                         if turkmenistan_lag is not None
                                         else None),
        },
    }, script_path="scripts/post_socialist_lag.py")


if __name__ == "__main__":
    main()
