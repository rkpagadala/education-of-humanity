"""
edu_to_gdp_lag_sweep.py
========================
Step 0 prior test (plan §0.4 / §0.6):
  Does edu(T) -> log GDP(T+lag) peak in the 15-25 year window predicted by
  the leader's-timeline framing (today's primary entrants reach workforce
  age 21+ around T+15; prime productive years T+15 to T+35)?

Prior under test:
  - Strongest edu-to-GDP signal at lag 15-25 years.
  - Should be weaker at lag 0 (cohort not yet productive) and at lag 50+
    (multi-generational decay).

Method:
  For each lag L from 0 to 50 years (step 5):
    - Predictor: lower secondary completion at year T-L  (WCDE v3, 1875-2015)
    - Outcome:   log GDP per capita at year T            (WDI, 1960-2015)
    - Country fixed effects (demean by country mean)
    - Record within-country R^2 and beta coefficient

Parallels scripts/figures/outcomes_r2_by_lag.py but flips edu and GDP:
  outcomes_r2_by_lag.py treats GDP as predictor (placebo for edu).
  This script treats edu as predictor of GDP (the direction the paper claims).

The pair {edu_gdp_r2_lag*, gdp_*_r2_lag* (from existing JSON)} lets us see
both directions on the same panel.

Output:
  checkin/edu_to_gdp_lag_sweep.json
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import PROC, DATA, write_checkin

LAG_MIN, LAG_MAX, LAG_STEP = 0, 50, 5
PANEL_START, PANEL_END = 1960, 2015
MIN_OBS, MIN_OBS_PER_C = 200, 3


def load_wide(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.str.lower().str.strip()
    df.columns = df.columns.astype(int)
    return df.clip(lower=0)


def load_edu():
    edu = pd.read_csv(os.path.join(PROC, "cohort_lower_sec_both.csv"),
                      index_col="country")
    edu.columns = [int(c) for c in edu.columns]
    all_yrs = list(range(min(edu.columns), max(edu.columns) + 1))
    edu = edu.reindex(columns=all_yrs).interpolate(axis=1).bfill(axis=1).ffill(axis=1)
    edu.index = edu.index.str.lower().str.strip()
    return edu.clip(lower=0)


def within_r2_and_beta(predictor_col, outcome_col, rows):
    if len(rows) < MIN_OBS:
        return np.nan, np.nan, 0, 0
    p = pd.DataFrame(rows)
    counts = p.groupby("country")[outcome_col].transform("count")
    p = p[counts >= MIN_OBS_PER_C].copy()
    if len(p) < MIN_OBS:
        return np.nan, np.nan, 0, 0
    n_obs = len(p)
    n_ctry = p["country"].nunique()
    for col in [predictor_col, outcome_col]:
        p[col] = p[col] - p.groupby("country")[col].transform("mean")
    reg = sm.OLS(p[outcome_col], p[[predictor_col]]).fit()
    return reg.rsquared, reg.params[predictor_col], n_obs, n_ctry


def main():
    print("Loading data...")
    edu = load_edu()
    gdp = load_wide(os.path.join(DATA, "gdppercapita_us_inflation_adjusted.csv"))

    countries = sorted(set(edu.index) & set(gdp.index))
    print(f"N countries: {len(countries)}")

    lags = list(range(LAG_MIN, LAG_MAX + 1, LAG_STEP))
    results = []

    print(f"\n{'Lag':>5} {'R²':>7} {'beta':>9} {'n':>6} {'ctry':>5}")
    print("-" * 40)

    for lag in lags:
        rows = []
        for country in countries:
            for yr in range(PANEL_START, PANEL_END + 1):
                g = gdp.loc[country, yr] if yr in gdp.columns else np.nan
                if pd.isna(g) or g <= 0:
                    continue
                log_g = np.log(g)
                yr_pred = yr - lag
                if yr_pred not in edu.columns:
                    continue
                e = edu.loc[country, yr_pred]
                if pd.isna(e):
                    continue
                rows.append({"country": country, "edu": e, "log_gdp": log_g})

        r2, beta, n, nc = within_r2_and_beta("edu", "log_gdp", rows)
        results.append({"lag": lag, "r2": r2, "beta": beta, "n": n, "n_ctry": nc})
        r2_str = f"{r2:.3f}" if not np.isnan(r2) else "  nan"
        beta_str = f"{beta:+.4f}" if not np.isnan(beta) else "    nan"
        print(f"{lag:>5} {r2_str:>7} {beta_str:>9} {n:>6} {nc:>5}")

    valid = [r for r in results if not np.isnan(r["r2"])]
    if valid:
        peak = max(valid, key=lambda r: r["r2"])
        print(f"\nPeak R²: {peak['r2']:.3f} at lag {peak['lag']}")
    else:
        peak = None

    numbers = {"n_countries": len(countries)}
    for r in results:
        numbers[f"edu_gdp_r2_lag{r['lag']}"] = round(r["r2"], 3) if not np.isnan(r["r2"]) else None
        numbers[f"edu_gdp_beta_lag{r['lag']}"] = round(r["beta"], 5) if not np.isnan(r["beta"]) else None
        numbers[f"edu_gdp_n_lag{r['lag']}"] = r["n"]
    if peak is not None:
        numbers["peak_lag"] = peak["lag"]
        numbers["peak_r2"] = round(peak["r2"], 3)
        numbers["peak_beta"] = round(peak["beta"], 5)

    write_checkin("edu_to_gdp_lag_sweep.json", {
        "notes": (f"{len(countries)} countries. Predictor: lower-sec completion (WCDE 1875-2015). "
                  f"Outcome: log GDP per capita (WDI 1960-2015). Within-country FE. "
                  f"Prior under test: peak in 15-25 year window."),
        "numbers": numbers,
    }, script_path="scripts/edu_to_gdp_lag_sweep.py")
    print("\nCheckin written: checkin/edu_to_gdp_lag_sweep.json")


if __name__ == "__main__":
    main()
