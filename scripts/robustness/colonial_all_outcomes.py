"""
robustness/colonial_all_outcomes.py
===================================
Extension of colonial_vs_institutions.py to all four outcomes using
WITHIN-COUNTRY FIXED-EFFECTS R² rather than cross-sectional R².

The parent script (colonial_vs_institutions.py) tests the AJR institutions
story cross-sectionally on a single snapshot (2020). This script answers
the complementary question:

    Within former colonies over time, which predictor — education or
    polity2 institutional quality — explains more of the within-country
    variation in each development outcome?

Outcomes:
  1. Life expectancy (years)
  2. Total fertility rate (children per woman)
  3. Log child mortality under 5 (per 1000)
  4. Child education — lower-secondary completion (WCDE, same generation)

Specification: FE R² from demeaned regression, per-country demeaning on
the sample of former colonies identified in COLONIES (the same panel used
by the parent script). Education lagged 25 years (T+25 outcome paradigm);
polity2 contemporaneous with outcome. Years 1960–2015 at five-year steps.
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _shared import (  # noqa: E402
    DATA,
    PROC,
    REGIONS,
    load_wb,
    NAME_MAP,
    write_checkin,
)
from robustness.colonial_vs_institutions import COLONIES, POLITY_MAP  # noqa: E402

LAG = 25
OUTCOME_YEARS = list(range(1985, 2016, 5))  # outcome observed at T+25 → edu at 1960..1990


def load_polity_panel() -> pd.DataFrame:
    """Return polity2 long-format: columns country (raw polity name), year, polity2."""
    df = pd.read_excel(os.path.join(DATA, "p5v2018.xls"))
    df = df[["country", "year", "polity2"]].dropna(subset=["polity2"]).copy()
    df["polity2"] = pd.to_numeric(df["polity2"], errors="coerce")
    return df.dropna()


def load_edu_wide() -> pd.DataFrame:
    """WCDE lower-secondary completion (both sexes), country × year wide."""
    df = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"))
    df = df[~df["country"].isin(REGIONS)].copy()
    df = df.set_index("country")
    df.columns = [int(c) for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def lookup_wb(df: pd.DataFrame, wcde_name: str, year: int) -> float:
    """Return a WB WDI value for a WCDE-named country at a specific year."""
    key = NAME_MAP.get(wcde_name, wcde_name).lower()
    for k in [wcde_name.lower(), key]:
        if k in df.index:
            try:
                v = float(df.loc[k, str(year)])
                return v if not np.isnan(v) else np.nan
            except (KeyError, ValueError):
                pass
    return np.nan


def build_panel(polity_long: pd.DataFrame, edu_wide: pd.DataFrame,
                outcomes: dict) -> pd.DataFrame:
    """Construct long panel for former colonies over OUTCOME_YEARS."""
    rows = []
    for country in COLONIES:
        pname = POLITY_MAP.get(country, country)
        p_sub = polity_long[polity_long["country"] == pname].set_index("year")
        for t in OUTCOME_YEARS:
            # Education at T-25 (interpolate 5-yr WCDE)
            if country not in edu_wide.index:
                continue
            row = edu_wide.loc[country].dropna()
            if row.empty:
                continue
            years = np.array(row.index, dtype=float)
            vals = row.values.astype(float)
            edu_lag_year = t - LAG
            if edu_lag_year < years.min() or edu_lag_year > years.max():
                continue
            edu_lag = float(np.interp(edu_lag_year, years, vals))

            # Contemporaneous child education for the "child_edu" outcome
            if t < years.min() or t > years.max():
                child_edu = np.nan
            else:
                child_edu = float(np.interp(t, years, vals))

            # Polity2 contemporaneous (average of ±2 years for smoothing)
            polity_val = np.nan
            if pname in polity_long["country"].values:
                window = polity_long[
                    (polity_long["country"] == pname)
                    & (polity_long["year"].between(t - 2, t + 2))
                ]["polity2"]
                if len(window) > 0:
                    polity_val = float(window.mean())

            le_t = lookup_wb(outcomes["le"], country, t)
            tfr_t = lookup_wb(outcomes["tfr"], country, t)
            u5_t = lookup_wb(outcomes["u5mr"], country, t)
            log_u5 = np.log(u5_t) if (u5_t is not None and not np.isnan(u5_t) and u5_t > 0) else np.nan

            rows.append({
                "country": country,
                "year": t,
                "edu_lag25": edu_lag,
                "polity2": polity_val,
                "le": le_t,
                "tfr": tfr_t,
                "log_u5mr": log_u5,
                "child_edu": child_edu,
            })
    return pd.DataFrame(rows)


def within_r2(panel: pd.DataFrame, xcol: str, ycol: str) -> tuple:
    """Country FE R²: demean then OLS. Returns (r2, n_obs, n_countries)."""
    sub = panel.dropna(subset=[xcol, ycol]).copy()
    # Keep countries with ≥2 obs so demeaning leaves variation
    counts = sub.groupby("country")[xcol].transform("count")
    sub = sub[counts >= 2]
    if sub.empty:
        return (np.nan, 0, 0)
    sub["x_dm"] = sub[xcol] - sub.groupby("country")[xcol].transform("mean")
    sub["y_dm"] = sub[ycol] - sub.groupby("country")[ycol].transform("mean")
    # Drop exactly-zero variance rows (single-obs after earlier filter)
    sub = sub[sub.groupby("country")["x_dm"].transform("std").fillna(0) > 0]
    if len(sub) < 5:
        return (np.nan, len(sub), sub["country"].nunique())
    X = sm.add_constant(sub[["x_dm"]].values)
    y = sub["y_dm"].values
    m = sm.OLS(y, X).fit()
    return (float(m.rsquared), int(len(sub)), int(sub["country"].nunique()))


def main() -> None:
    print("=" * 78)
    print("COLONIAL TEST — ALL FOUR OUTCOMES (within-country FE R²)")
    print("Former colonies only. Education lagged 25 years. Polity2 contemporaneous.")
    print("=" * 78)

    polity_long = load_polity_panel()
    edu_wide = load_edu_wide()
    outcomes = {
        "le": load_wb("life_expectancy_years.csv"),
        "tfr": load_wb("children_per_woman_total_fertility.csv"),
        "u5mr": load_wb("child_mortality_u5.csv"),
    }

    panel = build_panel(polity_long, edu_wide, outcomes)
    print(f"\nPanel: {len(panel)} country-year obs, "
          f"{panel['country'].nunique()} countries, "
          f"years {min(OUTCOME_YEARS)}-{max(OUTCOME_YEARS)}")

    outcome_labels = [
        ("LE", "le"),
        ("TFR", "tfr"),
        ("log U5MR", "log_u5mr"),
        ("Child edu (lower sec)", "child_edu"),
    ]

    results = {}
    print(f"\n{'Outcome':<25} {'edu R²':>10} {'polity2 R²':>12} {'edu − pol':>12} {'N':>6} {'k':>5} {'winner':>10}")
    print("-" * 85)
    for label, col in outcome_labels:
        r2_edu, n_e, k_e = within_r2(panel, "edu_lag25", col)
        r2_pol, n_p, k_p = within_r2(panel, "polity2", col)
        # Use the smaller intersection sample size for display
        diff = r2_edu - r2_pol
        winner = "edu" if r2_edu > r2_pol else ("polity2" if r2_pol > r2_edu else "tie")
        print(f"{label:<25} {r2_edu:>10.3f} {r2_pol:>12.3f} {diff:>+12.3f} "
              f"{min(n_e, n_p):>6d} {min(k_e, k_p):>5d} {winner:>10s}")
        results[col] = {
            "r2_education": round(r2_edu, 4) if not np.isnan(r2_edu) else None,
            "r2_polity2": round(r2_pol, 4) if not np.isnan(r2_pol) else None,
            "edu_minus_polity": round(diff, 4) if not np.isnan(diff) else None,
            "n_obs_edu": n_e,
            "n_obs_polity": n_p,
            "n_countries_edu": k_e,
            "n_countries_polity": k_p,
            "winner": winner,
        }

    # Rank outcomes by size of education advantage
    ranked = sorted(
        [(lbl, col, results[col]["edu_minus_polity"])
         for (lbl, col) in outcome_labels
         if results[col]["edu_minus_polity"] is not None],
        key=lambda x: -x[2],
    )
    print("\nRanked by education advantage (edu R² − polity2 R²):")
    for i, (lbl, col, diff) in enumerate(ranked, 1):
        print(f"  {i}. {lbl:<25}  Δ = {diff:+.3f}")

    all_edu_wins = all(results[col]["winner"] == "edu" for _, col in outcome_labels
                       if results[col]["winner"] is not None)

    write_checkin("colonial_all_outcomes.json", {
        "sample": "former colonies only, country-year panel",
        "years": f"{min(OUTCOME_YEARS)}-{max(OUTCOME_YEARS)} (every 5 yrs)",
        "lag_years": LAG,
        "n_panel_obs": int(len(panel)),
        "n_countries": int(panel["country"].nunique()),
        "outcomes": results,
        "ranked_edu_advantage": [
            {"outcome": lbl, "column": col, "edu_minus_polity": round(diff, 4)}
            for (lbl, col, diff) in ranked
        ],
        "education_wins_all": bool(all_edu_wins),
    }, script_path="scripts/robustness/colonial_all_outcomes.py")


if __name__ == "__main__":
    main()
