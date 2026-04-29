"""
Summary statistics and sample definition for the paper's main panel.

Produces two artefacts cited in the Appendix:

  1. Descriptive statistics (N, mean, SD, min, max) for the key analysis
     variables, pooled and by period (1975-89, 1990-04, 2005-15).

  2. Country-inclusion list: which countries appear in each main
     specification (education-only full panel; education+GDP; active-
     expansion <30% subsample).

Data sources:
  - Education (both sexes, age 20-24): wcde/data/processed/lower_sec_both.csv
  - GDP: data/gdppercapita_us_inflation_adjusted.csv (constant 2015 USD)
  - Life expectancy: data/life_expectancy_years.csv
  - TFR: data/children_per_woman_total_fertility.csv
  - U-5 mortality: data/child_mortality_u5.csv

Output: checkin/summary_stats.json
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from _shared import (PROC, DATA, REGIONS as NON_SOVEREIGN,
                     write_checkin, add_canonical_aliases, standardize_country_name)

PARENTAL_LAG = 25
OUTCOME_YEARS = list(range(1975, 2016, 5))   # 9 five-year points
PERIODS = [("1975-1989", range(1975, 1990, 5)),
           ("1990-2004", range(1990, 2005, 5)),
           ("2005-2015", range(2005, 2016, 5))]


def load_wb(filename):
    path = os.path.join(DATA, filename)
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(lambda s: standardize_country_name(str(s)))
    df = add_canonical_aliases(df)
    return df


def describe(series):
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {
        "n": int(len(s)),
        "mean": round(float(s.mean()), 3),
        "sd":   round(float(s.std(ddof=1)), 3),
        "min":  round(float(s.min()), 3),
        "max":  round(float(s.max()), 3),
    }


def main():
    edu = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"), index_col="country")
    gdp = load_wb("gdppercapita_us_inflation_adjusted.csv")
    le  = load_wb("life_expectancy_years.csv")
    tfr = load_wb("children_per_woman_total_fertility.csv")
    u5m = load_wb("child_mortality_u5.csv")

    # ── Build panel ─────────────────────────────────────────────────
    rows = []
    for country in edu.index:
        if country in NON_SOVEREIGN:
            continue
        c = country.lower()
        for y in OUTCOME_YEARS:
            sy, sy_lag = str(y), str(y - PARENTAL_LAG)
            if sy not in edu.columns or sy_lag not in edu.columns:
                continue
            parent = edu.loc[country, sy_lag]
            child  = edu.loc[country, sy]
            if np.isnan(child) or np.isnan(parent):
                continue
            rows.append({
                "country": country,
                "year":    y,
                "parent_edu": parent,
                "child_edu":  child,
                "log_gdp":    np.log(float(gdp.loc[c, sy])) if (c in gdp.index and sy in gdp.columns and pd.notna(gdp.loc[c, sy]) and float(gdp.loc[c, sy]) > 0) else np.nan,
                "log_gdp_parent": np.log(float(gdp.loc[c, sy_lag])) if (c in gdp.index and sy_lag in gdp.columns and pd.notna(gdp.loc[c, sy_lag]) and float(gdp.loc[c, sy_lag]) > 0) else np.nan,
                "life_exp":   float(le.loc[c, sy]) if (c in le.index and sy in le.columns and pd.notna(le.loc[c, sy])) else np.nan,
                "tfr":        float(tfr.loc[c, sy]) if (c in tfr.index and sy in tfr.columns and pd.notna(tfr.loc[c, sy])) else np.nan,
                "u5mr":       float(u5m.loc[c, sy]) if (c in u5m.index and sy in u5m.columns and pd.notna(u5m.loc[c, sy])) else np.nan,
            })
    df = pd.DataFrame(rows)

    # ── Descriptive statistics ──────────────────────────────────────
    vars_ = [
        ("parent_edu",  "Parental lower secondary completion (%, T$-$25)"),
        ("child_edu",   "Child lower secondary completion (%, T)"),
        ("log_gdp",     "Log GDP per capita (constant 2015 USD)"),
        ("life_exp",    "Life expectancy at birth (years)"),
        ("tfr",         "Total fertility rate (births per woman)"),
        ("u5mr",        "Under-5 mortality (per 1{,}000 live births)"),
    ]

    stats = {"pooled": {}, "by_period": {}}
    for v, _label in vars_:
        stats["pooled"][v] = describe(df[v])
        for pname, years in PERIODS:
            sub = df[df.year.isin(list(years))][v]
            stats["by_period"].setdefault(v, {})[pname] = describe(sub)

    # ── Country inclusion per specification ─────────────────────────
    full_countries = sorted(df["country"].unique())
    # "Edu + GDP" panel uses contemporaneous GDP (matches panel_full_fe Model 2).
    with_gdp = df.dropna(subset=["parent_edu", "log_gdp"])
    gdp_countries = sorted(with_gdp["country"].unique())
    dropped_for_gdp = sorted(set(full_countries) - set(gdp_countries))
    # Note: the headline Table 1 <30% subsample (105 countries / 629 obs) is
    # produced by scripts/residualization/by_gdp_cutoff.py on a different
    # education input file (cohort_completion_both_long.csv). We do not
    # re-derive those counts here; they are authoritative in
    # checkin/education_vs_gdp_by_cutoff.json under cutoff_30_n / countries.

    out = {
        "numbers": {
            "panel_obs":       int(len(df)),
            "panel_countries": len(full_countries),
            "gdp_panel_obs":   int(len(with_gdp)),
            "gdp_panel_countries": len(gdp_countries),
            "dropped_for_gdp_n":   len(dropped_for_gdp),
        },
        "descriptives": stats,
        "variable_labels": {v: lbl for v, lbl in vars_},
        "country_lists": {
            "full_panel":      full_countries,
            "edu_plus_gdp":    gdp_countries,
            "dropped_from_gdp": dropped_for_gdp,
        },
    }

    # ── Print ───────────────────────────────────────────────────────
    print("=" * 70)
    print("Summary statistics — pooled (1975-2015, 5-year intervals)")
    print("=" * 70)
    print(f"{'variable':45s}  {'n':>5s}  {'mean':>8s}  {'sd':>8s}  {'min':>8s}  {'max':>8s}")
    for v, lbl in vars_:
        s = stats["pooled"][v]
        print(f"{lbl[:45]:45s}  {s['n']:5d}  {s['mean']:8.3f}  {s['sd']:8.3f}  {s['min']:8.3f}  {s['max']:8.3f}")

    print()
    print(f"Full panel:        {len(full_countries):3d} countries, {len(df)} obs")
    print(f"Edu + GDP panel:   {len(gdp_countries):3d} countries, {len(with_gdp)} obs")
    print(f"  Dropped (no GDP): {len(dropped_for_gdp)} — {', '.join(dropped_for_gdp) if dropped_for_gdp else '(none)'}")

    write_checkin("summary_stats.json", out,
                  script_path="scripts/tables/summary_stats.py")


if __name__ == "__main__":
    main()
