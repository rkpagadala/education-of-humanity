"""
robustness/missing_by_period.py
================================
Reviewer R1.29: report missing countries by period and demonstrate that
the headline result is not driven by listwise deletion.

Method:
  1. For each five-year observation point in 1975-2015, count:
       - countries with parental education at T-25
       - countries with each outcome at T (LE, TFR, U-5)
       - countries with log GDP at T
       - intersection of all (the listwise-complete sample)
       - countries dropped by listwise filtering, by variable

  2. Re-estimate the headline parental-education coefficient
     (one-way country FE; child lower-sec on parental at T-25,
     active-expansion <30% subsample) on three sample definitions:
       - any-available: complete pairs (parent_edu, child_edu) only,
         do not require GDP / outcomes (largest sample)
       - listwise-complete on Table-1 inputs (current Table 1 sample)
       - listwise-complete on Table-1 + Table-7 inputs (intersection
         with the GDP-and-outcomes panel; smaller sample)

  Stable coefficients across the three definitions show that listwise
  deletion is not driving the headline.

Outputs:
  checkin/missing_by_period.json
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import (PROC, DATA, REGIONS as NON_SOVEREIGN, write_checkin,
                     add_canonical_aliases, standardize_country_name)

OUTCOME_YEARS = list(range(1975, 2016, 5))
PARENTAL_LAG = 25


def load_wb(filename):
    path = os.path.join(DATA, filename)
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(lambda s: standardize_country_name(str(s)))
    df = add_canonical_aliases(df)
    return df


def get_yr(df, country, year):
    if country not in df.index or str(year) not in df.columns:
        return np.nan
    v = df.loc[country, str(year)]
    return float(v) if pd.notna(v) else np.nan


def main():
    edu = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"), index_col="country")
    edu = edu[~edu.index.isin(NON_SOVEREIGN)]
    edu.index = edu.index.map(lambda s: standardize_country_name(str(s)))
    # Use canonical names only (no alias-row duplication) for the audit;
    # the per-period country counts must equal sovereign-country counts.
    edu = edu[~edu.index.duplicated(keep="first")]

    gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
    gdp_df = gdp_df[~gdp_df.index.duplicated(keep="first")]
    le_df = load_wb("life_expectancy_years.csv")
    le_df = le_df[~le_df.index.duplicated(keep="first")]
    tfr_df = load_wb("children_per_woman_total_fertility.csv")
    tfr_df = tfr_df[~tfr_df.index.duplicated(keep="first")]
    u5_df = load_wb("child_mortality_u5.csv")
    u5_df = u5_df[~u5_df.index.duplicated(keep="first")]

    countries = sorted(set(edu.index))

    # ── Per-period coverage matrix ──────────────────────────────────────
    coverage = {}
    rows = []
    for y in OUTCOME_YEARS:
        ny = y - PARENTAL_LAG
        per_var = {"parent_edu": [], "child_edu": [], "log_gdp": [],
                   "le": [], "tfr": [], "u5": []}
        for c in countries:
            pe = get_yr(edu, c, ny)
            ce = get_yr(edu, c, y)
            lg = get_yr(gdp_df, c, y)
            lev = get_yr(le_df, c, y)
            tfv = get_yr(tfr_df, c, y)
            u5v = get_yr(u5_df, c, y)
            if not np.isnan(pe):
                per_var["parent_edu"].append(c)
            if not np.isnan(ce):
                per_var["child_edu"].append(c)
            if not np.isnan(lg) and lg > 0:
                per_var["log_gdp"].append(c)
            if not np.isnan(lev):
                per_var["le"].append(c)
            if not np.isnan(tfv):
                per_var["tfr"].append(c)
            if not np.isnan(u5v):
                per_var["u5"].append(c)
            rows.append({
                "country": c, "year": y,
                "parent_edu": pe, "child_edu": ce,
                "log_gdp": np.log(lg) if not np.isnan(lg) and lg > 0 else np.nan,
                "le": lev, "tfr": tfv, "u5": u5v,
            })
        sets = {k: set(v) for k, v in per_var.items()}
        intersection = sets["parent_edu"] & sets["child_edu"] & sets["log_gdp"] \
                       & sets["le"] & sets["tfr"] & sets["u5"]
        # Edu-only (Table 1 col 1) — needs parent_edu and child_edu
        edu_only = sets["parent_edu"] & sets["child_edu"]
        coverage[y] = {
            "any_var_max": max(len(s) for s in sets.values()),
            "parent_edu":  len(sets["parent_edu"]),
            "child_edu":   len(sets["child_edu"]),
            "log_gdp":     len(sets["log_gdp"]),
            "le":          len(sets["le"]),
            "tfr":         len(sets["tfr"]),
            "u5":          len(sets["u5"]),
            "edu_only":    len(edu_only),
            "all_vars":    len(intersection),
            "listwise_drop_pct": round(100 * (1 - len(intersection) / max(1, len(edu_only))), 1),
        }

    panel = pd.DataFrame(rows)

    # ── Headline coefficient under three sample definitions ─────────────
    def fe_beta(sub_df, x_col="parent_edu", y_col="child_edu"):
        if len(sub_df) < 50:
            return None
        d = sub_df.copy()
        d["x_dem"] = d[x_col] - d.groupby("country")[x_col].transform("mean")
        d["y_dem"] = d[y_col] - d.groupby("country")[y_col].transform("mean")
        d = d.dropna(subset=["x_dem", "y_dem"])
        if len(d) < 50:
            return None
        x = d["x_dem"].to_numpy()
        y = d["y_dem"].to_numpy()
        beta = float((x * y).sum() / (x * x).sum())
        return {"beta": round(beta, 3), "n": int(len(d)),
                "countries": int(d["country"].nunique())}

    active = panel[(panel["parent_edu"] < 30) &
                   (~panel["parent_edu"].isna()) &
                   (~panel["child_edu"].isna())]
    listwise_t1 = active.dropna(subset=["parent_edu", "child_edu", "log_gdp"])
    listwise_full = active.dropna(subset=["parent_edu", "child_edu", "log_gdp",
                                            "le", "tfr", "u5"])

    sample_results = {
        "any_available_t1_pair":   fe_beta(active),
        "listwise_t1_inputs":      fe_beta(listwise_t1),
        "listwise_t1_plus_t7":     fe_beta(listwise_full),
    }

    # ── Print ───────────────────────────────────────────────────────────
    print("Per-period coverage (active expansion is post-filter, table 1)")
    print("=" * 90)
    print(f"{'Year':>4}  {'parent':>7}  {'child':>6}  {'gdp':>4}  {'le':>4}  "
          f"{'tfr':>4}  {'u5':>4}  {'edu-only':>9}  {'all':>4}  {'drop %':>7}")
    print("-" * 90)
    for y in OUTCOME_YEARS:
        c = coverage[y]
        print(f"{y:>4}  {c['parent_edu']:>7d}  {c['child_edu']:>6d}  "
              f"{c['log_gdp']:>4d}  {c['le']:>4d}  {c['tfr']:>4d}  {c['u5']:>4d}  "
              f"{c['edu_only']:>9d}  {c['all_vars']:>4d}  "
              f"{c['listwise_drop_pct']:>6.1f}%")

    print()
    print("Headline parental-education coefficient (active-expansion <30%, country FE)")
    print("=" * 90)
    for label, r in sample_results.items():
        if r is None:
            print(f"{label:30s}  insufficient n")
        else:
            print(f"{label:30s}  beta = {r['beta']:>5.3f}   "
                  f"n = {r['n']:>5d}   countries = {r['countries']:>3d}")

    write_checkin(
        "missing_by_period.json",
        {
            "method": (
                "Per-period country coverage by variable (1975-2015, 5-yr). "
                "Listwise-deletion stress test: re-estimate the active-expansion "
                "(<30%) parental-education coefficient on three nested samples."
            ),
            "coverage_by_year": {str(y): coverage[y] for y in OUTCOME_YEARS},
            "sample_results": sample_results,
        },
        script_path="scripts/robustness/missing_by_period.py",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
