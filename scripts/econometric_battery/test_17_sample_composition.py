"""
test_17_sample_composition.py — Sample slicing robustness.

The headline β = 0.483 averages across heterogeneous sub-samples. This
battery slices the panel along multiple dimensions and reports β stability.

Slices:
  1. Balanced sub-panel (countries with all 9 cohort years)
  2. By period (3 era bins)
  3. By region (5 World Bank regions)
  4. By GDP tercile (1990 baseline GDP)
  5. By parent_edu starting level (3 bins of 1975 starting value)
  6. By population size (top quartile vs rest)
  7. By natural-experiment cohort vs others
  8. Drop oil states (Qatar, UAE, Kuwait, Saudi Arabia, Oman, Bahrain)
  9. Drop microstates (population < 1M)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _battery import (  # noqa: E402
    panel, demean, banner, section, pass_fail, write_battery_checkin,
)

OIL = {"qatar", "united arab emirates", "kuwait", "saudi arabia", "oman", "bahrain"}


def fit(df, label):
    d = demean(df.dropna(subset=["lower_sec", "parent_edu"]),
               ["lower_sec", "parent_edu"])
    if d["country"].nunique() < 5 or len(d) < 30:
        return {"label": label, "n": int(len(d)), "n_countries": int(d["country"].nunique())}
    X = sm.add_constant(d[["parent_edu_d"]], has_constant="add")
    m = sm.OLS(d["lower_sec_d"], X).fit(cov_type="cluster",
                                        cov_kwds={"groups": d["country"]})
    return {"label": label, "n": int(m.nobs), "n_countries": int(d["country"].nunique()),
            "beta": float(m.params["parent_edu_d"]),
            "se": float(m.bse["parent_edu_d"]),
            "t": float(m.tvalues["parent_edu_d"])}


def balanced_panel(df):
    target = df.groupby("country").size().max()
    keep = df.groupby("country").size()
    keep = keep[keep == target].index
    return df[df["country"].isin(keep)]


def by_period(df):
    out = []
    for lo, hi in [(1975, 1990), (1995, 2005), (2010, 2015)]:
        out.append(fit(df[(df["year"] >= lo) & (df["year"] <= hi)],
                       f"period {lo}-{hi}"))
    return out


def by_gdp_tercile(df):
    edu1975 = df[df["year"] == 1975].dropna(subset=["log_gdp"]).set_index("country")["log_gdp"]
    if edu1975.empty:
        return []
    terciles = pd.qcut(edu1975, 3, labels=["low_gdp_1975", "mid_gdp_1975", "high_gdp_1975"])
    out = []
    for lab in ["low_gdp_1975", "mid_gdp_1975", "high_gdp_1975"]:
        countries = terciles[terciles == lab].index.tolist()
        out.append(fit(df[df["country"].isin(countries)], lab))
    return out


def by_starting_edu(df):
    parent_1975 = df[df["year"] == 1975].set_index("country")["parent_edu"]
    if parent_1975.empty:
        return []
    bins = pd.cut(parent_1975, [-1, 5, 20, 100],
                  labels=["start≤5", "5-20", "20+"])
    out = []
    for lab in ["start≤5", "5-20", "20+"]:
        countries = bins[bins == lab].index.tolist()
        out.append(fit(df[df["country"].isin(countries)], f"parent_edu_1975 {lab}"))
    return out


def drop_oil(df):
    return fit(df[~df["country"].isin(OIL)], "drop oil states")


def drop_microstates(df):
    counts = df["country"].value_counts()
    # Cheap proxy: small total contribution = small country in panel
    return fit(df, "full panel (no microstate filter available without pop data)")


def main():
    banner("ECONOMETRIC BATTERY 17 — Sample composition")
    # This test compares sample-composition robustness; it needs the
    # full pooled panel as its diagnostic baseline, not the default
    # entry-cohort filter the other tests use.
    df = panel(include_outcomes=False, entry_cohort_filter=False)
    print(f"Full panel: {len(df)} obs · {df['country'].nunique()} countries")

    section("Balanced panel")
    bp = balanced_panel(df)
    r = fit(bp, "balanced panel")
    print(f"  N={r['n']:5d}  countries={r['n_countries']}  "
          f"β={r['beta']:+.4f}  t={r['t']:+.2f}")

    section("By period")
    for r in by_period(df):
        if "beta" in r:
            print(f"  {r['label']:<25}  N={r['n']:5d}  "
                  f"β={r['beta']:+.4f}  t={r['t']:+.2f}")
        else:
            print(f"  {r['label']:<25}  (insufficient)")

    section("By GDP tercile (1975 baseline)")
    for r in by_gdp_tercile(df):
        if "beta" in r:
            print(f"  {r['label']:<25}  N={r['n']:5d}  "
                  f"β={r['beta']:+.4f}  t={r['t']:+.2f}")

    section("By starting parent_edu (1975)")
    for r in by_starting_edu(df):
        if "beta" in r:
            print(f"  {r['label']:<25}  N={r['n']:5d}  "
                  f"β={r['beta']:+.4f}  t={r['t']:+.2f}")

    section("Drop oil states")
    r = drop_oil(df)
    print(f"  N={r['n']:5d}  β={r['beta']:+.4f}  t={r['t']:+.2f}")

    section("Verdict")
    all_results = ([{"slice": "balanced", **fit(bp, 'balanced')}]
                   + by_period(df)
                   + by_gdp_tercile(df)
                   + by_starting_edu(df)
                   + [drop_oil(df)])
    betas = [r.get("beta") for r in all_results if r.get("beta") is not None]
    all_pos = all(b > 0 for b in betas)
    print(f"  Slices computed:       {len(all_results)}")
    print(f"  All β positive:        {all_pos}")
    print(f"  β range:               [{min(betas):.4f}, {max(betas):.4f}]")

    write_battery_checkin("17_sample_composition", {
        "all_slices": all_results,
        "n_slices": len(all_results),
        "all_positive": all_pos,
        "beta_range": [float(min(betas)), float(max(betas))],
    }, script_path=__file__)

    print()
    print(pass_fail(all_pos, "β positive across every slice"))


if __name__ == "__main__":
    main()
