# =============================================================================
# PAPER REFERENCE
# Script:  scripts/robustness/cross_cohort_within_year.py
# Paper:   "Education of Humanity"
#
# Produces:
#   Independent test of the intergenerational-transmission mechanism that
#   does NOT rely on the 25-year forward lag. Uses WCDE v3 population-share
#   microdata (prop_both.csv), which reports the education distribution by
#   five-year age group within each country-year.
#
#   In each (country, year), we compute lower-secondary-completion shares
#   for two age cohorts measured simultaneously:
#     child  = age 20-24  (the age cohort used everywhere else in the paper)
#     parent = age 45-49  (25 years older — one generation)
#
#   Regression:
#     child_{c,y} = β · parent_{c,y} + α_c + ε_{c,y}
#     (country fixed effects, country-clustered SEs)
#
#   Because parent and child are observed in the same year, this specification
#   is immune to cross-country shocks that occurred between year t-25 and
#   year t (global MDGs, HIV pandemic, end of the Cold War, etc.), which
#   were the principal concern the reviewer raised about the forward-lag
#   design.
#
# Output: checkin/cross_cohort_within_year.json
# =============================================================================
"""Within-year cross-cohort test: 20-24 on 45-49 (WCDE v3 microdata)."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from _shared import REPO_ROOT, REGIONS, write_checkin  # noqa: E402

# Any WCDE education category at or above "Lower Secondary" counts as
# "lower secondary complete." These are the four WCDE-v3 categories
# counted, in order of attainment.
COMPLETE_CATEGORIES = {"Lower Secondary", "Upper Secondary", "Post Secondary"}
RAW_PATH = os.path.join(REPO_ROOT, "wcde", "data", "raw", "prop_both.csv")

# WCDE standard scenario (SSP2 medium).
SCENARIO = 2
# Child and parent age groups.
CHILD_AGE = "20--24"
PARENT_AGE = "45--49"
# One generation for parent-child cohort = 25 years.
ACTIVE_EXPANSION_CUTOFF = 30


def load_completion_by_age():
    """Return DataFrame: country, year, age, lower_sec_complete_pct."""
    df = pd.read_csv(RAW_PATH)
    df = df[df["scenario"] == SCENARIO].copy()
    df = df[df["sex"] == "Both"].copy()
    df = df[df["age"].isin({CHILD_AGE, PARENT_AGE})].copy()
    df = df[~df["name"].isin(REGIONS)].copy()
    df["complete"] = df["education"].isin(COMPLETE_CATEGORIES).astype(float)
    # sum of prop for the completion categories per (country,year,age)
    grp = (df.assign(w=df["prop"] * df["complete"])
             .groupby(["name", "year", "age"], as_index=False)["w"].sum())
    grp = grp.rename(columns={"name": "country", "w": "completion_pct"})
    return grp


def build_panel(long):
    wide = long.pivot(index=["country", "year"], columns="age",
                       values="completion_pct").reset_index()
    wide = wide.rename(columns={CHILD_AGE: "child", PARENT_AGE: "parent"})
    wide = wide.dropna(subset=["child", "parent"])
    return wide


def fit(d, label):
    panel = d.set_index(["country", "year"])
    mod = PanelOLS(panel["child"], panel[["parent"]],
                   entity_effects=True, drop_absorbed=True, check_rank=False)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    b = float(res.params["parent"])
    se = float(res.std_errors["parent"])
    p = float(res.pvalues["parent"])
    r2 = float(res.rsquared_within)
    n = int(res.nobs)
    n_ctry = int(res.entity_info.total)
    print(f"{label:40s}  β={b:.4f}  SE={se:.4f}  p={p:.4f}  "
          f"R²={r2:.3f}  n={n}  ctry={n_ctry}")
    return {
        "parent_beta": round(b, 4),
        "parent_se":   round(se, 4),
        "parent_p":    round(p, 4),
        "r2_within":   round(r2, 4),
        "n": n,
        "countries": n_ctry,
    }


def main():
    long = load_completion_by_age()
    panel = build_panel(long)
    print(f"Cross-cohort panel: {len(panel)} country-years, "
          f"{panel['country'].nunique()} countries\n")

    # Need ≥2 observations per country for within-FE
    counts = panel.groupby("country").size()
    panel = panel[panel["country"].isin(counts[counts >= 2].index)].copy()

    full = fit(panel, "full sample (within-year, country FE)")

    active = panel[panel["parent"] < ACTIVE_EXPANSION_CUTOFF].copy()
    counts_a = active.groupby("country").size()
    active = active[active["country"].isin(counts_a[counts_a >= 2].index)].copy()
    act = fit(active, f"active expansion (parent 45-49 <{ACTIVE_EXPANSION_CUTOFF}%)")

    write_checkin("cross_cohort_within_year.json", {
        "notes": (
            "Within-year cross-cohort test of parental transmission. "
            "For each country-year in WCDE v3 microdata, lower-secondary "
            "completion share of age 20-24 is regressed on the same share "
            "for age 45-49 (one generation older, measured simultaneously). "
            "Country FE, country-clustered SEs. This specification does not "
            "rely on the 25-year forward lag: parent and child are observed "
            "in the same year, so any common cross-country time shock in "
            "the intervening decades is absorbed by design."
        ),
        "results": {"full_sample": full, "active_expansion": act},
        "numbers": {
            "full_beta":      full["parent_beta"],
            "full_se":        full["parent_se"],
            "full_p":         full["parent_p"],
            "full_r2":        full["r2_within"],
            "full_n":         full["n"],
            "full_countries": full["countries"],
            "active_beta":      act["parent_beta"],
            "active_se":        act["parent_se"],
            "active_p":         act["parent_p"],
            "active_r2":        act["r2_within"],
            "active_n":         act["n"],
            "active_countries": act["countries"],
        },
    }, script_path="scripts/robustness/cross_cohort_within_year.py")
    print("\nDone.")


if __name__ == "__main__":
    main()
