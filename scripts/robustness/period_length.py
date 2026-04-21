# =============================================================================
# PAPER REFERENCE
# Script:  scripts/robustness/period_length.py
# Paper:   "Education of Humanity"
#
# Produces:
#   Period-length robustness for the Table 1 Column 1 headline
#   (child_t ~ parent_{t-25}, country FE, clustered SE).
#
#   Three aggregation horizons:
#     5-year  (baseline):   child cohorts 1975..2015 step 5
#     10-year:              child cohorts 1975..2015 step 10
#     annual (interpolated): 1975..2015 step 1, linear interpolation
#                             of 5-year WCDE cohort values
#
#   Headline and active-expansion (parent <30%) specifications reported
#   for each horizon.
#
#   The annual specification mechanically inflates the within-country
#   observation count (~5×) and tightens the SE; the coefficient is
#   interpreted as a check on time-aggregation, not as independent evidence.
#
# Output: checkin/period_length.json
# =============================================================================
"""Period-length (5y / 10y / annual) robustness for Table 1 headline."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from _shared import PROC, REGIONS, write_checkin  # noqa: E402

PARENTAL_LAG = 25
ACTIVE_EXPANSION_CUTOFF = 30


def load_5yr():
    long = pd.read_csv(os.path.join(PROC, "cohort_completion_both_long.csv"))
    long = long[~long["country"].isin(REGIONS)]
    return long.pivot(index="country", columns="cohort_year",
                       values="lower_sec")


def load_annual(wide_5yr):
    """Linear interpolation of the 5-year WCDE cohort grid to annual."""
    dft = wide_5yr.T
    dft.index = dft.index.astype(int)
    yrs = sorted(dft.index)
    full = range(min(yrs), max(yrs) + 1)
    dft = dft.reindex(full).interpolate(method="linear")
    return dft.T


def build(wide, child_years):
    def edu(c, y):
        try:
            v = float(wide.loc[c, int(y)])
            return v if not np.isnan(v) else np.nan
        except (KeyError, ValueError):
            return np.nan

    rows = []
    for c in wide.index:
        for ch in child_years:
            p_yr = ch - PARENTAL_LAG
            child = edu(c, ch)
            parent = edu(c, p_yr)
            if np.isnan(child) or np.isnan(parent):
                continue
            rows.append({"country": c, "year": ch,
                         "child": child, "parent": parent})
    return pd.DataFrame(rows)


def fit(d):
    panel = d.set_index(["country", "year"])
    mod = PanelOLS(panel["child"], panel[["parent"]],
                   entity_effects=True, drop_absorbed=True, check_rank=False)
    return mod.fit(cov_type="clustered", cluster_entity=True)


def summarize(res, label):
    b = float(res.params["parent"])
    se = float(res.std_errors["parent"])
    p = float(res.pvalues["parent"])
    r2 = float(res.rsquared_within)
    n = int(res.nobs)
    n_ctry = int(res.entity_info.total)
    print(f"{label:40s}  β={b:.4f}  SE={se:.4f}  p={p:.4f}  "
          f"R²={r2:.3f}  n={n:5d}  ctry={n_ctry}")
    return {
        "parent_beta": round(b, 4),
        "parent_se":   round(se, 4),
        "parent_p":    round(p, 4),
        "r2_within":   round(r2, 4),
        "n": n,
        "countries": n_ctry,
    }


def run_horizon(label, panel):
    print(f"\n=== {label} ===")
    full = summarize(fit(panel), "full")
    active = panel[panel["parent"] < ACTIVE_EXPANSION_CUTOFF].copy()
    counts = active.groupby("country").size()
    active = active[active["country"].isin(counts[counts >= 2].index)].copy()
    act = summarize(fit(active), f"active expansion (<{ACTIVE_EXPANSION_CUTOFF}%)")
    return {"full": full, "active_expansion": act}


def main():
    w5 = load_5yr()
    print(f"5-year WCDE cohort grid: {w5.shape[0]} countries, "
          f"{w5.shape[1]} cohort years")

    wa = load_annual(w5)
    print(f"Annual (interpolated) grid: {wa.shape[0]} countries, "
          f"{wa.shape[1]} cohort years\n")

    cohorts_5  = list(range(1975, 2016, 5))
    cohorts_10 = list(range(1975, 2016, 10))
    cohorts_1  = list(range(1975, 2016, 1))

    p5  = build(w5, cohorts_5)
    p10 = build(w5, cohorts_10)
    pa  = build(wa, cohorts_1)

    five = run_horizon("5-year (baseline)", p5)
    ten  = run_horizon("10-year", p10)
    ann  = run_horizon("annual (interpolated)", pa)

    write_checkin("period_length.json", {
        "notes": (
            "Period-length robustness. Child cohort years 1975-2015 "
            "stepped by 5, 10, and 1 (annual, via linear interpolation "
            "of the 5-year WCDE grid). Each specification: child_t ~ "
            "parent_{t-25}, country FE, country-clustered SEs. "
            "The annual grid mechanically inflates n; the coefficient "
            "value is the quantity of interest, not the SE."
        ),
        "results": {
            "five_year":  five,
            "ten_year":   ten,
            "annual":     ann,
        },
        "numbers": {
            "five_full_beta":   five["full"]["parent_beta"],
            "five_full_n":      five["full"]["n"],
            "five_active_beta": five["active_expansion"]["parent_beta"],
            "five_active_n":    five["active_expansion"]["n"],
            "ten_full_beta":    ten["full"]["parent_beta"],
            "ten_full_n":       ten["full"]["n"],
            "ten_active_beta":  ten["active_expansion"]["parent_beta"],
            "ten_active_n":     ten["active_expansion"]["n"],
            "annual_full_beta":   ann["full"]["parent_beta"],
            "annual_full_n":      ann["full"]["n"],
            "annual_active_beta": ann["active_expansion"]["parent_beta"],
            "annual_active_n":    ann["active_expansion"]["n"],
        },
    }, script_path="scripts/robustness/period_length.py")
    print("\nDone.")


if __name__ == "__main__":
    main()
