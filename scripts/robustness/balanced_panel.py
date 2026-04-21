# =============================================================================
# PAPER REFERENCE
# Script:  scripts/robustness/balanced_panel.py
# Paper:   "Education of Humanity"
#
# Produces:
#   Re-estimates the Table 1 Column 1 headline (child_t ~ parent_{t-25},
#   country FE, clustered SE) on a balanced subpanel where every country
#   appears in every 5-year cohort year 1975..2015 (9 child cohorts).
#
#   Unbalanced (headline): 1665 obs / 185 countries.
#   Balanced:              N(countries) × 9, where N is the subset of
#                          countries observed in all 9 cohort years.
#
#   Also reports the balanced counterpart for the active-expansion
#   (parent <30%) sample — restricted to countries whose entire
#   cohort sequence satisfies the cutoff in every year they appear
#   below 30%, and that contribute ≥2 obs (required for within-FE).
#
# Output: checkin/balanced_panel.json
# =============================================================================
"""Balanced-panel robustness for the Table 1 specification."""

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
PERIOD = 5
CHILD_COHORTS = list(range(1975, 2016, PERIOD))   # 9 cohort years
N_EXPECTED_PERIODS = len(CHILD_COHORTS)
ACTIVE_EXPANSION_CUTOFF = 30


def load_panel():
    long = pd.read_csv(os.path.join(PROC, "cohort_completion_both_long.csv"))
    long = long[~long["country"].isin(REGIONS)]
    wide = long.pivot(index="country", columns="cohort_year", values="lower_sec")

    def edu(c, y):
        try:
            v = float(wide.loc[c, int(y)])
            return v if not np.isnan(v) else np.nan
        except (KeyError, ValueError):
            return np.nan

    rows = []
    for c in wide.index:
        for ch in CHILD_COHORTS:
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


def summarize(res, label, d):
    b = float(res.params["parent"])
    se = float(res.std_errors["parent"])
    p = float(res.pvalues["parent"])
    r2 = float(res.rsquared_within)
    n = int(res.nobs)
    n_ctry = int(res.entity_info.total)
    per = d.groupby("country").size()
    print(f"\n{label}: n={n}, countries={n_ctry}, "
          f"obs/country min={per.min()} max={per.max()} mean={per.mean():.2f}")
    print(f"  β(parent) = {b:.4f}   SE={se:.4f}   p={p:.4f}   R²={r2:.3f}")
    return {
        "parent_beta": round(b, 4),
        "parent_se":   round(se, 4),
        "parent_p":    round(p, 4),
        "r2_within":   round(r2, 4),
        "n": n,
        "countries": n_ctry,
        "obs_per_country_min":  int(per.min()),
        "obs_per_country_max":  int(per.max()),
        "obs_per_country_mean": round(float(per.mean()), 2),
    }


def main():
    panel = load_panel()
    print(f"Unbalanced full panel: {len(panel)} obs, "
          f"{panel['country'].nunique()} countries")
    full = summarize(fit(panel), "FULL (unbalanced)", panel)

    counts = panel.groupby("country").size()
    balanced_ctys = counts[counts == N_EXPECTED_PERIODS].index
    balanced = panel[panel["country"].isin(balanced_ctys)].copy()
    print(f"\nBalanced subset: countries with all "
          f"{N_EXPECTED_PERIODS} cohort years present = "
          f"{len(balanced_ctys)} countries")
    bal = summarize(fit(balanced), "FULL (balanced)", balanced)

    active = panel[panel["parent"] < ACTIVE_EXPANSION_CUTOFF].copy()
    counts_a = active.groupby("country").size()
    keep = counts_a[counts_a >= 2].index
    active = active[active["country"].isin(keep)].copy()
    act_full = summarize(fit(active), "ACTIVE EXPANSION (unbalanced)", active)

    active_bal_ctys = counts_a[counts_a == counts_a.max()].index
    active_bal = active[active["country"].isin(active_bal_ctys)].copy()
    if len(active_bal_ctys) >= 2:
        act_bal = summarize(fit(active_bal),
                            "ACTIVE EXPANSION (balanced at max per-country obs)",
                            active_bal)
    else:
        act_bal = None

    write_checkin("balanced_panel.json", {
        "notes": (
            "Balanced-panel robustness. 'Balanced' (full) = countries "
            f"observed in all {N_EXPECTED_PERIODS} cohort years "
            f"{CHILD_COHORTS[0]}..{CHILD_COHORTS[-1]}. "
            "'Balanced' (active expansion) = countries with the maximum "
            "number of below-30% obs in the active-expansion sample "
            "(the deepest-observed subset, since balance by strict equality "
            "is not identified once the cutoff is applied). Country FE, "
            "country-clustered SEs."
        ),
        "results": {
            "full_unbalanced":   full,
            "full_balanced":     bal,
            "active_unbalanced": act_full,
            "active_balanced_max": act_bal,
        },
        "numbers": {
            "full_unbal_beta": full["parent_beta"],
            "full_unbal_n":    full["n"],
            "full_unbal_countries": full["countries"],
            "full_bal_beta":   bal["parent_beta"],
            "full_bal_n":      bal["n"],
            "full_bal_countries": bal["countries"],
            "active_unbal_beta": act_full["parent_beta"],
            "active_unbal_n":    act_full["n"],
            "active_unbal_countries": act_full["countries"],
            "active_bal_max_beta": (act_bal or {}).get("parent_beta"),
            "active_bal_max_n":    (act_bal or {}).get("n"),
            "active_bal_max_countries": (act_bal or {}).get("countries"),
        },
    }, script_path="scripts/robustness/balanced_panel.py")
    print("\nDone.")


if __name__ == "__main__":
    main()
