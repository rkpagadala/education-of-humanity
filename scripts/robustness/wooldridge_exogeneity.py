# =============================================================================
# PAPER REFERENCE
# Script:  scripts/robustness/wooldridge_exogeneity.py
# Paper:   "Education of Humanity"
#
# Produces:
#   Wooldridge (2010, §10.5) regression-based F-test for strict exogeneity
#   in the country fixed-effects specification. If the error is uncorrelated
#   with the regressors at all leads and lags, the coefficient on a future
#   value of the regressor is zero. Reject => strict exogeneity violated.
#
# Test equation:
#   child_t = beta * parent_{t-25} + gamma * parent_{t-25+5} + alpha_i + u_t
#   H0: gamma = 0  (strict exogeneity)
#
#   parent_{t-25+5} is the lead of the regressor: the parental education
#   value observed five years after the primary regressor period. Because
#   the panel is 5-year spaced, this is the natural lead.
#
# Samples reported:
#   1. Full panel (1665 obs / 185 countries, Table A1 Model 4 sample).
#   2. Active-expansion (parent <30%, matches Table 1).
#
# Output: checkin/wooldridge_exogeneity.json
# =============================================================================
"""Wooldridge regression-based test for strict exogeneity (FE panel)."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from _shared import PROC, DATA, REGIONS, write_checkin  # noqa: E402

PARENTAL_LAG = 25
PERIOD = 5
CHILD_COHORTS = list(range(1975, 2016, PERIOD))
ACTIVE_EXPANSION_CUTOFF = 30


def load_panel():
    """Panel with parent_{t-25}, child_t, and the one-period lead parent_{t-20}."""
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
            parent_yr = ch - PARENTAL_LAG
            child = edu(c, ch)
            parent = edu(c, parent_yr)
            parent_lead = edu(c, parent_yr + PERIOD)
            if any(np.isnan(v) for v in (child, parent, parent_lead)):
                continue
            rows.append({
                "country": c,
                "year": ch,
                "child": child,
                "parent": parent,
                "parent_lead": parent_lead,
            })
    return pd.DataFrame(rows)


def fit(d, x_cols):
    panel = d.set_index(["country", "year"])
    mod = PanelOLS(panel["child"], panel[x_cols],
                   entity_effects=True, drop_absorbed=True, check_rank=False)
    return mod.fit(cov_type="clustered", cluster_entity=True)


def report(label, d):
    base = fit(d, ["parent"])
    test = fit(d, ["parent", "parent_lead"])
    lead_b = float(test.params["parent_lead"])
    lead_se = float(test.std_errors["parent_lead"])
    lead_t = float(test.tstats["parent_lead"])
    lead_p = float(test.pvalues["parent_lead"])
    f_stat = lead_t ** 2  # single-restriction F = t²; cluster-robust
    n = int(test.nobs)
    n_ctry = int(test.entity_info.total)
    print(f"\n{label}: n={n}, countries={n_ctry}")
    print(f"  β(parent)         = {float(base.params['parent']):.3f} "
          f"(SE {float(base.std_errors['parent']):.3f})   "
          f"[baseline, no lead]")
    print(f"  β(parent|+lead)   = {float(test.params['parent']):.3f} "
          f"(SE {float(test.std_errors['parent']):.3f})")
    print(f"  γ(parent_lead)    = {lead_b:.3f} "
          f"(SE {lead_se:.3f}, t={lead_t:.3f}, p={lead_p:.4f})")
    reject = lead_p < 0.05
    print(f"  Wooldridge F(1,C-1) ≈ t² = {f_stat:.3f}   "
          f"H0 strict exog {'REJECTED' if reject else 'NOT rejected'} at 5%")
    return {
        "n": n,
        "countries": n_ctry,
        "parent_beta_baseline": round(float(base.params["parent"]), 3),
        "parent_se_baseline":   round(float(base.std_errors["parent"]), 3),
        "parent_beta_with_lead": round(float(test.params["parent"]), 3),
        "parent_se_with_lead":   round(float(test.std_errors["parent"]), 3),
        "lead_beta": round(lead_b, 3),
        "lead_se":   round(lead_se, 3),
        "lead_t":    round(lead_t, 3),
        "lead_p":    round(lead_p, 4),
        "f_stat":    round(f_stat, 3),
        "strict_exog_rejected_5pct": bool(reject),
    }


def main():
    panel = load_panel()
    print(f"Raw panel (with lead): {len(panel)} obs, "
          f"{panel['country'].nunique()} countries")
    full = report("FULL PANEL", panel)
    active = panel[panel["parent"] < ACTIVE_EXPANSION_CUTOFF].copy()
    # need ≥2 obs per country for within-FE after lead restriction
    counts = active.groupby("country").size()
    active = active[active["country"].isin(counts[counts >= 2].index)].copy()
    active_res = report(f"ACTIVE EXPANSION (parent<{ACTIVE_EXPANSION_CUTOFF}%)", active)

    write_checkin("wooldridge_exogeneity.json", {
        "notes": (
            "Wooldridge (2010, §10.5) regression-based F-test for strict "
            "exogeneity. Add parent education at t-20 (one-period lead of "
            "parent_{t-25}) to the FE regression; test H0: γ=0. "
            "Non-rejection is consistent with strict exogeneity. "
            "5-year panel, country FE, country-clustered SEs."
        ),
        "results": {"full_panel": full, "active_expansion": active_res},
        "numbers": {
            "full_n": full["n"],
            "full_countries": full["countries"],
            "full_lead_beta": full["lead_beta"],
            "full_lead_se":   full["lead_se"],
            "full_lead_p":    full["lead_p"],
            "full_f_stat":    full["f_stat"],
            "full_strict_exog_rejected": int(full["strict_exog_rejected_5pct"]),
            "active_n": active_res["n"],
            "active_countries": active_res["countries"],
            "active_lead_beta": active_res["lead_beta"],
            "active_lead_se":   active_res["lead_se"],
            "active_lead_p":    active_res["lead_p"],
            "active_f_stat":    active_res["f_stat"],
            "active_strict_exog_rejected": int(active_res["strict_exog_rejected_5pct"]),
        },
    }, script_path="scripts/robustness/wooldridge_exogeneity.py")
    print("\nDone.")


if __name__ == "__main__":
    main()
