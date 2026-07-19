"""
female_vs_both_panel.py
=======================
Life-history label (§the-generational-lag):
  Predictors: female 20-24 cohort vs male 20-24 cohort completion
  at T. Outcomes at T+28 (cross-generation step). Use: identification
  of the maternal channel vs the household-channel. Female and male
  completion correlate ~0.9 within country; the horse race tests
  whether female-specific signal survives.

Does female schooling specifically drive T+28 outcomes, beyond what
male schooling drives at the same point? Horse-race regression:

    outcome(T+28) = β_F · female_lower_sec(T) + β_M · male_lower_sec(T)
                    + α_country + ε

with country FE, country-clustered SEs, entry-cohort design.

A raw β_female vs β_both comparison on the *same* (country, t) panel is
misleading: in early-transition rows male completion is much higher than
female completion, so the both-sexes measure at T is dominated by men
already schooled, while the development outcome at T+28 is produced by
female cohorts schooled *between* T and T+28. The horse race separates
which sex's schooling carries the within-country prediction once both
are in the regression. It also reports standardised (z-score) betas so
the comparison does not depend on the different ranges of the two
measures (female spans a wider 0-90 range; male a narrower 30-90 range
during the transition window).

Paper hooks: §gdp-has-no-independent-effect ("Female education produces
stronger results than the aggregate on every outcome").

Output: checkin/female_vs_both_panel.json
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from _shared import (
    load_education, load_wb, interpolate_to_annual, write_checkin,
    LAG_TFR, LAG_CHILDREARING, LAG_GENERATION,
)
from residualization._shared import (
    build_panel, precompute_entry_years, filter_panel, clustered_fe,
)

T_YEARS = list(range(1960, 1995, 5))
# Outcome-specific lag: TFR uses biological parent timing (LAG_TFR=5;
# cohort age 20-24 at T is at the reproductive peak by T+5). U5MR uses the
# childrearing window (LAG_CHILDREARING=12). LE keeps the time-to-agency lag
# (LAG_GENERATION=28).
OUTCOME_LAG = {
    "le":   LAG_GENERATION,
    "tfr":  LAG_TFR,
    "u5mr": LAG_CHILDREARING,
}
ENTRY = 10
CEILING = 90

print("Loading education (both-sexes, female, male) and outcomes...")
edu_both = load_education("completion_both_long.csv")
edu_female = load_education("completion_female_long.csv")
edu_male = load_education("completion_male_long.csv")

gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
le_df = load_wb("life_expectancy_years.csv")
tfr_df = load_wb("children_per_woman_total_fertility.csv")
u5mr_df = load_wb("child_mortality_u5.csv")

edu_both_annual = interpolate_to_annual(edu_both, "lower_sec")
edu_female_annual = interpolate_to_annual(edu_female, "lower_sec")
edu_male_annual = interpolate_to_annual(edu_male, "lower_sec")

OUTCOMES = {
    "LE": ("le", le_df),
    "TFR": ("tfr", tfr_df),
    "U5MR": ("u5mr", u5mr_df),
}


def build_paired_panel(outcome_col, outcome_df):
    """Single panel with both-sexes, female, and male edu on the same rows."""
    lag = OUTCOME_LAG[outcome_col]
    both_panel = build_panel(
        edu_both_annual, outcome_df, gdp_df, T_YEARS, lag, outcome_col,
    )
    entry_both = precompute_entry_years(
        edu_both_annual, thresholds=[ENTRY],
    )[ENTRY]
    sub = filter_panel(both_panel, entry_both, CEILING)

    female_panel = build_panel(
        edu_female_annual, outcome_df, gdp_df, T_YEARS, lag, outcome_col,
    )
    male_panel = build_panel(
        edu_male_annual, outcome_df, gdp_df, T_YEARS, lag, outcome_col,
    )
    fl = female_panel.set_index(["country", "t"])["edu_t"].to_dict()
    ml = male_panel.set_index(["country", "t"])["edu_t"].to_dict()
    keys = list(zip(sub["country"], sub["t"]))
    sub = sub.copy()
    sub["edu_t_both"] = sub["edu_t"]
    sub["edu_t_female"] = [fl.get(k, np.nan) for k in keys]
    sub["edu_t_male"] = [ml.get(k, np.nan) for k in keys]
    sub = sub.dropna(
        subset=["edu_t_both", "edu_t_female", "edu_t_male", outcome_col],
    )
    return sub


def demean_by_country(panel, cols):
    """Within-country demean each column. Returns a new DataFrame."""
    out = panel.copy()
    for c in cols:
        out[c + "_dm"] = out[c] - out.groupby("country")[c].transform("mean")
    return out


def horse_race(panel, outcome_col):
    """outcome ~ female + male (country FE, country-clustered SEs).

    Also runs single-regressor sex-specific fits (female alone, male alone)
    on the same panel for comparison.
    """
    d = demean_by_country(panel, ["edu_t_female", "edu_t_male", outcome_col])
    X = d[["edu_t_female_dm", "edu_t_male_dm"]].values
    y = d[outcome_col + "_dm"].values
    m = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": d["country"].values},
    )
    horse = {
        "beta_female": float(m.params[0]),
        "se_female": float(m.bse[0]),
        "pval_female": float(m.pvalues[0]),
        "beta_male": float(m.params[1]),
        "se_male": float(m.bse[1]),
        "pval_male": float(m.pvalues[1]),
        "r2": float(m.rsquared),
        "n": int(len(d)),
        "countries": int(d["country"].nunique()),
    }
    # Single-regressor fits (β per pp).
    female_only = clustered_fe("edu_t_female", outcome_col, panel)
    male_only = clustered_fe("edu_t_male", outcome_col, panel)
    both_only = clustered_fe("edu_t_both", outcome_col, panel)
    # Standardised (z-score) single-regressor betas: divide β by within-country
    # SD of the regressor and multiply by within-country SD of outcome → effect
    # in outcome-SDs per 1-SD-of-edu change.
    def z_beta(col):
        wx = panel[col] - panel.groupby("country")[col].transform("mean")
        wy = panel[outcome_col] - panel.groupby("country")[outcome_col].transform("mean")
        sx = wx.std(ddof=1)
        sy = wy.std(ddof=1)
        if sx == 0 or sy == 0:
            return None
        result = clustered_fe(col, outcome_col, panel)
        if result is None:
            return None
        return float(result["beta"] * sx / sy)
    return {
        "horse_race": horse,
        "female_only": {
            "beta": float(female_only["beta"]),
            "se": float(female_only["se"]),
            "pval": float(female_only["pval"]),
            "r2": float(female_only["r2"]),
        },
        "male_only": {
            "beta": float(male_only["beta"]),
            "se": float(male_only["se"]),
            "pval": float(male_only["pval"]),
            "r2": float(male_only["r2"]),
        },
        "both_only": {
            "beta": float(both_only["beta"]),
            "se": float(both_only["se"]),
            "pval": float(both_only["pval"]),
            "r2": float(both_only["r2"]),
        },
        "standardised_beta_female": z_beta("edu_t_female"),
        "standardised_beta_male": z_beta("edu_t_male"),
        "standardised_beta_both": z_beta("edu_t_both"),
    }


results = {}
print(
    f"\n{'Outcome':<6} {'n':>5} {'ctry':>4}  "
    f"{'βF|M':>10} {'pF|M':>6}  "
    f"{'βM|F':>10} {'pM|F':>6}  "
    f"{'βF(z)':>7} {'βM(z)':>7} {'βB(z)':>7}"
)
print("-" * 80)

for label, (col, df) in OUTCOMES.items():
    panel = build_paired_panel(col, df)
    r = horse_race(panel, col)
    h = r["horse_race"]
    print(
        f"{label:<6} {h['n']:>5} {h['countries']:>4}  "
        f"{h['beta_female']:>10.5f} {h['pval_female']:>6.3f}  "
        f"{h['beta_male']:>10.5f} {h['pval_male']:>6.3f}  "
        f"{r['standardised_beta_female']:>7.3f} "
        f"{r['standardised_beta_male']:>7.3f} "
        f"{r['standardised_beta_both']:>7.3f}"
    )
    results[label] = r

# Summary: by what factor does standardised female-beta exceed male-beta
# (in absolute value, since signs differ across outcomes).
summary = {}
for label, r in results.items():
    bf = abs(r["standardised_beta_female"])
    bm = abs(r["standardised_beta_male"])
    bb = abs(r["standardised_beta_both"])
    summary[label] = {
        "z_female_minus_male": round(bf - bm, 3),
        "z_female_over_both": round(bf / bb if bb else float("nan"), 3),
        "female_dominates_horse_race": bool(
            abs(r["horse_race"]["beta_female"]) > abs(r["horse_race"]["beta_male"])
            and r["horse_race"]["pval_female"] < 0.10
        ),
    }
print("\nstandardised |β_female| − |β_male|, by outcome:")
for label, s in summary.items():
    print(f"  {label}: {s['z_female_minus_male']:+.3f} SD "
          f"(female dominates horse race: {s['female_dominates_horse_race']})")

write_checkin(
    "female_vs_both_panel.json",
    {
        "method": (
            "Horse race: outcome(T+lag) ~ female_lower_sec(T) + "
            "male_lower_sec(T), country FE, country-clustered SEs. "
            f"Entry-cohort design (entry>={ENTRY}%, ceiling<={CEILING}% in "
            f"both-sexes lower-sec). T={T_YEARS[0]}-{T_YEARS[-1]}. "
            f"Outcome-specific lag: TFR uses LAG_TFR={LAG_TFR} (biological "
            f"parent timing); U5MR uses LAG_CHILDREARING={LAG_CHILDREARING} "
            f"(childrearing window); LE uses LAG_GENERATION={LAG_GENERATION} "
            "(time-to-agency). Also reports single-regressor fits for female, "
            "male, both, and standardised (z-score) betas on the identical panel."
        ),
        "params": {
            "entry": ENTRY, "ceiling": CEILING,
            "t_start": T_YEARS[0], "t_end": T_YEARS[-1],
            "outcome_lag": OUTCOME_LAG,
        },
        "outcomes": results,
        "summary": summary,
    },
    script_path="scripts/female_vs_both_panel.py",
)
