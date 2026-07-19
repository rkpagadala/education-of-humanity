"""
permutation_null_symmetric.py

PAPER ANCHOR — Chapter 9 §GDP Has No Independent Effect
  Question: run education and GDP through one identical permutation-null
            harness, side by side, so the asymmetry is direct: education's
            relationship to each outcome is real (many SDs above its null)
            and survives stripping GDP out; GDP's relationship survives
            only as long as it carries education, and collapses into its
            null once education is stripped out.
  Produces: the symmetric head-to-head the residualisation table asserts
            but does not test by permutation — "education sits N SDs above
            its null; GDP sits inside its."
  Checkin:  checkin/permutation_null_symmetric.json

Why this and not a raw-vs-raw test:

  Raw GDP, run alone through the null, sits ABOVE its null — because GDP is
  downstream of education and carries education's signal. A raw-vs-raw
  head-to-head therefore shows only the naive fact that both predict
  outcomes, which says nothing about which is doing the work. The
  asymmetry appears only under mutual residualisation (Frisch-Waugh-Lovell,
  country FE): strip GDP out of education and education survives; strip
  education out of GDP and GDP vanishes.

  This script runs FOUR predictor variants for each outcome, through the
  same two permutation schemes as permutation_null_gdp.py, so the full
  story is visible end to end:

    raw_edu     — education, country-demeaned                 (above null)
    raw_gdp     — log GDP, country-demeaned                   (above null:
                                                               carries edu)
    edu_resid   — education residualised against log GDP      (above null:
                  within country (edu ⊥ GDP)                   survives)
    gdp_resid   — log GDP residualised against education      (inside null:
                  within country (GDP ⊥ edu)                   vanishes)

  gdp_resid here reproduces permutation_null_gdp.py's residualised-GDP
  result, so its Z-scores serve as an internal consistency check against
  that script. edu_resid is the symmetric counterpart that script does not
  compute.

Method (parallel to permutation_null_gdp.py):

  Common-sample panel (inner-join of LE/child-edu at T+LAG_GENERATION=28,
  TFR at T+LAG_TFR=5, U5MR at T+LAG_CHILDREARING=12, with edu(T) and
  log GDP(T); entry-cohort filter entry=10, ceiling=90). Outcomes in logs
  (LE, TFR, U5MR) and levels (child education), matching the
  residualisation table.

  For each (outcome, predictor) the real β is the within-country slope of
  the country-demeaned outcome on the predictor. For the two residualised
  predictors the predictor is the FWL residual (orthogonal to the other
  regressor within country), so the slope equals
  cov(y_dm, predictor) / var(predictor).

  Two permutation schemes, identical to permutation_null_gdp.py:
    1. Full shuffle: permute the predictor across all country-years —
       breaks every systematic link. This is the PRIMARY null here: it
       treats all four predictors on equal footing, and its Z-scores track
       the within-country regression t-statistics (full Z ≈ t·√(1−R²)).
    2. Within-year shuffle: permute among countries within the same year —
       preserves the global time trend. This is the wrong null for the
       trended raw predictors (education rises monotonically within every
       country), because it absorbs the very within-country education wave
       that IS the channel — the same objection the paper raises against
       year fixed effects. It deflates the Z of any trended predictor
       (raw_edu, edu_resid) toward zero while leaving the de-trended
       gdp_resid unchanged, so it understates education and is reported for
       completeness only, not as the head-to-head null.

Output: checkin/permutation_null_symmetric.json with the real β, null
mean, null SD, null range, Z, and two-sided p-value for each
outcome × predictor × scheme.
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
RESID_DIR = os.path.join(SCRIPTS_DIR, "residualization")
# Put residualization first so its richer _shared shadows scripts/_shared,
# matching permutation_null_gdp.py's import setup.
sys.path.insert(0, RESID_DIR)
sys.path.insert(1, SCRIPTS_DIR)
sys.path.insert(2, SCRIPT_DIR)

from _shared import write_checkin, LAG_TFR, LAG_CHILDREARING, LAG_GENERATION  # noqa: E402

# Reuse the exact panel builder and null harness from the GDP-only script
# so the two scripts share one common-sample panel and one summary method.
from permutation_null_gdp import (  # noqa: E402
    build_common_panel,
    country_demean,
    beta_resid,
    summarize,
    N_ITER,
    SEED,
    OUTCOME_LAG,
)

# Outcome column → predictor sign convention is irrelevant here; we report
# the Z-score magnitude of the real β against each permutation null.
OUTCOMES = [
    ("log_le", "log_le_tp28"),
    ("log_tfr", "log_tfr_tp5"),
    ("log_u5mr", "log_u5mr_tp12"),
    ("child_edu", "ced_tp28"),
]


def fe_residualize(target, control, country_codes, n_c):
    """Within-country Frisch-Waugh-Lovell residual of `target` on `control`.

    Both series are first country-demeaned (country FE); `target_dm` is then
    regressed on `control_dm` and the residual returned. The residual is
    orthogonal to `control` within country and is itself country-demeaned.
    Also returns the within-country R^2 of `control` explaining `target`.
    """
    t_dm = country_demean(target, country_codes, n_c)
    c_dm = country_demean(control, country_codes, n_c)
    sxx = float(np.dot(c_dm, c_dm))
    if sxx <= 0.0:
        return t_dm.copy(), 0.0
    b = float(np.dot(c_dm, t_dm) / sxx)
    resid = t_dm - b * c_dm
    ss_tot = float(np.dot(t_dm, t_dm))
    r2 = 1.0 - float(np.dot(resid, resid)) / ss_tot if ss_tot > 0.0 else 0.0
    return resid, r2


def permute_betas(y_dm, predictor, country_codes, n_c, year_arr, rng, scheme, n_iter):
    """Null distribution of the within-country slope of y_dm on a shuffled
    predictor. Matches permutation_null_gdp.py: after shuffling, re-demean
    by country to stay in the within-country regression space."""
    out = []
    years = np.unique(year_arr)
    for _ in range(n_iter):
        if scheme == "within_year":
            permuted = np.empty_like(predictor)
            for yr in years:
                idx = np.where(year_arr == yr)[0]
                permuted[idx] = rng.permutation(predictor[idx])
        else:
            permuted = rng.permutation(predictor)
        permuted_dm = country_demean(permuted, country_codes, n_c)
        out.append(beta_resid(y_dm, permuted_dm))
    return out


def main():
    rng = np.random.default_rng(SEED)

    panel = build_common_panel()
    # Drop singleton countries (one observation): under country FE their
    # demeaned value is identically zero, so they contribute nothing to any
    # within-country slope or to the permutation null. Removing them leaves
    # every coefficient and Z unchanged and makes the reported sample equal
    # the estimation sample of Table~residualisation (n=645, 123 countries)
    # rather than the merged-panel header (n=652, 130).
    obs_per_country = panel["country"].value_counts()
    keep = obs_per_country[obs_per_country > 1].index
    panel = panel[panel["country"].isin(keep)].copy()
    n_obs = len(panel)
    n_countries = panel["country"].nunique()
    print(
        f"Common-sample panel: n={n_obs}, countries={n_countries}, "
        f"lag (TFR={LAG_TFR}, U5MR={LAG_CHILDREARING}, others={LAG_GENERATION})"
    )

    country_codes, country_uniq = pd.factorize(panel["country"], sort=False)
    n_c = len(country_uniq)
    year_arr = panel["t"].to_numpy()

    edu = panel["edu_t"].to_numpy(dtype=float)
    log_gdp = panel["log_gdp_t"].to_numpy(dtype=float)

    # Four predictor variants (all country-demeaned).
    raw_edu_dm = country_demean(edu, country_codes, n_c)
    raw_gdp_dm = country_demean(log_gdp, country_codes, n_c)
    edu_resid, edu_on_gdp_r2 = fe_residualize(edu, log_gdp, country_codes, n_c)
    gdp_resid, gdp_on_edu_r2 = fe_residualize(log_gdp, edu, country_codes, n_c)

    print(f"Within-country R²: GDP→edu {edu_on_gdp_r2:.3f}, "
          f"edu→GDP {gdp_on_edu_r2:.3f}")

    predictors = [
        ("raw_edu",   raw_edu_dm,  "education, country-demeaned"),
        ("raw_gdp",   raw_gdp_dm,  "log GDP, country-demeaned"),
        ("edu_resid", edu_resid,   "education residualised against log GDP (edu ⊥ GDP)"),
        ("gdp_resid", gdp_resid,   "log GDP residualised against education (GDP ⊥ edu)"),
    ]

    results = {}
    for oname, ocol in OUTCOMES:
        y = panel[ocol].to_numpy(dtype=float)
        y_dm = country_demean(y, country_codes, n_c)
        print("\n" + "=" * 72)
        print(f"OUTCOME: {oname}")
        print("=" * 72)

        results[oname] = {"outcome_col": ocol, "predictors": {}}
        for pname, pvec, pdesc in predictors:
            real = beta_resid(y_dm, pvec)
            within = summarize(
                permute_betas(y_dm, pvec, country_codes, n_c, year_arr, rng,
                              "within_year", N_ITER),
                real,
                f"  {oname} ~ {pname} — within-year shuffle",
            )
            full = summarize(
                permute_betas(y_dm, pvec, country_codes, n_c, year_arr, rng,
                              "full", N_ITER),
                real,
                f"  {oname} ~ {pname} — full shuffle",
            )
            results[oname]["predictors"][pname] = {
                "description": pdesc,
                "real_beta": real,
                "within_year": within,
                "full": full,
            }

    # Full-shuffle |Z| bounds across the four outcomes, per predictor —
    # the rounded ranges cited in the paper's permutation-null sentence
    # (raw education 14–18, education⊥GDP 11–13, residualised GDP within 2).
    def abs_z(pred):
        return [abs(results[o]["predictors"][pred]["full"]["z"]) for o, _ in OUTCOMES]

    full_shuffle_abs_z_bounds = {
        p: {"min": float(min(abs_z(p))), "max": float(max(abs_z(p)))}
        for p in ("raw_edu", "raw_gdp", "edu_resid", "gdp_resid")
    }

    checkin = {
        "method": (
            "Symmetric permutation null. One harness, four predictor "
            "variants per outcome (raw education, raw log GDP, education "
            "residualised against GDP, GDP residualised against education; "
            "all country-demeaned, residuals via within-country FWL). "
            "Common-sample panel (inner-join of LE and child edu at "
            "T+LAG_GENERATION=28, TFR at T+LAG_TFR=5, U5MR at "
            "T+LAG_CHILDREARING=12, with edu(T) and log GDP(T); entry=10, "
            "ceiling=90). For each (outcome, predictor) the real coefficient "
            "is the within-country slope of the demeaned outcome on the "
            "predictor. Two permutation schemes: within-year shuffle "
            "preserves year-level predictor distribution; full shuffle "
            "breaks every systematic link. Z and p-values are two-sided "
            "against each null. gdp_resid reproduces permutation_null_gdp.py "
            "as an internal consistency check; edu_resid is its symmetric "
            "counterpart."
        ),
        "n_obs": int(n_obs),
        "n_countries": int(n_countries),
        "n_iter": N_ITER,
        "seed": SEED,
        "outcome_lag": OUTCOME_LAG,
        "lag_tfr": LAG_TFR,
        "lag_childrearing": LAG_CHILDREARING,
        "lag_generation": LAG_GENERATION,
        "within_r2_gdp_explains_edu": float(edu_on_gdp_r2),
        "within_r2_edu_explains_gdp": float(gdp_on_edu_r2),
        "full_shuffle_abs_z_bounds": full_shuffle_abs_z_bounds,
        "outcomes": results,
    }

    write_checkin(
        "permutation_null_symmetric.json",
        checkin,
        script_path="scripts/robustness/permutation_null_symmetric.py",
    )

    # Headline table — FULL-shuffle Z by outcome × predictor (primary null).
    def print_table(scheme, title):
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        print(f"{'Outcome':<11}" + "".join(f"{p:>13}" for p, _, _ in predictors))
        for oname, _ in OUTCOMES:
            row = f"{oname:<11}"
            for pname, _, _ in predictors:
                z = results[oname]["predictors"][pname][scheme]["z"]
                row += f"{z:>+13.1f}"
            print(row)

    print_table("full", "SUMMARY — full-shuffle Z (PRIMARY null; SDs vs each null)")
    print("\nEducation towers above its null (raw_edu) and stays there after "
          "GDP is stripped out (edu_resid); GDP, once education is stripped "
          "out (gdp_resid), sits inside its null (|Z|<2) for every outcome.")
    print_table("within_year",
                "within-year-shuffle Z (trend-preserving; understates "
                "trended predictors — see docstring)")


if __name__ == "__main__":
    main()
