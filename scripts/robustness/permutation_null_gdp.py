"""
permutation_null_gdp.py

PAPER ANCHOR — Chapter 9 §GDP Has No Independent Effect
  Question: could common trends, panel autocorrelation, or serial
            correlation generate the residualised-GDP coefficient on
            outcomes by chance, even when its real value is near zero?
  Produces: paper claim that the real residualised-GDP β on each of
            four outcomes (log LE, log TFR, log U5MR, child education)
            sits inside the permutation null distribution — Z within
            ~1 SD, p well above 0.05 — confirming the no-signal
            reading directly.
  Checkin:  checkin/permutation_null_gdp.json

Reviewer ask (RA3): the existing permutation_null.py test defends the
intergenerational-transmission β (parent edu → child edu), not §9.8's
question (does GDP have an independent effect once education is
controlled?). The §9.8 test needs its own null, run on the
residualised-GDP regression.

Method (parallel to permutation_null.py):

  Build the common-sample panel from residualization_common_sample.py
  — inner-join of LE and child-education observations at T+28, TFR at
  T+5, and U5MR at T+12, given edu(T) and log GDP(T), entry-cohort
  filter (entry=10, ceiling=90). Apply the same log transforms to LE,
  TFR, U5MR that Table~residualisation uses.

  Residualise log GDP on lower-secondary education with country FE
  (Frisch-Waugh-Lovell) → GDP_resid, orthogonal to education within
  country. The two-step setup matches Panel B of Table~residualisation.

  For each outcome, compute the real β_GDP-resid in the country-FE
  regression outcome ~ edu + GDP_resid. Because GDP_resid is
  orthogonal to edu by construction, this equals
  cov(y_dm, GDP_resid) / var(GDP_resid) where _dm is country-demeaning.

  Two permutation schemes parallel to permutation_null.py:

    1. Within-year shuffle: permute GDP_resid among countries within
       the same year. Preserves any year-level distribution of GDP
       residuals; breaks the country-year match to the outcome.

    2. Full shuffle: permute GDP_resid across all country-years —
       breaks every systematic link.

A genuinely null coefficient (the §9.8 finding for log LE, log TFR,
log U5MR) sits inside the permutation null distribution. The headline
education coefficient on the same outcome — if also tested by
permutation — sits many SDs above its own null (see permutation_null.py
for the intergenerational version).

Output: checkin/permutation_null_gdp.json with the real β, null mean,
null SD, null range, Z, and two-sided p-value for each outcome × scheme.
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
RESID_DIR = os.path.join(SCRIPTS_DIR, "residualization")
# Put residualization first so its richer _shared shadows scripts/_shared
sys.path.insert(0, RESID_DIR)
sys.path.insert(1, SCRIPTS_DIR)

from _shared import (  # noqa: E402
    load_education,
    load_wb,
    interpolate_to_annual,
    build_panel,
    build_child_edu_panel,
    precompute_entry_years,
    filter_panel,
    fe_residualize_gdp,
    write_checkin,
    LAG_TFR,
    LAG_CHILDREARING,
    LAG_GENERATION,
)

N_ITER = 200
SEED = 2026
T_YEARS = list(range(1960, 1995, 5))
# Per-outcome lag. Each column's "_tpN" suffix names its true forward lag:
# LE/child-edu at T+LAG_GENERATION=28, TFR at T+LAG_TFR=5, U5MR at
# T+LAG_CHILDREARING=12.
OUTCOME_LAG = {
    "le_tp28":    LAG_GENERATION,
    "tfr_tp5":   LAG_TFR,
    "u5mr_tp12":  LAG_CHILDREARING,  # childrearing window
    "ced_tp28":   LAG_GENERATION,  # child education (edu→edu, one-step)
}
ENTRY = 10
CEILING = 90
EDU_COL = "lower_sec"


def beta_resid(y_dm, gdp_resid):
    """Coefficient of y on gdp_resid, both already country-demeaned (or
    in the case of gdp_resid, FE-residualised)."""
    sxx = float(np.dot(gdp_resid, gdp_resid))
    if sxx <= 0.0:
        return np.nan
    return float(np.dot(gdp_resid, y_dm) / sxx)


def country_demean(values, country_codes, n_countries):
    """Subtract country means using bincount — matches _demean_and_filter."""
    counts = np.bincount(country_codes, minlength=n_countries)
    sums = np.bincount(country_codes, weights=values, minlength=n_countries)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return values - means[country_codes]


def summarize(perm_betas, real_beta, label):
    arr = np.asarray(perm_betas)
    null_mean = float(arr.mean())
    null_sd = float(arr.std())
    z = float((real_beta - null_mean) / null_sd) if null_sd > 0 else float("inf")
    # Two-sided p-value: fraction of |null| >= |real - null_mean|
    centered = arr - null_mean
    p_two = float((np.abs(centered) >= abs(real_beta - null_mean)).mean())
    print(f"\n{label}")
    print(f"  Null mean β:    {null_mean: .6e}")
    print(f"  Null SD β:      {null_sd: .6e}")
    print(f"  Null range:     [{arr.min(): .4e}, {arr.max(): .4e}]")
    print(f"  Real β:         {real_beta: .4e}")
    print(f"  Z (two-sided):  {z:+.2f} SDs")
    print(f"  p (two-sided):  {p_two:.4f}")
    return {
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_min": float(arr.min()),
        "null_max": float(arr.max()),
        "z": z,
        "p_two": p_two,
        "n_iter": int(len(arr)),
    }


def build_common_panel():
    """Common-sample panel (n=645, 123 countries) — matches
    residualization_common_sample.py."""
    edu_raw = load_education("completion_both_long.csv")
    le_raw = load_wb("life_expectancy_years.csv")
    tfr_raw = load_wb("children_per_woman_total_fertility.csv")
    u5mr_raw = load_wb("child_mortality_u5.csv")
    gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")

    edu_annual = interpolate_to_annual(edu_raw, EDU_COL)

    # Each column's suffix names its true lag; TFR is at LAG_TFR=5.
    panel_le = build_panel(
        edu_annual, le_raw, gdp_raw, T_YEARS, OUTCOME_LAG["le_tp28"], "le_tp28",
    )
    panel_tfr = build_panel(
        edu_annual, tfr_raw, gdp_raw, T_YEARS, OUTCOME_LAG["tfr_tp5"], "tfr_tp5",
    )
    panel_u5mr = build_panel(
        edu_annual, u5mr_raw, gdp_raw, T_YEARS, OUTCOME_LAG["u5mr_tp12"], "u5mr_tp12",
    )
    panel_ced = build_child_edu_panel(
        edu_annual, gdp_raw, T_YEARS, OUTCOME_LAG["ced_tp28"],
    )
    panel_ced = panel_ced.rename(columns={"child_edu": "ced_tp28"})

    panel_u5mr["log_u5mr_tp12"] = np.log(
        panel_u5mr["u5mr_tp12"].where(panel_u5mr["u5mr_tp12"] > 0)
    )
    panel_le["log_le_tp28"] = np.log(panel_le["le_tp28"].where(panel_le["le_tp28"] > 0))
    panel_tfr["log_tfr_tp5"] = np.log(
        panel_tfr["tfr_tp5"].where(panel_tfr["tfr_tp5"] > 0)
    )

    key = ["country", "t"]
    merged = (
        panel_le[key + ["edu_t", "log_gdp_t", "le_tp28", "log_le_tp28"]]
        .merge(panel_tfr[key + ["tfr_tp5", "log_tfr_tp5"]], on=key, how="inner")
        .merge(panel_u5mr[key + ["u5mr_tp12", "log_u5mr_tp12"]], on=key, how="inner")
        .merge(panel_ced[key + ["ced_tp28"]], on=key, how="inner")
    )

    entry_years = precompute_entry_years(edu_annual)
    cohort = entry_years[ENTRY]
    panel = filter_panel(merged, cohort, CEILING)
    panel = panel.dropna(
        subset=[
            "edu_t",
            "log_gdp_t",
            "u5mr_tp12",
            "log_le_tp28",
            "log_tfr_tp5",
            "log_u5mr_tp12",
            "ced_tp28",
        ]
    ).copy()
    # Post-2000 outcome-year dummy for the level-U5MR exception test
    # (U5MR uses LAG_CHILDREARING=12).
    panel["post2000"] = ((panel["t"] + OUTCOME_LAG["u5mr_tp12"]) >= 2000).astype(int)
    return panel


def fit_interaction(y_dm, gdp_resid, post_dm, inter_dm):
    """Fit y_dm = b1·gdp_resid + b2·post_dm + b3·inter_dm (no intercept;
    all variables already country-demeaned). Return (b1, b2, b3).

    Solves via normal equations on the 3-column design matrix.
    """
    X = np.column_stack([gdp_resid, post_dm, inter_dm])
    # OLS: β = (X'X)^-1 X'y
    XtX = X.T @ X
    Xty = X.T @ y_dm
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan, np.nan])
    return beta


def demean_by_country(values, country_codes, n_countries):
    """Subtract country-specific means (alias of country_demean for clarity)."""
    return country_demean(values, country_codes, n_countries)


def main():
    rng = np.random.default_rng(SEED)

    panel = build_common_panel()
    n_obs = len(panel)
    n_countries = panel["country"].nunique()
    print(
        f"Common-sample panel: n={n_obs}, countries={n_countries}, "
        f"years T={T_YEARS[0]}-{T_YEARS[-1]}, "
        f"lag (TFR={LAG_TFR}, others={LAG_GENERATION})"
    )

    # Build GDP_resid via country-FE FWL
    resid_out = fe_residualize_gdp(panel)
    if resid_out is None:
        raise RuntimeError("residualisation failed")
    panel_r, edu_gdp_r2 = resid_out
    print(f"Edu↔log-GDP within-country R²: {edu_gdp_r2:.3f}")

    # Country codes (used to demean outcomes & permute within year)
    country_codes, country_uniq = pd.factorize(panel_r["country"], sort=False)
    n_c = len(country_uniq)
    year_arr = panel_r["t"].to_numpy()
    gdp_resid = panel_r["gdp_resid"].to_numpy()

    outcomes = [
        ("log_le", "log_le_tp28"),
        ("log_tfr", "log_tfr_tp5"),
        ("log_u5mr", "log_u5mr_tp12"),
        ("child_edu", "ced_tp28"),
    ]

    results = {}
    for name, col in outcomes:
        y = panel_r[col].to_numpy()
        y_dm = country_demean(y, country_codes, n_c)

        real = beta_resid(y_dm, gdp_resid)
        print(f"\n=== {name} === real β_GDP-resid = {real:.6e}")

        # Within-year permutation
        within = []
        for _ in range(N_ITER):
            permuted = np.empty_like(gdp_resid)
            for yr in np.unique(year_arr):
                mask = year_arr == yr
                idx = np.where(mask)[0]
                permuted[idx] = rng.permutation(gdp_resid[idx])
            # Re-demean permuted GDP-resid by country to stay in the
            # within-country regression space
            permuted_dm = country_demean(permuted, country_codes, n_c)
            within.append(beta_resid(y_dm, permuted_dm))
        within_summary = summarize(
            within,
            real,
            f"PERMUTATION 1 — within-year shuffle of GDP_resid ({name})",
        )

        # Full shuffle
        full = []
        for _ in range(N_ITER):
            permuted = rng.permutation(gdp_resid)
            permuted_dm = country_demean(permuted, country_codes, n_c)
            full.append(beta_resid(y_dm, permuted_dm))
        full_summary = summarize(
            full,
            real,
            f"PERMUTATION 2 — full shuffle of GDP_resid ({name})",
        )

        results[name] = {
            "outcome_col": col,
            "real_beta": real,
            "within_year": within_summary,
            "full": full_summary,
        }

    # ── Level U5MR pre/post-2000 (the paper's "one bounded exception") ──
    print("\n" + "=" * 72)
    print("LEVEL U5MR — pre/post-2000 split and interaction permutation")
    print("=" * 72)

    u5mr_level = panel_r["u5mr_tp12"].to_numpy()
    post = panel_r["post2000"].to_numpy().astype(float)
    pre_mask = post == 0
    post_mask = post == 1
    n_pre = int(pre_mask.sum())
    n_post = int(post_mask.sum())
    n_c_pre = int(pd.Series(country_codes[pre_mask]).nunique())
    n_c_post = int(pd.Series(country_codes[post_mask]).nunique())
    print(f"Pre-2000 subsample:  n={n_pre},  countries={n_c_pre}  "
          f"(outcome years 1985-1999)")
    print(f"Post-2000 subsample: n={n_post}, countries={n_c_post}  "
          f"(outcome years 2000-2015)")

    # Country-demean level U5MR on the full common-sample panel; for the
    # subsample tests we country-demean within each period instead
    # (matches the paper's separate pre/post regressions).

    def subsample_resid_perm(mask, label):
        codes_sub = country_codes[mask]
        # Re-factorize so country codes are dense in the sub
        uniq, codes_sub = np.unique(codes_sub, return_inverse=True)
        n_c_sub = len(uniq)
        years_sub = year_arr[mask]
        gdp_sub = gdp_resid[mask]
        # Re-demean gdp_resid within country on the subsample
        gdp_sub_dm = country_demean(gdp_sub, codes_sub, n_c_sub)
        y_sub_dm = country_demean(u5mr_level[mask], codes_sub, n_c_sub)
        real = beta_resid(y_sub_dm, gdp_sub_dm)
        print(f"\n--- {label} (n={int(mask.sum())}) real β = {real:.4f} ---")

        within = []
        for _ in range(N_ITER):
            permuted = np.empty_like(gdp_sub)
            for yr in np.unique(years_sub):
                idx = np.where(years_sub == yr)[0]
                permuted[idx] = rng.permutation(gdp_sub[idx])
            permuted_dm = country_demean(permuted, codes_sub, n_c_sub)
            within.append(beta_resid(y_sub_dm, permuted_dm))
        within_s = summarize(within, real,
                             f"  {label} — within-year shuffle")

        full = []
        for _ in range(N_ITER):
            permuted = rng.permutation(gdp_sub)
            permuted_dm = country_demean(permuted, codes_sub, n_c_sub)
            full.append(beta_resid(y_sub_dm, permuted_dm))
        full_s = summarize(full, real,
                           f"  {label} — full shuffle")
        return {
            "n_obs": int(mask.sum()),
            "n_countries": n_c_sub,
            "real_beta": real,
            "within_year": within_s,
            "full": full_s,
        }

    level_u5mr_results = {
        "pre_2000":  subsample_resid_perm(pre_mask,  "Pre-2000 (T+12 < 2000)"),
        "post_2000": subsample_resid_perm(post_mask, "Post-2000 (T+12 >= 2000)"),
    }

    # ── Formal interaction permutation: residGDP × Post-2000 ─────────────
    # Spec: u5mr_dm = b1·gdp_resid_dm + b2·post_dm + b3·(gdp_resid·post)_dm.
    # b3 is the test of interest (paper: β=+24.0, t=2.75, p=0.006).
    # Permuting gdp_resid recomputes the interaction and refits the
    # 3-column demeaned regression — null distribution for b3.
    print("\n" + "-" * 72)
    print("Formal interaction permutation — residGDP × Post-2000 on level U5MR")
    print("-" * 72)

    post_dm = country_demean(post, country_codes, n_c)
    y_full_dm = country_demean(u5mr_level, country_codes, n_c)
    gdp_resid_dm = country_demean(gdp_resid, country_codes, n_c)
    inter_real = gdp_resid * post
    inter_real_dm = country_demean(inter_real, country_codes, n_c)
    real_betas = fit_interaction(y_full_dm, gdp_resid_dm, post_dm, inter_real_dm)
    real_interaction = float(real_betas[2])
    real_pre_slope = float(real_betas[0])
    print(f"Real β_GDP-resid (pre-2000 slope): {real_pre_slope:+.4f}")
    print(f"Real β_interaction (Δ slope post-2000): {real_interaction:+.4f}")

    def interaction_null(scheme):
        out = []
        for _ in range(N_ITER):
            if scheme == "within_year":
                permuted = np.empty_like(gdp_resid)
                for yr in np.unique(year_arr):
                    idx = np.where(year_arr == yr)[0]
                    permuted[idx] = rng.permutation(gdp_resid[idx])
            else:
                permuted = rng.permutation(gdp_resid)
            permuted_dm = country_demean(permuted, country_codes, n_c)
            inter_perm = permuted * post
            inter_perm_dm = country_demean(inter_perm, country_codes, n_c)
            betas = fit_interaction(y_full_dm, permuted_dm, post_dm, inter_perm_dm)
            out.append(float(betas[2]))
        return out

    inter_within = interaction_null("within_year")
    inter_within_s = summarize(
        inter_within, real_interaction,
        "  Interaction null — within-year shuffle of GDP_resid",
    )
    inter_full = interaction_null("full")
    inter_full_s = summarize(
        inter_full, real_interaction,
        "  Interaction null — full shuffle of GDP_resid",
    )

    level_u5mr_results["interaction"] = {
        "spec": (
            "u5mr_level_dm ~ gdp_resid_dm + post2000_dm "
            "+ (gdp_resid x post2000)_dm; country FE via demeaning. "
            "Permutes gdp_resid then recomputes the interaction term."
        ),
        "n_obs": int(n_obs),
        "n_countries": int(n_countries),
        "real_beta_interaction": real_interaction,
        "real_beta_pre2000": real_pre_slope,
        "within_year": inter_within_s,
        "full": inter_full_s,
    }

    # ── Baseline-controlled interaction ─────────────────────────────────
    # Pre-2000 cells sit at systematically lower parental-completion
    # baselines than post-2000 cells (median 28.6% vs 38.2% in the n=645
    # common-sample panel). The unadjusted interaction therefore conflates
    # a temporal shift (GAVI/donor interventions kicked in post-2000)
    # with a baseline-composition shift (post-2000 active-expansion
    # countries are further along in their expansion).
    #
    # Spec adds parental-completion baseline (edu_t) and edu_t × post2000:
    #   u5mr_dm = b1·gdp_resid_dm + b2·post_dm + b3·(gdp_resid × post)_dm
    #           + b4·edu_dm + b5·(edu × post)_dm
    # If b3 survives the addition of b4/b5 with similar magnitude, the
    # GAVI interpretation holds independent of baseline composition.
    # If b3 collapses, the apparent "post-2000 effect" was partly just
    # "post-2000 cells have higher parental-education baselines."
    print("\n" + "-" * 72)
    print("Baseline-controlled interaction — adds edu_t + edu_t × Post-2000")
    print("-" * 72)

    edu = panel_r["edu_t"].to_numpy()
    edu_dm = country_demean(edu, country_codes, n_c)
    edu_x_post = edu * post
    edu_x_post_dm = country_demean(edu_x_post, country_codes, n_c)

    def fit_ols_5col(y_dm_, X_cols):
        """OLS with no intercept on demeaned regressors. Returns betas."""
        X = np.column_stack(X_cols)
        XtX = X.T @ X
        Xty = X.T @ y_dm_
        try:
            return np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            return np.full(len(X_cols), np.nan)

    real_5 = fit_ols_5col(
        y_full_dm,
        [gdp_resid_dm, post_dm, inter_real_dm, edu_dm, edu_x_post_dm],
    )
    real_b3_ctrl = float(real_5[2])
    coef_names = ["gdp_resid", "post2000", "gdp_resid × post", "edu_t", "edu_t × post"]
    print("Real coefficients (5-column spec):")
    for name, b in zip(coef_names, real_5):
        print(f"  {name:<22} {float(b):+.4f}")
    print(f"Coefficient of interest β_(gdp_resid × post) = {real_b3_ctrl:+.4f}")
    print(f"  (uncontrolled was {real_interaction:+.4f} — Δ = {real_b3_ctrl - real_interaction:+.4f})")

    def baseline_ctrl_null(scheme):
        out = []
        for _ in range(N_ITER):
            if scheme == "within_year":
                permuted = np.empty_like(gdp_resid)
                for yr in np.unique(year_arr):
                    idx = np.where(year_arr == yr)[0]
                    permuted[idx] = rng.permutation(gdp_resid[idx])
            else:
                permuted = rng.permutation(gdp_resid)
            permuted_dm = country_demean(permuted, country_codes, n_c)
            inter_perm = permuted * post
            inter_perm_dm = country_demean(inter_perm, country_codes, n_c)
            betas = fit_ols_5col(
                y_full_dm,
                [permuted_dm, post_dm, inter_perm_dm, edu_dm, edu_x_post_dm],
            )
            out.append(float(betas[2]))
        return out

    ctrl_within = baseline_ctrl_null("within_year")
    ctrl_within_s = summarize(
        ctrl_within, real_b3_ctrl,
        "  Baseline-controlled interaction — within-year shuffle",
    )
    ctrl_full = baseline_ctrl_null("full")
    ctrl_full_s = summarize(
        ctrl_full, real_b3_ctrl,
        "  Baseline-controlled interaction — full shuffle",
    )

    level_u5mr_results["interaction_baseline_controlled"] = {
        "spec": (
            "u5mr_level_dm ~ gdp_resid_dm + post2000_dm "
            "+ (gdp_resid x post2000)_dm + edu_dm + (edu x post2000)_dm; "
            "country FE via demeaning. Adds parental-completion baseline "
            "and its interaction with post-2000 to disentangle the "
            "temporal shift from the active-expansion baseline shift."
        ),
        "n_obs": int(n_obs),
        "n_countries": int(n_countries),
        "real_coefficients": {
            name: float(b) for name, b in zip(coef_names, real_5)
        },
        "real_beta_interaction_controlled": real_b3_ctrl,
        "real_beta_interaction_uncontrolled": real_interaction,
        "delta_from_uncontrolled": real_b3_ctrl - real_interaction,
        "within_year": ctrl_within_s,
        "full": ctrl_full_s,
    }

    checkin = {
        "method": (
            "Permutation null for the residualised-GDP coefficient. "
            "Common-sample panel (inner-join of LE and child edu at "
            "T+LAG_GENERATION=28, TFR at T+LAG_TFR=5, and U5MR at "
            "T+LAG_CHILDREARING=12, with edu(T) and "
            "log GDP(T); entry=10, ceiling=90). "
            "log GDP residualised against lower-secondary education with "
            "country FE (Frisch-Waugh-Lovell). For each outcome, the real "
            "coefficient β_GDP-resid is the country-FE regression of the "
            "outcome on (edu, GDP_resid). Two permutation schemes: "
            "within-year shuffle preserves year-level GDP_resid "
            "distribution; full shuffle breaks every systematic link. "
            "Z and p-values are two-sided against each null. "
            "level_u5mr block tests the paper's 'one bounded exception' "
            "(GAVI/donor child-health interventions, post-2000) on "
            "raw U5MR levels: pre/post-2000 subsamples plus formal "
            "interaction residGDP x Post-2000."
        ),
        "n_obs": int(n_obs),
        "n_countries": int(n_countries),
        "n_iter": N_ITER,
        "seed": SEED,
        "outcome_lag": OUTCOME_LAG,
        "lag_tfr": LAG_TFR,
        "lag_childrearing": LAG_CHILDREARING,
        "lag_generation": LAG_GENERATION,
        "t_years": [int(t) for t in T_YEARS],
        "entry_threshold": ENTRY,
        "ceiling": CEILING,
        "edu_gdp_within_r2": float(edu_gdp_r2),
        "outcomes": results,
        "level_u5mr": level_u5mr_results,
    }

    write_checkin(
        "permutation_null_gdp.json",
        checkin,
        script_path="scripts/robustness/permutation_null_gdp.py",
    )

    # Headline summary
    print("\n" + "=" * 72)
    print("SUMMARY — real β_GDP-resid vs permutation nulls")
    print("=" * 72)
    print(f"{'Outcome':<14} {'Real β':>12} {'Within-yr Z':>12} {'p':>8} "
          f"{'Full Z':>10} {'p':>8}")
    for name, r in results.items():
        print(
            f"{name:<14} {r['real_beta']:>12.4e} "
            f"{r['within_year']['z']:>+12.2f} {r['within_year']['p_two']:>8.3f} "
            f"{r['full']['z']:>+10.2f} {r['full']['p_two']:>8.3f}"
        )

    print()
    print("LEVEL U5MR — pre/post-2000 split + interaction:")
    print(f"{'Block':<22} {'Real β':>12} {'Within-yr Z':>12} {'p':>8} "
          f"{'Full Z':>10} {'p':>8}")
    for block in ["pre_2000", "post_2000"]:
        r = level_u5mr_results[block]
        print(
            f"{block:<22} {r['real_beta']:>12.4f} "
            f"{r['within_year']['z']:>+12.2f} {r['within_year']['p_two']:>8.3f} "
            f"{r['full']['z']:>+10.2f} {r['full']['p_two']:>8.3f}"
        )
    ri = level_u5mr_results["interaction"]
    print(
        f"{'interaction':<22} {ri['real_beta_interaction']:>12.4f} "
        f"{ri['within_year']['z']:>+12.2f} {ri['within_year']['p_two']:>8.3f} "
        f"{ri['full']['z']:>+10.2f} {ri['full']['p_two']:>8.3f}"
    )
    rc = level_u5mr_results["interaction_baseline_controlled"]
    print(
        f"{'interaction (ctrl)':<22} {rc['real_beta_interaction_controlled']:>12.4f} "
        f"{rc['within_year']['z']:>+12.2f} {rc['within_year']['p_two']:>8.3f} "
        f"{rc['full']['z']:>+10.2f} {rc['full']['p_two']:>8.3f}"
    )


if __name__ == "__main__":
    main()
