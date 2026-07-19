"""
robustness/robustness_tests.py
====================
Three econometric robustness tests for the education-development paper.

Test 1: Nickell Bias — Anderson-Hsiao IV vs standard FE for the home-niche regression
Test 2: Nonlinearity in residualization first stage (quadratic education)
Test 3: Bootstrapped CIs on R² comparisons (education vs residualized GDP → LE)

Outputs results to paper/robustness_results.txt
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from _shared import REPO_ROOT, write_checkin, LAG_TFR, LAG_CHILDREARING, LAG_GENERATION
from residualization._shared import (
    load_education, load_wb, get_wb_val, interpolate_to_annual,
    precompute_entry_years, build_panel, filter_panel, fe_r2,
)

# Output buffer
output_lines = []
# Collect values for checkin JSON (first ceiling = 60 is the one cited in paper)
_checkin_numbers = {}

def out(line=""):
    print(line)
    output_lines.append(line)


# ── Load data ────────────────────────────────────────────────────────────

out("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
le_df = load_wb("life_expectancy_years.csv")

COL_NAME = "lower_sec"
T_YEARS = list(range(1960, 1995, 5))
# Per-outcome lag. TFR uses LAG_TFR=5 (biological parent timing);
# U5MR uses LAG_CHILDREARING=12 (childrearing window); LE, child education
# use LAG_GENERATION=28 (time-to-agency / edu→edu).
OUTCOME_LAG = {
    "le":        LAG_GENERATION,
    "tfr":       LAG_TFR,
    "u5mr":      LAG_CHILDREARING,
    "child_edu": LAG_GENERATION,
}
# Default LAG retained for the LE / child-edu blocks built up-front (both
# use LAG_GENERATION); per-outcome lookup is applied inside Test 3.
LAG = LAG_GENERATION

edu_annual = interpolate_to_annual(edu_raw, COL_NAME)
entry_years = precompute_entry_years(edu_annual)

# Build LE panel (same as tables/regression_tables.py)
panel_le = build_panel(edu_annual, le_df, gdp_df, T_YEARS, OUTCOME_LAG["le"], "le")

# Build child-education panel (parent edu → child edu,
# LAG_GENERATION=28-year lag)
ce_lag = OUTCOME_LAG["child_edu"]
rows_ce = []
for c in sorted(edu_annual.keys()):
    s = edu_annual[c]
    for t in T_YEARS:
        if t not in s.index or (t + ce_lag) not in s.index:
            continue
        parent_edu = s[t]
        child_edu = s[t + ce_lag]
        gdp_t = get_wb_val(gdp_df, c, t)
        if np.isnan(parent_edu) or np.isnan(child_edu):
            continue
        rows_ce.append({
            "country": c, "t": t, "edu_t": parent_edu,
            "log_gdp_t": np.log(gdp_t) if not np.isnan(gdp_t) and gdp_t > 0 else np.nan,
            "child_edu": child_edu,
        })
panel_ce = pd.DataFrame(rows_ce)


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: NICKELL BIAS — Anderson-Hsiao IV vs standard FE
# ═══════════════════════════════════════════════════════════════════════

out("\n" + "=" * 80)
out("TEST 1: NICKELL BIAS — Anderson-Hsiao IV vs Standard FE")
out("=" * 80)
out("""
Concern: In a short-T dynamic panel with country FE, Nickell (1981) bias
inflates the autoregressive coefficient (child_edu ~ parent_edu + country FE).
Anderson-Hsiao (1982): first-difference the equation, use lagged levels
(edu_{t-1}) as instruments for the differenced regressor.

Model: child_edu(t+28) = beta * parent_edu(t) + alpha_i + epsilon
AH:    delta_child_edu  = beta * delta_parent_edu + delta_epsilon
       instrument for delta_parent_edu: parent_edu(t-1) = level at previous period

CAVEAT (do not cite AH results as bias-correction evidence in body text):
the lagged-level instrument parent_edu(t-1) is structurally correlated with
the outcome child_edu(t+28) through a channel that does not pass through
parent_edu(t). The paper's own grandparent-channel result (paper §The
Shape of the Response: β_gp ≠ 0 at low parental baselines) implies that
parent_edu(t-1) has an independent effect on child_edu(t+28) via the
grandparent radius. The exclusion restriction therefore fails by the
paper's own mechanism, and AH point estimates here are not a clean
bias correction. AH is retained as a battery diagnostic; the headline
identification rests on the biological argument, the natural experiments,
and the USSR falsification, not on AH.
""")

for ceiling in [60, 90]:
    cohort = entry_years.get(10, {})
    # Filter panel
    mask = panel_ce.apply(
        lambda r: (
            r["country"] in cohort
            and r["t"] >= cohort[r["country"]]
            and r["edu_t"] <= ceiling
        ), axis=1
    )
    sub = panel_ce[mask].copy()

    # ── Standard FE ──
    fe_sub = sub.dropna(subset=["edu_t", "child_edu"]).copy()
    counts = fe_sub.groupby("country").size()
    fe_sub = fe_sub[fe_sub["country"].isin(counts[counts >= 2].index)]

    xdm = fe_sub["edu_t"] - fe_sub.groupby("country")["edu_t"].transform("mean")
    ydm = fe_sub["child_edu"] - fe_sub.groupby("country")["child_edu"].transform("mean")
    ok = ~np.isnan(xdm.values) & ~np.isnan(ydm.values)
    beta_fe = np.sum(xdm.values[ok] * ydm.values[ok]) / np.sum(xdm.values[ok] ** 2)
    resid_fe = ydm.values[ok] - beta_fe * xdm.values[ok]
    r2_fe = 1 - np.sum(resid_fe ** 2) / np.sum(ydm.values[ok] ** 2)

    # Clustered SE for FE
    countries_fe = fe_sub["country"].values[ok]
    unique_c = np.unique(countries_fe)
    G = len(unique_c)
    meat = 0.0
    for cc in unique_c:
        idx = countries_fe == cc
        meat += np.sum(xdm.values[ok][idx] * resid_fe[idx]) ** 2
    bread = 1.0 / np.sum(xdm.values[ok] ** 2)
    n_fe = ok.sum()
    correction = (G / (G - 1)) * ((n_fe - 1) / (n_fe - 2))
    se_fe = np.sqrt(bread ** 2 * meat * correction)
    t_fe = beta_fe / se_fe
    p_fe = 2 * stats.t.sf(np.abs(t_fe), df=G - 1)

    # ── Anderson-Hsiao IV ──
    # Sort by country and time, compute first differences
    ah_sub = sub[["country", "t", "edu_t", "child_edu"]].dropna().sort_values(["country", "t"])

    # First differences within country
    ah_sub["d_child_edu"] = ah_sub.groupby("country")["child_edu"].diff()
    ah_sub["d_edu_t"] = ah_sub.groupby("country")["edu_t"].diff()
    # Instrument: lagged level of parent education (edu_t at previous period)
    ah_sub["edu_t_lag"] = ah_sub.groupby("country")["edu_t"].shift(1)

    ah_clean = ah_sub.dropna(subset=["d_child_edu", "d_edu_t", "edu_t_lag"]).copy()

    # Need at least some observations
    n_ah = len(ah_clean)
    n_c_ah = ah_clean["country"].nunique()

    if n_ah >= 10 and n_c_ah >= 3:
        # IV: two-stage least squares
        # Stage 1: d_edu_t = gamma * edu_t_lag + v
        z = ah_clean["edu_t_lag"].values
        d_x = ah_clean["d_edu_t"].values
        d_y = ah_clean["d_child_edu"].values

        # Stage 1
        gamma = np.sum(z * d_x) / np.sum(z * z)
        d_x_hat = gamma * z

        # First stage F-stat (correlation-based, always non-negative)
        corr_zx = np.corrcoef(z, d_x)[0, 1]
        r2_s1 = corr_zx ** 2
        f_stat_s1 = (r2_s1 / 1) / ((1 - r2_s1) / (n_ah - 2)) if r2_s1 < 1 else np.inf

        # Stage 2: d_y = beta_ah * d_x_hat + u
        beta_ah = np.sum(d_x_hat * d_y) / np.sum(d_x_hat * d_x_hat)
        resid_ah = d_y - beta_ah * d_x

        # Clustered SE for IV
        countries_ah = ah_clean["country"].values
        unique_c_ah = np.unique(countries_ah)
        G_ah = len(unique_c_ah)
        meat_ah = 0.0
        for cc in unique_c_ah:
            idx = countries_ah == cc
            meat_ah += np.sum(d_x_hat[idx] * resid_ah[idx]) ** 2
        bread_ah = 1.0 / np.sum(d_x_hat ** 2)
        correction_ah = (G_ah / (G_ah - 1)) * ((n_ah - 1) / (n_ah - 2))
        se_ah = np.sqrt(bread_ah ** 2 * meat_ah * correction_ah)
        t_ah = beta_ah / se_ah
        p_ah = 2 * stats.t.sf(np.abs(t_ah), df=G_ah - 1)

        out(f"\n  Ceiling = {ceiling}%, entry = 10%")
        out(f"  {'Method':<25} {'beta':>8} {'SE':>8} {'p':>10} {'n':>6} {'Ctry':>5}")
        out(f"  {'-' * 65}")
        out(f"  {'Standard FE':<25} {beta_fe:>8.4f} {se_fe:>8.4f} {p_fe:>10.4f} {n_fe:>6} {G:>5}")
        out(f"  {'Anderson-Hsiao IV':<25} {beta_ah:>8.4f} {se_ah:>8.4f} {p_ah:>10.4f} {n_ah:>6} {G_ah:>5}")
        out(f"  First-stage F-stat: {f_stat_s1:.1f} (>10 = strong instrument)")
        out(f"  FE beta / AH beta ratio: {beta_fe / beta_ah:.3f}")

        if beta_ah > 1.0:
            out(f"  --> beta > 1 SURVIVES under Anderson-Hsiao (beta_AH = {beta_ah:.4f})")
            out(f"      Nickell bias is NOT driving the result.")
        elif beta_ah > 0 and p_ah < 0.05:
            out(f"  --> AH beta is positive and significant (p = {p_ah:.4f})")
            out(f"      Coefficient is {'larger' if beta_ah > beta_fe else 'smaller'} under AH.")
            if beta_fe > 1.0 and beta_ah <= 1.0:
                out(f"      NOTE: FE beta > 1 but AH beta <= 1. Some Nickell inflation possible.")
            else:
                out(f"      Nickell bias is not materially affecting the result.")
        else:
            out(f"  --> AH result: beta = {beta_ah:.4f}, p = {p_ah:.4f}")
    else:
        out(f"\n  Ceiling = {ceiling}%: insufficient AH observations ({n_ah} obs, {n_c_ah} countries)")


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: NONLINEARITY IN RESIDUALIZATION FIRST STAGE
# ═══════════════════════════════════════════════════════════════════════

out("\n\n" + "=" * 80)
out("TEST 2: NONLINEARITY IN RESIDUALIZATION FIRST STAGE")
out("=" * 80)
out("""
Concern: The residualization regresses log_GDP on education (linear, with
country FE) in the first stage. If the true relationship is nonlinear,
the linear residuals might retain education signal, making residualized
GDP look weaker than it is.

Method: Add education^2 to the first stage. Compare residualized GDP R^2
for life expectancy at T+25 under linear vs quadratic first stage.
If both are near zero, the result is not an artifact of linear specification.
""")

for ceiling in [60, 90]:
    cohort = entry_years.get(10, {})
    sub = filter_panel(panel_le, cohort, ceiling)
    sub = sub.dropna(subset=["edu_t", "log_gdp_t", "le"]).copy()
    counts = sub.groupby("country").size()
    sub = sub[sub["country"].isin(counts[counts >= 2].index)]

    if sub["country"].nunique() < 3 or len(sub) < 10:
        out(f"\n  Ceiling = {ceiling}%: insufficient data")
        continue

    # Demean by country
    edu_dm = (sub["edu_t"] - sub.groupby("country")["edu_t"].transform("mean")).values
    gdp_dm = (sub["log_gdp_t"] - sub.groupby("country")["log_gdp_t"].transform("mean")).values
    le_dm = (sub["le"] - sub.groupby("country")["le"].transform("mean")).values

    ok = ~np.isnan(edu_dm) & ~np.isnan(gdp_dm) & ~np.isnan(le_dm)
    edu_dm, gdp_dm, le_dm = edu_dm[ok], gdp_dm[ok], le_dm[ok]
    countries = sub["country"].values[ok]

    # ── Linear first stage ──
    X_lin = edu_dm.reshape(-1, 1)
    reg_lin = sm.OLS(gdp_dm, X_lin).fit()
    resid_lin = gdp_dm - reg_lin.predict(X_lin)
    r2_stage1_lin = reg_lin.rsquared

    # Second stage: resid → LE
    X_resid_lin = resid_lin.reshape(-1, 1)
    reg_s2_lin = sm.OLS(le_dm, X_resid_lin).fit()
    r2_resid_lin = reg_s2_lin.rsquared

    # ── Quadratic first stage ──
    edu_sq_dm = edu_dm ** 2
    # Demean the squared term by country as well
    sub_tmp = pd.DataFrame({"country": countries, "edu_sq": sub["edu_t"].values[ok] ** 2})
    edu_sq_dm_proper = (sub_tmp["edu_sq"] - sub_tmp.groupby("country")["edu_sq"].transform("mean")).values

    X_quad = np.column_stack([edu_dm, edu_sq_dm_proper])
    reg_quad = sm.OLS(gdp_dm, X_quad).fit()
    resid_quad = gdp_dm - reg_quad.predict(X_quad)
    r2_stage1_quad = reg_quad.rsquared

    # Second stage: resid → LE
    X_resid_quad = resid_quad.reshape(-1, 1)
    reg_s2_quad = sm.OLS(le_dm, X_resid_quad).fit()
    r2_resid_quad = reg_s2_quad.rsquared

    # Also get education R² for LE (for comparison)
    reg_edu_le = sm.OLS(le_dm, edu_dm.reshape(-1, 1)).fit()
    r2_edu_le = reg_edu_le.rsquared

    n = ok.sum()
    nc = len(np.unique(countries))

    out(f"\n  Ceiling = {ceiling}%, entry = 10%, n = {n}, countries = {nc}")
    out(f"  {'First stage':<25} {'Stage1 R²':>10} {'Resid GDP→LE R²':>16}")
    out(f"  {'-' * 55}")
    out(f"  {'Linear (edu)':<25} {r2_stage1_lin:>10.4f} {r2_resid_lin:>16.4f}")
    out(f"  {'Quadratic (edu + edu²)':<25} {r2_stage1_quad:>10.4f} {r2_resid_quad:>16.4f}")
    out(f"  {'Education → LE (ref)':<25} {'':>10} {r2_edu_le:>16.4f}")
    out(f"")
    if r2_resid_quad < 0.05:
        out(f"  --> Quadratic first stage: residualized GDP R² = {r2_resid_quad:.4f} (near zero)")
        out(f"      Result is NOT an artifact of linear specification.")
    else:
        out(f"  --> Quadratic first stage increased residual GDP R² to {r2_resid_quad:.4f}")
        out(f"      Linear specification may understate GDP's independent role.")

    # Capture first ceiling's values for checkin
    if "Rob-quad-resid-R2" not in _checkin_numbers:
        _checkin_numbers["Rob-quad-resid-R2"] = round(r2_resid_quad, 2)


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: BOOTSTRAPPED CONFIDENCE INTERVALS — β AND R² ACROSS FOUR OUTCOMES
# ═══════════════════════════════════════════════════════════════════════
#
# Outcomes and predictors follow `tab:edu-outcomes` Panel A:
#   - log(LE) at T+25, predicted by lower-secondary completion at T
#   - log(TFR) at T+5, predicted by PRIMARY completion at T (TFR row in
#     tab:edu-outcomes uses primary because the steepest fertility decline
#     occurs at primary; see paper §demographic-structure-and-the-fertility-
#     transition)
#   - log(U5MR) at T+12, predicted by lower-secondary completion at T
#   - child education at T+28 (level — already a percentage), predicted by
#     parent lower-secondary completion at T
#
# Primary metric is β (coefficient); R² is reported as a secondary
# descriptive statistic.

out("\n\n" + "=" * 80)
out("TEST 3: BOOTSTRAPPED CIs ON β (PRIMARY) AND R² (SECONDARY)")
out("=" * 80)
out("""
Method: Bootstrap 1000 replications, resampling countries with replacement.
For each (outcome, ceiling), compute (outcome read at its per-outcome lag:
LE T+25, TFR T+5, U5MR T+12, child education T+25):
  - β_education: coefficient of education at T on outcome at T+lag (country FE)
  - β_GDP_resid: coefficient of residualised log GDP at T (FWL: log GDP
                 stripped of education's contribution) on outcome at T+lag
                 (country FE)
Report point estimates and 95% percentile CIs. The null tested is whether
β_GDP_resid is statistically distinguishable from zero; if the CI includes
zero across outcomes, residualised GDP has no independent predictive power.

Outcome transformation matches tab:edu-outcomes log rows for LE/TFR/U5MR;
child education stays in levels (it is already a percentage).
""")

# Load primary completion for the TFR row (matching tab:edu-outcomes).
edu_primary_annual = interpolate_to_annual(edu_raw, "primary")

# Load TFR and U5MR (LE is already loaded as le_df).
tfr_df = load_wb("children_per_woman_total_fertility.csv")
u5mr_df = load_wb("child_mortality_u5.csv")

N_BOOT = 1000
RNG_SEED = 42


def _build_log_panel(edu_annual_, outcome_df_, gdp_df_, t_years_, lag_, outcome_name_):
    """build_panel + log transform of outcome. NaN outcomes dropped."""
    p = build_panel(edu_annual_, outcome_df_, gdp_df_, t_years_, lag_, outcome_name_)
    if p.empty:
        return p
    with np.errstate(invalid="ignore", divide="ignore"):
        p[outcome_name_] = np.where(p[outcome_name_] > 0,
                                    np.log(p[outcome_name_]), np.nan)
    return p


# Build four panels. TFR uses biological parent timing (LAG_TFR=5);
# U5MR uses the childrearing window (LAG_CHILDREARING=12); LE uses
# time-to-agency (LAG_GENERATION=28). panel_child reuses panel_ce built
# upstream at LAG_GENERATION.
panel_le_log = _build_log_panel(
    edu_annual, le_df, gdp_df, T_YEARS, OUTCOME_LAG["le"], "le")
panel_tfr_log = _build_log_panel(
    edu_primary_annual, tfr_df, gdp_df, T_YEARS, OUTCOME_LAG["tfr"], "tfr")
panel_u5mr_log = _build_log_panel(
    edu_annual, u5mr_df, gdp_df, T_YEARS, OUTCOME_LAG["u5mr"], "u5mr")
panel_child = panel_ce  # already built upstream; level outcome (child_edu).

# Cohort lookup (entry threshold 10%) is shared across outcomes.
cohort_10 = entry_years.get(10, {})


def _bootstrap_outcome(sub_, outcome_col_, n_boot_=N_BOOT, seed_=RNG_SEED):
    """Country-resample bootstrap of β and R² for a single outcome.

    Uses per-country sufficient statistics so each replicate is
    O(n_countries) rather than O(n_obs).

    Returns dict with point estimates and 95% percentile CIs for β_edu,
    β_GDP_resid, R²_edu, R²_GDP_resid; or None if the sample is too small.
    """
    sub_ = sub_.dropna(subset=["edu_t", "log_gdp_t", outcome_col_]).copy()
    counts_ = sub_.groupby("country").size()
    sub_ = sub_[sub_["country"].isin(counts_[counts_ >= 2].index)]

    countries_ = sub_["country"].unique()
    n_c = len(countries_)
    if n_c < 5:
        return None

    edu_dm = (sub_["edu_t"] - sub_.groupby("country")["edu_t"].transform("mean")).values
    gdp_dm = (sub_["log_gdp_t"] - sub_.groupby("country")["log_gdp_t"].transform("mean")).values
    y_dm = (sub_[outcome_col_] - sub_.groupby("country")[outcome_col_].transform("mean")).values

    c2i = {c: i for i, c in enumerate(countries_)}
    row_idx = sub_["country"].map(c2i).values

    len_c = np.bincount(row_idx, minlength=n_c)
    see_c = np.bincount(row_idx, weights=edu_dm * edu_dm, minlength=n_c)
    sgg_c = np.bincount(row_idx, weights=gdp_dm * gdp_dm, minlength=n_c)
    syy_c = np.bincount(row_idx, weights=y_dm * y_dm, minlength=n_c)
    seg_c = np.bincount(row_idx, weights=edu_dm * gdp_dm, minlength=n_c)
    sey_c = np.bincount(row_idx, weights=edu_dm * y_dm, minlength=n_c)
    sgy_c = np.bincount(row_idx, weights=gdp_dm * y_dm, minlength=n_c)

    def _one(mult_):
        see = float(mult_ @ see_c)
        sgg = float(mult_ @ sgg_c)
        syy = float(mult_ @ syy_c)
        seg = float(mult_ @ seg_c)
        sey = float(mult_ @ sey_c)
        sgy = float(mult_ @ sgy_c)
        if see <= 0.0 or syy <= 0.0:
            return None
        beta_e = sey / see
        r2_e = (sey * sey) / (see * syy)
        ssr_r = sgg - (seg * seg) / see
        if ssr_r <= 0.0:
            return None
        cross_rg = sgy - (seg * sey) / see
        beta_r = cross_rg / ssr_r
        r2_r = 1.0 - (syy - 2.0 * beta_r * cross_rg + beta_r * beta_r * ssr_r) / syy
        return beta_e, r2_e, beta_r, r2_r

    full_mult = np.ones(n_c, dtype=np.intp)
    point_est = _one(full_mult)
    if point_est is None:
        return None
    beta_edu_point, r2_edu_point, beta_resid_point, r2_resid_point = point_est

    rng = np.random.RandomState(seed_)
    boot_be, boot_re, boot_br, boot_rr = [], [], [], []

    for _ in range(n_boot_):
        boot_countries = rng.choice(countries_, size=n_c, replace=True)
        idx_ = np.fromiter((c2i[c] for c in boot_countries),
                           dtype=np.intp, count=n_c)
        mult_ = np.bincount(idx_, minlength=n_c)

        n_obs_ = int(mult_ @ len_c)
        n_unique_ = int(np.count_nonzero(mult_))
        if n_obs_ < 10 or n_unique_ < 3:
            continue

        res = _one(mult_)
        if res is None:
            continue
        be, re, br, rr = res
        boot_be.append(be)
        boot_re.append(re)
        boot_br.append(br)
        boot_rr.append(rr)

    def _pct(arr, p):
        return float(np.percentile(np.asarray(arr), p)) if len(arr) >= 20 else float("nan")

    return {
        "n_obs": int(len(edu_dm)),
        "n_countries": int(n_c),
        "n_valid_boots": int(len(boot_be)),
        "beta_edu": {
            "point": float(beta_edu_point),
            "ci_lo": _pct(boot_be, 2.5),
            "ci_hi": _pct(boot_be, 97.5),
        },
        "beta_resid_gdp": {
            "point": float(beta_resid_point),
            "ci_lo": _pct(boot_br, 2.5),
            "ci_hi": _pct(boot_br, 97.5),
        },
        "r2_edu": {
            "point": float(r2_edu_point),
            "ci_lo": _pct(boot_re, 2.5),
            "ci_hi": _pct(boot_re, 97.5),
        },
        "r2_resid_gdp": {
            "point": float(r2_resid_point),
            "ci_lo": _pct(boot_rr, 2.5),
            "ci_hi": _pct(boot_rr, 97.5),
        },
    }


# Define the four (outcome label, panel, outcome column) tuples.
# Order matters: LE first so the backward-compat flat keys (Rob-boot-*)
# capture LE results at ceiling=60, matching the prior script behaviour.
outcome_specs = [
    ("LE (log)",        panel_le_log,   "le"),
    ("TFR (log, primary→TFR)", panel_tfr_log, "tfr"),
    ("U5MR (log)",      panel_u5mr_log, "u5mr"),
    ("Child edu (level)", panel_child,  "child_edu"),
]

# Nested results structure: {outcome_key: {ceiling_str: result_dict}}.
bootstrap_results = {}


def _outcome_key(label_):
    return label_.split()[0].lower().replace("le", "le").replace("tfr", "tfr").replace("u5mr", "u5mr")


for outcome_label, outcome_panel, outcome_col in outcome_specs:
    out("\n" + "-" * 80)
    out(f"OUTCOME: {outcome_label}")
    out("-" * 80)

    if outcome_col == "child_edu":
        # Child-education panel has no entry/ceiling filter built in
        # (it's intergenerational home-niche transmission, not the GDP-merge sample). Use the
        # full panel without further restriction.
        ceilings_to_run = [None]
    else:
        ceilings_to_run = [60, 90]

    for ceiling in ceilings_to_run:
        if ceiling is None:
            sub = outcome_panel.copy()
            sample_label = "full panel (child-edu)"
        else:
            sub = filter_panel(outcome_panel, cohort_10, ceiling)
            sample_label = f"entry≥10%, ceiling≤{ceiling}%"

        res = _bootstrap_outcome(sub, outcome_col)
        if res is None:
            out(f"\n  {sample_label}: insufficient data")
            continue

        n_obs = res["n_obs"]
        n_c = res["n_countries"]
        nv = res["n_valid_boots"]
        out(f"\n  {sample_label}: n={n_obs}, countries={n_c}, "
            f"{nv}/{N_BOOT} valid replications")
        out("")
        out(f"  {'Quantity':<28} {'Point':>10} {'95% CI lo':>12} {'95% CI hi':>12}")
        out(f"  {'-' * 64}")
        for qname, qkey in [
            ("β_education",   "beta_edu"),
            ("β_GDP_resid",   "beta_resid_gdp"),
            ("R²_education",  "r2_edu"),
            ("R²_GDP_resid",  "r2_resid_gdp"),
        ]:
            q = res[qkey]
            star = " *" if (qkey.startswith("beta_resid")
                            and not (np.isnan(q["ci_lo"]) or np.isnan(q["ci_hi"]))
                            and (q["ci_lo"] > 0 or q["ci_hi"] < 0)) else ""
            out(f"  {qname:<28} {q['point']:>10.4f} {q['ci_lo']:>12.4f} "
                f"{q['ci_hi']:>12.4f}{star}")

        # Verdict on the null
        bg = res["beta_resid_gdp"]
        if not np.isnan(bg["ci_lo"]) and not np.isnan(bg["ci_hi"]):
            if bg["ci_lo"] <= 0.0 <= bg["ci_hi"]:
                out(f"  --> 95% CI for β_GDP_resid INCLUDES ZERO "
                    f"[{bg['ci_lo']:.4f}, {bg['ci_hi']:.4f}]: residualised GDP "
                    f"has no independent predictive power.")
            else:
                out(f"  --> 95% CI for β_GDP_resid EXCLUDES ZERO "
                    f"[{bg['ci_lo']:.4f}, {bg['ci_hi']:.4f}]: residualised "
                    f"GDP retains a signal in this sample.")

        # Store result
        okey = outcome_col  # "le", "tfr", "u5mr", "child_edu"
        ckey = "full" if ceiling is None else f"ceil_{ceiling}"
        bootstrap_results.setdefault(okey, {})[ckey] = res

        # Backward-compatible flat keys: LE at ceiling=60, R² CIs.
        # Preserves verify regs Rob-boot-edu-* / Rob-boot-gdp-* but the
        # values now reflect log(LE) instead of level LE.
        if okey == "le" and ckey == "ceil_60" and "Rob-boot-edu-lo" not in _checkin_numbers:
            _checkin_numbers["Rob-boot-edu-lo"] = round(res["r2_edu"]["ci_lo"], 2)
            _checkin_numbers["Rob-boot-edu-hi"] = round(res["r2_edu"]["ci_hi"], 2)
            _checkin_numbers["Rob-boot-gdp-lo"] = round(res["r2_resid_gdp"]["ci_lo"], 2)
            _checkin_numbers["Rob-boot-gdp-hi"] = round(res["r2_resid_gdp"]["ci_hi"], 2)


# ── Write output ────────────────────────────────────────────────────────

out_path = os.path.join(REPO_ROOT, "paper", "robustness_results.txt")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    f.write("\n".join(output_lines) + "\n")
out(f"\nResults written to {out_path}")

# ── Write checkin JSON ────────────────────────────────────────────────────
write_checkin("robustness_tests.json", {
    "numbers": _checkin_numbers,
    "bootstrap_betas": bootstrap_results,
}, script_path="scripts/robustness/robustness_tests.py")
