"""
robustness/robustness_tests.py
====================
Three econometric robustness tests for the education-development paper.

Test 1: Nickell Bias — Anderson-Hsiao IV vs standard FE for PTE regression
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
from _shared import REPO_ROOT, write_checkin
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
LAG = 25

edu_annual = interpolate_to_annual(edu_raw, COL_NAME)
entry_years = precompute_entry_years(edu_annual)

# Build LE panel (same as tables/regression_tables.py)
panel_le = build_panel(edu_annual, le_df, gdp_df, T_YEARS, LAG, "le")

# Build child-education panel (parent edu → child edu, 25-year lag)
rows_ce = []
for c in sorted(edu_annual.keys()):
    s = edu_annual[c]
    for t in T_YEARS:
        if t not in s.index or (t + LAG) not in s.index:
            continue
        parent_edu = s[t]
        child_edu = s[t + LAG]
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

Model: child_edu(t+25) = beta * parent_edu(t) + alpha_i + epsilon
AH:    delta_child_edu  = beta * delta_parent_edu + delta_epsilon
       instrument for delta_parent_edu: parent_edu(t-1) = level at previous period
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
# TEST 3: BOOTSTRAPPED CONFIDENCE INTERVALS ON R² COMPARISONS
# ═══════════════════════════════════════════════════════════════════════

out("\n\n" + "=" * 80)
out("TEST 3: BOOTSTRAPPED CONFIDENCE INTERVALS ON R² COMPARISONS")
out("=" * 80)
out("""
Method: Bootstrap 1000 replications, resampling countries with replacement.
For each replication, compute:
  - Education R² for LE at T+25 (country FE)
  - Residualized GDP R² for LE at T+25 (country FE)
Report 95% CI for both. If CIs don't overlap, the difference is robust.
""")

np.random.seed(42)
N_BOOT = 1000

for ceiling in [60, 90]:
    cohort = entry_years.get(10, {})
    sub = filter_panel(panel_le, cohort, ceiling)
    sub = sub.dropna(subset=["edu_t", "log_gdp_t", "le"]).copy()
    counts = sub.groupby("country").size()
    sub = sub[sub["country"].isin(counts[counts >= 2].index)]

    countries_list = sub["country"].unique()
    n_countries = len(countries_list)

    if n_countries < 5:
        out(f"\n  Ceiling = {ceiling}%: insufficient countries ({n_countries})")
        continue

    # Precompute per-country sufficient statistics once. Bootstrap that
    # resamples countries with replacement is equivalent to summing these
    # with integer multiplicities — each replicate costs O(n_countries)
    # ops instead of rebuilding a DataFrame + groupby demean.
    edu_dm_full = (sub["edu_t"] - sub.groupby("country")["edu_t"].transform("mean")).values
    gdp_dm_full = (sub["log_gdp_t"] - sub.groupby("country")["log_gdp_t"].transform("mean")).values
    le_dm_full = (sub["le"] - sub.groupby("country")["le"].transform("mean")).values
    country_to_idx = {c: i for i, c in enumerate(countries_list)}
    row_idx = sub["country"].map(country_to_idx).values
    len_c = np.bincount(row_idx, minlength=n_countries)
    see_c = np.bincount(row_idx, weights=edu_dm_full * edu_dm_full, minlength=n_countries)
    sgg_c = np.bincount(row_idx, weights=gdp_dm_full * gdp_dm_full, minlength=n_countries)
    sll_c = np.bincount(row_idx, weights=le_dm_full * le_dm_full, minlength=n_countries)
    seg_c = np.bincount(row_idx, weights=edu_dm_full * gdp_dm_full, minlength=n_countries)
    sel_c = np.bincount(row_idx, weights=edu_dm_full * le_dm_full, minlength=n_countries)
    sgl_c = np.bincount(row_idx, weights=gdp_dm_full * le_dm_full, minlength=n_countries)

    boot_edu_r2 = []
    boot_resid_r2 = []
    boot_diff = []

    for b in range(N_BOOT):
        # np.random.choice once per rep preserves the exact draw sequence
        # the old loop produced, so replicates stay bit-identical.
        boot_countries = np.random.choice(countries_list, size=n_countries, replace=True)
        idx = np.fromiter((country_to_idx[c] for c in boot_countries),
                          dtype=np.intp, count=n_countries)
        mult = np.bincount(idx, minlength=n_countries)

        n_obs = int(mult @ len_c)
        n_unique = int(np.count_nonzero(mult))
        if n_obs < 10 or n_unique < 3:
            continue

        see = float(mult @ see_c)
        sgg = float(mult @ sgg_c)
        sll = float(mult @ sll_c)
        seg = float(mult @ seg_c)
        sel = float(mult @ sel_c)
        sgl = float(mult @ sgl_c)

        if sll <= 0.0 or see <= 0.0:
            continue

        # Edu → LE R² without intercept: 1 - (sll - sel²/see)/sll = sel²/(see·sll)
        r2_e = (sel * sel) / (see * sll)

        # Residualize GDP on edu: gdp_resid = gdp_dm - (seg/see)·edu_dm
        # sum(gdp_resid²)      = sgg - seg²/see
        # sum(gdp_resid · le)  = sgl - (seg·sel)/see
        ssr_resid = sgg - (seg * seg) / see
        cross_rg = sgl - (seg * sel) / see
        if ssr_resid <= 0.0:
            continue
        beta_rg = cross_rg / ssr_resid
        # 1 - (sll - 2·β·cross + β²·ssr_resid)/sll
        r2_rg = 1.0 - (sll - 2.0 * beta_rg * cross_rg + beta_rg * beta_rg * ssr_resid) / sll

        boot_edu_r2.append(r2_e)
        boot_resid_r2.append(r2_rg)
        boot_diff.append(r2_e - r2_rg)

    boot_edu_r2 = np.array(boot_edu_r2)
    boot_resid_r2 = np.array(boot_resid_r2)
    boot_diff = np.array(boot_diff)

    n_valid = len(boot_edu_r2)

    # Point estimates from full sample
    r2_edu_point, _, _ = fe_r2("edu_t", "le", sub)

    # Residualize for point estimate
    edu_dm_full = (sub["edu_t"] - sub.groupby("country")["edu_t"].transform("mean")).values
    gdp_dm_full = (sub["log_gdp_t"] - sub.groupby("country")["log_gdp_t"].transform("mean")).values
    le_dm_full = (sub["le"] - sub.groupby("country")["le"].transform("mean")).values
    ok_full = ~np.isnan(edu_dm_full) & ~np.isnan(gdp_dm_full) & ~np.isnan(le_dm_full)
    edu_f = edu_dm_full[ok_full]
    gdp_f = gdp_dm_full[ok_full]
    le_f = le_dm_full[ok_full]
    beta_eg_f = np.sum(edu_f * gdp_f) / np.sum(edu_f ** 2)
    gdp_resid_f = gdp_f - beta_eg_f * edu_f
    beta_rg_f = np.sum(gdp_resid_f * le_f) / np.sum(gdp_resid_f ** 2)
    resid_rg_f = le_f - beta_rg_f * gdp_resid_f
    r2_resid_point = 1 - np.sum(resid_rg_f ** 2) / np.sum(le_f ** 2)

    # CIs (percentile method)
    ci_edu = np.percentile(boot_edu_r2, [2.5, 97.5])
    ci_resid = np.percentile(boot_resid_r2, [2.5, 97.5])
    ci_diff = np.percentile(boot_diff, [2.5, 97.5])

    out(f"\n  Ceiling = {ceiling}%, entry = 10%, {n_valid}/{N_BOOT} valid replications")
    out(f"  n = {ok_full.sum()}, countries = {n_countries}")
    out(f"")
    out(f"  {'Measure':<30} {'Point':>8} {'95% CI lower':>14} {'95% CI upper':>14}")
    out(f"  {'-' * 70}")
    out(f"  {'Education R²':<30} {r2_edu_point:>8.4f} {ci_edu[0]:>14.4f} {ci_edu[1]:>14.4f}")
    out(f"  {'Residualized GDP R²':<30} {r2_resid_point:>8.4f} {ci_resid[0]:>14.4f} {ci_resid[1]:>14.4f}")
    out(f"  {'Difference (Edu - Resid)':<30} {r2_edu_point - r2_resid_point:>8.4f} {ci_diff[0]:>14.4f} {ci_diff[1]:>14.4f}")
    out(f"")

    overlap = ci_edu[0] <= ci_resid[1] and ci_resid[0] <= ci_edu[1]
    if not overlap:
        out(f"  --> CIs DO NOT OVERLAP. The education R² advantage is statistically robust.")
    else:
        out(f"  --> CIs overlap. Difference may not be statistically significant at 95% level.")

    if ci_diff[0] > 0:
        out(f"  --> 95% CI for difference is entirely above zero [{ci_diff[0]:.4f}, {ci_diff[1]:.4f}]")
        out(f"      Education R² is robustly higher than residualized GDP R².")

    # Capture first ceiling's bootstrap CIs for checkin
    if "Rob-boot-edu-lo" not in _checkin_numbers:
        _checkin_numbers["Rob-boot-edu-lo"] = round(ci_edu[0], 2)
        _checkin_numbers["Rob-boot-edu-hi"] = round(ci_edu[1], 2)
        _checkin_numbers["Rob-boot-gdp-lo"] = round(ci_resid[0], 2)
        _checkin_numbers["Rob-boot-gdp-hi"] = round(ci_resid[1], 2)


# ── Write output ────────────────────────────────────────────────────────

out_path = os.path.join(REPO_ROOT, "paper", "robustness_results.txt")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    f.write("\n".join(output_lines) + "\n")
out(f"\nResults written to {out_path}")

# ── Write checkin JSON ────────────────────────────────────────────────────
write_checkin("robustness_tests.json", {
    "numbers": _checkin_numbers,
}, script_path="scripts/robustness/robustness_tests.py")
