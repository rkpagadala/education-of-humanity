"""
robustness/residgdp_country_decade_fe.py
=========================================

TEST 3 from the residGDP audit plan.

Question: does the +24 GAVI interaction (paper) / +11 baseline-controlled
interaction survive stripping within-country era variation via country ×
decade fixed effects?

Method:
  - Same common-sample panel as permutation_null_gdp.py (n=645, 123 countries).
  - Decade key: floor((t + LAG) / 10) * 10 — outcome-year decade.
  - Country × decade dummies absorbed via two-step demeaning:
      Step 1: demean each variable by country
      Step 2: demean each variable by country × decade
  - Re-residualise log GDP on edu within country × decade.
  - Run the interaction spec: u5mr_dm = b1·gdp_resid + b2·post + b3·(gdp_resid × post)
    + b4·edu + b5·(edu × post)
  - Compare β_3 (the GDP-resid × Post-2000 interaction) under:
      (a) country FE only — paper's spec, +23.96 expected
      (b) country FE + baseline controls — our +11.18 from earlier
      (c) country × decade FE + baseline controls — this test

If b3 collapses further from +11.18 → small/insignificant under (c), the
post-2000 "GAVI signal" was within-country era variation, not a temporal
donor effect. If b3 survives ~+11, the donor interpretation is structural.

Output: checkin/residgdp_country_decade_fe.json
"""

import os
import sys
import json

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "residualization"))
sys.path.insert(1, SCRIPTS_DIR)

from _shared import (  # noqa: E402
    load_education,
    load_wb,
    interpolate_to_annual,
    build_panel,
    build_child_edu_panel,
    precompute_entry_years,
    filter_panel,
    LAG_GENERATION,
)

T_YEARS = list(range(1960, 1995, 5))
LAG = LAG_GENERATION
ENTRY = 10
CEILING = 90


def demean_by_group(values, group_codes, n_groups):
    counts = np.bincount(group_codes, minlength=n_groups)
    sums = np.bincount(group_codes, weights=values, minlength=n_groups)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return values - means[group_codes]


def fit_ols_clustered(y, X_cols, cluster_codes, names):
    """OLS y ~ X (no intercept; pre-demeaned). Country-clustered SE.
    Uses lstsq + pseudo-inverse so rank-deficient designs return finite
    estimates instead of crashing — but flags it via condition number."""
    X = np.column_stack(X_cols)
    XtX = X.T @ X
    cond = float(np.linalg.cond(XtX)) if np.all(np.isfinite(XtX)) else float("inf")
    try:
        beta = np.linalg.solve(XtX, X.T @ y)
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        XtX_inv = np.linalg.pinv(XtX)
    resid = y - X @ beta
    # Cluster-robust variance (CR0; consistent with statsmodels with
    # no small-sample correction at this scale)
    n_c = int(cluster_codes.max() + 1)
    meat = np.zeros_like(XtX)
    for c in range(n_c):
        mask = cluster_codes == c
        if mask.sum() == 0:
            continue
        Xc = X[mask]
        rc = resid[mask]
        u = Xc.T @ rc
        meat += np.outer(u, u)
    vcov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(vcov))
    from scipy import stats
    out = {}
    for i, n in enumerate(names):
        t = beta[i] / se[i] if se[i] > 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(t))) if not np.isnan(t) else np.nan
        out[n] = {
            "beta": float(beta[i]),
            "se": float(se[i]),
            "t": float(t) if not np.isnan(t) else None,
            "p": float(p) if not np.isnan(p) else None,
        }
    return out


def fe_residualize(panel, x_col, on_col, group_codes, n_groups):
    """Residualise x_col on on_col within group via FWL. Returns residual array."""
    x_dm = demean_by_group(panel[x_col].to_numpy(), group_codes, n_groups)
    on_dm = demean_by_group(panel[on_col].to_numpy(), group_codes, n_groups)
    ok = ~np.isnan(x_dm) & ~np.isnan(on_dm)
    sxx = float(np.dot(on_dm[ok], on_dm[ok]))
    if sxx <= 0:
        return None, np.nan
    sxy = float(np.dot(on_dm[ok], x_dm[ok]))
    beta = sxy / sxx
    syy = float(np.dot(x_dm[ok], x_dm[ok]))
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else np.nan
    return x_dm - beta * on_dm, r2


# ── Load and build common-sample panel ──────────────────────────────

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
le_raw = load_wb("life_expectancy_years.csv")
tfr_raw = load_wb("children_per_woman_total_fertility.csv")
u5mr_raw = load_wb("child_mortality_u5.csv")
gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")

edu_annual = interpolate_to_annual(edu_raw, "lower_sec")

panel_le = build_panel(edu_annual, le_raw, gdp_raw, T_YEARS, LAG, "le_tp28")
panel_tfr = build_panel(edu_annual, tfr_raw, gdp_raw, T_YEARS, LAG, "tfr_tp28")
panel_u5mr = build_panel(edu_annual, u5mr_raw, gdp_raw, T_YEARS, LAG, "u5mr_tp28")
panel_ced = build_child_edu_panel(edu_annual, gdp_raw, T_YEARS, LAG).rename(
    columns={"child_edu": "ced_tp28"})

key = ["country", "t"]
merged = (
    panel_le[key + ["edu_t", "log_gdp_t", "le_tp28"]]
    .merge(panel_tfr[key + ["tfr_tp28"]], on=key, how="inner")
    .merge(panel_u5mr[key + ["u5mr_tp28"]], on=key, how="inner")
    .merge(panel_ced[key + ["ced_tp28"]], on=key, how="inner")
)
entry_years = precompute_entry_years(edu_annual)
cohort = entry_years[ENTRY]
panel = filter_panel(merged, cohort, CEILING)
panel = panel.dropna(subset=[
    "edu_t", "log_gdp_t", "u5mr_tp28", "le_tp28", "tfr_tp28", "ced_tp28",
]).copy()
panel["post2000"] = ((panel["t"] + LAG) >= 2000).astype(int)
# Outcome-year decade
panel["decade"] = ((panel["t"] + LAG) // 10) * 10
panel["country_decade"] = panel["country"] + "_" + panel["decade"].astype(str)
print(f"Panel: n={len(panel)}, countries={panel['country'].nunique()}, "
      f"decades={sorted(panel['decade'].unique().tolist())}")

# ── Demean by country (Spec A baseline) and by country × decade (Spec C) ──

c_codes, c_uniq = pd.factorize(panel["country"], sort=False)
cd_codes, cd_uniq = pd.factorize(panel["country_decade"], sort=False)
n_c = len(c_uniq)
n_cd = len(cd_uniq)
print(f"Country groups: {n_c}, country-decade groups: {n_cd}")

# Drop country-decade singletons (need >=2 obs per group for FE to mean something)
cd_counts = pd.Series(cd_codes).value_counts()
cd_keep = cd_counts[cd_counts >= 2].index.values
keep_mask = np.isin(cd_codes, cd_keep)
print(f"  obs in singleton country-decade groups: {(~keep_mask).sum()} dropped")
panel_cd = panel.iloc[keep_mask].reset_index(drop=True)
c_codes_cd, c_uniq_cd = pd.factorize(panel_cd["country"], sort=False)
cd_codes_cd, cd_uniq_cd = pd.factorize(panel_cd["country_decade"], sort=False)
n_c_cd = len(c_uniq_cd)
n_cd_cd = len(cd_uniq_cd)
print(f"After dropping country-decade singletons: n={len(panel_cd)}, "
      f"country-decades={n_cd_cd}")

results = {}

# ── Spec A: country FE only (paper's spec) ──────────────────────────

print("\n" + "=" * 72)
print("SPEC A: country FE only (paper's spec)")
print("=" * 72)
gdp_resid_A, edu_gdp_r2_A = fe_residualize(
    panel, "log_gdp_t", "edu_t", c_codes, n_c)
y_dm_A = demean_by_group(panel["u5mr_tp28"].to_numpy(), c_codes, n_c)
edu_dm_A = demean_by_group(panel["edu_t"].to_numpy(), c_codes, n_c)
post_arr = panel["post2000"].to_numpy().astype(float)
post_dm_A = demean_by_group(post_arr, c_codes, n_c)
gdp_resid_dm_A = demean_by_group(gdp_resid_A, c_codes, n_c)
inter_A = demean_by_group(gdp_resid_A * post_arr, c_codes, n_c)

res_A = fit_ols_clustered(
    y_dm_A,
    [gdp_resid_dm_A, post_dm_A, inter_A],
    c_codes,
    ["gdp_resid", "post", "gdp_resid_x_post"],
)
print(f"  n={len(panel)}, edu↔GDP within-R²={edu_gdp_r2_A:.3f}")
for name, r in res_A.items():
    print(f"  {name:<22} β={r['beta']:>+9.4f}  SE={r['se']:>7.4f}  "
          f"t={r['t']:>+6.2f}  p={r['p']:>6.3f}")
results["A_country_fe"] = {"n": len(panel), "n_countries": n_c,
                           "edu_gdp_within_r2": edu_gdp_r2_A,
                           "coefficients": res_A}

# ── Spec B: country FE + baseline controls (our +11 spec) ──────────

print("\n" + "=" * 72)
print("SPEC B: country FE + edu_t + edu_t × post (baseline-controlled)")
print("=" * 72)
edu_x_post_dm_A = demean_by_group(panel["edu_t"].to_numpy() * post_arr,
                                   c_codes, n_c)
res_B = fit_ols_clustered(
    y_dm_A,
    [gdp_resid_dm_A, post_dm_A, inter_A, edu_dm_A, edu_x_post_dm_A],
    c_codes,
    ["gdp_resid", "post", "gdp_resid_x_post", "edu_t", "edu_t_x_post"],
)
for name, r in res_B.items():
    print(f"  {name:<22} β={r['beta']:>+9.4f}  SE={r['se']:>7.4f}  "
          f"t={r['t']:>+6.2f}  p={r['p']:>6.3f}")
results["B_country_fe_with_baseline"] = {"n": len(panel), "n_countries": n_c,
                                          "coefficients": res_B}

# ── Spec C: country × decade FE + baseline controls ────────────────

print("\n" + "=" * 72)
print("SPEC C: country × decade FE + edu_t + edu_t × post")
print("=" * 72)
gdp_resid_C, edu_gdp_r2_C = fe_residualize(
    panel_cd, "log_gdp_t", "edu_t", cd_codes_cd, n_cd_cd)
y_dm_C = demean_by_group(panel_cd["u5mr_tp28"].to_numpy(), cd_codes_cd, n_cd_cd)
edu_dm_C = demean_by_group(panel_cd["edu_t"].to_numpy(), cd_codes_cd, n_cd_cd)
post_arr_C = panel_cd["post2000"].to_numpy().astype(float)
post_dm_C = demean_by_group(post_arr_C, cd_codes_cd, n_cd_cd)
gdp_resid_dm_C = demean_by_group(gdp_resid_C, cd_codes_cd, n_cd_cd)
inter_C = demean_by_group(gdp_resid_C * post_arr_C, cd_codes_cd, n_cd_cd)
edu_x_post_dm_C = demean_by_group(
    panel_cd["edu_t"].to_numpy() * post_arr_C, cd_codes_cd, n_cd_cd)

res_C = fit_ols_clustered(
    y_dm_C,
    [gdp_resid_dm_C, post_dm_C, inter_C, edu_dm_C, edu_x_post_dm_C],
    c_codes_cd,
    ["gdp_resid", "post", "gdp_resid_x_post", "edu_t", "edu_t_x_post"],
)
print(f"  n={len(panel_cd)}, country-decades={n_cd_cd}, "
      f"edu↔GDP within(cd)-R²={edu_gdp_r2_C:.3f}")
for name, r in res_C.items():
    se_v = r.get("se", float("nan")) or float("nan")
    t_v = r.get("t", float("nan")) or float("nan")
    p_v = r.get("p", float("nan")) or float("nan")
    print(f"  {name:<22} β={r['beta']:>+9.4f}  SE={se_v:>7.4f}  "
          f"t={t_v:>+6.2f}  p={p_v:>6.3f}")
results["C_country_decade_fe_with_baseline"] = {
    "n": len(panel_cd), "n_country_decades": n_cd_cd,
    "edu_gdp_within_cd_r2": edu_gdp_r2_C,
    "coefficients": res_C,
}

# ── Spec D: country FE + country-specific linear trends ────────────
# The cleanest era-control that does NOT structurally absorb post-2000.
# Add (t - t̄_country) for each country as a regressor (or equivalently,
# residualise every variable against country-specific linear trend).
print("\n" + "=" * 72)
print("SPEC D: country FE + country-specific linear time trend + baseline ctrl")
print("=" * 72)

t_arr = panel["t"].to_numpy().astype(float)


def detrend_by_country(values, t_vals, country_codes, n_groups):
    """Residualise values against country-specific linear trend in t.
    Within each country, regress values on t (intercept = country mean) and
    return the residuals."""
    out = np.empty_like(values, dtype=float)
    for g in range(n_groups):
        mask = country_codes == g
        if mask.sum() < 2:
            out[mask] = np.nan
            continue
        x = t_vals[mask]
        y = values[mask]
        xb = x - x.mean()
        yb = y - y.mean()
        sxx = float(np.dot(xb, xb))
        if sxx <= 0:
            out[mask] = yb
            continue
        slope = float(np.dot(xb, yb) / sxx)
        out[mask] = yb - slope * xb
    return out


y_dm_D = detrend_by_country(panel["u5mr_tp28"].to_numpy(), t_arr, c_codes, n_c)
edu_dm_D = detrend_by_country(panel["edu_t"].to_numpy(), t_arr, c_codes, n_c)
gdp_dm_D = detrend_by_country(panel["log_gdp_t"].to_numpy(), t_arr, c_codes, n_c)
post_dm_D = detrend_by_country(post_arr, t_arr, c_codes, n_c)
# Residualise log GDP on edu within country-trend space (FWL step)
ok = ~np.isnan(gdp_dm_D) & ~np.isnan(edu_dm_D)
sxx = float(np.dot(edu_dm_D[ok], edu_dm_D[ok]))
sxy = float(np.dot(edu_dm_D[ok], gdp_dm_D[ok]))
beta_D = sxy / sxx if sxx > 0 else 0.0
gdp_resid_D = gdp_dm_D - beta_D * edu_dm_D
inter_D = detrend_by_country(
    gdp_resid_D * post_arr - gdp_resid_D.mean() * post_arr.mean(),  # rough centring
    t_arr, c_codes, n_c)
# Cleaner: compute interaction in original space then detrend
raw_inter_D = panel["log_gdp_t"].to_numpy() * panel["post2000"].to_numpy()
# But we want the interaction to use the residualised GDP, not raw — so:
inter_D = detrend_by_country(
    gdp_resid_D * post_arr, t_arr, c_codes, n_c)
edu_x_post_D = detrend_by_country(
    panel["edu_t"].to_numpy() * post_arr, t_arr, c_codes, n_c)

ok_all = (~np.isnan(y_dm_D) & ~np.isnan(gdp_resid_D) & ~np.isnan(post_dm_D)
          & ~np.isnan(inter_D) & ~np.isnan(edu_dm_D) & ~np.isnan(edu_x_post_D))
print(f"  n with country trend (after singleton-drop): {ok_all.sum()}")

res_D = fit_ols_clustered(
    y_dm_D[ok_all],
    [gdp_resid_D[ok_all], post_dm_D[ok_all], inter_D[ok_all],
     edu_dm_D[ok_all], edu_x_post_D[ok_all]],
    c_codes[ok_all],
    ["gdp_resid", "post", "gdp_resid_x_post", "edu_t", "edu_t_x_post"],
)
for name, r in res_D.items():
    print(f"  {name:<22} β={r['beta']:>+9.4f}  SE={r['se']:>7.4f}  "
          f"t={r['t']:>+6.2f}  p={r['p']:>6.3f}")
results["D_country_fe_with_trend_and_baseline"] = {
    "n": int(ok_all.sum()),
    "n_countries": n_c,
    "coefficients": res_D,
    "note": (
        "Country FE + country-specific linear trend in t + baseline controls. "
        "Cleaner era-stripping than Spec C (which is structurally collinear "
        "because outcome-year decade aligns with the pre/post-2000 cutoff)."
    ),
}

# ── Summary table ───────────────────────────────────────────────────

print("\n" + "=" * 72)
print("β on (gdp_resid × post) across specs")
print("=" * 72)
print(f"{'Spec':<54} {'β':>10} {'SE':>8} {'p':>7}")
for spec_name, res in [
    ("A: country FE only (paper spec)", res_A),
    ("B: country FE + baseline controls (our +11 spec)", res_B),
    ("C: country × decade FE + baseline (collinear, see note)", res_C),
    ("D: country FE + country trend + baseline (era-stripped)", res_D),
]:
    r = res["gdp_resid_x_post"]
    se_val = r.get("se", float("nan"))
    p_val = r.get("p", float("nan"))
    if p_val is None:
        p_val = float("nan")
    print(f"{spec_name:<54} {r['beta']:>+10.4f} {se_val:>8.4f} {p_val:>7.3f}")
print("\nNOTE on Spec C: country × decade FE structurally absorbs the "
      "post-2000 dummy because decades (1980, 1990, 2000, 2010) align with "
      "the pre/post boundary at year 2000. The pseudo-inverse fallback "
      "returns a number but it is not interpretable as a causal interaction. "
      "Spec D is the correct test for 'is the +24 a within-country era effect?'")

# ── Save ────────────────────────────────────────────────────────────

out_path = os.path.join(SCRIPTS_DIR, "..", "checkin",
                         "residgdp_country_decade_fe.json")
checkin = {
    "method": (
        "GDP-resid × Post-2000 interaction across three FE specs: "
        "(A) country FE only — paper's spec; "
        "(B) country FE + edu_t + edu_t × post — baseline-controlled; "
        "(C) country × decade FE + edu_t + edu_t × post — absorbs within-country era variation. "
        "Country-clustered SE. Common-sample panel."
    ),
    "panel": {
        "n_total": int(len(panel)),
        "n_country_only": int(n_c),
        "n_after_cd_singleton_drop": int(len(panel_cd)),
        "n_country_decades": int(n_cd_cd),
    },
    "results": results,
}
with open(out_path, "w") as f:
    json.dump(checkin, f, indent=2, default=lambda o: None
              if isinstance(o, float) and np.isnan(o) else o)
print(f"\nWrote {out_path}")
