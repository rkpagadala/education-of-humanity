"""
robustness/log_u5mr_pre2000_trends.py
========================================

Residual check from the audit: log U5MR pre-2000 (broader panel, n=203)
gives β_(residGDP) = −0.368, p = 0.018 — a real signal in the traditional
direction (more residGDP → less mortality).

Does this survive country-specific linear trends? If yes → separable
residGDP channel pre-2000 that the paper's framework has to explain. If
no → consistent with the paper (signal is country-specific
education-driven trajectory).

Method: same broader U5MR panel as residgdp_by_baseline_fine.py
(entry=10, ceiling=90, T_YEARS = 1960–1990, lag = 25). Restrict to
pre-2000 outcome years (T+25 < 2000). Compare four specs:

  A. Country FE only — replicates the −0.368
  B. Country FE + linear time trend (global) — controls for any global
     mortality trend
  C. Country FE + country-specific linear trends — absorbs each
     country's own (education-driven) trajectory
  D. Country FE + country-specific linear trends + edu_t as additional
     regressor — stricter still: any residGDP signal must survive
     conditioning on the country's own education at T

Output: checkin/log_u5mr_pre2000_trends.json
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
    precompute_entry_years,
    filter_panel,
    fe_residualize_gdp,
    LAG_CHILDREARING,
)

T_YEARS = list(range(1960, 1995, 5))
LAG = LAG_CHILDREARING  # U5MR childrearing window (was hardcoded 25)
ENTRY = 10
CEILING = 90


def country_demean(values, codes, n_groups):
    counts = np.bincount(codes, minlength=n_groups)
    sums = np.bincount(codes, weights=values, minlength=n_groups)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return values - means[codes]


def detrend_country(values, t_vals, codes, n_groups):
    """Residualise values against country-specific linear trend in t."""
    out = np.empty_like(values, dtype=float)
    for g in range(n_groups):
        mask = codes == g
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


def fit_ols_clustered(y, X_cols, cluster_codes, names):
    X = np.column_stack(X_cols)
    XtX = X.T @ X
    try:
        beta = np.linalg.solve(XtX, X.T @ y)
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        XtX_inv = np.linalg.pinv(XtX)
    resid = y - X @ beta
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
            "se": float(se[i]) if not np.isnan(se[i]) else None,
            "t": float(t) if not np.isnan(t) else None,
            "p": float(p) if not np.isnan(p) else None,
        }
    return out


# ── Load and build the broader U5MR pre-2000 subsample ──────────────

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
u5mr_raw = load_wb("child_mortality_u5.csv")
gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")
edu_annual = interpolate_to_annual(edu_raw, "lower_sec")
entry_years = precompute_entry_years(edu_annual)
cohort = entry_years[ENTRY]

panel = build_panel(edu_annual, u5mr_raw, gdp_raw, T_YEARS, LAG, "u5mr_tp12")
panel = filter_panel(panel, cohort, CEILING).copy()
panel = panel.dropna(subset=["edu_t", "log_gdp_t", "u5mr_tp12"])
panel["log_u5mr_tp12"] = np.log(panel["u5mr_tp12"].where(panel["u5mr_tp12"] > 0))
panel = panel.dropna(subset=["log_u5mr_tp12"])
panel_pre = panel[(panel["t"] + LAG) < 2000].reset_index(drop=True).copy()
print(f"Pre-2000 panel: n={len(panel_pre)}, "
      f"countries={panel_pre['country'].nunique()}")

c_codes, c_uniq = pd.factorize(panel_pre["country"], sort=False)
n_c = len(c_uniq)
t_arr = panel_pre["t"].to_numpy().astype(float)

# Residualise log GDP against edu within country FE (FWL)
resid_out = fe_residualize_gdp(panel_pre)
if resid_out is None:
    raise RuntimeError("residualisation failed")
panel_r, edu_gdp_r2 = resid_out
print(f"edu↔log-GDP within-R² (pre-2000 subset): {edu_gdp_r2:.3f}")

# Sample sizes after fe_residualize_gdp (which drops singletons)
print(f"After fe_residualize: n={len(panel_r)}, "
      f"countries={panel_r['country'].nunique()}")

# Re-factorize after potential row drops
c_codes, c_uniq = pd.factorize(panel_r["country"], sort=False)
n_c = len(c_uniq)
t_arr = panel_r["t"].to_numpy().astype(float)

y = panel_r["log_u5mr_tp12"].to_numpy()
gdp_resid = panel_r["gdp_resid"].to_numpy()
edu = panel_r["edu_t"].to_numpy()

# ── Spec A: country FE only ─────────────────────────────────────────

print("\n" + "=" * 72)
print("SPEC A: country FE only")
print("=" * 72)
y_dm = country_demean(y, c_codes, n_c)
gdp_dm = country_demean(gdp_resid, c_codes, n_c)
res_A = fit_ols_clustered(
    y_dm, [gdp_dm], c_codes, ["gdp_resid"],
)
for n, r in res_A.items():
    se_v = r.get("se", float("nan")) or float("nan")
    t_v = r.get("t", float("nan")) or float("nan")
    p_v = r.get("p", float("nan")) or float("nan")
    print(f"  {n:<22} β={r['beta']:>+9.4f}  SE={se_v:>7.4f}  "
          f"t={t_v:>+6.2f}  p={p_v:>6.3f}")

# ── Spec B: country FE + global linear trend ───────────────────────

print("\n" + "=" * 72)
print("SPEC B: country FE + global linear time trend")
print("=" * 72)
t_dm = country_demean(t_arr, c_codes, n_c)
res_B = fit_ols_clustered(
    y_dm, [gdp_dm, t_dm], c_codes, ["gdp_resid", "t"],
)
for n, r in res_B.items():
    se_v = r.get("se", float("nan")) or float("nan")
    t_v = r.get("t", float("nan")) or float("nan")
    p_v = r.get("p", float("nan")) or float("nan")
    print(f"  {n:<22} β={r['beta']:>+9.4f}  SE={se_v:>7.4f}  "
          f"t={t_v:>+6.2f}  p={p_v:>6.3f}")

# ── Spec C: country FE + country-specific linear trends ────────────

print("\n" + "=" * 72)
print("SPEC C: country FE + country-specific linear time trends")
print("=" * 72)
y_dm_C = detrend_country(y, t_arr, c_codes, n_c)
gdp_dm_C = detrend_country(gdp_resid, t_arr, c_codes, n_c)
ok_C = ~np.isnan(y_dm_C) & ~np.isnan(gdp_dm_C)
n_C_eff = int(ok_C.sum())
print(f"  Effective n after singleton-drop: {n_C_eff}")
res_C = fit_ols_clustered(
    y_dm_C[ok_C], [gdp_dm_C[ok_C]],
    c_codes[ok_C], ["gdp_resid"],
)
for n, r in res_C.items():
    se_v = r.get("se", float("nan")) or float("nan")
    t_v = r.get("t", float("nan")) or float("nan")
    p_v = r.get("p", float("nan")) or float("nan")
    print(f"  {n:<22} β={r['beta']:>+9.4f}  SE={se_v:>7.4f}  "
          f"t={t_v:>+6.2f}  p={p_v:>6.3f}")

# ── Spec D: country FE + country-specific trends + edu_t ───────────

print("\n" + "=" * 72)
print("SPEC D: country FE + country trends + edu_t as additional control")
print("=" * 72)
edu_dm_D = detrend_country(edu, t_arr, c_codes, n_c)
ok_D = ok_C & ~np.isnan(edu_dm_D)
res_D = fit_ols_clustered(
    y_dm_C[ok_D],
    [gdp_dm_C[ok_D], edu_dm_D[ok_D]],
    c_codes[ok_D],
    ["gdp_resid", "edu_t"],
)
print(f"  Effective n: {int(ok_D.sum())}")
for n, r in res_D.items():
    se_v = r.get("se", float("nan")) or float("nan")
    t_v = r.get("t", float("nan")) or float("nan")
    p_v = r.get("p", float("nan")) or float("nan")
    print(f"  {n:<22} β={r['beta']:>+9.4f}  SE={se_v:>7.4f}  "
          f"t={t_v:>+6.2f}  p={p_v:>6.3f}")

# ── Summary ─────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("β on residGDP under widening era-controls (pre-2000, log U5MR)")
print("=" * 72)
print(f"{'Spec':<54} {'β':>10} {'SE':>8} {'p':>7}")
for label, res in [
    ("A: country FE only", res_A),
    ("B: + global linear trend", res_B),
    ("C: + country-specific linear trends", res_C),
    ("D: + country trends + edu_t control", res_D),
]:
    r = res["gdp_resid"]
    se_v = r.get("se", float("nan")) or float("nan")
    p_v = r.get("p", float("nan")) or float("nan")
    print(f"{label:<54} {r['beta']:>+10.4f} {se_v:>8.4f} {p_v:>7.3f}")

# ── Save ────────────────────────────────────────────────────────────

out_path = os.path.join(SCRIPTS_DIR, "..", "checkin",
                         "log_u5mr_pre2000_trends.json")
checkin = {
    "method": (
        "Test whether the pre-2000 log U5MR residGDP signal (β=-0.368, "
        "p=0.018 under country FE; from residgdp_by_baseline_fine.py) "
        "survives country-specific linear time trends. If it survives, "
        "it is a separable residGDP channel the paper's framework would "
        "have to explain. If it collapses, it is education-driven "
        "trajectory."
    ),
    "panel": {
        "n": int(len(panel_r)),
        "n_countries": int(n_c),
        "edu_gdp_within_r2": float(edu_gdp_r2),
    },
    "results": {
        "A_country_fe": res_A,
        "B_country_fe_plus_global_trend": res_B,
        "C_country_fe_plus_country_trends": res_C,
        "D_country_fe_plus_country_trends_plus_edu": res_D,
    },
}
with open(out_path, "w") as f:
    json.dump(checkin, f, indent=2, default=lambda o: None
              if isinstance(o, float) and np.isnan(o) else o)
print(f"\nWrote {out_path}")
