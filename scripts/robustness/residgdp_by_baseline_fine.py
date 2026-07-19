"""
robustness/residgdp_by_baseline_fine.py
========================================

TEST 1 from the residGDP audit plan.

Question: where in baseline-education space does the post-2000 residGDP
signal on level U5MR live? Is it concentrated in a narrow low-baseline
band (consistent with Lutz/GAVI's high-mortality-target story), or
diffuse?

Method:
  - Broader U5MR panel from build_panel; entry-cohort filter
    (entry >= 10, ceiling <= 90) — same as paper's u5mr_residual_by_year.
  - Fine baseline bands: 5pp steps for active-expansion range.
  - Within each band × period:
      (1) Within-country FE regression Y on residGDP (FWL with within-band
          residualisation on edu_t).
      (2) Country-clustered SE on β.
      (3) Country-bootstrap 95% CI on β (N_BOOT reps).
  - Outcomes: level U5MR (the paper's bounded-exception variable) and
    log U5MR (the spec where the paper says signal vanishes).

Compares against:
  - checkin/u5mr_by_edu_level.json (existing, no-entry-filter sample)
  - checkin/u5mr_residual_by_year.json (existing, no-band stratification)

Output: checkin/residgdp_by_baseline_fine.json

This is a DIAGNOSTIC — not yet a paper-anchor number.
"""

import os
import sys

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
    LAG_GENERATION,
)
import json  # noqa: E402

T_YEARS = list(range(1960, 1995, 5))
LAG = LAG_GENERATION
ENTRY = 10
CEILING = 90
N_BOOT = 200
SEED = 2026

# Baseline bands: 5pp steps for active-expansion range.
BANDS = [
    (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40),
    (40, 45), (45, 50), (50, 55), (55, 60), (60, 70), (70, 80), (80, 90),
    # Wider bands for comparison
    (10, 20), (20, 30), (30, 50), (50, 90), (10, 90),
]


def fe_regress_clustered(panel, x_col, y_col):
    """Country-FE OLS of y on x, country-clustered SE.
    Returns (beta, se, t, p, r2, n, n_countries)."""
    p = panel.dropna(subset=[x_col, y_col]).copy()
    counts = p.groupby("country").size()
    keep = counts[counts >= 2].index
    p = p[p["country"].isin(keep)]
    if len(p) < 10:
        return None
    x = (p[x_col] - p.groupby("country")[x_col].transform("mean")).to_numpy()
    y = (p[y_col] - p.groupby("country")[y_col].transform("mean")).to_numpy()
    sxx = float(np.dot(x, x))
    sxy = float(np.dot(x, y))
    syy = float(np.dot(y, y))
    if sxx <= 0:
        return None
    beta = sxy / sxx
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else np.nan
    resid = y - beta * x
    codes, _ = pd.factorize(p["country"], sort=False)
    xe = x * resid
    sum_g = np.bincount(codes, weights=xe)
    meat = float(np.dot(sum_g, sum_g))
    var = meat / (sxx ** 2)
    se = float(np.sqrt(var)) if var > 0 else np.nan
    from scipy import stats
    t = beta / se if se and se > 0 else np.nan
    pval = (2 * (1 - stats.norm.cdf(abs(t)))) if not np.isnan(t) else np.nan
    return {
        "beta": float(beta), "se": float(se), "t": float(t), "p": float(pval),
        "r2": float(r2), "n": int(len(p)), "n_countries": int(p["country"].nunique()),
    }


def bootstrap_beta(panel, x_col, y_col, n_boot, rng):
    """Country-cluster bootstrap β (vectorised on country groups).
    Returns dict with 95% CI."""
    countries = panel["country"].unique()
    if len(countries) < 5:
        return None
    # Pre-group by country for fast resample
    groups = {c: panel[panel["country"] == c][[x_col, y_col]].to_numpy()
              for c in countries}
    n_c = len(countries)
    betas = []
    for _ in range(n_boot):
        sample = rng.choice(countries, size=n_c, replace=True)
        # Each replicate: country mean = mean of that country's rows
        xs, ys, gid = [], [], []
        for i, c in enumerate(sample):
            g = groups[c]
            if len(g) < 2:
                continue
            xm = g[:, 0].mean()
            ym = g[:, 1].mean()
            xs.append(g[:, 0] - xm)
            ys.append(g[:, 1] - ym)
            gid.append(np.full(len(g), i))
        if not xs:
            continue
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        ok = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[ok], y[ok]
        if len(x) < 10:
            continue
        sxx = float(np.dot(x, x))
        if sxx <= 0:
            continue
        betas.append(float(np.dot(x, y) / sxx))
    if len(betas) < n_boot // 4:
        return None
    arr = np.asarray(betas)
    return {
        "median": float(np.median(arr)),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n_replicates": int(len(arr)),
    }


def analyse_band_period(panel, band, period_label, outcome_col, rng,
                       with_bootstrap=True):
    """Run within-band, within-period analysis. Returns dict."""
    lo, hi = band
    sub = panel[(panel["edu_t"] >= lo) & (panel["edu_t"] <= hi)].copy()
    if period_label == "pre_2000":
        sub = sub[(sub["t"] + LAG) < 2000]
    elif period_label == "post_2000":
        sub = sub[(sub["t"] + LAG) >= 2000]
    if len(sub) < 15 or sub["country"].nunique() < 3:
        return None

    resid = fe_residualize_gdp(sub)
    if resid is None:
        return None
    sub_r, edu_gdp_r2 = resid
    sub_r = sub_r.dropna(subset=["gdp_resid", outcome_col])
    if len(sub_r) < 15:
        return None

    res = fe_regress_clustered(sub_r, "gdp_resid", outcome_col)
    if res is None:
        return None

    if with_bootstrap and len(sub_r) >= 30 and sub_r["country"].nunique() >= 10:
        boot = bootstrap_beta(sub_r, "gdp_resid", outcome_col, N_BOOT, rng)
    else:
        boot = None

    return {
        **res,
        "edu_gdp_within_r2": float(edu_gdp_r2),
        "bootstrap": boot,
    }


# ── Load data ────────────────────────────────────────────────────────

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
u5mr_raw = load_wb("child_mortality_u5.csv")
gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")

edu_annual = interpolate_to_annual(edu_raw, "lower_sec")
entry_years = precompute_entry_years(edu_annual)
cohort = entry_years[ENTRY]

# ── Build U5MR panel ────────────────────────────────────────────────

panel = build_panel(edu_annual, u5mr_raw, gdp_raw, T_YEARS, LAG, "u5mr_tp28")
panel = filter_panel(panel, cohort, CEILING).copy()
panel = panel.dropna(subset=["edu_t", "log_gdp_t", "u5mr_tp28"])
panel["log_u5mr_tp28"] = np.log(
    panel["u5mr_tp28"].where(panel["u5mr_tp28"] > 0)
)
panel["post2000"] = ((panel["t"] + LAG) >= 2000).astype(int)
print(f"U5MR panel (entry={ENTRY}, ceiling={CEILING}): "
      f"n={len(panel)}, countries={panel['country'].nunique()}")
print(f"  pre-2000:  n={(panel.post2000==0).sum()}, "
      f"countries={panel[panel.post2000==0]['country'].nunique()}")
print(f"  post-2000: n={(panel.post2000==1).sum()}, "
      f"countries={panel[panel.post2000==1]['country'].nunique()}")

# ── Run bands ────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)

results = {"level_u5mr": {}, "log_u5mr": {}}

for outcome_label, outcome_col in [
    ("level_u5mr", "u5mr_tp28"),
    ("log_u5mr", "log_u5mr_tp28"),
]:
    print("\n" + "=" * 90)
    print(f"OUTCOME: {outcome_label}")
    print("=" * 90)
    print(f"  {'Band':<10} {'Period':<10} {'n':>4} {'cn':>3} | "
          f"{'beta':>10} {'SE':>7} {'t':>6} {'p':>7} | "
          f"{'boot 95% CI':>22}")
    for band in BANDS:
        for period_label in ["all", "pre_2000", "post_2000"]:
            res = analyse_band_period(
                panel, band, period_label, outcome_col, rng,
                with_bootstrap=(period_label == "all"
                                or (band[1] - band[0]) >= 10),
            )
            key = f"{band[0]}-{band[1]}|{period_label}"
            if res is None:
                results[outcome_label][key] = None
                print(f"  {band[0]}-{band[1]:<6} {period_label:<10} ---")
                continue
            results[outcome_label][key] = res
            boot_str = "---"
            if res["bootstrap"] is not None:
                b = res["bootstrap"]
                boot_str = f"[{b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}]"
            print(f"  {band[0]}-{band[1]:<6} {period_label:<10} "
                  f"{res['n']:>4} {res['n_countries']:>3} | "
                  f"{res['beta']:>+10.4f} {res['se']:>7.4f} {res['t']:>+6.2f} "
                  f"{res['p']:>7.3f} | {boot_str:>22}")

# ── Save ────────────────────────────────────────────────────────────

out_dir = os.path.join(SCRIPTS_DIR, "..", "checkin")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "residgdp_by_baseline_fine.json")

checkin = {
    "method": (
        "Baseline-stratified residGDP β on level and log U5MR. Broader U5MR "
        "panel from build_panel; entry-cohort filter (entry=10, ceiling=90). "
        "FWL residualisation of log GDP on lower-sec edu WITHIN each band × "
        "period subset (local residualisation). Country-clustered SE. "
        "Country-cluster bootstrap 95% CI (N_BOOT reps) for bands with n>=30 "
        "and >=10 countries. Diagnostic — NOT a paper-anchor."
    ),
    "parameters": {
        "T_years": [int(t) for t in T_YEARS],
        "lag": LAG,
        "entry": ENTRY,
        "ceiling": CEILING,
        "n_boot": N_BOOT,
        "seed": SEED,
    },
    "panel": {
        "n_total": int(len(panel)),
        "n_countries": int(panel["country"].nunique()),
        "n_pre_2000": int((panel.post2000 == 0).sum()),
        "n_post_2000": int((panel.post2000 == 1).sum()),
    },
    "results": results,
}

with open(out_path, "w") as f:
    json.dump(checkin, f, indent=2, default=lambda o: None
              if isinstance(o, float) and np.isnan(o) else o)
print(f"\nWrote {out_path}")
