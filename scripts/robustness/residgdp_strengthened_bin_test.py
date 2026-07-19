"""
robustness/residgdp_strengthened_bin_test.py
==============================================

PAPER ANCHOR — Chapter 9 §Robustness to country trends
  Question: does the +5.45 within-bin meta β residual on level U5MR
            post-2000 (from residgdp_within_between_decomp.py) survive
            (a) Maddison GDP backfill, (b) 5pp bin resolution, and
            (c) a country-cluster bootstrap that accounts for the same
            countries appearing in multiple bins over time?

  If yes → real residual donor margin, paper soft-edges its strong
           "no separable channel" claim
  If no  → within-meta was noise + non-independence artefact, paper
           strengthens its claim

Paper's anchor: GDP is invariant in poor countries pre-decolonization
(subsistence floor ~$400-600). The bin test should give residGDP no
signal at low baselines because there is nothing to vary. If the
+5.45 within-meta survives a properly-bootstrapped test under
Maddison-backfilled long-history GDP and 5pp bins, the residual is
real. If it doesn't, the prior result was inverse-variance machinery
weighting the high-baseline bin where the donor mechanism doesn't
apply.

Method:
  - Broader U5MR panel (entry=10, ceiling=90, T_YEARS = 1960–1990,
    lag = 28). Maddison-backfill log GDP.
  - 5pp baseline bins: 10-15, 15-20, ..., 85-90.
  - For each bin × period (all, pre-2000, post-2000):
      (1) Country-FE residualise log GDP on edu within the bin.
      (2) Country-FE regress level U5MR on residGDP with country-
          clustered SE. Record β and SE.
  - Inverse-variance-weighted within-meta β across bins.
  - 2000-rep country-cluster bootstrap on the within-meta:
      Resample countries with replacement; in each bootstrap sample,
      re-bin and re-compute the within-meta β. Report 95% CI on the
      within-meta.

Output: checkin/residgdp_strengthened_bin_test.json
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
    standardize_country_name,
    DATA,
    LAG_GENERATION,
)

T_YEARS = list(range(1960, 1995, 5))
LAG = LAG_GENERATION
ENTRY = 10
CEILING = 90
N_BOOT = 2000
SEED = 2026

# 5pp bins across the active-expansion window
BINS = [(lo, lo + 5) for lo in range(10, 90, 5)]


def load_maddison_wide():
    path = os.path.join(DATA, "external", "maddison_gdppc_wb_equivalent.csv")
    mpd = pd.read_csv(path)
    mpd["country_canon"] = mpd["country_canon"].astype(str).str.lower()
    wide = (mpd.pivot_table(index="country_canon", columns="year",
                            values="gdppc_wb_eq", aggfunc="first"))
    wide.columns = [str(int(c)) for c in wide.columns]
    wide.index.name = "Country"
    return wide


def backfill_with_maddison(wb, maddison):
    mpd_canon = maddison.copy()
    mpd_canon.index = [standardize_country_name(c) for c in mpd_canon.index]
    mpd_canon = mpd_canon[~pd.isna(mpd_canon.index)]
    mpd_canon = mpd_canon[~mpd_canon.index.duplicated(keep="first")]
    filled = wb.copy()
    for col in filled.columns:
        if col not in mpd_canon.columns:
            continue
        nan_mask = filled[col].isna()
        if not nan_mask.any():
            continue
        canon = [standardize_country_name(c) for c in filled.index[nan_mask]]
        for wb_idx, c in zip(filled.index[nan_mask], canon):
            if c in mpd_canon.index:
                v = mpd_canon.at[c, col]
                if pd.notna(v):
                    filled.at[wb_idx, col] = v
    return filled


def fe_regress(panel, x_col, y_col):
    p = panel.dropna(subset=[x_col, y_col]).copy()
    counts = p.groupby("country").size()
    keep = counts[counts >= 2].index
    p = p[p["country"].isin(keep)]
    if len(p) < 10:
        return None
    x = (p[x_col] - p.groupby("country")[x_col].transform("mean")).to_numpy()
    y = (p[y_col] - p.groupby("country")[y_col].transform("mean")).to_numpy()
    sxx = float(np.dot(x, x))
    if sxx <= 0:
        return None
    sxy = float(np.dot(x, y))
    beta = sxy / sxx
    resid = y - beta * x
    codes, _ = pd.factorize(p["country"], sort=False)
    xe = x * resid
    sum_g = np.bincount(codes, weights=xe)
    meat = float(np.dot(sum_g, sum_g))
    se = float(np.sqrt(meat / (sxx ** 2))) if meat > 0 else np.nan
    return {"beta": float(beta), "se": se,
            "n": int(len(p)), "n_countries": int(p["country"].nunique())}


def within_meta(panel, outcome_col, bins):
    """Compute within-bin βs and inverse-variance meta-aggregate."""
    bin_results = {}
    betas, ses = [], []
    for lo, hi in bins:
        sub = panel[(panel["edu_t"] >= lo) & (panel["edu_t"] <= hi)]
        if len(sub) < 15:
            bin_results[f"{lo}-{hi}"] = None
            continue
        out = fe_residualize_gdp(sub)
        if out is None:
            bin_results[f"{lo}-{hi}"] = None
            continue
        sub_r, _ = out
        sub_r = sub_r.dropna(subset=["gdp_resid", outcome_col])
        res = fe_regress(sub_r, "gdp_resid", outcome_col)
        bin_results[f"{lo}-{hi}"] = res
        if res is None or not np.isfinite(res["se"]) or res["se"] <= 0:
            continue
        betas.append(res["beta"])
        ses.append(res["se"])
    if not betas:
        return None, bin_results
    w = 1.0 / np.asarray(ses) ** 2
    beta_meta = float(np.sum(np.asarray(betas) * w) / np.sum(w))
    se_meta = float(1.0 / np.sqrt(np.sum(w)))
    from scipy import stats
    t = beta_meta / se_meta if se_meta > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(t))) if not np.isnan(t) else np.nan
    return {"beta": beta_meta, "se_meta_iv": se_meta,
            "t_iv": float(t), "p_iv": float(p),
            "n_bins": len(betas)}, bin_results


def bootstrap_meta(panel, outcome_col, bins, n_boot, rng):
    """Country-cluster bootstrap on the within-meta β.
    Each replicate: resample countries with replacement, re-bin,
    re-compute meta. Returns array of bootstrap β_meta values."""
    countries = panel["country"].unique()
    n_c = len(countries)
    # Pre-group for speed
    by_country = {c: panel[panel["country"] == c] for c in countries}
    betas = []
    for rep in range(n_boot):
        sample = rng.choice(countries, size=n_c, replace=True)
        parts = []
        for i, c in enumerate(sample):
            df = by_country[c].copy()
            df["country"] = f"{c}__{i}"  # unique label per draw
            parts.append(df)
        boot = pd.concat(parts, ignore_index=True)
        meta_res, _ = within_meta(boot, outcome_col, bins)
        if meta_res is None:
            continue
        betas.append(meta_res["beta"])
        if (rep + 1) % 250 == 0:
            print(f"    boot {rep+1}/{n_boot}: latest β_meta = "
                  f"{meta_res['beta']:+.3f}")
    return np.asarray(betas)


# ── Load data ────────────────────────────────────────────────────────

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
u5mr_raw = load_wb("child_mortality_u5.csv")
gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")
maddison = load_maddison_wide()
gdp_back = backfill_with_maddison(gdp_raw, maddison)
edu_annual = interpolate_to_annual(edu_raw, "lower_sec")
entry_years = precompute_entry_years(edu_annual)
cohort = entry_years[ENTRY]

panel = build_panel(edu_annual, u5mr_raw, gdp_back, T_YEARS, LAG, "u5mr_tp28")
panel = filter_panel(panel, cohort, CEILING).copy()
panel = panel.dropna(subset=["edu_t", "log_gdp_t", "u5mr_tp28"])
panel["post2000"] = ((panel["t"] + LAG) >= 2000).astype(int)
print(f"Panel (Maddison-backfilled): n={len(panel)}, "
      f"countries={panel['country'].nunique()}")
print(f"  pre-2000:  n={(panel.post2000==0).sum()}, "
      f"countries={panel[panel.post2000==0]['country'].nunique()}")
print(f"  post-2000: n={(panel.post2000==1).sum()}, "
      f"countries={panel[panel.post2000==1]['country'].nunique()}")

# Compare against WB-only sample
panel_wb_only = build_panel(edu_annual, u5mr_raw, gdp_raw, T_YEARS, LAG, "u5mr_tp28")
panel_wb_only = filter_panel(panel_wb_only, cohort, CEILING).copy()
panel_wb_only = panel_wb_only.dropna(subset=["edu_t", "log_gdp_t", "u5mr_tp28"])
print(f"  (WB-only sample for comparison: n={len(panel_wb_only)}, "
      f"countries={panel_wb_only['country'].nunique()})")

# GDP variance by baseline bin (the user's anchor: GDP invariant in
# poor countries pre-decolonization)
print("\nGDP variance by baseline bin (within-country, log GDP):")
for lo, hi in [(10, 20), (20, 30), (30, 50), (50, 90)]:
    sub = panel[(panel["edu_t"] >= lo) & (panel["edu_t"] <= hi)]
    if len(sub) < 10:
        continue
    within_var = sub.groupby("country")["log_gdp_t"].var().mean()
    print(f"  baseline {lo}-{hi}%: n={len(sub):>4}, "
          f"mean within-country log-GDP variance = {within_var:.4f}")

# ── Run within-meta for level U5MR, with bootstrap ──────────────────

rng = np.random.default_rng(SEED)

results = {}

for period_label, period_mask in [
    ("all", panel["post2000"].notna()),
    ("pre_2000", panel["post2000"] == 0),
    ("post_2000", panel["post2000"] == 1),
]:
    print("\n" + "=" * 78)
    print(f"PERIOD: {period_label}")
    print("=" * 78)
    sub_panel = panel[period_mask]
    if len(sub_panel) < 50:
        print("  too few obs")
        continue
    meta_res, bin_results = within_meta(sub_panel, "u5mr_tp28", BINS)
    if meta_res is None:
        print("  meta failed")
        continue
    print(f"  Bins with valid β: {meta_res['n_bins']}")
    for bin_label, res in bin_results.items():
        if res is None:
            continue
        print(f"    {bin_label:<7} β={res['beta']:>+9.3f}  SE={res['se']:>7.3f}  "
              f"n={res['n']:>4}")
    print(f"  Within-meta β (IV-weighted): {meta_res['beta']:+.4f}  "
          f"SE_iv={meta_res['se_meta_iv']:.4f}  p_iv={meta_res['p_iv']:.4f}")

    print(f"  Bootstrap ({N_BOOT} reps, country-cluster)...")
    boot_betas = bootstrap_meta(sub_panel, "u5mr_tp28", BINS, N_BOOT, rng)
    if len(boot_betas) < N_BOOT // 4:
        print(f"  Bootstrap insufficient — {len(boot_betas)} successful reps")
        boot_summary = None
    else:
        ci_lo = float(np.percentile(boot_betas, 2.5))
        ci_hi = float(np.percentile(boot_betas, 97.5))
        median = float(np.median(boot_betas))
        excludes_zero = (ci_lo > 0) or (ci_hi < 0)
        print(f"  Bootstrap median: {median:+.3f}")
        print(f"  Bootstrap 95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]  "
              f"excludes zero: {excludes_zero}")
        boot_summary = {
            "median": median,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "excludes_zero_95": bool(excludes_zero),
            "n_reps_successful": int(len(boot_betas)),
        }
    results[period_label] = {
        "meta": meta_res,
        "bins": bin_results,
        "bootstrap": boot_summary,
    }

# ── Print summary ──────────────────────────────────────────────────

print("\n" + "=" * 78)
print("STRENGTHENED BIN TEST SUMMARY (Maddison-backfilled, 5pp bins, 2000-rep boot)")
print("=" * 78)
print(f"{'Period':<12} {'within-meta β':>14} {'IV-SE p':>9} | "
      f"{'boot 95% CI':>22} {'≠0?':>5}")
for period_label, r in results.items():
    m = r["meta"]
    b = r["bootstrap"]
    if b is None:
        boot_str = "---"
        excl = "?"
    else:
        boot_str = f"[{b['ci_lo']:+.2f}, {b['ci_hi']:+.2f}]"
        excl = "YES" if b["excludes_zero_95"] else "no"
    print(f"{period_label:<12} {m['beta']:>+14.4f} {m['p_iv']:>9.4f} | "
          f"{boot_str:>22} {excl:>5}")

# ── Save ────────────────────────────────────────────────────────────

out_path = os.path.join(SCRIPTS_DIR, "..", "checkin",
                         "residgdp_strengthened_bin_test.json")
checkin = {
    "method": (
        "Strengthened bin test for level-U5MR residGDP residual. "
        "Maddison-backfilled GDP. 5pp baseline bins on lower-sec "
        "completion (10-15, 15-20, ..., 85-90). Within each bin, "
        "country-FE residualisation of log GDP on edu, then "
        "country-FE regression of level U5MR on residGDP with "
        "country-clustered SE. Inverse-variance-weighted within-meta "
        "across bins. 2000-rep country-cluster bootstrap on the "
        "within-meta to account for non-independence (countries appear "
        "in multiple bins over time)."
    ),
    "parameters": {
        "T_years": [int(t) for t in T_YEARS],
        "lag": LAG, "entry": ENTRY, "ceiling": CEILING,
        "n_bins": len(BINS), "n_boot": N_BOOT, "seed": SEED,
    },
    "panel": {
        "n_maddison_backfilled": int(len(panel)),
        "n_wb_only": int(len(panel_wb_only)),
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
