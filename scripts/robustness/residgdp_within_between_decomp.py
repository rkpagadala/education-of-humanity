"""
robustness/residgdp_within_between_decomp.py
==============================================

PAPER ANCHOR — Chapter 9 §Robustness to country trends
  Question: when the pooled residualised-GDP signal on U5MR (post-2000
            +14.12, p=0.047; pre-2000 log -0.37, p=0.018) is decomposed
            into within-baseline-bin and between-baseline-bin components,
            how much is composition versus within-bin causal?

The observational test of "is there another factor beyond education":
stratify by parental-education baseline, ask whether within each tier
residGDP predicts mortality. If yes within-bin → another factor. If
the signal lives only between bins → composition, education alone
explains it.

Method:
  - Broader U5MR panel (entry=10, ceiling=90); same as
    residgdp_by_baseline_fine.py.
  - Bins on parental lower-sec completion edu_t: 10-20, 20-30, 30-50,
    50-90 (also report 10-90 pooled for reference).
  - For each (outcome, period) cell:
      (1) Pooled β_pool from country-FE regression on full bin-union.
      (2) Within-bin β: for each bin, country-FE β + clustered SE.
          Inverse-variance-weighted meta-aggregate across bins.
      (3) Between-bin β: collapse each bin to one (mean residGDP, mean
          outcome) point and regress (weighted by bin size).
      (4) Decomposition: β_pool ≈ β_within + composition residual.

Outcomes: level U5MR, log U5MR.
Periods: all, pre-2000, post-2000.

Output: checkin/residgdp_within_between_decomp.json
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
    LAG_GENERATION,
)

T_YEARS = list(range(1960, 1995, 5))
LAG = LAG_GENERATION
ENTRY = 10
CEILING = 90

BINS = [(10, 20), (20, 30), (30, 50), (50, 90)]
POOL = (10, 90)


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
    syy = float(np.dot(y, y))
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
    p_val = 2 * (1 - stats.norm.cdf(abs(t))) if not np.isnan(t) else np.nan
    return {
        "beta": float(beta), "se": float(se), "t": float(t), "p": float(p_val),
        "r2": float(r2), "n": int(len(p)),
        "n_countries": int(p["country"].nunique()),
    }


def residualise_and_regress(panel, outcome_col):
    """Residualise log GDP on edu in the panel, then regress outcome
    on the residual with country FE."""
    out = fe_residualize_gdp(panel)
    if out is None:
        return None
    sub, _ = out
    sub = sub.dropna(subset=["gdp_resid", outcome_col])
    return fe_regress(sub, "gdp_resid", outcome_col)


# ── Load data ────────────────────────────────────────────────────────

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
u5mr_raw = load_wb("child_mortality_u5.csv")
gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")
edu_annual = interpolate_to_annual(edu_raw, "lower_sec")
entry_years = precompute_entry_years(edu_annual)
cohort = entry_years[ENTRY]

panel = build_panel(edu_annual, u5mr_raw, gdp_raw, T_YEARS, LAG, "u5mr_tp28")
panel = filter_panel(panel, cohort, CEILING).copy()
panel = panel.dropna(subset=["edu_t", "log_gdp_t", "u5mr_tp28"])
panel["log_u5mr_tp28"] = np.log(panel["u5mr_tp28"].where(panel["u5mr_tp28"] > 0))
panel = panel.dropna(subset=["log_u5mr_tp28"])
panel["post2000"] = ((panel["t"] + LAG) >= 2000).astype(int)

print(f"Panel: n={len(panel)}, countries={panel['country'].nunique()}")

# ── Run decomposition for each outcome × period ─────────────────────

def slice_period(p, label):
    if label == "all":
        return p
    if label == "pre_2000":
        return p[p["post2000"] == 0]
    if label == "post_2000":
        return p[p["post2000"] == 1]


def decompose(panel, outcome_col):
    """Within / Between decomposition.
    Returns dict with pooled, within_meta, between, and per-bin results."""
    # (a) Pooled β
    sub_pool = panel[(panel["edu_t"] >= POOL[0]) & (panel["edu_t"] <= POOL[1])]
    pooled = residualise_and_regress(sub_pool, outcome_col)

    # (b) Within-bin βs + inverse-variance meta-aggregate
    bins_out = {}
    betas, ses, ns = [], [], []
    for lo, hi in BINS:
        sub = panel[(panel["edu_t"] >= lo) & (panel["edu_t"] <= hi)]
        res = residualise_and_regress(sub, outcome_col)
        if res is None:
            bins_out[f"{lo}-{hi}"] = None
            continue
        bins_out[f"{lo}-{hi}"] = res
        if res["se"] > 0 and not np.isnan(res["se"]):
            betas.append(res["beta"])
            ses.append(res["se"])
            ns.append(res["n"])

    # Inverse-variance-weighted meta-β
    if betas:
        w = 1.0 / np.asarray(ses) ** 2
        beta_meta = float(np.average(betas, weights=w))
        se_meta = float(1.0 / np.sqrt(np.sum(w)))
        from scipy import stats
        t_meta = beta_meta / se_meta if se_meta > 0 else np.nan
        p_meta = 2 * (1 - stats.norm.cdf(abs(t_meta))) if not np.isnan(t_meta) else np.nan
        within_meta = {"beta": beta_meta, "se": se_meta,
                       "t": float(t_meta), "p": float(p_meta),
                       "n_bins": len(betas)}
    else:
        within_meta = None

    # (c) Between-bin: collapse each bin to (mean gdp_resid, mean outcome),
    # weight by bin size, regress
    between_pts = []
    for lo, hi in BINS:
        sub = panel[(panel["edu_t"] >= lo) & (panel["edu_t"] <= hi)]
        out = fe_residualize_gdp(sub)
        if out is None:
            continue
        sub_r, _ = out
        # Use ALL-country mean of GDP-resid and outcome in the bin
        # (no further within-country demeaning — the residualisation
        # already country-demeaned the GDP side)
        between_pts.append({
            "bin": f"{lo}-{hi}",
            "mean_gdp_resid": float(sub_r["gdp_resid"].mean()),
            "mean_outcome": float(sub_r[outcome_col].mean()),
            "n": int(len(sub_r)),
        })
    if len(between_pts) >= 2:
        bx = np.array([p["mean_gdp_resid"] for p in between_pts])
        by = np.array([p["mean_outcome"] for p in between_pts])
        bw = np.array([p["n"] for p in between_pts])
        bxm = np.average(bx, weights=bw)
        bym = np.average(by, weights=bw)
        bsxx = np.sum(bw * (bx - bxm) ** 2)
        bsxy = np.sum(bw * (bx - bxm) * (by - bym))
        between_beta = float(bsxy / bsxx) if bsxx > 0 else np.nan
        between = {"beta": between_beta, "n_bins": len(between_pts),
                   "bins_used": between_pts}
    else:
        between = None

    return {"pooled": pooled, "within_meta": within_meta,
            "between": between, "by_bin": bins_out}


results = {}
for outcome_label, outcome_col in [
    ("level_u5mr", "u5mr_tp28"),
    ("log_u5mr", "log_u5mr_tp28"),
]:
    results[outcome_label] = {}
    for period_label in ["all", "pre_2000", "post_2000"]:
        sub_panel = slice_period(panel, period_label)
        if len(sub_panel) < 30:
            continue
        results[outcome_label][period_label] = decompose(sub_panel, outcome_col)

# ── Print ────────────────────────────────────────────────────────────

print("\n" + "=" * 90)
print(f"{'Outcome / Period':<28} {'Pooled β':>10} {'p':>7} | "
      f"{'Within meta β':>13} {'p':>7} | {'Between β':>10}")
print("=" * 90)
for outcome_label, d in results.items():
    for period_label, r in d.items():
        pool_b = r["pooled"]["beta"] if r["pooled"] else float("nan")
        pool_p = r["pooled"]["p"] if r["pooled"] else float("nan")
        wm_b = r["within_meta"]["beta"] if r["within_meta"] else float("nan")
        wm_p = r["within_meta"]["p"] if r["within_meta"] else float("nan")
        bt_b = r["between"]["beta"] if r["between"] else float("nan")
        print(f"{outcome_label:<14} {period_label:<14} "
              f"{pool_b:>+10.4f} {pool_p:>7.3f} | "
              f"{wm_b:>+13.4f} {wm_p:>7.3f} | {bt_b:>+10.4f}")

# Bin-by-bin detail for level_u5mr post_2000 and log_u5mr pre_2000
# (the two cells where the pooled signal was significant)
print("\n" + "=" * 90)
print("BIN-BY-BIN DETAIL — the two cells where pooled signal was significant")
print("=" * 90)
for outcome_label, period_label in [("level_u5mr", "post_2000"),
                                     ("log_u5mr", "pre_2000")]:
    r = results[outcome_label][period_label]
    print(f"\n{outcome_label} / {period_label}: pooled β = "
          f"{r['pooled']['beta']:+.4f} (p={r['pooled']['p']:.3f})")
    print(f"  Within-meta β = {r['within_meta']['beta']:+.4f} "
          f"(p={r['within_meta']['p']:.3f}); "
          f"Between-bin β = {r['between']['beta']:+.4f}")
    for bin_label, res in r["by_bin"].items():
        if res is None:
            print(f"  bin {bin_label}: ---")
            continue
        print(f"  bin {bin_label:<7} β={res['beta']:>+9.4f} "
              f"SE={res['se']:>7.4f} p={res['p']:>5.3f} n={res['n']:>4}")

# ── Save ────────────────────────────────────────────────────────────

out_path = os.path.join(SCRIPTS_DIR, "..", "checkin",
                         "residgdp_within_between_decomp.json")
checkin = {
    "method": (
        "Within-bin vs between-bin decomposition of the residGDP-U5MR "
        "relationship on the broader U5MR panel (entry=10, ceiling=90). "
        "Bins on parental lower-sec completion at T: 10-20, 20-30, 30-50, "
        "50-90. For each bin, country-FE residualisation of log GDP on edu, "
        "then country-FE regression of outcome on the residual. "
        "Within-meta beta = inverse-variance-weighted mean of bin-specific "
        "betas. Between-bin beta = regression of bin-mean outcomes on "
        "bin-mean residGDP, bin-size-weighted. Tests whether the pooled "
        "residGDP signal lives within-bin (real causal channel) or "
        "between-bin (education-baseline composition)."
    ),
    "bins": [{"lo": lo, "hi": hi} for lo, hi in BINS],
    "pool": {"lo": POOL[0], "hi": POOL[1]},
    "panel": {"n": int(len(panel)), "n_countries": int(panel["country"].nunique())},
    "results": results,
}
with open(out_path, "w") as f:
    json.dump(checkin, f, indent=2, default=lambda o: None
              if isinstance(o, float) and np.isnan(o) else o)
print(f"\nWrote {out_path}")
