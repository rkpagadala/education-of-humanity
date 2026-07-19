"""
robustness/residgdp_strengthened_bin_test_lags.py
===================================================

PAPER ANCHOR — Chapter 9 §Robustness to baseline stratification +
               §The Generational Lag

Question: does the bin-test verdict (no within-bin residGDP signal on
U5MR) hold at all three timescales the paper defines?

  - LAG_CONTEMPORANEOUS = 0    (real-time response)
  - LAG_CHILDREARING    = 12   (cohort's own children — the
                                mechanistic primary for U5MR per the
                                paper, line 2826)
  - LAG_GENERATION      = 25   (cross-generation, paper headline)

Previous bounded-exception work used only lag 25. The Lutz/GAVI
story (schooled mothers using donor vaccines on their own children)
is mechanistically a childrearing-lag story; this test asks whether
the bin-test null at lag 25 holds at lag 12 too.

Method: re-run the strengthened bin test (Maddison-backfilled, 5pp
bins, 1000-rep country-cluster bootstrap on the within-meta β) on
LEVEL U5MR at each of the three lags. Bootstrap reps trimmed from
2000 to 1000 to keep runtime tractable across 3 lags.

If all three lags show null bootstrap CIs → no separable channel at
any timescale. Paper's "no separable channels on U5MR" claim is
fully robust to the timescale question.

If lag 12 shows a signal that lag 25 missed → reconsider; the
bounded-exception was the wrong-spec dismissal.

Output: checkin/residgdp_strengthened_bin_test_lags.json
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
    LAG_CONTEMPORANEOUS,
    LAG_CHILDREARING,
    LAG_GENERATION,
)

T_YEARS = list(range(1960, 1995, 5))
ENTRY = 10
CEILING = 90
N_BOOT = 1000
SEED = 2026

BINS = [(lo, lo + 5) for lo in range(10, 90, 5)]

LAGS = [
    ("contemporaneous", LAG_CONTEMPORANEOUS),
    ("childrearing",    LAG_CHILDREARING),
    ("generation",      LAG_GENERATION),
]


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
    countries = panel["country"].unique()
    n_c = len(countries)
    by_country = {c: panel[panel["country"] == c] for c in countries}
    betas = []
    for rep in range(n_boot):
        sample = rng.choice(countries, size=n_c, replace=True)
        parts = []
        for i, c in enumerate(sample):
            df = by_country[c].copy()
            df["country"] = f"{c}__{i}"
            parts.append(df)
        boot = pd.concat(parts, ignore_index=True)
        meta_res, _ = within_meta(boot, outcome_col, bins)
        if meta_res is None:
            continue
        betas.append(meta_res["beta"])
        if (rep + 1) % 250 == 0:
            print(f"      boot {rep+1}/{n_boot}: latest β = {meta_res['beta']:+.3f}")
    return np.asarray(betas)


# ── Load data once ───────────────────────────────────────────────────

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
u5mr_raw = load_wb("child_mortality_u5.csv")
gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")
maddison = load_maddison_wide()
gdp_back = backfill_with_maddison(gdp_raw, maddison)
edu_annual = interpolate_to_annual(edu_raw, "lower_sec")
entry_years = precompute_entry_years(edu_annual)
cohort = entry_years[ENTRY]

rng = np.random.default_rng(SEED)
all_results = {}

for lag_name, lag in LAGS:
    print("\n" + "#" * 78)
    print(f"# LAG = {lag} ({lag_name})")
    print("#" * 78)

    panel = build_panel(edu_annual, u5mr_raw, gdp_back, T_YEARS, lag, "u5mr_tp")
    panel = filter_panel(panel, cohort, CEILING).copy()
    panel = panel.dropna(subset=["edu_t", "log_gdp_t", "u5mr_tp"])
    n = len(panel)
    n_c = panel["country"].nunique()
    out_yrs = sorted(set(panel["t"] + lag))
    print(f"Panel: n={n}, countries={n_c}, outcome years {out_yrs[0]}-{out_yrs[-1]}")

    meta_res, bin_res = within_meta(panel, "u5mr_tp", BINS)
    if meta_res is None:
        print("  meta failed")
        continue

    print(f"  Within-meta β (IV): {meta_res['beta']:+.4f}  "
          f"SE_iv={meta_res['se_meta_iv']:.4f}  p_iv={meta_res['p_iv']:.4f}  "
          f"({meta_res['n_bins']} bins)")
    print(f"  Bootstrapping ({N_BOOT} reps)...")
    boot = bootstrap_meta(panel, "u5mr_tp", BINS, N_BOOT, rng)

    if len(boot) < N_BOOT // 4:
        boot_summary = None
        print(f"  Bootstrap insufficient — only {len(boot)} successful")
    else:
        ci_lo = float(np.percentile(boot, 2.5))
        ci_hi = float(np.percentile(boot, 97.5))
        median = float(np.median(boot))
        excl = (ci_lo > 0) or (ci_hi < 0)
        print(f"  Bootstrap median: {median:+.3f}")
        print(f"  Bootstrap 95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]  excludes 0: {excl}")
        boot_summary = {
            "median": median, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "excludes_zero_95": bool(excl),
            "n_reps": int(len(boot)),
        }

    all_results[lag_name] = {
        "lag": lag,
        "panel_n": int(n),
        "panel_n_countries": int(n_c),
        "outcome_year_range": [int(out_yrs[0]), int(out_yrs[-1])],
        "meta": meta_res,
        "bins": bin_res,
        "bootstrap": boot_summary,
    }

# ── Summary table ──────────────────────────────────────────────────

print("\n" + "=" * 86)
print("ALL-LAG SUMMARY — within-bin residGDP signal on level U5MR")
print("=" * 86)
print(f"{'Lag':<22} {'n':>5} {'within-meta β':>14} {'IV-p':>7} | "
      f"{'boot 95% CI':>24} {'≠0?':>5}")
for lag_name, r in all_results.items():
    label = f"{lag_name} (lag {r['lag']})"
    m = r["meta"]
    b = r["bootstrap"]
    if b is None:
        boot_str = "---"
        excl = "?"
    else:
        boot_str = f"[{b['ci_lo']:+.2f}, {b['ci_hi']:+.2f}]"
        excl = "YES" if b["excludes_zero_95"] else "no"
    print(f"{label:<22} {r['panel_n']:>5} {m['beta']:>+14.4f} "
          f"{m['p_iv']:>7.3f} | {boot_str:>24} {excl:>5}")

# ── Save ───────────────────────────────────────────────────────────

out_path = os.path.join(SCRIPTS_DIR, "..", "checkin",
                         "residgdp_strengthened_bin_test_lags.json")
checkin = {
    "method": (
        "Strengthened bin test for level-U5MR residGDP residual at all "
        "three timescales the paper defines (§The Generational Lag): "
        "contemporaneous (lag 0), childrearing (lag 12 — midpoint of "
        "10-15), and cross-generation (lag 25). Maddison-backfilled "
        "GDP, 5pp bins on parental lower-secondary completion, "
        "1000-rep country-cluster bootstrap on the within-meta β. The "
        "paper's framework treats lag 12 as the mechanistic primary "
        "for U5MR (cohort's own children) and lag 25 as the "
        "system-building cross-generation step. The Lutz/GAVI story "
        "of donor-channel-mediated mortality is mechanistically a "
        "lag-12 story."
    ),
    "parameters": {
        "T_years": [int(t) for t in T_YEARS],
        "entry": ENTRY, "ceiling": CEILING,
        "lags_tested": [{"name": n, "lag": l} for n, l in LAGS],
        "n_bins": len(BINS), "n_boot": N_BOOT, "seed": SEED,
    },
    "results": all_results,
}
with open(out_path, "w") as f:
    json.dump(checkin, f, indent=2, default=lambda o: None
              if isinstance(o, float) and np.isnan(o) else o)
print(f"\nWrote {out_path}")
