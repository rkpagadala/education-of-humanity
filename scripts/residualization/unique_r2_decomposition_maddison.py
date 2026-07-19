"""
unique_r2_decomposition_maddison.py
====================================

PAPER ANCHOR — Chapter 9 §Education vs GDP at Population Scale (Notes block)
  Question: does the unique-R^2 asymmetry survive when missing World Bank
            GDP cells are backfilled with rescaled Maddison Project (2020) data?
  Produces: paper Notes claim that Maddison backfill expands the GDP
            intersection from n=640/122 to n=746/137, with the asymmetry
            tightening — education's unique R^2 runs 0.25-0.39, GDP's
            runs 0.000-0.007.
  Checkin:  checkin/unique_r2_decomposition_maddison.json

Companion to unique_r2_decomposition.py. Same decomposition, but backfills
missing World Bank GDP cells with the Maddison Project Database 2020 series
already rescaled to WB-equivalent constant 2015 USD (produced by
scripts/ml/build_maddison_backfill.py, output at
data/external/maddison_gdppc_wb_equivalent.csv).

The motivation: the GDP intersection in the WB-only sample is N=640 / 122
countries — the missing cells are systematically poor countries in early
years, exactly where the within-country GDP signal would carry the most
information. Maddison backfill expands the panel by reintroducing those
cells, so the unique-R^2 decomposition can be reported on a sample closer
to the full expansion window (N=1069 / 148).

For each outcome (LE, TFR, log U5MR, child education), run three
within-country FE regressions on the entry-cohort sample (entry >= 10%,
ceiling 90%, T=1960-1990). Outcome-specific lags: LE and child education at
T+28, TFR at T+5, log U5MR at T+12:
  GDP alone, Edu alone, Both -> R^2.

Edu unique = R^2(both) - R^2(GDP alone)
GDP unique = R^2(both) - R^2(Edu alone)
Ratio      = Edu unique / GDP unique

Output: checkin/unique_r2_decomposition_maddison.json
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from _shared import (  # noqa: E402
    load_education, load_wb, interpolate_to_annual,
    build_panel, build_child_edu_panel,
    precompute_entry_years, filter_panel,
    fe_r2, _demean_and_filter, write_checkin,
    DATA, LAG_TFR, LAG_CHILDREARING, LAG_GENERATION,
)
from _shared import standardize_country_name  # noqa: E402


T_YEARS = list(range(1960, 1995, 5))
ENTRY = 10
CEILING = 90
EDU_COL = "lower_sec"

# Outcome → lag. TFR uses LAG_TFR=5 (biological parent-of-newborn timing).
# U5MR uses LAG_CHILDREARING=12 (childrearing window). LE and child education
# use LAG_GENERATION=28 (time-to-agency / edu→edu transmission). Column names
# encode each outcome's true lag: tfr_tp5, u5mr_tp12, le_tp28, ced_tp28.
OUTCOME_LAG = {
    "le_tp28":   LAG_GENERATION,
    "tfr_tp5":   LAG_TFR,
    "u5mr_tp12": LAG_CHILDREARING,
    "ced_tp28":  LAG_GENERATION,
}

MADDISON_CSV = os.path.join(DATA, "external", "maddison_gdppc_wb_equivalent.csv")


def load_maddison_wide():
    """Load Maddison rescaled CSV and pivot to country x year wide format
    matching the WB GDP DataFrame layout."""
    mpd = pd.read_csv(MADDISON_CSV)
    mpd["country_canon"] = mpd["country_canon"].astype(str).str.lower()
    wide = (mpd.pivot_table(index="country_canon",
                            columns="year",
                            values="gdppc_wb_eq",
                            aggfunc="first"))
    wide.columns = [str(int(c)) for c in wide.columns]
    wide.index.name = "Country"
    return wide


def merge_wb_maddison(wb, maddison):
    """Backfill missing WB GDP cells with Maddison rescaled values.

    Preserves the WB DataFrame's index structure (including the aliased
    duplicate rows from add_canonical_aliases — load_wb relies on those for
    name-variant lookups). For each WB row, canonicalize its country name,
    then for each year-column that is NaN in WB, fill from the canonicalized
    Maddison series if a value exists.
    """
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
        canon_for_nan = [standardize_country_name(c) for c in filled.index[nan_mask]]
        # Look up each canonicalized name in Maddison; only fill where match exists
        for wb_idx, canon in zip(filled.index[nan_mask], canon_for_nan):
            if canon in mpd_canon.index:
                v = mpd_canon.at[canon, col]
                if pd.notna(v):
                    filled.at[wb_idx, col] = v
    return filled


def fe_joint(x1_col, x2_col, y_col, data):
    """Two-regressor country-FE regression with country-clustered SEs.
    Returns dict with R^2, beta/SE/p per regressor, n, n_countries.
    """
    result = _demean_and_filter(data, [x1_col, x2_col, y_col])
    if result is None:
        return None
    sub, dm, n_countries = result
    X1 = dm[x1_col].to_numpy()
    X2 = dm[x2_col].to_numpy()
    y  = dm[y_col].to_numpy()
    ok = ~np.isnan(X1) & ~np.isnan(X2) & ~np.isnan(y)
    n = int(ok.sum())
    if n < 10:
        return None
    X = np.column_stack([X1[ok], X2[ok]])
    yv = y[ok]
    countries = sub.iloc[ok]["country"].to_numpy()
    codes, _ = pd.factorize(countries, sort=False)

    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ yv)
    yhat = X @ beta
    resid = yv - yhat
    sse = float(np.dot(resid, resid))
    sst = float(np.dot(yv, yv))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    n_g = int(codes.max()) + 1
    Xe = X * resid[:, None]
    sums_g = np.zeros((n_g, X.shape[1]))
    for j in range(X.shape[1]):
        sums_g[:, j] = np.bincount(codes, weights=Xe[:, j], minlength=n_g)
    meat = sums_g.T @ sums_g
    V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    z = np.where(se > 0, beta / se, np.nan)
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))

    return dict(
        r2=float(r2),
        beta=[float(beta[0]), float(beta[1])],
        se=[float(se[0]), float(se[1])],
        p=[float(pvals[0]), float(pvals[1])],
        n=n,
        n_countries=int(n_countries),
    )


def decompose(panel, edu_col, gdp_col, y_col):
    sub = panel.dropna(subset=[edu_col, gdp_col, y_col]).copy()
    edu_r2, _, _ = fe_r2(edu_col, y_col, sub)
    gdp_r2, _, _ = fe_r2(gdp_col, y_col, sub)
    joint = fe_joint(edu_col, gdp_col, y_col, sub)
    if joint is None:
        return None
    both_r2 = joint["r2"]
    edu_unique = both_r2 - gdp_r2
    gdp_unique = both_r2 - edu_r2
    ratio = edu_unique / gdp_unique if gdp_unique > 0 else float("inf")
    return dict(
        edu_alone_r2=round(edu_r2, 4),
        gdp_alone_r2=round(gdp_r2, 4),
        both_r2=round(both_r2, 4),
        edu_unique_r2=round(edu_unique, 4),
        gdp_unique_r2=round(gdp_unique, 4),
        ratio=round(ratio, 2) if np.isfinite(ratio) else None,
        edu_beta_joint=round(joint["beta"][0], 4),
        edu_se_joint=round(joint["se"][0], 4),
        edu_p_joint=round(joint["p"][0], 4),
        gdp_beta_joint=round(joint["beta"][1], 4),
        gdp_se_joint=round(joint["se"][1], 4),
        gdp_p_joint=round(joint["p"][1], 4),
        n=int(joint["n"]),
        n_countries=int(joint["n_countries"]),
    )


print("Loading data...")
edu_raw  = load_education("completion_both_long.csv")
le_raw   = load_wb("life_expectancy_years.csv")
tfr_raw  = load_wb("children_per_woman_total_fertility.csv")
u5mr_raw = load_wb("child_mortality_u5.csv")
gdp_wb   = load_wb("gdppercapita_us_inflation_adjusted.csv")

maddison = load_maddison_wide()
print(f"  Maddison: {len(maddison)} countries x {len(maddison.columns)} years "
      f"(years {min(int(c) for c in maddison.columns)}-{max(int(c) for c in maddison.columns)})")

# Count WB-only nonnulls
wb_cells_in_window = sum(int(gdp_wb[str(y)].notna().sum())
                          for y in T_YEARS if str(y) in gdp_wb.columns)
print(f"  WB GDP nonnull cells in T_YEARS window: {wb_cells_in_window}")

gdp_merged = merge_wb_maddison(gdp_wb, maddison)
mpd_cells_in_window = sum(int(gdp_merged[str(y)].notna().sum())
                          for y in T_YEARS if str(y) in gdp_merged.columns)
print(f"  WB+Maddison merged nonnull cells in T_YEARS window: {mpd_cells_in_window} "
      f"(+{mpd_cells_in_window - wb_cells_in_window} backfilled)")

edu_annual = interpolate_to_annual(edu_raw, EDU_COL)

panel_le   = build_panel(edu_annual, le_raw,   gdp_merged, T_YEARS,
                         OUTCOME_LAG["le_tp28"],   "le_tp28")
panel_tfr  = build_panel(edu_annual, tfr_raw,  gdp_merged, T_YEARS,
                         OUTCOME_LAG["tfr_tp5"],  "tfr_tp5")
panel_u5mr = build_panel(edu_annual, u5mr_raw, gdp_merged, T_YEARS,
                         OUTCOME_LAG["u5mr_tp12"], "u5mr_tp12")
panel_ced  = build_child_edu_panel(edu_annual, gdp_merged, T_YEARS,
                                    OUTCOME_LAG["ced_tp28"])
panel_ced  = panel_ced.rename(columns={"child_edu": "ced_tp28"})

# Log-transform LE, TFR, U5MR to match Table 13 convention.
panel_le["log_le_tp28"]     = np.log(panel_le["le_tp28"].where(panel_le["le_tp28"] > 0))
panel_tfr["log_tfr_tp5"]   = np.log(panel_tfr["tfr_tp5"].where(panel_tfr["tfr_tp5"] > 0))
panel_u5mr["log_u5mr_tp12"] = np.log(panel_u5mr["u5mr_tp12"].where(panel_u5mr["u5mr_tp12"] > 0))

entry_years = precompute_entry_years(edu_annual)
cohort = entry_years[ENTRY]

panel_le_f   = filter_panel(panel_le,   cohort, CEILING)
panel_tfr_f  = filter_panel(panel_tfr,  cohort, CEILING)
panel_u5mr_f = filter_panel(panel_u5mr, cohort, CEILING)
panel_ced_f  = filter_panel(panel_ced,  cohort, CEILING)

outcomes = [
    ("life_expectancy_log", panel_le_f,   "log_le_tp28"),
    ("tfr_log",             panel_tfr_f,  "log_tfr_tp5"),
    ("u5mr_log",            panel_u5mr_f, "log_u5mr_tp12"),
    ("child_education",     panel_ced_f,  "ced_tp28"),
]

per_outcome = {}
print("\nPer-outcome unique-R^2 decomposition (entry-cohort, WB+Maddison merged GDP):")
print(f"{'Outcome':<18} {'Edu R^2':>8} {'GDP R^2':>8} {'Both R^2':>9} "
      f"{'Edu uniq':>9} {'GDP uniq':>9} {'Ratio':>7} "
      f"{'GDP β':>8} {'GDP p':>8} {'n':>5} {'cn':>4}")
for name, panel, ycol in outcomes:
    res = decompose(panel, "edu_t", "log_gdp_t", ycol)
    per_outcome[name] = res
    r_str = f"{res['ratio']:.1f}" if res['ratio'] is not None else "inf"
    print(f"{name:<18} {res['edu_alone_r2']:>8.3f} {res['gdp_alone_r2']:>8.3f} "
          f"{res['both_r2']:>9.3f} {res['edu_unique_r2']:>9.3f} "
          f"{res['gdp_unique_r2']:>9.3f} {r_str:>7} "
          f"{res['gdp_beta_joint']:>8.3f} {res['gdp_p_joint']:>8.3f} "
          f"{res['n']:>5d} {res['n_countries']:>4d}")

# U5MR pre/post-2000 split
print("\nU5MR pre/post-2000 unique-R^2 split (outcome year T+12):")
print(f"{'Window':<18} {'Edu R^2':>8} {'GDP R^2':>8} {'Both R^2':>9} "
      f"{'Edu uniq':>9} {'GDP uniq':>9} {'Ratio':>7} "
      f"{'GDP β':>8} {'GDP p':>8} {'n':>5} {'cn':>4}")
u5mr_split = {}
u5mr_panel = panel_u5mr_f.copy()
u5mr_panel["outcome_year"] = u5mr_panel["t"] + OUTCOME_LAG["u5mr_tp12"]
windows = [
    ("pre_2000",  u5mr_panel[u5mr_panel["outcome_year"] <  2000]),
    ("post_2000", u5mr_panel[u5mr_panel["outcome_year"] >= 2000]),
]
for label, p in windows:
    if len(p) < 10:
        continue
    res = decompose(p, "edu_t", "log_gdp_t", "log_u5mr_tp12")
    if res is None:
        continue
    u5mr_split[label] = res
    r_str = f"{res['ratio']:.1f}" if res['ratio'] is not None else "inf"
    print(f"{label:<18} {res['edu_alone_r2']:>8.3f} {res['gdp_alone_r2']:>8.3f} "
          f"{res['both_r2']:>9.3f} {res['edu_unique_r2']:>9.3f} "
          f"{res['gdp_unique_r2']:>9.3f} {r_str:>7} "
          f"{res['gdp_beta_joint']:>8.3f} {res['gdp_p_joint']:>8.3f} "
          f"{res['n']:>5d} {res['n_countries']:>4d}")

key = ["country", "t"]
merged = (
    panel_le[key + ["edu_t", "log_gdp_t", "le_tp28", "log_le_tp28"]]
    .merge(panel_tfr[key + ["tfr_tp5", "log_tfr_tp5"]], on=key, how="inner")
    .merge(panel_u5mr[key + ["u5mr_tp12", "log_u5mr_tp12"]], on=key, how="inner")
    .merge(panel_ced[key + ["ced_tp28"]], on=key, how="inner")
)
common = filter_panel(merged, cohort, CEILING)
common = common.dropna(subset=["edu_t", "log_gdp_t", "log_le_tp28",
                                "log_tfr_tp5", "log_u5mr_tp12", "ced_tp28"])
print(f"\nCommon-sample decomposition (n={len(common)}, "
      f"countries={common['country'].nunique()}):")
print(f"{'Outcome':<18} {'Edu R^2':>8} {'GDP R^2':>8} {'Both R^2':>9} "
      f"{'Edu uniq':>9} {'GDP uniq':>9} {'Ratio':>7} "
      f"{'GDP β':>8} {'GDP p':>8}")
common_results = {}
for name, _, ycol in outcomes:
    res = decompose(common, "edu_t", "log_gdp_t", ycol)
    if res is None:
        continue
    common_results[name] = res
    r_str = f"{res['ratio']:.1f}" if res['ratio'] is not None else "inf"
    print(f"{name:<18} {res['edu_alone_r2']:>8.3f} {res['gdp_alone_r2']:>8.3f} "
          f"{res['both_r2']:>9.3f} {res['edu_unique_r2']:>9.3f} "
          f"{res['gdp_unique_r2']:>9.3f} {r_str:>7} "
          f"{res['gdp_beta_joint']:>8.3f} {res['gdp_p_joint']:>8.3f}")

write_checkin("unique_r2_decomposition_maddison.json", {
    "method": (
        "Per-outcome unique-R^2 decomposition, with World Bank GDP cells "
        "backfilled by Maddison Project 2020 (rescaled to WB-equivalent "
        "constant 2015 USD via median overlap ratio). Companion to "
        "unique_r2_decomposition.json (WB only). Same regressions, same "
        "entry-cohort filter (entry >= 10%, ceiling 90%, T=1960-1990); only "
        "the GDP source layer differs. Outcome-specific lags: TFR at "
        f"T+LAG_TFR={LAG_TFR} (biological parent-of-newborn timing); U5MR at "
        f"T+LAG_CHILDREARING={LAG_CHILDREARING} (childrearing window); LE and "
        f"child education at T+LAG_GENERATION={LAG_GENERATION} "
        "(time-to-agency / edu→edu transmission)."
    ),
    "parameters": {
        "T_years": T_YEARS,
        "outcome_lags": OUTCOME_LAG,
        "entry_threshold": ENTRY,
        "ceiling": CEILING,
        "edu_measure": EDU_COL,
        "gdp_source": "WB primary, Maddison 2020 backfill for missing WB cells",
    },
    "gdp_coverage": {
        "wb_only_cells_in_t_window": int(wb_cells_in_window),
        "wb_plus_maddison_cells_in_t_window": int(mpd_cells_in_window),
        "backfilled": int(mpd_cells_in_window - wb_cells_in_window),
    },
    "per_outcome": per_outcome,
    "u5mr_pre_post_2000": u5mr_split,
    "common_sample": {
        "n": int(len(common)),
        "n_countries": int(common["country"].nunique()),
        "results": common_results,
    },
}, script_path="scripts/residualization/unique_r2_decomposition_maddison.py")

print("\nDone.")
