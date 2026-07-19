"""
unique_r2_ceiling_sweep.py
==========================

PAPER ANCHOR — Chapter 9 §The one bounded exception (U5MR / Lutz reading)
  Question: walking the entry-cohort ceiling from 20% to 90%, where does
            log GDP retain any unique signal above education?
  Produces: paper claim that GDP's unique R^2 on log U5MR is at most 0.002
            at every ceiling (the residualisation null survives the spec
            sweep), while the levels-spec post-2000 child-mortality signal
            is the Lutz/Kebede 1990-2010 child-health-aid exception
            (donor-funded, not domestic income).
  Checkin:  checkin/unique_r2_ceiling_sweep.json

Walk the education ceiling from 20 to 90 in steps of 10, holding entry=10%,
and report the unique-R^2 decomposition at each stop. The question: does
GDP's unique contribution to each outcome appear at specific education
baselines and vanish elsewhere?

Hypothesis (consistent with Lutz & Kebede 2018's documented 1990-2010
child-mortality exception): GDP/donor-provision picks up incremental signal
in high-mortality, low-education country-years (low ceiling), and vanishes
as countries reach saturation (high ceiling).

For each ceiling C in {20, 30, 40, 50, 60, 70, 80, 90}:
  entry-cohort = countries that crossed 10% lower-sec, child education
                 at T at or below C.
  Run 3 nested FE regressions per outcome (GDP alone, Edu alone, Both)
  on the Maddison-backfilled GDP sample.
  Report R^2 decomposition + joint-regression GDP beta/p.

Outcomes (Table 13 convention): log LE, log TFR, log U5MR, child education
(levels - already a percentage).

Output: checkin/unique_r2_ceiling_sweep.json
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
CEILINGS = [20, 30, 40, 50, 60, 70, 80, 90]
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
    mpd = pd.read_csv(MADDISON_CSV)
    mpd["country_canon"] = mpd["country_canon"].astype(str).str.lower()
    wide = mpd.pivot_table(index="country_canon", columns="year",
                            values="gdppc_wb_eq", aggfunc="first")
    wide.columns = [str(int(c)) for c in wide.columns]
    wide.index.name = "Country"
    return wide


def merge_wb_maddison(wb, maddison):
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
        for wb_idx, canon in zip(filled.index[nan_mask], canon_for_nan):
            if canon in mpd_canon.index:
                v = mpd_canon.at[canon, col]
                if pd.notna(v):
                    filled.at[wb_idx, col] = v
    return filled


def fe_joint(x1_col, x2_col, y_col, data):
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
    return dict(r2=float(r2),
                beta=[float(beta[0]), float(beta[1])],
                se=[float(se[0]), float(se[1])],
                p=[float(pvals[0]), float(pvals[1])],
                n=n, n_countries=int(n_countries))


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
gdp_merged = merge_wb_maddison(gdp_wb, load_maddison_wide())

edu_annual = interpolate_to_annual(edu_raw, EDU_COL)
entry_years = precompute_entry_years(edu_annual)
cohort = entry_years[ENTRY]

panel_le   = build_panel(edu_annual, le_raw,   gdp_merged, T_YEARS,
                         OUTCOME_LAG["le_tp28"],   "le_tp28")
panel_tfr  = build_panel(edu_annual, tfr_raw,  gdp_merged, T_YEARS,
                         OUTCOME_LAG["tfr_tp5"],  "tfr_tp5")
panel_u5mr = build_panel(edu_annual, u5mr_raw, gdp_merged, T_YEARS,
                         OUTCOME_LAG["u5mr_tp12"], "u5mr_tp12")
panel_ced  = build_child_edu_panel(edu_annual, gdp_merged, T_YEARS,
                                    OUTCOME_LAG["ced_tp28"])
panel_ced  = panel_ced.rename(columns={"child_edu": "ced_tp28"})

panel_le["log_le_tp28"]     = np.log(panel_le["le_tp28"].where(panel_le["le_tp28"] > 0))
panel_tfr["log_tfr_tp5"]   = np.log(panel_tfr["tfr_tp5"].where(panel_tfr["tfr_tp5"] > 0))
panel_u5mr["log_u5mr_tp12"] = np.log(panel_u5mr["u5mr_tp12"].where(panel_u5mr["u5mr_tp12"] > 0))

outcomes = [
    ("life_expectancy_log", panel_le,   "log_le_tp28"),
    ("tfr_log",             panel_tfr,  "log_tfr_tp5"),
    ("u5mr_log",            panel_u5mr, "log_u5mr_tp12"),
    ("child_education",     panel_ced,  "ced_tp28"),
]


sweep = {}
print("\nUnique-R^2 decomposition: ceiling sweep at entry=10%, Maddison-backfilled GDP")
print("=" * 110)
for name, base_panel, ycol in outcomes:
    print(f"\n{name.upper()}")
    print(f"{'Ceil':<5} {'Edu R^2':>8} {'GDP R^2':>8} {'Both':>8} "
          f"{'Edu uniq':>9} {'GDP uniq':>9} {'Ratio':>9} "
          f"{'GDP β':>9} {'GDP SE':>8} {'GDP p':>8} {'n':>5} {'cn':>4}")
    print("-" * 110)
    rows = {}
    for c in CEILINGS:
        sub = filter_panel(base_panel, cohort, c)
        res = decompose(sub, "edu_t", "log_gdp_t", ycol)
        if res is None:
            continue
        rows[str(c)] = res
        r_str = f"{res['ratio']:.1f}" if res['ratio'] is not None else "inf"
        print(f"{c:<5} {res['edu_alone_r2']:>8.3f} {res['gdp_alone_r2']:>8.3f} "
              f"{res['both_r2']:>8.3f} {res['edu_unique_r2']:>9.3f} "
              f"{res['gdp_unique_r2']:>9.3f} {r_str:>9} "
              f"{res['gdp_beta_joint']:>9.3f} {res['gdp_se_joint']:>8.3f} "
              f"{res['gdp_p_joint']:>8.3f} {res['n']:>5d} {res['n_countries']:>4d}")
    sweep[name] = rows


write_checkin("unique_r2_ceiling_sweep.json", {
    "method": (
        "Unique-R^2 decomposition swept across education ceilings 20-90 in "
        "steps of 10, entry threshold fixed at 10%. For each ceiling, run "
        "three within-country FE regressions per outcome (GDP alone, Edu "
        "alone, Both) on the Maddison-backfilled GDP sample, report R^2 "
        "decomposition and joint-regression GDP coefficient/SE/p. Tests "
        "whether GDP's unique contribution concentrates at low ceilings "
        "(high-mortality / low-education countries — the Lutz & Kebede "
        "1990-2010 child-mortality exception) and vanishes at high ceilings. "
        f"Outcome-specific lags: TFR at T+LAG_TFR={LAG_TFR} (biological "
        f"parent-of-newborn timing); U5MR at T+LAG_CHILDREARING={LAG_CHILDREARING} "
        f"(childrearing window); LE and child education at "
        f"T+LAG_GENERATION={LAG_GENERATION} (time-to-agency / edu→edu)."
    ),
    "parameters": {
        "T_years": T_YEARS,
        "outcome_lags": OUTCOME_LAG,
        "entry_threshold": ENTRY,
        "ceilings_swept": CEILINGS,
        "edu_measure": EDU_COL,
        "gdp_source": "WB primary, Maddison 2020 backfill",
    },
    "sweep": sweep,
}, script_path="scripts/residualization/unique_r2_ceiling_sweep.py")

print("\nDone.")
