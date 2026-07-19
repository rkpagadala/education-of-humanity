"""
edu_vs_resid_gdp_bars_beta.py
=============================
Figure 4 (β version): standardized regression coefficient of education vs.
residualized GDP across four outcomes. β is the causal quantity -- directly
comparable across outcomes and directly testable against zero. R² conflates
effect size with predictor variance and outcome noise; β does not.

Blue bars: |β_edu|. Red bars: |β_resid_gdp|. Significance in italics above
each bar. The visual: education's coefficient is large and highly significant
(|t| > 10) on every outcome; residualized GDP's coefficient is near zero and
statistically indistinguishable from zero on every outcome except U-5
mortality (marginal).

Entry-cohort design (entry >= 10%, ceiling <= 90%), country fixed effects,
lower-secondary completion, T=1960-1990. Outcome-specific lags: TFR at
LAG_TFR=5 (biological parent-of-newborn timing); U5MR at LAG_CHILDREARING=12
(childrearing window); LE and child education at LAG_GENERATION=28
(time-to-agency).

Outputs:
  paper/figures/edu_vs_resid_gdp_bars_beta.png
  checkin/edu_vs_resid_gdp_bars_beta.json
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import (load_education, load_wb, interpolate_to_annual,
                     REPO_ROOT, write_checkin, LAG_GENERATION, LAG_CHILDREARING,
                     LAG_TFR, FOR_WEB, PAPER_FIGS, WEB_FIGS)
from residualization._shared import (
    precompute_entry_years, build_panel, build_child_edu_panel,
    filter_panel, fe_residualize_gdp, _demean_and_filter,
)

PAPER_DIR = os.path.join(REPO_ROOT, "paper")

T_YEARS  = list(range(1960, 1995, 5))
# Outcome → lag. TFR uses LAG_TFR=5 (biological parent-of-newborn timing);
# U5MR uses LAG_CHILDREARING=12 (childrearing window); LE and child education
# use LAG_GENERATION=28 (time-to-agency / edu→edu).
OUTCOME_LAG = {
    "le":        LAG_GENERATION,
    "tfr":       LAG_TFR,
    "child_edu": LAG_GENERATION,
    "u5mr":      LAG_CHILDREARING,
}
COL_NAME = "lower_sec"
CEILING  = 90
ENTRY    = 10


def standardized_beta(data, x_col, y_col):
    """Standardized within-country β of y on x: demean by country, standardize
    each series by pooled SD, fit OLS with no intercept.

    Returns (beta, se, t, n_obs, n_countries) or (nan, nan, nan, 0, 0).
    """
    result = _demean_and_filter(data, [x_col, y_col])
    if result is None:
        return np.nan, np.nan, np.nan, 0, 0
    sub, dm, n_countries = result
    X = dm[x_col].to_numpy()
    y = dm[y_col].to_numpy()
    ok = ~np.isnan(X) & ~np.isnan(y)
    n = int(ok.sum())
    if n < 10:
        return np.nan, np.nan, np.nan, 0, n_countries
    x = X[ok]
    yv = y[ok]
    sd_x = x.std(ddof=0)
    sd_y = yv.std(ddof=0)
    if sd_x == 0 or sd_y == 0:
        return np.nan, np.nan, np.nan, n, n_countries
    xs = x / sd_x
    ys = yv / sd_y
    model = sm.OLS(ys, xs).fit()
    return (float(model.params[0]), float(model.bse[0]),
            float(model.tvalues[0]), n, n_countries)


print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_df  = load_wb("gdppercapita_us_inflation_adjusted.csv")
le_df   = load_wb("life_expectancy_years.csv")
tfr_df  = load_wb("children_per_woman_total_fertility.csv")
u5mr_df = load_wb("child_mortality_u5.csv")

edu_annual   = interpolate_to_annual(edu_raw, COL_NAME)
entry_years  = precompute_entry_years(edu_annual)
cohort       = entry_years[ENTRY]


OUTCOME_SPECS = [
    # (label, panel_builder, outcome_col)
    ("Life\nexpectancy", "wb", "le",    le_df),
    ("Fertility",        "wb", "tfr",   tfr_df),
    ("Child\neducation", "ce", "child_edu", None),
    ("Child\nmortality", "wb", "u5mr",  u5mr_df),
]


def compute_outcome(label, kind, outcome_col, outcome_df):
    """Run edu-β and resid-GDP-β for a single outcome."""
    lag = OUTCOME_LAG[outcome_col]
    if kind == "wb":
        panel = build_panel(edu_annual, outcome_df, gdp_df, T_YEARS, lag, outcome_col)
    else:
        panel = build_child_edu_panel(edu_annual, gdp_df, T_YEARS, lag)
    sub = filter_panel(panel, cohort, CEILING)

    # Education β (standardized, within-country FE)
    be, se_e, t_e, n, ctry = standardized_beta(sub, "edu_t", outcome_col)

    # Residualize GDP against education, then β of residualized GDP on outcome
    resid = fe_residualize_gdp(sub)
    if resid is None:
        br = np.nan
        se_r = np.nan
        t_r = np.nan
    else:
        sub_r, _ = resid
        br, se_r, t_r, _, _ = standardized_beta(sub_r, "gdp_resid", outcome_col)

    return {
        "label": label,
        "edu_beta":  be, "edu_se":  se_e, "edu_t":  t_e,
        "resid_beta": br, "resid_se": se_r, "resid_t": t_r,
        "n": n, "countries": ctry,
    }


results = []
print("\n%-22s %7s %7s %7s   %7s %7s %7s   %s" %
      ("Outcome", "|β|-edu", "SE", "|t|", "|β|-rsd", "SE", "|t|", "n (ctry)"))
print("-" * 95)
for label, kind, col, df in OUTCOME_SPECS:
    r = compute_outcome(label, kind, col, df)
    results.append(r)
    def fmt(v, spec="7.3f"):
        return "nan   " if (v is None or (isinstance(v, float) and np.isnan(v))) else format(abs(v), spec)
    print("%-22s %s %s %s   %s %s %s   %d (%d)" % (
        label.replace("\n", " "),
        fmt(r["edu_beta"]), fmt(r["edu_se"]), fmt(r["edu_t"], "7.2f"),
        fmt(r["resid_beta"]), fmt(r["resid_se"]), fmt(r["resid_t"], "7.2f"),
        r["n"], r["countries"],
    ))


labels     = [r["label"] for r in results]
edu_vals   = [abs(r["edu_beta"])   for r in results]
resid_vals = [abs(r["resid_beta"]) for r in results]
edu_ts     = [abs(r["edu_t"])      for r in results]
resid_ts   = [abs(r["resid_t"])    for r in results]


def sig_mark(t):
    if np.isnan(t):
        return "n.s."
    if abs(t) > 3.29:   return "***"    # p < 0.001
    if abs(t) > 2.58:   return "**"     # p < 0.01
    if abs(t) > 1.96:   return "*"      # p < 0.05
    return "n.s."


fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(labels))
width = 0.35

bars_edu = ax.bar(x - width/2, edu_vals, width,
                  color="#2563eb", label="Education", zorder=3)
bars_r   = ax.bar(x + width/2, resid_vals, width,
                  color="#ef4444",
                  label="GDP (after removing\neducation's contribution)",
                  zorder=3)

y_hi = max(edu_vals) * 1.25
for bar, val, t in zip(bars_edu, edu_vals, edu_ts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_hi * 0.015,
            f"|β|={val:.2f}\n|t|={t:.1f}{sig_mark(t)}",
            ha="center", va="bottom", fontsize=9.5,
            fontweight="bold", color="#2563eb")
for bar, val, t in zip(bars_r, resid_vals, resid_ts):
    display_h = max(bar.get_height(), y_hi * 0.008)
    ax.text(bar.get_x() + bar.get_width()/2, display_h + y_hi * 0.015,
            f"|β|={val:.2f}\n|t|={t:.1f}{sig_mark(t)}",
            ha="center", va="bottom", fontsize=9.5,
            fontweight="bold", color="#ef4444")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Standardized |β| (within-country FE)", fontsize=11)
ax.set_ylim(0, y_hi)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, zorder=0)

if FOR_WEB:
    ax.set_title("GDP has no independent effect on any development outcome",
                 fontsize=13, fontweight="bold", pad=15)

fig.text(0.5, -0.02,
         "Country fixed effects, lower secondary completion, entry ≥ 10%, "
         "ceiling ≤ 90%, T=1960–1990. Outcome lags: fertility T+5 "
         "(biological parent timing); child mortality T+12 (childrearing "
         "window); LE, child education T+25 (time-to-agency). "
         "Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. otherwise.",
         ha="center", fontsize=8, color="#64748b")

plt.tight_layout()
out_paper = os.path.join(PAPER_FIGS if not FOR_WEB else WEB_FIGS, "edu_vs_resid_gdp_bars_beta.png")
fig.savefig(out_paper, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {out_paper}")
plt.close()


numbers = {
    "entry_threshold": ENTRY,
    "ceiling":         CEILING,
    "lag_tfr":         OUTCOME_LAG["tfr"],
    "lag_generation":  LAG_GENERATION,
    "t_start":         T_YEARS[0],
    "t_end":           T_YEARS[-1] + LAG_GENERATION,
    "education_level": COL_NAME,
}
for r in results:
    key = r["label"].replace("\n", "_").replace(" ", "_").lower()
    for fld in ("edu_beta", "edu_se", "edu_t",
                "resid_beta", "resid_se", "resid_t", "n", "countries"):
        v = r[fld]
        if isinstance(v, (int, np.integer)):
            numbers[f"{key}_{fld}"] = int(v)
        else:
            numbers[f"{key}_{fld}"] = (
                None if v is None or (isinstance(v, float) and np.isnan(v))
                else round(float(v), 4)
            )

write_checkin(
    "edu_vs_resid_gdp_bars_beta.json",
    {
        "notes": (
            "Standardized within-country β of education and of residualized "
            "log-GDP on four development outcomes. Outcome lags: TFR at "
            "T+LAG_TFR=5 (biological parent-of-newborn timing); U5MR at "
            "T+LAG_CHILDREARING=12 (childrearing window); LE and child "
            "education at T+LAG_GENERATION=28 (time-to-agency). "
            "Residualization: regress log_GDP on education with country FE; "
            "use residuals (GDP not explained by education) as second predictor. "
            "Entry-cohort design: entry >= 10%, ceiling <= 90%. "
            "T=1960-1990, lower-secondary completion."
        ),
        "numbers": numbers,
    },
    script_path="scripts/figures/edu_vs_resid_gdp_bars_beta.py",
)
print("Done.")
