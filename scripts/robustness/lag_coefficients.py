"""
lag_coefficients.py

β version of outcomes_r2_by_lag.py. At each lag L in {0, 5, ..., 100}
reports the standardized within-country regression coefficient (beta) of
lower-secondary completion on each of four development outcomes, along
with its standard error and t-statistic. Also sweeps log-GDP as a
predictor across the same lags (raw WDI, not backfilled; the curve
naturally terminates where WDI data runs out). Produces a figure
paralleling outcomes_r2_by_lag.png but in |β| instead of R².

Motivation:
  R^2 conflates effect size with predictor variance and outcome noise and
  is not directly comparable across outcomes. The standardized beta -- the
  expected outcome change (in outcome SDs) for a one-SD change in the
  predictor -- is the causal quantity and is comparable across outcomes.
  The t-statistic is the clean test of whether a signal is present.

Method (mirrors backfill_all_outcomes.py for data alignment):
  - Education: WCDE v3 lower-secondary completion, 1875-2015 (cohort, both
    sexes), interpolated to annual.
  - Outcomes: life expectancy (WDI), total fertility (WDI), under-5
    mortality logged (WDI), child education (WCDE, autoregression of the
    predictor).
  - Country fixed effects: demean every series by country mean.
  - Standardize both predictor and outcome by pooled within-country SD
    (SD of the demeaned series), then regress outcome on predictor with
    no intercept (FE absorbed by demean; standardized so intercept = 0).
  - Panel: 142 common countries (intersection of edu / GDP / LE / TFR / U-5),
    outcome years 1960-2015. Same as backfill_all_outcomes.py.

Outputs:
  checkin/lag_coefficients.json
    <outcome>_beta_lag<L>        education standardized coefficient
    <outcome>_se_lag<L>          education standard error
    <outcome>_t_lag<L>           education t-statistic
    <outcome>_n_lag<L>           observations
    gdp_<outcome>_beta_lag<L>    log-GDP standardized coefficient (raw WDI)
    gdp_<outcome>_se_lag<L>      log-GDP standard error
    gdp_<outcome>_t_lag<L>       log-GDP t-statistic
  paper/figures/outcomes_beta_by_lag.png
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import PROC as WCDE_PROC, DATA as WB_DIR, REPO_ROOT, write_checkin

FIG_PATH      = os.path.join(REPO_ROOT, "paper", "figures", "outcomes_beta_by_lag.png")
FIG_PATH_4A   = os.path.join(REPO_ROOT, "paper", "figures", "outcomes_beta_by_lag_4a.png")
FIG_PATH_4B   = os.path.join(REPO_ROOT, "paper", "figures", "outcomes_beta_by_lag_4b.png")
GDP_LAG_MAX = 45  # cap for raw (non-backfilled) log-GDP predictor; reflects
                  # WDI coverage (data start 1960), feasible up to lag ~45
                  # given outcome panel 1960-2015 and MIN_OBS / MIN_OBS_PER_C.
LAG_4A_MAX  = 45  # x-axis cap for Figure 4A (the GDP-availability window)

LAG_MIN     = 0
LAG_MAX     = 100
LAG_STEP    = 5
PANEL_START = 1960
PANEL_END   = 2015

MIN_OBS       = 500
MIN_OBS_PER_C = 3


def load_wide(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.str.lower().str.strip()
    df.columns = df.columns.astype(int)
    return df.clip(lower=0)


_edu_raw = pd.read_csv(os.path.join(WCDE_PROC, "cohort_lower_sec_both.csv"),
                       index_col="country")
_edu_raw.columns = [int(c) for c in _edu_raw.columns]
_all_yrs = list(range(min(_edu_raw.columns), max(_edu_raw.columns) + 1))
_edu_raw = _edu_raw.reindex(columns=_all_yrs).interpolate(axis=1) \
                    .bfill(axis=1).ffill(axis=1)
_edu_raw.index = _edu_raw.index.str.lower().str.strip()
edu = _edu_raw.clip(lower=0)

gdp_orig = load_wide(os.path.join(WB_DIR, "gdppercapita_us_inflation_adjusted.csv"))
le_df    = load_wide(os.path.join(WB_DIR, "life_expectancy_years.csv"))
tfr_df   = load_wide(os.path.join(WB_DIR, "children_per_woman_total_fertility.csv"))
u5_df    = load_wide(os.path.join(WB_DIR, "child_mortality_u5.csv"))

countries = sorted(
    set(edu.index)
    & set(gdp_orig.index)
    & set(le_df.index)
    & set(tfr_df.index)
    & set(u5_df.index)
)
print(f"N countries: {len(countries)}")


def _log_nonpos_to_nan(mat):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mat > 0, np.log(mat), np.nan)


def _mat(df, cols):
    return df.reindex(index=countries, columns=cols).to_numpy(dtype=float, na_value=np.nan)


def standardized_beta(pred, out, country_labels):
    valid = ~np.isnan(pred) & ~np.isnan(out)
    if valid.sum() < MIN_OBS:
        return np.nan, np.nan, np.nan, 0
    df = pd.DataFrame({"country": country_labels[valid],
                       "p": pred[valid], "o": out[valid]})
    cnt = df.groupby("country")["o"].transform("count")
    df = df[cnt >= MIN_OBS_PER_C]
    if len(df) < MIN_OBS:
        return np.nan, np.nan, np.nan, 0
    df["p"] = df["p"] - df.groupby("country")["p"].transform("mean")
    df["o"] = df["o"] - df.groupby("country")["o"].transform("mean")
    sd_p = df["p"].std(ddof=0)
    sd_o = df["o"].std(ddof=0)
    if sd_p == 0 or sd_o == 0:
        return np.nan, np.nan, np.nan, int(len(df))
    df["p"] = df["p"] / sd_p
    df["o"] = df["o"] / sd_o
    model = sm.OLS(df["o"], df[["p"]]).fit()
    beta = float(model.params["p"])
    se   = float(model.bse["p"])
    t    = float(model.tvalues["p"])
    return beta, se, t, int(len(df))


OUTCOMES = [
    ("le",    "Life expectancy",          le_df,  False, False),
    ("tfr",   "Total fertility rate",     tfr_df, False, False),
    ("u5log", "Under-5 mortality (log)",  u5_df,  True,  False),
    ("cedu",  "Child education",          None,   False, True),
]

lags = list(range(LAG_MIN, LAG_MAX + 1, LAG_STEP))

_outcome_yrs = list(range(PANEL_START, PANEL_END + 1))
_country_labels_flat = np.repeat(np.array(countries, dtype=object), len(_outcome_yrs))

results = {key: {"beta": [], "se": [], "t": [], "n": []} for key, *_ in OUTCOMES}

print(f"\nSweeping {len(lags)} lags x {len(OUTCOMES)} outcomes...\n")

for key, label, outcome_df, logit, use_edu in OUTCOMES:
    print(f"\n### {label}")
    out_src = edu if use_edu else outcome_df
    out_mat = _mat(out_src, _outcome_yrs)
    if logit:
        out_mat = _log_nonpos_to_nan(out_mat)
    out_flat = out_mat.ravel()

    for lag in lags:
        if use_edu and lag == 0:
            results[key]["beta"].append(np.nan)
            results[key]["se"].append(np.nan)
            results[key]["t"].append(np.nan)
            results[key]["n"].append(0)
            print(f"  lag={lag:3d}  skipped (identity: predictor == outcome)")
            continue
        pred_cols = [y - lag for y in _outcome_yrs]
        edu_pred = _mat(edu, pred_cols).ravel()
        b, s, t, n = standardized_beta(edu_pred, out_flat, _country_labels_flat)
        results[key]["beta"].append(b)
        results[key]["se"].append(s)
        results[key]["t"].append(t)
        results[key]["n"].append(n)
        b_s = f"{b:>+6.3f}" if not np.isnan(b) else "   nan"
        t_s = f"{t:>+7.2f}" if not np.isnan(t) else "    nan"
        print(f"  lag={lag:3d}  beta={b_s}  se={s:>5.3f}  t={t_s}  n={n}")


# ── GDP sweep (raw WDI log-GDP, no backfill; terminates past lag 25) ─────────
print("\n\n### GDP sweep (raw WDI log-GDP, capped at lag %d)" % GDP_LAG_MAX)
gdp_results = {key: {"beta": [], "se": [], "t": []} for key, *_ in OUTCOMES}
for key, label, outcome_df, logit, use_edu in OUTCOMES:
    out_src = edu if use_edu else outcome_df
    out_mat = _mat(out_src, _outcome_yrs)
    if logit:
        out_mat = _log_nonpos_to_nan(out_mat)
    out_flat = out_mat.ravel()
    for lag in lags:
        if lag > GDP_LAG_MAX:
            gdp_results[key]["beta"].append(np.nan)
            gdp_results[key]["se"].append(np.nan)
            gdp_results[key]["t"].append(np.nan)
            continue
        if use_edu and lag == 0:
            gdp_results[key]["beta"].append(np.nan)
            gdp_results[key]["se"].append(np.nan)
            gdp_results[key]["t"].append(np.nan)
            continue
        pred_cols = [y - lag for y in _outcome_yrs]
        gdp_pred = _log_nonpos_to_nan(_mat(gdp_orig, pred_cols)).ravel()
        b, s, t, _ = standardized_beta(gdp_pred, out_flat, _country_labels_flat)
        gdp_results[key]["beta"].append(b)
        gdp_results[key]["se"].append(s)
        gdp_results[key]["t"].append(t)


# ── Plot: |β| by lag, four outcomes, education vs GDP ─────────────────────────
PLOT_ORDER = [
    # (key, short label, color)
    ("u5log", "Under-5 mortality (log)", "#6a3d9a"),   # purple
    ("cedu",  "Child education",         "#e67e22"),   # orange
    ("tfr",   "Total fertility rate",    "#2ca25f"),   # green
    ("le",    "Life expectancy",         "#1f6feb"),   # blue
]


def _plot_lag(ax, lags_, results_, gdp_results_, *, x_max, show_gdp,
              gen_marks):
    for key, long_label, color in PLOT_ORDER:
        y_edu = [abs(b) if not np.isnan(b) else np.nan for b in results_[key]["beta"]]
        ax.plot(lags_, y_edu, linestyle="-", linewidth=2.0, marker="o", markersize=3.8,
                color=color, label=f"Education -> {long_label}")
        if show_gdp:
            y_gdp = [abs(b) if not np.isnan(b) else np.nan
                     for b in gdp_results_[key]["beta"]]
            ax.plot(lags_, y_gdp, linestyle="--", linewidth=1.6, marker="s", markersize=3.5,
                    color=color, alpha=0.75,
                    label=f"GDP -> {long_label}")
    for lag_mark, text in gen_marks:
        ax.axvline(lag_mark, color="0.7", linestyle=":", linewidth=0.9, zorder=0)
        ax.text(lag_mark, 1.00, text, rotation=90, va="top", ha="right",
                fontsize=8, color="0.35")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Lag (years) between predictor and outcome")
    ax.set_ylabel(r"Standardized $|\beta|$ (within-country FE)")
    ax.grid(True, linestyle=":", linewidth=0.6, color="0.8", zorder=0)
    ax.set_axisbelow(True)
    legend = ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9)
    legend.get_frame().set_edgecolor("0.8")


# Figure 4A — lags 0..GDP_LAG_MAX, both education and GDP on the same scale.
fig_a, ax_a = plt.subplots(figsize=(9, 5.5))
_plot_lag(ax_a, lags, results, gdp_results,
          x_max=LAG_4A_MAX, show_gdp=True,
          gen_marks=[(25, "1 generation")])
ax_a.set_title("Figure 4A: Education vs. log GDP per capita, lags 0-45\n"
               "GDP-availability window (raw WDI 1960+); same axis as education",
               fontsize=11)
plt.tight_layout()
plt.savefig(FIG_PATH_4A, dpi=200, bbox_inches="tight")
print(f"\nSaved: {FIG_PATH_4A}")
plt.close()

# Figure 4B — lags 0..100, education only.
fig_b, ax_b = plt.subplots(figsize=(9, 5.5))
_plot_lag(ax_b, lags, results, gdp_results,
          x_max=100, show_gdp=False,
          gen_marks=[(25, "1 generation"), (50, "2 generations"),
                     (75, "3 generations")])
ax_b.set_title("Figure 4B: Education across 0-100 year lags\n"
               "WCDE 1875-2015; long-run persistence beyond GDP's coverage window",
               fontsize=11)
plt.tight_layout()
plt.savefig(FIG_PATH_4B, dpi=200, bbox_inches="tight")
print(f"Saved: {FIG_PATH_4B}")
plt.close()

# Combined figure (kept for backward compatibility with anything that still
# references outcomes_beta_by_lag.png; the paper now uses 4A and 4B).
fig, ax = plt.subplots(figsize=(9, 5.5))
_plot_lag(ax, lags, results, gdp_results,
          x_max=100, show_gdp=True,
          gen_marks=[(25, "1 generation"), (50, "2 generations"),
                     (75, "3 generations")])
ax.set_title("Education vs. GDP across 0-100 year lags\n"
             "Solid = education (WCDE 1875-2015); dashed = log GDP (WDI 1960+, raw)",
             fontsize=11)
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
print(f"Saved: {FIG_PATH}")
plt.close()


print("\n" + "=" * 90)
print("SUMMARY: standardized beta by outcome, at generation anchors")
print("=" * 90)
print(f"{'Lag':>5}  {'Gen':>4}  "
      f"{'LE beta (t)':>15}  {'TFR beta (t)':>15}  "
      f"{'U-5 beta (t)':>15}  {'CE beta (t)':>15}")
print("-" * 90)
for i, lag in enumerate(lags):
    if lag not in (0, 25, 50, 75, 100):
        continue
    gen = {0: "-", 25: "1", 50: "2", 75: "3", 100: "4"}[lag]
    parts = [f"{lag:>5}", f"{gen:>4}"]
    for key, *_ in OUTCOMES:
        b = results[key]["beta"][i]
        t = results[key]["t"][i]
        parts.append(f"{b:>+6.3f} ({t:>+6.2f})")
    print("  ".join(parts))


numbers = {
    "n_countries": len(countries),
    "lags": lags,
    "panel_start": PANEL_START,
    "panel_end": PANEL_END,
}
for key, *_ in OUTCOMES:
    for i, lag in enumerate(lags):
        for tag in ("beta", "se", "t", "n"):
            v = results[key][tag][i]
            if isinstance(v, (int, np.integer)):
                numbers[f"{key}_{tag}_lag{lag}"] = int(v)
            else:
                numbers[f"{key}_{tag}_lag{lag}"] = (
                    None if v is None or (isinstance(v, float) and np.isnan(v))
                    else round(float(v), 4)
                )
# GDP predictor sweep (raw WDI; NaN past lag GDP_LAG_MAX)
for key, *_ in OUTCOMES:
    for i, lag in enumerate(lags):
        for tag in ("beta", "se", "t"):
            v = gdp_results[key][tag][i]
            numbers[f"gdp_{key}_{tag}_lag{lag}"] = (
                None if v is None or (isinstance(v, float) and np.isnan(v))
                else round(float(v), 4)
            )

write_checkin(
    "lag_coefficients.json",
    {
        "notes": (
            f"{len(countries)} countries, outcome years {PANEL_START}-{PANEL_END}. "
            f"Standardized within-country OLS of development outcome on "
            f"lower-secondary completion at T-lag. Each series demeaned by "
            f"country mean and standardized by pooled within-country SD, then "
            f"regressed with no intercept. Panel matches backfill_all_outcomes.py."
        ),
        "numbers": numbers,
    },
    script_path="scripts/robustness/lag_coefficients.py",
)

print("\nDone.")
