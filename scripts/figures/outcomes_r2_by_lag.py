"""
figures/outcomes_r2_by_lag.py

Generates a 4-panel version of Figure 3 (le_r2_by_lag.py), extending the
lag-decay analysis from life expectancy to all four development outcomes
the paper cares about.

Output:
  paper/figures/outcomes_r2_by_lag.png
  checkin/outcomes_r2_by_lag.json

What it does:
  For each lag L from 0 to 100 years (step 5):
    - Predictor: lower secondary completion at year T-L  (WCDE v3, 1875-2015)
    - Outcome (four panels):
        (a) life expectancy at birth               (WDI, 1960-2015)
        (b) total fertility rate                   (WDI, 1960-2015)
        (c) under-5 mortality (log)                (WDI, 1960-2015)
        (d) lower secondary completion itself      (WCDE v3, autoregression)
    - Country fixed effects: demean each variable by country mean
    - Record within-country R² for each outcome separately.

  Plots each outcome's R² curve vs lag on its own panel, with generational
  horizons (25, 50, 75, 100) marked.

Note on child education panel:
  At lag 0 the predictor and outcome are identical, so R² = 1.0 trivially.
  The curve becomes informative from lag 5 onward, measuring how much of
  today's education is predicted by past generations' education within the
  same country (autoregressive persistence of educational investment).

Verified output format:
  For each lag, prints R² for all four outcomes, plus n and country count.

Key parameters:
  LAG_MIN    = 0    # years
  LAG_MAX    = 100  # years
  LAG_STEP   = 5    # years
  PANEL_START = 1960 # first outcome year (WDI coverage)
  PANEL_END  = 2015 # last outcome year
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import PROC, DATA, REPO_ROOT, write_checkin

WCDE_PROC = PROC
WB_DIR    = DATA
OUT       = os.path.join(REPO_ROOT, "paper", "figures", "outcomes_r2_by_lag.png")

# ── parameters ────────────────────────────────────────────────────────────────
LAG_MIN     = 0
LAG_MAX     = 100
LAG_STEP    = 5
PANEL_START = 1960
PANEL_END   = 2015

MIN_OBS       = 500
MIN_OBS_PER_C = 3

# ── load data ─────────────────────────────────────────────────────────────────
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

le   = load_wide(os.path.join(WB_DIR, "life_expectancy_years.csv"))
tfr  = load_wide(os.path.join(WB_DIR, "children_per_woman_total_fertility.csv"))
u5   = load_wide(os.path.join(WB_DIR, "child_mortality_u5.csv"))

# ── within-country R² ────────────────────────────────────────────────────────
def within_r2(predictor_col, outcome_col, rows):
    if len(rows) < MIN_OBS:
        return np.nan, 0, 0
    p = pd.DataFrame(rows)
    counts = p.groupby("country")[outcome_col].transform("count")
    p = p[counts >= MIN_OBS_PER_C].copy()
    if len(p) < MIN_OBS:
        return np.nan, 0, 0
    n_obs = len(p)
    n_ctry = p["country"].nunique()
    for col in [predictor_col, outcome_col]:
        p[col] = p[col] - p.groupby("country")[col].transform("mean")
    reg = sm.OLS(p[outcome_col], p[[predictor_col]]).fit()
    return reg.rsquared, n_obs, n_ctry


# ── build outcome rows generically ────────────────────────────────────────────
def collect_rows(outcome_df, outcome_name, lag, countries,
                 log_outcome=False, use_edu_as_outcome=False):
    rows = []
    for country in countries:
        for yr in range(PANEL_START, PANEL_END + 1):
            if use_edu_as_outcome:
                out_val = edu.loc[country, yr] if yr in edu.columns else np.nan
            else:
                out_val = outcome_df.loc[country, yr] if yr in outcome_df.columns else np.nan
            if pd.isna(out_val):
                continue
            if log_outcome:
                if out_val <= 0:
                    continue
                out_val = np.log(out_val)
            yr_pred = yr - lag
            if yr_pred not in edu.columns:
                continue
            edu_val = edu.loc[country, yr_pred]
            if pd.isna(edu_val):
                continue
            rows.append({"country": country, "edu": edu_val, outcome_name: out_val})
    return rows


OUTCOMES = [
    # (label, short key, dataframe, log?, use_edu_as_outcome?)
    ("Life expectancy",      "le",    le,  False, False),
    ("Total fertility rate", "tfr",   tfr, False, False),
    ("Under-5 mortality (log)", "u5log", u5,  True,  False),
    ("Child education",      "cedu",  None, False, True),
]

# Common countries (intersection of all outcome datasets plus edu)
countries = sorted(set(edu.index) & set(le.index) & set(tfr.index) & set(u5.index))
print(f"N countries: {len(countries)}")

# ── compute R² at each lag for each outcome ───────────────────────────────────
lags = list(range(LAG_MIN, LAG_MAX + 1, LAG_STEP))
results = {label: [] for (label, _, _, _, _) in OUTCOMES}
ns = {label: [] for (label, _, _, _, _) in OUTCOMES}
ncs = {label: [] for (label, _, _, _, _) in OUTCOMES}

print(f"\nComputing within-country R² at {len(lags)} lag values × {len(OUTCOMES)} outcomes...\n")

for lag in lags:
    line_parts = [f"lag={lag:3d}"]
    for (label, key, df, logit, use_edu) in OUTCOMES:
        rows = collect_rows(df, key, lag, countries,
                            log_outcome=logit, use_edu_as_outcome=use_edu)
        r2, n, nc = within_r2("edu", key, rows)
        results[label].append(r2)
        ns[label].append(n)
        ncs[label].append(nc)
        r2_str = f"{r2:.3f}" if not np.isnan(r2) else "  nan"
        line_parts.append(f"{label[:10]}={r2_str} (n={n},c={nc})")
    print("  " + "  ".join(line_parts))


# ── plot: single-axis, 4 edu curves + 4 matched GDP curves ───────────────────
# Solid = education; dashed = log GDP per capita (raw WDI, no backfill).
# Colors are matched within outcomes: same hue for edu and GDP on that outcome.
# Legend order matches the visual top-to-bottom ordering at the 1-generation
# anchor (lag 25), listing each outcome's edu-then-GDP pair together.
PLOT_ORDER = [
    ("Under-5 mortality (log)",  "u5log", "#6a3d9a"),   # purple
    ("Child education",          "cedu",  "#e67e22"),   # orange
    ("Total fertility rate",     "tfr",   "#2ca25f"),   # green
    ("Life expectancy",          "le",    "#1f6feb"),   # blue
]

fig, ax = plt.subplots(figsize=(9, 5.5))

# Compute GDP -> outcome curves (raw WDI, no backfill; terminates where data ends)
gdp = load_wide(os.path.join(WB_DIR, "gdppercapita_us_inflation_adjusted.csv"))
gdp_results = {key: [] for _, key, _ in PLOT_ORDER}

# For child edu: predictor is GDP, outcome is edu (tests whether income predicts
# future education — the reverse causality direction).
outcome_df_map = {"u5log": (u5, True), "cedu": (edu, False),
                  "tfr": (tfr, False), "le": (le, False)}

for lag in lags:
    for _, key, _ in PLOT_ORDER:
        odf, olog = outcome_df_map[key]
        rows = []
        for country in countries:
            for yr in range(PANEL_START, PANEL_END + 1):
                out_val = odf.loc[country, yr] if yr in odf.columns else np.nan
                if pd.isna(out_val):
                    continue
                if olog:
                    if out_val <= 0:
                        continue
                    out_val = np.log(out_val)
                yr_pred = yr - lag
                if yr_pred not in gdp.columns:
                    continue
                g = gdp.loc[country, yr_pred]
                if pd.isna(g) or g <= 0:
                    continue
                rows.append({"country": country, "log_gdp": np.log(g), "out": out_val})
        r2, _, _ = within_r2("log_gdp", "out", rows)
        gdp_results[key].append(r2)

# Plot: for each outcome, add solid edu curve then dashed GDP curve paired
key_to_label = {key: label for (label, key, _, _, _) in OUTCOMES}
for label_short, key, color in PLOT_ORDER:
    long_label = key_to_label[key]
    y_edu = results[long_label]
    y_gdp = gdp_results[key]
    ax.plot(lags, y_edu, linestyle="-", linewidth=2.0, marker="o", markersize=3.8,
            color=color, label=f"Education -> {label_short}")
    ax.plot(lags, y_gdp, linestyle="--", linewidth=1.6, marker="s", markersize=3.5,
            color=color, alpha=0.75,
            label=f"GDP -> {label_short}")

# Generation anchors
for lag_mark in [25, 50, 75]:
    ax.axvline(lag_mark, color="0.7", linestyle=":", linewidth=0.9, zorder=0)
anchor_labels = {25: "1 generation", 50: "2 generations", 75: "3 generations"}
for lag_mark, text in anchor_labels.items():
    ax.text(lag_mark, 1.00, text, rotation=90, va="top", ha="right",
            fontsize=8, color="0.35")

ax.set_xlim(0, 100)
ax.set_ylim(0.0, 1.02)
ax.set_xlabel("Lag (years) between predictor and outcome")
ax.set_ylabel("Within-country $R^2$ (fixed effects)")
ax.set_title("Education vs. GDP across 0-100 year lags\n"
             "Solid = education (WCDE 1875-2015); dashed = log GDP (WDI 1960+, raw)",
             fontsize=11)
ax.grid(True, linestyle=":", linewidth=0.6, color="0.8", zorder=0)
ax.set_axisbelow(True)
legend = ax.legend(loc="upper right", frameon=True, framealpha=0.95,
                   fontsize=9)
legend.get_frame().set_edgecolor("0.8")

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT}")

# ── summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SUMMARY: within-country R² across lags (education → four outcomes)")
print("=" * 90)
print(f"{'Lag':>5}  {'LE':>7}  {'TFR':>7}  {'U5 log':>7}  {'ChildEdu':>9}")
print("-" * 50)
for i, lag in enumerate(lags):
    marker = (" ← gen " + {25: "1", 50: "2", 75: "3", 100: "4"}[lag]
              if lag in (25, 50, 75, 100) else "")
    cells = []
    for (label, _, _, _, _) in OUTCOMES:
        v = results[label][i]
        cells.append(f"{v:>7.3f}" if not np.isnan(v) else "    nan")
    print(f"{lag:>5}  {'  '.join(cells)}{marker}")

# ── checkin JSON ──────────────────────────────────────────────────────────────
numbers = {"n_countries": len(countries)}
for (label, key, _, _, _) in OUTCOMES:
    for i, lag in enumerate(lags):
        v = results[label][i]
        numbers[f"{key}_r2_lag{lag}"] = round(v, 3) if not np.isnan(v) else None
# GDP -> outcome curves (raw, no backfill)
for (_, key, _) in PLOT_ORDER:
    for i, lag in enumerate(lags):
        v = gdp_results[key][i]
        numbers[f"gdp_{key}_r2_lag{lag}"] = round(v, 3) if not np.isnan(v) else None

write_checkin("outcomes_r2_by_lag.json", {
    "notes": (f"{len(countries)} countries. Education uses WCDE 1875-2015; "
              f"outcomes use WDI 1960-2015. Child education uses WCDE as outcome. "
              f"U-5 mortality logged. All within-country FE."),
    "numbers": numbers,
}, script_path="scripts/figures/outcomes_r2_by_lag.py")

print("\nDone.")
