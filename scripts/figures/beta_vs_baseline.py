"""
figures/beta_vs_baseline.py

Generates Figure 3 for:
  "Education of Nations"

Output:
  paper/beta_vs_baseline.png

What it does:
  For each country, computes the intergenerational education transmission
  coefficient (β) using a sliding window of 6 child cohorts (25 years),
  stepping forward 10 years at a time.

  Plots β against average parental baseline education for each window,
  showing that β varies systematically with baseline: β>1 at low baselines
  (state + PTE compounding), β→0 near ceiling, β always positive
  (durability).

Data source:
  WCDE v3 long-run cohort data (1875-2015), processed by education-rupture
  pipeline. Lower secondary completion, age 20-24, both sexes.

Key parameters:
  WINDOW_SIZE = 25 years (6 child cohorts at 5-year intervals)
  STEP        = 10 years
  LAG         = 25 years (one PTE cycle)
  MIN_OBS     = 3 (minimum data points per window)
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
from _shared import PROC, DATA, CHECKIN, REGIONS, REPO_ROOT, write_checkin

PTE_PROC   = PROC
OUT        = os.path.join(REPO_ROOT, "paper", "figures", "beta_vs_baseline.png")

# ── parameters ────────────────────────────────────────────────────────────────
WINDOW_SIZE = 25    # years (6 cohorts at 5-year intervals)
STEP        = 10    # years between window starts
LAG         = 25    # one PTE cycle
MIN_OBS     = 3     # minimum observations per window

# ── load data ─────────────────────────────────────────────────────────────────
longrun = pd.read_csv(os.path.join(PTE_PROC, "cohort_completion_both_long.csv"))
longrun = longrun[~longrun["country"].isin(REGIONS)]
wide = longrun.pivot(index="country", columns="cohort_year", values="lower_sec")

def v(country, year):
    try:
        val = float(wide.loc[country, int(year)])
        return val if not np.isnan(val) else np.nan
    except (KeyError, ValueError):
        return np.nan

def beta_for_window(country, child_start, child_end):
    """OLS β for child cohorts in [child_start, child_end], parent = child - LAG."""
    rows = []
    for cy in range(child_start, child_end + 1, 5):
        py = cy - LAG
        child = v(country, cy)
        parent = v(country, py)
        if not np.isnan(child) and not np.isnan(parent):
            rows.append({"child": child, "parent": parent})
    if len(rows) < MIN_OBS:
        return np.nan, np.nan
    df = pd.DataFrame(rows)
    reg = sm.OLS(df["child"], sm.add_constant(df[["parent"]])).fit()
    return reg.params.iloc[1], df["parent"].mean()

# ── countries to plot ─────────────────────────────────────────────────────────
# Selected to show full range: early developer (USA), rapid state (Korea),
# current low-baseline (Bangladesh), large developing (India)
# start_year: first sliding window start. Korea/Taiwan start later to avoid
# mechanically unstable β at near-zero denominators.
COUNTRIES = [
    ("United States of America",       "USA",         "#2166ac", "o",  1900),
    ("Republic of Korea",              "Korea",       "#d6604d", "s",  1920),
    ("Taiwan Province of China",       "Taiwan",      "#e08214", "v",  1930),
    ("Philippines",                    "Philippines", "#878787", "P",  1920),
    ("Bangladesh",                     "Bangladesh",  "#1b7837", "^",  1900),
    ("India",                          "India",       "#762a83", "D",  1900),
]

# ── compute sliding-window β ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

for country_name, label, color, marker, start_year in COUNTRIES:
    baselines = []
    betas = []
    for start in range(start_year, 1996, STEP):
        end = start + WINDOW_SIZE
        beta, avg_parent = beta_for_window(country_name, start, end)
        if not np.isnan(beta) and not np.isnan(avg_parent):
            baselines.append(avg_parent)
            betas.append(beta)

    ax.plot(baselines, betas, color=color, linewidth=2, marker=marker,
            markersize=6, label=label, zorder=3)

# ── reference lines ──────────────────────────────────────────────────────────
ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)
ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="-", alpha=0.4, zorder=1)

# annotate β=1 line
ax.text(85, 1.05, "β = 1 (unity)", fontsize=8, color="grey", va="bottom")

# ── formatting ───────────────────────────────────────────────────────────────
ax.set_xlabel("Average parental baseline education (% lower secondary completion)",
              fontsize=11)
ax.set_ylabel("Generational transmission coefficient (β)", fontsize=11)
ax.set_title(
    "Figure 3. Generational β Varies With Baseline Education Level\n"
    "Sliding window (25 years), within-country OLS, 1900–2015",
    fontsize=12,
)
ax.legend(fontsize=10, loc="upper right")
ax.set_xlim(-2, 100)
ax.set_ylim(-0.5, None)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)

# cap y-axis to keep the chart readable (Korea/Bangladesh β stays ≤14)
ymax = min(ax.get_ylim()[1], 15)
ax.set_ylim(-0.5, ymax)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")

# ── summary table ─────────────────────────────────────────────────────────────
print("\nKey values:")
for country_name, label, _, _, start_year in COUNTRIES:
    print(f"\n  {label}:")
    print(f"  {'Window':<15} {'β':>8} {'Avg parent%':>12}")
    for start in range(start_year, 1996, STEP):
        end = start + WINDOW_SIZE
        beta, avg_parent = beta_for_window(country_name, start, end)
        if not np.isnan(beta):
            print(f"  {start}-{end:<10} {beta:>8.3f} {avg_parent:>11.1f}%")

# ── Write checkin JSON ───────────────────────────────────────────────────────

# Compute specific beta values needed by verify_humanity.py
def get_beta(country_name, child_start):
    child_end = child_start + WINDOW_SIZE
    beta, avg_parent = beta_for_window(country_name, child_start, child_end)
    return round(beta, 2) if not np.isnan(beta) else None

write_checkin("beta_vs_baseline.json", {
    "numbers": {
        "Fig1-USA-beta-high": get_beta("United States of America", 1900),
        "Fig1-USA-beta-low": get_beta("United States of America", 1980),
        "Fig1-Korea-beta-high": get_beta("Republic of Korea", 1920),
        "Fig1-Korea-beta-3.6": get_beta("Republic of Korea", 1930),
        "Fig1-Korea-beta-1.8": get_beta("Republic of Korea", 1960),
        "Fig1-Korea-beta-low": get_beta("Republic of Korea", 1980),
        "Fig1-Taiwan-beta": get_beta("Taiwan Province of China", 1930),
        "Fig1-Phil-beta-high": get_beta("Philippines", 1920),
        "Fig1-Phil-beta-low": get_beta("Philippines", 1990),
    },
}, script_path="scripts/figures/beta_vs_baseline.py")
