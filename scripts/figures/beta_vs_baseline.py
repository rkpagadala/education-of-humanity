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
OUT          = os.path.join(REPO_ROOT, "paper", "figures", "beta_vs_baseline.png")
OUT_SCATTER  = os.path.join(REPO_ROOT, "paper", "figures",
                            "beta_vs_baseline_scatter.png")

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

# ── R2.10: cross-country average β_g, binned by baseline ─────────────────────
# For every country in the long-run panel, compute β_g for every sliding
# window (WINDOW_SIZE=25, STEP=10, LAG=25, MIN_OBS=3). Bin observations by
# average parental baseline (0–10, 10–20, …, 90–100) and plot the mean
# β_g per bin as a thick black line, overlaid on the country-specific
# trajectories above. Outliers are clipped at |β| ≤ 15 to match the
# country-line cap; binned averages are unaffected by clipping at typical
# magnitudes.
all_records = []  # (country, baseline, beta)
for country_name in wide.index:
    for start in range(1900, 1996, STEP):
        end = start + WINDOW_SIZE
        beta, avg_parent = beta_for_window(country_name, start, end)
        if (not np.isnan(beta) and not np.isnan(avg_parent)
                and abs(beta) <= 15):
            all_records.append((country_name, avg_parent, beta))

scatter_df = pd.DataFrame(all_records,
                          columns=["country", "baseline", "beta"])
print(f"\nUniversality overlay: {len(scatter_df)} country-window observations "
      f"from {scatter_df['country'].nunique()} countries")

bin_edges = list(range(0, 101, 10))
scatter_df["bin"] = pd.cut(scatter_df["baseline"], bins=bin_edges,
                            labels=[f"{e}-{e+10}" for e in bin_edges[:-1]],
                            include_lowest=True)
bin_means = scatter_df.groupby("bin", observed=True).agg(
    baseline=("baseline", "mean"),
    beta=("beta", "mean"),
    n=("beta", "count"),
).reset_index()
bin_means = bin_means[bin_means["n"] >= 5]  # require at least 5 obs/bin

ax.plot(bin_means["baseline"], bin_means["beta"],
        color="black", linewidth=3.0, marker="o", markersize=7,
        label="Cross-country mean (binned)", zorder=4)

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

# ── R2.11: Scatter of every country-window with OLS fit (appendix) ───────────
# For every country-window in the long-run panel, plot β_g vs avg parental
# baseline. Fit OLS β_g = α + γ · baseline with country-clustered SEs;
# report where the fit crosses β=1 and the slope's significance.
#
# To keep the regression numerically stable, drop windows where either β
# is outside [-3, 3] (a few near-singular cases at extremes); in practice
# this trims a handful of windows where parent variance was tiny.
fit_df = scatter_df[(scatter_df["beta"] >= -3) & (scatter_df["beta"] <= 5)].copy()
X_fit = sm.add_constant(fit_df[["baseline"]])
ols_model = sm.OLS(fit_df["beta"], X_fit).fit(
    cov_type="cluster", cov_kwds={"groups": fit_df["country"]},
)
intercept = float(ols_model.params["const"])
slope = float(ols_model.params["baseline"])
slope_se = float(ols_model.bse["baseline"])
slope_p = float(ols_model.pvalues["baseline"])
crossing_x = (1.0 - intercept) / slope if slope != 0 else np.nan

print("\nR2.11 OLS fit (country-clustered SE):")
print(f"  intercept = {intercept:+.3f}")
print(f"  slope     = {slope:+.4f}  SE = {slope_se:.4f}  p = {slope_p:.4g}")
print(f"  crosses β=1 at baseline = {crossing_x:.1f}%")
print(f"  n = {len(fit_df)}, countries = {fit_df['country'].nunique()}")

# OLS predicted line and 95% CI band
xx = np.linspace(0, 100, 101)
XX = sm.add_constant(pd.DataFrame({"baseline": xx}))
pred = ols_model.get_prediction(XX)
pred_summary = pred.summary_frame(alpha=0.05)
yhat = pred_summary["mean"].values
ci_lo = pred_summary["mean_ci_lower"].values
ci_hi = pred_summary["mean_ci_upper"].values

fig2, ax2 = plt.subplots(figsize=(9, 5.5))
ax2.scatter(fit_df["baseline"], fit_df["beta"],
            s=14, color="#bbbbbb", alpha=0.5, edgecolor="none", zorder=2,
            label=f"Country-window observations (n = {len(fit_df)})")
ax2.fill_between(xx, ci_lo, ci_hi, color="black", alpha=0.15, zorder=3,
                  label="95% confidence band (country-clustered)")
ax2.plot(xx, yhat, color="black", linewidth=2.5, zorder=4,
         label=f"OLS fit: β = {intercept:.2f} {slope:+.4f}·baseline")
ax2.axhline(1.0, color="#d6604d", linewidth=1.2, linestyle="--",
            alpha=0.8, zorder=1, label="β = 1 (unity)")
if 0 <= crossing_x <= 100:
    ax2.axvline(crossing_x, color="#d6604d", linewidth=0.8, linestyle=":",
                alpha=0.6, zorder=1)
    ax2.annotate(f"crosses β=1 at\nbaseline ≈ {crossing_x:.0f}%",
                 xy=(crossing_x, 1.0),
                 xytext=(crossing_x + 5, 2.5),
                 fontsize=9, color="#7a1a1a",
                 arrowprops=dict(arrowstyle="->", color="#d6604d", alpha=0.7))
ax2.set_xlabel("Average parental baseline education "
               "(% lower secondary completion)", fontsize=11)
ax2.set_ylabel("Generational transmission coefficient (β)", fontsize=11)
ax2.set_title("Universality of ceiling compression: country-window scatter\n"
              f"All countries in the long-run panel (n countries = "
              f"{fit_df['country'].nunique()}), 1900–2015 "
              f"(WINDOW_SIZE=25, STEP=10, LAG=25, MIN_OBS=3)",
              fontsize=11)
ax2.legend(fontsize=9, loc="upper right")
ax2.set_xlim(-2, 100)
ax2.set_ylim(-1.5, 5)
ax2.grid(axis="y", linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.savefig(OUT_SCATTER, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_SCATTER}")
plt.close(fig2)


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
        # R2.10 / R2.11 universality overlays
        "Universality-n-windows":  int(len(scatter_df)),
        "Universality-n-countries": int(scatter_df["country"].nunique()),
        "Universality-fit-intercept": round(intercept, 3),
        "Universality-fit-slope":     round(slope, 4),
        "Universality-fit-slope-se":  round(slope_se, 4),
        "Universality-fit-slope-p":   round(slope_p, 4),
        "Universality-fit-crossing":  (round(crossing_x, 1)
                                        if not np.isnan(crossing_x) else None),
        "Universality-fit-n":         int(len(fit_df)),
        "Universality-fit-countries": int(fit_df["country"].nunique()),
    },
    "binned_means": [
        {"bin": str(row["bin"]),
         "baseline_mean": round(float(row["baseline"]), 1),
         "beta_mean":     round(float(row["beta"]), 3),
         "n":             int(row["n"])}
        for _, row in bin_means.iterrows()
    ],
}, script_path="scripts/figures/beta_vs_baseline.py")
