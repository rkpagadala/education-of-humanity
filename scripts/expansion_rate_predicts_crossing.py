"""
expansion_rate_predicts_crossing.py
Does expansion RATE predict development crossing better than education LEVEL?

Uses Table 4 case studies: Taiwan, Korea, Cuba, Bangladesh, Sri Lanka, China, Kerala.
"Crossing" = crossing developed-country life expectancy or fertility thresholds.

Tests:
  1. Rate vs lag (R², slope, p-value)
  2. Rate + starting base (does base add predictive power?)
  3. Rate alone vs level alone (R² comparison)
  4. Predicted vs actual lag from rate-only model
  5. Gap-adjusted rate: rate × (100 - base) / 100
"""

import os
import sys
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import REPO_ROOT

# ── Data ──────────────────────────────────────────────────────────────────────
countries = ["Taiwan", "Korea", "Cuba", "Bangladesh", "Sri Lanka", "China", "Kerala"]
rate       = np.array([2.15, 2.13, 2.27, 1.30, 1.20, 1.50, 0.87])   # pp/yr
start_yr   = np.array([1950, 1953, 1961, 1990, 1950, 1950, 1920])
cross_yr   = np.array([1972, 1987, 1974, 2014, 1993, 1994, 1982])
lag        = cross_yr - start_yr                                       # years
base       = np.array([18, 25, 40, 20, 15, 10, 5])                    # starting %

# ── India verification (crossed 2017 — compute rate from WCDE) ────────────
# India: onset 1950s, WCDE v3 lower secondary completion (both sexes, 20-24)
# 1950: 9.9%, 2015: 65.8% (WCDE v3)
india_start_yr = 1950
india_end_yr = 2015
india_start_pct = 9.9
india_end_pct = 65.8
india_rate = (india_end_pct - india_start_pct) / (india_end_yr - india_start_yr)
print(f"India rate verification: ({india_end_pct} - {india_start_pct}) / "
      f"({india_end_yr} - {india_start_yr}) = {india_rate:.2f} pp/yr")
print(f"  Paper claims: 0.87 pp/yr")
print(f"  Computed:     {india_rate:.2f} pp/yr")
if abs(india_rate - 0.87) < 0.05:
    print(f"  MATCH (within 0.05 pp/yr)")
else:
    print(f"  *** MISMATCH *** — difference = {india_rate - 0.87:.2f} pp/yr")
print()

# Derived: completion at crossing (approximate: base + rate × lag)
completion_at_crossing = base + rate * lag
# Gap-adjusted rate: how fast you close the remaining gap to 100%
gap_adj_rate = rate * (100 - base) / 100

n = len(countries)
sep = "=" * 72

print(sep)
print("DATA SUMMARY")
print(sep)
print(f"{'Country':<14} {'Rate':>6} {'Base%':>6} {'Lag':>5} {'Compl@Cross':>12} {'GapAdj':>8}")
for i in range(n):
    print(f"{countries[i]:<14} {rate[i]:6.2f} {base[i]:6.1f} {lag[i]:5d} "
          f"{completion_at_crossing[i]:12.1f} {gap_adj_rate[i]:8.3f}")

# ── TEST 1: Rate vs lag ──────────────────────────────────────────────────────
print(f"\n{sep}")
print("TEST 1: Lag ~ Rate (OLS)")
print(sep)

X1 = sm.add_constant(rate)
m1 = sm.OLS(lag, X1).fit()
print(m1.summary2().tables[1].to_string())
print(f"\n  R²    = {m1.rsquared:.4f}")
print(f"  Adj-R² = {m1.rsquared_adj:.4f}")
print(f"  Slope  = {m1.params[1]:.2f} years per pp/yr")
print(f"  p(rate)= {m1.pvalues[1]:.4f}")

# Pearson correlation for clarity
r_val, p_val = stats.pearsonr(rate, lag)
print(f"  Pearson r = {r_val:.4f}, p = {p_val:.4f}")

# ── TEST 2: Rate + starting base ─────────────────────────────────────────────
print(f"\n{sep}")
print("TEST 2: Lag ~ Rate + Starting Base (OLS)")
print(sep)

X2 = sm.add_constant(np.column_stack([rate, base]))
m2 = sm.OLS(lag, X2).fit()
print(m2.summary2().tables[1].to_string())
print(f"\n  R²    = {m2.rsquared:.4f}")
print(f"  Adj-R² = {m2.rsquared_adj:.4f}")
print(f"  p(rate)= {m2.pvalues[1]:.4f}")
print(f"  p(base)= {m2.pvalues[2]:.4f}")

delta_r2 = m2.rsquared - m1.rsquared
print(f"\n  Adding base increases R² by {delta_r2:.4f}")
if m2.pvalues[2] > 0.10:
    print("  --> Base is NOT significant (p > 0.10). Rate is sufficient.")
else:
    print("  --> Base adds significant predictive power.")

# ── TEST 3: R² comparison — rate vs base vs completion at crossing ───────────
print(f"\n{sep}")
print("TEST 3: R² comparison — which predictor best explains lag?")
print(sep)

predictors = {
    "Rate (pp/yr)":             rate,
    "Starting base (%)":        base,
    "Completion at crossing (%)": completion_at_crossing,
    "Gap-adjusted rate":        gap_adj_rate,
}

results = {}
for name, x in predictors.items():
    Xp = sm.add_constant(x)
    mp = sm.OLS(lag, Xp).fit()
    rp, pp = stats.pearsonr(x, lag)
    results[name] = {"R2": mp.rsquared, "adjR2": mp.rsquared_adj,
                     "slope": mp.params[1], "p": mp.pvalues[1], "r": rp}
    print(f"\n  {name}")
    print(f"    R²     = {mp.rsquared:.4f}")
    print(f"    Adj-R² = {mp.rsquared_adj:.4f}")
    print(f"    Slope  = {mp.params[1]:.3f}")
    print(f"    p      = {mp.pvalues[1]:.4f}")
    print(f"    r      = {rp:.4f}")

best = max(results, key=lambda k: results[k]["R2"])
print(f"\n  ** Best single predictor: {best}  (R² = {results[best]['R2']:.4f}) **")

# ── TEST 4: Predicted vs actual lag from rate-only model ─────────────────────
print(f"\n{sep}")
print("TEST 4: Predicted vs actual lag (rate-only model)")
print(sep)

pred_lag = m1.predict(X1)
print(f"{'Country':<14} {'Actual':>7} {'Predicted':>10} {'Error':>7} {'%Error':>8}")
for i in range(n):
    err = pred_lag[i] - lag[i]
    pct = 100 * err / lag[i]
    print(f"{countries[i]:<14} {lag[i]:7d} {pred_lag[i]:10.1f} {err:7.1f} {pct:7.1f}%")

mae = np.mean(np.abs(pred_lag - lag))
rmse = np.sqrt(np.mean((pred_lag - lag) ** 2))
print(f"\n  MAE  = {mae:.1f} years")
print(f"  RMSE = {rmse:.1f} years")

# ── TEST 5: Gap-adjusted rate ────────────────────────────────────────────────
print(f"\n{sep}")
print("TEST 5: Gap-adjusted rate = rate × (100 - base) / 100")
print(sep)

X5 = sm.add_constant(gap_adj_rate)
m5 = sm.OLS(lag, X5).fit()
pred5 = m5.predict(X5)

print(f"  R²    = {m5.rsquared:.4f}")
print(f"  Adj-R² = {m5.rsquared_adj:.4f}")
print(f"  Slope  = {m5.params[1]:.2f}")
print(f"  p      = {m5.pvalues[1]:.4f}")

print(f"\n{'Country':<14} {'GapAdj':>8} {'Actual':>7} {'Predicted':>10} {'Error':>7}")
for i in range(n):
    err = pred5[i] - lag[i]
    print(f"{countries[i]:<14} {gap_adj_rate[i]:8.3f} {lag[i]:7d} {pred5[i]:10.1f} {err:7.1f}")

mae5 = np.mean(np.abs(pred5 - lag))
rmse5 = np.sqrt(np.mean((pred5 - lag) ** 2))
print(f"\n  MAE  = {mae5:.1f} years")
print(f"  RMSE = {rmse5:.1f} years")

improvement = (results["Rate (pp/yr)"]["R2"] - results["Gap-adjusted rate"]["R2"])
print(f"\n  Rate-only R² = {results['Rate (pp/yr)']['R2']:.4f}")
print(f"  Gap-adj  R²  = {results['Gap-adjusted rate']['R2']:.4f}")
if improvement > 0:
    print(f"  --> Raw rate is better by {improvement:.4f} R²")
else:
    print(f"  --> Gap-adjusted rate is better by {-improvement:.4f} R²")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("SUMMARY")
print(sep)
print(f"  {'Predictor':<30} {'R²':>8} {'p-value':>10}")
print(f"  {'-'*50}")
for name in predictors:
    r2 = results[name]["R2"]
    p = results[name]["p"]
    flag = " <-- BEST" if name == best else ""
    print(f"  {name:<30} {r2:8.4f} {p:10.4f}{flag}")

print(f"\n  Rate + Base (multivariate):  R² = {m2.rsquared:.4f}, "
      f"p(base) = {m2.pvalues[2]:.4f}")
print(f"\n  Rate-only model: MAE = {mae:.1f} years, RMSE = {rmse:.1f} years")

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Rate vs Lag
ax = axes[0]
ax.scatter(rate, lag, s=80, c="steelblue", zorder=5)
for i in range(n):
    ax.annotate(countries[i], (rate[i], lag[i]), fontsize=8,
                xytext=(5, 5), textcoords="offset points")
x_fit = np.linspace(rate.min() - 0.1, rate.max() + 0.1, 100)
y_fit = m1.params[0] + m1.params[1] * x_fit
ax.plot(x_fit, y_fit, "r-", lw=2, label=f"R² = {m1.rsquared:.3f}")
ax.set_xlabel("Expansion rate (pp/yr)")
ax.set_ylabel("Lag to crossing (years)")
ax.set_title("A. Rate predicts lag")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Starting base vs Lag
ax = axes[1]
ax.scatter(base, lag, s=80, c="darkorange", zorder=5)
for i in range(n):
    ax.annotate(countries[i], (base[i], lag[i]), fontsize=8,
                xytext=(5, 5), textcoords="offset points")
Xb = sm.add_constant(base)
mb = sm.OLS(lag, Xb).fit()
x_fit2 = np.linspace(base.min() - 2, base.max() + 2, 100)
y_fit2 = mb.params[0] + mb.params[1] * x_fit2
ax.plot(x_fit2, y_fit2, "r-", lw=2, label=f"R² = {mb.rsquared:.3f}")
ax.set_xlabel("Starting completion (%)")
ax.set_ylabel("Lag to crossing (years)")
ax.set_title("B. Starting level predicts lag")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel C: Gap-adjusted rate vs Lag
ax = axes[2]
ax.scatter(gap_adj_rate, lag, s=80, c="seagreen", zorder=5)
for i in range(n):
    ax.annotate(countries[i], (gap_adj_rate[i], lag[i]), fontsize=8,
                xytext=(5, 5), textcoords="offset points")
x_fit3 = np.linspace(gap_adj_rate.min() - 0.05, gap_adj_rate.max() + 0.05, 100)
y_fit3 = m5.params[0] + m5.params[1] * x_fit3
ax.plot(x_fit3, y_fit3, "r-", lw=2, label=f"R² = {m5.rsquared:.3f}")
ax.set_xlabel("Gap-adjusted rate")
ax.set_ylabel("Lag to crossing (years)")
ax.set_title("C. Gap-adjusted rate predicts lag")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(REPO_ROOT, "scripts", "expansion_rate_predicts_crossing.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n  Figure saved: {fig_path}")

# ── Checkin JSON ──────────────────────────────────────────────────────────────
import json
import math

generations = [round(l / 25) for l in lag]
# Kerala (62 yr, 2.48 gen) rounds to 2 but spans parts of 3 generations
kerala_idx = countries.index("Kerala")
generations[kerala_idx] = 3
checkin = {}
for i, c in enumerate(countries):
    checkin[c] = {
        "rate": float(rate[i]),
        "base": float(base[i]),
        "lag": int(lag[i]),
        "generations": generations[i],
    }

checkin_path = os.path.join(REPO_ROOT, "checkin", "expansion_rate_predicts_crossing.json")
with open(checkin_path, "w") as f:
    json.dump(checkin, f, indent=2)
print(f"  Checkin written to {checkin_path}")
print(sep)
