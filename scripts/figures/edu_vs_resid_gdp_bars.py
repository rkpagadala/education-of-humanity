"""
edu_vs_resid_gdp_bars.py
=======================
Figure 5: Education R² vs Residualized GDP R² across four outcomes.

Blue bars (education) at 0.28-0.52. Red bars (residualized GDP) near zero.
The visual: education predicts everything, GDP predicts nothing independently.

Entry-cohort design (entry ≥ 10%, ceiling ≤ 90%), country FE,
lower secondary completion, T=1960-1990, lag=25.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import load_education, load_wb, interpolate_to_annual, REPO_ROOT
from residualization._shared import (
    precompute_entry_years, build_panel, build_child_edu_panel,
    filter_panel, fe_r2, fe_residualize_gdp, compare_predictors,
)

PAPER_DIR = os.path.join(REPO_ROOT, "paper")

# ── Compute results ──────────────────────────────────────────────────

T_YEARS = list(range(1960, 1995, 5))
LAG = 25
COL_NAME = "lower_sec"
CEILING = 90

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
le_df = load_wb("life_expectancy_years.csv")
tfr_df = load_wb("children_per_woman_total_fertility.csv")
u5mr_df = load_wb("child_mortality_u5.csv")

edu_annual = interpolate_to_annual(edu_raw, COL_NAME)
entry_years = precompute_entry_years(edu_annual)
cohort = entry_years[10]

outcomes = {}

# LE, TFR, U5MR from WB data
for label, outcome_col, outcome_df in [
    ("Life\nexpectancy", "le", le_df),
    ("Fertility", "tfr", tfr_df),
    ("Child\nmortality", "u5mr", u5mr_df),
]:
    panel = build_panel(edu_annual, outcome_df, gdp_df, T_YEARS, LAG, outcome_col)
    sub = filter_panel(panel, cohort, CEILING)
    cp = compare_predictors(sub, outcome_col)
    r2_r = cp["resid_gdp_r2"] if not np.isnan(cp["resid_gdp_r2"]) else 0.0
    outcomes[label] = {"edu": cp["edu_r2"], "resid": r2_r, "n": cp["n"], "ctry": cp["countries"]}
    print(f"  {label.replace(chr(10), ' ')}: edu={cp['edu_r2']:.3f}, resid={r2_r:.3f}, n={cp['n']}, ctry={cp['countries']}")

# Child education (parent→child from WCDE)
panel_ce = build_child_edu_panel(edu_annual, gdp_df, T_YEARS, LAG)
sub = filter_panel(panel_ce, cohort, CEILING)
cp = compare_predictors(sub, "child_edu")
r2_r = cp["resid_gdp_r2"] if not np.isnan(cp["resid_gdp_r2"]) else 0.0
outcomes["Child\neducation"] = {"edu": cp["edu_r2"], "resid": r2_r, "n": cp["n"], "ctry": cp["countries"]}
print(f"  Child education: edu={cp['edu_r2']:.3f}, resid={r2_r:.3f}, n={cp['n']}, ctry={cp['countries']}")

# ── Plot ─────────────────────────────────────────────────────────────

labels = ["Life\nexpectancy", "Fertility", "Child\neducation", "Child\nmortality"]
edu_vals = [outcomes[l]["edu"] for l in labels]
resid_vals = [outcomes[l]["resid"] for l in labels]

fig, ax = plt.subplots(figsize=(8, 4.5))

x = np.arange(len(labels))
width = 0.35

bars_edu = ax.bar(x - width/2, edu_vals, width, color='#2563eb', label='Education', zorder=3)
bars_resid = ax.bar(x + width/2, resid_vals, width, color='#ef4444', label='GDP (after removing\neducation\'s contribution)', zorder=3)

# Labels on bars
for bar, val in zip(bars_edu, edu_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2563eb')

for bar, val in zip(bars_resid, resid_vals):
    label_text = f'{val:.1%}' if val >= 0.005 else '0%'
    ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0.005) + 0.01,
            label_text, ha='center', va='bottom', fontsize=11, fontweight='bold', color='#ef4444')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Within-country R² (predictive power)', fontsize=11)
ax.set_ylim(0, 0.65)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, zorder=0)

ax.set_title('GDP has no independent effect on any development outcome',
             fontsize=13, fontweight='bold', pad=15)

fig.text(0.5, -0.02,
         'Country fixed effects, lower secondary completion, entry ≥ 10%, ceiling ≤ 90%, T=1960–1990, lag=25 years.',
         ha='center', fontsize=8, color='#64748b')

plt.tight_layout()

# Save for paper
out_paper = os.path.join(PAPER_DIR, "edu_vs_resid_gdp_bars.png")
fig.savefig(out_paper, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved to {out_paper}")

plt.close()
