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
from _shared import (load_education, load_wb, interpolate_to_annual, REPO_ROOT,
                     FOR_WEB, FOR_ZH, L, fig_file, PAPER_FIGS, WEB_FIGS,
                     LAG_GENERATION)
from residualization._shared import (
    precompute_entry_years, build_panel, build_child_edu_panel,
    filter_panel, fe_r2, fe_residualize_gdp, compare_predictors,
)

PAPER_DIR = os.path.join(REPO_ROOT, "paper")
FIGURES_DIR = PAPER_FIGS if not FOR_WEB else WEB_FIGS
CHECKIN_DIR = os.path.join(REPO_ROOT, "checkin")

# ── Compute results ──────────────────────────────────────────────────

T_YEARS = list(range(1960, 1995, 5))
LAG = LAG_GENERATION
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

bars_edu = ax.bar(x - width/2, edu_vals, width, color='#2563eb',
                  label=L('Education', '教育'), zorder=3)
bars_resid = ax.bar(x + width/2, resid_vals, width, color='#ef4444',
                    label=L('GDP (after removing\neducation\'s contribution)',
                            'GDP（剔除教育\n贡献之后）'), zorder=3)

# Labels on bars
for bar, val in zip(bars_edu, edu_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2563eb')

for bar, val in zip(bars_resid, resid_vals):
    label_text = f'{val:.1%}' if val >= 0.005 else '0%'
    ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0.005) + 0.01,
            label_text, ha='center', va='bottom', fontsize=11, fontweight='bold', color='#ef4444')

ax.set_xticks(x)
display_labels = [
    L("Life\nexpectancy", "预期\n寿命"),
    L("Fertility", "生育率"),
    L("Child\neducation", "下一代\n教育"),
    L("Child\nmortality", "儿童\n死亡率"),
]
ax.set_xticklabels(display_labels, fontsize=11)
ax.set_ylabel(L('Within-country R² (predictive power)', '国家内部 R²（预测力）'), fontsize=11)
ax.set_ylim(0, 0.65)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, zorder=0)

if FOR_WEB:
    ax.set_title(L('GDP has no independent effect on any development outcome',
                   'GDP 对任何发展产出都没有独立作用'),
                 fontsize=13, fontweight='bold', pad=15)

fig.text(0.5, -0.02,
         L(f'Country fixed effects, lower secondary completion, entry ≥ 10%, ceiling ≤ 90%, T=1960–1990, lag={LAG} years.',
           f'国家固定效应，初中完成率，进入 ≥ 10%，天花板 ≤ 90%，T=1960–1990，滞后 {LAG} 年。'),
         ha='center', fontsize=8, color='#64748b')

plt.tight_layout()

# Save for paper
out_paper = os.path.join(FIGURES_DIR, fig_file("edu_vs_resid_gdp_bars.png"))
fig.savefig(out_paper, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved to {out_paper}")

plt.close()

# Checkin stamp — values shown on the figure (identical across languages;
# only the English/paper build writes it to avoid a redundant rewrite).
import json
if not FOR_ZH:
    stamp = {
        "figure": "edu_vs_resid_gdp_bars.png",
        "outcomes": {l.replace("\n", " "): outcomes[l] for l in labels},
    }
    stamp_path = os.path.join(CHECKIN_DIR, "edu_vs_resid_gdp_bars.json")
    with open(stamp_path, "w") as f:
        json.dump(stamp, f, indent=2)
    print(f"Stamp written to {stamp_path}")
