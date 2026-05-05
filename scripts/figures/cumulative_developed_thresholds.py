"""
cumulative_developed_thresholds.py

Sensitivity analysis: cumulative share of world population in 'developed'
countries over time, under multiple threshold definitions.

Specs:
  Loose:   TFR < 4.5,  LE > 65
  Main:    TFR < 3.65, LE > 69.8  (1960 USA — paper default)
  Strict:  TFR < 2.1,  LE > 72.6  (replacement fertility, 1972 USA LE)

Output: paper/cumulative_developed_thresholds.png
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from _shared import (
    REPO_ROOT, REGIONS,
    WB_REGIONS, WB_TO_WCDE,
    load_wide_indicator, load_population_by_year,
)

OUT_PATH = os.path.join(REPO_ROOT, "paper", "figures", "cumulative_developed_thresholds.png")

START_YEAR = 1960
END_YEAR = 2022

SPECS = [
    ("Loose (TFR<4.5, LE>65)",     4.5,  65.0,  "#93c5fd"),
    ("Main (TFR<3.65, LE>69.8)",   3.65, 69.8,  "#2563eb"),
    ("Strict (TFR<2.5, LE>72.6)",  2.5,  72.6,  "#1e3a5f"),
]


def compute_series(tfr_all, le_all, pop_all, tfr_thresh, le_thresh):
    pop_years = sorted(pop_all.columns)
    years = list(range(START_YEAR, END_YEAR + 1))
    pct_list = []

    for yr in years:
        yr_str = str(yr)
        if yr_str not in tfr_all.columns or yr_str not in le_all.columns:
            pct_list.append(None)
            continue

        tfr = tfr_all[yr_str].dropna()
        le = le_all[yr_str].dropna()
        crossed = set(tfr[tfr < tfr_thresh].index) & set(le[le > le_thresh].index)

        nearest_pop_yr = min(pop_years, key=lambda y: abs(y - yr))
        pop = pop_all[nearest_pop_yr].dropna()
        pop_crossed = sum(pop.get(c, 0) for c in crossed)
        pop_world = pop.sum()
        pct_list.append(100.0 * pop_crossed / pop_world if pop_world > 0 else 0)

    return years, pct_list


def main():
    print("Loading data...")
    tfr_all = load_wide_indicator("children_per_woman_total_fertility.csv")
    le_all = load_wide_indicator("life_expectancy_years.csv")
    pop_all = load_population_by_year()

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, tfr_t, le_t, color in SPECS:
        years, pct = compute_series(tfr_all, le_all, pop_all, tfr_t, le_t)
        ax.plot(years, pct, label=label, color=color, linewidth=2.5)
        # Print final value
        final = [p for p in pct if p is not None][-1]
        n_crossed = sum(1 for yr in [str(END_YEAR)]
                        if yr in tfr_all.columns
                        for c in set(tfr_all[yr].dropna()[tfr_all[yr].dropna() < tfr_t].index)
                        & set(le_all[yr].dropna()[le_all[yr].dropna() > le_t].index))
        print(f"  {label:40s}  {final:5.1f}%  ({END_YEAR})")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Share of world population (%)", fontsize=12)
    ax.set_title("Development threshold sensitivity", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlim(START_YEAR, END_YEAR)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
