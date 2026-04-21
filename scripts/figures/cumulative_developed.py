"""
Cumulative share of world population living in 'developed' countries over time.

Developed = TFR < 3.65 AND LE > 69.8 (1960 USA benchmarks).

For each year from 1960 to 2022, identifies which countries have crossed BOTH
thresholds and sums their population as a share of the world total.

Population source: WCDE v3 scenario 2 (5-year grid). Each year is matched to
the nearest WCDE population year. TFR/LE sources: World Bank WDI wide-format CSVs.

Output: paper/cumulative_developed.png
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Paths ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from _shared import (
    REPO_ROOT, DATA, REGIONS,
    TFR_THRESHOLD, LE_THRESHOLD,
    WB_REGIONS, WB_TO_WCDE,
    load_wide_indicator, load_population_by_year,
)

OUT_PATH = os.path.join(REPO_ROOT, "paper", "figures", "cumulative_developed.png")

# ── Constants ────────────────────────────────────────────────────
START_YEAR = 1960
END_YEAR = 2022


def main():
    tfr_all = load_wide_indicator("children_per_woman_total_fertility.csv")
    le_all = load_wide_indicator("life_expectancy_years.csv")
    pop_all = load_population_by_year()

    # Available WCDE population years (5-year grid)
    pop_years = sorted(pop_all.columns)

    years = list(range(START_YEAR, END_YEAR + 1))
    pct_developed = []
    n_countries = []

    for yr in years:
        yr_str = str(yr)
        if yr_str not in tfr_all.columns or yr_str not in le_all.columns:
            pct_developed.append(None)
            n_countries.append(0)
            continue

        tfr = tfr_all[yr_str].dropna()
        le = le_all[yr_str].dropna()

        crossed_tfr = set(tfr[tfr < TFR_THRESHOLD].index)
        crossed_le = set(le[le > LE_THRESHOLD].index)
        crossed = crossed_tfr & crossed_le

        # Philippines crossed in 2020 (LE 70.1, TFR 2.08); COVID pulled LE
        # below threshold in 2021-22. Count as crossed from 2020 onward.
        if yr >= 2020 and "philippines" in crossed_tfr:
            crossed = crossed | {"philippines"}

        # Find nearest WCDE population year
        nearest_pop_yr = min(pop_years, key=lambda y: abs(y - yr))
        pop = pop_all[nearest_pop_yr].dropna()

        pop_crossed = sum(pop.get(c, 0) for c in crossed)
        pop_world = pop.sum()

        pct = 100.0 * pop_crossed / pop_world if pop_world > 0 else 0
        pct_developed.append(pct)
        n_countries.append(len(crossed))

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_pct = "#1f4e79"
    ax1.fill_between(years, pct_developed, alpha=0.15, color=color_pct)
    ax1.plot(years, pct_developed, color=color_pct, linewidth=2.5,
             label="% of world population")
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Share of world population (%)", fontsize=12, color=color_pct)
    ax1.tick_params(axis="y", labelcolor=color_pct)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_ylim(0, 100)
    ax1.set_xlim(START_YEAR, END_YEAR)

    # Second axis: number of countries
    color_n = "#c0392b"
    ax2 = ax1.twinx()
    ax2.plot(years, n_countries, color=color_n, linewidth=1.5,
             linestyle="--", alpha=0.7, label="Number of countries")
    ax2.set_ylabel("Number of countries crossed", fontsize=12, color=color_n)
    ax2.tick_params(axis="y", labelcolor=color_n)
    ax2.set_ylim(0, 180)

    # Annotations for key crossings
    annotations = [
        (1972, "Taiwan"),
        (1987, "S. Korea"),
        (1994, "China"),
        (2014, "Bangladesh"),
        (2022, f"{n_countries[-1]} countries"),
    ]
    for yr, label in annotations:
        idx = yr - START_YEAR
        if 0 <= idx < len(pct_developed) and pct_developed[idx] is not None:
            ax1.annotate(
                label, xy=(yr, pct_developed[idx]),
                xytext=(0, 18), textcoords="offset points",
                fontsize=8, ha="center", color="#333333",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8),
            )

    ax1.set_title(
        "Cumulative share of world population in 'developed' countries\n"
        f"(TFR < {TFR_THRESHOLD}, LE > {LE_THRESHOLD} — 1960 USA benchmarks)",
        fontsize=13, fontweight="bold", pad=15,
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")

    # Print key milestones
    print("\nKey milestones:")
    for target in [10, 25, 50, 75]:
        for i, pct in enumerate(pct_developed):
            if pct is not None and pct >= target:
                print(f"  {target}% of world population: {years[i]}  "
                      f"({n_countries[i]} countries, {pct:.1f}%)")
                break


if __name__ == "__main__":
    main()
