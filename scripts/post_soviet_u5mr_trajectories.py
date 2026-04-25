"""
post_soviet_u5mr_trajectories.py

Add under-5 mortality (U5MR) to the post-Soviet phenotype comparison.

U5MR is the most education-sensitive outcome in the paper's framework —
it moves most directly with maternal decision-making (sanitation,
vaccination, timely care-seeking, home health practice). If reported
Soviet education was real, U5MR should fall faster in Soviet Central
Asia than in Iran/Turkey starting around 1970 (parental-transmission
lag after reported 95% lower-sec completion). If reported education
was credential-inflated, U5MR should track the neighbors.

Produces two figures:
  1. Five U5MR panels: Baltics, Slavic west, Caucasus, Central Asia +
     Iran/Turkey, with log scale so the low-mortality tail is visible.
  2. A single overlay plot comparing Central Asia vs Iran/Turkey.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import REPO_ROOT, DATA, load_wide_indicator

OUT_PANEL = os.path.join(
    REPO_ROOT, "paper", "figures",
    "post_soviet_u5mr_trajectories.png",
)
OUT_OVERLAY = os.path.join(
    REPO_ROOT, "paper", "figures",
    "central_asia_vs_iran_turkey_u5mr.png",
)

GROUPS = [
    ("Baltics (real education)",
     ["latvia", "estonia", "lithuania"],
     ["#2a6fb8", "#3a8fd8", "#5badee"]),
    ("Slavic west",
     ["russian federation", "ukraine", "belarus"],
     ["#c0392b", "#e55d47", "#f08a78"]),
    ("Caucasus",
     ["armenia", "azerbaijan", "georgia"],
     ["#7a3f9e", "#a568c2", "#c596db"]),
    ("Central Asia + Iran/Turkey",
     ["kazakhstan", "kyrgyzstan", "tajikistan", "turkmenistan",
      "uzbekistan", "iran", "turkey"],
     ["#d97706", "#f59e0b", "#fbbf24", "#b45309", "#92400e",
      "#0f766e", "#14b8a6"]),
]

START, END = 1960, 2022


def pretty(n):
    n = n.title()
    return n if len(n) < 22 else n[:20] + "."


def plot_u5mr_panel(ax, u5mr, countries, colors, title):
    years = [c for c in u5mr.columns
             if str(c).isdigit() and START <= int(c) <= END]
    years_sorted = sorted(years, key=int)
    for c, col in zip(countries, colors):
        if c not in u5mr.index:
            continue
        row = u5mr.loc[c, years_sorted]
        ax.plot([int(y) for y in years_sorted], row.values,
                color=col, linewidth=1.8, alpha=0.9, label=pretty(c))
    # 1960 US U5MR reference (~30 per 1,000) — for orientation only
    ax.axhline(30, color="#222222", linewidth=0.8,
               linestyle="--", alpha=0.6)
    ax.text(END + 0.3, 30, " 30 (≈1960 US)",
            va="center", fontsize=8, color="#222")
    ax.set_xlim(START, END)
    ax.set_yscale("log")
    ax.set_ylim(4, 500)
    ax.grid(alpha=0.25, which="both")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="best", frameon=False)


def main():
    u5mr = load_wide_indicator("child_mortality_u5.csv")

    # ── Figure 1: small multiples per group ────────────────────────
    fig, axes = plt.subplots(len(GROUPS), 1, figsize=(9, 2.8 * len(GROUPS)),
                             sharex=True)
    for i, (group_name, countries, colors) in enumerate(GROUPS):
        plot_u5mr_panel(axes[i], u5mr, countries, colors, group_name)
        if i == 0:
            axes[i].set_ylabel("U5MR (per 1,000, log scale)", fontsize=9)
        else:
            axes[i].set_ylabel("U5MR (log)", fontsize=9)
    axes[-1].set_xlabel("Year", fontsize=10)
    fig.suptitle(
        "Under-5 mortality, post-Soviet groups vs non-Soviet neighbors, "
        "1960–2022\n"
        "Reported Soviet lower-sec completion in 1970: "
        "Baltics 90%+ • Russia 95% • Caucasus 95% • "
        "Central Asia 90%+ • Iran/Turkey 22%",
        fontsize=11, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(OUT_PANEL), exist_ok=True)
    fig.savefig(OUT_PANEL, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUT_PANEL}")
    plt.close(fig)

    # ── Figure 2: direct overlay, Central Asia vs Iran/Turkey ─────
    fig, ax = plt.subplots(figsize=(10, 6))
    ca = ["kazakhstan", "kyrgyzstan", "tajikistan",
          "turkmenistan", "uzbekistan"]
    ca_colors = ["#d97706", "#f59e0b", "#fbbf24", "#b45309", "#92400e"]
    neighbors = ["iran", "turkey"]
    n_colors = ["#0f766e", "#14b8a6"]

    years = sorted([int(c) for c in u5mr.columns
                    if str(c).isdigit() and START <= int(c) <= END])
    yr_str = [str(y) for y in years]

    for c, col in zip(ca, ca_colors):
        if c in u5mr.index:
            ax.plot(years, u5mr.loc[c, yr_str].values,
                    color=col, linewidth=2.0, alpha=0.9,
                    label=f"{pretty(c)} (Soviet CA)")
    for c, col in zip(neighbors, n_colors):
        if c in u5mr.index:
            ax.plot(years, u5mr.loc[c, yr_str].values,
                    color=col, linewidth=2.2, alpha=0.95,
                    linestyle="--", label=f"{pretty(c)} (non-Soviet)")

    ax.axhline(30, color="#222", linewidth=0.8,
               linestyle=":", alpha=0.7)
    ax.text(END + 0.3, 30, " 30 (≈1960 US)",
            va="center", fontsize=9, color="#222")
    ax.set_yscale("log")
    ax.set_ylim(4, 400)
    ax.set_xlim(START, END)
    ax.grid(alpha=0.25, which="both")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("U5MR (per 1,000 live births, log scale)", fontsize=11)
    ax.set_title(
        "Under-5 mortality: Soviet Central Asia vs Iran/Turkey, 1960–2022\n"
        "If the Soviet 1970 reported 95% lower-sec completion were real\n"
        "human capital, the solid lines should have dropped far faster than\n"
        "the dashed ones starting around 1970. They did not.",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_OVERLAY, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUT_OVERLAY}")
    plt.close(fig)

    # ── Quantitative summary ──────────────────────────────────────
    print()
    print("U5MR at key dates (per 1,000):")
    rows = []
    for group_name, countries, _ in GROUPS:
        for c in countries:
            if c not in u5mr.index:
                continue
            vals = {}
            for y in [1960, 1970, 1980, 1990, 2000, 2010, 2020]:
                ys = str(y)
                if ys in u5mr.columns:
                    vals[y] = u5mr.loc[c, ys]
            rows.append({"group": group_name, "country": c, **vals})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:5.0f}"))


if __name__ == "__main__":
    main()
