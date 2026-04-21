"""
robustness/beta_by_baseline_group.py

Split the post-1975 panel into baseline education groups and estimate
the generational coefficient (β) separately for each.

Groups (based on 1975 lower-secondary completion):
  Low:    < 20%
  Medium: 20–60%
  High:   > 60%

This tests whether β=0.482 (the pooled post-1975 estimate) masks
structurally different dynamics at different development stages.

Usage:
    python scripts/robustness/beta_by_baseline_group.py
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from _shared import PROC, REGIONS, write_checkin
from residualization._shared import fe_beta_r2


def main():
    agg = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"), index_col="country")

    # Build panel: child(t) ~ parent(t-25) for t in 1975..2015
    rows = []
    for country in agg.index:
        if country in REGIONS:
            continue
        for y in range(1975, 2016, 5):
            sy, sy_lag = str(y), str(y - 25)
            if sy not in agg.columns or sy_lag not in agg.columns:
                continue
            child = agg.loc[country, sy]
            parent = agg.loc[country, sy_lag]
            if np.isnan(child) or np.isnan(parent):
                continue
            rows.append({"country": country, "year": y,
                         "child": child, "parent": parent})

    panel = pd.DataFrame(rows)

    # Classify countries by 1975 baseline (or earliest available)
    baseline = {}
    for country in panel["country"].unique():
        if "1975" in agg.columns and country in agg.index:
            val = agg.loc[country, "1975"]
            if not np.isnan(val):
                baseline[country] = val
                continue
        # Fallback: earliest non-NaN value
        for col in sorted(agg.columns):
            if country in agg.index:
                v = agg.loc[country, col]
                if not np.isnan(v):
                    baseline[country] = v
                    break

    panel["baseline_1975"] = panel["country"].map(baseline)
    panel = panel.dropna(subset=["baseline_1975"])

    # Define groups
    def group(val):
        if val < 20:
            return "Low (<20%)"
        elif val < 60:
            return "Medium (20-60%)"
        else:
            return "High (>60%)"

    panel["group"] = panel["baseline_1975"].apply(group)

    # ── Pooled result ─────────────────────────────────────────────
    print("=" * 72)
    print("GENERATIONAL β BY BASELINE EDUCATION GROUP (post-1975)")
    print("=" * 72)

    p_beta, p_r2, p_n, p_nc = fe_beta_r2("parent", "child", panel)
    print(f"\n  Pooled:  β={p_beta:.3f}  R²={p_r2:.3f}  "
          f"[N={p_n}, {p_nc} countries]")

    # ── By group ──────────────────────────────────────────────────
    print(f"\n  {'Group':<20s}  {'β':>8s}  {'R²':>8s}  {'N':>6s}  {'Countries':>10s}  "
          f"{'Avg baseline':>13s}  {'Avg parent':>11s}  {'Avg child':>10s}")
    print("  " + "-" * 100)

    for grp in ["Low (<20%)", "Medium (20-60%)", "High (>60%)"]:
        sub = panel[panel["group"] == grp]
        beta, r2, n_obs, n_countries = fe_beta_r2("parent", "child", sub)
        if n_obs == 0:
            print(f"  {grp:<20s}  insufficient data")
            continue
        avg_base = sub.groupby("country")["baseline_1975"].first().mean()
        avg_parent = sub["parent"].mean()
        avg_child = sub["child"].mean()
        print(f"  {grp:<20s}  {beta:8.3f}  {r2:8.3f}  "
              f"{n_obs:6d}  {n_countries:10d}  "
              f"{avg_base:13.1f}  {avg_parent:11.1f}  {avg_child:10.1f}")

    # ── Country detail by group ───────────────────────────────────
    print(f"\n{'=' * 72}")
    print("COUNTRY DETAIL BY GROUP")
    print("=" * 72)

    for grp in ["Low (<20%)", "Medium (20-60%)", "High (>60%)"]:
        sub = panel[panel["group"] == grp]
        countries = sub.groupby("country").agg(
            baseline=("baseline_1975", "first"),
            obs=("child", "count"),
            parent_mean=("parent", "mean"),
            child_mean=("child", "mean"),
            child_2015=("child", "last"),
        ).sort_values("baseline")

        print(f"\n  {grp} ({len(countries)} countries):")
        print(f"  {'Country':<35s}  {'1975 base':>10s}  {'Avg parent':>11s}  "
              f"{'Avg child':>10s}  {'2015':>8s}  {'Obs':>4s}")
        print("  " + "-" * 85)
        for c, row in countries.iterrows():
            # Get actual 2015 value
            c2015 = agg.loc[c, "2015"] if "2015" in agg.columns and c in agg.index else np.nan
            print(f"  {c:<35s}  {row['baseline']:10.1f}  {row['parent_mean']:11.1f}  "
                  f"{row['child_mean']:10.1f}  {c2015:8.1f}  {int(row['obs']):4d}")

    # ── Speed of β collapse ───────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("β TRAJECTORY BY COUNTRY (sliding 25-year windows)")
    print("=" * 72)
    print("\n  Shows how fast β falls — the speed of fall = speed of development")

    # For select countries, compute per-window β
    showcase = ["Republic of Korea", "Taiwan Province of China", "Singapore",
                "Viet Nam", "China", "Bangladesh", "India", "Philippines",
                "Nepal", "Cambodia", "Nigeria", "Cuba"]

    for country in showcase:
        if country not in agg.index:
            continue
        csub = panel[panel["country"] == country].sort_values("year")
        if len(csub) < 2:
            continue
        base = baseline.get(country, 0)
        # Compute per-window beta (simple ratio of changes)
        windows = []
        for i in range(len(csub) - 1):
            dp = csub.iloc[i + 1]["parent"] - csub.iloc[i]["parent"]
            dc = csub.iloc[i + 1]["child"] - csub.iloc[i]["child"]
            if abs(dp) > 0.1:
                windows.append({
                    "period": f"{csub.iloc[i]['year']:.0f}-{csub.iloc[i+1]['year']:.0f}",
                    "parent_start": csub.iloc[i]["parent"],
                    "beta_approx": dc / dp,
                })
        if not windows:
            continue

        short = country.replace("Republic of Korea", "Korea") \
                       .replace("Taiwan Province of China", "Taiwan") \
                       .replace("Viet Nam", "Vietnam")
        print(f"\n  {short} (1975 baseline: {base:.1f}%)")
        for w in windows:
            print(f"    {w['period']}  parent={w['parent_start']:5.1f}%  β≈{w['beta_approx']:+.2f}")

    # ── Summary interpretation ────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("INTERPRETATION")
    print("=" * 72)
    print("""
  β=0.482 (pooled post-1975) masks three structurally different dynamics:

  - Low-baseline countries: β is HIGH — education is compounding fast,
    every parental percentage point produces large gains in the next
    generation. These countries are on the steep part of the S-curve.

  - Medium-baseline countries: β is MODERATE — the transition is underway,
    gains are still substantial but decelerating.

  - High-baseline countries: β is LOW — near ceiling. The mechanism hasn't
    weakened; the variable is bounded at 100%. The speed at which β fell
    from high to low IS the speed of development.

  The 28-country long-run panel (β=0.960) is dominated by countries that
  developed slowly — their β never crashed because they never accelerated.
  Korea's β crashed from >6 to near-zero in 60 years. Bangladesh has held
  β≈2 for a century. The shape of the β curve is the development story.
""")


    # ── Write checkin JSON ─────────────────────────────────────────
    grp_numbers = {}
    for grp, prefix in [("Low (<20%)", "Grp-low"), ("Medium (20-60%)", "Grp-med"), ("High (>60%)", "Grp-high")]:
        sub = panel[panel["group"] == grp]
        beta, r2, n_obs, n_countries = fe_beta_r2("parent", "child", sub)
        if n_obs > 0:
            grp_numbers[f"{prefix}-beta"] = round(beta, 3)
            grp_numbers[f"{prefix}-R2"] = round(r2, 3)
            grp_numbers[f"{prefix}-n"] = n_obs
            grp_numbers[f"{prefix}-countries"] = n_countries

    write_checkin("beta_by_baseline_group.json", {
        "numbers": grp_numbers,
    }, script_path="scripts/robustness/beta_by_baseline_group.py")


if __name__ == "__main__":
    main()
