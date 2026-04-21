#!/usr/bin/env python3
"""
cases/china_mean_years.py
==========================
Redraw the China vs peers comparison using mean years of schooling
(entire 20-24 population) instead of lower secondary completion rate.

Mean years captures the full distribution — including China's massive
primary-educated base that the lower-sec threshold misses.

WCDE education categories → years of schooling:
  No Education:        0
  Incomplete Primary:  3
  Primary:             6
  Lower Secondary:     9
  Upper Secondary:    12
  Post Secondary:     15
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _shared import REPO_ROOT, DATA, PROC, NAME_MAP, REGIONS, load_wb, write_checkin

WCDE_RAW = os.path.join(REPO_ROOT, "wcde", "data", "raw")

# Years of schooling assigned to each WCDE education level
YEARS_MAP = {
    "No Education": 0,
    "Incomplete Primary": 3,
    "Primary": 6,
    "Lower Secondary": 9,
    "Upper Secondary": 12,
    "Post Secondary": 15,
}

BAND = 0.5  # ±years for peer matching


def wb_name(wcde):
    return NAME_MAP.get(wcde, wcde).lower()


def compute_mean_years():
    """
    Compute mean years of schooling for age 20-24, both sexes,
    for every country-year in WCDE.
    Returns DataFrame: country, year, mean_yrs.
    """
    prop = pd.read_csv(os.path.join(WCDE_RAW, "prop_both.csv"))
    # Filter: age 20-24, both sexes, scenario 2
    mask = (
        (prop["age"] == "20--24") &
        (prop["sex"] == "Both") &
        (prop["scenario"] == 2)
    )
    sub = prop[mask].copy()

    # Map education level to years
    sub["yrs"] = sub["education"].map(YEARS_MAP)
    # Drop unmapped (shouldn't happen with our 6 categories)
    sub = sub.dropna(subset=["yrs"])

    # Weighted average: mean_yrs = sum(prop * yrs) / 100
    sub["weighted"] = sub["prop"] * sub["yrs"]
    grouped = sub.groupby(["name", "year"]).agg(
        mean_yrs=("weighted", lambda x: x.sum() / 100),
        total_prop=("prop", "sum"),
    ).reset_index()

    # Filter to countries (not aggregates) with valid totals
    grouped = grouped[
        (~grouped["name"].isin(REGIONS)) &
        (grouped["total_prop"] > 95)  # should be ~100
    ]
    return grouped[["name", "year", "mean_yrs"]].rename(columns={"name": "country"})


def interpolate_mean_years(mys):
    """Interpolate 5-year mean years to annual."""
    records = {}
    for c, grp in mys.groupby("country"):
        s = grp.set_index("year")["mean_yrs"].sort_index()
        s = s[(s.index >= 1950) & (s.index <= 2015)]
        if len(s) < 2:
            continue
        full = pd.Series(dtype=float, index=range(s.index.min(), s.index.max() + 1))
        full.update(s)
        records[c] = full.interpolate(method="linear")
    return records


def main():
    print("Computing mean years of schooling from WCDE proportions...")
    mys = compute_mean_years()
    print(f"  {len(mys)} country-year observations")

    # Show China's trajectory
    china_mys = mys[mys["country"] == "China"].sort_values("year")
    print("\nChina mean years of schooling (age 20-24):")
    for _, r in china_mys.iterrows():
        if r["year"] <= 2015:
            print(f"  {int(r['year'])}: {r['mean_yrs']:.2f} years")

    # Interpolate to annual
    print("\nInterpolating to annual...")
    annual = interpolate_mean_years(mys)

    # Load WB data
    le_wb = load_wb("life_expectancy_years.csv")
    u5_wb = load_wb("child_mortality_u5.csv")

    # Build annual panel
    print("Building panel...")
    rows = []
    for c, mys_s in annual.items():
        wbn = wb_name(c)
        for yr in range(1960, 2016):
            if yr not in mys_s.index:
                continue
            my = mys_s[yr]
            if pd.isna(my):
                continue

            le_val = np.nan
            u5_val = np.nan
            y_str = str(yr)

            if wbn in le_wb.index and y_str in le_wb.columns:
                le_val = le_wb.at[wbn, y_str]
            if wbn in u5_wb.index and y_str in u5_wb.columns:
                u5_val = u5_wb.at[wbn, y_str]

            if pd.notna(le_val) or pd.notna(u5_val):
                rows.append({
                    "country": c, "year": yr, "mean_yrs": my,
                    "le": le_val, "u5mr": u5_val,
                })

    panel = pd.DataFrame(rows)
    print(f"  Panel: {len(panel):,} obs, {panel['country'].nunique()} countries")

    # China trajectory
    china = panel[panel["country"] == "China"].sort_values("year")

    # For each year, find peer mean LE and U5MR (±0.5 years of schooling)
    print(f"Computing peer expectations (±{BAND} years of schooling)...")
    peer_rows = []
    for _, cr in china.iterrows():
        yr = cr["year"]
        my = cr["mean_yrs"]
        peers = panel[
            (abs(panel["mean_yrs"] - my) <= BAND) &
            (panel["country"] != "China")
        ]

        p_le = peers["le"].dropna()
        p_u5 = peers["u5mr"].dropna()

        peer_rows.append({
            "year": yr,
            "mean_yrs": round(my, 2),
            "china_le": cr["le"],
            "china_u5": cr["u5mr"],
            "peer_le_mean": p_le.mean() if len(p_le) > 0 else np.nan,
            "peer_le_p25": p_le.quantile(0.25) if len(p_le) > 0 else np.nan,
            "peer_le_p75": p_le.quantile(0.75) if len(p_le) > 0 else np.nan,
            "peer_u5_mean": p_u5.mean() if len(p_u5) > 0 else np.nan,
            "peer_u5_p25": p_u5.quantile(0.25) if len(p_u5) > 0 else np.nan,
            "peer_u5_p75": p_u5.quantile(0.75) if len(p_u5) > 0 else np.nan,
            "n_peers_le": len(p_le),
            "n_peers_u5": len(p_u5),
        })

    result = pd.DataFrame(peer_rows)

    # ── Print table ──
    print()
    print("CHINA vs EDUCATION-MATCHED PEERS (MEAN YEARS OF SCHOOLING)")
    print(f"Peers = all country-years within ±{BAND} mean years, any calendar year")
    print("=" * 105)
    print(f"{'Year':>5} {'MYS':>6} │ {'China_LE':>9} {'Peer_LE':>8} {'Δ_LE':>7} │ "
          f"{'China_U5':>9} {'Peer_U5':>8} {'Δ_U5':>7} │ {'N':>5}")
    print("─" * 105)

    for _, r in result.iterrows():
        yr = int(r["year"])
        if yr % 5 != 0:
            continue
        d_le = r["china_le"] - r["peer_le_mean"] if pd.notna(r["china_le"]) and pd.notna(r["peer_le_mean"]) else np.nan
        d_u5 = r["china_u5"] - r["peer_u5_mean"] if pd.notna(r["china_u5"]) and pd.notna(r["peer_u5_mean"]) else np.nan

        def f(v, fmt=".1f"):
            return f"{v:{fmt}}" if pd.notna(v) else "n/a"

        print(f"{yr:>5} {r['mean_yrs']:>6.2f} │ {f(r['china_le']):>9} {f(r['peer_le_mean']):>8} "
              f"{f(d_le, '+.1f'):>7} │ {f(r['china_u5']):>9} {f(r['peer_u5_mean']):>8} "
              f"{f(d_u5, '+.1f'):>7} │ {r['n_peers_le']:>5}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Life expectancy
    ax = axes[0]
    valid_le = result.dropna(subset=["china_le", "peer_le_mean"])
    ax.fill_between(valid_le["year"], valid_le["peer_le_p25"], valid_le["peer_le_p75"],
                    alpha=0.2, color="steelblue", label="Peer IQR")
    ax.plot(valid_le["year"], valid_le["peer_le_mean"], color="steelblue",
            linewidth=2, label=f"Peer mean (same MYS ±{BAND}yr)")
    ax.plot(valid_le["year"], valid_le["china_le"], color="red",
            linewidth=2.5, label="China")
    ax.set_ylabel("Life expectancy (years)", fontsize=12)
    ax.set_title("China vs education-matched peers: Life expectancy\n"
                 "(matched on mean years of schooling, age 20-24)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add mean years annotation on right axis
    ax2 = ax.twinx()
    ax2.plot(valid_le["year"], valid_le["mean_yrs"], color="gray",
             linewidth=1, linestyle="--", alpha=0.5)
    ax2.set_ylabel("China mean years of schooling", fontsize=10, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # U5MR
    ax = axes[1]
    valid_u5 = result.dropna(subset=["china_u5", "peer_u5_mean"])
    ax.fill_between(valid_u5["year"], valid_u5["peer_u5_p25"], valid_u5["peer_u5_p75"],
                    alpha=0.2, color="steelblue", label="Peer IQR")
    ax.plot(valid_u5["year"], valid_u5["peer_u5_mean"], color="steelblue",
            linewidth=2, label=f"Peer mean (same MYS ±{BAND}yr)")
    ax.plot(valid_u5["year"], valid_u5["china_u5"], color="red",
            linewidth=2.5, label="China")
    ax.set_ylabel("Under-5 mortality (per 1,000)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_title("China vs education-matched peers: Under-5 mortality\n"
                 "(matched on mean years of schooling, age 20-24)", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(REPO_ROOT, "figures", "china_mys_vs_peers.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")
    plt.close()

    # ── Structural break test: barefoot doctor removal (1981) ──
    break_results = structural_break_test(result)

    # ── Save checkin JSON ──
    save_checkin(result, break_results, panel)


# ================================================================
# STRUCTURAL BREAK TEST
# ================================================================

BREAK_YEAR = 1981  # barefoot doctors dismantled with decollectivization


def structural_break_test(result):
    """
    Test whether barefoot doctor removal (1981) changed China's LE or U5MR
    trajectory relative to education-matched peers.

    Model: gap = α + β₁·t + β₂·POST + β₃·POST×t + ε
    If β₂ or β₃ significant → removal had an effect.
    """
    print(f"\n{'=' * 75}")
    print("STRUCTURAL BREAK TEST: Barefoot doctor removal (1981)")
    print(f"{'=' * 75}")

    # Compute gap series (1965-2000)
    gaps = result[(result["year"] >= 1965) & (result["year"] <= 2000)].copy()
    gaps["le_gap"] = gaps["china_le"] - gaps["peer_le_mean"]
    gaps["u5_gap"] = gaps["china_u5"] - gaps["peer_u5_mean"]

    all_results = {}

    for label, col in [("LE gap (China - peers)", "le_gap"),
                        ("U5MR gap (China - peers)", "u5_gap")]:
        d = gaps.dropna(subset=[col]).copy()
        d["t"] = d["year"] - 1965
        d["post"] = (d["year"] >= BREAK_YEAR).astype(int)
        d["post_t"] = d["post"] * (d["year"] - BREAK_YEAR)

        # Full model: gap = α + β₁·t + β₂·post + β₃·post_t
        X = np.column_stack([
            np.ones(len(d)),
            d["t"].values,
            d["post"].values,
            d["post_t"].values,
        ])
        y = d[col].values

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        n, k = len(y), X.shape[1]
        se = np.sqrt(np.diag(
            np.linalg.inv(X.T @ X) * (resid @ resid / (n - k))
        ))
        t_stats = beta / se
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
        r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))

        pre_slope = beta[1]
        post_slope = beta[1] + beta[3]

        print(f"\n{'─' * 75}")
        print(f"{label}")
        print(f"{'─' * 75}")
        print(f"Model: gap = α + β₁·t + β₂·POST_{BREAK_YEAR} + β₃·POST_{BREAK_YEAR}×t")
        print(f"n = {n}, R² = {r2:.4f}")
        print(f"\n{'Parameter':<20} {'β':>10} {'SE':>10} {'t':>10} {'p':>10}")
        print(f"{'-' * 60}")
        names = ["Intercept (α)", "Trend (β₁)",
                 "Break level (β₂)", "Break slope (β₃)"]
        for nm, b, s, t, p in zip(names, beta, se, t_stats, p_vals):
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else
                  ("*" if p < 0.05 else ""))
            print(f"{nm:<20} {b:>10.4f} {s:>10.4f} {t:>10.3f} {p:>10.4f} {sig}")

        print(f"\nPre-{BREAK_YEAR} trend:  {pre_slope:+.3f}/year")
        print(f"Post-{BREAK_YEAR} trend: {post_slope:+.3f}/year")
        print(f"Slope change:      {beta[3]:+.3f}/year (p = {p_vals[3]:.4f})")
        print(f"Level shift:       {beta[2]:+.3f} (p = {p_vals[2]:.4f})")

        # Chow test
        X_all = np.column_stack([np.ones(len(d)), d["t"].values])
        beta_all = np.linalg.lstsq(X_all, y, rcond=None)[0]
        rss_all = np.sum((y - X_all @ beta_all) ** 2)

        pre = d[d["year"] < BREAK_YEAR]
        post = d[d["year"] >= BREAK_YEAR]
        X_pre = np.column_stack([np.ones(len(pre)), pre["t"].values])
        X_post = np.column_stack([np.ones(len(post)), post["t"].values])
        y_pre, y_post = pre[col].values, post[col].values
        rss_pre = np.sum((y_pre - X_pre @ np.linalg.lstsq(X_pre, y_pre, rcond=None)[0]) ** 2)
        rss_post = np.sum((y_post - X_post @ np.linalg.lstsq(X_post, y_post, rcond=None)[0]) ** 2)

        F_chow = ((rss_all - rss_pre - rss_post) / 2) / ((rss_pre + rss_post) / (n - 4))
        p_chow = 1 - stats.f.cdf(F_chow, 2, n - 4)

        print(f"\nChow test: F = {F_chow:.3f}, p = {p_chow:.4f}")

        outcome = "le" if "LE" in label else "u5mr"
        all_results[outcome] = {
            "pre_slope": round(pre_slope, 4),
            "post_slope": round(post_slope, 4),
            "beta_break_slope": round(beta[3], 4),
            "se_break_slope": round(se[3], 4),
            "p_break_slope": round(p_vals[3], 4),
            "beta_break_level": round(beta[2], 4),
            "p_break_level": round(p_vals[2], 4),
            "r2": round(r2, 4),
            "n": n,
            "chow_F": round(F_chow, 3),
            "chow_p": round(p_chow, 4),
        }

    return all_results


# ================================================================
# CHECKIN JSON
# ================================================================

def compute_peer_le_gains(panel):
    """
    Compute two supplementary verification outputs:

    1. peer_le_gains: LE gain from T=0 to T+20 for each target country,
       using the provision-discontinuity framework (countries anchored
       to the year they crossed 30% lower secondary completion).
       Uses WCDE e0.csv (life expectancy) and lower_sec_both.csv.

    2. n_provision_countries: number of countries that crossed 45%
       lower secondary completion in the provision discontinuity
       analysis, with sufficient LE data (≥4 offsets with data).

    Returns (peer_le_gains dict, n_provision_countries int).
    """
    # Load WCDE education and LE data
    edu_wide = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"))
    edu_wide = edu_wide[~edu_wide["country"].isin(REGIONS)].copy()
    e0_wide = pd.read_csv(os.path.join(PROC, "e0.csv"))
    e0_wide = e0_wide[~e0_wide["country"].isin(REGIONS)].copy()

    # Lowercase for consistent matching
    edu_wide["country"] = edu_wide["country"].str.lower()
    e0_wide["country"] = e0_wide["country"].str.lower()
    edu_wide = edu_wide.set_index("country")
    e0_wide = e0_wide.set_index("country")

    common = edu_wide.index.intersection(e0_wide.index)
    edu_wide = edu_wide.loc[common]
    e0_wide = e0_wide.loc[common]

    years_5yr = [str(y) for y in range(1950, 2020, 5)]
    edu_num = edu_wide[years_5yr].apply(pd.to_numeric, errors="coerce")
    e0_num = e0_wide[[y for y in years_5yr if y in e0_wide.columns]].apply(
        pd.to_numeric, errors="coerce"
    )

    def find_crossing_year(row, threshold):
        for i in range(len(years_5yr) - 1):
            v1, v2 = row[years_5yr[i]], row[years_5yr[i + 1]]
            if pd.isna(v1) or pd.isna(v2):
                continue
            if v1 < threshold <= v2:
                frac = (threshold - v1) / (v2 - v1)
                return int(years_5yr[i]) + frac * 5
        return None

    def le_at_year(country, yr):
        """Interpolate LE at a fractional year from 5-year e0 data."""
        if country not in e0_num.index:
            return None
        e0_years = [int(y) for y in years_5yr if y in e0_num.columns]
        for i in range(len(e0_years) - 1):
            y1, y2 = e0_years[i], e0_years[i + 1]
            if y1 <= yr < y2:
                frac = (yr - y1) / (y2 - y1)
                v1 = e0_num.loc[country, str(y1)]
                v2 = e0_num.loc[country, str(y2)]
                if pd.notna(v1) and pd.notna(v2):
                    return float(v1 + frac * (v2 - v1))
        # Check exact last year
        last_yr = e0_years[-1]
        if abs(yr - last_yr) < 0.01:
            v = e0_num.loc[country, str(last_yr)]
            return float(v) if pd.notna(v) else None
        return None

    # ── 1. Peer LE gains at 30% threshold ──
    # (All four target countries cross 30% in the data window)
    threshold_peers = 30
    crossings_30 = {}
    for country in edu_num.index:
        yr = find_crossing_year(edu_num.loc[country], threshold_peers)
        if yr is not None and 1950 <= yr <= 2010:
            crossings_30[country] = yr

    target_map = {
        "Philippines": "philippines",
        "Peru": "peru",
        "Panama": "panama",
        "Viet Nam": "viet nam",
    }

    peer_gains = {}
    offsets = [-10, -5, 0, 5, 10, 15, 20]
    for display, lc in target_map.items():
        if lc not in crossings_30:
            peer_gains[display] = None
            continue
        cross_yr = crossings_30[lc]
        # Collect LE at each offset
        le_data = {}
        for offset in offsets:
            target_yr = cross_yr + offset
            if 1950 <= target_yr <= 2015:
                val = le_at_year(lc, target_yr)
                if val is not None:
                    le_data[offset] = val
        if 0 in le_data and 20 in le_data:
            peer_gains[display] = round(le_data[20] - le_data[0], 1)
        elif 0 in le_data and 15 in le_data:
            peer_gains[display] = round(le_data[15] - le_data[0], 1)
        else:
            peer_gains[display] = None

    # ── 2. Provision discontinuity country count at 45% ──
    threshold_prov = 45
    crossings_45 = {}
    for country in edu_num.index:
        yr = find_crossing_year(edu_num.loc[country], threshold_prov)
        if yr is not None and 1950 <= yr <= 2010:
            crossings_45[country] = yr

    # Filter to countries with sufficient LE data (≥4 offsets)
    n_provision = 0
    for country, cross_yr in crossings_45.items():
        if country == "china":
            continue
        n_offsets = 0
        for offset in offsets:
            target_yr = cross_yr + offset
            if 1950 <= target_yr <= 2015:
                val = le_at_year(country, target_yr)
                if val is not None:
                    n_offsets += 1
        if n_offsets >= 4:
            n_provision += 1

    return peer_gains, n_provision


def save_checkin(result, break_results, panel):
    """Save results to checkin JSON for number traceability."""
    # Key data points for the paper
    r65 = result[result["year"] == 1965].iloc[0]
    r80 = result[result["year"] == 1980].iloc[0]
    r91 = result[result["year"] == 1991].iloc[0]
    r92 = result[result["year"] == 1992].iloc[0]
    r00 = result[result["year"] == 2000].iloc[0]
    r15 = result[result["year"] == 2015].iloc[0]

    def gap_le(r):
        return round(r["china_le"] - r["peer_le_mean"], 1)

    def gap_u5(r):
        if pd.notna(r["china_u5"]) and pd.notna(r["peer_u5_mean"]):
            return round(r["china_u5"] - r["peer_u5_mean"], 1)
        return None

    # Compute peer LE gains and provision country count
    peer_le_gains, n_provision = compute_peer_le_gains(panel)

    checkin_data = {
        "method": (
            "China LE and U5MR vs education-matched peers. "
            "Matching on mean years of schooling (WCDE v3 proportions, "
            "age 20-24, both sexes, ±0.5 years). "
            "LE and U5MR from World Bank WDI (annual). "
            "Structural break test at 1981 (barefoot doctor removal)."
        ),
        "band": BAND,
        "key_data_points": {
            "china_mys_1965": r65["mean_yrs"],
            "china_mys_1980": r80["mean_yrs"],
            "china_mys_2000": r00["mean_yrs"],
            "china_le_1965": r65["china_le"],
            "peer_le_1965": round(r65["peer_le_mean"], 1),
            "le_gap_1965": gap_le(r65),
            "le_gap_1980": gap_le(r80),
            "le_gap_1991": gap_le(r91),
            "le_gap_1992": gap_le(r92),
            "le_gap_2000": gap_le(r00),
            "le_gap_2015": gap_le(r15),
            "u5_gap_1965": gap_u5(r65),
            "u5_gap_1980": gap_u5(r80),
            "u5_gap_2000": gap_u5(r00),
            "u5_gap_2015": gap_u5(r15),
            "le_crossover_year": int(
                result[(result["year"] >= 1965) &
                       (result["china_le"] >= result["peer_le_mean"])]
                ["year"].min()
            ),
        },
        "structural_break_1981": break_results,
        "peer_le_gains": peer_le_gains,
        "n_provision_countries": n_provision,
        "annual_data": [
            {
                "year": int(r["year"]),
                "mean_yrs": round(r["mean_yrs"], 2),
                "china_le": round(r["china_le"], 1) if pd.notna(r["china_le"]) else None,
                "peer_le": round(r["peer_le_mean"], 1) if pd.notna(r["peer_le_mean"]) else None,
                "china_u5": round(r["china_u5"], 1) if pd.notna(r["china_u5"]) else None,
                "peer_u5": round(r["peer_u5_mean"], 1) if pd.notna(r["peer_u5_mean"]) else None,
            }
            for _, r in result[
                (result["year"] >= 1960) & (result["year"] <= 2015)
            ].iterrows()
        ],
    }

    write_checkin("china_mean_yrs_vs_peers.json", checkin_data,
                  script_path="scripts/cases/china_mean_years.py")


if __name__ == "__main__":
    main()
