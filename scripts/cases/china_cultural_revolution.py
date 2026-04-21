"""
cases/china_cultural_revolution.py

Verify China Cultural Revolution cohort gains from WCDE data.

Expected numbers:
  - CR cohort gains: +10.6pp for 1975 cohort, +15.0pp for 1980 cohort
    (5-year gain in lower_sec completion)
  - China education trajectory: 30.9% (1965), 62.0% (1980), 75.0% (1990)
  - China provision discontinuity: countries at ~30% completion in 1965
    had LE 54-65 years, peer average 57.4; China's was ~53
  - China LE gain: from ~53 to 64 by 1980

Usage:
    python scripts/cases/china_cultural_revolution.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _shared import PROC, DATA, REGIONS, write_checkin

COMP_PATH = os.path.join(PROC, "completion_both_long.csv")
E0_PATH = os.path.join(PROC, "e0.csv")
WDI_LE_PATH = os.path.join(DATA, "life_expectancy_years.csv")

CHINA_WCDE = "China"


def main():
    comp = pd.read_csv(COMP_PATH)
    e0 = pd.read_csv(E0_PATH)

    # ── 1. China lower-secondary completion trajectory ───────────────────
    china_comp = comp[comp["country"] == CHINA_WCDE].set_index("year")

    print("=" * 70)
    print("CHINA CULTURAL REVOLUTION COHORT VERIFICATION")
    print("=" * 70)

    print("\n--- China lower_sec completion trajectory ---")
    trajectory_years = [1950, 1955, 1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000]
    for y in trajectory_years:
        if y in china_comp.index:
            val = china_comp.loc[y, "lower_sec"]
            print(f"  {y}: {val:.1f}%")

    # Expected trajectory values
    expected_trajectory = {1965: 30.9, 1980: 62.0, 1990: 75.0}
    results = {"trajectory": {}}
    traj_pass = True

    print("\n--- Trajectory check ---")
    for year, expected in expected_trajectory.items():
        actual = float(china_comp.loc[year, "lower_sec"])
        diff = abs(actual - expected)
        status = "PASS" if diff <= 1.0 else "MISMATCH"
        if status == "MISMATCH":
            traj_pass = False
        print(f"  {year}: expected={expected:.1f}, actual={actual:.1f}, diff={diff:.1f} -> {status}")
        results["trajectory"][str(year)] = {
            "expected": expected,
            "actual": round(actual, 2),
            "diff": round(diff, 2),
            "status": status,
        }

    # ── 2. CR cohort gains (5-year increments in lower_sec) ──────────────
    print("\n--- CR cohort gains (5-year delta in lower_sec) ---")
    cohort_gains = {}
    gain_years = [1955, 1960, 1965, 1970, 1975, 1980, 1985, 1990]
    for y in gain_years:
        prev = y - 5
        if y in china_comp.index and prev in china_comp.index:
            gain = float(china_comp.loc[y, "lower_sec"] - china_comp.loc[prev, "lower_sec"])
            cohort_gains[y] = gain
            print(f"  {prev}->{y}: +{gain:.1f}pp")

    # Expected CR gains
    expected_gains = {1975: 10.6, 1980: 15.0}
    gains_pass = True
    results["cohort_gains"] = {}

    print("\n--- CR cohort gain check ---")
    for year, expected in expected_gains.items():
        actual = cohort_gains.get(year)
        if actual is not None:
            diff = abs(actual - expected)
            status = "PASS" if diff <= 1.5 else "MISMATCH"
            if status == "MISMATCH":
                gains_pass = False
            print(f"  {year} cohort: expected=+{expected:.1f}pp, actual=+{actual:.1f}pp, diff={diff:.1f} -> {status}")
            results["cohort_gains"][str(year)] = {
                "expected": expected,
                "actual": round(actual, 2),
                "diff": round(diff, 2),
                "status": status,
            }
        else:
            print(f"  {year} cohort: data not available")
            results["cohort_gains"][str(year)] = {"status": "MISSING"}
            gains_pass = False

    # ── 3. Provision discontinuity: peers at ~30% in 1965 ───────────────
    print("\n--- Provision discontinuity (1965, lower_sec 25-38%) ---")
    comp_1965 = comp[comp["year"] == 1965][["country", "lower_sec", "primary"]].copy()
    # Filter to countries with lower_sec between 25% and 38% in 1965
    peers_1965 = comp_1965[
        (comp_1965["lower_sec"] >= 25) & (comp_1965["lower_sec"] <= 38)
    ].copy()

    # Exclude aggregates, non-sovereign territories, oil city-states, primary outliers
    exclude = REGIONS | {
        # Non-sovereign territories and tiny islands
        "Occupied Palestinian Territory", "Reunion", "Aruba",
        "Macao Special Administrative Region of China",
        "Hong Kong Special Administrative Region of China",
        "Taiwan Province of China",
        "Kiribati", "Samoa", "Fiji", "Belize",
        # Oil city-states
        "Qatar", "Kuwait", "Bahrain", "Saudi Arabia",
        # Primary outlier (89% primary, LE driven by near-universal primary)
        "Greece",
    }
    peers_1965 = peers_1965[~peers_1965["country"].isin(exclude)]

    # Get LE for 1965 from WCDE e0
    e0_wide = e0.set_index("country")
    if 1965 in e0_wide.columns:
        e0_col = 1965
    else:
        e0_wide.columns = [int(c) if str(c).isdigit() else c for c in e0_wide.columns]
        e0_col = 1965

    peers_1965["le"] = peers_1965["country"].map(
        lambda c: float(e0_wide.loc[c, e0_col]) if c in e0_wide.index else np.nan
    )
    peers_1965 = peers_1965.dropna(subset=["le"])

    print(f"  Countries at 25-38% lower_sec in 1965: {len(peers_1965)}")
    for _, row in peers_1965.sort_values("le").iterrows():
        marker = " <-- CHINA" if row["country"] == CHINA_WCDE else ""
        print(f"    {row['country']:40s}  lower_sec={row['lower_sec']:.1f}%  LE={row['le']:.1f}{marker}")

    # China LE in 1965
    china_le_1965 = float(e0_wide.loc[CHINA_WCDE, e0_col]) if CHINA_WCDE in e0_wide.index else None
    # Peer average excluding China
    peer_no_china = peers_1965[peers_1965["country"] != CHINA_WCDE]
    peer_avg_le = float(peer_no_china["le"].mean()) if len(peer_no_china) > 0 else None
    peer_le_range = (float(peer_no_china["le"].min()), float(peer_no_china["le"].max())) if len(peer_no_china) > 0 else None

    print(f"\n  China LE 1965: {china_le_1965:.1f}" if china_le_1965 else "  China LE 1965: N/A")
    print(f"  Peer avg LE (excl China): {peer_avg_le:.1f}" if peer_avg_le else "  Peer avg: N/A")
    if peer_le_range:
        print(f"  Peer LE range: {peer_le_range[0]:.1f} - {peer_le_range[1]:.1f}")

    results["provision_discontinuity"] = {
        "china_le_1965": round(china_le_1965, 2) if china_le_1965 else None,
        "peer_avg_le": round(peer_avg_le, 2) if peer_avg_le else None,
        "peer_le_range": [round(x, 2) for x in peer_le_range] if peer_le_range else None,
        "n_peers": len(peers_1965),
        "expected_china_le": 54.4,
        "expected_peer_avg": 60,
        "expected_peer_range": [48, 69],
    }

    disc_pass = True
    if china_le_1965 is not None:
        if abs(china_le_1965 - 54.4) > 1:
            disc_pass = False
            print(f"  NOTE: China LE 1965 = {china_le_1965:.1f}, expected ~54.4")
    if peer_avg_le is not None:
        if abs(peer_avg_le - 60) > 2:
            disc_pass = False
            print(f"  NOTE: Peer avg LE = {peer_avg_le:.1f}, expected ~60")

    # ── 4. China LE gain: ~53 to ~64 by 1980 ────────────────────────────
    print("\n--- China LE gain ---")
    china_le_1980 = float(e0_wide.loc[CHINA_WCDE, 1980]) if CHINA_WCDE in e0_wide.index else None
    if china_le_1965 and china_le_1980:
        le_gain = china_le_1980 - china_le_1965
        print(f"  1965: {china_le_1965:.1f}, 1980: {china_le_1980:.1f}, gain: {le_gain:.1f} years")
        le_pass = abs(china_le_1980 - 64) <= 3 and abs(china_le_1965 - 53) <= 3
        if not le_pass:
            print(f"  NOTE: expected ~53 -> ~64")
    else:
        le_gain = None
        le_pass = False
        print("  LE data not available for China")

    results["le_gain"] = {
        "le_1965": round(china_le_1965, 2) if china_le_1965 else None,
        "le_1980": round(china_le_1980, 2) if china_le_1980 else None,
        "gain": round(le_gain, 2) if le_gain else None,
        "expected_1965": 53,
        "expected_1980": 64,
    }

    # ── Summary ──────────────────────────────────────────────────────────
    all_pass = traj_pass and gains_pass and disc_pass and le_pass
    overall = "PASS" if all_pass else "SOME_NOTES"

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Trajectory:     {'PASS' if traj_pass else 'CHECK'}")
    print(f"  Cohort gains:   {'PASS' if gains_pass else 'CHECK'}")
    print(f"  Discontinuity:  {'PASS' if disc_pass else 'CHECK'}")
    print(f"  LE gain:        {'PASS' if le_pass else 'CHECK'}")
    print(f"  Overall: {overall}")

    # ── Write checkin JSON ───────────────────────────────────────────────
    write_checkin("china_cr.json", {
        "results": results,
        "overall": overall,
    }, script_path="scripts/cases/china_cultural_revolution.py")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
