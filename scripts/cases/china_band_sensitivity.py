#!/usr/bin/env python3
"""
cases/china_band_sensitivity.py
===============================
Peer-pool bandwidth sensitivity test for the China-vs-peers mean-years
comparison used to rebut Drèze & Sen's support-led-security thesis.

Main spec uses ±0.5 mean years of schooling (china_mean_years.py). This
script reruns the comparison at ±0.25 and ±1.0 to show the result holds
across band choices — specifically that China's LE was below and U5MR
was above education-matched peers before ~1991/2000.

Output: JSON checkin with China's LE and U5MR gap vs peers at the four
key years cited in the paper (1965, 1980, 1991, 2000) for each band.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _shared import REGIONS, load_wb, write_checkin  # noqa: E402

WCDE_RAW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "wcde", "data", "raw",
)

YEARS_MAP = {
    "No Education": 0,
    "Incomplete Primary": 3,
    "Primary": 6,
    "Lower Secondary": 9,
    "Upper Secondary": 12,
    "Post Secondary": 15,
}

BANDS = [0.25, 0.5, 1.0]
KEY_YEARS = [1965, 1980, 1991, 2000]

# WCDE → World Bank name map for China (only country we query)
WB_CHINA = "china"


def compute_mean_years():
    prop = pd.read_csv(os.path.join(WCDE_RAW, "prop_both.csv"))
    mask = (
        (prop["age"] == "20--24")
        & (prop["sex"] == "Both")
        & (prop["scenario"] == 2)
    )
    sub = prop[mask].copy()
    sub["yrs"] = sub["education"].map(YEARS_MAP)
    sub = sub.dropna(subset=["yrs"])
    sub["weighted"] = sub["prop"] * sub["yrs"]
    grouped = sub.groupby(["name", "year"]).agg(
        mean_yrs=("weighted", lambda x: x.sum() / 100),
        total_prop=("prop", "sum"),
    ).reset_index()
    grouped = grouped[
        (~grouped["name"].isin(REGIONS)) & (grouped["total_prop"] > 95)
    ]
    return grouped[["name", "year", "mean_yrs"]].rename(columns={"name": "country"})


def interpolate_annual(mys):
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


def build_panel():
    mys = compute_mean_years()
    annual = interpolate_annual(mys)
    le_wb = load_wb("life_expectancy_years.csv")
    u5_wb = load_wb("child_mortality_u5.csv")

    # Local lowercase name lookup
    from _shared import NAME_MAP
    rows = []
    for c, mys_s in annual.items():
        wbn = NAME_MAP.get(c, c).lower()
        for yr in range(1960, 2016):
            if yr not in mys_s.index:
                continue
            my = mys_s[yr]
            if pd.isna(my):
                continue
            y_str = str(yr)
            le_val = le_wb.at[wbn, y_str] if (wbn in le_wb.index and y_str in le_wb.columns) else np.nan
            u5_val = u5_wb.at[wbn, y_str] if (wbn in u5_wb.index and y_str in u5_wb.columns) else np.nan
            if pd.notna(le_val) or pd.notna(u5_val):
                rows.append({"country": c, "year": yr, "mean_yrs": my, "le": le_val, "u5mr": u5_val})
    return pd.DataFrame(rows)


def gap_at_year(panel, year, band):
    china = panel[(panel["country"] == "China") & (panel["year"] == year)]
    if china.empty:
        return None
    cr = china.iloc[0]
    my = cr["mean_yrs"]
    # Match the main spec: peers are country-years from ANY calendar year
    # within ±band mean years of schooling of China's value at this year.
    peers = panel[
        (abs(panel["mean_yrs"] - my) <= band)
        & (panel["country"] != "China")
    ]
    p_le = peers["le"].dropna()
    p_u5 = peers["u5mr"].dropna()
    return {
        "year": year,
        "china_mys": round(float(my), 2),
        "n_peers_le": int(len(p_le)),
        "china_le": round(float(cr["le"]), 1) if pd.notna(cr["le"]) else None,
        "peer_le_mean": round(float(p_le.mean()), 1) if len(p_le) else None,
        "le_gap": round(float(cr["le"] - p_le.mean()), 1) if len(p_le) and pd.notna(cr["le"]) else None,
        "n_peers_u5": int(len(p_u5)),
        "china_u5": round(float(cr["u5mr"]), 1) if pd.notna(cr["u5mr"]) else None,
        "peer_u5_mean": round(float(p_u5.mean()), 1) if len(p_u5) else None,
        "u5_gap": round(float(cr["u5mr"] - p_u5.mean()), 1) if len(p_u5) and pd.notna(cr["u5mr"]) else None,
    }


def main():
    print("Building panel (same machinery as china_mean_years.py)...")
    panel = build_panel()
    print(f"  Panel: {len(panel):,} obs, {panel['country'].nunique()} countries")
    print()

    results = {}
    for band in BANDS:
        print(f"=== Band = ±{band} mean years ===")
        print(f"{'Year':>5} {'MYS':>5} {'N_LE':>5} {'ΔLE':>7} {'ΔU5':>7}")
        rows = []
        for yr in KEY_YEARS:
            g = gap_at_year(panel, yr, band)
            if g is None:
                continue
            rows.append(g)
            le_gap = f"{g['le_gap']:+.1f}" if g['le_gap'] is not None else "—"
            u5_gap = f"{g['u5_gap']:+.1f}" if g['u5_gap'] is not None else "—"
            print(f"{g['year']:>5} {g['china_mys']:>5} {g['n_peers_le']:>5} {le_gap:>7} {u5_gap:>7}")
        results[str(band)] = rows
        print()

    # Flat checkin with one number per (band, year, metric),
    # plus the band bounds themselves so paper references are traceable.
    checkin = {
        "band_lo": min(BANDS),
        "band_hi": max(BANDS),
    }
    for band, rows in results.items():
        for r in rows:
            tag = f"China-peers-band{band}-{r['year']}"
            if r.get("le_gap") is not None:
                checkin[f"{tag}-LE-gap"] = r["le_gap"]
            if r.get("u5_gap") is not None:
                checkin[f"{tag}-U5-gap"] = r["u5_gap"]

    # Summary: sign of LE gap and U5 gap at each year, across all bands
    print("Sign-robustness across bands:")
    print(f"{'Year':>5} | {'LE sign':>20} | {'U5 sign':>20}")
    for yr in KEY_YEARS:
        le_signs = []
        u5_signs = []
        for band in BANDS:
            rows = results[str(band)]
            match = [r for r in rows if r["year"] == yr]
            if not match:
                continue
            r = match[0]
            if r.get("le_gap") is not None:
                le_signs.append("−" if r["le_gap"] < 0 else "+")
            if r.get("u5_gap") is not None:
                u5_signs.append("+" if r["u5_gap"] > 0 else "−")
        print(f"{yr:>5} | {''.join(le_signs):>20} | {''.join(u5_signs):>20}")

    write_checkin(
        "china_band_sensitivity.json",
        checkin,
        script_path="scripts/cases/china_band_sensitivity.py",
    )


if __name__ == "__main__":
    main()
