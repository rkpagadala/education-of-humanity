#!/usr/bin/env python3
"""
cases/support_led_peers.py
==========================
Education-substrate peer matching for the Drèze-Sen support-led canon
(Cuba, Sri Lanka, Costa Rica). Reports LE and U5MR vs education-matched
peer pools under five matching regimes:

  A. mys15p          — MYS15+ ±0.5 (the Lutz/WCDE population-stock measure)
  B. mys15p_female   — Female MYS15+ ±0.5 (mothers route the channel; sharper)
  C. multi           — MYS15+ ±0.5 AND % of 15+ with no education ±5pp
                       AND F-M MYS gap ±0.5 (multi-dim snapshot)
  D. depth           — MYS15+ ±0.5 AND MYS15+ at t-25 ±0.5 (generational
                       depth: matches the parents' substrate too)
  E. full_substrate  — MYS15+ ±0.5 AND MYS15+ at t-25 ±0.5 AND % no-edu ±5pp
                       AND F-M MYS gap ±0.5 (maximal substrate snapshot+lag25)
  F. depth_50        — MYS15+ ±0.5 AND MYS15+ at t-25 ±0.5 AND MYS15+ at t-50
                       ±0.5 (three-generation depth: parents AND grandparents)
  G. full_50         — D + lag50 + %no-edu + F-M gap (maximal substrate;
                       only valid for years >= 2000 since WCDE starts 1950)

The China case (scripts/cases/china_mean_years.py) is left untouched.
This script writes a SEPARATE checkin file (support_led_peers.json) and
does not register any number with verify_humanity.py.

The home-niche substrate is the older living generations — parents,
grandparents, older siblings — so MYS15+ replaces the 20-24 cohort metric.
Two further dimensions of the substrate that mean-MYS-alone misses:

  - Educational floor (% of 15+ with NO education): same MYS can be
    achieved by very different distributions; the floor matters most for
    U5MR because the at-risk population is the lowest-educated households.
  - Gender parity (F MYS - M MYS): mothers' education routes the channel
    to TFR/U5MR more heavily than fathers'; same population MYS with
    near-parity vs a year-wide gap is a different substrate.

If the residual collapses under regime B or C, the substrate claim
strengthens (it's just three numbers, not one). If a residual survives,
a non-education channel may be operating.

Reads:
  wcde/data/raw/pop_both.csv
  data/life_expectancy_years.csv
  data/child_mortality_u5.csv

Writes:
  checkin/support_led_peers.json

Kerala is excluded — WCDE is country-level only. Sub-national matching
is its own effort.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _shared import REPO_ROOT, NAME_MAP, REGIONS, load_wb, write_checkin

WCDE_RAW = os.path.join(REPO_ROOT, "wcde", "data", "raw")

YEARS_MAP = {
    "No Education": 0,
    "Incomplete Primary": 3,
    "Primary": 6,
    "Lower Secondary": 9,
    "Upper Secondary": 12,
    "Post Secondary": 15,
}

SNAPSHOT_YEARS = [1965, 1975, 1985, 1995, 2005, 2015]
TARGETS = ["Cuba", "Sri Lanka", "Costa Rica"]

# Matching regimes: each maps a label to a tuple of:
#   (description, peer-selection function)
# The selection function takes (panel_DataFrame, target_row, country_name)
# and returns a DataFrame of peer rows.
MYS_BAND = 0.5
NO_EDU_BAND = 5.0  # percentage points
GAP_BAND = 0.5     # years


def wb_name(wcde):
    return NAME_MAP.get(wcde, wcde).lower()


# ── Feature computation ──────────────────────────────────────────────

def compute_features():
    """Compute (country, year, mys15p, mys_female, mys_male, gap_f_m, pct_no_edu)
    at the 5-year WCDE observation grid."""
    pop = pd.read_csv(os.path.join(WCDE_RAW, "pop_both.csv"))
    pop = pop[pop["scenario"] == 2].copy()
    pop["yrs"] = pop["education"].map(YEARS_MAP)
    pop = pop.dropna(subset=["yrs"])  # drops "Under 15" placeholder rows
    pop["weighted"] = pop["pop"] * pop["yrs"]

    # MYS15+ both sexes
    g_total = pop.groupby(["name", "year"]).agg(
        pop=("pop", "sum"),
        wsum=("weighted", "sum"),
    ).reset_index()
    g_total["mys15p"] = g_total["wsum"] / g_total["pop"]

    # MYS15+ by sex
    g_sex = pop.groupby(["name", "year", "sex"]).agg(
        pop=("pop", "sum"),
        wsum=("weighted", "sum"),
    ).reset_index()
    g_sex["mys"] = g_sex["wsum"] / g_sex["pop"]
    sex_pivot = g_sex.pivot(index=["name", "year"], columns="sex", values="mys").reset_index()
    sex_pivot.columns.name = None
    sex_pivot = sex_pivot.rename(columns={"Female": "mys_female", "Male": "mys_male"})
    sex_pivot["gap_f_m"] = sex_pivot["mys_female"] - sex_pivot["mys_male"]

    # % 15+ with No Education
    no_edu_rows = pop[pop["education"] == "No Education"]
    g_no = no_edu_rows.groupby(["name", "year"]).agg(no_edu_pop=("pop", "sum")).reset_index()
    feat = g_total.merge(g_no, on=["name", "year"], how="left")
    feat["no_edu_pop"] = feat["no_edu_pop"].fillna(0.0)
    feat["pct_no_edu"] = feat["no_edu_pop"] / feat["pop"] * 100

    out = feat.merge(sex_pivot, on=["name", "year"])
    out = out[~out["name"].isin(REGIONS)]
    return out[[
        "name", "year",
        "mys15p", "mys_female", "mys_male", "gap_f_m", "pct_no_edu",
    ]].rename(columns={"name": "country"})


def interpolate_features(feat):
    """For each country, interpolate each feature column annually
    between the 5-year observation points."""
    cols = ["mys15p", "mys_female", "mys_male", "gap_f_m", "pct_no_edu"]
    annual = {}
    for c, grp in feat.groupby("country"):
        grp = grp.sort_values("year")
        years = grp["year"].values
        if len(years) < 2:
            continue
        idx = list(range(int(years.min()), int(years.max()) + 1))
        df = pd.DataFrame({"year": idx})
        for col in cols:
            s = pd.Series(grp[col].values, index=grp["year"].values)
            df[col] = df["year"].map(s).interpolate(method="linear").values
        df["country"] = c
        annual[c] = df
    return annual


def build_panel(annual, le_wb, u5_wb):
    rows = []
    for c, df in annual.items():
        wbn = wb_name(c)
        for _, r in df.iterrows():
            yr = int(r["year"])
            if yr < 1960 or yr > 2015:
                continue
            y_str = str(yr)
            le_val = le_wb.at[wbn, y_str] if (wbn in le_wb.index and y_str in le_wb.columns) else np.nan
            u5_val = u5_wb.at[wbn, y_str] if (wbn in u5_wb.index and y_str in u5_wb.columns) else np.nan
            if not (pd.notna(le_val) or pd.notna(u5_val)):
                continue
            if pd.isna(r["mys15p"]):
                continue
            rows.append({
                "country": c, "year": yr,
                "mys15p": r["mys15p"],
                "mys_female": r["mys_female"],
                "mys_male": r["mys_male"],
                "gap_f_m": r["gap_f_m"],
                "pct_no_edu": r["pct_no_edu"],
                "le": le_val, "u5mr": u5_val,
            })
    return pd.DataFrame(rows)


def build_feature_panel(annual):
    """Education features only, 1950-2015. Used for 25-year-lagged lookups
    that need pre-1960 mys15p data the main (LE/U5-required) panel drops."""
    rows = []
    for c, df in annual.items():
        for _, r in df.iterrows():
            yr = int(r["year"])
            if yr < 1950 or yr > 2015:
                continue
            if pd.isna(r["mys15p"]):
                continue
            rows.append({
                "country": c, "year": yr,
                "mys15p": r["mys15p"],
            })
    return pd.DataFrame(rows)


def add_lag_column(panel, feature_panel, n_years=25):
    """Add mys15p_lag<N> column to panel via self-merge with feature_panel
    at year - N. Rows where the lagged year falls before 1950 get NaN."""
    src = feature_panel[["country", "year", "mys15p"]].rename(
        columns={"mys15p": f"mys15p_lag{n_years}"}
    )
    src["year"] = src["year"] + n_years
    return panel.merge(src, on=["country", "year"], how="left")


# ── Matching regimes ─────────────────────────────────────────────────

def match_mys15p(panel, target_row, country):
    return panel[
        (abs(panel["mys15p"] - target_row["mys15p"]) <= MYS_BAND) &
        (panel["country"] != country)
    ]


def match_mys15p_female(panel, target_row, country):
    sub = panel.dropna(subset=["mys_female"])
    return sub[
        (abs(sub["mys_female"] - target_row["mys_female"]) <= MYS_BAND) &
        (sub["country"] != country)
    ]


def match_multi(panel, target_row, country):
    sub = panel.dropna(subset=["mys15p", "pct_no_edu", "gap_f_m"])
    return sub[
        (abs(sub["mys15p"] - target_row["mys15p"]) <= MYS_BAND) &
        (abs(sub["pct_no_edu"] - target_row["pct_no_edu"]) <= NO_EDU_BAND) &
        (abs(sub["gap_f_m"] - target_row["gap_f_m"]) <= GAP_BAND) &
        (sub["country"] != country)
    ]


def match_depth(panel, target_row, country):
    if pd.isna(target_row.get("mys15p_lag25")):
        return panel.iloc[0:0]
    sub = panel.dropna(subset=["mys15p", "mys15p_lag25"])
    return sub[
        (abs(sub["mys15p"] - target_row["mys15p"]) <= MYS_BAND) &
        (abs(sub["mys15p_lag25"] - target_row["mys15p_lag25"]) <= MYS_BAND) &
        (sub["country"] != country)
    ]


def match_full_substrate(panel, target_row, country):
    if pd.isna(target_row.get("mys15p_lag25")):
        return panel.iloc[0:0]
    sub = panel.dropna(subset=["mys15p", "mys15p_lag25", "pct_no_edu", "gap_f_m"])
    return sub[
        (abs(sub["mys15p"] - target_row["mys15p"]) <= MYS_BAND) &
        (abs(sub["mys15p_lag25"] - target_row["mys15p_lag25"]) <= MYS_BAND) &
        (abs(sub["pct_no_edu"] - target_row["pct_no_edu"]) <= NO_EDU_BAND) &
        (abs(sub["gap_f_m"] - target_row["gap_f_m"]) <= GAP_BAND) &
        (sub["country"] != country)
    ]


def match_depth_50(panel, target_row, country):
    if pd.isna(target_row.get("mys15p_lag25")) or pd.isna(target_row.get("mys15p_lag50")):
        return panel.iloc[0:0]
    sub = panel.dropna(subset=["mys15p", "mys15p_lag25", "mys15p_lag50"])
    return sub[
        (abs(sub["mys15p"] - target_row["mys15p"]) <= MYS_BAND) &
        (abs(sub["mys15p_lag25"] - target_row["mys15p_lag25"]) <= MYS_BAND) &
        (abs(sub["mys15p_lag50"] - target_row["mys15p_lag50"]) <= MYS_BAND) &
        (sub["country"] != country)
    ]


def match_full_50(panel, target_row, country):
    if pd.isna(target_row.get("mys15p_lag25")) or pd.isna(target_row.get("mys15p_lag50")):
        return panel.iloc[0:0]
    sub = panel.dropna(subset=[
        "mys15p", "mys15p_lag25", "mys15p_lag50", "pct_no_edu", "gap_f_m"
    ])
    return sub[
        (abs(sub["mys15p"] - target_row["mys15p"]) <= MYS_BAND) &
        (abs(sub["mys15p_lag25"] - target_row["mys15p_lag25"]) <= MYS_BAND) &
        (abs(sub["mys15p_lag50"] - target_row["mys15p_lag50"]) <= MYS_BAND) &
        (abs(sub["pct_no_edu"] - target_row["pct_no_edu"]) <= NO_EDU_BAND) &
        (abs(sub["gap_f_m"] - target_row["gap_f_m"]) <= GAP_BAND) &
        (sub["country"] != country)
    ]


REGIMES = [
    ("mys15p", "MYS15+ ±0.5", match_mys15p),
    ("mys15p_female", "Female MYS15+ ±0.5", match_mys15p_female),
    ("multi", "MYS15+ ±0.5 AND %no-edu ±5pp AND F-M gap ±0.5", match_multi),
    ("depth", "MYS15+ ±0.5 AND MYS15+(t-25) ±0.5", match_depth),
    ("full_substrate",
     "MYS15+ ±0.5 AND MYS15+(t-25) ±0.5 AND %no-edu ±5pp AND F-M gap ±0.5",
     match_full_substrate),
    ("depth_50",
     "MYS15+ ±0.5 AND MYS15+(t-25) ±0.5 AND MYS15+(t-50) ±0.5",
     match_depth_50),
    ("full_50",
     "MYS15+ ±0.5 AND MYS15+(t-25) ±0.5 AND MYS15+(t-50) ±0.5 AND %no-edu ±5pp AND F-M gap ±0.5",
     match_full_50),
]


# ── Per-country, per-regime computation ──────────────────────────────

def country_vs_peers(panel, country, match_fn):
    target = panel[panel["country"] == country].sort_values("year")
    if target.empty:
        raise ValueError(f"{country} not present in panel")
    rows = []
    for _, cr in target.iterrows():
        peers = match_fn(panel, cr, country)
        p_le = peers["le"].dropna()
        p_u5 = peers["u5mr"].dropna()
        rows.append({
            "year": int(cr["year"]),
            "mys15p": float(cr["mys15p"]),
            "mys_female": float(cr["mys_female"]) if pd.notna(cr["mys_female"]) else None,
            "pct_no_edu": float(cr["pct_no_edu"]) if pd.notna(cr["pct_no_edu"]) else None,
            "gap_f_m": float(cr["gap_f_m"]) if pd.notna(cr["gap_f_m"]) else None,
            "country_le": float(cr["le"]) if pd.notna(cr["le"]) else None,
            "country_u5": float(cr["u5mr"]) if pd.notna(cr["u5mr"]) else None,
            "peer_le_mean": float(p_le.mean()) if len(p_le) > 0 else None,
            "peer_u5_mean": float(p_u5.mean()) if len(p_u5) > 0 else None,
            "n_peers_le": int(len(p_le)),
            "n_peers_u5": int(len(p_u5)),
        })
    return pd.DataFrame(rows)


def _r(v, n=1):
    if v is None or (isinstance(v, float) and (pd.isna(v) or np.isnan(v))):
        return None
    return round(float(v), n)


def regime_payload(result):
    by_year = result.set_index("year")
    key = {}
    for yr in SNAPSHOT_YEARS:
        if yr not in by_year.index:
            continue
        r = by_year.loc[yr]
        c_le = r["country_le"]
        p_le = r["peer_le_mean"]
        c_u5 = r["country_u5"]
        p_u5 = r["peer_u5_mean"]
        key[f"mys15p_{yr}"] = _r(r["mys15p"], 2)
        key[f"country_le_{yr}"] = _r(c_le, 1)
        key[f"peer_le_{yr}"] = _r(p_le, 1)
        key[f"le_gap_{yr}"] = _r(c_le - p_le, 1) if (c_le is not None and p_le is not None) else None
        key[f"country_u5_{yr}"] = _r(c_u5, 1)
        key[f"peer_u5_{yr}"] = _r(p_u5, 1)
        key[f"u5_gap_{yr}"] = _r(c_u5 - p_u5, 1) if (c_u5 is not None and p_u5 is not None) else None
        key[f"n_peers_le_{yr}"] = int(r["n_peers_le"])

    annual = []
    for _, r in result.iterrows():
        c_le = r["country_le"]
        p_le = r["peer_le_mean"]
        c_u5 = r["country_u5"]
        p_u5 = r["peer_u5_mean"]
        annual.append({
            "year": int(r["year"]),
            "mys15p": _r(r["mys15p"], 2),
            "country_le": _r(c_le, 1),
            "peer_le": _r(p_le, 1),
            "le_gap": _r(c_le - p_le, 1) if (c_le is not None and p_le is not None) else None,
            "country_u5": _r(c_u5, 1),
            "peer_u5": _r(p_u5, 1),
            "u5_gap": _r(c_u5 - p_u5, 1) if (c_u5 is not None and p_u5 is not None) else None,
            "n_peers_le": int(r["n_peers_le"]),
            "n_peers_u5": int(r["n_peers_u5"]),
        })
    return {"key_data_points": key, "annual_data": annual}


# ── Reporting ───────────────────────────────────────────────────────

def print_summary(country, regime_label, result):
    print()
    print(f"{country.upper()} — {regime_label}")
    print("=" * 100)
    print(f"{'Year':>5} {'MYS':>5} {'Fmys':>5} {'%NoEd':>5} {'F-M':>5} | "
          f"{'Cnt_LE':>7} {'Peer_LE':>8} {'ΔLE':>6} | {'Cnt_U5':>7} {'Peer_U5':>8} {'ΔU5':>6} | {'N':>4}")
    print("-" * 100)
    for _, r in result.iterrows():
        yr = int(r["year"])
        if yr % 5 != 0:
            continue
        d_le = (r["country_le"] - r["peer_le_mean"]) if (r["country_le"] is not None and r["peer_le_mean"] is not None) else None
        d_u5 = (r["country_u5"] - r["peer_u5_mean"]) if (r["country_u5"] is not None and r["peer_u5_mean"] is not None) else None

        def f(v, fmt=".1f"):
            return f"{v:{fmt}}" if v is not None else "n/a"

        print(f"{yr:>5} {r['mys15p']:>5.2f} {f(r['mys_female'], '.2f'):>5} "
              f"{f(r['pct_no_edu'], '.1f'):>5} {f(r['gap_f_m'], '+.2f'):>5} | "
              f"{f(r['country_le']):>7} {f(r['peer_le_mean']):>8} {f(d_le, '+.1f'):>6} | "
              f"{f(r['country_u5']):>7} {f(r['peer_u5_mean']):>8} {f(d_u5, '+.1f'):>6} | "
              f"{r['n_peers_le']:>4}")


def main():
    print("Computing education features (MYS15+, female MYS, %no-edu, gender gap)...")
    feat = compute_features()
    annual = interpolate_features(feat)

    print("Loading World Bank LE and U5MR...")
    le_wb = load_wb("life_expectancy_years.csv")
    u5_wb = load_wb("child_mortality_u5.csv")

    print("Building annual panel...")
    panel = build_panel(annual, le_wb, u5_wb)
    feature_panel = build_feature_panel(annual)
    panel = add_lag_column(panel, feature_panel, n_years=25)
    panel = add_lag_column(panel, feature_panel, n_years=50)
    print(f"  Panel: {len(panel):,} obs, {panel['country'].nunique()} countries")
    print(f"  Rows with mys15p_lag25 available: {panel['mys15p_lag25'].notna().sum():,}")
    print(f"  Rows with mys15p_lag50 available: {panel['mys15p_lag50'].notna().sum():,}")

    countries_payload = {}
    for country in TARGETS:
        regimes = {}
        for label, desc, fn in REGIMES:
            result = country_vs_peers(panel, country, fn)
            print_summary(country, desc, result)
            regimes[label] = regime_payload(result)
            regimes[label]["description"] = desc
        countries_payload[country] = {"regimes": regimes}

    checkin_data = {
        "method": (
            "Education-substrate peer matching for the Drèze-Sen support-led canon "
            "under three regimes. (A) MYS15+ alone (Lutz/WCDE population-stock "
            "measure). (B) Female MYS15+ alone (mothers route the channel). "
            "(C) Multi-dim: MYS15+ ±0.5 AND %15+ with no education ±5pp AND "
            "F-M MYS gap ±0.5 — same MYS achieved by very different distributions; "
            "the multi-dim match controls for distribution and gender parity. "
            "WCDE v3 pop_both.csv (scenario 2); LE and U5MR from World Bank WDI. "
            "If the residual collapses under (B) or (C), the substrate is "
            "distribution + gender parity, not state provision."
        ),
        "bands": {
            "mys15p": MYS_BAND,
            "pct_no_edu_pp": NO_EDU_BAND,
            "gap_f_m": GAP_BAND,
        },
        "snapshot_years": SNAPSHOT_YEARS,
        "regimes": [{"label": l, "description": d} for l, d, _ in REGIMES],
        "countries": countries_payload,
    }

    write_checkin(
        "support_led_peers.json",
        checkin_data,
        script_path="scripts/cases/support_led_peers.py",
    )


if __name__ == "__main__":
    main()
