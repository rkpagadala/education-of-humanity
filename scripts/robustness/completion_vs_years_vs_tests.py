# =============================================================================
# PAPER REFERENCE
# Script:  scripts/robustness/completion_vs_years_vs_tests.py
# Paper:   "Education of Nations" — §4.3 Completion as the Operative Variable
#
# Produces:
#   4-horse race extending completion_vs_test_scores.py. Asks whether
#   duration of exposure (years of schooling) beats fidelity (test scores)
#   at predicting development outcomes — the specific claim in §4.3
#   ("the dose is duration of exposure, not fidelity of instruction").
#
#   Four measures raced on the overlap sample where all four exist:
#     1. WCDE lower secondary completion (threshold measure)
#     2. WCDE mean years of schooling (continuous duration, same dataset)
#     3. Barro-Lee mean years of schooling (continuous duration, independent dataset)
#     4. HLO harmonised test scores (fidelity measure, Angrist et al. 2021)
#
#   Within-WCDE consistency: completion vs mean years from the same source.
#   Cross-dataset robustness: WCDE mean years vs Barro-Lee mean years.
#   Direct comparison: duration (both encodings) vs fidelity.
#
# Inputs:
#   wcde/data/raw/prop_both.csv             — WCDE cohort proportions
#   wcde/data/processed/completion_both_long.csv  — WCDE completion
#   data/barro_lee_v3.csv                   — Barro-Lee education data
#   data/hlo_raw.csv                        — HLO test scores
#   data/life_expectancy_years.csv          — WB LE
#   data/children_per_woman_total_fertility.csv   — WB TFR
#   data/child_mortality_u5.csv             — WB U5MR
#   data/gdppercapita_us_inflation_adjusted.csv   — WB GDP
#
# Outputs:
#   checkin/completion_vs_years_vs_tests.json
# =============================================================================
"""
completion_vs_years_vs_tests.py

4-horse race: WCDE completion, WCDE mean years, Barro-Lee mean years,
HLO test scores — on the overlap sample (countries/years with all four).
Country FE, clustered SEs, lag 0 and lag 10. Settles the duration-vs-fidelity
claim on internally consistent data.
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from residualization._shared import (
    load_education, load_wb, interpolate_to_annual,
    build_panel, clustered_fe, get_wb_val, DATA,
)
from _shared import fmt_r2, write_checkin, REGIONS, NAME_MAP

WCDE_RAW = os.path.join(REPO_ROOT, "wcde", "data", "raw")

# ── WCDE education level → years of schooling ──────────────────────
YEARS_MAP = {
    "No Education": 0,
    "Incomplete Primary": 3,
    "Primary": 6,
    "Lower Secondary": 9,
    "Upper Secondary": 12,
    "Post Secondary": 15,
}

# HLO country name → WB lowercase (subset from completion_vs_test_scores.py)
HLO_NAME_MAP = {
    "czech republic": "czechia",
    "egypt": "egypt, arab rep.",
    "hong kong, sar china": "hong kong sar, china",
    "iran, islamic republic of": "iran, islamic rep.",
    "korea\u00ac\u2020(south)": "korea, rep.",
    "kyrgyzstan": "kyrgyz republic",
    "macao, sar china": "macao sar, china",
    "macedonia, republic of": "north macedonia",
    "serbia and montenegro": "serbia",
    "slovakia": "slovak republic",
    "syrian arab republic\u00ac\u2020(syria)": "syrian arab republic",
    "taiwan, republic of china": "taiwan",
    "turkey": "turkiye",
    "united states of america": "united states",
}

# Barro-Lee country name → WB lowercase
BL_TO_WB = {
    "Korea, Rep.": "korea, rep.",
    "Iran (Islamic Republic of)": "iran, islamic rep.",
    "Egypt": "egypt, arab rep.",
    "Venezuela": "venezuela, rb",
    "Congo, Dem. Rep.": "congo, dem. rep.",
    "Gambia": "gambia, the",
    "Lao People's Democratic Republic": "lao pdr",
    "Syrian Arab Republic": "syrian arab republic",
    "Yemen": "yemen, rep.",
    "Kyrgyzstan": "kyrgyz republic",
    "Czech Republic": "czechia",
    "Republic of Moldova": "moldova",
    "Russian Federation": "russian federation",
    "Slovakia": "slovak republic",
    "Taiwan": "taiwan",
    "TFYR Macedonia": "north macedonia",
    "Turkey": "turkiye",
    "Viet Nam": "vietnam",
}


# ── WCDE mean years (from cohort proportions) ──────────────────────

def load_wcde_mean_years():
    """Returns {country_lc: pd.Series(year → mean_yrs annual)}."""
    prop = pd.read_csv(os.path.join(WCDE_RAW, "prop_both.csv"))
    mask = (
        (prop["age"] == "20--24") &
        (prop["sex"] == "Both") &
        (prop["scenario"] == 2)
    )
    sub = prop[mask].copy()
    sub["yrs"] = sub["education"].map(YEARS_MAP)
    sub = sub.dropna(subset=["yrs"])
    sub["weighted"] = sub["prop"] * sub["yrs"]

    grouped = sub.groupby(["name", "year"]).agg(
        mean_yrs=("weighted", lambda x: x.sum() / 100),
        total=("prop", "sum"),
    ).reset_index()
    grouped = grouped[
        (~grouped["name"].isin(REGIONS)) &
        (grouped["total"] > 95)
    ]

    out = {}
    for name, g in grouped.groupby("name"):
        country_lc = NAME_MAP.get(name, name).lower()
        s = g.set_index("year")["mean_yrs"].sort_index()
        s = s[(s.index >= 1950) & (s.index <= 2015)]
        if len(s) < 2:
            continue
        full = pd.Series(dtype=float, index=range(s.index.min(), s.index.max() + 1))
        full.update(s)
        out[country_lc] = full.interpolate(method="linear")
    return out


# ── Barro-Lee mean years (age 15-24, MF) ───────────────────────────

def load_bl_mean_years():
    """Returns {country_lc: pd.Series(year → yr_sch)}."""
    df = pd.read_csv(os.path.join(DATA, "barro_lee_v3.csv"))
    df = df[(df["agefrom"] == 15) & (df["ageto"] == 24) & (df["sex"] == "MF")].copy()
    df["country_lc"] = df["country"].apply(
        lambda c: BL_TO_WB.get(c, c.lower())
    )
    out = {}
    for c, g in df.groupby("country_lc"):
        s = g.set_index("year")["yr_sch"].sort_index()
        s = pd.to_numeric(s, errors="coerce").dropna()
        s = s[(s.index >= 1950) & (s.index <= 2015)]
        if len(s) < 2:
            continue
        full = pd.Series(dtype=float, index=range(s.index.min(), s.index.max() + 1))
        full.update(s)
        out[c] = full.interpolate(method="linear")
    return out


# ── HLO test scores (secondary, nationally representative) ─────────

def load_hlo_scores():
    raw = pd.read_csv(os.path.join(DATA, "hlo_raw.csv"))
    sec = raw[(raw["level"] == "sec") & (raw["n_res"] == 1)].copy()
    sec["country"] = sec["country"].str.lower().replace(HLO_NAME_MAP)
    avg = sec.groupby(["country", "year"])["hlo"].mean().reset_index()
    wide = avg.pivot(index="country", columns="year", values="hlo")
    wide.columns = [str(int(c)) for c in wide.columns]
    for c in wide.columns:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")
    return wide


# ── Build overlap panel ─────────────────────────────────────────────

T_YEARS_HLO = [2000, 2003, 2006, 2007, 2009, 2011, 2012, 2015]


def val_from_dict(d, country, year):
    # Normalize to the lowercase WB key (same convention as get_wb_val)
    key = NAME_MAP.get(country, country).lower()
    s = d.get(key, d.get(country.lower()))
    if s is None or year not in s.index:
        return np.nan
    v = s.loc[year]
    if isinstance(v, pd.Series):
        v = v.iloc[0]
    return float(v) if pd.notna(v) else np.nan


def build_overlap_panel(edu_annual, wcde_mys, bl_mys, hlo_wide,
                        outcome_df, gdp_df, lag):
    panel = build_panel(edu_annual, outcome_df, gdp_df, T_YEARS_HLO, lag, "outcome")
    rows = []
    for _, r in panel.iterrows():
        country = r["country"]
        t = int(r["t"])
        wys = val_from_dict(wcde_mys, country, t)
        bly = val_from_dict(bl_mys, country, t)
        hlo = get_wb_val(hlo_wide, country, t)
        if any(pd.isna(x) for x in [r["edu_t"], wys, bly, hlo, r["outcome"]]):
            continue
        rows.append({
            "country": country, "t": t,
            "edu_t": r["edu_t"],
            "wcde_mys_t": wys,
            "bl_mys_t": bly,
            "test_t": hlo,
            "outcome": r["outcome"],
        })
    return pd.DataFrame(rows)


def main():
    print("Loading WCDE completion, WCDE mean years, Barro-Lee mean years, HLO...")
    edu_raw = load_education("completion_both_long.csv")
    edu_annual = interpolate_to_annual(edu_raw, "lower_sec")
    wcde_mys = load_wcde_mean_years()
    bl_mys = load_bl_mean_years()
    hlo_wide = load_hlo_scores()

    gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
    le_df = load_wb("life_expectancy_years.csv")
    tfr_df = load_wb("children_per_woman_total_fertility.csv")
    u5_df = load_wb("child_mortality_u5.csv")

    OUTCOMES = {
        "le": ("Life expectancy", le_df),
        "tfr": ("TFR", tfr_df),
        "u5mr": ("U5MR", u5_df),
    }

    print(f"  WCDE mean years: {len(wcde_mys)} countries")
    print(f"  BL mean years:   {len(bl_mys)} countries")
    print(f"  HLO scores:      {len(hlo_wide)} countries")

    results = {}
    horses = [
        ("edu_t",       "WCDE completion"),
        ("wcde_mys_t",  "WCDE mean years"),
        ("bl_mys_t",    "Barro-Lee mean years"),
        ("test_t",      "HLO test scores"),
    ]

    for lag in [0, 10]:
        print(f"\n{'=' * 90}")
        print(f"4-HORSE RACE — lag={lag}, overlap sample, country FE")
        print(f"{'=' * 90}")
        lag_out = {}

        for key, (label, outcome_df) in OUTCOMES.items():
            ol = build_overlap_panel(edu_annual, wcde_mys, bl_mys, hlo_wide,
                                     outcome_df, gdp_df, lag)
            n = len(ol)
            nc = ol["country"].nunique()
            if n < 20 or nc < 3:
                print(f"\n  {label}: insufficient (n={n}, countries={nc})")
                lag_out[key] = None
                continue

            print(f"\n  {label}  (n={n}, countries={nc})")
            print(f"  {'Horse':<25} {'R²':>8} {'β':>12} {'p':>10}")
            print("  " + "-" * 60)

            horse_results = {}
            for col, hname in horses:
                res = clustered_fe(col, "outcome", ol)
                if res is None:
                    print(f"  {hname:<25} {'n/a':>8}")
                    continue
                print(f"  {hname:<25} {fmt_r2(res['r2']):>8} "
                      f"{fmt_r2(res['beta']):>12} {res['pval']:>10.4f}")
                horse_results[col] = {
                    "r2": round(res["r2"], 4),
                    "beta": round(res["beta"], 6),
                    "pval": round(res["pval"], 6),
                    "n": res["n"],
                    "countries": res["countries"],
                }
            lag_out[key] = horse_results

        results[f"lag_{lag}"] = lag_out

    checkin = {
        "method": (
            "4-horse race on overlap sample (countries with WCDE completion, "
            "WCDE mean years, Barro-Lee mean years, and HLO test scores). "
            "Tests whether duration of exposure (years) or fidelity of "
            "instruction (test scores) better predicts LE, TFR, U5MR. "
            "Country FE, clustered SEs, lags 0 and 10."
        ),
        "results": results,
    }
    write_checkin("completion_vs_years_vs_tests.json", checkin,
                  "scripts/robustness/completion_vs_years_vs_tests.py")
    print("\n✓ Results written to checkin/completion_vs_years_vs_tests.json")


if __name__ == "__main__":
    main()
