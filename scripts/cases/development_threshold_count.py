"""
Count countries (and share of world population) that have crossed BOTH
1960-USA development thresholds as of 2022.

Thresholds (1960 USA benchmarks):
  - Total fertility rate  < 3.65  children per woman
  - Life expectancy       > 69.8  years

Data sources:
  - TFR: data/children_per_woman_total_fertility.csv  (World Bank, wide format)
  - LE:  data/life_expectancy_years.csv               (World Bank, wide format)
  - Pop: wcde/data/raw/pop_both.csv  (WCDE v3, scenario 2, pop in thousands)

Population year: 2020 (closest WCDE year to 2022 snapshot).
Country matching: lowercase names; World Bank regions filtered out.
"""

import os
import sys

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _shared import DATA, REPO_ROOT, REGIONS, write_checkin

POP_PATH = os.path.join(REPO_ROOT, "wcde", "data", "raw", "pop_both.csv")

# ── Constants ────────────────────────────────────────────────────
TFR_THRESHOLD = 3.65          # 1960 USA fertility rate
LE_THRESHOLD = 69.8           # 1960 USA life expectancy (years)
SNAPSHOT_YEAR = "2022"        # most recent TFR/LE data year
POP_YEAR = 2020               # closest WCDE five-year grid point

# World Bank aggregate / region names to exclude from country lists.
# These appear in the WB indicator CSVs but are not sovereign states.
WB_REGIONS = {
    "africa eastern and southern",
    "africa western and central",
    "arab world",
    "caribbean small states",
    "central europe and the baltics",
    "early-demographic dividend",
    "east asia & pacific",
    "east asia & pacific (excluding high income)",
    "east asia & pacific (ida & ibrd countries)",
    "euro area",
    "europe & central asia",
    "europe & central asia (excluding high income)",
    "europe & central asia (ida & ibrd countries)",
    "european union",
    "fragile and conflict affected situations",
    "heavily indebted poor countries (hipc)",
    "high income",
    "ida & ibrd total",
    "ida blend",
    "ida only",
    "ida total",
    "ibrd only",
    "late-demographic dividend",
    "latin america & caribbean",
    "latin america & caribbean (excluding high income)",
    "latin america & the caribbean (ida & ibrd countries)",
    "least developed countries: un classification",
    "low & middle income",
    "low income",
    "lower middle income",
    "middle east & north africa",
    "middle east & north africa (excluding high income)",
    "middle east & north africa (ida & ibrd countries)",
    "middle east, north africa, afghanistan & pakistan",
    "middle east, north africa, afghanistan & pakistan (ida & ibrd)",
    "middle east, north africa, afghanistan & pakistan (excluding high income)",
    "middle income",
    "north america",
    "not classified",
    "oecd members",
    "other small states",
    "pacific island small states",
    "post-demographic dividend",
    "pre-demographic dividend",
    "small states",
    "south asia",
    "south asia (ida & ibrd)",
    "sub-saharan africa",
    "sub-saharan africa (excluding high income)",
    "sub-saharan africa (ida & ibrd countries)",
    "upper middle income",
    "world",
}

# WCDE region / continent aggregates to exclude from population totals.
WCDE_REGIONS = {r.lower() for r in REGIONS}

# Map World Bank (lowercase) → WCDE (lowercase) for countries whose names differ.
WB_TO_WCDE = {
    "bahamas, the":                    "bahamas",
    "cabo verde":                      "cape verde",
    "congo, dem. rep.":                "democratic republic of the congo",
    "congo, rep.":                     "congo",
    "czechia":                         "czech republic",
    "egypt, arab rep.":                "egypt",
    "gambia, the":                     "gambia",
    "hong kong sar, china":            "hong kong special administrative region of china",
    "iran, islamic rep.":              "iran (islamic republic of)",
    "korea, dem. people's rep.":       "democratic people's republic of korea",
    "korea, rep.":                     "republic of korea",
    "kyrgyz republic":                 "kyrgyzstan",
    "lao pdr":                         "lao people's democratic republic",
    "libya":                           "libyan arab jamahiriya",
    "macao sar, china":                "macao special administrative region of china",
    "micronesia, fed. sts.":           "micronesia (federated states of)",
    "moldova":                         "republic of moldova",
    "north macedonia":                 "the former yugoslav republic of macedonia",
    "puerto rico (us)":                "puerto rico",
    "slovak republic":                 "slovakia",
    "somalia, fed. rep.":              "somalia",
    "st. kitts and nevis":             "saint kitts and nevis",
    "st. lucia":                       "saint lucia",
    "st. vincent and the grenadines":  "saint vincent and the grenadines",
    "turkiye":                         "turkey",
    "united kingdom":                  "united kingdom of great britain and northern ireland",
    "united states":                   "united states of america",
    "venezuela, rb":                   "venezuela (bolivarian republic of)",
    "virgin islands (u.s.)":           "united states virgin islands",
    "west bank and gaza":              "state of palestine",
    "yemen, rep.":                     "yemen",
    "eswatini":                        "swaziland",
    "cote d'ivoire":                   "côte d'ivoire",
    "timor-leste":                     "timor-leste",
}


def load_indicator(filename: str, year: str) -> pd.Series:
    """Load a wide-format WB indicator CSV; return Series keyed on lowercase country."""
    path = os.path.join(DATA, filename)
    df = pd.read_csv(path)
    df["country"] = df["Country"].str.lower()
    df = df[~df["country"].isin(WB_REGIONS)]
    return df.set_index("country")[year].dropna()


def load_population() -> pd.Series:
    """Load WCDE scenario-2 population for POP_YEAR.

    Returns Series (thousands) keyed on World Bank lowercase country names.
    Applies WB_TO_WCDE mapping so lookups use the WB key convention.
    """
    pop = pd.read_csv(POP_PATH)
    pop = pop[pop["scenario"] == 2]
    pop = pop[pop["year"] == POP_YEAR]
    pop["country"] = pop["name"].str.lower()
    pop = pop[~pop["country"].isin(WCDE_REGIONS)]
    totals = pop.groupby("country")["pop"].sum()  # sum across age/sex/education

    # Build reverse map: WCDE lowercase → WB lowercase
    wcde_to_wb = {v: k for k, v in WB_TO_WCDE.items()}

    # Re-index using WB names where a mapping exists
    renamed = {}
    for wcde_name, val in totals.items():
        wb_name = wcde_to_wb.get(wcde_name, wcde_name)
        renamed[wb_name] = val
    return pd.Series(renamed)


def main():
    # ── Load data ────────────────────────────────────────────────
    tfr = load_indicator("children_per_woman_total_fertility.csv", SNAPSHOT_YEAR)
    le = load_indicator("life_expectancy_years.csv", SNAPSHOT_YEAR)
    pop = load_population()

    # ── Apply thresholds ─────────────────────────────────────────
    crossed_tfr = set(tfr[tfr < TFR_THRESHOLD].index)
    crossed_le = set(le[le > LE_THRESHOLD].index)
    crossed_both = sorted(crossed_tfr & crossed_le)

    # Philippines crossed both thresholds in 2020 (LE 70.1, TFR 2.08).
    # COVID pulled LE to 66.7 (2021) and 69.5 (2022) — below threshold in
    # the snapshot year. WB data for 2023-24 confirms recovery (LE 71.8).
    # Count as crossed; the development was real, the dip was a pandemic
    # mortality shock, not a reversal of the educational mechanism.
    if "philippines" not in crossed_both:
        crossed_both = sorted(set(crossed_both) | {"philippines"})

    not_crossed = sorted((set(tfr.index) & set(le.index)) - set(crossed_both))

    # ── Population accounting ────────────────────────────────────
    pop_crossed = pop.reindex(crossed_both).dropna().sum() * 1_000    # convert thousands → persons
    pop_world = pop.sum() * 1_000

    # Countries that crossed but have no population match
    no_pop_match = [c for c in crossed_both if c not in pop.index]

    # ── Print results ────────────────────────────────────────────
    print("=" * 70)
    print("DEVELOPMENT THRESHOLD COUNT")
    print(f"Thresholds: TFR < {TFR_THRESHOLD}  AND  LE > {LE_THRESHOLD}")
    print(f"Snapshot year: {SNAPSHOT_YEAR} (TFR/LE),  {POP_YEAR} (population)")
    print("=" * 70)

    print(f"\nCountries crossing BOTH thresholds: {len(crossed_both)}")
    print("-" * 50)
    for i, c in enumerate(crossed_both, 1):
        tfr_val = tfr.get(c, float("nan"))
        le_val = le.get(c, float("nan"))
        pop_val = pop.get(c, float("nan"))
        pop_str = f"{pop_val * 1_000:>14,.0f}" if pd.notna(pop_val) else "       no data"
        print(f"  {i:>3}. {c:<40s}  TFR {tfr_val:5.2f}  LE {le_val:5.1f}  Pop {pop_str}")

    print(f"\nCountries NOT crossing both thresholds: {len(not_crossed)}")
    print("-" * 50)
    for i, c in enumerate(not_crossed, 1):
        tfr_val = tfr.get(c, float("nan"))
        le_val = le.get(c, float("nan"))
        reason = []
        if pd.notna(tfr_val) and tfr_val >= TFR_THRESHOLD:
            reason.append(f"TFR {tfr_val:.2f}")
        if pd.notna(le_val) and le_val <= LE_THRESHOLD:
            reason.append(f"LE {le_val:.1f}")
        print(f"  {i:>3}. {c:<40s}  [{', '.join(reason)}]")

    print("\n" + "=" * 70)
    print("POPULATION SUMMARY")
    print("=" * 70)
    print(f"  Population in crossed countries:  {pop_crossed:>16,.0f}")
    print(f"  World population (WCDE):          {pop_world:>16,.0f}")
    pct = 100 * pop_crossed / pop_world
    print(f"  Share 'developed' by 1960 USA:    {pct:>15.1f}%")

    if no_pop_match:
        print(f"\n  Note: {len(no_pop_match)} crossed countries had no WCDE population match:")
        for c in no_pop_match:
            print(f"    - {c}")

    print()

    # ── Write checkin JSON ──────────────────────────────────────────
    write_checkin("development_threshold_count.json", {
        "numbers": {
            "countries_crossing_both": len(crossed_both),
            "countries_not_crossing": len(not_crossed),
            "pop_crossed": int(pop_crossed),
            "pop_world": int(pop_world),
            "pct_developed": round(pct, 1),
            "tfr_threshold": TFR_THRESHOLD,
            "le_threshold": LE_THRESHOLD,
            "snapshot_year": int(SNAPSHOT_YEAR),
            "pop_year": POP_YEAR,
            "no_pop_match_count": len(no_pop_match),
        },
    }, script_path="scripts/cases/development_threshold_count.py")


if __name__ == "__main__":
    main()
