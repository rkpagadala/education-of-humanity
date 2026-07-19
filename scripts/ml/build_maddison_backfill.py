# =============================================================================
# Script:  scripts/ml/build_maddison_backfill.py
# Paper:   "The Long Childhood: On the Convergence of Humanity"
#
# Produces:
#   data/external/maddison_gdppc_wb_equivalent.csv  (country_canon, year, gdppc)
#
# Rationale:
#   World Bank GDP per capita (constant 2015 USD) only starts in 1960 and has
#   country-specific gaps thereafter. The transformer panel's window starts at
#   T-15, so for T=1965 the sequence includes 1950 and 1955 (both blank in WB).
#   Cells that are missing are systematically the poor countries in early
#   years — exactly where GDP would carry the most information.
#
#   Mean-imputation (the previous default) decorrelates GDP from outcomes for
#   those cells and biases the G2 ablation gate toward "GDP barely matters."
#
#   Maddison Project Database 2020 covers most countries back to 1820 in
#   2011 USD international (PPP). We rescale country-by-country into
#   WB-equivalent 2015 USD using the median overlap-year ratio, then use the
#   rescaled series as backfill ONLY where WB is missing.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from _shared import DATA, load_wb, standardize_country_name

MADDISON_XLSX = os.path.join(DATA, "external", "mpd2020.xlsx")
OUT_CSV = os.path.join(DATA, "external", "maddison_gdppc_wb_equivalent.csv")

# Overlap window for computing rescale ratio. Years where both WB (constant
# 2015 USD) and Maddison (2011 PPP USD) tend to have data for most countries.
# Use the median ratio across observed years in this window per country.
OVERLAP_YEARS = list(range(1990, 2011))


def main():
    print(f"[build_maddison] loading {MADDISON_XLSX}")
    mpd = pd.read_excel(MADDISON_XLSX, sheet_name="Full data")
    mpd = mpd[["country", "year", "gdppc"]].copy()
    mpd["country_canon"] = mpd["country"].map(standardize_country_name)
    mpd = mpd.dropna(subset=["country_canon", "gdppc"])
    mpd["year"] = mpd["year"].astype(int)
    print(f"[build_maddison] {len(mpd):,} (country, year) rows in Maddison after canonicalization")

    print("[build_maddison] loading WB constant 2015 USD series")
    wb = load_wb("gdppercapita_us_inflation_adjusted.csv")
    # canonicalize WB index
    wb.index = [standardize_country_name(c) for c in wb.index]
    wb = wb[~wb.index.isna()]
    wb = wb[~wb.index.duplicated(keep="first")]
    wb.columns = [int(c) for c in wb.columns]

    # Long-format WB
    wb_long = wb.stack().reset_index()
    wb_long.columns = ["country_canon", "year", "gdppc_wb"]
    wb_long = wb_long.dropna(subset=["gdppc_wb"])

    # Merge for overlap rows
    overlap = mpd.merge(wb_long, on=["country_canon", "year"], how="inner")
    overlap = overlap[overlap["year"].isin(OVERLAP_YEARS)].copy()
    overlap["ratio"] = overlap["gdppc_wb"] / overlap["gdppc"]
    overlap = overlap[overlap["ratio"] > 0]
    overlap = overlap[np.isfinite(overlap["ratio"])]

    # Country-specific median ratio
    ratios = overlap.groupby("country_canon")["ratio"].median().to_dict()
    n_with_ratio = len(ratios)
    print(f"[build_maddison] computed country-specific ratios for {n_with_ratio} countries")

    # Global fallback ratio for countries with no overlap (rare; usually only
    # very small states or USSR-era successor states with sparse WB pre-1991)
    global_ratio = float(np.median(list(ratios.values())))
    print(f"[build_maddison] global fallback ratio = {global_ratio:.4f}")

    # Apply rescaling to ALL Maddison rows
    rows = []
    no_ratio = set()
    for _, r in mpd.iterrows():
        cc = r["country_canon"]
        ratio = ratios.get(cc)
        if ratio is None:
            no_ratio.add(cc)
            ratio = global_ratio
        rows.append({"country_canon": cc, "year": int(r["year"]),
                     "gdppc_wb_eq": float(r["gdppc"]) * float(ratio),
                     "used_fallback": ratio == global_ratio and cc in no_ratio})
    out = pd.DataFrame(rows)
    out = out.sort_values(["country_canon", "year"]).reset_index(drop=True)

    print(f"[build_maddison] {len(out):,} rescaled rows; "
          f"{len(no_ratio)} countries used global fallback: {sorted(no_ratio)[:10]}...")

    # Sanity: spot-check a few countries' rescaled vs WB values in overlap years
    print("\n[build_maddison] spot check (Maddison rescaled vs WB) at 1995:")
    for cc in ["korea, rep.", "bangladesh", "ghana", "india", "united states"]:
        sub = out[(out["country_canon"] == cc) & (out["year"] == 1995)]
        wb_val = wb_long[(wb_long["country_canon"] == cc) & (wb_long["year"] == 1995)]
        if len(sub) and len(wb_val):
            print(f"  {cc:30s}  maddison_rescaled=${sub.iloc[0]['gdppc_wb_eq']:>10,.0f}   "
                  f"wb=${wb_val.iloc[0]['gdppc_wb']:>10,.0f}")

    out[["country_canon", "year", "gdppc_wb_eq"]].to_csv(OUT_CSV, index=False)
    print(f"\n[build_maddison] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
