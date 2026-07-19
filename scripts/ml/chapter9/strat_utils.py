# =============================================================================
# scripts/ml/chapter9/strat_utils.py
#
# Stratification helpers: country → region, country → income-tier masks.
# Used by transformer_strat jobs to subset the panel.
#
# Region buckets (UN M49 sub-regions, mapped to WB-style coarse regions):
#   sub_saharan_africa          : Sub-Saharan Africa
#   south_asia                  : Southern Asia
#   east_asia_pacific           : Eastern Asia + South-eastern Asia + Oceania
#   latin_america_caribbean     : Latin America and the Caribbean
#   europe_central_asia         : Europe + Central Asia
#   middle_east_north_africa    : Northern Africa + Western Asia
#
# Income tiers — based on each country's median GDP per capita over its
# panel-coverage years, split into thirds.
# =============================================================================
"""
strat_utils.py

Region + income membership masks for the panel data.
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(ML_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from _shared import DATA, standardize_country_name
EXTERNAL = os.path.join(DATA, "external")


# UN M49 sub-region → coarse region mapping
SUBREGION_TO_REGION = {
    "Sub-Saharan Africa":             "sub_saharan_africa",
    "Eastern Africa":                  "sub_saharan_africa",   # legacy code
    "Middle Africa":                   "sub_saharan_africa",
    "Southern Africa":                 "sub_saharan_africa",
    "Western Africa":                  "sub_saharan_africa",
    "Southern Asia":                   "south_asia",
    "Eastern Asia":                    "east_asia_pacific",
    "South-eastern Asia":              "east_asia_pacific",
    "Oceania":                         "east_asia_pacific",
    "Australia and New Zealand":       "east_asia_pacific",
    "Melanesia":                       "east_asia_pacific",
    "Micronesia":                      "east_asia_pacific",
    "Polynesia":                       "east_asia_pacific",
    "Latin America and the Caribbean": "latin_america_caribbean",
    "Caribbean":                       "latin_america_caribbean",
    "Central America":                 "latin_america_caribbean",
    "South America":                   "latin_america_caribbean",
    "Northern America":                "europe_central_asia",     # fold US/Canada into ECA for OECD bucket
    "Eastern Europe":                  "europe_central_asia",
    "Northern Europe":                 "europe_central_asia",
    "Southern Europe":                 "europe_central_asia",
    "Western Europe":                  "europe_central_asia",
    "Central Asia":                    "europe_central_asia",
    "Northern Africa":                 "middle_east_north_africa",
    "Western Asia":                    "middle_east_north_africa",
}


_CC_CACHE = None


def _load_country_to_region():
    """Build a canonical-country → coarse-region dict."""
    global _CC_CACHE
    if _CC_CACHE is not None:
        return _CC_CACHE
    df = pd.read_csv(os.path.join(EXTERNAL, "country_codes.csv"),
                      low_memory=False)
    out = {}
    for _, r in df.iterrows():
        subregion = r.get("Sub-region Name")
        intermediate = r.get("Intermediate Region Name")
        # Prefer intermediate region if it exists (gives Sub-Saharan Africa)
        if isinstance(intermediate, str) and intermediate.strip():
            key = intermediate.strip()
        elif isinstance(subregion, str) and subregion.strip():
            key = subregion.strip()
        else:
            continue
        region = SUBREGION_TO_REGION.get(key)
        if region is None:
            continue
        for name_col in ("official_name_en", "CLDR display name"):
            nm = r.get(name_col)
            if isinstance(nm, str) and nm.strip():
                c = standardize_country_name(nm)
                if c and c not in out:
                    out[c] = region
                    break
    _CC_CACHE = out
    return out


def country_region_mask(panel, region):
    """Boolean array over panel rows: True if that row's country is in `region`."""
    cmap = _load_country_to_region()
    region_country_ids = set()
    for cid, name in enumerate(panel["countries"]):
        if cmap.get(name) == region:
            region_country_ids.add(cid)
    return np.array([cid in region_country_ids for cid in panel["country_ids"]])


def country_income_mask(panel, tier):
    """Income tier mask using the panel's standardized GDP feature.

    Strategy: for each country, take the median of its observed GDP feature
    values across the panel window. Split countries into thirds based on
    median GDP. tier ∈ {"low", "lower_middle", "upper_middle_plus"}.

    The GDP feature is already standardized inside the panel so we work
    in standard-deviation space. We use sorted tertile cuts to ensure
    balanced groups.
    """
    gdp_indices = panel["feature_groups"].get("gdp", [])
    if not gdp_indices:
        raise RuntimeError("income mask requires gdp feature group in panel")
    gdp_idx = gdp_indices[0]
    n_countries = len(panel["countries"])
    medians = np.zeros(n_countries)
    for cid in range(n_countries):
        cmask = panel["country_ids"] == cid
        if cmask.sum() == 0:
            medians[cid] = np.nan
            continue
        # Mean over the country's panel rows × window, weighted by mask
        Xc = panel["X"][cmask, :, gdp_idx]
        Mc = panel["mask"][cmask]
        obs = Mc > 0
        if obs.sum() == 0:
            medians[cid] = np.nan
        else:
            medians[cid] = float(np.nanmedian(Xc[obs]))

    finite = ~np.isnan(medians)
    if finite.sum() == 0:
        raise RuntimeError("no countries with observed GDP")
    p33 = np.percentile(medians[finite], 100/3)
    p67 = np.percentile(medians[finite], 200/3)

    tier_ids = set()
    for cid in range(n_countries):
        m = medians[cid]
        if np.isnan(m):
            continue
        if tier == "low" and m <= p33:
            tier_ids.add(cid)
        elif tier == "lower_middle" and p33 < m <= p67:
            tier_ids.add(cid)
        elif tier == "upper_middle_plus" and m > p67:
            tier_ids.add(cid)
    return np.array([cid in tier_ids for cid in panel["country_ids"]])


def main():
    """Smoke test: print region counts and tier counts."""
    sys.path.insert(0, ML_DIR)
    from data_loader import load_panel
    panel = load_panel(mode="joint", verbose=False)
    print(f"panel: n={panel['X'].shape[0]}, "
          f"countries={len(panel['countries'])}")
    print("\nRegions:")
    for r in ("sub_saharan_africa", "south_asia", "east_asia_pacific",
              "latin_america_caribbean", "europe_central_asia",
              "middle_east_north_africa"):
        mask = country_region_mask(panel, r)
        cids = set(int(c) for c in panel["country_ids"][mask])
        countries = [panel["countries"][i] for i in sorted(cids)]
        print(f"  {r:<32} n_rows={mask.sum():>4}  "
              f"countries={len(countries)}")
    print("\nIncome tiers:")
    for t in ("low", "lower_middle", "upper_middle_plus"):
        mask = country_income_mask(panel, t)
        cids = set(int(c) for c in panel["country_ids"][mask])
        print(f"  {t:<24} n_rows={mask.sum():>4}  "
              f"countries={len(cids)}")


if __name__ == "__main__":
    main()
