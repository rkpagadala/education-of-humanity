# =============================================================================
# PAPER REFERENCE
# Script:  scripts/tables/table_1_stepwise.py
# Paper:   "Education of Humanity"
#
# Produces:
#   Table 1 stepwise specification. Four columns of increasing controls
#   on the same common sample so the parental-education coefficient is
#   directly comparable across columns.
#
#     (1) child ~ parent,                         C-FE
#     (2) child ~ parent + log_gdp,               C-FE
#     (3) child ~ parent + parent² + log_gdp,     C-FE
#     (4) child ~ parent + parent² + log_gdp,     C+Y-FE
#
#   - log_gdp: contemporaneous log GDP per capita (bad-control test).
#   - parent²: squared parental completion (functional-form check;
#              probes whether the linear specification understates the
#              effect at low baselines where β_g>1).
#
# Sample: parental completion below 30% (active expansion). Common
# sample across columns: every observation must have contemporaneous
# log GDP non-missing. Matches the 629-obs / 105-country active-expansion
# sample produced by scripts/residualization/by_gdp_cutoff.py at the
# 30% cutoff.
#
# Inputs:
#   wcde/data/processed/cohort_completion_both_long.csv
#   data/gdppercapita_us_inflation_adjusted.csv
#
# Output: checkin/table_1_stepwise.json
# =============================================================================
"""Stepwise Table 1: five columns of increasing controls, same sample."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from _shared import PROC, DATA, REGIONS, write_checkin  # noqa: E402

PARENTAL_LAG = 25
ACTIVE_EXPANSION_CUTOFF = 30
CHILD_COHORTS = list(range(1975, 2016, 5))

# ── Country-name bridge (WCDE → World Bank) ─────────────────────────────────
MANUAL_MAP = {
    "republic of korea": "south korea",
    "united states of america": "united states",
    "united kingdom of great britain and northern ireland": "united kingdom",
    "taiwan province of china": "taiwan",
    "hong kong special administrative region of china": "hong kong, china",
    "democratic people's republic of korea": "north korea",
    "iran (islamic republic of)": "iran",
    "bolivia (plurinational state of)": "bolivia",
    "venezuela (bolivarian republic of)": "venezuela",
    "lao people's democratic republic": "lao",
    "viet nam": "vietnam",
    "russian federation": "russia",
    "syrian arab republic": "syria",
    "republic of moldova": "moldova",
    "united republic of tanzania": "tanzania",
    "democratic republic of the congo": "congo, dem. rep.",
}


def load_panel():
    """Assemble the country-year panel with every variable needed by Col 5."""
    long = pd.read_csv(os.path.join(PROC, "cohort_completion_both_long.csv"))
    long = long[~long["country"].isin(REGIONS)]
    low_w = long.pivot(index="country", columns="cohort_year", values="lower_sec")

    gdp = pd.read_csv(os.path.join(DATA, "gdppercapita_us_inflation_adjusted.csv"))
    gdp_long = gdp.melt(id_vars="Country", var_name="year", value_name="gdp")
    gdp_long["year"] = pd.to_numeric(gdp_long["year"], errors="coerce")
    gdp_long = gdp_long.dropna(subset=["year", "gdp"])
    gdp_long["year"] = gdp_long["year"].astype(int)
    gdp_long["country_lc"] = gdp_long["Country"].str.lower().str.strip()
    gdp_lc_set = set(gdp_long["country_lc"].unique())

    wcde_to_lc = {}
    for wc in low_w.index:
        wl = wc.lower()
        if wl in gdp_lc_set:
            wcde_to_lc[wc] = wl
        elif wl in MANUAL_MAP:
            wcde_to_lc[wc] = MANUAL_MAP[wl]
        else:
            for gc in gdp_lc_set:
                if wl.split()[0] == gc.split()[0] and len(wl.split()[0]) > 4:
                    wcde_to_lc[wc] = gc
                    break

    def gdp_lookup(country_lc, year):
        if country_lc is None:
            return np.nan
        rows = gdp_long[(gdp_long["country_lc"] == country_lc) & (gdp_long["year"] == year)]
        if len(rows) == 0:
            return np.nan
        val = rows["gdp"].iloc[0]
        return np.log(val) if pd.notna(val) and val > 0 else np.nan

    def edu_lookup(country, year):
        try:
            val = float(low_w.loc[country, int(year)])
            return val if not np.isnan(val) else np.nan
        except (KeyError, ValueError):
            return np.nan

    rows = []
    for c in low_w.index:
        gdp_lc = wcde_to_lc.get(c)
        for child_yr in CHILD_COHORTS:
            parent_yr = child_yr - PARENTAL_LAG
            child_low = edu_lookup(c, child_yr)
            parent_low = edu_lookup(c, parent_yr)
            if np.isnan(child_low) or np.isnan(parent_low):
                continue
            rows.append({
                "country": c,
                "year": child_yr,
                "child": child_low,
                "parent": parent_low,
                "log_gdp": gdp_lookup(gdp_lc, child_yr),
            })
    df = pd.DataFrame(rows)
    df["parent_sq"] = df["parent"] ** 2
    return df


def fit_fe(d, x_cols, y_col, time_effects=False):
    """One-/two-way FE with country-clustered SEs via linearmodels.PanelOLS."""
    panel = d.set_index(["country", "year"])
    y = panel[y_col]
    X = panel[x_cols]
    mod = PanelOLS(y, X, entity_effects=True, time_effects=time_effects,
                   drop_absorbed=True, check_rank=False)
    return mod.fit(cov_type="clustered", cluster_entity=True)


def stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def _get(series, key, default=np.nan):
    return float(series[key]) if key in series.index else default


def main():
    panel = load_panel()
    print(f"Raw panel: {len(panel):4d} obs, {panel['country'].nunique():3d} countries")

    # Active-expansion subsample (parent <30%)
    active = panel[panel["parent"] < ACTIVE_EXPANSION_CUTOFF].copy()
    print(f"Parent < {ACTIVE_EXPANSION_CUTOFF}%: "
          f"{len(active):4d} obs, {active['country'].nunique():3d} countries")

    # Common sample: drop obs missing anything Col 4 needs (parent² is
    # derived from parent, so only raw covariates gate the sample).
    cov_cols = ["parent", "log_gdp"]
    common = active.dropna(subset=cov_cols + ["child"]).copy()
    # Need ≥2 obs per country for within-country demeaning
    counts = common.groupby("country").size()
    keep = counts[counts >= 2].index
    common = common[common["country"].isin(keep)].copy()
    print(f"Common sample: "
          f"{len(common):4d} obs, {common['country'].nunique():3d} countries")

    specs = [
        ("m1", ["parent"],                              False, "child ~ parent [C-FE]"),
        ("m2", ["parent", "log_gdp"],                   False, "+ log GDP"),
        ("m3", ["parent", "parent_sq", "log_gdp"],      False, "+ parent²"),
        ("m4", ["parent", "parent_sq", "log_gdp"],      True,  "+ year FE"),
    ]

    results = {}
    numbers = {
        "cutoff": ACTIVE_EXPANSION_CUTOFF,
        "parental_lag_years": PARENTAL_LAG,
    }

    print()
    print(f"{'Col':>3}  {'Spec':35s}  {'β_parent':>10}  {'SE':>7}  {'p':>6}  "
          f"{'β_gdp':>8}  {'R²_within':>10}  {'N':>5}  {'Ctry':>5}")
    print("-" * 108)

    for key, x_cols, year_fx, label in specs:
        res = fit_fe(common, x_cols, "child", time_effects=year_fx)
        params = res.params
        bse = res.std_errors
        pvals = res.pvalues
        rw = float(res.rsquared_within)
        n = int(res.nobs)
        n_ctry = int(res.entity_info.total)

        parent_b, parent_se, parent_p = (
            _get(params, "parent"), _get(bse, "parent"), _get(pvals, "parent"),
        )
        gdp_b, gdp_se, gdp_p = (
            _get(params, "log_gdp"), _get(bse, "log_gdp"), _get(pvals, "log_gdp"),
        )
        psq_b, psq_se, psq_p = (
            _get(params, "parent_sq"),
            _get(bse, "parent_sq"),
            _get(pvals, "parent_sq"),
        )

        print(f"{key:>3}  {label:35s}  {parent_b:10.3f}  {parent_se:7.3f}  "
              f"{parent_p:6.3f}  "
              f"{(gdp_b if not np.isnan(gdp_b) else 0):8.2f}  "
              f"{rw:10.3f}  {n:5d}  {n_ctry:5d}")

        results[key] = {
            "label": label,
            "parent_beta": round(parent_b, 3),
            "parent_se":   round(parent_se, 3),
            "parent_p":    round(parent_p, 4),
            "parent_stars": stars(parent_p),
            "gdp_beta":    None if np.isnan(gdp_b) else round(gdp_b, 2),
            "gdp_se":      None if np.isnan(gdp_se) else round(gdp_se, 2),
            "gdp_p":       None if np.isnan(gdp_p) else round(gdp_p, 4),
            "gdp_stars":   stars(gdp_p) if not np.isnan(gdp_p) else "",
            "parent_sq_beta":  None if np.isnan(psq_b) else round(psq_b, 4),
            "parent_sq_se":    None if np.isnan(psq_se) else round(psq_se, 4),
            "parent_sq_p":     None if np.isnan(psq_p) else round(psq_p, 4),
            "parent_sq_stars": stars(psq_p) if not np.isnan(psq_p) else "",
            "r2_within": round(rw, 3),
            "n": n,
            "countries": n_ctry,
            "country_fe": True,
            "year_fe": bool(year_fx),
        }

        for field, value in results[key].items():
            if field in ("label", "country_fe", "year_fe"):
                continue
            numbers[f"{key}_{field}"] = value

    numbers["sample_obs"]       = int(results["m1"]["n"])
    numbers["sample_countries"] = int(results["m1"]["countries"])

    write_checkin("table_1_stepwise.json", {
        "notes": (f"Stepwise Table 1. Parental education <{ACTIVE_EXPANSION_CUTOFF}% "
                  "(active expansion). Common sample across all four columns: "
                  "every observation must have contemporaneous log GDP "
                  "non-missing. Country-clustered SEs. Matches the 629-obs / "
                  "105-country sample in scripts/residualization/by_gdp_cutoff.py. "
                  "Produced by scripts/tables/table_1_stepwise.py."),
        "models": results,
        "numbers": numbers,
    }, script_path="scripts/tables/table_1_stepwise.py")
    print("\nDone.")


if __name__ == "__main__":
    main()
