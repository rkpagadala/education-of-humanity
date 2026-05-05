# =============================================================================
# PAPER REFERENCE
# Script:  scripts/tables/table_1_subgroups.py
# Paper:   "Education of Humanity"
#
# Produces:
#   Subgroup-robustness table for the headline Column 1 specification
#   of Table 1 (child lower-sec ~ parental, country FE, active-expansion
#   <30% cutoff). Splits the 105-country / 629-obs sample by:
#
#     - World Bank region (SSA / MENA / South Asia / East Asia & Pacific /
#                          Latin America & Caribbean / Europe & Central Asia)
#     - Time period (pre-1990 vs 1990-onward child cohorts)
#     - GDP tercile at the child-cohort year
#
#   Answers: is the headline β driven by a particular group of countries,
#   a particular era, or a particular income band?
#
# Inputs:
#   wcde/data/processed/cohort_completion_both_long.csv
#   data/gdppercapita_us_inflation_adjusted.csv
#
# Output: checkin/table_1_subgroups.json
# =============================================================================
"""Subgroup robustness for Table 1 headline specification."""

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

# ── Country name bridge (WCDE → WB) ─────────────────────────────────────────
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

# ── WB-style region classification (WCDE country names) ─────────────────────
# Roughly tracks World Bank's 7 regions; North America folded into
# "Europe & Central Asia + North America" because n=2 (Canada, US).
REGION_MAP = {
    "SSA": {  # Sub-Saharan Africa (incl. Reunion/Mauritius Indian-Ocean)
        "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon",
        "Cape Verde", "Central African Republic", "Chad", "Comoros", "Congo",
        "Cote d'Ivoire", "Democratic Republic of the Congo",
        "Equatorial Guinea", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
        "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Madagascar", "Malawi",
        "Mali", "Mauritius", "Mozambique", "Namibia", "Niger", "Nigeria",
        "Reunion", "Rwanda", "Sao Tome and Principe", "Senegal",
        "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan",
        "Swaziland", "Togo", "Uganda", "United Republic of Tanzania", "Zambia",
        "Zimbabwe",
    },
    "MENA": {  # Middle East & North Africa
        "Algeria", "Bahrain", "Egypt", "Iran (Islamic Republic of)", "Iraq",
        "Israel", "Jordan", "Kuwait", "Lebanon", "Morocco",
        "Occupied Palestinian Territory", "Oman", "Qatar", "Saudi Arabia",
        "Syrian Arab Republic", "Tunisia", "Turkey",
        "United Arab Emirates", "Yemen",
    },
    "South Asia": {
        "Afghanistan", "Bangladesh", "Bhutan", "India", "Maldives", "Nepal",
        "Pakistan", "Sri Lanka",
    },
    "East Asia & Pacific": {
        "Australia", "Cambodia", "China",
        "Democratic People's Republic of Korea", "Fiji", "French Polynesia",
        "Hong Kong Special Administrative Region of China", "Indonesia",
        "Japan", "Kiribati", "Lao People's Democratic Republic",
        "Macao Special Administrative Region of China", "Malaysia",
        "Micronesia (Federated States of)", "Mongolia", "Myanmar",
        "New Caledonia", "New Zealand", "Philippines", "Republic of Korea",
        "Samoa", "Singapore", "Solomon Islands", "Taiwan Province of China",
        "Thailand", "Timor-Leste", "Tonga", "Vanuatu", "Viet Nam",
    },
    "LAC": {  # Latin America & Caribbean
        "Argentina", "Aruba", "Bahamas", "Belize",
        "Bolivia (Plurinational State of)", "Brazil", "Chile", "Colombia",
        "Costa Rica", "Cuba", "Curaçao", "Dominican Republic", "Ecuador",
        "El Salvador", "French Guiana", "Guadeloupe", "Guatemala", "Guyana",
        "Haiti", "Honduras", "Jamaica", "Martinique", "Mexico", "Nicaragua",
        "Panama", "Paraguay", "Peru", "Puerto Rico", "Saint Lucia",
        "Saint Vincent and the Grenadines", "Suriname", "Trinidad and Tobago",
        "Uruguay", "Venezuela (Bolivarian Republic of)",
    },
    "Europe & N.America": {
        "Albania", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium",
        "Bosnia and Herzegovina", "Bulgaria", "Canada", "Croatia", "Cyprus",
        "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Georgia",
        "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy",
        "Kazakhstan", "Kyrgyzstan", "Latvia", "Lithuania", "Luxembourg",
        "Malta", "Montenegro", "Netherlands", "Norway", "Poland", "Portugal",
        "Republic of Moldova", "Romania", "Russian Federation", "Serbia",
        "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Tajikistan",
        "The former Yugoslav Republic of Macedonia", "Turkmenistan", "Ukraine",
        "United Kingdom of Great Britain and Northern Ireland",
        "United States of America",
    },
}


def region_of(country):
    for r, members in REGION_MAP.items():
        if country in members:
            return r
    return None


def load_panel():
    """Assemble the 30%-cutoff active-expansion panel."""
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
        reg = region_of(c)
        for child_yr in CHILD_COHORTS:
            parent_yr = child_yr - PARENTAL_LAG
            child_low = edu_lookup(c, child_yr)
            parent_low = edu_lookup(c, parent_yr)
            if np.isnan(child_low) or np.isnan(parent_low):
                continue
            rows.append({
                "country": c,
                "region": reg,
                "year": child_yr,
                "child": child_low,
                "parent": parent_low,
                "log_gdp": gdp_lookup(gdp_lc, child_yr),
            })
    return pd.DataFrame(rows)


def fit_parent_only(df):
    """Col-1 headline spec: child ~ parent, country FE, clustered SEs."""
    d = df.dropna(subset=["parent", "child"]).copy()
    # Need ≥2 obs per country for within-country FE
    counts = d.groupby("country").size()
    d = d[d["country"].isin(counts[counts >= 2].index)]
    if len(d) < 10 or d["country"].nunique() < 3:
        return None
    panel = d.set_index(["country", "year"])
    mod = PanelOLS(panel["child"], panel[["parent"]],
                   entity_effects=True, drop_absorbed=True, check_rank=False)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return {
        "beta": float(res.params["parent"]),
        "se":   float(res.std_errors["parent"]),
        "p":    float(res.pvalues["parent"]),
        "r2":   float(res.rsquared_within),
        "n":    int(res.nobs),
        "countries": int(res.entity_info.total),
    }


def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def fmt_row(label, r):
    if r is None:
        return f"{label:30s}  {'—':>8} {'—':>8} {'—':>6} {'—':>8} {'—':>5} {'—':>5}"
    return (f"{label:30s}  {r['beta']:8.3f} {r['se']:8.3f} {r['p']:6.3f} "
            f"{r['r2']:8.3f} {r['n']:5d} {r['countries']:5d}")


def main():
    panel = load_panel()
    # Drop any country we haven't assigned a region (should be 0 if mapping is complete)
    unassigned = sorted(panel[panel["region"].isna()]["country"].unique())
    if unassigned:
        print(f"WARNING: {len(unassigned)} unassigned countries: {unassigned}")
    print(f"Raw panel: {len(panel):4d} obs, {panel['country'].nunique():3d} countries")

    # Active-expansion headline sample (parent < 30%, log_gdp present to match Table 1)
    active = panel[panel["parent"] < ACTIVE_EXPANSION_CUTOFF].copy()
    active = active.dropna(subset=["log_gdp"])
    print(f"Active-expansion <{ACTIVE_EXPANSION_CUTOFF}%, GDP present: "
          f"{len(active):4d} obs, {active['country'].nunique():3d} countries")

    # ── Headline on the common sample (should match Table 1 Col 1) ──
    results = {}
    print()
    print(f"{'Subset':30s}  {'β':>8} {'SE':>8} {'p':>6} "
          f"{'R²':>8} {'N':>5} {'Ctry':>5}")
    print("-" * 80)
    hl = fit_parent_only(active)
    print(fmt_row("FULL (Table 1 Col 1)", hl))
    results["headline"] = hl

    # ── By region ──
    print()
    print("By WB region:")
    by_region = {}
    for reg in ["SSA", "MENA", "South Asia", "East Asia & Pacific",
                "LAC", "Europe & N.America"]:
        sub = active[active["region"] == reg]
        r = fit_parent_only(sub)
        by_region[reg] = r
        print(fmt_row(f"  {reg}", r))
    results["by_region"] = by_region

    # ── By period (child-cohort year) ──
    print()
    print("By period (child cohort year):")
    pre = active[active["year"] < 1990]
    post = active[active["year"] >= 1990]
    r_pre  = fit_parent_only(pre)
    r_post = fit_parent_only(post)
    print(fmt_row("  Pre-1990 (1975-1985)", r_pre))
    print(fmt_row("  1990-2015",            r_post))
    results["by_period"] = {"pre_1990": r_pre, "post_1990": r_post}

    # ── By GDP tercile ──
    # Tercile computed on the active-expansion sample's log_gdp pooled.
    print()
    print("By GDP tercile (within active-expansion sample):")
    q33, q67 = active["log_gdp"].quantile([1/3, 2/3]).values
    tercile_def = {
        "low":    active[active["log_gdp"] <  q33],
        "middle": active[(active["log_gdp"] >= q33) & (active["log_gdp"] < q67)],
        "high":   active[active["log_gdp"] >= q67],
    }
    by_tercile = {}
    for k, sub in tercile_def.items():
        r = fit_parent_only(sub)
        by_tercile[k] = r
        print(fmt_row(f"  GDP tercile: {k}", r))
    results["by_gdp_tercile"] = {"q33": round(float(q33), 3),
                                 "q67": round(float(q67), 3),
                                 **by_tercile}

    # ── Write checkin ──
    numbers = {
        "headline_beta":     hl["beta"],
        "headline_n":        hl["n"],
        "headline_countries": hl["countries"],
    }
    for reg, r in by_region.items():
        tag = reg.replace(" ", "").replace(".", "").replace("&", "")
        if r:
            numbers[f"region_{tag}_beta"]     = round(r["beta"], 3)
            numbers[f"region_{tag}_se"]       = round(r["se"], 3)
            numbers[f"region_{tag}_p"]        = round(r["p"], 4)
            numbers[f"region_{tag}_r2"]       = round(r["r2"], 3)
            numbers[f"region_{tag}_n"]        = r["n"]
            numbers[f"region_{tag}_countries"] = r["countries"]
    for period, r in results["by_period"].items():
        if r:
            numbers[f"{period}_beta"]     = round(r["beta"], 3)
            numbers[f"{period}_se"]       = round(r["se"], 3)
            numbers[f"{period}_n"]        = r["n"]
            numbers[f"{period}_countries"] = r["countries"]
            numbers[f"{period}_r2"]       = round(r["r2"], 3)
    for tercile in ("low", "middle", "high"):
        r = by_tercile[tercile]
        if r:
            numbers[f"gdp_{tercile}_beta"]     = round(r["beta"], 3)
            numbers[f"gdp_{tercile}_se"]       = round(r["se"], 3)
            numbers[f"gdp_{tercile}_n"]        = r["n"]
            numbers[f"gdp_{tercile}_countries"] = r["countries"]
            numbers[f"gdp_{tercile}_r2"]       = round(r["r2"], 3)

    write_checkin("table_1_subgroups.json", {
        "notes": (f"Table 1 Column 1 spec (child ~ parent, country FE, "
                  f"clustered SE) re-estimated on subsamples of the "
                  f"{hl['n']}-obs / {hl['countries']}-country active-expansion "
                  f"sample (<{ACTIVE_EXPANSION_CUTOFF}% parental completion, "
                  f"contemporaneous GDP present). Produced by "
                  f"scripts/tables/table_1_subgroups.py."),
        "numbers": numbers,
        "results": results,
    }, script_path="scripts/tables/table_1_subgroups.py")
    print("\nDone.")


if __name__ == "__main__":
    main()
