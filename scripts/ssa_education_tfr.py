"""
ssa_education_tfr.py
====================
Sub-Saharan Africa-only test of the paper's primary -> TFR claim.

The literature flags that primary education's individual-level effect on
TFR is weak or null in early-transition settings (Cleland 2002 UN paper;
Jejeebhoy 1995 threshold result). The paper's claim is at the cohort/
generational level, not individual cross-section.  This script tests
whether the cohort-level relationship survives when we restrict the
panel to Sub-Saharan Africa, which is where the literature concern is
sharpest.

Three cuts:

  (1) SSA-only FE panel: education(T) -> TFR(T+25), country fixed
      effects, three education levels (primary, lower-sec, upper-sec).
      Compare R^2 to global benchmark from
      residualization/education_vs_tfr.py (global primary R^2 = 0.65).

  (2) SSA-only crossing distribution: for SSA countries that crossed
      TFR < 3.65, what was primary completion at the year of crossing?
      Compare to global clean-set median 79% / p10 57%.

  (3) SSA-only composition-by-level: which education level has the
      strongest cohort-level link to TFR within SSA?  Tests whether the
      literature's "lower-sec is the threshold" claim holds within SSA.

T = 1960..1990 (5yr step), lag = 25 years.  Same as global spec.

UN M49 Sub-Saharan Africa: Eastern + Middle + Southern + Western
Africa, 48 countries total.  Mayotte and Reunion (dependencies) are
omitted; Sudan is reclassified to Northern Africa by UN M49 since 2017
and so is excluded.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import (
    PROC, CHECKIN, TFR_THRESHOLD,
    load_education, load_wide_indicator, get_wb_val,
    interpolate_to_annual, completion_at_year, fe_regression,
    REGIONS, WB_TO_WCDE,
)


# ── SSA country list (WCDE naming) ───────────────────────────────────
# UN M49 Sub-Saharan Africa.  Names match completion_both_long.csv.
SSA_WCDE = {
    # Eastern Africa
    "Burundi", "Comoros", "Djibouti", "Eritrea", "Ethiopia", "Kenya",
    "Madagascar", "Malawi", "Mauritius", "Mozambique", "Rwanda",
    "Seychelles", "Somalia", "South Sudan", "Uganda",
    "United Republic of Tanzania", "Zambia", "Zimbabwe",
    # Middle Africa
    "Angola", "Cameroon", "Central African Republic", "Chad", "Congo",
    "Democratic Republic of the Congo", "Equatorial Guinea", "Gabon",
    "Sao Tome and Principe",
    # Southern Africa
    "Botswana", "Lesotho", "Namibia", "South Africa", "Swaziland",
    # Western Africa
    "Benin", "Burkina Faso", "Cape Verde", "Cote d'Ivoire", "Gambia",
    "Ghana", "Guinea", "Guinea-Bissau", "Liberia", "Mali", "Mauritania",
    "Niger", "Nigeria", "Senegal", "Sierra Leone", "Togo",
}

EDU_LEVELS = ("primary", "lower_sec", "upper_sec")
T_YEARS = list(range(1960, 1995, 5))
LAG = 25


# ── (1) + (3): FE panel regressions, three education levels ──────────

def build_panel(edu_annual, tfr_df, level_name):
    """Long panel: rows = country x T, with edu(T) and TFR(T+lag)."""
    rows = []
    for c in sorted(edu_annual.keys()):
        if c not in SSA_WCDE:
            continue
        s = edu_annual[c]
        for t in T_YEARS:
            if t not in s.index:
                continue
            edu_val = s[t]
            tfr_val = get_wb_val(tfr_df, c, t + LAG)
            if np.isnan(edu_val) or np.isnan(tfr_val):
                continue
            rows.append({
                "country": c,
                "t": t,
                "edu_t": float(edu_val),
                "tfr_tp25": float(tfr_val),
            })
    return pd.DataFrame(rows)


def run_panel():
    print("=" * 78)
    print("(1) + (3) SSA-ONLY FE PANEL: education(T) -> TFR(T+25)")
    print("=" * 78)

    edu_raw = load_education("completion_both_long.csv")
    tfr = load_wide_indicator("children_per_woman_total_fertility.csv")

    panel_results = {}
    print(f"{'Level':<14}  {'beta':>7}  {'SE':>6}  {'R2 within':>10}  "
          f"{'n obs':>5}  {'n countries':>11}")
    print("-" * 70)

    for level in EDU_LEVELS:
        edu_annual = interpolate_to_annual(edu_raw, level)
        panel = build_panel(edu_annual, tfr, level)
        if len(panel) == 0:
            print(f"{level:<14}  (no SSA observations)")
            continue
        model, n_obs, n_countries = fe_regression(
            panel, ["edu_t"], "tfr_tp25"
        )
        beta = float(model.params["edu_t_dm"])
        se = float(model.bse["edu_t_dm"])
        r2 = float(model.rsquared)
        print(f"{level:<14}  {beta:>7.4f}  {se:>6.4f}  {r2:>10.3f}  "
              f"{n_obs:>5}  {n_countries:>11}")
        panel_results[level] = {
            "beta": round(beta, 4),
            "se": round(se, 4),
            "r2_within": round(r2, 3),
            "n_obs": int(n_obs),
            "n_countries": int(n_countries),
        }

    print()
    print("Reference: global primary -> TFR(T+25) R^2 = 0.65")
    print("           (residualization/education_vs_tfr.py, "
          "entry=10%, ceiling=90%)")
    return panel_results


# ── (2) SSA-only crossing distribution ───────────────────────────────

def run_crossing():
    print()
    print("=" * 78)
    print("(2) SSA-ONLY CROSSING DISTRIBUTION: primary at year TFR<3.65")
    print("=" * 78)

    tfr = load_wide_indicator("children_per_woman_total_fertility.csv")
    prim = pd.read_csv(os.path.join(PROC, "primary_both.csv"),
                       index_col="country")
    prim.columns = prim.columns.astype(int)
    prim.index = [s.lower() for s in prim.index]

    # First year TFR < 3.65 per country
    first_cross = {}
    for yr in range(1960, 2023):
        yr_str = str(yr)
        if yr_str not in tfr.columns:
            continue
        tfr_y = tfr[yr_str].dropna()
        for c in tfr_y[tfr_y < TFR_THRESHOLD].index:
            if c not in first_cross:
                first_cross[c] = yr

    # Build SSA crosser table.  WCDE-side SSA names lowercased so they
    # match the lowercased TFR/prim indices.
    ssa_lc = {s.lower() for s in SSA_WCDE}
    # Add WB-name variants for SSA countries that differ between
    # registries (Cape Verde / Cabo Verde, Swaziland / Eswatini, etc.).
    ssa_lc |= {
        "cabo verde", "eswatini", "tanzania",
        "congo, rep.", "congo, dem. rep.",
        "gambia, the",
    }

    recs = []
    for wdi_lc, cross_y in first_cross.items():
        if wdi_lc not in ssa_lc:
            continue
        wcde_lc = WB_TO_WCDE.get(wdi_lc, wdi_lc)
        if wcde_lc in REGIONS:
            continue
        if wcde_lc not in prim.index:
            continue
        p_at = completion_at_year(prim, wcde_lc, cross_y)
        if pd.isna(p_at):
            continue
        recs.append({
            "country": wdi_lc,
            "crossing_year": cross_y,
            "primary_at_cross": float(p_at),
        })

    df = pd.DataFrame(recs).sort_values("primary_at_cross")
    if df.empty:
        print("No SSA crossers in TFR<3.65 set.")
        return {"n": 0}

    print(f"\nSSA countries that crossed TFR<3.65, n = {len(df)}")
    print()
    print("Country (cross year):  primary completion at cross")
    print("-" * 60)
    for _, r in df.iterrows():
        print(f"  {r['country']:<28} ({int(r['crossing_year'])})   "
              f"{r['primary_at_cross']:>5.1f}%")

    s = df["primary_at_cross"]
    summary = {
        "n": int(len(df)),
        "min": round(float(s.min()), 1),
        "p10": round(float(s.quantile(0.10)), 1),
        "p25": round(float(s.quantile(0.25)), 1),
        "median": round(float(s.median()), 1),
        "p75": round(float(s.quantile(0.75)), 1),
        "p90": round(float(s.quantile(0.90)), 1),
        "max": round(float(s.max()), 1),
    }
    print()
    print(f"  min    = {summary['min']:>5.1f}%")
    print(f"  p10    = {summary['p10']:>5.1f}%")
    print(f"  p25    = {summary['p25']:>5.1f}%")
    print(f"  median = {summary['median']:>5.1f}%")
    print(f"  p75    = {summary['p75']:>5.1f}%")
    print(f"  p90    = {summary['p90']:>5.1f}%")
    print(f"  max    = {summary['max']:>5.1f}%")
    print()
    print("Reference: global clean set median = 79.2%, p10 = 57.4%, "
          "n = 88 (primary_at_tfr_crossing.py)")
    return summary


# ── Main ─────────────────────────────────────────────────────────────

def main():
    panel_results = run_panel()
    crossing_summary = run_crossing()

    checkin = {
        "method": (
            "Sub-Saharan Africa restriction (UN M49, 48 countries) of "
            "the paper's primary -> TFR claim. Three cuts: (1) FE panel, "
            "edu(T) -> TFR(T+25), three education levels (primary, "
            "lower-sec, upper-sec); (2) primary completion at TFR<3.65 "
            "crossing year for SSA crossers; (3) composition-by-level "
            "comparison within SSA. T=1960-1990 (5yr step), lag=25."
        ),
        "ssa_n_countries": len(SSA_WCDE),
        "panel_by_level": panel_results,
        "crossing": crossing_summary,
    }
    out_path = os.path.join(CHECKIN, "ssa_education_tfr.json")
    with open(out_path, "w") as f:
        json.dump(checkin, f, indent=2)
    print()
    print(f"Checkin written to {out_path}")


if __name__ == "__main__":
    main()
