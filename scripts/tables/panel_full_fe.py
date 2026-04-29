"""
Full-panel one-way FE diagnostic (child lower secondary completion on
parental education and log GDP per capita).

Full WCDE panel: 1,665 country-years, 185 countries, 1975-2015, 5-year
intervals. These are the coefficients cited in the paper's year-FE
discussion and footnotes (β=0.483 education-only; Parental Income
collapse test PI-alone/PI-cond). Also reports the female completion
robustness check.

NOTE: The headline Table 1 shown in the paper (β=1.376, GDP β=13.659,
629 country-years, 105 countries) uses the <30% active-expansion
subsample and is produced by scripts/residualization/by_gdp_cutoff.py,
not this script.

Data sources:
  - Education: wcde/data/processed/lower_sec_both.csv (WCDE v3, completion %)
  - Female education: wcde/data/processed/lower_sec_female.csv
  - GDP: data/gdppercapita_us_inflation_adjusted.csv (World Bank, constant 2015 USD)

Output: checkin/panel_full_fe.json
"""

import os
import sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from _shared import (PROC, DATA, CHECKIN, REGIONS as NON_SOVEREIGN,
                     write_checkin, fe_regression, add_canonical_aliases)

# ── Constants ─────────────────────────────────────────────────────
PARENTAL_LAG = 25                    # one PTE generation
OUTCOME_YEARS = list(range(1975, 2016, 5))  # 9 five-year points

# ── Load data ─────────────────────────────────────────────────────
agg = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"), index_col="country")
fem = pd.read_csv(os.path.join(PROC, "lower_sec_female.csv"), index_col="country")

gdp_raw = pd.read_csv(os.path.join(DATA, "gdppercapita_us_inflation_adjusted.csv"),
                       index_col="Country")
gdp_raw.index = gdp_raw.index.str.lower()
gdp_raw = add_canonical_aliases(gdp_raw)

# ── Build panel ───────────────────────────────────────────────────
rows = []
for country in agg.index:
    if country in NON_SOVEREIGN:
        continue
    for y in OUTCOME_YEARS:
        y_lag = y - PARENTAL_LAG
        sy, sy_lag = str(y), str(y_lag)

        if sy not in agg.columns or sy_lag not in agg.columns:
            continue
        child = agg.loc[country, sy]
        parent = agg.loc[country, sy_lag]
        if np.isnan(child) or np.isnan(parent):
            continue

        parent_f = np.nan
        if country in fem.index and sy_lag in fem.columns:
            parent_f = fem.loc[country, sy_lag]

        # GDP lookup (country names differ between WCDE and World Bank)
        log_gdp = np.nan
        log_gdp_parent = np.nan
        c_lower = country.lower()
        if c_lower in gdp_raw.index:
            if str(y) in gdp_raw.columns:
                try:
                    g = float(gdp_raw.loc[c_lower, str(y)])
                    if g > 0:
                        log_gdp = np.log(g)
                except (ValueError, TypeError):
                    pass
            if sy_lag in gdp_raw.columns:
                try:
                    g = float(gdp_raw.loc[c_lower, sy_lag])
                    if g > 0:
                        log_gdp_parent = np.log(g)
                except (ValueError, TypeError):
                    pass

        rows.append({
            "country": country,
            "year": y,
            "child": child,
            "parent": parent,
            "parent_f": parent_f,
            "log_gdp": log_gdp,
            "log_gdp_parent": log_gdp_parent,
        })

panel = pd.DataFrame(rows)
print(f"Full panel: {len(panel)} obs, {panel['country'].nunique()} countries")

# ── Table 1 ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 1. Country FE regressions: child lower secondary completion")
print("=" * 70)

# Model (1): child ~ parent education
m1, n1, nc1 = fe_regression(panel, ["parent"], "child")
print(f"\n(1) child ~ parent_edu  [N={n1}, {nc1} countries]")
print(f"    β = {m1.params.iloc[0]:.3f}  (SE={m1.bse.iloc[0]:.3f}, p={m1.pvalues.iloc[0]:.4f})")
print(f"    R² (within) = {m1.rsquared:.3f}")

# Model (2): child ~ log GDP
m2, n2, nc2 = fe_regression(panel, ["log_gdp"], "child")
print(f"\n(2) child ~ log_gdp  [N={n2}, {nc2} countries]")
print(f"    β = {m2.params.iloc[0]:.3f}  (SE={m2.bse.iloc[0]:.3f}, p={m2.pvalues.iloc[0]:.4f})")
print(f"    R² (within) = {m2.rsquared:.3f}")

# Model (3): child ~ parent + log GDP
m3, n3, nc3 = fe_regression(panel, ["parent", "log_gdp"], "child")
print(f"\n(3) child ~ parent_edu + log_gdp  [N={n3}, {nc3} countries]")
print(f"    β_edu = {m3.params.iloc[0]:.3f}  (SE={m3.bse.iloc[0]:.3f}, p={m3.pvalues.iloc[0]:.4f})")
print(f"    β_gdp = {m3.params.iloc[1]:.3f}  (SE={m3.bse.iloc[1]:.3f}, p={m3.pvalues.iloc[1]:.4f})")
print(f"    R² (within) = {m3.rsquared:.3f}")

# ── Female specification robustness (Table 1 footnote) ────────────
print("\n" + "=" * 70)
print("FEMALE SPECIFICATION ROBUSTNESS (same-sample comparison)")
print("=" * 70)

both = panel.dropna(subset=["parent", "parent_f", "child"])
print(f"\nSame-sample panel: {len(both)} obs, {both['country'].nunique()} countries")

m_agg, n_agg, _ = fe_regression(both, ["parent"], "child")
m_fem, n_fem, _ = fe_regression(both, ["parent_f"], "child")

print(f"\n  Aggregate: β = {m_agg.params.iloc[0]:.3f}  R² = {m_agg.rsquared:.3f}  [N={n_agg}]")
print(f"  Female:    β = {m_fem.params.iloc[0]:.3f}  R² = {m_fem.rsquared:.3f}  [N={n_fem}]")
print(f"  Δβ = {m_fem.params.iloc[0] - m_agg.params.iloc[0]:+.3f}")
print(f"  ΔR² = {m_fem.rsquared - m_agg.rsquared:+.3f}")
if m_fem.rsquared < m_agg.rsquared:
    print("  Female is WEAKER (as expected: PTE is a household mechanism)")
else:
    print("  Female is STRONGER")

print("\n" + "=" * 70)
print("Numbers for paper:")
print("=" * 70)
print(f"  Table 1 Model (1): β={m1.params.iloc[0]:.3f}, R²={m1.rsquared:.3f}")
print(f"  Table 1 Model (2): β={m2.params.iloc[0]:.3f}, R²={m2.rsquared:.3f}")
print(f"  Table 1 Model (3): β_edu={m3.params.iloc[0]:.3f}, β_gdp={m3.params.iloc[1]:.3f}, R²={m3.rsquared:.3f}")
print(f"  Footnote: female β={m_fem.params.iloc[0]:.3f}, R²={m_fem.rsquared:.3f}")
print(f"            vs agg  β={m_agg.params.iloc[0]:.3f}, R²={m_agg.rsquared:.3f} (same sample)")

# ── Parental income test (GDP alone → child edu, with FE) ────────
print("\n" + "=" * 70)
print("Parental income collapse test")
print("=" * 70)

gdp_sub = panel.dropna(subset=["log_gdp_parent"]).copy()
# GDP alone → child edu
m_pi1, n_pi1, nc_pi1 = fe_regression(gdp_sub, ["log_gdp_parent"], "child")
# Edu alone on GDP subsample
m_pi2, n_pi2, nc_pi2 = fe_regression(gdp_sub, ["parent"], "child")
# Both
m_pi3, n_pi3, nc_pi3 = fe_regression(gdp_sub, ["parent", "log_gdp_parent"], "child")

print(f"  GDP alone:  β={m_pi1.params.iloc[0]:.1f}, R²={m_pi1.rsquared:.3f}")
print(f"  Edu alone:  β={m_pi2.params.iloc[0]:.3f}, R²={m_pi2.rsquared:.3f}")
print(f"  Both:       β_edu={m_pi3.params.iloc[0]:.3f}, β_gdp={m_pi3.params.iloc[1]:.1f} (p={m_pi3.pvalues.iloc[1]:.2f}), R²={m_pi3.rsquared:.3f}")

# ── Write checkin JSON ───────────────────────────────────────────
write_checkin("panel_full_fe.json", {
    "numbers": {
        "panel_obs": n1,
        "panel_countries": nc1,
        "table1_m1_edu_beta": round(m1.params.iloc[0], 3),
        "table1_m1_edu_se": round(m1.bse.iloc[0], 3),
        "table1_m1_r2_within": round(m1.rsquared, 3),
        "table1_m1_n": n1,
        "table1_m1_countries": nc1,
        "table1_m2_gdp_beta": round(m2.params.iloc[0], 3),
        "table1_m2_gdp_se": round(m2.bse.iloc[0], 3),
        "table1_m2_r2_within": round(m2.rsquared, 3),
        "table1_m2_n": n2,
        "table1_m2_countries": nc2,
        "table1_m3_edu_beta": round(m3.params.iloc[0], 3),
        "table1_m3_gdp_beta": round(m3.params.iloc[1], 3),
        "table1_m3_r2_within": round(m3.rsquared, 3),
        "table1_m3_n": n3,
        "table1_m3_countries": nc3,
        "footnote_female_beta": round(m_fem.params.iloc[0], 3),
        "footnote_female_r2": round(m_fem.rsquared, 3),
        "footnote_agg_beta": round(m_agg.params.iloc[0], 3),
        "footnote_agg_r2": round(m_agg.rsquared, 3),
        "footnote_delta_beta": round(m_fem.params.iloc[0] - m_agg.params.iloc[0], 3),
        "footnote_delta_r2": round(m_fem.rsquared - m_agg.rsquared, 3),
        "T1-obs": n1,
        "T1-countries": nc1,
        "PI-alone-beta": round(m_pi1.params.iloc[0], 1),
        "PI-alone-R2": round(m_pi1.rsquared, 3),
        "PI-cond-beta": round(m_pi3.params.iloc[1], 1),
        "PI-cond-p": round(m_pi3.pvalues.iloc[1], 2),
        "PI-edu-alone": round(m_pi2.params.iloc[0], 3),
        "PI-edu-cond": round(m_pi3.params.iloc[0], 3),
        "PI-joint-R2": round(m_pi3.rsquared, 3),
        "PI-cond-R2": round(m_pi3.rsquared - m_pi2.rsquared, 3),
    },
}, script_path="scripts/tables/panel_full_fe.py")
