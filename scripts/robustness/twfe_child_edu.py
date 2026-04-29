"""
Reproduce Table A1 from "Education of Nations."

Table A1: Two-way fixed effects (country + year) regressions — child lower
secondary completion on parental education and log GDP per capita.
187 countries, 1975–2015, 5-year intervals (WCDE v3).

Year dummies absorb the global post-decolonization educational expansion.
The near-zero R² confirms that almost all within-country variation in
education over time is shared across countries (the global expansion),
not idiosyncratic.

Data sources:
  - Education: wcde/data/processed/lower_sec_both.csv (WCDE v3, completion %)
  - GDP: data/gdppercapita_us_inflation_adjusted.csv (World Bank, constant 2015 USD)

Output: Table A1 numbers (β coefficients, within R², observation counts).
"""

import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from _shared import (PROC, DATA, CHECKIN, REGIONS as NON_SOVEREIGN,
                     write_checkin, add_canonical_aliases)

# ── Constants ─────────────────────────────────────────────────────
PARENTAL_LAG = 25
OUTCOME_YEARS = list(range(1975, 2016, 5))

# ── Load data ─────────────────────────────────────────────────────
agg = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"), index_col="country")

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

        log_gdp = np.nan
        c_lower = country.lower()
        if c_lower in gdp_raw.index and str(y) in gdp_raw.columns:
            try:
                g = float(gdp_raw.loc[c_lower, str(y)])
                if g > 0:
                    log_gdp = np.log(g)
            except (ValueError, TypeError):
                pass

        rows.append({
            "country": country,
            "year": y,
            "child": child,
            "parent": parent,
            "log_gdp": log_gdp,
        })

panel = pd.DataFrame(rows)
print(f"Full panel: {len(panel)} obs, {panel['country'].nunique()} countries")
gdp_panel = panel.dropna(subset=["log_gdp"])
print(f"With GDP:   {len(gdp_panel)} obs, {gdp_panel['country'].nunique()} countries")

# ── Two-way FE regression: demean country + year, clustered SEs ──
def two_way_fe_regression(df, x_cols, y_col, cluster_col="country"):
    """
    Two-way FE: demean within country, then within year.
    Equivalent to including country + year dummies.
    Cluster-robust SEs by country.
    """
    d = df.dropna(subset=x_cols + [y_col]).copy()

    # Step 1: demean within country
    for col in x_cols + [y_col]:
        d[col + "_dm1"] = d.groupby(cluster_col)[col].transform(
            lambda x: x - x.mean()
        )

    # Step 2: demean within year (absorb year FE)
    for col in x_cols + [y_col]:
        d[col + "_dm2"] = d.groupby("year")[col + "_dm1"].transform(
            lambda x: x - x.mean()
        )

    X = d[[c + "_dm2" for c in x_cols]]
    y = d[y_col + "_dm2"]

    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": d[cluster_col]},
    )
    return model, len(d), d[cluster_col].nunique()

# ── Table A1 ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE A1. Two-way FE (country + year): child lower secondary completion")
print("=" * 70)

# Model (1): child ~ parent education (full panel)
m1, n1, nc1 = two_way_fe_regression(panel, ["parent"], "child")
print(f"\n(1) FE + year: child ~ parent_edu  [N={n1}, {nc1} countries]")
print(f"    β = {m1.params.iloc[0]:.3f}  (SE={m1.bse.iloc[0]:.3f}, p={m1.pvalues.iloc[0]:.4f})")
print(f"    R² (within) = {m1.rsquared:.3f}")

# Model (2): child ~ log GDP
m2, n2, nc2 = two_way_fe_regression(panel, ["log_gdp"], "child")
print(f"\n(2) FE + year: child ~ log_gdp  [N={n2}, {nc2} countries]")
print(f"    β = {m2.params.iloc[0]:.3f}  (SE={m2.bse.iloc[0]:.3f}, p={m2.pvalues.iloc[0]:.4f})")
print(f"    R² (within) = {m2.rsquared:.3f}")

# Model (3): child ~ parent + log GDP
m3, n3, nc3 = two_way_fe_regression(panel, ["parent", "log_gdp"], "child")
print(f"\n(3) FE + year: child ~ parent_edu + log_gdp  [N={n3}, {nc3} countries]")
print(f"    β_edu = {m3.params.iloc[0]:.3f}  (SE={m3.bse.iloc[0]:.3f}, p={m3.pvalues.iloc[0]:.4f})")
print(f"    β_gdp = {m3.params.iloc[1]:.3f}  (SE={m3.bse.iloc[1]:.3f}, p={m3.pvalues.iloc[1]:.4f})")
print(f"    R² (within) = {m3.rsquared:.3f}")

print("\n" + "=" * 70)
print("Numbers for paper:")
print("=" * 70)
print(f"  Table A1 Model (1): β={m1.params.iloc[0]:.3f}, R²={m1.rsquared:.3f}, N={n1}")
print(f"  Table A1 Model (2): β={m2.params.iloc[0]:.3f}, R²={m2.rsquared:.3f}, N={n2}")
print(f"  Table A1 Model (3): β_edu={m3.params.iloc[0]:.3f}, β_gdp={m3.params.iloc[1]:.3f}, R²={m3.rsquared:.3f}, N={n3}")

# ── Write checkin JSON ───────────────────────────────────────────
write_checkin("twfe_child_edu.json", {
    "numbers": {
        "panel_obs": n1,
        "panel_countries": nc1,
        "gdp_panel_obs": n2,
        "gdp_panel_countries": nc2,
        "ta1_m1_edu_beta": round(m1.params.iloc[0], 3),
        "ta1_m1_edu_se": round(m1.bse.iloc[0], 3),
        "ta1_m1_edu_p": round(m1.pvalues.iloc[0], 4),
        "ta1_m1_r2_within": round(m1.rsquared, 3),
        "ta1_m1_n": n1,
        "ta1_m2_gdp_beta": round(m2.params.iloc[0], 3),
        "ta1_m2_gdp_se": round(m2.bse.iloc[0], 3),
        "ta1_m2_gdp_p": round(m2.pvalues.iloc[0], 4),
        "ta1_m2_r2_within": round(m2.rsquared, 3),
        "ta1_m2_n": n2,
        "ta1_m3_edu_beta": round(m3.params.iloc[0], 3),
        "ta1_m3_gdp_beta": round(m3.params.iloc[1], 3),
        "ta1_m3_r2_within": round(m3.rsquared, 3),
        "ta1_m3_n": n3,
        "TA1-M1-beta": round(m1.params.iloc[0], 3),
        "TA1-M1-R2": round(m1.rsquared, 3),
    },
}, script_path="scripts/robustness/twfe_child_edu.py")
