"""
robustness/u5mr_residual_by_year.py
========================
Verify paper claim: residualized GDP → U5MR p-value changes over time.

Lutz argues that at low education levels, recent health interventions
(vaccines, oral rehydration, bed nets) since the MDG era genuinely
reduce child mortality independent of domestic education.

Test: split the residualization analysis by outcome year cutoff.
For each cutoff year Y, restrict to observations where T+lag <= Y,
then report residualized GDP R² and p-value. If Lutz is right,
the p-value should be high (no GDP signal) for early cutoffs and
drop as post-2000 MDG-era outcomes enter the sample.

Uses clustered standard errors (matching tables/regression_tables.py).
Entry-cohort design (entry >= 10%, ceiling <= 90%), country FE,
lower-secondary completion. T runs 1960-2000 (extended past the usual
1990 cutoff; see note below) and U5MR is read at the childrearing window
lag (LAG_CHILDREARING=12).
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from residualization._shared import (
    load_education, load_wb, interpolate_to_annual, precompute_entry_years,
    build_panel, filter_panel, fe_residualize_gdp, clustered_fe,
    write_checkin,
)
from _shared import fmt_r2, LAG_GENERATION, LAG_CHILDREARING

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Parameters (match Table 2b) ────────────────────────────────────

# This by-outcome-year diagnostic extends T past the usual 1990 cutoff so
# that, at the childrearing lag (12), the post-2000 outcome window still
# exists (T=2000 -> outcome 2012). The forward-prediction tables stop at
# T=1990 because their 28-year lag already reaches 2015.
T_YEARS = list(range(1960, 2005, 5))
# U5MR registers at the childrearing window (LAG_CHILDREARING=12), not the
# time-to-agency generation lag. The explicit 0/12/28 bootstrap sweep below
# still reports all three timescales.
LAG = LAG_CHILDREARING
COL_NAME = "lower_sec"
CEILING = 90
ENTRY_THRESHOLD = 10


# ── Load data ───────────────────────────────────────────────────────

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
u5mr_df = load_wb("child_mortality_u5.csv")

edu_annual = interpolate_to_annual(edu_raw, COL_NAME)
entry_years = precompute_entry_years(edu_annual)
cohort = entry_years[ENTRY_THRESHOLD]

# ── Build full panel ────────────────────────────────────────────────

panel = build_panel(edu_annual, u5mr_df, gdp_df, T_YEARS, LAG, "u5mr")
sub_full = filter_panel(panel, cohort, CEILING)

print(f"Full panel: {len(sub_full)} obs, {sub_full['country'].nunique()} countries")
print(f"Outcome years range: {(sub_full['t'] + LAG).min()} to {(sub_full['t'] + LAG).max()}")

# ── Verify against tables/regression_tables.py (full sample) ──────────────

print("\n── Verification: full sample should match Table 2b ──")
resid_full = fe_residualize_gdp(sub_full)
if resid_full is not None:
    sub_r_full, _ = resid_full
    res = clustered_fe("gdp_resid", "u5mr", sub_r_full)
    if res:
        print(f"  Resid R²={res['r2']:.3f}  p={res['pval']:.4f}  "
              f"n={res['n']}  countries={res['countries']}")
        print(f"  (full-sample residualised-GDP R² at lag={LAG}, "
              f"childrearing window)")

# ── Sweep by outcome year cutoff ────────────────────────────────────

print("\n" + "=" * 90)
print("RESIDUALIZED GDP → U5MR: SWEEP BY OUTCOME YEAR CUTOFF (CLUSTERED SEs)")
print("=" * 90)
print(f"Entry >= {ENTRY_THRESHOLD}%, ceiling <= {CEILING}%, "
      f"T={T_YEARS[0]}-{T_YEARS[-1]}, lag={LAG}")
print()
print(f"{'Outcome ≤':<12} {'Edu R²':>7} {'Edu p':>8} {'Raw GDP R²':>11} "
      f"{'Resid R²':>9} {'Resid p':>9} {'n':>5} {'Ctry':>5}")
print("-" * 75)

results = []

for cutoff in range(1990, 2021, 5):
    mask = (sub_full["t"] + LAG) <= cutoff
    sub = sub_full[mask].copy()

    if len(sub) < 10 or sub["country"].nunique() < 3:
        continue

    # Education
    res_e = clustered_fe("edu_t", "u5mr", sub)

    # Raw GDP
    res_g = clustered_fe("log_gdp_t", "u5mr", sub)

    # Residualized GDP
    resid = fe_residualize_gdp(sub)
    res_r = None
    if resid is not None:
        sub_r, edu_gdp_r2 = resid
        res_r = clustered_fe("gdp_resid", "u5mr", sub_r)

    def fmtp(v):
        return f"{v:.4f}" if v is not None and not np.isnan(v) else "n/a"

    r2_e = res_e["r2"] if res_e else np.nan
    p_e = res_e["pval"] if res_e else np.nan
    r2_g = res_g["r2"] if res_g else np.nan
    r2_r = res_r["r2"] if res_r else np.nan
    p_r = res_r["pval"] if res_r else np.nan
    n = res_e["n"] if res_e else 0
    ctry = res_e["countries"] if res_e else 0

    print(f"  ≤ {cutoff:<7} {fmt_r2(r2_e):>7} {fmtp(p_e):>8} {fmt_r2(r2_g):>11} "
          f"{fmt_r2(r2_r):>9} {fmtp(p_r):>9} {n:>5} {ctry:>5}")

    results.append({
        "outcome_year_cutoff": cutoff,
        "edu_r2": round(r2_e, 4) if not np.isnan(r2_e) else None,
        "edu_pval": round(p_e, 4) if not np.isnan(p_e) else None,
        "raw_gdp_r2": round(r2_g, 4) if not np.isnan(r2_g) else None,
        "resid_gdp_r2": round(r2_r, 4) if not np.isnan(r2_r) else None,
        "resid_pvalue": round(p_r, 4) if not np.isnan(p_r) else None,
        "n_obs": n,
        "n_countries": ctry,
    })

# ── Before / after 2000 split ──────────────────────────────────────

print("\n" + "=" * 90)
print("BEFORE vs AFTER 2000 (MDG ERA) — CLUSTERED SEs")
print("=" * 90)

split_results = {}

for label, mask in [
    ("Before 2000", (sub_full["t"] + LAG) < 2000),
    ("After 2000",  (sub_full["t"] + LAG) >= 2000),
    ("All years",   pd.Series(True, index=sub_full.index)),
]:
    sub = sub_full[mask].copy()
    if len(sub) < 10 or sub["country"].nunique() < 3:
        print(f"  {label}: insufficient data")
        continue

    res_e = clustered_fe("edu_t", "u5mr", sub)

    resid = fe_residualize_gdp(sub)
    res_r = None
    if resid is not None:
        sub_r, _ = resid
        res_r = clustered_fe("gdp_resid", "u5mr", sub_r)

    r2_e = res_e["r2"] if res_e else np.nan
    r2_r = res_r["r2"] if res_r else np.nan
    p_r = res_r["pval"] if res_r else np.nan
    n = res_e["n"] if res_e else 0
    ctry = res_e["countries"] if res_e else 0

    def fmtp(v):
        return f"{v:.4f}" if v is not None and not np.isnan(v) else "n/a"

    print(f"  {label:<15}  Edu R²={fmt_r2(r2_e)}  Resid R²={fmt_r2(r2_r)}  "
          f"p={fmtp(p_r)}  n={n}  countries={ctry}")

    split_results[label] = {
        "edu_r2": round(r2_e, 4) if not np.isnan(r2_e) else None,
        "resid_gdp_r2": round(r2_r, 4) if not np.isnan(r2_r) else None,
        "resid_pvalue": round(p_r, 4) if not np.isnan(p_r) else None,
        "n_obs": n,
        "n_countries": ctry,
    }

# ── Formal interaction test: residualised_GDP × post-2000 ───────────
# The Before/After split above compares two separate regressions. A formal
# interaction term tests whether the slope of residualised GDP on U5MR
# differs significantly between the two periods. Spec:
#   u5mr_it = α_i + γ_1·gdp_resid_it + γ_2·D_post2000_it
#                 + γ_3·(gdp_resid × D_post2000)_it + ε_it
# Country FE via demeaning; country-clustered SE. γ_3 is the test of
# interest: significant γ_3 means the post-2000 GDP-residual slope is
# statistically different from the pre-2000 slope.

print("\n" + "=" * 90)
print("FORMAL INTERACTION TEST: residualised GDP × post-2000")
print("=" * 90)

import statsmodels.api as sm  # local import; main script doesn't otherwise need it

resid_full = fe_residualize_gdp(sub_full)
interaction_result = None
if resid_full is not None:
    sub_r, _ = resid_full
    sub_r = sub_r.dropna(subset=["gdp_resid", "u5mr"]).copy()
    sub_r["post2000"] = ((sub_r["t"] + LAG) >= 2000).astype(int)
    sub_r["gdp_resid_x_post"] = sub_r["gdp_resid"] * sub_r["post2000"]

    # Country demean
    for col in ["gdp_resid", "post2000", "gdp_resid_x_post", "u5mr"]:
        sub_r[f"{col}_dm"] = (
            sub_r[col] - sub_r.groupby("country")[col].transform("mean")
        )

    X = sub_r[["gdp_resid_dm", "post2000_dm", "gdp_resid_x_post_dm"]].values
    y = sub_r["u5mr_dm"].values
    countries_clust = sub_r["country"].values

    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": countries_clust},
    )

    int_beta = float(model.params[2])
    int_se = float(model.bse[2])
    int_t = float(model.tvalues[2])
    int_p = float(model.pvalues[2])
    main_beta = float(model.params[0])
    main_se = float(model.bse[0])

    print(f"  n={len(sub_r)}, countries={sub_r['country'].nunique()}")
    print(f"  β_gdp_resid (pre-2000 slope):                {main_beta:+.4f} "
          f"(SE {main_se:.4f})")
    print(f"  β_interaction (Δ slope post-2000):           {int_beta:+.4f} "
          f"(SE {int_se:.4f}, t={int_t:+.3f}, p={int_p:.4f})")
    if int_p < 0.05:
        print(f"  --> Interaction is significant at p<0.05: post-2000 slope "
              f"differs from pre-2000.")
    else:
        print(f"  --> Interaction is NOT significant at p<0.05: the "
              f"before/after split is descriptive, not a statistically "
              f"distinguishable structural shift.")

    interaction_result = {
        "n_obs": int(len(sub_r)),
        "n_countries": int(sub_r["country"].nunique()),
        "beta_gdp_resid_pre2000": round(main_beta, 4),
        "se_gdp_resid_pre2000":   round(main_se, 4),
        "beta_interaction":       round(int_beta, 4),
        "se_interaction":         round(int_se, 4),
        "t_interaction":          round(int_t, 3),
        "p_interaction":          round(int_p, 4),
    }

# ── Paper number verification ───────────────────────────────────────

print("\n" + "=" * 90)
print("PAPER VERIFICATION")
print("=" * 90)

before = split_results.get("Before 2000", {})
after = split_results.get("After 2000", {})

print(f"\n  Paper claims (Section 6.2.1):")
print(f"    'before 2000, residualized GDP explains 0.3% of child mortality'")
print(f"    'after 2000, it rises to 2.3%'")
print(f"\n  Script produces:")
print(f"    Before 2000: Resid R² = {before.get('resid_gdp_r2', 'n/a')}")
print(f"    After 2000:  Resid R² = {after.get('resid_gdp_r2', 'n/a')}")

b_r2 = before.get("resid_gdp_r2")
a_r2 = after.get("resid_gdp_r2")
if b_r2 is not None and a_r2 is not None:
    b_pct = f"{b_r2 * 100:.1f}%"
    a_pct = f"{a_r2 * 100:.1f}%"
    b_match = "MATCH" if abs(b_r2 - 0.003) < 0.002 else "MISMATCH"
    a_match = "MATCH" if abs(a_r2 - 0.023) < 0.005 else "MISMATCH"
    print(f"\n  Before 2000: {b_pct} vs paper 0.3%  → {b_match}")
    print(f"  After 2000:  {a_pct} vs paper 2.3%  → {a_match}")

# ── Checkin ─────────────────────────────────────────────────────────

write_checkin("u5mr_residual_by_year.json", {
    "method": (
        "Residualized GDP → U5MR swept by outcome year cutoff. "
        "Clustered SEs by country (matching tables/regression_tables.py). "
        "Tests Lutz hypothesis: MDG-era health interventions create "
        "GDP signal for child mortality independent of education. "
        f"Entry >= {ENTRY_THRESHOLD}%, ceiling <= {CEILING}%, "
        f"T=1960-2000, lag={LAG}."
    ),
    "sweep": results,
    "before_after_2000": split_results,
    "post2000_interaction": interaction_result,
}, script_path="scripts/robustness/u5mr_residual_by_year.py")
