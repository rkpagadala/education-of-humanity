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
lower secondary completion, T=1960-1990, lag=25.
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
from _shared import fmt_r2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Parameters (match Table 2b) ────────────────────────────────────

T_YEARS = list(range(1960, 1995, 5))
LAG = 25
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
        print(f"  Paper Table 2b: R²=0.023, p=0.11")

# ── Sweep by outcome year cutoff ────────────────────────────────────

print("\n" + "=" * 90)
print("RESIDUALIZED GDP → U5MR: SWEEP BY OUTCOME YEAR CUTOFF (CLUSTERED SEs)")
print("=" * 90)
print(f"Entry >= {ENTRY_THRESHOLD}%, ceiling <= {CEILING}%, "
      f"T=1960-1990, lag={LAG}")
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
        f"T=1960-1990, lag={LAG}."
    ),
    "sweep": results,
    "before_after_2000": split_results,
}, script_path="scripts/robustness/u5mr_residual_by_year.py")
