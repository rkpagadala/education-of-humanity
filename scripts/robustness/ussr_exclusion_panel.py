"""
ussr_exclusion_panel.py

Robustness check: rerun the paper's headline Table 1 Model (1) parental-
education country-FE regression, and the LE/TFR 25-year forward-prediction
specifications, on the 170-country panel that drops the 15 USSR republics.

Motivation:
  Segments 5–12 of the 2026-04-22 session established that WCDE v3
  inflates reported lower-secondary completion for the 15 USSR republics
  in 1960–90 (Goskomstat reporting artifact). The paper's §9 documents
  the four signatures of the anomaly. This script is the numerical gate:
  if headline results shift materially under USSR exclusion, the paper's
  claims would not be robust. If they don't, the Appendix Data-Source
  Robustness table gets one new row reporting the exclusion panel's
  LE/TFR R² alongside the existing WCDE/B-L/post-1970 rows.

  Gate thresholds (plan-specified):
    - parental-education coefficient shifts <±0.05
    - LE R² and TFR R² shift <±0.02

USSR REPUBLICS (15, WCDE canonical names):
  Russian Federation, Ukraine, Belarus, Republic of Moldova,
  Estonia, Latvia, Lithuania,
  Georgia, Armenia, Azerbaijan,
  Kazakhstan, Kyrgyzstan, Tajikistan, Turkmenistan, Uzbekistan

Warsaw Pact and Yugoslav successors remain in the panel — they used
national statistical offices separate from Goskomstat and pass phenotype-
consistency under Barro-Lee (session Segment 9).

Data sources (same as panel_full_fe.py):
  - WCDE v3 lower-sec completion both sexes: wcde/data/processed/lower_sec_both.csv
  - World Bank WDI outcomes: data/life_expectancy_years.csv,
                             data/children_per_woman_total_fertility.csv,
                             data/child_mortality_u5.csv

Output: checkin/ussr_exclusion_panel.json
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from _shared import (PROC, DATA, REGIONS as NON_SOVEREIGN,
                     write_checkin, fe_regression, load_wide_indicator,
                     wcde_to_wdi)

PARENTAL_LAG = 25
OUTCOME_YEARS = list(range(1975, 2016, 5))

USSR_WCDE = frozenset({
    "Russian Federation", "Ukraine", "Belarus", "Republic of Moldova",
    "Estonia", "Latvia", "Lithuania",
    "Georgia", "Armenia", "Azerbaijan",
    "Kazakhstan", "Kyrgyzstan", "Tajikistan", "Turkmenistan", "Uzbekistan",
})


def build_edu_panel(edu_df, exclude=None):
    """Build (country, year) rows with child lsec completion at year Y and
    parent lsec completion at year Y-25. Mirrors panel_full_fe.py."""
    exclude = exclude or set()
    rows = []
    for country in edu_df.index:
        if country in NON_SOVEREIGN or country in exclude:
            continue
        for y in OUTCOME_YEARS:
            sy, sy_lag = str(y), str(y - PARENTAL_LAG)
            if sy not in edu_df.columns or sy_lag not in edu_df.columns:
                continue
            child = edu_df.loc[country, sy]
            parent = edu_df.loc[country, sy_lag]
            if np.isnan(child) or np.isnan(parent):
                continue
            rows.append({
                "country": country,
                "year": y,
                "child": child,
                "parent": parent,
            })
    return pd.DataFrame(rows)


def build_outcome_panel(edu_df, outcome_df, exclude=None):
    """Build (country, year) rows pairing education at year T-25 with
    outcome at year T. Country names are lowercased for WDI join via
    wcde_to_wdi()."""
    exclude = exclude or set()
    rows = []
    for country in edu_df.index:
        if country in NON_SOVEREIGN or country in exclude:
            continue
        wdi_name = wcde_to_wdi(country).lower()
        if wdi_name not in outcome_df.index:
            continue
        for y in OUTCOME_YEARS:
            sy, sy_lag = str(y), str(y - PARENTAL_LAG)
            if sy_lag not in edu_df.columns:
                continue
            if sy not in outcome_df.columns:
                continue
            edu_val = edu_df.loc[country, sy_lag]
            out_val = outcome_df.loc[wdi_name, sy]
            if np.isnan(edu_val) or np.isnan(out_val):
                continue
            rows.append({
                "country": country,
                "year": y,
                "edu_lag": edu_val,
                "outcome": out_val,
            })
    return pd.DataFrame(rows)


def run_edu_fe(panel, label):
    model, n, nc = fe_regression(panel, ["parent"], "child")
    beta = float(model.params.iloc[0])
    se = float(model.bse.iloc[0])
    r2 = float(model.rsquared)
    print(f"  [{label}]  β={beta:+.3f}  SE={se:.3f}  "
          f"R²={r2:.3f}  N={n}  countries={nc}")
    return {"beta": round(beta, 3), "se": round(se, 3),
            "r2": round(r2, 3), "n": int(n), "countries": int(nc)}


def run_outcome_fe(panel, label):
    model, n, nc = fe_regression(panel, ["edu_lag"], "outcome")
    beta = float(model.params.iloc[0])
    se = float(model.bse.iloc[0])
    r2 = float(model.rsquared)
    print(f"  [{label}]  β={beta:+.3f}  SE={se:.3f}  "
          f"R²={r2:.3f}  N={n}  countries={nc}")
    return {"beta": round(beta, 3), "se": round(se, 3),
            "r2": round(r2, 3), "n": int(n), "countries": int(nc)}


def delta(a, b, key):
    return round(b[key] - a[key], 3)


def main():
    edu = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"),
                      index_col="country")
    le = load_wide_indicator("life_expectancy_years.csv")
    tfr = load_wide_indicator("children_per_woman_total_fertility.csv")
    u5 = load_wide_indicator("child_mortality_u5.csv")

    print("=" * 78)
    print("USSR EXCLUSION ROBUSTNESS")
    print("  Headline 185-country panel vs 170-country USSR-excluded panel")
    print(f"  USSR republics excluded ({len(USSR_WCDE)}):")
    for name in sorted(USSR_WCDE):
        print(f"    - {name}")
    print("=" * 78)

    # ── Child-education forward FE (Table 1 Model 1) ────────────────
    print("\nTable 1 Model (1): child lsec ~ parent lsec  (country FE)")
    edu_full = build_edu_panel(edu, exclude=set())
    edu_clean = build_edu_panel(edu, exclude=USSR_WCDE)
    m_full = run_edu_fe(edu_full, "185 full")
    m_clean = run_edu_fe(edu_clean, "170 USSR-excl")
    beta_delta = delta(m_full, m_clean, "beta")
    r2_delta = delta(m_full, m_clean, "r2")
    print(f"  Δβ = {beta_delta:+.3f}   ΔR² = {r2_delta:+.3f}   "
          f"Δn = {m_clean['n'] - m_full['n']}")

    # ── LE 25-yr forward FE ─────────────────────────────────────────
    print("\nLE(T) ~ lsec(T-25)  (country FE)")
    le_full = build_outcome_panel(edu, le, exclude=set())
    le_clean = build_outcome_panel(edu, le, exclude=USSR_WCDE)
    le_m_full = run_outcome_fe(le_full, "185 full")
    le_m_clean = run_outcome_fe(le_clean, "170 USSR-excl")
    le_beta_delta = delta(le_m_full, le_m_clean, "beta")
    le_r2_delta = delta(le_m_full, le_m_clean, "r2")
    print(f"  Δβ = {le_beta_delta:+.3f}   ΔR² = {le_r2_delta:+.3f}")

    # ── TFR 25-yr forward FE ────────────────────────────────────────
    print("\nTFR(T) ~ lsec(T-25)  (country FE)")
    tfr_full = build_outcome_panel(edu, tfr, exclude=set())
    tfr_clean = build_outcome_panel(edu, tfr, exclude=USSR_WCDE)
    tfr_m_full = run_outcome_fe(tfr_full, "185 full")
    tfr_m_clean = run_outcome_fe(tfr_clean, "170 USSR-excl")
    tfr_beta_delta = delta(tfr_m_full, tfr_m_clean, "beta")
    tfr_r2_delta = delta(tfr_m_full, tfr_m_clean, "r2")
    print(f"  Δβ = {tfr_beta_delta:+.3f}   ΔR² = {tfr_r2_delta:+.3f}")

    # ── U5MR 25-yr forward FE (log) ─────────────────────────────────
    print("\nlog U5MR(T) ~ lsec(T-25)  (country FE)")
    u5_log = u5.copy().clip(lower=0.1)
    u5_log = np.log(u5_log.astype(float))
    u5_full = build_outcome_panel(edu, u5_log, exclude=set())
    u5_clean = build_outcome_panel(edu, u5_log, exclude=USSR_WCDE)
    u5_m_full = run_outcome_fe(u5_full, "185 full")
    u5_m_clean = run_outcome_fe(u5_clean, "170 USSR-excl")
    u5_beta_delta = delta(u5_m_full, u5_m_clean, "beta")
    u5_r2_delta = delta(u5_m_full, u5_m_clean, "r2")
    print(f"  Δβ = {u5_beta_delta:+.3f}   ΔR² = {u5_r2_delta:+.3f}")

    # ── Gate check ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("GATE CHECK  (plan thresholds: |Δβ| < 0.05, |ΔR²| < 0.02)")
    print("=" * 78)
    checks = [
        ("Table 1 M1 β", beta_delta, 0.05),
        ("LE R²",         le_r2_delta, 0.02),
        ("TFR R²",        tfr_r2_delta, 0.02),
    ]
    all_pass = True
    for name, val, thresh in checks:
        pass_ = abs(val) < thresh
        marker = "PASS" if pass_ else "FAIL"
        print(f"  {name:<20} Δ={val:+.3f}   threshold=±{thresh:.3f}   [{marker}]")
        if not pass_:
            all_pass = False
    print(f"\n  Overall: {'PASS — proceed with §9' if all_pass else 'FAIL — revise plan before proceeding'}")

    # ── JSON ────────────────────────────────────────────────────────
    write_checkin("ussr_exclusion_panel.json", {
        "numbers": {
            "ussr_excluded_n": len(USSR_WCDE),
            # Table 1 Model (1)
            "T1M1_full_beta": m_full["beta"],
            "T1M1_full_r2": m_full["r2"],
            "T1M1_full_n": m_full["n"],
            "T1M1_full_countries": m_full["countries"],
            "T1M1_clean_beta": m_clean["beta"],
            "T1M1_clean_r2": m_clean["r2"],
            "T1M1_clean_n": m_clean["n"],
            "T1M1_clean_countries": m_clean["countries"],
            "T1M1_delta_beta": beta_delta,
            "T1M1_delta_r2": r2_delta,
            # LE
            "LE_full_r2": le_m_full["r2"],
            "LE_full_n": le_m_full["n"],
            "LE_clean_r2": le_m_clean["r2"],
            "LE_clean_n": le_m_clean["n"],
            "LE_delta_r2": le_r2_delta,
            # TFR
            "TFR_full_r2": tfr_m_full["r2"],
            "TFR_full_n": tfr_m_full["n"],
            "TFR_clean_r2": tfr_m_clean["r2"],
            "TFR_clean_n": tfr_m_clean["n"],
            "TFR_delta_r2": tfr_r2_delta,
            # U5MR (log)
            "U5log_full_r2": u5_m_full["r2"],
            "U5log_clean_r2": u5_m_clean["r2"],
            "U5log_delta_r2": u5_r2_delta,
            # Gate
            "gate_pass": int(all_pass),
        },
    }, script_path="scripts/robustness/ussr_exclusion_panel.py")


if __name__ == "__main__":
    main()
