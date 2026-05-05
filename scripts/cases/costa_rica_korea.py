"""
cases/costa_rica_korea.py

Verify the Korea vs Costa Rica comparison (Section 9):
  - Costa Rica 1960 GDP: $3,609; Korea 1960 GDP: $1,038 (constant 2015 USD)
  - Ratio: 3.5x
  - By 1990: Korea $9,673 (9-fold increase); Costa Rica $6,037 (1.7-fold increase)
  - Korea expansion rate: 8-14 pp per five years (WCDE lower_sec)
  - Costa Rica expansion rate: 3-6 pp per five years

Data sources:
  - GDP: data/gdppercapita_us_inflation_adjusted.csv (World Bank, constant 2015 USD)
  - Education: wcde/data/processed/lower_sec_both.csv (WCDE v3)

Output: checkin/costa_rica_korea.json
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _shared import PROC, load_wb, write_checkin

# ── Load GDP data ────────────────────────────────────────────────────────────
gdp = load_wb("gdppercapita_us_inflation_adjusted.csv")

# Country names in World Bank data (lowercase, as returned by load_wb)
CR_GDP_NAME = "costa rica"
KR_GDP_NAME = "korea, rep."

# ── Load WCDE lower secondary completion ─────────────────────────────────────
edu = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"), index_col="country")

# Country names in WCDE data
CR_EDU_NAME = "Costa Rica"
KR_EDU_NAME = "Republic of Korea"

# ── GDP comparison ───────────────────────────────────────────────────────────
def get_gdp(country, year):
    try:
        val = float(gdp.loc[country, str(year)])
        return val if not np.isnan(val) else None
    except (KeyError, ValueError):
        return None

cr_1960 = get_gdp(CR_GDP_NAME, 1960)
kr_1960 = get_gdp(KR_GDP_NAME, 1960)
cr_1990 = get_gdp(CR_GDP_NAME, 1990)
kr_1990 = get_gdp(KR_GDP_NAME, 1990)

print("=" * 70)
print("KOREA vs COSTA RICA GDP COMPARISON")
print("=" * 70)
print(f"\n  Costa Rica 1960 GDP: ${cr_1960:,.0f}")
print(f"  Korea 1960 GDP:      ${kr_1960:,.0f}")
ratio_1960 = cr_1960 / kr_1960 if (cr_1960 and kr_1960) else None
print(f"  Ratio (CR/KR):       {ratio_1960:.1f}x")

print(f"\n  Costa Rica 1990 GDP: ${cr_1990:,.0f}")
print(f"  Korea 1990 GDP:      ${kr_1990:,.0f}")
kr_fold = kr_1990 / kr_1960 if (kr_1990 and kr_1960) else None
cr_fold = cr_1990 / cr_1960 if (cr_1990 and cr_1960) else None
print(f"  Korea fold increase:       {kr_fold:.1f}x")
print(f"  Costa Rica fold increase:  {cr_fold:.1f}x")

# Paper claims
print("\n  Paper claims:")
print(f"    Costa Rica 1960 GDP: $3,609  →  actual: ${cr_1960:,.0f}")
print(f"    Korea 1960 GDP: $1,038       →  actual: ${kr_1960:,.0f}")
print(f"    Ratio: 3.5x                  →  actual: {ratio_1960:.1f}x")
print(f"    Korea 1990: $9,673 (9x)      →  actual: ${kr_1990:,.0f} ({kr_fold:.1f}x)")
print(f"    CR 1990: $6,037 (1.7x)       →  actual: ${cr_1990:,.0f} ({cr_fold:.1f}x)")

# ── Education expansion rates ────────────────────────────────────────────────
def get_edu(country, year):
    try:
        val = float(edu.loc[country, str(year)])
        return val if not np.isnan(val) else None
    except (KeyError, ValueError):
        return None

print("\n" + "=" * 70)
print("EDUCATION EXPANSION RATES (lower secondary completion, WCDE)")
print("=" * 70)

# Available WCDE years (5-year intervals)
years = sorted([int(c) for c in edu.columns if c.isdigit()])

kr_gains = []
cr_gains = []

print(f"\n  {'Period':<12} {'Korea %':>10} {'Korea gain':>12} {'CR %':>10} {'CR gain':>10}")
print("  " + "-" * 58)

for i in range(len(years) - 1):
    y1, y2 = years[i], years[i + 1]
    interval = y2 - y1
    if interval != 5:
        continue

    kr1 = get_edu(KR_EDU_NAME, y1)
    kr2 = get_edu(KR_EDU_NAME, y2)
    cr1 = get_edu(CR_EDU_NAME, y1)
    cr2 = get_edu(CR_EDU_NAME, y2)

    kr_gain = (kr2 - kr1) if (kr1 is not None and kr2 is not None) else None
    cr_gain = (cr2 - cr1) if (cr1 is not None and cr2 is not None) else None

    kr_str = f"{kr2:.1f}" if kr2 is not None else "N/A"
    cr_str = f"{cr2:.1f}" if cr2 is not None else "N/A"
    kr_g_str = f"{kr_gain:+.1f} pp" if kr_gain is not None else "N/A"
    cr_g_str = f"{cr_gain:+.1f} pp" if cr_gain is not None else "N/A"

    print(f"  {y1}-{y2:<7} {kr_str:>10} {kr_g_str:>12} {cr_str:>10} {cr_g_str:>10}")

    if kr_gain is not None:
        kr_gains.append({"period": f"{y1}-{y2}", "gain_pp": round(kr_gain, 2)})
    if cr_gain is not None:
        cr_gains.append({"period": f"{y1}-{y2}", "gain_pp": round(cr_gain, 2)})

# Summarize ranges -- focus on the expansion period (before ceiling/data artifacts)
# Korea's expansion period: 1950-1985 (before hitting 94%+)
# Costa Rica's expansion period: 1950-2015 (still expanding)
kr_expansion = [g for g in kr_gains
                if g["gain_pp"] > 0
                and int(g["period"].split("-")[0]) >= 1950
                and int(g["period"].split("-")[0]) <= 1980]
cr_expansion = [g for g in cr_gains
                if g["gain_pp"] > 0
                and int(g["period"].split("-")[0]) >= 1950
                and int(g["period"].split("-")[0]) <= 2015]

kr_exp_vals = [g["gain_pp"] for g in kr_expansion]
cr_exp_vals = [g["gain_pp"] for g in cr_expansion]

print(f"\n  Korea expansion period (1950-1985) gains: "
      f"{min(kr_exp_vals):.1f} to {max(kr_exp_vals):.1f} pp per 5 years")
print(f"  Costa Rica gains (1950-2015): "
      f"{min(cr_exp_vals):.1f} to {max(cr_exp_vals):.1f} pp per 5 years")
print(f"  Paper claims: Korea 8-14 pp, Costa Rica 3-6 pp")

# ── Write checkin ────────────────────────────────────────────────────────────
write_checkin("costa_rica_korea.json", {
    "numbers": {
        "cr_1960_gdp": round(cr_1960, 0) if cr_1960 else None,
        "kr_1960_gdp": round(kr_1960, 0) if kr_1960 else None,
        "gdp_ratio_1960": round(ratio_1960, 2) if ratio_1960 else None,
        "cr_1990_gdp": round(cr_1990, 0) if cr_1990 else None,
        "kr_1990_gdp": round(kr_1990, 0) if kr_1990 else None,
        "kr_fold_increase": round(kr_fold, 2) if kr_fold else None,
        "cr_fold_increase": round(cr_fold, 2) if cr_fold else None,
        "kr_5yr_gains": kr_gains,
        "cr_5yr_gains": cr_gains,
        "kr_expansion_gains": kr_expansion,
        "cr_expansion_gains": cr_expansion,
        "kr_expansion_range_pp": [round(min(kr_exp_vals), 1), round(max(kr_exp_vals), 1)],
        "cr_expansion_range_pp": [round(min(cr_exp_vals), 1), round(max(cr_exp_vals), 1)],
    },
    "paper_claims": {
        "cr_1960_gdp": 3609,
        "kr_1960_gdp": 1038,
        "ratio_1960": 3.5,
        "kr_1990_gdp": 9673,
        "kr_fold": 9.0,
        "cr_1990_gdp": 6037,
        "cr_fold": 1.7,
        "kr_expansion_pp": "8-14",
        "cr_expansion_pp": "3-6",
    },
}, script_path="scripts/cases/costa_rica_korea.py")
