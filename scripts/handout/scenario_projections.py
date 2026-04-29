"""
handout/scenario_projections.py
================================
Extended scenario projection table for the Stanford King Center handout
(5 May 2026). For each SSA country below the 30% active-expansion
threshold, project four outcomes (LE, TFR, log GDP, U5MR) under two
scenarios over 25 years (one generation):

  • Current pace: country continues at its own 1990–2015 lower-sec
    growth rate; ΔLE = 25 × growth × β.
  • Bangladesh pace: country matches Bangladesh's 1990–2015 expansion
    of +30pp over 25 years (= 1.20 pp/yr).

Coefficients used (matching handout §3 spec — paper's "+init Y" spec):
  • LE β:        SSA-specific (this script's run) where significant;
                 falls back to global if not.
  • TFR β:       SSA-specific.
  • log_GDP β:   global (SSA estimate not significant).
  • U5MR β:      SSA-specific (pure-FE; +init-Y not significant for
                 either sample, so revert to pure-FE for U5MR alone).

All βs sourced from checkin/handout_ssa_betas.json (run
scripts/handout/ssa_subsample_betas.py first).

Defaults (changeable at top of script):
  THRESHOLD_PCT       = 30      # only SSA countries below this in 2015
  BANGLADESH_PACE_PP  = 30      # +30pp over 25 yrs
  BASELINE_YEAR       = 2015
  SCENARIO_YEARS      = 25

The output is a per-country table with current LE/TFR/U5MR/GDP, and
projected values under both scenarios. Written to
checkin/handout_scenario_projections.json.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, ".."))

from _shared import (  # noqa: E402
    REPO_ROOT, CHECKIN, load_education, load_wb, get_wb_val,
    interpolate_to_annual, write_checkin,
)

# ── Parameters ───────────────────────────────────────────────────────
THRESHOLD_PCT = 30
BANGLADESH_PACE_PP = 30
BASELINE_YEAR = 2015
SCENARIO_YEARS = 25
GROWTH_BASELINE_START = 1990

# ── SSA list (UN M49) ────────────────────────────────────────────────
SSA_WCDE = {
    "Burundi", "Comoros", "Djibouti", "Eritrea", "Ethiopia", "Kenya",
    "Madagascar", "Malawi", "Mauritius", "Mozambique", "Rwanda",
    "Seychelles", "Somalia", "South Sudan", "Uganda",
    "United Republic of Tanzania", "Zambia", "Zimbabwe",
    "Angola", "Cameroon", "Central African Republic", "Chad", "Congo",
    "Democratic Republic of the Congo", "Equatorial Guinea", "Gabon",
    "Sao Tome and Principe",
    "Botswana", "Lesotho", "Namibia", "South Africa", "Swaziland",
    "Benin", "Burkina Faso", "Cape Verde", "Cote d'Ivoire", "Gambia",
    "Ghana", "Guinea", "Guinea-Bissau", "Liberia", "Mali", "Mauritania",
    "Niger", "Nigeria", "Senegal", "Sierra Leone", "Togo",
}

# ── Pull β values from prior run ─────────────────────────────────────
betas_path = os.path.join(CHECKIN, "handout_ssa_betas.json")
if not os.path.exists(betas_path):
    raise SystemExit(
        "Run scripts/handout/ssa_subsample_betas.py first "
        "(checkin/handout_ssa_betas.json missing)."
    )

with open(betas_path) as f:
    betas_all = json.load(f)["results"]


def beta(name, spec, sample):
    r = betas_all.get(name, {}).get(spec, {}).get(sample)
    return r["beta"] if r else None


# Choose β per outcome — matches handout's "+init Y" spec where defensible
BETAS = {
    "LE":   {"value": beta("life_expectancy", "with_init_outcome", "ssa"),
             "source": "SSA β (+init Y), this script"},
    "TFR":  {"value": beta("tfr", "with_init_outcome", "ssa"),
             "source": "SSA β (+init Y), this script"},
    "GDP":  {"value": beta("log_gdp", "with_init_outcome", "global"),
             "source": "global β (+init Y); SSA not significant"},
    "U5MR": {"value": beta("u5mr", "pure_fe", "ssa"),
             "source": "SSA β (pure FE); +init Y not significant"},
}

print("=" * 86)
print("SSA SCENARIO PROJECTIONS — countries below "
      f"{THRESHOLD_PCT}% lower-sec completion in {BASELINE_YEAR}")
print(f"Bangladesh-pace scenario: +{BANGLADESH_PACE_PP}pp over "
      f"{SCENARIO_YEARS} years (= {BANGLADESH_PACE_PP/SCENARIO_YEARS:.2f} pp/yr)")
print("=" * 86)

print("\nβ used in projections:")
for k, v in BETAS.items():
    print(f"  {k:<5}: β = {v['value']:+.4f}    ({v['source']})")

# ── Load data ────────────────────────────────────────────────────────
edu_raw = load_education("completion_both_long.csv")
edu_annual = interpolate_to_annual(edu_raw, "lower_sec")
le = load_wb("life_expectancy_years.csv")
tfr = load_wb("children_per_woman_total_fertility.csv")
u5 = load_wb("child_mortality_u5.csv")
gdp = load_wb("gdppercapita_us_inflation_adjusted.csv")


def edu_at(country, year):
    s = edu_annual.get(country)
    if s is None or year not in s.index:
        return np.nan
    return float(s[year])


def own_pace(country):
    """Country's own lower-sec growth rate, 1990 → 2015 (pp/yr)."""
    e_old = edu_at(country, GROWTH_BASELINE_START)
    e_new = edu_at(country, BASELINE_YEAR)
    if np.isnan(e_old) or np.isnan(e_new):
        return np.nan
    return (e_new - e_old) / (BASELINE_YEAR - GROWTH_BASELINE_START)


# ── Build per-country table ─────────────────────────────────────────
rows = []
for c in sorted(SSA_WCDE):
    e_2015 = edu_at(c, BASELINE_YEAR)
    if np.isnan(e_2015) or e_2015 >= THRESHOLD_PCT:
        continue
    pace = own_pace(c)
    if np.isnan(pace):
        continue

    le_now = get_wb_val(le, c, BASELINE_YEAR)
    tfr_now = get_wb_val(tfr, c, BASELINE_YEAR)
    u5_now = get_wb_val(u5, c, BASELINE_YEAR)
    gdp_now = get_wb_val(gdp, c, BASELINE_YEAR)

    delta_current = pace * SCENARIO_YEARS
    delta_bgd = float(BANGLADESH_PACE_PP)

    def project(delta, init):
        return {
            "Δedu": delta,
            "edu_next": e_2015 + delta,
            "LE": init["LE"] + delta * BETAS["LE"]["value"]
                  if not np.isnan(init["LE"]) else np.nan,
            "TFR": init["TFR"] + delta * BETAS["TFR"]["value"]
                   if not np.isnan(init["TFR"]) else np.nan,
            "U5MR": init["U5MR"] + delta * BETAS["U5MR"]["value"]
                    if not np.isnan(init["U5MR"]) else np.nan,
            "GDP": init["GDP"] * np.exp(delta * BETAS["GDP"]["value"])
                   if not np.isnan(init["GDP"]) else np.nan,
        }

    init = {"LE": le_now, "TFR": tfr_now, "U5MR": u5_now, "GDP": gdp_now}
    cur = project(delta_current, init)
    bgd = project(delta_bgd, init)

    rows.append({
        "country": c,
        "edu_2015": e_2015,
        "pace_pp_per_yr": pace,
        "le_2015": le_now,
        "tfr_2015": tfr_now,
        "u5mr_2015": u5_now,
        "gdp_2015": gdp_now,
        "current_pace_2040": cur,
        "bgd_pace_2040": bgd,
    })

# ── Print table ──────────────────────────────────────────────────────

def fmt(v, kind):
    if v is None or pd.isna(v):
        return "  N/A"
    if kind == "pct":
        return f"{v:5.1f}%"
    if kind == "yrs":
        return f"{v:5.1f}"
    if kind == "tfr":
        return f"{v:5.2f}"
    if kind == "u5":
        return f"{v:5.0f}"
    if kind == "gdp":
        return f"${v:>6,.0f}"
    if kind == "pace":
        return f"{v:+5.2f}"
    return f"{v:.2f}"


print(f"\n{'Country':<22} {'Edu':>6} {'pace':>6}   {'LE':>7} → {'cur':>5}  "
      f"{'BGD':>5}    {'TFR':>5} → {'cur':>4} {'BGD':>4}    "
      f"{'U5MR':>5} → {'cur':>4} {'BGD':>4}    {'GDP':>7} → {'cur':>7} {'BGD':>7}")
print("-" * 130)

for r in sorted(rows, key=lambda x: x["edu_2015"]):
    cur = r["current_pace_2040"]
    bgd = r["bgd_pace_2040"]
    print(
        f"{r['country']:<22} "
        f"{fmt(r['edu_2015'], 'pct'):>6} "
        f"{fmt(r['pace_pp_per_yr'], 'pace'):>6}   "
        f"{fmt(r['le_2015'], 'yrs'):>7} → "
        f"{fmt(cur['LE'], 'yrs'):>5}  "
        f"{fmt(bgd['LE'], 'yrs'):>5}    "
        f"{fmt(r['tfr_2015'], 'tfr'):>5} → "
        f"{fmt(cur['TFR'], 'tfr'):>4} "
        f"{fmt(bgd['TFR'], 'tfr'):>4}    "
        f"{fmt(r['u5mr_2015'], 'u5'):>5} → "
        f"{fmt(cur['U5MR'], 'u5'):>4} "
        f"{fmt(bgd['U5MR'], 'u5'):>4}    "
        f"{fmt(r['gdp_2015'], 'gdp'):>7} → "
        f"{fmt(cur['GDP'], 'gdp'):>7} "
        f"{fmt(bgd['GDP'], 'gdp'):>7}"
    )

# ── Niger spotlight ──────────────────────────────────────────────────
niger = next((r for r in rows if r["country"] == "Niger"), None)
if niger:
    print("\n" + "=" * 86)
    print("NIGER SPOTLIGHT (matches handout §3 framing, with SSA β substituted)")
    print("=" * 86)
    print(f"  Baseline 2015: edu = {niger['edu_2015']:.1f}%, "
          f"pace = {niger['pace_pp_per_yr']:+.2f} pp/yr, "
          f"LE = {niger['le_2015']:.1f}, TFR = {niger['tfr_2015']:.2f}, "
          f"U5MR = {niger['u5mr_2015']:.0f}, GDP = ${niger['gdp_2015']:,.0f}")
    print()
    cur = niger["current_pace_2040"]
    bgd = niger["bgd_pace_2040"]
    print(f"  At current pace (+{cur['Δedu']:.1f}pp over 25 yrs):")
    print(f"    edu  → {cur['edu_next']:5.1f}%")
    print(f"    LE   → {cur['LE']:5.1f}  (Δ = {cur['LE']-niger['le_2015']:+.1f})")
    print(f"    TFR  → {cur['TFR']:5.2f}  (Δ = {cur['TFR']-niger['tfr_2015']:+.2f})")
    print(f"    U5MR → {cur['U5MR']:5.0f}  (Δ = {cur['U5MR']-niger['u5mr_2015']:+.0f})")
    print(f"    GDP  → ${cur['GDP']:,.0f}  (Δ = "
          f"{(cur['GDP']/niger['gdp_2015']-1)*100:+.1f}%)")
    print()
    print(f"  At Bangladesh pace (+{bgd['Δedu']:.0f}pp over 25 yrs):")
    print(f"    edu  → {bgd['edu_next']:5.1f}%")
    print(f"    LE   → {bgd['LE']:5.1f}  (Δ = {bgd['LE']-niger['le_2015']:+.1f}) "
          f"[handout: +3.3 yrs → 64.4]")
    print(f"    TFR  → {bgd['TFR']:5.2f}  (Δ = {bgd['TFR']-niger['tfr_2015']:+.2f}) "
          f"[handout: -0.96 → 6.44]")
    print(f"    U5MR → {bgd['U5MR']:5.0f}  (Δ = {bgd['U5MR']-niger['u5mr_2015']:+.0f}) "
          f"[handout: not projected]")
    print(f"    GDP  → ${bgd['GDP']:,.0f}  (Δ = "
          f"{(bgd['GDP']/niger['gdp_2015']-1)*100:+.1f}%) "
          f"[handout: +33% → $479]")

print("\n" + "=" * 86)
print("CAVEATS")
print("=" * 86)
print("  • Linear projection. Coefficients estimated on entry-cohort sample")
print("    (entry≥10%, ceiling≤90%). Niger's 2015 baseline of 8.9% is below")
print("    the 10% entry threshold; first ~1pp of expansion is below the")
print("    estimated range. Bangladesh-pace scenario takes Niger to 38.9%")
print("    (Niger) or 47.7% (Mali) — well within the estimated range.")
print("  • U5MR projection uses pure-FE β (handout doesn't project U5MR).")
print("    The +init-Y spec for U5MR is not significant in either sample,")
print("    so pure-FE is the only defensible choice.")
print("  • GDP projection uses global β (SSA estimate not significant).")
print("  • All βs from country FE on within-country variation; cross-country")
print("    levels not identified. Use as IMPROVEMENT over status quo, not")
print("    forecast of absolute level.")

write_checkin(
    "handout_scenario_projections.json",
    {
        "method": (
            "Per-country linear projection of LE/TFR/U5MR/GDP under two "
            "scenarios (current pace and Bangladesh pace +30pp). Country FE "
            "βs from handout_ssa_betas.json: LE/TFR use SSA-specific +init-Y "
            "spec (matches handout §3); GDP uses global +init-Y (SSA n.s.); "
            "U5MR uses SSA pure-FE (+init-Y n.s. for both)."
        ),
        "params": {
            "threshold_pct": THRESHOLD_PCT,
            "bangladesh_pace_pp": BANGLADESH_PACE_PP,
            "baseline_year": BASELINE_YEAR,
            "scenario_years": SCENARIO_YEARS,
            "growth_baseline_start": GROWTH_BASELINE_START,
        },
        "betas_used": BETAS,
        "rows": rows,
    },
    "scripts/handout/scenario_projections.py",
)
