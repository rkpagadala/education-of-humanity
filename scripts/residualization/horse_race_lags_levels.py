"""
residualization/horse_race_lags_levels.py
==========================================

PAPER ANCHOR — Chapter 9 §The Generational Lag, outcome-specific lag structure
  Question: do the cohort's own schooling (the T block) and the prior
            generation's schooling (the T-28 block) at all three completion
            levels each carry orthogonal information for each outcome, once
            each outcome is read forward at its own biological horizon?
  Produces: the four delta-R^2 numbers cited inline in the paper
            (TFR / U5MR carry signal in both blocks; LE / GDP add little
            in the T-28 block once the T block is in).
  Checkin:  checkin/horse_race_lags_levels.json

Horse-race regressions with multiple education variables at the cohort's
own schooling year (T) and one generation earlier (T-28). Crucially, each
OUTCOME is read forward at its biological horizon (not at T): TFR at T+5
(parent-of-newborn timing), U5MR and LE at T+12 (the childrearing window),
GDP at T+0 (the educated worker's current output). Holding the two education
blocks at T and T-28 while moving the outcome to T+lag makes the T block the
parent generation (it leads the outcome by exactly its biological lag) and
the T-28 block the grandparent generation — reproducing the per-outcome
parent/grandparent lags of Section 9.6 (TFR: parent at outcome-5, grandparent
at outcome-33; U5MR: outcome-12 / outcome-40; GDP: outcome-0 / outcome-28).

For each outcome (log TFR, log LE, log U5MR, log GDP_pc), runs the
following country-FE specifications on a SINGLE sample so the columns
are directly comparable:

  Univariate, lag 0:
     [1]  primary(T)
     [2]  lower_sec(T)
     [3]  upper_sec(T)
  Univariate, lag 28:
     [4]  primary(T-28)
     [5]  lower_sec(T-28)
     [6]  upper_sec(T-28)
  Same-lag horse race (which level matters when all three are in):
     [7]  primary(T)     + lower_sec(T)     + upper_sec(T)
     [8]  primary(T-28)  + lower_sec(T-28)  + upper_sec(T-28)
  Same-level horse race (which lag matters when both are in):
     [9]  primary(T)     + primary(T-28)
     [10] lower_sec(T)   + lower_sec(T-28)
     [11] upper_sec(T)   + upper_sec(T-28)
  Kitchen sink:
     [12] all six regressors together

Sample: entry=10%/ceiling=90% on lower_sec at T (paper's headline spec).
Same rows for every spec for a given outcome — apples-to-apples.

Each spec reports: β, SE, p (country-clustered), within-R², n, countries.
"""

import os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _shared import (
    load_education, load_wb, interpolate_to_annual, write_checkin, get_wb_val,
    LAG_GENERATION, LAG_TFR, LAG_CHILDREARING, LAG_LE, LAG_CONTEMPORANEOUS,
)

# Per-outcome biological horizon at which each outcome is read forward of the
# education year T (see _shared.py for the anchoring). TFR at the parent-of-
# newborn lag, U5MR/LE at the childrearing window, GDP contemporaneous.
OUTCOME_BIO_LAG = {
    "tfr": LAG_TFR,            # 5  — cohort age 20-24 → its own newborn
    "u5":  LAG_CHILDREARING,   # 12 — cohort's children must reach age 5
    "le":  LAG_LE,             # 12 — LE at birth is U5MR-dominated; same window
    "gdp": LAG_CONTEMPORANEOUS, # 0 — the educated worker's current output
}

# ── Load data ────────────────────────────────────────────────────────

print("Loading data...")
edu_raw = load_education("completion_both_long.csv")
tfr_raw = load_wb("children_per_woman_total_fertility.csv")
le_raw  = load_wb("life_expectancy_years.csv")
u5_raw  = load_wb("child_mortality_u5.csv")
gdp_raw = load_wb("gdppercapita_us_inflation_adjusted.csv")

# Interpolate each WCDE level to annual
pri_a = interpolate_to_annual(edu_raw, "primary")
low_a = interpolate_to_annual(edu_raw, "lower_sec")
upp_a = interpolate_to_annual(edu_raw, "upper_sec")

# ── Build panel: edu at T and T-28, outcomes at T ───────────────────

T_YEARS = list(range(1975, 2016, 5))   # T-28 ≥ 1950, T ≤ 2015; WCDE starts 1950
LAG_OLD = LAG_GENERATION

rows = []
all_countries = sorted(set(pri_a.keys()) & set(low_a.keys()) & set(upp_a.keys()))
for c in all_countries:
    p = pri_a.get(c); l = low_a.get(c); u = upp_a.get(c)
    if p is None or l is None or u is None:
        continue
    for t in T_YEARS:
        if (t not in p.index) or ((t - LAG_OLD) not in p.index):
            continue
        rows.append({
            "country":      c,
            "t":            t,
            "pri_t":        p[t],
            "low_t":        l[t],
            "upp_t":        u[t],
            "pri_t28":      p[t - LAG_OLD],
            "low_t28":      l[t - LAG_OLD],
            "upp_t28":      u[t - LAG_OLD],
            "tfr":          get_wb_val(tfr_raw, c, t + OUTCOME_BIO_LAG["tfr"]),
            "le":           get_wb_val(le_raw,  c, t + OUTCOME_BIO_LAG["le"]),
            "u5":           get_wb_val(u5_raw,  c, t + OUTCOME_BIO_LAG["u5"]),
            "gdp":          get_wb_val(gdp_raw, c, t + OUTCOME_BIO_LAG["gdp"]),
        })

panel = pd.DataFrame(rows)
for col in ["tfr","le","u5","gdp"]:
    panel[f"log_{col}"] = np.where(panel[col] > 0, np.log(panel[col]), np.nan)

# Apply entry-cohort: country enters when lower_sec ≥ 10, observations kept
# while lower_sec ≤ 90 (the paper's headline expansion spec). Use lower_sec
# at T as the gating variable. Drop countries that never crossed 10%.
panel["lower_sec_t"] = panel["low_t"]
entry_year_low10 = {}
for c, g in panel.groupby("country"):
    above = g[g["lower_sec_t"] >= 10]
    if len(above) > 0:
        entry_year_low10[c] = above["t"].min()
panel = panel[panel.apply(lambda r: r["country"] in entry_year_low10 and
                                     r["t"] >= entry_year_low10[r["country"]] and
                                     r["lower_sec_t"] <= 90, axis=1)].copy()

print(f"Panel: {len(panel)} country-year observations, "
      f"{panel['country'].nunique()} countries, "
      f"T ∈ [{panel['t'].min()}, {panel['t'].max()}]")


# ── Within-country FE regression helper ─────────────────────────────

def fe_reg(df, y_col, x_cols):
    """Country FE within-OLS with country-clustered SEs.
    Returns dict per regressor + R² + n + countries — or None if too small.
    """
    cols = [y_col] + x_cols + ["country"]
    sub = df[cols].dropna().copy()
    if len(sub) < 30:
        return None
    # Demean y and each x within country
    for col in [y_col] + x_cols:
        sub[col + "_dm"] = sub.groupby("country")[col].transform(lambda s: s - s.mean()).astype(float)
    # Drop singleton countries (zero within variation)
    ok_country = sub.groupby("country")[y_col].transform("count") > 1
    sub = sub[ok_country]
    if sub["country"].nunique() < 3:
        return None
    Y = sub[y_col + "_dm"].values
    X = sub[[c + "_dm" for c in x_cols]].values
    try:
        model = sm.OLS(Y, X).fit(cov_type="cluster",
                                 cov_kwds={"groups": sub["country"].values})
    except Exception:
        return None
    out = {
        "n":         int(len(sub)),
        "countries": int(sub["country"].nunique()),
        "r2":        float(model.rsquared),
    }
    for i, c in enumerate(x_cols):
        out[c] = {
            "beta": float(model.params[i]),
            "se":   float(model.bse[i]),
            "p":    float(model.pvalues[i]),
        }
    return out


# ── Define specs ─────────────────────────────────────────────────────

SPECS = [
    # (label, regressor list)
    ("[1]  pri(T) alone",                   ["pri_t"]),
    ("[2]  low(T) alone",                   ["low_t"]),
    ("[3]  upp(T) alone",                   ["upp_t"]),
    ("[4]  pri(T-28) alone",                ["pri_t28"]),
    ("[5]  low(T-28) alone",                ["low_t28"]),
    ("[6]  upp(T-28) alone",                ["upp_t28"]),
    ("[7]  pri+low+upp at T",               ["pri_t","low_t","upp_t"]),
    ("[8]  pri+low+upp at T-28",            ["pri_t28","low_t28","upp_t28"]),
    ("[9]  pri(T) + pri(T-28)",             ["pri_t","pri_t28"]),
    ("[10] low(T) + low(T-28)",             ["low_t","low_t28"]),
    ("[11] upp(T) + upp(T-28)",             ["upp_t","upp_t28"]),
    ("[12] kitchen sink (all 6)",           ["pri_t","low_t","upp_t","pri_t28","low_t28","upp_t28"]),
]

OUTCOMES = [
    ("log(TFR)", "log_tfr"),
    ("log(LE)",  "log_le"),
    ("log(U5)",  "log_u5"),
    ("log(GDP)", "log_gdp"),
]


# ── Run and print ───────────────────────────────────────────────────

results = {}
for o_label, o_col in OUTCOMES:
    print(f"\n{'#'*100}")
    print(f"# OUTCOME: {o_label}")
    print(f"{'#'*100}")

    # Same sample for this outcome across all specs
    needed = ["pri_t","low_t","upp_t","pri_t28","low_t28","upp_t28", o_col, "country"]
    sub = panel[needed].dropna().copy()

    # Drop singleton countries
    ok = sub.groupby("country")[o_col].transform("count") > 1
    sub = sub[ok]

    print(f"  Sample: n={len(sub)}, countries={sub['country'].nunique()}\n")
    print(f"  {'Spec':<32}  {'Regressor':<10}  {'β':>10}  {'SE':>9}  {'p':>9}  {'R²':>6}  {'n':>5}")
    print("  " + "-"*98)

    outcome_results = {}
    for label, x_cols in SPECS:
        r = fe_reg(sub, o_col, x_cols)
        if r is None:
            print(f"  {label:<32}  (insufficient data)")
            continue
        outcome_results[label] = r
        first = True
        for x in x_cols:
            row = r[x]
            star = ""
            if row["p"] < 0.001: star = "***"
            elif row["p"] < 0.01: star = "**"
            elif row["p"] < 0.05: star = "*"
            if first:
                print(f"  {label:<32}  {x:<10}  {row['beta']:>+10.5f}  {row['se']:>9.5f}  "
                      f"{row['p']:>9.2e}  {r['r2']:>6.3f}  {r['n']:>5d}  {star}")
                first = False
            else:
                print(f"  {'':<32}  {x:<10}  {row['beta']:>+10.5f}  {row['se']:>9.5f}  "
                      f"{row['p']:>9.2e}  {'':>6}  {'':>5}  {star}")
        print()

    results[o_label] = outcome_results


# ── Headline take-aways ─────────────────────────────────────────────

print("\n" + "="*100)
print("KITCHEN-SINK TAKE-AWAYS  (β with country-clustered p — unique contribution after all controls)")
print("="*100)
print(f"{'Outcome':<10}  {'pri(T)':>20}  {'low(T)':>20}  {'upp(T)':>20}")
print(f"{'':<10}  {'pri(T-28)':>20}  {'low(T-28)':>20}  {'upp(T-28)':>20}")
print("-"*100)
for o_label, _ in OUTCOMES:
    r = results.get(o_label, {}).get("[12] kitchen sink (all 6)", None)
    if r is None:
        continue
    def cell(k):
        x = r[k]
        star = "***" if x["p"]<0.001 else "**" if x["p"]<0.01 else "*" if x["p"]<0.05 else " "
        return f"{x['beta']:+9.5f} ({x['p']:.0e}){star}".rjust(20)
    print(f"{o_label:<10}  {cell('pri_t')}  {cell('low_t')}  {cell('upp_t')}")
    print(f"{'':<10}  {cell('pri_t28')}  {cell('low_t28')}  {cell('upp_t28')}")
    print()


# ── Variance decomposition (incremental R²) ─────────────────────────

print("="*100)
print("INCREMENTAL R²: how much does each block add?")
print(" 'lag0 only' = R² of {pri(T),low(T),upp(T)}")
print(" 'lag28 only' = R² of {pri(T-28),low(T-28),upp(T-28)}")
print(" 'kitchen' = R² of all 6")
print(" gain_from_lag28 = kitchen - lag0 (how much new info do parental-gen edus add)")
print(" gain_from_lag0  = kitchen - lag28 (how much new info do current-cohort edus add)")
print("="*100)
print(f"{'Outcome':<10}  {'lag0 only':>10}  {'lag28 only':>11}  {'kitchen':>9}  "
      f"{'gain_lag28':>11}  {'gain_lag0':>10}")
print("-"*100)
incrementals = {}
for o_label, _ in OUTCOMES:
    r0  = results.get(o_label, {}).get("[7]  pri+low+upp at T", {})
    r28 = results.get(o_label, {}).get("[8]  pri+low+upp at T-28", {})
    rk  = results.get(o_label, {}).get("[12] kitchen sink (all 6)", {})
    if not (r0 and r28 and rk):
        continue
    gain_28 = rk["r2"] - r0["r2"]
    gain_0  = rk["r2"] - r28["r2"]
    incrementals[o_label] = dict(lag0_r2=r0["r2"], lag28_r2=r28["r2"],
                                  kitchen_r2=rk["r2"],
                                  gain_from_lag28=gain_28, gain_from_lag0=gain_0)
    print(f"{o_label:<10}  {r0['r2']:>10.3f}  {r28['r2']:>11.3f}  {rk['r2']:>9.3f}  "
          f"{gain_28:>+11.3f}  {gain_0:>+10.3f}")


# ── Checkin ──────────────────────────────────────────────────────────

flat = {}
for o_label, _ in OUTCOMES:
    o_key = o_label.replace("(", "").replace(")", "").replace(" ", "_").lower()
    for spec_label, r in results.get(o_label, {}).items():
        # canonical spec id from leading [n]
        spec_id = spec_label.split("]")[0].lstrip("[").strip()
        flat[f"{o_key}.spec{spec_id}.r2"] = round(r["r2"], 4)
        flat[f"{o_key}.spec{spec_id}.n"]  = r["n"]
        for k, v in r.items():
            if isinstance(v, dict):
                for stat in ("beta","se","p"):
                    flat[f"{o_key}.spec{spec_id}.{k}.{stat}"] = round(v[stat], 6)
    for k, v in incrementals.get(o_label, {}).items():
        flat[f"{o_key}.{k}"] = round(v, 4)

checkin = {
    "method": (
        "Horse-race country-FE regressions with the cohort's own education (T) "
        "and the prior generation's education (T-28) at three levels (primary, "
        "lower secondary, upper secondary). Each OUTCOME is read forward at its "
        "biological horizon: TFR at T+5, U5MR and LE at T+12, GDP at T+0. With "
        "the two education blocks fixed at T and T-28 while the outcome sits at "
        "T+lag, the T block is the parent generation (leading the outcome by its "
        "biological lag) and the T-28 block is the grandparent generation, "
        "matching the per-outcome lags of the §The Grandparent Channel "
        "decomposition. Same sample within each outcome across all 12 specs "
        "(entry-cohort gating on lower_sec≥10 at T, ceiling 90). Country-"
        "clustered SEs. T = 1975-2015 5yr stops (so T-28 ≥ 1950 within WCDE "
        "coverage); U5MR's sample is shorter because WDI U5MR ends in 2015, so "
        "T+12 ≤ 2015 limits it to T ≤ 2003. Tests which level/block carries "
        "unique signal after the others are controlled for."
    ),
    "specs": {
        s_label: x_cols for s_label, x_cols in SPECS
    },
    "results":      results,
    "incremental":  incrementals,
    "numbers":      flat,
}

write_checkin("horse_race_lags_levels.json", checkin,
              "scripts/residualization/horse_race_lags_levels.py")
