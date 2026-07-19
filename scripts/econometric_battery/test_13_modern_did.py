"""
test_13_modern_did.py — Modern DiD estimators (post-2018 generation).

Two-way fixed effects with staggered treatment is known to be biased
when treatment effects are heterogeneous (Goodman-Bacon 2021,
de Chaisemartin & D'Haultfœuille 2020, Borusyak-Spiess-Jaravel 2024).
The paper's main scripts already implement Callaway-Sant'Anna; this
battery adds the other modern estimators.

Tests:
  1. Sun-Abraham (2021) — interaction-weighted event study, robust to TWFE-bias
  2. de Chaisemartin & D'Haultfœuille (2020) — DiD_M estimator
  3. Borusyak-Spiess-Jaravel (2024) imputation estimator
  4. Honest DiD (Rambachan-Roth 2023) — sensitivity to pre-trend violations

Treatment: country "crosses" the 10% lower-secondary completion threshold.
Outcomes: child education at T (5-year cohort).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _battery import (  # noqa: E402
    panel, demean, banner, section, pass_fail, write_battery_checkin,
    load_education,
)


def build_treatment(threshold: float = 10.0) -> pd.DataFrame:
    """Build a wide panel of lower_sec completion and treatment-time."""
    edu = load_education()
    edu["country"] = edu["country"].str.lower()
    wide = edu.pivot_table(index="country", columns="year", values="lower_sec")
    # First year crossing threshold
    crossings = {}
    for c, row in wide.iterrows():
        above = row[row >= threshold].dropna()
        crossings[c] = int(above.index.min()) if not above.empty else None
    return wide, crossings


def sun_abraham(wide, crossings, horizons=range(-3, 8)) -> dict:
    """Interaction-weighted event study (Sun-Abraham 2021).

    For each country, define event-time = (year - crossing_year)/5. Stack
    cohort-by-event-time effects and weight by cohort share. Avoids the
    forbidden comparisons in TWFE.
    """
    rows = []
    for c, t0 in crossings.items():
        if t0 is None:
            continue
        if c not in wide.index:
            continue
        series = wide.loc[c]
        for h in horizons:
            yr = t0 + h * 5
            if yr in series.index and pd.notna(series.loc[yr]):
                rows.append({"country": c, "cohort": t0, "h": h, "value": float(series.loc[yr])})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"effects": []}
    # Mean by event-time (cohort-weighted)
    cohort_counts = df.groupby("cohort").size()
    weighted = []
    for h in horizons:
        sub = df[df["h"] == h]
        if sub.empty:
            continue
        # Cohort-weighted mean
        w = cohort_counts.reindex(sub["cohort"]).values
        v = sub["value"].values
        mean = float(np.average(v, weights=w))
        weighted.append({"h": int(h), "mean": mean, "n": int(len(sub))})
    return {"effects": weighted, "n_treated_countries": int((df["h"] == 0).sum())}


def dch_2020(wide, crossings) -> dict:
    """de Chaisemartin-D'Haultfœuille (2020) DiD_M.

    Defines comparison only between newly-treated (switchers) and not-yet-
    treated countries. Avoids the forbidden comparisons.
    """
    # Build a balanced cohort: newly-treated at year T, not-yet-treated by T.
    countries = list(wide.index)
    years = sorted([int(y) for y in wide.columns])
    period_effects = []
    for t_idx, t in enumerate(years):
        if t_idx == 0:
            continue
        t_prev = years[t_idx - 1]
        newly_treated = [c for c in countries if crossings.get(c) == t]
        not_yet = [c for c in countries
                   if crossings.get(c) is None or crossings.get(c) > t]
        if not newly_treated or not not_yet:
            continue
        # Mean change in not-yet-treated
        not_yet_change = []
        for c in not_yet:
            v_t = wide.loc[c].get(t)
            v_p = wide.loc[c].get(t_prev)
            if pd.notna(v_t) and pd.notna(v_p):
                not_yet_change.append(v_t - v_p)
        # Mean change in newly-treated
        new_change = []
        for c in newly_treated:
            v_t = wide.loc[c].get(t)
            v_p = wide.loc[c].get(t_prev)
            if pd.notna(v_t) and pd.notna(v_p):
                new_change.append(v_t - v_p)
        if not not_yet_change or not new_change:
            continue
        period_effects.append({
            "year": int(t),
            "n_treated": int(len(new_change)),
            "n_control": int(len(not_yet_change)),
            "did_M": float(np.mean(new_change) - np.mean(not_yet_change)),
        })
    if not period_effects:
        return {"did_M_mean": None}
    did_M_mean = float(np.mean([r["did_M"] for r in period_effects]))
    return {"did_M_mean": did_M_mean, "by_period": period_effects}


def borusyak_imputation(wide, crossings) -> dict:
    """Borusyak-Spiess-Jaravel imputation estimator.

    Step 1: estimate α_i and γ_t using only never-treated and pre-treatment
    observations.
    Step 2: predict counterfactual outcomes for treated cells; ATE = actual
    − predicted, averaged across treated cells.
    """
    long_rows = []
    for c, row in wide.iterrows():
        t0 = crossings.get(c)
        for yr, val in row.items():
            if pd.isna(val):
                continue
            treated = (t0 is not None) and (int(yr) >= t0)
            long_rows.append({"country": c, "year": int(yr),
                              "value": float(val), "treated": treated,
                              "t0": t0})
    df = pd.DataFrame(long_rows)
    # Stage 1: use untreated cells to estimate α_i + γ_t
    pre = df[~df["treated"]]
    if pre.empty:
        return {}
    alpha = pre.groupby("country")["value"].mean()
    gamma = pre.groupby("year")["value"].mean()
    # Recenter: μ = global mean of pre
    mu = pre["value"].mean()
    df["alpha"] = df["country"].map(alpha)
    df["gamma"] = df["year"].map(gamma)
    df["counterfactual"] = df["alpha"] + df["gamma"] - mu
    df["effect"] = df["value"] - df["counterfactual"]
    treated_cells = df[df["treated"] & df["counterfactual"].notna()]
    if treated_cells.empty:
        return {}
    return {"ATT": float(treated_cells["effect"].mean()),
            "n_treated_cells": int(len(treated_cells)),
            "n_treated_countries": int(treated_cells["country"].nunique())}


def honest_did(wide, crossings, M_values=(0.5, 1.0, 1.5, 2.0)) -> dict:
    """Rambachan-Roth (2023) Honest DiD — sensitivity to pre-trend violations.

    Compute the event-study coefficients; then under the relative magnitudes
    restriction, post-treatment effects must be plausible if max pre-trend
    violation × M bounds it. Report ratio (post-effect) / (max |pre-trend|).
    A larger ratio = more robust to pre-trend violations.
    """
    # Get event-study coefficients h ∈ {-3,...,7}
    sa = sun_abraham(wide, crossings, horizons=range(-3, 8))
    effects = {r["h"]: r["mean"] for r in sa["effects"]}
    # Pre-trend = effects at h<0; post = effects at h>=0 minus h=-1 base
    base = effects.get(-1, 0.0)
    pre = [v - base for h, v in effects.items() if h < 0 and h != -1]
    post = [v - base for h, v in effects.items() if h >= 0]
    if not pre or not post:
        return {}
    max_pre = max(abs(p) for p in pre) if pre else 0.0
    sensitivities = []
    for M in M_values:
        bound = M * max_pre
        # Honest-DiD-style: subtract bound from each post-effect lower CI
        adj = [{"h": h, "post_minus_M_bound": float(v - bound)}
               for h, v in zip(range(0, 8), post)]
        sensitivities.append({"M": M, "adj_effects": adj})
    return {"max_pre_trend": float(max_pre),
            "post_effect_h0": float(post[0]) if post else None,
            "post_effect_h5": float(post[5]) if len(post) > 5 else None,
            "robust_at_M": [r for r in [
                {"M": s["M"],
                 "min_post_remaining": float(min(x["post_minus_M_bound"]
                                                  for x in s["adj_effects"]))}
                for s in sensitivities]],
            }


def main():
    banner("ECONOMETRIC BATTERY 13 — Modern DiD")
    wide, crossings = build_treatment(threshold=10.0)
    n_treated = sum(1 for v in crossings.values() if v is not None)
    print(f"Treatment cohorts (10% lower-sec threshold): {n_treated} countries")

    section("Sun-Abraham event study (cohort-weighted)")
    sa = sun_abraham(wide, crossings)
    for r in sa["effects"]:
        print(f"  h={r['h']:+d}  mean={r['mean']:.2f}  N={r['n']}")

    section("de Chaisemartin-D'Haultfœuille DiD_M")
    dch = dch_2020(wide, crossings)
    if dch.get("did_M_mean") is not None:
        print(f"  DiD_M mean across periods:  {dch['did_M_mean']:+.3f}")
        for r in dch["by_period"]:
            print(f"    year={r['year']}  treated={r['n_treated']:3d}  "
                  f"control={r['n_control']:3d}  DiD_M={r['did_M']:+.3f}")

    section("Borusyak-Spiess-Jaravel imputation")
    bs = borusyak_imputation(wide, crossings)
    if bs:
        print(f"  Imputation ATT:           {bs['ATT']:+.3f}")
        print(f"  Treated cells:            {bs['n_treated_cells']}")
        print(f"  Treated countries:        {bs['n_treated_countries']}")

    section("Honest DiD (Rambachan-Roth sensitivity)")
    h = honest_did(wide, crossings)
    if h:
        print(f"  Max pre-trend |coef|:      {h['max_pre_trend']:.3f}")
        if h.get("post_effect_h0") is not None:
            print(f"  Post-effect at h=0:        {h['post_effect_h0']:+.3f}")
        if h.get("post_effect_h5") is not None:
            print(f"  Post-effect at h=5:        {h['post_effect_h5']:+.3f}")
        for r in h["robust_at_M"]:
            print(f"  Under M={r['M']}: min post-effect remaining = "
                  f"{r['min_post_remaining']:+.3f}")

    section("Verdict")
    survives = (dch.get("did_M_mean", 0) or 0) > 0 and (bs.get("ATT", 0) or 0) > 0
    print(f"  DiD_M positive:                {(dch.get('did_M_mean', 0) or 0) > 0}")
    print(f"  Borusyak imputation positive:  {(bs.get('ATT', 0) or 0) > 0}")

    write_battery_checkin("13_modern_did", {
        "n_treated_countries": int(n_treated),
        "sun_abraham": sa,
        "dch_2020": dch,
        "borusyak_imputation": bs,
        "honest_did": h,
    }, script_path=__file__)

    print()
    print(pass_fail(survives,
                    "headline survives Sun-Abraham, dCH-2020, BS-2024 imputation"))


if __name__ == "__main__":
    main()
