"""
test_14_synthetic_control.py — Synthetic control for canonical cases.

Implements Abadie-Diamond-Hainmueller (2010) synthetic control for the
canonical singular-priority and over-performer cases:
  - South Korea (singular priority, post-1953)
  - Taiwan (singular priority, post-1949)
  - Cuba (singular priority, post-1959)
  - Sri Lanka (competing priority, free education 1945)
  - Bangladesh (over-performer, post-1990)
  - Cambodia (Khmer Rouge schooling disruption 1975-79)
  - Philippines (counterfactual to Korea — same 1950 starting position)

For each, we construct a synthetic donor pool from countries with
comparable 1950 income and education levels, fit optimal weights on
pre-period attainment, and compute the gap between actual and synthetic
trajectories on outcomes (LE, TFR, U5MR).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _battery import (  # noqa: E402
    banner, section, pass_fail, write_battery_checkin, _wb_long, load_education,
)

CASES = [
    {"name": "South Korea",   "country": "republic of korea",  "T0": 1960},
    {"name": "Taiwan",        "country": "taiwan province of china", "T0": 1955},
    {"name": "Cuba",          "country": "cuba",                "T0": 1965},
    {"name": "Sri Lanka",     "country": "sri lanka",           "T0": 1955},
    {"name": "Bangladesh",    "country": "bangladesh",          "T0": 1995},
    {"name": "Cambodia",      "country": "cambodia",            "T0": 1980},
    {"name": "Philippines",   "country": "philippines",         "T0": 1960},
]

OUTCOMES = {
    "le":   "life_expectancy_years.csv",
    "tfr":  "children_per_woman_total_fertility.csv",
    "u5mr": "child_mortality_u5.csv",
}


def build_indicator(filename: str) -> pd.DataFrame:
    return _wb_long(filename, "value").rename(columns={"value": "v"})


def sc_weights(treated_pre: np.ndarray, donors_pre: np.ndarray) -> np.ndarray:
    """Solve: min ||y_t - W' Y_donors||² s.t. w ≥ 0, sum w = 1."""
    n = donors_pre.shape[1]
    w0 = np.ones(n) / n

    def loss(w):
        return float(np.sum((treated_pre - donors_pre @ w) ** 2))
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n)]
    res = minimize(loss, w0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    if not res.success:
        return w0
    return res.x


def synthetic_control_case(case, indicators: dict) -> dict:
    edu = load_education()
    edu["country"] = edu["country"].str.lower()
    edu_wide = edu.pivot_table(index="country", columns="year", values="lower_sec")
    treated = case["country"]
    T0 = case["T0"]
    pre_years = [y for y in edu_wide.columns if y <= T0]
    if treated not in edu_wide.index:
        return {"available": False}
    treated_pre = edu_wide.loc[treated, pre_years].dropna()
    if len(treated_pre) < 3:
        return {"available": False, "n_pre": int(len(treated_pre))}
    common_pre = treated_pre.index.tolist()
    # Donor pool: countries with full pre-period attainment data
    donor_pool = edu_wide.dropna(subset=common_pre, how="any").index.tolist()
    donor_pool = [c for c in donor_pool if c != treated]
    donors_pre = edu_wide.loc[donor_pool, common_pre].values.T
    treated_vec = treated_pre.loc[common_pre].values
    w = sc_weights(treated_vec, donors_pre)
    top_donors = sorted(zip(donor_pool, w), key=lambda x: -x[1])[:5]
    # Compute gap on each outcome
    gaps = {}
    for name, file_ in OUTCOMES.items():
        ind = indicators[name]
        wide = ind.pivot_table(index="country", columns="year", values="v")
        if treated not in wide.index:
            continue
        years = sorted(wide.columns)
        # Use only years present in donors' availability
        treated_y = wide.loc[treated].dropna()
        synth_y = pd.Series(index=treated_y.index, dtype=float)
        for yr in treated_y.index:
            donor_vals = []
            donor_w = []
            for c, wt in zip(donor_pool, w):
                if c in wide.index and yr in wide.columns and pd.notna(wide.loc[c, yr]):
                    donor_vals.append(wide.loc[c, yr])
                    donor_w.append(wt)
            if donor_w and sum(donor_w) > 0:
                synth_y.loc[yr] = float(np.average(donor_vals, weights=donor_w))
        gap = (treated_y - synth_y).dropna()
        post_gap = gap[gap.index > T0]
        pre_gap = gap[gap.index <= T0]
        gaps[name] = {
            "pre_rmse":  float(np.sqrt((pre_gap ** 2).mean())) if len(pre_gap) else None,
            "post_mean": float(post_gap.mean()) if len(post_gap) else None,
            "post_end":  float(post_gap.iloc[-1]) if len(post_gap) else None,
            "T_end":     int(post_gap.index[-1]) if len(post_gap) else None,
        }
    return {
        "available": True,
        "case": case["name"],
        "country": treated,
        "T0": T0,
        "n_pre_years": int(len(common_pre)),
        "top_5_donors": [{"country": c, "weight": float(w)} for c, w in top_donors],
        "gaps": gaps,
    }


def main():
    banner("ECONOMETRIC BATTERY 14 — Synthetic control (canonical cases)")
    indicators = {n: build_indicator(f) for n, f in OUTCOMES.items()}

    out = []
    for case in CASES:
        section(f"{case['name']} (T0 = {case['T0']})")
        r = synthetic_control_case(case, indicators)
        out.append(r)
        if not r.get("available"):
            print(f"  insufficient data (n_pre={r.get('n_pre', 0)})")
            continue
        print(f"  Top 5 donors:")
        for d in r["top_5_donors"]:
            print(f"    {d['country']:<30}  w={d['weight']:.3f}")
        for outcome, g in r["gaps"].items():
            pre_rmse = f"{g['pre_rmse']:.2f}" if g['pre_rmse'] is not None else "n/a"
            post_mean = f"{g['post_mean']:+.2f}" if g['post_mean'] is not None else "n/a"
            post_end = f"{g['post_end']:+.2f}" if g['post_end'] is not None else "n/a"
            print(f"  {outcome:<6}  pre-RMSE={pre_rmse}  post-mean gap={post_mean}  post-end gap={post_end}")

    section("Summary")
    table = []
    for r in out:
        if not r.get("available"):
            continue
        row = {"case": r["case"]}
        for o in OUTCOMES.keys():
            row[f"{o}_post"] = r["gaps"].get(o, {}).get("post_end")
        table.append(row)
    print(pd.DataFrame(table).to_string(index=False))

    write_battery_checkin("14_synthetic_control", {
        "cases": out,
    }, script_path=__file__)

    print()
    print(pass_fail(True, "synthetic-control battery complete"))


if __name__ == "__main__":
    main()
