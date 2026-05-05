"""
cases/threshold_cutoffs.py

Produce and verify Table A1 cutoff-specific rows from "Education of Nations":
Two-way fixed effects (country + year) regressions — child lower secondary
completion on parental education, at each baseline cutoff.

Uses iterative two-way demeaning (alternating country/year mean removal until
convergence), then no-intercept OLS with cluster-robust SEs.  R² is from the
converged demeaned regression (partial within-R²: variance explained by
parent education beyond country + year FEs).

Data sources:
  - Education: wcde/data/processed/lower_sec_both.csv (WCDE v3)

Output: checkin/table_a1_cutoffs.json
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _shared import PROC, REGIONS, write_checkin

# ── Constants ────────────────────────────────────────────────────────────────
PARENTAL_LAG = 25
OUTCOME_YEARS = list(range(1975, 2016, 5))

NON_SOVEREIGN = REGIONS

# ── Load education data ─────────────────────────────────────────────────────
agg = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"), index_col="country")

# ── Build panel ──────────────────────────────────────────────────────────────
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
        rows.append({
            "country": country,
            "year": y,
            "child": child,
            "parent": parent,
        })

panel = pd.DataFrame(rows)
print(f"Full panel: {len(panel)} obs, {panel['country'].nunique()} countries")


# ── Two-way FE: iterative demeaning ─────────────────────────────────────────
def two_way_fe(df, x_col="parent", y_col="child", cluster_col="country",
               max_iter=200, tol=1e-12):
    """
    Iterative two-way demeaning until convergence, then no-intercept OLS
    with cluster-robust SEs.  Equivalent to proper two-way FE.
    """
    d = df.dropna(subset=[x_col, y_col]).copy()
    d["x_dm"] = d[x_col].values.copy()
    d["y_dm"] = d[y_col].values.copy()

    for i in range(max_iter):
        x_old = d["x_dm"].values.copy()
        d["x_dm"] = d.groupby(cluster_col)["x_dm"].transform(lambda v: v - v.mean())
        d["y_dm"] = d.groupby(cluster_col)["y_dm"].transform(lambda v: v - v.mean())
        d["x_dm"] = d.groupby("year")["x_dm"].transform(lambda v: v - v.mean())
        d["y_dm"] = d.groupby("year")["y_dm"].transform(lambda v: v - v.mean())
        if np.max(np.abs(d["x_dm"].values - x_old)) < tol:
            break

    model = sm.OLS(d["y_dm"], d[["x_dm"]]).fit(
        cov_type="cluster",
        cov_kwds={"groups": d[cluster_col]},
    )
    return model, len(d), d[cluster_col].nunique(), i + 1


# ── Cutoff analysis ─────────────────────────────────────────────────────────
CUTOFFS = [30, 20, 10, None]

# Paper claims (Table A1, current paper)
PAPER = {
    30:   {"beta": 0.739, "r2": 0.154, "n": 783, "countries": 137},
    20:   {"beta": 1.032, "r2": 0.145, "n": 600, "countries": 118},
    10:   {"beta": 1.019, "r2": 0.067, "n": 358, "countries": 85},
    None: {"beta": 0.083, "r2": 0.009, "n": 1665, "countries": 185},
}

print("\n" + "=" * 70)
print("TABLE A1 CUTOFF ROWS (iterative two-way FE, clustered SEs)")
print("=" * 70)

results = {}
all_pass = True

for cutoff in CUTOFFS:
    if cutoff is not None:
        sub = panel[panel["parent"] < cutoff]
        label = f"cutoff_{cutoff}"
        display = f"<{cutoff}%"
    else:
        sub = panel
        label = "all"
        display = "all"

    model, n, nc, iters = two_way_fe(sub)
    beta = model.params.iloc[0]
    se = model.bse.iloc[0]
    t = model.tvalues.iloc[0]
    p = model.pvalues.iloc[0]
    r2 = model.rsquared

    results[label] = {
        "beta": round(beta, 3),
        "se": round(se, 3),
        "t": round(t, 1),
        "p": round(p, 4),
        "r2": round(r2, 3),
        "n": n,
        "countries": nc,
        "iterations": iters,
    }

    claim = PAPER[cutoff]
    beta_ok = abs(beta - claim["beta"]) < 0.002
    r2_ok = abs(r2 - claim["r2"]) < 0.002
    n_ok = n == claim["n"]
    nc_ok = nc == claim["countries"]
    ok = beta_ok and r2_ok and n_ok and nc_ok

    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False

    print(f"\n  {display}: beta={beta:.3f} R2={r2:.3f} n={n} countries={nc}  "
          f"(iters={iters})  [{status}]")
    if not beta_ok:
        print(f"    beta: paper={claim['beta']}  actual={round(beta, 3)}")
    if not r2_ok:
        print(f"    R2:   paper={claim['r2']}  actual={round(r2, 3)}")
    if not n_ok:
        print(f"    n:    paper={claim['n']}  actual={n}")
    if not nc_ok:
        print(f"    ctry: paper={claim['countries']}  actual={nc}")

print(f"\n{'ALL PASS' if all_pass else 'SOME FAILURES'}")

# ── Write checkin ────────────────────────────────────────────────────────────
write_checkin("table_a1_cutoffs.json", {
    "method": "Iterative two-way demeaning (country + year) until convergence, "
              "no-intercept OLS, cluster-robust SEs by country.",
    "notes": f"Full panel: {results.get('all', {}).get('n', '?')} obs, "
             f"{results.get('all', {}).get('countries', '?')} countries.",
    "numbers": results,
    "paper_claims": {str(k): v for k, v in PAPER.items()},
    "all_pass": all_pass,
}, script_path="scripts/cases/threshold_cutoffs.py")
