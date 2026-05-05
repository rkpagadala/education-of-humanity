"""
cases/figure2_betas.py

Verify Figure 3 country-specific sliding-window betas. The paper cites:
  - USA: beta=1.9 at 13% baseline, declining to beta=0.08 at 92%
  - Korea: beta=6.5 at 1%, beta=3.6, beta=1.8, beta=0.2 at 59%
  - Taiwan: beta=5.1 at 1.2%
  - Philippines: beta=4.4 at 1.5%, declining to beta=0.4 at 49%
  - Bangladesh: beta~2 through 1-18% baseline
  - India: beta~2 through 28% baseline

Runs the same logic as scripts/figures/beta_vs_baseline.py and extracts
the specific betas mentioned in the paper.

Data source:
  wcde/data/processed/cohort_completion_both_long.csv

Output: checkin/figure2_betas.json
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _shared import PROC, write_checkin

# ── Parameters (same as figures/beta_vs_baseline.py) ────────────────────────────
WINDOW_SIZE = 25
STEP = 10
LAG = 25
MIN_OBS = 3

# ── Load data ────────────────────────────────────────────────────────────────
longrun = pd.read_csv(os.path.join(PROC, "cohort_completion_both_long.csv"))
wide = longrun.pivot(index="country", columns="cohort_year", values="lower_sec")


def v(country, year):
    try:
        val = float(wide.loc[country, int(year)])
        return val if not np.isnan(val) else np.nan
    except (KeyError, ValueError):
        return np.nan


def beta_for_window(country, child_start, child_end):
    """OLS beta for child cohorts in [child_start, child_end], parent = child - LAG."""
    rows = []
    for cy in range(child_start, child_end + 1, 5):
        py = cy - LAG
        child = v(country, cy)
        parent = v(country, py)
        if not np.isnan(child) and not np.isnan(parent):
            rows.append({"child": child, "parent": parent})
    if len(rows) < MIN_OBS:
        return np.nan, np.nan
    df = pd.DataFrame(rows)
    reg = sm.OLS(df["child"], sm.add_constant(df[["parent"]])).fit()
    return reg.params.iloc[1], df["parent"].mean()


# ── Countries (same as figures/beta_vs_baseline.py) ─────────────────────────────
COUNTRIES = [
    ("United States of America",       "USA",         1900),
    ("Republic of Korea",              "Korea",       1920),
    ("Taiwan Province of China",       "Taiwan",      1930),
    ("Philippines",                    "Philippines", 1920),
    ("Bangladesh",                     "Bangladesh",  1900),
    ("India",                          "India",       1900),
]

# ── Compute all sliding-window betas ────────────────────────────────────────
print("=" * 70)
print("FIGURE 2 VERIFICATION: Sliding-window betas by country")
print("=" * 70)

all_results = {}
for country_name, label, start_year in COUNTRIES:
    windows = []
    print(f"\n  {label}:")
    print(f"  {'Window':<15} {'beta':>8} {'Avg parent%':>12}")
    for start in range(start_year, 1996, STEP):
        end = start + WINDOW_SIZE
        beta, avg_parent = beta_for_window(country_name, start, end)
        if not np.isnan(beta):
            print(f"  {start}-{end:<10} {beta:>8.3f} {avg_parent:>11.1f}%")
            windows.append({
                "window": f"{start}-{end}",
                "beta": round(beta, 3),
                "avg_parent_pct": round(avg_parent, 1),
            })
    all_results[label] = windows

# ── Extract paper-cited values ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("PAPER CLAIM VERIFICATION")
print("=" * 70)

paper_claims = {
    "USA": [
        {"baseline_approx": 13, "beta_claimed": 1.9},
        {"baseline_approx": 92, "beta_claimed": 0.08},
    ],
    "Korea": [
        {"baseline_approx": 1, "beta_claimed": 6.5},
        {"baseline_approx": 3, "beta_claimed": 3.6},
        {"baseline_approx": 23, "beta_claimed": 1.8},
        {"baseline_approx": 59, "beta_claimed": 0.2},
    ],
    "Taiwan": [
        {"baseline_approx": 1.2, "beta_claimed": 5.1},
    ],
    "Philippines": [
        {"baseline_approx": 1.5, "beta_claimed": 4.4},
        {"baseline_approx": 49, "beta_claimed": 0.4},
    ],
    "Bangladesh": [
        {"baseline_approx": "1-18", "beta_claimed": "~2"},
    ],
    "India": [
        {"baseline_approx": 28, "beta_claimed": "~2"},
    ],
}

verification = {}
for label, claims in paper_claims.items():
    windows = all_results.get(label, [])
    if not windows:
        print(f"\n  {label}: NO DATA")
        verification[label] = {"status": "no_data"}
        continue

    print(f"\n  {label}:")
    country_verification = []
    for claim in claims:
        bl = claim["baseline_approx"]
        bc = claim["beta_claimed"]

        if isinstance(bc, str):
            # Approximate claim (e.g., "~2")
            target_beta = float(bc.replace("~", ""))
            if isinstance(bl, str):
                # Range like "1-18"
                lo, hi = [float(x) for x in bl.split("-")]
                matching = [w for w in windows
                            if lo - 2 <= w["avg_parent_pct"] <= hi + 2]
                if matching:
                    actual_betas = [w["beta"] for w in matching]
                    avg_actual = np.mean(actual_betas)
                    print(f"    Claim: beta{bc} at {bl}% → actual avg beta={avg_actual:.2f} "
                          f"(range {min(actual_betas):.2f}-{max(actual_betas):.2f})")
                    country_verification.append({
                        "claim_baseline": bl,
                        "claim_beta": bc,
                        "actual_avg_beta": round(avg_actual, 3),
                        "actual_range": [round(min(actual_betas), 3), round(max(actual_betas), 3)],
                    })
                else:
                    print(f"    Claim: beta{bc} at {bl}% → no matching windows")
                    country_verification.append({
                        "claim_baseline": bl,
                        "claim_beta": bc,
                        "actual": "no matching windows",
                    })
            else:
                # Single baseline value
                closest = min(windows, key=lambda w: abs(w["avg_parent_pct"] - bl))
                print(f"    Claim: beta{bc} at {bl}% → closest: beta={closest['beta']:.2f} "
                      f"at {closest['avg_parent_pct']}% ({closest['window']})")
                country_verification.append({
                    "claim_baseline": bl,
                    "claim_beta": bc,
                    "actual_beta": closest["beta"],
                    "actual_baseline": closest["avg_parent_pct"],
                    "window": closest["window"],
                })
        else:
            # Exact claim
            closest = min(windows, key=lambda w: abs(w["avg_parent_pct"] - bl))
            diff_beta = abs(closest["beta"] - bc)
            status = "OK" if diff_beta < 0.5 else "MISMATCH"
            print(f"    Claim: beta={bc} at {bl}% → actual: beta={closest['beta']:.2f} "
                  f"at {closest['avg_parent_pct']}% ({closest['window']}) [{status}]")
            country_verification.append({
                "claim_baseline": bl,
                "claim_beta": bc,
                "actual_beta": closest["beta"],
                "actual_baseline": closest["avg_parent_pct"],
                "window": closest["window"],
                "diff": round(diff_beta, 3),
                "status": status,
            })

    verification[label] = country_verification

# ── Write checkin ────────────────────────────────────────────────────────────
write_checkin("figure2_betas.json", {
    "parameters": {
        "window_size": WINDOW_SIZE,
        "step": STEP,
        "lag": LAG,
        "min_obs": MIN_OBS,
    },
    "all_windows": {label: windows for label, windows in all_results.items()},
    "verification": verification,
}, script_path="scripts/cases/figure2_betas.py")
