"""
le_independent_of_u5mr.py
=========================
Life-history label (§the-generational-lag):
  Predictor: 20-24 cohort completion at T-28. Outcome: LE at T,
  controlling for U5MR at T. Decomposes LE into child-mortality and
  adult-mortality channels. The adult channel (female AMRT) shows
  ~26.7% independent contribution. Generational relation: cross-
  generation step (one generation), with U5MR as mediator.

Does education have an independent effect on life expectancy beyond
what flows through U5MR?

QUESTION
--------
LE at birth is heavily pulled down by child mortality. If a country
cuts U5MR, LE rises mechanically — fewer deaths near zero raises the
mean age at death. The question: when we control for U5MR(T+28), does
education(T) still predict LE(T+28)? Or is education's LE benefit
entirely the U5MR effect?

If education has an LE channel beyond U5MR, that channel is the adult-
side benefit: longer lives among those who reach adulthood, driven by
the cognitive architecture loaded during the long childhood — planning
horizon, health knowledge, self-care, occupational hazards avoided,
adherence to treatment, etc.

METHOD
------
Country fixed effects, three specifications:

  Model A:  LE(T+28)   = β_edu * edu(T) + α_c + ε             (baseline)
  Model B:  U5MR(T+28) = β_edu * edu(T) + α_c + ε             (channel)
  Model C:  LE(T+28)   = β_edu * edu(T) + β_u5 * U5MR(T+28) + α_c + ε
                                                              (mediation)

Interpretation:
  - β_edu in A: total LE effect of education
  - β_edu in C: LE effect of education NOT routed through U5MR
  - (β_edu_A - β_edu_C) / β_edu_A: share routed through U5MR

Caveat: U5MR(T+28) is a post-treatment mediator of edu(T). Including it
gives a partial-mediation estimate, not a clean causal channel
decomposition. The interpretation is: how much of the LE coefficient
survives statistical control for the U5MR pathway.

DATA
----
- Education: WCDE v3 lower secondary completion (both sexes, age 20-24)
- LE: World Bank WDI SP.DYN.LE00.IN
- U5MR: World Bank WDI SH.DYN.MORT
- Panel: T ∈ {1975, 1980, 1985, 1990}, outcomes at T+28
- USSR republics excluded (Goskomstat anomaly, see paper §Hollow Education)

OUTPUT
------
Prints regression tables; writes checkin/le_independent_of_u5mr.json.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from _shared import (
    CHECKIN,
    fe_regression,
    get_wb_val,
    LAG_GENERATION,
    load_education,
    load_wb,
)

USSR_REPUBLICS = {
    "Armenia", "Azerbaijan", "Belarus", "Estonia", "Georgia",
    "Kazakhstan", "Kyrgyzstan", "Latvia", "Lithuania", "Moldova",
    "Russia", "Russian Federation", "Tajikistan", "Turkmenistan",
    "Ukraine", "Uzbekistan",
}

T_YEARS = [1975, 1980, 1985, 1990]
LAG = LAG_GENERATION  # 28; Human Cost uses the generational vantage: the cohort's own adult longevity one generation forward (T+28)


# ── Load data ───────────────────────────────────────────────────────

print("Loading data...")
edu = load_education("completion_both_long.csv")
le_raw = load_wb("life_expectancy_years.csv")
u5_raw = load_wb("child_mortality_u5.csv")
print(f"  Education: {edu['country'].nunique()} countries, "
      f"years {edu['year'].min()}-{edu['year'].max()}")
print(f"  LE: {len(le_raw)} countries")
print(f"  U5MR: {len(u5_raw)} countries")


# ── Build panel ─────────────────────────────────────────────────────

print(f"\nBuilding panel (T={T_YEARS}, lag={LAG} yr, USSR excluded)...")

rows = []
for c in sorted(edu["country"].unique()):
    if c in USSR_REPUBLICS:
        continue
    edu_c = edu[edu["country"] == c].set_index("year")
    for t in T_YEARS:
        if t not in edu_c.index:
            continue
        tp28 = t + LAG
        low = edu_c.loc[t, "lower_sec"]
        le_tp28 = get_wb_val(le_raw, c, tp28)
        u5_tp28 = get_wb_val(u5_raw, c, tp28)
        if np.isnan(low) or np.isnan(le_tp28) or np.isnan(u5_tp28):
            continue
        rows.append({
            "country": c,
            "t": t,
            "edu_t": low,
            "le_tp28": le_tp28,
            "u5_tp28": u5_tp28,
            "log_u5_tp28": np.log(u5_tp28) if u5_tp28 > 0 else np.nan,
        })

panel = pd.DataFrame(rows)
print(f"  Panel: {len(panel)} obs, {panel['country'].nunique()} countries")


# ── Regressions ─────────────────────────────────────────────────────

def report(name, model, n, k):
    coef = model.params
    se = model.bse
    t = model.tvalues
    p = model.pvalues
    r2 = model.rsquared
    print(f"\n{name}")
    print(f"  n={n}, countries={k}, R²(within)={r2:.4f}")
    for var in coef.index:
        print(f"  {var:18s}  β={coef[var]:+8.4f}  SE={se[var]:6.4f}  "
              f"t={t[var]:+6.2f}  p={p[var]:.4f}")
    return {
        "n": int(n), "countries": int(k), "r2_within": float(r2),
        "coefs": {
            v: {
                "beta": float(coef[v]),
                "se": float(se[v]),
                "t": float(t[v]),
                "p": float(p[v]),
            }
            for v in coef.index
        },
    }


print("\n" + "=" * 76)
print("FIXED-EFFECTS REGRESSIONS — does education predict LE beyond U5MR?")
print("=" * 76)

# Model A: total effect of education on LE
print("\nModel A: LE(T+28) = β·edu(T) + α_c [baseline total effect]")
mA, nA, kA = fe_regression(panel, ["edu_t"], "le_tp28")
A = report("[A] total LE effect", mA, nA, kA)

# Model B: effect of education on U5MR (confirms channel exists)
print("\nModel B: log U5MR(T+28) = β·edu(T) + α_c [channel verification]")
mB, nB, kB = fe_regression(panel.dropna(subset=["log_u5_tp28"]),
                            ["edu_t"], "log_u5_tp28")
B = report("[B] U5MR channel", mB, nB, kB)

# Model C: LE on edu controlling for U5MR (does edu survive?)
print("\nModel C: LE(T+28) = β·edu(T) + γ·log U5MR(T+28) + α_c [mediation]")
mC, nC, kC = fe_regression(panel.dropna(subset=["log_u5_tp28"]),
                            ["edu_t", "log_u5_tp28"], "le_tp28")
C = report("[C] LE | edu, U5MR", mC, nC, kC)

# Magnitudes (fe_regression demeans columns; coef keys carry _dm suffix)
beta_A = A["coefs"]["edu_t_dm"]["beta"]
beta_C = C["coefs"]["edu_t_dm"]["beta"]
p_C = C["coefs"]["edu_t_dm"]["p"]
share_through_u5 = (beta_A - beta_C) / beta_A if beta_A != 0 else float("nan")

print("\n" + "=" * 76)
print("INTERPRETATION")
print("=" * 76)
print(f"  Total edu→LE coefficient (Model A):              β = {beta_A:+.4f} yr/pp")
print(f"  Edu→LE coefficient net of U5MR (Model C):        β = {beta_C:+.4f} yr/pp")
print(f"  Share of total LE effect routed through U5MR:    {share_through_u5*100:.1f}%")
print(f"  Share of total LE effect from non-U5MR channels: {(1-share_through_u5)*100:.1f}%")
print(f"  Edu coefficient in C is {'SIGNIFICANT' if p_C < 0.05 else 'NOT significant'}"
      f" (p={p_C:.4f})")

if p_C < 0.05 and beta_C > 0:
    verdict = "Education predicts LE beyond what U5MR alone explains. The independent adult-LE channel is real."
elif beta_C <= 0 or p_C >= 0.05:
    verdict = "Education's LE effect is fully (or nearly) mediated by U5MR. No independent adult-LE channel detected in this panel."
else:
    verdict = "Inconclusive — check sign and significance."

print(f"\n  Verdict: {verdict}")


# ── Write checkin ───────────────────────────────────────────────────

out = {
    "spec": {
        "t_years": T_YEARS,
        "lag_years": LAG,
        "education": "WCDE v3 lower-secondary completion, both sexes, age 20-24",
        "outcomes": "WDI LE at birth, WDI U5MR (per 1000)",
        "fixed_effects": "country",
        "ussr_excluded": True,
        "n_ussr_excluded_from_pool": len(USSR_REPUBLICS),
    },
    "model_A_total": A,
    "model_B_u5_channel": B,
    "model_C_mediation": C,
    "magnitudes": {
        "beta_total_edu_on_le_yr_per_pp": float(beta_A),
        "beta_direct_edu_on_le_yr_per_pp": float(beta_C),
        "share_through_u5mr": float(share_through_u5),
        "share_non_u5mr": float(1 - share_through_u5),
    },
    "verdict": verdict,
}

path = os.path.join(CHECKIN, "le_independent_of_u5mr.json")
with open(path, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nCheckin: {path}")
