"""
mothers_own_longevity.py
========================
Life-history label (§the-generational-lag):
  Predictor: 20-24 cohort female completion at T. Outcome: female
  adult mortality (AMRT 15-60) at T+28. Tests the adult-mortality
  channel (her OWN longevity) separately from the child-survival
  channel. Generational relation: same-cohort effect (the educated
  woman's own life), realized over T to T+40 (her adult years).
  Use: identification of LE channels beyond U5MR mediation.

Does an educated woman live longer herself — independent of her
children's survival?

QUESTION
--------
The prior test (le_independent_of_u5mr.py) showed education's effect
on LE at birth collapses to zero once U5MR is controlled. But LE at
birth is mechanically dominated by infant/child deaths. Our World in
Data documents LE rising at every age (5, 15, 30, 60), so the
adult-side gain exists at the global level. The question for the
panel: does it exist within countries, attributable to education?

The cleanest test is adult mortality directly. Adult mortality rate
(₄₅q₁₅): the probability that a 15-year-old will die before age 60,
per 1000, computed period-style from age-specific rates of the
reference year. Sex-specific. The question becomes: does female
education(T) predict female adult mortality(T+28)?

METHOD
------
Country fixed effects, three specifications. Outcome variable is
log(AMRT_FE) — the natural log because mortality follows a roughly
log-linear gradient against income/education at the cross-section.

  Model A:  log AMRT_FE(T+28) = β·fem_edu(T) + α_c + ε
  Model B:  log AMRT_FE(T+28) = β·fem_edu(T) + γ·log U5MR(T+28) + α_c + ε
  Model C:  log AMRT_MA(T+28) = β·fem_edu(T) + α_c + ε      (placebo)

Interpretation:
  - Model A: does female education predict female adult mortality?
  - Model B: does it survive once child mortality is also accounted
    for? (The mother-herself channel, net of the children-not-dying
    statistical correlation.)
  - Model C: does female education predict MALE adult mortality?
    If yes, the channel is general country development, not
    specifically the educated woman's own longevity.
    If much weaker than female effect, the female effect is
    specifically about women's own outcomes.

DATA
----
- Female education: WCDE v3 lower-secondary completion, female,
  age 20-24 cohort, at year T
- AMRT_FE / AMRT_MA: World Bank WDI SP.DYN.AMRT.FE / .MA at year
  T+28 (when the cohort is ~age 45, midlife of the 15-60 window)
- U5MR: World Bank WDI SH.DYN.MORT at T+28
- Panel: T ∈ {1975, 1980, 1985, 1990}
- USSR republics excluded (Goskomstat anomaly)

OUTPUT
------
Prints regression tables; writes checkin/mothers_own_longevity.json.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from _shared import (
    CHECKIN,
    LAG_GENERATION,
    fe_regression,
    get_wb_val,
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
LAG = LAG_GENERATION


# ── Load data ───────────────────────────────────────────────────────

print("Loading data...")
fem_edu = load_education("completion_female_long.csv")
amrt_fe = load_wb("adult_mortality_female.csv")
amrt_ma = load_wb("adult_mortality_male.csv")
u5_raw = load_wb("child_mortality_u5.csv")
print(f"  Female education: {fem_edu['country'].nunique()} countries")
print(f"  AMRT_FE: {len(amrt_fe)} countries")
print(f"  AMRT_MA: {len(amrt_ma)} countries")
print(f"  U5MR:    {len(u5_raw)} countries")


# ── Build panel ─────────────────────────────────────────────────────

print(f"\nBuilding panel (T={T_YEARS}, lag={LAG} yr, USSR excluded)...")

rows = []
for c in sorted(fem_edu["country"].unique()):
    if c in USSR_REPUBLICS:
        continue
    edu_c = fem_edu[fem_edu["country"] == c].set_index("year")
    for t in T_YEARS:
        if t not in edu_c.index:
            continue
        tp28 = t + LAG
        fem = edu_c.loc[t, "lower_sec"]
        a_fe = get_wb_val(amrt_fe, c, tp28)
        a_ma = get_wb_val(amrt_ma, c, tp28)
        u5 = get_wb_val(u5_raw, c, tp28)
        if (np.isnan(fem) or np.isnan(a_fe) or np.isnan(a_ma)
                or np.isnan(u5)):
            continue
        rows.append({
            "country": c,
            "t": t,
            "fem_edu_t": fem,
            "amrt_fe_tp28": a_fe,
            "amrt_ma_tp28": a_ma,
            "u5_tp28": u5,
            "log_amrt_fe": np.log(a_fe) if a_fe > 0 else np.nan,
            "log_amrt_ma": np.log(a_ma) if a_ma > 0 else np.nan,
            "log_u5": np.log(u5) if u5 > 0 else np.nan,
        })

panel = pd.DataFrame(rows).dropna()
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
        print(f"  {var:22s}  β={coef[var]:+8.5f}  SE={se[var]:7.5f}  "
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
print("FIXED-EFFECTS REGRESSIONS — does female education lower female adult")
print("mortality, independent of child mortality?")
print("=" * 76)

# Model A: female ed → log AMRT_FE
print("\nModel A: log AMRT_FE(T+28) = β·fem_edu(T) + α_c")
mA, nA, kA = fe_regression(panel, ["fem_edu_t"], "log_amrt_fe")
A = report("[A] female adult mortality on female education", mA, nA, kA)

# Model B: female ed → log AMRT_FE controlling for U5MR
print("\nModel B: log AMRT_FE(T+28) = β·fem_edu(T) + γ·log U5MR(T+28) + α_c")
mB, nB, kB = fe_regression(panel, ["fem_edu_t", "log_u5"], "log_amrt_fe")
B = report("[B] mother-herself channel net of U5MR", mB, nB, kB)

# Model C (placebo): female ed → log AMRT_MA
print("\nModel C (placebo): log AMRT_MA(T+28) = β·fem_edu(T) + α_c")
mC, nC, kC = fe_regression(panel, ["fem_edu_t"], "log_amrt_ma")
C = report("[C] male adult mortality on female education", mC, nC, kC)

# Convert log coefficients to percent change for readability
beta_A = A["coefs"]["fem_edu_t_dm"]["beta"]
beta_B = B["coefs"]["fem_edu_t_dm"]["beta"]
beta_C = C["coefs"]["fem_edu_t_dm"]["beta"]
p_A = A["coefs"]["fem_edu_t_dm"]["p"]
p_B = B["coefs"]["fem_edu_t_dm"]["p"]
p_C = C["coefs"]["fem_edu_t_dm"]["p"]

pct_A = (np.exp(beta_A) - 1) * 100
pct_B = (np.exp(beta_B) - 1) * 100
pct_C = (np.exp(beta_C) - 1) * 100

print("\n" + "=" * 76)
print("INTERPRETATION (1 pp rise in female lower-sec completion at T")
print(" → % change in adult mortality 28 years later, within-country)")
print("=" * 76)
print(f"  [A] Female AMRT, no controls:          {pct_A:+.3f}%/pp  "
      f"(p={p_A:.4f})")
print(f"  [B] Female AMRT, U5MR controlled:      {pct_B:+.3f}%/pp  "
      f"(p={p_B:.4f})")
print(f"  [C] Male AMRT (placebo):               {pct_C:+.3f}%/pp  "
      f"(p={p_C:.4f})")

if p_B < 0.05 and beta_B < 0:
    verdict_b = ("Female education predicts lower female adult mortality "
                 "even after controlling for child mortality. The mother "
                 "herself lives longer, independent of the children-survive "
                 "effect.")
elif p_B >= 0.05:
    verdict_b = ("Female education does not predict female adult mortality "
                 "independently of U5MR. The adult-LE channel for the "
                 "mother herself is not separately identified.")
else:
    verdict_b = "Unexpected sign or magnitude — check spec."

share_through_u5 = (beta_A - beta_B) / beta_A if beta_A != 0 else float("nan")

print(f"\n  Verdict (B): {verdict_b}")
print(f"  Share of total female-AMRT effect routed through U5MR: "
      f"{share_through_u5*100:.1f}%")
print(f"  Share net of U5MR (mother-herself channel): "
      f"{(1-share_through_u5)*100:.1f}%")

if abs(beta_C) < abs(beta_A) / 2:
    placebo_msg = "(Male AMRT placebo: female education predicts MALE adult mortality less than half as strongly — the female channel is sex-specific, not just general country development.)"
elif abs(beta_C) > abs(beta_A) * 0.8:
    placebo_msg = "(Male AMRT placebo: female education predicts male AMRT nearly as strongly — the effect is general country development, not specifically the educated woman's own longevity.)"
else:
    placebo_msg = "(Male AMRT placebo: effect on males is smaller than on females; mostly female-specific.)"

print(f"  {placebo_msg}")


# ── Write checkin ───────────────────────────────────────────────────

out = {
    "spec": {
        "t_years": T_YEARS,
        "lag_years": LAG,
        "education": "WCDE v3 lower-secondary completion, female, age 20-24",
        "outcomes_amrt": "WDI SP.DYN.AMRT.FE / .MA — adult mortality per 1000",
        "outcome_u5": "WDI SH.DYN.MORT — under-5 mortality per 1000",
        "fixed_effects": "country",
        "ussr_excluded": True,
    },
    "model_A_amrt_fe_baseline": A,
    "model_B_amrt_fe_with_u5": B,
    "model_C_amrt_ma_placebo": C,
    "magnitudes_pct_per_pp": {
        "amrt_fe_baseline": float(pct_A),
        "amrt_fe_net_of_u5": float(pct_B),
        "amrt_ma_placebo": float(pct_C),
    },
    "share_through_u5mr": float(share_through_u5),
    "share_mother_herself_channel": float(1 - share_through_u5),
    "verdict_net_of_u5": verdict_b,
}

path = os.path.join(CHECKIN, "mothers_own_longevity.json")
with open(path, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nCheckin: {path}")
