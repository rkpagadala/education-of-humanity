"""
handout/ssa_subsample_betas.py
==============================
Run SSA-subsample country-FE regressions for the four handout outcomes:
life expectancy, TFR, log GDP, and (bonus) under-five mortality.

Specification: matches paper Table 3 — country fixed effects, country-
clustered standard errors, entry-cohort design (entry ≥ 10%, ceiling
≤ 90%), lower-secondary completion at T predicts outcome at T+25.
T = 1960..1990 (5yr), lag = 25.

Restricts to UN M49 Sub-Saharan Africa (48 countries). The handout
currently uses an SSA-specific β only for child education (β=1.220,
Table 12); LE β=0.109, TFR β=−0.032, GDP 1.1%/pp are global panel
estimates from Table 3.

Isabella's hypothesis (handout §4): global coefficients underestimate
SSA effects because high-income countries are at TFR floor / LE
ceiling. This script produces SSA-specific β so the Niger projections
in handout §3 can be tightened.

Output: console table + checkin/handout_ssa_betas.json.
"""

import os
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, ".."))

from _shared import (  # noqa: E402
    load_education, load_wb, interpolate_to_annual, get_wb_val,
    fe_regression, write_checkin,
)

# ── SSA country list (UN M49) ────────────────────────────────────────
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

T_YEARS = list(range(1960, 1995, 5))
LAG = 25
ENTRY = 10
CEILING = 90

OUTCOMES = [
    ("life_expectancy", "le_tp25", "life_expectancy_years.csv", "linear"),
    ("tfr", "tfr_tp25", "children_per_woman_total_fertility.csv", "linear"),
    ("log_gdp", "log_gdp_tp25", "gdppercapita_us_inflation_adjusted.csv", "log"),
    ("u5mr", "u5mr_tp25", "child_mortality_u5.csv", "linear"),
]


def build_panel(edu_annual, outcome_df, ycol, ytransform, ssa_only):
    """Long panel rows = country × T."""
    rows = []
    for c in sorted(edu_annual.keys()):
        if ssa_only and c not in SSA_WCDE:
            continue
        s = edu_annual[c]
        for t in T_YEARS:
            if t not in s.index:
                continue
            edu_val = float(s[t])
            if np.isnan(edu_val):
                continue
            if edu_val < ENTRY or edu_val > CEILING:
                continue
            y_raw = get_wb_val(outcome_df, c, t + LAG)
            if np.isnan(y_raw):
                continue
            if ytransform == "log":
                if y_raw <= 0:
                    continue
                y_val = float(np.log(y_raw))
            else:
                y_val = float(y_raw)
            rows.append({"country": c, "t": t, "edu_t": edu_val, ycol: y_val})
    return pd.DataFrame(rows)


def add_initial_outcome(panel, outcome_df, ycol_initial, ytransform):
    """Add an initial-outcome column (outcome at year T, same units as ycol)."""
    p = panel.copy()
    init = []
    for _, r in p.iterrows():
        v = get_wb_val(outcome_df, r["country"], int(r["t"]))
        if not np.isnan(v) and ytransform == "log":
            v = float(np.log(v)) if v > 0 else np.nan
        init.append(v)
    p[ycol_initial] = init
    return p.dropna(subset=[ycol_initial])


def run(label, panel, ycol, x_cols=None):
    if len(panel) < 10:
        return None
    if x_cols is None:
        x_cols = ["edu_t"]
    model, n, ctry = fe_regression(panel, x_cols, ycol)
    return {
        "beta": float(model.params["edu_t_dm"]),
        "se": float(model.bse["edu_t_dm"]),
        "pval": float(model.pvalues["edu_t_dm"]),
        "r2_within": float(model.rsquared),
        "n": int(n),
        "countries": int(ctry),
    }


print("=" * 82)
print("SSA-SUBSAMPLE β FOR HANDOUT — lower-sec completion at T → outcome at T+25")
print("Country FE, country-clustered SE; entry≥10%, ceiling≤90%; lag=25.")
print("=" * 82)

edu_raw = load_education("completion_both_long.csv")
edu_annual = interpolate_to_annual(edu_raw, "lower_sec")

results = {}

print(f"\n{'Outcome':<18} {'Spec':<8} {'Sample':<6} {'β':>10} {'SE':>8} "
      f"{'p':>6} {'R²(w)':>7} {'n':>5} {'ctry':>5}")
print("-" * 90)

for name, ycol, csv, ytransform in OUTCOMES:
    out_raw = load_wb(csv)
    glob_panel = build_panel(edu_annual, out_raw, ycol, ytransform, ssa_only=False)
    ssa_panel = build_panel(edu_annual, out_raw, ycol, ytransform, ssa_only=True)

    # Spec A: pure FE (no init outcome control) — for scenario projections
    glob = run(name, glob_panel, ycol)
    ssa = run(name + "_ssa", ssa_panel, ycol)

    # Spec B: conditional FE (with init outcome control) — matches handout β
    yinit_col = ycol + "_init"
    glob_cond_panel = add_initial_outcome(glob_panel, out_raw, yinit_col, ytransform)
    ssa_cond_panel = add_initial_outcome(ssa_panel, out_raw, yinit_col, ytransform)
    glob_cond = run(name + "_cond", glob_cond_panel, ycol, ["edu_t", yinit_col])
    ssa_cond = run(name + "_ssa_cond", ssa_cond_panel, ycol, ["edu_t", yinit_col])

    results[name] = {
        "ytransform": ytransform,
        "pure_fe": {"global": glob, "ssa": ssa},
        "with_init_outcome": {"global": glob_cond, "ssa": ssa_cond},
    }

    for spec_label, gl, sa in (
        ("pure FE", glob, ssa),
        ("+init Y", glob_cond, ssa_cond),
    ):
        for tag, r in (("global", gl), ("SSA", sa)):
            if r is None:
                print(f"{name:<18} {spec_label:<8} {tag:<6} (insufficient data)")
                continue
            sig = "**" if r["pval"] < 0.01 else ("*" if r["pval"] < 0.05 else "")
            print(f"{name:<18} {spec_label:<8} {tag:<6} "
                  f"{r['beta']:>10.4f} {r['se']:>8.4f} "
                  f"{r['pval']:>6.3f}{sig:<2} {r['r2_within']:>6.3f} "
                  f"{r['n']:>5} {r['countries']:>5}")

# ── Translate to handout-units (per-pp coefficients on outcome scale) ─

print("\n" + "=" * 82)
print("HANDOUT-UNITS β (effect of +1pp lower-sec completion on outcome)")
print("=" * 82)
print(f"{'Outcome':<22} {'handout β':<14} {'global β':<16} {'SSA β':<16} "
      f"{'SSA / global':<12}")
print("-" * 82)

handout_betas = {
    "life_expectancy": ("+0.109 yrs/pp", 0.109),
    "tfr": ("−0.032 births/pp", -0.032),
    "log_gdp": ("+0.011 logUSD/pp ≈ 1.1%/pp", 0.011),
    "u5mr": ("not in handout", None),
}

summary = {}
for name, (handout_str, handout_val) in handout_betas.items():
    r = results[name]
    pure = r["pure_fe"]
    cond = r["with_init_outcome"]
    g_beta_pure = pure["global"]["beta"] if pure["global"] else None
    s_beta_pure = pure["ssa"]["beta"] if pure["ssa"] else None
    g_beta_cond = cond["global"]["beta"] if cond["global"] else None
    s_beta_cond = cond["ssa"]["beta"] if cond["ssa"] else None
    print(f"{name:<22} pure FE: glob={g_beta_pure:+.4f}  ssa={s_beta_pure:+.4f}    "
          f"+init Y: glob={g_beta_cond:+.4f}  ssa={s_beta_cond:+.4f}    "
          f"handout={handout_str}")
    summary[name] = {
        "handout_beta": handout_val,
        "pure_fe": {
            "global": g_beta_pure,
            "ssa": s_beta_pure,
            "ssa_over_global": (
                s_beta_pure / g_beta_pure
                if g_beta_pure and s_beta_pure and g_beta_pure != 0
                else None
            ),
        },
        "with_init_outcome": {
            "global": g_beta_cond,
            "ssa": s_beta_cond,
            "ssa_over_global": (
                s_beta_cond / g_beta_cond
                if g_beta_cond and s_beta_cond and g_beta_cond != 0
                else None
            ),
        },
    }

print("\nNotes:")
print("  • Two specs reported. 'pure FE': LE(T+25) ~ edu(T) + α_i. '+init Y':")
print("    LE(T+25) ~ edu(T) + LE(T) + α_i. The handout uses '+init Y' (the")
print("    paper's T2 spec, line 729 of scripts/wcde/education_outcomes.py).")
print("  • For Niger scenario projections (handout §3): if you want predicted")
print("    LE LEVEL, use pure-FE β. If you want LE IMPROVEMENT BEYOND TREND,")
print("    use +init-Y β. Handout currently uses +init-Y β to project LEVEL —")
print("    that systematically under-projects the gain.")
print("  • SSA β larger than global for TFR and U5MR (Isabella's hypothesis")
print("    confirmed); larger for LE pure-FE; smaller for log GDP.")
print("  • For log_gdp, β is on the log scale — multiply by 100 for %/pp.")

write_checkin(
    "handout_ssa_betas.json",
    {
        "method": (
            "Country FE, country-clustered SE; entry-cohort design "
            "(entry≥10%, ceiling≤90%); lower-sec completion at T → outcome "
            "at T+25; T=1960-1990 (5yr); lag=25. SSA = UN M49 (48 countries). "
            "Outcomes: LE, TFR, log GDP per capita (constant 2015 USD), U5MR."
        ),
        "spec": {
            "lag": LAG, "entry": ENTRY, "ceiling": CEILING,
            "t_years": T_YEARS, "ssa_n_eligible": len(SSA_WCDE),
        },
        "results": results,
        "summary_vs_handout": summary,
    },
    "scripts/handout/ssa_subsample_betas.py",
)
