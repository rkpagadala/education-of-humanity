"""
robustness/iv_2sls_colonial_icrg.py
===================================
Same 2SLS contest as iv_2sls_colonial.py, but using AJR's own
institutional measure: average protection against expropriation risk
1985-1995 (avexpr) from Political Risk Services / ICRG.

The companion script (iv_2sls_colonial.py) used Polity5's polity2
(democracy-autocracy axis) as the institutional proxy. Polity2 is not
the variable AJR (2001) used. AJR used avexpr — a property-rights /
contract-enforcement measure on a 0-10 scale. The two constructs differ:
Singapore scores high on avexpr but low on polity2; many post-Soviet
democracies score the reverse.

This script closes the measurement gap by re-running the contest on
AJR's actual variable. The exclusion-restriction failure is structural
and survives any measurement choice; this script is a robustness check
on the secondary first-stage finding only.

Two competing 2SLS regressions using the same instrument:
  IV-Education:    protestant -> education -> development
  IV-Institution:  protestant -> avexpr     -> development

Source for avexpr: Acemoglu, Johnson & Robinson (2001) replication
archive, maketable1.dta, baseco==1 sample (64 former colonies).
"""

import os, sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _shared import PROC, DATA, REGIONS, load_wb, NAME_MAP, write_checkin

# ── Load colony classifications from existing script ─────────────────
COLONIES_CODE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "colonial_vs_institutions.py")).read()
start = COLONIES_CODE.index("COLONIES = {")
depth = 0
for i, ch in enumerate(COLONIES_CODE[start:]):
    if ch == '{':
        depth += 1
    elif ch == '}':
        depth -= 1
        if depth == 0:
            end = start + i + 1
            break
exec(COLONIES_CODE[start:end])

# AJR's base sample includes a few colonies/territories not in COLONIES;
# extend locally so the full 64-row AJR sample is usable.
COLONIES_AJR_EXT = {
    "Bahamas":   ("Britain", "protestant"),
    "Gabon":     ("France",  "catholic"),
    "Hong Kong, China":   ("Britain", "protestant"),
}
COLONIES.update(COLONIES_AJR_EXT)

# ── Map AJR shortnam (ISO3) -> WCDE country name ─────────────────────
ISO3_TO_WCDE = {
    "AGO": "Angola", "ARG": "Argentina", "AUS": "Australia",
    "BFA": "Burkina Faso", "BGD": "Bangladesh", "BHS": "Bahamas",
    "BOL": "Bolivia (Plurinational State of)", "BRA": "Brazil",
    "CAN": "Canada", "CHL": "Chile", "CIV": "Côte d'Ivoire",
    "CMR": "Cameroon", "COG": "Congo", "COL": "Colombia",
    "CRI": "Costa Rica", "DOM": "Dominican Republic", "DZA": "Algeria",
    "ECU": "Ecuador", "EGY": "Egypt", "ETH": "Ethiopia",
    "GAB": "Gabon", "GHA": "Ghana", "GIN": "Guinea", "GMB": "Gambia",
    "GTM": "Guatemala", "GUY": "Guyana", "HKG": "Hong Kong, China",
    "HND": "Honduras", "HTI": "Haiti", "IDN": "Indonesia",
    "IND": "India", "JAM": "Jamaica", "KEN": "Kenya", "LKA": "Sri Lanka",
    "MAR": "Morocco", "MDG": "Madagascar", "MEX": "Mexico", "MLI": "Mali",
    "MLT": "Malta", "MYS": "Malaysia", "NER": "Niger", "NGA": "Nigeria",
    "NIC": "Nicaragua", "NZL": "New Zealand", "PAK": "Pakistan",
    "PAN": "Panama", "PER": "Peru", "PRY": "Paraguay", "SDN": "Sudan",
    "SEN": "Senegal", "SGP": "Singapore", "SLE": "Sierra Leone",
    "SLV": "El Salvador", "TGO": "Togo", "TTO": "Trinidad and Tobago",
    "TUN": "Tunisia", "TZA": "United Republic of Tanzania",
    "UGA": "Uganda", "URY": "Uruguay", "USA": "United States of America",
    "VEN": "Venezuela (Bolivarian Republic of)", "VNM": "Viet Nam",
    "ZAF": "South Africa", "ZAR": "Democratic Republic of the Congo",
}


# ── Load data ────────────────────────────────────────────────────────

ajr = pd.read_csv(os.path.join(DATA, "ajr2001", "ajr2001_baseco.csv"))
ajr["country"] = ajr["shortnam"].map(ISO3_TO_WCDE)
missing_iso = ajr[ajr["country"].isna()]["shortnam"].tolist()
if missing_iso:
    raise SystemExit(f"Unmapped ISO3 codes: {missing_iso}")
avexpr = ajr.set_index("country")["avexpr"]

edu_wide = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"))
edu_wide = edu_wide[~edu_wide["country"].isin(REGIONS)].copy()
edu_1950 = edu_wide[["country", "1950"]].copy()
edu_1950.columns = ["country", "edu_1950"]
edu_1950["edu_1950"] = pd.to_numeric(edu_1950["edu_1950"], errors="coerce")
edu_1950 = edu_1950.set_index("country")

gdp = load_wb("gdppercapita_us_inflation_adjusted.csv")
le = load_wb("life_expectancy_years.csv")
tfr = load_wb("children_per_woman_total_fertility.csv")


def get_wb(df, wcde_name, year="2020"):
    key = NAME_MAP.get(wcde_name, wcde_name).lower()
    for k in [wcde_name.lower(), key]:
        if k in df.index:
            try:
                v = float(df.loc[k, year])
                return v if not np.isnan(v) else np.nan
            except (KeyError, ValueError):
                pass
    return np.nan


# ── Build analysis dataset (AJR base sample) ─────────────────────────

rows = []
for country in avexpr.index:
    if country not in COLONIES:
        # AJR includes a few never-colonized observations (e.g. ETH);
        # skip them to keep the colonizer-religion instrument well-defined.
        continue
    colonizer, religion = COLONIES[country]
    if colonizer is None:
        continue
    edu50 = float(edu_1950.loc[country, "edu_1950"]) if country in edu_1950.index else np.nan
    gdp20 = get_wb(gdp, country, "2020")
    le20 = get_wb(le, country, "2020")
    tfr20 = get_wb(tfr, country, "2020")

    rows.append({
        "country": country,
        "colonizer": colonizer,
        "religion": religion,
        "protestant": 1 if religion == "protestant" else 0,
        "edu_1950": edu50,
        "avexpr": float(avexpr.loc[country]),
        "log_gdp_2020": np.log(gdp20) if gdp20 and not np.isnan(gdp20) and gdp20 > 0 else np.nan,
        "le_2020": le20,
        "tfr_2020": tfr20,
    })

df = pd.DataFrame(rows)

print("=" * 78)
print("2SLS INSTRUMENTAL VARIABLE TEST (ICRG): EDUCATION vs INSTITUTIONS")
print("Instrument: colonizer religion (protestant = 1)")
print("Institutional measure: AJR (2001) avexpr 1985-1995")
print("=" * 78)
print(f"\nAJR base-sample colonies in analysis: {len(df)}")
print(f"  Protestant colonizer: {(df.protestant == 1).sum()}")
print(f"  Catholic colonizer:   {(df.protestant == 0).sum()}")


# ── Manual 2SLS implementation (mirrors iv_2sls_colonial.py) ─────────

def run_2sls(df, endog_col, outcome_col, instrument_col="protestant", label=""):
    sub = df.dropna(subset=[endog_col, outcome_col, instrument_col]).copy()
    n = len(sub)

    Z = sm.add_constant(sub[[instrument_col]].values)
    X_endog = sub[endog_col].values
    Y = sub[outcome_col].values

    first_stage = sm.OLS(X_endog, Z).fit()
    f_stat = first_stage.fvalue
    f_pval = first_stage.f_pvalue
    fs_coef = first_stage.params[1]
    fs_se = first_stage.bse[1]
    fs_t = first_stage.tvalues[1]

    endog_hat = first_stage.fittedvalues
    X2 = sm.add_constant(endog_hat)
    second_stage_naive = sm.OLS(Y, X2).fit()
    ss_coef = second_stage_naive.params[1]

    resid_2s = Y - (second_stage_naive.params[0] + ss_coef * X_endog)
    sigma2 = np.sum(resid_2s**2) / (n - 2)
    var_ols = sigma2 * np.linalg.inv(X2.T @ X2)
    ss_se_corrected = np.sqrt(var_ols[1, 1])
    ss_t_corrected = ss_coef / ss_se_corrected

    X_actual = sm.add_constant(X_endog)
    ols = sm.OLS(Y, X_actual).fit()
    ols_coef = ols.params[1]

    resid_fs = first_stage.resid
    X_hausman = np.column_stack([X_actual, resid_fs])
    hausman_reg = sm.OLS(Y, X_hausman).fit()
    hausman_t = hausman_reg.tvalues[2]
    hausman_p = hausman_reg.pvalues[2]

    reduced = sm.OLS(Y, Z).fit()
    rf_coef = reduced.params[1]
    rf_t = reduced.tvalues[1]
    rf_p = reduced.pvalues[1]

    wald = rf_coef / fs_coef

    return {
        "label": label, "n": n,
        "first_stage_F": f_stat, "first_stage_F_pval": f_pval,
        "first_stage_coef": fs_coef, "first_stage_se": fs_se, "first_stage_t": fs_t,
        "second_stage_coef": ss_coef, "second_stage_se": ss_se_corrected,
        "second_stage_t": ss_t_corrected,
        "ols_coef": ols_coef, "wald_estimate": wald,
        "reduced_form_coef": rf_coef, "reduced_form_t": rf_t,
        "reduced_form_p": rf_p,
        "hausman_t": hausman_t, "hausman_p": hausman_p,
    }


def print_2sls(r):
    print(f"\n  {r['label']}  (n = {r['n']})")
    print(f"  {'─' * 60}")
    strong = "STRONG" if r["first_stage_F"] > 10 else "WEAK"
    print(f"  First stage F-stat:   {r['first_stage_F']:>8.2f}  ({strong} instrument)")
    print(f"  First stage coef:     {r['first_stage_coef']:>+8.3f}  (se={r['first_stage_se']:.3f}, t={r['first_stage_t']:.2f})")
    print(f"  Second stage coef:    {r['second_stage_coef']:>+8.4f}  (se={r['second_stage_se']:.4f}, t={r['second_stage_t']:.2f})")
    print(f"  OLS coef:             {r['ols_coef']:>+8.4f}")
    print(f"  Wald (IV) estimate:   {r['wald_estimate']:>+8.4f}")
    print(f"  Reduced form:         {r['reduced_form_coef']:>+8.3f}  (t={r['reduced_form_t']:.2f}, p={r['reduced_form_p']:.4f})")
    print(f"  Wu-Hausman:           t={r['hausman_t']:+.2f}, p={r['hausman_p']:.4f}")


# ── Run the contest ──────────────────────────────────────────────────

results = {}

for outcome, olabel in [("log_gdp_2020", "log GDP 2020"),
                         ("le_2020", "Life expectancy 2020"),
                         ("tfr_2020", "TFR 2020")]:
    print(f"\n{'═' * 78}")
    print(f"OUTCOME: {olabel}")
    print(f"{'═' * 78}")

    r_edu = run_2sls(df, "edu_1950", outcome, label=f"IV-Education: protestant -> edu_1950 -> {olabel}")
    print_2sls(r_edu)

    r_inst = run_2sls(df, "avexpr", outcome, label=f"IV-Institution: protestant -> avexpr -> {olabel}")
    print_2sls(r_inst)

    results[outcome] = {"education": r_edu, "institution": r_inst}

    print(f"\n  VERDICT:")
    if r_edu["first_stage_F"] > 10 and r_inst["first_stage_F"] < 10:
        print(f"  -> Protestant colonizer is a STRONG instrument for education (F={r_edu['first_stage_F']:.1f})")
        print(f"     but a WEAK instrument for AJR avexpr (F={r_inst['first_stage_F']:.1f})")
        print(f"  -> The channel runs through education, not institutions.")
    elif r_edu["first_stage_F"] > r_inst["first_stage_F"]:
        print(f"  -> Stronger first stage for education (F={r_edu['first_stage_F']:.1f} vs {r_inst['first_stage_F']:.1f})")
    else:
        print(f"  -> First stages: education F={r_edu['first_stage_F']:.1f}, avexpr F={r_inst['first_stage_F']:.1f}")
        print(f"     Even with strong institutions first stage, exclusion restriction is broken")
        print(f"     by the documented education channel: 2SLS coefs cannot identify either.")


# ── Summary ──────────────────────────────────────────────────────────

print(f"\n{'═' * 78}")
print("SUMMARY: FIRST-STAGE F-STATISTICS (AJR avexpr instead of polity2)")
print(f"{'═' * 78}")
print(f"\n  {'Outcome':<25} {'Edu F-stat':>12} {'Avexpr F':>12} {'Winner':>10}")
print(f"  {'─' * 60}")
for outcome, olabel in [("log_gdp_2020", "log GDP"),
                         ("le_2020", "Life expectancy"),
                         ("tfr_2020", "TFR")]:
    r = results[outcome]
    fe = r["education"]["first_stage_F"]
    fi = r["institution"]["first_stage_F"]
    winner = "Education" if fe > fi else "Institution"
    print(f"  {olabel:<25} {fe:>12.1f} {fi:>12.1f} {winner:>10}")

print(f"\n  Instrument relevance threshold: F > 10 (Stock & Yogo 2005)")

fe_gdp = results["log_gdp_2020"]["education"]["first_stage_F"]
fi_gdp = results["log_gdp_2020"]["institution"]["first_stage_F"]

print(f"\n{'═' * 78}")
print("INTERPRETATION")
print(f"{'═' * 78}")
print(f"""
This is the AJR-on-AJR contest: AJR's own institutional measure
(avexpr 1985-1995) tested against education (lower-secondary completion
1950) using the colonizer-religion instrument that absorbs the variance
their settler-mortality instrument was designed to capture.

  protestant -> education:  F = {fe_gdp:.1f}  ({'strong' if fe_gdp > 10 else 'weak'} instrument)
  protestant -> avexpr:     F = {fi_gdp:.1f}  ({'strong' if fi_gdp > 10 else 'weak'} instrument)

The exclusion restriction failure documented by the companion contest
holds regardless of which institutional proxy is used: protestant
colonizers brought BOTH schools AND institutions, so the instrument
cannot identify either channel cleanly. This script tests whether the
weak-instrument finding for institutions survives the switch from
polity2 to AJR's own avexpr. {'It does.' if fi_gdp < 10 else 'It does not — but the structural identification problem remains.'}
""")

# ── Save checkin ─────────────────────────────────────────────────────

def _p_from_t(t, n):
    return 2 * (1 - stats.t.cdf(abs(t), df=max(n - 2, 1)))

checkin = {
    "n_ajr_base": int(len(df)),
    "n_ajr_complete": int(results["log_gdp_2020"]["education"]["n"]),
}
for outcome, prefix in [("log_gdp_2020", "gdp"),
                         ("le_2020", "le"),
                         ("tfr_2020", "tfr")]:
    for arm, akey in [("education", "edu"), ("institution", "inst")]:
        r = results[outcome][arm]
        checkin[f"{prefix}_{akey}_first_stage_F"] = round(r["first_stage_F"], 2)
        checkin[f"{prefix}_{akey}_2sls_coef"] = round(r["second_stage_coef"], 4)
        checkin[f"{prefix}_{akey}_2sls_t"] = round(r["second_stage_t"], 2)
        checkin[f"{prefix}_{akey}_2sls_p"] = round(float(_p_from_t(r["second_stage_t"], r["n"])), 4)
        checkin[f"{prefix}_{akey}_reduced_coef"] = round(r["reduced_form_coef"], 4)
        checkin[f"{prefix}_{akey}_reduced_p"] = round(r["reduced_form_p"], 4)
checkin["gdp_edu_wald"] = round(results["log_gdp_2020"]["education"]["wald_estimate"], 4)
write_checkin("iv_2sls_colonial_icrg.json", checkin,
              script_path="scripts/robustness/iv_2sls_colonial_icrg.py")
