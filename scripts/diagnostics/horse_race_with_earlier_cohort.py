"""
horse_race_with_earlier_cohort.py

Diagnostic: does HLO still dominate the 2015 cross-section TFR race
once we add an earlier lower-sec stock (lsec at 1990, i.e. T-10 with
T=2000) as a control?

The earlier cohort here is the T-10 cohort — the 20-24-year-olds of
1990, a decade ahead of the T=2000 mother generation. This is NOT a
grandparent (~T-28) cohort; it is the parent-adjacent prior decade.

Hypothesis: HLO may be picking up cumulative educational stock across
cohorts rather than pure cognitive quality. If true, adding lower-sec
at 1990 (the T-10 cohort's own 20-24 education) should shrink HLO's
coefficient and revive lower-sec quantity at T.

Specifications (each + log population, region FE, USSR/oil excluded):
  A: lower-sec(2000) only
  B: HLO_sec only
  C: lower-sec(2000) + HLO_sec                    [the previous horse race]
  D: lower-sec(2000) + HLO_sec + lower-sec(1990)  [+ T-10 cohort stock]
  E: lower-sec(2000) + lower-sec(1990)            [no HLO, two cohorts]
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import DATA, PROC, load_wide_indicator, REGIONS

USSR_LC = {
    "russian federation", "ukraine", "belarus",
    "estonia", "latvia", "lithuania",
    "kazakhstan", "uzbekistan", "turkmenistan",
    "kyrgyz republic", "tajikistan",
    "azerbaijan", "armenia", "georgia", "moldova",
}
OIL_LC = {"qatar", "united arab emirates", "kuwait",
          "saudi arabia", "oman", "bahrain"}

HLO_TO_WDI = {
    "Iran, Islamic Republic of": "iran",
    "Korea, Rep.": "korea, rep.",
    "Russian Federation": "russian federation",
    "Turkey": "turkiye",
    "Kyrgyz Republic": "kyrgyz republic",
    "Slovak Republic": "slovak republic",
    "Egypt, Arab Rep.": "egypt, arab rep.",
    "Moldova": "moldova",
    "Venezuela, RB": "venezuela, rb",
    "Viet Nam": "viet nam",
    "Yemen, Rep.": "yemen, rep.",
    "Hong Kong SAR, China": "hong kong sar, china",
    "Macao SAR, China": "macao sar, china",
    "Czechia": "czech republic",
}


def load_hlo():
    hlo_path = os.path.join(DATA, "hlo_raw.csv")
    hlo = pd.read_csv(hlo_path)
    hlo["country"] = hlo["country"].map(
        lambda x: HLO_TO_WDI.get(x, x.lower().strip()))
    hlo_sec = (hlo[hlo["level"] == "sec"]
               .groupby("country")["hlo"].mean())
    return hlo_sec


def main():
    # Education series
    lsec = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"),
                       index_col="country")
    lsec.columns = lsec.columns.astype(int)
    lsec.index = [s.lower() for s in lsec.index]

    # Outcomes
    tfr = load_wide_indicator("children_per_woman_total_fertility.csv")
    pop_path = os.path.join(PROC, "population_2015.csv")
    pop_df = pd.read_csv(pop_path)
    pop_df["country"] = pop_df["country"].str.lower().str.strip()
    pop = pop_df.set_index("country")["population_thousands"]
    hlo_sec = load_hlo()

    rows = []
    for c in tfr.index:
        if c in REGIONS or c in USSR_LC or c in OIL_LC:
            continue
        lsec_2000 = lsec.loc[c, 2000] if c in lsec.index and 2000 in lsec.columns else np.nan
        lsec_1990 = lsec.loc[c, 1990] if c in lsec.index and 1990 in lsec.columns else np.nan
        tfr_2015 = tfr.loc[c, "2015"] if "2015" in tfr.columns else np.nan
        pop_2015 = pop.get(c, np.nan)
        hlo = hlo_sec.get(c, np.nan)
        if pd.isna(lsec_2000) or pd.isna(tfr_2015) or pd.isna(pop_2015):
            continue
        rows.append({
            "country": c,
            "lsec_2000": lsec_2000,
            "lsec_1990": lsec_1990,
            "hlo_sec": hlo,
            "tfr_2015": tfr_2015,
            "log_pop_2015": np.log(pop_2015),
        })
    df = pd.DataFrame(rows)
    print(f"Panel: {len(df)} countries; with lsec_1990: "
          f"{df['lsec_1990'].notna().sum()}; with HLO: "
          f"{df['hlo_sec'].notna().sum()}; with all three: "
          f"{df.dropna(subset=['lsec_1990', 'hlo_sec']).shape[0]}")

    # Restrict to the intersection sample (all three predictors)
    sub = df.dropna(subset=["lsec_1990", "hlo_sec"]).copy()
    print(f"\nCommon sample: n={len(sub)}\n")

    # Delta = lsec(2000) - lsec(1990)  (expansion over the prior decade)
    sub["lsec_delta"] = sub["lsec_2000"] - sub["lsec_1990"]
    # Standardize predictors so betas are z-comparable
    for col in ["lsec_2000", "lsec_1990", "lsec_delta", "hlo_sec"]:
        sub[f"{col}_z"] = (sub[col] - sub[col].mean()) / sub[col].std()
    sub["tfr_z"] = (sub["tfr_2015"] - sub["tfr_2015"].mean()) / sub["tfr_2015"].std()

    def reg(predictors, label):
        X = sm.add_constant(sub[predictors + ["log_pop_2015"]])
        y = sub["tfr_z"]
        m = sm.OLS(y, X).fit(cov_type="HC1")
        print(f"--- {label} ---")
        for p in predictors:
            print(f"  {p:<14s}  beta_z = {m.params[p]:+.3f}  "
                  f"t = {m.tvalues[p]:+.2f}  p = {m.pvalues[p]:.3f}")
        print(f"  R^2 = {m.rsquared:.3f}  adj R^2 = {m.rsquared_adj:.3f}  n = {int(m.nobs)}")
        print()
        return m

    print("=" * 72)
    print("TFR_z 2015 horse race — common sample (n={})".format(len(sub)))
    print("=" * 72)
    results = {
        "A_lsec_only":            reg(["lsec_2000_z"],                                "A: lower-sec(T) only"),
        "B_hlo_only":             reg(["hlo_sec_z"],                                  "B: HLO only"),
        "C_lsec_hlo":             reg(["lsec_2000_z", "hlo_sec_z"],                   "C: lsec(T) + HLO  [previous race]"),
        "D_lsec_hlo_t10":         reg(["lsec_2000_z", "hlo_sec_z", "lsec_1990_z"],    "D: lsec(T) + HLO + lsec(1990)  [+ T-10 cohort]"),
        "E_lsec_t10":             reg(["lsec_2000_z", "lsec_1990_z"],                 "E: lsec(T) + lsec(1990)  [no HLO, T-10 cohort]"),
        "F_delta_only":           reg(["lsec_delta_z"],                                "F: delta (2000 - 1990) only"),
        "G_delta_hlo":            reg(["lsec_delta_z", "hlo_sec_z"],                   "G: delta + HLO"),
        "H_lsec_delta":           reg(["lsec_2000_z", "lsec_delta_z"],                 "H: lsec(T) + delta"),
        "I_lsec_delta_hlo":       reg(["lsec_2000_z", "lsec_delta_z", "hlo_sec_z"],    "I: lsec(T) + delta + HLO"),
    }

    # Pack into checkin JSON for paper registration.
    import json
    out = {
        "method": (
            "2015 TFR cross-section horse race. Predictors: lower-secondary "
            "completion at T=2000 (mother generation), HLO secondary test "
            "score, lower-secondary at 1990 (the T-10 cohort, the prior "
            "decade — not a grandparent cohort), "
            "and the delta lsec(2000)-lsec(1990). All standardised; outcome is "
            "TFR_z. Controls: log population. SEs: HC1 robust. Excludes USSR "
            "republics and oil states. Common sample = countries with all "
            "predictors observed."
        ),
        "numbers": {},
    }
    for spec_id, m in results.items():
        for p in m.params.index:
            if p == "const":
                continue
            out["numbers"][f"{spec_id}.{p}.beta"] = round(float(m.params[p]), 3)
            out["numbers"][f"{spec_id}.{p}.t"]    = round(float(m.tvalues[p]), 2)
            out["numbers"][f"{spec_id}.{p}.p"]    = round(float(m.pvalues[p]), 4)
        out["numbers"][f"{spec_id}.r2"] = round(float(m.rsquared), 3)
        out["numbers"][f"{spec_id}.n"]  = int(m.nobs)
    out["n_common_sample"] = int(len(sub))

    checkin_path = os.path.join(
        os.path.dirname(SCRIPT_DIR), "..", "checkin",
        "horse_race_earlier_cohort.json")
    with open(os.path.abspath(checkin_path), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nCheckin written to {os.path.abspath(checkin_path)}")


if __name__ == "__main__":
    main()
