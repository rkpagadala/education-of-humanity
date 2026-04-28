"""
hanushek_horse_race_comprehensive.py

Comprehensive horse race between the paper's quantity measures and
Hanushek-style quality measures, with four refinements:

  1. Outcome-specific educational depth:
       - TFR       ← primary completion  (primary literacy → mother
                                          fertility via grandparent-
                                          level norm transmission)
       - U5MR      ← lower-secondary completion (maternal cognitive
                                          reorganization drives care-
                                          seeking, dosing, sanitation)
       - LE        ← mean years of schooling across living adults
                                          (integrated; never stops)

  2. Quality adjusted for selection (HLO sampled only students
     present at test age; multiply by completion rate to get a
     population-scale 'real education' measure):
       real_edu_primary  = primary_completion  × HLO_primary  / 500
       real_edu_lsec     = lsec_completion     × HLO_secondary/ 500
       real_edu_lifetime = BL_yr_sch           × HLO_secondary/ 500

  3. Cohort alignment: use education measured at 2000 (when today's
     25-34 adults were 10-20 years old) for TFR/U5MR outcomes in 2015.
     LE uses 2010 B-L since LE integrates across older cohorts.

  4. Controls: log population, region fixed effects.

USSR republics excluded (fail the phenotype-consistency test).
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import DATA, PROC, REPO_ROOT, load_wide_indicator, REGIONS, write_checkin

USSR_LC = {
    "russian federation", "ukraine", "belarus",
    "estonia", "latvia", "lithuania",
    "kazakhstan", "uzbekistan", "turkmenistan",
    "kyrgyz republic", "tajikistan",
    "azerbaijan", "armenia", "georgia", "moldova",
}

NAME_TO_WDI = {
    "Iran, Islamic Republic of": "iran",
    "Iran (Islamic Republic of)": "iran",
    "Korea, Rep.": "korea, rep.",
    "Korea, Republic of": "korea, rep.",
    "Russian Federation": "russian federation",
    "Turkey": "turkiye",
    "Kyrgyz Republic": "kyrgyz republic",
    "Kyrgyzstan": "kyrgyz republic",
    "Slovak Republic": "slovak republic",
    "Slovakia": "slovak republic",
    "Egypt, Arab Rep.": "egypt, arab rep.",
    "Egypt": "egypt, arab rep.",
    "Moldova": "moldova",
    "Republic of Moldova": "moldova",
    "Venezuela, RB": "venezuela, rb",
    "Viet Nam": "viet nam",
    "Vietnam": "viet nam",
    "Yemen, Rep.": "yemen, rep.",
    "Yemen": "yemen, rep.",
    "Hong Kong SAR, China": "hong kong sar, china",
    "Hong Kong Special Administrative Region of China":
        "hong kong sar, china",
    "Macao SAR, China": "macao sar, china",
    "Czechia": "czech republic",
    "Czech Republic": "czech republic",
    "North Macedonia": "north macedonia",
    "The former Yugoslav Republic of Macedonia": "north macedonia",
    "Congo, Rep.": "congo, rep.",
    "Congo, Dem. Rep.": "congo, dem. rep.",
    "Lao PDR": "lao pdr",
    "Lao People's Democratic Republic": "lao pdr",
    "United States of America": "united states",
    "United States": "united states",
    "United Kingdom of Great Britain and Northern Ireland":
        "united kingdom",
    "United Kingdom": "united kingdom",
    "Bahamas, The": "bahamas, the",
    "Gambia, The": "gambia, the",
    "Cabo Verde": "cabo verde",
    "Swaziland": "eswatini",
    "Eswatini": "eswatini",
    "Republic of Korea": "korea, rep.",
}

WCDE_TO_WDI = {
    "republic of korea": "korea, rep.",
    "iran (islamic republic of)": "iran",
    "viet nam": "viet nam",
    "united kingdom of great britain and northern ireland":
        "united kingdom",
    "united states of america": "united states",
    "turkey": "turkiye",
    "republic of moldova": "moldova",
}


def load_hlo_by_level():
    """Return two dicts: primary HLO country-mean, secondary HLO
    country-mean. Averaged across years 2000-2017 and across
    math/reading/science subjects (when present)."""
    hlo = pd.read_csv(os.path.join(DATA, "hlo_raw.csv"))
    hlo = hlo[hlo['subject'].isin(['math', 'reading', 'science'])]
    pri = hlo[hlo['level'] == 'pri'].groupby('country')['hlo'].mean()
    sec = hlo[hlo['level'] == 'sec'].groupby('country')['hlo'].mean()
    pri_out = {NAME_TO_WDI.get(c, c.lower()): float(v)
               for c, v in pri.items()}
    sec_out = {NAME_TO_WDI.get(c, c.lower()): float(v)
               for c, v in sec.items()}
    return pri_out, sec_out


def load_wcde_completion(filename, year):
    """Load % completion for a level (both sexes, age 20-24) at `year`."""
    df = pd.read_csv(os.path.join(PROC, filename), index_col="country")
    df.columns = df.columns.astype(int)
    df.index = [s.lower() for s in df.index]
    out = {}
    for c in df.index:
        if c in [r.lower() for r in REGIONS]:
            continue
        v = df.loc[c, year] if year in df.columns else np.nan
        if pd.isna(v):
            continue
        wdi = WCDE_TO_WDI.get(c, c)
        out[wdi] = float(v)
    return out


def load_bl_yrsch(year):
    bl = pd.read_csv(os.path.join(DATA, "barro_lee_v3.csv"))
    bl = bl[(bl['agefrom'] == 25) & (bl['year'] == year)].copy()
    return {NAME_TO_WDI.get(r['country'], r['country'].lower()):
            float(r['yr_sch']) for _, r in bl.iterrows()}


def load_population():
    """Country total population at 2015, in thousands.

    Read from preprocessed `wcde/data/processed/population_2015.csv` so the
    public replication repo does not need the 91 MB raw `pop_both.csv`.
    The processed file is generated by `wcde/scripts/build_population_2015.py`
    and is a row-per-country aggregate (scenario=2, year=2015, summed across
    age/sex/education).
    """
    pop_path = os.path.join(REPO_ROOT, "wcde", "data", "processed",
                            "population_2015.csv")
    pop = pd.read_csv(pop_path)
    out = {}
    for _, r in pop.iterrows():
        c = r['country']
        wdi = WCDE_TO_WDI.get(c, c)
        out[wdi] = float(r['population_thousands'])
    return out


def zscore(s):
    m, sd = s.mean(), s.std(ddof=0)
    return (s - m) / sd if sd > 0 else s * 0


def _fit(d, y, x_cols):
    X = sm.add_constant(d[x_cols].astype(float))
    m = sm.OLS(d[y].astype(float), X).fit()
    return m


def run_race(panel, y_col, y_label, quantity_col, quality_col,
             real_col, pop_col):
    d = panel.dropna(subset=[y_col, quantity_col, quality_col,
                             real_col, pop_col]).copy()
    # Standardize outcome and all predictors
    d["y_z"] = zscore(d[y_col])
    d["quant_z"] = zscore(d[quantity_col])
    d["qual_z"] = zscore(d[quality_col])
    d["real_z"] = zscore(d[real_col])
    d["logpop_z"] = zscore(np.log(d[pop_col]))

    n = len(d)
    print(f"\n{'=' * 78}")
    print(f"{y_label}   —   n = {n}")
    print(f"  quantity   = {quantity_col}")
    print(f"  quality    = {quality_col}")
    print(f"  real = q×Q = {real_col}")
    print(f"  control    = log(population) at 2015")
    print(f"{'=' * 78}")

    m_q = _fit(d, "y_z", ["quant_z", "logpop_z"])
    m_h = _fit(d, "y_z", ["qual_z", "logpop_z"])
    m_r = _fit(d, "y_z", ["real_z", "logpop_z"])
    m_qh = _fit(d, "y_z", ["quant_z", "qual_z", "logpop_z"])
    m_qr = _fit(d, "y_z", ["quant_z", "real_z", "logpop_z"])

    def _show(name, m, params):
        parts = []
        for p in params:
            parts.append(f"{p}: β={m.params[p]:+.3f} "
                         f"(t={m.tvalues[p]:+.2f}, "
                         f"p={m.pvalues[p]:.3f})")
        print(f"  {name:<28} R²={m.rsquared:.3f}  | " +
              "  ".join(parts))

    _show("(A) quantity + logpop", m_q, ["quant_z"])
    _show("(B) quality  + logpop", m_h, ["qual_z"])
    _show("(C) real q×Q + logpop", m_r, ["real_z"])
    _show("(D) quant + qual + pop", m_qh, ["quant_z", "qual_z"])
    _show("(E) quant + real + pop", m_qr, ["quant_z", "real_z"])

    # Verdict: which model has highest R²?
    scores = {
        "quantity only (A)": m_q.rsquared,
        "quality only (B)": m_h.rsquared,
        "real (q×Q) only (C)": m_r.rsquared,
        "quantity+quality (D)": m_qh.rsquared,
        "quantity+real (E)": m_qr.rsquared,
    }
    winner = max(scores, key=scores.get)
    print(f"\n  VERDICT (best R²): {winner}  "
          f"(R²={scores[winner]:.3f})")

    # Capture numbers for JSON
    return {
        "n": int(n),
        "A_quant_only_r2": round(float(m_q.rsquared), 3),
        "A_quant_beta": round(float(m_q.params["quant_z"]), 3),
        "A_quant_t": round(float(m_q.tvalues["quant_z"]), 2),
        "B_qual_only_r2": round(float(m_h.rsquared), 3),
        "B_qual_beta": round(float(m_h.params["qual_z"]), 3),
        "B_qual_t": round(float(m_h.tvalues["qual_z"]), 2),
        "C_real_only_r2": round(float(m_r.rsquared), 3),
        "D_both_r2": round(float(m_qh.rsquared), 3),
        "D_quant_beta": round(float(m_qh.params["quant_z"]), 3),
        "D_quant_t": round(float(m_qh.tvalues["quant_z"]), 2),
        "D_quant_p": round(float(m_qh.pvalues["quant_z"]), 3),
        "D_qual_beta": round(float(m_qh.params["qual_z"]), 3),
        "D_qual_t": round(float(m_qh.tvalues["qual_z"]), 2),
        "D_qual_p": round(float(m_qh.pvalues["qual_z"]), 3),
        "winner_model": winner,
    }


def main():
    print("Comprehensive Hanushek horse race:")
    print("  outcome-specific depth • HLO × completion for real edu")
    print("  • log(pop) control • USSR excluded")
    print()

    # Load primitives
    hlo_pri, hlo_sec = load_hlo_by_level()
    pri_comp = load_wcde_completion("primary_both.csv", 2000)
    lsec_comp = load_wcde_completion("lower_sec_both.csv", 2000)
    lsec_comp_2015 = load_wcde_completion("lower_sec_both.csv", 2015)
    bl_2010 = load_bl_yrsch(2010)
    pop = load_population()

    tfr = load_wide_indicator(
        "children_per_woman_total_fertility.csv")
    le = load_wide_indicator("life_expectancy_years.csv")
    u5 = load_wide_indicator("child_mortality_u5.csv")

    # Build one row per country with everything we need
    rows = []
    country_universe = (set(hlo_pri) | set(hlo_sec)) \
        & set(pri_comp) & set(lsec_comp) & set(bl_2010) & set(pop)
    for c in country_universe:
        if c in USSR_LC:
            continue
        if c not in tfr.index or c not in le.index or c not in u5.index:
            continue
        tv = tfr.loc[c, "2015"] if "2015" in tfr.columns else np.nan
        lv = le.loc[c, "2015"] if "2015" in le.columns else np.nan
        uv = u5.loc[c, "2015"] if "2015" in u5.columns else np.nan
        if pd.isna(tv) or pd.isna(lv) or pd.isna(uv):
            continue
        pc = pri_comp[c]
        lc = lsec_comp[c]
        lc15 = lsec_comp_2015.get(c, np.nan)
        ys = bl_2010[c]
        hp = hlo_pri.get(c, np.nan)
        hs = hlo_sec.get(c, np.nan)
        row = {
            "country": c,
            "tfr_2015": tv,
            "u5mr_2015": uv,
            "log_u5mr_2015": np.log(uv),
            "le_2015": lv,
            "pri_comp_2000": pc,
            "lsec_comp_2000": lc,
            "lsec_comp_2015": lc15,
            "bl_yrsch_2010": ys,
            "hlo_pri": hp,
            "hlo_sec": hs,
            # Real (quantity × quality)
            "real_pri": pc * hp / 500 if not pd.isna(hp) else np.nan,
            "real_lsec": lc * hs / 500 if not pd.isna(hs) else np.nan,
            "real_lifetime": ys * hs / 500 if not pd.isna(hs)
                             else np.nan,
            "pop_2015": pop[c],
        }
        rows.append(row)
    panel = pd.DataFrame(rows)
    print(f"Panel: {len(panel)} countries")
    print(f"  with primary-HLO: "
          f"{panel['hlo_pri'].notna().sum()}")
    print(f"  with secondary-HLO: "
          f"{panel['hlo_sec'].notna().sum()}")

    # Race 1: TFR ← primary
    tfr_res = run_race(panel, "tfr_2015",
             "TFR at 2015  (primary-driven hypothesis)",
             quantity_col="pri_comp_2000",
             quality_col="hlo_pri",
             real_col="real_pri",
             pop_col="pop_2015")

    # Race 2: U5MR ← lower-sec
    u5_res = run_race(panel, "log_u5mr_2015",
             "log U5MR at 2015  (lower-sec-driven hypothesis)",
             quantity_col="lsec_comp_2000",
             quality_col="hlo_sec",
             real_col="real_lsec",
             pop_col="pop_2015")

    # Race 3: LE ← total lifetime education
    le_res = run_race(panel, "le_2015",
             "LE at 2015  (lifetime-integrated education)",
             quantity_col="bl_yrsch_2010",
             quality_col="hlo_sec",
             real_col="real_lifetime",
             pop_col="pop_2015")

    write_checkin("hanushek_horse_race.json", {
        "numbers": {
            "panel_n": int(len(panel)),
            "panel_with_primary_hlo": int(panel["hlo_pri"].notna().sum()),
            "panel_with_sec_hlo": int(panel["hlo_sec"].notna().sum()),
            "tfr": tfr_res,
            "u5mr": u5_res,
            "le": le_res,
        },
    }, script_path="scripts/hanushek_horse_race_comprehensive.py")


if __name__ == "__main__":
    main()
