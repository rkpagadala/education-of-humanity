"""
which_edu_measure_is_correct.py

Head-to-head statistical test: which reconstruction of educational
attainment — WCDE v3 lower-secondary completion, or Barro-Lee v3
attainment — better predicts the downstream phenotype 25 years later?

Assumption behind the test:
  Both WCDE and B-L are noisy measures of a latent true education
  stock X*. The phenotype (TFR, LE, U5MR) is produced by the true X*
  through the paper's parental-transmission channel. Under OLS with
  country and year fixed effects, the education measure with less
  noise around X* produces:
    - higher within-R²,
    - larger |coefficient| (less attenuation bias),
    - tighter residuals for contested observations.

If WCDE is inflated for Soviet countries (reports ~95% where truth is
much lower), WCDE's global fit will be degraded relative to B-L, and
the Soviet observations specifically will show systematic residuals.

Test structure
---------------
 (1) Build a panel on overlapping years {1960, 1970, 1980, 1990}
     with WCDE lsec (completed lower sec, age 20-24),
     B-L ls+lh (reached secondary+, age 25-34), and
     B-L yr_sch (mean years of schooling, age 25-34).
 (2) For each outcome (TFR, LE, log U5MR) at T+25, run three OLS
     specifications with country+year FEs:
       (a) outcome(T+25) ~ WCDE(T)
       (b) outcome(T+25) ~ BL_ls_lh(T)
       (c) outcome(T+25) ~ BL_yr_sch(T)
     Report within-R², coefficient, residual SD, N.
 (3) Dig into the contested post-Soviet cohort. For each test-case
     country-year pair, compute residuals under specs (a) and (b).
     Positive residual = actual phenotype WORSE than the model predicts
     given that country's reported education (i.e., the education
     number was too high for the observed phenotype).

A measure that systematically over-predicts good phenotypes for Soviet
countries has a positive residual bias on that subset — direct
evidence of upward bias in that measure.
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import DATA, PROC, load_wide_indicator, write_checkin

YEARS_EDU = [1960, 1970, 1980, 1990]  # years we observe education
LAG = 25                              # forward lag to outcome

# Test-case cohort: expanded to separate USSR republics from
# Warsaw-Pact / Yugoslav satellites (which had their own national
# statistical offices, not Soviet Goskomstat).
USSR_REPUBLICS_BL = {
    "Kazakhstan", "Kyrgyzstan", "Tajikistan",
    "Russian Federation", "Ukraine",
    "Armenia", "Azerbaijan", "Georgia",
    "Latvia", "Estonia", "Lithuania",
    "Republic of Moldova",
}
WARSAW_YUGO_BL = {
    "Poland", "Hungary", "Romania", "Bulgaria",
    "Czech Republic", "Czechia", "Slovakia", "Slovak Republic",
    "Croatia", "Slovenia", "Serbia",
    "Bosnia and Herzegovina", "North Macedonia",
    "The former Yugoslav Republic of Macedonia",
    "Montenegro", "Albania",
}
TEST_CASES_BL = USSR_REPUBLICS_BL | WARSAW_YUGO_BL
# Mapping BL name -> WDI-lowercase name for TFR/LE/U5MR lookup
BL_TO_WDI = {
    "Russian Federation": "russian federation",
    "Iran (Islamic Republic of)": "iran",
    "Turkey": "turkey",
    "Kazakhstan": "kazakhstan",
    "Kyrgyzstan": "kyrgyz republic",
    "Tajikistan": "tajikistan",
    "Ukraine": "ukraine",
    "Armenia": "armenia",
    "Azerbaijan": "azerbaijan",
    "Georgia": "georgia",
    "Republic of Moldova": "moldova",
    "Latvia": "latvia",
    "Estonia": "estonia",
    "Lithuania": "lithuania",
    "Poland": "poland",
    "Hungary": "hungary",
    "Romania": "romania",
    "Bulgaria": "bulgaria",
    "Czech Republic": "czech republic",
    "Czechia": "czech republic",
    "Slovakia": "slovak republic",
    "Slovak Republic": "slovak republic",
    "Croatia": "croatia",
    "Slovenia": "slovenia",
    "Serbia": "serbia",
    "Bosnia and Herzegovina": "bosnia and herzegovina",
    "North Macedonia": "north macedonia",
    "The former Yugoslav Republic of Macedonia": "north macedonia",
    "Montenegro": "montenegro",
    "Albania": "albania",
}


def load_panel():
    """Build a long panel: rows indexed by (country, edu_year), with
    WCDE lsec, B-L ls+lh, B-L yr_sch, and TFR/LE/U5MR at edu_year+25."""
    bl = pd.read_csv(os.path.join(DATA, "barro_lee_v3.csv"))
    # Age 25-34 is the young-adult cohort closest to WCDE's 20-24.
    bl = bl[bl['agefrom'] == 25].copy()
    bl['bl_reach_sec'] = bl['ls'] + bl['lh']
    bl['bl_yr_sch'] = bl['yr_sch']
    bl = bl[['country', 'year', 'bl_reach_sec', 'bl_yr_sch']]

    wcde = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"),
                       index_col="country")
    wcde.columns = wcde.columns.astype(int)
    wcde_long = wcde.stack().reset_index()
    wcde_long.columns = ['country', 'year', 'wcde_lsec']

    # Merge WCDE and B-L on BL's country names. WCDE uses long names
    # that mostly match BL (e.g., "Iran (Islamic Republic of)").
    merged = bl.merge(wcde_long, on=['country', 'year'], how='inner')
    merged = merged[merged['year'].isin(YEARS_EDU)].dropna()

    # Now pull TFR / LE / U5MR at year + 25 from WDI.
    tfr = load_wide_indicator("children_per_woman_total_fertility.csv")
    le = load_wide_indicator("life_expectancy_years.csv")
    u5 = load_wide_indicator("child_mortality_u5.csv")

    # Build WDI name map from BL names. BL uses long names; WDI uses
    # short lowercase names.
    def wdi_name(bl_name):
        return BL_TO_WDI.get(bl_name, bl_name.lower())

    outs = []
    for _, r in merged.iterrows():
        wn = wdi_name(r['country'])
        tyr = r['year'] + LAG
        if str(tyr) not in tfr.columns or str(tyr) not in le.columns:
            continue
        try:
            tv = tfr.loc[wn, str(tyr)] if wn in tfr.index else np.nan
            lv = le.loc[wn, str(tyr)] if wn in le.index else np.nan
            uv = u5.loc[wn, str(tyr)] if wn in u5.index else np.nan
        except KeyError:
            continue
        outs.append({
            'country': r['country'],
            'edu_year': r['year'],
            'out_year': tyr,
            'wcde_lsec': r['wcde_lsec'],
            'bl_reach_sec': r['bl_reach_sec'],
            'bl_yr_sch': r['bl_yr_sch'],
            'tfr': tv, 'le': lv, 'u5mr': uv,
        })
    p = pd.DataFrame(outs)
    p['log_u5mr'] = np.log(p['u5mr'])
    return p


def run_fe_ols(df, y, x, drop_countries=None):
    """Fit pooled OLS with YEAR FE only (no country FE), on training set.
    Year FE absorbs secular trends; country deviations remain visible
    in residuals. If `drop_countries` provided, fit on the complement
    and return predictions / residuals on the full panel.
    """
    d = df[[y, x, 'country', 'edu_year']].dropna().copy()
    d = d.reset_index(drop=True)
    if len(d) < 30:
        return None
    X_full = pd.get_dummies(d[['edu_year']], drop_first=True, dtype=float)
    X_full[x] = d[x].values
    X_full = sm.add_constant(X_full).astype(float)
    y_arr = d[y].astype(float).values

    if drop_countries:
        train_mask = ~d['country'].isin(drop_countries)
    else:
        train_mask = np.ones(len(d), dtype=bool)
    model = sm.OLS(y_arr[train_mask], X_full.loc[train_mask]).fit()
    pred = np.asarray(model.predict(X_full))
    resid = y_arr - pred
    train_resid = resid[train_mask]
    return {
        'n_train': int(train_mask.sum()),
        'n_full': len(d),
        'coef': model.params[x],
        'se': model.bse[x],
        't': model.tvalues[x],
        'p': model.pvalues[x],
        'r2': model.rsquared,
        'rmse_train': float(np.sqrt(np.mean(train_resid ** 2))),
        'resid': resid,            # numpy array, positional, full panel
        'pred': pred,
        'country': d['country'].values,
        'edu_year': d['edu_year'].values,
        'y': y_arr,
    }


def main():
    p = load_panel()
    print(f"Panel: n={len(p)} obs, "
          f"{p['country'].nunique()} countries, "
          f"years {sorted(p['edu_year'].unique())}")
    print()

    # ── HEAD-TO-HEAD FIT ──────────────────────────────────────
    outcomes = {'tfr': 'TFR (T+25)',
                'le': 'Life expectancy (T+25)',
                'log_u5mr': 'log U5MR (T+25)'}
    measures = {'wcde_lsec': 'WCDE lower-sec completion',
                'bl_reach_sec': 'B-L reached secondary+',
                'bl_yr_sch': 'B-L mean years schooling'}

    print("=" * 78)
    print("GLOBAL FIT — pooled OLS with year FE (country-held-out training)")
    print("  Training set: all countries EXCEPT the contested cohort.")
    print("  Year FE absorbs secular trends. Coefficient on education")
    print("  is the pooled cross-country slope.")
    print("=" * 78)
    print(f"{'Outcome':<22}  {'Measure':<28}  "
          f"{'n_tr':>5}  {'coef':>8}  {'t':>6}  {'R²':>6}  {'RMSE':>6}")
    fits = {}
    for y, ylabel in outcomes.items():
        for x, xlabel in measures.items():
            res = run_fe_ols(p, y, x, drop_countries=TEST_CASES_BL)
            if res is None:
                continue
            fits[(y, x)] = res
            print(f"{ylabel[:22]:<22}  {xlabel[:28]:<28}  "
                  f"{res['n_train']:>5}  {res['coef']:>8.4f}  "
                  f"{res['t']:>6.2f}  {res['r2']:>6.3f}  "
                  f"{res['rmse_train']:>6.3f}")
        print()

    # ── CONTESTED-COUNTRY RESIDUALS ──────────────────────────
    # For each test-case country-year, what was the residual
    # (actual - predicted) under each model?
    print("=" * 78)
    print("CONTESTED COHORT RESIDUALS: actual outcome minus predicted")
    print("  Positive residual (for LE) = phenotype BETTER than the model")
    print("    expected given reported education.")
    print("  Positive residual (for TFR, log U5MR) = phenotype WORSE than")
    print("    the model expected given reported education.")
    print("  A consistently wrong-sign residual on a subset = that")
    print("    measure over-predicts good outcomes for that subset =")
    print("    that measure's education values are too high.")
    print("=" * 78)

    for y, ylabel in outcomes.items():
        print(f"\n[{ylabel}]  (higher = better for LE; worse for TFR/U5MR)")
        print(f"  {'country':<22}  {'edu':>4}→{'out':>4}  "
              f"{'WCDE-resid':>10}  {'B-L-resid':>10}")
        for country in sorted(TEST_CASES_BL):
            sub = p[p['country'] == country]
            if not len(sub):
                continue
            for _, row in sub.iterrows():
                wc_res = fits.get((y, 'wcde_lsec'))
                bl_res = fits.get((y, 'bl_reach_sec'))
                # Find this row's residual in each model
                def find_resid(res_obj):
                    for i in range(len(res_obj['country'])):
                        if (res_obj['country'][i] == country
                                and res_obj['edu_year'][i] == row['edu_year']):
                            return res_obj['resid'][i]
                    return None
                r_wc = find_resid(wc_res)
                r_bl = find_resid(bl_res)
                if r_wc is None or r_bl is None:
                    continue
                print(f"  {country[:22]:<22}  "
                      f"{int(row['edu_year']):>4}→{int(row['out_year']):>4}  "
                      f"{r_wc:>10.3f}  {r_bl:>10.3f}")

    # Aggregate bias on contested cohort: mean out-of-sample residual
    print()
    print("=" * 78)
    print("OUT-OF-SAMPLE RESIDUAL ON CONTESTED COHORT")
    print("  Model was trained WITHOUT the contested countries.")
    print("  Residual = actual outcome − predicted outcome given that")
    print("  country's reported education value.")
    print("  If a measure reports TOO-HIGH education values, the model")
    print("  over-predicts good outcomes; residuals should be:")
    print("    - TFR:      positive (actual TFR > predicted)")
    print("    - LE:       negative (actual LE < predicted)")
    print("    - log U5MR: positive (actual U5MR > predicted)")
    print("=" * 78)
    print(f"{'Outcome':<22}  {'Measure':<28}  "
          f"{'n':>3}  {'mean_resid':>11}  {'se':>8}  "
          f"{'t':>6}  {'bias_std':>9}")
    # bias_std = mean_resid / RMSE — dimensionless bias size
    for y, ylabel in outcomes.items():
        for x in ['wcde_lsec', 'bl_reach_sec']:
            res = fits.get((y, x))
            if res is None:
                continue
            mask = np.array([c in TEST_CASES_BL for c in res['country']])
            r = res['resid'][mask]
            if len(r) == 0:
                continue
            mean_r = float(r.mean())
            se_r = float(r.std(ddof=1) / np.sqrt(len(r)))
            t_r = mean_r / se_r if se_r > 0 else np.nan
            bias_std = mean_r / res['rmse_train']
            print(f"{ylabel[:22]:<22}  "
                  f"{measures[x][:28]:<28}  "
                  f"{len(r):>3}  {mean_r:>11.4f}  {se_r:>8.4f}  "
                  f"{t_r:>6.2f}  {bias_std:>9.3f}")
        print()

    # RMSE on contested cohort: which measure predicts contested
    # countries' outcomes with less error?
    print("=" * 78)
    print("OUT-OF-SAMPLE RMSE ON CONTESTED COHORT")
    print("  Lower RMSE = measure produces predictions closer to actual")
    print("  outcomes for these countries = closer to truth.")
    print("=" * 78)
    print(f"{'Outcome':<22}  "
          f"{'WCDE RMSE':>10}  {'B-L RMSE':>10}  "
          f"{'Δ (B-L − WCDE)':>16}  {'winner':>8}")
    for y, ylabel in outcomes.items():
        wc = fits.get((y, 'wcde_lsec'))
        bl = fits.get((y, 'bl_reach_sec'))
        if not (wc and bl):
            continue
        mask_w = np.array([c in TEST_CASES_BL for c in wc['country']])
        mask_b = np.array([c in TEST_CASES_BL for c in bl['country']])
        rmse_w = float(np.sqrt(np.mean(wc['resid'][mask_w] ** 2)))
        rmse_b = float(np.sqrt(np.mean(bl['resid'][mask_b] ** 2)))
        winner = "B-L" if rmse_b < rmse_w else "WCDE"
        print(f"{ylabel[:22]:<22}  "
              f"{rmse_w:>10.3f}  {rmse_b:>10.3f}  "
              f"{rmse_b - rmse_w:>+16.3f}  {winner:>8}")

    # ── SPLIT: Central-Asia-Caucasus (suspected hollow) vs Baltics
    # ── (suspected real) ────────────────────────────────────────
    print()
    print("=" * 78)
    print("SPLIT BY SUBGROUP — is the bias uniform or concentrated?")
    print("  Group A (hollow-suspected): Kazakhstan, Kyrgyzstan,")
    print("    Tajikistan, Armenia. Reported 90%+ lower-sec in 1970")
    print("    under Soviet system, but Kyrgyzstan HLO 2009 = 350.")
    print("  Group B (real-suspected):  Latvia, Estonia, Lithuania.")
    print("    Pre-1940 educational infrastructure, Baltic HLO 500+.")
    print("  Same pooled-OLS model, same held-out training set.")
    print("=" * 78)
    # Three subgroups: Central Asia + Caucasus (hollow), Baltics,
    # Warsaw Pact / Yugoslav (separate statistical offices).
    g_central_caucasus = {
        "Kazakhstan", "Kyrgyzstan", "Tajikistan",
        "Armenia", "Azerbaijan", "Georgia",
    }
    g_baltics = {"Latvia", "Estonia", "Lithuania"}
    g_slavic = {"Russian Federation", "Ukraine"}
    g_warsaw = {
        "Poland", "Hungary", "Romania", "Bulgaria",
        "Czech Republic", "Czechia", "Slovakia", "Slovak Republic",
    }
    g_yugoslav = {
        "Croatia", "Slovenia", "Serbia",
        "Bosnia and Herzegovina", "North Macedonia",
        "The former Yugoslav Republic of Macedonia",
        "Montenegro", "Albania",
    }
    subgroup_results = {}
    for y, ylabel in outcomes.items():
        for x in ['wcde_lsec', 'bl_reach_sec']:
            res = fits.get((y, x))
            if res is None:
                continue
            print(f"\n  [{ylabel} / {measures[x]}]")
            for group_name, group_set in [
                ("central_caucasus", g_central_caucasus),
                ("baltics", g_baltics),
                ("slavic_west", g_slavic),
                ("warsaw_pact", g_warsaw),
                ("yugoslavia_albania", g_yugoslav),
            ]:
                mask = np.array([c in group_set for c in res['country']])
                r = res['resid'][mask]
                if len(r) == 0:
                    print(f"    {group_name}  (no observations)")
                    continue
                mean_r = float(r.mean())
                se_r = float(r.std(ddof=1) / np.sqrt(len(r)))
                t_r = mean_r / se_r if se_r > 0 else np.nan
                bias_std = mean_r / res['rmse_train']
                print(f"    {group_name:<22}  "
                      f"n={len(r):>2}  "
                      f"mean_resid={mean_r:>+8.3f}  "
                      f"SE={se_r:>6.3f}  t={t_r:>5.2f}  "
                      f"bias_std={bias_std:>+6.3f}")
                subgroup_results[f"{y}_{x}_{group_name}"] = {
                    "n": int(len(r)),
                    "mean_resid": round(mean_r, 3),
                    "se": round(se_r, 3),
                    "t": round(float(t_r), 2),
                    "bias_sds": round(bias_std, 2),
                }

    # ── JSON emission ──────────────────────────────────────────────
    numbers = {"panel_n": int(len(p)),
               "panel_countries": int(p['country'].nunique())}
    # Global fit coefficients
    for y in outcomes:
        for x in ['wcde_lsec', 'bl_reach_sec', 'bl_yr_sch']:
            res = fits.get((y, x))
            if not res:
                continue
            numbers[f"{y}_{x}_coef"] = round(float(res['coef']), 4)
            numbers[f"{y}_{x}_t"] = round(float(res['t']), 2)
            numbers[f"{y}_{x}_r2"] = round(float(res['r2']), 3)
            numbers[f"{y}_{x}_rmse"] = round(float(res['rmse_train']), 3)
    # Aggregate contested-cohort bias
    for y in outcomes:
        for x in ['wcde_lsec', 'bl_reach_sec']:
            res = fits.get((y, x))
            if not res:
                continue
            mask = np.array([c in TEST_CASES_BL for c in res['country']])
            r = res['resid'][mask]
            if len(r) == 0:
                continue
            mean_r = float(r.mean())
            numbers[f"{y}_{x}_agg_bias_sds"] = round(
                mean_r / res['rmse_train'], 2)
    # Subgroup split
    numbers["subgroup_split"] = subgroup_results

    write_checkin("edu_measure_horse_race.json", {
        "numbers": numbers,
    }, script_path="scripts/which_edu_measure_is_correct.py")


if __name__ == "__main__":
    main()
