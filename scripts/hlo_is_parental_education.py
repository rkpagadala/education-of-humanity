"""
hlo_is_parental_education.py

Test whether Hanushek's HLO (cognitive skills of 14-15-year-olds) is
largely a downstream function of their parents' education 25 years
ago — i.e., what the paper's PT mechanism predicts.

HLO tests 15-year-olds today. Their parents are ~30 today, and had
their formative education ~25 years ago (when the parents were
~20-24). So:

  HLO_today(15yo) = f( parental_education_25yrs_ago(20-24) )

If this relationship is tight, Hanushek is measuring a lagged
function of the paper's quantity measure — not an independent
"quality" alternative.

Test:
  1. Regress country-mean HLO (secondary, math+reading+science) on
     lower-secondary completion at 1990 (parents' 20-24 education
     when today's 15-year-olds were born).
  2. Report R², slope, and plot.
  3. Also test with BL yr_sch at 1990 as a continuous alternative.
  4. Then re-run the TFR / U5MR / LE horse race, but decompose HLO
     into (a) predicted-by-parental-education component and (b)
     residual ("pure school quality" independent of PT).

USSR republics excluded — their reported 1990 education fails the
phenotype-consistency test.
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "Republic of Moldova": "moldova",
    "Moldova": "moldova",
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


def load_hlo_secondary():
    hlo = pd.read_csv(os.path.join(DATA, "hlo_raw.csv"))
    sec = hlo[(hlo['level'] == 'sec')
              & (hlo['subject'].isin(['math', 'reading', 'science']))]
    out = {}
    for c, v in sec.groupby('country')['hlo'].mean().items():
        out[NAME_TO_WDI.get(c, c.lower())] = float(v)
    return out


def load_wcde_lsec_1990():
    lsec = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"),
                       index_col="country")
    lsec.columns = lsec.columns.astype(int)
    lsec.index = [s.lower() for s in lsec.index]
    out = {}
    for c in lsec.index:
        if c in [r.lower() for r in REGIONS]:
            continue
        if 1990 not in lsec.columns:
            continue
        v = lsec.loc[c, 1990]
        if pd.isna(v):
            continue
        wdi = WCDE_TO_WDI.get(c, c)
        out[wdi] = float(v)
    return out


def load_bl_yrsch_1990():
    bl = pd.read_csv(os.path.join(DATA, "barro_lee_v3.csv"))
    bl = bl[(bl['agefrom'] == 25) & (bl['year'] == 1990)].copy()
    return {NAME_TO_WDI.get(r['country'], r['country'].lower()):
            float(r['yr_sch']) for _, r in bl.iterrows()}


def main():
    hlo = load_hlo_secondary()
    lsec_1990 = load_wcde_lsec_1990()
    yrs_1990 = load_bl_yrsch_1990()

    rows = []
    for c, h in hlo.items():
        if c in USSR_LC:
            continue
        row = {"country": c, "hlo_sec": h,
               "lsec_1990": lsec_1990.get(c),
               "yrs_1990": yrs_1990.get(c)}
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"Countries with HLO: {len(df)} (USSR excluded)")
    print(f"  with lsec 1990: {df['lsec_1990'].notna().sum()}")
    print(f"  with BL yr_sch 1990: {df['yrs_1990'].notna().sum()}")
    print()

    # ── Test 1: HLO ~ lsec_1990 ──────────────────────────────────
    d1 = df.dropna(subset=["hlo_sec", "lsec_1990"]).copy()
    X = sm.add_constant(d1["lsec_1990"].astype(float))
    m1 = sm.OLS(d1["hlo_sec"].astype(float), X).fit()
    print("=" * 72)
    print(f"TEST 1: HLO_secondary_today ~ lsec_completion_1990")
    print(f"  (parents' 20-24 education 25 years ago → "
          f"15-year-old test score today)")
    print("=" * 72)
    print(f"  n = {len(d1)}")
    print(f"  β (lsec_1990) = {m1.params['lsec_1990']:+.3f} "
          f"HLO points per pp  (t = {m1.tvalues['lsec_1990']:.2f})")
    print(f"  intercept    = {m1.params['const']:+.1f}")
    print(f"  R²           = {m1.rsquared:.3f}")
    print(f"  Corr(HLO, lsec_1990) = "
          f"{d1['hlo_sec'].corr(d1['lsec_1990']):.3f}")
    print()

    # ── Test 2: HLO ~ BL yr_sch_1990 ────────────────────────────
    d2 = df.dropna(subset=["hlo_sec", "yrs_1990"]).copy()
    X = sm.add_constant(d2["yrs_1990"].astype(float))
    m2 = sm.OLS(d2["hlo_sec"].astype(float), X).fit()
    print("=" * 72)
    print(f"TEST 2: HLO_secondary_today ~ BL_yr_sch_1990 (age 25-34)")
    print("=" * 72)
    print(f"  n = {len(d2)}")
    print(f"  β (yrs_1990) = {m2.params['yrs_1990']:+.2f} "
          f"HLO points per year  (t = {m2.tvalues['yrs_1990']:.2f})")
    print(f"  R²           = {m2.rsquared:.3f}")
    print(f"  Corr(HLO, yrs_1990) = "
          f"{d2['hlo_sec'].corr(d2['yrs_1990']):.3f}")
    print()

    # ── Decompose HLO into PT-predicted + residual ──────────────
    d = df.dropna(subset=["hlo_sec", "lsec_1990"]).copy()
    X = sm.add_constant(d["lsec_1990"].astype(float))
    m = sm.OLS(d["hlo_sec"].astype(float), X).fit()
    d["hlo_predicted_from_pt"] = m.predict(X)
    d["hlo_residual"] = d["hlo_sec"] - d["hlo_predicted_from_pt"]

    print("=" * 72)
    print("HLO decomposition: PT-predicted vs residual (school quality "
          "independent of parental education)")
    print("=" * 72)
    print(f"  Total variance of HLO:   "
          f"{d['hlo_sec'].var():.1f}")
    print(f"  Explained by lsec_1990:  "
          f"{m.rsquared * 100:.1f}%")
    print(f"  Residual variance:       "
          f"{d['hlo_residual'].var():.1f}")
    print()
    print("  Top 10 POSITIVE residuals (HLO higher than PT predicts —")
    print("    school quality adding on top of parental education):")
    top = d.nlargest(10, "hlo_residual")
    for _, r in top.iterrows():
        print(f"    {r['country']:<28}  HLO={r['hlo_sec']:.0f}  "
              f"lsec_1990={r['lsec_1990']:.1f}%  "
              f"residual=+{r['hlo_residual']:.0f}")
    print()
    print("  Top 10 NEGATIVE residuals (HLO lower than PT predicts —")
    print("    school quality undershooting parental education):")
    bot = d.nsmallest(10, "hlo_residual")
    for _, r in bot.iterrows():
        print(f"    {r['country']:<28}  HLO={r['hlo_sec']:.0f}  "
              f"lsec_1990={r['lsec_1990']:.1f}%  "
              f"residual={r['hlo_residual']:.0f}")
    print()

    # ── Scatter plot ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(d["lsec_1990"], d["hlo_sec"],
               s=50, color="#1f4e79", alpha=0.7,
               edgecolor="white", linewidth=0.6)
    xx = np.linspace(0, 100, 200)
    yy = m.params["const"] + m.params["lsec_1990"] * xx
    ax.plot(xx, yy, color="#c0392b", linewidth=2,
            label=f"HLO = {m.params['const']:.0f} + "
                  f"{m.params['lsec_1990']:.2f}·lsec_1990  "
                  f"(R²={m.rsquared:.2f})")
    # Label extreme cases
    for _, r in pd.concat([top.head(5), bot.head(5)]).iterrows():
        ax.annotate(r["country"][:18],
                    xy=(r["lsec_1990"], r["hlo_sec"]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=7, color="#444")
    ax.axhline(500, color="#222", linewidth=0.8,
               linestyle=":", alpha=0.6)
    ax.text(2, 500, " OECD ≈ 500",
            fontsize=8, va="bottom", color="#222")
    ax.set_xlabel("Lower-secondary completion at 1990 "
                  "(parents' education 25 years ago, %)",
                  fontsize=11)
    ax.set_ylabel("HLO secondary today (math + reading + science mean)",
                  fontsize=11)
    ax.set_title(
        "Hanushek's HLO of today's 15-year-olds is largely a function "
        "of their parents'\neducation 25 years ago — the paper's PT "
        "mechanism in action\n"
        f"{m.rsquared * 100:.0f}% of HLO variance is explained by "
        f"lower-sec completion 1990 (n={len(d)}, USSR excluded)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10, frameon=False)
    ax.set_xlim(0, 105)
    fig.tight_layout()

    out_path = os.path.join(REPO_ROOT, "paper", "figures",
                            "hlo_vs_parental_education_1990.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")

    write_checkin("hlo_is_pt.json", {
        "numbers": {
            # Test 1: HLO ~ lsec_1990
            "t1_n": int(len(d1)),
            "t1_beta_lsec": round(float(m1.params["lsec_1990"]), 3),
            "t1_t_lsec": round(float(m1.tvalues["lsec_1990"]), 2),
            "t1_intercept": round(float(m1.params["const"]), 1),
            "t1_r2": round(float(m1.rsquared), 3),
            "t1_corr": round(
                float(d1["hlo_sec"].corr(d1["lsec_1990"])), 3),
            # Test 2: HLO ~ BL yr_sch 1990
            "t2_n": int(len(d2)),
            "t2_beta_yrs": round(float(m2.params["yrs_1990"]), 2),
            "t2_r2": round(float(m2.rsquared), 3),
            # Top/bottom residuals (school overperformers / underperformers)
            "top_positive_residuals": [
                {"country": r["country"],
                 "hlo": round(float(r["hlo_sec"]), 0),
                 "lsec_1990": round(float(r["lsec_1990"]), 1),
                 "residual": round(float(r["hlo_residual"]), 0)}
                for _, r in top.head(6).iterrows()],
            "top_negative_residuals": [
                {"country": r["country"],
                 "hlo": round(float(r["hlo_sec"]), 0),
                 "lsec_1990": round(float(r["lsec_1990"]), 1),
                 "residual": round(float(r["hlo_residual"]), 0)}
                for _, r in bot.head(6).iterrows()],
        },
    }, script_path="scripts/hlo_is_parental_education.py")


if __name__ == "__main__":
    main()
