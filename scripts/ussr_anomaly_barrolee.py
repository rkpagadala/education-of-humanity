"""
ussr_anomaly_barrolee.py

Same as ussr_anomaly_wcde.py but uses Barro-Lee v3 mean years of
schooling (age 25-34) as the educational measure instead of WCDE
lower-secondary completion.

Contemporaneous (not lagged). Trajectory plot: each USSR country
appears as a line connecting its observations at t ∈ {1960, 1970,
1980, 1990, 2000, 2010} (Barro-Lee ends at 2015, so 2010 is the last
clean 10-year point).

Why two scripts: Barro-Lee is an independent reconstruction with more
conservative Soviet-era figures than WCDE. Comparing the two figures
side by side shows how much of the USSR anomaly survives the
switch to B-L's more cautious reading — the part of the anomaly
that persists is not a WCDE measurement artifact.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import (
    REPO_ROOT, DATA, load_wide_indicator,
    TFR_THRESHOLD, LE_THRESHOLD, write_checkin,
)

YEARS = [1960, 1970, 1980, 1990, 2000, 2010]

USSR_BL = {
    "Russian Federation", "Ukraine",
    "Estonia", "Latvia", "Lithuania",
    "Kazakhstan", "Kyrgyzstan", "Tajikistan",
    "Armenia",
    "Republic of Moldova",
}

BL_TO_WDI = {
    "Russian Federation": "russian federation",
    "Iran (Islamic Republic of)": "iran",
    "Turkey": "turkiye",
    "Republic of Korea": "korea, rep.",
    "Kyrgyzstan": "kyrgyz republic",
    "Slovakia": "slovak republic",
    "Republic of Moldova": "moldova",
    "Venezuela (Bolivarian Republic of)": "venezuela, rb",
    "Bolivia (Plurinational State of)": "bolivia",
    "Lao People's Democratic Republic": "lao pdr",
    "Viet Nam": "viet nam",
    "Congo": "congo, rep.",
    "Democratic Republic of the Congo": "congo, dem. rep.",
    "United States of America": "united states",
    "United Republic of Tanzania": "tanzania",
    "Hong Kong Special Administrative Region of China":
        "hong kong sar, china",
    "Macao Special Administrative Region of China":
        "macao sar, china",
    "Swaziland": "eswatini",
    "Eswatini": "eswatini",
    "Cabo Verde": "cabo verde",
    "Yemen": "yemen, rep.",
    "Egypt": "egypt, arab rep.",
    "Syrian Arab Republic": "syrian arab republic",
    "North Macedonia": "north macedonia",
    "The former Yugoslav Republic of Macedonia": "north macedonia",
    "Czech Republic": "czech republic",
    "Czechia": "czech republic",
    "Gambia": "gambia, the",
    "Bahamas": "bahamas, the",
}

OUT_DIR = os.path.join(REPO_ROOT, "paper", "figures")


def build_panel():
    bl = pd.read_csv(os.path.join(DATA, "barro_lee_v3.csv"))
    bl = bl[bl['agefrom'] == 25].copy()
    yrs = bl.pivot_table(index='country', columns='year',
                         values='yr_sch')

    tfr = load_wide_indicator("children_per_woman_total_fertility.csv")
    le = load_wide_indicator("life_expectancy_years.csv")

    rows = []
    for c in yrs.index:
        wdi = BL_TO_WDI.get(c, c.lower())
        if wdi not in tfr.index or wdi not in le.index:
            continue
        for t in YEARS:
            if t not in yrs.columns:
                continue
            edu = yrs.loc[c, t]
            if str(t) not in tfr.columns or str(t) not in le.columns:
                continue
            tval = tfr.loc[wdi, str(t)]
            lval = le.loc[wdi, str(t)]
            if pd.isna(edu) or pd.isna(tval) or pd.isna(lval):
                continue
            rows.append({
                "country": c, "t": t,
                "edu_t": float(edu),
                "tfr_t": float(tval),
                "le_t": float(lval),
                "is_ussr": c in USSR_BL,
            })
    return pd.DataFrame(rows)


def short_name(c):
    return {
        "Russian Federation": "Russia",
        "Republic of Moldova": "Moldova",
    }.get(c, c.split()[0])[:10]


def fit_and_plot(ax, p, outcome_col, outcome_label, threshold_val):
    non = p[~p["is_ussr"]]
    ussr = p[p["is_ussr"]]

    x = non["edu_t"].values
    y = non[outcome_col].values
    beta1, beta0 = np.polyfit(x, y, 1)
    xx = np.linspace(0, 15, 200)
    yy = beta0 + beta1 * xx
    resid = y - (beta0 + beta1 * x)
    sigma = float(np.std(resid, ddof=2))

    ax.fill_between(xx, yy - 2 * sigma, yy + 2 * sigma,
                    color="#cfdcee", alpha=0.5,
                    label=f"non-USSR ±2σ ({2 * sigma:.1f})")
    ax.plot(xx, yy, color="#1f4e79", linewidth=2.0,
            label=f"non-USSR fit: y = {beta0:.1f} + {beta1:.3f}·yrs")
    ax.scatter(non["edu_t"], non[outcome_col],
               s=16, color="#1f4e79", alpha=0.35,
               edgecolor="white", linewidth=0.3,
               label=f"non-USSR (n={len(non)})")

    n_u = ussr["country"].nunique()
    cmap = plt.cm.get_cmap("tab20", 20)
    for i, (name, grp) in enumerate(ussr.groupby("country")):
        grp = grp.sort_values("t")
        color = cmap(i % 20)
        ax.plot(grp["edu_t"], grp[outcome_col], "-",
                color=color, linewidth=1.5, alpha=0.85, zorder=4)
        ax.scatter(grp["edu_t"], grp[outcome_col],
                   s=35, color=color, edgecolor="#7a1b12",
                   linewidth=0.6, marker="o", zorder=5)
        last = grp.iloc[-1]
        ax.annotate(short_name(name),
                    xy=(last["edu_t"], last[outcome_col]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=7, color=color, alpha=0.98,
                    fontweight="bold")
    ax.scatter([], [], s=40, color="#c0392b", edgecolor="#7a1b12",
               marker="o",
               label=f"USSR trajectories ({n_u} countries, up to "
                     f"{len(YEARS)} points each)")

    ax.axhline(threshold_val, color="#222", linewidth=0.8,
               linestyle=":", alpha=0.7)
    ax.text(0.3, threshold_val,
            f"  {outcome_label.split(' ')[0]} "
            f"threshold = {threshold_val}",
            fontsize=8,
            va="bottom" if "TFR" in outcome_label else "top",
            color="#222")
    ax.set_xlabel("Mean years of schooling at t, age 25-34 (Barro-Lee)",
                  fontsize=10)
    ax.set_ylabel(f"{outcome_label} at t", fontsize=10)
    ax.set_xlim(0, 15)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best", frameon=False)

    ussr_pred = beta0 + beta1 * ussr["edu_t"].values
    ussr_resid = ussr[outcome_col].values - ussr_pred
    return {
        "beta0": beta0, "beta1": beta1, "sigma": sigma,
        "n_non": len(non), "n_ussr": len(ussr),
        "ussr_mean_resid": float(np.mean(ussr_resid)),
        "ussr_resid_bias_std": float(np.mean(ussr_resid) / sigma),
        "n_countries": n_u,
    }


def main():
    p = build_panel()
    print(f"Panel (B-L yr_sch, contemporaneous, "
          f"years {YEARS}): {len(p)} obs")
    print(f"  non-USSR: {(~p['is_ussr']).sum()} obs "
          f"({p.loc[~p['is_ussr'], 'country'].nunique()} countries)")
    print(f"  USSR:     {p['is_ussr'].sum()} obs "
          f"({p.loc[p['is_ussr'], 'country'].nunique()} countries)")
    print()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    r1 = fit_and_plot(ax1, p, "le_t", "Life expectancy",
                      LE_THRESHOLD)
    r2 = fit_and_plot(ax2, p, "tfr_t", "TFR (children per woman)",
                      TFR_THRESHOLD)

    ax1.set_title(
        f"LE(t) vs yrs-schooling(t) — USSR mean residual: "
        f"{r1['ussr_mean_resid']:+.1f} yrs "
        f"({r1['ussr_resid_bias_std']:+.2f} SDs)",
        fontsize=11, fontweight="bold",
    )
    ax2.set_title(
        f"TFR(t) vs yrs-schooling(t) — USSR mean residual: "
        f"{r2['ussr_mean_resid']:+.2f} "
        f"({r2['ussr_resid_bias_std']:+.2f} SDs)",
        fontsize=11, fontweight="bold",
    )
    fig.suptitle(
        f"USSR education anomaly — Barro-Lee mean years of schooling, "
        f"1960–2010 trajectories\n"
        f"Global fit trained on non-USSR countries only; "
        f"each USSR country (n={r1['n_countries']}) shown as a "
        f"connected trajectory",
        fontsize=12, fontweight="bold", y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "ussr_anomaly_barrolee.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")

    print()
    print(f"LE(t) ~ yrs_sch(t): β = {r1['beta1']:.3f} yrs LE per yr "
          f"(n={r1['n_non']} non-USSR)")
    print(f"  USSR (n={r1['n_ussr']}) mean residual = "
          f"{r1['ussr_mean_resid']:+.2f} yrs = "
          f"{r1['ussr_resid_bias_std']:+.2f} SDs")
    print(f"TFR(t) ~ yrs_sch(t): β = {r2['beta1']:.4f} TFR per yr "
          f"(n={r2['n_non']} non-USSR)")
    print(f"  USSR (n={r2['n_ussr']}) mean residual = "
          f"{r2['ussr_mean_resid']:+.3f} = "
          f"{r2['ussr_resid_bias_std']:+.2f} SDs")

    # Total Barro-Lee panel countries at age 15-24 (cited in §appendix-robust
    # alternative-measure table footnote)
    bl_age15 = pd.read_csv(os.path.join(DATA, "barro_lee_v3.csv"))
    bl_age15 = bl_age15[(bl_age15['agefrom'] == 15) &
                        (bl_age15['ageto'] == 24)]
    bl_15_24_countries = int(bl_age15['country'].nunique())

    write_checkin("ussr_anomaly_barrolee.json", {
        "numbers": {
            "n_ussr_countries": r1["n_countries"],
            "n_non_ussr_obs": r1["n_non"],
            "n_ussr_obs": r1["n_ussr"],
            "bl_age_15_24_n_countries": bl_15_24_countries,
            # LE panel
            "le_beta_per_yr": round(r1["beta1"], 3),
            "le_sigma_global": round(r1["sigma"], 2),
            "le_ussr_mean_resid_yrs": round(r1["ussr_mean_resid"], 2),
            "le_ussr_bias_sds": round(r1["ussr_resid_bias_std"], 2),
            # TFR panel
            "tfr_beta_per_yr": round(r2["beta1"], 4),
            "tfr_sigma_global": round(r2["sigma"], 3),
            "tfr_ussr_mean_resid": round(r2["ussr_mean_resid"], 3),
            "tfr_ussr_bias_sds": round(r2["ussr_resid_bias_std"], 2),
        },
    }, script_path="scripts/ussr_anomaly_barrolee.py")


if __name__ == "__main__":
    main()
