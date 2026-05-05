"""
ussr_le_residual_by_year.py

Decompose USSR LE and TFR residuals by year to distinguish two
hypotheses about the LE anomaly:

H1 (1990s-crisis theory): LE residual spiked in 1990s transition from
    alcohol / tobacco / collapse of primary care. Implies a narrow
    peak around 2000 with small residuals before and after.

H2 (inflated-cohort-still-alive theory): older USSR cohorts had less
    real education than their era's inflated reporting suggested,
    and their mortality reflects real-not-reported schooling. As
    those cohorts die off, the LE residual shrinks. Implies a broad
    peak 1970-2000 and gradual decline 2000-2020 as the cohorts
    aged out of the mortality distribution.

The two outcomes test H2 directly:
  - LE: integrated across all living cohorts → affected by
    inflated-past cohorts still alive. H2 predicts large persistent
    residual that decays slowly as cohorts die.
  - TFR: decided by today's young adults → reflects TODAY's real
    education. Under Barro-Lee (which corrects current-cohort
    reporting), H2 predicts small TFR residuals at all years.

If H2 is correct: LE residual large under B-L and persistent; TFR
residual small under B-L and stable. That is the signature of
"measurement matters for currently-decided outcomes (TFR), doesn't
matter for currently-dying cohorts (LE, which reflects reality)."
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
    REPO_ROOT, PROC, DATA, load_wide_indicator, REGIONS, write_checkin,
)

OUT = os.path.join(REPO_ROOT, "paper", "figures",
                   "ussr_residual_by_year.png")

YEARS = [1960, 1970, 1980, 1990, 2000, 2010, 2020]

USSR_WCDE = {
    "russian federation", "ukraine", "belarus",
    "estonia", "latvia", "lithuania",
    "kazakhstan", "uzbekistan", "turkmenistan",
    "kyrgyz republic", "tajikistan",
    "azerbaijan", "armenia", "georgia",
    "moldova",
}
USSR_BL = {
    "Russian Federation", "Ukraine", "Estonia", "Latvia", "Lithuania",
    "Kazakhstan", "Kyrgyzstan", "Tajikistan", "Armenia",
    "Republic of Moldova",
}

WCDE_TO_WDI = {
    "republic of korea": "korea, rep.",
    "iran (islamic republic of)": "iran",
    "viet nam": "viet nam",
    "united kingdom of great britain and northern ireland":
        "united kingdom",
    "united states of america": "united states",
    "turkey": "turkiye",
}
BL_TO_WDI = {
    "Russian Federation": "russian federation",
    "Iran (Islamic Republic of)": "iran",
    "Turkey": "turkiye",
    "Republic of Korea": "korea, rep.",
    "Kyrgyzstan": "kyrgyz republic",
    "Republic of Moldova": "moldova",
    "Slovakia": "slovak republic",
    "Viet Nam": "viet nam",
    "Czechia": "czech republic",
    "Czech Republic": "czech republic",
}


def _residuals_wcde(outcome_file):
    lsec = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"),
                       index_col="country")
    lsec.columns = lsec.columns.astype(int)
    lsec.index = [s.lower() for s in lsec.index]
    out = load_wide_indicator(outcome_file)
    rows_by_year = {}
    for t in YEARS:
        if t not in lsec.columns or str(t) not in out.columns:
            continue
        xs, ys, is_u = [], [], []
        for c in lsec.index:
            if c in [r.lower() for r in REGIONS]:
                continue
            wdi = WCDE_TO_WDI.get(c, c)
            if wdi not in out.index:
                continue
            lv = lsec.loc[c, t]
            ov = out.loc[wdi, str(t)]
            if pd.isna(lv) or pd.isna(ov):
                continue
            xs.append(float(lv))
            ys.append(float(ov))
            is_u.append(c in USSR_WCDE)
        xs, ys, is_u = np.array(xs), np.array(ys), np.array(is_u)
        non_mask = ~is_u
        beta1, beta0 = np.polyfit(xs[non_mask], ys[non_mask], 1)
        sigma = float(np.std(ys[non_mask] -
                             (beta0 + beta1 * xs[non_mask]), ddof=2))
        resid_u = ys[is_u] - (beta0 + beta1 * xs[is_u])
        rows_by_year[t] = {
            "mean_resid": float(np.mean(resid_u)),
            "se_resid": float(np.std(resid_u, ddof=1) /
                              np.sqrt(len(resid_u))),
            "sigma_global": sigma,
            "n_ussr": int(is_u.sum()),
        }
    return rows_by_year


def _residuals_bl(outcome_file):
    bl = pd.read_csv(os.path.join(DATA, "barro_lee_v3.csv"))
    bl = bl[bl['agefrom'] == 25].copy()
    yrs = bl.pivot_table(index='country', columns='year',
                         values='yr_sch')
    out = load_wide_indicator(outcome_file)
    rows = {}
    for t in [1960, 1970, 1980, 1990, 2000, 2010]:
        if t not in yrs.columns or str(t) not in out.columns:
            continue
        xs, ys, is_u = [], [], []
        for c in yrs.index:
            wdi = BL_TO_WDI.get(c, c.lower())
            if wdi not in out.index:
                continue
            edu = yrs.loc[c, t]
            ov = out.loc[wdi, str(t)]
            if pd.isna(edu) or pd.isna(ov):
                continue
            xs.append(float(edu))
            ys.append(float(ov))
            is_u.append(c in USSR_BL)
        xs, ys, is_u = np.array(xs), np.array(ys), np.array(is_u)
        non_mask = ~is_u
        beta1, beta0 = np.polyfit(xs[non_mask], ys[non_mask], 1)
        sigma = float(np.std(ys[non_mask] -
                             (beta0 + beta1 * xs[non_mask]), ddof=2))
        resid_u = ys[is_u] - (beta0 + beta1 * xs[is_u])
        rows[t] = {
            "mean_resid": float(np.mean(resid_u)),
            "se_resid": float(np.std(resid_u, ddof=1) /
                              np.sqrt(len(resid_u))),
            "sigma_global": sigma,
            "n_ussr": int(is_u.sum()),
        }
    return rows


def compute_le_residuals_wcde():
    return _residuals_wcde("life_expectancy_years.csv")


def compute_le_residuals_bl():
    return _residuals_bl("life_expectancy_years.csv")


def compute_tfr_residuals_wcde():
    return _residuals_wcde(
        "children_per_woman_total_fertility.csv")


def compute_tfr_residuals_bl():
    return _residuals_bl("children_per_woman_total_fertility.csv")


def compute_logu5mr_residuals_wcde():
    # Wrap the outcome file loader so U5MR is log-transformed before
    # regression — mortality is lognormal-ish globally.
    return _residuals_wcde_log("child_mortality_u5.csv")


def compute_logu5mr_residuals_bl():
    return _residuals_bl_log("child_mortality_u5.csv")


def _residuals_wcde_log(outcome_file):
    lsec = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"),
                       index_col="country")
    lsec.columns = lsec.columns.astype(int)
    lsec.index = [s.lower() for s in lsec.index]
    out = load_wide_indicator(outcome_file)
    rows_by_year = {}
    for t in YEARS:
        if t not in lsec.columns or str(t) not in out.columns:
            continue
        xs, ys, is_u = [], [], []
        for c in lsec.index:
            if c in [r.lower() for r in REGIONS]:
                continue
            wdi = WCDE_TO_WDI.get(c, c)
            if wdi not in out.index:
                continue
            lv = lsec.loc[c, t]
            ov = out.loc[wdi, str(t)]
            if pd.isna(lv) or pd.isna(ov) or ov <= 0:
                continue
            xs.append(float(lv))
            ys.append(float(np.log(ov)))
            is_u.append(c in USSR_WCDE)
        xs, ys, is_u = np.array(xs), np.array(ys), np.array(is_u)
        non_mask = ~is_u
        beta1, beta0 = np.polyfit(xs[non_mask], ys[non_mask], 1)
        sigma = float(np.std(ys[non_mask] -
                             (beta0 + beta1 * xs[non_mask]), ddof=2))
        resid_u = ys[is_u] - (beta0 + beta1 * xs[is_u])
        rows_by_year[t] = {
            "mean_resid": float(np.mean(resid_u)),
            "se_resid": float(np.std(resid_u, ddof=1) /
                              np.sqrt(len(resid_u))),
            "sigma_global": sigma,
            "n_ussr": int(is_u.sum()),
        }
    return rows_by_year


def _residuals_bl_log(outcome_file):
    bl = pd.read_csv(os.path.join(DATA, "barro_lee_v3.csv"))
    bl = bl[bl['agefrom'] == 25].copy()
    yrs = bl.pivot_table(index='country', columns='year',
                         values='yr_sch')
    out = load_wide_indicator(outcome_file)
    rows = {}
    for t in [1960, 1970, 1980, 1990, 2000, 2010]:
        if t not in yrs.columns or str(t) not in out.columns:
            continue
        xs, ys, is_u = [], [], []
        for c in yrs.index:
            wdi = BL_TO_WDI.get(c, c.lower())
            if wdi not in out.index:
                continue
            edu = yrs.loc[c, t]
            ov = out.loc[wdi, str(t)]
            if pd.isna(edu) or pd.isna(ov) or ov <= 0:
                continue
            xs.append(float(edu))
            ys.append(float(np.log(ov)))
            is_u.append(c in USSR_BL)
        xs, ys, is_u = np.array(xs), np.array(ys), np.array(is_u)
        non_mask = ~is_u
        beta1, beta0 = np.polyfit(xs[non_mask], ys[non_mask], 1)
        sigma = float(np.std(ys[non_mask] -
                             (beta0 + beta1 * xs[non_mask]), ddof=2))
        resid_u = ys[is_u] - (beta0 + beta1 * xs[is_u])
        rows[t] = {
            "mean_resid": float(np.mean(resid_u)),
            "se_resid": float(np.std(resid_u, ddof=1) /
                              np.sqrt(len(resid_u))),
            "sigma_global": sigma,
            "n_ussr": int(is_u.sum()),
        }
    return rows


def _print_table(label, wcde, blee):
    print(f"USSR mean {label} residual vs non-USSR fit — by year")
    print(f"  {'Year':<6}  {'WCDE resid':>12}  {'WCDE SE':>9}  "
          f"{'B-L resid':>12}  {'B-L SE':>9}")
    for t in YEARS:
        w = wcde.get(t)
        b = blee.get(t)
        wr = f"{w['mean_resid']:+7.3f}" if w else "   --"
        ws = f"{w['se_resid']:7.3f}" if w else "   --"
        br = f"{b['mean_resid']:+7.3f}" if b else "   --"
        bs = f"{b['se_resid']:7.3f}" if b else "   --"
        print(f"  {t:<6}  {wr:>12}  {ws:>9}  {br:>12}  {bs:>9}")
    print()


def _plot_panel(ax, wcde, blee, ylabel, title):
    yrs_w = [t for t in YEARS if t in wcde]
    vals_w = [wcde[t]["mean_resid"] for t in yrs_w]
    ses_w = [wcde[t]["se_resid"] for t in yrs_w]
    yrs_b = [t for t in YEARS if t in blee]
    vals_b = [blee[t]["mean_resid"] for t in yrs_b]
    ses_b = [blee[t]["se_resid"] for t in yrs_b]
    ax.errorbar(yrs_w, vals_w, yerr=ses_w, marker='o', markersize=8,
                linewidth=2, color='#c0392b',
                label='WCDE (lsec)', capsize=4)
    ax.errorbar(yrs_b, vals_b, yerr=ses_b, marker='s', markersize=7,
                linewidth=2, color='#1f4e79',
                label='Barro-Lee (yrs schooling)', capsize=4)
    ax.axhline(0, color='#222', linewidth=0.8, linestyle='--',
               alpha=0.6)
    ax.axvspan(1991, 1998, alpha=0.10, color='gray',
               label='Soviet transition (1991–98)')
    ax.set_xlabel("Year t", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='best', frameon=False)


def main():
    le_wcde = compute_le_residuals_wcde()
    le_bl = compute_le_residuals_bl()
    tfr_wcde = compute_tfr_residuals_wcde()
    tfr_bl = compute_tfr_residuals_bl()
    u5_wcde = compute_logu5mr_residuals_wcde()
    u5_bl = compute_logu5mr_residuals_bl()

    _print_table("LE", le_wcde, le_bl)
    _print_table("TFR", tfr_wcde, tfr_bl)
    _print_table("log U5MR", u5_wcde, u5_bl)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    _plot_panel(
        ax1, le_wcde, le_bl,
        "USSR mean LE residual (years below non-USSR fit)",
        "LE (integrated over cohorts)\n"
        "Persistent large negative residual; peaks 1990–2000;\n"
        "decays 2000→2020 as inflated cohorts die off.",
    )
    _plot_panel(
        ax2, tfr_wcde, tfr_bl,
        "USSR mean TFR residual (children above non-USSR fit)",
        "TFR (today's decision)\n"
        "Small under B-L across all years.\n"
        "Current cohort → current real education.",
    )
    _plot_panel(
        ax3, u5_wcde, u5_bl,
        "USSR mean log U5MR residual (log per-1,000 above fit)",
        "U5MR (young mother's current behaviour)\n"
        "Key diagnostic: large under B-L → B-L also inflated;\n"
        "small under B-L → B-L current-cohort is real.",
    )
    fig.suptitle(
        "USSR education anomaly by year — three outcomes, diagnostic "
        "of whether B-L's current-cohort correction is itself adequate",
        fontsize=12, fontweight='bold', y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches='tight')
    print(f"Saved: {OUT}")

    def _pack(byyr, keys=("mean_resid", "se_resid", "n_ussr")):
        return {
            str(y): {k: round(float(byyr[y][k]), 3) for k in keys}
            for y in sorted(byyr)
        }

    numbers = {}
    for out, lbl in [("le", "LE"), ("tfr", "TFR"), ("u5log", "log_U5MR")]:
        if out == "le":
            w, b = le_wcde, le_bl
        elif out == "tfr":
            w, b = tfr_wcde, tfr_bl
        else:
            w, b = u5_wcde, u5_bl
        numbers[f"{out}_wcde_resid_by_year"] = _pack(w)
        numbers[f"{out}_bl_resid_by_year"] = _pack(b)

    # Scalars the paper cites directly
    scalars = {}
    for t in (1960, 1970, 1980, 1990, 2000, 2010):
        if t in le_bl:
            scalars[f"le_bl_resid_{t}"] = round(le_bl[t]["mean_resid"], 2)
        if t in tfr_bl:
            scalars[f"tfr_bl_resid_{t}"] = round(tfr_bl[t]["mean_resid"], 3)
        if t in u5_bl:
            scalars[f"u5log_bl_resid_{t}"] = round(u5_bl[t]["mean_resid"], 2)
    numbers.update(scalars)

    write_checkin("ussr_residual_by_year.json", {
        "numbers": numbers,
    }, script_path="scripts/ussr_residual_by_year.py")


if __name__ == "__main__":
    main()
