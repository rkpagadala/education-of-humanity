"""
hlo_vs_education_lag_sweep.py

Continuous lag sweep: regress HLO_today (country-mean, roughly
centered at 2010) on lower-secondary completion at year 2010 - L,
for L ∈ [0, 60] in 5-year steps.

Question: what lag best predicts today's cognitive skill? If the peak
is at L ≈ 0, Hanushek's "school quality" is about today's schools.
If peak at L ≈ 25, it's parent-generation PT. If peak at L ≈ 50, it's
grandparent PT. If the curve is flat or broad, HLO reflects
cumulative multi-generational education stock.

Secondary sweep with primary completion (where WCDE goes back to
1950 cleanly, we can lag further — to L ≈ 60, i.e., great-grandparent
primary).

USSR republics excluded.
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
from _shared import DATA, PROC, REPO_ROOT, REGIONS, write_checkin

HLO_CENTER = 2010  # representative year for HLO country-mean

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
    return {NAME_TO_WDI.get(c, c.lower()): float(v)
            for c, v in sec.groupby('country')['hlo'].mean().items()}


def load_wcde_wide(filename):
    df = pd.read_csv(os.path.join(PROC, filename), index_col="country")
    df.columns = df.columns.astype(int)
    df.index = [s.lower() for s in df.index]
    df = df[~df.index.isin([r.lower() for r in REGIONS])]
    df.index = [WCDE_TO_WDI.get(c, c) for c in df.index]
    return df


def sweep(hlo_dict, edu_df, label, lags):
    """For each lag L, fit HLO ~ edu(HLO_CENTER - L). Report R²/β/n."""
    results = []
    for L in lags:
        year = HLO_CENTER - L
        if year not in edu_df.columns:
            continue
        xs, ys = [], []
        for country, h in hlo_dict.items():
            if country in USSR_LC:
                continue
            if country not in edu_df.index:
                continue
            v = edu_df.loc[country, year]
            if pd.isna(v):
                continue
            xs.append(float(v))
            ys.append(float(h))
        if len(xs) < 20:
            continue
        x = np.array(xs)
        y = np.array(ys)
        beta1, beta0 = np.polyfit(x, y, 1)
        ss_res = np.sum((y - (beta0 + beta1 * x)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        results.append({
            "lag": L, "year": year, "n": len(x),
            "beta": beta1, "r2": r2,
        })
    return pd.DataFrame(results)


def _r2_for_arrays(x, y):
    if len(x) < 5:
        return np.nan
    beta1, beta0 = np.polyfit(x, y, 1)
    ss_res = np.sum((y - (beta0 + beta1 * x)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def bootstrap_ci(hlo_dict, edu_df, lags, n_boot=2000, seed=42):
    """
    Country-resampling bootstrap. Returns:
      - per-lag 95% CI on R²
      - joint draws (n_boot × len(lags)) for difference CIs
        (peak-vs-lag-0)

    Same country sample is used across all lags within a single draw,
    so CIs on R² differences correctly account for the joint
    cross-country covariance.
    """
    countries = [c for c in hlo_dict
                 if c not in USSR_LC and c in edu_df.index]
    rng = np.random.default_rng(seed)

    # Pre-build per-country (hlo, edu-at-each-lag) panel.
    h_arr = np.array([hlo_dict[c] for c in countries])
    edu_panel = np.full((len(countries), len(lags)), np.nan)
    for j, L in enumerate(lags):
        year = HLO_CENTER - L
        if year not in edu_df.columns:
            continue
        for i, c in enumerate(countries):
            v = edu_df.loc[c, year]
            if not pd.isna(v):
                edu_panel[i, j] = float(v)

    boot = np.full((n_boot, len(lags)), np.nan)
    n = len(countries)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        h_b = h_arr[idx]
        edu_b = edu_panel[idx]
        for j in range(len(lags)):
            mask = ~np.isnan(edu_b[:, j])
            if mask.sum() < 20:
                continue
            boot[b, j] = _r2_for_arrays(edu_b[mask, j], h_b[mask])

    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    return lo, hi, boot


def main():
    hlo = load_hlo_secondary()
    lsec = load_wcde_wide("lower_sec_both.csv")
    pri = load_wcde_wide("primary_both.csv")

    lags_lsec = list(range(0, 61, 5))     # lsec goes back to 1950
    lags_pri = list(range(0, 61, 5))      # pri goes back to 1950

    res_lsec = sweep(hlo, lsec, "lower-sec completion", lags_lsec)
    res_pri = sweep(hlo, pri, "primary completion", lags_pri)

    n_boot = 2000
    lo_lsec, hi_lsec, boot_lsec = bootstrap_ci(
        hlo, lsec, lags_lsec, n_boot=n_boot, seed=42)
    lo_pri, hi_pri, boot_pri = bootstrap_ci(
        hlo, pri, lags_pri, n_boot=n_boot, seed=43)
    res_lsec["r2_lo"] = lo_lsec
    res_lsec["r2_hi"] = hi_lsec
    res_pri["r2_lo"] = lo_pri
    res_pri["r2_hi"] = hi_pri

    print(f"HLO → lower-sec completion, lag sweep (USSR excluded, "
          f"95% CI from {n_boot} country-resample bootstraps)")
    print(f"{'Lag':>4}  {'Year':>5}  {'n':>4}  {'β':>7}  "
          f"{'R²':>6}  {'CI_lo':>6}  {'CI_hi':>6}")
    for _, r in res_lsec.iterrows():
        print(f"{int(r['lag']):>4}  {int(r['year']):>5}  "
              f"{int(r['n']):>4}  {r['beta']:>7.3f}  "
              f"{r['r2']:>6.3f}  {r['r2_lo']:>6.3f}  "
              f"{r['r2_hi']:>6.3f}")

    print()
    print(f"HLO → primary completion, lag sweep (USSR excluded, "
          f"95% CI from {n_boot} country-resample bootstraps)")
    print(f"{'Lag':>4}  {'Year':>5}  {'n':>4}  {'β':>7}  "
          f"{'R²':>6}  {'CI_lo':>6}  {'CI_hi':>6}")
    for _, r in res_pri.iterrows():
        print(f"{int(r['lag']):>4}  {int(r['year']):>5}  "
              f"{int(r['n']):>4}  {r['beta']:>7.3f}  "
              f"{r['r2']:>6.3f}  {r['r2_lo']:>6.3f}  "
              f"{r['r2_hi']:>6.3f}")

    # Joint CIs on R² differences (peak vs lag 0) — same bootstrap
    # draw used across lags, so the difference distribution accounts
    # for cross-country covariance.
    def _diff_ci(boot, lags, peak_lag, ref_lag=0):
        j_peak = lags.index(peak_lag)
        j_ref = lags.index(ref_lag)
        diffs = boot[:, j_peak] - boot[:, j_ref]
        diffs = diffs[~np.isnan(diffs)]
        return (float(np.percentile(diffs, 2.5)),
                float(np.percentile(diffs, 97.5)),
                float(np.mean(diffs > 0)))

    peak_lag_lsec = int(res_lsec.loc[res_lsec["r2"].idxmax(), "lag"])
    peak_lag_pri = int(res_pri.loc[res_pri["r2"].idxmax(), "lag"])
    diff_lo_lsec, diff_hi_lsec, p_lsec = _diff_ci(
        boot_lsec, lags_lsec, peak_lag_lsec, 0)
    diff_lo_pri, diff_hi_pri, p_pri = _diff_ci(
        boot_pri, lags_pri, peak_lag_pri, 0)

    print()
    print(f"R²(peak) − R²(lag 0), 95% CI:")
    print(f"  lower-sec: peak={peak_lag_lsec}yr, "
          f"ΔR²∈[{diff_lo_lsec:+.3f}, {diff_hi_lsec:+.3f}], "
          f"P(peak>lag 0)={p_lsec:.3f}")
    print(f"  primary:   peak={peak_lag_pri}yr, "
          f"ΔR²∈[{diff_lo_pri:+.3f}, {diff_hi_pri:+.3f}], "
          f"P(peak>lag 0)={p_pri:.3f}")

    # Identify peak lag for each
    peak_l = res_lsec.loc[res_lsec["r2"].idxmax()]
    peak_p = res_pri.loc[res_pri["r2"].idxmax()]
    print()
    print(f"Peak lag (lower-sec): L = {int(peak_l['lag'])} years "
          f"(year {int(peak_l['year'])}) → R² = {peak_l['r2']:.3f}")
    print(f"Peak lag (primary):   L = {int(peak_p['lag'])} years "
          f"(year {int(peak_p['year'])}) → R² = {peak_p['r2']:.3f}")

    # ── Plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.fill_between(res_lsec["lag"], res_lsec["r2_lo"],
                    res_lsec["r2_hi"], color="#c0392b", alpha=0.15)
    ax.fill_between(res_pri["lag"], res_pri["r2_lo"],
                    res_pri["r2_hi"], color="#1f4e79", alpha=0.15)
    ax.plot(res_lsec["lag"], res_lsec["r2"],
            marker="o", markersize=7, linewidth=2,
            color="#c0392b",
            label="HLO ~ lower-sec completion (95% CI shaded)")
    ax.plot(res_pri["lag"], res_pri["r2"],
            marker="s", markersize=7, linewidth=2,
            color="#1f4e79",
            label="HLO ~ primary completion (95% CI shaded)")

    # Shade PT-generation windows
    ax.axvspan(20, 30, alpha=0.10, color="#2ecc71",
               label="parent window (~25 yr lag)")
    ax.axvspan(45, 55, alpha=0.10, color="#f39c12",
               label="grandparent window (~50 yr lag)")

    ax.set_xlabel(f"Lag L (years before HLO test year ~{HLO_CENTER})",
                  fontsize=11)
    ax.set_ylabel("R² (variance of HLO explained)", fontsize=11)
    ax.set_title(
        "HLO today explained by past education — continuous lag sweep\n"
        "Peak lag tells us whose education the HLO test is actually "
        "measuring",
        fontsize=12, fontweight="bold",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="best", frameon=False)
    ax.set_ylim(0, max(res_lsec["r2"].max(),
                       res_pri["r2"].max()) * 1.1)
    fig.tight_layout()
    out_path = os.path.join(REPO_ROOT, "paper", "figures",
                            "hlo_vs_education_lag_sweep.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    def _pack(df):
        return {
            f"lag_{int(r['lag'])}": {
                "year": int(r["year"]),
                "n": int(r["n"]),
                "beta": round(float(r["beta"]), 3),
                "r2": round(float(r["r2"]), 3),
                "r2_ci_lo": round(float(r["r2_lo"]), 3),
                "r2_ci_hi": round(float(r["r2_hi"]), 3),
            }
            for _, r in df.iterrows()
        }

    numbers = {
        "lsec_sweep": _pack(res_lsec),
        "primary_sweep": _pack(res_pri),
        "peak_lag_lsec": int(peak_l["lag"]),
        "peak_r2_lsec": round(float(peak_l["r2"]), 3),
        "peak_lag_primary": int(peak_p["lag"]),
        "peak_r2_primary": round(float(peak_p["r2"]), 3),
        # Scalars cited in §9.6
        "r2_lsec_lag_0": round(float(res_lsec.loc[
            res_lsec["lag"] == 0, "r2"].iloc[0]), 3),
        "r2_lsec_lag_25": round(float(res_lsec.loc[
            res_lsec["lag"] == 25, "r2"].iloc[0]), 3),
        "r2_lsec_lag_50": round(float(res_lsec.loc[
            res_lsec["lag"] == 50, "r2"].iloc[0]), 3),
        "r2_lsec_lag_60": round(float(res_lsec.loc[
            res_lsec["lag"] == 60, "r2"].iloc[0]), 3)
        if (res_lsec["lag"] == 60).any() else None,
        "r2_primary_lag_0": round(float(res_pri.loc[
            res_pri["lag"] == 0, "r2"].iloc[0]), 3),
        "r2_primary_lag_25": round(float(res_pri.loc[
            res_pri["lag"] == 25, "r2"].iloc[0]), 3),
        "r2_primary_lag_60": round(float(res_pri.loc[
            res_pri["lag"] == 60, "r2"].iloc[0]), 3)
        if (res_pri["lag"] == 60).any() else None,
        # Top-level scalars for paper citation.
        "bootstrap_n": int(n_boot),
        "lsec_peak_p_pct": int(round(p_lsec * 100)),
        "primary_peak_p_pct": int(round(p_pri * 100)),
        # Bootstrap diagnostics — n_boot country-resamples, 95% CIs.
        "bootstrap": {
            "n_boot": n_boot,
            "lsec_peak_minus_lag0": {
                "peak_lag": peak_lag_lsec,
                "diff_ci_lo": round(diff_lo_lsec, 3),
                "diff_ci_hi": round(diff_hi_lsec, 3),
                "p_peak_gt_lag0": round(p_lsec, 3),
            },
            "primary_peak_minus_lag0": {
                "peak_lag": peak_lag_pri,
                "diff_ci_lo": round(diff_lo_pri, 3),
                "diff_ci_hi": round(diff_hi_pri, 3),
                "p_peak_gt_lag0": round(p_pri, 3),
            },
        },
    }

    write_checkin("hlo_lag_sweep.json", {
        "numbers": numbers,
    }, script_path="scripts/hlo_vs_education_lag_sweep.py")


if __name__ == "__main__":
    main()
