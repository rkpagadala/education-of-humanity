"""
ussr_per_country_residuals.py

Per-country breakdown of the USSR education anomaly.

The paper excludes 15 USSR countries because their reported education
numbers (WCDE lower-secondary completion, age 20-24, both sexes) sit systematically above
what the non-USSR global fit predicts for their LE / TFR / U5MR.

This script computes each of the 15 countries' mean residual
(in raw units AND in σ units) against a non-USSR global fit, across
all years 1960–2020, for three outcomes:

  - LE(t)        — life expectancy
  - TFR(t)       — total fertility rate
  - log U5MR(t)  — under-5 mortality, log scale

A country whose residual is small is consistent with the global fit:
its reported education matches the phenotype the rest of the world
shows at that education level. A country with a large residual
deviates from the global pattern.

We do this for both WCDE lower-secondary completion AND Barro-Lee years-of-schooling, so
we can see which countries are inflated under WCDE only, B-L only,
or both.

Output: ranked tables + a checkin JSON.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import (
    REPO_ROOT, PROC, DATA, load_wide_indicator, REGIONS, write_checkin,
)

YEARS = [1960, 1970, 1980, 1990, 2000, 2010, 2020]
BL_YEARS = [1960, 1970, 1980, 1990, 2000, 2010]

USSR_WCDE = [
    "russian federation", "ukraine", "belarus",
    "estonia", "latvia", "lithuania",
    "kazakhstan", "uzbekistan", "turkmenistan",
    "kyrgyzstan", "tajikistan",
    "azerbaijan", "armenia", "georgia",
    "republic of moldova",
]
# Barro-Lee covers 10 of the 15
USSR_BL = [
    "Russian Federation", "Ukraine", "Estonia", "Latvia", "Lithuania",
    "Kazakhstan", "Kyrgyzstan", "Tajikistan", "Armenia",
    "Republic of Moldova",
]
BL_TO_DISPLAY = {
    "Russian Federation": "russian federation",
    "Ukraine": "ukraine",
    "Estonia": "estonia",
    "Latvia": "latvia",
    "Lithuania": "lithuania",
    "Kazakhstan": "kazakhstan",
    "Kyrgyzstan": "kyrgyzstan",
    "Tajikistan": "tajikistan",
    "Armenia": "armenia",
    "Republic of Moldova": "republic of moldova",
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
    "kyrgyzstan": "kyrgyz republic",
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


def _load_outcome(name, log=False):
    df = load_wide_indicator(name)
    if log:
        df = df.where(df > 0).apply(np.log)
    return df


def _resolve(df, name, aliases):
    """Try a list of candidate names against df.index, return first hit."""
    if name in df.index:
        return name
    for a in aliases:
        if a in df.index:
            return a
    return None


# Per-country fallbacks: file-specific names that differ from WDI
# convention. e.g. child_mortality_u5.csv uses "russia" not
# "russian federation".
EXTRA_ALIASES = {
    "russian federation": ["russia"],
}


def per_country_wcde(outcome_file, log=False):
    """For each USSR country and each year, compute the residual vs
    the non-USSR global fit, using WCDE lower-secondary completion as the education x."""
    lsec = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"),
                       index_col="country")
    lsec.columns = lsec.columns.astype(int)
    lsec.index = [s.lower() for s in lsec.index]
    out = _load_outcome(outcome_file, log=log)

    rows = []
    for t in YEARS:
        if t not in lsec.columns or str(t) not in out.columns:
            continue
        xs, ys, names = [], [], []
        for c in lsec.index:
            if c in [r.lower() for r in REGIONS]:
                continue
            wdi_guess = WCDE_TO_WDI.get(c, c)
            wdi = _resolve(out, wdi_guess,
                           EXTRA_ALIASES.get(wdi_guess, []))
            if wdi is None:
                continue
            lv = lsec.loc[c, t]
            ov = out.loc[wdi, str(t)]
            if pd.isna(lv) or pd.isna(ov):
                continue
            xs.append(float(lv))
            ys.append(float(ov))
            names.append(c)
        xs, ys = np.array(xs), np.array(ys)
        names = np.array(names)
        is_u = np.array([n in USSR_WCDE for n in names])
        non_mask = ~is_u
        if non_mask.sum() < 10 or is_u.sum() == 0:
            continue
        beta1, beta0 = np.polyfit(xs[non_mask], ys[non_mask], 1)
        sigma = float(np.std(ys[non_mask] -
                             (beta0 + beta1 * xs[non_mask]), ddof=2))
        for i in np.where(is_u)[0]:
            pred = beta0 + beta1 * xs[i]
            rows.append({
                "country": names[i],
                "year": t,
                "edu": float(xs[i]),
                "outcome": float(ys[i]),
                "pred": float(pred),
                "resid": float(ys[i] - pred),
                "sigma": sigma,
                "resid_sd": float((ys[i] - pred) / sigma),
            })
    return pd.DataFrame(rows)


def per_country_bl(outcome_file, log=False):
    """Per-country residuals using Barro-Lee yrs-of-schooling, age 25+."""
    bl = pd.read_csv(os.path.join(DATA, "barro_lee_v3.csv"))
    bl = bl[bl["agefrom"] == 25].copy()
    yrs = bl.pivot_table(index="country", columns="year",
                         values="yr_sch")
    out = _load_outcome(outcome_file, log=log)

    rows = []
    for t in BL_YEARS:
        if t not in yrs.columns or str(t) not in out.columns:
            continue
        xs, ys, names = [], [], []
        for c in yrs.index:
            wdi_guess = BL_TO_WDI.get(c, c.lower())
            wdi = _resolve(out, wdi_guess,
                           EXTRA_ALIASES.get(wdi_guess, []))
            if wdi is None:
                continue
            edu = yrs.loc[c, t]
            ov = out.loc[wdi, str(t)]
            if pd.isna(edu) or pd.isna(ov):
                continue
            xs.append(float(edu))
            ys.append(float(ov))
            names.append(c)
        xs, ys = np.array(xs), np.array(ys)
        names = np.array(names)
        is_u = np.array([n in USSR_BL for n in names])
        non_mask = ~is_u
        if non_mask.sum() < 10 or is_u.sum() == 0:
            continue
        beta1, beta0 = np.polyfit(xs[non_mask], ys[non_mask], 1)
        sigma = float(np.std(ys[non_mask] -
                             (beta0 + beta1 * xs[non_mask]), ddof=2))
        for i in np.where(is_u)[0]:
            pred = beta0 + beta1 * xs[i]
            display = BL_TO_DISPLAY.get(names[i], names[i].lower())
            rows.append({
                "country": display,
                "year": t,
                "edu": float(xs[i]),
                "outcome": float(ys[i]),
                "pred": float(pred),
                "resid": float(ys[i] - pred),
                "sigma": sigma,
                "resid_sd": float((ys[i] - pred) / sigma),
            })
    return pd.DataFrame(rows)


def _summarize(df, label):
    """Mean residual per country across years; ranked by |resid_sd|."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("country").agg(
        mean_resid=("resid", "mean"),
        mean_resid_sd=("resid_sd", "mean"),
        abs_resid_sd=("resid_sd", lambda s: float(np.mean(np.abs(s)))),
        n=("year", "count"),
    ).reset_index()
    g["outcome"] = label
    return g.sort_values("abs_resid_sd")


def _print_country_table(label, le, tfr, u5):
    """Side-by-side country ranking across the three outcomes."""
    countries = sorted(set(le["country"]) | set(tfr["country"]) |
                       set(u5["country"]))
    print(f"\n=== Per-country residuals — {label} ===")
    print(f"  Sign convention: LE residual <0 = country lives shorter "
          f"than its education predicts (i.e., reported edu inflated).")
    print(f"  TFR residual >0 = country has more children than its "
          f"education predicts (reported edu inflated).")
    print(f"  log U5MR residual >0 = country has higher U5MR than its "
          f"education predicts (reported edu inflated).")
    print()
    hdr = f"  {'country':<22} | {'LE resid':>9} ({'σ':>5}) | " \
          f"{'TFR resid':>9} ({'σ':>5}) | {'lnU5 resid':>10} ({'σ':>5}) | " \
          f"{'LE+U5 |σ|':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = []
    for c in countries:
        le_row = le[le["country"] == c]
        tfr_row = tfr[tfr["country"] == c]
        u5_row = u5[u5["country"] == c]
        le_r = le_row["mean_resid"].iloc[0] if not le_row.empty else np.nan
        le_s = le_row["mean_resid_sd"].iloc[0] if not le_row.empty else np.nan
        tfr_r = tfr_row["mean_resid"].iloc[0] if not tfr_row.empty else np.nan
        tfr_s = tfr_row["mean_resid_sd"].iloc[0] if not tfr_row.empty else np.nan
        u5_r = u5_row["mean_resid"].iloc[0] if not u5_row.empty else np.nan
        u5_s = u5_row["mean_resid_sd"].iloc[0] if not u5_row.empty else np.nan
        sigmas = [abs(s) for s in (le_s, tfr_s, u5_s) if not np.isnan(s)]
        mean_abs = float(np.mean(sigmas)) if sigmas else np.nan
        rows.append({
            "country": c, "le_r": le_r, "le_s": le_s,
            "tfr_r": tfr_r, "tfr_s": tfr_s,
            "u5_r": u5_r, "u5_s": u5_s, "mean_abs": mean_abs,
        })
    # Compute LE+U5 only mean (ignore TFR — outcome that least
    # reflects measurement; LE and U5MR are the better indicators)
    for r in rows:
        sigmas = [abs(s) for s in (r["le_s"], r["u5_s"])
                  if not np.isnan(s)]
        r["mean_abs_le_u5"] = float(np.mean(sigmas)) if sigmas else np.nan

    rows.sort(key=lambda r: (np.isnan(r["mean_abs_le_u5"]),
                             r["mean_abs_le_u5"]))
    for r in rows:
        def _fmt(v, w, prec):
            if np.isnan(v):
                return f"{'  --':>{w}}"
            return f"{v:+{w}.{prec}f}"
        print(f"  {r['country']:<22} | "
              f"{_fmt(r['le_r'], 9, 2)} ({_fmt(r['le_s'], 5, 2)}) | "
              f"{_fmt(r['tfr_r'], 9, 3)} ({_fmt(r['tfr_s'], 5, 2)}) | "
              f"{_fmt(r['u5_r'], 10, 2)} ({_fmt(r['u5_s'], 5, 2)}) | "
              f"{_fmt(r['mean_abs_le_u5'], 8, 2)}")
    return rows


def main():
    le_w = per_country_wcde("life_expectancy_years.csv")
    tfr_w = per_country_wcde("children_per_woman_total_fertility.csv")
    u5_w = per_country_wcde("child_mortality_u5.csv", log=True)

    le_b = per_country_bl("life_expectancy_years.csv")
    tfr_b = per_country_bl("children_per_woman_total_fertility.csv")
    u5_b = per_country_bl("child_mortality_u5.csv", log=True)

    le_w_s = _summarize(le_w, "LE")
    tfr_w_s = _summarize(tfr_w, "TFR")
    u5_w_s = _summarize(u5_w, "logU5")
    le_b_s = _summarize(le_b, "LE")
    tfr_b_s = _summarize(tfr_b, "TFR")
    u5_b_s = _summarize(u5_b, "logU5")

    print(f"USSR per-country residuals — fits trained on non-USSR only")
    print(f"WCDE: 15 countries × up to 7 years (1960-2020)")
    print(f"B-L:  10 countries × up to 6 years (1960-2010)")

    rows_w = _print_country_table("WCDE lsec age 20-24",
                                  le_w_s, tfr_w_s, u5_w_s)
    rows_b = _print_country_table("Barro-Lee yrs schooling 25+",
                                  le_b_s, tfr_b_s, u5_b_s)

    # LE + U5MR year-by-year detail for the cleanest tier + Russia.
    # σ uses each year's global residual SD.
    detail_countries = [
        "lithuania", "belarus", "estonia", "latvia",
        "ukraine", "russian federation",
    ]
    print(f"\n=== LE & log U5MR year-by-year (WCDE) — clean tier + Russia ===")
    print(f"  Cell shows: actual / predicted / resid (σ)")
    for c in detail_countries:
        print(f"\n  {c}")
        le_c = le_w[le_w["country"] == c].set_index("year")
        u5_c = u5_w[u5_w["country"] == c].set_index("year")
        print(f"    {'year':<6} {'edu':>6} | "
              f"{'LE actual':>10} {'pred':>7} {'resid':>10}     "
              f"{'U5':>5}/1k {'pred':>7} {'lnU5 resid':>11}")
        for t in YEARS:
            edu = (le_c.loc[t, "edu"] if t in le_c.index
                   else u5_c.loc[t, "edu"] if t in u5_c.index
                   else None)
            if edu is None:
                continue
            le_actual = le_c.loc[t, "outcome"] if t in le_c.index else None
            le_pred = le_c.loc[t, "pred"] if t in le_c.index else None
            le_r = le_c.loc[t, "resid"] if t in le_c.index else None
            le_sd = le_c.loc[t, "resid_sd"] if t in le_c.index else None
            u5_actual = (np.exp(u5_c.loc[t, "outcome"])
                         if t in u5_c.index else None)
            u5_pred = (np.exp(u5_c.loc[t, "pred"])
                       if t in u5_c.index else None)
            u5_r = u5_c.loc[t, "resid"] if t in u5_c.index else None
            u5_sd = u5_c.loc[t, "resid_sd"] if t in u5_c.index else None
            le_str = (f"{le_actual:>10.1f} {le_pred:>7.1f} "
                      f"{le_r:>+6.2f}({le_sd:+.2f}σ)"
                      if le_actual is not None else
                      f"{'  --':>10} {'  --':>7} {'  --':>10}     ")
            u5_str = (f"{u5_actual:>5.1f} {u5_pred:>7.1f} "
                      f"{u5_r:>+6.2f}({u5_sd:+.2f}σ)"
                      if u5_actual is not None else
                      f"{'  --':>5} {'  --':>7} {'  --':>11}")
            print(f"    {t:<6} {edu:>6.1f} | {le_str}     {u5_str}")

    # Pack to checkin JSON
    def _to_record(rows):
        return {
            r["country"]: {
                "le_resid": round(float(r["le_r"]), 3) if not np.isnan(r["le_r"]) else None,
                "le_resid_sd": round(float(r["le_s"]), 3) if not np.isnan(r["le_s"]) else None,
                "tfr_resid": round(float(r["tfr_r"]), 4) if not np.isnan(r["tfr_r"]) else None,
                "tfr_resid_sd": round(float(r["tfr_s"]), 3) if not np.isnan(r["tfr_s"]) else None,
                "u5log_resid": round(float(r["u5_r"]), 3) if not np.isnan(r["u5_r"]) else None,
                "u5log_resid_sd": round(float(r["u5_s"]), 3) if not np.isnan(r["u5_s"]) else None,
                "mean_abs_le_u5_sd": round(float(r["mean_abs_le_u5"]), 3) if not np.isnan(r["mean_abs_le_u5"]) else None,
            }
            for r in rows
        }

    write_checkin("ussr_per_country_residuals.json", {
        "wcde": _to_record(rows_w),
        "barro_lee": _to_record(rows_b),
        "wcde_ranking_low_to_high_le_u5": [r["country"] for r in rows_w],
        "barro_lee_ranking_low_to_high_le_u5": [r["country"] for r in rows_b],
    }, script_path="scripts/ussr_per_country_residuals.py")


if __name__ == "__main__":
    main()
