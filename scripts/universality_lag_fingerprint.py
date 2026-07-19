# =============================================================================
# PAPER REFERENCE
# Script:  scripts/universality_lag_fingerprint.py
# Paper:   "The Long Childhood: On the Convergence of Humanity"
#          Chapter "The Panel", section "Universality Across Subsamples"
#
# Purpose:
#   §Universality re-estimates the headline generational coefficient on eleven
#   subsamples and finds eleven positive, significant betas. That demonstrates
#   the *relationship* does not vanish anywhere; on its own it does not show the
#   *mechanism* is the same everywhere. A single positive coefficient is
#   consistent with heterogeneous causal pathways across regions.
#
#   This script tests the stronger claim. The mechanism's signature is not one
#   coefficient but a coordinated four-outcome pattern at outcome-specific lags
#   (per scripts/wcde/table_contemporaneous.py and §The Shape of the Response):
#
#       log GDP   at lag 0   (LAG_CONTEMPORANEOUS) : positive
#       log(TFR)  at lag 5   (LAG_TFR)             : negative
#       log(LE)   at lag 12  (LAG_LE)              : positive
#       log(U5MR) at lag 12  (LAG_CHILDREARING)    : negative
#
#   If this *fingerprint* — the four signs together, each at its canonical lag —
#   repeats inside every region, era, and income band, the universality is of
#   the mechanism, not merely of a reduced-form coefficient. Heterogeneous
#   mechanisms would not reproduce the same coordinated sign pattern at the same
#   biological lags everywhere.
#
#   The test:
#     Canonical-lag fingerprint: per subsample, all four outcomes at their
#     canonical lags. Records sign-match and significance (44 cells:
#     11 subsamples x 4 outcomes), and the across-subsample band of each
#     semi-elasticity. Not just every sign correct everywhere, but every
#     magnitude in a narrow band — the coordinated quadruple, not a lone
#     coefficient, is what heterogeneous mechanisms could not reproduce.
#
#   Why no lag-recovery test here: regressing one outcome across a lag grid and
#   asking where significance peaks does NOT recover the canonical lag even in
#   the full panel — shorter lags retain more observations and the outcomes are
#   autocorrelated, so the p-value peaks at lag 0 regardless of mechanism. The
#   lag at which each outcome's mechanism runs is established biologically and
#   in §The Shape of the Response, not by a within-panel p-value search.
#
# Subsamples (eleven, matching §Universality):
#   six World Bank regions; two child/education eras (T<1990, T>=1990);
#   three within-sample GDP terciles (GDP at the education year T).
#
# Inputs:
#   wcde/data/processed/cohort_completion_both_long.csv
#   data/gdppercapita_us_inflation_adjusted.csv
#   data/life_expectancy_years.csv
#   data/children_per_woman_total_fertility.csv
#   data/child_mortality_u5.csv
#
# Output: checkin/universality_lag_fingerprint.json
# =============================================================================
"""Mechanism-fingerprint universality test across the eleven subsamples."""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as _st

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import (PROC, REGIONS, write_checkin, load_wb,  # noqa: E402
                     NAME_MAP as _SHARED_NAME_MAP,
                     LAG_CONTEMPORANEOUS, LAG_LE, LAG_TFR, LAG_CHILDREARING)

EDU_YEARS = list(range(1960, 2016, 5))
ERA_SPLIT = 1990

# Canonical per-outcome lag and expected sign (per scripts/wcde/table_contemporaneous.py).
#   tag -> (lag, expected_sign, label)
OUTCOMES = {
    "log_gdp": (LAG_CONTEMPORANEOUS, +1, "log GDP  (lag 0)"),
    "log_tfr": (LAG_TFR,             -1, "log(TFR) (lag 5)"),
    "log_le":  (LAG_LE,              +1, "log(LE)  (lag 12)"),
    "log_u5":  (LAG_CHILDREARING,    -1, "log(U5MR)(lag 12)"),
}

# ── World Bank region membership, lowercased WCDE country names ─────────────
# Mirrors scripts/tables/table_1_subgroups.py REGION_MAP (the §Universality
# region split), lowercased to match the lowercase country keys in
# cohort_completion_both_long after the standard normalisation below.
_REGION_MAP_TITLE = {
    "SSA": {
        "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon",
        "Cape Verde", "Central African Republic", "Chad", "Comoros", "Congo",
        "Cote d'Ivoire", "Democratic Republic of the Congo",
        "Equatorial Guinea", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
        "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Madagascar", "Malawi",
        "Mali", "Mauritius", "Mozambique", "Namibia", "Niger", "Nigeria",
        "Reunion", "Rwanda", "Sao Tome and Principe", "Senegal",
        "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan",
        "Swaziland", "Togo", "Uganda", "United Republic of Tanzania", "Zambia",
        "Zimbabwe",
    },
    "MENA": {
        "Algeria", "Bahrain", "Egypt", "Iran (Islamic Republic of)", "Iraq",
        "Israel", "Jordan", "Kuwait", "Lebanon", "Morocco",
        "Occupied Palestinian Territory", "Oman", "Qatar", "Saudi Arabia",
        "Syrian Arab Republic", "Tunisia", "Turkey",
        "United Arab Emirates", "Yemen",
    },
    "South Asia": {
        "Afghanistan", "Bangladesh", "Bhutan", "India", "Maldives", "Nepal",
        "Pakistan", "Sri Lanka",
    },
    "East Asia & Pacific": {
        "Australia", "Cambodia", "China",
        "Democratic People's Republic of Korea", "Fiji", "French Polynesia",
        "Hong Kong Special Administrative Region of China", "Indonesia",
        "Japan", "Kiribati", "Lao People's Democratic Republic",
        "Macao Special Administrative Region of China", "Malaysia",
        "Micronesia (Federated States of)", "Mongolia", "Myanmar",
        "New Caledonia", "New Zealand", "Philippines", "Republic of Korea",
        "Samoa", "Singapore", "Solomon Islands", "Taiwan Province of China",
        "Thailand", "Timor-Leste", "Tonga", "Vanuatu", "Viet Nam",
    },
    "LAC": {
        "Argentina", "Aruba", "Bahamas", "Belize",
        "Bolivia (Plurinational State of)", "Brazil", "Chile", "Colombia",
        "Costa Rica", "Cuba", "Curaçao", "Dominican Republic", "Ecuador",
        "El Salvador", "French Guiana", "Guadeloupe", "Guatemala", "Guyana",
        "Haiti", "Honduras", "Jamaica", "Martinique", "Mexico", "Nicaragua",
        "Panama", "Paraguay", "Peru", "Puerto Rico", "Saint Lucia",
        "Saint Vincent and the Grenadines", "Suriname", "Trinidad and Tobago",
        "Uruguay", "Venezuela (Bolivarian Republic of)",
    },
    "Europe & N.America": {
        "Albania", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium",
        "Bosnia and Herzegovina", "Bulgaria", "Canada", "Croatia", "Cyprus",
        "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Georgia",
        "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy",
        "Kazakhstan", "Kyrgyzstan", "Latvia", "Lithuania", "Luxembourg",
        "Malta", "Montenegro", "Netherlands", "Norway", "Poland", "Portugal",
        "Republic of Moldova", "Romania", "Russian Federation", "Serbia",
        "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Tajikistan",
        "The former Yugoslav Republic of Macedonia", "Turkmenistan", "Ukraine",
        "United Kingdom of Great Britain and Northern Ireland",
        "United States of America",
    },
}
REGION_MAP = {r: {m.lower().strip() for m in members}
              for r, members in _REGION_MAP_TITLE.items()}
REGION_ORDER = ["SSA", "MENA", "South Asia", "East Asia & Pacific",
                "LAC", "Europe & N.America"]


def region_of(country):
    for r, members in REGION_MAP.items():
        if country in members:
            return r
    return None


# ── Data loading (mirrors scripts/wcde/table_contemporaneous.py) ────────────
def _name(df, c):
    if c in df.index:
        return c
    cand = _SHARED_NAME_MAP.get(c, c)
    return cand if cand in df.index else None


def _val(df, c, year):
    nm = _name(df, c)
    if nm is None or str(year) not in df.columns:
        return np.nan
    v = df.loc[nm, str(year)]
    return float(v) if pd.notna(v) else np.nan


def build_panel():
    """Canonical-lag panel: lower-sec at T plus each outcome at its canonical
    lag. Region and GDP-at-T attached for subsetting."""
    edu_long = pd.read_csv(os.path.join(PROC, "cohort_completion_both_long.csv"))
    edu_long = edu_long[~edu_long["country"].isin(REGIONS)]
    edu_long = edu_long.rename(columns={"cohort_year": "year"})
    edu_long["country"] = edu_long["country"].str.lower().str.strip()

    gdp_df = load_wb("gdppercapita_us_inflation_adjusted.csv")
    le_df  = load_wb("life_expectancy_years.csv")
    tfr_df = load_wb("children_per_woman_total_fertility.csv")
    u5_df  = load_wb("child_mortality_u5.csv")
    src = {"log_gdp": gdp_df, "log_le": le_df, "log_tfr": tfr_df, "log_u5": u5_df}

    rows = []
    for c in sorted(edu_long["country"].unique()):
        sub = edu_long[edu_long["country"] == c].set_index("year")
        reg = region_of(c)
        for t in EDU_YEARS:
            if t not in sub.index:
                continue
            low_t = sub.loc[t, "lower_sec"]
            if np.isnan(low_t):
                continue
            gdp_ctrl_t = _val(gdp_df, c, t)  # GDP at the education year (tercile basis)
            row = {
                "country": c, "t": t, "region": reg, "low_t": low_t,
                "log_gdp_ctrl_t": (np.log(gdp_ctrl_t)
                                   if pd.notna(gdp_ctrl_t) and gdp_ctrl_t > 0 else np.nan),
            }
            # Each outcome at its canonical lag (raw -> log).
            for tag, df in src.items():
                lag = OUTCOMES[tag][0]
                v = _val(df, c, t + lag)
                row[tag] = np.log(v) if pd.notna(v) and v > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def fe_clustered(x_col, y_col, data):
    """Country-FE, country-clustered-SE univariate slope (within estimator).
    Identical estimator to scripts/wcde/table_contemporaneous.py."""
    sub = data.dropna(subset=[x_col, y_col]).copy()
    counts = sub.groupby("country").size()
    sub = sub[sub["country"].isin(counts[counts >= 2].index)]
    if sub["country"].nunique() < 3 or len(sub) < 10:
        return None
    for col in (x_col, y_col):
        sub[col] = sub[col] - sub.groupby("country")[col].transform("mean")
    Xd = sub[[x_col]].to_numpy(dtype=float)
    yd = sub[y_col].to_numpy(dtype=float)
    countries = sub["country"].to_numpy()
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ yd
    resid = yd - Xd @ beta
    meat = np.zeros((1, 1))
    for cc in np.unique(countries):
        idx = countries == cc
        u = Xd[idx].T @ resid[idx]
        meat += np.outer(u, u)
    G = len(np.unique(countries))
    N = len(yd)
    cluster_adj = (G / (G - 1)) * ((N - 1) / (N - 1))
    vcov = cluster_adj * (XtX_inv @ meat @ XtX_inv)
    se = float(np.sqrt(vcov[0, 0]))
    tval = float(beta[0]) / se if se > 0 else np.nan
    p = float(2 * (1 - _st.t.cdf(np.abs(tval), df=G - 1))) if se > 0 else np.nan
    return {"beta": float(beta[0]), "se": se, "p": p, "n": int(N), "countries": int(G)}


def make_subsamples(panel):
    """The eleven §Universality subsamples, returned as (name, dataframe)."""
    subs = []
    for reg in REGION_ORDER:
        subs.append((f"region:{reg}", panel[panel["region"] == reg]))
    subs.append((f"era:T<{ERA_SPLIT}", panel[panel["t"] < ERA_SPLIT]))
    subs.append((f"era:T>={ERA_SPLIT}", panel[panel["t"] >= ERA_SPLIT]))
    # GDP terciles on GDP at the education year, within the analysis sample.
    g = panel.dropna(subset=["log_gdp_ctrl_t"])
    q33, q67 = g["log_gdp_ctrl_t"].quantile([1/3, 2/3]).values
    subs.append(("gdp_tercile:low",
                 g[g["log_gdp_ctrl_t"] < q33]))
    subs.append(("gdp_tercile:mid",
                 g[(g["log_gdp_ctrl_t"] >= q33) & (g["log_gdp_ctrl_t"] < q67)]))
    subs.append(("gdp_tercile:high",
                 g[g["log_gdp_ctrl_t"] >= q67]))
    return subs, float(q33), float(q67)


def stars(p):
    if p is None or np.isnan(p):
        return "  "
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "  "


def main():
    panel = build_panel()
    print(f"Panel: {len(panel)} obs, {panel['country'].nunique()} countries, "
          f"{panel['region'].notna().sum()} region-assigned obs")
    unassigned = sorted(panel[panel["region"].isna()]["country"].unique())
    if unassigned:
        print(f"WARNING: {len(unassigned)} unassigned countries: {unassigned}")

    subs, q33, q67 = make_subsamples(panel)
    # Prepend the full sample as the reference fingerprint.
    samples = [("FULL", panel)] + subs

    # ── Canonical-lag fingerprint per subsample ─────────────────────────────
    print("\n" + "=" * 78)
    print("FINGERPRINT — four outcomes at canonical lags (expected: GDP+, TFR-, LE+, U5MR-)")
    print("=" * 78)
    fingerprint = {}
    header = f"{'subsample':<24}" + "".join(f"{OUTCOMES[t][2]:>20}" for t in OUTCOMES)
    print(header)
    for name, df in samples:
        cells = {}
        line = f"{name:<24}"
        for tag, (lag, sign, _lab) in OUTCOMES.items():
            res = fe_clustered("low_t", tag, df)
            if res is None:
                cells[tag] = None
                line += f"{'  n/a':>20}"
                continue
            sign_ok = (np.sign(res["beta"]) == sign)
            sig = (res["p"] is not None and not np.isnan(res["p"]) and res["p"] < 0.05)
            cells[tag] = {**res, "sign_ok": bool(sign_ok),
                          "sig05": bool(sig), "expected_sign": sign}
            line += f"{res['beta']:>+12.4f}{stars(res['p']):<3}{('OK' if sign_ok else 'XX'):>5}"
        fingerprint[name] = cells
        print(line)

    # Tally over the eleven subsamples (exclude FULL).
    sub_names = [n for n, _ in subs]
    n_cells = 0
    n_sign_ok = 0
    n_sign_and_sig = 0
    full_pattern_subs = 0
    max_p = 0.0
    for name in sub_names:
        cells = fingerprint[name]
        present = [c for c in cells.values() if c is not None]
        all_sign = all(c["sign_ok"] for c in present) and len(present) == len(OUTCOMES)
        all_sig = all(c["sign_ok"] and c["sig05"] for c in present) and len(present) == len(OUTCOMES)
        if all_sign:
            full_pattern_subs += 1
        for c in present:
            n_cells += 1
            n_sign_ok += int(c["sign_ok"])
            n_sign_and_sig += int(c["sign_ok"] and c["sig05"])
            if c["p"] is not None and not np.isnan(c["p"]):
                max_p = max(max_p, c["p"])
        tag = "ALL-4 sign-correct" + (" & p<.05" if all_sig else "")
        print(f"   {name:<22} -> {tag if all_sign else 'sign mismatch'}")

    print(f"\n  Cells (subsample x outcome) evaluated:        {n_cells}")
    print(f"  Cells with correct sign:                      {n_sign_ok}/{n_cells}")
    print(f"  Cells correct sign AND p<0.05:                {n_sign_and_sig}/{n_cells}")
    print(f"  Subsamples w/ all four signs correct:         {full_pattern_subs}/{len(sub_names)}")

    # ── Magnitude band: each semi-elasticity across the eleven subsamples ───
    # Not just every sign correct, but every magnitude in a narrow band. A lone
    # coefficient could be reproduced by chance or by a different mechanism; the
    # four semi-elasticities holding the same scale everywhere cannot.
    print("\n" + "=" * 78)
    print("MAGNITUDE BAND — semi-elasticity across the eleven subsamples")
    print("=" * 78)
    bands = {}
    print(f"{'outcome':<22}{'min':>12}{'max':>12}{'full':>12}{'|max/min|':>12}")
    for tag, (_lag, _sign, lab) in OUTCOMES.items():
        betas = [fingerprint[n][tag]["beta"] for n in sub_names
                 if fingerprint[n][tag] is not None]
        bmin, bmax = min(betas), max(betas)
        full_beta = fingerprint["FULL"][tag]["beta"]
        ratio = abs(bmax / bmin) if bmin != 0 else float("nan")
        bands[tag] = {"min": round(bmin, 4), "max": round(bmax, 4),
                      "full": round(full_beta, 4), "max_over_min": round(ratio, 2),
                      "n_subsamples": len(betas)}
        print(f"{lab:<22}{bmin:>+12.4f}{bmax:>+12.4f}{full_beta:>+12.4f}{ratio:>12.2f}")

    # ── Pack checkin ────────────────────────────────────────────────────────
    numbers = {
        "n_subsamples": len(sub_names),
        "n_outcomes": len(OUTCOMES),
        "n_cells": n_cells,
        "cells_sign_correct": n_sign_ok,
        "cells_sign_and_sig05": n_sign_and_sig,
        "max_p_across_cells": float(f"{max_p:.4g}"),
        "subsamples_all_four_signs": full_pattern_subs,
        "gdp_tercile_q33_loggdp": round(q33, 3),
        "gdp_tercile_q67_loggdp": round(q67, 3),
    }
    for tag, b in bands.items():
        numbers[f"band_{tag}_min"] = b["min"]
        numbers[f"band_{tag}_max"] = b["max"]
        numbers[f"band_{tag}_max_over_min"] = b["max_over_min"]
    # Per-cell betas/p for traceability.
    detail = {}
    for name in [n for n, _ in samples]:
        detail[name] = {}
        for tag, cell in fingerprint[name].items():
            if cell is None:
                detail[name][tag] = None
            else:
                detail[name][tag] = {
                    "beta": round(cell["beta"], 4),
                    "se": round(cell["se"], 4),
                    "p": float(f"{cell['p']:.4g}") if cell["p"] is not None else None,
                    "n": cell["n"], "countries": cell["countries"],
                    "expected_sign": cell["expected_sign"],
                    "sign_ok": cell["sign_ok"], "sig05": cell["sig05"],
                }
    write_checkin("universality_lag_fingerprint.json", {
        "notes": (
            "Mechanism-fingerprint universality test. The canonical per-outcome "
            "bio-lag spec of scripts/wcde/table_contemporaneous.py (lower-sec at "
            "T -> log GDP @0, log TFR @5, log LE @12, log U5MR @12; country FE, "
            "country-clustered SE) re-estimated on the eleven subsamples of "
            "the §Universality section (six WB regions, two eras at T<1990/>=1990, "
            "three GDP terciles on GDP at T). Reports, for each subsample, "
            "whether all four signs hold at canonical lags with p<0.05, and the "
            "across-subsample band of each semi-elasticity. Produced by "
            "scripts/universality_lag_fingerprint.py."),
        "numbers": numbers,
        "fingerprint": detail,
        "bands": bands,
    }, script_path="scripts/universality_lag_fingerprint.py")
    print("\nDone.")


if __name__ == "__main__":
    main()
