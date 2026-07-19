"""
necessity_floor.py
==================
Empirical floor for the framework's necessity claim:
    "Lower-secondary completion is necessary for the joint crossing of
     TFR<3.65 and LE>69.8."

For each country, identify T = first year both TFR<3.65 and LE>69.8 hold.
Read lsec(20-24) at T (educational stock at the moment of crossing).
Among non-oil crossers with valid T and valid lsec at T, report:

  - L*       = minimum lsec(T) — the necessity floor
  - binding  = the country that defines L*
  - p10, median, p90, max for distribution context

The floor L* is the empirical minimum below which no historical joint
crossing has occurred. Future joint crossings with lsec(T) below L*
break the necessity claim.

Exclusions:
  - oil states (LE crossings driven by rents, not the channel under test)
  - countries left-censored on T (already past at least one threshold at
    first observation — pre-WB data)
"""

import os
import sys
import json
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import (
    REPO_ROOT, PROC, TFR_THRESHOLD, LE_THRESHOLD, OIL_STATES,
    load_wb, standardize_country_name, REGIONS, write_checkin,
)

LAG = 0  # measure lsec at the year of joint crossing (T), not T-lag
FLOOR = 35.0  # rounded necessity floor — used for the sufficiency window test

# Sufficiency-subset criterion: countries that sustained lsec expansion at
# this rate or faster from below 10% through the transition. Rate computed
# 10%→90% (or 10%→last observation if 90% not yet reached). Floor 1.25 pp/yr
# is set by Sri Lanka (1.26 pp/yr) — the slowest of the paper's canonical
# fast-crossing cases. Bangladesh (0.89 pp/yr) sits below this floor.
#
# We do NOT label this subset "singular priority" — that is the paper's
# qualitative concept about state intent, and rate alone cannot adjudicate
# it. The rate filter is a clean empirical criterion that approximates the
# subset but admits cases (Indonesia, Malaysia, Tunisia) where the singular-
# priority label is contested.
SUSTAINED_RATE_FLOOR = 1.25  # pp/yr

# Countries excluded from the binding bound because the channel was disrupted
# during the lag window — either by war (outcomes suppressed while lsec kept
# expanding) or by hollow education (lsec attainment rose but the educated
# population didn't accumulate at home, by emigration or by content).
# These are the two failure modes §10 of the paper already names; lag
# observations under disruption do not test the framework's sufficiency
# claim, which is conditioned on an undisrupted channel.
DISRUPTED = {
    "sri lanka": "war — civil war 1983–2009 overlaps the back half of the 1960→1993 window",
    "guyana": "hollow (emigration) — sustained net emigration of educated population since the 1966 Burnham era; lsec attainment rose to 93% but home-niche stock never accumulated (population 760k in 1980 → 800k in 2022 despite high TFR; ~70%+ of college-educated emigrated)",
    "mongolia": "hollow (content + collapse) — Soviet-era schooling content was politically constrained (the §10 mechanism), then the 1990s post-Soviet macro collapse disrupted the demographic transition while lsec stayed high",
    "tonga": "hollow (emigration) — Pacific small-island emigration to NZ/Australia/US; same mechanism as Guyana, diaspora exceeds home population",
    "palestine": "occupation — channel-disruption analogue of war; territorial governance fragmented across the lag window",
    "trinidad and tobago": "hollow (emigration) — Caribbean brain drain; population 1.0M in 1980 → 1.4M in 2022 despite high TFR; substantial diaspora in UK/US/Canada",
    "egypt": "hollow (content, rural lsec) — post-2008 rural TFR rose 18.8% (rural Upper Egypt 3.6→4.1) while urban rose 7.4%, and by 2013 TFRs converged across education levels (UN/UNFPA fertility-trends report); the lsec credential exists but cohort-driven transmission isn't running, especially in rural areas; TIMSS 2015 ranks Egypt 49/50 reading, 38/39 science",
}


def crossing_year(series, threshold, direction):
    """Return (year, left_censored).

    direction: 'below' for TFR<threshold, 'above' for LE>threshold.
    left_censored=True means the threshold was already met at the first
    observation, so the true crossing is pre-data.
    Returns (None, False) if the threshold is never met.
    """
    s = series.dropna()
    if len(s) == 0:
        return None, False
    cmp = (s < threshold) if direction == "below" else (s > threshold)
    if not cmp.any():
        return None, False
    if bool(cmp.iloc[0]):
        return int(s.index[0]), True
    idx = cmp[cmp].index[0]
    return int(idx), False


def lsec_at(year, yrs, vals):
    """Linear interpolation of lsec at `year`. NaN if out of observed range."""
    mask = ~np.isnan(vals)
    if mask.sum() < 2:
        return np.nan
    yrs_obs = yrs[mask]
    vals_obs = vals[mask]
    if year < yrs_obs.min() or year > yrs_obs.max():
        return np.nan
    return float(np.interp(year, yrs_obs, vals_obs))


def year_at_level(yrs, vals, level):
    """First year lsec reaches `level`, linearly interpolated.

    Returns (year, left_censored). left_censored=True iff lsec was already
    above `level` at the first observation, in which case `year` is the first
    observation year (an upper bound on the true crossing).
    """
    mask = ~np.isnan(vals)
    if mask.sum() < 2:
        return np.nan, False
    yo = yrs[mask]
    vo = vals[mask]
    if vo[0] >= level:
        return float(yo[0]), True
    above = vo >= level
    if not above.any():
        return np.nan, False
    i = int(np.argmax(above))
    if vo[i] == vo[i - 1]:
        return float(yo[i]), False
    frac = (level - vo[i - 1]) / (vo[i] - vo[i - 1])
    return float(yo[i - 1] + frac * (yo[i] - yo[i - 1])), False


def main():
    edu = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"))
    edu = edu[~edu["country"].isin(REGIONS)].copy()
    edu["country_canon"] = edu["country"].map(
        lambda x: standardize_country_name(x) or x.lower()
    )
    year_cols = [c for c in edu.columns if c.isdigit()]
    yrs = np.array([int(c) for c in year_cols], dtype=float)

    le_df = load_wb("life_expectancy_years.csv")
    tfr_df = load_wb("children_per_woman_total_fertility.csv")
    le_df.columns = [int(c) for c in le_df.columns]
    tfr_df.columns = [int(c) for c in tfr_df.columns]

    oil_canon = {standardize_country_name(c) or c.lower() for c in OIL_STATES}

    crossed, left_censored_T, left_censored_lag, no_outcome = [], [], [], []
    oil_recs = []
    for _, r in edu.iterrows():
        canon = r["country_canon"]
        vals = r[year_cols].values.astype(float)

        if canon not in le_df.index or canon not in tfr_df.index:
            continue
        le_s = le_df.loc[canon]
        tfr_s = tfr_df.loc[canon]
        if isinstance(le_s, pd.DataFrame):
            le_s = le_s.iloc[0]
        if isinstance(tfr_s, pd.DataFrame):
            tfr_s = tfr_s.iloc[0]

        tfr_year, tfr_lc = crossing_year(tfr_s, TFR_THRESHOLD, "below")
        le_year, le_lc = crossing_year(le_s, LE_THRESHOLD, "above")
        if tfr_year is None or le_year is None:
            no_outcome.append(canon)
            continue
        T = int(max(tfr_year, le_year))
        if tfr_lc or le_lc:
            left_censored_T.append({"country": canon, "T_observed": T})
            continue

        target = T - LAG
        L = lsec_at(target, yrs, vals)
        if np.isnan(L):
            left_censored_lag.append(
                {"country": canon, "T": T, "T_minus_lag": target}
            )
            continue

        year_floor, floor_lc = year_at_level(yrs, vals, FLOOR)
        if np.isnan(year_floor):
            lag_from_floor = np.nan
        else:
            lag_from_floor = T - year_floor

        # lsec expansion rate (10→90 pp/yr; if 90 not reached, 10→last obs)
        y10, _ = year_at_level(yrs, vals, 10.0)
        y90, _ = year_at_level(yrs, vals, 90.0)
        obs_mask = ~np.isnan(vals)
        if obs_mask.any():
            last_yr = float(yrs[obs_mask][-1])
            last_val = float(vals[obs_mask][-1])
        else:
            last_yr = last_val = np.nan
        if np.isnan(y10):
            rate = np.nan
        elif not np.isnan(y90):
            rate = 80.0 / (y90 - y10)
        elif last_val > 10:
            rate = (last_val - 10.0) / (last_yr - y10)
        else:
            rate = np.nan

        rec = {
            "country": canon,
            "T_joint": T,
            "tfr_year": int(tfr_year),
            "le_year": int(le_year),
            "lsec_at_T": round(L, 1),
            "year_reached_floor": (
                round(year_floor, 1) if not np.isnan(year_floor) else None
            ),
            "floor_left_censored": floor_lc,
            "lag_from_floor": (
                round(lag_from_floor, 1) if not np.isnan(lag_from_floor) else None
            ),
            "lsec_rate_pp_per_yr": (
                round(rate, 2) if not np.isnan(rate) else None
            ),
            "in_rate_subset": (
                bool(
                    (not np.isnan(rate))
                    and (rate >= SUSTAINED_RATE_FLOOR)
                    and (not floor_lc)
                )
            ),
        }
        if canon in oil_canon:
            oil_recs.append(rec)
        else:
            crossed.append(rec)

    df = pd.DataFrame(crossed).sort_values("lsec_at_T").reset_index(drop=True)
    if df.empty:
        print("No countries with valid T and lsec(T). Exiting.")
        return

    floor = float(df["lsec_at_T"].min())
    binding = df.iloc[0]["country"]
    binding_T = int(df.iloc[0]["T_joint"])
    p10 = float(df["lsec_at_T"].quantile(0.10))
    median = float(df["lsec_at_T"].median())
    p90 = float(df["lsec_at_T"].quantile(0.90))
    maximum = float(df["lsec_at_T"].max())

    sep = "=" * 92
    print(sep)
    print("NECESSITY FLOOR — lsec(20-24) at T across joint crossers")
    print(f"  T = first year both TFR<{TFR_THRESHOLD} and LE>{LE_THRESHOLD} hold")
    print(f"  measurement at year of crossing (lag = {LAG})")
    print(f"  n = {len(df)} non-oil crossers with valid T and valid lsec(T)")
    print(sep)
    print(f"  L* (binding minimum):  {floor:5.1f}%   ({binding}, T={binding_T})")
    print(f"  p10:                   {p10:5.1f}%")
    print(f"  median:                {median:5.1f}%")
    print(f"  p90:                   {p90:5.1f}%")
    print(f"  max:                   {maximum:5.1f}%")
    print()
    print("  Bottom 15 (closest to the floor):")
    print(f"  {'country':28s} {'T':>5} {'tfr_yr':>7} {'le_yr':>6} {'lsec(T)':>11}")
    print(f"  {'-'*28} {'-'*5:>5} {'-'*7:>7} {'-'*6:>6} {'-'*11:>11}")
    for _, row in df.head(15).iterrows():
        print(
            f"  {row['country']:28s} {int(row['T_joint']):5d} "
            f"{int(row['tfr_year']):7d} {int(row['le_year']):6d} "
            f"{row['lsec_at_T']:10.1f}%"
        )

    print()
    print("  Excluded:")
    print(f"    oil states (rent-driven LE):           {len(oil_recs):3d}")
    print(f"    left-censored T (joint pre-data):      {len(left_censored_T):3d}")
    print(f"    lsec out of WCDE range at T:           {len(left_censored_lag):3d}")
    print(f"    never crossed both thresholds:         {len(no_outcome):3d}")
    if left_censored_T:
        names = sorted({c["country"] for c in left_censored_T})
        print(f"\n    pre-data crossers ({len(names)}): "
              + ", ".join(names))
    if left_censored_lag:
        names = sorted({c["country"] for c in left_censored_lag})
        print(f"\n    out-of-range ({len(names)}): "
              + ", ".join(names))

    # ── sufficiency window: lag from passing the floor to joint crossing ────
    suff = df[df["lag_from_floor"].notna()].copy()
    if suff.empty:
        suff_max = suff_p90 = suff_median = float("nan")
        sp_max = sp_p90 = float("nan")
    else:
        suff_max = float(suff["lag_from_floor"].max())
        suff_p90 = float(suff["lag_from_floor"].quantile(0.90))
        suff_median = float(suff["lag_from_floor"].median())
        sp = suff[suff["in_rate_subset"] == True]
        sp_max = float(sp["lag_from_floor"].max()) if not sp.empty else float("nan")
        sp_p90 = float(sp["lag_from_floor"].quantile(0.90)) if not sp.empty else float("nan")
        sp_median = float(sp["lag_from_floor"].median()) if not sp.empty else float("nan")
        sp_clean = sp[~sp["country"].isin(DISRUPTED)]
        sp_clean_max = (
            float(sp_clean["lag_from_floor"].max()) if not sp_clean.empty else float("nan")
        )
        sp_clean_binding = (
            sp_clean.sort_values("lag_from_floor").iloc[-1]["country"]
            if not sp_clean.empty else None
        )

    print()
    print(sep)
    print(f"SUFFICIENCY WINDOW — lag (years) from lsec ≥ {FLOOR:.0f}% to joint crossing")
    print(f"  rate-subset criterion: lsec rate ≥ {SUSTAINED_RATE_FLOOR:.2f} pp/yr "
          f"(10%→90%, or 10%→last) AND not LC at floor")
    print(sep)
    print(f"  All non-oil crossers (n={len(suff)}):")
    print(f"    median: {suff_median:5.1f}  p90: {suff_p90:5.1f}  max: {suff_max:5.1f}")
    if not suff.empty:
        binding_max = suff.sort_values("lag_from_floor").iloc[-1]
        print(f"    longest lag: {binding_max['country']} "
              f"({binding_max['lag_from_floor']:.0f} yr; "
              f"floor {binding_max['year_reached_floor']:.0f} → T {int(binding_max['T_joint'])}; "
              f"rate {binding_max['lsec_rate_pp_per_yr']} pp/yr)")
    print()
    print(f"  Rate-subset crossers (n={len(sp) if not suff.empty else 0}):")
    if not suff.empty and not sp.empty:
        for _, row in sp.sort_values("lag_from_floor").iterrows():
            tag = f"  [{DISRUPTED[row['country']].split('—')[0].strip()}-disrupted]" if row["country"] in DISRUPTED else ""
            print(f"    {row['country']:28s}  rate {row['lsec_rate_pp_per_yr']:.2f}  "
                  f"floor {row['year_reached_floor']:.0f}  T {int(row['T_joint'])}  "
                  f"lag {row['lag_from_floor']:.0f} yr  "
                  f"lsec(T) {row['lsec_at_T']:.0f}%{tag}")
        print(f"\n    rate-subset median lag:                       {sp_median:.0f} yr")
        print(f"    rate-subset p90 lag:                          {sp_p90:.0f} yr")
        print(f"    rate-subset max lag (all):                    {sp_max:.0f} yr")
        print(f"    rate-subset max lag (no channel disruption):  {sp_clean_max:.0f} yr "
              f"({sp_clean_binding})")
        print()
        print("    channel-disruption notes:")
        for c, note in DISRUPTED.items():
            print(f"      {c}: {note}")

    write_checkin(
        "necessity_floor.json",
        {
            "method": (
                f"min lsec(20-24) at T across non-oil joint crossers, "
                f"where T = first year both TFR<{TFR_THRESHOLD} and "
                f"LE>{LE_THRESHOLD} hold and neither threshold was already "
                f"met at first observation"
            ),
            "lag_years": LAG,
            "tfr_threshold": TFR_THRESHOLD,
            "le_threshold": LE_THRESHOLD,
            "n_crossers_used": len(df),
            "n_oil_excluded": len(oil_recs),
            "n_left_censored_T": len(left_censored_T),
            "n_left_censored_lag": len(left_censored_lag),
            "n_never_crossed": len(no_outcome),
            "floor_pct": round(floor, 1),
            "binding_country": binding,
            "binding_T_joint": binding_T,
            "p10_pct": round(p10, 1),
            "median_pct": round(median, 1),
            "p90_pct": round(p90, 1),
            "max_pct": round(maximum, 1),
            "sufficiency_floor_pct": FLOOR,
            "sufficiency_lag_median_yr": (
                round(suff_median, 1) if not np.isnan(suff_median) else None
            ),
            "sufficiency_lag_p90_yr": (
                round(suff_p90, 1) if not np.isnan(suff_p90) else None
            ),
            "sufficiency_lag_max_yr": (
                round(suff_max, 1) if not np.isnan(suff_max) else None
            ),
            "sustained_rate_floor_pp_per_yr": SUSTAINED_RATE_FLOOR,
            "channel_disruption_notes": DISRUPTED,
            "rate_subset_n": int(len(sp)) if not suff.empty else 0,
            "rate_subset_lag_median_yr": (
                round(sp_median, 1) if not np.isnan(sp_median) else None
            ),
            "rate_subset_lag_p90_yr": (
                round(sp_p90, 1) if not np.isnan(sp_p90) else None
            ),
            "rate_subset_lag_max_yr_all": (
                round(sp_max, 1) if not np.isnan(sp_max) else None
            ),
            "rate_subset_lag_max_yr_no_disruption": (
                round(sp_clean_max, 1) if not np.isnan(sp_clean_max) else None
            ),
            "rate_subset_lag_max_yr_no_disruption_binding": sp_clean_binding,
            "all_crossers": df.to_dict(orient="records"),
            "oil_excluded": oil_recs,
            "left_censored_T": left_censored_T,
            "left_censored_lag": left_censored_lag,
        },
        script_path="scripts/necessity_floor.py",
    )


if __name__ == "__main__":
    main()
