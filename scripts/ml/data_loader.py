# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/data_loader.py
# Paper:   "The Long Childhood: On the Convergence of Humanity"
#
# Produces:
#   Country x year panel for the evidential-AI transformer (Phase A).
#   Sequence inputs [T-15, T-10, T-5, T] -> targets (LE, TFR, U5MR) at
#   T+PRED_HORIZON (25 legacy grid; LAG_GENERATION=28 under UNIFY_GENERATION).
#
# Inputs:
#   wcde/data/processed/{primary,lower_sec,upper_sec,college}_{both,female}.csv
#   data/barro_lee.csv
#   data/gdppercapita_us_inflation_adjusted.csv
#   data/life_expectancy_years.csv
#   data/children_per_woman_total_fertility.csv
#   data/child_mortality_u5.csv
#
# Outputs (in-memory):
#   load_panel(mode, window) -> dict with X, y, mask, country_ids, years,
#                                feature_names, countries, feature_groups,
#                                feat_mean, feat_std, mode.
# =============================================================================
"""
data_loader.py

Assembles the country x year panel for the panel transformer.

Each training sample is one (country, T) row where outcomes at T+PRED_HORIZON
are available. The feature sequence is the country's trajectory over the
preceding window [T - 5*(W-1), ..., T-5, T]. Targets are LE, TFR, U5MR at
year T+PRED_HORIZON, all from the World Bank WDI to keep targets observation-
based (no WCDE projections in the y). PRED_HORIZON is 25 on the legacy WCDE
grid and LAG_GENERATION=28 under the UNIFY_GENERATION flag.

Three variants control which feature group the model sees:
  - 'joint':     all features
  - 'wcde_only': BL education features zeroed (post-standardization)
  - 'bl_only':   WCDE education features zeroed
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from _shared import (PROC, DATA, REGIONS, standardize_country_name, load_wb,
                     LAG_GENERATION)
sys.path.insert(0, SCRIPT_DIR)
from broader_features import BroaderFeatures

# Grid + horizons
# ---------------------------------------------------------------------------
# Generation lag. The pipeline runs in one of two regimes, selected by the
# UNIFY_GENERATION flag (env ML_UNIFY_GENERATION=1):
#
#   legacy grid  (UNIFY_GENERATION=False — the committed, trained default):
#     PRED_HORIZON=25, COHORT_LAGS=[25,50,75], on the WCDE 5-year birth-cohort
#     grid. Reproduces the committed GCS checkpoints and scripts/ml/checkin
#     numbers byte-for-byte, so `make verify` keeps passing untouched.
#
#   unified generation (UNIFY_GENERATION=True):
#     PRED_HORIZON=LAG_GENERATION (28), COHORT_LAGS=[28,56,84] — the paper's
#     analysis generation (scripts/_shared.py::LAG_GENERATION). 28 is off the
#     5-year cohort grid, so the birth-cohort attainment CSVs are interpolated
#     to annual resolution at load (see _interp_cohort_to_annual) and the
#     cohort_year = T-lag lookups resolve. WB targets (LE/TFR/U5MR) are already
#     annual, so the 28-year PRED_HORIZON needs no interpolation on the y side.
#
# The flag stays OFF by default (opt-in). The 28-generation retrain has been
# run: the committed parent-vantage artifacts — universal_evidence_parent_lag.json
# (aggregate_parent_vantage.py over the cloud trf_parent_* results) and
# spec_curve_parent.json (spec_curve.py --parent) — are the ML_UNIFY_GENERATION=1
# (28) products and only reproduce with the flag SET. Everything else stays on
# the legacy 25-year grid: the bloc holdout (chapter9_bloc_holdout.json) and the
# long-run pipeline (data_loader_longrun.py) are deliberately pinned to 25 as
# historical-window probes — the bloc test reads post-1990 Soviet republics that
# the 28-year horizon (which caps the joint panel at T<=1990) would starve, and
# longrun's 5-year cohort grid is intrinsic to its pre-medicine windows. The flag
# is NOT flipped to ON because that would silently push the bloc test (and the
# joint-panel ch9 diagnostics still committed at 25) onto the 28 grid.
UNIFY_GENERATION = os.environ.get("ML_UNIFY_GENERATION", "0") == "1"

YEARS = list(range(1950, 2026, 5))
PRED_HORIZON = LAG_GENERATION if UNIFY_GENERATION else 25
WINDOW = 4


def _t_values_for(horizon, window=WINDOW):
    """T values where T-15 >= 1950 AND T+horizon <= 2020 (WB outcome coverage)."""
    return [t for t in YEARS if t - 5 * (window - 1) >= YEARS[0] and t + horizon <= 2020]


# Default T values (lag=25)
T_VALUES = _t_values_for(PRED_HORIZON)

# WCDE wide files used as input features — 4 attainment levels × 3 sexes.
# Sex disaggregation matters: gender gap is a load-bearing mechanism term
# in the paper (mother-weighted parental routing). Including male as well
# as female lets the model see the gap rather than collapsing it.
WCDE_LEVELS = [
    "primary_both", "primary_female", "primary_male",
    "lower_sec_both", "lower_sec_female", "lower_sec_male",
    "upper_sec_both", "upper_sec_female", "upper_sec_male",
    "college_both", "college_female", "college_male",
]
# WCDE cohort historical files (wide, indexed by birth cohort year, back
# to 1870). At year T, look up cohort_year=T-25, T-50, T-75. Relative to a
# parent-vantage outcome (features at t = outcome - bio_lag, so the current
# cohort is the parent), these are the grandparent, great-grandparent and
# great-great-grandparent generations. The paper's generational mechanism
# (§grandparent effect, §multi-generational kin transmission) needs this
# explicit exposure for the model to see across multiple generations.
COHORT_LEVELS = [
    "primary_both", "primary_female", "primary_male",
    "lower_sec_both", "lower_sec_female", "lower_sec_male",
]
# Cohort lags in multiples of one biological generation. In the parent-vantage
# panels lag1 = grandparent / lag2 = great-grandparent / lag3 = great-great-
# grandparent (the current cohort is the parent). With the parent that is four
# generations, matching the §"Education persists across four generations"
# finding (U5MR persistence 66% → 67% → 44% → 25% → 16%). Legacy regime uses
# the WCDE 5-year-grid turnover (25/50/75); the unified regime uses the paper's
# LAG_GENERATION (28/56/84), read off the annual-interpolated cohort panels.
COHORT_LAGS = ([LAG_GENERATION, 2 * LAG_GENERATION, 3 * LAG_GENERATION]
               if UNIFY_GENERATION else [25, 50, 75])

# Barro-Lee v3 attainment shares. Seven shares + mean years per (country,
# year, sex, age-group). We pull THREE age cohorts at year T to expose
# generational structure within the population alive at T:
#   15-24 — current young adults / future parents
#   25-34 — current parents
#   35-44 — older parents / younger grandparents
# For each cohort, BOTH MF (both sexes) and F (female only).
BL_SHARE_VARS = ["lu", "lp", "lpc", "ls", "lsc", "lh", "lhc", "yr_sch"]
BL_AGE_GROUPS = [(15, 24), (25, 34), (35, 44)]


def _load_wcde_wide(filename):
    """Load WCDE wide CSV; return DataFrame indexed by canonical country name,
    columns = integer year."""
    df = pd.read_csv(os.path.join(PROC, filename))
    df = df[~df["country"].isin(REGIONS)].copy()
    df["country_canon"] = df["country"].map(standardize_country_name)
    df = df.dropna(subset=["country_canon"]).set_index("country_canon")
    df = df.drop(columns=["country"])
    df.columns = [int(c) for c in df.columns]
    return df


def _load_bl_v3_cohort(filename, sex_filter, age_from, age_to):
    """Load Barro-Lee v3 long-format CSV; filter to one age cohort + sex.

    Returns dict (country_canon, year) -> {var: value} keyed by BL_SHARE_VARS.
    """
    bl = pd.read_csv(os.path.join(DATA, filename))
    bl = bl[(bl["sex"] == sex_filter) & (bl["agefrom"] == age_from) & (bl["ageto"] == age_to)].copy()
    bl["country_canon"] = bl["country"].map(standardize_country_name)
    bl = bl.dropna(subset=["country_canon"])
    lookup = {}
    for _, row in bl.iterrows():
        key = (row["country_canon"], int(row["year"]))
        lookup[key] = {v: row[v] for v in BL_SHARE_VARS}
    return lookup


def _load_wcde_cohort(filename):
    """Load a WCDE cohort wide CSV (indexed by birth COHORT year).

    Same structure as _load_wcde_wide but the column years are interpreted
    as birth cohort years, not observation years. Goes back to 1870.
    """
    return _load_wcde_wide(filename)


def _interp_cohort_to_annual(df):
    """Interpolate a birth-cohort-indexed wide panel (integer year columns on
    the 5-year grid) to annual columns so off-grid generational lags
    (cohort_year = T - 28/56/84 under LAG_GENERATION) resolve by direct column
    lookup. Fills only inside each country's observed cohort span — no
    extrapolation past the first/last observed cohort. Mirrors the annual
    interpolation used by grandparent_effect.py and the placebo loaders."""
    full = range(int(df.columns.min()), int(df.columns.max()) + 1)
    return df.reindex(columns=full).interpolate(axis=1, limit_area="inside")


def _wb_lookup(df, country, year):
    """Look up a WB wide-format value for a canonical country and integer year."""
    if country not in df.index:
        return np.nan
    col = str(year)
    if col not in df.columns:
        return np.nan
    val = df.loc[country, col]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def _build_wb_canon(df):
    """Re-index a WB DataFrame from raw country names to canonical names."""
    out = df.copy()
    canon = [standardize_country_name(c) for c in out.index]
    out.index = canon
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="first")]
    return out


def _load_maddison_backfill():
    """Load Maddison Project 2020 rescaled to WB-equivalent constant 2015 USD.

    Returns dict (country_canon, year) -> gdppc. Used to fill WB GDP cells
    that are NaN (typically pre-1960 and a long tail of country-specific
    early-year gaps). The companion observed-flag _obs_gdp lets the model
    distinguish backfill from original WB observation.
    """
    path = os.path.join(DATA, "external", "maddison_gdppc_wb_equivalent.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {(r["country_canon"], int(r["year"])): float(r["gdppc_wb_eq"])
            for _, r in df.iterrows()}


def load_panel(mode="joint", window=WINDOW, verbose=True, pred_horizon=PRED_HORIZON):
    """Assemble the panel. Returns a dict with arrays and metadata.

    `pred_horizon` is the lag in years between the feature timestep T
    and the target year. Default 25 (the paper's structural lag); pass
    other values for lag-sweep experiments.
    """
    t_values = _t_values_for(pred_horizon, window=window)

    # Broader (non-education) features: geography, region, colonizer,
    # religion fractions, colonial binary, settler mortality, polity,
    # oil rents, malaria, trade, CO2. See broader_features.py.
    broader = BroaderFeatures()

    # Maddison GDP backfill for WB-missing cells (rescaled to WB-equivalent
    # 2015 USD per build_maddison_backfill.py). Empty dict if file absent.
    maddison_gdp = _load_maddison_backfill()

    # WCDE current attainment (age 20-24 at observation year T)
    wcde_data = {lvl: _load_wcde_wide(f"{lvl}.csv") for lvl in WCDE_LEVELS}

    # WCDE cohort (birth-cohort-indexed) — gives parent and grandparent
    # generations' attainment at year T via cohort_year = T - lag lookup.
    # Under the unified generation the lags (28/56/84) fall off the 5-year
    # cohort grid, so interpolate the cohort panels to annual resolution.
    cohort_data = {lvl: _load_wcde_cohort(f"cohort_{lvl}.csv") for lvl in COHORT_LEVELS}
    if UNIFY_GENERATION:
        cohort_data = {lvl: _interp_cohort_to_annual(df)
                       for lvl, df in cohort_data.items()}

    # Barro-Lee v3 — three age groups at year T, MF + F per group.
    bl_data = {}
    for ag_from, ag_to in BL_AGE_GROUPS:
        bl_data[("MF", ag_from)] = _load_bl_v3_cohort("barro_lee_v3.csv", "MF", ag_from, ag_to)
        bl_data[("F",  ag_from)] = _load_bl_v3_cohort("barro_lee_v3_female.csv", "F", ag_from, ag_to)

    # WB outcome and GDP series (canonicalize the index for joins)
    gdp_wb = _build_wb_canon(load_wb("gdppercapita_us_inflation_adjusted.csv"))
    le_wb = _build_wb_canon(load_wb("life_expectancy_years.csv"))
    tfr_wb = _build_wb_canon(load_wb("children_per_woman_total_fertility.csv"))
    u5mr_wb = _build_wb_canon(load_wb("child_mortality_u5.csv"))

    # Country universe: union of WCDE country sets, filtered to canonical names
    countries = set()
    for df in wcde_data.values():
        countries.update(df.index.dropna())
    countries = sorted(c for c in countries if c and not pd.isna(c))

    # Feature schema
    # Feature schema (v3 — exhaustive):
    #   wcde_current   — WCDE current age 20-24 attainment, 4 levels × 3 sexes = 12
    #   wcde_cohort    — WCDE cohort attainment at birth-years T-COHORT_LAGS
    #                    (25/50/75 legacy, 28/56/84 unified),
    #                    2 levels × 3 sexes × 3 lags = 18
    #   bl_education   — BL v3 shares at 3 age groups × 2 sexes × 8 vars = 48
    #   derived        — Gender gaps (4 levels, both sexes diff) + incomplete
    #                    primary explicit (2 sexes) = 6
    #   gdp            — GDP per capita, constant 2015 USD = 1
    # Total: 79 features per timestep.
    # Deliberately excludes prior outcomes — see feedback_no_prior_outcomes_as_features.md.
    feature_groups = {
        "wcde_education": [],
        "bl_education": [],
        "gdp": [],
        "derived": [],
    }
    feature_names = []

    # WCDE current (12 features)
    for lvl in WCDE_LEVELS:
        feature_groups["wcde_education"].append(len(feature_names))
        feature_names.append(f"wcde_{lvl}")

    # WCDE cohort: 2 levels × 3 sexes × 3 lags = 18 features. Named by LAG,
    # not generation — which generation a lag denotes depends on the panel's
    # vantage. In the parent-vantage panels (features at t = outcome − bio_lag)
    # the first lag = grandparent, second = great-grandparent, third = great-
    # great-grandparent; in the joint panel each is one generation shallower.
    # The lag value in the feature name follows the active regime (25/50/75
    # legacy, 28/56/84 unified). The paper prose attaches the generation per
    # vantage.
    cohort_feature_specs = []   # (idx, level, lag)
    for lag in COHORT_LAGS:
        for lvl in COHORT_LEVELS:
            cohort_feature_specs.append((len(feature_names), lvl, lag))
            feature_groups["wcde_education"].append(len(feature_names))
            feature_names.append(f"wcde_cohort_lag{lag}_{lvl}")

    # BL features (48 features: 3 age groups × 2 sexes × 8 vars)
    bl_feature_specs = []        # (idx, sex_key, age_from, var)
    for ag_from, ag_to in BL_AGE_GROUPS:
        for sex_key, sex_suffix in (("MF", "mf"), ("F", "f")):
            for v in BL_SHARE_VARS:
                bl_feature_specs.append((len(feature_names), sex_key, ag_from, v))
                feature_groups["bl_education"].append(len(feature_names))
                feature_names.append(f"bl_{v}_{sex_suffix}_{ag_from}_{ag_to}")

    # Derived: gender gaps (lower_sec, primary, upper_sec, college) using
    # current WCDE male - female. 4 features.
    gap_feature_specs = []       # (idx, level_root)
    for lvl_root in ["primary", "lower_sec", "upper_sec", "college"]:
        gap_feature_specs.append((len(feature_names), lvl_root))
        feature_groups["derived"].append(len(feature_names))
        feature_names.append(f"wcde_gap_{lvl_root}")

    # Derived: incomplete primary explicit (lp - lpc) at age 15-24, both
    # MF and F. 2 features.
    inc_prim_specs = []          # (idx, sex_key)
    for sex_key, sex_suffix in (("MF", "mf"), ("F", "f")):
        inc_prim_specs.append((len(feature_names), sex_key))
        feature_groups["derived"].append(len(feature_names))
        feature_names.append(f"bl_inc_prim_{sex_suffix}_15_24")

    # GDP
    feature_groups["gdp"].append(len(feature_names))
    feature_names.append("gdp")
    gdp_idx = feature_names.index("gdp")

    # Broader (non-education) features — appended after education + GDP.
    # Each broader-feature group becomes its own ablatable feature group.
    broader_offset = len(feature_names)
    broader_names = broader.feature_names()
    broader_groups = broader.feature_groups()
    for bn in broader_names:
        feature_names.append(bn)
    for gname, indices in broader_groups.items():
        # shift broader-local indices into the global feature space
        feature_groups[gname] = [broader_offset + i for i in indices]

    # Missingness indicators (1 if feature observed, 0 if NaN/imputed).
    # Added for the broader features with notable missingness — AJR
    # settler mortality (61/200 countries), polity2, religion fractions,
    # oil rents, malaria, trade, CO2. Lets the model distinguish "this
    # country genuinely has a low polity score" from "polity score was
    # never observed". Fair-ablation requirement: when we ablate a
    # broader-feature group, this indicator carries no information
    # because we'd zero the group's feature(s) and the indicator becomes
    # redundant.
    # GDP gets an indicator that fires 1 only for WB-observed cells (i.e.
    # cells where Maddison backfill was NOT needed). This lets the model
    # distinguish "WB observed" from "Maddison-reconstructed pre-WB or gap".
    INDICATOR_TARGETS = [
        "religion_frac_christian", "religion_frac_muslim",
        "religion_frac_hindu", "religion_frac_buddhist",
        "religion_frac_unaffiliated", "ajr_logem4", "polity2",
        "oil_rents_pct_gdp", "malaria_incidence_per_1000",
        "trade_openness_pct_gdp", "co2_per_capita_tonnes",
        "gdp",
    ]
    indicator_pairs = []   # (indicator_idx, source_idx, src_name)
    feature_groups["missingness_indicators"] = []
    for src_name in INDICATOR_TARGETS:
        if src_name not in feature_names:
            continue
        src_idx = feature_names.index(src_name)
        ind_idx = len(feature_names)
        feature_groups["missingness_indicators"].append(ind_idx)
        feature_names.append(f"_obs_{src_name}")
        indicator_pairs.append((ind_idx, src_idx, src_name))

    F = len(feature_names)
    backfill_log = {"wb_observed": 0, "maddison_backfill": 0, "both_missing": 0}

    def features_at(country, year):
        x = np.full(F, np.nan)

        # WCDE current (age 20-24 at year T)
        for i, lvl in enumerate(WCDE_LEVELS):
            df = wcde_data[lvl]
            if country in df.index and year in df.columns:
                x[i] = df.loc[country, year]

        # WCDE cohort: look up cohort_year = T - lag
        for idx, lvl, lag in cohort_feature_specs:
            cohort_year = year - lag
            df = cohort_data[lvl]
            if country in df.index and cohort_year in df.columns:
                x[idx] = df.loc[country, cohort_year]

        # BL: per (sex, age_group, var)
        for idx, sex_key, ag_from, var in bl_feature_specs:
            row = bl_data[(sex_key, ag_from)].get((country, year))
            if row is not None:
                x[idx] = row[var]

        # Derived: gender gap (male - female) for current WCDE
        for idx, lvl_root in gap_feature_specs:
            m = wcde_data.get(f"{lvl_root}_male", pd.DataFrame())
            f_ = wcde_data.get(f"{lvl_root}_female", pd.DataFrame())
            if country in m.index and year in m.columns and country in f_.index and year in f_.columns:
                mv = m.loc[country, year]
                fv = f_.loc[country, year]
                if pd.notna(mv) and pd.notna(fv):
                    x[idx] = float(mv) - float(fv)

        # Derived: incomplete primary = lp - lpc at age 15-24
        for idx, sex_key in inc_prim_specs:
            row = bl_data[(sex_key, 15)].get((country, year))
            if row is not None and pd.notna(row.get("lp")) and pd.notna(row.get("lpc")):
                x[idx] = float(row["lp"]) - float(row["lpc"])

        # GDP — WB first, Maddison backfill if WB missing.
        wb_val = _wb_lookup(gdp_wb, country, year)
        gdp_is_wb_observed = not np.isnan(wb_val)
        if gdp_is_wb_observed:
            x[gdp_idx] = wb_val
            backfill_log["wb_observed"] += 1
        else:
            maddison_val = maddison_gdp.get((country, year))
            if maddison_val is not None and np.isfinite(maddison_val):
                x[gdp_idx] = maddison_val
                backfill_log["maddison_backfill"] += 1
            else:
                x[gdp_idx] = np.nan
                backfill_log["both_missing"] += 1

        # Broader features (geography, region, colonizer, religion,
        # colonial_binary, settler_mortality, institutions, resources,
        # climate_disease, trade)
        bx = broader.features_at(country, year)
        for i, v in enumerate(bx):
            x[broader_offset + i] = v

        # Missingness indicators — set BEFORE standardization fills NaN.
        # GDP's indicator means "WB-observed", NOT "any value present" —
        # Maddison-backfilled cells get _obs_gdp=0 so the model can learn
        # to discount them if reconstruction quality matters.
        for ind_idx, src_idx, src_name in indicator_pairs:
            if src_name == "gdp":
                x[ind_idx] = 1.0 if gdp_is_wb_observed else 0.0
            else:
                x[ind_idx] = 0.0 if np.isnan(x[src_idx]) else 1.0
        return x

    def target_at(country, year):
        return np.array([
            _wb_lookup(le_wb, country, year),
            _wb_lookup(tfr_wb, country, year),
            _wb_lookup(u5mr_wb, country, year),
        ])

    country_to_idx = {c: i for i, c in enumerate(countries)}
    X_list, y_list, mask_list, cid_list, year_list = [], [], [], [], []

    for country in countries:
        for t in t_values:
            y = target_at(country, t + pred_horizon)
            if np.any(np.isnan(y)):
                continue
            seq = np.zeros((window, F), dtype=np.float64)
            seq_mask = np.zeros(window, dtype=np.float64)
            for k in range(window):
                yr = t - 5 * (window - 1 - k)
                if yr < YEARS[0]:
                    continue
                feats = features_at(country, yr)
                if np.all(np.isnan(feats)):
                    continue
                seq[k] = np.where(np.isnan(feats), np.nan, feats)  # keep NaN for now
                seq_mask[k] = 1.0
            if seq_mask.sum() == 0:
                continue
            X_list.append(seq)
            y_list.append(y)
            mask_list.append(seq_mask)
            cid_list.append(country_to_idx[country])
            year_list.append(t)

    if not X_list:
        raise RuntimeError("No samples in panel; check data paths and country names.")

    X = np.stack(X_list).astype(np.float64)
    y = np.stack(y_list).astype(np.float32)
    mask = np.stack(mask_list).astype(np.float32)
    country_ids = np.array(cid_list, dtype=np.int64)
    years = np.array(year_list, dtype=np.int64)

    # Per-feature standardization using only observed (non-NaN) values
    feat_mean = np.zeros(F)
    feat_std = np.ones(F)
    for i in range(F):
        col = X[:, :, i]
        observed = ~np.isnan(col)
        if observed.sum() > 0:
            vals = col[observed]
            feat_mean[i] = vals.mean()
            s = vals.std()
            feat_std[i] = s if s > 1e-8 else 1.0
        X[:, :, i] = np.where(observed, (col - feat_mean[i]) / feat_std[i], 0.0)

    X = X.astype(np.float32)

    # Mode filter (zero out feature groups not present in the variant).
    # The "no_education" variant strips ALL education-related groups
    # (WCDE current + cohort, BL all age groups, derived gender-gap and
    # incomplete-primary). What remains: GDP + the 29 broader features.
    # This is the proper training-time ablation that adversarial critics
    # demand: not just inference-time zeroing, but a model that NEVER
    # saw education features during fitting. If R² stays high, the
    # country embedding + broader features carry the work. If R² falls,
    # education was actually doing it in the loss surface.
    if mode == "wcde_only":
        for i in feature_groups["bl_education"]:
            X[:, :, i] = 0.0
    elif mode == "bl_only":
        for i in feature_groups["wcde_education"]:
            X[:, :, i] = 0.0
    elif mode == "no_education":
        for grp in ("wcde_education", "bl_education", "derived"):
            for i in feature_groups[grp]:
                X[:, :, i] = 0.0
    elif mode == "education_only":
        # Keep WCDE + BL + derived; zero everything else (GDP, broader
        # features, missingness indicators). Isolates education's
        # predictive contribution without broader-feature proxies.
        keep = {"wcde_education", "bl_education", "derived"}
        for grp, indices in feature_groups.items():
            if grp not in keep:
                for i in indices:
                    X[:, :, i] = 0.0
    elif mode == "gdp_only":
        # Keep GDP only; zero education, broader, and indicators.
        # Head-to-head counterpart to education_only.
        keep = {"gdp"}
        for grp, indices in feature_groups.items():
            if grp not in keep:
                for i in indices:
                    X[:, :, i] = 0.0
    elif mode == "country_emb_only":
        # All features zeroed; only the country embedding is active in
        # the model. Baseline for what country identity alone predicts.
        for grp, indices in feature_groups.items():
            for i in indices:
                X[:, :, i] = 0.0
    elif mode != "joint":
        raise ValueError(f"Unknown mode: {mode}")

    if verbose:
        print(f"[data_loader] mode={mode}  X={X.shape}  y={y.shape}  "
              f"mask_mean={mask.mean():.3f}  countries={len(countries)}  "
              f"T_values={t_values}  pred_horizon={pred_horizon}")
        total_gdp_cells = sum(backfill_log.values())
        if total_gdp_cells > 0:
            print(f"[data_loader] GDP cell coverage: "
                  f"WB-observed {backfill_log['wb_observed']:,} "
                  f"({100 * backfill_log['wb_observed'] / total_gdp_cells:.1f}%), "
                  f"Maddison-backfilled {backfill_log['maddison_backfill']:,} "
                  f"({100 * backfill_log['maddison_backfill'] / total_gdp_cells:.1f}%), "
                  f"both-missing {backfill_log['both_missing']:,} "
                  f"({100 * backfill_log['both_missing'] / total_gdp_cells:.1f}%)")

    return {
        "X": X,
        "y": y,
        "mask": mask,
        "country_ids": country_ids,
        "years": years,
        "feature_names": feature_names,
        "countries": countries,
        "feature_groups": feature_groups,
        "feat_mean": feat_mean.astype(np.float32),
        "feat_std": feat_std.astype(np.float32),
        "mode": mode,
        "window": window,
        "pred_horizon": pred_horizon,
    }


def country_holdout_mask(panel, n_folds=5, fold=0, seed=42):
    """Return boolean array of length N with True for validation samples.

    Splits countries (not samples) so a held-out country is fully unseen.
    """
    n_countries = len(panel["countries"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_countries)
    fold_size = n_countries // n_folds
    val_start = fold * fold_size
    val_end = val_start + fold_size if fold < n_folds - 1 else n_countries
    val_countries = set(perm[val_start:val_end].tolist())
    return np.array([cid in val_countries for cid in panel["country_ids"]])


def era_holdout_mask(panel, cutoff_year=1985):
    """Return boolean array of length N with True for validation samples.

    Splits by year, not country: all samples with T >= cutoff_year are
    held out. Tests temporal generalization (predict outcomes in years
    the model never trained on) rather than cross-country generalization.

    Note: country embeddings stay 'warm' for all countries in this split,
    since every country appears in both train and validation. So era-
    holdout R² should be higher than country-holdout R² as a rule. The
    test is specifically whether the *temporal* signature holds out of
    the training era — i.e., does the model trained on 1965-1980 features
    predict outcomes at 2010-2020 from 1985-1995 features?
    """
    return panel["years"] >= cutoff_year


def feature_ablation(panel, drop_groups):
    """Return a deep-copy panel with specified feature groups zeroed.

    drop_groups: iterable of group names from panel['feature_groups'].
    Use this for G1/G2 ablation gates.
    """
    new = {k: v for k, v in panel.items()}
    X = panel["X"].copy()
    for g in drop_groups:
        if g not in panel["feature_groups"]:
            raise KeyError(f"Unknown feature group: {g}")
        for i in panel["feature_groups"][g]:
            X[:, :, i] = 0.0
    new["X"] = X
    new["ablated"] = list(drop_groups)
    return new


if __name__ == "__main__":
    for mode in ("joint", "wcde_only", "bl_only"):
        panel = load_panel(mode=mode)
        print(f"  features: {panel['feature_names']}")
        print(f"  T_VALUES: {T_VALUES}")
        print(f"  groups:   {panel['feature_groups']}")
        is_val = country_holdout_mask(panel, n_folds=5, fold=0)
        print(f"  fold0 val={int(is_val.sum())} / total={len(is_val)}\n")
