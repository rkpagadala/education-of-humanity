# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/data_loader_longrun.py
# Paper:   "The Long Childhood: On the Convergence of Humanity"
#
# Produces:
#   Pre-1950 country × year panel for the longrun (pre-medicine) test.
#   Modeled on data_loader.py but extended back to 1870 and using HMD +
#   Gapminder outcome series instead of WB.
#
# Two modes:
#   'within':  T ∈ {1885,...,1920}; predicts (LE, U5MR, TFR) at T+25
#              from windows [T-15,T-10,T-5,T]. All outcomes pre-1945.
#   'forward': train T ∈ {1885,...,1940}; test T ∈ {1965,...,1995}.
#              Trained on pre-medicine, tested on modern (1990-2020 outcomes).
#
# Inputs:
#   wcde/data/processed/cohort_{primary,lower_sec}_{both,female,male}.csv
#   data/external/hmd/{e0,u5mr}_1880_1950.csv
#   data/external/gapminder/{life_expectancy,child_mortality,tfr}_1800_2020.csv
#   data/external/maddison_gdppc_wb_equivalent.csv
#   data/external/country_latlong.csv
#
# Outputs:
#   load_panel_longrun(mode, tier, ...) -> dict with same schema as
#       data_loader.load_panel: X, y, mask, country_ids, years,
#       feature_names, countries, feature_groups, feat_mean, feat_std,
#       mode, window, pred_horizon, tier.
# =============================================================================
"""
data_loader_longrun.py — parallel loader for the longrun (pre-medicine) test.

DOES NOT mutate scripts/ml/data_loader.py. The 1960+ panel produced by
that loader remains the canonical evidence; this loader extends the
training era backwards to ~1880.
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from _shared import PROC, DATA, REGIONS, standardize_country_name   # noqa: E402

HMD_DIR = os.path.join(REPO_ROOT, "data", "external", "hmd")
GAP_DIR = os.path.join(REPO_ROOT, "data", "external", "gapminder")
EXT_DIR = os.path.join(REPO_ROOT, "data", "external")

WINDOW = 4
# PRED_HORIZON / COHORT_LAGS are on the WCDE 5-year cohort grid by design, NOT
# the analysis constant _shared.py::LAG_GENERATION (=28). Unifying to 28 is
# deferred to a cloud-retrain session (see data_loader.py note).
PRED_HORIZON = 25

# Within-era T values: T ∈ {1885,...,1920}, so T-15 ≥ 1870 and T+25 ≤ 1945.
T_VALUES_WITHIN = list(range(1885, 1925, 5))

# Forward-mode train T values: T ∈ {1885,...,1940}, outcomes at T+25 ∈ {1910,...,1965}.
# Forward-mode test  T values: T ∈ {1965,...,1995}, outcomes at T+25 ∈ {1990,...,2020}.
T_VALUES_FWD_TRAIN = list(range(1885, 1945, 5))
T_VALUES_FWD_TEST  = list(range(1965, 2000, 5))

COHORT_LEVELS = [
    "primary_both", "primary_female", "primary_male",
    "lower_sec_both", "lower_sec_female", "lower_sec_male",
]
COHORT_LAGS = [25, 50]   # parent + grandparent generations

TARGETS = ["LE", "U5MR", "TFR"]

# Mode-dependent target sets. Pre-1945 fertility (TFR) is too sparse in the
# historical sources to admit more than ~4 countries, so the within-era
# headline rests on the two MORTALITY outcomes (LE, U5MR) — which are also
# precisely the outcomes the medical-tech confound is about (antibiotics and
# vaccines act on death, not fertility), so the pre-medicine test is strongest
# exactly where the data is richest. The forward-mode test trains pre-1950 and
# predicts 1960+, where post-1950 TFR is abundant, so it keeps all three.
TARGETS_BY_MODE = {
    "within":  ["LE", "U5MR"],
    "forward": ["LE", "U5MR", "TFR"],
}


def _load_wcde_cohort_features():
    """Load all six WCDE cohort wide CSVs; return dict[level -> DataFrame]
    indexed by canonical country name, columns = integer year (1870..2015).

    Filters out region rows (Africa, Asia, etc) using _shared.REGIONS.
    """
    out = {}
    for level in COHORT_LEVELS:
        path = os.path.join(PROC, f"cohort_{level}.csv")
        df = pd.read_csv(path)
        df = df[~df["country"].isin(REGIONS)].copy()
        df["country_canon"] = df["country"].map(standardize_country_name)
        df = df.dropna(subset=["country_canon"]).set_index("country_canon")
        df = df.drop(columns=["country"])
        df.columns = [int(c) for c in df.columns]
        out[level] = df
    return out


def _load_maddison():
    """Maddison GDP per capita, WB-equivalent units, log-transformed.

    Returns long-format DataFrame: country_canon, year, log_gdppc.
    """
    path = os.path.join(EXT_DIR, "maddison_gdppc_wb_equivalent.csv")
    df = pd.read_csv(path)
    gdp = df["gdppc_wb_eq"].astype(float)
    # Maddison contains a few rows with gdppc==0 (data gaps coded as zero).
    # log(0) = -inf would contaminate the panel's standardization; NaN them
    # so the loader's mask channel handles them cleanly.
    gdp = gdp.mask(gdp <= 0)
    df["log_gdppc"] = np.log(gdp)
    return df[["country_canon", "year", "log_gdppc"]].copy()


def _load_latlong():
    """Country latlong from data/external/country_latlong.csv.

    Returns DataFrame: country_canon, lat, lon. Drops countries that
    don't map to a canonical name.
    """
    path = os.path.join(EXT_DIR, "country_latlong.csv")
    df = pd.read_csv(path)
    df["country_canon"] = df["name"].map(standardize_country_name)
    df = df.dropna(subset=["country_canon"])
    return df[["country_canon", "latitude", "longitude"]].rename(
        columns={"latitude": "lat", "longitude": "lon"}
    )


def forward_train_mask(panel):
    """For forward-mode panel, return boolean array: True = training row
    (pre-1945 outcomes), False = test row (post-1990 outcomes)."""
    years = panel["years"]
    return years < 1945


def forward_standardize(panel):
    """Re-standardize features using TRAIN ROWS ONLY. Returns a new panel
    dict with X re-standardized and feat_mean/feat_std updated. Used by
    forward-mode runners to prevent train-test leakage."""
    is_train = forward_train_mask(panel)
    if not is_train.any():
        raise RuntimeError("forward_standardize: no training rows")
    # Undo the mode-agnostic standardization first
    X = panel["X"] * panel["feat_std"] + panel["feat_mean"]
    feat_mean = X[is_train].reshape(-1, X.shape[-1]).mean(axis=0)
    feat_std = X[is_train].reshape(-1, X.shape[-1]).std(axis=0)
    feat_std[feat_std == 0] = 1.0
    X_new = (X - feat_mean) / feat_std
    out = dict(panel)
    out["X"] = X_new
    out["feat_mean"] = feat_mean
    out["feat_std"] = feat_std
    return out


# HMD uses constituent-nation labels that standardize_country_name doesn't
# resolve, while WCDE education is reported at the sovereign-state level.
# Map HMD's mortality units onto the WCDE country they share education with.
# England and Wales is ~88% of UK population and WCDE education is UK-wide,
# so it is the best single UK proxy; Scotland is dropped from the core panel
# to avoid two HMD rows colliding on one WCDE country (noted in PROVENANCE).
HMD_TO_WCDE_NAME = {
    "England and Wales": "United Kingdom",
}
HMD_DROP = {"Scotland"}


def _load_hmd_wide(filename):
    """Load HMD CSV (country, 1880, 1885, ..., 1950 wide); return
    DataFrame indexed by canonical country name.

    Applies HMD_TO_WCDE_NAME before canonicalization so HMD's constituent-
    nation labels (England and Wales) map onto the WCDE sovereign state
    (United Kingdom) that carries the matching education series.
    """
    df = pd.read_csv(os.path.join(HMD_DIR, filename))
    df = df[~df["country"].isin(HMD_DROP)].copy()
    df["country"] = df["country"].replace(HMD_TO_WCDE_NAME)
    df["country_canon"] = df["country"].map(standardize_country_name)
    df = df.dropna(subset=["country_canon"]).set_index("country_canon")
    df = df.drop(columns=["country"])
    df.columns = [int(c) for c in df.columns]
    return df


def _load_gapminder_long(filename, value_col):
    """Load Gapminder OWID long-format CSV; return wide DataFrame
    indexed by canonical country name, columns = integer year (5-yr grid).

    The Gapminder files use OWID format: entity, code, year, <indicator>.
    The 'entity' column contains full country names; we map directly
    through standardize_country_name.
    """
    df = pd.read_csv(os.path.join(GAP_DIR, filename))
    df["country_canon"] = df["entity"].map(standardize_country_name)
    df = df.dropna(subset=["country_canon"])
    df["year"] = df["year"].astype(int)
    df = df[df["year"] % 5 == 0]   # 5-year grid
    wide = df.pivot_table(index="country_canon", columns="year",
                          values=value_col, aggfunc="first")
    wide.columns = [int(c) for c in wide.columns]
    return wide


def _load_outcomes(tier="core"):
    """Load LE, U5MR, TFR outcome panels.

    tier='core':     HMD for LE+U5MR (1880-1950); Gapminder for TFR
                     (HMD doesn't carry fertility). WB 1960+ for forward-mode
                     test years.
    tier='extended': Gapminder for all three (wider country set, lower quality).

    Returns dict[target -> wide DataFrame: country × year]. Years are 5-yr
    grid 1880..2015. Missing values stay NaN.
    """
    grid_pre  = list(range(1880, 1955, 5))
    grid_post = list(range(1960, 2025, 5))

    if tier == "core":
        le_pre  = _load_hmd_wide("e0_1880_1950.csv")
        u5_pre  = _load_hmd_wide("u5mr_1880_1950.csv")
    elif tier == "extended":
        le_pre  = _load_gapminder_long(
            "life_expectancy_1800_2020.csv", "life_expectancy_0")
        # OWID's `child_mortality_rate` is per 100 (PERCENT). HMD and WB
        # convention is per 1000. Multiply by 10 so the units match across
        # tiers and across the pre/post merge with wb_u5.
        u5_pre  = _load_gapminder_long(
            "child_mortality_1800_2020.csv",
            "child_mortality_rate") * 10.0
    else:
        raise ValueError(f"unknown tier: {tier}")

    tfr_full = _load_gapminder_long(
        "tfr_1800_2020.csv", "fertility_rate_hist")

    # WB post-1960 for forward-mode test years
    wb_le = _load_wb_wide(os.path.join(DATA, "life_expectancy_years.csv"))
    wb_u5 = _load_wb_wide(os.path.join(DATA, "child_mortality_u5.csv"))
    wb_tfr = _load_wb_wide(os.path.join(DATA, "children_per_woman_total_fertility.csv"))

    def _merge(pre, post):
        # combine_first: prefer post (measured WB) where present, fall back to pre.
        merged = post.combine_first(pre)
        return merged[[y for y in (grid_pre + grid_post) if y in merged.columns]]

    return {
        "LE":   _merge(le_pre, wb_le),
        "U5MR": _merge(u5_pre, wb_u5),
        "TFR":  _merge(tfr_full, wb_tfr),
    }


def _build_features_for_country_year(country, T, cohort, maddison_idx, latlong_idx):
    """Build the feature vector at (country, T) for one timestep.

    Features:
      - 6 WCDE cohort education levels × 2 cohort lags (25, 50) = 12 features
      - Maddison log GDPpc at year T (1 feature)
      - lat, lon (2 features, time-invariant)
    Returns (feat_vector, valid_flag). valid_flag=False if WCDE features
    are entirely missing for this country.
    """
    vec = []
    for level in COHORT_LEVELS:
        df = cohort[level]
        for lag in COHORT_LAGS:
            cohort_year = T - lag
            try:
                v = float(df.loc[country, cohort_year])
            except KeyError:
                v = np.nan
            vec.append(v)
    # Maddison
    v = maddison_idx.get((country, T), np.nan)
    vec.append(v)
    # latlong
    if country in latlong_idx.index:
        vec.append(float(latlong_idx.loc[country, "lat"]))
        vec.append(float(latlong_idx.loc[country, "lon"]))
    else:
        vec.append(np.nan); vec.append(np.nan)
    arr = np.asarray(vec, dtype=float)
    valid = bool(np.isfinite(arr[:12]).any())   # any WCDE cohort value valid
    return arr, valid


FEATURE_NAMES = (
    [f"cohort_{lvl}_lag{lag}" for lvl in COHORT_LEVELS for lag in COHORT_LAGS]
    + ["log_gdppc_t"]
    + ["lat", "lon"]
)


def load_panel_longrun(mode="within", tier="core", verbose=True):
    """Assemble the longrun panel.

    mode: 'within' or 'forward'.
    tier: 'core' (HMD) or 'extended' (Gapminder).
    """
    cohort = _load_wcde_cohort_features()
    maddison_df = _load_maddison()
    maddison_idx = maddison_df.set_index(["country_canon", "year"])["log_gdppc"].to_dict()
    latlong_idx = _load_latlong().set_index("country_canon")
    outcomes = _load_outcomes(tier=tier)

    if mode == "within":
        t_values = T_VALUES_WITHIN
    elif mode == "forward":
        # Includes both train and test T values; the runner splits on
        # year >= 1965 (the gap between train and test).
        t_values = T_VALUES_FWD_TRAIN + T_VALUES_FWD_TEST
    else:
        raise ValueError(f"unknown mode: {mode}")

    active_targets = TARGETS_BY_MODE[mode]

    # Countries: intersection of cohort education ∩ the ACTIVE outcome panels.
    # Only the active targets prune the country set — within-mode must not be
    # pruned by TFR coverage it never uses.
    candidate_countries = set(cohort["primary_both"].index)
    for tgt in active_targets:
        candidate_countries &= set(outcomes[tgt].index)

    countries = sorted(candidate_countries)
    country_idx = {c: i for i, c in enumerate(countries)}

    Xs, ys, masks, cids, years_out = [], [], [], [], []
    for c in countries:
        for T in t_values:
            # Target check: all ACTIVE outcomes at T+25 must be available
            target_year = T + PRED_HORIZON
            y_vec = [outcomes[tgt].loc[c].get(target_year, np.nan)
                     for tgt in active_targets]
            if not all(np.isfinite(y_vec)):
                continue
            # Build window [T-15, T-10, T-5, T]
            window = []
            mask = []
            valid_any = False
            for w in range(WINDOW):
                t_step = T - 5 * (WINDOW - 1 - w)   # T-15, T-10, T-5, T
                vec, valid = _build_features_for_country_year(
                    c, t_step, cohort, maddison_idx, latlong_idx)
                window.append(vec)
                mask.append(1.0 if valid else 0.0)
                if valid:
                    valid_any = True
            if not valid_any:
                continue
            Xs.append(np.stack(window))
            ys.append(np.asarray(y_vec, dtype=float))
            masks.append(np.asarray(mask, dtype=float))
            cids.append(country_idx[c])
            years_out.append(T)

    X = np.stack(Xs) if Xs else np.zeros((0, WINDOW, len(FEATURE_NAMES)))
    # Replace NaN with 0 after stacking; mask channel handles missingness.
    X = np.nan_to_num(X, nan=0.0)
    y = np.asarray(ys) if ys else np.zeros((0, len(active_targets)))
    mask = np.asarray(masks) if masks else np.zeros((0, WINDOW))
    cids = np.asarray(cids, dtype=int)
    years_out = np.asarray(years_out, dtype=int)

    # Remap country ids to the CONTRIBUTING countries only. `countries` above
    # is the candidate set (cohort ∩ active outcomes), but most candidates
    # contribute no admissible (country, T) sample — their outcomes at T+25
    # are missing in the pre-1945 window. Keep only countries that actually
    # appear, so `countries`/`country_ids`/n_countries describe the real panel.
    surviving = sorted(set(int(c) for c in cids))
    remap = {old: new for new, old in enumerate(surviving)}
    contributing_countries = [countries[o] for o in surviving]
    cids = np.asarray([remap[int(c)] for c in cids], dtype=int)
    countries = contributing_countries

    # Z-score within feature (training pool only — for within mode
    # that's everything; for forward mode this is overridden by the
    # runner using train-only stats).
    feat_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    feat_std = X.reshape(-1, X.shape[-1]).std(axis=0)
    feat_std[feat_std == 0] = 1.0
    X = (X - feat_mean) / feat_std

    if verbose:
        print(f"longrun panel ({mode}, tier={tier}): "
              f"{X.shape[0]} samples, {len(countries)} countries, "
              f"years {years_out.min() if len(years_out) else 'N/A'}"
              f"-{years_out.max() if len(years_out) else 'N/A'}")

    return {
        "X": X, "y": y, "mask": mask,
        "country_ids": cids, "years": years_out,
        "feature_names": FEATURE_NAMES,
        "countries": countries,
        "feature_groups": {
            "wcde_education": list(range(12)),
            "derived": [],
            "bl_education": [],
            "gdp": [12],
            "geo": [13, 14],
        },
        "feat_mean": feat_mean, "feat_std": feat_std,
        "mode": mode, "tier": tier,
        "target_names": active_targets,
        "window": WINDOW, "pred_horizon": PRED_HORIZON,
    }


def _load_wb_wide(path):
    """Load WB CSV (Country, 1960, 1961, ... wide); return wide DataFrame
    indexed by canonical country name, columns = integer year (5-yr grid)."""
    df = pd.read_csv(path)
    df["country_canon"] = df.iloc[:, 0].map(standardize_country_name)
    df = df.dropna(subset=["country_canon"]).set_index("country_canon")
    df = df.drop(columns=[df.columns[0]])
    year_cols = [c for c in df.columns if str(c).isdigit() and int(c) % 5 == 0]
    df = df[year_cols]
    df.columns = [int(c) for c in df.columns]
    return df
