# scripts/ml/longrun/tests/test_data_loader.py
import os
import sys
import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "ml"))


def test_wcde_cohort_returns_dict_of_dataframes():
    from data_loader_longrun import _load_wcde_cohort_features
    feats = _load_wcde_cohort_features()
    # Returns dict[level -> DataFrame] indexed by country_canon
    assert isinstance(feats, dict)
    assert "primary_both" in feats
    assert "lower_sec_female" in feats
    df = feats["primary_both"]
    # Has 1870..2015 in 5-year steps
    assert 1870 in df.columns
    assert 1920 in df.columns
    assert 2015 in df.columns
    # Has Sweden (canonical)
    assert "sweden" in df.index


def test_maddison_loader_returns_country_year_long():
    from data_loader_longrun import _load_maddison
    df = _load_maddison()
    assert {"country_canon", "year", "log_gdppc"} <= set(df.columns)
    # Sweden 1900 should be a real number
    s_1900 = df[(df.country_canon == "sweden") & (df.year == 1900)]
    assert len(s_1900) == 1
    assert np.isfinite(s_1900["log_gdppc"].iloc[0])


def test_latlong_loader_returns_lat_lon_per_country():
    from data_loader_longrun import _load_latlong
    df = _load_latlong()
    assert {"country_canon", "lat", "lon"} <= set(df.columns)
    assert "sweden" in df.country_canon.values


def test_outcomes_core_tier_uses_hmd_for_le_and_u5mr():
    from data_loader_longrun import _load_outcomes
    outcomes = _load_outcomes(tier="core")
    # Should be dict keyed by ('LE','U5MR','TFR'), each a wide DataFrame
    # indexed by country_canon, columns = year
    assert set(outcomes) == {"LE", "U5MR", "TFR"}
    le = outcomes["LE"]
    # HMD core should have Sweden 1900 (pre-medicine)
    assert 1900 in le.columns
    assert "sweden" in le.index
    assert np.isfinite(le.loc["sweden", 1900])
    # LE for Sweden in 1900 should be reasonable: 45-60 years.
    assert 40 < le.loc["sweden", 1900] < 65


def test_outcomes_extended_tier_falls_back_to_gapminder():
    from data_loader_longrun import _load_outcomes
    outcomes = _load_outcomes(tier="extended")
    le = outcomes["LE"]
    # Extended panel adds countries NOT in HMD; brazil should be in
    # Gapminder but not HMD.
    assert "brazil" in le.index


def test_load_panel_within_core_shape_and_targets():
    from data_loader_longrun import load_panel_longrun, T_VALUES_WITHIN, WINDOW
    panel = load_panel_longrun(mode="within", tier="core", verbose=False)
    # Schema check (matches data_loader.load_panel)
    for key in ("X", "y", "mask", "country_ids", "years",
                "feature_names", "countries", "feat_mean", "feat_std",
                "mode", "window", "pred_horizon", "tier", "target_names"):
        assert key in panel, f"missing key: {key}"
    # Within-era targets are the two MORTALITY outcomes (LE, U5MR); pre-1945
    # fertility is too sparse to admit (see TARGETS_BY_MODE in the loader).
    assert panel["target_names"] == ["LE", "U5MR"]
    # Shapes
    n = panel["X"].shape[0]
    assert panel["X"].shape == (n, WINDOW, len(panel["feature_names"]))
    assert panel["y"].shape == (n, 2)   # LE, U5MR
    assert panel["mask"].shape == (n, WINDOW)
    # T_VALUES_WITHIN years should appear in panel["years"]
    assert set(panel["years"].tolist()) <= set(T_VALUES_WITHIN)
    # The European first-mover panel: ~11 countries with continuous HMD
    # LE+U5MR coverage back to 1880.
    assert len(panel["countries"]) >= 8
    # Sweden expected in core
    assert any("sweden" in c for c in panel["countries"])


def test_load_panel_forward_includes_train_and_test_years():
    from data_loader_longrun import load_panel_longrun
    panel = load_panel_longrun(mode="forward", tier="core", verbose=False)
    years = set(panel["years"].tolist())
    assert any(y <= 1940 for y in years), "missing pre-1945 train years"
    assert any(y >= 1965 for y in years), "missing post-1960 test years"


def test_forward_split_mask_separates_train_from_test():
    from data_loader_longrun import load_panel_longrun, forward_train_mask
    panel = load_panel_longrun(mode="forward", tier="core", verbose=False)
    is_train = forward_train_mask(panel)
    # Train years should all be pre-1945; test should all be post-1960
    train_years = panel["years"][is_train]
    test_years  = panel["years"][~is_train]
    assert train_years.max() < 1945
    assert test_years.min() >= 1965
