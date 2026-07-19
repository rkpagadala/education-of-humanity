# scripts/ml/chapter9/tests/test_aggregate_longrun.py
"""
Tests for aggregate_longrun.py.

Two test groups:
  1. transformer_longrun_within (2-target: LE, U5MR) — verifies r2_baseline_median.
  2. panel_fe_longrun_within   — verifies ols_beta_edu_median.
"""

import json
import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "ml", "chapter9"))


# ---------------------------------------------------------------------------
# Helpers to write fake result.json files
# ---------------------------------------------------------------------------

def _write_transformer_within(dir_, seed, r2_le, r2_u5mr):
    """Write a fake transformer_longrun_within result.json (2 targets: LE, U5MR)."""
    sub = os.path.join(dir_, f"longrun_twithin_s{seed}")
    os.makedirs(sub, exist_ok=True)
    with open(os.path.join(sub, "result.json"), "w") as f:
        json.dump(
            {
                "kind": "transformer_longrun_within",
                "params": {"tier": "core", "seed": seed},
                "tier": "core",
                "n_samples": 85,
                "n_countries": 15,
                "r2_baseline": {"LE": r2_le, "U5MR": r2_u5mr},
                "r2_ablated_edu": {"LE": 0.0, "U5MR": 0.0},
                "r2_drop_absolute": {"LE": r2_le, "U5MR": r2_u5mr},
            },
            f,
        )


def _write_panel_fe_within(dir_, seed, beta_le, cluster_se_le, r2_oof_le,
                            beta_u5mr, cluster_se_u5mr, r2_oof_u5mr):
    """Write a fake panel_fe_longrun_within result.json (2 targets: LE, U5MR)."""
    sub = os.path.join(dir_, f"longrun_fwithin_s{seed}")
    os.makedirs(sub, exist_ok=True)
    with open(os.path.join(sub, "result.json"), "w") as f:
        json.dump(
            {
                "kind": "panel_fe_longrun_within",
                "params": {"tier": "core", "seed": seed},
                "tier": "core",
                "n_samples": 85,
                "n_countries": 15,
                "ols_by_target": {
                    "LE": {
                        "beta_edu": beta_le,
                        "cluster_se": cluster_se_le,
                        "r2_oof": r2_oof_le,
                        "n": 85,
                        "n_countries": 15,
                        "target": "LE",
                    },
                    "U5MR": {
                        "beta_edu": beta_u5mr,
                        "cluster_se": cluster_se_u5mr,
                        "r2_oof": r2_oof_u5mr,
                        "n": 85,
                        "n_countries": 15,
                        "target": "U5MR",
                    },
                },
                "gbm_by_target": {
                    "LE": {
                        "r2_baseline": 0.5,
                        "r2_ablated_edu": 0.3,
                        "r2_drop": 0.2,
                        "n": 85,
                        "n_countries": 15,
                        "target": "LE",
                    },
                    "U5MR": {
                        "r2_baseline": 0.6,
                        "r2_ablated_edu": 0.4,
                        "r2_drop": 0.2,
                        "n": 85,
                        "n_countries": 15,
                        "target": "U5MR",
                    },
                },
            },
            f,
        )


# ---------------------------------------------------------------------------
# Test 1: transformer within — r2_baseline_median
# ---------------------------------------------------------------------------

# Known r2_le values across 5 seeds; sorted: [0.50, 0.55, 0.58, 0.60, 0.62]
# Median = 0.58
_TRANSFORMER_LE_VALUES = [0.50, 0.55, 0.60, 0.62, 0.58]
_TRANSFORMER_U5MR_VALUES = [0.40, 0.45, 0.50, 0.52, 0.48]
_TRANSFORMER_LE_MEDIAN = 0.58     # statistics.median of [0.50,0.55,0.58,0.60,0.62]
_TRANSFORMER_U5MR_MEDIAN = 0.48   # statistics.median of [0.40,0.45,0.48,0.50,0.52]


def test_aggregate_within_transformer_le_median():
    """aggregate_within_from_dir returns correct LE r2_baseline_median for transformer."""
    from aggregate_longrun import aggregate_within_from_dir

    with tempfile.TemporaryDirectory() as td:
        for s, r2_le, r2_u5mr in zip(
            [42, 43, 44, 45, 46],
            _TRANSFORMER_LE_VALUES,
            _TRANSFORMER_U5MR_VALUES,
        ):
            _write_transformer_within(td, s, r2_le, r2_u5mr)

        agg = aggregate_within_from_dir(td)

    assert "by_method" in agg
    assert "transformer" in agg["by_method"]
    tr = agg["by_method"]["transformer"]
    assert "LE" in tr, f"LE missing from transformer block; keys={list(tr.keys())}"
    median = tr["LE"]["r2_baseline_median"]
    assert abs(median - _TRANSFORMER_LE_MEDIAN) < 1e-9, (
        f"LE median={median!r}, expected {_TRANSFORMER_LE_MEDIAN}"
    )


def test_aggregate_within_transformer_u5mr_median():
    """aggregate_within_from_dir returns correct U5MR r2_baseline_median for transformer."""
    from aggregate_longrun import aggregate_within_from_dir

    with tempfile.TemporaryDirectory() as td:
        for s, r2_le, r2_u5mr in zip(
            [42, 43, 44, 45, 46],
            _TRANSFORMER_LE_VALUES,
            _TRANSFORMER_U5MR_VALUES,
        ):
            _write_transformer_within(td, s, r2_le, r2_u5mr)

        agg = aggregate_within_from_dir(td)

    tr = agg["by_method"]["transformer"]
    assert "U5MR" in tr
    median = tr["U5MR"]["r2_baseline_median"]
    assert abs(median - _TRANSFORMER_U5MR_MEDIAN) < 1e-9, (
        f"U5MR median={median!r}, expected {_TRANSFORMER_U5MR_MEDIAN}"
    )


def test_aggregate_within_transformer_n_seeds():
    """n_seeds is 5 when 5 seeds are written."""
    from aggregate_longrun import aggregate_within_from_dir

    with tempfile.TemporaryDirectory() as td:
        for s, r2_le, r2_u5mr in zip(
            [42, 43, 44, 45, 46],
            _TRANSFORMER_LE_VALUES,
            _TRANSFORMER_U5MR_VALUES,
        ):
            _write_transformer_within(td, s, r2_le, r2_u5mr)

        agg = aggregate_within_from_dir(td)

    assert agg["by_method"]["transformer"]["LE"]["n_seeds"] == 5


def test_aggregate_within_mode_field():
    """Top-level mode key is 'within'."""
    from aggregate_longrun import aggregate_within_from_dir

    with tempfile.TemporaryDirectory() as td:
        _write_transformer_within(td, 42, 0.5, 0.4)
        agg = aggregate_within_from_dir(td)

    assert agg["mode"] == "within"


# ---------------------------------------------------------------------------
# Test 2: panel_fe within — ols_beta_edu_median
# ---------------------------------------------------------------------------

# Known beta_edu for LE across 5 seeds; sorted: [0.38, 0.39, 0.40, 0.41, 0.42]
# Median = 0.40
_PANEL_FE_BETA_LE = [0.38, 0.42, 0.40, 0.39, 0.41]
_PANEL_FE_BETA_U5MR = [-3.2, -2.8, -3.0, -2.9, -3.1]
_PANEL_FE_BETA_LE_MEDIAN = 0.40
_PANEL_FE_BETA_U5MR_MEDIAN = -3.0   # median of [-3.2,-3.1,-3.0,-2.9,-2.8]


def test_aggregate_within_panel_fe_ols_beta_edu_le_median():
    """aggregate_within_from_dir returns correct ols_beta_edu_median for panel_fe LE."""
    from aggregate_longrun import aggregate_within_from_dir

    with tempfile.TemporaryDirectory() as td:
        for s, b_le, b_u5mr in zip(
            [42, 43, 44, 45, 46],
            _PANEL_FE_BETA_LE,
            _PANEL_FE_BETA_U5MR,
        ):
            _write_panel_fe_within(
                td, s,
                beta_le=b_le, cluster_se_le=0.09, r2_oof_le=-0.05,
                beta_u5mr=b_u5mr, cluster_se_u5mr=0.6, r2_oof_u5mr=0.003,
            )

        agg = aggregate_within_from_dir(td)

    assert "panel_fe" in agg["by_method"]
    pf = agg["by_method"]["panel_fe"]
    assert "LE" in pf, f"LE missing from panel_fe block; keys={list(pf.keys())}"
    median = pf["LE"]["ols_beta_edu_median"]
    assert abs(median - _PANEL_FE_BETA_LE_MEDIAN) < 1e-9, (
        f"ols_beta_edu_median LE={median!r}, expected {_PANEL_FE_BETA_LE_MEDIAN}"
    )


def test_aggregate_within_panel_fe_ols_beta_edu_u5mr_median():
    """aggregate_within_from_dir returns correct ols_beta_edu_median for panel_fe U5MR."""
    from aggregate_longrun import aggregate_within_from_dir

    with tempfile.TemporaryDirectory() as td:
        for s, b_le, b_u5mr in zip(
            [42, 43, 44, 45, 46],
            _PANEL_FE_BETA_LE,
            _PANEL_FE_BETA_U5MR,
        ):
            _write_panel_fe_within(
                td, s,
                beta_le=b_le, cluster_se_le=0.09, r2_oof_le=-0.05,
                beta_u5mr=b_u5mr, cluster_se_u5mr=0.6, r2_oof_u5mr=0.003,
            )

        agg = aggregate_within_from_dir(td)

    pf = agg["by_method"]["panel_fe"]
    assert "U5MR" in pf
    median = pf["U5MR"]["ols_beta_edu_median"]
    assert abs(median - _PANEL_FE_BETA_U5MR_MEDIAN) < 1e-9, (
        f"ols_beta_edu_median U5MR={median!r}, expected {_PANEL_FE_BETA_U5MR_MEDIAN}"
    )


def test_aggregate_within_panel_fe_n_seeds():
    """panel_fe n_seeds is 5 when 5 panel_fe within files are present."""
    from aggregate_longrun import aggregate_within_from_dir

    with tempfile.TemporaryDirectory() as td:
        for s, b_le, b_u5mr in zip(
            [42, 43, 44, 45, 46],
            _PANEL_FE_BETA_LE,
            _PANEL_FE_BETA_U5MR,
        ):
            _write_panel_fe_within(
                td, s,
                beta_le=b_le, cluster_se_le=0.09, r2_oof_le=-0.05,
                beta_u5mr=b_u5mr, cluster_se_u5mr=0.6, r2_oof_u5mr=0.003,
            )

        agg = aggregate_within_from_dir(td)

    assert agg["by_method"]["panel_fe"]["LE"]["n_seeds"] == 5


# ---------------------------------------------------------------------------
# Test 3: empty dir — no crash, empty dicts
# ---------------------------------------------------------------------------

def test_aggregate_within_empty_dir_no_crash():
    """aggregate_within_from_dir with no result files returns empty by_method entries."""
    from aggregate_longrun import aggregate_within_from_dir

    with tempfile.TemporaryDirectory() as td:
        agg = aggregate_within_from_dir(td)

    # Transformer block should be an empty dict (no files)
    assert agg["by_method"]["transformer"] == {}
    assert agg["by_method"]["panel_fe"] == {}


# ---------------------------------------------------------------------------
# Test 4: mixed dir — transformer and panel_fe coexist
# ---------------------------------------------------------------------------

def test_aggregate_within_mixed_dir():
    """transformer and panel_fe results can coexist in same dir."""
    from aggregate_longrun import aggregate_within_from_dir

    with tempfile.TemporaryDirectory() as td:
        # 3 transformer seeds
        for s, r2 in zip([42, 43, 44], [0.5, 0.6, 0.55]):
            _write_transformer_within(td, s, r2, r2 - 0.1)
        # 2 panel_fe seeds
        for s, b in zip([42, 43], [0.40, 0.42]):
            _write_panel_fe_within(
                td, s,
                beta_le=b, cluster_se_le=0.09, r2_oof_le=-0.05,
                beta_u5mr=-3.0, cluster_se_u5mr=0.6, r2_oof_u5mr=0.003,
            )

        agg = aggregate_within_from_dir(td)

    assert agg["by_method"]["transformer"]["LE"]["n_seeds"] == 3
    assert agg["by_method"]["panel_fe"]["LE"]["n_seeds"] == 2
