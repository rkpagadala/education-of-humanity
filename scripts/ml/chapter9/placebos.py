# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/chapter9/placebos.py
# Paper:   "The Long Childhood: On the Convergence of Humanity"
# Section: Chapter 9 — The Panel
#
# Produces:
#   The falsification suite. Each placebo transforms the panel in a way
#   that, if the model is genuinely learning the education→outcomes
#   relationship, MUST collapse the education R² contribution to ~0:
#
#     P1 outcome_scramble     — shuffle Y across observations (kills signal)
#     P2 year_scramble        — shuffle years within country (kills temporal)
#     P3 country_scramble     — shuffle country labels (kills cross-country)
#     P4 edu_replaced_noise   — replace edu features with N(0,1) noise
#     P5 edu_replaced_lat     — replace edu features with absolute latitude
#     P6 edu_replaced_gdprank — replace edu features with within-year GDP rank
#
#   For each placebo we report the spec-curve education-R²-drop. The
#   nulls should be near zero across all methods; the real run should be
#   22-31%.
#
# Outputs:
#   scripts/ml/checkin/placebos.json
#
# Compute:
#   CPU-only (uses spec_curve infrastructure). ~30-90 minutes for the
#   full 6-placebo × 5-method × 3-outcome grid.
# =============================================================================
"""
placebos.py

Falsification of the spec-curve attribution. If education is genuinely
load-bearing, breaking the link in any of six ways should erase the
R² drop. Six nulls is decisive evidence; one is not.
"""

import os
import sys
import time
import warnings

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(ML_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, ML_DIR)
sys.path.insert(0, SCRIPT_DIR)

from _shared import write_checkin, ML_CHECKIN
from data_loader import load_panel
from spec_curve import flatten_feature_groups, run_one_method

warnings.filterwarnings("ignore")

PLACEBO_METHODS = ["ridge", "random_forest", "gradient_boosting"]
DEFAULT_SEED = 42


def _copy_panel(panel):
    """Shallow copy with deep-copied arrays for safe mutation."""
    return {
        **panel,
        "X": panel["X"].copy(),
        "y": panel["y"].copy(),
        "country_ids": panel["country_ids"].copy(),
        "years": panel["years"].copy(),
        "mask": panel["mask"].copy(),
    }


def _edu_indices_3d(panel):
    """Education indices in [F] space (not flat)."""
    idx = []
    for g in ("wcde_education", "bl_education", "derived"):
        idx.extend(panel["feature_groups"].get(g, []))
    return sorted(set(idx))


def placebo_outcome_scramble(panel, seed):
    """P1: shuffle Y across observations. No predictor structure should
    explain anything; baseline R² should collapse to ~0, edu drop ~0."""
    p = _copy_panel(panel)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(p["y"].shape[0])
    p["y"] = p["y"][perm]
    return p, "outcome_scramble"


def placebo_year_scramble(panel, seed):
    """P2: within each country, shuffle the year labels on the (T, target)
    pairs. Each country's sequence of (year, outcome) is permuted so the
    temporal structure within country breaks while the cross-country
    structure persists. Concretely we shuffle y within each country."""
    p = _copy_panel(panel)
    rng = np.random.default_rng(seed)
    for cid in np.unique(p["country_ids"]):
        mask = p["country_ids"] == cid
        idx = np.where(mask)[0]
        perm = rng.permutation(idx)
        p["y"][idx] = p["y"][perm]
    return p, "year_scramble_within_country"


def placebo_country_scramble(panel, seed):
    """P3: pair each country's features with ANOTHER country's outcomes.

    Breaks the cross-country features<->outcome link while keeping each
    outcome block a coherent (real) country trajectory. If the model's
    education signal is genuine cross-country structure, this collapses the
    education R² drop to ~0.

    Only countries that actually have samples are remapped (the panel lists
    ~200 countries but only a subset appear in country_ids). Source and
    target blocks differ in length, so target rows are filled by sampling
    the source country's rows with replacement — size-safe by construction.
    """
    p = _copy_panel(panel)
    rng = np.random.default_rng(seed)
    cids = np.unique(p["country_ids"])
    # Derangement over countries-with-samples (no country keeps its own
    # outcomes) where the set has more than one country.
    perm = cids.copy()
    rng.shuffle(perm)
    for i in range(len(cids)):
        if perm[i] == cids[i] and len(cids) > 1:
            j = (i + 1) % len(cids)
            perm[i], perm[j] = perm[j], perm[i]
    src_of = dict(zip(cids.tolist(), perm.tolist()))
    new_y = p["y"].copy()
    for cid in cids:
        tgt_idx = np.where(p["country_ids"] == cid)[0]
        src_idx = np.where(p["country_ids"] == src_of[cid])[0]
        pick = rng.choice(src_idx, size=len(tgt_idx), replace=True)
        new_y[tgt_idx] = p["y"][pick]
    p["y"] = new_y
    return p, "country_scramble"


def placebo_edu_replaced_noise(panel, seed):
    """P4: replace all education features with standard-normal noise.
    No real education signal in the inputs; baseline R² should fall;
    edu R² drop on the noise features should be ~0."""
    p = _copy_panel(panel)
    rng = np.random.default_rng(seed)
    edu_idx = _edu_indices_3d(p)
    for i in edu_idx:
        # Replace, preserving the mask: where mask=0 keep 0; else random
        m = p["mask"]
        noise = rng.standard_normal(p["X"].shape[:2])
        p["X"][:, :, i] = np.where(m > 0, noise, 0.0)
    return p, "edu_replaced_with_noise"


def placebo_edu_replaced_latitude(panel, seed):
    """P5: replace all education features with each country's absolute
    latitude (constant within country). Climate / geography-style placebo.

    Requires the broader-features panel to carry a 'geography' group;
    we use the first geography feature as the latitude proxy. The
    transformation is country-constant — every timestep gets the same
    value — which is what we want for a "feature that varies across
    countries but not within."
    """
    p = _copy_panel(panel)
    geo = p["feature_groups"].get("geography", [])
    if not geo:
        raise RuntimeError("placebo_edu_replaced_latitude requires geography group")
    # Use index 0 of geography (typically latitude in the BroaderFeatures
    # ordering). The feature is already standardized.
    lat_idx = geo[0]
    edu_idx = _edu_indices_3d(p)
    for i in edu_idx:
        # Copy latitude (constant within country across the window) into edu slot
        p["X"][:, :, i] = p["X"][:, :, lat_idx]
    return p, "edu_replaced_with_latitude_proxy"


def placebo_edu_replaced_gdprank(panel, seed):
    """P6: replace all education features with within-year GDP rank.

    Constructs a synthetic feature: for each year, rank countries by GDP
    and emit the rank percentile [0,1]. Variable that varies across both
    country and year, BUT carries only income-rank information. If
    income-rank alone reproduces the education R² drop, then the
    education attribution was just GDP rank in disguise.
    """
    p = _copy_panel(panel)
    gdp_idx_list = p["feature_groups"].get("gdp", [])
    if not gdp_idx_list:
        raise RuntimeError("placebo_edu_replaced_gdprank requires gdp group")
    gdp_idx = gdp_idx_list[0]
    # Build a rank-within-year tensor on the same [N, W] grid using the
    # existing gdp feature (already standardized; we'll re-rank).
    rank_tensor = np.zeros(p["X"].shape[:2], dtype=np.float64)
    for w in range(p["X"].shape[1]):
        col = p["X"][:, w, gdp_idx]
        # Rank only over observed (mask=1) rows
        m = p["mask"][:, w]
        valid_rows = np.where(m > 0)[0]
        if len(valid_rows) == 0:
            continue
        vals = col[valid_rows]
        ranks = np.argsort(np.argsort(vals)) / max(len(vals) - 1, 1)
        rank_tensor[valid_rows, w] = ranks - 0.5    # center
    edu_idx = _edu_indices_3d(p)
    for i in edu_idx:
        p["X"][:, :, i] = np.where(p["mask"] > 0, rank_tensor, 0.0)
    return p, "edu_replaced_with_gdp_rank"


PLACEBOS = [
    placebo_outcome_scramble,
    placebo_year_scramble,
    placebo_country_scramble,
    placebo_edu_replaced_noise,
    placebo_edu_replaced_latitude,
    placebo_edu_replaced_gdprank,
]


def run_placebos(methods=None, placebos=None, seeds=(42, 43, 44),
                  verbose=True):
    methods = methods or PLACEBO_METHODS
    placebos = placebos or PLACEBOS

    base = load_panel(mode="joint", verbose=verbose)
    flat_groups = flatten_feature_groups(base)

    out = {}
    t_total = time.time()
    for placebo_fn in placebos:
        # Use only first seed for non-stochastic placebos (cheaper);
        # use all seeds for stochastic ones (outcome_scramble, year_scramble,
        # edu_replaced_noise, country_scramble).
        is_stochastic = placebo_fn in (
            placebo_outcome_scramble, placebo_year_scramble,
            placebo_country_scramble, placebo_edu_replaced_noise,
        )
        seeds_to_use = seeds if is_stochastic else (seeds[0],)
        rows = []
        for s in seeds_to_use:
            p, name = placebo_fn(base, seed=s)
            if verbose:
                print(f"\n[{name}] seed={s}")
            for m in methods:
                r = run_one_method(p, flat_groups, m, verbose=False)
                rows.append({
                    "placebo": name,
                    "seed": s,
                    "method": m,
                    "r2_baseline": r["r2_baseline"],
                    "r2_drop_absolute": r["r2_drop_absolute"],
                    "r2_drop_abs_mean": r["r2_drop_abs_mean"],
                })
                if verbose:
                    drops = r["r2_drop_absolute"]
                    print(f"  {m:<22} drop: LE={drops['LE']:+.3f} "
                          f"TFR={drops['TFR']:+.3f} U5MR={drops['U5MR']:+.3f} "
                          f"mean={r['r2_drop_abs_mean']:+.3f}")
        out[placebo_fn.__name__] = rows

    return {
        "method": (
            "Six placebo transformations of the panel. Each placebo breaks "
            "a different aspect of the education-outcomes link. If the "
            "education R² drop measured by the spec curve is genuinely "
            "education's contribution (not an artifact), every placebo "
            "should return an R² drop near zero."
        ),
        "methods": list(methods),
        "seeds": list(seeds),
        "placebo_rows": out,
        "total_elapsed_sec": time.time() - t_total,
    }


def run_placebos_parent(methods=None, placebos=None, seeds=(42, 43, 44),
                         verbose=True):
    """Parent-vantage placebo suite. Mirrors run_placebos but operates on the
    three biological-lag single-target base panels (LE@T+12, TFR@T+5,
    U5MR@T+12) instead of the single T+25 joint panel. Each placebo is applied
    to each single-target base panel and scored with the same spec-curve
    methods; results are keyed by target then by placebo function name.

    The base panels come from data_loader_parent_lag.load_parent_lag_panels,
    so the row coverage and feature pipeline are identical to the joint placebo
    run, varying only the target horizon (apples-to-apples).
    """
    from data_loader_parent_lag import load_parent_lag_panels

    methods = methods or PLACEBO_METHODS
    placebos = placebos or PLACEBOS

    base_panels = load_parent_lag_panels(mode="joint", verbose=verbose)

    by_target = {}
    t_total = time.time()
    for tname, base in base_panels.items():
        flat_groups = flatten_feature_groups(base)
        out = {}
        for placebo_fn in placebos:
            # Same stochastic/deterministic seed policy as run_placebos.
            is_stochastic = placebo_fn in (
                placebo_outcome_scramble, placebo_year_scramble,
                placebo_country_scramble, placebo_edu_replaced_noise,
            )
            seeds_to_use = seeds if is_stochastic else (seeds[0],)
            rows = []
            for s in seeds_to_use:
                p, name = placebo_fn(base, seed=s)
                if verbose:
                    print(f"\n[{tname}][{name}] seed={s}")
                for m in methods:
                    r = run_one_method(p, flat_groups, m, verbose=False)
                    rows.append({
                        "target": tname,
                        "placebo": name,
                        "seed": s,
                        "method": m,
                        # Single-target panel: r2_baseline / r2_drop_absolute
                        # are keyed by this one target name only.
                        "r2_baseline": r["r2_baseline"],
                        "r2_drop_absolute": r["r2_drop_absolute"],
                        "r2_drop_abs_mean": r["r2_drop_abs_mean"],
                    })
                    if verbose:
                        print(f"  {m:<22} drop_mean={r['r2_drop_abs_mean']:+.3f}")
            out[placebo_fn.__name__] = rows
        by_target[tname] = out

    return {
        "method": (
            "Parent-vantage placebo suite. Six placebo transformations applied "
            "to each biological-lag single-target panel (LE@T+12, TFR@T+5, "
            "U5MR@T+12). If the spec-curve education R² drop is genuinely "
            "education's contribution, every placebo should return an R² drop "
            "near zero for every outcome."
        ),
        "vantage": "parent",
        "methods": list(methods),
        "seeds": list(seeds),
        "placebo_rows_by_target": by_target,
        "total_elapsed_sec": time.time() - t_total,
    }


def main():
    out = run_placebos()
    write_checkin("placebos.json", out, base_dir=ML_CHECKIN,
                  script_path="scripts/ml/chapter9/placebos.py")
    print(f"\n=== placebo suite complete in {out['total_elapsed_sec']/60:.1f} min ===")


if __name__ == "__main__":
    main()
