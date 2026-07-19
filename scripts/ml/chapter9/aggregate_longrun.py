# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/chapter9/aggregate_longrun.py
#
# Aggregates longrun job results (20 jobs: 4 kinds × 5 seeds) from a local
# directory or GCS bucket. Reads whatever result.json files are present,
# medians across seeds, and writes:
#
#   checkin/longrun_within.json
#   checkin/longrun_forward.json
#
# Handles 2-target within results (LE, U5MR) and 3-target forward results
# (LE, U5MR, TFR) by iterating targets present in the data, NOT a hardcoded
# 3-list. If a method has no result files in the input directory, an empty
# dict is emitted for that method rather than raising.
#
# Schemas consumed:
#
#   transformer_longrun_within / transformer_longrun_forward:
#     {"kind":..., "n_samples":..., "n_countries":...,
#      "r2_baseline": {"LE":f, "U5MR":f[, "TFR":f]},
#      "r2_ablated_edu": {...}, "r2_drop_absolute": {...}}
#
#   panel_fe_longrun_within:
#     {"kind":..., "n_samples":..., "n_countries":...,
#      "ols_by_target": {"LE": {"beta_edu":f, "cluster_se":f, "r2_oof":f, ...}, ...},
#      "gbm_by_target": {"LE": {"r2_baseline":f, "r2_ablated_edu":f, "r2_drop":f}, ...}}
#
#   panel_fe_longrun_forward:
#     {"kind":..., "n_train":..., "n_test":...,
#      "ols_by_target": {"LE": {"beta_edu":f, "r2_oof_forward":f, ...}, ...},
#      "gbm_by_target": {"LE": {"r2_baseline":f, "r2_ablated_edu":f, "r2_drop":f}, ...}}
#     NOTE: forward OLS uses "r2_oof_forward" not "r2_oof"; both handled via .get().
#
# Output JSON shape:
#
#   {"mode": "within"|"forward",
#    "by_method": {
#      "transformer": {
#        "<target>": {"r2_baseline_median": f, "r2_baseline_range": [min,max],
#                     "r2_drop_median": f, "n_seeds": i}
#      },
#      "panel_fe": {
#        "<target>": {"ols_beta_edu_median": f, "ols_cluster_se_median": f,
#                     "ols_r2_median": f, "gbm_r2_median": f,
#                     "gbm_drop_median": f, "n_seeds": i}
#      }
#    }}
# =============================================================================
"""
aggregate_longrun.py — median across seeds for all four longrun job kinds.
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load_results(dir_):
    """Load all result.json files found one level deep under dir_."""
    rows = []
    for path in sorted(glob.glob(os.path.join(dir_, "*", "result.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def _median(xs):
    """Median of a list, filtering None/non-finite. Returns None if no values."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return statistics.median(xs)


def _range_or_none(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return [min(xs), max(xs)]


# ---------------------------------------------------------------------------
# Transformer aggregation (r2_baseline, r2_drop_absolute dicts)
# ---------------------------------------------------------------------------

def _aggregate_transformer(rows, kind):
    """Aggregate transformer results for one kind.

    Returns dict keyed by target (whatever is present in the data), each value:
      {"r2_baseline_median": f, "r2_baseline_range": [min,max],
       "r2_drop_median": f, "n_seeds": i}

    Returns {} if no matching rows.
    """
    matching = [r for r in rows if r.get("kind") == kind]
    if not matching:
        return {}

    # Discover targets present in any matching row
    all_targets = set()
    for r in matching:
        all_targets.update(r.get("r2_baseline", {}).keys())

    by_target = {}
    for tgt in sorted(all_targets):
        r2_vals = [
            r["r2_baseline"][tgt]
            for r in matching
            if tgt in r.get("r2_baseline", {})
        ]
        drop_vals = [
            r["r2_drop_absolute"][tgt]
            for r in matching
            if tgt in r.get("r2_drop_absolute", {})
        ]
        by_target[tgt] = {
            "r2_baseline_median": _median(r2_vals),
            "r2_baseline_range":  _range_or_none(r2_vals),
            "r2_drop_median":     _median(drop_vals),
            "n_seeds":            len(r2_vals),
        }
    return by_target


# ---------------------------------------------------------------------------
# Panel-FE aggregation (ols_by_target + gbm_by_target dicts)
# ---------------------------------------------------------------------------

def _aggregate_panel_fe(rows, kind):
    """Aggregate panel_fe results for one kind.

    Handles both within (ols r2 key = "r2_oof") and forward (key = "r2_oof_forward")
    via d.get("r2_oof", d.get("r2_oof_forward")).

    Returns dict keyed by target, each value:
      {"ols_beta_edu_median": f, "ols_cluster_se_median": f,
       "ols_r2_median": f, "gbm_r2_median": f,
       "gbm_drop_median": f, "n_seeds": i}

    Returns {} if no matching rows.
    """
    matching = [r for r in rows if r.get("kind") == kind]
    if not matching:
        return {}

    # Discover targets present in ols_by_target of any matching row
    all_targets = set()
    for r in matching:
        all_targets.update(r.get("ols_by_target", {}).keys())

    by_target = {}
    for tgt in sorted(all_targets):
        # OLS fields
        ols_rows = [
            r["ols_by_target"][tgt]
            for r in matching
            if tgt in r.get("ols_by_target", {})
        ]
        beta_vals      = [d["beta_edu"] for d in ols_rows if "beta_edu" in d]
        cluster_se_vals = [d["cluster_se"] for d in ols_rows if "cluster_se" in d]
        # r2: within uses "r2_oof", forward uses "r2_oof_forward"
        r2_ols_vals = [
            d.get("r2_oof", d.get("r2_oof_forward"))
            for d in ols_rows
        ]
        r2_ols_vals = [v for v in r2_ols_vals if v is not None]

        # GBM fields
        gbm_rows = [
            r["gbm_by_target"][tgt]
            for r in matching
            if tgt in r.get("gbm_by_target", {})
        ]
        gbm_r2_vals   = [d["r2_baseline"] for d in gbm_rows if "r2_baseline" in d]
        gbm_drop_vals = [d["r2_drop"] for d in gbm_rows if "r2_drop" in d]

        by_target[tgt] = {
            "ols_beta_edu_median":  _median(beta_vals),
            "ols_cluster_se_median": _median(cluster_se_vals),
            "ols_r2_median":        _median(r2_ols_vals),
            "gbm_r2_median":        _median(gbm_r2_vals),
            "gbm_drop_median":      _median(gbm_drop_vals),
            "n_seeds":              len(ols_rows),
        }
    return by_target


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate_within_from_dir(dir_):
    """Aggregate all within-mode results from a local directory.

    Returns the within-mode output dict (does NOT write any files).
    """
    rows = _load_results(dir_)
    return {
        "mode": "within",
        "by_method": {
            "transformer": _aggregate_transformer(rows, "transformer_longrun_within"),
            "panel_fe":    _aggregate_panel_fe(rows,    "panel_fe_longrun_within"),
        },
    }


def aggregate_forward_from_dir(dir_):
    """Aggregate all forward-mode results from a local directory.

    Returns the forward-mode output dict (does NOT write any files).
    """
    rows = _load_results(dir_)
    return {
        "mode": "forward",
        "by_method": {
            "transformer": _aggregate_transformer(rows, "transformer_longrun_forward"),
            "panel_fe":    _aggregate_panel_fe(rows,    "panel_fe_longrun_forward"),
        },
    }


# ---------------------------------------------------------------------------
# GCS helper
# ---------------------------------------------------------------------------

def _gcs_pull(bucket_prefix, dest_dir):
    """Pull all result.json files from a GCS prefix into dest_dir."""
    cmd = ["gsutil", "-m", "cp", "-r", f"{bucket_prefix}/*", dest_dir]
    subprocess.check_call(cmd)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Aggregate longrun job results and write "
            "checkin/longrun_within.json + checkin/longrun_forward.json"
        )
    )
    ap.add_argument(
        "--from-gcs", default="",
        help="GCS prefix, e.g. gs://YOUR_BUCKET/results",
    )
    ap.add_argument(
        "--from-dir", default="",
        help="Local directory of <jobid>/result.json subdirs",
    )
    ap.add_argument(
        "--out-dir", default=os.path.join(REPO_ROOT, "checkin"),
        help="Output directory for the two checkin JSONs",
    )
    args = ap.parse_args()

    if not args.from_gcs and not args.from_dir:
        print("ERROR: provide --from-gcs or --from-dir")
        sys.exit(2)

    with tempfile.TemporaryDirectory() as td:
        if args.from_gcs:
            print(f"Pulling from GCS: {args.from_gcs}")
            _gcs_pull(args.from_gcs, td)
            src = td
        else:
            src = args.from_dir

        within  = aggregate_within_from_dir(src)
        forward = aggregate_forward_from_dir(src)

    os.makedirs(args.out_dir, exist_ok=True)

    within_path = os.path.join(args.out_dir, "longrun_within.json")
    forward_path = os.path.join(args.out_dir, "longrun_forward.json")

    with open(within_path, "w") as f:
        json.dump(within, f, indent=2, default=float)
    with open(forward_path, "w") as f:
        json.dump(forward, f, indent=2, default=float)

    print(f"Wrote {within_path}")
    print(f"Wrote {forward_path}")


if __name__ == "__main__":
    main()
