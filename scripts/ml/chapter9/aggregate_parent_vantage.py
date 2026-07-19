"""
aggregate_parent_vantage.py
===========================

Aggregate the cloud parent-vantage transformer jobs (group B:
trf_parent_{LE,TFR,U5MR}_s{seed}, 30 seeds x 3 outcomes) into the canonical
checkin `universal_evidence_parent_lag.json`, averaging across seeds.

Per-outcome biological horizons (data_loader_parent_lag.PARENT_LAG_HORIZONS):
  LE @ T+12 (LAG_LE — LE at birth is mortality-dominated, shares the
  childrearing window), TFR @ T+5, U5MR @ T+12 (childrearing window).

Reads:  scripts/ml/checkin/panel_runs/trf_parent_*/result.json
Writes: scripts/ml/checkin/universal_evidence_parent_lag.json

The output keeps the key structure consumed by compare_parent_lag_vs_t25.py
(per_outcome[t]["B0_joint_5fold"]["overall_r2_oof"][0],
 per_outcome[t]["B1_B2_gates_on_joint"]["g1_education_ablation"]
              ["r2_drop_fraction"][t]) and adds seed-level mean/std.

Run: python scripts/ml/chapter9/aggregate_parent_vantage.py
"""

import json
import os
import sys
import glob

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(ML_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, ML_DIR)

from _shared import write_checkin, ML_CHECKIN          # noqa: E402
from data_loader_parent_lag import PARENT_LAG_HORIZONS  # noqa: E402

PANEL_RUNS = os.path.join(ML_CHECKIN, "panel_runs")
TARGETS = ["LE", "TFR", "U5MR"]


def _load_seeds(target):
    """Load every trf_parent_<target>_s* result.json."""
    pattern = os.path.join(PANEL_RUNS, f"trf_parent_{target}_s*", "result.json")
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            out.append(json.load(fh))
    return out


def _mean_std(vals):
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def aggregate_target(target):
    results = _load_seeds(target)
    if not results:
        raise SystemExit(f"No cloud results for trf_parent_{target}_s* in {PANEL_RUNS}")

    r2_oof = [r["summary"]["overall_r2_oof"][0] for r in results]
    fold_means = [r["summary"]["fold_r2_mean"][0] for r in results]
    baseline = [r["gates"]["baseline_r2"][target] for r in results]
    edu_ablated = [r["gates"]["g1_education_ablation"]["ablated_r2"][target] for r in results]
    edu_dropfrac = [r["gates"]["g1_education_ablation"]["r2_drop_fraction"][target] for r in results]
    gdp_ablated = [r["gates"]["g2_gdp_ablation"]["ablated_r2"][target] for r in results]
    gdp_dropfrac = [r["gates"]["g2_gdp_ablation"]["r2_drop_fraction"][target] for r in results]
    n_samples = results[0]["summary"]["n_samples"]

    r2_m, r2_s = _mean_std(r2_oof)
    base_m, _ = _mean_std(baseline)
    edu_ab_m, _ = _mean_std(edu_ablated)
    edu_df_m, edu_df_s = _mean_std(edu_dropfrac)
    gdp_ab_m, _ = _mean_std(gdp_ablated)
    gdp_df_m, _ = _mean_std(gdp_dropfrac)

    return {
        "horizon": PARENT_LAG_HORIZONS[target],
        "n_seeds": len(results),
        "B0_joint_5fold": {
            "overall_r2_oof": [r2_m],
            "overall_r2_oof_std": [r2_s],
            "fold_r2_mean": [float(np.mean(fold_means))],
            "n_samples": n_samples,
        },
        "B1_B2_gates_on_joint": {
            "n_evaluated": n_samples,
            "baseline_r2": {target: base_m},
            "g1_education_ablation": {
                "ablated_r2": {target: edu_ab_m},
                "r2_drop_fraction": {target: edu_df_m},
                "r2_drop_fraction_std": {target: edu_df_s},
                "r2_drop_absolute": {target: base_m - edu_ab_m},
                "max_drop": edu_df_m,
            },
            "g2_gdp_ablation": {
                "ablated_r2": {target: gdp_ab_m},
                "r2_drop_fraction": {target: gdp_df_m},
                "r2_drop_absolute": {target: base_m - gdp_ab_m},
                "max_drop": gdp_df_m,
            },
        },
    }


def main():
    per_outcome = {t: aggregate_target(t) for t in TARGETS}
    out = {
        "method": (
            "Parent-vantage UniversalTransformer (no country embedding), "
            "30-seed cloud aggregate (mean across seeds). Three SEPARATE "
            "single-target models, one per outcome at its biological horizon: "
            "LE at T+12 (childrearing window) (LAG_LE), TFR at T+5 "
            "(LAG_TFR), U5MR at T+12 (LAG_CHILDREARING). Same encoder, feature "
            "pipeline and fold construction as the T+25 multi-target build; "
            "only the target horizon varies. Education / GDP ablation gates "
            "averaged across seeds."
        ),
        "horizons": {t: PARENT_LAG_HORIZONS[t] for t in TARGETS},
        "n_seeds": per_outcome[TARGETS[0]]["n_seeds"],
        "per_outcome": per_outcome,
    }
    write_checkin("universal_evidence_parent_lag.json", out, base_dir=ML_CHECKIN,
                  script_path="scripts/ml/chapter9/aggregate_parent_vantage.py")

    print("\nPARENT-VANTAGE TRANSFORMER (30-seed cloud aggregate)")
    print(f"{'outcome':<8} {'horizon':>7} {'OOF R²':>10} {'edu drop':>10} "
          f"{'gdp drop':>10} {'n':>6}")
    for t in TARGETS:
        po = per_outcome[t]
        b0 = po["B0_joint_5fold"]
        g = po["B1_B2_gates_on_joint"]
        print(f"{t:<8} {po['horizon']:>7} {b0['overall_r2_oof'][0]:>10.3f} "
              f"{g['g1_education_ablation']['r2_drop_fraction'][t]:>10.3f} "
              f"{g['g2_gdp_ablation']['r2_drop_fraction'][t]:>10.3f} "
              f"{b0['n_samples']:>6}")


if __name__ == "__main__":
    main()
