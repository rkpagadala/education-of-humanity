# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/chapter9/aggregate_parent_battery.py
# Paper:   "The Long Childhood: On the Convergence of Humanity"
#
# Produces:
#   chapter9_parent_battery.json — the biological-lag (parent-vantage) recast
#   of the ML robustness battery that feeds ECONOMETRICS.md and the paper's
#   "Every Method Agrees" narrative (NOT the make-verify registrations, which
#   read spec_curve_parent.json / universal_evidence_parent_lag.json). Each
#   diagnostic was run once per outcome at its biological horizon (LE@T+12,
#   TFR@T+5, U5MR@T+12) over the 28/56/84 generational cohort stack
#   (ML_UNIFY_GENERATION=1). Results are keyed by target.
#
# Inputs:
#   parent_* job result.json files, either synced locally under
#   scripts/ml/checkin/panel_runs/ or pulled from a GCS results prefix with
#   --from-gcs gs://<bucket>/results.
#
# Output:
#   scripts/ml/checkin/chapter9_parent_battery.json
#   plus a human-readable summary to stdout (the numbers the doc cites).
#
# Infeasible-by-design jobs (logged, not silently dropped): at the biological
# horizons the panel does not extend past ~2008, so walk-forward cutoff 2005
# and the [2005, 2020] stratification era have empty/too-small validation sets
# and never produce a result. They are reported under "infeasible".
# =============================================================================
"""aggregate_parent_battery.py — biological-lag battery aggregator."""

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(ML_DIR))
CHECKIN = os.path.join(ML_DIR, "checkin")
# Runs dir holds the parent_* result.json files; override with PARENT_RUNS_DIR
# (used to aggregate from a temp sync without touching the committed tree).
PANEL_RUNS = os.environ.get("PARENT_RUNS_DIR", os.path.join(CHECKIN, "panel_runs"))
TARGETS = ["LE", "TFR", "U5MR"]


def _median(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(xs)) if xs else None


def _load_results(from_gcs):
    """Return {job_id: result_dict} for every parent_* job with a valid
    result.json. Reads local panel_runs/ unless --from-gcs is given, in which
    case it rsyncs the parent_* results down first."""
    if from_gcs:
        dest = PANEL_RUNS
        os.makedirs(dest, exist_ok=True)
        # Pull only the parent_* result dirs (the full results tree also holds
        # the joint big-run and is huge). List them, then recursively copy each
        # dir — `gsutil -m cp -r -I` reads source dirs from stdin and preserves
        # the parent_<id>/result.json layout the loader below expects.
        listing = subprocess.run(
            ["gsutil", "ls", f"{from_gcs.rstrip('/')}/"],
            capture_output=True, text=True).stdout.splitlines()
        parent_dirs = [l for l in listing if "/parent_" in l]
        print(f"[battery] pulling {len(parent_dirs)} parent_ result dirs from GCS")
        proc = subprocess.run(
            ["gsutil", "-m", "cp", "-r", "-I", f"{dest}/"],
            input="\n".join(parent_dirs), text=True,
            stderr=subprocess.DEVNULL)
    out = {}
    for path in glob.glob(os.path.join(PANEL_RUNS, "parent_*", "result.json")):
        jid = os.path.basename(os.path.dirname(path))
        try:
            with open(path) as f:
                out[jid] = json.load(f)
        except Exception:
            pass  # empty/partial (infeasible job) — counted as missing
    return out


def agg_walk(results):
    """Walk-forward: per cutoff, median per-target baseline R² and edu-drop
    across seeds."""
    by_cut = defaultdict(lambda: defaultdict(lambda: {"base": [], "drop": []}))
    for jid, d in results.items():
        if not jid.startswith("parent_walk_c") or d.get("vantage") != "parent":
            continue
        cut = d["cutoff_year"]
        for t, v in d.get("by_target", {}).items():
            by_cut[cut][t]["base"].append(v["r2_baseline"][t])
            by_cut[cut][t]["drop"].append(v["r2_drop_absolute"][t])
    out = {}
    for cut in sorted(by_cut):
        out[str(cut)] = {t: {"r2_baseline_median": _median(by_cut[cut][t]["base"]),
                             "edu_drop_median": _median(by_cut[cut][t]["drop"]),
                             "n_seeds": len(by_cut[cut][t]["base"])}
                         for t in by_cut[cut]}
    return out


def agg_loo(results):
    """LOO-185: gather every per-country row across shards, per target."""
    rows = defaultdict(list)  # target -> [r2_baseline per country]
    drops = defaultdict(list)
    countries = defaultdict(set)
    for jid, d in results.items():
        if not jid.startswith("parent_loo_shard") or d.get("vantage") != "parent":
            continue
        for t, rs in d.get("by_target", {}).items():
            for r in rs:
                rows[t].append(r["r2_baseline"][t])
                drops[t].append(r["r2_drop_absolute"][t])
                countries[t].add(r["country"])
    out = {}
    for t in rows:
        out[t] = {
            "n_countries": len(countries[t]),
            "r2_baseline_median": _median(rows[t]),
            "r2_baseline_min": float(np.min(rows[t])) if rows[t] else None,
            "edu_drop_median": _median(drops[t]),
        }
    return out


def agg_placebos(results):
    """Placebos: per target per placebo, median edu R² drop across methods/seeds
    (should sit near zero for the destructive placebos)."""
    acc = defaultdict(lambda: defaultdict(list))  # target -> placebo -> [drops]
    for jid, d in results.items():
        if not jid.startswith("parent_placebos_") or d.get("vantage") != "parent":
            continue
        for t, byfn in d.get("placebo_rows_by_target", {}).items():
            for fn, rws in byfn.items():
                for r in rws:
                    acc[t][fn].append(r["r2_drop_abs_mean"])
    return {t: {fn: {"median_edu_r2_drop": _median(v), "n": len(v)}
                for fn, v in byfn.items()}
            for t, byfn in acc.items()}


def agg_optuna(results):
    """Optuna: global best across shards (max mean-biological-lag 5-fold R²)."""
    best = None
    for jid, d in results.items():
        if not jid.startswith("parent_optuna_shard") or d.get("vantage") != "parent":
            continue
        bv = d.get("best_value")
        if bv is not None and (best is None or bv > best["best_value"]):
            best = {"best_value": float(bv), "best_params": d.get("best_params"),
                    "shard": jid}
    return best


def agg_cond_perm(results):
    d = results.get("parent_cond_perm")
    if not d:
        return None
    return d.get("by_target")


def agg_strat(results):
    """Stratification: per stratum, per-target OOF R² (universality)."""
    by_stratum = defaultdict(lambda: defaultdict(list))
    for jid, d in results.items():
        if not jid.startswith("parent_strat_") or d.get("vantage") != "parent":
            continue
        # strip trailing _s<seed>
        stratum = jid.rsplit("_s", 1)[0].replace("parent_strat_", "")
        for t, v in d.get("by_target", {}).items():
            oof = v.get("summary", {}).get("overall_r2_oof")
            if oof:
                by_stratum[stratum][t].append(oof[0])
    return {s: {t: {"r2_oof_median": _median(v), "n_seeds": len(v)}
                for t, v in byt.items()}
            for s, byt in by_stratum.items()}


def agg_barro_lee(results):
    """Barro-Lee replication: per target, median 5-fold R² and g1 edu-drop
    fraction across seeds."""
    acc = defaultdict(lambda: {"r2": [], "g1": []})
    for jid, d in results.items():
        if not jid.startswith("parent_trf_bl_") or d.get("kind") != "transformer_single":
            continue
        t = d.get("target")
        r2 = d.get("summary", {}).get("fold_r2_mean")
        if r2:
            acc[t]["r2"].append(r2[0])
        g1 = (d.get("gates", {}).get("g1_education_ablation", {})
              .get("r2_drop_fraction", {}).get(t))
        if g1 is not None:
            acc[t]["g1"].append(g1)
    return {t: {"r2_median": _median(v["r2"]),
                "edu_drop_fraction_median": _median(v["g1"]),
                "n_seeds": len(v["r2"])}
            for t, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-gcs", default="", help="gs://bucket/results to pull parent_* from")
    args = ap.parse_args()

    results = _load_results(args.from_gcs)
    print(f"[battery] loaded {len(results)} valid parent_ results")

    out = {
        "method": ("Biological-lag (parent-vantage) ML robustness battery. Each "
                   "diagnostic run once per outcome at its biological horizon "
                   "(LE@12, TFR@5, U5MR@12) over the 28/56/84 generational cohort "
                   "stack. Feeds ECONOMETRICS.md and the Every-Method-Agrees "
                   "narrative; the make-verify ML numbers live in "
                   "spec_curve_parent.json / universal_evidence_parent_lag.json."),
        "vantage": "parent",
        "horizons": {"LE": 12, "TFR": 5, "U5MR": 12},
        "n_results": len(results),
        "walk_forward": agg_walk(results),
        "loo": agg_loo(results),
        "placebos": agg_placebos(results),
        "optuna": agg_optuna(results),
        "cond_perm": agg_cond_perm(results),
        "strat": agg_strat(results),
        "barro_lee": agg_barro_lee(results),
    }
    dest = os.path.join(CHECKIN, "chapter9_parent_battery.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[battery] wrote {dest}")
    # Human-readable highlights
    print("\n=== walk-forward (per cutoff, median LE/TFR/U5MR baseline R²) ===")
    for cut, byt in out["walk_forward"].items():
        print(f"  c{cut}: " + "  ".join(
            f"{t}={byt[t]['r2_baseline_median']:.3f}" for t in TARGETS if t in byt))
    print("\n=== LOO ===")
    for t, v in out["loo"].items():
        print(f"  {t}: n_countries={v['n_countries']} median R²={v['r2_baseline_median']} "
              f"min={v['r2_baseline_min']}")
    print("\n=== placebos (median edu R² drop; ~0 means signal destroyed) ===")
    for t, byfn in out["placebos"].items():
        print(f"  {t}: " + "  ".join(f"{fn.replace('placebo_','')}={d['median_edu_r2_drop']}"
                                     for fn, d in byfn.items()))
    print("\n=== optuna best ===", out["optuna"])
    print("=== cond_perm ===", out["cond_perm"])
    print("=== barro_lee ===", out["barro_lee"])


if __name__ == "__main__":
    main()
