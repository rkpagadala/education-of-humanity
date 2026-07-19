# =============================================================================
# scripts/ml/chapter9/aggregate_results.py
#
# Reads merged per-job results from a Panel cloud run (downloaded via
# launch.py aggregate) and produces:
#
#   1. spec_curve_table.json        — final method × outcome table with CIs
#   2. placebo_nulls.json           — every placebo's R² drop (should be ~0)
#   3. walk_forward.json            — train-on-past, test-on-future R²
#   4. headline_numbers.json        — the 6-8 numbers the Panel will cite
#
# All outputs land in checkin/ and feed verify_the_long_childhood.py.
# =============================================================================
"""
aggregate_results.py — Panel result post-processing.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(ML_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from _shared import write_checkin, ML_CHECKIN

PANEL_RUNS_DIR = os.path.join(ML_DIR, "checkin", "panel_runs")
TARGETS = ["LE", "TFR", "U5MR"]

# Linear estimators that are NOT valid out-of-sample cross-country predictors
# under country-clustered CV, and so are excluded from the spec curve and the
# placebo nulls (see scripts/ml/chapter9/spec_curve.py header):
#   - ols_within_fe: a held-out country's fixed effect is unidentified, so the
#       out-of-fold R² diverges (≈ -53 baseline; placebo drops to -8/+65).
#   - ols_pooled:    unregularized OLS with ~500 features overfits out-of-fold;
#       ablating education sends the ablated R² strongly negative and the
#       education "drop" past 1.0 (uninterpretable).
# The linear class is represented by its regularized members (ridge, lasso).
# The in-sample within-FE β lives in the headline panel table instead.
EXCLUDE_METHODS = {"ols_within_fe", "ols_pooled"}


def _percentile(values, q):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def _ci95(values):
    return [_percentile(values, 2.5), _percentile(values, 97.5)]


def aggregate_transformer_t25(merged):
    """Aggregate the 15-seed T+25 transformer runs into baseline R² and
    education R² drop per target, with seed-CI bands."""
    rows = []
    for jid, jres in merged.items():
        if not jid.startswith("trf_t25_s"):
            continue
        if "gates" not in jres or "summary" not in jres:
            continue
        gates = jres["gates"]
        baseline = gates["baseline_r2"]
        edu_drop = gates["g1_education_ablation"]["ablated_r2"]
        for t in TARGETS:
            rows.append({
                "job_id": jid, "target": t,
                "baseline_r2": baseline[t],
                "ablated_r2": edu_drop[t],
                "edu_r2_drop_abs": baseline[t] - edu_drop[t],
            })
    by_target = defaultdict(list)
    for r in rows:
        by_target[r["target"]].append(r)
    out = {}
    for t in TARGETS:
        rs = by_target[t]
        if not rs:
            continue
        baselines = [r["baseline_r2"] for r in rs]
        drops = [r["edu_r2_drop_abs"] for r in rs]
        out[t] = {
            "n_seeds": len(rs),
            "baseline_r2_median": float(np.median(baselines)),
            "baseline_r2_ci95": _ci95(baselines),
            "edu_r2_drop_median": float(np.median(drops)),
            "edu_r2_drop_ci95": _ci95(drops),
        }
    return out


def aggregate_spec_curve(merged):
    """Merge per-method spec-curve shards (id `spec_curve_<method>`) into
    method × target rows of baseline R² and education R² drop."""
    rows = []
    for jid, jres in merged.items():
        if not jid.startswith("spec_curve_"):
            continue
        results = jres.get("results", {})
        for method, r in results.items():
            if method in EXCLUDE_METHODS:
                continue
            for t in TARGETS:
                rows.append({
                    "method": method, "target": t,
                    "baseline_r2": r["r2_baseline"][t],
                    "edu_r2_drop_abs": r["r2_drop_absolute"][t],
                })
    return rows


def aggregate_placebos(merged):
    """Merge per-placebo shards (id `placebos_<name>`) into per-placebo null
    summaries. Headline: max drop across all placebos should be ~0."""
    nulls = defaultdict(list)
    for jid, jres in merged.items():
        if not jid.startswith("placebos_"):
            continue
        for placebo_name, rows in jres.get("placebo_rows", {}).items():
            for r in rows:
                if r["method"] in EXCLUDE_METHODS:
                    continue
                for t in TARGETS:
                    nulls[placebo_name].append({
                        "method": r["method"], "target": t,
                        "edu_r2_drop_abs": r["r2_drop_absolute"][t],
                    })
    summary = {}
    for placebo_name, rows in nulls.items():
        drops = [row["edu_r2_drop_abs"] for row in rows]
        summary[placebo_name] = {
            "n_rows": len(rows),
            "median_drop": float(np.median(drops)),
            "max_abs_drop": float(np.max(np.abs(drops))),
            "drop_ci95": _ci95(drops),
        }
    return summary


def aggregate_dml(merged):
    """Merge per-seed DML shards (id `dml_s<seed>`): pool the per-seed theta
    estimates per target and recompute the median + 95% CI across all seeds."""
    per_seed = {t: [] for t in TARGETS}
    for jid, jres in merged.items():
        if not jid.startswith("dml_s") or jres.get("kind") != "dml":
            continue
        psp = jres.get("per_seed_per_target", {})
        for t in TARGETS:
            per_seed[t].extend(psp.get(t, []))
    out = {}
    for t in TARGETS:
        thetas = [d["theta_mean"] for d in per_seed[t]]
        if not thetas:
            continue
        out[t] = {
            "n_seeds": len(thetas),
            "theta_seed_estimates": thetas,
            "theta_median": float(np.median(thetas)),
            "theta_ci95": _ci95(thetas),
        }
    return out


def aggregate_cond_perm(merged):
    """Merge per-target conditional-permutation shards (id `cond_perm_<T>`)."""
    out = {}
    for jid, jres in merged.items():
        if not jid.startswith("cond_perm_") or jres.get("kind") != "cond_perm":
            continue
        for t, s in jres.get("summary", {}).items():
            out[t] = s
    return out


def aggregate_walk_forward(merged):
    """Walk-forward R² by cutoff year × target."""
    by_cutoff = defaultdict(list)
    for jid, jres in merged.items():
        if not jid.startswith("walk_c"):
            continue
        if "summary" not in jres:
            continue
        cutoff = int(jid.split("_")[1].lstrip("c"))
        for i, t in enumerate(TARGETS):
            by_cutoff[(cutoff, t)].append(jres["summary"]["fold_r2_mean"][i])
    out = {}
    for (cutoff, t), values in by_cutoff.items():
        out[f"{cutoff}_{t}"] = {
            "cutoff": cutoff, "target": t,
            "n_seeds": len(values),
            "r2_median": float(np.median(values)),
            "r2_ci95": _ci95(values),
        }
    return out


def aggregate_loo(merged):
    """Merge LOO shards into one 185-country table of per-country R² and
    education R² drop. Shards are disjoint (round-robin over countries), so
    concatenation reconstructs the full set; dedup by country guards against
    any accidental overlap."""
    by_country = {}
    shards_seen = set()
    for jid, jres in merged.items():
        if not jid.startswith("loo_shard_"):
            continue
        if jres.get("kind") != "loo_transformer":
            continue
        shards_seen.add(jres.get("shard_idx"))
        for rec in jres.get("per_country", []):
            by_country[rec["country"]] = rec
    if not by_country:
        return {}
    recs = list(by_country.values())

    # "Out of spec" = the model (trained on every OTHER country) mispredicts
    # this country. Per-country R² is unstable on ~6 points, so we rank on
    # natural-unit residuals. The three targets have different scales
    # (LE-years vs TFR-children vs U5MR-per-1000), so each country's
    # mean-absolute-residual is normalized by the cross-country median for
    # that target, then averaged — a unitless "x times the typical miss".
    have_resid = any("mean_abs_resid" in r for r in recs)
    if have_resid:
        # Per-target scale = median absolute residual across countries. Use
        # nanmedian: a few countries have missing actuals for some target
        # (nan residual), and a plain median would propagate nan into every
        # country's score.
        scale = {}
        for t in TARGETS:
            vals = [r.get("mean_abs_resid", {}).get(t) for r in recs]
            vals = [v for v in vals if v is not None]
            arr = np.array(vals, dtype=float)
            med = float(np.nanmedian(arr)) if arr.size else 1.0
            scale[t] = med if (med and not np.isnan(med)) else 1.0

        def oos_score(rec):
            mar = rec.get("mean_abs_resid", {})
            zs = [mar[t] / scale[t] for t in TARGETS
                  if mar.get(t) is not None and np.isfinite(mar[t])
                  and scale[t] > 0]
            return float(np.mean(zs)) if zs else float("nan")

        for r in recs:
            r["out_of_spec_score"] = oos_score(r)
        # Sort by score descending, nan scores (all-missing countries) last.
        recs.sort(key=lambda r: (r["out_of_spec_score"]
                                 if np.isfinite(r["out_of_spec_score"])
                                 else -np.inf), reverse=True)
        return {
            "n_countries": len(recs),
            "n_shards": len(shards_seen),
            "target_scale_median_abs_resid": scale,
            "ranking_metric": "mean over targets of (country mean_abs_resid / "
                              "cross-country median mean_abs_resid)",
            "per_country": recs,
        }

    # Fallback (old result format without residuals).
    def mean_drop(rec):
        d = rec.get("r2_drop_absolute", {})
        vals = [d.get(t) for t in TARGETS if d.get(t) is not None]
        return float(np.mean(vals)) if vals else float("nan")
    recs.sort(key=mean_drop, reverse=True)
    return {"n_countries": len(recs), "n_shards": len(shards_seen),
            "per_country": recs}


def aggregate_optuna(merged):
    """Pick the global-best architecture across all Optuna shards."""
    best = None
    n_trials_total = 0
    n_shards = 0
    for jid, jres in merged.items():
        if not jid.startswith("optuna_shard_"):
            continue
        if jres.get("kind") != "optuna":
            continue
        n_shards += 1
        n_trials_total += len(jres.get("all_values", []))
        bv = jres.get("best_value")
        if bv is not None and (best is None or bv > best["best_value"]):
            best = {"best_value": float(bv),
                    "best_params": jres.get("best_params"),
                    "from_shard": jres.get("shard_idx")}
    if best is None:
        return {}
    return {
        "n_shards": n_shards,
        "n_trials_total": n_trials_total,
        "global_best": best,
    }


def aggregate_all(input_dir=PANEL_RUNS_DIR):
    """Read merged JSON and produce all Panel outputs."""
    merged_path = os.path.join(input_dir, "_merged.json")
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"Missing {merged_path}. Run "
                                 "`launch.py aggregate` first.")
    with open(merged_path) as f:
        merged = json.load(f)
    print(f"Loaded {len(merged)} job results from {merged_path}")

    trf = aggregate_transformer_t25(merged)
    spec = aggregate_spec_curve(merged)
    placebos = aggregate_placebos(merged)
    walk_fwd = aggregate_walk_forward(merged)
    loo = aggregate_loo(merged)
    optuna = aggregate_optuna(merged)
    dml = aggregate_dml(merged)
    cond_perm = aggregate_cond_perm(merged)

    # Headline numbers: education R² drop per target across all methods
    headline = {
        "transformer_t25_15_seed": trf,
        "spec_curve_5_methods": spec,
        "placebo_nulls": placebos,
        "walk_forward": walk_fwd,
    }
    if loo:
        headline["loo_185"] = {
            "n_countries": loo["n_countries"],
            "n_shards": loo["n_shards"],
        }
    if optuna:
        headline["optuna"] = optuna
    if dml:
        headline["dml"] = dml
    if cond_perm:
        headline["cond_perm"] = cond_perm

    write_checkin("chapter9_spec_curve.json",
                   {"transformer_t25": trf, "spec_curve": spec},
                   base_dir=ML_CHECKIN,
                   script_path="scripts/ml/chapter9/aggregate_results.py")
    write_checkin("chapter9_placebos.json",
                   {"placebo_nulls": placebos},
                   base_dir=ML_CHECKIN,
                   script_path="scripts/ml/chapter9/aggregate_results.py")
    write_checkin("chapter9_walk_forward.json",
                   {"walk_forward": walk_fwd},
                   base_dir=ML_CHECKIN,
                   script_path="scripts/ml/chapter9/aggregate_results.py")
    if loo:
        write_checkin("chapter9_loo.json",
                       loo,
                       base_dir=ML_CHECKIN,
                       script_path="scripts/ml/chapter9/aggregate_results.py")
    if optuna:
        write_checkin("chapter9_optuna.json",
                       optuna,
                       base_dir=ML_CHECKIN,
                       script_path="scripts/ml/chapter9/aggregate_results.py")
    if dml:
        write_checkin("chapter9_dml.json",
                       {"dml": dml},
                       base_dir=ML_CHECKIN,
                       script_path="scripts/ml/chapter9/aggregate_results.py")
    if cond_perm:
        write_checkin("chapter9_cond_perm.json",
                       {"cond_perm": cond_perm},
                       base_dir=ML_CHECKIN,
                       script_path="scripts/ml/chapter9/aggregate_results.py")
    write_checkin("chapter9_headline.json",
                   headline,
                   base_dir=ML_CHECKIN,
                   script_path="scripts/ml/chapter9/aggregate_results.py")

    # Console summary
    print("\n=== TRANSFORMER T+25 (15-seed median + 95% CI) ===")
    for t, r in trf.items():
        print(f"  {t}: baseline R² = {r['baseline_r2_median']:.3f} "
              f"[{r['baseline_r2_ci95'][0]:.3f}, {r['baseline_r2_ci95'][1]:.3f}]")
        print(f"      edu R² drop = {r['edu_r2_drop_median']:.3f} "
              f"[{r['edu_r2_drop_ci95'][0]:.3f}, {r['edu_r2_drop_ci95'][1]:.3f}]")

    print("\n=== SPEC CURVE (5 methods, single seed) ===")
    by_method = defaultdict(dict)
    for r in spec:
        by_method[r["method"]][r["target"]] = r["edu_r2_drop_abs"]
    for m, drops in by_method.items():
        line = "  " + m.ljust(22)
        for t in TARGETS:
            line += f"  {t}={drops.get(t, float('nan')):+.3f}"
        print(line)

    print("\n=== PLACEBO NULLS (max |R² drop| should be ~0) ===")
    for p, summary in placebos.items():
        flag = "✓" if abs(summary["median_drop"]) < 0.03 else "⚠"
        print(f"  {flag} {p:<32} median={summary['median_drop']:+.3f}  "
              f"max|drop|={summary['max_abs_drop']:.3f}")

    print("\n=== WALK-FORWARD (train pre-cutoff, predict post-cutoff) ===")
    cutoffs = sorted(set(v["cutoff"] for v in walk_fwd.values()))
    for c in cutoffs:
        line = f"  cutoff {c}:"
        for t in TARGETS:
            r = walk_fwd.get(f"{c}_{t}")
            if r:
                line += f"  {t}={r['r2_median']:.3f}"
        print(line)

    if loo:
        print(f"\n=== LOO-185 ({loo['n_countries']} countries) — MOST OUT OF SPEC "
              f"(model trained on all others mispredicts) ===")
        print(f"  {'country':<26}{'score':>6}  {'LE_err':>7}{'TFR_err':>8}{'U5MR_err':>9}  (signed: + = over-predicted)")
        for rec in loo["per_country"][:15]:
            mr = rec.get("mean_resid", {})
            sc = rec.get("out_of_spec_score", float("nan"))
            print(f"  {rec['country']:<26}{sc:>6.2f}  "
                  f"{mr.get('LE', float('nan')):>+7.1f}{mr.get('TFR', float('nan')):>+8.2f}"
                  f"{mr.get('U5MR', float('nan')):>+9.1f}")

    if optuna:
        gb = optuna["global_best"]
        print(f"\n=== OPTUNA ({optuna['n_trials_total']} trials, "
              f"{optuna['n_shards']} shards) ===")
        print(f"  best OOF R² = {gb['best_value']:.4f}  params = {gb['best_params']}")

    if dml:
        print("\n=== DML (doubly-robust education effect, median over seeds) ===")
        for t in TARGETS:
            if t in dml:
                r = dml[t]
                print(f"  {t}: theta = {r['theta_median']:+.4f} "
                      f"[{r['theta_ci95'][0]:+.4f}, {r['theta_ci95'][1]:+.4f}] "
                      f"(n={r['n_seeds']})")

    if cond_perm:
        print("\n=== CONDITIONAL PERMUTATION (edu R² drop) ===")
        for t in TARGETS:
            if t in cond_perm:
                r = cond_perm[t]
                print(f"  {t}: drop = {r['drop_mean']:+.4f} ± {r['drop_std']:.4f}")

    return headline


def main():
    aggregate_all()


if __name__ == "__main__":
    main()
