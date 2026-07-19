# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/residual_atlas.py
# Paper:   "The Long Childhood: On the Convergence of Humanity"
#
# Produces:
#   Layer 2 of the evidential AI build, BIO-LAG edition. Reads out-of-fold
#   predictions from the three single-target parent-vantage transformers
#   (each trained at its own biological horizon) and produces a per-country
#   residual atlas:
#     - signed residuals per (country, T, outcome) at the outcome's bio-lag
#     - per-country aggregate residual signature
#     - cluster labels (HDBSCAN if available, else k-means fallback)
#     - cross-reference with WCDE-vs-Barro-Lee divergence
#
# Bio-lags (from scripts/_shared.py, via data_loader_parent_lag):
#     LE   at T + LAG_LE          = T+12   (LE at birth is mortality-dominated)
#     TFR  at T + LAG_TFR         = T+5    (cohort -> own newborn)
#     U5MR at T + LAG_CHILDREARING = T+12  (childrearing window)
#   No flat T+25 anywhere — each outcome lands at its own biological horizon.
#
# Inputs:
#   scripts/ml/checkpoints/universal_parent_lag_{le,tfr,u5mr}_fold{0..4}.pt
#   data_loader_parent_lag.load_parent_lag_panels(mode='joint')
#
# Outputs:
#   scripts/ml/checkin/residual_atlas_per_country.json  (one row per country;
#       read by the chatbot's query_residual_atlas tool)
#   scripts/ml/checkin/residual_atlas.json              (per-(country,T) detail)
#   scripts/ml/checkin/ml_gate_g4.json                  (labeled-anomaly recovery)
# =============================================================================
"""
residual_atlas.py

The model's misfits are diagnostic. A perfect fit is overfit; the
strongest result is a model with small, structured residuals that sort
into a small number of mechanism-consistent buckets.

This module assembles those residuals into an atlas, clusters them by
signature, cross-references with the WCDE-vs-Barro-Lee data-quality
divergence, and exposes the result both as JSON (for the evidence chain)
and as a lookup the chatbot can interrogate at chat-time.

Bio-lag edition: residuals come from three single-target models, each at
its own biological horizon, rather than one joint T+25 model.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, SCRIPT_DIR)

from _shared import ML_CHECKIN, write_checkin
from data_loader_parent_lag import load_parent_lag_panels, PARENT_LAG_HORIZONS
from run_universal_evidence_parent_lag import _oof_predict_single

OUTCOMES = ("LE", "TFR", "U5MR")

# Canonical names of labeled anomalies. The atlas's job is to recover these
# clusters automatically. Keys correspond to expected residual signatures.
LABELED_ANOMALIES = {
    "soviet_bloc_hollow_education": [
        "russia", "kazakhstan", "uzbekistan", "kyrgyzstan", "tajikistan",
        "turkmenistan", "belarus", "ukraine", "moldova", "georgia",
        "armenia", "azerbaijan", "latvia", "lithuania", "estonia",
    ],
    "out_migration_artifact": ["guyana", "suriname"],
    "oil_rent_bypass": [
        "saudi arabia", "kuwait", "qatar", "united arab emirates",
        "oman", "bahrain", "equatorial guinea",
    ],
}


def _wcde_minus_bl_divergence(panel):
    """WCDE-vs-Barro-Lee mean-years divergence per (country, T), in years.

    Input-side, lag-independent: computed from the feature window's last
    position (time T). Undoes the loader's standardization to recover raw
    attainment shares, converts to a WCDE mean-years proxy, and subtracts
    the Barro-Lee mean-years feature. Returns {(country, T): signed years}.
    """
    feat_mean = panel["feat_mean"]
    feat_std = panel["feat_std"]
    fn = panel["feature_names"]

    idx_prim = fn.index("wcde_primary_both")
    idx_lsec = fn.index("wcde_lower_sec_both")
    idx_usec = fn.index("wcde_upper_sec_both")
    idx_coll = fn.index("wcde_college_both")
    idx_bl_yr = fn.index("bl_yr_sch_mf_15_24")

    X = panel["X"][:, -1, :]            # (N, F) at time T
    raw = X * feat_std + feat_mean

    primary = raw[:, idx_prim]
    lsec = raw[:, idx_lsec]
    usec = raw[:, idx_usec]
    college = raw[:, idx_coll]
    bl_yr = raw[:, idx_bl_yr]

    # WCDE attainment shares -> mean years (primary=6, lower-sec=9,
    # upper-sec=12, college=16).
    wcde_proxy = (
        (primary - lsec) / 100 * 6
        + (lsec - usec) / 100 * 9
        + (usec - college) / 100 * 12
        + college / 100 * 16
    )
    div = wcde_proxy - bl_yr

    out = {}
    for i, (cid, T) in enumerate(zip(panel["country_ids"], panel["years"])):
        c = panel["countries"][cid]
        out[(c, int(T))] = float(div[i])
    return out


def _per_outcome_rows(panels):
    """Long-form per-(country, T, outcome) residual rows across all three
    single-target bio-lag models, plus the input-side divergence."""
    rows = []
    div_seen = {}
    for tname in OUTCOMES:
        panel = panels[tname]
        horizon = PARENT_LAG_HORIZONS[tname]
        prefix = f"universal_parent_lag_{tname.lower()}"
        pred, seen = _oof_predict_single(panel, prefix)
        resid = panel["y"][:, 0] - pred[:, 0]    # signed: actual - predicted
        div_map = _wcde_minus_bl_divergence(panel)
        div_seen.update(div_map)
        for i in range(len(panel["country_ids"])):
            if not seen[i]:
                continue
            c = panel["countries"][panel["country_ids"][i]]
            T = int(panel["years"][i])
            rows.append({
                "country": c,
                "outcome": tname,
                "T": T,
                "horizon": horizon,
                "T_outcome": T + horizon,
                "actual": float(panel["y"][i, 0]),
                "pred": float(pred[i, 0]),
                "resid": float(resid[i]),
                "wcde_bl_divergence_years": div_map.get((c, T)),
            })
    return pd.DataFrame(rows), div_seen


def _cluster_signatures(X):
    """Cluster country signatures. Try HDBSCAN; fall back to k-means."""
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=2)
        return clusterer.fit_predict(X).tolist(), "hdbscan"
    except ImportError:
        pass
    from sklearn.cluster import KMeans
    # k=8: with ~185 countries, k=5 collapsed most into one undiscriminating
    # cluster. Finer k gives more meaningful cluster structure.
    km = KMeans(n_clusters=8, random_state=42, n_init=10)
    return km.fit_predict(X).tolist(), "kmeans"


def _nearest_neighbours(countries, feats_z, k=5):
    """Top-k most-similar countries per country by Euclidean distance in
    standardized residual-signature space. Robust 'similar countries' that
    does not depend on coarse cluster membership."""
    X = np.asarray(feats_z, dtype=float)
    out = {}
    for i, c in enumerate(countries):
        d = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
        order = np.argsort(d)
        nn = [countries[j] for j in order if j != i][:k]
        out[c] = nn
    return out


def compute_atlas():
    """Build the bio-lag residual atlas from the three parent-vantage models."""
    panels = load_parent_lag_panels(mode="joint", verbose=False)
    long_df, div_seen = _per_outcome_rows(panels)

    # Per-country mean/std residual per outcome (each at its own bio-lag).
    pivot_mean = long_df.pivot_table(index="country", columns="outcome",
                                     values="resid", aggfunc="mean")
    pivot_std = long_df.pivot_table(index="country", columns="outcome",
                                    values="resid", aggfunc="std")
    pivot_n = long_df.pivot_table(index="country", columns="outcome",
                                  values="resid", aggfunc="count")

    # Per-country mean divergence (input-side, dedup by (country, T)).
    div_by_country = {}
    for (c, _T), v in div_seen.items():
        div_by_country.setdefault(c, []).append(v)
    mean_div = {c: float(np.nanmean(vs)) for c, vs in div_by_country.items()}

    sig = pd.DataFrame(index=pivot_mean.index)
    for tn in OUTCOMES:
        sig[f"mean_resid_{tn}"] = pivot_mean.get(tn)
        sig[f"std_resid_{tn}"] = pivot_std.get(tn)
    sig["n"] = pivot_n.reindex(columns=list(OUTCOMES)).min(axis=1)
    sig["mean_divergence"] = sig.index.map(mean_div)
    sig = sig.reset_index().rename(columns={"index": "country"})

    # Cluster on the standardized 3-d residual signature; only countries
    # observed on all three outcomes are clustered.
    cols = [f"mean_resid_{tn}" for tn in OUTCOMES]
    complete = sig.dropna(subset=cols).copy()
    feats = complete[cols]
    feats_z = (feats - feats.mean()) / feats.std().replace(0, 1.0)
    labels, cluster_method = _cluster_signatures(feats_z.values)
    complete["cluster"] = labels
    # Precompute genuine nearest-neighbour similar countries (distance in
    # standardized residual space), independent of cluster granularity.
    nn = _nearest_neighbours(list(complete["country"]), feats_z.values, k=5)
    complete["similar_countries"] = complete["country"].map(nn)
    sig = sig.merge(complete[["country", "cluster", "similar_countries"]],
                    on="country", how="left")
    sig["cluster_method"] = cluster_method

    sig["abs_resid_sum"] = (
        sig["mean_resid_LE"].abs().fillna(0)
        + sig["mean_resid_TFR"].abs().fillna(0)
        + sig["mean_resid_U5MR"].abs().fillna(0)
    )
    sig = sig.sort_values("abs_resid_sum", ascending=False).reset_index(drop=True)
    return long_df, sig, cluster_method


def write_outputs(long_df, sig, cluster_method):
    out = {
        "mode": "parent_lag_bio",
        "horizons": dict(PARENT_LAG_HORIZONS),
        "cluster_method": cluster_method,
        "n_countries": int(len(sig)),
        "countries": json.loads(sig.where(pd.notnull(sig), None).to_json(orient="records")),
    }
    write_checkin("residual_atlas_per_country.json", out, base_dir=ML_CHECKIN,
                  script_path="scripts/ml/residual_atlas.py")

    out_detail = {
        "mode": "parent_lag_bio",
        "horizons": dict(PARENT_LAG_HORIZONS),
        "n_rows": int(len(long_df)),
        "rows": json.loads(long_df.where(pd.notnull(long_df), None).to_json(orient="records")),
    }
    write_checkin("residual_atlas.json", out_detail, base_dir=ML_CHECKIN,
                  script_path="scripts/ml/residual_atlas.py")


def gate_g4(sig):
    """Labeled-anomaly recovery: do the named buckets show the expected
    residual direction? (resid = actual - predicted.)"""
    by_country = sig.set_index("country").to_dict(orient="index")

    def _agg(group):
        present = [c for c in group if c in by_country]
        if not present:
            return None
        rows = [by_country[c] for c in present]
        def _m(key):
            vals = [r.get(key) for r in rows if r.get(key) is not None and not (isinstance(r.get(key), float) and np.isnan(r.get(key)))]
            return float(np.mean(vals)) if vals else None
        return {
            "countries_present": present,
            "countries_missing": [c for c in group if c not in by_country],
            "mean_resid_LE": _m("mean_resid_LE"),
            "mean_resid_TFR": _m("mean_resid_TFR"),
            "mean_resid_U5MR": _m("mean_resid_U5MR"),
            "mean_divergence": _m("mean_divergence"),
        }

    soviet = _agg(LABELED_ANOMALIES["soviet_bloc_hollow_education"])
    guyana = _agg(LABELED_ANOMALIES["out_migration_artifact"])
    oil = _agg(LABELED_ANOMALIES["oil_rent_bypass"])

    # resid = actual - predicted.
    #   LE   resid < 0  <=> worse than predicted
    #   TFR  resid > 0  <=> worse than predicted (higher fertility)
    #   U5MR resid > 0  <=> worse than predicted (higher mortality)
    def _ok(d, cond):
        return bool(d is not None and cond(d))

    soviet_ok = _ok(soviet, lambda d: d["mean_resid_LE"] < 0 and d["mean_resid_TFR"] > 0 and d["mean_resid_U5MR"] > 0)
    guy_ok = _ok(guyana, lambda d: d["mean_resid_LE"] < 0 and abs(d["mean_resid_TFR"]) < 0.5)
    oil_ok = _ok(oil, lambda d: d["mean_resid_LE"] > 0 and d["mean_resid_TFR"] < 0 and d["mean_resid_U5MR"] < 0)

    out = {
        "soviet_bloc_hollow_education": {"signature": soviet, "direction_matches": soviet_ok,
                                         "wcde_bl_divergence_high": bool(soviet is not None and (soviet["mean_divergence"] or 0) > 1.0)},
        "out_migration_artifact": {"signature": guyana, "direction_matches": guy_ok},
        "oil_rent_bypass": {"signature": oil, "direction_matches": oil_ok},
        "pass": bool(soviet_ok and guy_ok and oil_ok),
    }
    write_checkin("ml_gate_g4.json", out, base_dir=ML_CHECKIN,
                  script_path="scripts/ml/residual_atlas.py")
    return out


def main():
    long_df, sig, cluster_method = compute_atlas()
    write_outputs(long_df, sig, cluster_method)
    g4 = gate_g4(sig)
    print(f"[atlas] {len(sig)} countries, cluster_method={cluster_method}, "
          f"horizons={dict(PARENT_LAG_HORIZONS)}")
    print(f"[atlas] G4 labeled-anomaly recovery: pass={g4['pass']} "
          f"(soviet={g4['soviet_bloc_hollow_education']['direction_matches']}, "
          f"guyana={g4['out_migration_artifact']['direction_matches']}, "
          f"oil={g4['oil_rent_bypass']['direction_matches']})")
    top = sig.head(8)[["country", "mean_resid_LE", "mean_resid_TFR", "mean_resid_U5MR", "cluster"]]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
