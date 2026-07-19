# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/run_universal_evidence.py
# Paper:   "The Long Childhood: On the Convergence of Humanity"
#
# Library module (no longer a standalone runner). Provides the out-of-fold
# ablation gate for the UniversalTransformer (no country embedding):
#
#   oof_predict(panel, ckpt_prefix)  — country-held-out OOF predictions from
#                                      the per-fold checkpoints.
#   run_g1_g2(panel, ckpt_prefix)    — G1 (zero education at inference) and
#                                      G2 (zero GDP) R²-drop gates.
#
# run_g1_g2 is imported by scripts/ml/cloud/run_job.py. The per-outcome
# parent-vantage evidence suite is produced by
# run_universal_evidence_parent_lag.py.
#
# History: the standalone joint-T+25 entrypoint (main(), the B0/B3/B4 5-fold
# training and the Korea/Niger saturation test) and its universal_evidence.json
# output were retired when the headline ablation and the country-counterfactual
# moved to the per-outcome biological horizons (parent-lag). See
# run_universal_evidence_parent_lag.py and chapter9/counterfactuals.py.
# =============================================================================
"""
run_universal_evidence.py

Out-of-fold ablation gates for the UniversalTransformer (no country
embedding) — the honest test of whether the panel, stripped of any
per-country memorization, still tells the education-is-load-bearing
story. Imported by the cloud worker; not run directly.
"""

import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, SCRIPT_DIR)

from data_loader import country_holdout_mask, feature_ablation
from universal_transformer import build_model_from_checkpoint, CHECKPOINT_DIR

TARGETS = ["LE", "TFR", "U5MR"]
N_FOLDS = 5


def oof_predict(panel, ckpt_prefix):
    Xt = torch.tensor(panel["X"], dtype=torch.float32)
    mt = torch.tensor(panel["mask"], dtype=torch.float32)
    n_features = panel["X"].shape[-1]
    window = panel["window"]
    pred_oof = np.zeros_like(panel["y"])
    seen = np.zeros(panel["y"].shape[0], dtype=bool)
    for fold in range(N_FOLDS):
        path = os.path.join(CHECKPOINT_DIR, f"{ckpt_prefix}_fold{fold}.pt")
        ck = torch.load(path, weights_only=False)
        model = build_model_from_checkpoint(ck, n_features, window)
        is_val = country_holdout_mask(panel, n_folds=N_FOLDS, fold=fold, seed=ck["hp"]["seed"])
        with torch.no_grad():
            pred_std = model(Xt[is_val], mt[is_val]).numpy()
        pred_oof[is_val] = pred_std * ck["y_std"] + ck["y_mean"]
        seen |= is_val
    return pred_oof, seen


def _r2_per_target(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def run_g1_g2(panel, ckpt_prefix):
    baseline_pred, seen = oof_predict(panel, ckpt_prefix)
    baseline_r2 = _r2_per_target(panel["y"][seen], baseline_pred[seen])

    p_no_edu = feature_ablation(panel, drop_groups=["wcde_education", "bl_education", "derived"])
    pred_no_edu, _ = oof_predict(p_no_edu, ckpt_prefix)
    r2_no_edu = _r2_per_target(panel["y"][seen], pred_no_edu[seen])
    g1_drop = (baseline_r2 - r2_no_edu) / np.maximum(np.abs(baseline_r2), 1e-12)

    p_no_gdp = feature_ablation(panel, drop_groups=["gdp"])
    pred_no_gdp, _ = oof_predict(p_no_gdp, ckpt_prefix)
    r2_no_gdp = _r2_per_target(panel["y"][seen], pred_no_gdp[seen])
    g2_drop = (baseline_r2 - r2_no_gdp) / np.maximum(np.abs(baseline_r2), 1e-12)

    return {
        "n_evaluated": int(seen.sum()),
        "baseline_r2": {t: float(v) for t, v in zip(TARGETS, baseline_r2)},
        "g1_education_ablation": {
            "ablated_r2": {t: float(v) for t, v in zip(TARGETS, r2_no_edu)},
            "r2_drop_fraction": {t: float(v) for t, v in zip(TARGETS, g1_drop)},
            "max_drop": float(np.max(g1_drop)),
            "min_drop": float(np.min(g1_drop)),
        },
        "g2_gdp_ablation": {
            "ablated_r2": {t: float(v) for t, v in zip(TARGETS, r2_no_gdp)},
            "r2_drop_fraction": {t: float(v) for t, v in zip(TARGETS, g2_drop)},
            "max_drop": float(np.max(g2_drop)),
        },
    }
