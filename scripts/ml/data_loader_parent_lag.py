# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/data_loader_parent_lag.py
# Paper:   "The Long Childhood: On the Convergence of Humanity"
#
# Produces:
#   Three single-target country x year panels for the parent-vantage
#   transformer variant. Each outcome is predicted at its biological
#   horizon rather than the canonical T+25 grandparent vantage:
#     - TFR  at T + LAG_TFR             (parent timing, biological birth)
#     - U5MR at T + LAG_CHILDREARING    (childrearing window, ~10-15 yrs)
#     - LE   at T + LAG_LE              (childrearing window — LE at birth
#                                        is mortality-dominated, mostly
#                                        U5MR, so it shares U5MR's lag, not
#                                        a generation-back signal)
#
#   Same feature construction as scripts/ml/data_loader.py — this file
#   is a thin wrapper that calls load_panel() three times with the
#   per-outcome horizon and slices to a single target. The architecture
#   (UniversalTransformer) and feature pipeline are identical to the
#   T+25 build; only the target horizon changes.
#
# Inputs:
#   Same as scripts/ml/data_loader.py (WCDE, Barro-Lee, WB, Maddison,
#   BroaderFeatures).
#
# Outputs (in-memory):
#   load_parent_lag_panels(mode) -> dict with three sub-panels:
#       'TFR'  (horizon=5),
#       'U5MR' (horizon=12),
#       'LE'   (horizon=12).
#   Each sub-panel has the same keys as load_panel(): X, y, mask,
#   country_ids, years, feature_names, countries, feature_groups,
#   feat_mean, feat_std, mode, window, pred_horizon — plus an extra
#   'target_name' field. y has shape [N, 1] (single outcome).
# =============================================================================
"""
data_loader_parent_lag.py

Wrapper around data_loader.load_panel that returns one single-target
panel per outcome, with each outcome at its biological horizon. Feature
construction is delegated to the original loader so the comparison
between T+25 grandparent vantage and biological-lag parent vantage
varies ONLY the target horizon (apples-to-apples on features and
architecture).
"""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, SCRIPT_DIR)

from _shared import LAG_TFR, LAG_CHILDREARING, LAG_LE
from data_loader import load_panel

# Order in load_panel().y: [LE, TFR, U5MR]
TARGET_INDEX = {"LE": 0, "TFR": 1, "U5MR": 2}

# Per-outcome biological horizons.
#   LE at T+12  — life expectancy at birth is mechanically dominated by
#                 child mortality (within-country, log U5MR alone carries
#                 ~0.70 of LE's variation; child + adult mortality together
#                 ~0.90). The schooling that moves LE is therefore the same
#                 childrearing-window schooling that moves U5MR (LAG_LE =
#                 LAG_CHILDREARING), not a generation-back signal.
#   TFR at T+5  — schooled cohort becomes biological parents ~5 years
#                 after measurement (LAG_TFR).
#   U5MR at T+12 — schooled parents' children are in the under-5 window
#                  roughly 10-15 years after measurement (LAG_CHILDREARING).
PARENT_LAG_HORIZONS = {
    "LE":   LAG_LE,
    "TFR":  LAG_TFR,
    "U5MR": LAG_CHILDREARING,
}


def _single_target_slice(panel, target_name):
    """Return a new panel dict with y restricted to a single column.

    The base panel's y has shape [N, 3] (LE, TFR, U5MR) and rows are
    only kept when ALL three outcomes are observed at T + pred_horizon.
    For a single-target evaluation we still require all three to be
    observed (which is what load_panel produces); this keeps the row
    coverage aligned with the original loader's filtering rules and
    avoids divergent N across the three sub-panels for spurious reasons.
    """
    idx = TARGET_INDEX[target_name]
    new = {k: v for k, v in panel.items()}
    new["y"] = panel["y"][:, idx:idx + 1].copy()
    new["target_name"] = target_name
    new["targets"] = [target_name]
    return new


def load_parent_lag_panels(mode="joint", verbose=True):
    """Assemble three single-target panels, each at its biological lag.

    Calls scripts/ml/data_loader.load_panel() three times with the
    per-outcome horizon and slices the resulting y to a single column.
    All three panels share the SAME feature construction code as the
    original T+25 loader.

    Returns a dict keyed by target name ('TFR', 'U5MR', 'LE'), each
    value a panel dict (same schema as load_panel(), with y now shape
    [N, 1] and an added 'target_name' field).
    """
    out = {}
    for tname in ("TFR", "U5MR", "LE"):
        horizon = PARENT_LAG_HORIZONS[tname]
        if verbose:
            print(f"[parent_lag] loading panel for {tname} at horizon T+{horizon}")
        base = load_panel(mode=mode, verbose=verbose, pred_horizon=horizon)
        out[tname] = _single_target_slice(base, tname)
    return out


if __name__ == "__main__":
    panels = load_parent_lag_panels(mode="joint")
    for tname, p in panels.items():
        print(f"  {tname}: X={p['X'].shape}  y={p['y'].shape}  "
              f"horizon=T+{p['pred_horizon']}  n_samples={p['y'].shape[0]}")
