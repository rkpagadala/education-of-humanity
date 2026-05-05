# =============================================================================
# PAPER REFERENCE
# Script:  scripts/robustness/beta_by_ceiling_cutoff.py
# Paper:   "Education of Nations"
#
# Produces:
#   Table showing β and R² at parental education cutoffs from <10% to no cutoff
#   Two panels: (A) all countries 1900–2015, (B) post-1975 panel
#   Key finding: β > 1 at every cutoff below 90% — education amplifies
#               across generations, dragged below unity only by ceiling noise
#
# Inputs:
#   wcde/data/processed/cohort_completion_both_long.csv
#
# Key parameters:
#   GENERATIONAL_LAG = 25
#   CUTOFFS = 10, 20, ..., 90
# =============================================================================
"""
robustness/beta_by_ceiling_cutoff.py

Shows that the pooled β is mechanically suppressed by ceiling observations.
When countries near 100% completion are excluded, β exceeds unity at every
cutoff — education gains amplify across generations rather than depreciating.

Section 6.1 of the paper.
"""

import os
import sys

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from _shared import write_checkin, load_education
from residualization._shared import build_panel, precompute_entry_years, filter_panel, find_entry_year, fe_beta_r2

# ── Load data ────────────────────────────────────────────────────────────────
long = load_education("cohort_completion_both_long.csv")
low_w = long.pivot(index="country", columns="cohort_year", values="lower_sec")


def v(df_w, country, year):
    """Look up a value from wide-format DataFrame."""
    try:
        val = float(df_w.loc[country, int(year)])
        return val if not np.isnan(val) else np.nan
    except (KeyError, ValueError):
        return np.nan


def build_cohort_panel(child_cohorts):
    """Build parent-child panel for all countries."""
    rows = []
    for c in low_w.index:
        for child_yr in child_cohorts:
            parent_yr = child_yr - 25
            child_low = v(low_w, c, child_yr)
            parent_low = v(low_w, c, parent_yr)
            if np.isnan(child_low) or np.isnan(parent_low):
                continue
            rows.append({
                "country": c,
                "cohort_year": child_yr,
                "child_low": child_low,
                "parent_low": parent_low,
            })
    return pd.DataFrame(rows)


CUTOFFS = list(range(10, 100, 10))


numbers = {}


def print_panel(panel, label, prefix):
    """Print β and R² at each cutoff for a panel."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {len(panel)} obs, {panel['country'].nunique()} countries")
    print(f"{'='*60}")
    print(f"{'Cutoff':>10} {'beta':>8} {'R2':>8} {'n':>8} {'Countries':>10}")
    print("-" * 50)

    for cutoff in CUTOFFS:
        sub = panel[panel["parent_low"] < cutoff]
        beta, r2, n, nc = fe_beta_r2("parent_low", "child_low", sub)
        if n == 0:
            continue
        print(f"    <{cutoff:3d}%  {beta:8.3f} {r2:8.3f} {n:8d} {nc:10d}")
        numbers[f"{prefix}_cutoff_{cutoff}_beta"] = round(beta, 3)
        numbers[f"{prefix}_cutoff_{cutoff}_r2"] = round(r2, 3)

    beta, r2, n, nc = fe_beta_r2("parent_low", "child_low", panel)
    print(f"  no cut  {beta:8.3f} {r2:8.3f} {n:8d} {nc:10d}")
    numbers[f"{prefix}_no_cutoff_beta"] = round(beta, 3)
    numbers[f"{prefix}_no_cutoff_r2"] = round(r2, 3)


# ── Panel A: All countries, 1900–2015 ────────────────────────────────────────
panel_full = build_cohort_panel(list(range(1900, 2016, 5)))
print_panel(panel_full, "Panel A: All countries, 1900-2015", "panelA")

# ── Panel B: Post-1975 (matching Table 1 time range) ────────────────────────
panel_post75 = build_cohort_panel(list(range(1975, 2016, 5)))
print_panel(panel_post75, "Panel B: Post-1975 (Table 1 comparison)", "panelB")

# ── Write checkin JSON ──────────────────────────────────────────────────────
write_checkin("beta_by_ceiling_cutoff.json", {
    "notes": f"Panel A: All countries 1900-2015 ({len(panel_full)} obs, "
             f"{panel_full['country'].nunique()} countries). "
             f"Panel B: Post-1975 ({len(panel_post75)} obs, "
             f"{panel_post75['country'].nunique()} countries).",
    "numbers": numbers,
}, script_path="scripts/robustness/beta_by_ceiling_cutoff.py")
print("Done.")
