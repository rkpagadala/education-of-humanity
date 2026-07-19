# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/compare_parent_lag_vs_t25.py
# Paper:   "The Long Childhood: On the Convergence of Humanity"
#
# Produces:
#   Side-by-side comparison of the T+25 grandparent-vantage universal
#   transformer (universal_evidence.json) vs the parent-vantage variant
#   at biological horizons (universal_evidence_parent_lag.json).
#
# Inputs:
#   scripts/ml/checkin/universal_evidence.json
#   scripts/ml/checkin/universal_evidence_parent_lag.json
#
# Outputs:
#   scripts/ml/checkin/parent_lag_vs_t25_comparison.md  (and stdout)
# =============================================================================
"""
compare_parent_lag_vs_t25.py

Reads both evidence JSONs and produces a side-by-side comparison
report. Run AFTER both runners have completed.
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from _shared import ML_CHECKIN

T25_JSON = os.path.join(ML_CHECKIN, "universal_evidence.json")
PARENT_JSON = os.path.join(ML_CHECKIN, "universal_evidence_parent_lag.json")
OUT_MD = os.path.join(ML_CHECKIN, "parent_lag_vs_t25_comparison.md")

# Order in universal_evidence.json B0_joint_5fold arrays: [LE, TFR, U5MR]
T25_TARGET_IDX = {"LE": 0, "TFR": 1, "U5MR": 2}


def _signed(x):
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.3f}"


def main():
    if not os.path.exists(T25_JSON):
        raise FileNotFoundError(f"Missing {T25_JSON}")
    if not os.path.exists(PARENT_JSON):
        raise FileNotFoundError(f"Missing {PARENT_JSON}")

    with open(T25_JSON) as f:
        t25 = json.load(f)
    with open(PARENT_JSON) as f:
        pl = json.load(f)

    t25_b0 = t25["B0_joint_5fold"]
    t25_gates = t25["B1_B2_gates_on_joint"]
    pl_per = pl["per_outcome"]
    horizons = pl["horizons"]

    lines = []
    lines.append("## Parent-vantage transformer vs T+25 grandparent vantage\n")

    lines.append("### Setup")
    lines.append(
        "Architecture is the encoder-only `UniversalTransformer` (no country "
        "embedding) from `scripts/ml/universal_transformer.py`. Same encoder "
        "depth / width, same window of four 5-year timesteps [T-15, T-10, "
        "T-5, T], same 126-feature input (WCDE current + cohort, Barro-Lee "
        "v3, gender gaps, GDP with Maddison backfill, broader features). "
        "5-fold country-holdout cross-validation, same seed (42) so fold "
        "assignment is identical between the two builds. The T+25 build "
        "uses a single multi-target head (LE/TFR/U5MR jointly). The "
        "parent-vantage build trains three SEPARATE single-target models, "
        "each at its biological horizon: LE at T+12, TFR at T+5, U5MR at "
        "T+12. Hyperparameters held fixed: "
        f"d_model={pl['hyperparameters']['d_model']}, "
        f"nhead={pl['hyperparameters']['nhead']}, "
        f"num_layers={pl['hyperparameters']['num_layers']}, "
        f"dropout={pl['hyperparameters']['dropout']}, "
        f"lr={pl['hyperparameters']['lr']}, "
        f"batch={pl['hyperparameters']['batch_size']}, "
        f"epochs={pl['hyperparameters']['epochs']}, "
        f"patience={pl['hyperparameters']['patience']}.\n"
    )

    lines.append("### Per-outcome OOF R²\n")
    lines.append("| Outcome | T+25 R² (grandparent) | Parent-lag R² (biological) | Δ |")
    lines.append("|---|---|---|---|")
    for tname in ("LE", "TFR", "U5MR"):
        idx = T25_TARGET_IDX[tname]
        r2_t25 = t25_b0["overall_r2_oof"][idx]
        r2_pl = pl_per[tname]["B0_joint_5fold"]["overall_r2_oof"][0]
        delta = r2_pl - r2_t25
        h = horizons[tname]
        lines.append(
            f"| {tname}  | {r2_t25:.3f} | {r2_pl:.3f} at T+{h} | {_signed(delta)} |"
        )

    lines.append("")
    lines.append("### Education zero-out R² drop\n")
    lines.append("| Outcome | T+25 drop fraction | Parent-lag drop fraction |")
    lines.append("|---|---|---|")
    for tname in ("LE", "TFR", "U5MR"):
        t25_drop = t25_gates["g1_education_ablation"]["r2_drop_fraction"][tname]
        pl_drop = pl_per[tname]["B1_B2_gates_on_joint"][
            "g1_education_ablation"]["r2_drop_fraction"][tname]
        lines.append(f"| {tname} | {t25_drop:.3f} | {pl_drop:.3f} |")

    lines.append("")
    lines.append("### GDP zero-out R² drop (sanity / G2)\n")
    lines.append("| Outcome | T+25 drop fraction | Parent-lag drop fraction |")
    lines.append("|---|---|---|")
    for tname in ("LE", "TFR", "U5MR"):
        t25_drop = t25_gates["g2_gdp_ablation"]["r2_drop_fraction"][tname]
        pl_drop = pl_per[tname]["B1_B2_gates_on_joint"][
            "g2_gdp_ablation"]["r2_drop_fraction"][tname]
        lines.append(f"| {tname} | {t25_drop:.3f} | {pl_drop:.3f} |")

    lines.append("")
    lines.append("### Sample sizes\n")
    lines.append("| Outcome | T+25 N | Parent-lag N |")
    lines.append("|---|---|---|")
    for tname in ("LE", "TFR", "U5MR"):
        n_pl = pl_per[tname]["B0_joint_5fold"]["n_samples"]
        lines.append(f"| {tname} | {t25_b0['n_samples']} | {n_pl} |")

    lines.append("")
    lines.append("### Characterization\n")

    # Per-outcome paragraphs — generated programmatically from the deltas.
    for tname in ("LE", "TFR", "U5MR"):
        idx = T25_TARGET_IDX[tname]
        r2_t25 = t25_b0["overall_r2_oof"][idx]
        r2_pl = pl_per[tname]["B0_joint_5fold"]["overall_r2_oof"][0]
        delta = r2_pl - r2_t25
        t25_edu = t25_gates["g1_education_ablation"]["r2_drop_fraction"][tname]
        pl_edu = pl_per[tname]["B1_B2_gates_on_joint"][
            "g1_education_ablation"]["r2_drop_fraction"][tname]
        h = horizons[tname]
        n_pl = pl_per[tname]["B0_joint_5fold"]["n_samples"]
        direction = ("higher" if delta > 0.01 else
                     "lower" if delta < -0.01 else "roughly equal")
        edu_direction = ("stronger" if pl_edu > t25_edu + 0.01 else
                         "weaker" if pl_edu < t25_edu - 0.01 else "comparable")
        lines.append(
            f"- **{tname} at parent lag T+{h} vs T+25 grandparent lag**: "
            f"parent-vantage OOF R²={r2_pl:.3f} ({direction} than T+25's "
            f"{r2_t25:.3f}, Δ={_signed(delta)}). Education-ablation drop is "
            f"{pl_edu:.3f} at the biological horizon vs {t25_edu:.3f} at T+25 "
            f"({edu_direction} education dependence). N={n_pl} parent-vantage "
            f"samples vs {t25_b0['n_samples']} at T+25."
        )

    lines.append("")
    lines.append("### Bottom line\n")

    # Compute aggregate signal for the bottom-line paragraph.
    deltas = {}
    edu_deltas = {}
    for tname in ("LE", "TFR", "U5MR"):
        idx = T25_TARGET_IDX[tname]
        deltas[tname] = (pl_per[tname]["B0_joint_5fold"]["overall_r2_oof"][0]
                          - t25_b0["overall_r2_oof"][idx])
        edu_deltas[tname] = (pl_per[tname]["B1_B2_gates_on_joint"][
                                "g1_education_ablation"]["r2_drop_fraction"][tname]
                              - t25_gates["g1_education_ablation"][
                                "r2_drop_fraction"][tname])

    best = max(deltas, key=lambda k: deltas[k])
    worst = min(deltas, key=lambda k: deltas[k])
    edu_best = max(edu_deltas, key=lambda k: edu_deltas[k])
    edu_worst = min(edu_deltas, key=lambda k: edu_deltas[k])

    lines.append(
        f"- R² wins: parent-vantage is highest-relative on **{best}** "
        f"(Δ={_signed(deltas[best])}) and lowest-relative on **{worst}** "
        f"(Δ={_signed(deltas[worst])})."
    )
    lines.append(
        f"- Education signal: the education-ablation drop is largest on "
        f"**{edu_best}** at the biological horizon (Δ vs T+25 = "
        f"{_signed(edu_deltas[edu_best])}) and smallest on **{edu_worst}** "
        f"(Δ vs T+25 = {_signed(edu_deltas[edu_worst])})."
    )
    lines.append(
        "- The T+25 grandparent vantage corresponds to the paper's primary "
        "structural lag (time-to-agency); the biological-lag variant tests "
        "whether each outcome's natural cohort horizon yields a stronger "
        "education signal than collapsing all three to T+25. Use the table "
        "above to decide whether the parent-vantage build joins, replaces, "
        "or sits alongside the T+25 evidence in the paper."
    )
    lines.append("")
    lines.append(
        f"_Generated from `{os.path.basename(T25_JSON)}` and "
        f"`{os.path.basename(PARENT_JSON)}`._\n"
    )

    text = "\n".join(lines)
    with open(OUT_MD, "w") as f:
        f.write(text)
    print(text)
    print(f"\n[wrote] {OUT_MD}")


if __name__ == "__main__":
    main()
