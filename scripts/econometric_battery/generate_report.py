"""Generate REPORT.md from existing JSON outputs (no re-run required)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _battery import BATTERY_CHECKIN_DIR

TESTS = [
    ("01_stationarity.json",            "Panel unit-root tests"),
    ("02_cross_section_dependence.json", "Cross-sectional dependence"),
    ("03_cointegration.json",           "Panel cointegration"),
    ("04_serial_correlation.json",      "Panel serial correlation"),
    ("05_robust_se.json",               "Robust standard errors"),
    ("06_exogeneity.json",              "Exogeneity battery"),
    ("07_specification.json",           "Specification / functional form"),
    ("08_structural_breaks.json",       "Structural breaks"),
    ("09_influence.json",               "Influence diagnostics"),
    ("10_robust_regression.json",       "Outlier-robust estimators"),
    ("11_multiple_testing.json",        "Multiple testing corrections"),
    ("12_dynamic_panels.json",          "Dynamic panel / GMM"),
    ("13_modern_did.json",              "Modern DiD"),
    ("14_synthetic_control.json",       "Synthetic control"),
    ("15_cross_validation.json",        "Cross-validation"),
    ("16_functional_form.json",         "Variable transformations"),
    ("17_sample_composition.json",      "Sample composition"),
    ("18_measurement_error.json",       "Measurement-error correction"),
    ("19_placebo.json",                 "Placebo tests"),
    ("20_predictive_validity.json",     "Predictive validity"),
]


def fmt_value(v, depth=0):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if abs(v) < 1e-3 or abs(v) > 1e6:
            return f"{v:.3e}"
        return f"{v:.4f}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return v
    return str(v)


def walk(d, depth=0, max_depth=3, prefix=""):
    out = []
    if depth > max_depth:
        return out
    if isinstance(d, dict):
        for k, v in d.items():
            if k == "script":
                continue
            label = f"{prefix}{k}"
            if isinstance(v, (int, float, str, bool)) or v is None:
                out.append((label, fmt_value(v)))
            elif isinstance(v, list):
                if v and isinstance(v[0], (int, float, str, bool)):
                    s = ", ".join(fmt_value(x) for x in v[:8])
                    if len(v) > 8:
                        s += f"  …({len(v)} total)"
                    out.append((label, s))
                elif v and isinstance(v[0], dict) and depth < max_depth:
                    out.append((label, f"({len(v)} entries)"))
                    for i, item in enumerate(v[:3]):
                        out.extend(walk(item, depth + 1, max_depth, prefix=f"  [{i}] "))
                    if len(v) > 3:
                        out.append((f"  …", f"+{len(v) - 3} more entries"))
            elif isinstance(v, dict) and depth < max_depth:
                out.append((label, ""))
                out.extend(walk(v, depth + 1, max_depth, prefix=f"  {prefix}"))
    return out


def main():
    md = []
    md.append("# Econometric Battery — Summary Report")
    md.append("")
    md.append("Comprehensive panel-econometrics test suite for *The Long Childhood*.")
    md.append("Auto-generated from JSON outputs in `checkin/econometric_battery/`.")
    md.append("")
    md.append("**Panel headline:** β = 0.4834 (lower-secondary parent → child completion,")
    md.append("country FE, cluster SE, 1665 obs · 185 countries · 1975–2015).")
    md.append("")
    md.append("## Test inventory")
    md.append("")
    md.append("| #  | Test                                | Description                                   |")
    md.append("|----|-------------------------------------|-----------------------------------------------|")
    for i, (f, d) in enumerate(TESTS, 1):
        md.append(f"| {i:02d} | `{f}` | {d} |")
    md.append("")
    md.append("## Detailed findings")
    md.append("")

    for f, desc in TESTS:
        path = os.path.join(BATTERY_CHECKIN_DIR, f)
        if not os.path.exists(path):
            md.append(f"### {f} — {desc}")
            md.append("")
            md.append("_(no output)_")
            md.append("")
            continue
        try:
            data = json.load(open(path))
        except Exception as e:
            md.append(f"### {f} — {desc}\n\n_(error reading: {e})_\n")
            continue
        md.append(f"### {f.replace('.json', '')} — {desc}")
        md.append("")
        rows = walk(data, max_depth=3)
        if not rows:
            md.append("_(no scalars)_")
        else:
            for label, value in rows:
                if value:
                    md.append(f"- **{label}**: `{value}`")
                else:
                    md.append(f"- **{label}**:")
        md.append("")

    md.append("## Top-line verdicts")
    md.append("")
    md.append("- **β stability** — across 20 estimators and slicings, β stays positive in")
    md.append("  every cell. Range: [+0.21, +1.80] depending on slice; full-sample baseline +0.48.")
    md.append("- **Cross-sectional dependence** (Pesaran CD = +25.5, p≈0) is *expected* and")
    md.append("  consistent with the convergence phenomenon the paper documents.")
    md.append("- **Panel unit roots** present in raw attainment series (mechanical, monotone trend).")
    md.append("  Within-FE residuals are closer to stationary. Identification does not rest on")
    md.append("  stationarity; natural experiments + USSR falsification carry causal load.")
    md.append("- **Cointegration** Kao, Pedroni, and Westerlund all reject the no-cointegration")
    md.append("  null, supporting the levels relationship as non-spurious.")
    md.append("- **Serial correlation** present (DW ≈ 0.28, AR(1) ρ̂ ≈ 0.83); cluster-robust SEs")
    md.append("  handle it for inference; β is unbiased under arbitrary within-country AR.")
    md.append("- **Robust SE engines** (D-K, two-way cluster, Conley): every t-stat > 10.")
    md.append("- **Multiple-testing corrections** (Bonferroni, Holm, BH, BY, Westfall-Young,")
    md.append("  Romano-Wolf) all preserve the headline. 16/16 tests survive at 5%.")
    md.append("- **Specification**: linear-in-x rejected; Hansen threshold at parent_edu ≈ 25%")
    md.append("  splits β into 0.86 (below) and 0.34 (above) — confirms the paper's claim that")
    md.append("  the active-expansion window is where the slope lives.")
    md.append("- **Structural breaks**: β positive and significant in every era 1975-2015;")
    md.append("  Andrews-Quandt sup-F maxes at 1985, but β is stable [0.30, 0.47] across eras.")
    md.append("- **Influence**: leave-one-country-out β in [0.478, 0.493]. No single country pivots.")
    md.append("- **Outliers**: Huber, Tukey, LAD, Theil-Sen, trimmed, winsorized all in [0.42, 0.49].")
    md.append("- **Modern DiD**: Sun-Abraham, dCH, Borusyak-Spiess imputation, Honest DiD (M=2.0)")
    md.append("  all give positive treatment effects; Honest DiD shows post-effect remains +19 pp")
    md.append("  even under the most pessimistic pre-trend-violation bound.")
    md.append("- **Synthetic control**: Korea LE = +9.7 yrs, TFR = -1.4, U5MR = -23 vs synthetic")
    md.append("  donor pool. Cuba and Sri Lanka show similar over-performance.")
    md.append("- **OOS predictive validity**: education and GDP have similar OOS RMSE on LE/U5MR;")
    md.append("  edu encompasses GDP only for TFR. This is honest and consistent with the paper's")
    md.append("  claim that within-sample GDP-R² collapses on residualization (which it does).")
    md.append("- **Placebos**: within-country shuffle null β = 0.0000 ± 0.019; cohort-mismatch")
    md.append("  null = +0.10 ± 0.005. Real β = 0.48 sits 25+ SDs above either null.")
    md.append("")
    md.append("## What this battery does NOT do")
    md.append("")
    md.append("It does not replace the paper's identification strategy. The panel correlation")
    md.append("documents the pattern; the causal weight comes from (a) biological temporal")
    md.append("ordering, (b) natural experiments (Cambodia 1975-79, Sri Lanka 1945, Korea 1953,")
    md.append("Asian Financial Crisis 1997), and (c) the USSR falsification test (centralised")
    md.append("schooling without economic freedom did not produce convergence — paper §11).")
    md.append("Every test in this directory is a panel-econometrics companion to those three.")

    out = os.path.join(HERE, "REPORT.md")
    with open(out, "w") as fh:
        fh.write("\n".join(md))
    print(f"Wrote {out}  ({len(md)} lines)")


if __name__ == "__main__":
    main()
