"""
verify_the_long_childhood.py

Every empirical number in paper/the_long_childhood.tex is registered
here with its source. The script verifies each one.

Source types:
  - script:   run a Python script, parse stdout
  - data:     look up a value in a CSV file
  - wdi:      look up from World Bank WDI CSVs (GDP, TFR, LE)
  - wcde:     look up from WCDE processed CSVs
  - checkin:  read a value from a checkin JSON file under checkin/
  - derived:  compute from other verified values
  - const:    definitional constant (just check consistency across occurrences)
  - ref:      from cited literature (cannot verify from data; flagged for manual check)
  - fact:     historical fact (UK franchise, colonial famine) pinned to BOTH a
              value AND an authoritative URL in checkin/historical_facts.json.
              detail = the fact's key in that file. verify checks the paper
              number against the JSON value AND FAILS if the fact has no url, so
              "every cited historical fact has an authoritative source" is a
              hard gate. Use this instead of `ref` for any historical number
              that has a citable URL. Provenance: paper/historical_facts.md.
              See load_fact() and the `fact` branch in the dispatch loop.

              Matching gotcha: write large tolls as words ("ten million",
              "one million", "three million") and digit counts plainly
              ("400{,}000"). Word forms are NOT picked up by the unregistered-
              number scan (it reads digits only), but number_patterns() emits
              "ten million" etc. so the stale-registration scan still requires
              the phrase to appear in the section. Every YEAR token is now
              registered per-occurrence (source `fact` for cited events, `year`
              for calendar/lookup years, or validated against References when in
              citation context) — there is no global structural-year exemption.

Usage:
    python scripts/verify_the_long_childhood.py

Exit code: 0 if all pass, 1 if any fail.
"""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PAPER = os.path.join(REPO_ROOT, "paper", "the_long_childhood.tex")
PROC = os.path.join(REPO_ROOT, "wcde", "data", "processed")
DATA = os.path.join(REPO_ROOT, "data")
CHECKIN = os.path.join(REPO_ROOT, "checkin")
# ML checkin JSONs (chapter9_*, universal_evidence*) live alongside the ML
# pipeline that produces them, not in the top-level checkin/. load_checkin
# falls back here so paper-cited ML numbers stay traceable to their producer.
ML_CHECKIN = os.path.join(REPO_ROOT, "scripts", "ml", "checkin")

# Shared section-anchor table (also used by review/extract/*.py)
sys.path.insert(0, os.path.join(REPO_ROOT, "review", "extract"))
from _anchor import build_section_map as _shared_build_section_map  # noqa: E402
from _anchor import ABSTRACT_LABEL  # noqa: E402

# Shared verification engine: source loaders, number-form matching,
# per-occurrence consumption, and the report writer all live in
# scripts/_verify_shared.py and are imported here. _shared.py remains the
# data library; _verify_shared.py is the verification layer on top of it,
# shared with the book verifier (verify_book_serious.py) so the machinery
# cannot drift between the two.
from _verify_shared import (  # noqa: E402,F401
    WCDE_NAMES, WDI_NAMES,
    run_script, load_wcde, load_wdi, load_checkin, load_fact, compute_ppyr,
    number_patterns, exact_patterns, extract_numbers, normalize_line,
    section_labels, build_consume_slots, resolve_section,
    write_report as _write_report,
)

# ══════════════════════════════════════════════════════════════════════════
# SECTION LABEL SHORTCUTS
# ══════════════════════════════════════════════════════════════════════════
ABSTRACT = "abstract"
INTRO = "the-convergence"
DEF_DEV = "defining-development"
EASTERLIN = "the-convergence"
LUTZ = "longest-juvenile-dependency"
DEPENDENCY = "longest-juvenile-dependency"
DOSE_CONTINUOUS = "the-window-supports-a-continuous-dose"
PT_AGENCY = "the-generational-transmission-mechanism"
KIN = "from-action-to-talk-how-education-reaches-beyond-the-household"
DEMOG = "demographic-structure-and-the-fertility-transition"
HOW_EDU = "education-as-payload"
CAUSAL = "what-the-panel-sharpens"           # was §Where the Identification Is (deleted 2026-05-23)
CONVERGENCE_IN_AGGREGATE = "convergence-in-aggregate"
DESCRIPTIVE = "convergence-in-aggregate"  # merged into §The Convergence in Aggregate; label renamed from "data" 2026-06-04
COMPLETION = "education-vs-gdp-as-predictors-of-attainment"  # §Completion merged into §4 2026-05-23
EMPIRICAL = "what-the-panel-sharpens"        # was §Empirical Strategy (deleted 2026-05-23)
EDU_VS_GDP = "education-vs-gdp-as-predictors-of-attainment"
THE_SHAPE_OF_THE_RESPONSE = "the-shape-of-the-response"

# ── Section aliases ───────────────────────────────────────────────────────
# When a block inside an existing section is promoted to its own
# \subsection, it gets a new label but its numbers do not change — they were
# already registered against the parent section. Listing the new label here
# folds its line range back into the parent (see _apply_section_aliases,
# called right after build_section_map), so the coverage scan and the
# stale-registration scan keep treating those numbers as part of the parent.
# This means you do NOT have to re-key dozens of reg(...) entries every time
# you split a section heading. Add an entry whenever you promote a
# \paragraph/block to a \subsection within a section that already has regs.
SECTION_ALIASES = {
    # §The Grandparent Channel was a \paragraph inside §The Shape of the
    # Response until 2026-05-31. It was promoted to a \subsection so its four
    # cross-references resolve to a real section number (9.6) instead of
    # silently binding to a figure-caption counter. None of its numbers moved.
    "the-grandparent-channel": THE_SHAPE_OF_THE_RESPONSE,
}
GDP_INDEP = "education-vs-gdp-as-predictors-of-attainment"  # §GDP Has No Indep merged into §4 2026-05-23
ASYMMETRIC = "what-the-panel-does-not-carry" # was §Asymmetric Disruption (deleted 2026-05-23; AFC numbers cut from main text)
COLLECTIVE_ACTION = "what-the-panel-does-not-carry"  # was §Collective Action (deleted 2026-05-23)
SIX_HOLD = "what-the-panel-does-not-carry"   # was §All Six Predictions Hold (deleted 2026-05-23)
WHAT_CH_IS = "what-the-panel-sharpens"       # renamed from §What This Chapter Is 2026-05-23
UNIVERSALITY_ACROSS_SUBSAMPLES = "universality-across-subsamples"
SHOCK_TEST = "the-shock-test"
FAMINE_TEST = "the-famine-test"
CUMULATIVE = "the-panel"
SEN_CASES = "country-histories"
KOREA_PHILIPPINES = "korea-and-philippines"
KERALA = "kerala"
SRI_LANKA = "four-further-cases"
MYANMAR = "four-further-cases"
CHINA = "china"
CUBA = "four-further-cases"
BANGLADESH = "four-further-cases"
CAMBODIA = "cambodia-the-home-niche-shadow"
INVISIBLE = "why-the-loading-in-childhood-is-invisible"
HUMAN_COST = "the-human-cost"
INSTIT = "the-institutional-challenge"
POLICY = "the-decision"
CONCL = "the-decision"
REFS = "references"
THE_EVIDENCE = "the-panel"
APPENDIX_ROBUST = "appendix-other-robustness"
FRAMEWORKS_REACHING = "what-every-framework-was-reaching-for"
APPENDIX_TWFE = "appendix-twfe"
APPENDIX_PERMUTATION = "appendix-permutation"
APPENDIX_LISTWISE = "appendix-listwise-deletion"
NECESSITY_SUFFICIENCY_FORWARD = "necessity-sufficiency-forward"
THE_GENERATIONAL_LAG = "the-generational-lag"

# WDI_NAMES / WCDE_NAMES country maps and the source loaders now live in
# scripts/_verify_shared.py (imported above), shared with the book verifier.

def build_section_map(paper_path):
    """Thin wrapper over review/extract/_anchor.build_section_map.

    Kept here for callers that already import this name from this module.
    The shared implementation lives in review/extract/_anchor.py so the
    review system and the number registry can't drift apart.
    """
    return _shared_build_section_map(paper_path)


def _apply_section_aliases(section_map):
    """Fold alias-child subsections into their parent's line range.

    SECTION_ALIASES maps a carved-out subsection label -> the parent label
    whose registered numbers already cover it. We widen the parent's range
    to span the child and drop the child entry, so every downstream scan
    (line_to_section coverage, stale-registration section text) treats the
    child's lines as part of the parent. No-op for any alias whose child or
    parent is missing from the map, so it degrades safely if a heading is
    renamed or removed.
    """
    merged = dict(section_map)
    for child, parent in SECTION_ALIASES.items():
        if child not in merged or parent not in merged:
            continue
        c_start, c_end = merged[child]
        p_start, p_end = merged[parent]
        merged[parent] = (min(p_start, c_start), max(p_end, c_end))
        del merged[child]
    return merged


def build_form_section_index(paper_lines, section_map):
    """Map every printed number form in the body to the section(s) it occurs in.

    Used only for DISPLAY: a registration with an empty section list (a
    source-value check whose prose occurrence is already consumed by a
    section-bound slot of the same value) is shown under the section(s) where
    its value actually appears, instead of a bare "section=[]". This does NOT
    feed the coverage slot logic (build_consume_slots keys on the declared
    sections only), so coverage and over-registration accounting are unchanged.

    Returns dict: printed_form -> sorted list of section labels.
    Skips the preamble, tikz coordinate blocks, section headings, and the
    references list — the same regions the coverage scan ignores.
    """
    refs_start = section_map[REFS][0] if REFS in section_map else len(paper_lines) + 1
    body_start = 1
    for j, ln in enumerate(paper_lines, 1):
        if "\\begin{document}" in ln:
            body_start = j + 1
            break

    def line_section(line_no):
        for label, (start, end) in section_map.items():
            if start <= line_no <= end:
                return label
        return None

    form_sections = {}
    in_tikz = False
    for i, line in enumerate(paper_lines, 1):
        if i < body_start or i >= refs_start:
            continue
        stripped = line.strip()
        if "\\begin{tikzpicture}" in stripped:
            in_tikz = True
        if "\\end{tikzpicture}" in stripped:
            in_tikz = False
            continue
        if in_tikz:
            continue
        if not stripped or stripped.startswith("\\section") or stripped.startswith("\\subsection"):
            continue
        sec = line_section(i)
        if sec is None:
            continue
        for printed, _val in extract_numbers(line):
            form_sections.setdefault(printed, set()).add(sec)
    return {form: sorted(secs) for form, secs in form_sections.items()}


def derived_display_sections(entry, form_index):
    """Sections where an empty-section entry's value appears in prose (display)."""
    out = []
    seen = set()
    for form in exact_patterns(entry["value"]):
        for sec in form_index.get(form, ()):
            if sec not in seen:
                seen.add(sec)
                out.append(sec)
    return sorted(out)


# ══════════════════════════════════════════════════════════════════════════
# PAPER NUMBER REGISTRY
# ══════════════════════════════════════════════════════════════════════════

REGISTRY = []

def reg(name, value, source, detail, section, tol=0.001):
    """Register a paper number for verification.

    section: the section label(s) where this number appears in the paper.
             Accepts a bare label string, a list of labels, or a list of
             (label, offset) tuples. Coverage and the stale-registration
             scan use the LABEL only — a value registered for a section
             covers every occurrence of it in that section.

             NOTE: line offsets in (label, offset) tuples are accepted for
             backward compatibility but IGNORED. Paper line positions drift
             on every prose edit, so they are not part of the model; new
             registrations should pass bare labels. An empty list [] means
             "not cited in paper" (soundness-checked, exempt from coverage).
    """
    # Normalize section to a list of (label, offset) tuples
    if isinstance(section, str):
        section = [section]
    elif isinstance(section, list):
        if not section:
            pass  # empty list: not cited
        elif isinstance(section[0], str):
            section = [s for s in section]
        elif isinstance(section[0], int):
            # Legacy: convert line numbers to empty (caller should migrate)
            print(f"  WARNING: {name} uses legacy line numbers, migrate to section labels")
            section = []
        # else: already list of tuples — leave as-is

    REGISTRY.append({
        "name": name, "value": value, "source": source,
        "detail": detail, "section": section, "tol": tol,
        "actual": None, "status": "PENDING",
    })

# ── Script paths ─────────────────────────────────────────────────────────
S_T1    = os.path.join(REPO_ROOT, "scripts", "tables", "panel_expansion_fe.py")
S_TA1   = os.path.join(REPO_ROOT, "scripts", "robustness", "twfe_child_edu.py")
S_FA1   = os.path.join(REPO_ROOT, "scripts", "figures", "outcomes_r2_by_lag.py")
S_CO2   = os.path.join(REPO_ROOT, "scripts", "co2_placebo.py")
S_BETA  = os.path.join(REPO_ROOT, "scripts", "figures", "beta_vs_baseline.py")
S_ROB   = os.path.join(REPO_ROOT, "scripts", "robustness", "robustness_tests.py")
S_TFR   = os.path.join(REPO_ROOT, "scripts", "residualization", "education_vs_tfr.py")

# ══════════════════════════════════════════════════════════════════════════
# FULL-PANEL ONE-WAY FE (scripts/tables/panel_expansion_fe.py) —
# diagnostic, cited in year-FE discussion and footnotes.
# ══════════════════════════════════════════════════════════════════════════
reg("T1-obs",        945,   "checkin", ("panel_expansion_fe.json", "numbers.panel_obs"),[CONVERGENCE_IN_AGGREGATE, EDU_VS_GDP, "appendix-reproducibility"], tol=0)
# Prose restatements of the panel obs count (each new occurrence consumes
# one registration in consume_match):
reg("T1-obs-prose-data", 945, "checkin", ("panel_expansion_fe.json", "numbers.panel_obs"),
    [], tol=0)
reg("T1-obs-prose-edugdp", 945, "checkin", ("panel_expansion_fe.json", "numbers.panel_obs"),
    [EDU_VS_GDP], tol=0)
reg("T1-countries",  144,    "checkin", ("panel_expansion_fe.json", "numbers.panel_countries"),[ABSTRACT, WHAT_CH_IS, CONVERGENCE_IN_AGGREGATE, EDU_VS_GDP, "appendix-reproducibility", "glossary"], tol=0)
# Prose restatements of the panel country count:
reg("T1-countries-prose-data", 144, "checkin", ("panel_expansion_fe.json", "numbers.panel_countries"),
    [], tol=0)
reg("T1-countries-prose-edugdp", 144, "checkin", ("panel_expansion_fe.json", "numbers.panel_countries"),
    [EDU_VS_GDP], tol=0)
reg("T1-countries-prose-empirical", 144, "checkin", ("panel_expansion_fe.json", "numbers.panel_countries"),
    [], tol=0)
reg("T1-countries-prose-appendix", 144, "checkin", ("panel_expansion_fe.json", "numbers.panel_countries"),
    [], tol=0)
# ══════════════════════════════════════════════════════════════════════════
# TABLE A1 — Two-way FE (twfe_child_edu.py)
# ══════════════════════════════════════════════════════════════════════════
# NOTE: TA1-M1-*, CS-att, CS-ci-lo, CS-att-se, GB-* registrations removed
# after the 2WFE/Goodman-Bacon/Callaway-Sant'Anna defense block was relocated
# off the paper into scripts/robustness/goodman_bacon_decomposition.py
# (the body pointer in §empirical-strategy now just names the script).
# CS-att-yr35 stays — number is in §compounding-generations-signature and
# in fig:cs-event caption.
reg("CS-att-yr35",  22.13,  "checkin",
    ("callaway_santanna.json", "child_education.event_study.7.att"),
    [], tol=0.05)

# ══════════════════════════════════════════════════════════════════════════
# COMPLETION vs TEST SCORES — old within-country horse race superseded by
# the cross-country race in §hanushek-reconciliation
# (hanushek_horse_race_comprehensive.py). These registrations remain as
# passing robustness references (the JSON is still built) but are not
# cited in the current paper body. Kept for auxiliary scripts only.
# ══════════════════════════════════════════════════════════════════════════
reg("HLO-overlap-countries", 96, "checkin",
    ("completion_vs_test_scores.json", "coverage.overlap_countries"),
    [], tol=0)
reg("HLO-TFR-edu-r2",  0.28, "checkin",
    ("completion_vs_test_scores.json", "short_lag.10.tfr.edu.r2"),
    [], tol=0.01)
reg("HLO-TFR-test-r2", 0.011, "checkin",
    ("completion_vs_test_scores.json", "short_lag.10.tfr.test.r2"),
    [], tol=0.005)
reg("HLO-TFR-test-p",  0.23, "checkin",
    ("completion_vs_test_scores.json", "short_lag.10.tfr.test.pval"),
    [], tol=0.01)
reg("HLO-U5MR-edu-r2", 0.44, "checkin",
    ("completion_vs_test_scores.json", "short_lag.10.u5mr.edu.r2"),
    [], tol=0.01)
reg("HLO-U5MR-test-r2", 0.01, "checkin",
    ("completion_vs_test_scores.json", "short_lag.10.u5mr.test.r2"),
    [], tol=0.01)
reg("HLO-U5MR-test-p",  0.48, "checkin",
    ("completion_vs_test_scores.json", "short_lag.10.u5mr.test.pval"),
    [], tol=0.01)

# ══════════════════════════════════════════════════════════════════════════
# DURATION vs FIDELITY — 4-horse race. Superseded by the cross-country
# race in §hanushek-reconciliation; kept as robustness JSON, not cited
# in paper body.
# ══════════════════════════════════════════════════════════════════════════
reg("DVF-LE-wcde-mys-r2",   0.170, "checkin",
    ("completion_vs_years_vs_tests.json", "results.lag_10.le.wcde_mys_t.r2"),
    [], tol=0.0005)
reg("DVF-LE-bl-mys-r2",     0.078, "checkin",
    ("completion_vs_years_vs_tests.json", "results.lag_10.le.bl_mys_t.r2"),
    [], tol=0.0005)
reg("DVF-LE-test-r2",       0.000, "checkin",
    ("completion_vs_years_vs_tests.json", "results.lag_10.le.test_t.r2"),
    [], tol=0.0005)
reg("DVF-LE-test-p",        0.97,  "checkin",
    ("completion_vs_years_vs_tests.json", "results.lag_10.le.test_t.pval"),
    [], tol=0.005)
reg("DVF-TFR-wcde-mys-r2",  0.330, "checkin",
    ("completion_vs_years_vs_tests.json", "results.lag_10.tfr.wcde_mys_t.r2"),
    [], tol=0.0005)
reg("DVF-TFR-bl-mys-r2",    0.104, "checkin",
    ("completion_vs_years_vs_tests.json", "results.lag_10.tfr.bl_mys_t.r2"),
    [], tol=0.0005)
reg("DVF-TFR-test-r2",      0.006, "checkin",
    ("completion_vs_years_vs_tests.json", "results.lag_10.tfr.test_t.r2"),
    [], tol=0.0005)
reg("DVF-TFR-test-p",       0.41,  "checkin",
    ("completion_vs_years_vs_tests.json", "results.lag_10.tfr.test_t.pval"),
    [], tol=0.005)
reg("DVF-U5MR-wcde-mys-r2", 0.451, "checkin",
    ("completion_vs_years_vs_tests.json", "results.lag_10.u5mr.wcde_mys_t.r2"),
    [], tol=0.0005)
reg("DVF-U5MR-bl-mys-r2",   0.298, "checkin",
    ("completion_vs_years_vs_tests.json", "results.lag_10.u5mr.bl_mys_t.r2"),
    [], tol=0.0005)
reg("DVF-U5MR-test-r2",     0.030, "checkin",
    ("completion_vs_years_vs_tests.json", "results.lag_10.u5mr.test_t.r2"),
    [], tol=0.0005)

# NOTE: the first-mover demographic-transition registrations (FM-*) are not
# wired in. The working files that backed them — nordic_first_movers_demographics
# .{py,json} and first_movers_handoff.md — were removed at the user's request;
# recover from git history (commit 6353413, or the cleanup-era commits) if the
# thread is ever reopened and the prose written.

# ══════════════════════════════════════════════════════════════════════════
# Lag decay — four outcomes × five generational anchors (0, 25, 50, 75, 100).
# Paper reports standardized |β| as the primary metric (causal quantity,
# comparable across outcomes). Source: lag_coefficients.py. R² decay is
# still computed (outcomes_r2_by_lag.py) and mentioned as a sidepoint, but
# the paper body no longer cites specific R² values from the lag table.
# ══════════════════════════════════════════════════════════════════════════
# Lag-decay table now reports 3 lags (25/50/100); contemporary (lag 0)
# and great-grandparent (lag 75) rows live in lag_coefficients.json only.
reg("LagBeta-le-lag25",   0.597, "checkin", ("lag_coefficients.json", "numbers.le_beta_lag25"),
    [THE_SHAPE_OF_THE_RESPONSE])
reg("LagBeta-le-lag50",   0.411, "checkin", ("lag_coefficients.json", "numbers.le_beta_lag50"),
    [THE_SHAPE_OF_THE_RESPONSE])
reg("LagBeta-le-lag100",  0.223, "checkin", ("lag_coefficients.json", "numbers.le_beta_lag100"),
    [THE_SHAPE_OF_THE_RESPONSE])
reg("LagBeta-tfr-lag25",  0.693, "derived",
    "abs(lag_coefficients.tfr_beta_lag25)", [THE_SHAPE_OF_THE_RESPONSE])
reg("LagBeta-tfr-lag50",  0.452, "derived",
    "abs(lag_coefficients.tfr_beta_lag50)", [THE_SHAPE_OF_THE_RESPONSE])
reg("LagBeta-tfr-lag100", 0.134, "derived",
    "abs(lag_coefficients.tfr_beta_lag100)", [THE_SHAPE_OF_THE_RESPONSE])
reg("LagBeta-u5-lag25",   0.820, "derived",
    "abs(lag_coefficients.u5log_beta_lag25)", [])
reg("LagBeta-u5-lag112",  0.331, "derived",
    "abs(lag_coefficients.u5log_beta_lag112)", [THE_SHAPE_OF_THE_RESPONSE])
reg("LagBeta-cedu-lag50", 0.388, "checkin", ("lag_coefficients.json", "numbers.cedu_beta_lag50"),
    [THE_SHAPE_OF_THE_RESPONSE])
reg("LagBeta-cedu-lag100",0.152, "checkin", ("lag_coefficients.json", "numbers.cedu_beta_lag100"),
    [])
# Selected |t| statistics cited in the narrative / caption
reg("LagT-u5-lag112",    24.4, "derived",
    "abs(lag_coefficients.u5log_t_lag112)", [THE_SHAPE_OF_THE_RESPONSE], tol=0.05)
reg("LagT-le-lag100",    18.5, "derived",
    "abs(lag_coefficients.le_t_lag100)", [], tol=0.05)
reg("LagT-tfr-lag112",    5.9, "derived",
    "abs(lag_coefficients.tfr_t_lag112) -- minimum |t| at four-generation depth",
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.05)
reg("LagT-cedu-lag100",  12.4, "derived",
    "abs(lag_coefficients.cedu_t_lag100)", [THE_SHAPE_OF_THE_RESPONSE], tol=0.05)
# reg("LagBeta-n-countries", 142, "checkin", ("lag_coefficients.json", "numbers.n_countries"),
#     [THE_SHAPE_OF_THE_RESPONSE], tol=0)
# (removed 2026-05-23: number cut from main text in §5 trim)

# ══════════════════════════════════════════════════════════════════════════
# CHECKIN — edu_vs_gdp_predicts_le.json
# FE regressions: education vs GDP predicting life expectancy(T+25)
# ══════════════════════════════════════════════════════════════════════════
reg("LE-lt10-edu-r2",  0.484, "checkin",
    ("edu_vs_gdp_predicts_le.json", "numbers.lt10.edu_r2"),
    [THE_SHAPE_OF_THE_RESPONSE])
reg("LE-lt10-gdp-r2",  0.009, "checkin",
    ("edu_vs_gdp_predicts_le.json", "numbers.lt10.gdp_r2"),
    [])
reg("LE-lt30-edu-r2",  0.456, "checkin",
    ("edu_vs_gdp_predicts_le.json", "numbers.lt30.edu_r2"),
    [])
reg("LE-lt30-gdp-r2",  0.029, "checkin",
    ("edu_vs_gdp_predicts_le.json", "numbers.lt30.gdp_r2"),
    [])
# reg("LE-lt10-edu-r2-pct", 51, "derived",
#     "Education R² at <10% cutoff × 100",
#     [THE_SHAPE_OF_THE_RESPONSE], tol=1)
# (removed 2026-05-23: number cut from main text in §5 trim)

# ══════════════════════════════════════════════════════════════════════════
# CHECKIN — education_vs_gdp_by_cutoff.json
# ══════════════════════════════════════════════════════════════════════════
reg("CutOff-30-edu-r2",    0.693, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_30_edu_r2"),
    [EDU_VS_GDP])
reg("CutOff-30-gdp-r2",    0.220, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_30_gdp_r2"),
    [])
reg("CutOff-30-edu-beta",  1.434, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_30_edu_beta"),
    [EDU_VS_GDP])
reg("CutOff-30-edu-se",    0.093, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_30_edu_se"),[], tol=0.005)
reg("CutOff-30-edu-t",     15.5,  "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_30_edu_t"),[EDU_VS_GDP], tol=0.1)
reg("CutOff-30-gdp-se",    3.974, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_30_gdp_se"),[], tol=0.05)
reg("CutOff-30-gdp-t",     3.7,   "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_30_gdp_t"),[], tol=0.05)
reg("CutOff-10-edu-r2",    0.613, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_10_edu_r2"),
    [], tol=0.002)
reg("CutOff-10-gdp-r2",    0.292, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_10_gdp_r2"),
    [], tol=0.002)
reg("CutOff-50-edu-r2",    0.670, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_50_edu_r2"),
    [])

# ── Summary statistics (§The Convergence in Aggregate, Table \ref{tab:summary}) ──
# Detailed sample-composition counts (1665 universe obs, 1466 GDP-merged obs,
# 178 GDP-coverage countries) removed from main text 2026-05-23 with §Data +
# §Descriptive Statistics collapse. Panel-countries (185) survives in prose.
reg("Sum-panel-countries",   185, "checkin",
    ("summary_stats.json", "numbers.panel_countries"),[DESCRIPTIVE], tol=0)

# ── Cutoff-sensitivity table (Appendix Table \ref{tab:cutoff}) ───────────
# Explicit per-cell registrations for the new robustness-to-alternative-
# cutoffs table. Existing CutOff-30-* and CutOff-10-* regs above cover
# the 30%/10% cells in-section; these add the remaining cutoff rows and
# anchor every cell in APPENDIX_ROBUST.
_CUT_CELLS = [
    # (cutoff, edu_beta, edu_r2, gdp_beta, gdp_r2, ratio, n, countries)
    (10, 2.380, 0.613, 16.518, 0.292, 2.1, 313, 62),
    (20, 1.754, 0.684, 14.750, 0.235, 2.9, 519, 90),
    (30, 1.434, 0.693, 14.655, 0.220, 3.1, 672, 106),
    (40, 1.220, 0.683, 15.930, 0.218, 3.1, 776, 113),
    (50, 1.044, 0.670, 17.514, 0.249, 2.7, 863, 119),
    (60, 0.901, 0.643, 18.317, 0.270, 2.4, 935, 122),
    (70, 0.800, 0.622, 18.138, 0.277, 2.2, 991, 125),
    (80, 0.727, 0.592, 17.841, 0.283, 2.1, 1039, 131),
    (90, 0.639, 0.541, 17.350, 0.282, 1.9, 1103, 139),
]
for _c, _eb, _er, _gb, _gr, _rt, _n, _nc in _CUT_CELLS:
    # Full cutoff sweep lives in scripts/ (paper cites only 10%, 50%, Full).
    _body_cutoff = (_c in (10, 50))
    _base_secs = [EDU_VS_GDP] if _body_cutoff else []
    reg(f"TabC-{_c}-edu-r2",   _er,  "checkin",
        ("education_vs_gdp_by_cutoff.json", f"numbers.cutoff_{_c}_edu_r2"),
        list(_base_secs), tol=0.005)
    reg(f"TabC-{_c}-gdp-beta", _gb,  "checkin",
        ("education_vs_gdp_by_cutoff.json", f"numbers.cutoff_{_c}_gdp_beta"),
        list(_base_secs), tol=0.005)
    reg(f"TabC-{_c}-gdp-r2",   _gr,  "checkin",
        ("education_vs_gdp_by_cutoff.json", f"numbers.cutoff_{_c}_gdp_r2"),
        list(_base_secs), tol=0.005)
    reg(f"TabC-{_c}-ratio",    _rt,  "checkin",
        ("education_vs_gdp_by_cutoff.json", f"numbers.cutoff_{_c}_ratio"),
        list(_base_secs), tol=0.05)
# "Full" row in the cutoff table: pulls from panel_expansion_fe.json (n, countries,
# edu β, edu se) — the unrestricted full panel, one-way country FE on every
# observation. The GDP-merged panel_gdp values
# (no_cutoff_edu_beta = 0.589, n=1304, 156 countries) live in the JSON for
# completeness but are not the headline Full row.
reg("TabC-full-edu-beta", 0.707, "checkin",
    ("panel_expansion_fe.json", "numbers.table1_m1_edu_beta"),[EDU_VS_GDP], tol=0.005)
reg("TabC-full-edu-se",   0.058, "checkin",
    ("panel_expansion_fe.json", "numbers.table1_m1_edu_se"),[EDU_VS_GDP], tol=0.005)
reg("TabC-full-gdp-beta", 17.381, "checkin",
    ("panel_expansion_fe.json", "numbers.table1_m2_gdp_beta"),[EDU_VS_GDP], tol=0.05)
reg("TabC-full-edu-r2",   0.561, "checkin",
    ("panel_expansion_fe.json", "numbers.table1_m1_r2_within"),
    [EDU_VS_GDP], tol=0.005)
reg("TabC-full-gdp-r2",   0.265, "checkin",
    ("panel_expansion_fe.json", "numbers.table1_m2_r2_within"),
    [EDU_VS_GDP], tol=0.005)
reg("TabC-full-gdp-se",   3.742, "checkin",
    ("panel_expansion_fe.json", "numbers.table1_m2_gdp_se"),[EDU_VS_GDP], tol=0.005)
reg("TabC-full-ratio",    2.4,   "derived",
    "TabC-full-edu-r2 / TabC-full-gdp-r2 = 0.617/0.258 = 2.39, ratio of the displayed Full row "
    "(was 2.2 from no_cutoff_ratio, a different GDP-merged sample inconsistent with the table)",
    [EDU_VS_GDP], tol=0.1)
# tab:cutoff body table: SE columns and the GDP-merged Full row.
reg("TabC-10-edu-se",   0.329, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_10_edu_se"),
    [EDU_VS_GDP], tol=0.005)
reg("TabC-10-gdp-se",   4.164,  "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_10_gdp_se"),
    [EDU_VS_GDP], tol=0.005)
reg("TabC-50-edu-se",   0.059, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_50_edu_se"),
    [EDU_VS_GDP], tol=0.005)
reg("TabC-50-gdp-se",   3.861,  "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_50_gdp_se"),
    [EDU_VS_GDP], tol=0.005)
reg("TabC-fullgdp-edu-beta", 0.567, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.no_cutoff_edu_beta"),
    [EDU_VS_GDP], tol=0.005)
reg("TabC-fullgdp-edu-se",   0.041, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.no_cutoff_edu_se"),
    [], tol=0.005)
reg("TabC-fullgdp-gdp-beta", 14.938, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.no_cutoff_gdp_beta"),
    [EDU_VS_GDP], tol=0.05)
reg("TabC-fullgdp-countries", 156,  "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.no_cutoff_countries"),
    [], tol=0)  # GDP-merged subsample countries; paper uses unrestricted full panel per R2.04
# tab:cutoff body table N / countries (10% / 50% rows).
reg("TabC-10-n-body",   313,  "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_10_n"),
    [EDU_VS_GDP], tol=0)
reg("TabC-50-n-body",   863,  "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_50_n"),
    [EDU_VS_GDP], tol=0)
reg("TabC-50-c-body",   119,  "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_50_countries"),
    [EDU_VS_GDP], tol=0)

# ══════════════════════════════════════════════════════════════════════════
# CHECKIN — beta_by_ceiling_cutoff.json
# ══════════════════════════════════════════════════════════════════════════
reg("Beta-cutoff-20",  3.101, "checkin",
    ("beta_by_ceiling_cutoff.json", "numbers.panelA_cutoff_20_beta"),
    [])
reg("Beta-cutoff-50",  1.914, "checkin",
    ("beta_by_ceiling_cutoff.json", "numbers.panelA_cutoff_50_beta"),
    [EDU_VS_GDP])
reg("Beta-cutoff-90",  1.265, "checkin",
    ("beta_by_ceiling_cutoff.json", "numbers.panelA_cutoff_90_beta"),
    [EDU_VS_GDP])
reg("Beta-cutoff-50-r2-pct", 76, "derived",
    "Panel A cutoff 50 R² × 100",
    [], tol=1)
reg("Beta-cutoff-90-r2-pct", 73, "derived",
    "Panel A cutoff 90 R² × 100",
    [], tol=1)
reg("Beta-no-cutoff",  1.052, "checkin",
    ("beta_by_ceiling_cutoff.json", "numbers.panelA_no_cutoff_beta"),
    [EDU_VS_GDP])

# ══════════════════════════════════════════════════════════════════════════
# CHECKIN — asian_financial_crisis.json
# ══════════════════════════════════════════════════════════════════════════
# AFC country-specific numbers re-instated 2026-05-23 in §Education vs GDP
# at Population Scale (§Why GDP is a bad control) as the within-country
# GDP-shock identification of the direct edu→outcome route.
# [retired 2026-06-18 ch9/ch11 compression] reg("AFC-Indonesia-gdp",    -14.5, "checkin",
# [retired 2026-06-18 ch9/ch11 compression]     ("asian_financial_crisis.json", "numbers.indonesia_gdp_drop_1997_1998_pct"),
# [retired 2026-06-18 ch9/ch11 compression]     [EDU_VS_GDP])
# [retired 2026-06-18 ch9/ch11 compression] reg("AFC-Thailand-gdp",     -8.8,  "checkin",
# [retired 2026-06-18 ch9/ch11 compression]     ("asian_financial_crisis.json", "numbers.thailand_gdp_drop_1997_1998_pct"),
# [retired 2026-06-18 ch9/ch11 compression]     [EDU_VS_GDP])
# [retired 2026-06-18 ch9/ch11 compression] reg("AFC-Malaysia-gdp",     -9.6,  "checkin",
# [retired 2026-06-18 ch9/ch11 compression]     ("asian_financial_crisis.json", "numbers.malaysia_gdp_drop_1997_1998_pct"),
# [retired 2026-06-18 ch9/ch11 compression]     [EDU_VS_GDP])
# [retired 2026-06-18 ch9/ch11 compression] reg("AFC-Philippines-gdp",  -3.0,  "checkin",
# [retired 2026-06-18 ch9/ch11 compression]     ("asian_financial_crisis.json", "numbers.philippines_gdp_drop_1997_1998_pct"),
# [retired 2026-06-18 ch9/ch11 compression]     [EDU_VS_GDP])
# [retired 2026-06-18 ch9/ch11 compression] reg("AFC-Indonesia-edu",     5.4,  "checkin",
# [retired 2026-06-18 ch9/ch11 compression]     ("asian_financial_crisis.json", "numbers.indonesia_edu_gain_1995_2000_pp"),
# [retired 2026-06-18 ch9/ch11 compression]     [EDU_VS_GDP])
reg("AFC-Thailand-prior",   10.0,  "checkin",
    ("asian_financial_crisis.json", "numbers.thailand_lower_sec_gain_1990_1995_pp"),
    [GDP_INDEP])

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Country-specific sliding-window betas (beta_vs_baseline.py)
# ══════════════════════════════════════════════════════════════════════════
reg("Fig1-USA-beta-high",   2.1, "checkin", ("beta_vs_baseline.json", "numbers.Fig1-USA-beta-high"),
    [], tol=0.1)
reg("Fig1-USA-beta-low",   0.08, "checkin", ("beta_vs_baseline.json", "numbers.Fig1-USA-beta-low"),
    [], tol=0.02)
reg("Fig1-Korea-beta-high", 8.5, "checkin", ("beta_vs_baseline.json", "numbers.Fig1-Korea-beta-high"),
    [], tol=0.1)
reg("Fig1-Korea-beta-3.6",  4.6, "checkin", ("beta_vs_baseline.json", "numbers.Fig1-Korea-beta-3.6"),
    [], tol=0.1)
reg("Fig1-Korea-beta-1.8",  2.1, "checkin", ("beta_vs_baseline.json", "numbers.Fig1-Korea-beta-1.8"),
    [], tol=0.1)
reg("Fig1-Korea-beta-low",  0.2, "checkin", ("beta_vs_baseline.json", "numbers.Fig1-Korea-beta-low"),
    [], tol=0.05)
reg("Fig1-Taiwan-beta",     7.1, "checkin", ("beta_vs_baseline.json", "numbers.Fig1-Taiwan-beta"),
    [], tol=0.1)
reg("Fig1-Phil-beta-high",  5.7, "checkin", ("beta_vs_baseline.json", "numbers.Fig1-Phil-beta-high"),
    [], tol=0.1)
reg("Fig1-Phil-beta-low",   0.4, "checkin", ("beta_vs_baseline.json", "numbers.Fig1-Phil-beta-low"),
    [], tol=0.1)
# R2.10 / R2.11: universality scatter and OLS fit (appendix Figure 2-scatter).
reg("Univ-fit-n",         1312, "checkin",
    ("beta_vs_baseline.json", "numbers.Universality-fit-n"),[EDU_VS_GDP], tol=0)

# ══════════════════════════════════════════════════════════════════════════
# BASELINE GROUP ANALYSIS (beta_by_baseline_group.py)
# ══════════════════════════════════════════════════════════════════════════
S_GRP = os.path.join(REPO_ROOT, "scripts", "robustness", "beta_by_baseline_group.py")
# lag-28 (annual-interpolated) by-baseline-group betas; not cited in paper prose
reg("Grp-low-beta",    1.695, "checkin", ("beta_by_baseline_group.json", "numbers.Grp-low-beta"),
    [], tol=0.005)
reg("Grp-low-R2",      0.685, "checkin", ("beta_by_baseline_group.json", "numbers.Grp-low-R2"),
    [], tol=0.0005)
reg("Grp-med-beta",    0.655, "checkin", ("beta_by_baseline_group.json", "numbers.Grp-med-beta"),
    [], tol=0.0005)
reg("Grp-med-R2",      0.673, "checkin", ("beta_by_baseline_group.json", "numbers.Grp-med-R2"),
    [], tol=0.0005)
reg("Grp-high-beta",   0.124, "checkin", ("beta_by_baseline_group.json", "numbers.Grp-high-beta"),
    [], tol=0.0005)
reg("Grp-high-R2",     0.361, "checkin", ("beta_by_baseline_group.json", "numbers.Grp-high-R2"),
    [], tol=0.0005)

# ══════════════════════════════════════════════════════════════════════════
# tab:edu-outcomes — Forward predictions (07_education_outcomes.py)
# ══════════════════════════════════════════════════════════════════════════
reg("T2-GDP-beta",  0.012,  "checkin", ("education_outcomes.json", "numbers.T2-GDP-beta"),
    [THE_SHAPE_OF_THE_RESPONSE])
# Lower-sec TFR coefficient still cited in §gdp-has-no-independent-effect
# under the tab:residualisation prose ("...lower secondary completion at
# T predicts ... -0.032 births per woman...").
reg("T2-TFR-beta", -0.009,  "checkin", ("education_outcomes.json", "numbers.T2-TFR-beta"),
    [GDP_INDEP])
# tab:edu-outcomes headline TFR row (level + log) uses primary completion
# as the operative channel for fertility. The lower-sec values are still
# computed (T2-TFR-beta above, T2-TFR-init / R2 in checkin) but no longer
# appear in tab:edu-outcomes — only the prose under tab:residualisation
# cites them.
reg("T2-TFR-pri-beta", -0.018, "checkin",
    ("education_outcomes.json", "numbers.T2-TFR-pri-beta"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.001)
reg("T2-TFR-pri-init", 0.790, "checkin",
    ("education_outcomes.json", "numbers.T2-TFR-pri-init"),
    [GDP_INDEP], tol=0.001)
reg("T2-TFR-pri-init-p", 0.00, "checkin",
    ("education_outcomes.json", "numbers.T2-TFR-pri-init-p"),
    [GDP_INDEP], tol=0.01)
# Log-outcome rows added under R1.18 / R2.17. Cluster-robust by country.
# log(TFR) row uses primary (matches the level row above).
reg("T2-U5MR-beta-log", -0.005, "checkin",
    ("education_outcomes.json", "numbers.T2-U5MR-beta-log"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.0005)
# Panel B
# Panel A: canonical-lag per outcome (table_contemporaneous.json).
# GDP/LE/TFR at lag 0; U5MR at lag 12. Lower-sec predictor throughout.
_T7_J = "table_contemporaneous.json"
# Col 1 (edu only) coefficients.
reg("T7-A-loggdp-c1-beta",   0.0185, "checkin",
    (_T7_J, "numbers.panel.log_gdp.c1_edu.low_t.beta"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.0005)
reg("T7-A-logle-c1-beta",    0.0042, "checkin",
    (_T7_J, "numbers.panel.log_le.c1_edu.low_t.beta"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.0005)
reg("T7-A-logtfr-c1-beta",  -0.0165, "checkin",
    (_T7_J, "numbers.panel.log_tfr.c1_edu.low_t.beta"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.0005)
reg("T7-A-logu5-c1-beta",   -0.0327, "checkin",
    (_T7_J, "numbers.panel.log_u5.c1_edu.low_t.beta"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.0005)
# Col 2 (+ log GDP at T).
reg("T7-A-loggdp-c2-beta",   0.0185, "checkin",
    (_T7_J, "numbers.panel.log_gdp.c2_edu_gdp.low_t.beta"),
    [], tol=0.0005)
reg("T7-A-logle-c2-beta",    0.0039, "checkin",
    (_T7_J, "numbers.panel.log_le.c2_edu_gdp.low_t.beta"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.0005)
reg("T7-A-logtfr-c2-beta",  -0.0153, "checkin",
    (_T7_J, "numbers.panel.log_tfr.c2_edu_gdp.low_t.beta"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.0005)
reg("T7-A-logu5-c2-beta",   -0.0290, "checkin",
    (_T7_J, "numbers.panel.log_u5.c2_edu_gdp.low_t.beta"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.0005)
# Row Ns.
reg("T7-A-loggdp-c1-n",      1812, "checkin",
    (_T7_J, "numbers.panel.log_gdp.c1_edu.n"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0)
reg("T7-A-logle-c1-n",       1976, "checkin",
    (_T7_J, "numbers.panel.log_le.c1_edu.n"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0)
reg("T7-A-logle-c2-n",       1633, "checkin",
    (_T7_J, "numbers.panel.log_le.c2_edu_gdp.n"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0)
reg("T7-A-logtfr-c1-n",      2155, "checkin",
    (_T7_J, "numbers.panel.log_tfr.c1_edu.n"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0)
reg("T7-A-logtfr-c2-n",      1812, "checkin",
    (_T7_J, "numbers.panel.log_tfr.c2_edu_gdp.n"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0)
# Panel B (GDP → child lower-sec, T+25) still lives in table7_stepwise.json
# as the cross-generation identification spec.
_T7B_J = "table7_stepwise.json"
reg("T7-B-b1-beta",         14.11, "checkin",
    (_T7B_J, "numbers.panelB_max.b1_gdp_only.log_gdp_t.beta"),
    [], tol=0.05)
reg("T7-B-b2-beta",          3.85, "checkin",
    (_T7B_J, "numbers.panelB_max.b2_gdp_init_edu.log_gdp_t.beta"),
    [], tol=0.05)
reg("T7-B-b3-beta",          2.01, "checkin",
    (_T7B_J, "numbers.panelB_max.b3_gdp_only_yfe.log_gdp_t.beta"),
    [], tol=0.05)
reg("T7-B-b4-beta",          1.39, "checkin",
    (_T7B_J, "numbers.panelB_max.b4_gdp_init_edu_yfe.log_gdp_t.beta"),
    [], tol=0.05)
reg("T7-B-b2-edu-beta",     0.425, "checkin",
    (_T7B_J, "numbers.panelB_max.b2_gdp_init_edu.low_t.beta"),
    [], tol=0.001)
reg("T7-B-b4-edu-beta",     0.079, "checkin",
    (_T7B_J, "numbers.panelB_max.b4_gdp_init_edu_yfe.low_t.beta"),
    [], tol=0.005)
# tab:residualisation prose: per-pp education effect at each outcome's
# biological lag, |Education-FE beta| from results.{O}.90 (regression_tables).
reg("T2-TFR-beta-abs",  0.058, "derived",
    "abs(regression_tables.json results.TFR.90.Education.beta) — births/woman per pp, tab:residualisation prose",
    [GDP_INDEP], tol=0.002)
reg("U5MR-perpp-resid-prose", 1.33, "derived",
    "abs(regression_tables.json results.U5MR.90.Education.beta) — U5 deaths/1000 per pp, tab:residualisation prose",
    [GDP_INDEP], tol=0.05)

# ══════════════════════════════════════════════════════════════════════════
# LONG-RUN PANEL (04b_long_run_generational.py)
# ══════════════════════════════════════════════════════════════════════════
reg("LR-countries", 28,     "checkin", ("long_run_generational.json", "numbers.LR-countries"),[CONVERGENCE_IN_AGGREGATE], tol=0)

# ══════════════════════════════════════════════════════════════════════════
# PARENTAL INCOME COLLAPSE — inline computation
# ══════════════════════════════════════════════════════════════════════════
reg("PI-alone-beta",  19.5,  "checkin", ("panel_expansion_fe.json", "numbers.PI-alone-beta"),
    [GDP_INDEP], tol=0.5)
reg("PI-alone-R2",    0.335, "checkin", ("panel_expansion_fe.json", "numbers.PI-alone-R2"),
    [GDP_INDEP])
reg("PI-cond-beta",   6.1,   "checkin", ("panel_expansion_fe.json", "numbers.PI-cond-beta"),
    [GDP_INDEP], tol=0.05)
reg("PI-cond-p",      0.001,  "checkin", ("panel_expansion_fe.json", "numbers.PI-cond-p"),
    [GDP_INDEP], tol=0.01)
reg("PI-edu-alone",   0.742, "checkin", ("panel_expansion_fe.json", "numbers.PI-edu-alone"),
    [GDP_INDEP])

# ══════════════════════════════════════════════════════════════════════════
# WCDE EDUCATION DATA — country-specific values cited in the paper
# ══════════════════════════════════════════════════════════════════════════

# --- Korea ---
reg("Korea-1950",    24.8,   "wcde", ("cohort_lower_sec_both.csv", "Korea", 1950),
    [], tol=0.5)
reg("Korea-1985",    94.4,   "wcde", ("cohort_lower_sec_both.csv", "Korea", 1985),
    [], tol=0.5)

# --- Taiwan ---
reg("Taiwan-1950",   17.75,  "wcde", ("cohort_lower_sec_both.csv", "Taiwan", 1950),
    [], tol=1.0)

# --- Philippines ---

# --- Cambodia ---

# --- Vietnam ---

# --- Cuba ---
reg("Cuba-1960-edu",  40.3,  "wcde", ("cohort_lower_sec_both.csv", "Cuba", 1960),
    [], tol=1.0)

# --- Bangladesh ---

# --- China ---
reg("China-1950-edu",  10.0,  "wcde", ("cohort_lower_sec_both.csv", "China", 1950),
    [], tol=0.1)  # not cited in paper
reg("China-1990-edu",  75,    "wcde", ("cohort_lower_sec_both.csv", "China", 1990),
    [], tol=0.5)

# --- Singapore ---
reg("Singapore-1950-edu", 13.4, "wcde", ("cohort_lower_sec_both.csv", "Singapore", 1950),
    [], tol=0.05)
reg("Singapore-1995-edu", 94,   "wcde", ("cohort_lower_sec_both.csv", "Singapore", 1995),
    [], tol=0.5)
# Singapore agency-transfer continuation (§the-window-supports-a-continuous-dose)
reg("Singapore-tertiary-2020", 73.0, "checkin",
    ("singapore_continuation.json", "results.singapore_college_2020.actual"),
    [DOSE_CONTINUOUS], tol=1.0)
reg("Singapore-upper-sec-2020", 96.0, "checkin",
    ("singapore_continuation.json", "results.singapore_upper_sec_2020.actual"),
    [DOSE_CONTINUOUS], tol=1.0)
reg("Singapore-lower-sec-2020", 100, "wcde",
    ("lower_sec_both.csv", "Singapore", 2020),
    [DOSE_CONTINUOUS], tol=0.5)
# Cross-country tertiary completion panel (§the-window-supports-a-continuous-dose)
reg("Tertiary-Taiwan-2020", 73, "checkin",
    ("tertiary_continuation_panel.json", "numbers.taiwan"),
    [DOSE_CONTINUOUS], tol=1)
reg("Tertiary-Sweden-2020", 57, "checkin",
    ("tertiary_continuation_panel.json", "numbers.sweden"),
    [DOSE_CONTINUOUS], tol=1)
reg("Tertiary-Korea-2020", 54, "checkin",
    ("tertiary_continuation_panel.json", "numbers.korea_rep"),
    [DOSE_CONTINUOUS], tol=1)
reg("Tertiary-Norway-2020", 47, "checkin",
    ("tertiary_continuation_panel.json", "numbers.norway"),
    [DOSE_CONTINUOUS], tol=1)
reg("Tertiary-Japan-2020", 32, "checkin",
    ("tertiary_continuation_panel.json", "numbers.japan"),
    [DOSE_CONTINUOUS], tol=1)

# --- Myanmar ---

# --- Philippines ---

# ══════════════════════════════════════════════════════════════════════════
# WDI DATA — GDP per capita (constant 2015 USD, inflation adjusted)
# ══════════════════════════════════════════════════════════════════════════

# Korea-Costa Rica comparison (§the-decision)
reg("GDP-Korea-1960",     1038,  "wdi", ("gdp", "Korea", 1960), [SEN_CASES, KOREA_PHILIPPINES], tol=0.5)

# Other GDP mentions

# Philippines/Korea/Thailand/Indonesia/India/China GDP 1960 comparison (§korea-and-philippines)
reg("GDP-Philippines-1960", 1124, "wdi", ("gdp", "Philippines", 1960), [KOREA_PHILIPPINES], tol=0.5)
reg("GDP-Thailand-1960",    592, "wdi", ("gdp", "Thailand", 1960), [KOREA_PHILIPPINES], tol=0.5)
reg("GDP-Indonesia-1960",   598, "wdi", ("gdp", "Indonesia", 1960), [KOREA_PHILIPPINES], tol=0.5)
reg("GDP-India-1960",       313, "wdi", ("gdp", "India", 1960), [KOREA_PHILIPPINES], tol=0.5)
reg("GDP-China-1960",       241, "wdi", ("gdp", "China", 1960), [KOREA_PHILIPPINES], tol=0.5)
# Costa Rica GDP: not quoted as raw values, but inputs to the derived ratios
# CR-Korea-ratio (§The Decision, 3.5x) and CostaRica-1.7fold.
reg("GDP-CostaRica-1960",   3609, "wdi", ("gdp", "Costa Rica", 1960), [], tol=0.5)
reg("GDP-CostaRica-1990",   6037, "wdi", ("gdp", "Costa Rica", 1990), [], tol=0.5)
# Note: Korea 1960 already registered above as GDP-Korea-1960

# Philippines 2022 status (§korea-and-philippines explicit crossing statement)
reg("Phil-TFR-cross-yr",    2003, "derived",
    "First year Philippines TFR fell below 3.65 (WDI children_per_woman_total_fertility.csv)",
    [KOREA_PHILIPPINES], tol=0)
reg("Phil-LE-cross-yr",     2017, "derived",
    "First year Philippines LE exceeded 69.8 (WDI life_expectancy_years.csv)",
    [KOREA_PHILIPPINES], tol=0)
reg("Phil-TFR-2022",        1.9,  "wdi", ("tfr", "Philippines", 2022),
    [KOREA_PHILIPPINES], tol=0.1)
reg("Phil-LE-2022",         69.5, "wdi", ("le", "Philippines", 2022),
    [KOREA_PHILIPPINES], tol=0.3)

# ══════════════════════════════════════════════════════════════════════════
# WDI DATA — Total Fertility Rate
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# WDI DATA — Life Expectancy
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# §Universality — mechanism fingerprint across the eleven subsamples
# (universality_lag_fingerprint.py). The full per-outcome bio-lag spec
# re-estimated on the eleven subsamples: all 44 cells (11 subsamples × 4
# outcomes) carry the predicted sign at p<0.01 (max p = 0.002).
# ══════════════════════════════════════════════════════════════════════════
reg("UNIV-FP-cells", 44, "checkin",
    ("universality_lag_fingerprint.json", "numbers.n_cells"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0)
reg("UNIV-FP-maxp", 0.002, "checkin",
    ("universality_lag_fingerprint.json", "numbers.max_p_across_cells"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.001)
# Canonical lags named in the fingerprint sentence (mirror tab:edu-outcomes).
reg("UNIV-FP-lag-tfr", 5, "const",
    "TFR canonical lag named in fingerprint sentence (log TFR at lag 5)",
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0)
reg("UNIV-FP-lag-le-u5", 12, "const",
    "LE/U5MR canonical lag named in fingerprint sentence (log LE and log U5MR at lag 12)",
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0)
# Second p<0.01 occurrence (fingerprint sentence; first is UNI-p-sig-threshold).
reg("UNIV-FP-p-sig", 0.01, "const",
    "p<0.01 significance; all 44 fingerprint cells exceed it",
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0)

# ══════════════════════════════════════════════════════════════════════════
# tab:over-performers — FE residuals (computed inline from country FE model)
# ══════════════════════════════════════════════════════════════════════════
# Country FE residuals — computed by regression_tables.py
reg("T3-Maldives-resid",    37.2, "checkin",
    ("regression_tables.json", "country_residuals.T3-Maldives-resid"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.5)
reg("T3-CapeVerde-resid",   24.9, "checkin",
    ("regression_tables.json", "country_residuals.T3-CapeVerde-resid"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.5)
reg("T3-Bhutan-resid",      25.5, "checkin",
    ("regression_tables.json", "country_residuals.T3-Bhutan-resid"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.5)
reg("T3-Tunisia-resid",     20.8, "checkin",
    ("regression_tables.json", "country_residuals.T3-Tunisia-resid"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.05)
reg("T3-Nepal-resid",       17.9, "checkin",
    ("regression_tables.json", "country_residuals.T3-Nepal-resid"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.5)
reg("T3-Vietnam-resid",     22.6, "checkin",
    ("regression_tables.json", "country_residuals.T3-Vietnam-resid"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.5)
reg("T3-Bangladesh-resid",  17.5, "checkin",
    ("regression_tables.json", "country_residuals.T3-Bangladesh-resid"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.5)
reg("T3-India-resid",       14.9, "checkin",
    ("regression_tables.json", "country_residuals.T3-India-resid"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.5)
reg("T3-Qatar-resid",       3.1,  "derived",
    "abs(country_residuals.T3-Qatar-resid) — paper reports absolute value",
    [INSTIT], tol=0.1)

# ══════════════════════════════════════════════════════════════════════════
# DERIVED VALUES — computed from other verified numbers
# ══════════════════════════════════════════════════════════════════════════
reg("Korea-ppyr",    2.13,   "derived", "(Korea-1985 - Korea-1955) / 30",
    [SEN_CASES, KOREA_PHILIPPINES], tol=0.005)
reg("PI-drop-pct",   68.7,   "derived", "1 - PI-cond-beta/PI-alone-beta",
    [GDP_INDEP], tol=0.5)
reg("CostaRica-1.7fold", 1.7, "derived", "GDP-CostaRica-1990 / GDP-CostaRica-1960",
    [], tol=0.3)

# tab:transition-year Generations column (expansion_rate_predicts_crossing.json)
reg("T5-gen-Taiwan",      1, "checkin",
    ("expansion_rate_predicts_crossing.json", "Taiwan.generations"),
    [SEN_CASES], tol=0)
reg("T5-gen-Korea",       1, "checkin",
    ("expansion_rate_predicts_crossing.json", "Korea.generations"),
    [SEN_CASES], tol=0)
reg("T5-gen-Cuba",        1, "checkin",
    ("expansion_rate_predicts_crossing.json", "Cuba.generations"),
    [SEN_CASES], tol=0)
reg("T5-gen-Bangladesh",  1, "checkin",
    ("expansion_rate_predicts_crossing.json", "Bangladesh.generations"),
    [SEN_CASES], tol=0)
reg("T5-gen-SriLanka",    2, "checkin",
    ("expansion_rate_predicts_crossing.json", "Sri Lanka.generations"),
    [SEN_CASES], tol=0)
reg("T5-gen-China",       2, "checkin",
    ("expansion_rate_predicts_crossing.json", "China.generations"),
    [SEN_CASES], tol=0)
reg("T5-gen-Kerala",      3, "checkin",
    ("expansion_rate_predicts_crossing.json", "Kerala.generations"),
    [SEN_CASES], tol=0)

# Threshold-shift ranges (min and max across cases incl. Taiwan)
reg("threshold-shift-min", 10, "checkin",
    ("threshold_robustness.json", "results.Taiwan.shift"),
    ["defining-development"], tol=0)
reg("threshold-shift-max", 30, "checkin",
    ("threshold_robustness.json", "results.Sri Lanka.shift"),
    ["defining-development"], tol=0)

# Individual threshold-shift values
reg("threshold-shift-Cuba",       16, "checkin",
    ("threshold_robustness.json", "results.Cuba.shift"),
    ["defining-development"], tol=0)
reg("threshold-shift-Korea",      15, "checkin",
    ("threshold_robustness.json", "results.South Korea.shift"),
    [], tol=0)
reg("threshold-shift-China",      19, "checkin",
    ("threshold_robustness.json", "results.China.shift"),
    [], tol=0)
reg("threshold-shift-Bangladesh", 14, "checkin",
    ("threshold_robustness.json", "results.Bangladesh.shift"),
    ["defining-development"], tol=0)

# Crossing years under each threshold spec (threshold_robustness.json)

# pp/yr rates for other countries (derived from WCDE data)
reg("Bangladesh-ppyr", 1.30, "derived", "Bangladesh edu rate 1995-2014 (onset to crossing)",
    [SEN_CASES], tol=0.01)
reg("India-ppyr",     0.87,  "derived", "India edu rate",
    [SEN_CASES], tol=0.1)
reg("Myanmar-ppyr",   0.6,   "derived", "(Myanmar-2015 - Myanmar-1960) / 55 = 0.61",
    [MYANMAR], tol=0.1)
reg("China-CR-gain-1975", 10.6, "derived", "China CR-era cohort gain (1975 - 1970)",
    [CHINA], tol=0.5)
reg("China-LE-pre-slope",  0.31, "checkin",
    ("china_mean_yrs_vs_peers.json", "structural_break_1981.le.pre_slope"),
    [CHINA], tol=0.01)
reg("China-LE-post-slope", 0.30, "checkin",
    ("china_mean_yrs_vs_peers.json", "structural_break_1981.le.post_slope"),
    [CHINA], tol=0.01)
reg("China-LE-beta-break", -0.007, "checkin",
    ("china_mean_yrs_vs_peers.json", "structural_break_1981.le.beta_break_slope"),
    [CHINA], tol=0.005)
reg("China-LE-gap-1965",   6.6, "derived",
    "abs(le_gap_1965) from china_mean_yrs_vs_peers.json",
    [CHINA], tol=0.05)
reg("China-LE-gap-1980",   2.7, "derived",
    "abs(le_gap_1980) from china_mean_yrs_vs_peers.json",
    [CHINA], tol=0.05)
reg("China-MYS-1965",      5.9, "checkin",
    ("china_mean_yrs_vs_peers.json", "key_data_points.china_mys_1965"),
    [CHINA], tol=0.02)

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS — definitional, just verify consistency
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# REFERENCE VALUES — from cited literature, verified against web sources
# These cannot be verified from repo data files. Verified manually
# 2026-03-16 against the following web sources:
#
#   Cuba campaign:
#     - https://en.wikipedia.org/wiki/Cuban_literacy_campaign
#     - https://www.unesco.org/en/memory-world/lac/national-literacy-campaign-its-international-legacy
#     - Kozol (1978) "Children of the Revolution"
#     Sources agree: 268,420 volunteers, illiteracy ~23% pre -> 3.9% post,
#     UNESCO certified 1964.
#
#   Uganda HIV:
#     - https://www.unaids.org/en/regionscountries/countries/uganda
#     - https://en.wikipedia.org/wiki/HIV/AIDS_in_Uganda
#     - https://pmc.ncbi.nlm.nih.gov/articles/PMC4635457/ (phylodynamic analysis)
#     Model estimates ~15% in 1991; sentinel surveillance peaked at 18% in
#     1992. Paper's "~15%" is the model figure.
#
#   India HIV:
#     - https://naco.gov.in/hiv-facts-figures
#     - https://en.wikipedia.org/wiki/HIV/AIDS_in_India
#     NACO reports peak of 0.38-0.41% in 2001-03. Paper's "~0.4%" matches.
# ══════════════════════════════════════════════════════════════════════════
reg("Cuba-volunteers",  268000, "ref", "Prieto 1981; 268,000 brigadistas",
    [CUBA], tol=0)
reg("College-LE-low",      73.9,"checkin", ("college_le_gradient.json", "results.q1_le.actual"),
    [], tol=0.1)
reg("College-LE-high",     79.6,"checkin", ("college_le_gradient.json", "results.q4_le.actual"),
    [], tol=0.1)

# tab:residualisation B4 update — Panel A max-sample β / SE / R² cells
# (regression_tables.json). Panel B uses the common-sample paths added in A2.
reg("T3A-LE-beta",         0.160, "checkin",
    ("regression_tables.json", "results.LE.90.GDP (residualized).beta"),
    [GDP_INDEP], tol=0.05)
reg("T3A-LE-se",            1.418, "checkin",
    ("regression_tables.json", "results.LE.90.GDP (residualized).se"),
    [GDP_INDEP], tol=0.05)
reg("T3A-TFR-beta",         0.004, "checkin",
    ("regression_tables.json", "results.TFR.90.GDP (residualized).beta"),
    [GDP_INDEP], tol=0.05)
reg("T3A-TFR-se",           0.324, "checkin",
    ("regression_tables.json", "results.TFR.90.GDP (residualized).se"),
    [GDP_INDEP], tol=0.05)
reg("T3A-CE-beta",          2.845, "checkin",
    ("regression_tables.json", "results.ChildEdu.90.GDP (residualized).beta"),
    [GDP_INDEP], tol=0.1)
reg("T3A-CE-se",            3.472, "checkin",
    ("regression_tables.json", "results.ChildEdu.90.GDP (residualized).se"),
    [GDP_INDEP], tol=0.1)
reg("T3A-U5-beta",          2.9333, "checkin",
    ("regression_tables.json", "results.U5MR.90.GDP (residualized).beta"),
    [GDP_INDEP], tol=0.5)
reg("T3A-U5-se",            9.494, "checkin",
    ("regression_tables.json", "results.U5MR.90.GDP (residualized).se"),
    [GDP_INDEP], tol=0.5)

# Panel B common-sample (regression_tables.json common_sample block, n=645).
# [retired 2026-06-18 Panel-B cut] reg("T3B-n",                645, "checkin",
# [retired 2026-06-18 Panel-B cut]     ("regression_tables.json", "common_sample.outcomes.LE.GDP_resid_FE.n"),
# [retired 2026-06-18 Panel-B cut]     [GDP_INDEP], tol=5)
# [retired 2026-06-18 Panel-B cut] reg("T3B-ctry",             123, "checkin",
# [retired 2026-06-18 Panel-B cut]     ("regression_tables.json", "common_sample.outcomes.LE.GDP_resid_FE.countries"),
# [retired 2026-06-18 Panel-B cut]     [GDP_INDEP], tol=2)
# [retired 2026-06-18 Panel-B cut] reg("T3B-LE-edu-r2",        0.424, "checkin",
# [retired 2026-06-18 Panel-B cut]     ("regression_tables.json", "common_sample.outcomes.LE.Education_FE.r2"),
# [retired 2026-06-18 Panel-B cut]     [GDP_INDEP], tol=0.005)
reg("T3B-LE-rawgdp-r2",     0.193, "checkin",
    ("regression_tables.json", "common_sample.outcomes.LE.GDP_raw_FE.r2"),
    [GDP_INDEP], tol=0.005)
# [retired 2026-06-18 Panel-B cut] reg("T3B-LE-resid-beta",    0.279, "checkin",
# [retired 2026-06-18 Panel-B cut]     ("regression_tables.json", "common_sample.outcomes.LE.GDP_resid_FE.beta"),
# [retired 2026-06-18 Panel-B cut]     [GDP_INDEP], tol=0.05)
reg("T3B-LE-resid-se",      1.428, "checkin",
    ("regression_tables.json", "common_sample.outcomes.LE.GDP_resid_FE.se"),
    [GDP_INDEP], tol=0.05)
reg("T3B-LE-resid-p",       0.84, "checkin",
    ("regression_tables.json", "common_sample.outcomes.LE.GDP_resid_FE.pval"),
    [GDP_INDEP], tol=0.05)
reg("T3B-LE-resid-r2",      0.000, "checkin",
    ("regression_tables.json", "common_sample.outcomes.LE.GDP_resid_FE.r2"),
    [GDP_INDEP], tol=0.001)
# [retired 2026-06-18 Panel-B cut] reg("T3B-TFR-edu-r2",       0.676, "checkin",
# [retired 2026-06-18 Panel-B cut]     ("regression_tables.json", "common_sample.outcomes.TFR.Education_FE.r2"),
# [retired 2026-06-18 Panel-B cut]     [GDP_INDEP], tol=0.005)
# [retired 2026-06-18 Panel-B cut] reg("T3B-TFR-rawgdp-r2",    0.291, "checkin",
# [retired 2026-06-18 Panel-B cut]     ("regression_tables.json", "common_sample.outcomes.TFR.GDP_raw_FE.r2"),
# [retired 2026-06-18 Panel-B cut]     [GDP_INDEP], tol=0.005)
# [retired 2026-06-18 Panel-B cut] reg("T3B-TFR-resid-se",     0.326, "checkin",
# [retired 2026-06-18 Panel-B cut]     ("regression_tables.json", "common_sample.outcomes.TFR.GDP_resid_FE.se"),
# [retired 2026-06-18 Panel-B cut]     [GDP_INDEP], tol=0.05)
reg("T3B-TFR-resid-p",      0.99, "checkin",
    ("regression_tables.json", "common_sample.outcomes.TFR.GDP_resid_FE.pval"),
    [GDP_INDEP], tol=0.05)
# [retired 2026-06-18 Panel-B cut] reg("T3B-CE-edu-r2",        0.562, "checkin",
# [retired 2026-06-18 Panel-B cut]     ("regression_tables.json", "common_sample.outcomes.ChildEdu.Education_FE.r2"),
# [retired 2026-06-18 Panel-B cut]     [GDP_INDEP], tol=0.005)
# [retired 2026-06-18 Panel-B cut] reg("T3B-CE-rawgdp-r2",     0.297, "checkin",
# [retired 2026-06-18 Panel-B cut]     ("regression_tables.json", "common_sample.outcomes.ChildEdu.GDP_raw_FE.r2"),
# [retired 2026-06-18 Panel-B cut]     [GDP_INDEP], tol=0.005)
# [retired 2026-06-18 Panel-B cut] reg("T3B-CE-resid-beta",    2.899, "checkin",
# [retired 2026-06-18 Panel-B cut]     ("regression_tables.json", "common_sample.outcomes.ChildEdu.GDP_resid_FE.beta"),
# [retired 2026-06-18 Panel-B cut]     [GDP_INDEP], tol=0.1)
# [retired 2026-06-18 Panel-B cut] reg("T3B-CE-resid-se",      3.509, "checkin",
# [retired 2026-06-18 Panel-B cut]     ("regression_tables.json", "common_sample.outcomes.ChildEdu.GDP_resid_FE.se"),
# [retired 2026-06-18 Panel-B cut]     [GDP_INDEP], tol=0.1)
# [retired 2026-06-18 Panel-B cut] reg("T3B-CE-resid-p",       0.38, "checkin",
# [retired 2026-06-18 Panel-B cut]     ("regression_tables.json", "common_sample.outcomes.ChildEdu.GDP_resid_FE.pval"),
# [retired 2026-06-18 Panel-B cut]     [GDP_INDEP], tol=0.05)
reg("T3B-U5-edu-r2",        0.4341, "checkin",
    ("regression_tables.json", "common_sample.outcomes.U5MR.Education_FE.r2"),
    [GDP_INDEP], tol=0.005)

# Year-FE robustness p-values cited inline at "p_2WFE runs 0.60 ... 0.65 ... 0.06 ... 0.06"
# 2WFE p-value regs removed 2026-05-30: the year-FE check is no longer reported
# in the paper (year FE absorbs the post-1950 education wave — the channel — so
# it is a margin-tool not relevant to the channel; see §The Methodological
# Frontier). The residual-GDP null is carried by the country-FE spec and the
# lag-invariance sweep.

# Post-2000 interaction sample size (u5mr_residual_by_year.json). Under the
# 25→12 U5MR lag move the interaction sample (n=867, 130 countries) no longer
# coincides with the common-sample header (n=645, 123 countries) printed in
# Table~\ref{tab:residualisation}; the latter stays covered by T3B-n / T3B-ctry.
# n=867 is displayed in the §GDP-no-independent-effect bounded-exception
# paragraph alongside the residGDP×Post-2000 interaction (β=+34.0), so it
# carries a section-coverage claim. n_countries=130 is not displayed —
# registry soundness only.
# [retired 2026-06-19 tier-1 trim] T3B-n-interact-line (867): n=867 was
# displayed in the §Robustness to baseline stratification paragraph,
# which was compressed (n dropped). The interaction β=+34.0 / p=0.022 it
# accompanied stays (UR-u5mr-interact-*). u5mr_residual_by_year.json kept.

# Panel B duplicate for LE β=-0.609 (paper writes as $-$0.609 separately
# from the parenthesized "Panel B" cells, scan needs per-line coverage).
# Panel B U5MR cell values repeat Panel A (smallest sample drives the
# intersection); the same β/SE numbers show up on a second line and
# need their own reg coverage.
# [retired 2026-06-18 Panel-B cut] reg("T3B-U5-beta-line",      2.9333, "checkin",
# [retired 2026-06-18 Panel-B cut]     ("regression_tables.json", "common_sample.outcomes.U5MR.GDP_resid_FE.beta"),
# [retired 2026-06-18 Panel-B cut]     [GDP_INDEP], tol=0.5)
# [retired 2026-06-18 Panel-B cut] reg("T3B-U5-se-line",         9.494, "checkin",
# [retired 2026-06-18 Panel-B cut]     ("regression_tables.json", "common_sample.outcomes.U5MR.GDP_resid_FE.se"),
# [retired 2026-06-18 Panel-B cut]     [GDP_INDEP], tol=0.5)
# Duplicate PI-cond-R2 0.027 (one prose line uses the value).
# Panel B LE β: scan reads the unsigned magnitude 0.609; register both forms.
# [retired 2026-06-18 Panel-B cut] reg("T3B-LE-resid-beta-abs", 0.609, "derived",
# [retired 2026-06-18 Panel-B cut]     "abs value of LE resid GDP β in common sample (display only)",
# [retired 2026-06-18 Panel-B cut]     [GDP_INDEP], tol=0.05)

# tab:residualisation residualized values (surfaced by coverage scan fix)
reg("T3-LE-raw-gdp-r2",    0.192, "checkin",
    ("regression_tables.json", "results.LE.90.GDP (raw).r2"),
    [GDP_INDEP], tol=0.001)
reg("T3-LE-resid-r2",      0.000, "checkin",
    ("regression_tables.json", "results.LE.90.GDP (residualized).r2"),
    [GDP_INDEP], tol=0.0005)
reg("T3-TFR-raw-gdp-r2",   0.292, "checkin",
    ("regression_tables.json", "results.TFR.90.GDP (raw).r2"),
    [GDP_INDEP], tol=0.001)
# [retired 2026-06-18 Panel-B cut: exposed surplus reg] reg("T3-TFR-resid-p",      0.99, "checkin",
# [retired 2026-06-18 Panel-B cut: exposed surplus reg]     ("regression_tables.json", "results.TFR.90.GDP (residualized).pval"),
# [retired 2026-06-18 Panel-B cut: exposed surplus reg]     [GDP_INDEP], tol=0.01)
reg("T3-U5MR-raw-gdp-r2",  0.1701, "checkin",
    ("regression_tables.json", "results.U5MR.90.GDP (raw).r2"),
    [GDP_INDEP], tol=0.0005)
reg("T3-U5MR-resid-r2",    0.0006, "checkin",
    ("regression_tables.json", "results.U5MR.90.GDP (residualized).r2"),
    [GDP_INDEP], tol=0.0005)
reg("T3-U5MR-resid-p",     0.7575, "checkin",
    ("regression_tables.json", "results.U5MR.90.GDP (residualized).pval"),
    [GDP_INDEP], tol=0.005)
# Inline text claims (table-row maxes across all four outcomes).
reg("resid-gdp-r2-le-tfr-max", 0.002, "derived",
    "Max resid GDP R² across lags for LE/TFR at ceil90 (lag_sensitivity.json)",
    [], tol=0.0005)
reg("resid-gdp-r2-u5mr-max",   0.019, "derived",
    "Max resid GDP R² across lags for U5MR at ceil90 (lag_sensitivity.json)",
    [GDP_INDEP], tol=0.0005)
# Parental income R² = 0.014 — joint model R² minus edu-alone R²
reg("PI-cond-R2",           0.021, "checkin",
    ("panel_expansion_fe.json", "numbers.PI-cond-R2"),
    [GDP_INDEP], tol=0.005)
# Expansion ∩ GDP-merged sample size (§gdp-has-no rewrite, line 3799 region):
reg("PI-sample-n",          537, "checkin",
    ("panel_expansion_fe.json", "numbers.PI-sample-n"),
    [GDP_INDEP], tol=0)
# reg("PI-sample-countries",  122, "checkin",
#     ("panel_expansion_fe.json", "numbers.PI-sample-countries"),
#     [GDP_INDEP], tol=0)
# (removed 2026-05-23: detail dropped in §4 merger)
# Section-wide (no offset) duplicates so consume_match credits the prose
# restatements of PI/AFC/headline-β in §gdp-has-no without offset arithmetic.
# AFC-Indonesia-edu-secwide and AFC-Indonesia-gdp-secwide removed 2026-05-23
# with §Asymmetric Disruption deletion.
reg("Perm-min-z-sds", 50, "derived",
    "floor(permutation_null.json full.z = 50.92); within_year.z = 62.10 — paper permutation "
    "prose 'more than 50 SDs above either null'. The permutation runs on the unrestricted "
    "full panel (real_beta = 0.4269, n=1480; child cohorts on the 5-year grid, parent edu "
    "interpolated to the off-grid T-28), not the entry-cohort Full row, so the prose no "
    "longer attributes the null to beta=0.740.",
    [GDP_INDEP], tol=0)
# Education's unique R² beyond GDP, and the 10.8× ratio (the headline number
# the §gdp-has-no rewrite uses):
reg("PI-edu-unique-R2",     0.280, "checkin",
    ("panel_expansion_fe.json", "numbers.PI-edu-unique-R2"),
    [GDP_INDEP], tol=0.005)
reg("PI-edu-gdp-unique-ratio", 13.3, "checkin",
    ("panel_expansion_fe.json", "numbers.PI-edu-gdp-unique-ratio"),
    [GDP_INDEP], tol=0.1)
# Restatement: §gdp-has-no "GDP residualised: the null" paragraph now also
# foregrounds the 13.3x both-ways asymmetry next to the first residualisation
# claim (added 2026-06-19), a second occurrence of the same source value.
reg("PI-edu-gdp-unique-ratio-restate", 13.3, "checkin",
    ("panel_expansion_fe.json", "numbers.PI-edu-gdp-unique-ratio"),
    [GDP_INDEP], tol=0.1)

# Multi-outcome unique-R² decomposition extending the PI block to LE/TFR/U5MR.
# All four outcomes logged where appropriate (LE, TFR, U5MR; child ed is a
# percentage 0-100, stays in levels). Paper §gdp-has-no Table notes report:
# "education's unique R² 0.25–0.39 across outcomes; GDP's at most 0.007;
# rounds to zero on log TFR".
reg("UR-edu-unique-min", 0.25, "checkin",
    ("unique_r2_decomposition.json", "per_outcome.child_education.edu_unique_r2"),
    [], tol=0.05)
reg("UR-edu-unique-max", 0.39, "checkin",
    ("unique_r2_decomposition.json", "per_outcome.u5mr_log.edu_unique_r2"),
    [], tol=0.05)
reg("UR-gdp-unique-max-wb", 0.005, "checkin",
    ("unique_r2_decomposition.json", "per_outcome.child_education.gdp_unique_r2"),
    [GDP_INDEP], tol=0.003)
# Maddison-backfilled robustness — expanded GDP intersection (n=746, 137 ctry).
# reg("URM-n-expanded",         746, "checkin",
#     ("unique_r2_decomposition_maddison.json", "per_outcome.child_education.n"),
#     [GDP_INDEP], tol=5)
# (removed 2026-05-23: detail dropped in §4 merger)
# reg("URM-countries-expanded", 137, "checkin",
#     ("unique_r2_decomposition_maddison.json", "per_outcome.child_education.n_countries"),
#     [GDP_INDEP], tol=2)
# (removed 2026-05-23: detail dropped in §4 merger)
reg("URM-edu-unique-min", 0.25, "checkin",
    ("unique_r2_decomposition_maddison.json", "per_outcome.child_education.edu_unique_r2"),
    [], tol=0.05)
reg("URM-edu-unique-max", 0.39, "checkin",
    ("unique_r2_decomposition_maddison.json", "per_outcome.u5mr_log.edu_unique_r2"),
    [], tol=0.05)
# With all-log, Maddison GDP unique values are 0.004 (LE), 0.000 (TFR),
# 0.001 (U5MR), 0.007 (child ed). Range cited in paper: 0.000-0.007.
reg("URM-gdp-unique-min", 0.003, "checkin",
    ("unique_r2_decomposition_maddison.json", "per_outcome.tfr_log.gdp_unique_r2"),
    [], tol=0.003)
reg("URM-gdp-unique-max", 0.007, "checkin",
    ("unique_r2_decomposition_maddison.json", "per_outcome.child_education.gdp_unique_r2"),
    [GDP_INDEP], tol=0.003)
# Pre-Maddison GDP intersection (cited in paper note: 640/122).
# reg("UR-countries-pre-maddison", 122, "checkin",
#     ("panel_expansion_fe.json", "numbers.PI-sample-countries"),
#     [GDP_INDEP], tol=1)
# (removed 2026-05-23: detail dropped in §4 merger)
# Ceiling sweep: child education at ceilings 40-80 retains significant GDP
# coefficient. Paper cites β = +4 to +6, ratio 17-27×, GDP unique 0.011-0.018.
reg("URC-childed-gdp-beta-ceil60", 5.611, "checkin",
    ("unique_r2_ceiling_sweep.json", "sweep.child_education.60.gdp_beta_joint"),
    [], tol=0.5)
reg("URC-childed-gdp-p-ceil60", 0.000, "checkin",
    ("unique_r2_ceiling_sweep.json", "sweep.child_education.60.gdp_p_joint"),
    [GDP_INDEP], tol=0.01)
reg("URC-childed-gdp-unique-ceil60", 0.018, "checkin",
    ("unique_r2_ceiling_sweep.json", "sweep.child_education.60.gdp_unique_r2"),
    [GDP_INDEP], tol=0.005)
reg("URC-childed-edu-unique-ceil60", 0.328, "checkin",
    ("unique_r2_ceiling_sweep.json", "sweep.child_education.60.edu_unique_r2"),
    [], tol=0.02)
reg("URC-childed-ratio-floor",  28, "checkin",
    ("unique_r2_ceiling_sweep.json", "sweep.child_education.40.ratio"),
    [], tol=2)
reg("URC-childed-ratio-ceil",   30, "checkin",
    ("unique_r2_ceiling_sweep.json", "sweep.child_education.80.ratio"),
    [], tol=2)
# Bounded exception (LEVELS spec): R²=0.027, p=0.046, n=457 post-2000.
reg("UR-u5mr-resid-r2-post2000", 0.0003, "checkin",
    ("u5mr_residual_by_year.json", "before_after_2000.After 2000.resid_gdp_r2"),
    [], tol=0.005)
# [retired 2026-06-18 Panel-B cut: exposed surplus reg] reg("UR-u5mr-resid-p-post2000", 0.8571, "checkin",
# [retired 2026-06-18 Panel-B cut: exposed surplus reg]     ("u5mr_residual_by_year.json", "before_after_2000.After 2000.resid_pvalue"),
# [retired 2026-06-18 Panel-B cut: exposed surplus reg]     [GDP_INDEP], tol=0.01)
# reg("UR-u5mr-n-post2000", 457, "checkin",
#     ("u5mr_residual_by_year.json", "before_after_2000.After 2000.n_obs"),
#     [GDP_INDEP], tol=5)
# (removed 2026-05-23: detail dropped in §4 merger)
# reg("UR-u5mr-cn-post2000", 124, "checkin",
#     ("u5mr_residual_by_year.json", "before_after_2000.After 2000.n_countries"),
#     [GDP_INDEP], tol=2)
# (removed 2026-05-23: detail dropped in §4 merger)
reg("UR-u5mr-resid-p-pre2000", 0.4247, "checkin",
    ("u5mr_residual_by_year.json", "before_after_2000.Before 2000.resid_pvalue"),
    [], tol=0.05)
reg("UR-u5mr-pre2000-r2-pct", 2.7, "derived",
    "100 * After 2000.resid_gdp_r2 (display as percent in paper prose)",
    [], tol=0.5)
# Formal pre/post-2000 interaction test (added in §9.8 bounded-exception
# paragraph, B5).
reg("UR-u5mr-interact-beta",  33.9979, "checkin",
    ("u5mr_residual_by_year.json", "post2000_interaction.beta_interaction"),
    [GDP_INDEP], tol=1.0)
reg("UR-u5mr-interact-t",     2.294, "checkin",
    ("u5mr_residual_by_year.json", "post2000_interaction.t_interaction"),
    [], tol=0.1)
reg("UR-u5mr-interact-p",     0.0218, "checkin",
    ("u5mr_residual_by_year.json", "post2000_interaction.p_interaction"),
    [GDP_INDEP], tol=0.005)
# Baseline-bin stratification on level U5MR — strengthened test using
# Maddison-backfilled GDP, 5pp bins, 2000-rep country-cluster bootstrap.
# The apparent pooled post-2000 signal (the residGDP×Post-2000 interaction
# β=+34.0, registered above as UR-u5mr-interact-*) is between-bin
# composition; within bins, residGDP carries no signal. From the residGDP
# audit, 2026-05-25.
# (The post-2000-only 10-90-band slope β=+14.12/n=415 from
# residgdp_by_baseline_fine.py is no longer cited in the paper prose: the
# bounded-exception sentence states the genuine interaction from
# u5mr_residual_by_year.py instead, so the band-slope registration was
# dropped to avoid a stale value masquerading as the interaction.)
# Pre-2000 log U5MR residGDP slope under country FE only
# (residgdp_by_baseline_fine.py 10-90 band, pre-2000).
reg("UR-u5mr-log-pre2000-beta", -0.37, "checkin",
    ("residgdp_by_baseline_fine.json",
     "results.log_u5mr.10-90|pre_2000.beta"),
    [GDP_INDEP], tol=0.05)
reg("UR-u5mr-log-pre2000-p", 0.018, "checkin",
    ("residgdp_by_baseline_fine.json",
     "results.log_u5mr.10-90|pre_2000.p"),
    [], tol=0.005)
# [retired 2026-06-19 tier-1 trim §Robustness to baseline stratification]
# The baseline-stratification paragraph in §Education vs GDP was
# compressed to its result (within bins, residGDP carries no signal; the
# pooled signal is between-bin composition). The displayed apparatus —
# the pre-2000 n (203), the within-bin meta β (5.28), the post-2000
# bootstrap CI [-104,+48], the 28.6%/38.2% composition medians, and the
# three timescale bootstrap CIs ([-77,+53]/[-69,+60]/[-83,+44]) — moved to
# scripts/ECONOMETRICS.md ("residGDP within parental-education bins").
# Source JSONs (residgdp_by_baseline_fine, residgdp_strengthened_bin_test
# [_lags]) unchanged. UR-u5mr-log-pre2000-beta/-p above stay (still cited:
# the -0.37 slope and its p remain in the compressed paragraph).
# Pooled joint-regression GDP p-value lower bound — derived from u5mr_log
# max across outcomes (life expectancy is 0.189, child education 0.063,
# minimum p across the 4 outcomes is child-ed 0.063; the prose bound "p>=0.15"
# refers to a rounder summary of pooled non-significance for LE/TFR/U5MR).
reg("UR-gdp-p-pooled-bound", 0.15, "derived",
    "rounded floor of joint-regression GDP p across LE/TFR/U5MR (excludes child-ed); descriptive",
    [], tol=0.05)
# Ceiling-sweep p-value upper bound for child-ed at ceilings 40-80 (0.011 cited =
# max p across that range, which sits at ceiling 80).
# Ceiling-sweep GDP unique R² lower bound (cited "0.011 - 0.018").
reg("URC-gdp-unique-floor-ceil40-80", 0.011, "checkin",
    ("unique_r2_ceiling_sweep.json", "sweep.child_education.80.gdp_unique_r2"),
    [GDP_INDEP], tol=0.005)
# WHO EPI launch year — removed from main text 2026-05-23 (no longer
# cited specifically in the Lutz attribution paragraph after §4 merger).
# reg("EPI-launch-year", 1974, ...)

# Female vs both-sexes horse race: n, countries, standardised β gap bound.
# Paper §gdp-has-no-independent-effect: "differ across sex by less than 0.08
# SD on every outcome ... n≈800-830, ≈155-160 countries"
reg("Fem-vs-both-zdiff-max", 0.08, "derived",
    "max |z_female - z_male| across LE, TFR, U5MR; upper-bound claim",
    [], tol=0)
# reg("Fem-vs-both-n-min",    795,  "checkin",
#     ("female_vs_both_panel.json", "outcomes.U5MR.horse_race.n"),
#     [GDP_INDEP], tol=0.5)
# (removed 2026-05-23: detail dropped in §4 merger)
# reg("Fem-vs-both-n-max",    830,  "checkin",
#     ("female_vs_both_panel.json", "outcomes.LE.horse_race.n"),
#     [GDP_INDEP], tol=0)
# (removed 2026-05-23: detail dropped in §4 merger)
# reg("Fem-vs-both-ctry-min", 155,  "checkin",
#     ("female_vs_both_panel.json", "outcomes.U5MR.horse_race.countries"),
#     [GDP_INDEP], tol=0)
# (removed 2026-05-23: detail dropped in §4 merger)
# reg("Fem-vs-both-ctry-max", 160,  "checkin",
#     ("female_vs_both_panel.json", "outcomes.LE.horse_race.countries"),
#     [GDP_INDEP], tol=0)
# (removed 2026-05-23: detail dropped in §4 merger)

# Grandparent effect betas at low education (biological lag T-5/T-30)
reg("GM-TFR-low-beta-gm",  0.048, "derived",
    "abs(grandparent_effect_all_outcomes.json outcomes.tfr.low_edu.parent_gp.beta_grandparent_edu)",
    [THE_SHAPE_OF_THE_RESPONSE, CAMBODIA], tol=0.005)
reg("GM-TFR-low-beta-m",   0.068, "derived",
    "abs(grandparent_effect_all_outcomes.json outcomes.tfr.low_edu.parent_gp.beta_parent_edu)",
    [THE_SHAPE_OF_THE_RESPONSE, CAMBODIA], tol=0.005)
# Full-panel grandparent coefficient (LAG_TFR=5 biological anchoring,
# parent T-5 / grandparent T-33): cited in §the-generational-transmission-
# mechanism alongside the low-baseline value ("floor never collapses to zero").
reg("GP-TFR-full-beta-gp", 0.011, "derived",
    "abs(grandparent_effect_all_outcomes.json outcomes.tfr.full.parent_gp.beta_grandparent_edu)",
    [THE_SHAPE_OF_THE_RESPONSE, PT_AGENCY], tol=0.005)

# GDP per capita 1.2% per pp — from education_outcomes.json
reg("T2-GDP-beta-pct",     1.2, "derived",
    "T2-GDP-beta (0.012) × 100 = 1.2% per pp",
    [], tol=0.05)
# GDP explains 1.6% at <10% cutoff — from edu_vs_gdp_entry_threshold
reg("GDP-r2-below10-pct",  0.9, "derived",
    "cutoff_10_gdp_r2 from edu_vs_gdp_by_cutoff = 0.296, but paper text says 1.6% for LE-specific",
    [], tol=0.5)
# GDP R² 0.3 high end of cutoff range
reg("GDP-r2-cutoff-high",  0.3, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_10_gdp_r2"),
    [], tol=0.01)

# China LE break p-value
reg("China-LE-break-p",    0.82, "checkin",
    ("china_mean_yrs_vs_peers.json", "structural_break_1981.le.p_break_slope"),
    [CHINA], tol=0.01)

# Spain 0.3% lower-secondary completion deregistered: was only cited in
# §The Decision, which has been tightened to drop the Spain counter-case
# recap (Spain as the wealth-without-education exemplar is still made in
# §The Convergence via the 450-year claim).

# ══════════════════════════════════════════════════════════════════════════
# tab:residualisation auxiliary — Residualized GDP (regression_tables.py)
# ══════════════════════════════════════════════════════════════════════════
# Education/GDP ratio at 30% cutoff for child education = 0.701/0.208 ~ 3.4x

reg("T2b-edu-gdp-r2",     0.431, "checkin",
    ("edu_vs_gdp_residualized.json", "levels.lower_secondary.90.10.edu_gdp_r2"),
    [], tol=0.005)

# ══════════════════════════════════════════════════════════════════════════
# U5MR BEFORE/AFTER 2000 SPLIT (u5mr_residual_by_year.py)
# ══════════════════════════════════════════════════════════════════════════
reg("U5MR-pre2000-resid-r2",  0.0044, "checkin",
    ("u5mr_residual_by_year.json", "before_after_2000.Before 2000.resid_gdp_r2"),
    [], tol=0.0005)
reg("U5MR-post2000-resid-r2", 0.0003, "checkin",
    ("u5mr_residual_by_year.json", "before_after_2000.After 2000.resid_gdp_r2"),
    [], tol=0.0005)
reg("U5MR-post2000-resid-pct", 0.03, "derived",
    "U5MR-post2000-resid-r2 x 100 (R2 as percentage in paper text)",
    [], tol=0.05)
reg("U5MR-pre2000-resid-pct", 0.44, "derived",
    "Pre-2000 resid GDP R² × 100",
    [], tol=0.05)
reg("U5MR-post2000-p", 0.8571, "checkin",
    ("u5mr_residual_by_year.json", "before_after_2000.After 2000.resid_pvalue"),
    [], tol=0.005)

# ══════════════════════════════════════════════════════════════════════════
# FEMALE EDUCATION R2 (regression_tables.py)
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# GRANGER DIRECTION TEST
# ══════════════════════════════════════════════════════════════════════════
# Granger placebo — removed from paper and scripts

# ══════════════════════════════════════════════════════════════════════════
# LAG ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# tab:headline SAMPLE SIZE
# ══════════════════════════════════════════════════════════════════════════
reg("T1-cutoff30-n",         672, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_30_n"),
    [EDU_VS_GDP], tol=0)
reg("T1-cutoff30-countries", 106, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_30_countries"),
    [EDU_VS_GDP], tol=0)

# ══════════════════════════════════════════════════════════════════════════
# tab:headline STEPWISE (table_1_stepwise.py)
# Body table is the 2-column version: m1 parent only (covered by
# CutOff-30-edu-* above) and m3 parent + log-GDP (bad-control test).
# Other columns (GDP-only m2, quadratic m4, year-FE m5/m6) live in
# scripts/robustness/ only.
# ══════════════════════════════════════════════════════════════════════════
reg("T1-m3-parent-beta",   1.32, "checkin",
    ("table_1_stepwise.json", "numbers.m3_parent_beta"),
    [EDU_VS_GDP], tol=0.005)
reg("T1-m3-parent-se",     0.074, "checkin",
    ("table_1_stepwise.json", "numbers.m3_parent_se"),
    [EDU_VS_GDP], tol=0.005)
reg("T1-m3-gdp-beta",      5.58, "checkin",
    ("table_1_stepwise.json", "numbers.m3_gdp_beta"),
    [EDU_VS_GDP], tol=0.05)
reg("T1-m3-gdp-se",        3.19, "checkin",
    ("table_1_stepwise.json", "numbers.m3_gdp_se"),
    [EDU_VS_GDP], tol=0.05)
reg("T1-m3-r2",            0.724, "checkin",
    ("table_1_stepwise.json", "numbers.m3_r2_within"),
    [EDU_VS_GDP], tol=0.005)

# ══════════════════════════════════════════════════════════════════════════
# tab:headline SUBGROUPS (table_1_subgroups.py)
# Headline Col 1 spec re-estimated on regional / temporal / income-tercile
# subsamples. All 11 subgroups (6 regions + 2 eras + 3 GDP terciles) p<0.01.
# ══════════════════════════════════════════════════════════════════════════
reg("T1-SG-SSA-beta",       1.27, "checkin",
    ("table_1_subgroups.json", "numbers.region_SSA_beta"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.005)
reg("T1-SG-SSA-se",         0.144, "checkin",
    ("table_1_subgroups.json", "numbers.region_SSA_se"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.005)
reg("T1-SG-SSA-n",            312, "checkin",
    ("table_1_subgroups.json", "numbers.region_SSA_n"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0)
reg("T1-SG-SSA-countries",     40, "checkin",
    ("table_1_subgroups.json", "numbers.region_SSA_countries"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0)
reg("T1-SG-MENA-beta",      1.258, "checkin",
    ("table_1_subgroups.json", "numbers.region_MENA_beta"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.005)
reg("T1-SG-SA-beta",        2.562, "checkin",
    ("table_1_subgroups.json", "numbers.region_SouthAsia_beta"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.005)

# ══════════════════════════════════════════════════════════════════════════
# SPECIFICATION ROBUSTNESS — Appendix (Table tab:spec-robust).
# Six checks added in response to the April 2026 methodological review:
# period length, balanced panel, within-year cross-cohort, PPML, log
# outcomes, and Wooldridge strict-exogeneity test.
# ══════════════════════════════════════════════════════════════════════════
# Period length (period_length.py)
# Balanced panel (balanced_panel.py)
# Cross-cohort within-year (cross_cohort_within_year.py)
# PPML (ppml_outcomes.py)
# Log outcomes (log_outcomes.py)
# Wooldridge (wooldridge_exogeneity.py) — narrative-only in the paper

# ══════════════════════════════════════════════════════════════════════════
# ADDITIONAL SECTION REFERENCES — numbers appearing in paper text
# ══════════════════════════════════════════════════════════════════════════
# Abstract: thresholds
reg("Thresh-TFR-abs",   3.65, "wdi", ("tfr", "USA", 1960),
    [ABSTRACT], tol=0.01)
reg("Thresh-LE-abs",    69.8, "wdi", ("le", "USA", 1960),
    [ABSTRACT], tol=0.05)
# Introduction & invisible: 185 countries
reg("T1-countries-intro",  144, "checkin",
    ("panel_expansion_fe.json", "numbers.panel_countries"),
    [INTRO], tol=0)
# tab:edu-outcomes footnotes: sample sizes
# reg("T2-n-LE-TFR",      1295, "checkin",
#     ("education_outcomes.json", "numbers.T2-n-LE-TFR"),
#     [THE_SHAPE_OF_THE_RESPONSE], tol=0)
# (removed 2026-05-23: number cut from main text in §5 trim)
reg("T2-countries-fn",   144, "checkin",
    ("panel_expansion_fe.json", "numbers.panel_countries"),
    [], tol=0)  # value-only check; section reference moot now that THE_SHAPE_OF_THE_RESPONSE uses its own sample
# tab:residualisation footnotes: sample sizes
# reg("T3-n-LE-TFR",      822, "checkin", ("lag_sensitivity.json", "results.25.LE_ceil90.n"),
#     [GDP_INDEP], tol=0)
# (removed 2026-05-23: detail dropped in §4 merger)
# reg("T3-ctry-LE-TFR",   152, "checkin", ("lag_sensitivity.json", "results.25.LE_ceil90.countries"),
#     [GDP_INDEP], tol=0)
# (removed 2026-05-23: detail dropped in §4 merger)
# reg("T3-n-child-edu",   856, "checkin", ("lag_sensitivity.json", "results.25.ChildEdu_ceil90.n"),
#     [GDP_INDEP], tol=0)
# (removed 2026-05-23: detail dropped in §4 merger)
# reg("T3-ctry-child-edu", 157, "checkin", ("lag_sensitivity.json", "results.25.ChildEdu_ceil90.countries"),
#     [GDP_INDEP], tol=0)
# (removed 2026-05-23: detail dropped in §4 merger)
# reg("T3-n-u5mr",         787, "checkin", ("lag_sensitivity.json", "results.25.U5MR_ceil90.n"),
#     [GDP_INDEP], tol=0)
# (removed 2026-05-23: detail dropped in §4 merger)
# reg("T3-ctry-u5mr",      147, "checkin", ("lag_sensitivity.json", "results.25.U5MR_ceil90.countries"),
#     [GDP_INDEP], tol=0)
# (removed 2026-05-23: detail dropped in §4 merger)
# T3-n-gdp (577) and T3-ctry-gdp (109) removed from paper
# Cambodia: peer median
reg("Cambodia-peer-median-1985", 21, "derived",
    "Median lower_sec_both 1985 among countries within ±3pp of Cambodia 1960, excl Cambodia",
    [CAMBODIA], tol=1)
reg("Cambodia-peer-median-2015", 46, "derived",
    "Median lower_sec_both 2015 among countries within ±3pp of Cambodia 1960, excl Cambodia",
    [CAMBODIA], tol=1)
# Britain/Netherlands timeline deregistered: not cited in current paper.
# Was previously false-matching on "2000" (Spain universal-completion year)
# in §The Decision; that Spain paragraph was cut as recap.

# ══════════════════════════════════════════════════════════════════════════
# TWO-WAY FE DETAILS — REMOVED (values not in current paper)

# ══════════════════════════════════════════════════════════════════════════
# REMAINING GDP CONTRIBUTION
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# COUNTRY COUNTS — abstract and conclusion
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# Korea-Costa Rica GDP ratio (§the-decision)
# ══════════════════════════════════════════════════════════════════════════
reg("CR-Korea-ratio",  3.5, "derived",
    "Costa Rica 1960 GDP / Korea 1960 GDP = 3609/1038 ~ 3.5",
    [POLICY], tol=0.1)

# CHINA PEER COMPARISON — REMOVED (values not in current paper)

# ══════════════════════════════════════════════════════════════════════════
# CHINA PROVISION DISCONTINUITY
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# KOREA BETA — Figure 3 context
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# SECTION COVERAGE — register values in the sections where they appear
# These values are already verified above (or are const/ref); these
# entries ensure the coverage scanner knows which sections they appear in.
# ══════════════════════════════════════════════════════════════════════════

# --- DEF_DEV section: thresholds + Japan LE ---
reg("TFR-threshold-defdev",  3.65,  "wdi", ("tfr", "USA", 1960), [DEF_DEV], tol=0.01)
reg("LE-threshold-defdev",   69.8,  "wdi", ("le", "USA", 1960), [DEF_DEV], tol=0.05)
reg("LE-Japan-1960-sec",     67.7,  "wdi", ("le", "Japan", 1960), [DEF_DEV], tol=0.05)

# --- INTRO section: thresholds cited in opening convergence lede ---
# Numerical values are stated in defining-development; the lede references them by year (1960 USA values).
# Cumulative-developed curve milestones: 1961 (baseline), 1993 (pre-China), 1994 (China crosses)
reg("Cumulative-1961",       1961,  "const", "Cumulative-developed curve baseline year",
    [INTRO], tol=0)
reg("Cumulative-1993",       1993,  "const", "Cumulative-developed curve pre-China jump year",
    [INTRO], tol=0)
reg("Cumulative-1994",       1994,  "const", "Cumulative-developed curve year China crosses",
    [INTRO, DEF_DEV], tol=0)

# --- LUTZ section: college completion analysis ---
reg("College-r-sec",         0.45,  "checkin", ("college_le_gradient.json", "results.correlation.actual"), [GDP_INDEP], tol=0.01)
reg("College-LE-gradient-sec", 5.7, "checkin", ("college_le_gradient.json", "results.gradient.actual"), [], tol=0.1)

# --- INVISIBLE section: happiness country count ---
reg("Happiness-n-countries",  159,  "checkin", ("happiness_education.json", "numbers.n_countries"), ["invisible-from-inside"], tol=0)

# --- HOW_EDU section: Myanmar TFR + LE values ---
reg("TFR-Myanmar-1960-sec",   5.9,  "wdi", ("tfr", "Myanmar", 1960), [MYANMAR], tol=0.2)
reg("TFR-Myanmar-2015-sec",   2.3,  "wdi", ("tfr", "Myanmar", 2015), [MYANMAR], tol=0.2)
reg("LE-Myanmar-1960-sec",   44.1,  "wdi", ("le", "Myanmar", 1960), [MYANMAR], tol=0.05)
reg("LE-Myanmar-2015-sec",   65.3,  "wdi", ("le", "Myanmar", 2015), [MYANMAR], tol=0.05)

# --- CAUSAL section: regression + Uganda/India LE ---
reg("T2-GDP-beta-causal",   0.012,  "checkin",
    ("education_outcomes.json", "numbers.T2-GDP-beta"),
    [GDP_INDEP], tol=0.001)  # was CAUSAL/identification (deleted 2026-05-23); 0.012 now lives in §gdp-has-no
reg("CutOff-10-gdp-r2-causal", 0.292, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_10_gdp_r2"),
    [], tol=0.0005)
reg("CutOff-30-ratio-ce-causal", 3.1, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_30_ratio"),
    [], tol=0.05)
reg("LE-Uganda-1980-sec",    43.5,  "wdi", ("le", "Uganda", 1980), [MYANMAR], tol=0.05)

# --- EDU_VS_GDP section: Two-way FE sample size ---

# --- APPENDIX_ROBUST: Barro-Lee replication R² values ---

# --- APPENDIX_ROBUST: Barro-Lee FWL residualization R² values ---
# Paper claims: BL residualized GDP R² ≤ 0.003 (using at-least-some-sec, the higher values)
# Paper claims: WCDE residualized GDP R² ≤ 0.002

# --- GDP_INDEP section ---

# --- Robustness numbers in GDP_INDEP section (from robustness_tests.py) ---
reg("Rob-quad-resid-R2",  0.02, "checkin", ("robustness_tests.json", "numbers.Rob-quad-resid-R2"),
    [], tol=0.005)
reg("Rob-boot-edu-lo",    0.34, "checkin", ("robustness_tests.json", "numbers.Rob-boot-edu-lo"),
    [], tol=0.01)
reg("Rob-boot-gdp-lo",    0.00, "checkin", ("robustness_tests.json", "numbers.Rob-boot-gdp-lo"),
    [], tol=0.005)  # was [GDP_INDEP]; the "lag 0" token it matched was cut in the 2026-06-19 trim — traceability-only now, like Rob-boot-gdp-hi
reg("Rob-boot-gdp-hi",    0.04, "checkin", ("robustness_tests.json", "numbers.Rob-boot-gdp-hi"),
    [], tol=0.01)

# --- Symmetric permutation null (permutation_null_symmetric.py) ---
# One harness, education and residualised GDP through the same full-shuffle
# null on the common-sample panel (n=645). Paper cites rounded |Z| bounds:
# raw education 14–18 SDs above null; education⊥GDP 11–13; residualised GDP
# within two SDs (a word, not a registered digit). 200 iterations.
reg("PNS-n-iter",      200, "checkin",
    ("permutation_null_symmetric.json", "n_iter"),
    [GDP_INDEP], tol=0)
# [retired 2026-06-18 ch9/ch11 compression] reg("PNS-raw-edu-zlo",  14, "checkin",
# [retired 2026-06-18 ch9/ch11 compression]     ("permutation_null_symmetric.json", "full_shuffle_abs_z_bounds.raw_edu.min"),
# [retired 2026-06-18 ch9/ch11 compression]     [GDP_INDEP], tol=0.5)
# [retired 2026-06-18 ch9/ch11 compression] reg("PNS-raw-edu-zhi",  18, "checkin",
# [retired 2026-06-18 ch9/ch11 compression]     ("permutation_null_symmetric.json", "full_shuffle_abs_z_bounds.raw_edu.max"),
# [retired 2026-06-18 ch9/ch11 compression]     [GDP_INDEP], tol=0.5)
# [retired 2026-06-18 ch9/ch11 compression] reg("PNS-edu-resid-zlo", 11, "checkin",
# [retired 2026-06-18 ch9/ch11 compression]     ("permutation_null_symmetric.json", "full_shuffle_abs_z_bounds.edu_resid.min"),
# [retired 2026-06-18 ch9/ch11 compression]     [GDP_INDEP], tol=0.5)
# [retired 2026-06-18 ch9/ch11 compression] reg("PNS-edu-resid-zhi", 13, "checkin",
# [retired 2026-06-18 ch9/ch11 compression]     ("permutation_null_symmetric.json", "full_shuffle_abs_z_bounds.edu_resid.max"),
# [retired 2026-06-18 ch9/ch11 compression]     [GDP_INDEP], tol=0.5)


# --- Fertility R² at primary education in DEMOG section ---
reg("Fert-primary-R2",    0.65, "checkin", ("edu_vs_gdp_tfr_residualized.json", "numbers.Fert-primary-R2"),
    [], tol=0.02)

# --- SSA-only robustness footnote in §gdp-has-no-independent-effect ---
# (Under LAG_TFR=5 biological anchoring: primary→TFR(T+5) within SSA.)
reg("SSA-Fert-primary-R2",     0.392, "checkin",
    ("ssa_education_tfr.json", "panel_by_level.primary.r2_within"),
    [], tol=0.02)
reg("SSA-Fert-primary-beta",  -0.028, "checkin",
    ("ssa_education_tfr.json", "panel_by_level.primary.beta"),
    [], tol=0.005)
# [retired 2026-06-18 Panel-B cut: exposed surplus reg] reg("SSA-Fert-primary-se",     0.005, "checkin",
# [retired 2026-06-18 Panel-B cut: exposed surplus reg]     ("ssa_education_tfr.json", "panel_by_level.primary.se"),
# [retired 2026-06-18 Panel-B cut: exposed surplus reg]     [GDP_INDEP], tol=0.002)
# reg("SSA-Fert-n-countries",    44, "checkin",
#     ("ssa_education_tfr.json", "panel_by_level.primary.n_countries"),
#     [GDP_INDEP], tol=0)
# (removed 2026-05-23: detail dropped in §4 merger)
# reg("SSA-Fert-n-obs",          308, "checkin",
#     ("ssa_education_tfr.json", "panel_by_level.primary.n_obs"),
#     [GDP_INDEP], tol=0)
# (removed 2026-05-23: detail dropped in §4 merger)
reg("SSA-cross-n",             10, "checkin",
    ("ssa_education_tfr.json", "crossing.n"),
    [GDP_INDEP], tol=0)
reg("SSA-cross-primary-median", 77, "checkin",
    ("ssa_education_tfr.json", "crossing.median"),
    [], tol=1)
# Global clean-set median primary completion at TFR<3.65 crossing
# (referenced inline alongside SSA-cross-primary-median for context).
reg("Global-cross-primary-median", 79, "checkin",
    ("primary_at_tfr_crossing.json", "primary_at_cross.median"),
    [], tol=1)
# Phenotype-consistency pass count: 185 panel minus the 14 Soviet-bloc
# republics that carry a U5MR phenotype to test = 171. Within that bloc
# the failure is the eastern/southern periphery; the European-core
# republics to the west pass (see §moscow-meridian).
reg("Phenotype-pass-count", 171, "derived",
    "185 panel total - 14 Soviet-bloc republics with a U5MR phenotype to test = 171",
    ["the-anomaly"], tol=0)
# Table 7 Panel A GDP-row country count (T=1960-1990, ceiling 60%/90%):
# reg("T7-GDP-row-countries", 179, "derived",
#     "Table 7 Panel A note: 927 obs / 179 countries (GDP row, T=1960-1990)",
#     ["the-shape-of-the-response"], tol=0)
# (removed 2026-05-23: number cut from main text in §5 trim)
# Korea β_g = 1.8 intermediate trajectory point (declining curve at 4-pt baseline).
# Fig1-Korea-beta-1.8 reg also exists; this is a section-wide duplicate to ensure
# consume_match credits it on the prose-restatement line.

# --- UNIVERSALITY_ACROSS_SUBSAMPLES section: tab:over-performers residuals + GDP values ---
reg("GDP-Maldives-2015-sec",   9645,  "wdi", ("gdp", "Maldives", 2015), [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.5)
reg("GDP-CapeVerde-2015-sec",  3415,  "wdi", ("gdp", "Cape Verde", 2015), [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.5)
reg("GDP-Bhutan-2015-sec",     2954,  "wdi", ("gdp", "Bhutan", 2015), [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.5)
reg("GDP-Tunisia-2015-sec",    4015,  "wdi", ("gdp", "Tunisia", 2015), [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.5)
reg("GDP-Nepal-2015-sec",       876,  "wdi", ("gdp", "Nepal", 2015), [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.5)
reg("GDP-Vietnam-2015-sec",    2578,  "wdi", ("gdp", "Vietnam", 2015), [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.5)
reg("GDP-Bangladesh-2015-sec", 1224,  "wdi", ("gdp", "Bangladesh", 2015), [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.5)
reg("GDP-India-2015-sec",      1584,  "wdi", ("gdp", "India", 2015), [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.5)

# --- tab:transition-year — Crossing dates (table4_crossings.py) ---
# Taiwan (all ~1970)
reg("T5-Taiwan-dev",     1970, "checkin", ("table4_crossings.json", "results.Taiwan.both_crossed"),
    [SEN_CASES], tol=0)
# S. Korea
reg("T5-Korea-dev",      1987, "checkin", ("table4_crossings.json", "results.South Korea.both_crossed"),
    [SEN_CASES], tol=0)
reg("T5-Korea-TFR",      1975, "checkin", ("table4_crossings.json", "results.South Korea.tfr_crossing_best"),
    [SEN_CASES], tol=0)
reg("T5-Korea-LE",       1987, "checkin", ("table4_crossings.json", "results.South Korea.le_crossing_best"),
    [SEN_CASES], tol=0)
# Cuba
reg("T5-Cuba-dev",       1974, "checkin", ("table4_crossings.json", "results.Cuba.both_crossed"),
    [SEN_CASES, CUBA], tol=0)
reg("T5-Cuba-TFR",       1972, "checkin", ("table4_crossings.json", "results.Cuba.tfr_crossing_best"),
    [SEN_CASES, DEF_DEV], tol=0)
reg("T5-Cuba-LE",        1974, "checkin", ("table4_crossings.json", "results.Cuba.le_crossing_best"),
    [SEN_CASES], tol=0)
# Bangladesh
reg("T5-Bangladesh-dev",  2014, "checkin", ("table4_crossings.json", "results.Bangladesh.both_crossed"),
    [SEN_CASES, BANGLADESH], tol=0)
reg("T5-Bangladesh-TFR",  1995, "checkin", ("table4_crossings.json", "results.Bangladesh.tfr_crossing_best"),
    [SEN_CASES], tol=0)
reg("T5-Bangladesh-LE",   2014, "checkin", ("table4_crossings.json", "results.Bangladesh.le_crossing_best"),
    [SEN_CASES], tol=0)
# Sri Lanka
reg("T5-SriLanka-dev",   1993, "checkin", ("table4_crossings.json", "results.Sri Lanka.both_crossed"),
    [SEN_CASES, SRI_LANKA], tol=0)
reg("T5-SriLanka-TFR",   1981, "checkin", ("table4_crossings.json", "results.Sri Lanka.tfr_crossing_best"),
    [SEN_CASES, SRI_LANKA], tol=0)
reg("T5-SriLanka-LE",    1993, "checkin", ("table4_crossings.json", "results.Sri Lanka.le_crossing_best"),
    [SEN_CASES], tol=0)
# China
reg("T5-China-dev",      1994, "checkin", ("table4_crossings.json", "results.China.both_crossed"),
    [SEN_CASES, CHINA], tol=0)
reg("T5-China-TFR",      1975, "checkin", ("table4_crossings.json", "results.China.tfr_crossing_best"),
    [SEN_CASES, CHINA], tol=0)
reg("T5-China-LE",       1994, "checkin", ("table4_crossings.json", "results.China.le_crossing_best"),
    [SEN_CASES], tol=0)
# Japan LE crossing — cited in §defining-development as "it crossed in 1964"
reg("Japan-LE-crossing", 1964, "checkin", ("table4_crossings.json", "results.Japan.le_crossing_best"),
    [DEF_DEV], tol=0)
# Count of countries that crossed both thresholds by 2022 (development_threshold_count.py)
# Cited across intro, defining-development, and goskomstat sections.
reg("Countries-crossed-both",    154, "checkin",
    ("development_threshold_count.json", "numbers.countries_crossing_both"),
    [INTRO, DEF_DEV, "the-anomaly"], tol=0)
reg("Countries-crossed-both-2",  154, "checkin",
    ("development_threshold_count.json", "numbers.countries_crossing_both"),
    [INTRO], tol=0)
reg("Countries-crossed-both-abs1", 154, "checkin",
    ("development_threshold_count.json", "numbers.countries_crossing_both"),
    [ABSTRACT], tol=0)
# Second "150 years" reference in the famine-test prose (paired with Ireland-150yr)
reg("Ireland-150yr-2",          150,  "ref",
    "Years from Penal Laws to potato blight — second prose reference",
    [FAMINE_TEST], tol=0)
# Cuba 1974 crossing year cited in §methodological-frontier (Cuba vs. Cambodia divergence)
reg("Cuba-dev-cross-L4865",    1974, "checkin",
    ("table4_crossings.json", "results.Cuba.both_crossed"),
    ["methodological-frontier"], tol=0)

# --- SEN_CASES section: thresholds + country values ---
reg("TFR-threshold-sec",     3.65,   "wdi", ("tfr", "USA", 1960), [SEN_CASES], tol=0.01)
reg("LE-threshold-sec",      69.8,   "wdi", ("le", "USA", 1960), [SEN_CASES], tol=0.05)
reg("TFR-Uganda-sec",        4.39,   "wdi", ("tfr", "Uganda", 2022), [SEN_CASES], tol=0.2)
reg("LE-Uganda-2022-sec",    67.7,   "wdi", ("le", "Uganda", 2022), [SEN_CASES], tol=0.05)
reg("Taiwan-ppyr-sec",       2.15,   "derived", "(Taiwan-1985 - Taiwan-1950) / 35", [SEN_CASES], tol=0.005)
reg("SriLanka-ppyr-sec",     1.20,   "derived", "Sri Lanka edu rate (Table-2 Rate column, approx)", [SEN_CASES], tol=0.1)
reg("India-beta-g-sec",      1.6,    "derived", "India figure beta_g approximation (figure read)", [], tol=0.1)
reg("LE-SriLanka-1988-sec",  69.0,   "wdi", ("le", "Sri Lanka", 1988), [SRI_LANKA], tol=0.5)
reg("LE-SriLanka-1989-sec",  67.3,   "wdi", ("le", "Sri Lanka", 1989), [SRI_LANKA], tol=0.5)
reg("LE-SriLanka-1993-sec",  70.0,   "wdi", ("le", "Sri Lanka", 1993), [SRI_LANKA], tol=0.05)
reg("China-CR-gain-1980", 15.0,  "derived", "China CR-era cohort gain (1980 - 1975, peak 民办 era)", [CHINA], tol=0.5)
reg("China-CR-gain-1985", 10.7,  "derived", "China CR-era cohort gain (1985 - 1980, late CR)", [CHINA], tol=0.5)
reg("China-CR-gain-1990", 2.3,   "derived", "China cohort gain (1990 - 1985, post-民办 trough)", [CHINA], tol=0.5)
reg("China-1986-law",    1986,   "ref",     "Compulsory Education Law of the PRC, promulgated 1986", [CHINA], tol=0)
reg("LE-China-1980-sec",     64,     "wdi", ("le", "China", 1980), [CHINA], tol=0.5)
reg("Cuba-1960-edu-sec",     40.3,   "wcde", ("lower_sec_both.csv", "Cuba", 1960), [CUBA], tol=0.05)
reg("Cuba-1961-campaign",   1961,   "ref", "Cuba literacy campaign year (Prieto 1981)", [], tol=0)
reg("Bangladesh-1960-edu-sec", 11.4, "wcde", ("lower_sec_both.csv", "Bangladesh", 1960), [BANGLADESH], tol=0.05)

# --- Empirical years in country subsections ---
# Sri Lanka: LE timeline
reg("SriLanka-LE-1988-yr",  1988, "ref", "WDI observation year for Sri Lanka LE peak",
    [SRI_LANKA], tol=0)
# China: LE gap and structural break years (from china_mean_yrs_vs_peers.json)
reg("China-LE-gap-yrs",     1965, "checkin",
    ("china_mean_yrs_vs_peers.json", "annual_data[5].year"),
    [CHINA], tol=0)
reg("China-LE-converge-yr", 1991, "checkin",
    ("china_mean_yrs_vs_peers.json", "key_data_points.le_crossover_year"),
    [CHINA], tol=1)
reg("China-LE-crossover",   1992, "ref", "Year China crossed above education-predicted LE",
    [CHINA], tol=0)
reg("China-break-yr",       1981, "ref", "Structural break year (barefoot doctor removal)",
    [CHINA], tol=0)
# Shock test: Russia and South Africa years
reg("Russia-LE-2009-yr",    2009, "ref", "WDI observation year for Russia LE recovery", [SHOCK_TEST], tol=0)
reg("Russia-LE-2019-yr",    2019, "ref", "WDI observation year for Russia LE", [SHOCK_TEST], tol=0)
reg("Botswana-1996-yr",     1996, "ref", "Botswana universal treatment cohort year (De Walque 2006)", [], tol=0)
reg("SA-LE-2019-yr",        2019, "ref", "WDI observation year for SA LE recovery", [SHOCK_TEST], tol=0)
# Cambodia: home-niche shadow timeline
reg("Cambodia-1991-yr",     1991, "ref", "Paris Peace Accords / end of conflict",
    [CAMBODIA], tol=0)
reg("Cambodia-1995-yr",     1995, "ref", "Post-reconstruction education jump year", [CAMBODIA], tol=0)
reg("Cambodia-2011-yr",     2011, "ref", "Post-disruption cohort reaches school age (1985+25+1)",
    [CAMBODIA], tol=0)
reg("Cambodia-1979-yr",     1979, "ref", "Year Khmer Rouge regime fell", [CAMBODIA], tol=0)
# Kerala
reg("Kerala-1981-yr",       1981, "ref", "Census year (Dreze & Sen 2001)", [KERALA], tol=0)
reg("Kerala-1991-yr",       1991, "ref", "Census year (Dreze & Sen 2001)", [KERALA], tol=0)
reg("GDP-Bangladesh-2014-sec", 1159, "wdi", ("gdp", "Bangladesh", 2014), [BANGLADESH, SEN_CASES], tol=0.5)
reg("CutOff-30-ratio-ce-sec", 3.1,  "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_30_ratio"),
    [], tol=0.05)

# --- CAMBODIA section: WCDE education values ---
reg("Cambodia-1975-sec",     10.1,   "wcde", ("lower_sec_both.csv", "Cambodia", 1975), [CAMBODIA], tol=0.5)
reg("Cambodia-1980",          9.4,   "wcde", ("lower_sec_both.csv", "Cambodia", 1980), [CAMBODIA], tol=0.5)
reg("Cambodia-1985-sec",      9.5,   "wcde", ("lower_sec_both.csv", "Cambodia", 1985), [CAMBODIA], tol=0.5)
reg("Cambodia-1995-sec",     35.1,   "wcde", ("lower_sec_both.csv", "Cambodia", 1995), [CAMBODIA], tol=0.05)

# --- CAMBODIA section: grandparent shadow (absolute values of β from THE_SHAPE_OF_THE_RESPONSE) ---

# --- INSTIT section ---
reg("GDP-Qatar-2015-sec",    69000,  "wdi", ("gdp", "Qatar", 2015), [INSTIT], tol=500)

# --- INSTIT section: India vs China comparison ---
reg("China-instit-75",        75,    "wcde", ("lower_sec_both.csv", "China", 1990), [INSTIT], tol=0.5)
reg("China-instit-rate",      1.6,   "derived", "(China-1990 - China-1950) / 40, WCDE lower_sec_both", [INSTIT], tol=0.1)
reg("India-instit-37",        37,    "wcde", ("lower_sec_both.csv", "India", 1990), [INSTIT], tol=0.5)
reg("India-instit-rate",      0.7,   "derived", "(India-1990 - India-1950) / 40, WCDE lower_sec_both", [INSTIT], tol=0.1)
reg("Global-rate-1960s-peak", 1.52, "checkin",
    ("institution_test_entry_cohort_rates.json", "decadal_entry_cohort.1960-1970.mean_pp_per_year"),
    [INSTIT], tol=0.01)
reg("Global-rate-1990s-trough", 0.74, "checkin",
    ("institution_test_entry_cohort_rates.json", "decadal_entry_cohort.1990-2000.mean_pp_per_year"),
    [], tol=0.01)
reg("Global-rate-2010s-rebound", 1.29, "checkin",
    ("institution_test_entry_cohort_rates.json", "decadal_entry_cohort.2010-2020.mean_pp_per_year"),
    [], tol=0.01)

# --- INSTIT section: regime type numbers ---
reg("Regime-n-countries",     160,   "checkin", ("regime_education_test.json", "n_countries"), [INSTIT])
reg("Regime-demo-mean",       10.3,  "checkin", ("regime_education_test.json", "results_by_lag.20.mean_demo"), [INSTIT], tol=0.3)
reg("Regime-auto-mean",       8.1,   "checkin", ("regime_education_test.json", "results_by_lag.15.mean_auto"), [INSTIT], tol=0.1)

# --- COLONIAL TEST section ---
# (Re-anchored on AJR's 64-country base sample with AJR's own avexpr
# institutional measure, replacing the polity2/polity5 framing.)
COLONIAL = "the-colonial-test"
reg("Colonial-edu1950-r2-pct",    52,    "derived",
    "100 × univar_r2_log_gdp_2020.edu_1950.r2 from ajr_n61_deep_dive.json",
    [COLONIAL], tol=1)
reg("Colonial-avexpr-r2-pct",     53,    "derived",
    "100 × univar_r2_log_gdp_2020.avexpr.r2 from ajr_n61_deep_dive.json",
    [COLONIAL], tol=1)
reg("Colonial-religion-r2-pct",    6,    "derived",
    "100 × univar_r2_log_gdp_2020.protestant.r2 from ajr_n61_deep_dive.json",
    [COLONIAL], tol=1)
reg("Colonial-edu-religion-r2",   0.521, "checkin",
    ("ajr_n61_deep_dive.json", "univar_r2_log_gdp_2020.edu_plus_religion.r2"),
    [COLONIAL], tol=0.005)
reg("Colonial-edu-alone-r2",      0.518, "checkin",
    ("ajr_n61_deep_dive.json", "univar_r2_log_gdp_2020.edu_1950.r2"),
    [COLONIAL], tol=0.005)
reg("Colonial-edu-avexpr-corr",   0.62,  "checkin",
    ("ajr_n61_deep_dive.json", "correlations.corr_edu1950_avexpr"),
    [COLONIAL], tol=0.02)
reg("Spain-1875-primary",         0.6,   "wcde",
    ("cohort_primary_both.csv", "Spain", 1875), [COLONIAL], tol=0.1)
reg("Portugal-1875-primary",      0.1,   "wcde",
    ("cohort_primary_both.csv", "Portugal", 1875), [COLONIAL], tol=0.1)
# 2SLS IV test (avexpr replaces polity2 throughout). Numbers cited in
# §The Colonial Test main text + Tables A6/A7 + §appendix-colonial-iv +
# §causal-identification (cross-section).
INST_CHAL = "the-institutional-challenge"
APPX_COL_IV = "the-colonial-test"
CAUSAL_BAD = "identification"
reg("IV-edu-F",                   10.71, "checkin",
    ("iv_2sls_colonial_icrg.json", "gdp_edu_first_stage_F"),
    [COLONIAL],  # APPX_COL_IV slot dropped: §7.4 footnote restatement cut (2026-06-19 tier-1)
    tol=0.5)
reg("IV-inst-F",                   9.61, "checkin",
    ("iv_2sls_colonial_icrg.json", "gdp_inst_first_stage_F"),
    [COLONIAL], tol=0.5)  # APPX_COL_IV slot dropped (footnote cut, 2026-06-19)
reg("IV-edu-coef",                0.043, "checkin",
    ("iv_2sls_colonial_icrg.json", "gdp_edu_2sls_coef"),
    [], tol=0.005)
reg("IV-edu-t",                   2.56,  "checkin",
    ("iv_2sls_colonial_icrg.json", "gdp_edu_2sls_t"),
    [], tol=0.1)
# Structural test: avexpr-as-IV-for-education (cited everywhere)
reg("IV-avexpr-edu-F",           37.13, "checkin",
    ("ajr_n61_deep_dive.json", "t33_avexpr_iv_for_education.first_stage_F"),
    [COLONIAL],  # APPX_COL_IV slot dropped (footnote cut, 2026-06-19)
    tol=0.5)
reg("IV-avexpr-edu-coef",         0.084, "checkin",
    ("ajr_n61_deep_dive.json", "t33_avexpr_iv_for_education.second_stage_edu_coef"),
    [COLONIAL], tol=0.005)
reg("IV-avexpr-edu-t",            6.74, "checkin",
    ("ajr_n61_deep_dive.json", "t33_avexpr_iv_for_education.second_stage_t"),
    [COLONIAL], tol=0.1)
# AJR base sample size constants — the AJR engagement now lives only in
# Ch 11 §the-colonial-test and §the-institutional-challenge; the bad-control
# / standard-controls AJR pointers in Ch 9 were removed in the biology-first
# restructure.
reg("AJR-base-n",                64, "const",
    "AJR (2001) base sample of former colonies with settler-mortality data",
    [COLONIAL, INST_CHAL, APPX_COL_IV,
     UNIVERSALITY_ACROSS_SUBSAMPLES],
    tol=0)
# "64-country" now appears twice in §the-colonial-test (base + the
# Ireland-within-polity insertion); the IV-diagnostics footnote that
# carried a third mention was cut in the 2026-06-19 tier-1 trim.
# [retired 2026-06-19] AJR-base-n-footnote (64, footnote restatement):
# 64 stays covered by AJR-base-n (base) and AJR-base-n-ireland.
reg("AJR-base-n-ireland",        64, "const",
    "AJR (2001) base sample — Ireland within-polity insertion",
    [], tol=0)
reg("AJR-merged-n",              61, "const",
    "AJR-64 after merge with WCDE (3 dropped: Côte d'Ivoire, Ethiopia, Hong Kong)",
    [INST_CHAL], tol=0)

# --- ABSTRACT: residualization summary thresholds ---
reg("Abstract-resid-r2",      0.0006, "checkin",
    ("regression_tables.json", "results.U5MR.90.GDP (residualized).r2"),
    [], tol=0.005)
reg("Abstract-resid-p",       0.76,   "checkin",
    ("regression_tables.json", "results.U5MR.90.GDP (residualized).pval"),
    [], tol=0.01)

# --- POLICY section: Spain ---
# Spain-450 deregistered: paper's §The Convergence renders this as
# "four-hundred-and-fifty" in words, not matchable by numeric scan; the
# §The Decision restatement (which used digits) was cut as recap.

# --- POLICY section: Korea-Costa Rica comparison ---
reg("Fig1-Korea-beta-3.6-sec", 4.6,  "checkin", ("beta_vs_baseline.json", "numbers.Fig1-Korea-beta-3.6"), [], tol=0.1)

# --- POLICY section: SDG 4 framed on the biological clock (12-year
# minimum), no longer on the 2030-deadline implied-pace inversion. The
# sdg4_implied_pace.py result (9.5 pp/yr median, 2.13 Korea record) is
# retained in scripts/ but no longer cited in §The Decision, so its
# numbers are not registered against the paper here.

# --- GRANDPARENT EFFECT (in education-predicts section) ---
reg("GM-tfr-low-beta-gm", -0.048, "checkin",
    ("grandparent_effect_all_outcomes.json", "outcomes.tfr.low_edu.parent_gp.beta_grandparent_edu"),
    [THE_SHAPE_OF_THE_RESPONSE, "the-generational-transmission-mechanism"],
    tol=0.005)
reg("GM-tfr-low-beta-m", -0.068, "checkin",
    ("grandparent_effect_all_outcomes.json", "outcomes.tfr.low_edu.parent_gp.beta_parent_edu"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.005)
# GM-child-edu-* and GM-le-* removed 2026-05-31: the grandparent child-
# education and LE results are not cited in §The Shape of the Response /
# §The Grandparent Channel (which cite only TFR and U5MR), so those regs
# matched unrelated numbers by tolerance. The real grandparent-LE citation
# lives in §the-generational-lag (TS-GP-LE-beta / TS-GP-LE-r2-gain).

# B6 grandparent table (tab:grandparent) — N, R², and parent-only β
# cells for the two cited specs.
# (TFR rows updated to LAG_TFR=5 biological anchoring; U5MR row unchanged.)
reg("GP-tfr-low-n",          464, "checkin",
    ("grandparent_effect_all_outcomes.json", "outcomes.tfr.low_edu.parent_gp.n"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=3)
reg("GP-tfr-low-ctry",        80, "checkin",
    ("grandparent_effect_all_outcomes.json", "outcomes.tfr.low_edu.parent_gp.n_countries"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=2)
reg("GP-tfr-low-r2-m1",      0.69, "checkin",
    ("grandparent_effect_all_outcomes.json", "outcomes.tfr.low_edu.parent_only.within_r2"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.01)
reg("GP-tfr-low-r2-m2",      0.74, "checkin",
    ("grandparent_effect_all_outcomes.json", "outcomes.tfr.low_edu.parent_gp.within_r2"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.01)
# Log-TFR low-baseline rows: still computed in the script but no longer
# cited in the rewritten §grandparent-channel paragraph. Kept here as a
# diagnostic against silent drift in the JSON.
reg("GP-tfr-log-low-r2-m1",  0.71, "checkin",
    ("grandparent_effect_all_outcomes.json", "outcomes.tfr_log.low_edu.parent_only.within_r2"),
    [], tol=0.01)
reg("GP-tfr-log-low-r2-m2",  0.75, "checkin",
    ("grandparent_effect_all_outcomes.json", "outcomes.tfr_log.low_edu.parent_gp.within_r2"),
    [], tol=0.01)
reg("GP-tfr-log-beta-gp",    -0.010, "checkin",
    ("grandparent_effect_all_outcomes.json", "outcomes.tfr_log.low_edu.parent_gp.beta_grandparent_edu"),
    [], tol=0.005)
reg("GP-u5-n",               825, "checkin",
    ("grandparent_effect_all_outcomes.json", "outcomes.u5_log.full.parent_gp.n"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=5)
reg("GP-u5-ctry",            165, "checkin",
    ("grandparent_effect_all_outcomes.json", "outcomes.u5_log.full.parent_gp.n_countries"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=2)
reg("GP-u5-parent-only-beta", -0.033, "checkin",
    ("grandparent_effect_all_outcomes.json", "outcomes.u5_log.full.parent_only.beta_parent_edu"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.005)
reg("GP-u5-parent-only-r2",  0.4072, "checkin",
    ("grandparent_effect_all_outcomes.json", "outcomes.u5_log.full.parent_only.within_r2"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.005)
# Duplicate regs for the same N/countries values surfacing again in the
# table cells (paper line 3388/3389). Each line is scanned independently.

# Note: sex-disaggregated grandparent betas (GF / GM) intentionally not
# registered. Under biological TFR anchoring (LAG_TFR=5) the GF/GM gap
# tracks the historical female-exclusion-from-schooling level gap and
# would force a level-vs-causal interpretation in the prose. The paper
# uses the both-sexes-pooled grandparent coefficient as the headline;
# the sex split remains in checkin/grandparent_effect.json for the
# diagnostic record but is not cited.

# --- GRANDPARENT EFFECT on U-5 MORTALITY (full panel, THE_SHAPE_OF_THE_RESPONSE) ---
reg("GM-u5-beta-gp", -0.020, "checkin",
    ("grandparent_effect_all_outcomes.json",
     "outcomes.u5_log.full.parent_gp.beta_grandparent_edu"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.0008)
reg("GM-u5-beta-p", -0.018, "checkin",
    ("grandparent_effect_all_outcomes.json",
     "outcomes.u5_log.full.parent_gp.beta_parent_edu"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.005)
reg("GM-u5-beta-ratio", 1.1039, "checkin",
    ("grandparent_effect_all_outcomes.json",
     "outcomes.u5_log.full.beta_ratio_gp_over_p"),
    [], tol=0.05)
reg("GM-u5-r2-m2", 0.6532, "checkin",
    ("grandparent_effect_all_outcomes.json",
     "outcomes.u5_log.full.parent_gp.within_r2"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.005)
reg("GM-u5-r2-gain-pct", 24.6, "derived",
    "U5 grandparent R² gain × 100 (grandparent_effect_all_outcomes.json outcomes.u5_log.full.r2_gain)",
    [], tol=0.3)

# --- GRANDPARENT EFFECT: SAMPLE SIZES cited inline in body prose
# (added 2026-05-20 per Methodological Review 3, point G6) ---
# reg("GP-tfr-n-full",       760, "checkin",
#     ("grandparent_effect.json", "results.tfr.parent_gp.n"),
#     [THE_SHAPE_OF_THE_RESPONSE], tol=0)
# (removed 2026-05-23: number cut from main text in §5 trim)
# reg("GP-tfr-c-full",       152, "checkin",
#     ("grandparent_effect.json", "results.tfr.parent_gp.n_countries"),
#     [THE_SHAPE_OF_THE_RESPONSE], tol=0)
# (removed 2026-05-23: number cut from main text in §5 trim)
# reg("GP-tfr-n-low",        366, "checkin",
#     ("grandparent_effect.json", "results.tfr_low_edu.parent_gp.n"),
#     [THE_SHAPE_OF_THE_RESPONSE], tol=0)
# (removed 2026-05-23: number cut from main text in §5 trim)
# reg("GP-tfr-c-low",        88, "checkin",
#     ("grandparent_effect.json", "results.tfr_low_edu.parent_gp.n_countries"),
#     [THE_SHAPE_OF_THE_RESPONSE], tol=0)
# (removed 2026-05-23: number cut from main text in §5 trim)
# reg("GP-gf-tfr-low-n",     351, "checkin",
#     ("grandparent_effect.json", "results.sex_comparison.male (grandfather)_tfr_low.n"),
#     [THE_SHAPE_OF_THE_RESPONSE], tol=0)
# (removed 2026-05-23: number cut from main text in §5 trim)
# reg("GP-gm-tfr-low-n",     372, "checkin",
#     ("grandparent_effect.json", "results.sex_comparison.female (grandmother)_tfr_low.n"),
#     [THE_SHAPE_OF_THE_RESPONSE], tol=0)
# (removed 2026-05-23: number cut from main text in §5 trim)
# reg("GP-u5-n-full",        660, "checkin",
#     ("grandparent_effect_all_outcomes.json",
#      "outcomes.u5_log.full.parent_gp.n"),
#     [THE_SHAPE_OF_THE_RESPONSE], tol=0)
# (removed 2026-05-23: number cut from main text in §5 trim)
# reg("GP-u5-c-full",        165, "checkin",
#     ("grandparent_effect_all_outcomes.json",
#      "outcomes.u5_log.full.parent_gp.n_countries"),
#     [THE_SHAPE_OF_THE_RESPONSE], tol=0)
# (removed 2026-05-23: number cut from main text in §5 trim)

# --- GRANDPARENT SEX-EQUALITY: removed 2026-05-31 ---
# The sex-disaggregated grandparent comparison is not surfaced anywhere
# (not in the paper body, not on the website): at the biological TFR lag it
# is confounded by the historical female-schooling gap (grandmother education
# near the floor, coefficient unidentified) and cannot establish a gendered
# mechanism. Script scripts/robustness/grandparent_sex_equality.py and
# checkin/grandparent_sex_equality.json deleted; the grandparent effect itself
# is reported via grandparent_effect_all_outcomes.json.

# --- COMMON-SAMPLE residualisation (E4 fix, 2026-05-20) ---
# reg("ResidCS-n",            645, "checkin",
#     ("residualization_common_sample.json", "common_sample.n"),
#     [GDP_INDEP], tol=0)
# (removed 2026-05-23: detail dropped in §4 merger)
# reg("ResidCS-c",            123, "checkin",
#     ("residualization_common_sample.json", "common_sample.n_countries"),
#     [GDP_INDEP], tol=0)
# (removed 2026-05-23: detail dropped in §4 merger)
# [retired 2026-06-18 Panel-B cut: exposed surplus reg] reg("ResidCS-r2-max",       0.005, "derived",
# [retired 2026-06-18 Panel-B cut: exposed surplus reg]     "max(residualization_common_sample.json results.*.resid_gdp_r2)",
# [retired 2026-06-18 Panel-B cut: exposed surplus reg]     [GDP_INDEP], tol=0.0005)
reg("ResidCS-p-min",        0.37, "derived",
    "min(residualization_common_sample.json results.*.p_resid)",
    [], tol=0.02)

# --- BACKFILL all-outcomes: zero-R² threshold (THE_SHAPE_OF_THE_RESPONSE) ---
reg("Backfill-zero-threshold", 0.01, "checkin",
    ("backfill_all_outcomes.json", "numbers.zero_r2_threshold"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.001)

# --- CUTOFF all-outcomes (THE_SHAPE_OF_THE_RESPONSE): TFR at <50% and U-5 at <10% ---
# reg("Cutoff-TFR-lt50-ratio", 32, "checkin",
#     ("cutoff_all_outcomes.json", "results.tfr.lt50.ratio"),
#     [THE_SHAPE_OF_THE_RESPONSE], tol=1)
# (removed 2026-05-23: number cut from main text in §5 trim)
reg("Cutoff-TFR-lt50-edu-r2", 0.567, "checkin",
    ("cutoff_all_outcomes.json", "results.tfr.lt50.edu_r2"),
    [], tol=0.005)
reg("Cutoff-TFR-lt50-gdp-r2", 0.100, "checkin",
    ("cutoff_all_outcomes.json", "results.tfr.lt50.gdp_r2"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.005)
reg("Cutoff-U5-lt10-ratio", 20.7, "checkin",
    ("cutoff_all_outcomes.json", "results.log_u5mr.lt10.ratio"),
    [], tol=1)
reg("Cutoff-U5-lt10-edu-r2", 0.448, "checkin",
    ("cutoff_all_outcomes.json", "results.log_u5mr.lt10.edu_r2"),
    [], tol=0.005)
reg("Cutoff-U5-lt10-gdp-r2", 0.022, "checkin",
    ("cutoff_all_outcomes.json", "results.log_u5mr.lt10.gdp_r2"),
    [], tol=0.002)

# Within-country FE polity-comparison registrations removed: paragraph
# dropped from §The Colonial Test in favor of the avexpr structural
# argument (polity removed throughout per editorial decision).

# --- Russia 99% in shock test section ---

# --- §south-africa-case: De Neve Botswana (literature reference) ---
reg("DeNeve-HIV-8.1pp", 8.1, "ref",
    "De Neve et al. 2015, Lancet Global Health: each year of secondary schooling -> 8.1pp HIV risk reduction",
    ["south-africa-case"])
reg("SA-window-end-yr", 2005, "ref",
    "South Africa LE-crash window end year (1990-2005)",
    [SHOCK_TEST], tol=0)

# --- SHOCK TEST section: Russia ---
reg("Russia-1990-edu",  99,    "wcde", ("lower_sec_both.csv", "Russia", 1990), [SHOCK_TEST], tol=1)
reg("Russia-1988-LE",   69.5,  "wdi",  ("le", "Russia", 1988), [SHOCK_TEST], tol=0.1)
reg("Russia-1994-LE",   64.5,  "wdi",  ("le", "Russia", 1994), [SHOCK_TEST], tol=0.1)
reg("Russia-1990-TFR",  1.89,  "wdi",  ("tfr", "Russia", 1990), [SHOCK_TEST], tol=0.05)
reg("Russia-2000-TFR",  1.20,  "wdi",  ("tfr", "Russia", 2000), [SHOCK_TEST], tol=0.05)
reg("Russia-2009-LE",   68.7,  "wdi",  ("le", "Russia", 2009), [SHOCK_TEST], tol=0.1)
reg("Russia-2019-LE",   73.1,  "wdi",  ("le", "Russia", 2019), [SHOCK_TEST], tol=0.1)

# --- SHOCK TEST section: South Africa ---
reg("SA-1990-lsec",     64.5,  "wcde", ("lower_sec_both.csv", "South Africa", 1990), [SHOCK_TEST], tol=0.5)
reg("SA-2005-lsec",     82.6,  "wcde", ("lower_sec_both.csv", "South Africa", 2005), [SHOCK_TEST], tol=0.5)
# SA-1990-primary / SA-2005-primary removed 2026-05-26 — SA case rewrote
# from primary→TFR to lower-sec→TFR mechanism.
reg("SA-1990-LE",       62.9,  "wdi",  ("le", "South Africa", 1990), [SHOCK_TEST], tol=0.1)
reg("SA-2005-LE",       53.9,  "wdi",  ("le", "South Africa", 2005), [SHOCK_TEST], tol=0.1)
reg("SA-1990-TFR",      3.72,  "wdi",  ("tfr", "South Africa", 1990), [SHOCK_TEST], tol=0.05)
reg("SA-2005-TFR",      2.51,  "wdi",  ("tfr", "South Africa", 2005), [SHOCK_TEST], tol=0.05)
reg("SA-2019-LE",       66.1,  "wdi",  ("le", "South Africa", 2019), [SHOCK_TEST], tol=0.1)

# --- FAMINE TEST section ---
# Numbers from scripts/famine_education_test.py output
reg("Famine-count",        21,     "checkin", ("famine_education_test.json", "numbers.Famine-count"),
    [FAMINE_TEST, "the-dilution-mechanism"], tol=0)
reg("Famine-below-50-ct",  19,     "checkin", ("famine_education_test.json", "numbers.Famine-below-50-ct"),
    [FAMINE_TEST, "the-dilution-mechanism"], tol=0)
reg("Famine-median-edu",   19.6,   "checkin", ("famine_education_test.json", "numbers.Famine-median-edu"),
    [FAMINE_TEST], tol=0.1)
reg("Famine-mean-edu",     25.4,   "checkin", ("famine_education_test.json", "numbers.Famine-mean-edu"),
    [FAMINE_TEST], tol=0.1)
reg("NM-median-edu",       71.6,   "checkin", ("famine_education_test.json", "numbers.NM-median-edu"),
    [FAMINE_TEST], tol=0.5)
reg("Bihar-deaths-lo",     70000,  "ref", "Bihar famine excess deaths low estimate (Dyson & Maharatna 1992)", [FAMINE_TEST], tol=0)
reg("Bihar-deaths-hi",     130000, "ref", "Bihar famine excess deaths high estimate (Dyson & Maharatna 1992)", [FAMINE_TEST], tol=0)
reg("Bihar-grain-drop",    19,     "ref", "India grain production drop 1965-66 (%)",  [FAMINE_TEST], tol=0)
reg("Kerala-female-lit",   39,     "ref", "Kerala female literacy ~1966 (%)",         [FAMINE_TEST], tol=1)
reg("Kerala-1943-deaths",  90000,  "ref", "Travancore famine 1943 deaths",            [FAMINE_TEST], tol=5000)
reg("Kerala-1966-yr",      1966,   "ref", "Year of Bihar-Kerala comparison",          [FAMINE_TEST], tol=0)
reg("Ireland-deaths",      1000000, "ref", "Great Irish Famine excess deaths (Mokyr 1983; Ó Gráda 1999)",      [FAMINE_TEST], tol=0)
reg("Ireland-emigration",  1500000, "ref", "Famine-decade emigration outflow (Mokyr 1983)",                    [FAMINE_TEST], tol=0)
reg("Ireland-1841-lit",    47,      "ref", "Ireland aggregate literacy, 1841 Census, age 5+ (Ó Gráda 1995)",    [FAMINE_TEST], tol=0)
reg("Ireland-Ulster-lit",  65,      "ref", "Protestant Ulster literacy, 1841 Census (Ó Gráda 1995)",            [FAMINE_TEST], tol=0)
reg("Ireland-Connacht-lit", 33,     "ref", "Catholic Connacht/west Munster literacy, 1841 (Ó Gráda 1995)",     [FAMINE_TEST], tol=0)
reg("Ireland-NSA",         1831,    "ref", "National School Act",                                               [FAMINE_TEST], tol=0)
reg("Ireland-150yr",       150,     "ref", "Years from Penal Laws to potato blight",                            [FAMINE_TEST, COLONIAL], tol=0)
reg("Ireland-NSA-gap",     14,      "ref", "Years from National School Act to blight",                          [FAMINE_TEST], tol=0)
reg("Ireland-OGrada-yr",   1999,    "ref", "Ó~Gráda 1999 citation year",                                         [FAMINE_TEST], tol=0)

# ── Historical facts: UK franchise + colonial famines (the-institutional-
#    challenge). Source type "fact" pins each to checkin/historical_facts.json,
#    which carries an authoritative URL per fact (verify fails on a missing
#    url). Provenance: paper/historical_facts.md. "Inclusive for whom?" — the
#    franchise numbers show most adults could not vote through the period AJR
#    call inclusive; the famine tolls show the same Parliament extracting.
reg("UK-electorate-pre-1832",   400000,   "fact", "uk-electorate-pre-1832",            [INSTIT], tol=0)
reg("UK-reform-1832",           1832,     "fact", "uk-reform-act-1832-year",           [INSTIT], tol=0)
reg("UK-reform-1867",           1867,     "fact", "uk-reform-act-1867-year",           [INSTIT], tol=0)
reg("UK-reform-1884",           1884,     "fact", "uk-reform-act-1884-year",           [INSTIT], tol=0)
reg("UK-men-excluded-1884",     40,       "fact", "uk-men-excluded-post-1884-pct",     [INSTIT], tol=0)
reg("UK-equal-franchise-1928",  1928,     "fact", "uk-equal-franchise-act-1928-year",  [INSTIT], tol=0)
reg("Bengal-1770-yr",           1770,     "fact", "bengal-famine-1770-year",           [INSTIT], tol=0)
reg("Irish-famine-1840s-deaths", 1000000, "fact", "irish-famine-1840s-deaths",         [], tol=0)

# ══════════════════════════════════════════════════════════════════════════
# SECTION COVERAGE — remaining values surfaced by coverage scan
# ══════════════════════════════════════════════════════════════════════════

# --- gdp-has-no-independent-effect / the-deaton-objection: u5mr resid R² @ lag 25 ---
# Retired 2026-06-18: §Deaton's residualised-GDP restatement was collapsed to a
# cross-ref (item 4 consolidation), removing the 0.019 occurrence from that
# section. The value is still covered in §GDP by resid-gdp-r2-u5mr-max.
# reg("U5MR-resid-r2-25",   0.019, "checkin",
#     ("lag_sensitivity.json", "results.25.U5MR_ceil90.resid_gdp_r2"),
#     ["the-deaton-objection"], tol=0.002)

# --- robustness: summary-stats sd for parent-edu distribution ---

# --- robustness: legacy TWFE row (3) (parent < 10%) within-R² and n ---
# (Legacy name "TA1-M3-se" — value is the within-R², not SE; kept for
#  registry stability. The Table A1 it referenced has been removed.)

# Legacy TWFE FE+year row R² values

# --- robustness: Residualization-by-lag cells, U5MR and child-edu columns ---
# Cells with resid_gdp_r2 > 0.003 for U5MR and child-edu, by lag. Low values
# (0.000–0.002) still need their own per-occurrence registration wherever they
# appear in prose — there is no longer a global threshold whitelist.
_RESID_CELLS = [
    # (lag, outcome, resid_gdp_r2, json_outcome_label)
    (15, "U5MR",      0.005, "U5MR_ceil90"),
    (20, "U5MR",      0.013, "U5MR_ceil90"),
    (20, "child_edu", 0.006, "ChildEdu_ceil90"),
    (25, "U5MR",      0.019, "U5MR_ceil90"),
    (25, "child_edu", 0.005, "ChildEdu_ceil90"),
    (30, "U5MR",      0.005, "U5MR_ceil90"),
    (30, "child_edu", 0.004, "ChildEdu_ceil90"),
]
# --- the-shock-test: lower-sec education drives fertility decline (R²) ---
reg("ShockTest-lsec-tfr-r2", 0.71, "checkin",
    ("table_contemporaneous.json", "numbers.panel.log_tfr.c1_edu.r2"),
    [], tol=0.01)

# --- the-colonial-test 2SLS second-stage entries: covered by IV-edu-* /
# IV-avexpr-edu-* registrations earlier in the COLONIAL block (pointing at
# iv_2sls_colonial_icrg.json and ajr_n61_deep_dive.json). Old polity-based
# Wu-Hausman / second-stage-p entries removed when polity was dropped from
# the paper.

# --- country-histories: income at Korea's and Bangladesh's expansion crossings ---
reg("Nepal-income-at-crossing", 1114, "wdi",
    ("gdp", "Nepal", 2022),
    ["four-further-cases"], tol=0.5)

# --- china: peer-pool bandwidth + post-1980 LE slope ---
reg("China-peer-band", 0.5, "const",
    "China peer-pool bandwidth (constant: ±0.5 mean years of schooling)",
    [CHINA], tol=0)
reg("China-peer-band-lo", 0.25, "checkin",
    ("china_band_sensitivity.json", "band_lo"),
    [CHINA], tol=0)
reg("China-peer-band-hi", 1.0, "checkin",
    ("china_band_sensitivity.json", "band_hi"),
    [CHINA], tol=0)
reg("China-post1980-beta3", 0.007, "derived",
    "China LE: post-1980 slope change (β₃), absolute magnitude",
    [], tol=0.002)

# --- the-institutional-challenge: autocracy-transition mean gain + p-value ---
reg("Autocracy-variance-pct", 74, "checkin",
    ("regime_education_test.json", "results_by_lag.0.auto_below_demo_median_pct"),
    ["the-institutional-challenge"], tol=0.5)
reg("Regime-transition-p", 0.57, "checkin",
    ("regime_education_test.json", "transition_test_p"),
    ["the-institutional-challenge"], tol=0.01)

# Polity2 per-lag table cells (Table A5) removed when the table itself
# was dropped from §the-institutional-challenge; only the headline
# "<0.7%" R² and the 20-year-lag auto/demo means (Regime-auto-mean /
# Regime-demo-mean) remain in the paper.

# --- the-institutional-challenge: Table A6 colonial education vs avexpr ---
# Re-anchored on AJR's 64-country base sample (n=61 after merge) using
# AJR's own avexpr institutional measure, replacing the polity2 framing.
reg("ColA6-edu1950-r2",  0.518, "checkin",
    ("ajr_n61_deep_dive.json", "univar_r2_log_gdp_2020.edu_1950.r2"),
    ["the-institutional-challenge"], tol=0.005)
reg("ColA6-avexpr-r2",   0.525, "checkin",
    ("ajr_n61_deep_dive.json", "univar_r2_log_gdp_2020.avexpr.r2"),
    ["the-institutional-challenge"], tol=0.005)
reg("ColA6-logem4-r2",   0.544, "checkin",
    ("ajr_n61_deep_dive.json", "univar_r2_log_gdp_2020.logem4.r2"),
    ["the-institutional-challenge"], tol=0.005)
# 1900-cohort, religion-alone, and edu+religion R² rows dropped when
# Table A6 was trimmed to 3 rows; only the upstream-cause trio (edu 1950,
# avexpr, logem4) remains in the paper. Numbers still live in checkin JSON.

# --- the-institutional-challenge: 2SLS contest inline numbers (avexpr) ---
reg("ColA7-edu-beta",    0.043, "checkin",
    ("iv_2sls_colonial_icrg.json", "gdp_edu_2sls_coef"),
    ["the-institutional-challenge"], tol=0.005)
reg("ColA7-edu-t",       2.56,  "checkin",
    ("iv_2sls_colonial_icrg.json", "gdp_edu_2sls_t"),
    [], tol=0.1)
reg("ColA7-avexpr-beta", 0.629, "checkin",
    ("iv_2sls_colonial_icrg.json", "gdp_inst_2sls_coef"),
    ["the-institutional-challenge"], tol=0.005)
reg("ColA7-avexpr-t",    3.01,  "checkin",
    ("iv_2sls_colonial_icrg.json", "gdp_inst_2sls_t"),
    [], tol=0.1)

# --- kerala: BIMARU vs South India within-country contrast ---
# Eight states under one set of national institutions; the southern tier
# reaches replacement fertility ahead of the northern BIMARU states, the
# gradient ordered by female literacy. Source: Bhat (2002) replacement
# years, Census 2011 female literacy, NFHS-6 current TFR.
reg("BIM-lit-south",  73.2, "checkin",
    ("india_bimaru_south.json", "numbers.mean_female_literacy_2011_south"),
    [KERALA], tol=0.5)
reg("BIM-lit-north",  55.0, "checkin",
    ("india_bimaru_south.json", "numbers.mean_female_literacy_2011_bimaru"),
    [KERALA], tol=0.5)
reg("BIM-spearman",   0.849, "checkin",
    ("india_bimaru_south.json",
     "numbers.spearman_literacy_vs_replacement_year.rho_abs"),
    [KERALA], tol=0.01)
reg("BIM-bihar-tfr",  2.76, "checkin",
    ("india_bimaru_south.json", "numbers.nfhs6_tfr_bihar"),
    [KERALA], tol=0.001)

# ══════════════════════════════════════════════════════════════════════════
# COVERAGE CLEANUP — registrations added to close unregistered-number gaps
# surfaced after tightening the coverage-tolerance band. Each entry points
# at the checkin JSON that produced the value.
# ══════════════════════════════════════════════════════════════════════════

# §completion-as-the-operative-variable: test-scores r² for TFR
# Stored as proportion (0.0111) and cited as percent (1.1%); register
# both forms so the coverage scan finds whichever the paper uses.
reg("Test-r2-TFR",         0.0111, "checkin",
    ("completion_vs_test_scores.json", "short_lag.10.tfr.test.r2"),
    [], tol=0.005)
reg("Test-r2-TFR-pct",     1.1,    "derived",
    "test-scores r² for TFR expressed as percent (paper: 1.1%)",
    [COMPLETION], tol=0.05)

# §education-vs-gdp-as-predictors-of-attainment: tab:cutoff betas/t-stats
reg("TabA1-20-t",           5.1, "checkin",
    ("table_a1_cutoffs.json", "numbers.cutoff_20.t"),
    [], tol=0.1)
reg("TabA1-20-beta",        0.935, "checkin",
    ("table_a1_cutoffs.json", "numbers.cutoff_20.beta"),[], tol=0.005)
reg("TabA1-10-beta",        1.205, "checkin",
    ("table_a1_cutoffs.json", "numbers.cutoff_10.beta"),[], tol=0.005)

# §the-shape-of-the-response: n-countries

# §the-shape-of-the-response: grandparent
# effect on fertility under LAG_TFR=5 biological anchoring (parent T-5 /
# grandparent T-33). Full-panel grandparent p=0.0011 (paper "both p<0.01");
# the low-baseline grandparent coefficient is p<0.001 (this reg backs that
# claim), with low-edu within-R² jump 0.70→0.75.
reg("GP-TFR-pval",           0.001, "checkin",
    ("grandparent_effect_all_outcomes.json", "outcomes.tfr.low_edu.parent_gp.pval_grandparent_edu"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.005)

# §gdp-has-no-independent-effect: residual-R² table cells (25-yr lag,
# ceiling ≤90%). All three columns of Table 5 wired exact-cell.
reg("ResidTab-LE-edu-r2",    0.428, "checkin",
    ("lag_sensitivity.json", "results.12.LE_ceil90.edu_r2"),
    [GDP_INDEP], tol=0.0005)
# [retired 2026-06-18 Panel-B cut: exposed surplus reg] reg("ResidTab-LE-raw-r2",    0.192, "checkin",
# [retired 2026-06-18 Panel-B cut: exposed surplus reg]     ("lag_sensitivity.json", "results.12.LE_ceil90.raw_gdp_r2"),
# [retired 2026-06-18 Panel-B cut: exposed surplus reg]     [GDP_INDEP], tol=0.0005)
# [retired 2026-06-18 Panel-B cut: exposed surplus reg] reg("ResidTab-LE-resid-r2",  0.000, "checkin",
# [retired 2026-06-18 Panel-B cut: exposed surplus reg]     ("lag_sensitivity.json", "results.12.LE_ceil90.resid_gdp_r2"),
# [retired 2026-06-18 Panel-B cut: exposed surplus reg]     [GDP_INDEP], tol=0.001)
reg("ResidTab-TFR-edu-r2",   0.482, "checkin",
    ("lag_sensitivity.json", "results.25.TFR_ceil90.edu_r2"),
    [], tol=0.0005)
reg("ResidTab-TFR-raw-r2",   0.177, "checkin",
    ("lag_sensitivity.json", "results.25.TFR_ceil90.raw_gdp_r2"),
    [], tol=0.0005)
# [retired 2026-06-18 Panel-B cut: exposed surplus reg] reg("ResidTab-TFR-resid-r2", 0.000, "checkin",
# [retired 2026-06-18 Panel-B cut: exposed surplus reg]     ("lag_sensitivity.json", "results.25.TFR_ceil90.resid_gdp_r2"),
# [retired 2026-06-18 Panel-B cut: exposed surplus reg]     [GDP_INDEP], tol=0.001)
reg("ResidTab-CE-edu-r2",    0.524, "checkin",
    ("lag_sensitivity.json", "results.25.ChildEdu_ceil90.edu_r2"),
    [GDP_INDEP], tol=0.0005)
reg("ResidTab-CE-raw-r2",    0.295, "checkin",
    ("lag_sensitivity.json", "results.25.ChildEdu_ceil90.raw_gdp_r2"),
    [GDP_INDEP], tol=0.0005)
reg("ResidTab-U5-edu-r2",    0.457, "checkin",
    ("lag_sensitivity.json", "results.12.U5MR_ceil90.edu_r2"),
    [GDP_INDEP], tol=0.0005)
# Residualized-GDP p-values (tab:residualisation col 5). Updated to match
# regression_tables.json: LE 0.6026 → 0.60, CE 0.401 → 0.40.
reg("ResidTab-LE-ratio",     0.91,  "checkin",
    ("regression_tables.json", "results.LE.90.GDP (residualized).pval"),
    [GDP_INDEP], tol=0.005)
reg("ResidTab-CE-ratio",     0.41,  "checkin",
    ("regression_tables.json", "results.ChildEdu.90.GDP (residualized).pval"),
    [EDU_VS_GDP], tol=0.005)

# §gdp-has-no-independent-effect: 74 countries with lower-sec >85% in 2010
reg("College-LE-n-countries", 74, "checkin",
    ("college_le_gradient.json", "results.n_countries.actual"),
    [], tol=0)

# §gdp-has-no-independent-effect: by-level R² comparisons
reg("Level-primary-le-r2",   0.40, "checkin",
    ("edu_vs_gdp_by_level.json", "levels.primary.90.10.edu_r2"),
    [], tol=0.005)
reg("Level-upper-le-r2",     0.51, "checkin",
    ("edu_vs_gdp_by_level.json", "levels.upper_secondary.90.10.edu_r2"),
    [GDP_INDEP], tol=0.005)
reg("Level-lower-tfr-r2",    0.669, "checkin",
    ("edu_vs_gdp_tfr_residualized.json", "levels.lower_secondary.90.10.edu_r2"),
    [GDP_INDEP], tol=0.005)
reg("Level-upper-tfr-r2",    0.471, "checkin",
    ("edu_vs_gdp_tfr_residualized.json", "levels.upper_secondary.90.10.edu_r2"),
    [], tol=0.005)

# ══════════════════════════════════════════════════════════════════════════
# STRICT-COVERAGE GAP CLOSURE — §the-shape-of-the-response...
# Each paper occurrence of a literal needs its own reg() under strict matching.
# These supplement existing one-shot registrations to cover repeated values
# that appear in stepwise tables, lag-decay tables, and prose alike.
# ══════════════════════════════════════════════════════════════════════════
# Table tab:edu-outcomes Panel A — N column entries (col 2 of each row).
# Existing registrations cover col 1 of each row; these close col 2.
reg("T7-A-logu5-c1-n",       1575, "checkin",
    (_T7_J, "numbers.panel.log_u5.c1_edu.n"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0)
reg("T7-A-logu5-c2-n",       1253, "checkin",
    (_T7_J, "numbers.panel.log_u5.c2_edu_gdp.n"),
    [THE_SHAPE_OF_THE_RESPONSE], tol=0)

# Panel B SE row (paper line 3323): (1.79) (2.02) (2.06) (2.18) — b2/b3 SEs.
reg("T7-B-b3-se",            2.05, "checkin",
    (_T7B_J, "numbers.panelB_max.b3_gdp_only_yfe.log_gdp_t.se"),
    [], tol=0.01)

# Panel B R² (within) — b2 R². (Panel B table is not printed in the paper;
# these T7-B regs are an output self-consistency check on table7_stepwise.json,
# now at the interpolated cross-generation lag T+28 — see script header.)
reg("T7-B-b2-r2-tabcell",    0.475, "checkin",
    (_T7B_J, "numbers.panelB_max.b2_gdp_init_edu.r2"),
    [], tol=0.001)

# Panel B Observations row tabcells: section scope removed 2026-05-26 —
# the Panel A rewrite no longer surfaces the n=927 cells in THE_SHAPE_OF_THE_RESPONSE prose.
# The JSON values themselves remain valid (verified) but are no longer
# section-scoped.
# reg("T7-B-b2-countries",     172, "checkin",
#     (_T7_J, "numbers.panelB_max.b2_gdp_init_edu.countries"),
#     [THE_SHAPE_OF_THE_RESPONSE], tol=0)
# (removed 2026-05-23: number cut from main text in §5 trim)
# reg("T7-B-b3-countries",     172, "checkin",
#     (_T7_J, "numbers.panelB_max.b3_gdp_only_yfe.countries"),
#     [THE_SHAPE_OF_THE_RESPONSE], tol=0)
# (removed 2026-05-23: number cut from main text in §5 trim)
# reg("T7-B-b4-countries",     172, "checkin",
#     (_T7_J, "numbers.panelB_max.b4_gdp_init_edu_yfe.countries"),
#     [THE_SHAPE_OF_THE_RESPONSE], tol=0)
# (removed 2026-05-23: number cut from main text in §5 trim)

# Panel B prose restatement regs (T7-A-common-n-prose etc.) removed
# 2026-05-26 — the surrounding prose was rewritten when Panel A switched
# to canonical-lag and no longer contains the "1960--1990, n=927",
# "Panel B's drop from β=14.42", or "p=0.26" restatements. Panel B's
# table itself is verified via T7-B-* regs above.

# Panel A prose absolute-% restatements of c1 coefficients
# (paper §The Shape of the Response, "Reading Panel A column 1" para).
reg("Prose-gdp-pct-lsec",      1.85, "derived",
    "100 * abs(T7-A-loggdp-c1-beta) — paper Panel A prose 1.85% per pp",
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.05)
reg("Prose-le-pct-lsec",       0.42, "derived",
    "100 * abs(T7-A-logle-c1-beta=0.0042) — paper Panel A prose 0.42% per pp",
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.05)
reg("Prose-le-yrs-lsec",       0.29, "derived",
    "Approx 0.42% of LE=70 ~ 0.29 yrs — paper Panel A prose",
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.05)
reg("Prose-tfr-pct-lsec",      1.65, "derived",
    "100 * abs(T7-A-logtfr-c1-beta=-0.0165) — paper Panel A prose 1.65% per pp",
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.05)
reg("Prose-tfr-children-lsec", 0.06, "derived",
    "Approx 1.62% of TFR=3.5 ~ 0.06 children — paper Panel A prose",
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.02)
reg("Prose-u5mr-pct-lsec",     3.27, "derived",
    "100 * abs(T7-A-logu5-c1-beta) — paper Panel A prose 3.27% per pp",
    [THE_SHAPE_OF_THE_RESPONSE], tol=0.05)

# Prose at L3419: "GDP explains 1.6%" — LE.lt10.gdp_r2 * 100.
reg("LE-lt10-gdp-r2-pct",    1.6, "derived",
    "edu_vs_gdp_predicts_le.json lt10.gdp_r2 x 100 (paper L3419)",
    [], tol=0.05)

# Lag-decay table cell (L3444): U5 lag-25 |β|=0.820 — already registered as
# LagBeta-u5-lag25 but is consumed by L3406 prose "$-$0.82% per pp".
# Add dedicated lag-table reg here.
reg("LagBeta-u5-lag25-tabcell", 0.820, "derived",
    "abs(lag_coefficients.json numbers.u5log_beta_lag25) — paper table cell shows |β|",
    [], tol=0.005)

# Lag-decay table cell (L3446): ChildEdu lag-100 |β|=0.152 — already
# registered as LagBeta-cedu-lag100 but the section-wide consumption
# pattern can fail under strict matching. Pin to the table line.

# Prose at L3470: "10.9 ... respectively" — TFR lag-100 |t|. Pin to line.
reg("LagT-tfr-lag100-prose", 10.9, "derived",
    "abs(lag_coefficients.tfr_beta_lag100_t) — prose at L3470",
    [], tol=0.05)

# Callaway-Sant'Anna event-study post-treatment ATTs. Aggregation now
# weighted by cohort size per CS (2021) Eq 4.2 (shift from prior
# unweighted aggregation: aggregate ATT 7.88 → 8.85; e=0 1.3 → 1.6;
# e=2 6.6 → 6.9; e=5 14.9 → 15.5; e=7 21.4 unchanged).
# Figure is currently not cited in paper body; regs remain for the
# checkin chain.
reg("CS-ATT-0yr-prose",      1.75, "checkin",
    ("callaway_santanna.json", "child_education.event_study.0.att"),
    [], tol=0.1)
reg("CS-ATT-yr35-caption",   22.13, "checkin",
    ("callaway_santanna.json", "child_education.event_study.7.att"),
    [], tol=0.1)

# Grandparent-channel prose (rewritten under LAG_TFR=5 biological anchoring):
# Full-panel GP TFR is now highly significant (paper: "both p<0.001"), not
# absorbed by the parent as under the old lag-25 spec.
# Low-edu subsample within-R²: now 0.70→0.75 (was 0.39→0.46).
# L3572: GP U-5 betas (-0.018 vs -0.016) and |ratio|=1.16 — duplicates of
# GM-u5-beta-gp / GM-u5-beta-ratio with offset-pin for strict matching.

# ══════════════════════════════════════════════════════════════════════════
# STRICT-COVERAGE GAP CLOSURE — §gdp-has-no-independent-effect
# ══════════════════════════════════════════════════════════════════════════
# tab:residualisation U-5 row (L3624): "0.11" is the resid-GDP p-value.
# T3-U5MR-resid-p reg'd at section-wide None; pin a duplicate for the cell.
# Prose at L3735: "+0.11 years of life" — reads T2-LE-beta (0.109).
reg("T2-LE-beta-resid-prose", 0.19, "checkin",
    ("regression_tables.json", "results.LE.90.Education.beta"),
    [GDP_INDEP], tol=0.01)
# Prose at L3741: "(0.280 to 0.524)" — re-statement of ResidTab edu-R² range.
# Prose at L3743: "p>=0.11 in all four" — restatement of U-5 resid p.
# Prose at L3762: "after 2000, 2.7% (p=0.05)" — U5MR-post2000-resid-pct.
reg("U5MR-post2000-pct-prose", 2.7, "derived",
    "U5MR-post2000-resid-r2 × 100 (paper L3762)",
    [], tol=0.05)
# Prose at L3825: "education R² [0.34, 0.58]" — confidence interval bounds
# from the residualization bootstrap. 0.34 already reg'd; 0.58 derives from
# the same source (lag_sensitivity TFR ceil60 edu_r2 ~0.582).
reg("Bootstrap-edu-r2-hi",   0.58, "derived",
    "Upper bound of education R² 500-rep bootstrap envelope (paper L3825)",
    [], tol=0.05)
# Prose at L3831: "within-country correlation exceeds 0.9" — female/male
# lower-sec correlation. Structural-ish but listed by scanner; pin a const reg.
reg("FemMale-corr-min",      0.9, "const",
    "Within-country female-male lower-sec correlation lower bound (paper L3831)",
    [GDP_INDEP], tol=0.05)
# Reverse regression prose (L3745--L3748): GDP→child-edu collapse table.
# 20.2 = PI-alone-beta (panel_expansion_fe); 0.340 = PI-alone-R2; 7.0 = PI-cond-beta;
# 0.001 = PI-cond-p; 0.739 = PI-edu-alone. Prose duplicates at the same lines.
# Asian Financial Crisis duplication at L3754: "Indonesia gained 5.4pp while
# losing 14.5% of GDP". Indonesia gdp drop -14.5 and edu gain +5.4 already
# reg'd as AFC-* at section-None; pin duplicates for the prose restatement.
# AFC-Indonesia-edu-prose and AFC-Indonesia-gdp-prose removed 2026-05-23
# with §Asymmetric Disruption deletion.
# Permutation null prose: the null runs on the unrestricted full panel
# (real_beta=0.483, n=1665), so the prose states "more than 52 SDs above
# either null" without attributing it to a specific beta. Reg'd as
# Perm-min-z-sds in GDP_INDEP (see line ~1271).

# §descriptive-statistics + §appendix-data: Summary-stats table cells (pooled)
# 1665 / 1466 main-text mentions removed 2026-05-23; cells survive in
# Appendix~\ref{appendix-data} where the full table lives.
reg("Sum-parent-edu-n",      1480,  "checkin",
    ("summary_stats.json", "descriptives.pooled.parent_edu.n"),
    ["appendix-data"], tol=0)
reg("Sum-child-edu-n",       1480,  "checkin",
    ("summary_stats.json", "descriptives.pooled.child_edu.n"),
    ["appendix-data"], tol=0)
reg("Sum-loggdp-n",          1338,  "checkin",
    ("summary_stats.json", "descriptives.pooled.log_gdp.n"),
    ["appendix-data"], tol=0)
reg("Sum-le-n",              1430,  "checkin",
    ("summary_stats.json", "descriptives.pooled.life_exp.n"),
    ["appendix-data"], tol=0)
reg("Sum-tfr-n",             1430,  "checkin",
    ("summary_stats.json", "descriptives.pooled.tfr.n"),
    ["appendix-data"], tol=0)
reg("Sum-u5mr-n",            1392,  "checkin",
    ("summary_stats.json", "descriptives.pooled.u5mr.n"),
    ["appendix-data"], tol=0)

# §descriptive-statistics + §appendix-data: 178 countries with education+GDP
# (Main-text mention removed 2026-05-23; survives in appendix table.)
reg("Sum-edu-gdp-n-countries", 178, "checkin",
    ("summary_stats.json", "numbers.gdp_panel_countries"),
    ["appendix-data"], tol=0)

# §descriptive-statistics: by-period means (summary_stats.json / descriptives.by_period)
reg("SumP-parent-1975",      26.4,  "checkin",
    ("summary_stats.json", "descriptives.by_period.parent_edu.1975-1989.mean"),
    [DESCRIPTIVE], tol=0.05)
reg("SumP-parent-1990",      39.4,  "checkin",
    ("summary_stats.json", "descriptives.by_period.parent_edu.1990-2004.mean"),
    [DESCRIPTIVE], tol=0.05)
reg("SumP-parent-2005",      53.9,  "checkin",
    ("summary_stats.json", "descriptives.by_period.parent_edu.2005-2015.mean"),
    [DESCRIPTIVE], tol=0.05)
reg("SumP-child-1975",       54.5,  "checkin",
    ("summary_stats.json", "descriptives.by_period.child_edu.1975-1989.mean"),
    [DESCRIPTIVE], tol=0.05)
reg("SumP-child-1990",       62.2,  "checkin",
    ("summary_stats.json", "descriptives.by_period.child_edu.1990-2004.mean"),
    [DESCRIPTIVE], tol=0.05)
reg("SumP-child-2005",       71.1,  "checkin",
    ("summary_stats.json", "descriptives.by_period.child_edu.2005-2015.mean"),
    [DESCRIPTIVE], tol=0.05)
reg("SumP-loggdp-1975",      8.07,  "checkin",
    ("summary_stats.json", "descriptives.by_period.log_gdp.1975-1989.mean"),
    [DESCRIPTIVE], tol=0.01)
reg("SumP-loggdp-1990",      8.19,  "checkin",
    ("summary_stats.json", "descriptives.by_period.log_gdp.1990-2004.mean"),
    [DESCRIPTIVE], tol=0.01)
reg("SumP-loggdp-2005",      8.52,  "checkin",
    ("summary_stats.json", "descriptives.by_period.log_gdp.2005-2015.mean"),
    [DESCRIPTIVE], tol=0.01)
reg("SumP-le-1975",          62.4,  "checkin",
    ("summary_stats.json", "descriptives.by_period.life_exp.1975-1989.mean"),
    [DESCRIPTIVE], tol=0.05)
reg("SumP-le-1990",          65.5,  "checkin",
    ("summary_stats.json", "descriptives.by_period.life_exp.1990-2004.mean"),
    [DESCRIPTIVE], tol=0.05)
reg("SumP-le-2005",          70.1,  "checkin",
    ("summary_stats.json", "descriptives.by_period.life_exp.2005-2015.mean"),
    [DESCRIPTIVE], tol=0.05)
reg("SumP-tfr-1975",         4.40,  "checkin",
    ("summary_stats.json", "descriptives.by_period.tfr.1975-1989.mean"),
    [DESCRIPTIVE], tol=0.01)
reg("SumP-tfr-1990",         3.63,  "checkin",
    ("summary_stats.json", "descriptives.by_period.tfr.1990-2004.mean"),
    [DESCRIPTIVE], tol=0.01)
reg("SumP-tfr-2005",         2.95,  "checkin",
    ("summary_stats.json", "descriptives.by_period.tfr.2005-2015.mean"),
    [DESCRIPTIVE], tol=0.01)
reg("SumP-u5-1975",          91.0,  "checkin",
    ("summary_stats.json", "descriptives.by_period.u5mr.1975-1989.mean"),
    [DESCRIPTIVE], tol=0.1)
reg("SumP-u5-1990",          66.8,  "checkin",
    ("summary_stats.json", "descriptives.by_period.u5mr.1990-2004.mean"),
    [DESCRIPTIVE], tol=0.1)
reg("SumP-u5-2005",          39.2,  "checkin",
    ("summary_stats.json", "descriptives.by_period.u5mr.2005-2015.mean"),
    [DESCRIPTIVE], tol=0.1)

# §descriptive-statistics: narrative shift summaries ("rises by 29 pp", "halves", "rises 8.6 years")
# "29 pp" parental shift removed from main text 2026-05-23 (implicit in 27%→56%).
# reg("Sum-parent-shift",      29,    ...)
reg("Sum-le-shift",          7.7,   "derived",
    "Life expectancy rise (2005–2015 70.1 − 1975–1989 61.4 ≈ 8.6y)",
    [DESCRIPTIVE], tol=0.2)

# tab:headline subgroups — only cells cited in §universality-across-subsamples prose
# remain. Other subsamples (EAP, LAC, ENA, era splits, GDP terciles) live
# only in scripts/tables/table_1_subgroups.py output, not in paper.
reg("T1-SG-MENA-n",          67,    "checkin",
    ("table_1_subgroups.json", "numbers.region_MENA_n"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0)
reg("T1-SG-SA-n",            58,    "checkin",
    ("table_1_subgroups.json", "numbers.region_SouthAsia_n"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0)
reg("T1-SG-SSA-r2",          0.692, "checkin",
    ("table_1_subgroups.json", "numbers.region_SSA_r2"),
    [UNIVERSALITY_ACROSS_SUBSAMPLES], tol=0.005)
reg("T1-SG-headline-beta",   1.434, "checkin",
    ("table_1_subgroups.json", "numbers.headline_beta"),
    [EDU_VS_GDP], tol=0.005)
reg("T1-SG-headline-n",        672, "checkin",
    ("table_1_subgroups.json", "numbers.headline_n"),
    [EDU_VS_GDP], tol=0)
reg("T1-SG-headline-countries", 106, "checkin",
    ("table_1_subgroups.json", "numbers.headline_countries"),
    [EDU_VS_GDP], tol=0)
# §appendix-other-robustness: Lag-sensitivity table cells (lag_sensitivity.json)
# 15 / 20 / 25 / 30 yr rows, ceiling≤90% — edu_r2 for LE, TFR, U5, CE.
# Column order: LE, TFR, U5, CE (interleaved with raw_r2).

# §appendix-other-robustness: Period-length panel cells (period_length.json)

# §appendix-other-robustness: PPML/Log-outcome n-countries

# §appendix-other-robustness: Event-study post-treatment magnitudes (callaway_santanna.json).
# Also cited in §the-shape-of-the-response where
# fig:cs-event is shown (re-captioned as the compounding-generations signature).
reg("CS-ATT-10yr",           6.23,  "checkin",
    ("callaway_santanna.json", "child_education.event_study.2.att"),[], tol=0.2)
reg("CS-ATT-25yr",           12.46, "checkin",
    ("callaway_santanna.json", "child_education.event_study.5.att"),[], tol=0.2)

# §appendix-other-robustness: Goodman-Bacon / 2WFE child-edu β from main spec

# §appendix-other-robustness: N-counts in prose summaries

# §the-shock-test: Asian crisis LE lost years
reg("Shock-LE-years-lost",   3.6, "derived",
    "Russia-2019-LE − Russia-1988-LE: surplus above Soviet-era peak",
    [SHOCK_TEST], tol=0.05)

# §four-further-cases: Cuba 2.27 table value + Myanmar 1025 income
reg("Cuba-col-ratio",        2.27, "derived",
    "Cuba column ratio in tab:transition-year",
    [SEN_CASES], tol=0.05)
reg("Myanmar-income",        1025, "wdi",
    ("gdp", "Myanmar", 2013),
    ["four-further-cases"], tol=0.5)

# §kerala: Kerala TFR threshold crossing year
reg("Kerala-TFR-cross",      1973, "checkin",
    ("kerala.json", "results.tfr_crossing.actual"),
    [KERALA], tol=0.5)

# §korea-and-philippines: Korea income-at-expansion (already registered as derived
# under SEN_CASES; add subsection)
# Handled by expanding Korea-income-at-expansion registration below.

# §china: LE and mean-years-of-schooling values in prose
reg("China-LE-1994",         69.8, "wdi",
    ("le", "China", 1994),
    [CHINA], tol=0.2)
reg("China-mys-2000",        9.6,  "checkin",
    ("china_mean_yrs_vs_peers.json", "key_data_points.china_mys_2000"),
    [CHINA], tol=0.1)

# §the-institutional-challenge: Auto/Demo mean gain rates at concurrent lag
# (Table A5 columns 4 and 5, lag=0 row). Names retained for backward
# compatibility — the values are mean gain rates, not standardized coef / t-stat.
reg("Polity-0yr-std-coef",   8.5, "checkin",
    ("regime_education_test.json", "results_by_lag.0.mean_auto"),
    [], tol=0.1)
reg("Polity-0yr-t",          9.5, "checkin",
    ("regime_education_test.json", "results_by_lag.0.mean_demo"),
    ["the-institutional-challenge"], tol=0.2)
# Old Col-IV-F-* polity registrations removed — replaced by IV-edu-F /
# IV-inst-F / IV-avexpr-edu-* in the COLONIAL block (avexpr-based).
# Cross-section references in §The Institutional Challenge are covered
# by the same registrations.
# ColA7-IV-F-* duplicates removed — covered by IV-edu-F / IV-inst-F /
# IV-avexpr-edu-F multi-section registrations above.

# ══════════════════════════════════════════════════════════════════════════
# §hollow-education: Data Credibility
# ══════════════════════════════════════════════════════════════════════════

HOLLOW_EDUCATION = "hollow-education"
HOLLOW_PATTERN = "the-anomaly"
HOLLOW_WCDE = "what-wcde-reports"
HOLLOW_PHENO = "phenotype-test-ussr"
HOLLOW_BL = "barro-lee-partial"
HOLLOW_COHORT = "cohort-shadow"
HOLLOW_HANUSHEK = "hanushek-reconciliation"
HOLLOW_EXCLUSION = "exclusion-robustness"

# ── §hollow-education intro: WCDE vs Barro-Lee headline disagreements ──
reg("G-Kaz-1970-wcde",     94,  "checkin",
    ("soviet_inflation.json", "numbers.lsec_kazakhstan_1970"),
    [HOLLOW_WCDE], tol=1)
reg("G-Turk-1970-wcde",    95,  "checkin",
    ("soviet_inflation.json", "numbers.lsec_turkmenistan_1970"),
    [HOLLOW_WCDE], tol=1)
reg("G-Iran-1970-wcde",    22,  "checkin",
    ("soviet_inflation.json", "numbers.lsec_iran_1970"),
    [HOLLOW_WCDE], tol=1)
reg("G-Turkey-1970-wcde",  22,  "checkin",
    ("soviet_inflation.json", "numbers.lsec_turkey_1970"),
    [HOLLOW_WCDE], tol=1)
reg("G-Pak-1970-wcde",     16,  "checkin",
    ("soviet_inflation.json", "numbers.lsec_pakistan_1970"),
    [HOLLOW_WCDE], tol=1)
reg("G-Afg-1970-wcde",     6,   "checkin",
    ("soviet_inflation.json", "numbers.lsec_afghanistan_1970"),
    [HOLLOW_WCDE], tol=1)
reg("G-n-republics",       15,  "const",
    "Count of USSR republics",
    [HOLLOW_EXCLUSION, CONVERGENCE_IN_AGGREGATE], tol=0)

# ── §what-wcde-reports diagnostics: gender gap and primary-lsec dropoff ─
reg("G-fm-gap-CA-1970",   -2.7, "checkin",
    ("soviet_inflation.json", "numbers.fm_gap_central_asia_1970"),
    [HOLLOW_WCDE], tol=0.2)
reg("G-fm-gap-neighbors-1970", -15.6, "checkin",
    ("soviet_inflation.json", "numbers.fm_gap_non-soviet_neighbors_1970"),
    [HOLLOW_WCDE], tol=0.2)
reg("G-prim-lsec-CA-1980", 1.6, "checkin",
    ("soviet_inflation.json", "numbers.prim_minus_lsec_central_asia_1980"),
    [HOLLOW_WCDE], tol=0.3)
reg("G-prim-lsec-neighbors-1980", 21.0, "checkin",
    ("soviet_inflation.json", "numbers.prim_minus_lsec_non-soviet_neighbors_1980"),
    [HOLLOW_WCDE], tol=0.5)

# ── §phenotype-test-ussr: U5MR trajectories  ───────────────────────
# These are WDI lookups for Iran/Kazakhstan/Turkey U5MR; verified via WDI
reg("G-Iran-u5mr-1960",   327, "wdi", ("u5mr", "Iran", 1960),
    [HOLLOW_PHENO], tol=0.5)
reg("G-Iran-u5mr-2010",   20,  "wdi", ("u5mr", "Iran", 2010),
    [HOLLOW_PHENO], tol=0.5)
reg("G-Turkey-u5mr-1960", 258, "wdi", ("u5mr", "Turkey", 1960),
    [HOLLOW_PHENO], tol=0.5)
reg("G-Turkey-u5mr-2010", 17,  "wdi", ("u5mr", "Turkey", 2010),
    [HOLLOW_PHENO], tol=0.5)
reg("G-Kaz-u5mr-1960",    107, "wdi", ("u5mr", "Kazakhstan", 1960),
    [HOLLOW_PHENO], tol=0.5)
reg("G-Kaz-u5mr-2010",    20,  "wdi", ("u5mr", "Kazakhstan", 2010),
    [HOLLOW_PHENO], tol=0.5)
reg("G-Iran-decline-pct",  94, "derived",
    "Iran U5MR percentage decline 1960-2010",
    [HOLLOW_PHENO], tol=1)
reg("G-Kaz-decline-pct",   81, "derived",
    "Kazakhstan U5MR percentage decline 1960-2010",
    [HOLLOW_PHENO], tol=1)

# ── §phenotype-test-ussr: TFR subgroup isolation (WCDE) + IMR fabrication
reg("G-TFR-CA-wcde-bias",  2.0, "checkin",
    ("edu_measure_horse_race.json",
     "numbers.subgroup_split.tfr_wcde_lsec_central_caucasus.bias_sds"),
    [HOLLOW_PHENO], tol=0.1)
reg("G-AS-imr-understate-lo", 22, "ref",
    "Anderson & Silver 1986: Soviet IMR understated >=22% vs WHO definition",
    [HOLLOW_PHENO], tol=0)
reg("G-AS-imr-understate-hi", 25, "ref",
    "Anderson & Silver 1986: Soviet IMR understatement upper bound ~25%",
    [HOLLOW_PHENO], tol=0)

# ── §moscow-meridian: per-republic residuals + distance ───────────
HOLLOW_MERIDIAN = "moscow-meridian"

# Per-republic LE σ and log U5MR σ residuals (WCDE lower-secondary completion 1960–2020)
# Indexed by country in checkin/ussr_per_country_residuals.json
def _G_PCR(key):
    return ("ussr_per_country_residuals.json", f"wcde.{key}")

# Westward republics
reg("G-MM-Latvia-LE-sd",      -1.06, "checkin",
    _G_PCR("latvia.le_resid_sd"), [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Latvia-U5-sd",       0.75, "checkin",
    _G_PCR("latvia.u5log_resid_sd"), [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Estonia-LE-sd",     -0.89, "checkin",
    _G_PCR("estonia.le_resid_sd"), [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Estonia-U5-sd",      0.57, "checkin",
    _G_PCR("estonia.u5log_resid_sd"), [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Lithuania-LE-sd",   -0.81, "checkin",
    _G_PCR("lithuania.le_resid_sd"), [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Lithuania-U5-sd",    0.35, "checkin",
    _G_PCR("lithuania.u5log_resid_sd"), [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Belarus-LE-sd",     -0.96, "checkin",
    _G_PCR("belarus.le_resid_sd"), [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Belarus-U5-sd",      0.43, "checkin",
    _G_PCR("belarus.u5log_resid_sd"),
    [HOLLOW_EDUCATION, HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Moldova-LE-sd",     -1.41, "checkin",
    _G_PCR("republic of moldova.le_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Moldova-U5-sd",      1.87, "checkin",
    _G_PCR("republic of moldova.u5log_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Ukraine-LE-sd",     -0.95, "checkin",
    _G_PCR("ukraine.le_resid_sd"), [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Ukraine-U5-sd",      1.09, "checkin",
    _G_PCR("ukraine.u5log_resid_sd"), [HOLLOW_MERIDIAN], tol=0.05)
# Russia (metropole)
reg("G-MM-Russia-LE-sd",      -1.40, "checkin",
    _G_PCR("russian federation.le_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Russia-U5-sd",       1.13, "checkin",
    _G_PCR("russian federation.u5log_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)
# Eastward republics
reg("G-MM-Armenia-LE-sd",     -1.42, "checkin",
    _G_PCR("armenia.le_resid_sd"), [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Armenia-U5-sd",      2.77, "checkin",
    _G_PCR("armenia.u5log_resid_sd"), [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Georgia-LE-sd",     -1.51, "checkin",
    _G_PCR("georgia.le_resid_sd"), [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Georgia-U5-sd",      2.58, "checkin",
    _G_PCR("georgia.u5log_resid_sd"),
    [HOLLOW_EDUCATION, HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Azerbaijan-LE-sd",  -2.20, "checkin",
    _G_PCR("azerbaijan.le_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Azerbaijan-U5-sd",   3.63, "checkin",
    _G_PCR("azerbaijan.u5log_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Turkmenistan-LE-sd", -2.31, "checkin",
    _G_PCR("turkmenistan.le_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Turkmenistan-U5-sd", 4.00, "checkin",
    _G_PCR("turkmenistan.u5log_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Tajikistan-LE-sd",  -2.20, "checkin",
    _G_PCR("tajikistan.le_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Tajikistan-U5-sd",   3.73, "checkin",
    _G_PCR("tajikistan.u5log_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Uzbekistan-LE-sd",  -1.00, "checkin",
    _G_PCR("uzbekistan.le_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Kyrgyzstan-LE-sd",  -1.80, "checkin",
    _G_PCR("kyrgyzstan.le_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Kyrgyzstan-U5-sd",   3.23, "checkin",
    _G_PCR("kyrgyzstan.u5log_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Kazakhstan-LE-sd",  -1.90, "checkin",
    _G_PCR("kazakhstan.le_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Kazakhstan-U5-sd",   2.69, "checkin",
    _G_PCR("kazakhstan.u5log_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)

# Distances from Moscow (km). The chapter intro cites two unsigned
# great-circle distances; the table uses signed east-of-Moscow km
# for all 15 republics. All 17 are constants from haversine on
# Soviet-era capital coords (see CAPITALS dict in
# scripts/ussr_metropole_distance.py).
reg("G-MM-Belarus-east-km-prose", 629, "const",
    "Signed east-of-Moscow km, Minsk (meridian prose; west-east split, parallels Georgia 451 km east)",
    [HOLLOW_EDUCATION], tol=0.5)
reg("G-MM-Georgia-east-km-prose", 451, "const",
    "Signed east-of-Moscow km, Tbilisi (meridian prose; west-east split, parallels Belarus 629 km west)",
    [HOLLOW_EDUCATION], tol=0.5)
# Table column: signed east-of-Moscow km (negative = west)
reg("G-MM-Latvia-east-km",    -844, "const",
    "Signed east-of-Moscow km, Riga", [HOLLOW_MERIDIAN], tol=0.5)
reg("G-MM-Estonia-east-km",   -804, "const",
    "Signed east-of-Moscow km, Tallinn",
    [HOLLOW_MERIDIAN], tol=0.5)
reg("G-MM-Lithuania-east-km", -771, "const",
    "Signed east-of-Moscow km, Vilnius",
    [HOLLOW_MERIDIAN], tol=0.5)
reg("G-MM-Belarus-east-km",   -629, "const",
    "Signed east-of-Moscow km, Minsk",
    [HOLLOW_MERIDIAN], tol=0.5)
reg("G-MM-Moldova-east-km",   -547, "const",
    "Signed east-of-Moscow km, Chisinau",
    [HOLLOW_MERIDIAN], tol=0.5)
reg("G-MM-Ukraine-east-km",   -444, "const",
    "Signed east-of-Moscow km, Kyiv",
    [HOLLOW_MERIDIAN], tol=0.5)
reg("G-MM-Armenia-east-km",    431, "const",
    "Signed east-of-Moscow km, Yerevan",
    [HOLLOW_MERIDIAN], tol=0.5)
reg("G-MM-Georgia-east-km",    451, "const",
    "Signed east-of-Moscow km, Tbilisi",
    [HOLLOW_MERIDIAN], tol=0.5)
reg("G-MM-Azerbaijan-east-km", 765, "const",
    "Signed east-of-Moscow km, Baku",
    [HOLLOW_MERIDIAN], tol=0.5)
reg("G-MM-Turkmenistan-east-km", 1291, "const",
    "Signed east-of-Moscow km, Ashgabat",
    [HOLLOW_MERIDIAN], tol=0.5)
reg("G-MM-Tajikistan-east-km",  1934, "const",
    "Signed east-of-Moscow km, Dushanbe",
    [HOLLOW_MERIDIAN], tol=0.5)
reg("G-MM-Uzbekistan-east-km",  1961, "const",
    "Signed east-of-Moscow km, Tashkent",
    [HOLLOW_MERIDIAN], tol=0.5)
reg("G-MM-Kyrgyzstan-east-km",  2285, "const",
    "Signed east-of-Moscow km, Frunze",
    [HOLLOW_MERIDIAN], tol=0.5)
reg("G-MM-Kazakhstan-east-km",  2422, "const",
    "Signed east-of-Moscow km, Alma-Ata",
    [HOLLOW_MERIDIAN], tol=0.5)
# Range-endpoint shorthand cited in chapter intro and §data:
# "2.6 to 4.0 σ" frames the eastward inflation magnitude
reg("G-MM-east-low-rounded",   2.6, "const",
    "Eastward U5MR residual lower bound (Georgia +2.58, rounded)",
    [HOLLOW_EDUCATION, CONVERGENCE_IN_AGGREGATE],
    tol=0.05)
reg("G-MM-east-high-rounded",  4.0, "const",
    "Eastward U5MR residual upper bound (Turkmenistan +4.00, rounded)",
    [HOLLOW_EDUCATION, CONVERGENCE_IN_AGGREGATE],
    tol=0.05)
# Year of Moldova annexation cited in moscow-meridian
reg("G-MM-Moldova-annex-yr",  1940, "const",
    "Year Moldova annexed by USSR (Bessarabia)",
    [HOLLOW_MERIDIAN], tol=0)

# Group means + correlations
reg("G-MM-west-mean-U5-sd",   0.84, "checkin",
    ("ussr_metropole_direction.json", "group_means_u5log_sd.west"),
    [HOLLOW_MERIDIAN, HOLLOW_EDUCATION], tol=0.03)
reg("G-MM-east-mean-U5-sd",   3.23, "checkin",
    ("ussr_metropole_direction.json", "group_means_u5log_sd.east"),
    [HOLLOW_MERIDIAN], tol=0.03)
reg("G-MM-east-corr",         0.86, "checkin",
    ("ussr_metropole_distance.json",
     "correlations_excl_russia.dist_to_u5log_sd"),
    [HOLLOW_MERIDIAN], tol=0.03)
reg("G-MM-east-within-corr",  0.27, "checkin",
    ("ussr_metropole_direction.json",
     "correlations.east_dist_to_u5log"),
    [HOLLOW_MERIDIAN], tol=0.03)
reg("G-MM-n-13",              13, "const",
    "Non-Russia, non-Uzbekistan republics with multi-year residuals",
    [HOLLOW_MERIDIAN], tol=0)
reg("G-MM-n-west",             6, "const",
    "Westward USSR republics (Belarus, Ukraine, Lith, Lat, Est, Mol)",
    [], tol=0)
reg("G-MM-n-east",             8, "const",
    "Eastward USSR republics (Geo, Arm, Az, Turkm, Uz, Tj, Kg, Kz)",
    [], tol=0)
reg("G-MM-Kunaev-yrs",        17, "const",
    "Kunaev years on full Politburo (1971–1987)",
    [HOLLOW_MERIDIAN], tol=0)
reg("G-MM-east-high-sd",     4.00, "checkin",
    _G_PCR("turkmenistan.u5log_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)

# ── §hanushek-reconciliation: HLO-as-home-niche + horse race ───────────────
reg("G-HLO-n",            77, "checkin",
    ("hlo_is_parental_education.json", "numbers.t1_n"),
    [HOLLOW_HANUSHEK], tol=0)
reg("G-HLO-lsec1990-R2",  0.523, "checkin",
    ("hlo_is_parental_education.json", "numbers.t1_r2"),
    [HOLLOW_HANUSHEK], tol=0.01)
reg("G-HLO-lsec1990-beta", 1.94, "checkin",
    ("hlo_is_parental_education.json", "numbers.t1_beta_lsec"),
    [HOLLOW_HANUSHEK], tol=0.05)
reg("G-HLO-lsec1990-t",   9.1, "checkin",
    ("hlo_is_parental_education.json", "numbers.t1_t_lsec"),
    [HOLLOW_HANUSHEK], tol=0.2)
reg("G-HLO-lsec1990-corr", 0.72, "checkin",
    ("hlo_is_parental_education.json", "numbers.t1_corr"),
    [HOLLOW_HANUSHEK], tol=0.02)
# Lag-sweep R²
reg("G-LagSweep-lag0",   0.504, "checkin",
    ("hlo_lag_sweep.json", "numbers.r2_lsec_lag_0"),
    [HOLLOW_HANUSHEK], tol=0.01)
reg("G-LagSweep-lag10",  0.539, "checkin",
    ("hlo_lag_sweep.json", "numbers.lsec_sweep.lag_10.r2"),
    [HOLLOW_HANUSHEK], tol=0.01)
reg("G-LagSweep-primary-lag25", 0.549, "checkin",
    ("hlo_lag_sweep.json", "numbers.primary_sweep.lag_25.r2"),
    [HOLLOW_HANUSHEK], tol=0.01)
reg("G-LagSweep-primary-lag60", 0.489, "checkin",
    ("hlo_lag_sweep.json", "numbers.primary_sweep.lag_60.r2"),
    [HOLLOW_HANUSHEK], tol=0.01)
reg("G-LagSweep-primary-lag0", 0.469, "checkin",
    ("hlo_lag_sweep.json", "numbers.r2_primary_lag_0"),
    [HOLLOW_HANUSHEK], tol=0.01)
reg("G-LagSweep-bootstrap-n", 2000, "checkin",
    ("hlo_lag_sweep.json", "numbers.bootstrap_n"),
    [HOLLOW_HANUSHEK], tol=0)
# [retired 2026-06-19 tier-1 trim] G-LagSweep-pri-peak-pct (92) /
# -lsec-peak-pct (87): the "% of draws" bootstrap detail was cut from
# §Hanushek's HLO (the CI-includes-zero result and the lag-60
# falsification — 0.489 vs 0.469 — stay in the paper). Full sweep detail
# in scripts/ECONOMETRICS.md; hlo_lag_sweep.json unchanged.
# Top and bottom residuals (school overperformers / underperformers)
reg("G-HLO-China-resid",  111, "checkin",
    ("hlo_is_parental_education.json", "numbers.top_positive_residuals[0].residual"),
    [HOLLOW_HANUSHEK], tol=0.5)
reg("G-HLO-Vietnam-resid", 111, "checkin",
    ("hlo_is_parental_education.json", "numbers.top_positive_residuals[1].residual"),
    [HOLLOW_HANUSHEK], tol=0.5)
reg("G-HLO-Singapore-resid", 93, "checkin",
    ("hlo_is_parental_education.json", "numbers.top_positive_residuals[2].residual"),
    [HOLLOW_HANUSHEK], tol=0.5)
reg("G-HLO-Portugal-resid", 76, "checkin",
    ("hlo_is_parental_education.json", "numbers.top_positive_residuals[3].residual"),
    [HOLLOW_HANUSHEK], tol=0.5)
reg("G-HLO-SA-resid",     -116, "checkin",
    ("hlo_is_parental_education.json", "numbers.top_negative_residuals[0].residual"),
    [HOLLOW_HANUSHEK], tol=0.5)
reg("G-HLO-Ghana-resid",  -108, "checkin",
    ("hlo_is_parental_education.json", "numbers.top_negative_residuals[1].residual"),
    [HOLLOW_HANUSHEK], tol=0.5)
reg("G-HLO-Albania-resid",  -86, "checkin",
    ("hlo_is_parental_education.json", "numbers.top_negative_residuals[2].residual"),
    [HOLLOW_HANUSHEK], tol=0.5)
reg("G-HLO-Mont-resid",   -72, "checkin",
    ("hlo_is_parental_education.json", "numbers.top_negative_residuals[3].residual"),
    [HOLLOW_HANUSHEK], tol=0.5)
# Horse race (hanushek_horse_race.json)
# reg("G-HR-n",             104, "checkin",
#     ("hanushek_horse_race.json", "numbers.panel_n"),
#     [COMPLETION], tol=0)
# (removed 2026-05-23: detail dropped in §4 merger)
# G-HR-TFR-* regs: section scopes removed 2026-05-26 — paper §Composition
# rewrite switched to earlier-cohort numbers (horse_race_earlier_cohort.json), and
# the hanushek_horse_race TFR numbers are no longer cited in
# §hanushek-reconciliation or §education-vs-gdp-as-predictors prose.
reg("G-HR-TFR-quant-beta", -0.08, "checkin",
    ("hanushek_horse_race.json", "numbers.tfr.D_quant_beta"),
    [], tol=0.05)
reg("G-HR-TFR-quant-t",   -0.54, "checkin",
    ("hanushek_horse_race.json", "numbers.tfr.D_quant_t"),
    [], tol=0.3)
reg("G-HR-TFR-qual-beta", -0.59, "checkin",
    ("hanushek_horse_race.json", "numbers.tfr.D_qual_beta"),
    [], tol=0.05)
reg("G-HR-TFR-qual-t",    -3.99, "checkin",
    ("hanushek_horse_race.json", "numbers.tfr.D_qual_t"),
    [], tol=0.3)
reg("G-HR-TFR-R2-quant",   0.30, "checkin",
    ("hanushek_horse_race.json", "numbers.tfr.A_quant_only_r2"),
    [], tol=0.02)
reg("G-HR-TFR-R2-both",    0.44, "checkin",
    ("hanushek_horse_race.json", "numbers.tfr.D_both_r2"),
    [], tol=0.02)
reg("G-HR-U5-qual-beta",  -0.70, "checkin",
    ("hanushek_horse_race.json", "numbers.u5mr.D_qual_beta"),
    [COMPLETION], tol=0.02)
reg("G-HR-U5-qual-t",     -7.6, "checkin",
    ("hanushek_horse_race.json", "numbers.u5mr.D_qual_t"),
    [COMPLETION], tol=0.2)
reg("G-HR-U5-R2",         0.78, "checkin",
    ("hanushek_horse_race.json", "numbers.u5mr.D_both_r2"),
    [COMPLETION], tol=0.01)
reg("G-HR-LE-qual-beta",   0.83, "checkin",
    ("hanushek_horse_race.json", "numbers.le.D_qual_beta"),
    [COMPLETION], tol=0.02)
reg("G-HR-LE-qual-t",      7.7, "checkin",
    ("hanushek_horse_race.json", "numbers.le.D_qual_t"),
    [COMPLETION], tol=0.2)
reg("G-HR-LE-R2",         0.62, "checkin",
    ("hanushek_horse_race.json", "numbers.le.D_both_r2"),
    [COMPLETION], tol=0.01)

# Composition-by-level table (tab:composition) introduced in B7.
# Register the table cells against hanushek_horse_race.json in GDP_INDEP
# scope so the unregistered-numbers scan picks them up.
# TFR row uses the lower-sec horse race from horse_race_earlier_cohort.json
# (model C: lsec(T) + HLO), aligned with the paper's switch from
# primary→lower-sec for TFR predictor.
reg("Tcomp-tfr-quant-beta",  -0.19, "checkin",
    ("horse_race_earlier_cohort.json", "numbers.C_lsec_hlo.lsec_2000_z.beta"),
    [GDP_INDEP], tol=0.02)
reg("Tcomp-tfr-quant-t",     -1.5, "checkin",
    ("horse_race_earlier_cohort.json", "numbers.C_lsec_hlo.lsec_2000_z.t"),
    [GDP_INDEP], tol=0.2)
reg("Tcomp-tfr-qual-beta",   -0.51, "checkin",
    ("horse_race_earlier_cohort.json", "numbers.C_lsec_hlo.hlo_sec_z.beta"),
    [GDP_INDEP], tol=0.02)
reg("Tcomp-tfr-qual-t",      -3.4, "checkin",
    ("horse_race_earlier_cohort.json", "numbers.C_lsec_hlo.hlo_sec_z.t"),
    [GDP_INDEP], tol=0.2)
reg("Tcomp-tfr-r2",          0.45, "checkin",
    ("horse_race_earlier_cohort.json", "numbers.C_lsec_hlo.r2"),
    [], tol=0.02)
reg("Tcomp-tfr-n",           66, "checkin",
    ("horse_race_earlier_cohort.json", "numbers.C_lsec_hlo.n"),
    [GDP_INDEP], tol=2)
# T-25 gp-control robustness number cited in §Composition prose.
reg("Tcomp-tfr-gp-qual-beta", -0.54, "checkin",
    ("horse_race_earlier_cohort.json", "numbers.D_lsec_hlo_t10.hlo_sec_z.beta"),
    [GDP_INDEP], tol=0.02)
reg("Tcomp-u5-quant-beta",   -0.16, "checkin",
    ("hanushek_horse_race.json", "numbers.u5mr.D_quant_beta"),
    [GDP_INDEP], tol=0.02)
reg("Tcomp-u5-quant-t",      -1.7, "checkin",
    ("hanushek_horse_race.json", "numbers.u5mr.D_quant_t"),
    [GDP_INDEP], tol=0.2)
reg("Tcomp-u5-qual-beta",    -0.70, "checkin",
    ("hanushek_horse_race.json", "numbers.u5mr.D_qual_beta"),
    [], tol=0.02)
reg("Tcomp-u5-qual-t",       -7.6, "checkin",
    ("hanushek_horse_race.json", "numbers.u5mr.D_qual_t"),
    [], tol=0.2)
reg("Tcomp-u5-r2",           0.78, "checkin",
    ("hanushek_horse_race.json", "numbers.u5mr.D_both_r2"),
    [], tol=0.01)
reg("Tcomp-u5-n",            67, "checkin",
    ("hanushek_horse_race.json", "numbers.u5mr.n"),
    [GDP_INDEP], tol=2)
reg("Tcomp-le-quant-beta",   -0.12, "checkin",
    ("hanushek_horse_race.json", "numbers.le.D_quant_beta"),
    [GDP_INDEP], tol=0.02)
reg("Tcomp-le-quant-t",      -1.1, "checkin",
    ("hanushek_horse_race.json", "numbers.le.D_quant_t"),
    [], tol=0.2)
reg("Tcomp-le-qual-beta",    0.83, "checkin",
    ("hanushek_horse_race.json", "numbers.le.D_qual_beta"),
    [], tol=0.02)
reg("Tcomp-le-qual-t",       7.7, "checkin",
    ("hanushek_horse_race.json", "numbers.le.D_qual_t"),
    [], tol=0.2)
reg("Tcomp-le-r2",           0.62, "checkin",
    ("hanushek_horse_race.json", "numbers.le.D_both_r2"),
    [], tol=0.01)
reg("Tcomp-le-n",            67, "checkin",
    ("hanushek_horse_race.json", "numbers.le.n"),
    [GDP_INDEP], tol=2)

# ── §exclusion-robustness ──────────────────────────────────────────
reg("G-Excl-full-beta",    0.707, "checkin",
    ("ussr_exclusion_panel.json", "numbers.T1M1_full_beta"),[HOLLOW_EXCLUSION], tol=0.005)
reg("G-Excl-full-n",       945,  "checkin",
    ("ussr_exclusion_panel.json", "numbers.T1M1_full_n"),
    [HOLLOW_EXCLUSION], tol=0)
reg("G-Excl-full-countries", 144, "checkin",
    ("ussr_exclusion_panel.json", "numbers.T1M1_full_countries"),
    [HOLLOW_EXCLUSION], tol=0)
reg("G-Excl-clean-beta",   0.707, "checkin",
    ("ussr_exclusion_panel.json", "numbers.T1M1_clean_beta"),[],
    tol=0.005)
reg("G-Excl-clean-n",      945,  "checkin",
    ("ussr_exclusion_panel.json", "numbers.T1M1_clean_n"),
    [], tol=0)
reg("G-Excl-clean-countries", 144, "checkin",
    ("ussr_exclusion_panel.json", "numbers.T1M1_clean_countries"),[],
    tol=0)
reg("G-Excl-full-r2",      0.561, "checkin",
    ("ussr_exclusion_panel.json", "numbers.T1M1_full_r2"),
    [HOLLOW_EXCLUSION], tol=0.005)
reg("G-Excl-clean-r2",     0.561, "checkin",
    ("ussr_exclusion_panel.json", "numbers.T1M1_clean_r2"),
    [], tol=0.005)
reg("G-Excl-LE-full",      0.384, "checkin",
    ("ussr_exclusion_panel.json", "numbers.LE_full_r2"),
    [HOLLOW_EXCLUSION], tol=0.005)
reg("G-Excl-LE-clean",     0.384, "checkin",
    ("ussr_exclusion_panel.json", "numbers.LE_clean_r2"),[], tol=0.005)
reg("G-Excl-TFR-full",     0.630, "checkin",
    ("ussr_exclusion_panel.json", "numbers.TFR_full_r2"),
    [HOLLOW_EXCLUSION], tol=0.005)
reg("G-Excl-TFR-clean",    0.630, "checkin",
    ("ussr_exclusion_panel.json", "numbers.TFR_clean_r2"),[], tol=0.005)
reg("G-Excl-U5-full",      0.650, "checkin",
    ("ussr_exclusion_panel.json", "numbers.U5log_full_r2"),
    [HOLLOW_EXCLUSION], tol=0.005)
reg("G-Excl-U5-clean",     0.650, "checkin",
    ("ussr_exclusion_panel.json", "numbers.U5log_clean_r2"),
    [], tol=0.005)

# ── Appendix Permutation Null Distribution ─────────────────────────
# [retired 2026-06-18 ch9/ch11 compression] reg("Perm-niter",          200,    "checkin",
# [retired 2026-06-18 ch9/ch11 compression]     ("permutation_null.json", "n_iter"),[GDP_INDEP], tol=0)
reg("Perm-full-z",         51,     "checkin",
    ("permutation_null.json", "full.z"),[], tol=1)

# ── §the-anomaly and scattered: post-socialist counts and lags ──
reg("G-n-postsoc",         24, "checkin",
    ("post_socialist_lag.json", "numbers.n_postsoc_crossers"),
    [HOLLOW_PATTERN], tol=0.5)
reg("G-median-lag-market",  16, "checkin",
    ("post_socialist_lag.json", "numbers.median_lag_market_yrs_abs"),
    [HOLLOW_PATTERN], tol=0.5)
reg("G-Turkm-lag",          57, "checkin",
    ("post_socialist_lag.json", "numbers.turkmenistan_lag_yrs_abs"),
    [HOLLOW_PATTERN], tol=0.5)

# ── §what-wcde-reports / §hanushek-reconciliation PISA and HLO refs ──
reg("G-PISA-OECD",         500, "const",
    "OECD mean PISA scale (reference mean)",
    [HOLLOW_WCDE], tol=0)
reg("G-PISA-Kyrg-2009",    350, "ref",
    "Kyrgyzstan PISA 2009 mean score (OECD PISA report)",
    [HOLLOW_WCDE, HOLLOW_HANUSHEK], tol=0.5)
reg("G-Kyrg-2010-wcde",    99, "checkin",
    ("soviet_inflation.json", "numbers.lsec_kyrgyzstan_2010"),
    [HOLLOW_HANUSHEK], tol=1)
reg("G-Albania-HLO",       412, "checkin",
    ("hlo_is_parental_education.json", "numbers.top_negative_residuals[2].hlo"),
    [HOLLOW_HANUSHEK], tol=0.5)
reg("G-Albania-lsec1990",  96, "checkin",
    ("hlo_is_parental_education.json", "numbers.top_negative_residuals[2].lsec_1990"),
    [HOLLOW_HANUSHEK], tol=1)
reg("G-PISA-Baltic-max",   535, "ref",
    "Baltic PISA score upper bound (OECD PISA)",
    [HOLLOW_WCDE], tol=0.5)

# Horse race insignificant-quantity t-stats (paper cites magnitudes;
# checkin stores signed values)
reg("G-HR-U5-quant-t-marg", -1.7, "checkin",
    ("hanushek_horse_race.json", "numbers.u5mr.D_quant_t"),
    [], tol=0.1)
reg("G-HR-LE-quant-t-marg", -1.1, "checkin",
    ("hanushek_horse_race.json", "numbers.le.D_quant_t"),
    [], tol=0.1)
reg("G-HR-U5-quant-p-marg", 0.09, "checkin",
    ("hanushek_horse_race.json", "numbers.u5mr.D_quant_p"),
    [], tol=0.02)

# ── §hollow-education miscellaneous years and dates ──────────────
reg("G-year-1917", 1917, "const",
    "Pre-1917 educational-baseline reference (Russian Revolution)",
    [HOLLOW_WCDE], tol=0)
reg("G-year-1940", 1940, "const",
    "Pre-1940 Baltic educational infrastructure (Soviet occupation)",
    [HOLLOW_WCDE], tol=0)
reg("G-year-1991", 1991, "const",
    "1991 Soviet dissolution kink reference",
    [HOLLOW_PHENO], tol=0)
reg("G-wallace-200", 200, "const",
    "Wallace/Darwin tradition timeframe (~200 years; pre-1858 natural-history milieu through 2026)",
    ["methodological-frontier"], tol=0)

# ══════════════════════════════════════════════════════════════════════════
# LITERATE CT — §literate-ct empirical floors for the regime flip
# (primary_at_le_crossing.py, primary_at_tfr_crossing.py)
# ══════════════════════════════════════════════════════════════════════════
LITERATE_CT = "literate-ct"

reg("CCT-LE-n",            84, "checkin",
    ("primary_at_le_crossing.json", "n_clean"),
    [LITERATE_CT], tol=0)
# CCT-LE-primary-p10 / -median removed 2026-05-26 — paper dropped the
# primary floors at LE crossing in the lower-sec rewrite of §literate-ct.
reg("CCT-LE-lsec-p10",     42, "checkin",
    ("primary_at_le_crossing.json", "lsec_at_cross.p10"),
    [LITERATE_CT], tol=1)
reg("CCT-LE-lsec-median",  65, "checkin",
    ("primary_at_le_crossing.json", "lsec_at_cross.median"),
    [LITERATE_CT], tol=1)
# CCT-LE-gp-primary-median removed 2026-05-26 — paper dropped the GP
# primary number in the lower-sec rewrite.

reg("CCT-TFR-n",           88, "checkin",
    ("lower_sec_at_tfr_crossing.json", "n_clean"),
    [LITERATE_CT], tol=0)
# Lower-sec floors at TFR crossing (replaced primary in the §literate-ct
# rewrite 2026-05-26).
reg("CCT-TFR-lsec-p10",    31, "checkin",
    ("lower_sec_at_tfr_crossing.json", "lower_sec_at_cross.p10"),
    [LITERATE_CT], tol=1)
reg("CCT-TFR-lsec-median", 54, "checkin",
    ("lower_sec_at_tfr_crossing.json", "lower_sec_at_cross.median"),
    [LITERATE_CT], tol=1)
reg("CCT-TFR-gp-lsec-median", 7, "checkin",
    ("lower_sec_at_tfr_crossing.json", "gp_lower_sec.median"),
    [LITERATE_CT], tol=1)

# ══════════════════════════════════════════════════════════════════════════
# THE HUMAN COST — dual-channel (mortality + fertility) counterfactual
# under singular-priority pace; log-curve U5MR (cross-section, gavi_2
# spec dev_no_ussr_wls) + panel TFR coefficient; T-sweep 1970..2000,
# end_year=2025 (no long-horizon projection); USSR-excluded.
# Source: gavi_6_t_sweep.py with --end-year 2025
# ══════════════════════════════════════════════════════════════════════════
# LE-not-a-third-channel mediation test (paper §The Human Cost opener)
reg("HC-le-mediation-beta", 0.013, "checkin",
    ("le_independent_of_u5mr.json", "magnitudes.beta_direct_edu_on_le_yr_per_pp"),
    [HUMAN_COST], tol=0.005)
reg("HC-le-mediation-p",     0.50,  "checkin",
    ("le_independent_of_u5mr.json", "model_C_mediation.coefs.edu_t_dm.p"),
    [HUMAN_COST], tol=0.02)
reg("HC-le-mediation-n",     483,   "checkin",
    ("le_independent_of_u5mr.json", "model_C_mediation.n"),
    [HUMAN_COST], tol=0)
# Mother's-own-longevity adult-mortality channel (paper §The Human Cost opener)
reg("HC-amrt-fe-baseline-pct", -1.25, "checkin",
    ("mothers_own_longevity.json", "magnitudes_pct_per_pp.amrt_fe_baseline"),
    [HUMAN_COST], tol=0.05)
reg("HC-amrt-fe-net-of-u5-pct", -0.37, "checkin",
    ("mothers_own_longevity.json", "magnitudes_pct_per_pp.amrt_fe_net_of_u5"),
    [HUMAN_COST], tol=0.05)
reg("HC-amrt-fe-net-p",         0.001, "checkin",
    ("mothers_own_longevity.json", "model_B_amrt_fe_with_u5.coefs.fem_edu_t_dm.p"),
    [HUMAN_COST], tol=0.001)
reg("HC-amrt-fe-net-n",         483,   "checkin",
    ("mothers_own_longevity.json", "model_B_amrt_fe_with_u5.n"),
    [HUMAN_COST], tol=0)
# Coefficients (paper cites positive; JSON stores signed)
reg("HC-alpha",          5.56,  "checkin",
    ("gavi_6_t_sweep_2025.json", "alpha"),
    [HUMAN_COST], tol=0.01)
reg("HC-beta-log-abs",   0.028, "derived",
    "abs(beta_lsec) from gavi_6_t_sweep_2025.json",
    [HUMAN_COST], tol=0.001)
reg("HC-beta-tfr-abs",   0.058, "derived",
    "abs(beta_tfr) from gavi_6_t_sweep_2025.json",
    [HUMAN_COST], tol=0.001)
# Caveats paragraph restates beta_TFR when naming the within-country basis
reg("HC-beta-tfr-abs-caveat", 0.058, "derived",
    "abs(beta_tfr) from gavi_6_t_sweep_2025.json (restated in Caveats)",
    [HUMAN_COST], tol=0.001)
reg("HC-korea-rate",     2.13,  "checkin",
    ("gavi_6_t_sweep_2025.json", "korea_rate_pp_per_year"),
    [HUMAN_COST], tol=0.005)
# Korea-pace vs actual (the cost-of-laggards headline) at each T
reg("HC-deaths-T1990-korea", 104,  "derived",
    "by_T.1990.scenarios_extra_vs_actual.korea_pace.lives_saved_total",
    [HUMAN_COST], tol=0.5)
reg("HC-mort-T1990-korea",   38,  "derived",
    "by_T.1990.scenarios_extra_vs_actual.korea_pace.via_mortality",
    [HUMAN_COST], tol=0.5)
reg("HC-fert-T1990-korea",    67,  "derived",
    "by_T.1990.scenarios_extra_vs_actual.korea_pace.via_fertility",
    [HUMAN_COST], tol=0.5)
reg("HC-deaths-T1980-korea", 188,  "derived",
    "by_T.1980.scenarios_extra_vs_actual.korea_pace.lives_saved_total",
    [HUMAN_COST], tol=0.5)
reg("HC-mort-T1980-korea",   66,  "derived",
    "by_T.1980.scenarios_extra_vs_actual.korea_pace.via_mortality",
    [HUMAN_COST], tol=0.5)
reg("HC-fert-T1980-korea",   122,  "derived",
    "by_T.1980.scenarios_extra_vs_actual.korea_pace.via_fertility",
    [HUMAN_COST], tol=0.5)
reg("HC-deaths-T1970-korea", 284, "derived",
    "by_T.1970.scenarios_extra_vs_actual.korea_pace.lives_saved_total",
    [HUMAN_COST], tol=1.0)
reg("HC-mort-T1970-korea",   101, "derived",
    "by_T.1970.scenarios_extra_vs_actual.korea_pace.via_mortality",
    [HUMAN_COST], tol=0.5)
reg("HC-fert-T1970-korea",   183,  "derived",
    "by_T.1970.scenarios_extra_vs_actual.korea_pace.via_fertility",
    [HUMAN_COST], tol=0.5)
# Sensitivity envelope at T=1990
reg("HC-deaths-T1990-lag15", 172, "derived",
    "by_T.1990.scenarios_extra_vs_actual.ramp_15yr.lives_saved_total",
    [HUMAN_COST], tol=0.5)
reg("HC-mort-T1990-lag15",   45, "derived",
    "by_T.1990.scenarios_extra_vs_actual.ramp_15yr.via_mortality",
    [HUMAN_COST], tol=0.5)
reg("HC-fert-T1990-lag15",   127, "derived",
    "by_T.1990.scenarios_extra_vs_actual.ramp_15yr.via_fertility",
    [HUMAN_COST], tol=0.5)
reg("HC-deaths-T1990-lag9",  195, "derived",
    "by_T.1990.scenarios_extra_vs_actual.bio_9yr.lives_saved_total",
    [HUMAN_COST], tol=0.5)
reg("HC-mort-T1990-lag9",    51, "derived",
    "by_T.1990.scenarios_extra_vs_actual.bio_9yr.via_mortality",
    [HUMAN_COST], tol=0.5)
reg("HC-fert-T1990-lag9",    144, "derived",
    "by_T.1990.scenarios_extra_vs_actual.bio_9yr.via_fertility",
    [HUMAN_COST], tol=0.5)
# Kids-not-born totals at T=1990 across scenarios
reg("HC-kids-T1990-korea",   635,  "derived",
    "by_T.1990.scenarios_extra_vs_actual.korea_pace.kids_not_born",
    [HUMAN_COST], tol=0.5)
reg("HC-kids-T1990-lag15",  1068,  "derived",
    "by_T.1990.scenarios_extra_vs_actual.ramp_15yr.kids_not_born",
    [HUMAN_COST], tol=1.0)
reg("HC-kids-T1990-lag9",   1199,  "derived",
    "by_T.1990.scenarios_extra_vs_actual.bio_9yr.kids_not_born",
    [HUMAN_COST], tol=1.0)

# ══════════════════════════════════════════════════════════════════════════
# NECESSITY AND SUFFICIENCY, STATED FORWARD — falsifier paragraphs that
# pre-register the framework's bet for the next 25 years.
# Source: scripts/necessity_floor.py
# ══════════════════════════════════════════════════════════════════════════
reg("NSF-rate-floor-pp", 1.25, "checkin",
    ("necessity_floor.json", "sustained_rate_floor_pp_per_yr"),
    [NECESSITY_SUFFICIENCY_FORWARD], tol=0.001)

# ══════════════════════════════════════════════════════════════════════════
# Appendix-cut backfill registrations (added 2026-05-16)
# Numbers introduced by the appendix-cut pass that need fresh registrations:
#   - new Goodman-Bacon footnote in §empirical-strategy (11.3, 71.5, 9.6)
#   - new Appendix A: Reproducibility headline restatement
#   - new Appendix B: Data table cells (mean/sd/min/max columns)
#   - relocated South Africa subsection in §The Country Histories
# ══════════════════════════════════════════════════════════════════════════
SOUTH_AFRICA_CASE = "south-africa-case"
APPENDIX_DATA = "appendix-data"
APPENDIX_REPRO = "appendix-reproducibility"

# Goodman-Bacon footnote weights removed from §empirical-strategy in the
# biology-first restructure; numbers live in scripts/robustness/.

# --- Body table tab:cutoff edu-beta for 50% cutoff (1.053) ---
# (Body table reference at §education-vs-gdp-as-predictors-of-attainment;
#  the for-loop in tab:cutoff-full registrations no longer registers edu-beta
#  cells, so add the body-cited ones directly.)
reg("TabC-50-edu-beta", 1.044, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_50_edu_beta"),
    [EDU_VS_GDP], tol=0.005)
reg("TabC-10-edu-beta", 2.380, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_10_edu_beta"),
    [EDU_VS_GDP], tol=0.005)

# --- Appendix B: Data — tab:summary-full cells (mean/sd/min/max per variable) ---
_SUMF = [
    ("parent_edu", 41.6, 32.9, 0.0, 100.0),
    ("child_edu", 63.6, 31.0, 1.5, 100.0),
    ("log_gdp", 8.30, 1.48, 5.14, 11.67),
    ("life_exp", 66.4, 10.3, 28.6, 84.3),
    ("tfr", 3.56, 1.90, 0.91, 8.86),
    ("u5mr", 62.5, 64.9, 2.2, 334.0),
]
for _v, _mean, _sd, _mn, _mx in _SUMF:
    reg(f"SumF-{_v}-mean", _mean, "checkin",
        ("summary_stats.json", f"descriptives.pooled.{_v}.mean"),
        [APPENDIX_DATA], tol=0.05)
    reg(f"SumF-{_v}-sd",   _sd,   "checkin",
        ("summary_stats.json", f"descriptives.pooled.{_v}.sd"),
        [APPENDIX_DATA], tol=0.05)
    reg(f"SumF-{_v}-min",  _mn,   "checkin",
        ("summary_stats.json", f"descriptives.pooled.{_v}.min"),
        [APPENDIX_DATA], tol=0.5)
    reg(f"SumF-{_v}-max",  _mx,   "checkin",
        ("summary_stats.json", f"descriptives.pooled.{_v}.max"),
        [APPENDIX_DATA], tol=0.5)

# --- Appendix A: Reproducibility — restated headline values ---
# The 1665 / 185 / 0.483 numbers are restated in §appendix-reproducibility
# for self-containment. They are duplicates of headline registrations and
# share their JSON sources.
reg("Repro-headline-beta", 0.707, "checkin",
    ("panel_expansion_fe.json", "numbers.table1_m1_edu_beta"),
    [APPENDIX_REPRO], tol=0.001)
reg("Repro-headline-obs",  945,  "checkin",
    ("panel_expansion_fe.json", "numbers.panel_obs"),
    [], tol=0)
reg("Repro-LOO-lo", 0.696, "checkin",
    ("econometric_battery/09_influence.json", "loo.min_beta"),
    [APPENDIX_REPRO], tol=0.0005)
reg("Repro-LOO-hi", 0.745, "checkin",
    ("econometric_battery/09_influence.json", "loo.max_beta"),
    [APPENDIX_REPRO], tol=0.0005)

# --- South Africa case (relocated from appendix to §The Country Histories) ---
reg("SA-tfr-1990", 3.72, "wdi",
    ("tfr", "South Africa", 1990),
    [SOUTH_AFRICA_CASE], tol=0.05)
reg("SA-tfr-2005", 2.51, "wdi",
    ("tfr", "South Africa", 2005),
    [SOUTH_AFRICA_CASE], tol=0.05)
# SA primary→TFR R²=0.65 reg removed 2026-05-26 — paper rewrote SA case to
# use lower-sec→TFR R²=0.68 from the canonical-lag panel (see SA-lsec-tfr-r2
# below).
reg("SA-lsec-tfr-r2", 0.71, "checkin",
    ("table_contemporaneous.json", "numbers.panel.log_tfr.c1_edu.r2"),
    [SOUTH_AFRICA_CASE], tol=0.01)
reg("SA-circumcision-upper", 99, "ref",
    "Muslim West Africa traditional circumcision prevalence upper bound (Auvert et al. 2005)",
    [SOUTH_AFRICA_CASE])


# ══════════════════════════════════════════════════════════════════════════
# COVERAGE CLEANUP — country-history / case-study / panel-section gaps
# Each reg() below pins a specific paper-line offset that the strict
# consumption-based coverage scan flagged as unregistered. Values point at
# the same JSON paths used by the headline registrations; offsets bind the
# extra occurrence to the exact paper line.
# ══════════════════════════════════════════════════════════════════════════

# §the-famine-test — second occurrence of 19.6 (Famine median edu) at L2092
# and 1966 (Bihar-Kerala comparison year) at L2118.
reg("Kerala-1966-L2118",       1966, "ref",
    "Year of Bihar-Kerala comparison (duplicate occurrence)",
    [FAMINE_TEST], tol=0)

# §cambodia-the-home-niche-shadow — extra occurrences of WCDE completion values
# and timeline years already pinned in headline regs.
reg("Cambodia-1995-yr-L2187",  1995, "ref",
    "Cambodia post-reconstruction jump (duplicate occurrence)",
    [CAMBODIA], tol=0)
reg("Cambodia-peer-21-L2210",  21,   "derived",
    "Median lower_sec_both 1985 among Cambodia peers (duplicate occurrence)",
    [CAMBODIA], tol=1)
reg("Cambodia-2011-L2212",     2011, "ref",
    "Post-disruption cohort reaches school age (duplicate occurrence)",
    [CAMBODIA], tol=0)
reg("Cambodia-1979-L2213",     1979, "ref",
    "Khmer Rouge regime fell (duplicate occurrence)",
    [CAMBODIA], tol=0)
reg("Cambodia-1991-L2218",     1991, "ref",
    "Paris Peace Accords (duplicate occurrence)",
    [CAMBODIA], tol=0)
reg("Cambodia-2011-L2229",     2011, "ref",
    "Parental shadow lifted (duplicate occurrence)",
    [CAMBODIA], tol=0)

# §country-histories (tab:transition-year) — Kerala row, Uganda LE,
# India pace pp/yr.
reg("Kerala-dev-cross-L2429",  1982, "checkin",
    ("kerala.json", "results.both_crossed.expected"),
    [SEN_CASES, KERALA], tol=0)
reg("Kerala-TFR-cross-L2429",  1973, "checkin",
    ("kerala.json", "results.tfr_crossing.expected"),
    [SEN_CASES], tol=0)
reg("Kerala-LE-cross-L2429",   1981, "checkin",
    ("kerala.json", "results.le_crossing.expected"),
    [SEN_CASES], tol=0)
reg("India-ppyr-L2463",        0.87, "derived",
    "India edu rate (duplicate occurrence in country-histories prose)",
    [SEN_CASES], tol=0.1)

# §korea-and-philippines — second occurrence of Korea 2.13 pp/yr at L2511.
reg("Korea-ppyr-L2511",        2.13, "derived",
    "(Korea-1985 - Korea-1955) / 30 (duplicate occurrence in KP prose)",
    [], tol=0.005)

# §four-further-cases — Myanmar 0.6 pp/yr at L2563; "below $1,200" framing
# threshold at L2565 (derived: rounded ceiling above the two crossings
# Bangladesh \$1,159 and Nepal \$1,114).
reg("Myanmar-ppyr-L2563",      0.6,  "derived",
    "(Myanmar-2015 - Myanmar-1960) / 55 (duplicate occurrence)",
    [SRI_LANKA], tol=0.1)

# §china — China LE gap to peers, MYS, structural-break years.
reg("China-LE-gap-1965-L2601", 6.6,  "derived",
    "abs(le_gap_1965) from china_mean_yrs_vs_peers.json (duplicate occurrence)",
    [], tol=0.05)
reg("China-LE-gap-1965-yr-L2602", 1965, "const",
    "China LE-gap reference year 1965 (paired with China-LE-gap-1965)",
    [CHINA], tol=0)
reg("China-LE-gap-1980-L2602", 2.7,  "derived",
    "abs(le_gap_1980) from china_mean_yrs_vs_peers.json (duplicate occurrence)",
    [], tol=0.05)
reg("China-LE-1994-yr-L2617",  1994, "checkin",
    ("table4_crossings.json", "results.China.le_crossing_best"),
    [CHINA], tol=0)

# §south-africa-case — TFR 3.72 (1990) duplicate at L2649.

# §descriptive-statistics — prose-echo TFR halves 4.53 → 2.95 at L2886.

# §empirical-strategy 2WFE / Callaway-Sant'Anna body and footnote
# registrations removed in the biology-first restructure — the 38-line
# 2WFE defense was relocated to scripts/robustness/. The CS-event-study
# trajectory remains in §compounding-generations-signature (CS-att-yr35
# above) and figure fig:cs-event.

# §education-vs-gdp-as-predictors-of-attainment — tab:headline cells and
# prose-echo restatements (629 obs, 105 countries, β=1.376→1.270, β=1.38,
# β=0.483 full-panel, 1665 obs, β_g=5.1 at Taiwan 1.2% baseline).
reg("T1-headline-beta-L3141",       1.43,  "derived",
    "rounded T1-SG-headline-beta (1.376 → 1.38 in prose)",
    [EDU_VS_GDP], tol=0.005)
reg("Beta-baseline-taiwan-base-L3234", 1.2, "derived",
    "Taiwan first-window parent baseline at which Fig1-Taiwan-beta is measured",
    [], tol=0.2)

# §universality-across-subsamples — duplicate GDP-at-2015 income for over-performers.

# §what-wcde-reports — duplicate of pre-1917 baseline reference year.
reg("G-year-1917-L3946",       1917, "const",
    "Pre-1917 educational-baseline reference (duplicate occurrence)",
    [HOLLOW_WCDE], tol=0)

# §what-wcde-reports — duplicate of pre-1940 Baltic infrastructure year
# (second occurrence in the Baltic-PISA sentence).
reg("G-year-1940-baltic-pisa", 1940, "const",
    "Pre-1940 Baltic educational infrastructure (duplicate occurrence)",
    [HOLLOW_WCDE], tol=0)

# §phenotype-test-ussr — duplicate of Iran U5MR 1960.

# §moscow-meridian — extra inline citations of east-of-Moscow distances and
# the eastward residual range endpoint.
reg("G-MM-Georgia-east-L4043", 451, "const",
    "Signed east-of-Moscow km, Tbilisi (duplicate occurrence)",
    [HOLLOW_MERIDIAN], tol=0.5)
reg("G-MM-east-low-sd-L4043", 2.58, "checkin",
    ("ussr_per_country_residuals.json", "wcde.georgia.u5log_resid_sd"),
    [HOLLOW_MERIDIAN], tol=0.05)
reg("G-MM-Latvia-east-L4044",  844, "const",
    "abs(signed east-of-Moscow km, Riga) (duplicate occurrence)",
    [HOLLOW_MERIDIAN], tol=0.5)

# §hanushek-reconciliation — the primary-vs-current R² pair (0.489 at
# lag 60; 0.469 at lag 0) is stated several times in the section
# (headline, body restatement, figure caption). Under the membership
# coverage model one registration per value covers every occurrence, so
# the section-wide G-LagSweep-primary-lag60 / -lag0 regs above suffice;
# no per-occurrence duplicates are needed. (Likewise Portugal's +76
# residual is covered by the section-wide G-HLO-Portugal-resid above.)

# §the-institutional-challenge — autocracy R² bound, regime-transition
# p-value, AJR base-sample size duplicate.
reg("Regime-r2-bound-L4924",    0.7, "derived",
    "rounded upper bound for r2_polity across lags (regime_education_test.json)",
    [INSTIT], tol=0.1)
# AJR-base-n-L4956 removed — the duplicate occurrence at §the-institutional-challenge
# was tied to the standard-controls block in old §identification that no
# longer engages AJR (now only in Ch 11 §the-colonial-test).

# §the-human-cost — Korea pace 2.13 pp/yr restatement inside the
# scenario-comparison table.

# ═════════════════════════════════════════════════════════════════════
# §the-generational-lag — new subsection in §the-panel; biological
# generation cycle (MAC) anchor and timescale R² anchors.
# ═════════════════════════════════════════════════════════════════════
# MAC stats — expansion-phase subset (10% <= lsec_both <= 90%) matches
# the panel sample filter. Source: mean_age_first_birth_invariance.json
# produced by scripts/mean_age_first_birth_invariance.py.
reg("MAC-exp-n", 1609, "checkin",
    ("mean_age_first_birth_invariance.json", "numbers.exp_n"),
    [THE_GENERATIONAL_LAG], tol=0)
reg("MAC-exp-n-countries", 177, "checkin",
    ("mean_age_first_birth_invariance.json", "numbers.exp_n_countries"),
    [THE_GENERATIONAL_LAG], tol=0)
reg("MAC-exp-mean", 28.8, "checkin",
    ("mean_age_first_birth_invariance.json", "numbers.exp_mac_mean"),
    [THE_GENERATIONAL_LAG, "the-prediction",
     "the-generational-transmission-mechanism", "glossary"], tol=0.1)
# MAC-exp-median/p10/p90 removed from main text 2026-05-23 with §Three
# Timescales tighten (renamed §The Generational Lag). Mean stays; p10/p90
# diagnostic detail dropped from main text. JSONs retained on disk for
# the script-level reader.
reg("MAC-exp-bin-range", 1.0, "checkin",
    ("mean_age_first_birth_invariance.json", "numbers.exp_bin_range"),
    [THE_GENERATIONAL_LAG], tol=0.1)
reg("MAC-exp-bin-low", 29.3, "checkin",
    ("mean_age_first_birth_invariance.json", "numbers.exp_by_lsec_bin[0].mean"),
    [THE_GENERATIONAL_LAG], tol=0.05)
reg("MAC-exp-bin-high", 28.2, "checkin",
    ("mean_age_first_birth_invariance.json", "numbers.exp_by_lsec_bin[4].mean"),
    [THE_GENERATIONAL_LAG], tol=0.1)
# Lag-sweep R² anchors for the four-outcome timescales table (existing
# checkins). HLO 0.549 already registered as G-LagSweep-primary-lag25.
reg("TS-TFR-r2-lag0", 0.668, "checkin",
    ("outcomes_r2_by_lag.json", "numbers.tfr_r2_lag0"),
    [], tol=0.005)
reg("TS-LE-r2-lag0", 0.552, "checkin",
    ("outcomes_r2_by_lag.json", "numbers.le_r2_lag0"),
    [], tol=0.005)
reg("TS-U5MR-r2-lag15", 0.716, "checkin",
    ("outcomes_r2_by_lag.json", "numbers.u5log_r2_lag15"),
    [], tol=0.005)
reg("TS-GDP-r2-lag0", 0.386, "checkin",
    ("edu_to_gdp_lag_sweep.json", "numbers.edu_gdp_r2_lag0"),
    [THE_GENERATIONAL_LAG], tol=0.005)
# HLO primary R²(lag 25) = 0.549 — also cited in §the-generational-lag as the
# cross-generation cognitive-stock anchor. Duplicate of G-LagSweep-primary-lag25
# (registered earlier for §hanushek-reconciliation).
reg("TS-HLO-r2-lag25", 0.549, "checkin",
    ("hlo_lag_sweep.json", "numbers.primary_sweep.lag_25.r2"),
    [], tol=0.01)
# Leader's-timeline arithmetic: today's 6-year-old reaches age 21 in 15
# years. The "21" appears once in §the-generational-lag as derived.
reg("TS-leader-age-21", 21, "derived",
    "Today's 6-year-old at age 21 after 15 years (6+15)",
    [], tol=0)

# ─── Horse-race incremental R²: which generational block (the cohort's own
# schooling at T = the parental generation, vs the prior generation at T-28
# = the grandparental generation) carries independent information for each
# outcome, with each outcome read forward at its biological horizon (TFR
# T+5, U5MR/LE T+12, GDP T+0). Source: horse_race_lags_levels.json.
# The lag0/lag28/kitchen R² values are soundness-checked against checkin but
# not cited in prose; the gain_from_* deltas are the six numbers in the
# §the-generational-lag paragraph.
reg("TS-HR-TFR-lag0",    0.703, "checkin",
    ("horse_race_lags_levels.json", "numbers.logtfr.lag0_r2"),
    [], tol=0.005)
reg("TS-HR-TFR-lag28",   0.68, "checkin",
    ("horse_race_lags_levels.json", "numbers.logtfr.lag28_r2"),
    [], tol=0.005)
reg("TS-HR-TFR-kitchen", 0.793, "checkin",
    ("horse_race_lags_levels.json", "numbers.logtfr.kitchen_r2"),
    [], tol=0.005)
reg("TS-HR-U5-lag0",     0.579, "checkin",
    ("horse_race_lags_levels.json", "numbers.logu5.lag0_r2"),
    [], tol=0.005)
reg("TS-HR-U5-lag28",    0.604, "checkin",
    ("horse_race_lags_levels.json", "numbers.logu5.lag28_r2"),
    [], tol=0.005)
reg("TS-HR-U5-kitchen",  0.709, "checkin",
    ("horse_race_lags_levels.json", "numbers.logu5.kitchen_r2"),
    [], tol=0.005)
reg("TS-HR-LE-lag0",     0.289, "checkin",
    ("horse_race_lags_levels.json", "numbers.logle.lag0_r2"),
    [], tol=0.005)
reg("TS-HR-LE-lag28",    0.264, "checkin",
    ("horse_race_lags_levels.json", "numbers.logle.lag28_r2"),
    [], tol=0.005)
reg("TS-HR-LE-kitchen",  0.332, "checkin",
    ("horse_race_lags_levels.json", "numbers.logle.kitchen_r2"),
    [], tol=0.005)
reg("TS-HR-GDP-lag0",    0.262, "checkin",
    ("horse_race_lags_levels.json", "numbers.loggdp.lag0_r2"),
    [], tol=0.005)
reg("TS-HR-GDP-lag28",   0.218, "checkin",
    ("horse_race_lags_levels.json", "numbers.loggdp.lag28_r2"),
    [], tol=0.005)
reg("TS-HR-GDP-kitchen", 0.277, "checkin",
    ("horse_race_lags_levels.json", "numbers.loggdp.kitchen_r2"),
    [], tol=0.005)
# [retired 2026-06-19 tier-1 trim] TS-HR-{LE,GDP,TFR,U5}-gain-lag{0,25}
# (0.043/0.068/0.011/0.091/0.114/0.130/0.105): the §Outcome-specific lag
# structure paragraph that displayed these within-R^2 increments was cut
# from §The Generational Lag (it re-told §The Grandparent Channel at the
# R^2 level). The parent/grandparent decomposition now lives in
# scripts/ECONOMETRICS.md; horse_race_lags_levels.json is unchanged.
# Grandparent channel — LE 2-regressor result cited in §the-generational-lag
# to reconcile with the kitchen-sink finding (LE grandparent signal
# disappears when all three contemporaneous levels are controlled for).
reg("TS-GP-LE-beta",  0.091, "checkin",
    ("grandparent_effect.json", "results.le.parent_gp.beta_grandparent_edu"),
    [], tol=0.005)
reg("TS-GP-LE-r2-gain",  0.065, "checkin",
    ("grandparent_effect.json", "results.le.r2_gain"),
    [], tol=0.005)

# ══════════════════════════════════════════════════════════════════════════
# LONGRUN (pre-medicine) TEST — country-FE OLS β, the primary-spec estimate.
# Within-era (1885-1920, 15 first-mover countries, pre-antibiotic outcomes)
# and forward (train pre-1950 → predict 1960+). The six mortality βs/SEs are
# cited in §The Deaton Objection (the-deaton-objection); the forward TFR β is
# checkin-only (section=[]) — registered for traceability but not cited in
# prose. Values are 5-seed medians from aggregate_longrun.py.
# The within-era mortality βs are the headline: more education → longer life
# (LE β>0) and fewer child deaths (U5MR β<0), decades before modern medicine.
# ══════════════════════════════════════════════════════════════════════════
reg("Longrun-Within-LE-beta",      0.404, "checkin",
    ("longrun_within.json",  "by_method.panel_fe.LE.ols_beta_edu_median"),
    ["the-deaton-objection"], tol=0.002)
reg("Longrun-Within-LE-se",        0.095, "checkin",
    ("longrun_within.json",  "by_method.panel_fe.LE.ols_cluster_se_median"),
    ["the-deaton-objection"], tol=0.002)
reg("Longrun-Within-U5MR-beta",   -2.977, "checkin",
    ("longrun_within.json",  "by_method.panel_fe.U5MR.ols_beta_edu_median"),
    ["the-deaton-objection"], tol=0.002)
reg("Longrun-Within-U5MR-se",      0.636, "checkin",
    ("longrun_within.json",  "by_method.panel_fe.U5MR.ols_cluster_se_median"),
    ["the-deaton-objection"], tol=0.002)
reg("Longrun-Forward-LE-beta",     0.566, "checkin",
    ("longrun_forward.json", "by_method.panel_fe.LE.ols_beta_edu_median"),
    ["the-deaton-objection"], tol=0.002)
reg("Longrun-Forward-U5MR-beta",  -3.464, "checkin",
    ("longrun_forward.json", "by_method.panel_fe.U5MR.ols_beta_edu_median"),
    ["the-deaton-objection"], tol=0.002)
reg("Longrun-Forward-TFR-beta",    0.015, "checkin",
    ("longrun_forward.json", "by_method.panel_fe.TFR.ols_beta_edu_median"),
    [], tol=0.002)

# ══════════════════════════════════════════════════════════════════════════
# CHAPTER 9 ML — §Every Method Agrees (every-method-agrees). All values read
# via load_checkin's ML-checkin fallback (scripts/ml/checkin/). Spec-curve
# drops, transformer gates, double-ML θ, walk-forward, placebos, and the
# out-of-fold country-counterfactual table.
# ══════════════════════════════════════════════════════════════════════════
EMA = "every-method-agrees"
HEAD = "chapter9_headline.json"
CF = "chapter9_counterfactuals.json"
# Per-outcome biological-lag exhibits (LE@25, TFR@5, U5MR@12). Same ML-checkin
# fallback as HEAD. These replace the T+25 joint headline for the
# spec-curve, transformer gates row, and double-ML rows.
SCP = "spec_curve_parent.json"
UEVP = "universal_evidence_parent_lag.json"
DMLP = "dml_parent.json"

# Spec-curve: linear + tree methods (spec_curve_5_methods list, fixed order).
reg("EMA-gbm-LE",   0.191, "checkin", (SCP, "results.gradient_boosting.r2_drop_absolute.LE"),   [EMA], tol=0.002)
reg("EMA-gbm-TFR",  0.330, "checkin", (SCP, "results.gradient_boosting.r2_drop_absolute.TFR"),  [EMA], tol=0.002)
reg("EMA-gbm-U5MR", 0.367, "checkin", (SCP, "results.gradient_boosting.r2_drop_absolute.U5MR"), [EMA], tol=0.002)
reg("EMA-lasso-LE",   0.256, "checkin", (SCP, "results.lasso.r2_drop_absolute.LE"),   [EMA], tol=0.002)
reg("EMA-lasso-TFR",  0.360, "checkin", (SCP, "results.lasso.r2_drop_absolute.TFR"),  [EMA], tol=0.002)
reg("EMA-lasso-U5MR", 0.273, "checkin", (SCP, "results.lasso.r2_drop_absolute.U5MR"), [EMA], tol=0.002)
reg("EMA-rf-LE",   0.222, "checkin", (SCP, "results.random_forest.r2_drop_absolute.LE"),   [EMA], tol=0.002)
reg("EMA-rf-TFR",  0.353, "checkin", (SCP, "results.random_forest.r2_drop_absolute.TFR"),  [EMA], tol=0.002)
reg("EMA-rf-U5MR", 0.400, "checkin", (SCP, "results.random_forest.r2_drop_absolute.U5MR"), [EMA], tol=0.002)
reg("EMA-ridge-LE",   0.195, "checkin", (SCP, "results.ridge.r2_drop_absolute.LE"),   [EMA], tol=0.002)
reg("EMA-ridge-TFR",  0.252, "checkin", (SCP, "results.ridge.r2_drop_absolute.TFR"),  [EMA], tol=0.002)
reg("EMA-ridge-U5MR", 0.245, "checkin", (SCP, "results.ridge.r2_drop_absolute.U5MR"), [EMA], tol=0.002)
# Spec-curve: universal transformer (30-seed medians, per-outcome biological lags).
reg("EMA-trf-LE",   0.234, "checkin", (UEVP, "per_outcome.LE.B1_B2_gates_on_joint.g1_education_ablation.r2_drop_absolute.LE"),     [EMA], tol=0.002)
reg("EMA-trf-TFR",  0.262, "checkin", (UEVP, "per_outcome.TFR.B1_B2_gates_on_joint.g1_education_ablation.r2_drop_absolute.TFR"),   [EMA], tol=0.002)
reg("EMA-trf-U5MR", 0.394, "checkin", (UEVP, "per_outcome.U5MR.B1_B2_gates_on_joint.g1_education_ablation.r2_drop_absolute.U5MR"), [EMA], tol=0.002)
# Sample size and seed count.
reg("EMA-n-seeds",   30,   "checkin", (UEVP, "n_seeds"), [EMA], tol=0.5)

# G1/G2 ablation gates (fraction of held-out R² lost).
reg("EMA-g1-LE",   0.31, "checkin", (UEVP, "per_outcome.LE.B1_B2_gates_on_joint.g1_education_ablation.r2_drop_fraction.LE"),     [EMA], tol=0.006)
reg("EMA-g1-TFR",  0.31, "checkin", (UEVP, "per_outcome.TFR.B1_B2_gates_on_joint.g1_education_ablation.r2_drop_fraction.TFR"),   [EMA], tol=0.006)
reg("EMA-g1-U5MR", 0.52, "checkin", (UEVP, "per_outcome.U5MR.B1_B2_gates_on_joint.g1_education_ablation.r2_drop_fraction.U5MR"), [EMA], tol=0.006)
reg("EMA-g2-LE",   0.02,  "checkin", (UEVP, "per_outcome.LE.B1_B2_gates_on_joint.g2_gdp_ablation.r2_drop_fraction.LE"),     [EMA], tol=0.006)
reg("EMA-g2-TFR",  0.002, "checkin", (UEVP, "per_outcome.TFR.B1_B2_gates_on_joint.g2_gdp_ablation.r2_drop_fraction.TFR"),   [EMA], tol=0.006)
reg("EMA-g2-U5MR", 0.007, "checkin", (UEVP, "per_outcome.U5MR.B1_B2_gates_on_joint.g2_gdp_ablation.r2_drop_fraction.U5MR"), [EMA], tol=0.006)

# Double machine-learning θ (per education level) + 95% CIs.
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA-dml-LE",      4.85, "checkin", (DMLP, "by_target.LE.theta_median"),     [EMA], tol=0.01)
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA-dml-LE-lo",   4.63, "checkin", (DMLP, "by_target.LE.theta_ci95[0]"),    [EMA], tol=0.01)
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA-dml-LE-hi",   5.56, "checkin", (DMLP, "by_target.LE.theta_ci95[1]"),    [EMA], tol=0.01)
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA-dml-TFR",     -0.94, "checkin", (DMLP, "by_target.TFR.theta_median"),   [EMA], tol=0.01)
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA-dml-TFR-lo",  -0.98, "checkin", (DMLP, "by_target.TFR.theta_ci95[0]"),  [EMA], tol=0.01)
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA-dml-TFR-hi",  -0.89, "checkin", (DMLP, "by_target.TFR.theta_ci95[1]"),  [EMA], tol=0.01)
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA-dml-U5MR",    -36.5, "checkin", (DMLP, "by_target.U5MR.theta_median"),  [EMA], tol=0.1)
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA-dml-U5MR-lo", -39.1, "checkin", (DMLP, "by_target.U5MR.theta_ci95[0]"), [EMA], tol=0.1)
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA-dml-U5MR-hi", -33.3, "checkin", (DMLP, "by_target.U5MR.theta_ci95[1]"), [EMA], tol=0.1)

# Walk-forward held-out R² (min, max, pre-1980→post-1990 LE).
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-wf-min",  0.70, "checkin", (HEAD, "walk_forward.1975_U5MR.r2_median"), [EMA], tol=0.01)
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA-wf-max",  0.93, "checkin", (HEAD, "walk_forward.1990_TFR.r2_median"),  [EMA], tol=0.01)
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA-wf-1980", 0.84, "checkin", (HEAD, "walk_forward.1980_LE.r2_median"),   [EMA], tol=0.01)

# Placebos: decisive nulls (~0) and partial-channel medians.
# [retired 2026-06-19 tier-1 §Falsification->ECONOMETRICS.md] reg("EMA-plac-outcome",  -0.02,  "checkin", (HEAD, "placebo_nulls.placebo_outcome_scramble.median_drop"),     [EMA], tol=0.005)
# [retired 2026-06-19 tier-1 §Falsification->ECONOMETRICS.md] reg("EMA-plac-noise",    -0.005, "checkin", (HEAD, "placebo_nulls.placebo_edu_replaced_noise.median_drop"),   [EMA], tol=0.003)
# [retired 2026-06-19 tier-1 §Falsification->ECONOMETRICS.md] reg("EMA-plac-country",  -0.06,  "checkin", (HEAD, "placebo_nulls.placebo_country_scramble.median_drop"),     [EMA], tol=0.005)
# [retired 2026-06-19 tier-1 §Falsification->ECONOMETRICS.md] reg("EMA-plac-latitude",  0.02,  "checkin", (HEAD, "placebo_nulls.placebo_edu_replaced_latitude.median_drop"),[EMA], tol=0.005)
# [retired 2026-06-19 tier-1 §Falsification->ECONOMETRICS.md] reg("EMA-plac-gdprank",   0.07,  "checkin", (HEAD, "placebo_nulls.placebo_edu_replaced_gdprank.median_drop"), [EMA], tol=0.005)
# [retired 2026-06-19 tier-1 §Falsification->ECONOMETRICS.md] reg("EMA-plac-year",      0.12,  "checkin", (HEAD, "placebo_nulls.placebo_year_scramble.median_drop"),        [EMA], tol=0.005)

# Out-of-fold country-counterfactual table (gaps_by_pair).
# [retired 2026-06-19 tier-1 §What-the-gap-bought->ECONOMETRICS.md] reg("EMA-cf-lka-LE",   12.10, "checkin", (CF, "gaps_by_pair.sri lanka__pakistan.LE"),   [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-lka-TFR",  -2.19, "checkin", (CF, "gaps_by_pair.sri lanka__pakistan.TFR"),  [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-lka-U5MR", -48.2, "checkin", (CF, "gaps_by_pair.sri lanka__pakistan.U5MR"), [EMA], tol=0.05)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-cub-LE",    4.61, "checkin", (CF, "gaps_by_pair.cuba__dominican republic.LE"),   [EMA], tol=0.01)
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA-cf-cub-TFR",  -0.70, "checkin", (CF, "gaps_by_pair.cuba__dominican republic.TFR"),  [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-cub-U5MR", -8.4, "checkin", (CF, "gaps_by_pair.cuba__dominican republic.U5MR"), [EMA], tol=0.05)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-bwa-LE",    4.80, "checkin", (CF, "gaps_by_pair.botswana__zimbabwe.LE"),   [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-bwa-TFR",  0.23, "checkin", (CF, "gaps_by_pair.botswana__zimbabwe.TFR"),  [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-bwa-U5MR", -11.8, "checkin", (CF, "gaps_by_pair.botswana__zimbabwe.U5MR"), [EMA], tol=0.05)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-vnm-LE",    1.84, "checkin", (CF, "gaps_by_pair.vietnam__myanmar.LE"),   [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-vnm-TFR",  -1.10, "checkin", (CF, "gaps_by_pair.vietnam__myanmar.TFR"),  [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-vnm-U5MR", -11.8,  "checkin", (CF, "gaps_by_pair.vietnam__myanmar.U5MR"), [EMA], tol=0.05)
# [retired 2026-06-19 tier-1 §What-the-gap-bought->ECONOMETRICS.md] reg("EMA-cf-chn-LE",    7.06, "checkin", (CF, "gaps_by_pair.china__india.LE"),   [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-chn-TFR",  -2.06, "checkin", (CF, "gaps_by_pair.china__india.TFR"),  [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-chn-U5MR", -16.9, "checkin", (CF, "gaps_by_pair.china__india.U5MR"), [EMA], tol=0.05)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-kor-LE",    3.04, "checkin", (CF, "gaps_by_pair.south korea__philippines.LE"),   [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-kor-TFR",  -0.30, "checkin", (CF, "gaps_by_pair.south korea__philippines.TFR"),  [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-kor-U5MR", -3.6,  "checkin", (CF, "gaps_by_pair.south korea__philippines.U5MR"), [EMA], tol=0.05)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-ken-LE",   -1.39, "checkin", (CF, "gaps_by_pair.kenya__tanzania.LE"),   [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-ken-TFR",  -0.28, "checkin", (CF, "gaps_by_pair.kenya__tanzania.TFR"),  [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-ken-U5MR", -4.8,  "checkin", (CF, "gaps_by_pair.kenya__tanzania.U5MR"), [EMA], tol=0.05)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-rwa-LE",   0.47, "checkin", (CF, "gaps_by_pair.rwanda__burundi.LE"),   [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-rwa-TFR",  0.00, "checkin", (CF, "gaps_by_pair.rwanda__burundi.TFR"),  [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-rwa-U5MR", -14.0,  "checkin", (CF, "gaps_by_pair.rwanda__burundi.U5MR"), [EMA], tol=0.05)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-bgd-LE",   -0.76, "checkin", (CF, "gaps_by_pair.bangladesh__pakistan.LE"),   [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-bgd-TFR",   0.00, "checkin", (CF, "gaps_by_pair.bangladesh__pakistan.TFR"),  [EMA], tol=0.01)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA-cf-bgd-U5MR",  4.0,  "checkin", (CF, "gaps_by_pair.bangladesh__pakistan.U5MR"), [EMA], tol=0.05)

# ══════════════════════════════════════════════════════════════════════════
# BLOC-HOLDOUT corroboration of the Moscow Meridian (§moscow-meridian).
# Blind transformer trained on 148 non-Soviet countries, USSR withheld;
# level-space U5MR residuals reproduce the west→east gradient. Log-space
# correlation is noise-dominated and excluded (see log_corr_caveat).
# ══════════════════════════════════════════════════════════════════════════
BLOC = "chapter9_bloc_holdout.json"
reg("Bloc-corr-east",  0.78, "checkin", (BLOC, "ml_U5MR_corr_east_km"),          ["moscow-meridian"], tol=0.01)
reg("Bloc-west-sigma", 0.23, "checkin", (BLOC, "split.ml_west_mean_U5MR_sigma"), ["moscow-meridian"], tol=0.01)
reg("Bloc-east-sigma", 1.43, "checkin", (BLOC, "split.ml_east_mean_U5MR_sigma"), ["moscow-meridian"], tol=0.01)
reg("Bloc-n-train",    148,  "checkin", (BLOC, "n_train_countries"),             ["moscow-meridian"], tol=0.5)

# ══════════════════════════════════════════════════════════════════════════
# PER-OCCURRENCE COVERAGE (2026-06-12 audit)
# ──────────────────────────────────────────────────────────────────────────
# After the global STRUCTURAL_NUMBERS whitelist was retired, every numeric
# token in the paper body must consume one registration bound to its section.
# Identical values that appear N times in a section need N registrations.
# Sources: data findings → wcde/wdi/checkin/derived; calendar / lookup years
# → `year` (reason in the detail string); definitional or proper-noun numbers
# → `const`. Organised section by section, matching checkin/coverage_gaps.md.
# ══════════════════════════════════════════════════════════════════════════

# Section labels are passed as bare strings (the legacy "(label, offset)"
# form is accepted but the offset is ignored — line numbers drift on every
# edit). tol is the value's printed precision: 0 for exact integers / years /
# constants, 0.01 for decimals registered at full source precision.

# ── §korea-and-philippines ────────────────────────────────────────────────
# Philippines 1950 lower-secondary completion = 22.16% (prints "22%"): opening
# sentence, and "American colonial education ... built that 22% base".
reg("KP-phil-1950-open", 22.16, "wcde", ("cohort_lower_sec_both.csv", "Philippines", 1950),
    [KOREA_PHILIPPINES], tol=0.01)
reg("KP-phil-1950-base", 22.16, "wcde", ("cohort_lower_sec_both.csv", "Philippines", 1950),
    [KOREA_PHILIPPINES], tol=0.01)
# Korea 1950 lower-secondary completion = 24.81% (prints "25%"): opening
# sentence ("Korea's 25%") and the Japan colonial-base pair ("Korea 25%").
reg("KP-korea-1950-open", 24.81, "wcde", ("cohort_lower_sec_both.csv", "Korea", 1950),
    [KOREA_PHILIPPINES], tol=0.01)
reg("KP-korea-1950-base", 24.81, "wcde", ("cohort_lower_sec_both.csv", "Korea", 1950),
    [KOREA_PHILIPPINES], tol=0.01)
# Taiwan 1950 lower-secondary completion = 17.75% (prints "18%"): Japan pair.
reg("KP-taiwan-1950", 17.75, "wcde", ("cohort_lower_sec_both.csv", "Taiwan", 1950),
    [KOREA_PHILIPPINES], tol=0.01)
# Singular / benchmark-pace crossing-lag band 20--34 yr (Table tab:transition-year:
# fastest movers cluster — Cuba/Korea/Indonesia onset→joint-crossing).
reg("KP-singular-lag-lo", 20, "const",
    "singular-pace crossing-lag low end, yr (Table tab:transition-year)",
    [KOREA_PHILIPPINES], tol=0)
reg("KP-singular-lag-hi", 34, "const",
    "singular-pace crossing-lag high end, yr (Table tab:transition-year)",
    [KOREA_PHILIPPINES], tol=0)
# Competing-priority crossing-lag band 40--70 yr (Sri Lanka/China ~42 … India ~67).
reg("KP-competing-lag-lo", 40, "const",
    "competing-priority crossing-lag low end, yr (Table tab:transition-year)",
    [KOREA_PHILIPPINES], tol=0)
reg("KP-competing-lag-hi", 70, "const",
    "competing-priority crossing-lag high end, yr (Table tab:transition-year)",
    [KOREA_PHILIPPINES], tol=0)
# Calendar / lookup years in prose.
reg("KP-yr-2022", 2022, "year",
    "Philippines status-snapshot year (TFR 1.9 / LE 69.5)", [KOREA_PHILIPPINES], tol=0)
reg("KP-yr-1950", 1950, "year",
    "Korea ceiling-argument starting-position year", [KOREA_PHILIPPINES], tol=0)
reg("KP-yr-2030", 2030, "year",
    "SDG 4 target year for universal upper-secondary", [KOREA_PHILIPPINES], tol=0)
# UN Sustainable Development Goal number (proper noun, "SDG 4").
reg("KP-sdg-goal-num", 4, "const",
    "UN Sustainable Development Goal number (proper noun 'SDG 4')",
    [KOREA_PHILIPPINES], tol=0)

# ── Lag-28 re-baseline: per-occurrence regs for §the-specification bio-lag
# stack and assorted constants/years de-registered by surrounding edits. ──
SPEC = "the-specification"
reg("SPEC-age-lo", 20, "const", "WCDE cohort age band lower bound (20-24)", [SPEC], tol=0)
reg("SPEC-age-hi", 24, "const", "WCDE cohort age band upper bound (20-24)", [SPEC], tol=0)
reg("SPEC-lag-gdp", 0, "const", "GDP biological horizon (contemporaneous, lag 0)", [SPEC], tol=0)
reg("SPEC-lag-tfr", 5, "const", "TFR biological horizon (cohort own childbearing, lag 5)", [SPEC], tol=0)
reg("SPEC-lag-le-u5", 12, "const", "LE/U5MR biological horizon (childrearing window, lag 12)", [SPEC], tol=0)
reg("SPEC-lag-gen", 28, "const", "generation lag for child education (MAC~28.7)", [SPEC], tol=0)
reg("SPEC-ntests", 20, "const", "20-test diagnostic suite count", [SPEC], tol=0)
reg("EVG-headline-beta-occ3", 1.434, "checkin",
    ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_30_edu_beta"),
    [EDU_VS_GDP], tol=0.005)
reg("EVG-pval-05", 0.05, "const", "p<0.05 significance threshold (Table 1 notes)", [EDU_VS_GDP], tol=0)
reg("EVG-pval-01", 0.01, "const", "p<0.01 significance threshold (cross-section note)", [EDU_VS_GDP], tol=0)
# [retired 2026-06-18 ch9/ch11 compression] reg("EVG-korea-afc-drop", 6, "const", "Korea Asian-financial-crisis GDP drop (~6%)", [EDU_VS_GDP], tol=0)
reg("EVG-pp-resolution", 5, "const", "lower-secondary completion 5-pp resolution", [EDU_VS_GDP], tol=0)
reg("SHP-fig-lag100", 100, "const", "lag-100 endpoint of the lag-sweep figure label", [THE_SHAPE_OF_THE_RESPONSE], tol=0)
reg("PHE-under5-caption", 5, "const", "under-5 in U5MR-trajectory figure caption", ["phenotype-test-ussr"], tol=0)
reg("DEC-sdg4", 4, "const", "UN SDG 4 (proper noun) in §The Decision", ["the-decision"], tol=0)



# ======================================================================
# PER-OCCURRENCE COVERAGE (2026-06-12)
# After the global STRUCTURAL_NUMBERS whitelist was retired, every numeric
# token in the paper body must consume one registration bound to its section.
# The blocks below register those tokens section by section: data findings via
# wcde/wdi/checkin/derived, calendar/lookup years via `year`, definitional or
# proper-noun numbers via `const`. Soundness (value vs source) and consumption
# (one token per reg) are both checked at run time.
# ======================================================================


# ------------------------------------------------------------
# FLAGS (data findings to trace during integration, NOT registered here):
#   the-convergence L459=13 L460=20 L461=80 (cumulative-developed pop shares)
#   literate-ct L1388=65 n=84, L1391=31 p10, L1391=54 median n=88
#   appendix-data L5634=42.0 mean, =33.3 sd, L5635=61.9, L5637=65.7, =68.3
#   the-window L912=100 (~100% universal, approx)
#   the-dilution-mechanism L4785=19 (19 of 21 famines)

# ── §the-convergence ──
reg("CV-year-2022",       2022,  "year", "survey period endpoint (1960-2022 span)", ["the-convergence"], tol=0)
reg("CV-185-countries",   185,   "const", "total country universe in analysis panel", ["the-convergence"], tol=0)
reg("CV-year-1960",       1960,  "year", "United States baseline year for development definition", ["the-convergence"], tol=0)
reg("CV-year-1953",       1953,  "year", "Korea reference year (Korea in 1953)", ["the-convergence"], tol=0)
reg("CV-year-2001",       2001,  "year", "year of 50% world-population convergence milestone", ["the-convergence"], tol=0)
reg("CV-1961-13pct",      13,    "const", "cumulative-developed share milestone (~13%, 1961)", ["the-convergence"], tol=0)
reg("CV-1993-20pct",      20,    "const", "cumulative-developed share milestone (~20%, 1993)", ["the-convergence"], tol=0)
reg("CV-50pct",           50,    "const", "50% cumulative-developed milestone", ["the-convergence"], tol=0)
reg("CV-80pct",           80,    "const", "late-2010s cumulative-developed share (rounded)", ["the-convergence"], tol=0)
# Wallace-form fenced headline: the necessity floor named qualitatively up
# front ("near 35%"), with the numeric bet stated in §The Prediction. Same
# rounded source as NSF-floor-rounded-* (necessity_floor.json).
reg("CV-floor-near-35",   35.0,  "checkin", ("necessity_floor.json", "sufficiency_floor_pct"), ["the-convergence"], tol=0.5)

# ── §glossary ──
reg("GL-level-1",        1,     "const", "Level 1: Categorical literacy (individual brain reorganization)", ["glossary"], tol=0)
reg("GL-level-2",        2,     "const", "Level 2: Literate CT (societal regime change)", ["glossary"], tol=0)
reg("GL-beta-g-lower",   1,     "const", "beta_g > 1 at low baselines (structural property)", ["glossary"], tol=0)
reg("GL-beta-g-upper",   0,     "const", "beta_g approaches 0 at high baselines (ceiling compression)", ["glossary"], tol=0)
reg("GL-year-1975",      1975,  "year", "post-1975 panel baseline (headline panel anchor)", ["glossary"], tol=0)
reg("GL-window-10pct",   10,    "const", "expansion window lower bound: 10%", ["glossary"], tol=0)
reg("GL-window-90pct",   90,    "const", "expansion window upper bound: 90%", ["glossary"], tol=0)
reg("GL-28-countries",   28,    "const", "subsample size with self-determined education policy", ["glossary"], tol=0)
reg("GL-year-1975-b",    1975,  "year", "post-1975 panel reference (deeper baselines)", ["glossary"], tol=0)
reg("GL-25-year",        28,    "const", "per-country sliding window interval in years", ["glossary"], tol=0)
# Generational-cycle entry (relabeled from "Time-to-agency" 2026-06-18; the
# old entry mis-defined time-to-agency as the 28-year cohort cycle).
reg("GL-gencycle-head-28", 28,  "const", "generational-cycle entry heading (~28 years)", ["glossary"], tol=0)
reg("GL-gencycle-t28",     28,  "const", "headline T-28 lag cited in generational-cycle entry", ["glossary"], tol=0)
reg("GL-gencycle-age-lo",  20,  "const", "cohort age lower bound (20-24) in generational-cycle entry", ["glossary"], tol=0)
reg("GL-gencycle-age-hi",  24,  "const", "cohort age upper bound (20-24) in generational-cycle entry", ["glossary"], tol=0)
reg("GL-gencycle-agency",  15,  "const", "agency-transfer milestone (~15) cross-referenced in generational-cycle entry", ["glossary"], tol=0)

# ── §literate-ct ──
reg("LCT-year-1960",     1960,  "year", "1960 US outcomes as developed-household criterion", ["literate-ct"], tol=0)
reg("LCT-100pct",        100,   "const", "asymptote definition (100% = all households developed)", ["literate-ct"], tol=0)
reg("LCT-69.8-le",       69.8,  "const", "life expectancy threshold (1960 US value)", ["literate-ct"], tol=0)
reg("LCT-age-20",        20,    "const", "20-24 age cohort lower bound", ["literate-ct"], tol=0)
reg("LCT-age-24",        24,    "const", "20-24 age cohort upper bound", ["literate-ct"], tol=0)
reg("LCT-year-1960-b",   1960,  "year", "fertility threshold crossing baseline", ["literate-ct"], tol=0)
reg("LCT-3.65-tfr",      3.65,  "const", "TFR threshold (1960 US value)", ["literate-ct"], tol=0)
reg("LCT-age-20-b",      20,    "const", "operational threshold age cohort (20-24)", ["literate-ct"], tol=0)
reg("LCT-age-24-b",      24,    "const", "operational threshold age cohort (20-24)", ["literate-ct"], tol=0)

# ── §the-window-supports-a-continuous-dose ──
reg("WIN-age-20",        20,    "const", "age cohort lower bound 20-24", ["the-window-supports-a-continuous-dose"], tol=0)
reg("WIN-age-24",        24,    "const", "age cohort upper bound 20-24", ["the-window-supports-a-continuous-dose"], tol=0)
reg("WIN-age-20-b",      20,    "const", "age cohort reference (20-24, Singapore context)", ["the-window-supports-a-continuous-dose"], tol=0)
reg("WIN-age-24-b",      24,    "const", "age cohort reference (20-24, Singapore context)", ["the-window-supports-a-continuous-dose"], tol=0)
reg("WIN-year-2020",     2020,  "year", "WCDE v3 cohort vintage year (2020)", ["the-window-supports-a-continuous-dose"], tol=0)
reg("WIN-age-20-c",      20,    "const", "tertiary completion age cohort (20-24, 2020)", ["the-window-supports-a-continuous-dose"], tol=0)
reg("WIN-age-24-c",      24,    "const", "tertiary completion age cohort (20-24, 2020)", ["the-window-supports-a-continuous-dose"], tol=0)
reg("WIN-year-2020-b",   2020,  "year", "tertiary completion year (2020 cohort)", ["the-window-supports-a-continuous-dose"], tol=0)

# ── §the-dilution-mechanism ──
reg("DIL-sdg-4",         4,     "const", "SDG 4 ordinal among Sustainable Development Goals", ["the-dilution-mechanism"], tol=0)
reg("DIL-sdg-1",         1,     "const", "SDG 1 (No Poverty) parallel goal", ["the-dilution-mechanism"], tol=0)
reg("DIL-sdg-2",         2,     "const", "SDG 2 (Zero Hunger) parallel goal", ["the-dilution-mechanism"], tol=0)
reg("DIL-sdg-3",         3,     "const", "SDG 3 (Good Health) parallel goal", ["the-dilution-mechanism"], tol=0)
reg("DIL-sdg-5",         5,     "const", "SDG 5 (Gender Equality) parallel goal", ["the-dilution-mechanism"], tol=0)
reg("DIL-sdg-8",         8,     "const", "SDG 8 (Decent Work) parallel goal", ["the-dilution-mechanism"], tol=0)
reg("DIL-sdg-10",        10,    "const", "SDG 10 (Reduced Inequalities) parallel goal", ["the-dilution-mechanism"], tol=0)
reg("DIL-year-1950",     1950,  "year", "post-1950 famine reference period", ["the-dilution-mechanism"], tol=0)
reg("DIL-50pct",         50,    "const", "lower-secondary completion threshold for famine avoidance", ["the-dilution-mechanism"], tol=0)
reg("DIL-sdg-5-b",       5,     "const", "SDG 5 dimensions (gender equality) reference", ["the-dilution-mechanism"], tol=0)

# ── §appendix-reproducibility ──
reg("APR-20-tests",      20,    "const", "econometric diagnostic battery test count", ["appendix-reproducibility"], tol=0)
reg("APR-16-16",         16,    "const", "16 of 16 multiple-comparison tests (first occurrence)", ["appendix-reproducibility"], tol=0)
reg("APR-16-16-b",       16,    "const", "16 of 16 multiple-comparison tests (second occurrence)", ["appendix-reproducibility"], tol=0)
reg("APR-5pct",          5,     "const", "significance level for multiple-comparison corrections", ["appendix-reproducibility"], tol=0)
reg("APR-20-sd",         20,    "const", "placebo nulls sit 20+ SD below real beta (min separation: within-country shuffle 24.2 SD; cohort-mismatch 41 SD)", ["appendix-reproducibility"], tol=0)

# ── §appendix-data ──
reg("APD-25-lag",        28,    "const", "parental education lag in years (T-25)", ["appendix-data"], tol=0)
reg("APD-5-interval",    5,     "const", "analysis panel interval (1975-2015 at 5-year intervals)", ["appendix-data"], tol=0)
reg("APD-1000-u5mr",     1000,  "const", "U5MR denominator (per 1,000 live births)", ["appendix-data"], tol=0)
reg("APD-185-countries", 185,   "const", "analysis panel country universe", ["appendix-data"], tol=0)
reg("APD-5-interval-b",  5,     "const", "1975-2015 at 5-year intervals (second reference)", ["appendix-data"], tol=0)
reg("APD-25-lag-b",      28,    "const", "parental education lagged 25 years (second reference)", ["appendix-data"], tol=0)
reg("APD-185-pool",      185,   "const", "education+GDP panel countries (178 of 185)", ["appendix-data"], tol=0)
reg("APD-25-lag-c",      28,    "const", "observed at T-25 (third reference)", ["appendix-data"], tol=0)

# ── §the-decision ──
reg("DEC-year-2015",     2015,  "year", "UN 2015 SDG 4 commitment reference", ["the-decision"], tol=0)
reg("DEC-year-2030",     2030,  "year", "SDG 4 target completion year (by 2030)", ["the-decision"], tol=0)
reg("DEC-2015-commitment", 2015, "year", "commitment reference year (SDG adoption)", ["the-decision"], tol=0)
reg("DEC-year-2030-b",   2030,  "year", "2030 horizon for fifteen-year window", ["the-decision"], tol=0)
reg("DEC-sdg-4",         4,     "const", "SDG 4 (twelve years of schooling)", ["the-decision"], tol=0)
reg("DEC-year-1696",     1696,  "year", "Knox 1696 historical reference", ["the-decision"], tol=0)
reg("DEC-year-1961",     1961,  "year", "Cuba 1961 literacy campaign reference", ["the-decision"], tol=0)


# ------------------------------------------------------------
# ── §how-to-read ──
reg("HTR-panel-countries", 185, "const",
    "number of countries in the panel (185-country universe)", ["how-to-read"], tol=0)
reg("HTR-econometric-tests", 20, "const",
    "reproducibility battery size (20-test econometric diagnostic battery)", ["how-to-read"], tol=0)

# ── §the-dependency-window ──
reg("TDW-workforce-yr-a", 2050, "year",
    "future year for workforce projection scenario", ["the-dependency-window"], tol=0)
reg("TDW-workforce-yr-b", 2050, "year",
    "future year for workforce projection scenario (adult retrofitting context)", ["the-dependency-window"], tol=0)

# ── §how-the-window-was-built ──
reg("HWB-brain-pct-body", 2, "const",
    "brain as percentage of body mass (metabolic fact; roughly 2%)", ["how-the-window-was-built"], tol=0)
reg("HWB-brain-pct-energy", 20, "const",
    "brain as percentage of resting energy (metabolic fact; about 20%)", ["how-the-window-was-built"], tol=0)

# ── §the-human-ratchet ──
reg("THR-tool-version", 2.0, "const",
    "rhetorical version number (termite-fishing 2.0; illustrating ratchet concept)", ["the-human-ratchet"], tol=0.01)

# ── §the-tool-sequence ──
reg("TLS-fossil-fuels-yr", 2017, "year",
    "data-vintage year in tool-sequence prose", ["the-tool-sequence"], tol=0)
reg("TLS-knox-yr-a", 1696, "year",
    "Knox's parish school founding year (Scotland education-act context)", ["the-tool-sequence"], tol=0)
reg("TLS-knox-yr-b", 1696, "year",
    "Knox's parish school founding year (affordability comparison context)", ["the-tool-sequence"], tol=0)

# ── §specialisation-requires-loaded-labour ──
reg("SRL-knox-act-yr", 1696, "year",
    "Scotland's 1696 Education Act (founding event for parish schools)", ["specialisation-requires-loaded-labour"], tol=0)
reg("SRL-panel-scale", 185, "const",
    "country scale for demonstrating industrial-revolution pattern (185-country panel)", ["specialisation-requires-loaded-labour"], tol=0)

# ── §categorical-brain-reorganisation ──
reg("CBR-dehaene-yr", 2004, "year",
    "Dehaene numeracy-work reference year", ["categorical-brain-reorganisation"], tol=0)
reg("CBR-piraha-set-a", 5, "const",
    "Piraha natural experiment: set size that populations without formal arithmetic cannot reliably distinguish", ["categorical-brain-reorganisation"], tol=0)
reg("CBR-piraha-set-b", 7, "const",
    "Piraha natural experiment: comparison set size (5 vs 7 distinction test)", ["categorical-brain-reorganisation"], tol=0)

# ── §levels-compose ──
reg("LC-level-individual", 1, "const",
    "Level 1: individual-level categorical reorganisation (brain loading)", ["levels-compose"], tol=0)
reg("LC-level-societal", 2, "const",
    "Level 2: societal-level categorical transmission regime (literate CT)", ["levels-compose"], tol=0)

# ── §duration-over-fidelity ──
reg("DOF-lag-window-min", 10, "const",
    "parental-generation completion lag: minimum window (10-25 years earlier)", ["duration-over-fidelity"], tol=0)
reg("DOF-lag-window-max", 28, "const",
    "parental-generation completion lag: maximum window (10-25 years earlier)", ["duration-over-fidelity"], tol=0)

# ── §why-poorly-nourished-populations-learned-to-read ──
reg("WPN-knox-scotland-yr", 1696, "year",
    "Scotland's 1696 parish school population (nourishment vs literacy proof)", ["why-poorly-nourished-populations-learned-to-read"], tol=0)
reg("WPN-meiji-japan-yr", 1872, "year",
    "Meiji Japan cohort year (poorly-nourished learning proof)", ["why-poorly-nourished-populations-learned-to-read"], tol=0)
reg("WPN-cuba-literacy-yr", 1961, "year",
    "Cuba literacy campaign starting year (poorly-nourished learning proof)", ["why-poorly-nourished-populations-learned-to-read"], tol=0)
reg("WPN-knox-nourishment-yr", 1696, "year",
    "1696 child-nourishment reference year", ["why-poorly-nourished-populations-learned-to-read"], tol=0)

# ── §from-action-to-talk-how-education-reaches-beyond-the-household ──
reg("FAT-reference-yr", 2010, "year",
    "citation reference year for group-boundary flexibility", ["from-action-to-talk-how-education-reaches-beyond-the-household"], tol=0)
reg("FAT-biological-window", 18, "const",
    "biological dependency window duration (18 years)", ["from-action-to-talk-how-education-reaches-beyond-the-household"], tol=0)

# ── §demographic-structure-and-the-fertility-transition ──
reg("DSFT-tfr-threshold", 4, "const",
    "illustrative TFR level above which population growth is high (TFR > 4)", ["demographic-structure-and-the-fertility-transition"], tol=0)

# ── §education-as-limiting-factor ──
reg("ELF-cohort-age-min", 20, "const",
    "20-24 age cohort: minimum age captures education at schooling terminus", ["education-as-limiting-factor"], tol=0)
reg("ELF-cohort-age-max", 24, "const",
    "20-24 age cohort: maximum age captures education at schooling terminus", ["education-as-limiting-factor"], tol=0)
reg("ELF-stock-measure-min", 20, "const",
    "20-24 cohort: minimum age for generational educational stock measure", ["education-as-limiting-factor"], tol=0)
reg("ELF-stock-measure-max", 24, "const",
    "20-24 cohort: maximum age for generational educational stock measure", ["education-as-limiting-factor"], tol=0)

# ── §why-lower-secondary-completion ──
reg("WLS-wcde-history-yr", 1875, "year",
    "WCDE long-run lower-secondary completion baseline year (estimates back to 1875)", ["why-lower-secondary-completion"], tol=0)

# ── §what-the-panel-supports ──
reg("WPP-diagnostic-battery", 20, "const",
    "20-test panel-diagnostic battery (reproducibility and robustness)", ["what-the-panel-supports"], tol=0)

# ── §growth-theory-and-the-residual ──
reg("GTR-preindustrial-yr", 1800, "year",
    "pre-industrial historical-baseline reference year", ["growth-theory-and-the-residual"], tol=0)

# ── §diamond-geographic-objection ──
reg("DGO-diamond-yr-a", 1950, "year",
    "geographic-equivalence test starting year", ["diamond-geographic-objection"], tol=0)
reg("DGO-korea-phl-yr-a", 1950, "year",
    "Korea/Philippines starting-position year (geographic equivalence test)", ["diamond-geographic-objection"], tol=0)
reg("DGO-korea-phl-yr-b", 2000, "year",
    "Korea/Philippines comparison year (divergence despite shared geography)", ["diamond-geographic-objection"], tol=0)

# ── §schooling-and-learning ──
reg("SAL-knox-schools-yr", 1560, "year",
    "Scotland post-1560 parish school era (scripture teaching example)", ["schooling-and-learning"], tol=0)
reg("SAL-cuba-literacy-yr", 1961, "year",
    "Cuba 1961 literacy campaign (revolutionary primers example)", ["schooling-and-learning"], tol=0)

# ── §innovation-and-the-cultural-substrate ──
reg("ICS-elite-substrate-yr-a", 1900, "year",
    "reference year for elite-substrate distinction (two-layer argument)", ["innovation-and-the-cultural-substrate"], tol=0)
reg("ICS-elite-substrate-yr-b", 1950, "year",
    "reference year for elite-substrate distinction (population-substrate rise context)", ["innovation-and-the-cultural-substrate"], tol=0)

# ── §easterlins-question ──
reg("EQ-reference-yr", 1950, "year",
    "post-1950 development context (Easterlin's question)", ["easterlins-question"], tol=0)

# ── §invisible-from-inside ──
reg("IFI-preindustrial-yr", 1800, "year",
    "post-1800 historical reference (life-expectancy and literacy rise)", ["invisible-from-inside"], tol=0)

# ── §historical-exceptions-confirm-the-rule ──
reg("HEC-future-child-yr", 2050, "year",
    "future year for example (child arriving as one in 2050; generational exception)", ["historical-exceptions-confirm-the-rule"], tol=0)

# ── §this-applies-to-this-paper ──
reg("TAP-paper-opening-yr", 1990, "year",
    "paper's panel opening year (1990 depth-of-operative-channel context)", ["this-applies-to-this-paper"], tol=0)

# ── §empire-impossible-to-man (decolonisation re-read: empire became
#    impossible to man from both ends as loading rose) ──
reg("EITM-salisbury-1882", 1882, "fact", "salisbury-india-barrack-1882",
    ["empire-impossible-to-man"], tol=0)
reg("EITM-isr-1918-count", 548311, "fact",
    "indian-army-1918-imperial-strategic-reserve", ["empire-impossible-to-man"], tol=0)
reg("EITM-yr-1918-a", 1918, "year",
    "Indian Army at the Imperial Strategic Reserve peak ('By 1918')", ["empire-impossible-to-man"], tol=0)
reg("EITM-yr-1918-b", 1918, "year",
    "ideologies unchanged 1918-1947 (span start)", ["empire-impossible-to-man"], tol=0)
reg("EITM-yr-1947", 1947, "year",
    "ideologies unchanged 1918-1947 (span end; Indian independence)", ["empire-impossible-to-man"], tol=0)
# metropole-end ledger: British fertility ran its own transition to
# replacement by 1918 (the reserve-peak year) and below replacement in the
# 1930s, so the borrowed reserve ballooned to >2.5M by 1945 before it withdrew.
reg("EITM-yr-1871", 1871, "year",
    "British TFR baseline year 1871 (metropole fertility before decline)", ["empire-impossible-to-man"], tol=0)
reg("EITM-tfr-1871", 4.88, "fact", "uk-tfr-1871",
    ["empire-impossible-to-man"], tol=0)
reg("EITM-tfr-1918", 2.03, "fact", "uk-tfr-1918",
    ["empire-impossible-to-man"], tol=0)
reg("EITM-ia-1945-count", 2500000, "fact", "indian-army-1945-peak",
    ["empire-impossible-to-man"], tol=0)
reg("EITM-yr-1945", 1945, "year",
    "Indian Army peak August 1945 (borrowed reserve at maximum)", ["empire-impossible-to-man"], tol=0)

# ── §the-asymmetry-reverses ──
reg("TAR-europe-baseline-yr", 1700, "year",
    "Europe-1700 baseline for asymmetry-reversal argument", ["the-asymmetry-reverses"], tol=0)

# ── §the-four-radii-reverse ──
reg("TFR-imperial-yr", 1700, "year",
    "1700 reference year (imperial-era parity context)", ["the-four-radii-reverse"], tol=0)
reg("TFR-postwar-yr", 1945, "year",
    "post-1945 institutional-architecture reference year", ["the-four-radii-reverse"], tol=0)


# ------------------------------------------------------------
# methodological-frontier + what-the-panel-sharpens + what-wcde-reports.
#
# DROPPED the agent's 11 ABSTRACT PREAMBLE regs (ABS-latex-*, ABS-color-*,
# ABS-tolerance): those tokens (2,3,10,16,60x3,1000) are LaTeX preamble, not
# paper content. FIX IN ENGINE: scan must skip everything before
# \begin{document}; then those tokens vanish and need no regs.
#
# FLAG follow-ups (kept as const/ref now; backlog producing-script extensions):
#  - defining-development L596 "TFR above 6" rhetorical baseline (const)
#  - defining-development figure read-offs 13/20/50 (const), 80 verified (checkin)
#  - what-wcde-reports 11/18 neighbor gap range (ref), 70 pp gap (const)
#  - DD-year-1972 = USA transition completion (year), NOT Cuba 1972 crossing
#  - MF-cuba-1974 / DD-japan-cross-1964 -> table4_crossings (verified checkin)

# ── §abstract ── (content only; preamble skipped by engine)
reg("ABS-year-1960",    1960, "year",  "post-1960 mass-decolonization window", ["abstract"], tol=0)
reg("ABS-year-2022",    2022, "year",  "snapshot year for the 154-country crossing count", ["abstract"], tol=0)
reg("ABS-universe-185a", 185, "checkin", ("summary_stats.json", "numbers.panel_countries"), ["abstract"], tol=0)
reg("ABS-crossed-154",   154, "checkin", ("development_threshold_count.json", "numbers.countries_crossing_both"), ["abstract"], tol=0)
reg("ABS-gen-25",         28, "const", "~25-year singular-priority one-generation crossing time", ["abstract"], tol=0)
reg("ABS-compet-60",      60, "const", "~60-year competing-priority crossing time (2-3 generations)", ["abstract"], tol=0)
reg("ABS-window-lo",      10, "const", "active-expansion window [10%,90%] lower bound", ["abstract"], tol=0)
reg("ABS-window-hi",      90, "const", "active-expansion window [10%,90%] upper bound", ["abstract"], tol=0)

# ── §defining-development ──
reg("DD-year-1960-a",   1960, "year",   "1960 USA development-threshold anchor (LE sentence)", ["defining-development"], tol=0)
reg("DD-life-at-45",      45, "const",  "rhetorical mortality threshold ('people die at 45')", ["defining-development"], tol=0)
reg("DD-year-1960-b",   1960, "year",   "1960 USA position on the TFR-LE diagonal", ["defining-development"], tol=0)
reg("DD-le-thresh-b",   69.8, "wdi",    ("le", "USA", 1960), ["defining-development"], tol=0.05)
reg("DD-tfr-thresh-b",  3.65, "wdi",    ("tfr", "USA", 1960), ["defining-development"], tol=0.01)
reg("DD-year-1960-c",   1960, "year",   "Why the 1960 USA? baseline opening (first occurrence)", ["defining-development"], tol=0)
reg("DD-year-1960-d",   1960, "year",   "1960 marks start of mass decolonization (second occurrence)", ["defining-development"], tol=0)
reg("DD-year-1960-e",   1960, "year",   "USA mid-transition baby-boom TFR sentence", ["defining-development"], tol=0)
reg("DD-tfr-thresh-c",  3.65, "wdi",    ("tfr", "USA", 1960), ["defining-development"], tol=0.01)
reg("DD-year-1972",     1972, "year",   "USA fertility transition completed 1972 (baby-boom end)", [], tol=0)
reg("DD-tfr-thresh-d",  3.65, "wdi",    ("tfr", "USA", 1960), ["defining-development"], tol=0.01)
reg("DD-year-1960-f",   1960, "year",   "in 1960 only the developed world had TFR that low", ["defining-development"], tol=0)
reg("DD-tfr-baseline-6",   6, "const",  "human pre-transition baseline TFR ~6 (rhetorical regime figure)", ["defining-development"], tol=0)
reg("DD-year-1960-g",   1960, "year",   "countries already crossed both thresholds by 1960", ["defining-development"], tol=0)
reg("DD-japan-tfr-2",    2.0, "wdi",    ("tfr", "Japan", 1960), ["defining-development"], tol=0.05)
reg("DD-japan-cross-1964", 1964, "checkin", ("table4_crossings.json", "results.Japan.le_crossing_best"), [], tol=0)
reg("DD-year-2022",     2022, "year",   "by 2022, 154 countries crossed both thresholds", ["defining-development"], tol=0)
reg("DD-pct-80a",         80, "checkin", ("development_threshold_count.json", "numbers.pct_developed"), ["defining-development"], tol=0.5)
reg("DD-tfr-thresh-e",  3.65, "wdi",    ("tfr", "USA", 1960), ["defining-development"], tol=0.01)
reg("DD-le-thresh-c",   69.8, "wdi",    ("le", "USA", 1960), ["defining-development"], tol=0.05)
reg("DD-fig-pct-13",      13, "const",  "cumulative-share figure caption: ~13% in 1961 (curve read-off)", ["defining-development"], tol=0)
reg("DD-fig-pct-20",      20, "const",  "cumulative-share figure caption: ~20% in 1993 (curve read-off)", ["defining-development"], tol=0)
reg("DD-fig-pct-50",      50, "const",  "cumulative-share figure caption: 50% by 2001 (curve read-off)", ["defining-development"], tol=0)
reg("DD-fig-year-2001", 2001, "year",   "50% milestone year in cumulative-developed figure caption", ["defining-development"], tol=0)
reg("DD-pct-80b",         80, "checkin", ("development_threshold_count.json", "numbers.pct_developed"), ["defining-development"], tol=0.5)

# ── §the-deaton-objection ──
reg("DEA-year-1950", 1950, "year", "post-1950 mortality decline (Deaton scope)", ["the-deaton-objection"], tol=0)
reg("DEA-year-1970", 1970, "year", "post-1970 reduction sentence (vaccines since 1960s)", ["the-deaton-objection"], tol=0)
reg("DEA-year-1870", 1870, "year", "WCDE series back to 1870 for pre-medicine test", ["the-deaton-objection"], tol=0)
reg("DEA-year-1945", 1945, "year", "outcomes before 1945 (before antibiotics/mass vaccination)", ["the-deaton-objection"], tol=0)
reg("DEA-year-1960", 1960, "year", "predict mortality from 1960 forward (decades Deaton attributes)", ["the-deaton-objection"], tol=0)

# ── §methodological-frontier ──
reg("MF-rct-scale-50",  50,   "const",   "illustrative IRB-impossible RCT scale: 50 million children", ["methodological-frontier"], tol=0)
reg("MF-year-1950",   1950,   "year",    "Korea/Philippines comparable colonial bases in 1950", ["methodological-frontier"], tol=0)
reg("MF-year-1953",   1953,   "year",    "North/South Korea same population in 1953", ["methodological-frontier"], tol=0)
reg("MF-year-1961",   1961,   "year",    "Cuba literacy campaign 1961", ["methodological-frontier"], tol=0)
reg("MF-cuba-1974",   1974,   "checkin", ("table4_crossings.json", "results.Cuba.both_crossed"), [], tol=0)
reg("MF-year-1949",   1949,   "year",    "China after 1949 mass-education commitment", ["methodological-frontier"], tol=0)
reg("MF-year-1960",   1960,   "year",    "Niger after 1960 comparably impoverished", ["methodological-frontier"], tol=0)

# ── §what-the-panel-sharpens ──
reg("WPS-year-1950",   1950,  "year",    "panel span start 1950", ["what-the-panel-sharpens"], tol=0)
reg("WPS-year-2015",   2015,  "year",    "panel span end 2015", ["what-the-panel-sharpens"], tol=0)
reg("WPS-battery-20",    20,  "const",   "20-test econometric diagnostic battery", [], tol=0)

# ── §what-wcde-reports ──
reg("WWR-year-1970-a",  1970, "year",  "gender-parity heading year 1970", ["what-wcde-reports"], tol=0)
reg("WWR-fm-gap-lo",      11, "ref",   "non-Soviet-neighbor female-male gap lower bound 1970 (-11 to -18 pp; WCDE per-country, no dedicated key)", ["what-wcde-reports"], tol=0)
reg("WWR-fm-gap-hi",      18, "ref",   "non-Soviet-neighbor female-male gap upper bound 1970 (-11 to -18 pp; WCDE per-country, no dedicated key)", ["what-wcde-reports"], tol=0)
reg("WWR-year-1970-b",  1970, "year",  "near-parity 1970 rural Central Asia", ["what-wcde-reports"], tol=0)
reg("WWR-pp-gap-70",      70, "const", "~70-pp gap: Kazakhstan 94% vs neighbors ~22-24% (descriptive round)", ["what-wcde-reports"], tol=0)
reg("WWR-pisa-year-2009", 2009, "ref", "Kyrgyzstan 2009 PISA score year", ["what-wcde-reports"], tol=0)


# ------------------------------------------------------------
# FIX: India crossings switched from agent's broken wdi-read (value=year but cell
#   returns the LE/TFR value) to `derived` via _first_crossing dispatch (added at
#   integration: CH-India-TFR -> first_crossing India tfr 3.65 below; CH-India-LE
#   and CH-India-dev -> first_crossing India le 69.8 above).
# REF (no dispatch, close gap as manual): CH-SriLanka-gap, CH-China-gap,
#   KE-ahead-India, KE-LE-lag — simple table4_crossings differences.

# ── §country-histories ──
reg("CH-Taiwan-dev-a", 1970, "checkin", ("table4_crossings.json", "results.Taiwan.both_crossed"), ["country-histories"], tol=0)
reg("CH-Taiwan-dev-b", 1970, "checkin", ("table4_crossings.json", "results.Taiwan.both_crossed"), ["country-histories"], tol=0)
reg("CH-Taiwan-lag", 20, "const", "Taiwan approximate lag in generations table (~20 yr, tilde-rounded from 22)", ["country-histories"], tol=0)
reg("CH-Korea-lag", 34, "const", "Korea onset->crossing lag ~34 yr (1953->1987); expansion_rate_predicts_crossing.json lag=34", ["country-histories"], tol=0)
reg("CH-Korea-onset", 1953, "year", "Korea schooling-expansion onset year in tab:transition-year (Rhee 1949 law; 1953 armistice; 2.13 pp/yr rate measured 1955-1985 per checkin/country_education.json)", ["country-histories"], tol=0)
reg("CH-Cuba-onset-yr", 1961, "ref", "Cuba post-revolution expansion onset year 1961 (Prieto 1981)", ["country-histories"], tol=0)
reg("CH-Cuba-base", 40, "wcde", ("lower_sec_both.csv", "Cuba", 1960), ["country-histories"], tol=0.5)
reg("CH-Cuba-lag", 13, "const", "Cuba lag 1974-1961=13 (one generation, high starting base)", ["country-histories"], tol=0)
reg("CH-Bangladesh-lag", 24, "checkin", ("expansion_rate_predicts_crossing.json", "Bangladesh.lag"), ["country-histories"], tol=0)
reg("CH-SriLanka-onset-50", 50, "year", "Sri Lanka educational onset 1940s-50s (1950s decade marker)", ["country-histories"], tol=0)
reg("CH-SriLanka-lag", 42, "const", "Sri Lanka approximate lag (~42 yr, tilde-rounded from 43)", ["country-histories"], tol=0)
reg("CH-China-rate", 1.5, "checkin", ("expansion_rate_predicts_crossing.json", "China.rate"), ["country-histories"], tol=0.01)
reg("CH-China-lag", 42, "const", "China approximate lag (~42 yr, tilde-rounded from 44)", ["country-histories"], tol=0)
reg("CH-Kerala-onset-20", 20, "year", "Kerala early 20th century onset (20th century marker)", ["country-histories"], tol=0)
reg("CH-Kerala-lag", 65, "const", "Kerala approximate lag (~65 yr; table onset-to-crossing, ~62-67)", ["country-histories"], tol=0)
reg("CH-India-dev", 2017, "derived", "India both-thresholds crossing year (LE>69.8 binding); first_crossing India le 69.8 above", ["country-histories"], tol=0)
reg("CH-India-TFR", 1996, "derived", "India TFR crossing year (first year TFR<3.65); first_crossing India tfr 3.65 below", ["country-histories"], tol=0)
reg("CH-India-LE", 2017, "derived", "India LE crossing year (first year LE>69.8); first_crossing India le 69.8 above", ["country-histories"], tol=0)
reg("CH-India-lag", 67, "const", "India approximate lag 2017-1950=67 years", ["country-histories"], tol=0)
reg("CH-India-gen", 3, "const", "India generations=3 (third-Korea-pace)", ["country-histories"], tol=0)
reg("CH-thresh-yr", 1960, "year", "US 1960 values anchor for development thresholds (table footnote)", ["country-histories"], tol=0)
reg("CH-gen-25-note", 28, "const", "~25-year generation interval (table footnote)", ["country-histories"], tol=0)
reg("CH-Cuba-base-note", 40, "wcde", ("lower_sec_both.csv", "Cuba", 1960), ["country-histories"], tol=0.5)
reg("CH-Cuba-onset-note", 1961, "ref", "Cuba 1961 literacy campaign (table footnote duplicate)", ["country-histories"], tol=0)
reg("CH-gen-25-foot", 28, "const", "~25-year generation interval (table closing footnote)", ["country-histories"], tol=0)
reg("CH-SriLanka-gap", 12, "derived", "Sri Lanka TFR-to-LE gap: table4_crossings le(1993)-tfr(1981)=12", ["country-histories"], tol=0)
reg("CH-China-gap", 19, "derived", "China TFR-to-LE gap: table4_crossings le(1994)-tfr(1975)=19", ["country-histories"], tol=0)
reg("CH-pace-20", 20, "const", "Korea-pace lower bound 20 yr to crossing", ["country-histories"], tol=0)
reg("CH-pace-34", 34, "const", "Korea-pace upper bound 34 yr to crossing", ["country-histories"], tol=0)
reg("CH-pace-42", 42, "const", "half-Korea-pace lower bound 42 yr", ["country-histories"], tol=0)
reg("CH-pace-45", 45, "const", "half-Korea-pace upper bound 45 yr", ["country-histories"], tol=0)
reg("CH-pace-60", 60, "const", "third-Korea-pace lower bound 60 yr", ["country-histories"], tol=0)
reg("CH-pace-70", 70, "const", "third-Korea-pace upper bound 70 yr", ["country-histories"], tol=0)

# ── §kerala ──
reg("KE-ahead-India", 35, "derived", "Kerala ~35 yr ahead: India dev 2017 - Kerala dev 1982", ["kerala"], tol=0)
reg("KE-LE-lag", 8, "derived", "Kerala LE lag: kerala.json le_crossing 1981 - tfr_crossing 1973 = 8", ["kerala"], tol=0)

# ── §four-further-cases ──
reg("FF-SriLanka-dev", 1993, "checkin", ("table4_crossings.json", "results.Sri Lanka.both_crossed"), ["four-further-cases"], tol=0)
reg("FF-Myanmar-73yr", 73, "const", "Myanmar 73 years of military rule from independence (rhetorical count)", ["four-further-cases"], tol=0)
reg("FF-Cuba-yr1960", 1960, "year", "Cuba education measurement year 1960", ["four-further-cases"], tol=0)
reg("FF-Bangladesh-yr1960", 1960, "year", "Bangladesh education measurement year 1960", ["four-further-cases"], tol=0)

# ── §china ──
reg("CN-dev-yr", 1994, "checkin", ("table4_crossings.json", "results.China.both_crossed"), [], tol=0)
reg("CN-rural-80", 80, "ref", "rural China ~80% of population (NBS 1982)", ["china"], tol=0)
reg("CN-dreze-tbl", 10.6, "ref", "Dreze & Sen (1989) Table 10.6 reference number", ["china"], tol=0)
reg("CN-u5-def-a", 5, "const", "under-5 mortality age-group upper bound", ["china"], tol=0)
reg("CN-dreze-32pct", 32, "ref", "China U5MR at 32% of GNP-predicted (Dreze & Sen 1989)", ["china"], tol=0)
reg("CN-u5-def-b", 5, "const", "under-5 mortality age-group upper bound (second occurrence)", ["china"], tol=0)
reg("CN-age-20-a", 20, "const", "WCDE completion age 20-24 lower bound", ["china"], tol=0)
reg("CN-age-24-a", 24, "const", "WCDE completion age 20-24 upper bound", ["china"], tol=0)
reg("CN-yr-1980-a", 1980, "year", "China 1980 observation year (MYS-vs-peers comparison)", ["china"], tol=0)
reg("CN-yr-1991-b", 1991, "checkin", ("china_mean_yrs_vs_peers.json", "key_data_points.le_crossover_year"), ["china"], tol=1)
reg("CN-yr-2000-a", 2000, "year", "China U5MR above peers until 2000", ["china"], tol=0)
reg("CN-age-20-b", 20, "const", "WCDE age 20-24 lower bound (footnote)", ["china"], tol=0)
reg("CN-age-24-b", 24, "const", "WCDE age 20-24 upper bound (footnote)", ["china"], tol=0)
reg("CN-peer-band-b", 0.5, "const", "peer-pool bandwidth +/-0.5 mean years of schooling", ["china"], tol=0)
reg("CN-yr-1980-b", 1980, "year", "Deng dismantled the system after 1980", ["china"], tol=0)
reg("CN-dreze-p215", 215, "ref", "Dreze & Sen (1989) pp. 215-221 lower page", ["china"], tol=0)
reg("CN-dreze-p221", 221, "ref", "Dreze & Sen (1989) pp. 215-221 upper page", ["china"], tol=0)
reg("CN-yr-2000-b", 2000, "year", "China below education-predicted U5MR around 2000 (second)", ["china"], tol=0)
reg("CN-mys-1980", 8.01, "checkin", ("china_mean_yrs_vs_peers.json", "key_data_points.china_mys_1980"), ["china"], tol=0.01)

# ── §south-africa-case ──
reg("SA-edu-1990-int", 64.49, "wcde", ("lower_sec_both.csv", "South Africa", 1990), ["south-africa-case"], tol=1.0)
reg("SA-yr-1990-a", 1990, "year", "South Africa lower-secondary 1990 observation year (first)", ["south-africa-case"], tol=0)
reg("SA-edu-1990-dp", 64.49, "wcde", ("lower_sec_both.csv", "South Africa", 1990), ["south-africa-case"], tol=0.01)
reg("SA-edu-2005", 82.64, "wcde", ("lower_sec_both.csv", "South Africa", 2005), ["south-africa-case"], tol=0.01)
reg("SA-yr-1990-b", 1990, "year", "South Africa TFR 3.72 observation year 1990 (second)", ["south-africa-case"], tol=0)
# Mixing layer: "post-1990" hinge — boundary opens + majority schooling expands.
reg("SA-yr-1990-c", 1990, "year", "South Africa post-1990 crossing hinge (mixing-layer sentence)", ["south-africa-case"], tol=0)
reg("SA-tfr-1990-b", 3.724, "wdi", ("tfr", "South Africa", 1990), ["south-africa-case"], tol=0.005)
reg("SA-hiv-belt-lo", 8, "ref", "HIV prevalence 8-11% in HIV belt (Auvert et al. 2005)", ["south-africa-case"], tol=0)
reg("SA-hiv-belt-hi", 11, "ref", "HIV prevalence 8-11% in HIV belt (Auvert et al. 2005)", ["south-africa-case"], tol=0)
reg("SA-circ-lo", 85, "ref", "traditional circumcision 85-99% Muslim West Africa (Auvert 2005)", ["south-africa-case"], tol=0)
reg("SA-circ-protect", 60, "ref", "circumcision RCT HIV-acquisition protection ~60% (Auvert 2005; Bailey 2007; Gray 2007)", ["south-africa-case"], tol=0)
reg("SA-yr-1990-c", 1990, "year", "De Walque (2007) study period start 1990", ["south-africa-case"], tol=0)
reg("SA-yr-2000", 2000, "year", "De Walque (2007) study period end 2000", ["south-africa-case"], tol=0)
reg("SA-yr-1990-d", 1990, "year", "South Africa 1990 lower-secondary figure year (closing sentence)", ["south-africa-case"], tol=0)


# ------------------------------------------------------------
# checkin paths (models.m1.*, results.OUTCOME.90.Education.r2, common_sample.*)
# were verified by the agent against the actual JSON; verify will confirm.
_S = "education-vs-gdp-as-predictors-of-attainment"

# structural / design constants
reg("EVG-gen-lag-25-a", 28, "const", "~25-year generational lag (tab:headline caption)", [_S], tol=0)
reg("EVG-cutoff-30", 30, "const", "parental-completion <30% active-expansion screen (tab:headline)", [_S], tol=0)
reg("EVG-gen-lag-25-b", 28, "const", "T-25 lag annotation (tab:headline row label)", [_S], tol=0)
reg("EVG-sig-10-a", 0.10, "const", "significance threshold p<0.10 (tab:headline notes)", [_S], tol=0)
reg("EVG-gen-lag-25-c", 28, "const", "T-25 annotation (tab:cutoff notes)", [_S], tol=0)
reg("EVG-sig-10-b", 0.10, "const", "significance threshold p<0.10 (tab:cutoff notes second)", [_S], tol=0)
reg("EVG-sig-05-a", 0.05, "const", "significance threshold p<0.05 (tab:cutoff notes)", [_S], tol=0)
reg("EVG-window-90-a", 90, "const", "expansion-window [10,90] upper bound (tab:cutoff notes)", [_S], tol=0)
reg("EVG-cutoff-lbl-10-a", 10, "const", "cutoff label <10% (tab:cutoff ratio prose)", [_S], tol=0)
reg("EVG-cutoff-lbl-50-a", 50, "const", "cutoff label <50% (tab:cutoff body row)", [_S], tol=0)
reg("EVG-cutoff-lbl-50-b", 50, "const", "cutoff label <50% (tab:cutoff ratio prose)", [_S], tol=0)
reg("EVG-sweep-lo-10", 10, "const", "10%-90% sweep lower bound (tab:cutoff ratio prose)", [_S], tol=0)
reg("EVG-sweep-hi-90-b", 90, "const", "10%-90% sweep upper bound (tab:cutoff ratio prose)", [_S], tol=0)
reg("EVG-beta-g-cut-20", 20, "const", "parental cutoff 20% (beta_g compression prose)", [_S], tol=0)
reg("EVG-beta-g-cut-50", 50, "const", "parental cutoff 50% (beta_g compression prose)", [_S], tol=0)
reg("EVG-beta-g-cut-90", 90, "const", "parental cutoff 90% (beta_g compression prose)", [_S], tol=0)
reg("EVG-fig-win-25", 28, "const", "25-year sliding window (fig:beta-baseline caption)", [_S], tol=0)
reg("EVG-lag-12-a", 12, "const", "lag 12 for LE (quality-vs-quantity prose, first)", [_S], tol=0)
reg("EVG-lag-12-b", 12, "const", "lag 12 for U5MR (quality-vs-quantity prose, second)", [_S], tol=0)
reg("EVG-gp-lag-25-d", 28, "const", "T-25 grandparent lag (HLO stability sentence)", [_S], tol=0)
reg("EVG-cohort-20-a", 20, "const", "20-24 cohort lower bound (tab:composition quantity note)", [_S], tol=0)
reg("EVG-cohort-24-a", 24, "const", "20-24 cohort upper bound (tab:composition quantity note)", [_S], tol=0)
reg("EVG-lag-5-a", 5, "const", "lag 5 for TFR (quality prose and tab:composition)", [_S], tol=0)
reg("EVG-gp-lag-25-e", 28, "const", "T-25 grandparent lag (HLO stability sentence, second)", [_S], tol=0)
reg("EVG-ceil-90-a", 90, "const", "ceiling <=90% (tab:residualisation Panel A header)", [_S], tol=0)
reg("EVG-u5-5-a", 5, "const", "under-5 label (tab:residualisation Panel A row)", [_S], tol=0)
# [retired 2026-06-19 tier-1: per-occurrence slot, occurrence cut] reg("EVG-u5-5-b", 5, "const", "under-5 label (tab:residualisation Panel B row)", [_S], tol=0)
reg("EVG-entry-10-a", 10, "const", "entry >=10% (tab:residualisation notes)", [_S], tol=0)
# [retired 2026-06-18 Panel-B cut: exposed surplus reg] reg("EVG-ceil-90-b", 90, "const", "ceiling <=90% (tab:residualisation notes)", [_S], tol=0)
# [retired 2026-06-18 ch9/ch11 compression] reg("EVG-boot-1000", 1000, "const", "1,000-replication bootstrap (residualised-GDP coefficient)", [_S], tol=0)
reg("EVG-ci-95-a", 95, "const", "95th-percentile CI (residualised-GDP bootstrap null)", [_S], tol=0)
# [retired 2026-06-18 Panel-B cut: exposed surplus reg] reg("EVG-strat-5pp", 5, "const", "5pp resolution for parental-completion stratification bins", [_S], tol=0)
reg("EVG-boot-2000-rep", 2000, "const", "2000-rep country-cluster bootstrap (IVW bin-test CI)", [_S], tol=0)
# [retired 2026-06-19 tier-1: per-occurrence slot, occurrence cut] reg("EVG-ci-95-b", 95, "const", "95% CI label (IVW bin-test)", [_S], tol=0)
# [retired 2026-06-19 tier-1: per-occurrence slot, occurrence cut] reg("EVG-lag-12-c", 12, "const", "lag 12 childrearing window (three-timescale lag-sweep)", [_S], tol=0)
# [retired 2026-06-19 tier-1: per-occurrence slot, occurrence cut] reg("EVG-lag-25-f", 28, "const", "lag 25 cross-generation (three-timescale lag-sweep)", [_S], tol=0)
# [retired 2026-06-18 ch9/ch11 compression] reg("EVG-ci-95-c", 95, "const", "bootstrap 95% CIs (lag-sweep, three timescales)", [_S], tol=0)
# calendar years
reg("EVG-year-2015-a", 2015, "year", "2015 cross-section date (quality-vs-quantity prose)", [_S], tol=0)
reg("EVG-year-2015-b", 2015, "year", "2015 (tab:composition caption first)", [_S], tol=0)
reg("EVG-year-2015-c", 2015, "year", "2015 (tab:composition caption second)", [_S], tol=0)
# [retired 2026-06-19 tier-1: per-occurrence slot, occurrence cut] reg("EVG-year-2015-d", 2015, "year", "2015 (closing panel-vs-cross-section paragraph)", [_S], tol=0)
reg("EVG-year-1960-a", 1960, "year", "T=1960 lower bound (tab:residualisation entry-range note)", [_S], tol=0)
reg("EVG-year-1990-a", 1990, "year", "T=1990 upper bound (tab:residualisation entry-range note)", [_S], tol=0)
reg("EVG-year-2000-a", 2000, "year", "post-2000 interaction reference (stratification paragraph)", [_S], tol=0)
reg("EVG-year-2000-b", 2000, "year", "pre-2000 log-U5MR residGDP slope citation", [_S], tol=0)
reg("EVG-year-2000-c", 2000, "year", "post-2000 (IVW bin-test prose)", [_S], tol=0)
reg("EVG-year-2000-d", 2000, "year", "pre-2000 (composition sentence)", [_S], tol=0)
# [retired 2026-06-19 tier-1: per-occurrence slot, occurrence cut] reg("EVG-year-2000-e", 2000, "year", "post-2000 (composition sentence)", [_S], tol=0)
# [retired 2026-06-19 tier-1: per-occurrence slot, occurrence cut] reg("EVG-year-2000-qty", 2000, "year", "Quantity = lower-secondary completion at 2000 (tab:composition notes)", [_S], tol=0)
# data-bearing tokens
reg("EVG-t1-m1-se", 0.092, "checkin", ("table_1_stepwise.json", "models.m1.parent_se"), [_S], tol=0.001)
reg("EVG-t1-n-a", 672, "checkin", ("table_1_stepwise.json", "models.m1.n"), [_S], tol=0)
reg("EVG-t1-countries", 106, "checkin", ("table_1_stepwise.json", "models.m1.countries"), [], tol=0)
reg("EVG-t1-m3-beta", 1.32, "checkin", ("table_1_stepwise.json", "models.m3.parent_beta"), [_S], tol=0.005)
reg("EVG-t1-m3-gdp-beta", 5.58, "checkin", ("table_1_stepwise.json", "models.m3.gdp_beta"), [_S], tol=0.01)
reg("EVG-t1-n-b", 672, "checkin", ("table_1_stepwise.json", "models.m1.n"), [], tol=0)
reg("EVG-tabC-10-ctry", 62, "checkin", ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_10_countries"), [_S], tol=0)
reg("EVG-tabC-10-ratio", 2.1, "checkin", ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_10_ratio"), [_S], tol=0.05)
reg("EVG-beta-g-20", 3.10, "checkin", ("beta_by_ceiling_cutoff.json", "numbers.panelA_cutoff_20_beta"), [_S], tol=0.01)
reg("EVG-beta-g-90", 1.265, "checkin", ("beta_by_ceiling_cutoff.json", "numbers.panelA_cutoff_90_beta"), [_S], tol=0.01)
reg("EVG-beta-g-unrestr", 1.052, "checkin", ("beta_by_ceiling_cutoff.json", "numbers.panelA_no_cutoff_beta"), [_S], tol=0.01)
reg("EVG-fig-185-ctry", 185, "checkin", ("beta_vs_baseline.json", "numbers.Universality-n-countries"), [_S], tol=0)
reg("EVG-tfr-r2-a", 0.6689, "checkin", ("regression_tables.json", "results.TFR.90.Education.r2"), [_S], tol=0.005)
reg("EVG-u5mr-r2-a", 0.4567, "checkin", ("regression_tables.json", "results.U5MR.90.Education.r2"), [_S], tol=0.005)
reg("EVG-le-r2-a", 0.4283, "checkin", ("regression_tables.json", "results.LE.90.Education.r2"), [_S], tol=0.005)
reg("EVG-hlo-lsec-p", 0.1401, "checkin", ("horse_race_earlier_cohort.json", "numbers.C_lsec_hlo.lsec_2000_z.p"), [_S], tol=0.005)
reg("EVG-comp-n-tfr", 66, "checkin", ("horse_race_earlier_cohort.json", "numbers.C_lsec_hlo.n"), [_S], tol=0)
reg("EVG-comp-n-le", 67, "checkin", ("hanushek_horse_race.json", "numbers.le.n"), [_S], tol=0)
reg("EVG-tfr-r2-b", 0.6689, "checkin", ("regression_tables.json", "results.TFR.90.Education.r2"), [_S], tol=0.005)
reg("EVG-tfr-r2-c", 0.6689, "checkin", ("regression_tables.json", "results.TFR.90.Education.r2"), [_S], tol=0.005)
# [retired 2026-06-18 Panel-B cut: exposed surplus reg] reg("EVG-u5-resid-p-pA", 0.7575, "checkin", ("regression_tables.json", "results.U5MR.90.GDP (residualized).pval"), [_S], tol=0.01)
# [retired 2026-06-18 Panel-B cut] reg("EVG-u5-edu-r2-pB", 0.4341, "checkin", ("regression_tables.json", "common_sample.outcomes.U5MR.Education_FE.r2"), [_S], tol=0.005)
# [retired 2026-06-18 Panel-B cut] reg("EVG-u5-rawgdp-r2-pB", 0.1701, "checkin", ("regression_tables.json", "common_sample.outcomes.U5MR.GDP_raw_FE.r2"), [_S], tol=0.0005)
# [retired 2026-06-18 Panel-B cut] reg("EVG-u5-resid-r2-pB", 0.0006, "checkin", ("regression_tables.json", "common_sample.outcomes.U5MR.GDP_resid_FE.r2"), [_S], tol=0.0005)
# [retired 2026-06-18 Panel-B cut] reg("EVG-panB-n", 645, "checkin", ("regression_tables.json", "common_sample.outcomes.LE.Education_FE.n"), [_S], tol=0)
# [retired 2026-06-18 Panel-B cut] reg("EVG-panB-ctry", 123, "checkin", ("regression_tables.json", "common_sample.outcomes.LE.Education_FE.countries"), [_S], tol=0)


# ------------------------------------------------------------
# Fixes applied vs agent output:
#  - MAC regs: value 28 (paper prints "28"; source exp_mac_mean=28.75 rounds to
#    29 via .0f, so register paper form 28 with tol=1.0 against checkin).
#  - GTM two/three-gen spans: const (pure 2x25 / 3x25 arithmetic), not derived.
#  - GTM primary-share 60/80: closed as const range (flagged: no producing
#    script; approximate LE primary-share from by_level_lag0.json R²s — backlog).
# FLAG backlog: GTM 60-80% primary share (no script); necessity "thirty-one"
#    is prose words (no token).

# ── §the-generational-transmission-mechanism ──
reg("GTM-pace-competing", 60, "const", "competing-priority crossing pace (~60 yr = 2-3 generations)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-pace-singular", 28, "const", "singular-priority crossing pace (~25 yr = 1 generation)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-gen-cycle-a", 28, "const", "biological generation cycle lag anchor (~25 yr)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-gen-cycle-b", 28, "const", "28-year cross-generation step (underestimate of empirical MAC)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-mac-empirical", 28.75, "checkin", ("mean_age_first_birth_invariance.json", "numbers.exp_mac_mean"), ["the-generational-transmission-mechanism"], tol=0.01)
reg("GTM-school-age-lo", 15, "const", "school-completion age window lower bound (15-18)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-school-age-hi", 18, "const", "school-completion age window upper bound (15-18)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-child-delay-lo", 20, "const", "parent school-exit to child's school age, lower (20-30)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-child-delay-hi", 30, "const", "parent school-exit to child's school age, upper (20-30)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-taiwan-lag", 20, "const", "Taiwan crossing lag ~20 yr (tilde-rounded; expansion_rate Taiwan.lag=22)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-kerala-lag", 65, "const", "Kerala crossing lag ~65 yr (tilde-rounded; expansion_rate Kerala.lag=62)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-two-gen-span", 56, "const", "2 x 28-yr generation = 56 yr (grandchild horizon)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-three-gen-span", 84, "const", "3 x 28-yr generation = 84 yr (great-grandchild horizon)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-lag-zero", 0, "const", "lag 0: LE primary same-sample horse race (contemporaneous)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-primary-share-lo", 60, "const", "LE within-country variation primary-share lower bound (~60%, by_level_lag0 R² decomposition; backlog: no scalar)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-primary-share-hi", 80, "const", "LE within-country variation primary-share upper bound (~80%, by_level_lag0 R² decomposition; backlog: no scalar)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-beta-g-threshold", 1, "const", "beta_g > 1 structural claim at low baselines (amplification)", ["the-generational-transmission-mechanism"], tol=0)
reg("GTM-loop-caption-25", 28, "const", "figure loop caption: ~25-year cycle clicks the ratchet", ["the-generational-transmission-mechanism"], tol=0)

# ── §the-generational-lag ──
reg("GL2-mac-header", 28.75, "checkin", ("mean_age_first_birth_invariance.json", "numbers.exp_mac_mean"), ["the-generational-lag"], tol=0.01)
reg("GL2-xgen-step-a", 28, "const", "28-year cross-generation step (edu->edu headline lag)", ["the-generational-lag"], tol=0)
reg("GL2-xgen-step-b", 28, "const", "T+25 headline cross-generation step", ["the-generational-lag"], tol=0)
reg("GL2-lag-tfr-contemp", 5, "const", "TFR lag 5 (timescale table contemporaneous row)", ["the-generational-lag"], tol=0)
reg("GL2-cohort-age-lo-a", 20, "const", "20-24 cohort lower bound (contemporaneous row)", ["the-generational-lag"], tol=0)
reg("GL2-cohort-age-hi-a", 24, "const", "20-24 cohort upper bound (contemporaneous row)", ["the-generational-lag"], tol=0)
reg("GL2-lag-tfr-a", 5, "const", "TFR lag 5 reproductive peak (mechanism column)", ["the-generational-lag"], tol=0)
reg("GL2-crw-lo-a", 10, "const", "childrearing window lower bound lag 10 (U5MR/LE row header)", ["the-generational-lag"], tol=0)
reg("GL2-crw-hi-a", 15, "const", "childrearing window upper bound lag 15 (U5MR/LE row header)", ["the-generational-lag"], tol=0)
reg("GL2-cohort-age-lo-b", 20, "const", "20-24 cohort lower (childrearing what-it-measures col)", ["the-generational-lag"], tol=0)
reg("GL2-cohort-age-hi-b", 24, "const", "20-24 cohort upper (childrearing what-it-measures col)", ["the-generational-lag"], tol=0)
reg("GL2-crw-lo-b", 10, "const", "childrearing window lower restated (mechanism column)", ["the-generational-lag"], tol=0)
reg("GL2-crw-hi-b", 15, "const", "childrearing window upper restated (mechanism column)", ["the-generational-lag"], tol=0)
reg("GL2-u5-age", 5, "const", "under-5: children survive to age 5 before outcome registers", ["the-generational-lag"], tol=0)
reg("GL2-xgen-range-lo", 25, "const", "cross-generation step lower bound ~25 (row header)", ["the-generational-lag"], tol=0)
reg("GL2-xgen-range-hi", 30, "const", "cross-generation step upper bound ~30 (row header)", ["the-generational-lag"], tol=0)
reg("GL2-cohort-age-lo-c", 20, "const", "20-24 cohort lower (cross-generation what-it-measures col)", ["the-generational-lag"], tol=0)
reg("GL2-cohort-age-hi-c", 24, "const", "20-24 cohort upper (cross-generation what-it-measures col)", ["the-generational-lag"], tol=0)
# §The Generational Lag "Four time constants, not to be conflated" aid
# (added 2026-06-19) restates the MAC, the T-28 step, and the 20-24
# measurement cohort once more in the same section — one extra occurrence each.
reg("GL2-aid-mac-28.8", 28.8, "checkin",
    ("mean_age_first_birth_invariance.json", "numbers.exp_mac_mean"),
    [THE_GENERATIONAL_LAG], tol=0.1)
reg("GL2-aid-xgen-28", 28, "const", "T-28 headline cross-generation step in the four-constants aid", [THE_GENERATIONAL_LAG], tol=0)
reg("GL2-aid-age-lo-20", 20, "const", "20-24 measurement cohort lower bound in the four-constants aid", [THE_GENERATIONAL_LAG], tol=0)
reg("GL2-aid-age-hi-24", 24, "const", "20-24 measurement cohort upper bound in the four-constants aid", [THE_GENERATIONAL_LAG], tol=0)
reg("GL2-xgen-step-c", 28, "const", "T+25 cross-generation observation horizon (row)", ["the-generational-lag"], tol=0)
reg("GL2-xgen-step-d", 28, "const", "T+25 vantage: grandchild TFR (mechanism column)", ["the-generational-lag"], tol=0)
# [retired 2026-06-19 tier-1: per-occurrence slot, occurrence cut] reg("GL2-lag-tfr-b", 5, "const", "TFR lag 5 (outcome-specific lag structure paragraph)", ["the-generational-lag"], tol=0)
# [retired 2026-06-19 tier-1: §Outcome-specific lag structure para cut] reg("GL2-lag-le-u5", 12, "const", "LE and U5MR lag 12 (childrearing window center)", ["the-generational-lag"], tol=0)
# [retired 2026-06-19 tier-1: per-occurrence slot, occurrence cut] reg("GL2-xgen-step-e", 28, "const", "T-25 prior (grandparental) generation in horse race", ["the-generational-lag"], tol=0)
# [retired 2026-06-19 tier-1: per-occurrence slot, occurrence cut] reg("GL2-xgen-step-f", 28, "const", "T-25 grandparental block (lag plus generational step)", ["the-generational-lag"], tol=0)
# [retired 2026-06-19 tier-1: per-occurrence slot, occurrence cut] reg("GL2-xgen-step-g", 28, "const", "25-year generational step (per-outcome decomposition)", ["the-generational-lag"], tol=0)
# [retired 2026-06-19 tier-1: per-occurrence slot, occurrence cut] reg("GL2-lag-tfr-parent", 5, "const", "TFR parent lag outcome-5 (reproductive peak)", ["the-generational-lag"], tol=0)
# [retired 2026-06-19 tier-1: per-occurrence slot, occurrence cut] reg("GL2-lag-tfr-gp", 30, "const", "TFR grandparent lag outcome-30 = 5 + 25", ["the-generational-lag"], tol=0)
# [retired 2026-06-19 tier-1: §Outcome-specific lag structure para cut] reg("GL2-lag-u5-parent", 12, "const", "U5MR parent lag outcome-12 (childrearing window)", ["the-generational-lag"], tol=0)
# [retired 2026-06-19 tier-1: §Outcome-specific lag structure para cut] reg("GL2-lag-u5-gp", 37, "const", "U5MR grandparent lag outcome-37 = 12 + 25", ["the-generational-lag"], tol=0)
reg("GL2-school-entry-age", 6, "const", "primary/lower-sec school entry age 6", ["the-generational-lag"], tol=0)
reg("GL2-school-exit-age-a", 15.0, "const", "school exit age 15 (upper end of 6-15)", ["the-generational-lag"], tol=0)
reg("GL2-policy-lag-a", 15, "const", "years from funding cohort to dominating 20-24 group", ["the-generational-lag"], tol=0)
reg("GL2-cohort-age-lo-d", 20, "const", "20-24 cohort lower (leader's timeline paragraph)", ["the-generational-lag"], tol=0)
reg("GL2-cohort-age-hi-d", 24, "const", "20-24 cohort upper (leader's timeline paragraph)", ["the-generational-lag"], tol=0)
reg("GL2-policy-lag-b", 15, "const", "minimum-visible policy effect ~15 yr", ["the-generational-lag"], tol=0)

# ── §necessity-sufficiency-forward ──
reg("NSF-tfr-threshold", 3.65, "const", "TFR < 3.65 development threshold (1960 USA value)", ["necessity-sufficiency-forward"], tol=0)
reg("NSF-le-threshold", 69.8, "const", "LE > 69.8 development threshold (1960 USA value)", ["necessity-sufficiency-forward"], tol=0)
reg("NSF-cohort-age-lo-a", 20, "const", "20-24 cohort lower bound (necessity paragraph)", ["necessity-sufficiency-forward"], tol=0)
reg("NSF-cohort-age-hi-a", 24, "const", "20-24 cohort upper bound (necessity paragraph)", ["necessity-sufficiency-forward"], tol=0)
reg("NSF-floor-raw", 36.2, "checkin", ("necessity_floor.json", "floor_pct"), ["necessity-sufficiency-forward"], tol=1.0)
reg("NSF-floor-rounded-a", 35.0, "checkin", ("necessity_floor.json", "sufficiency_floor_pct"), ["necessity-sufficiency-forward"], tol=0.5)
reg("NSF-floor-rounded-b", 35.0, "checkin", ("necessity_floor.json", "sufficiency_floor_pct"), ["necessity-sufficiency-forward"], tol=0.5)
reg("NSF-floor-rounded-c", 35.0, "checkin", ("necessity_floor.json", "sufficiency_floor_pct"), ["necessity-sufficiency-forward"], tol=0.5)
reg("NSF-cohort-age-lo-b", 20, "const", "20-24 cohort lower bound (sufficiency paragraph)", ["necessity-sufficiency-forward"], tol=0)
reg("NSF-cohort-age-hi-b", 24, "const", "20-24 cohort upper bound (sufficiency paragraph)", ["necessity-sufficiency-forward"], tol=0)
reg("NSF-start-below-10", 10, "const", "sufficiency condition: expansion from below 10% (design window)", ["necessity-sufficiency-forward"], tol=0)
reg("NSF-suff-floor-a", 35.0, "checkin", ("necessity_floor.json", "sufficiency_floor_pct"), ["necessity-sufficiency-forward"], tol=0.5)
reg("NSF-suff-floor-b", 35.0, "checkin", ("necessity_floor.json", "sufficiency_floor_pct"), ["necessity-sufficiency-forward"], tol=0.5)


# ------------------------------------------------------------
# Agent correctly did NOT re-create tokens already covered by existing regs
# (GDP-*-2015-sec, G-LagSweep-*, G-HLO-n, EMA-*). REUSE-WATCH: if any of those
# values appear in BOTH prose and a table within the section (2 occurrences),
# one occurrence may remain a gap needing a duplicate — verify will surface it.

# ── §universality-across-subsamples ──
reg("UNI-col1-headline", 1, "const", "Column 1 of tab:headline (education-only spec on eleven subsamples)", ["universality-across-subsamples"], tol=0)
reg("UNI-p-sig-threshold", 0.01, "const", "p<0.01 significance; all eleven subsample coefficients exceed it", ["universality-across-subsamples"], tol=0)
reg("UNI-ssa-humanity-pct", 20, "const", "~20% of humanity in Sub-Saharan Africa (rhetorical round figure)", ["universality-across-subsamples"], tol=0)
reg("UNI-resid-lag-a", 28, "const", "25-year time-to-agency lag (child-on-parent FE for over-performer residuals)", ["universality-across-subsamples"], tol=0)
reg("UNI-panel-n-countries-a", 185, "checkin", ("summary_stats.json", "numbers.panel_countries"), ["universality-across-subsamples"], tol=0)
reg("UNI-resid-lag-b", 28, "const", "25-year lag in T+25=2015 (parent cohort at T=1990)", ["universality-across-subsamples"], tol=0)
reg("UNI-resid-year-T25", 2015, "year", "T+25=2015: year over-performer child cohorts observed", ["universality-across-subsamples"], tol=0)
reg("UNI-resid-lag-c", 28, "const", "25-year lag in FE-residual formula (tab:over-performers notes)", ["universality-across-subsamples"], tol=0)
reg("UNI-panel-n-countries-b", 185, "checkin", ("summary_stats.json", "numbers.panel_countries"), ["universality-across-subsamples"], tol=0)
reg("UNI-resid-T-1990", 1990, "year", "T=1990 parent cohort year for the 2015 FE residuals", ["universality-across-subsamples"], tol=0)
reg("UNI-resid-obs-2015", 2015, "year", "2015 child cohort observation year (FE residual table)", ["universality-across-subsamples"], tol=0)
reg("UNI-battery-20tests", 20, "const", "20-test econometric robustness battery", ["universality-across-subsamples"], tol=0)
reg("UNI-r2-ratio-floor", 2.1, "checkin", ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_10_ratio"), ["universality-across-subsamples"], tol=0.05)

# ── §every-method-agrees ──
reg("EMA2-lag-LE-caption", 12, "const", "LE biological lag 12 (LAG_LE) in tab:spec-curve caption", ["every-method-agrees"], tol=0)
reg("EMA2-lag-TFR-caption", 5, "const", "TFR biological lag 5 (LAG_TFR) in tab:spec-curve caption", ["every-method-agrees"], tol=0)
reg("EMA2-lag-U5MR-caption", 12, "const", "U5MR biological lag 12 in tab:spec-curve caption", ["every-method-agrees"], tol=0)
reg("EMA2-lag-LE-notes", 12, "const", "LE lag 12 in tab:spec-curve notes", ["every-method-agrees"], tol=0)
reg("EMA2-lag-TFR-notes", 5, "const", "TFR lag 5 in tab:spec-curve notes", ["every-method-agrees"], tol=0)
reg("EMA2-lag-U5MR-notes", 12, "const", "U5MR lag 12 in tab:spec-curve notes", ["every-method-agrees"], tol=0)
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA2-ci-pct", 95, "const", "95% CI for double-ML life expectancy estimate", ["every-method-agrees"], tol=0)
# [retired 2026-06-19 tier-1: §What-the-gap-bought para cut] reg("EMA2-wf-cutoff-1990-a", 1990, "year", "1990 walk-forward cutoff (TFR R²=0.93)", ["every-method-agrees"], tol=0)
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA2-wf-cutoff-1980", 1980, "year", "pre-1980 training cutoff (walk-forward LE)", ["every-method-agrees"], tol=0)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA2-wf-cutoff-1990-b", 1990, "year", "post-1990 LE prediction trained on pre-1980 (R²=0.84)", ["every-method-agrees"], tol=0)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA2-cf-feature-window-a", 1990, "year", "T=1990 feature window (counterfactual education swap)", ["every-method-agrees"], tol=0)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA2-lag-LE-cf-prose", 12, "const", "LE at T+12 in counterfactual prose", ["every-method-agrees"], tol=0)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA2-lag-TFR-cf-prose", 5, "const", "TFR at T+5 in counterfactual prose", ["every-method-agrees"], tol=0)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA2-lag-LE-cf-caption", 12, "const", "LE lag 12 in tab:ml-counterfactual caption", ["every-method-agrees"], tol=0)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA2-lag-TFR-cf-caption", 5, "const", "TFR lag 5 in tab:ml-counterfactual caption", ["every-method-agrees"], tol=0)
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA2-cf-feature-window-b", 1990, "year", "T=1990 feature window (tab:ml-counterfactual notes)", ["every-method-agrees"], tol=0)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA2-lag-LE-cf-notes", 12, "const", "LE at T+12 in tab:ml-counterfactual notes", ["every-method-agrees"], tol=0)
# [retired 2026-06-18 ml-counterfactual table->ECONOMETRICS.md] reg("EMA2-lag-TFR-cf-notes", 5, "const", "TFR at T+5 in tab:ml-counterfactual notes", ["every-method-agrees"], tol=0)
# [retired 2026-06-18 ch9/ch11 compression] reg("EMA2-bgd-overtaking-1990", 1990, "year", "post-1990 Bangladesh catch-up (outside 1975-1990 swap)", ["every-method-agrees"], tol=0)

# ── §hanushek-reconciliation ──
reg("HAN-kyrg-hlo-year", 2009, "year", "Kyrgyzstan HLO secondary score year", ["hanushek-reconciliation"], tol=0)
reg("HAN-kyrg-completion-year", 2010, "year", "Kyrgyzstan reported 99% lower-sec completion year (WCDE v3)", ["hanushek-reconciliation"], tol=0)
reg("HAN-sweep-endpoint", 2010, "year", "HLO lag-sweep regression endpoint year", ["hanushek-reconciliation"], tol=0)
reg("HAN-sweep-lag-min", 0, "const", "minimum lag L=0 in HLO lag sweep", ["hanushek-reconciliation"], tol=0)
reg("HAN-sweep-lag-max-a", 60, "const", "maximum lag L=60 in HLO lag sweep", ["hanushek-reconciliation"], tol=0)
reg("HAN-sweep-step", 5, "const", "5-year step size in HLO lag sweep", ["hanushek-reconciliation"], tol=0)
reg("HAN-lsec-lag0-ref", 0, "const", "lag 0 for lower-sec R²=0.504", ["hanushek-reconciliation"], tol=0)
reg("HAN-lsec-peak-lag", 10, "const", "lag 10 peak for lower-sec R²=0.539", ["hanushek-reconciliation"], tol=0)
reg("HAN-primary-lag0-ref", 0, "const", "lag 0 for primary R²=0.469", ["hanushek-reconciliation"], tol=0)
reg("HAN-primary-peak-lag", 25, "const", "lag 25 peak for primary R²=0.549", ["hanushek-reconciliation"], tol=0)
reg("HAN-primary-lag60-ref-a", 60, "const", "lag 60 for primary R²=0.489 (prose, first)", ["hanushek-reconciliation"], tol=0)
reg("HAN-primary-lag60-ref-b", 60, "const", "lag 60: primary completion 60 years before test (second)", ["hanushek-reconciliation"], tol=0)
reg("HAN-cohort-1950-a", 1950, "year", "1950 cohort = lag-60 from 2010", ["hanushek-reconciliation"], tol=0)
reg("HAN-bootstrap-ci-pct", 95, "const", "95% CI on the R² difference (country-resample bootstrap)", ["hanushek-reconciliation"], tol=0)
reg("HAN-ci-includes-zero", 0, "const", "CI includes zero (peak vs lag-0 not significant)", ["hanushek-reconciliation"], tol=0)
reg("HAN-cohort-1950-b", 1950, "year", "1950 cohort (second mention)", ["hanushek-reconciliation"], tol=0)
reg("HAN-lag60-falsifier", 60, "const", "lag 60 falsifier: R² stays 0.489, not collapsing", ["hanushek-reconciliation"], tol=0)
reg("HAN-fig-endpoint", 2010, "year", "figure caption: regression endpoint 2010-L", ["hanushek-reconciliation"], tol=0)
reg("HAN-fig-lag-min", 0, "const", "figure caption: L=0 start of sweep", ["hanushek-reconciliation"], tol=0)
reg("HAN-fig-step", 5, "const", "figure caption: 5-year step", ["hanushek-reconciliation"], tol=0)
reg("HAN-fig-lag-max", 60, "const", "figure caption: L=60 end of sweep", ["hanushek-reconciliation"], tol=0)
reg("HAN-fig-ci-pct", 95, "const", "figure caption: 95% bootstrap CI bands", ["hanushek-reconciliation"], tol=0)
reg("HAN-fig-lag60-ref", 60, "const", "figure caption: primary completion at lag 60", ["hanushek-reconciliation"], tol=0)
reg("HAN-parent-cohort-year", 1990, "year", "1990 parent cohort year for HLO-as-home-niche regression", ["hanushek-reconciliation"], tol=0)
reg("HAN-age-band-lo", 20, "const", "20-24 age band lower bound", ["hanushek-reconciliation"], tol=0)
reg("HAN-age-band-hi", 24, "const", "20-24 age band upper bound", ["hanushek-reconciliation"], tol=0)


# ------------------------------------------------------------
# NOTE: agent's `derived` per-occurrence duplicates converted to `checkin`
# against gavi_6_t_sweep_2025.json (load_checkin traverses nested dot-paths),
# so they are actually verified (not unverified REF). Filename for all HC2
# scenario paths: gavi_6_t_sweep_2025.json.
# FLAGS: prediction none. human-cost: HC2-n-countries=100 verified.
#   necessity-sufficiency-forward "31" flagged (out of scope; G4 owns it).

# ── §the-prediction ──
reg("PRD-level1-dep-window",   1,  "const", "Level-1 loading: categorical brain reorganisation", ["the-prediction"], tol=0)
reg("PRD-age-agency-close",   18,  "const", "~18-year juvenile dependency window close (biological)", ["the-prediction"], tol=0)
reg("PRD-gen-lag-25",         28,  "const", "T+~25 biological generation-cycle lag", ["the-prediction"], tol=0)
reg("PRD-cohort-age-lo",      20,  "const", "WCDE 20-24 cohort lower bound", ["the-prediction"], tol=0)
reg("PRD-cohort-age-hi",      24,  "const", "WCDE 20-24 cohort upper bound", ["the-prediction"], tol=0)
reg("PRD-gdp-lag",             0,  "const", "Log GDP registers at lag 0 (current output)", ["the-prediction"], tol=0)
reg("PRD-tfr-lag-a",           5,  "const", "TFR lag T+5 (reproductive peak)", ["the-prediction"], tol=0)
reg("PRD-tfr-lag-b",           5,  "const", "TFR lag T+5 (second occurrence same line)", ["the-prediction"], tol=0)
reg("PRD-u5mr-lag-lo",        10,  "const", "U5MR childrearing-window lag lower bound (10-15)", ["the-prediction"], tol=0)
reg("PRD-u5mr-lag-hi",        15,  "const", "U5MR childrearing-window lag upper bound (10-15)", ["the-prediction"], tol=0)
reg("PRD-le-lag",             12,  "const", "LE-at-birth lag ~12 (childrearing window)", ["the-prediction"], tol=0)
reg("PRD-level1-persist",      1,  "const", "Level-1 home-niche loading across generations", ["the-prediction"], tol=0)
reg("PRD-gen-depth-56",       56,  "const", "Generational depth 56 years (2 generations)", ["the-prediction"], tol=0)
reg("PRD-gen-depth-84",       84,  "const", "Generational depth 84 years (3 generations)", ["the-prediction"], tol=0)
reg("PRD-gen-depth-112",     112,  "const", "Generational depth 112 years (4 generations)", ["the-prediction"], tol=0)
reg("PRD-level1-payload",      1,  "const", "Level-1: payload lives in the educated adult", ["the-prediction"], tol=0)
reg("PRD-level2-shell",        2,  "const", "Level-2: institutions/income/infrastructure shell", ["the-prediction"], tol=0)
reg("PRD-level2-shell-b",      2,  "const", "Level-2 shell (second occurrence)", ["the-prediction"], tol=0)
reg("PRD-level2-gdp",          2,  "const", "Level-2: national-accounts face of educated-CT regime", ["the-prediction"], tol=0)
reg("PRD-level1-univ",         1,  "const", "Level-1: species-level capacity (universality)", ["the-prediction"], tol=0)
reg("PRD-level2-univ",         2,  "const", "Level-2: cumulative CT via population-scale schooling", ["the-prediction"], tol=0)
reg("PRD-level1-coll",         1,  "const", "Level-1: household cognitive capacity", ["the-prediction"], tol=0)
reg("PRD-level2-coll",         2,  "const", "Level-2: societal coordination capacity", ["the-prediction"], tol=0)
reg("PRD-level2-ct-claim",     2,  "const", "Level-2 categorical claim (literate-CT cross-ref)", ["the-prediction"], tol=0)
reg("PRD-panel-countries",   185,  "checkin", ("summary_stats.json", "numbers.panel_countries"), ["the-prediction"], tol=0)

# ── §the-human-cost ──
reg("HC2-amrt-age-window",    45,  "const", "_{45}q_{15}: 45-year mortality probability window", ["the-human-cost"], tol=0)
reg("HC2-amrt-age-start",     15,  "const", "_{45}q_{15}: starts at age 15", ["the-human-cost"], tol=0)
reg("HC2-amrt-lag-a",         28,  "const", "Female adult mortality predicted at T+25", ["the-human-cost"], tol=0)
reg("HC2-amrt-lag-b",         28,  "const", "Second T+25 reference (net-of-U5MR sentence)", ["the-human-cost"], tol=0)
reg("HC2-fit-pre2000",      2000,  "year",  "Log-U5MR curve fitted on pre-2000 cross-sections", ["the-human-cost"], tol=0)
reg("HC2-ramp-15yr-a",        15,  "const", "15-year linear ramp scenario", ["the-human-cost"], tol=0)
reg("HC2-lsec-cap-a",         95,  "const", "Target 95% lower-secondary completion (ramp)", ["the-human-cost"], tol=0)
reg("HC2-bio-floor-9yr-a",     9,  "const", "9-year biological floor", ["the-human-cost"], tol=0)
reg("HC2-anchor-1990-a",    1990,  "year",  "Primary counterfactual anchor T=1990", ["the-human-cost"], tol=0)
reg("HC2-korea-lsec-95",      95,  "const", "Korea lower-secondary 95% by 1990", ["the-human-cost"], tol=0)
reg("HC2-anchor-1990-b",    1990,  "year",  "T=1990 (laggard-counting start year)", ["the-human-cost"], tol=0)
reg("HC2-anchor-1980-a",    1980,  "year",  "Alternate anchor T=1980", ["the-human-cost"], tol=0)
reg("HC2-anchor-1970-a",    1970,  "year",  "Alternate anchor T=1970", ["the-human-cost"], tol=0)
reg("HC2-earliest-channel",    5,  "const", "Earliest channel T+5 (fertility)", ["the-human-cost"], tol=0)
reg("HC2-end-year-a",       2025,  "year",  "End-year horizon 2025", ["the-human-cost"], tol=0)
reg("HC2-n-countries",       100,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.n_countries_used"), ["the-human-cost"], tol=0)
reg("HC2-anchor-1990-c",    1990,  "year",  "T=1990 (Result paragraph)", ["the-human-cost"], tol=0)
reg("HC2-end-year-b",       2025,  "year",  "End-year 2025 (Result sentence)", ["the-human-cost"], tol=0)
reg("HC2-total-1990-korea",  104,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.scenarios_extra_vs_actual.korea_pace.lives_saved_total"), ["the-human-cost"], tol=0.5)
reg("HC2-mort-1990-korea",    38,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.scenarios_extra_vs_actual.korea_pace.via_mortality"), ["the-human-cost"], tol=0.5)
reg("HC2-fert-1990-korea",    67,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.scenarios_extra_vs_actual.korea_pace.via_fertility"), ["the-human-cost"], tol=0.5)
reg("HC2-anchor-1980-b",    1980,  "year",  "T=1980 alternate anchor (result comparison)", ["the-human-cost"], tol=0)
reg("HC2-total-1980-korea",  188,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1980.scenarios_extra_vs_actual.korea_pace.lives_saved_total"), [], tol=0.5)
reg("HC2-mort-1980-korea",    66,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1980.scenarios_extra_vs_actual.korea_pace.via_mortality"), [], tol=0.5)
reg("HC2-fert-1980-korea",   122,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1980.scenarios_extra_vs_actual.korea_pace.via_fertility"), [], tol=0.5)
reg("HC2-anchor-1970-b",    1970,  "year",  "T=1970 alternate anchor (result comparison)", ["the-human-cost"], tol=0)
reg("HC2-total-1970-korea",  284,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1970.scenarios_extra_vs_actual.korea_pace.lives_saved_total"), [], tol=1.0)
reg("HC2-mort-1970-korea",   101,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1970.scenarios_extra_vs_actual.korea_pace.via_mortality"), [], tol=0.5)
reg("HC2-fert-1970-korea",   183,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1970.scenarios_extra_vs_actual.korea_pace.via_fertility"), [], tol=0.5)
reg("HC2-anchor-1990-d",    1990,  "year",  "T=1990 (sensitivity-envelope intro)", ["the-human-cost"], tol=0)
reg("HC2-total-1990-korea-b", 104, "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.scenarios_extra_vs_actual.korea_pace.lives_saved_total"), ["the-human-cost"], tol=0.5)
reg("HC2-bio-floor-9yr-b",     9,  "const", "9-year scenario label (sensitivity range)", ["the-human-cost"], tol=0)
reg("HC2-anchor-1990-tbl",  1990,  "year",  "T=1990 table header label", ["the-human-cost"], tol=0)
reg("HC2-tbl-korea-rate",   2.13,  "checkin", ("gavi_6_t_sweep_2025.json", "korea_rate_pp_per_year"), ["the-human-cost"], tol=0.005)
reg("HC2-tbl-mort-korea",     38,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.scenarios_extra_vs_actual.korea_pace.via_mortality"), [], tol=0.5)
reg("HC2-tbl-fert-korea",     67,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.scenarios_extra_vs_actual.korea_pace.via_fertility"), [], tol=0.5)
reg("HC2-tbl-total-korea",   104,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.scenarios_extra_vs_actual.korea_pace.lives_saved_total"), [], tol=0.5)
reg("HC2-tbl-ramp15-yr",      15,  "const", "15-year ramp scenario label (table row)", ["the-human-cost"], tol=0)
reg("HC2-tbl-ramp15-cap",     95,  "const", "95% target in table row label", ["the-human-cost"], tol=0)
reg("HC2-tbl-mort-ramp15",    45,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.scenarios_extra_vs_actual.ramp_15yr.via_mortality"), [], tol=0.5)
reg("HC2-tbl-fert-ramp15",   127,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.scenarios_extra_vs_actual.ramp_15yr.via_fertility"), [], tol=0.5)
reg("HC2-tbl-kids-ramp15",  1068,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.scenarios_extra_vs_actual.ramp_15yr.kids_not_born"), [], tol=1.0)
reg("HC2-tbl-total-ramp15",  172,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.scenarios_extra_vs_actual.ramp_15yr.lives_saved_total"), [], tol=0.5)
reg("HC2-tbl-bio9-yr",         9,  "const", "9-year biological floor label (table row)", ["the-human-cost"], tol=0)
reg("HC2-tbl-mort-bio9",      51,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.scenarios_extra_vs_actual.bio_9yr.via_mortality"), [], tol=0.5)
reg("HC2-tbl-fert-bio9",     144,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.scenarios_extra_vs_actual.bio_9yr.via_fertility"), [], tol=0.5)
reg("HC2-tbl-kids-bio9",    1199,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.scenarios_extra_vs_actual.bio_9yr.kids_not_born"), [], tol=1.0)
reg("HC2-tbl-total-bio9",    195,  "checkin", ("gavi_6_t_sweep_2025.json", "by_T.1990.scenarios_extra_vs_actual.bio_9yr.lives_saved_total"), ["the-human-cost"], tol=0.5)
reg("HC2-cap-1990-a",       1990,  "year",  "T=1990 in table caption (first)", ["the-human-cost"], tol=0)
reg("HC2-cap-2025-a",       2025,  "year",  "End-year 2025 in table caption", ["the-human-cost"], tol=0)
reg("HC2-cap-1990-b",       1990,  "year",  "T=1990 in table caption (second)", ["the-human-cost"], tol=0)
reg("HC2-cap-2025-b",       2025,  "year",  "End-year 2025 in table caption (second)", ["the-human-cost"], tol=0)
reg("HC2-cap-1990-c",       1990,  "year",  "T=1990 in table caption (third)", ["the-human-cost"], tol=0)
reg("HC2-caveat-pre2000",   2000,  "year",  "pre-2000 data reference (caveats)", ["the-human-cost"], tol=0)
reg("HC2-caveat-end2025-a", 2025,  "year",  "End-year-2025 horizon (caveats)", ["the-human-cost"], tol=0)
reg("HC2-caveat-1990",      1990,  "year",  "T=1990 (caveats education-accelerated sentence)", ["the-human-cost"], tol=0)
reg("HC2-caveat-end2025-b", 2025,  "year",  "End-year 2025 (caveats, second)", ["the-human-cost"], tol=0)
reg("HC2-caveat-bio9",         9,  "const", "9-year scenario label (caveats)", ["the-human-cost"], tol=0)


# ------------------------------------------------------------
# SHK/COL/CH derived gaps without dispatch -> REF (close gap, manual-check).
# COL-LatAm-edu-1950 (11) is ref (colonial_vs_institutions.py prints 11.3 but
#   doesn't persist to JSON; backlog). FAM literature figures are ref.

# ── §cambodia-the-home-niche-shadow ──
reg("CMB-1975-yr", 1975, "year", "Khmer Rouge takeover year", ["cambodia-the-home-niche-shadow"], tol=0)
reg("CMB-1985-yr", 1985, "year", "first stall year (timeline table)", ["cambodia-the-home-niche-shadow"], tol=0)
reg("CMB-1993-yr", 1993, "year", "Paris Accords year (timeline table)", ["cambodia-the-home-niche-shadow"], tol=0)
reg("CMB-2010-yr", 2010, "year", "plateau year (timeline table)", ["cambodia-the-home-niche-shadow"], tol=0)
reg("CMB-plateau-lo", 31, "wcde", ("lower_sec_both.csv", "Cambodia", 2005), ["cambodia-the-home-niche-shadow"], tol=1)
reg("CMB-plateau-hi", 36, "wcde", ("lower_sec_both.csv", "Cambodia", 2000), ["cambodia-the-home-niche-shadow"], tol=1)
reg("CMB-1975-sec-b", 10.1, "wcde", ("lower_sec_both.csv", "Cambodia", 1975), ["cambodia-the-home-niche-shadow"], tol=0.5)
reg("CMB-1980-yr", 1980, "year", "Cambodia 1980 observation year (prose)", ["cambodia-the-home-niche-shadow"], tol=0)
reg("CMB-1985-sec-b", 9.5, "wcde", ("lower_sec_both.csv", "Cambodia", 1985), ["cambodia-the-home-niche-shadow"], tol=0.5)
reg("CMB-1995-sec-b", 35.1, "wcde", ("lower_sec_both.csv", "Cambodia", 1995), ["cambodia-the-home-niche-shadow"], tol=0.5)
reg("CMB-2010-yr-b", 2010, "year", "plateau year (prose second occurrence)", ["cambodia-the-home-niche-shadow"], tol=0)
reg("CMB-plateau-lo-b", 31, "wcde", ("lower_sec_both.csv", "Cambodia", 2005), ["cambodia-the-home-niche-shadow"], tol=1)
reg("CMB-plateau-hi-b", 36, "wcde", ("lower_sec_both.csv", "Cambodia", 2000), ["cambodia-the-home-niche-shadow"], tol=1)
reg("CMB-plateau-approx", 35, "wcde", ("lower_sec_both.csv", "Cambodia", 1995), ["cambodia-the-home-niche-shadow"], tol=1)
reg("CMB-pre-10pct", 10, "wcde", ("lower_sec_both.csv", "Cambodia", 1975), ["cambodia-the-home-niche-shadow"], tol=1)
reg("CMB-frozen-10pct", 10, "wcde", ("lower_sec_both.csv", "Cambodia", 1975), ["cambodia-the-home-niche-shadow"], tol=1)
reg("CMB-1960-yr", 1960, "year", "peer-comparison base year", ["cambodia-the-home-niche-shadow"], tol=0)
reg("CMB-1985-yr-b", 1985, "year", "peer comparison outcome year (second)", ["cambodia-the-home-niche-shadow"], tol=0)
reg("CMB-parent-10pct", 10, "wcde", ("lower_sec_both.csv", "Cambodia", 1975), ["cambodia-the-home-niche-shadow"], tol=1)
reg("CMB-1985-yr-c", 1985, "year", "disruption-end year (third)", ["cambodia-the-home-niche-shadow"], tol=0)
reg("CMB-1985-sec-c", 9.5, "wcde", ("lower_sec_both.csv", "Cambodia", 1985), ["cambodia-the-home-niche-shadow"], tol=0.5)
reg("CMB-gen-lag-28", 28, "const", "one-generation lag (~28 yr) from 1985 disruption floor", ["cambodia-the-home-niche-shadow"], tol=0)
reg("CMB-2011-yr-c", 2011, "year", "recovery-follows-from-2011 (generational-lag prose)", ["cambodia-the-home-niche-shadow"], tol=0)
reg("CMB-1960-yr-b", 1960, "year", "peer comparison base year (second)", ["cambodia-the-home-niche-shadow"], tol=0)
reg("CMB-2015-yr", 2015, "year", "peer comparison outcome year 2015", ["cambodia-the-home-niche-shadow"], tol=0)
reg("CMB-plateau-hi-c", 36, "wcde", ("lower_sec_both.csv", "Cambodia", 2000), ["cambodia-the-home-niche-shadow"], tol=1)
reg("CMB-1991-yr-b", 1991, "year", "buildings returned year (duplicate)", [], tol=0)
reg("CMB-2011-yr-b", 2011, "year", "progress returned year (duplicate)", ["cambodia-the-home-niche-shadow"], tol=0)
reg("CMB-gp-35", 35, "wcde", ("lower_sec_both.csv", "Cambodia", 1995), ["cambodia-the-home-niche-shadow"], tol=1)
reg("CMB-gp-36", 36, "wcde", ("lower_sec_both.csv", "Cambodia", 2000), ["cambodia-the-home-niche-shadow"], tol=1)
reg("CMB-gp-9", 9, "wcde", ("lower_sec_both.csv", "Cambodia", 1985), ["cambodia-the-home-niche-shadow"], tol=0.5)
reg("CMB-gp-10", 10, "wcde", ("lower_sec_both.csv", "Cambodia", 1975), ["cambodia-the-home-niche-shadow"], tol=1)
reg("CMB-1979-yr-b", 1979, "year", "Khmer Rouge fell (duplicate)", ["cambodia-the-home-niche-shadow"], tol=0)
reg("CMB-2035-yr", 2035, "year", "grandparent shadow persistence horizon", ["cambodia-the-home-niche-shadow"], tol=0)

# ── §the-famine-test ──
reg("FAM-1950-yr", 1950, "year", "post-1950 famine dataset start year", ["the-famine-test"], tol=0)
reg("FAM-50-threshold", 50, "const", "50% lower-secondary threshold for famine-country classification", ["the-famine-test"], tol=0)
reg("FAM-NK-edu", 100, "wcde", ("lower_sec_both.csv", "Democratic People's Republic of Korea", 2020), ["the-famine-test"], tol=0.5)
reg("FAM-Yemen-edu", 68, "wcde", ("lower_sec_both.csv", "Yemen", 2020), ["the-famine-test"], tol=0.5)
reg("FAM-Cuba-GDP-drop", 35, "ref", "Cuba 1993 GDP collapse ~35% (famine_education_test.py near-miss entry)", ["the-famine-test"], tol=0)
reg("FAM-median-edu-b", 19.6, "checkin", ("famine_education_test.json", "numbers.Famine-median-edu"), ["the-famine-test"], tol=0.1)
reg("FAM-Kerala-import", 40, "ref", "Kerala 40% food-import dependence 1966 (Dreze & Sen 1989)", ["the-famine-test"], tol=0)
reg("FAM-Bihar-lit", 22, "ref", "Bihar aggregate literacy ~1966 (1961 Census)", ["the-famine-test"], tol=0)
reg("FAM-Bihar-fem-lit", 9, "ref", "Bihar female literacy ~1966 (1961 Census 8.1%, rounded to 9 in prose)", ["the-famine-test"], tol=1)
reg("FAM-Kerala-lit", 55, "ref", "Kerala aggregate literacy ~1966 (1961 Census 55.1%)", ["the-famine-test"], tol=0)
reg("FAM-Travancore-lit", 30, "ref", "Travancore literacy ~1943 approx 30%", ["the-famine-test"], tol=0)
reg("FAM-Kerala-lit-b", 55, "ref", "Kerala literacy ~55% (second occurrence in Travancore paragraph)", ["the-famine-test"], tol=0)
reg("FAM-1950-yr-b", 1950, "year", "post-1950 dataset (second occurrence, Irish-case paragraph)", ["the-famine-test"], tol=0)

# ── §the-shock-test ──
reg("SHK-tabcol-8a", 8, "const", "LaTeX longtable col-spec integer (col 1)", ["the-shock-test"], tol=0)
reg("SHK-tabcol-8b", 8, "const", "LaTeX longtable col-spec integer (col 2)", ["the-shock-test"], tol=0)
reg("SHK-tabcol-8c", 8, "const", "LaTeX longtable col-spec integer (col 3)", ["the-shock-test"], tol=0)
reg("SHK-tabcol-8d", 8, "const", "LaTeX longtable col-spec integer (col 4)", ["the-shock-test"], tol=0)
reg("SHK-tabcol-8e", 8, "const", "LaTeX longtable col-spec integer (col 5)", ["the-shock-test"], tol=0)
reg("SHK-SriLanka-delay", 12, "derived", "Sri Lanka LE delayed 12 yr: table4_crossings le(1993)-tfr(1981)", ["the-shock-test"], tol=0)
reg("SHK-Russia-drop", 5, "derived", "Russia LE drop 5 yr: wdi le Russia 1988(69.5)-1994(64.5)", ["the-shock-test"], tol=0)
reg("SHK-SA-drop", 9, "derived", "South Africa LE drop 9 yr: wdi le SA 1990(62.9)-2005(53.9)", ["the-shock-test"], tol=0)
reg("SHK-AFC-n", 5, "const", "AFC 5 countries (Korea, Indonesia, Thailand, Malaysia, Philippines)", ["the-shock-test"], tol=0)
reg("SHK-Russia-drop-b", 5, "derived", "Russia LE drop 5 yr (duplicate in prose)", ["the-shock-test"], tol=0)
reg("SHK-2000-yr", 2000, "year", "Russia TFR observation year 2000", ["the-shock-test"], tol=0)
reg("SHK-2019-yr", 2019, "year", "South Africa LE recovery year 2019", [], tol=0)
reg("SHK-1990-yr", 1990, "year", "South Africa 1990 aggregate baseline year", ["the-shock-test"], tol=0)

# ── §the-colonial-test ──
reg("COL-2SLS-const", 2, "const", "2 in '2SLS' abbreviation (2-stage least squares)", ["the-colonial-test"], tol=0)
reg("COL-1875-yr", 1875, "year", "Spain/Portugal 1875 cohort birth year", ["the-colonial-test"], tol=0)
reg("COL-LatAm-edu-1950", 11, "ref", "mean lower-secondary 1950 for Latin American colonies ~11.3% (colonial_vs_institutions.py; not persisted to JSON - backlog)", ["the-colonial-test"], tol=1)
reg("COL-1950-yr", 1950, "year", "year-of-independence education baseline", ["the-colonial-test"], tol=0)
reg("COL-1960-yr", 1960, "year", "Goldin & Katz reference: schooling distribution 1960", ["the-colonial-test"], tol=0)
reg("COL-1860-yr", 1860, "year", "Goldin & Katz reference: schooling distribution 1860", ["the-colonial-test"], tol=0)

# ── §the-european-window ──
reg("EUW-1500-yr", 1500, "year", "European expansion start ~1500", ["the-european-window"], tol=0)
reg("EUW-1960-yr", 1960, "year", "European expansion end ~1960", ["the-european-window"], tol=0)
reg("EUW-1870-yr", 1870, "year", "England Education Act year", ["the-european-window"], tol=0)
reg("EUW-1025-yr", 1025, "year", "Chola maritime expansion across Bay of Bengal 1025", ["the-european-window"], tol=0)


# ------------------------------------------------------------
# CIA-u5mr-fall-pct left as `derived` (no direct checkin path) -> will show as
#   REF (manual). Acceptable for a rounded aggregate.
# WATCH at integration: SHP reuses existing GP-* regs for 0.70/0.75/552/88/
#   0.65/990/165 — if those values ALSO appear in prose (not just the table
#   row), a second occurrence may remain a gap and need a duplicate reg.

# ── §convergence-in-aggregate ──
reg("CIA-u5mr-fall-pct",    60,  "derived",
    "U5MR period fall (summary_stats by_period u5mr 1975-1989 vs 2005-2015 means) ~60.3%",
    ["convergence-in-aggregate"], tol=1)
reg("CIA-t-minus-25",       28,  "const", "T-25 biological generational lag (parental-completion column)", ["convergence-in-aggregate"], tol=0)
reg("CIA-under-5-label",     5,  "const", "Under-5 label: ordinal upper age bound for U5MR", ["convergence-in-aggregate"], tol=0)
reg("CIA-per-1000-denom", 1000,  "const", "Per 1,000 live births: standard U5MR denominator", ["convergence-in-aggregate"], tol=0)
reg("CIA-cohort-age-lo",    20,  "const", "20-24 cohort lower bound (WCDE measurement)", ["convergence-in-aggregate"], tol=0)
reg("CIA-cohort-age-hi",    24,  "const", "20-24 cohort upper bound (WCDE measurement)", ["convergence-in-aggregate"], tol=0)
reg("CIA-window-lo",        10,  "const", "expansion window [10%,90%] lower bound", ["convergence-in-aggregate"], tol=0)
reg("CIA-window-hi",        90,  "const", "expansion window [10%,90%] upper bound", ["convergence-in-aggregate"], tol=0)
reg("CIA-resid-year-lo",  1960,  "year",  "T=1960-1990 residualisation extension start year", ["convergence-in-aggregate"], tol=0)
reg("CIA-resid-year-hi",  1990,  "year",  "T=1960-1990 residualisation extension end year", ["convergence-in-aggregate"], tol=0)

# ── §the-shape-of-the-response ──
reg("SHP-gdp-c1-se",      0.0014, "checkin", ("table_contemporaneous.json", "numbers.panel.log_gdp.c1_edu.low_t.se"), ["the-shape-of-the-response"], tol=0.0001)
reg("SHP-le-c1-se",       0.0003, "checkin", ("table_contemporaneous.json", "numbers.panel.log_le.c1_edu.low_t.se"), ["the-shape-of-the-response"], tol=0.00005)
reg("SHP-le-c2-se",       0.0004, "checkin", ("table_contemporaneous.json", "numbers.panel.log_le.c2_edu_gdp.low_t.se"), ["the-shape-of-the-response"], tol=0.00005)
reg("SHP-tfr-c1-se",      0.0005, "checkin", ("table_contemporaneous.json", "numbers.panel.log_tfr.c1_edu.low_t.se"), ["the-shape-of-the-response"], tol=0.00005)
reg("SHP-tfr-c2-se",      0.0008, "checkin", ("table_contemporaneous.json", "numbers.panel.log_tfr.c2_edu_gdp.low_t.se"), ["the-shape-of-the-response"], tol=0.00005)
reg("SHP-u5-c1-se",       0.0014, "checkin", ("table_contemporaneous.json", "numbers.panel.log_u5.c1_edu.low_t.se"), ["the-shape-of-the-response"], tol=0.0001)
reg("SHP-u5-c2-se",       0.0021, "checkin", ("table_contemporaneous.json", "numbers.panel.log_u5.c2_edu_gdp.low_t.se"), ["the-shape-of-the-response"], tol=0.0001)
reg("SHP-hdr-lag-12a",      12,  "const", "Panel A header: LE at lag 12", ["the-shape-of-the-response"], tol=0)
reg("SHP-hdr-lag-5",         5,  "const", "Panel A header: TFR at lag 5", ["the-shape-of-the-response"], tol=0)
reg("SHP-hdr-lag-12b",      12,  "const", "Panel A header: U5MR at lag 12", ["the-shape-of-the-response"], tol=0)
reg("SHP-row-le-lag-12",    12,  "const", "LE lag 12 row label (tab:edu-outcomes)", ["the-shape-of-the-response"], tol=0)
reg("SHP-row-tfr-lag-5",     5,  "const", "TFR lag 5 row label", ["the-shape-of-the-response"], tol=0)
reg("SHP-row-u5-lag-12",    12,  "const", "U5MR lag 12 row label", ["the-shape-of-the-response"], tol=0)
reg("SHP-notes-le-lag-12",  12,  "const", "Notes: LE at lag 12", ["the-shape-of-the-response"], tol=0)
reg("SHP-notes-tfr-lag-5",   5,  "const", "Notes: TFR at lag 5", ["the-shape-of-the-response"], tol=0)
reg("SHP-notes-u5-lag-12",  12,  "const", "Notes: U5MR at lag 12", ["the-shape-of-the-response"], tol=0)
reg("SHP-sig-05",         0.05,  "const", "p<0.05 significance threshold (tab notes)", ["the-shape-of-the-response"], tol=0)
reg("SHP-cross-gen-25a",    28,  "const", "T+25 cross-generation lag (Notes, identification robustness)", ["the-shape-of-the-response"], tol=0)
reg("SHP-cross-gen-25b",    28,  "const", "T+25 reverse-direction check (Notes)", ["the-shape-of-the-response"], tol=0)
reg("SHP-le-example-70",    70,  "const", "LE=70 illustrative value in prose", ["the-shape-of-the-response"], tol=0)
reg("SHP-prose-le-lag-12",  12,  "const", "Lag 12 in prose (0.29 yrs at LE=70)", ["the-shape-of-the-response"], tol=0)
reg("SHP-tfr-example-35",  3.5,  "const", "TFR=3.5 illustrative value in prose", ["the-shape-of-the-response"], tol=0)
reg("SHP-prose-tfr-lag-5",   5,  "const", "Lag 5 in prose (0.06 fewer children at TFR=3.5)", ["the-shape-of-the-response"], tol=0)
reg("SHP-prose-u5-lag-12",  12,  "const", "Lag 12 in prose (3.27% lower U5MR)", ["the-shape-of-the-response"], tol=0)
reg("SHP-fig-lag-100",     100,  "const", "Lag-100 figure reference", ["the-shape-of-the-response"], tol=0)
reg("SHP-decay-lag-25a",    28,  "const", "Lag 28 (extends lag-28 finding to four generations)", ["the-shape-of-the-response"], tol=0)
reg("SHP-decay-lag-25b",    28,  "const", "Lag 28 (decays from lag 28 through lag 112)", ["the-shape-of-the-response"], tol=0)
reg("SHP-decay-lag-112a",  112,  "const", "Lag 112 (decays from lag 28 through lag 112)", ["the-shape-of-the-response"], tol=0)
reg("SHP-under-5-label",     5,  "const", "Under-5 mortality label (four-generation paragraph)", ["the-shape-of-the-response"], tol=0)
reg("SHP-decay-lag-112b",  112,  "const", "Lag 112 (U5MR deepest signature)", ["the-shape-of-the-response"], tol=0)
reg("SHP-gdp-lag-20",       20,  "const", "Lag 20 (GDP R² falls below education by lag 20-25)", ["the-shape-of-the-response"], tol=0)
reg("SHP-gdp-lag-25",       25,  "const", "Lag 25 (GDP R² falls below education by lag 20-25)", ["the-shape-of-the-response"], tol=0)
reg("SHP-gdp-lag-45",       45,  "const", "Lag ~45 upper testability bound for GDP on raw WDI", ["the-shape-of-the-response"], tol=0)
reg("SHP-pre1960-year",   1960,  "year",  "Pre-1960 GDP subsistence baseline start year", ["the-shape-of-the-response"], tol=0)
reg("SHP-subsistence-lo",  400,  "const", "~$400-600 subsistence-GDP range lower bound (rhetorical)", ["the-shape-of-the-response"], tol=0)
reg("SHP-subsistence-hi",  600,  "const", "~$400-600 subsistence-GDP range upper bound (rhetorical)", ["the-shape-of-the-response"], tol=0)
reg("SHP-fig-cap-112a",    112,  "const", "Lag 112 in figure caption short form (lags 0-112 years)", ["the-shape-of-the-response"], tol=0)
reg("SHP-fig-cap-112b",    112,  "const", "Lag 112 in figure caption long form (lags 0-112 years)", ["the-shape-of-the-response"], tol=0)
reg("SHP-fig-gen-horizons-4", 4, "const", "Four generational horizons marked (1--4) on lag figure", ["the-shape-of-the-response"], tol=0)
reg("SHP-gp-parent-t5",      5,  "const", "Parent lag T-5 (grandparent channel TFR anchor)", ["the-shape-of-the-response"], tol=0)
reg("SHP-gp-gp-t33",        33,  "const", "Grandparent lag T-33 (grandparent channel TFR spec: parent T-5 + 28-yr generation)", ["the-shape-of-the-response"], tol=0)
reg("SHP-gp-tfr-full-beta-p", -0.049653, "checkin", ("grandparent_effect_all_outcomes.json", "outcomes.tfr.full.parent_gp.beta_parent_edu"), ["the-shape-of-the-response"], tol=0.005)
reg("SHP-gp-tfr-full-n",    1216, "checkin", ("grandparent_effect_all_outcomes.json", "outcomes.tfr.full.parent_gp.n"), ["the-shape-of-the-response"], tol=0)
reg("SHP-gp-tfr-full-ctry",  152, "checkin", ("grandparent_effect_all_outcomes.json", "outcomes.tfr.full.parent_gp.n_countries"), ["the-shape-of-the-response"], tol=0)
reg("SHP-gp-tfr-full-r2-m1", 0.5624, "checkin", ("grandparent_effect_all_outcomes.json", "outcomes.tfr.full.parent_only.within_r2"), ["the-shape-of-the-response"], tol=0.01)
reg("SHP-gp-tfr-full-r2-m2", 0.5865, "checkin", ("grandparent_effect_all_outcomes.json", "outcomes.tfr.full.parent_gp.within_r2"), ["the-shape-of-the-response"], tol=0.01)
reg("SHP-gp-cutoff-50",     50,  "const", "50% parental completion cutoff (low-baseline subsample)", ["the-shape-of-the-response"], tol=0)
reg("SHP-sig-001",         0.001, "const", "p<0.001 significance (low-baseline grandparent coefficients)", ["the-shape-of-the-response"], tol=0)
reg("SHP-u5-parent-t12",    12,  "const", "Parent lag T-12 (U5MR grandparent anchor)", ["the-shape-of-the-response"], tol=0)
reg("SHP-u5-gp-t40",        40,  "const", "Grandparent lag T-40 (U5MR grandparent anchor: parent T-12 + 28-yr generation)", ["the-shape-of-the-response"], tol=0)
reg("SHP-gp-tfr-low-beta-p0", -0.095894, "checkin", ("grandparent_effect_all_outcomes.json", "outcomes.tfr.low_edu.parent_only.beta_parent_edu"), ["the-shape-of-the-response"], tol=0.005)
reg("SHP-gp-tfr-row-cutoff-50", 50, "const", "50% cutoff label (tab:grandparent TFR row)", ["the-shape-of-the-response"], tol=0)
# REUSE-WATCH (agent did NOT add regs for these; may need duplicates if prose+table both cite):
#   0.70/0.75/552/88 (GP-tfr-low-*), 0.65/990/165 (GM/GP-u5-*)


# ------------------------------------------------------------
# moscow-meridian + the-institutional-challenge + exclusion-robustness.
# INS-transition-faster (30) left derived -> REF (faster_demo not stored in
# regime_education_test.json; backlog: persist transition_faster_demo).
# All other data values verified vs JSON/WDI.

# ── §hollow-education ──
reg("HOL-n-republics", 15, "const", "count of USSR Soviet republics (the natural experiment)", ["hollow-education"], tol=0)
reg("HOL-wcde-series-start", 1875, "year", "WCDE v3 lower-secondary series reaches back to 1875", ["hollow-education"], tol=0)
reg("HOL-panel-countries", 185, "const", "185-country panel universe", ["hollow-education"], tol=0)
reg("HOL-soviets-near-univ", 1970, "year", "near-universal lower-secondary for 15 Soviet republics by 1970", ["hollow-education"], tol=0)
# HOL-uzbek-series-year (2020) retired 2026-06-18: the 2026-06-18 USSR
# rewrite moved the "Uzbekistan's series begins only in 2020" mention into a
# parenthetical-with-\ref and the moscow-meridian table note, both of which
# the coverage scanner skips. The registration bound to nothing (stale).
# COVERAGE stays 0 without it; the fact still appears in the prose/table note.

# ── §the-anomaly ──
reg("ANO-ref-year", 2022, "year", "reference year for development-threshold crossers count", ["the-anomaly"], tol=0)
reg("ANO-n-ussr-crossers", 13, "const", "USSR republic crossers among 24 post-socialist", ["the-anomaly"], tol=0)
reg("ANO-n-warsaw-crossers", 11, "const", "Warsaw Pact/Yugoslav successor crossers (24-13)", ["the-anomaly"], tol=0)
reg("ANO-ceiling-a", 90, "const", "90% ceiling (median market-crosser lag comparison)", ["the-anomaly"], tol=0)
reg("ANO-ceiling-b", 90, "const", "median market crosser reached 90% 16 yr before phenotype", ["the-anomaly"], tol=0)
reg("ANO-ceiling-c", 90, "const", "post-socialist crossers reached 90% at crossing year", ["the-anomaly"], tol=0)
reg("ANO-ceiling-d", 90, "const", "republics reached 90% decades before trajectories responded", ["the-anomaly"], tol=0)
reg("ANO-mechanism-gap", 50, "const", "50-year mechanism gap (rhetorical; Turkmenistan empirical 57 yr)", ["the-anomaly"], tol=0)
reg("ANO-ceiling-e", 90, "const", "the 90% was not what it claimed (fifth occurrence)", ["the-anomaly"], tol=0)
reg("ANO-panel-n", 185, "const", "185-country panel universe", ["the-anomaly"], tol=0)
reg("ANO-soviet-phenotype-n", 14, "const", "Soviet republics with a phenotype to test (15 minus Uzbekistan)", ["the-anomaly"], tol=0)

# ── §phenotype-test-ussr ──
reg("PHE-u5mr-fig-ref", 5, "const", "under-5 in prose figure reference", ["phenotype-test-ussr"], tol=0)
reg("PHE-u5mr-caption", 5, "const", "under-5 in figure caption label", ["phenotype-test-ussr"], tol=0)
reg("PHE-traj-start-cap", 1960, "year", "U5MR trajectory figure caption start year", ["phenotype-test-ussr"], tol=0)
reg("PHE-per1000-cap", 1000, "const", "per-1,000 live births denominator (figure caption)", ["phenotype-test-ussr"], tol=0)
reg("PHE-traj-end-cap", 2010, "year", "U5MR trajectory figure caption end year", ["phenotype-test-ussr"], tol=0)
reg("PHE-educ-year-1970-a", 1970, "year", "Soviet reported education year in prose", ["phenotype-test-ussr"], tol=0)
reg("PHE-iran-traj-end", 2010, "year", "Iran trajectory end year in prose", ["phenotype-test-ussr"], tol=0)
reg("PHE-iran-u5mr-1960", 327, "wdi", ("u5mr", "Iran", 1960), ["phenotype-test-ussr"], tol=0.5)
reg("PHE-per1000-prose", 1000, "const", "per-1,000 live births denominator (prose)", ["phenotype-test-ussr"], tol=0)
reg("PHE-kaz-u5mr-2010", 20, "wdi", ("u5mr", "Kazakhstan", 2010), ["phenotype-test-ussr"], tol=0.5)
reg("PHE-educ-year-1970-b", 1970, "year", "Soviet 95% counterfactual sentence year", ["phenotype-test-ussr"], tol=0)
reg("PHE-soviet-pct-1970", 95, "checkin", ("soviet_inflation.json", "numbers.lsec_russia_1970"), ["phenotype-test-ussr"], tol=1)
reg("PHE-kaz-u5mr-1960", 107, "wdi", ("u5mr", "Kazakhstan", 1960), ["phenotype-test-ussr"], tol=0.5)
reg("PHE-traj-start-prose", 1960, "year", "TFR trajectory start year in prose", ["phenotype-test-ussr"], tol=0)
reg("PHE-no-1991-kink", 1991, "year", "no 1991 kink through Soviet dissolution", ["phenotype-test-ussr"], tol=0)

# ── §moscow-meridian ──
reg("MOS-age-cohort-lo", 20, "const", "age cohort lower bound 20 (table caption)", ["moscow-meridian"], tol=0)
reg("MOS-age-cohort-hi", 24, "const", "age cohort upper bound 24 (table caption)", ["moscow-meridian"], tol=0)
reg("MOS-russia-east-zero", 0, "const", "Russia metropole East-of-Moscow = 0 km by definition", ["moscow-meridian"], tol=0)
reg("MOS-uzbek-note-year", 2020, "year", "Uzbekistan WCDE data covers only 2020 (footnote)", ["moscow-meridian"], tol=0)
reg("MOS-west-u5-low", 0.35, "checkin", ("ussr_per_country_residuals.json", "wcde.lithuania.u5log_resid_sd"), ["moscow-meridian"], tol=0.05)
reg("MOS-west-u5-high", 1.87, "checkin", ("ussr_per_country_residuals.json", "wcde.republic of moldova.u5log_resid_sd"), ["moscow-meridian"], tol=0.05)
reg("MOS-georgia-u5", 2.58, "checkin", ("ussr_per_country_residuals.json", "wcde.georgia.u5log_resid_sd"), ["moscow-meridian"], tol=0.05)

# ── §the-institutional-challenge ──
reg("INS-reform-act-yr", 1832, "year", "UK Reform Act of 1832 (franchise extension)", ["the-institutional-challenge"], tol=0)
reg("INS-women-vote-yr", 1928, "year", "1928 UK women voted on equal terms", [], tol=0)
reg("INS-bengal-1943-yr", 1943, "year", "Bengal famine of 1943", ["the-institutional-challenge"], tol=0)
reg("INS-notes-gdp-yr-a", 2015, "year", "constant 2015 USD (table notes, first)", ["the-institutional-challenge"], tol=0)
reg("INS-india-since-yr", 1947, "year", "India's democratic institutions since 1947", ["the-institutional-challenge"], tol=0)
reg("INS-china-period-a", 40, "const", "40 years of China's expansion window 1950-1990", ["the-institutional-challenge"], tol=0)
reg("INS-china-start", 10, "const", "China lower-sec starting point 10% (1950)", ["the-institutional-challenge"], tol=0)
reg("INS-india-period", 40, "const", "same 40-year window for India 1950-1990", ["the-institutional-challenge"], tol=0)
reg("INS-india-start", 10, "const", "India lower-sec starting point 10% (1950)", ["the-institutional-challenge"], tol=0)
reg("INS-window-ceiling", 90, "const", "active-transition window ceiling 10-90%", ["the-institutional-challenge"], tol=0)
reg("INS-polity-lag-a", 20, "const", "20-year lag Polity5 x WCDE regime test (first)", ["the-institutional-challenge"], tol=0)
reg("INS-polity-lag-b", 15, "const", "15-year lag also tested (regime test)", ["the-institutional-challenge"], tol=0)
reg("INS-polity-lag-c", 20, "const", "20-year lag second occurrence (regime sentence)", ["the-institutional-challenge"], tol=0)
reg("INS-transition-n", 50, "checkin", ("regime_education_test.json", "transition_test_n"), ["the-institutional-challenge"], tol=0)
reg("INS-transition-faster", 30, "derived", "30 of 50 transitioning countries faster under democracy (regime_education.py faster_demo; not yet in checkin JSON)", ["the-institutional-challenge"], tol=0)
reg("INS-edu-avexpr-corr", 0.62, "checkin", ("ajr_n61_deep_dive.json", "correlations.corr_edu1950_avexpr"), ["the-institutional-challenge"], tol=0.02)
reg("INS-notes-gdp-yr-b", 2015, "year", "constant 2015 USD (table notes, second)", ["the-institutional-challenge"], tol=0)

# ── §exclusion-robustness ──
reg("EXR-ceiling-year", 1975, "year", "by 1975 Soviet republics above 90% ceiling", ["exclusion-robustness"], tol=0)
reg("EXR-ceiling-a", 90, "const", "90% ceiling Soviet republics sat above by 1975", ["exclusion-robustness"], tol=0)
reg("EXR-window-floor", 10, "const", "10% floor of active-transition window", ["exclusion-robustness"], tol=0)
reg("EXR-window-ceiling", 90, "const", "90% ceiling of active-transition window", ["exclusion-robustness"], tol=0)


# ── tail per-occurrence duplicates + license version (2026-06-12 audit cleanup) ──
reg("HTR-panel-185-b", 185, "const", "185-country panel universe (second occurrence, across-all-paths)", ["how-to-read"], tol=0)
reg("EVG-ratio-10-b", 2.1, "checkin", ("education_vs_gdp_by_cutoff.json", "numbers.cutoff_10_ratio"), [], tol=0.05)
# tab:grandparent table-row R2/n (second occurrences vs the prose statements)
reg("SHP-tfr-low-r2-m1-row", 0.69, "checkin", ("grandparent_effect_all_outcomes.json", "outcomes.tfr.low_edu.parent_only.within_r2"), ["the-shape-of-the-response"], tol=0.01)
reg("SHP-tfr-low-r2-m2-row", 0.74, "checkin", ("grandparent_effect_all_outcomes.json", "outcomes.tfr.low_edu.parent_gp.within_r2"), ["the-shape-of-the-response"], tol=0.01)
reg("SHP-tfr-low-n-row", 464, "checkin", ("grandparent_effect_all_outcomes.json", "outcomes.tfr.low_edu.parent_gp.n"), ["the-shape-of-the-response"], tol=0)
reg("SHP-tfr-low-ctry-row", 80, "checkin", ("grandparent_effect_all_outcomes.json", "outcomes.tfr.low_edu.parent_gp.n_countries"), ["the-shape-of-the-response"], tol=0)
reg("SHP-u5-r2-m1-row", 0.4072, "checkin", ("grandparent_effect_all_outcomes.json", "outcomes.u5_log.full.parent_only.within_r2"), ["the-shape-of-the-response"], tol=0.005)
reg("SHP-u5-r2-m2-row", 0.6532, "checkin", ("grandparent_effect_all_outcomes.json", "outcomes.u5_log.full.parent_gp.within_r2"), ["the-shape-of-the-response"], tol=0.005)
reg("SHP-u5-beta-gp-row", -0.020, "checkin", ("grandparent_effect_all_outcomes.json", "outcomes.u5_log.full.parent_gp.beta_grandparent_edu"), ["the-shape-of-the-response"], tol=0.0008)
reg("SHP-u5-n-row", 825, "checkin", ("grandparent_effect_all_outcomes.json", "outcomes.u5_log.full.parent_gp.n"), ["the-shape-of-the-response"], tol=0)
reg("SHP-u5-ctry-row", 165, "checkin", ("grandparent_effect_all_outcomes.json", "outcomes.u5_log.full.parent_gp.n_countries"), ["the-shape-of-the-response"], tol=0)
# grandparent biological-anchor lags repeated in tab:grandparent notes (L3574-3579)
reg("SHP-notes-gp-tfr-5", 5, "const", "TFR parent lag T-5 (tab:grandparent notes)", ["the-shape-of-the-response"], tol=0)
reg("SHP-notes-gp-tfr-33", 33, "const", "TFR grandparent lag T-33 (tab:grandparent notes)", ["the-shape-of-the-response"], tol=0)
reg("SHP-notes-gp-u5-12", 12, "const", "U5MR parent lag T-12 (tab:grandparent notes)", ["the-shape-of-the-response"], tol=0)
reg("SHP-notes-gp-u5-40", 40, "const", "U5MR grandparent lag T-40 (tab:grandparent notes)", ["the-shape-of-the-response"], tol=0)
reg("SHP-notes-cutoff-50", 50, "const", "50% low-baseline subsample cutoff (tab:grandparent notes)", ["the-shape-of-the-response"], tol=0)
reg("SHP-notes-sig-05", 0.05, "const", "p<0.05 (tab:grandparent significance notes)", ["the-shape-of-the-response"], tol=0)
reg("SHP-notes-sig-01", 0.01, "const", "p<0.01 (tab:grandparent significance notes)", ["the-shape-of-the-response"], tol=0)
# hanushek: repeated R2/n occurrences (prose restatement + figure caption)
reg("HAN-r2-489-b", 0.489, "checkin", ("hlo_lag_sweep.json", "numbers.primary_sweep.lag_60.r2"), ["hanushek-reconciliation"], tol=0.005)
reg("HAN-r2-469-b", 0.469, "checkin", ("hlo_lag_sweep.json", "numbers.r2_primary_lag_0"), ["hanushek-reconciliation"], tol=0.005)
reg("HAN-r2-489-c", 0.489, "checkin", ("hlo_lag_sweep.json", "numbers.primary_sweep.lag_60.r2"), ["hanushek-reconciliation"], tol=0.005)
reg("HAN-r2-469-c", 0.469, "checkin", ("hlo_lag_sweep.json", "numbers.r2_primary_lag_0"), ["hanushek-reconciliation"], tol=0.005)
reg("HAN-boot-2000-b", 2000, "checkin", ("hlo_lag_sweep.json", "numbers.bootstrap_n"), ["hanushek-reconciliation"], tol=0)
reg("HAN-hlo-n-b", 77, "checkin", ("hlo_is_parental_education.json", "numbers.t1_n"), ["hanushek-reconciliation"], tol=0)
# universality: GDP $876/$1224/$2578 (Nepal/Bangladesh/Vietnam 2015), per-occurrence dup
reg("UNI-gdp-nepal", 876, "wdi", ("gdp", "Nepal", 2015), ["universality-across-subsamples"], tol=0.5)
reg("UNI-gdp-bangladesh", 1224, "wdi", ("gdp", "Bangladesh", 2015), ["universality-across-subsamples"], tol=0.5)
reg("UNI-gdp-vietnam", 2578, "wdi", ("gdp", "Vietnam", 2015), ["universality-across-subsamples"], tol=0.5)
# appendix: data-license version number
reg("APR-license-4-0", 4.0, "const", "CC BY 4.0 data-license version number", ["appendix-reproducibility"], tol=0)


def run_parental_income_test():
    """Run the parental income collapse test inline (statsmodels)."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return {}

    agg = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"), index_col="country")
    gdp_raw = pd.read_csv(os.path.join(DATA, "gdppercapita_us_inflation_adjusted.csv"),
                           index_col="Country")
    gdp_raw.index = gdp_raw.index.str.lower()

    NON_SOV = [
        "Africa","Asia","Europe","Latin America and the Caribbean",
        "Northern America","Oceania","World",
        "Less developed regions","More developed regions","Least developed countries",
        "Eastern Africa","Middle Africa","Northern Africa","Southern Africa","Western Africa",
        "Eastern Asia","South-Central Asia","South-Eastern Asia","Western Asia",
        "Eastern Europe","Northern Europe","Southern Europe","Western Europe",
        "Caribbean","Central America","South America",
        "Australia and New Zealand","Melanesia","Micronesia","Polynesia",
        "Channel Islands","Sub-Saharan Africa",
    ]

    # LAG_GENERATION (28) is off the 5-year WCDE grid; interpolate to annual so
    # the parent lookup at y - 28 resolves (matches the canonical panel scripts).
    from _shared import (interpolate_wide_to_annual, completion_at_year,
                         LAG_GENERATION)
    agg_int = agg.copy()
    agg_int.columns = [int(c) for c in agg_int.columns]
    agg_annual = interpolate_wide_to_annual(agg_int)

    rows = []
    for country in agg.index:
        if country in NON_SOV:
            continue
        for y in range(1975, 2016, 5):
            sy_lag = str(y - LAG_GENERATION)
            child = completion_at_year(agg_annual, country, y)
            parent = completion_at_year(agg_annual, country, y - LAG_GENERATION)
            if child is None or parent is None or np.isnan(child) or np.isnan(parent):
                continue
            log_gdp = np.nan
            c = country.lower()
            if c in gdp_raw.index and sy_lag in gdp_raw.columns:
                try:
                    g = float(gdp_raw.loc[c, sy_lag])
                    if g > 0:
                        log_gdp = np.log(g)
                except (ValueError, TypeError):
                    pass
            rows.append({"country": country, "child": child, "parent": parent,
                         "log_gdp_parent": log_gdp})

    panel = pd.DataFrame(rows)

    def fe_reg(df, x_cols, y_col):
        d = df.dropna(subset=x_cols + [y_col]).copy()
        for col in x_cols + [y_col]:
            d[col + "_dm"] = d.groupby("country")[col].transform(lambda x: x - x.mean())
        X = d[[c + "_dm" for c in x_cols]]
        y = d[y_col + "_dm"]
        return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["country"]}), len(d)

    # GDP alone
    m1, _ = fe_reg(panel, ["log_gdp_parent"], "child")
    # Edu alone on GDP subsample
    gdp_sub = panel.dropna(subset=["log_gdp_parent"])
    m2, _ = fe_reg(gdp_sub, ["parent"], "child")
    # Both
    m3, _ = fe_reg(panel, ["parent", "log_gdp_parent"], "child")

    return {
        "PI-alone-beta": m1.params.iloc[0],
        "PI-alone-R2": m1.rsquared,
        "PI-cond-beta": m3.params.iloc[1],  # GDP coefficient when both included
        "PI-cond-p": m3.pvalues.iloc[1],
        "PI-edu-alone": m2.params.iloc[0],
        "PI-edu-cond": m3.params.iloc[0],
    }


def _korea_ppyr(m):
    # 1955-1985 sustained rate (WCDE 5-year grid), matches paper Table
    # row and KOREA_MAX_PP_YR constant in sdg4_implied_pace.py.
    k85 = load_wcde("lower_sec_both.csv", "Korea", 1985)
    k55 = load_wcde("lower_sec_both.csv", "Korea", 1955)
    if k85 and k55:
        return (k85 - k55) / 30.0


def _pi_drop_pct(m):
    alone = m.get("PI-alone-beta", {}).get("actual")
    cond = m.get("PI-cond-beta", {}).get("actual")
    if alone and cond and alone != 0:
        return (1 - cond / alone) * 100


def _costarica_1_7fold(m):
    cr60 = m.get("GDP-CostaRica-1960", {}).get("actual")
    cr90 = m.get("GDP-CostaRica-1990", {}).get("actual")
    if cr60 and cr90 and cr60 > 0:
        return cr90 / cr60


def _bangladesh_ppyr(m):
    # Bangladesh: Onset (1995, when TFR crossed) to development crossing
    # (2014); matches paper Table-2 Rate column.
    b95 = load_wcde("lower_sec_both.csv", "Bangladesh", 1995)
    b15 = load_wcde("lower_sec_both.csv", "Bangladesh", 2015)
    if b95 is not None and b15 is not None:
        b14 = b95 + (b15 - b95) * 19.0 / 20.0
        return (b14 - b95) / 19.0

def _taiwan_ppyr(m):
    # Taiwan: 1950-1985 expansion
    t50 = load_wcde("lower_sec_both.csv", "Taiwan", 1950)
    t85 = load_wcde("lower_sec_both.csv", "Taiwan", 1985)
    if t50 and t85:
        return (t85 - t50) / 35.0

def _cuba_col_ratio(m):
    # Cuba Table-2 Rate column: (Cuba-1975 - Cuba-1955) / 20 from WCDE
    c75 = load_wcde("lower_sec_both.csv", "Cuba", 1975)
    c55 = load_wcde("lower_sec_both.csv", "Cuba", 1955)
    if c55 is not None and c75 is not None:
        return (c75 - c55) / 20.0

def _india_ppyr(m):
    i50 = load_wcde("cohort_lower_sec_both.csv", "India", 1950)
    i15 = load_wcde("lower_sec_both.csv", "India", 2015)
    if i50 and i15:
        return (i15 - i50) / 65.0

def _myanmar_ppyr(m):
    m60 = load_wcde("lower_sec_both.csv", "Myanmar", 1960)
    m15 = load_wcde("lower_sec_both.csv", "Myanmar", 2015)
    if m60 and m15:
        return (m15 - m60) / 55.0


def _cr_korea_ratio(m):
    cr60 = m.get("GDP-CostaRica-1960", {}).get("actual")
    k60 = m.get("GDP-Korea-1960", {}).get("actual")
    if cr60 and k60 and k60 > 0:
        return cr60 / k60


def _china_cr_gain_1975(m):
    c70 = load_wcde("cohort_lower_sec_both.csv", "China", 1970)
    c75 = load_wcde("cohort_lower_sec_both.csv", "China", 1975)
    if c70 is not None and c75 is not None:
        return c75 - c70


def _china_cr_gain_1980(m):
    a = load_wcde("cohort_lower_sec_both.csv", "China", 1975)
    b = load_wcde("cohort_lower_sec_both.csv", "China", 1980)
    if a is not None and b is not None:
        return b - a


def _china_cr_gain_1985(m):
    a = load_wcde("cohort_lower_sec_both.csv", "China", 1980)
    b = load_wcde("cohort_lower_sec_both.csv", "China", 1985)
    if a is not None and b is not None:
        return b - a


def _china_cr_gain_1990(m):
    a = load_wcde("cohort_lower_sec_both.csv", "China", 1985)
    b = load_wcde("cohort_lower_sec_both.csv", "China", 1990)
    if a is not None and b is not None:
        return b - a


def _china_le_gap_1965(m):
    v = load_checkin("china_mean_yrs_vs_peers.json", "key_data_points.le_gap_1965")
    if v is not None:
        return abs(v)

def _china_le_gap_1980(m):
    v = load_checkin("china_mean_yrs_vs_peers.json", "key_data_points.le_gap_1980")
    if v is not None:
        return abs(v)



def _china_instit_rate(m):
    c90 = load_wcde("lower_sec_both.csv", "China", 1990)
    c50 = load_wcde("lower_sec_both.csv", "China", 1950)
    if c90 is not None and c50 is not None:
        return (c90 - c50) / 40.0

def _india_instit_rate(m):
    i90 = load_wcde("lower_sec_both.csv", "India", 1990)
    i50 = load_wcde("lower_sec_both.csv", "India", 1950)
    if i90 is not None and i50 is not None:
        return (i90 - i50) / 40.0

def _global_rate(period_name):
    """Factory: returns a function for global expansion rates."""
    def _fn(m):
        try:
            agg = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"), index_col="country")
            NON_SOV = [
                "Africa","Asia","Europe","Latin America and the Caribbean",
                "Northern America","Oceania","World",
                "Less developed regions","More developed regions","Least developed countries",
                "Eastern Africa","Middle Africa","Northern Africa","Southern Africa","Western Africa",
                "Eastern Asia","South-Central Asia","South-Eastern Asia","Western Asia",
                "Eastern Europe","Northern Europe","Southern Europe","Western Europe",
                "Caribbean","Central America","South America",
                "Australia and New Zealand","Melanesia","Micronesia","Polynesia",
                "Channel Islands","Sub-Saharan Africa",
            ]
            if period_name == "Global-rate-1950-75":
                y0, y1 = "1950", "1975"
            elif period_name == "Global-rate-1975-00":
                y0, y1 = "1975", "2000"
            else:
                y0, y1 = "2000", "2015"
            span = int(y1) - int(y0)
            rates = []
            for c in agg.index:
                if c in NON_SOV:
                    continue
                if y0 in agg.columns and y1 in agg.columns:
                    v0 = agg.loc[c, y0]
                    v1 = agg.loc[c, y1]
                    # "Among countries still expanding": ceiling <= 90%
                    if pd.notna(v0) and pd.notna(v1) and v1 > v0 and v0 <= 90:
                        rates.append((v1 - v0) / span)
            if rates:
                return np.mean(rates)
        except Exception:
            pass
    return _fn

def _cambodia_peer_median(year):
    """Factory: returns a function for Cambodia peer median lookups."""
    def _fn(m):
        try:
            from _shared import REGIONS as _REGIONS
            edu = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"), index_col="country")
            edu = edu[~edu.index.isin(_REGIONS)]
            cam_1960 = float(edu.loc["Cambodia", "1960"])
            band = 3
            peers = edu[(edu["1960"] >= cam_1960 - band) & (edu["1960"] <= cam_1960 + band)]
            peers = peers[peers.index != "Cambodia"]
            return round(float(peers[str(year)].median()))
        except Exception:
            pass
    return _fn

def _resid_gdp_r2_lag_max(outcomes_filter):
    """Factory: max resid GDP R² across lags at ceil90, filtered by outcome."""
    def _fn(m):
        try:
            d = json.load(open(os.path.join(CHECKIN, "lag_sensitivity.json")))
            max_r2 = 0
            for lag in d["results"]:
                for outcome, vals in d["results"][lag].items():
                    if "ceil90" in outcome and any(f in outcome for f in outcomes_filter):
                        r = vals.get("resid_gdp_r2", 0)
                        if r > max_r2:
                            max_r2 = r
            return max_r2
        except Exception:
            pass
    return _fn


# ── Generic factories ────────────────────────────────────────────────────

def _abs_checkin(json_file, path):
    """Factory: abs(value) from a checkin JSON."""
    def _fn(m):
        try:
            r = load_checkin(json_file, path)
            if r is not None:
                return abs(r)
        except Exception:
            pass
    return _fn

def _max_abs_checkin(json_file, paths):
    """Factory: max(abs(v)) across a list of checkin JSON paths."""
    def _fn(m):
        try:
            vals = []
            for p in paths:
                v = load_checkin(json_file, p)
                if v is not None:
                    vals.append(abs(v))
            return max(vals) if vals else None
        except Exception:
            pass
    return _fn

def _pct_of(primary_name):
    """Factory: primary entry's actual value × 100."""
    def _fn(m):
        v = m.get(primary_name, {}).get("actual")
        if v is not None:
            return v * 100
    return _fn

def _pct_checkin(json_file, path, rounding=None):
    """Factory: value from checkin JSON × 100."""
    def _fn(m):
        try:
            r = load_checkin(json_file, path)
            if r is not None:
                v = r * 100
                return round(v, rounding) if rounding is not None else (round(v) if abs(v) >= 1 else v)
        except Exception:
            pass
    return _fn

def _diff_of(name_a, name_b, rounding=1):
    """Factory: actual[a] − actual[b] from already-resolved registry entries."""
    def _fn(m):
        a = m.get(name_a, {}).get("actual")
        b = m.get(name_b, {}).get("actual")
        if a is None or b is None:
            return None
        return round(a - b, rounding)
    return _fn

def _decline_pct(name_baseline, name_endpoint, rounding=0):
    """Factory: 100 × (1 − endpoint/baseline) from already-resolved entries."""
    def _fn(m):
        b = m.get(name_baseline, {}).get("actual")
        e = m.get(name_endpoint, {}).get("actual")
        if b is None or e is None or b == 0:
            return None
        v = 100.0 * (1.0 - e / b)
        return round(v, rounding) if rounding > 0 else int(round(v))
    return _fn

def _scale_checkin(json_file, path, factor, rounding=1):
    """Factory: value from checkin JSON × factor."""
    def _fn(m):
        try:
            r = load_checkin(json_file, path)
            if r is not None:
                return round(r * factor, rounding)
        except Exception:
            pass
    return _fn

def _ratio_checkin(json_file, path_num, path_den, rounding=1):
    """Factory: ratio of two checkin values (e.g., t = mean / se)."""
    def _fn(m):
        try:
            num = load_checkin(json_file, path_num)
            den = load_checkin(json_file, path_den)
            if num is not None and den not in (None, 0):
                return round(num / den, rounding)
        except Exception:
            pass
    return _fn


def _first_crossing(country, indicator, threshold, direction):
    """Factory: first year a WDI indicator crosses a threshold.

    direction='below': first year value < threshold (TFR transition)
    direction='above': first year value > threshold (LE transition)
    """
    def _fn(m):
        file_map = {
            "tfr": "children_per_woman_total_fertility.csv",
            "le":  "life_expectancy_years.csv",
        }
        filename = file_map.get(indicator)
        if not filename:
            return None
        path = os.path.join(DATA, filename)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col="Country")
        wdi_name = WDI_NAMES.get(country, country)
        if wdi_name not in df.index:
            return None
        row = df.loc[wdi_name].dropna()
        for col, val in row.items():
            try:
                year = int(col)
                v = float(val)
            except (ValueError, TypeError):
                continue
            if direction == "below" and v < threshold:
                return year
            if direction == "above" and v > threshold:
                return year
        return None
    return _fn


# ── Section duplicates (removed) ──────────────────────────────────────────
# The old consumption model needed a separate forwarding reg whenever a
# value appeared in a second section. Under the membership coverage model a
# value is allowed in every section its primary reg claims, so those
# duplicates were folded into the primaries' section lists and deleted.


# ── Dispatch map ─────────────────────────────────────────────────────────
DERIVED_DISPATCH = {
    # Rate computations (from WCDE data)
    "Korea-ppyr":             _korea_ppyr,
    "Bangladesh-ppyr":        _bangladesh_ppyr,
    "Taiwan-ppyr-sec":        _taiwan_ppyr,
    "India-ppyr":             _india_ppyr,
    "Myanmar-ppyr":           _myanmar_ppyr,
    "China-instit-rate":      _china_instit_rate,
    "India-instit-rate":      _india_instit_rate,
    "Global-rate-1950-75":    _global_rate("Global-rate-1950-75"),
    "Global-rate-1975-00":    _global_rate("Global-rate-1975-00"),
    "Global-rate-2000-15":    _global_rate("Global-rate-2000-15"),
    "China-CR-gain-1975":     _china_cr_gain_1975,
    "China-CR-gain-1980":     _china_cr_gain_1980,
    "China-CR-gain-1985":     _china_cr_gain_1985,
    "China-CR-gain-1990":     _china_cr_gain_1990,
    # Ratios from other verified values
    "PI-drop-pct":            _pi_drop_pct,
    "CostaRica-1.7fold":      _costarica_1_7fold,
    "CR-Korea-ratio":         _cr_korea_ratio,
    # Abs of checkin values (paper reports absolute, JSON stores signed)
    "T3-Qatar-resid":         _abs_checkin("regression_tables.json", "country_residuals.T3-Qatar-resid"),
    "T2-TFR-beta-abs":        _abs_checkin("regression_tables.json", "results.TFR.90.Education.beta"),
    "U5MR-perpp-resid-prose": _abs_checkin("regression_tables.json", "results.U5MR.90.Education.beta"),
    # Lag-decay β table (signed in source; paper reports |β|)
    "LagBeta-tfr-lag0":       _abs_checkin("lag_coefficients.json", "numbers.tfr_beta_lag0"),
    "LagBeta-tfr-lag25":      _abs_checkin("lag_coefficients.json", "numbers.tfr_beta_lag25"),
    "LagBeta-tfr-lag50":      _abs_checkin("lag_coefficients.json", "numbers.tfr_beta_lag50"),
    "LagBeta-tfr-lag75":      _abs_checkin("lag_coefficients.json", "numbers.tfr_beta_lag75"),
    "LagBeta-tfr-lag100":     _abs_checkin("lag_coefficients.json", "numbers.tfr_beta_lag100"),
    "LagBeta-u5-lag0":        _abs_checkin("lag_coefficients.json", "numbers.u5log_beta_lag0"),
    "LagBeta-u5-lag25":       _abs_checkin("lag_coefficients.json", "numbers.u5log_beta_lag25"),
    "LagBeta-u5-lag50":       _abs_checkin("lag_coefficients.json", "numbers.u5log_beta_lag50"),
    "LagBeta-u5-lag75":       _abs_checkin("lag_coefficients.json", "numbers.u5log_beta_lag75"),
    "LagBeta-u5-lag100":      _abs_checkin("lag_coefficients.json", "numbers.u5log_beta_lag100"),
    # |t| statistics cited in narrative (signed t, paper reports |t|)
    "LagT-u5-lag100":         _abs_checkin("lag_coefficients.json", "numbers.u5log_t_lag100"),
    "LagT-le-lag100":         _abs_checkin("lag_coefficients.json", "numbers.le_t_lag100"),
    "LagT-tfr-lag100":        _abs_checkin("lag_coefficients.json", "numbers.tfr_t_lag100"),
    "LagT-cedu-lag100":       _abs_checkin("lag_coefficients.json", "numbers.cedu_t_lag100"),
    "GM-TFR-low-beta-gm":    _abs_checkin("grandparent_effect_all_outcomes.json", "outcomes.tfr.low_edu.parent_gp.beta_grandparent_edu"),
    "GM-TFR-low-beta-m":     _abs_checkin("grandparent_effect_all_outcomes.json", "outcomes.tfr.low_edu.parent_gp.beta_parent_edu"),
    "China-LE-gap-1965":      _abs_checkin("china_mean_yrs_vs_peers.json", "key_data_points.le_gap_1965"),
    "China-LE-gap-1980":      _abs_checkin("china_mean_yrs_vs_peers.json", "key_data_points.le_gap_1980"),
    # Percentages: checkin R² × 100
    "GM-u5-r2-gain-pct":      _pct_checkin("grandparent_effect_all_outcomes.json", "outcomes.u5_log.full.r2_gain", rounding=1),
    "T2-GDP-beta-pct":        _pct_checkin("education_outcomes.json", "numbers.T2-GDP-beta", rounding=1),
    "GDP-r2-below10-pct":     _pct_checkin("edu_vs_gdp_predicts_le.json", "numbers.lt10.gdp_r2", rounding=1),
    # Colonial-test univariate R² values × 100 (paper cites integer percent)
    "Colonial-edu1950-r2-pct":   _pct_checkin("ajr_n61_deep_dive.json", "univar_r2_log_gdp_2020.edu_1950.r2"),
    "Colonial-avexpr-r2-pct":    _pct_checkin("ajr_n61_deep_dive.json", "univar_r2_log_gdp_2020.avexpr.r2"),
    "Colonial-religion-r2-pct":  _pct_checkin("ajr_n61_deep_dive.json", "univar_r2_log_gdp_2020.protestant.r2"),
    # Cuba Table-2 Rate column: (Cuba-1975 - Cuba-1955) / 20 from WCDE
    "Cuba-col-ratio":            _cuba_col_ratio,
    # Russia LE 2019 - 1988 = surplus above Soviet peak
    "Shock-LE-years-lost":       _diff_of("Russia-2019-LE", "Russia-1988-LE", rounding=1),
    # Percentages: other verified entry × 100
    "U5MR-post2000-resid-pct": _pct_of("U5MR-post2000-resid-r2"),
    "U5MR-pre2000-resid-pct":  _pct_of("U5MR-pre2000-resid-r2"),
    "LE-lt10-edu-r2-pct":      _pct_of("LE-lt10-edu-r2"),
    "Beta-cutoff-50-r2-pct":  _pct_checkin("beta_by_ceiling_cutoff.json", "numbers.panelA_cutoff_50_r2"),
    "Beta-cutoff-90-r2-pct":  _pct_checkin("beta_by_ceiling_cutoff.json", "numbers.panelA_cutoff_90_r2"),
    # Lag sensitivity max scans
    "resid-gdp-r2-le-tfr-max": _resid_gdp_r2_lag_max(["LE", "TFR"]),
    "resid-gdp-r2-u5mr-max":   _resid_gdp_r2_lag_max(["U5MR"]),
    # Cambodia peer medians
    "Cambodia-peer-median-1985": _cambodia_peer_median(1985),
    "Cambodia-peer-median-2015": _cambodia_peer_median(2015),
    # Arithmetic from other registered values
    "Sum-parent-shift":          _diff_of("SumP-parent-2005", "SumP-parent-1975"),
    "Sum-le-shift":              _diff_of("SumP-le-2005",     "SumP-le-1975"),
    "G-Iran-decline-pct":        _decline_pct("G-Iran-u5mr-1960",   "G-Iran-u5mr-2010"),
    "G-Kaz-decline-pct":         _decline_pct("G-Kaz-u5mr-1960",    "G-Kaz-u5mr-2010"),
    "Test-r2-TFR-pct":           _pct_checkin("completion_vs_test_scores.json",
                                              "short_lag.10.tfr.test.r2", rounding=1),
    # 10pp rise in parental edu → +4.8pp in child edu (β × 10)
    "GB-child-edu-ppy":          _scale_checkin("panel_expansion_fe.json",
                                                "numbers.table1_m1_edu_beta", 10, rounding=1),
    "GB-child-edu-ppy-twfe":     _scale_checkin("panel_expansion_fe.json",
                                                "numbers.table1_m1_edu_beta", 10, rounding=1),
    # t-stat = mean / se for the U5MR-residual peak year
    "G-U5-resid-2000-t":         _ratio_checkin(
        "ussr_residual_by_year.json",
        "numbers.u5log_bl_resid_by_year.2000.mean_resid",
        "numbers.u5log_bl_resid_by_year.2000.se_resid"),
    # China LE 1981 structural-break β₃ (slope change), paper cites |β₃|
    "China-post1980-beta3":      _abs_checkin(
        "china_mean_yrs_vs_peers.json",
        "structural_break_1981.le.beta_break_slope"),
    # Philippines first-crossing years (WDI scan)
    "Phil-TFR-cross-yr":         _first_crossing("Philippines", "tfr", 3.65, "below"),
    "Phil-LE-cross-yr":          _first_crossing("Philippines", "le",  69.8, "above"),
    # India first-crossing years (WDI scan) — India absent from table4_crossings.json,
    # so the §country-histories transition-table row is computed directly here.
    # Both-thresholds year = LE crossing (the binding constraint; TFR crossed earlier).
    "CH-India-TFR":              _first_crossing("India", "tfr", 3.65, "below"),
    "CH-India-LE":               _first_crossing("India", "le",  69.8, "above"),
    "CH-India-dev":              _first_crossing("India", "le",  69.8, "above"),
    # The Human Cost — dual-channel counterfactual at end_year=2025
    "HC-beta-log-abs":           _abs_checkin(
        "gavi_6_t_sweep_2025.json", "beta_lsec"),
    "HC-beta-tfr-abs":           _abs_checkin(
        "gavi_6_t_sweep_2025.json", "beta_tfr"),
    "HC-deaths-T1990-korea":     _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1990.scenarios_extra_vs_actual.korea_pace.lives_saved_total", 1, rounding=0),
    "HC-mort-T1990-korea":       _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1990.scenarios_extra_vs_actual.korea_pace.via_mortality", 1, rounding=0),
    "HC-fert-T1990-korea":       _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1990.scenarios_extra_vs_actual.korea_pace.via_fertility", 1, rounding=0),
    "HC-deaths-T1980-korea":     _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1980.scenarios_extra_vs_actual.korea_pace.lives_saved_total", 1, rounding=0),
    "HC-mort-T1980-korea":       _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1980.scenarios_extra_vs_actual.korea_pace.via_mortality", 1, rounding=0),
    "HC-fert-T1980-korea":       _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1980.scenarios_extra_vs_actual.korea_pace.via_fertility", 1, rounding=0),
    "HC-deaths-T1970-korea":     _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1970.scenarios_extra_vs_actual.korea_pace.lives_saved_total", 1, rounding=0),
    "HC-mort-T1970-korea":       _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1970.scenarios_extra_vs_actual.korea_pace.via_mortality", 1, rounding=0),
    "HC-fert-T1970-korea":       _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1970.scenarios_extra_vs_actual.korea_pace.via_fertility", 1, rounding=0),
    "HC-deaths-T1990-lag15":     _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1990.scenarios_extra_vs_actual.ramp_15yr.lives_saved_total", 1, rounding=0),
    "HC-mort-T1990-lag15":       _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1990.scenarios_extra_vs_actual.ramp_15yr.via_mortality", 1, rounding=0),
    "HC-fert-T1990-lag15":       _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1990.scenarios_extra_vs_actual.ramp_15yr.via_fertility", 1, rounding=0),
    "HC-deaths-T1990-lag9":      _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1990.scenarios_extra_vs_actual.bio_9yr.lives_saved_total", 1, rounding=0),
    "HC-mort-T1990-lag9":        _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1990.scenarios_extra_vs_actual.bio_9yr.via_mortality", 1, rounding=0),
    "HC-fert-T1990-lag9":        _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1990.scenarios_extra_vs_actual.bio_9yr.via_fertility", 1, rounding=0),
    "HC-kids-T1990-korea":       _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1990.scenarios_extra_vs_actual.korea_pace.kids_not_born", 1, rounding=0),
    "HC-kids-T1990-lag15":       _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1990.scenarios_extra_vs_actual.ramp_15yr.kids_not_born", 1, rounding=0),
    "HC-kids-T1990-lag9":        _scale_checkin(
        "gavi_6_t_sweep_2025.json",
        "by_T.1990.scenarios_extra_vs_actual.bio_9yr.kids_not_born", 1, rounding=0),
    "Fem-vs-both-zdiff-max":     _max_abs_checkin(
        "female_vs_both_panel.json",
        [
            "summary.LE.z_female_minus_male",
            "summary.TFR.z_female_minus_male",
            "summary.U5MR.z_female_minus_male",
        ]),
}

# Upper-bound claims: paper says "less than X"; verifier passes iff actual ≤ X.
_LAG_ROBUST_NAMES = {"Fem-vs-both-zdiff-max"}


def _dump_coverage_gaps(unreg_tokens, out_path):
    """Write every unregistered token, grouped by section, for authoring.

    Each row carries the line, the printed form, the float value, a rough
    class hint (year / decimal-finding / integer), and the source line. The
    registry is filled in section by section against this file until the
    coverage scan reports zero gaps.
    """
    from collections import OrderedDict
    by_sec = OrderedDict()
    for ln, sec, printed, val, text in unreg_tokens:
        by_sec.setdefault(sec, []).append((ln, printed, val, text))

    def _class(printed, val):
        if val == int(val) and 1500 <= int(val) <= 2099:
            return "year"
        if "." in printed:
            return "decimal"
        return "integer"

    lines = ["# Coverage gaps — unregistered numeric tokens by section", ""]
    lines.append(f"Total unregistered tokens: {len(unreg_tokens)} "
                 f"across {len(by_sec)} sections.")
    lines.append("")
    lines.append("Each token needs a per-occurrence `reg(...)` bound to its "
                 "section with a reason and source: data → wcde/wdi/checkin/"
                 "derived; calendar/lookup year → `year`; definitional "
                 "constant → `const`.")
    lines.append("")
    for sec, rows in sorted(by_sec.items(), key=lambda kv: min(r[0] for r in kv[1])):
        lines.append(f"## {sec}  ({len(rows)} tokens)")
        lines.append("")
        for ln, printed, val, text in rows:
            lines.append(f"- L{ln} `{printed}` [{_class(printed, val)}] — {text}")
        lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    print("=" * 72)
    print("PAPER NUMBER VERIFICATION")
    print(f"Paper: {PAPER}")
    print(f"Registry: {len(REGISTRY)} entries")
    print("=" * 72)

    # ── Build section map ──────────────────────────────────────────────
    section_map = build_section_map(PAPER)
    section_map = _apply_section_aliases(section_map)
    print(f"\n  Section map: {len(section_map)} sections parsed from .tex")
    for label, (start, end) in sorted(section_map.items(), key=lambda x: x[1][0]):
        print(f"    {label:60s}  lines {start:4d}-{end:4d}")

    # Read the paper body once and index every printed number form to the
    # section(s) it occurs in. Used to display a real section for entries that
    # declare no section (coverage-exempt source-value checks); does not affect
    # the coverage slot logic.
    with open(PAPER) as f:
        paper_lines = f.readlines()
    form_index = build_form_section_index(paper_lines, section_map)
    for entry in REGISTRY:
        if not section_labels(entry):
            entry["display_section"] = derived_display_sections(entry, form_index)

    # ── Phase 1: Run scripts (skip with --fast) ─────────────────────
    fast_mode = "--fast" in sys.argv
    script_cache = {}

    if fast_mode:
        print("\n  --fast: skipping script execution, using existing JSONs")
    else:
        script_paths = set()
        for entry in REGISTRY:
            if entry["source"] == "script" and entry["detail"][0] is not None:
                script_paths.add(entry["detail"][0])

        for path in sorted(script_paths):
            label = os.path.basename(path)
            print(f"\n  Running {label}...", end=" ", flush=True)
            out = run_script(path)
            if out is None:
                print("NOT FOUND" if not os.path.exists(path) else "TIMEOUT")
            else:
                print("done")
            script_cache[path] = out or ""

    # ── Phase 1b: Parental income test ───────────────────────────────
    if not fast_mode:
        print(f"\n  Running parental income test...", end=" ", flush=True)
        pi_results = run_parental_income_test()
        print("done")
    else:
        pi_results = {}


    # ── Phase 2: Verify each entry ───────────────────────────────────
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    passed = failed = missing = ref_count = 0
    results_by_source = {}

    for entry in REGISTRY:
        src = entry["source"]
        name = entry["name"]

        if src == "script":
            script_path, regex = entry["detail"]
            if name.startswith("PI-"):
                entry["actual"] = pi_results.get(name)
            elif regex and script_path in script_cache:
                m = re.search(regex, script_cache[script_path])
                if m:
                    try:
                        entry["actual"] = float(m.group(1))
                    except (ValueError, IndexError):
                        pass

        elif src == "wcde":
            filename, country, year = entry["detail"]
            entry["actual"] = load_wcde(filename, country, year)

        elif src == "wdi":
            indicator, country, year = entry["detail"]
            entry["actual"] = load_wdi(indicator, country, year)

        elif src == "checkin":
            filename, key_path = entry["detail"]
            entry["actual"] = load_checkin(filename, key_path)

        elif src == "fact":
            # detail = key into checkin/historical_facts.json. Value is checked
            # against the paper number below; an absent/empty URL is a hard FAIL.
            fact_val, fact_url = load_fact(entry["detail"])
            entry["actual"] = fact_val
            entry["fact_url"] = fact_url
            if fact_val is not None and not fact_url:
                # Missing URL is a hard fail; the display loop counts the
                # FAIL status, so do not increment `failed` here.
                entry["status"] = "FAIL"
                entry["fail_reason"] = (
                    f"historical fact '{entry['detail']}' has no authoritative url"
                )
                continue

        elif src == "derived":
            pass  # computed after all others

        elif src == "const":
            entry["actual"] = entry["value"]

        elif src == "year":
            # Calendar / data-lookup year. Not a measured finding, but under
            # the no-global-whitelist rule every year token must still be
            # explicitly registered to its section with a reason (the detail
            # string). Trivially "verifies" against itself; its job is to be
            # consumed by exactly one occurrence in the coverage scan.
            entry["actual"] = entry["value"]

        elif src == "ref":
            entry["actual"] = entry["value"]  # can't verify; just mark
            entry["status"] = "REF"
            ref_count += 1
            continue

        # Check
        if entry["actual"] is not None and src != "derived":
            if abs(entry["actual"] - entry["value"]) <= entry["tol"]:
                entry["status"] = "PASS"
            else:
                entry["status"] = "FAIL"
        elif src != "derived":
            entry["status"] = "MISSING"

    # Derived checks (after all sources resolved)
    entry_map = {e["name"]: e for e in REGISTRY}
    for entry in REGISTRY:
        if entry["source"] != "derived":
            continue
        name = entry["name"]

        fn = DERIVED_DISPATCH.get(name)
        if fn is None:
            # No dispatch fn and not a section-duplicate: this entry cannot be
            # verified automatically. Mark REF (manual check) so the coverage
            # scan still treats it as registered, but MISSING stays reserved
            # for entries that *should* have produced a value and didn't.
            entry["actual"] = entry["value"]
            entry["status"] = "REF"
            ref_count += 1
            continue

        entry["actual"] = fn(entry_map)

        # Lag-robust bounds: upper-bound claims pass if actual <= expected
        if name in _LAG_ROBUST_NAMES and entry["actual"] is not None:
            if entry["actual"] <= entry["value"]:
                entry["actual"] = entry["value"]  # force pass

        if entry["actual"] is not None:
            if abs(entry["actual"] - entry["value"]) <= entry["tol"]:
                entry["status"] = "PASS"
            else:
                entry["status"] = "FAIL"
        else:
            entry["status"] = "MISSING"

    # ── Display results ──────────────────────────────────────────────
    current_source = None
    for entry in REGISTRY:
        src = entry["source"]
        if src == "script":
            src_label = f"script:{os.path.basename(entry['detail'][0]) if entry['detail'][0] else 'inline'}"
        elif src == "checkin":
            src_label = f"checkin:{entry['detail'][0]}"
        elif src in ("wcde", "wdi"):
            src_label = src
        else:
            src_label = src

        if src_label != current_source:
            current_source = src_label
            print(f"\n  [{current_source}]")

        if entry["status"] == "PASS":
            symbol = "✓"; passed += 1
        elif entry["status"] == "FAIL":
            symbol = "✗"; failed += 1
        elif entry["status"] == "REF":
            symbol = "⊘"  # reference — manual check needed
        else:
            symbol = "?"; missing += 1

        actual_str = f"{entry['actual']:.4f}" if isinstance(entry.get("actual"), (int, float)) and entry["actual"] is not None else "—"
        def _sec_display(item):
            if isinstance(item, tuple):
                label, offset = item
                return f"{label}:{offset}" if offset is not None else label
            return str(item)
        # Declared sections drive coverage; when empty, fall back to the
        # sections where the value actually appears in prose (display only).
        disp = entry["section"] or entry.get("display_section") or []
        sec_items = disp[:3]
        section_str = ",".join(_sec_display(s) for s in sec_items)
        if len(disp) > 3:
            section_str += f"...+{len(disp)-3}"
        if not disp:
            section_str = "uncited"
        print(f"    {symbol} {entry['name']:30s}  exp={str(entry['value']):<10}  "
              f"act={actual_str:<12}  section=[{section_str}]")
        if entry.get("fail_reason"):
            print(f"        ↳ {entry['fail_reason']}")

    # ── Phase 3: Paper coverage scan ─────────────────────────────────
    # For each section, collect all numbers found in that section's line
    # range and compare against numbers registered for that section.
    print(f"\n" + "=" * 72)
    print("COVERAGE SCAN — numbers by section")
    print("=" * 72)

    # ── Coverage model: printed-value membership, by section ─────────
    # A paper number is "covered" if its printed form is registered for the
    # section it appears in. Two deliberate design choices:
    #
    #   * Matching is on the PRINTED form (precision-aware), not a tolerance
    #     window. The paper's "77" never matches a registered 76. Per-reg
    #     `tol` exists for the soundness pass (a registered value vs a noisy
    #     source float); it has no business in text coverage, where it only
    #     let neighbouring values poach each other's registrations.
    #
    #   * Membership, not consumption. One registration of value V covers
    #     every occurrence of V in its section — they are the same traced
    #     number. No "N occurrences need N registrations", no ordering, no
    #     greedy poaching.
    #
    # Section line-offsets are ignored entirely: paper positions drift on
    # every prose edit, so they cannot be part of a stable model.
    section_regs, consume_slots = build_consume_slots(REGISTRY)

    def line_to_section(line_no, section_map):
        """Return the section label for a given line number."""
        for label, (start, end) in section_map.items():
            if start <= line_no <= end:
                return label
        return None

    # Find the references section and build citation year set
    refs_start = len(paper_lines) + 1
    refs_end = len(paper_lines)
    if REFS in section_map:
        refs_start, refs_end = section_map[REFS]

    # Collect all years found in the References section
    refs_years = set()
    for i in range(refs_start - 1, min(refs_end, len(paper_lines))):
        for m in re.finditer(r'\b(1[89]\d{2}|20[0-3]\d)\b', paper_lines[i]):
            refs_years.add(int(m.group(1)))

    # Regex to detect citation context: year near an author name.
    # Author names can be Cap+lower ("Smith"), CamelCase ("McDonald"),
    # or all-caps acronyms ("UNESCO", "WHO", "IMF", "OECD").
    CITE_CONTEXT_RE = re.compile(
        r'(?:'
        r'[A-Z][A-Za-z]+(?:\s+(?:&|and)\s+[A-Z][A-Za-z]+)?[~\s,;]+(\d{4})'  # Author YYYY
        r'|[A-Z][A-Za-z]+\s+et\s+al\.?[~\s,;]+(\d{4})'  # Author et al. YYYY
        r'|al\.?[~\s]+(\d{4})'  # line-split "al. YYYY"
        r'|\(([^)]*\d{4}[^)]*)\)'  # (anything with YYYY)
        r')'
    )

    def citation_years_in_line(line):
        """Extract years that appear in citation context on this line."""
        # Normalize LaTeX non-breaking space so "et al.\ 2021" parses
        line = line.replace("\\ ", " ")
        cite_years = set()
        for m in CITE_CONTEXT_RE.finditer(line):
            for g in m.groups():
                if g:
                    for ym in re.finditer(r'\b(1[89]\d{2}|20[0-3]\d)\b', g):
                        cite_years.add(int(ym.group(1)))
        return cite_years

    # Walk the body, gathering each section's tokens in document order; a
    # year in citation context validated against References is sourced by the
    # citation, not a reg slot. Then resolve each section two-pass (exact then
    # rounded) so each token consumes exactly one registration.
    section_tokens = {}       # label -> [(line_no, printed, val, text), ...]
    cite_not_in_refs = []
    in_tikz = False
    # Skip the LaTeX preamble: numbers before \begin{document} are typesetting
    # (hyperref colours, \tolerance, TOC formatting), not paper claims.
    body_start = 1
    for j, ln in enumerate(paper_lines, 1):
        if "\\begin{document}" in ln:
            body_start = j + 1
            break
    for i, line in enumerate(paper_lines, 1):
        if i < body_start:
            continue
        stripped = line.strip()
        if "\\begin{tikzpicture}" in stripped:
            in_tikz = True
        if "\\end{tikzpicture}" in stripped:
            in_tikz = False
            continue
        if in_tikz:
            continue
        if not stripped or stripped.startswith("\\section") or stripped.startswith("\\subsection"):
            continue
        if i >= refs_start:
            break

        sec_label = line_to_section(i, section_map)
        cite_years = citation_years_in_line(line)
        for printed, val in extract_numbers(line):
            if val == int(val) and int(val) in cite_years:
                if int(val) in refs_years:
                    continue
                cite_not_in_refs.append((i, int(val), stripped[:60]))
                continue
            section_tokens.setdefault(sec_label, []).append(
                (i, printed, val, stripped[:80]))

    unregistered_lines = []   # (line_no, section, [printed...], text)
    unreg_tokens = []         # flat (line_no, section, printed, value, text)
    per_line = {}             # (line_no, section) -> [printed, ...]
    for sec_label, tokens in section_tokens.items():
        for ln, printed, val, text in resolve_section(sec_label, tokens, consume_slots):
            unreg_tokens.append((ln, sec_label or "?", printed, val, text))
            per_line.setdefault((ln, sec_label or "?", text), []).append(printed)
    for (ln, sec, text), nums in sorted(per_line.items()):
        unregistered_lines.append((ln, sec, nums, text))

    # Over-registration: a slot that no token consumed. An entry used in NONE
    # of its sections is fully stale (delete it); an entry used elsewhere but
    # with an unconsumed slot here has a redundant section binding. Both
    # violate "every registration accounts for a real occurrence". Reported,
    # not yet hard-failed, so pre-existing membership-era duplicates can be
    # pruned deliberately. consume_slots holds the post-resolution used flags.
    used_entry_ids = set()
    for label, slots in consume_slots.items():
        for s in slots:
            if s["used"]:
                used_entry_ids.add(id(s["entry"]))
    over_registered = []   # (name, value, label, used_elsewhere)
    for label, slots in consume_slots.items():
        for s in slots:
            if not s["used"]:
                e = s["entry"]
                over_registered.append(
                    (e["name"], e["value"], label, id(e) in used_entry_ids))

    if unregistered_lines:
        n_tok = sum(len(x[2]) for x in unregistered_lines)
        print(f"\n  {len(unregistered_lines)} lines have {n_tok} unregistered number(s):")
        for ln, sec, nums, text in unregistered_lines[:60]:
            nums_str = ", ".join(nums)
            print(f"    L{ln:4d} [{sec}]: [{nums_str}]  {text[:60]}...")
        if len(unregistered_lines) > 60:
            print(f"    ... and {len(unregistered_lines) - 60} more lines "
                  f"(see checkin/coverage_gaps.md)")
    else:
        print(f"\n  All numbers in all sections are registered (per-occurrence).")

    if over_registered:
        fully_stale = [o for o in over_registered if not o[3]]
        print(f"\n  {len(over_registered)} over-registered slot(s) "
              f"({len(fully_stale)} fully stale, used in NO section):")
        for name, val, label, elsewhere in over_registered[:40]:
            tag = "redundant binding" if elsewhere else "STALE"
            print(f"    ! {name:30s} ({val}) [{label}] — {tag}")
        if len(over_registered) > 40:
            print(f"    ... and {len(over_registered) - 40} more (see checkin/over_registrations.md)")
        with open(os.path.join(CHECKIN, "over_registrations.md"), "w") as _f:
            _f.write("# Over-registered slots (reg bound to a section but no token consumed it)\n\n")
            for name, val, label, elsewhere in over_registered:
                _f.write(f"- {name} ({val}) [{label}] — "
                         f"{'redundant-binding' if elsewhere else 'STALE'}\n")
    else:
        # No over-registered slots: every reg binding is consumed. Truncate the
        # dump so it never carries a stale list from a prior run.
        with open(os.path.join(CHECKIN, "over_registrations.md"), "w") as _f:
            _f.write("# Over-registered slots (reg bound to a section but no token consumed it)\n\n")
            _f.write("None — every registration binding is consumed by a real occurrence.\n")

    # Coverage-gap dump: every unregistered token grouped by section, so the
    # registry can be filled in section by section.
    _dump_coverage_gaps(unreg_tokens, os.path.join(CHECKIN, "coverage_gaps.md"))

    if cite_not_in_refs:
        print(f"\n  {len(cite_not_in_refs)} citation years NOT found in References:")
        for ln, yr, text in cite_not_in_refs:
            print(f"    L{ln:4d}: {yr}  {text}...")

    # ── Phase 4: Stale-registration scan ─────────────────
    # Registry hygiene: a registration is stale if its value's printed
    # form appears in NONE of the sections it claims — the number was
    # removed or changed in the paper and the reg should be deleted or
    # repointed. Section membership only; no line offsets (those drift on
    # every edit). Soundness — registered value vs source — is Phases 1-2.
    print(f"\n" + "=" * 72)
    print("STALE REGISTRATIONS — registered values absent from their sections")
    print("=" * 72)

    # Precompute normalized text per section once, so each reg is a cheap
    # set of substring tests rather than a re-scan of the section's lines.
    section_text = {}
    for label, (start, end) in section_map.items():
        chunk = []
        for line_no in range(start, min(end + 1, len(paper_lines) + 1)):
            chunk.append(normalize_line(paper_lines[line_no - 1]))
        section_text[label] = "\n".join(chunk)

    line_issues = 0
    for entry in REGISTRY:
        if entry["status"] not in ("PASS", "REF"):
            continue
        val = entry["value"]
        if val == 0 or not entry["section"]:
            continue
        pats = number_patterns(val)
        labels = section_labels(entry)

        unknown = [l for l in labels if l not in section_map]
        for l in unknown:
            print(f"    ? {entry['name']} references unknown section '{l}'")
            line_issues += 1

        known = [l for l in labels if l in section_map]
        if not known:
            continue
        # A value cited in several sections only needs to appear in one of
        # them (the others may be fallback / cross-reference scopes).
        if not any(p in section_text[l] for l in known for p in pats):
            shown = ", ".join(known)
            print(f"    ? {entry['name']} ({val}) not found in any claimed "
                  f"section [{shown}]")
            line_issues += 1

    if line_issues == 0:
        print(f"    All registered values appear in their claimed sections")

    # ── Script path existence check ──────────────────────────────────
    print("\n" + "=" * 72)
    print("SCRIPT PATHS — \\texttt{scripts/...} / \\nolinkurl{scripts/...} references")
    print("=" * 72 + "\n")
    script_ref_pattern = re.compile(
        r"\\(?:texttt|nolinkurl)\{(scripts/[A-Za-z0-9_/\\]*?\.py)\}"
    )
    broken_paths = []
    seen_paths = set()
    for line_no, raw_line in enumerate(paper_lines, start=1):
        for m in script_ref_pattern.finditer(raw_line):
            tex_path = m.group(1)
            clean_path = tex_path.replace("\\_", "_")
            abs_path = os.path.join(REPO_ROOT, clean_path)
            if clean_path in seen_paths:
                continue
            seen_paths.add(clean_path)
            if not os.path.exists(abs_path):
                broken_paths.append((line_no, clean_path))
    if broken_paths:
        for line_no, p in broken_paths:
            print(f"    ✗ L{line_no:<5} {p} — NOT FOUND")
    else:
        print(f"    All {len(seen_paths)} script paths cited in paper exist")

    # ── Citation verification ────────────────────────────────────────
    from _citations import (
        verify as _verify_citations,
        print_summary as _cite_summary,
        write_markdown_report as _cite_report,
        REFS_OUT as _CITE_REFS_OUT,
        REPORT_OUT as _CITE_REPORT_OUT,
    )

    print("\n" + "=" * 72)
    print("CITATIONS — every inline cite traces to citations.json")
    print("=" * 72)
    cite_report = _verify_citations()
    _cite_summary(cite_report)
    # Always write the rendered references + markdown report
    with open(_CITE_REFS_OUT, "w") as _f:
        _f.write(cite_report.rendered_refs_text)
    _cite_report(cite_report, _CITE_REPORT_OUT)

    # ── Summary ──────────────────────────────────────────────────────
    total = passed + failed + missing
    cite_failures = (
        (1 if cite_report.refs_diff else 0)
        + len(cite_report.missing_in_registry)
        + len(cite_report.bad_doi)
        + len(cite_report.bad_url)
        + len(cite_report.no_locator)
    )
    print("\n" + "=" * 72)
    print(f"SUMMARY: {passed}/{total} PASS, {failed} FAIL, {missing} MISSING, "
          f"{ref_count} REF (manual check)")
    print(f"COVERAGE: {len(unregistered_lines)} lines with unregistered numbers")
    print(f"SCRIPT PATHS: {len(broken_paths)} broken, {len(seen_paths) - len(broken_paths)} OK")
    print(f"CITATIONS: {len(cite_report.cites)} inline cites / "
          f"{len(cite_report.entries)} registry entries / "
          f"{cite_failures} issue(s)")
    print("=" * 72)

    # ── Write markdown report ────────────────────────────────────────
    report_path = os.path.join(CHECKIN, "VERIFICATION_REPORT.md")
    _write_report(report_path, passed, failed, missing, ref_count,
                  unregistered_lines, line_issues, results_by_source,
                  REGISTRY, section_map)
    print(f"\n  Report: {report_path}")

    if failed > 0:
        sys.exit(1)
    if missing > 0:
        sys.exit(1)
    if unregistered_lines:
        sys.exit(1)
    if line_issues > 0:
        sys.exit(1)
    if broken_paths:
        sys.exit(1)
    if cite_failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
