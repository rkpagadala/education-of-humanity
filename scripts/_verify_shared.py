"""_verify_shared.py — shared engine for the number-traceability verifiers.

This is a LIBRARY, not a runnable verifier. It holds the machinery that the
prose-number verifiers share so they cannot drift apart:

  - source loaders     (script / wcde / wdi / checkin / fact, plus compute_ppyr)
  - number-form matching   (number_patterns / exact_patterns / extract_numbers)
  - the per-occurrence consumption model  (build_consume_slots / resolve_section)
  - the markdown report writer  (write_report)

`scripts/_shared.py` is the *data* library (panels, regressions, country-name
standardisation). This module is the *verification* layer on top of it. The
paper verifier (verify_the_long_childhood.py) and the book verifier
(verify_book_serious.py) both import from here; the registry of which numbers
exist and where they come from stays in each verifier.

Registry entry shape (a plain dict; each verifier builds its own list):
    {"name", "value", "source", "detail", "section", "tol",
     "actual": None, "status": "PENDING"}

Source types understood by the loaders / a verifier's dispatch loop:
    script   run a Python script, parse stdout with a regex
    wcde     look up (file, country, year) in a WCDE processed CSV
    wdi      look up (indicator, country, year) in a World Bank WDI CSV
    checkin  read a dot-path value from a checkin JSON
    fact     historical fact pinned to value + authoritative url
    derived  computed from other verified values (verifier-specific)
    const    definitional constant (consistency only)
    year     calendar / lookup year (consistency only)
    ref      from cited literature (manual check)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROC = os.path.join(REPO_ROOT, "wcde", "data", "processed")
DATA = os.path.join(REPO_ROOT, "data")
CHECKIN = os.path.join(REPO_ROOT, "checkin")
# ML checkin JSONs (chapter9_*, universal_evidence*) live alongside the ML
# pipeline that produces them, not in the top-level checkin/. load_checkin
# falls back here so cited ML numbers stay traceable to their producer.
ML_CHECKIN = os.path.join(REPO_ROOT, "scripts", "ml", "checkin")
HISTORICAL_FACTS = os.path.join(CHECKIN, "historical_facts.json")

# ══════════════════════════════════════════════════════════════════════════
# COUNTRY NAME MAPPING
# Maps paper/common names to the index names used by each source CSV. These
# are the defaults; a verifier may pass its own map to the loaders. Both the
# paper and the book draw from the same set, so it lives here.
# ══════════════════════════════════════════════════════════════════════════
WDI_NAMES = {
    "Korea": "Korea, Rep.",
    "South Korea": "Korea, Rep.",
    "Costa Rica": "Costa Rica",
    "Bangladesh": "Bangladesh",
    "Nepal": "Nepal",
    "Myanmar": "Myanmar",
    "Uganda": "Uganda",
    "India": "India",
    "Sri Lanka": "Sri Lanka",
    "Cuba": "Cuba",
    "China": "China",
    "Qatar": "Qatar",
    "Maldives": "Maldives",
    "Cape Verde": "Cabo Verde",
    "Bhutan": "Bhutan",
    "Tunisia": "Tunisia",
    "Vietnam": "Viet Nam",
    "Singapore": "Singapore",
    "Japan": "Japan",
    "USA": "United States",
    "Philippines": "Philippines",
    "Thailand": "Thailand",
    "Indonesia": "Indonesia",
    "Russia": "Russian Federation",
    "South Africa": "South Africa",
    "Iran": "Iran, Islamic Rep.",
    "Turkey": "Turkiye",
    "Kazakhstan": "Kazakhstan",
}

WCDE_NAMES = {
    "Korea": "Republic of Korea",
    "South Korea": "Republic of Korea",
    "Taiwan": "Taiwan Province of China",
    "Vietnam": "Viet Nam",
    "Myanmar": "Myanmar",
    "Cambodia": "Cambodia",
    "Cuba": "Cuba",
    "Bangladesh": "Bangladesh",
    "China": "China",
    "Singapore": "Singapore",
    "Philippines": "Philippines",
    "Nepal": "Nepal",
    "India": "India",
    "Sri Lanka": "Sri Lanka",
    "Portugal": "Portugal",
    "Sweden": "Sweden",
    "Germany": "Germany",
    "Spain": "Spain",
    "Nigeria": "Nigeria",
    "Qatar": "Qatar",
    "Maldives": "Maldives",
    "Russia": "Russian Federation",
    "South Africa": "South Africa",
}


# ══════════════════════════════════════════════════════════════════════════
# SOURCE LOADERS
# ══════════════════════════════════════════════════════════════════════════
def run_script(path, cwd=None):
    if not os.path.exists(path):
        return None
    if cwd is None:
        cwd = os.path.dirname(os.path.dirname(path))
    try:
        r = subprocess.run([sys.executable, path],
                           capture_output=True, text=True,
                           cwd=cwd, timeout=300)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return f"ERROR: {e}"


def load_wcde(filename, country, year, names=WCDE_NAMES):
    """Look up a value from a WCDE processed CSV."""
    wcde_name = names.get(country, country)
    path = os.path.join(PROC, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="country")
    if wcde_name not in df.index:
        return None
    col = str(year)
    if col not in df.columns:
        return None
    val = df.loc[wcde_name, col]
    if pd.isna(val):
        return None
    return float(val)


def load_wdi(indicator, country, year, names=WDI_NAMES):
    """Look up a value from World Bank WDI CSV files."""
    file_map = {
        "gdp": "gdppercapita_us_inflation_adjusted.csv",
        "tfr": "children_per_woman_total_fertility.csv",
        "le":  "life_expectancy_years.csv",
        "u5mr": "child_mortality_u5.csv",
    }
    wdi_name = names.get(country, country)
    filename = file_map.get(indicator)
    if not filename:
        return None
    path = os.path.join(DATA, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="Country")
    if wdi_name not in df.index:
        # Try case-insensitive match
        matches = [x for x in df.index if x.lower() == wdi_name.lower()]
        if matches:
            wdi_name = matches[0]
        elif country in df.index:
            # Fall back to bare country name (some CSVs use short forms)
            wdi_name = country
        else:
            return None
    col = str(year)
    if col not in df.columns:
        return None
    val = df.loc[wdi_name, col]
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def load_checkin(filename, key_path):
    """Read a value from a checkin JSON file.

    key_path is a dot-separated path into the JSON, e.g.
    "numbers.lt10.edu_r2" or "numbers.cutoff_30_edu_r2".
    Handles keys containing dots by trying progressively longer prefixes.
    """
    path = os.path.join(CHECKIN, filename)
    if not os.path.exists(path):
        # Fall back to the ML pipeline's checkin dir for chapter9_*/universal_* JSONs.
        ml_path = os.path.join(ML_CHECKIN, filename)
        if os.path.exists(ml_path):
            path = ml_path
        else:
            return None
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Malformed JSON in checkin file {path}: {e.msg}",
                e.doc, e.pos,
            ) from e
    parts = key_path.split(".")
    node = data
    i = 0
    while i < len(parts):
        part = parts[i]
        # Handle array indices like "Korea[0]"
        arr_match = re.match(r'^(.+)\[(\d+)\]$', part)
        if arr_match:
            key, idx = arr_match.group(1), int(arr_match.group(2))
            if isinstance(node, dict) and key in node:
                node = node[key]
                if isinstance(node, list) and idx < len(node):
                    node = node[idx]
                    i += 1
                    continue
            return None
        if not isinstance(node, dict):
            return None
        # Try progressively longer key segments to handle dots in key names
        found = False
        for j in range(len(parts), i, -1):
            candidate = ".".join(parts[i:j])
            if candidate in node:
                node = node[candidate]
                i = j
                found = True
                break
        if not found:
            return None
    if node is None:
        return None
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


# ── Historical facts (source type "fact") ─────────────────────────────────
# Facts not derivable from the data pipeline (UK franchise, colonial famines)
# live in checkin/historical_facts.json. Each record is
#   "<key>": {"value": <number>, "url": "<authoritative url>", ...}
# load_fact returns (value, url). A verifier's dispatch loop checks the cited
# number against `value` AND fails if `url` is missing/empty — making an
# authoritative source a hard requirement for every cited historical number.
_hist_facts_cache = None


def load_fact(key, facts_path=HISTORICAL_FACTS):
    """Return (value, url) for a historical fact, or (None, None) if absent."""
    global _hist_facts_cache
    if _hist_facts_cache is None:
        with open(facts_path) as f:
            _hist_facts_cache = json.load(f)
    rec = _hist_facts_cache.get("facts", {}).get(key)
    if rec is None:
        return None, None
    val = rec.get("value")
    try:
        val = float(val) if val is not None else None
    except (TypeError, ValueError):
        val = None
    return val, rec.get("url")


def compute_ppyr(wcde_file, country, start_year, end_year):
    """Compute percentage points per year from WCDE data."""
    v_start = load_wcde(wcde_file, country, start_year)
    v_end = load_wcde(wcde_file, country, end_year)
    if v_start is not None and v_end is not None:
        years = end_year - start_year
        return (v_end - v_start) / years
    return None


# ══════════════════════════════════════════════════════════════════════════
# NUMBER-FORM MATCHING
# A registered value is matched against the document's PRINTED forms, not a
# tolerance window: the paper's "77" never matches a registered 76. (Per-reg
# `tol` is for the soundness pass — registered value vs a noisy source float.)
# ══════════════════════════════════════════════════════════════════════════
_NUMERAL_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}


def number_patterns(val):
    """Every printed form a numeric value might take in prose."""
    pats = set()
    if isinstance(val, int) or (isinstance(val, float) and val == int(val)):
        iv = int(val)
        pats.update([str(iv), f"{iv:,}"])
    if isinstance(val, (float, int)):
        fv = float(val)
        for fmt in [".4f", ".3f", ".2f", ".1f", ".0f", "g"]:
            s = format(fv, fmt)
            pats.add(s)
            pats.add(f"~{s}")
            pats.add(f"+{s}")
            if fv < 0:
                pats.add(f"−{format(abs(fv), fmt)}")
                pats.add(f"-{format(abs(fv), fmt)}")
        # Suffix forms for large round values: "1.5 million", "one~million".
        # Pick the single largest applicable scale so we don't emit
        # "1000 thousand".
        afv = abs(fv)
        scale = None
        if afv >= 1e9 and afv < 1e12:
            scale = (1e9, "billion")
        elif afv >= 1e6 and afv < 1e9:
            scale = (1e6, "million")
        elif afv >= 1e3 and afv < 1e6:
            scale = (1e3, "thousand")
        if scale is not None:
            divisor, suffix = scale
            ratio = fv / divisor
            for fmt in [".0f", ".1f", ".2f"]:
                s = format(ratio, fmt)
                if abs(float(s) - ratio) > 1e-9:
                    continue
                for sep in (" ", "~"):
                    pats.add(f"{s}{sep}{suffix}")
                if fmt == ".0f":
                    iv = int(round(ratio))
                    if iv in _NUMERAL_WORDS:
                        for sep in (" ", "~"):
                            pats.add(f"{_NUMERAL_WORDS[iv]}{sep}{suffix}")
    return pats


def exact_patterns(val):
    """Printed forms that round-trip to `val` without changing it.

    The first consumption pass matches on these only, so a lossy form
    like "70" (which 69.5's ".0f" emits) cannot poach a literal 70
    before the real 69.5 token claims its own registration.
    """
    pats = set()
    fv = float(abs(val))
    if fv == int(fv):
        iv = int(fv)
        pats.update([str(iv), f"{iv:,}"])
    for fmt in (".4f", ".3f", ".2f", ".1f", ".0f", "g"):
        s = format(fv, fmt)
        try:
            if abs(float(s) - fv) < 1e-9:
                pats.update([s, f"~{s}", f"+{s}"])
        except ValueError:
            pass
    return pats


def normalize_line(line):
    """Normalise a source line for the substring-membership (stale) scan."""
    s = line.replace("\\*\\*\\*", "").replace("\\*\\*", "").replace("\\*", "")
    s = s.replace("**", "").replace("*", "")
    s = s.replace("−", "-")
    s = s.replace("≈", "~")
    s = s.replace("{,}", ",")
    # LaTeX $-$ minus sign: "$-$0.039" -> "-0.039"
    s = s.replace("$-$", "-")
    # Bare $...$ math wrappers around numbers (e.g. "$10.8$")
    s = s.replace("$", "")
    return s


SECTION_REF_RE = re.compile(r'[Ss]ection\s+(\d+\.\d+)')

NUMBER_RE = re.compile(
    r'(?<![a-zA-Z_/0-9])([−\-+~≈]?\$?[\d,]+\.?\d*%?)'
)


def extract_numbers(line):
    """Extract candidate empirical numbers from a source line.

    Returns a list of (printed_string, float_value). The printed string is
    what a registration's pattern set must contain for the token to be
    consumed — matching is precision-aware (the paper's "18" only satisfies a
    reg whose printed form includes "18").
    """
    clean = line.replace("**", "").replace("*", "").replace("|", " ")
    # LaTeX counter manipulations are structural, not empirical claims.
    clean = re.sub(r'\\setcounter\{[^}]*\}\{[^}]*\}', '', clean)
    clean = re.sub(r'\\addtocounter\{[^}]*\}\{[^}]*\}', '', clean)
    clean = clean.replace("−", "-").replace("≈", "~")
    clean = clean.replace("{,}", ",")
    # Strip \textasciitilde so numbers after ~ are visible to the regex
    clean = clean.replace("\\textasciitilde", "~")
    # LaTeX non-breaking space `\ ` → ` ` (so "et al.\ 2021" parses as citation)
    clean = clean.replace("\\ ", " ")
    # Parenthetical citations: (Author 2004), (Author et al. 2008; Other 2010).
    # Strip a parenthetical ONLY when its sole digits are 4-digit years —
    # i.e. it is a citation or a vintage-year aside like "(constant 2015
    # USD)". A parenthetical that also carries data numbers, e.g.
    # "(Taiwan 18%, Korea 25% completion by 1950)", is NOT a citation and
    # must keep its numbers for per-occurrence registration.
    def _strip_cite_paren(m):
        inner = m.group(0)[1:-1]
        non_year = re.sub(r'\b(1[5-9]\d{2}|20\d{2})\b', '', inner)
        return '' if not re.search(r'\d', non_year) else m.group(0)
    clean = re.sub(r'\([^()]*\)', _strip_cite_paren, clean)
    # Inline code spans
    clean = re.sub(r'`[^`]+`', '', clean)
    # URLs
    clean = re.sub(r'https?://\S+', '', clean)
    # Section cross-references
    clean = SECTION_REF_RE.sub('', clean)
    # Decade references: 1950s, 1990s--2010s
    clean = re.sub(r'\d{4}s[–\-]\d{2}s', '', clean)
    clean = re.sub(r'\d{4}s', '', clean)
    # Note: citation years are NOT stripped — they are validated against
    # the References section at scan time (see citation_years_in_refs).
    # Date ranges in methodology: YYYY--YYYY, YYYY-YYYY (as range, not subtraction)
    clean = re.sub(r'\d{4}\s*[–\-]{1,2}\s*\d{4}', '', clean)
    # Abbreviated year ranges: YYYY--YY (e.g., 1881--82)
    clean = re.sub(r'\d{4}\s*[–\-]{1,2}\s*\d{2}\b', '', clean)
    # LaTeX commands with years: \texttt{...}
    clean = re.sub(r'\\texttt\{[^}]*\}', '', clean)
    # Footnote script references
    clean = re.sub(r'\\footnote\{[^}]*\}', '', clean)
    # LaTeX table column widths: \real{0.2400}
    clean = re.sub(r'\\real\{[^}]*\}', '', clean)
    # LaTeX column specifications: p{6.3cm}, m{5cm}, etc.
    clean = re.sub(r'[pmb]\{[^}]*?[0-9.]+\s*c?m\}', '', clean)
    # LaTeX \includegraphics options: [width=0.95\linewidth]
    clean = re.sub(r'\\includegraphics\[[^\]]*\]', '', clean)
    # Math superscripts are notation, not prose numbers: R^2, R^{2},
    # 10^{-3}, t^2 → drop the exponent so the "2"/"3" is not a token.
    clean = re.sub(r'\^\{?-?\d+\}?', '', clean)
    clean = re.sub(r'\\textsuperscript\{[^}]*\}', '', clean)
    # LaTeX table-layout counts/spans, not findings:
    #   \multicolumn{2}{..}, \multirow{2}{..}, \cmidrule(lr){2-3},
    #   \arraystretch{1.15}, \setlength..{0.4pt}, dimension widths.
    clean = re.sub(r'\\(multicolumn|multirow)\{\d+\}', '', clean)
    clean = re.sub(r'\\cmidrule(\([lr]+\))?\{[\d-]+\}', '', clean)
    clean = re.sub(r'\\(arraystretch|arrayrulewidth|tabcolsep|extrarowheight)\}?\{?[\d.]+\}?', '', clean)
    clean = re.sub(r'\{[\d.]+\s*(?:em|ex|cm|mm|pt|in|bp)\}', '', clean)   # {2em}, {6.3cm}
    clean = re.sub(r'\b[\d.]+\s*(?:em|ex|cm|mm|pt|in|bp)\b', '', clean)    # 2em, 0.4pt
    clean = re.sub(r'[\d.]+\\(?:line|text|column)width', '', clean)        # 0.85\linewidth

    nums = []
    for m in NUMBER_RE.finditer(clean):
        raw = m.group(1)
        s = raw.lstrip("−-+~≈$").rstrip("%").replace(",", "")
        # Strip a trailing sentence period: "1985." -> "1985" (a real number
        # never ends in a dot; the regex greedily captured end-of-sentence
        # punctuation).
        s = s.rstrip(".")
        if not s or not s.replace(".", "").isdigit():
            continue
        try:
            val = float(s)
        except ValueError:
            continue
        nums.append((s, val))
    return nums


# ══════════════════════════════════════════════════════════════════════════
# SPELLED-OUT CARDINALS
# The companion books write empirical figures as words ("twenty-five per
# cent", "eighteen percent"). parse_number_token() converts a captured token —
# numeral OR spelled — to a float, and NUM_TOKEN_RE is the regex fragment that
# matches either form. A verifier composes anchor patterns by interpolating
# NUM_TOKEN_RE (as a named group) and PCT_RE. Scope spelled matching to curated
# anchor contexts: a blanket scan would swallow narrative prose ("four
# generations") that carries no empirical claim.
# ══════════════════════════════════════════════════════════════════════════
_CARD_ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_CARD_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_TENS_ALT = "|".join(_CARD_TENS)
_TEEN_ALT = "ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
_ONE_ALT = "one|two|three|four|five|six|seven|eight|nine"
# Numeral first, then tens(+ones), then teens, then ones (longest-match order).
NUM_TOKEN_RE = (
    r"\d{1,4}(?:[.,]\d+)?"
    rf"|(?:{_TENS_ALT})(?:[-\s](?:{_ONE_ALT}))?"
    rf"|(?:{_TEEN_ALT})"
    rf"|(?:{_ONE_ALT})"
)
# "per cent" / "percent" / "%" / LaTeX "\%". \b after "cent" so "percentage"
# is not matched as a percent unit.
PCT_RE = r"(?:per\s?cent\b|\\?%)"


def parse_number_token(tok):
    """Convert a captured number token (numeral or spelled cardinal) to float.

    Returns None if the token is not a parseable number. Handles "25", "2.13",
    "twenty-five", "eighteen", bare tens ("twenty"), and tens+ones.
    """
    if tok is None:
        return None
    t = tok.strip().lower().replace(",", "")
    try:
        return float(t)
    except ValueError:
        pass
    words = t.replace("-", " ").split()
    if not words:
        return None
    total = 0
    for w in words:
        if w in _CARD_TENS:
            total += _CARD_TENS[w]
        elif w in _CARD_ONES:
            total += _CARD_ONES[w]
        else:
            return None
    return float(total)


# ══════════════════════════════════════════════════════════════════════════
# PER-OCCURRENCE CONSUMPTION
# No global whitelist. Every numeric token in a section's body must be
# consumed by exactly one registration bound to that section. N identical
# occurrences in a section need N registrations — each with its own reason and
# source.
# ══════════════════════════════════════════════════════════════════════════
def section_labels(entry):
    """Section labels an entry claims (offsets, if any, discarded)."""
    out = []
    for sec_item in entry["section"]:
        out.append(sec_item[0] if isinstance(sec_item, tuple) else sec_item)
    return out


def build_consume_slots(registry_entries):
    """Build the per-section consumption structures from a registry list.

    Returns (section_regs, consume_slots):
      section_regs   label -> [entry, ...]
      consume_slots  label -> [{"exact", "all", "entry", "used"}, ...]
    One slot per registration bound to the section, carrying its exact and
    rounded printed-form pattern sets plus a consumed flag. Each token consumes
    one slot; N identical occurrences in a section need N registrations.
    """
    section_regs = {}
    for entry in registry_entries:
        for label in section_labels(entry):
            section_regs.setdefault(label, []).append(entry)

    consume_slots = {}
    for label, entries in section_regs.items():
        consume_slots[label] = [
            {"exact": exact_patterns(e["value"]),
             "all": number_patterns(abs(e["value"])),
             "entry": e, "used": False}
            for e in entries
        ]
    return section_regs, consume_slots


def resolve_section(label, tokens, consume_slots):
    """Two-pass consumption for one section. Returns unmatched tokens.

    Pass A consumes tokens that match a slot's EXACT forms; pass B lets
    the leftovers match rounded forms. Exact-first removes the ordering
    hazard where a rounded value steals a literal token's registration.
    Each token is (line_no, printed, val, text).
    """
    slots = consume_slots.get(label, [])
    for s in slots:
        s["used"] = False

    def grab(printed, key):
        for s in slots:
            if not s["used"] and printed in s[key]:
                s["used"] = True
                return True
        return False

    pending = [tok for tok in tokens if not grab(tok[1], "exact")]
    unmatched = [tok for tok in pending if not grab(tok[1], "all")]
    return unmatched


# ══════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════
def write_report(path, passed, failed, missing, ref_count,
                 unregistered_lines, line_issues, results_by_source,
                 registry, section_map, title="Verification Report"):
    """Write a human-readable markdown verification report."""
    from datetime import datetime
    total = passed + failed + missing

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Summary box
    status = "PASS" if failed == 0 and missing == 0 else "FAIL"
    lines.append(f"## Result: {passed}/{total} {status}")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Verified claims | {passed} |")
    lines.append(f"| Failed | {failed} |")
    lines.append(f"| Missing | {missing} |")
    lines.append(f"| Literature refs (manual) | {ref_count} |")
    lines.append(f"| Unregistered numbers | {len(unregistered_lines)} |")
    lines.append(f"| Section offset issues | {line_issues} |")
    lines.append("")

    # By source type
    source_counts = {}
    for entry in registry:
        src = entry["source"]
        st = entry.get("status", "UNKNOWN")
        if src not in source_counts:
            source_counts[src] = {"PASS": 0, "FAIL": 0, "MISSING": 0, "REF": 0}
        if st in source_counts[src]:
            source_counts[src][st] += 1

    lines.append("## By Source Type")
    lines.append("")
    lines.append("| Source | Pass | Fail | Missing |")
    lines.append("|--------|------|------|---------|")
    for src in sorted(source_counts.keys()):
        c = source_counts[src]
        lines.append(f"| {src} | {c['PASS']} | {c['FAIL']} | {c['MISSING']} |")
    lines.append("")

    # By section
    section_counts = {}
    for entry in registry:
        # Entries that declare no section (coverage-exempt source-value checks)
        # are attributed to the section(s) where their value appears in prose,
        # so the By-Section table accounts for every registration.
        entry_secs = entry["section"] or entry.get("display_section") or []
        for sec_item in entry_secs:
            sec_label = sec_item[0] if isinstance(sec_item, tuple) else sec_item
            if sec_label not in section_counts:
                section_counts[sec_label] = {"PASS": 0, "FAIL": 0, "MISSING": 0}
            st = entry.get("status", "UNKNOWN")
            if st in section_counts[sec_label]:
                section_counts[sec_label][st] += 1

    lines.append("## By Section")
    lines.append("")
    lines.append("| Section | Claims | Pass | Fail |")
    lines.append("|---------|--------|------|------|")
    for sec_label in sorted(section_counts.keys()):
        c = section_counts[sec_label]
        total_sec = c["PASS"] + c["FAIL"] + c["MISSING"]
        lines.append(f"| {sec_label} | {total_sec} | {c['PASS']} | {c['FAIL']} |")
    lines.append("")

    # Failed claims detail
    failures = [e for e in registry if e.get("status") == "FAIL"]
    if failures:
        lines.append("## Failed Claims")
        lines.append("")
        lines.append("| Name | Expected | Actual | Source |")
        lines.append("|------|----------|--------|--------|")
        for e in failures:
            actual = f"{e['actual']:.4f}" if isinstance(e.get("actual"), (int, float)) else "---"
            src = e["source"]
            lines.append(f"| {e['name']} | {e['value']} | {actual} | {src} |")
        lines.append("")

    # Unregistered numbers
    if unregistered_lines:
        lines.append("## Unregistered Numbers")
        lines.append("")
        lines.append(f"{len(unregistered_lines)} lines contain numbers not mapped to any verification entry:")
        lines.append("")
        for ln, sec, nums, text in unregistered_lines[:20]:
            nums_str = ", ".join(str(n) for n in nums)
            lines.append(f"- L{ln} [{sec}]: {nums_str}")
        if len(unregistered_lines) > 20:
            lines.append(f"- ... and {len(unregistered_lines) - 20} more")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
