"""
scripts/_citations.py

Citation verification for paper/the_long_childhood.tex.

Design (JSON-as-source):

  1. checkin/citations.json is the source of truth. Each entry has:
       - text  : full LaTeX text of the references-section entry
       - year  : int (for sanity / search)
       - doi   : DOI string (or null)
       - url   : canonical URL (or null)
     At least one of doi / url must be set.

     The dict's insertion order = the order entries appear in the paper.

  2. The verifier renders all entries' text fields, joined by blank
     lines, and string-compares against the paper's actual References
     section. If they differ, it writes the expected text to
     checkin/references_generated.tex and fails with a diff.

  3. Inline (Author Year) citations in the body must each resolve to a
     JSON key (or a declared alias). For each occurrence, the verifier
     emits a stable URL pointing to the section header on the public
     repo (the_long_childhood.tex on rkpagadala/the-long-childhood).

  4. DOI/URL format is validated.

Run standalone:

    python scripts/_citations.py            # verify (structure: refs match,
                                            #   inline cites resolve, DOI/URL format)
    python scripts/_citations.py --resolve  # live-check that every locator
                                            #   actually resolves (flags dead
                                            #   links; tolerates publisher bot-blocks)
    python scripts/_citations.py --bootstrap  # generate skeleton citations.json from paper
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PAPER = os.path.join(REPO_ROOT, "paper", "the_long_childhood.tex")
CITATIONS_JSON = os.path.join(REPO_ROOT, "checkin", "citations.json")
REFS_OUT = os.path.join(REPO_ROOT, "checkin", "references_generated.tex")
REPORT_OUT = os.path.join(REPO_ROOT, "checkin", "citations_report.md")

sys.path.insert(0, os.path.join(REPO_ROOT, "review", "extract"))
from _anchor import build_section_map, label_for_line  # noqa: E402

PUBLIC_REPO_BLOB = (
    "https://github.com/rkpagadala/the-long-childhood/blob/main/"
    "paper/the_long_childhood.tex"
)

# ──────────────────────────────────────────────────────────────────────────
# References-section extraction (from paper.tex)
# ──────────────────────────────────────────────────────────────────────────

_REF_HEADER_RE = re.compile(r"\\section\*\{References\}\\label\{references\}")
_END_DOC_RE = re.compile(r"\\end\{document\}")


def extract_references_block(paper_text: str) -> str:
    """Return the text of the paper's References section, between the
    header and \\end{document}, with leading/trailing whitespace stripped."""
    m = _REF_HEADER_RE.search(paper_text)
    if not m:
        return ""
    after = paper_text[m.end():]
    end_m = _END_DOC_RE.search(after)
    if end_m:
        after = after[: end_m.start()]
    return after.strip()


def split_entries(refs_block: str) -> List[str]:
    """Split the references block into individual entries on blank lines."""
    blocks = re.split(r"\n\s*\n", refs_block.strip())
    return [b.strip() for b in blocks if b.strip()]


def normalize_entry(entry_text: str) -> str:
    """Collapse internal whitespace (for diff-tolerant comparison)."""
    return re.sub(r"\s+", " ", entry_text).strip()


# ──────────────────────────────────────────────────────────────────────────
# Key derivation (must match registry keys)
# ──────────────────────────────────────────────────────────────────────────

def derive_key(entry_text: str) -> Optional[str]:
    """Derive '<first_surname>_<year>' from an entry's text.

    Used to bootstrap citations.json keys from the paper. The runtime
    verifier does NOT re-derive — it trusts the JSON's keys.
    """
    norm = normalize_entry(entry_text)
    norm = norm.replace(r"\&", "&").replace("~", " ")
    year_m = re.search(r"\((\d{4})[a-z]?(?:,\s*[^)]*)?\)", norm)
    if not year_m:
        return None
    year = year_m.group(1)
    authors_part = norm[: year_m.start()].strip().rstrip(",. ")
    head = re.split(r",\s+[A-Z]\.|\s+&\s+|\s+and\s+", authors_part, maxsplit=1)[0]
    head = head.strip().rstrip(",. ")
    tokens = [t.strip(",.") for t in head.split() if t.strip(",.")]
    if not tokens:
        return None
    t0 = tokens[0]
    t0_low = t0.lower()
    surname: str
    if len(tokens) >= 2 and tokens[1][:1].isupper():
        if t0_low in {"de", "do", "van", "von", "le", "la", "el", "ó",
                      "del", "della"}:
            surname = f"{t0}_{tokens[1]}"
        elif t0 in {"De", "Ó", "Da", "Le", "La"}:
            surname = f"{t0}_{tokens[1]}"
        elif t0 in {"National", "United", "World", "European", "Asian"}:
            surname = f"{t0}_{tokens[1]}"
        else:
            surname = t0
    else:
        surname = t0
    return f"{surname.lower()}_{year}"


# ──────────────────────────────────────────────────────────────────────────
# Registry I/O
# ──────────────────────────────────────────────────────────────────────────

def load_registry(path: str = CITATIONS_JSON) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("entries", data)  # support either flat or wrapped


def save_registry(entries: Dict[str, dict], path: str = CITATIONS_JSON) -> None:
    payload = {
        "_meta": {
            "schema_version": 1,
            "description": (
                "Citation registry for paper/the_long_childhood.tex. "
                "Insertion order = render order in the paper's References "
                "section. text is the full LaTeX paragraph for the entry. "
                "doi is preferred; url is fallback for books/reports/web. "
                "Verified by scripts/_citations.py via render-and-diff."
            ),
        },
        "entries": entries,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def render_references(entries: Dict[str, dict]) -> str:
    """Render the registry back to a References section body (entry texts
    separated by blank lines). Order = JSON insertion order."""
    texts = [e["text"].strip() for e in entries.values()]
    return "\n\n".join(texts) + "\n"


# ──────────────────────────────────────────────────────────────────────────
# Inline citation extraction
# ──────────────────────────────────────────────────────────────────────────

_PARTICLE = r"(?:d[eo']|del|della|le|la|van|von|el)\s+"
_NAME = (
    rf"(?:{_PARTICLE})?[A-ZÓÉÀ][A-Za-z\xc0-\xff'\-]+"
    rf"(?:\s+[A-ZÓÉÀ][A-Za-z\xc0-\xff'\-]+)?"
)
_AUTHORS = (
    rf"{_NAME}"
    rf"(?:,\s+{_NAME})?"
    rf"(?:\s+(?:and|&)\s+{_NAME})?"
    rf"(?:\s+et\s+al\.?)?"
)

_PAT_A = re.compile(rf"\b({_AUTHORS})(?:'s)?\s+\((\d{{4}})\)")
_PAT_B = re.compile(rf"\(({_AUTHORS})(?:'s)?\s+(\d{{4}})\)")
_PAT_C = re.compile(rf"\(({_AUTHORS})(?:'s)?,\s*(\d{{4}})\)")
# PAT_D: "(quoted in Brass 1986)" — only known prefixes, to avoid catching
# "log GDP 2020" or similar all-caps non-citations.
_PAT_D = re.compile(
    rf"\((?:quoted in|cited in|see e\.g\.|see also|see|e\.g\.,?|after|via)\s+"
    rf"({_AUTHORS})\s+(\d{{4}})\)"
)

_NOT_AUTHORS = {
    "section", "figure", "table", "appendix", "equation",
    "chapter", "page", "footnote", "listing", "see", "fig",
    "sections", "figures", "tables", "appendices", "chapters",
    "japan", "korea", "china", "india", "kerala", "cuba",
    "philippines", "bangladesh", "myanmar", "sri", "cambodia",
    "vietnam", "taiwan", "indonesia", "ghana", "rwanda", "botswana",
    "theory", "the", "an", "a",
    "in", "by", "for", "since", "from", "until", "about", "around",
}


@dataclass
class Citation:
    phrase: str
    first_surname: str
    year: int
    line: int
    section_label: Optional[str]
    section_start_line: Optional[int]

    @property
    def key(self) -> str:
        return f"{self.first_surname}_{self.year}"

    @property
    def url(self) -> str:
        target = self.section_start_line or self.line
        return f"{PUBLIC_REPO_BLOB}#L{target}"


def _normalize_phrase(phrase: str) -> str:
    a = re.sub(r"\\[a-zA-Z]+", "", phrase).strip()
    a = re.sub(r"\s+", " ", a)
    if a.endswith("'s"):
        a = a[:-2]
    return a


def _phrase_to_surname(phrase: str) -> str:
    """Map an inline citation phrase to a lowercase surname key.

    Handles:
      "Smith"                          -> "smith"
      "Adam Smith"                     -> "smith"  (Given+Surname → last)
      "de Waal"                        -> "de_waal"
      "De Walque"                      -> "de_walque"
      "Ó Gráda"                        -> "ó_gráda"
      "United Nations"                 -> "united_nations"
      "Hare and Woods"                 -> "hare"   (multi-author, lead surname)
      "Lutz, Muttarak and Striessnig"  -> "lutz"   (comma list → lead surname)
    """
    raw_tokens = phrase.split()
    # A trailing comma on the first token signals a comma-separated author
    # list ("Lutz, Muttarak and Striessnig"). The first token is the lead
    # surname; the Western Given+Surname heuristic must not fire.
    first_has_comma = bool(raw_tokens) and raw_tokens[0].endswith(",")

    tokens = [t.strip(",.") for t in raw_tokens if t.strip(",.")]
    while tokens and tokens[-1].lower().rstrip(".") in {"al", "et"}:
        tokens.pop()
    if not tokens:
        return ""
    # Stop at conjunction (drop subsequent authors for keying)
    cut: List[str] = []
    for t in tokens:
        if t.lower() in {"and", "&"}:
            break
        cut.append(t)
    tokens = cut
    if not tokens:
        return ""
    first = tokens[0]
    if first_has_comma:
        return first.lower().replace("'", "")
    if len(tokens) >= 2:
        t1 = tokens[1]
        # lowercase particle: "de Waal", "van Reenen"
        if first.lower() in {"de", "do", "van", "von", "le", "la", "el",
                             "ó", "del", "della"} and t1[:1].isupper():
            return f"{first}_{t1}".lower().replace("'", "")
        # capitalized particle: "De Walque", "Ó Gráda"
        if first in {"De", "Ó", "Da", "Le", "La"} and t1[:1].isupper():
            return f"{first}_{t1}".lower().replace("'", "")
        # Organization: "United Nations", "World Bank", "National Bureau"
        if first in {"United", "World", "National", "European", "Asian"}:
            return f"{first}_{t1}".lower().replace("'", "")
        # Two-capitalized-token Western name: "Adam Smith" → "smith"
        if first[:1].isupper() and t1[:1].isupper():
            return t1.lower().replace("'", "")
    return first.lower().replace("'", "")


def _line_start_offsets(text: str) -> List[int]:
    offsets = [0]
    for m in re.finditer("\n", text):
        offsets.append(m.end())
    return offsets


def _offset_to_line(offsets: List[int], pos: int) -> int:
    lo, hi = 0, len(offsets) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= pos:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi + 1


def extract_inline_citations(paper_text: str, paper_path: str) -> List[Citation]:
    end = paper_text.find(r"\section*{References}")
    body = paper_text if end == -1 else paper_text[:end]

    normalized = (body.replace(r"\&", " and").replace(r"\,", "   ")
                  .replace("~", " "))
    # Collapse runs of hyphens (em-dash "---", en-dash "--" in page ranges,
    # compound surnames "Mankiw--Romer--Weil") to a single hyphen so the
    # name regex sees them as single tokens.
    normalized = re.sub(r"-{2,}", "-", normalized)

    section_map = build_section_map(paper_path)
    line_starts = _line_start_offsets(body)

    hits: List[Citation] = []
    seen: Set[Tuple[int, int, str]] = set()

    for pat in (_PAT_A, _PAT_B, _PAT_C, _PAT_D):
        for m in pat.finditer(normalized):
            phrase = _normalize_phrase(m.group(1))
            try:
                year = int(m.group(2))
            except ValueError:
                continue
            surname = _phrase_to_surname(phrase)
            if not surname or surname.split("_")[0] in _NOT_AUTHORS:
                continue
            line_no = _offset_to_line(line_starts, m.start())
            sec = label_for_line(section_map, line_no)
            sec_start = section_map.get(sec, (None, None))[0] if sec else None
            uniq = (line_no, year, surname)
            if uniq in seen:
                continue
            seen.add(uniq)
            hits.append(Citation(
                phrase=phrase,
                first_surname=surname,
                year=year,
                line=line_no,
                section_label=sec,
                section_start_line=sec_start,
            ))
    return hits


# ──────────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────────

_DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9<>]+$")
_URL_RE = re.compile(r"^https?://\S+$")


@dataclass
class CitationReport:
    entries: Dict[str, dict]
    cites: List[Citation]
    paper_refs_text: str
    rendered_refs_text: str
    refs_diff: List[str] = field(default_factory=list)
    missing_in_registry: Set[str] = field(default_factory=set)
    bad_doi: List[Tuple[str, str]] = field(default_factory=list)
    bad_url: List[Tuple[str, str]] = field(default_factory=list)
    no_locator: List[str] = field(default_factory=list)
    unused_in_body: Set[str] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return (not self.refs_diff
                and not self.missing_in_registry
                and not self.bad_doi
                and not self.bad_url
                and not self.no_locator)


def _alias_keys(entry: dict) -> List[str]:
    return list(entry.get("aliases", []))


def _all_keys_for(entry_key: str, entry: dict) -> Set[str]:
    return {entry_key} | set(_alias_keys(entry))


def diff_references(paper_refs: str, rendered_refs: str) -> List[str]:
    """Compare entry-by-entry after normalizing whitespace.

    Returns a unified diff (list of lines) if they differ, else []."""
    paper_entries = [normalize_entry(e) for e in split_entries(paper_refs)]
    rendered_entries = [normalize_entry(e) for e in split_entries(rendered_refs)]
    if paper_entries == rendered_entries:
        return []
    diff = list(difflib.unified_diff(
        paper_entries,
        rendered_entries,
        fromfile="paper References section",
        tofile="rendered from citations.json",
        lineterm="",
        n=1,
    ))
    return diff


def verify(paper_path: str = PAPER, registry_path: str = CITATIONS_JSON) -> CitationReport:
    with open(paper_path) as f:
        paper_text = f.read()

    entries = load_registry(registry_path)
    cites = extract_inline_citations(paper_text, paper_path)
    paper_refs = extract_references_block(paper_text)
    rendered = render_references(entries) if entries else ""

    report = CitationReport(
        entries=entries,
        cites=cites,
        paper_refs_text=paper_refs,
        rendered_refs_text=rendered,
    )

    report.refs_diff = diff_references(paper_refs, rendered)

    # Build alias-aware key set
    all_known: Set[str] = set()
    for k, e in entries.items():
        all_known |= _all_keys_for(k, e)

    # Inline citations → registry
    for c in cites:
        if c.key not in all_known:
            report.missing_in_registry.add(c.key)

    # Unused registry entries (warning, not failure)
    cited_keys = {c.key for c in cites}
    for k, e in entries.items():
        keys_for_this = _all_keys_for(k, e)
        if not (keys_for_this & cited_keys):
            report.unused_in_body.add(k)

    # DOI / URL validation
    for key, entry in entries.items():
        doi = entry.get("doi")
        url = entry.get("url")
        if doi and not _DOI_RE.match(doi):
            report.bad_doi.append((key, doi))
        if url and not _URL_RE.match(url):
            report.bad_url.append((key, url))
        if not doi and not url:
            report.no_locator.append(key)

    return report


# ──────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────

def print_summary(report: CitationReport, verbose: bool = False) -> None:
    print()
    print("CITATION VERIFICATION")
    print("=" * 70)
    print(f"Registry entries:       {len(report.entries)}")
    print(f"Inline citations:       {len(report.cites)}")
    print()

    failures = 0

    if report.refs_diff:
        print("FAIL: References section does not match rendered citations.json")
        print(f"      Expected output written to {REFS_OUT}")
        for line in report.refs_diff[:40]:
            print(f"    {line}")
        if len(report.refs_diff) > 40:
            print(f"    ... ({len(report.refs_diff) - 40} more diff lines)")
        print()
        failures += 1

    if report.missing_in_registry:
        failures += len(report.missing_in_registry)
        print(f"FAIL: {len(report.missing_in_registry)} inline citation(s) "
              f"with no registry entry:")
        for k in sorted(report.missing_in_registry):
            occs = [c for c in report.cites if c.key == k]
            secs = sorted({c.section_label or "?" for c in occs})
            print(f"    {k}  in sections: {secs}")
        print()

    if report.bad_doi:
        failures += len(report.bad_doi)
        print(f"FAIL: {len(report.bad_doi)} malformed DOI(s):")
        for k, doi in report.bad_doi:
            print(f"    {k}: {doi}")
        print()

    if report.bad_url:
        failures += len(report.bad_url)
        print(f"FAIL: {len(report.bad_url)} malformed URL(s):")
        for k, url in report.bad_url:
            print(f"    {k}: {url}")
        print()

    if report.no_locator:
        failures += len(report.no_locator)
        print(f"FAIL: {len(report.no_locator)} registry entry(ies) with "
              f"neither doi nor url:")
        for k in report.no_locator:
            print(f"    {k}")
        print()

    if report.unused_in_body and verbose:
        print(f"WARN: {len(report.unused_in_body)} entry(ies) not cited "
              f"in body (allowed):")
        for k in sorted(report.unused_in_body):
            print(f"    {k}")
        print()

    if failures == 0:
        print(f"PASS: References section matches; all {len(report.cites)} "
              f"inline citations resolved.")
    else:
        print(f"FAILED: {failures} issue(s).")


def write_markdown_report(report: CitationReport, out_path: str) -> None:
    lines = []
    lines.append("# Citation Registry — the_long_childhood")
    lines.append("")
    lines.append("Generated by `scripts/_citations.py`.")
    lines.append("")
    lines.append(f"- Registry entries: **{len(report.entries)}**")
    lines.append(f"- Inline citations: **{len(report.cites)}**")
    lines.append(f"- Unused entries: **{len(report.unused_in_body)}**")
    lines.append("")
    lines.append("| Key | Year | DOI / URL | Sections in paper |")
    lines.append("|-----|------|-----------|-------------------|")

    cite_by_key: Dict[str, List[Citation]] = {}
    for c in report.cites:
        cite_by_key.setdefault(c.key, []).append(c)

    for key, entry in report.entries.items():
        doi = entry.get("doi")
        url = entry.get("url")
        year = entry.get("year", "")
        if doi:
            locator = f"[`{doi}`](https://doi.org/{doi})"
        elif url:
            locator = f"[link]({url})"
        else:
            locator = "—"

        occs = cite_by_key.get(key, [])
        for alias in _alias_keys(entry):
            occs.extend(cite_by_key.get(alias, []))
        seen_sec: Set[str] = set()
        cells = []
        for c in occs:
            if not c.section_label or c.section_label in seen_sec:
                continue
            seen_sec.add(c.section_label)
            cells.append(f"[{c.section_label}]({c.url})")
        secs = ", ".join(cells) if cells else "—"

        lines.append(f"| `{key}` | {year} | {locator} | {secs} |")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ──────────────────────────────────────────────────────────────────────────
# Bootstrap (one-shot generation from paper)
# ──────────────────────────────────────────────────────────────────────────

def bootstrap_from_paper(paper_path: str = PAPER) -> Dict[str, dict]:
    """Parse the paper's References section into a starter registry.

    Each entry has {year, text, doi: null, url: null}. The user fills
    in DOI/URL afterwards.
    """
    with open(paper_path) as f:
        paper_text = f.read()
    refs_block = extract_references_block(paper_text)
    raw_entries = split_entries(refs_block)

    entries: Dict[str, dict] = {}
    duplicates: List[str] = []
    for raw in raw_entries:
        key = derive_key(raw)
        if not key:
            print(f"  (skipped: could not derive key from) {raw[:60]}...")
            continue
        if key in entries:
            # Append suffix to disambiguate Mokyr 1990, etc. only if needed
            base = key
            suffix = "b"
            while key in entries:
                key = f"{base}{suffix}"
                suffix = chr(ord(suffix) + 1)
            duplicates.append(key)
        year_m = re.search(r"\((\d{4})", raw)
        year = int(year_m.group(1)) if year_m else None
        entries[key] = {
            "year": year,
            "doi": None,
            "url": None,
            "text": raw,
        }
    if duplicates:
        print(f"  (renamed {len(duplicates)} duplicate keys with suffix)")
    return entries


# ──────────────────────────────────────────────────────────────────────────
# Live link resolution (--resolve)
# ──────────────────────────────────────────────────────────────────────────

# Publishers routinely return these for non-browser clients even when the
# page is live (Cloudflare / anti-bot / paywall gates). Treat them as
# "reachable, not dead" so the check doesn't drown in false positives.
_BOT_BLOCK_CODES = {401, 403, 405, 406, 429}

_RESOLVE_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _locator_url(entry: dict) -> Optional[str]:
    """The single URL a reader follows for this entry: DOI if present
    (doi preferred), else the url field."""
    doi = entry.get("doi")
    if doi:
        return f"https://doi.org/{doi}"
    return entry.get("url")


def _is_transient(status: object) -> bool:
    """A transport error or a 5xx is server-side flakiness, not a dead link
    — worth a retry. (A PLOS DOI returns an intermittent 502 under load even
    though the article is live.)"""
    if isinstance(status, str):
        return True  # DNS / TLS / timeout
    return status >= 500


def _http_once(url: str, timeout: int) -> Tuple[object, str]:
    """One resolution attempt. Returns (status, final_url): an int HTTP code,
    or an 'ERR:<reason>' string on a transport/DNS failure. Tries HEAD first,
    falls back to GET when HEAD is refused."""
    import ssl
    import urllib.error
    import urllib.request

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    headers = {"User-Agent": _RESOLVE_UA, "Accept": "*/*"}

    last_err = "unknown"
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return r.status, r.geturl()
        except urllib.error.HTTPError as e:
            # A HEAD refusal is worth retrying with GET; a real status is final.
            if method == "HEAD" and e.code in (400, 405, 406, 501):
                continue
            return e.code, getattr(e, "url", url) or url
        except Exception as e:  # noqa: BLE001 — DNS, TLS, timeout, etc.
            last_err = f"{type(e).__name__}: {str(e)[:80]}"
            continue
    return f"ERR:{last_err}", url


def _http_status(url: str, timeout: int = 30, attempts: int = 3) -> Tuple[object, str]:
    """Resolve a URL, retrying transient failures (5xx / transport errors)
    with a short backoff so a flaky gateway is not reported as a dead link.
    The last non-transient result — or the last transient one if every
    attempt flaked — is returned."""
    import time

    status, final = _http_once(url, timeout)
    for attempt in range(1, attempts):
        if not _is_transient(status):
            break
        time.sleep(attempt)  # 1s, 2s, ... linear backoff
        status, final = _http_once(url, timeout)
    return status, final


def resolve_links(entries: Dict[str, dict],
                  workers: int = 12) -> List[Tuple[str, str, object, str]]:
    """Live-check every entry's locator URL concurrently.

    Returns a list of (key, url, status, final_url). A status is "dead"
    if it is a hard 404/410, a 5xx, or a transport error — those are real
    broken references. Bot-block codes and 2xx/3xx are reachable."""
    import concurrent.futures as cf

    targets = [(k, _locator_url(v)) for k, v in entries.items()]
    rows: List[Tuple[str, str, object, str]] = []

    def work(item):
        k, url = item
        if not url:
            return (k, "", "ERR:no-locator", "")
        status, final = _http_status(url)
        return (k, url, status, final)

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(work, targets))
    return rows


def _is_dead(status: object) -> bool:
    if isinstance(status, str):
        return True  # transport error / no locator
    if status in _BOT_BLOCK_CODES:
        return False
    return not (200 <= status < 400)


def run_resolve(registry_path: str = CITATIONS_JSON) -> int:
    entries = load_registry(registry_path)
    rows = resolve_links(entries)

    dead = [r for r in rows if _is_dead(r[2])]
    blocked = [r for r in rows
               if isinstance(r[2], int) and r[2] in _BOT_BLOCK_CODES]

    print()
    print("LINK RESOLUTION")
    print("=" * 70)
    print(f"Checked:        {len(rows)} locator(s)")
    print(f"Reachable:      {len(rows) - len(dead)}  "
          f"(incl. {len(blocked)} publisher bot-block, reachable)")
    print(f"Dead:           {len(dead)}")
    print()

    if dead:
        print("DEAD LINKS (broken references — fix the doi/url in "
              "citations.json):")
        for k, url, status, final in sorted(dead, key=lambda r: str(r[2])):
            print(f"    [{status}] {k}")
            print(f"        {url}")
            if final and final != url:
                print(f"        -> {final}")
        print()
        print(f"FAILED: {len(dead)} dead link(s).")
        return 1

    print(f"PASS: all {len(rows)} locators reachable.")
    return 0


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def _main() -> int:
    if "--bootstrap" in sys.argv:
        entries = bootstrap_from_paper()
        save_registry(entries)
        print(f"Wrote {len(entries)} entries to {CITATIONS_JSON}")
        return 0

    if "--resolve" in sys.argv:
        return run_resolve()

    report = verify()
    print_summary(report, verbose="-v" in sys.argv)

    # Always write the generated tex (it's checked against paper)
    with open(REFS_OUT, "w") as f:
        f.write(report.rendered_refs_text)

    write_markdown_report(report, REPORT_OUT)
    print(f"Report:    {REPORT_OUT}")
    print(f"Rendered:  {REFS_OUT}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(_main())
