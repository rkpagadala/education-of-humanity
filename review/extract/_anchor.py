"""
review/extract/_anchor.py

Section-label-to-line-range anchoring for paper/education_of_humanity.tex.

Single source of truth shared between:
  - scripts/verify_humanity.py  (number registry)
  - review/extract/*.py         (review system extractors)

Public API:
  build_section_map(paper_path) -> dict[label, (start_line, end_line)]
  label_for_line(section_map, line_no) -> label | None
"""

import re
from typing import Dict, Optional, Tuple

ABSTRACT_LABEL = "abstract"

_HEADER_RE = re.compile(r'\\(?:sub)?section\*?\{.*?\}\\label\{([^}]+)\}')


def build_section_map(paper_path: str) -> Dict[str, Tuple[int, int]]:
    """Parse the .tex file and return label -> (start_line, end_line).

    Conventions:
      - Each \\section / \\subsection with an immediate \\label{} becomes an entry.
      - Range runs from the header line to the next header line - 1
        (or end of file for the last section).
      - Everything before the first header is mapped to "abstract".
      - Line numbers are 1-based, inclusive on both ends.
    """
    with open(paper_path) as f:
        lines = f.readlines()

    headers = []  # list of (line_no, label)
    for i, line in enumerate(lines, 1):
        m = _HEADER_RE.search(line)
        if m:
            headers.append((i, m.group(1)))

    section_map: Dict[str, Tuple[int, int]] = {}

    if headers:
        section_map[ABSTRACT_LABEL] = (1, headers[0][0] - 1)
    else:
        section_map[ABSTRACT_LABEL] = (1, len(lines))

    for idx, (line_no, label) in enumerate(headers):
        end = headers[idx + 1][0] - 1 if idx + 1 < len(headers) else len(lines)
        section_map[label] = (line_no, end)

    return section_map


def label_for_line(section_map: Dict[str, Tuple[int, int]], line_no: int) -> Optional[str]:
    """Return the most-specific section label containing line_no, or None."""
    best_label = None
    best_span = None  # narrower span wins
    for label, (start, end) in section_map.items():
        if start <= line_no <= end:
            span = end - start
            if best_span is None or span < best_span:
                best_label = label
                best_span = span
    return best_label
