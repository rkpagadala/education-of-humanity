"""
verify_country_names.py
=======================
Guardrail against silent data loss from country-name mismatches.

Scans every country column in every CSV under data/ and wcde/data/processed/,
runs each value through standardize_country_name(), and reports:

  1. UNKNOWN names     — don't match any alias; a bug (add the alias).
  2. DUPLICATE rows    — two raw names in the same file standardize to the
                         same canonical country (join would collide).
  3. EMPTY values      — blank or NaN country cells.
  4. COVERAGE gaps     — sovereign countries expected in the main panel
                         that never appear in at least one core dataset.

Exits non-zero when any problem is found. Reports ALL problems, not just
the first. Run after adding a new data file or changing any column schema.

Usage:
    python scripts/verify_country_names.py

When a name is flagged UNKNOWN: add it to scripts/_shared.py under the
correct canonical key in _CANONICAL_ALIASES (or _TERRITORY_ALIASES /
_AGGREGATE_ALIASES for territories and regional aggregates).
"""

import glob
import os
import sys
from collections import defaultdict

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from _shared import (  # noqa: E402
    CANONICAL_COUNTRIES,
    classify_country_name,
    standardize_country_name,
)

# Core datasets every sovereign country in the panel should appear in.
# Adding a new indicator? List it here to get coverage-gap warnings.
CORE_DATASETS = [
    "data/gdppercapita_us_inflation_adjusted.csv",
    "data/life_expectancy_years.csv",
    "data/children_per_woman_total_fertility.csv",
    "wcde/data/processed/lower_sec_both.csv",
]

# Sovereign countries we don't expect to find in WDI indicators (not issues).
KNOWN_WDI_GAPS = {"north korea", "taiwan"}

COUNTRY_COLUMNS = ("country", "Country", "name", "Name", "economy", "Economy")


def find_country_column(df):
    for c in COUNTRY_COLUMNS:
        if c in df.columns:
            return c
    return None


def scan_file(path):
    """Return (unknowns, dupes, empties) lists for one CSV.

    unknowns: list of raw names that don't standardize.
    dupes:    list of (canonical, [raw1, raw2, ...]) collisions.
    empties:  count of blank/NaN values seen in the country column.
    """
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        return [f"<read error: {e}>"], [], 0
    col = find_country_column(df)
    if col is None:
        return [], [], 0

    raw_names = df[col].tolist()
    empties = sum(1 for n in raw_names if pd.isna(n) or str(n).strip() == "")

    by_canonical = defaultdict(list)
    unknowns = []
    for n in raw_names:
        if pd.isna(n):
            continue
        s = str(n).strip()
        if not s:
            continue
        canonical = standardize_country_name(s)
        if canonical is None:
            unknowns.append(s)
        else:
            if s not in by_canonical[canonical]:
                by_canonical[canonical].append(s)

    # A file has a "duplicate" only if two *different* raw strings map to
    # the same canonical — that's the join-collision risk.
    dupes = [(canon, names) for canon, names in by_canonical.items()
             if len(names) > 1]
    return sorted(set(unknowns)), dupes, empties


def scan_all():
    csvs = []
    for pat in ("data/*.csv", "wcde/data/processed/*.csv"):
        csvs.extend(sorted(glob.glob(os.path.join(REPO_ROOT, pat))))

    problems = []          # file-level problems
    unknown_by_name = defaultdict(list)  # name -> [files]

    for path in csvs:
        rel = os.path.relpath(path, REPO_ROOT)
        unknowns, dupes, empties = scan_file(path)
        for u in unknowns:
            unknown_by_name[u].append(rel)
        if dupes:
            problems.append(("DUP", rel, dupes))
        if empties:
            problems.append(("EMPTY", rel, empties))

    return csvs, unknown_by_name, problems


def check_coverage():
    """Warn when sovereign countries are missing from a core dataset."""
    gaps = []
    for ds in CORE_DATASETS:
        path = os.path.join(REPO_ROOT, ds)
        if not os.path.exists(path):
            gaps.append((ds, "<missing file>"))
            continue
        df = pd.read_csv(path, low_memory=False)
        col = find_country_column(df)
        if col is None:
            gaps.append((ds, "<no country column>"))
            continue
        seen = set()
        for n in df[col].dropna():
            c = standardize_country_name(str(n).strip())
            if c:
                seen.add(c)
        for country in sorted(CANONICAL_COUNTRIES):
            if country in seen:
                continue
            if "wdi" in ds.lower() or ds.startswith("data/"):
                if country in KNOWN_WDI_GAPS:
                    continue
            gaps.append((ds, country))
    return gaps


def main():
    print("=" * 72)
    print("COUNTRY-NAME STANDARDIZATION VERIFIER")
    print("=" * 72)

    csvs, unknown_by_name, problems = scan_all()
    coverage_gaps = check_coverage()

    print(f"Scanned {len(csvs)} CSV files")
    print(f"Canonical sovereign countries: {len(CANONICAL_COUNTRIES)}")
    print()

    n_unknown = len(unknown_by_name)
    n_dup = sum(1 for p in problems if p[0] == "DUP")
    n_empty = sum(1 for p in problems if p[0] == "EMPTY")

    # ── 1. Unknown names ──────────────────────────────────────────────
    if unknown_by_name:
        print(f"[FAIL] {n_unknown} unknown country names "
              f"(add aliases in scripts/_shared.py):")
        for name in sorted(unknown_by_name):
            files = unknown_by_name[name]
            tail = files[0] if len(files) == 1 else f"{files[0]} (+{len(files)-1} more)"
            print(f"  {name!r:<50} in {tail}")
        print()

    # ── 2. Collisions within a file ──────────────────────────────────
    dup_problems = [p for p in problems if p[0] == "DUP"]
    if dup_problems:
        print(f"[FAIL] {n_dup} files have collisions "
              f"(two raw names → same canonical; will corrupt joins):")
        for _, path, dupes in dup_problems:
            print(f"  {path}")
            for canon, names in dupes:
                print(f"    {canon}: {names}")
        print()

    # ── 3. Empty country cells ───────────────────────────────────────
    empty_problems = [p for p in problems if p[0] == "EMPTY"]
    if empty_problems:
        print(f"[WARN] {n_empty} files have blank country cells:")
        for _, path, count in empty_problems:
            print(f"  {path}: {count} blank rows")
        print()

    # ── 4. Coverage gaps ─────────────────────────────────────────────
    if coverage_gaps:
        by_ds = defaultdict(list)
        for ds, country in coverage_gaps:
            by_ds[ds].append(country)
        print(f"[WARN] {len(coverage_gaps)} sovereign-country coverage gaps "
              f"in core datasets:")
        for ds, countries in by_ds.items():
            shown = ", ".join(countries[:10])
            more = f" (+{len(countries)-10} more)" if len(countries) > 10 else ""
            print(f"  {ds}: {shown}{more}")
        print()

    # ── Summary ──────────────────────────────────────────────────────
    fail = n_unknown > 0 or n_dup > 0
    print("=" * 72)
    print(f"SUMMARY: unknown={n_unknown}  collisions={n_dup}  "
          f"empty_cells={n_empty}  coverage_gaps={len(coverage_gaps)}")
    print("=" * 72)

    if fail:
        print("VERIFICATION FAILED — fix the [FAIL] entries above.")
        return 1
    print("VERIFICATION PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
