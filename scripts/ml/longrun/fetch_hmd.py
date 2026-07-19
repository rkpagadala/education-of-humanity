# =============================================================================
# scripts/ml/longrun/fetch_hmd.py
#
# HMD requires registration; we cannot auto-download. This script:
#   (1) prints manual fetch instructions on first run if files missing,
#   (2) verifies the files have the expected schema once present,
#   (3) computes SHA256 + stamps PROVENANCE.md.
# =============================================================================
"""
HMD verifier and provenance stamper. The actual download happens via
the user's HMD account at https://www.mortality.org/Data/ZippedDataFiles
and is a one-time manual step.

Expected output files at data/external/hmd/:
  e0_1880_1950.csv   columns: country, 1880, 1885, ..., 1950 (5-yr grid)
  u5mr_1880_1950.csv columns: country, 1880, 1885, ..., 1950 (5-yr grid)

Both should have rows for ~22 countries with continuous coverage
1880-1950.
"""

import datetime as dt
import hashlib
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEST = os.path.join(REPO_ROOT, "data", "external", "hmd")

EXPECTED_FILES = ["e0_1880_1950.csv", "u5mr_1880_1950.csv"]
EXPECTED_YEAR_COLS = list(range(1880, 1955, 5))   # 1880..1950 step 5
MIN_COUNTRIES = 12   # actual HMD continuous-coverage 1880-1950 subset is
                     # 12 European countries; USA/Japan/Canada start later
                     # (national vital statistics rolled out later).
                     # Partial-coverage countries still appear in the CSV
                     # with NaN; loader handles masking.

MANUAL_FETCH_INSTRUCTIONS = """
HMD data requires registration. Manual one-time fetch:

  1. Register at https://www.mortality.org/Account/UserAgreement
  2. Download "Life Tables 1×1 (period)" ZIPs for these countries:
       SWE, NOR, DNK, ISL, FIN, GBRTENW, GBR_SCO, NLD, BEL, FRATNP,
       CHE, ITA, ESP, DEUTNP, AUT, HUN, USA, CAN, AUS, NZL_NP, JPN
     (HMD codes; ~22 country ZIPs)
  3. Unpack each ZIP under data/external/hmd/raw/<HMD_CODE>/
     (so the path data/external/hmd/raw/SWE/STATS/bltper_1x1.txt exists)
  4. Run `scripts/ml/longrun/build_hmd_csv.py` to extract e0 and q5,
     interpolate to the 5-year grid, and write the two CSVs.
  5. Re-run THIS script to verify schema + stamp provenance.
"""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_one(path):
    """Verify schema: header has 'country' + the 16 year columns 1880..1950."""
    import pandas as pd
    df = pd.read_csv(path)
    if "country" not in df.columns:
        raise ValueError(f"{path}: missing 'country' column")
    have_years = [int(c) for c in df.columns if str(c).isdigit()]
    missing = set(EXPECTED_YEAR_COLS) - set(have_years)
    if missing:
        raise ValueError(f"{path}: missing year columns: {sorted(missing)}")
    full_cov = df.dropna(subset=[str(y) for y in EXPECTED_YEAR_COLS])
    n_full = len(full_cov)
    if n_full < MIN_COUNTRIES:
        raise ValueError(f"{path}: only {n_full} countries with continuous "
                         f"1880-1950 coverage; expected >= {MIN_COUNTRIES}")
    return {"n_countries_full": n_full, "rows": len(df)}


def update_provenance(digests, summaries):
    prov_path = os.path.join(REPO_ROOT, "data", "external", "PROVENANCE.md")
    today = dt.date.today().isoformat()
    with open(prov_path) as f:
        text = f.read()
    for filename, digest in digests.items():
        pattern_sha = rf"(### File: `hmd/{re.escape(filename)}`[\s\S]*?Sha256: )`[^`]*`"
        pattern_dl  = rf"(### File: `hmd/{re.escape(filename)}`[\s\S]*?Downloaded: )`[^`]*`"
        text = re.sub(pattern_sha, rf"\1`{digest}`", text)
        text = re.sub(pattern_dl,  rf"\1`{today}`", text)
    with open(prov_path, "w") as f:
        f.write(text)
    print(f"Updated provenance: {prov_path}")


def main():
    missing = [f for f in EXPECTED_FILES
               if not os.path.exists(os.path.join(DEST, f))]
    if missing:
        print(f"Missing HMD files: {missing}")
        print(MANUAL_FETCH_INSTRUCTIONS)
        sys.exit(2)

    digests = {}
    summaries = {}
    for filename in EXPECTED_FILES:
        path = os.path.join(DEST, filename)
        summary = verify_one(path)
        digests[filename] = _sha256(path)
        summaries[filename] = summary
        print(f"{filename}: {summary['n_countries_full']} countries, "
              f"{summary['rows']} rows, sha256={digests[filename][:12]}...")

    update_provenance(digests, summaries)
    print("HMD files verified.")


if __name__ == "__main__":
    main()
