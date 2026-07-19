# =============================================================================
# scripts/ml/longrun/fetch_gapminder.py
#
# Fetches OWID-republished Gapminder/Riley long-run series for LE, child
# mortality, TFR back to ~1800. Writes to data/external/gapminder/*.csv.
# Computes SHA256 and stamps PROVENANCE.md.
# =============================================================================
"""
Idempotent Gapminder/OWID fetcher. Re-running this script overwrites the
CSVs and updates the SHA256 + download-date entries in PROVENANCE.md.

We use OWID's grapher CSV endpoints with explicit time filters because
OWID republishes Gapminder + Riley + Zijdeman + HMD + UN-WPP as merged
historical series with stable URLs. Citations for the underlying
sources are recorded in PROVENANCE.md.
"""

import datetime as dt
import hashlib
import os
import re
import sys
import urllib.request

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEST = os.path.join(REPO_ROOT, "data", "external", "gapminder")

# OWID grapher slug -> output filename. The ?time= filter is critical: without
# it the CSV defaults to the post-1950 UN WPP slice and the long-run data is
# silently dropped.
OWID_BASE = "https://ourworldindata.org/grapher/{slug}.csv?time=1543..2025&useColumnShortNames=true"
SOURCES = {
    "life_expectancy_1800_2020.csv":   "life-expectancy",
    "child_mortality_1800_2020.csv":   "child-mortality",
    "tfr_1800_2020.csv":               "fertility-rate-complete-gapminder",
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


USER_AGENT = "longrun-fetcher/1.0 (Human-Development-Prediction; +https://github.com/rkpagadala)"


def fetch_one(filename, slug):
    url = OWID_BASE.format(slug=slug)
    out_path = os.path.join(DEST, filename)
    print(f"Fetching {filename} from {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
        f.write(resp.read())
    digest = _sha256(out_path)
    size = os.path.getsize(out_path)
    print(f"  -> {out_path} ({size} bytes, sha256={digest[:12]}...)")
    return digest


def update_provenance(digests):
    prov_path = os.path.join(REPO_ROOT, "data", "external", "PROVENANCE.md")
    today = dt.date.today().isoformat()
    with open(prov_path) as f:
        text = f.read()
    for filename, digest in digests.items():
        pattern_sha = rf"(### File: `gapminder/{re.escape(filename)}`[\s\S]*?Sha256: )`[^`]*`"
        pattern_dl  = rf"(### File: `gapminder/{re.escape(filename)}`[\s\S]*?Downloaded: )`[^`]*`"
        text = re.sub(pattern_sha, rf"\1`{digest}`", text)
        text = re.sub(pattern_dl,  rf"\1`{today}`", text)
    with open(prov_path, "w") as f:
        f.write(text)
    print(f"Updated provenance: {prov_path}")


def main():
    os.makedirs(DEST, exist_ok=True)
    digests = {}
    for filename, slug in SOURCES.items():
        digests[filename] = fetch_one(filename, slug)
    update_provenance(digests)
    print("All OWID/Gapminder long-run files fetched.")


if __name__ == "__main__":
    main()
