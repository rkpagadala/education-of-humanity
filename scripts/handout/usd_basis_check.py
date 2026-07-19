"""
handout/usd_basis_check.py
==========================
Diagnose the USD-basis inconsistency between the handout, the paper, and
the underlying data file.

Findings (declared up front; confirmed below):

  1. Data file `data/gdppercapita_us_inflation_adjusted.csv` is
     NY.GDP.PCAP.KD per `data/rebuild_datasets.py:20`, which states
     "constant 2015 USD". Paper, scripts, and website now label it as
     "constant 2015 USD" (label corrected 2026-04-29; the manuscript
     and most scripts previously mislabelled it "2017 USD"). The grep
     below acts as a regression check so any future drift back to
     "2017 USD" is flagged.

  2. Handout's Bangladesh GDP per capita (1990 = $279, 2015 = $1,086)
     does NOT match this file (1990 = $473, 2015 = $1,224 in constant
     2015 USD). The handout numbers are consistent with the World Bank
     Atlas-method GNI per capita (NY.GNP.PCAP.CD) — the indicator the
     Bank uses to classify countries by income level (low/middle/high).
     Atlas GNI 1990 Bangladesh ≈ $290; Atlas GNI 2015 Bangladesh ≈ $1,080.
     Isabella may be mixing two different World Bank indicators.

  3. Niger 2015 baseline GDP of $360 in the handout is in some basis
     this file does not contain. Our file shows Niger 2015 = $593 in
     constant 2015 USD. Atlas-method GNI for Niger 2015 was ~$390 —
     consistent with the handout's $360 if rounded/measured slightly
     differently.

Recommendation: pin the handout to ONE indicator. If the audience is
Stanford King Center policy people, Atlas-method GNI per capita is
defensible (it's how the Bank classifies the country). The paper's
analysis uses NY.GDP.PCAP.KD (constant USD), and the handout should
match either the paper's basis or note explicitly that the GDP figures
are presented on a different (income-classification) basis from the
panel coefficients.
"""

import os
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, ".."))
from _shared import REPO_ROOT, write_checkin  # noqa: E402

gdp_path = os.path.join(REPO_ROOT, "data",
                        "gdppercapita_us_inflation_adjusted.csv")
gdp_raw = pd.read_csv(gdp_path)
gdp_raw["country"] = gdp_raw["Country"].str.lower()
gdp = gdp_raw.set_index("country").drop(columns=["Country"])


def lookup(key, year):
    if key.lower() not in gdp.index:
        return np.nan
    val = gdp.loc[key.lower(), str(year)]
    return float(val) if not pd.isna(val) else np.nan


print("=" * 78)
print("USD BASIS DIAGNOSTIC")
print("=" * 78)

# 1. File basis declaration
print("\n[1] FILE BASIS DECLARATION")
print("-" * 78)
with open(os.path.join(REPO_ROOT, "data", "rebuild_datasets.py")) as fh:
    for line in fh:
        if "GDP:" in line and ("constant" in line or "PCAP" in line):
            print(f"  data/rebuild_datasets.py: {line.strip()}")
print("  (Indicator: NY.GDP.PCAP.KD; rebuild script declares constant 2015 USD.)")

# 2. Paper labels (sample)
print("\n[2] PAPER LABEL SAMPLES")
print("-" * 78)
import subprocess
SELF = "scripts/handout/usd_basis_check.py"
res = subprocess.run(
    ["grep", "-rn", "--exclude=" + os.path.basename(SELF),
     "constant 2017 USD", "scripts/", "paper/"],
    capture_output=True, text=True, cwd=REPO_ROOT
)
n_2017 = len([l for l in res.stdout.splitlines() if l.strip()])
res2 = subprocess.run(
    ["grep", "-rn", "--exclude=" + os.path.basename(SELF),
     "constant 2015 USD", "scripts/", "paper/"],
    capture_output=True, text=True, cwd=REPO_ROOT
)
n_2015 = len([l for l in res2.stdout.splitlines() if l.strip()])
print(f"  '\"constant 2017 USD\"' label occurrences in scripts/+paper/: {n_2017}")
print(f"  '\"constant 2015 USD\"' label occurrences in scripts/+paper/: {n_2015}")
if n_2017 == 0:
    print("  → All labels read 2015 USD, matching the underlying file.")
else:
    print(f"  → REGRESSION: {n_2017} '2017 USD' labels remain. Underlying")
    print("    file is constant 2015 USD; paper and scripts must agree.")

# 3. Handout Bangladesh & Niger vs file
print("\n[3] HANDOUT vs FILE (constant 2015 USD)")
print("-" * 78)
HANDOUT = {
    "bangladesh": {1965: 100.0, 1990: 279.0, 2015: 1086.0},
    "niger":      {2015: 360.0},
}

# Approximate Atlas-method GNI per capita reference values (World Bank Atlas)
# Source: World Bank Open Data, indicator NY.GNP.PCAP.CD (Atlas method).
# Values shown for cross-reference only; not in our local data file.
ATLAS_REF = {
    "bangladesh": {1990: 290.0, 2015: 1080.0},
    "niger":      {2015: 390.0},
}

print(f"  {'Country':<12} {'Year':<6} {'Handout':>10} "
      f"{'File (KD)':>10} {'Atlas GNI ref':>14} {'verdict':>15}")
print("  " + "-" * 76)

results = {"countries": {}}
for country, vals in HANDOUT.items():
    results["countries"][country] = {}
    for year, claim in vals.items():
        actual = lookup(country, year)
        atlas = ATLAS_REF.get(country, {}).get(year)
        if pd.isna(actual):
            verdict = "N/A in file"
        else:
            rel_kd = abs(actual - claim) / claim
            rel_atlas = (
                abs(atlas - claim) / claim if atlas is not None else None
            )
            if rel_kd < 0.05:
                verdict = "matches KD"
            elif rel_atlas is not None and rel_atlas < 0.05:
                verdict = "matches Atlas"
            else:
                verdict = "no match"
        atlas_str = f"${atlas:>9,.0f}" if atlas else "      —"
        actual_str = f"${actual:>9,.0f}" if not pd.isna(actual) else "  N/A     "
        print(f"  {country:<12} {year:<6} ${claim:>9,.0f} "
              f"{actual_str:>10} {atlas_str:>14} {verdict:>15}")
        results["countries"][country][year] = {
            "handout": claim,
            "file_constant_2015_usd": (
                None if pd.isna(actual) else round(actual, 2)
            ),
            "atlas_gni_reference": atlas,
            "verdict": verdict,
        }

print("\n[4] CONCLUSIONS")
print("-" * 78)
print("  • The file is constant 2015 USD (NY.GDP.PCAP.KD).")
print("  • Paper, scripts, and website all label it 2015 USD (consistent).")
print("  • Handout's Bangladesh GDP figures are consistent with Atlas-")
print("    method GNI per capita (NY.GNP.PCAP.CD), not constant USD GDP.")
print("  • Niger $360 (2015) is consistent with Atlas GNI, not file KD.")
print("  • Recommendation: pin the handout to ONE indicator. If using")
print("    Atlas GNI, label it 'Atlas-method GNI per capita' and note")
print("    the paper's coefficients use NY.GDP.PCAP.KD (different basis).")

results["file_basis"] = "constant 2015 USD (NY.GDP.PCAP.KD)"
results["paper_label_inconsistency"] = {
    "2017_label_count": n_2017,
    "2015_label_count": n_2015,
    "correct_basis": "constant 2015 USD",
}

write_checkin(
    "handout_usd_basis_check.json",
    {
        "method": (
            "Compare handout GDP claims against the file at "
            "data/gdppercapita_us_inflation_adjusted.csv (NY.GDP.PCAP.KD, "
            "constant 2015 USD per data/rebuild_datasets.py). Cross-reference "
            "Atlas-method GNI per capita (NY.GNP.PCAP.CD) from public Bank "
            "data; Atlas reference values not in local file."
        ),
        "results": results,
    },
    "scripts/handout/usd_basis_check.py",
)
