"""
Verify Kerala TFR and LE crossing dates claimed in the paper (Table 4).

Source: India Sample Registration System (SRS) published reports.
Data file: data/kerala_srs.csv

Paper claims:
  - TFR first fell below 3.65 around 1973
  - LE first exceeded 69.8 around 1981-1982
  - Both crossed by ~1982
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _shared import DATA, write_checkin

# ── Load Kerala SRS data ──
df = pd.read_csv(os.path.join(DATA, "kerala_srs.csv"))

TFR_THRESHOLD = 3.65
LE_THRESHOLD = 69.8


def interpolate_crossing(series, threshold, direction="below"):
    """
    Find the year a series crosses a threshold by linear interpolation.
    direction='below': find when series first drops below threshold.
    direction='above': find when series first rises above threshold.
    """
    clean = series.dropna().sort_index()
    for i in range(len(clean) - 1):
        y1 = clean.index[i]
        y2 = clean.index[i + 1]
        v1 = clean.iloc[i]
        v2 = clean.iloc[i + 1]

        if direction == "below" and v1 >= threshold > v2:
            frac = (v1 - threshold) / (v1 - v2)
            return y1 + frac * (y2 - y1)
        elif direction == "above" and v1 <= threshold < v2:
            frac = (threshold - v1) / (v2 - v1)
            return y1 + frac * (y2 - y1)
    return None


# ── TFR crossing ──
tfr_series = df.dropna(subset=["tfr"]).set_index("year")["tfr"]
tfr_crossing = interpolate_crossing(tfr_series, TFR_THRESHOLD, direction="below")

print(f"TFR threshold: {TFR_THRESHOLD}")
print(f"TFR crossing year (interpolated): {tfr_crossing:.1f}" if tfr_crossing else "TFR: no crossing found")
print(f"Paper claims: ~1973")
print()

# Show surrounding values
print("TFR values around crossing:")
for yr in [1970, 1971, 1972, 1973, 1974, 1975]:
    row = df[df["year"] == yr]
    if not row.empty and pd.notna(row["tfr"].values[0]):
        marker = " <-- threshold" if row["tfr"].values[0] < TFR_THRESHOLD else ""
        print(f"  {yr}: {row['tfr'].values[0]:.1f}{marker}")
print()

# ── LE crossing ──
le_series = df.dropna(subset=["le"]).set_index("year")["le"]
le_crossing = interpolate_crossing(le_series, LE_THRESHOLD, direction="above")

print(f"LE threshold: {LE_THRESHOLD}")
print(f"LE crossing year (interpolated): {le_crossing:.1f}" if le_crossing else "LE: no crossing found")
print(f"Paper claims: ~1981-1982")
print()

# Show surrounding values
print("LE values around crossing:")
for yr in [1978, 1980, 1981, 1982, 1985]:
    row = df[df["year"] == yr]
    if not row.empty and pd.notna(row["le"].values[0]):
        marker = " <-- threshold" if row["le"].values[0] >= LE_THRESHOLD else ""
        print(f"  {yr}: {row['le'].values[0]:.1f}{marker}")
print()

# ── Both crossed ──
if tfr_crossing and le_crossing:
    both_crossed = max(tfr_crossing, le_crossing)
    print(f"Both thresholds crossed by: ~{both_crossed:.0f}")
    print(f"Paper claims: ~1982")
else:
    both_crossed = None
    print("Cannot determine joint crossing.")

# ── Verification summary ──
tfr_ok = tfr_crossing is not None and abs(tfr_crossing - 1973) <= 2
le_ok = le_crossing is not None and abs(le_crossing - 1981.5) <= 2
both_ok = both_crossed is not None and abs(both_crossed - 1982) <= 2

print()
print("=" * 50)
print("VERIFICATION SUMMARY")
print("=" * 50)
print(f"TFR crossing ~1973: {'PASS' if tfr_ok else 'FAIL'} (actual: {tfr_crossing:.1f})" if tfr_crossing else "TFR: FAIL")
print(f"LE crossing ~1981:  {'PASS' if le_ok else 'FAIL'} (actual: {le_crossing:.1f})" if le_crossing else "LE: FAIL")
print(f"Both by ~1982:      {'PASS' if both_ok else 'FAIL'} (actual: ~{both_crossed:.0f})" if both_crossed else "Both: FAIL")

# ── Write checkin ──
write_checkin("kerala.json", {
    "source": "data/kerala_srs.csv (India Sample Registration System)",
    "thresholds": {"TFR": TFR_THRESHOLD, "LE": LE_THRESHOLD},
    "results": {
        "tfr_crossing": {
            "expected": 1973,
            "actual": round(tfr_crossing, 1) if tfr_crossing else None,
            "status": "PASS" if tfr_ok else "FAIL",
        },
        "le_crossing": {
            "expected": 1981,
            "actual": round(le_crossing, 1) if le_crossing else None,
            "status": "PASS" if le_ok else "FAIL",
        },
        "both_crossed": {
            "expected": 1982,
            "actual": round(both_crossed) if both_crossed else None,
            "status": "PASS" if both_ok else "FAIL",
        },
    },
    "overall": "PASS" if (tfr_ok and le_ok and both_ok) else "FAIL",
}, script_path="scripts/cases/kerala.py")
