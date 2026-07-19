# =============================================================================
# scripts/ml/longrun/build_hmd_csv.py
#
# Reads HMD raw life-table files (period, 1x1) for the ~22 first-mover
# countries, extracts e0 and computes u5mr = 1 - prod_{x=0..4}(1 - q_x) * 1000,
# interpolates to the 5-year grid 1880..1950, writes the two CSVs.
# =============================================================================
"""
HMD raw -> longrun CSV builder.

Expects HMD ZIPs to be unpacked under data/external/hmd/raw/<HMD_CODE>/
e.g. data/external/hmd/raw/SWE/STATS/E0per.txt        (period e0, F/M/Total)
     data/external/hmd/raw/SWE/STATS/bltper_1x1.txt   (period life table, both sexes)

HMD code mapping to canonical country names is below. Outputs:
  data/external/hmd/e0_1880_1950.csv
  data/external/hmd/u5mr_1880_1950.csv
"""

import os
import sys
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from _shared import standardize_country_name   # noqa: E402

HMD_DIR = os.path.join(REPO_ROOT, "data", "external", "hmd")
RAW_DIR = os.path.join(HMD_DIR, "raw")

# HMD code -> canonical country name (used in CSV's `country` column).
HMD_CODES = {
    "SWE": "Sweden", "NOR": "Norway", "DNK": "Denmark", "ISL": "Iceland",
    "FIN": "Finland", "GBRTENW": "England and Wales", "GBR_SCO": "Scotland",
    "NLD": "Netherlands", "BEL": "Belgium", "FRATNP": "France",
    "CHE": "Switzerland", "ITA": "Italy", "ESP": "Spain",
    "DEUTNP": "Germany", "AUT": "Austria", "HUN": "Hungary",
    "USA": "United States", "CAN": "Canada", "AUS": "Australia",
    "NZL_NP": "New Zealand", "JPN": "Japan",
}

YEAR_GRID = list(range(1880, 1955, 5))   # 1880..1950 step 5


def _read_e0_period(code):
    """Read HMD E0per.txt for one country. Returns Series indexed by year.

    HMD uses '.' as the missing-value sentinel; we coerce to NaN.
    """
    path = os.path.join(RAW_DIR, code, "STATS", "E0per.txt")
    df = pd.read_csv(path, sep=r"\s+", skiprows=2, na_values=["."])
    if "Year" not in df.columns or "Total" not in df.columns:
        raise ValueError(f"unexpected HMD E0 schema in {path}: cols={df.columns.tolist()}")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
    df = df.dropna(subset=["Year", "Total"])
    return df.set_index("Year")["Total"].astype(float)


def _read_u5mr_period(code):
    """Read HMD both-sexes period life table; compute u5mr per year as
    1 - prod_{x=0..4}(1 - q_x), expressed per 1000.

    HMD uses '.' as the missing-value sentinel; we coerce to NaN and drop.
    """
    path = os.path.join(RAW_DIR, code, "STATS", "bltper_1x1.txt")
    df = pd.read_csv(path, sep=r"\s+", skiprows=2, na_values=["."])
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Age"]  = pd.to_numeric(df["Age"], errors="coerce")
    df["qx"]   = pd.to_numeric(df["qx"], errors="coerce")
    df = df.dropna(subset=["Year", "Age", "qx"])
    df = df[df["Age"] < 5]

    def _calc(group):
        survive = (1.0 - group.set_index("Age")["qx"]).prod()
        return (1.0 - survive) * 1000.0

    return df.groupby("Year").apply(_calc).astype(float)


def _interpolate_to_grid(series, grid=YEAR_GRID):
    """Linear-interpolate a yearly series onto the 5-year grid."""
    if len(series) == 0:
        return {y: np.nan for y in grid}
    full = pd.Series(index=range(int(min(series.index)), int(max(series.index)) + 1),
                      dtype=float)
    full.update(series)
    full = full.interpolate(method="linear").ffill().bfill()
    out = {}
    for y in grid:
        out[y] = float(full.get(y, np.nan)) if y in full.index else np.nan
    return out


def build_e0():
    rows = []
    for code, name in HMD_CODES.items():
        try:
            ser = _read_e0_period(code)
            grid = _interpolate_to_grid(ser)
        except FileNotFoundError:
            print(f"  skipped (missing): {code} {name}")
            continue
        rows.append({"country": name, **grid})
    df = pd.DataFrame(rows)
    df = df[["country"] + YEAR_GRID]
    out = os.path.join(HMD_DIR, "e0_1880_1950.csv")
    df.to_csv(out, index=False)
    print(f"e0: wrote {len(df)} rows -> {out}")


def build_u5mr():
    rows = []
    for code, name in HMD_CODES.items():
        try:
            ser = _read_u5mr_period(code)
            grid = _interpolate_to_grid(ser)
        except FileNotFoundError:
            print(f"  skipped (missing): {code} {name}")
            continue
        rows.append({"country": name, **grid})
    df = pd.DataFrame(rows)
    df = df[["country"] + YEAR_GRID]
    out = os.path.join(HMD_DIR, "u5mr_1880_1950.csv")
    df.to_csv(out, index=False)
    print(f"u5mr: wrote {len(df)} rows -> {out}")


def main():
    os.makedirs(HMD_DIR, exist_ok=True)
    if not os.path.isdir(RAW_DIR):
        print(f"ERROR: HMD raw directory not found at {RAW_DIR}")
        print("Unpack HMD country ZIPs there first "
              "(see scripts/ml/longrun/fetch_hmd.py for instructions).")
        sys.exit(2)
    build_e0()
    build_u5mr()


if __name__ == "__main__":
    main()
