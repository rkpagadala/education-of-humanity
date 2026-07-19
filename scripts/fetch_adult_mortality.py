"""
fetch_adult_mortality.py
========================
Fetch WB adult mortality rates (₄₅q₁₅ — probability of dying between
age 15 and 60, per 1000) for both sexes, save as wide CSVs to data/.

Used by: scripts/mothers_own_longevity.py
"""

import json
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")


def json_to_wide(json_path, out_csv):
    with open(json_path) as f:
        payload = json.load(f)
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"unexpected JSON shape in {json_path}")
    rows = payload[1]
    records = []
    for r in rows:
        if r.get("value") is None:
            continue
        records.append({
            "country": r["country"]["value"],
            "year": int(r["date"]),
            "value": float(r["value"]),
        })
    long = pd.DataFrame(records)
    wide = long.pivot(index="country", columns="year", values="value")
    wide = wide.reindex(columns=sorted(wide.columns))
    wide.index.name = "Country"  # match WB CSV convention used by load_wb()
    wide.to_csv(out_csv)
    print(f"  Wrote {out_csv}  shape={wide.shape}")
    return wide


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: fetch_adult_mortality.py <fe_json> <ma_json>")
        print("  (typical: /tmp/adult_mort_fe.json /tmp/adult_mort_ma.json)")
        sys.exit(2)
    fe_json, ma_json = sys.argv[1], sys.argv[2]
    json_to_wide(fe_json, os.path.join(DATA, "adult_mortality_female.csv"))
    json_to_wide(ma_json, os.path.join(DATA, "adult_mortality_male.csv"))
