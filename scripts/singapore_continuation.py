"""
Singapore post-secondary continuation rate from WCDE v3.

Supports the Chapter 2 / Chapter 5 claim that when education is offered
past the dependency window's close (age ~18), most of the cohort chooses
to continue — evidence that educational dose is continuous within and
just past the biological dependency window, not thresholded at 9 years.

Measure: WCDE college completion among 20-24-year-olds (both sexes) for
Singapore, latest observed vintage (2020). Also reports the conditional
continuation rate: college completion as a share of upper-secondary
completion.

Source: WCDE v3 processed CSVs (wcde/data/processed/).
"""

import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WCDE = ROOT / "wcde" / "data" / "processed"
CHECKIN = ROOT / "checkin" / "singapore_continuation.json"

YEAR = "2020"
COUNTRY = "Singapore"

college = pd.read_csv(WCDE / "college_both.csv")
upper = pd.read_csv(WCDE / "upper_sec_both.csv")

col_sg = float(college.loc[college["country"] == COUNTRY, YEAR].iloc[0])
up_sg = float(upper.loc[upper["country"] == COUNTRY, YEAR].iloc[0])
conditional = 100.0 * col_sg / up_sg

print(f"Singapore, {YEAR} (WCDE v3, both sexes, age 20-24)")
print(f"  Upper secondary completion: {up_sg:.2f}%")
print(f"  College (tertiary) completion: {col_sg:.2f}%")
print(f"  Conditional continuation (college | upper secondary): {conditional:.2f}%")

results = {
    "results": {
        "singapore_college_2020": {
            "expected": 73.0,
            "actual": round(col_sg, 2),
            "status": "PASS" if abs(col_sg - 73.0) < 1.0 else "FAIL",
        },
        "singapore_upper_sec_2020": {
            "expected": 96.0,
            "actual": round(up_sg, 2),
            "status": "PASS" if abs(up_sg - 96.0) < 1.0 else "FAIL",
        },
        "singapore_conditional_continuation": {
            "expected": 76.0,
            "actual": round(conditional, 2),
            "status": "PASS" if abs(conditional - 76.0) < 1.0 else "FAIL",
        },
    },
    "overall": "PASS",
    "script": "scripts/singapore_continuation.py",
    "source": "WCDE v3 (wcde/data/processed/college_both.csv, upper_sec_both.csv)",
}

overall = "PASS" if all(r["status"] == "PASS" for r in results["results"].values()) else "FAIL"
results["overall"] = overall

CHECKIN.parent.mkdir(parents=True, exist_ok=True)
with open(CHECKIN, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nCheckin written to {CHECKIN.relative_to(ROOT)}")
