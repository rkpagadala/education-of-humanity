"""
SDG 4 implied-pace verification.

SDG 4.1 (UN 2015) commits every signatory to universal primary and
secondary completion by 2030. From the most recent WCDE upper-secondary
completion baseline (20-24 cohort, both sexes), this script computes the
per-country pace required to reach universal coverage by 2030, restricted
to countries still meaningfully below the threshold.

Output: per-country median and mean implied pace; comparison to Korea's
WCDE-record sustained pace (2.13 pp/yr lower-secondary, 1955-1985, the
historical maximum across the panel).
"""
import sys
from pathlib import Path
import json
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _shared import load_education  # noqa: E402

REPO = SCRIPT_DIR.parent
CHECKIN = REPO / "checkin" / "sdg4_implied_pace.json"

BASELINE_YEAR = 2025
TARGET_YEAR = 2030
TARGET_PCT = 100.0
BELOW_THRESHOLD = 90.0   # "still meaningfully below" universal
KOREA_MAX_PP_YR = 2.13   # sustained Korean lower-sec pace, 1955-1985


def main():
    df = load_education("upper_sec_both.csv")
    if str(BASELINE_YEAR) not in df.columns:
        raise SystemExit(f"missing year {BASELINE_YEAR} in upper_sec_both.csv")

    base = df[["country", str(BASELINE_YEAR)]].rename(
        columns={str(BASELINE_YEAR): "upper_sec_pct"}
    ).dropna()

    below = base[base["upper_sec_pct"] < BELOW_THRESHOLD].copy()
    years_remaining = TARGET_YEAR - BASELINE_YEAR
    below["implied_pp_per_yr"] = (
        TARGET_PCT - below["upper_sec_pct"]
    ) / years_remaining

    paces = below["implied_pp_per_yr"].sort_values().tolist()
    n = len(paces)
    median_pace = below["implied_pp_per_yr"].median()
    mean_pace = below["implied_pp_per_yr"].mean()
    p25 = below["implied_pp_per_yr"].quantile(0.25)
    p75 = below["implied_pp_per_yr"].quantile(0.75)

    n_above_korea_max = (below["implied_pp_per_yr"] > KOREA_MAX_PP_YR).sum()
    n_above_2x_korea = (below["implied_pp_per_yr"] > 2 * KOREA_MAX_PP_YR).sum()
    n_above_4pp = (below["implied_pp_per_yr"] > 4.0).sum()

    out = {
        "baseline_year": BASELINE_YEAR,
        "target_year": TARGET_YEAR,
        "target_pct": TARGET_PCT,
        "below_threshold_filter": BELOW_THRESHOLD,
        "years_remaining": years_remaining,
        "korea_record_pp_per_yr": KOREA_MAX_PP_YR,
        "n_countries_below_threshold": int(n),
        "implied_pace_pp_per_yr": {
            "median": round(median_pace, 2),
            "mean": round(mean_pace, 2),
            "p25": round(p25, 2),
            "p75": round(p75, 2),
            "min": round(min(paces), 2),
            "max": round(max(paces), 2),
        },
        "n_countries_above_korea_max": int(n_above_korea_max),
        "n_countries_above_2x_korea_max": int(n_above_2x_korea),
        "n_countries_above_4pp_per_yr": int(n_above_4pp),
        "median_pace_exceeds_korea_max": bool(median_pace > KOREA_MAX_PP_YR),
        "median_pace_exceeds_4pp": bool(median_pace > 4.0),
    }

    print(json.dumps(out, indent=2))
    CHECKIN.parent.mkdir(parents=True, exist_ok=True)
    CHECKIN.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
