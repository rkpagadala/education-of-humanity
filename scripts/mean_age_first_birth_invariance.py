"""
mean_age_first_birth_invariance.py
====================================
Step 0 prior test (plan §0.2):

Prior under test (and FINDING):
  Mean age at childbearing (MAC) is roughly invariant across populations
  and education levels — a stable biological anchor for the generational
  cycle in the paper. The DATA put that anchor at MAC ≈ 28.7 (mean 28.69,
  median 28.81, SD 1.42; UN WPP 2024, 237 countries, 16,827 country-years;
  expansion-phase panel subset 28.75), NOT the round 25 an earlier draft
  assumed. The paper's generational lag is therefore 28 years (LAG_GENERATION),
  not 25. Note the education gradient runs OPPOSITE to the naive "uneducated
  have kids earlier" intuition: less-educated populations have HIGHER MAC
  (29.4 at <10% lower-sec) than more-educated (28.0 at ≥90%), because MAC
  is births-weighted across all parities — uneducated women start earlier
  but keep bearing later, raising the mean.

What we actually test (data permitting):
  1. Cross-country distribution of MAC, 1950-2020.
  2. MAC by lower-secondary completion bin (proxy for "education level"
     of the female population producing the births at that time).
  3. Within-country trajectory: how much does MAC shift over decades
     as a country's education stock rises?

Data:
  - UN WPP 2024 Demographic Indicators (Medium variant). MAC column.
    Downloaded from population.un.org/wpp/.
  - WCDE v3 lower-secondary completion (female, both sexes), 1950-2020.

Output:
  checkin/mean_age_first_birth_invariance.json

Interpretation rule:
  The anchor holds: MAC is tightly clustered (SD 1.42; p10-p90 span 3.6
  years) around ≈ 28.7 across populations and education bins, justifying a
  single ~28-year generational cycle. The dispersion is small enough that
  one round number (28) is defensible; the paper uses 28.
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import PROC, DATA, write_checkin

WPP_PATH = os.path.join(DATA, "external", "wpp",
                       "WPP2024_Demographic_Indicators_Medium.csv")
WCDE_LSEC = os.path.join(PROC, "lower_sec_both.csv")
WCDE_LSEC_F = os.path.join(PROC, "lower_sec_female.csv")


def load_wpp_mac():
    df = pd.read_csv(WPP_PATH, low_memory=False)
    df = df[df["LocTypeName"] == "Country/Area"].copy()
    df = df.dropna(subset=["MAC"])
    df["country"] = df["Location"].str.lower().str.strip()
    df["year"] = df["Time"].astype(int)
    # Restrict to observed years (1950-2023), not the projection range
    df = df[df["year"] <= 2023]
    return df[["country", "ISO3_code", "year", "MAC"]]


def load_wcde_lsec_female():
    df = pd.read_csv(WCDE_LSEC_F, index_col=0)
    df.columns = df.columns.astype(int)
    df.index = df.index.str.lower().str.strip()
    long = df.stack().reset_index()
    long.columns = ["country", "year", "lsec_female"]
    return long


def load_wcde_lsec_both():
    df = pd.read_csv(os.path.join(PROC, "lower_sec_both.csv"), index_col=0)
    df.columns = df.columns.astype(int)
    df.index = df.index.str.lower().str.strip()
    long = df.stack().reset_index()
    long.columns = ["country", "year", "lsec_both"]
    return long


def main():
    print("Loading UN WPP 2024 MAC...")
    mac = load_wpp_mac()
    print(f"  WPP MAC: {len(mac)} country-year obs, "
          f"{mac['country'].nunique()} countries, "
          f"years {mac['year'].min()}-{mac['year'].max()}")

    print("\nLoading WCDE female lower-secondary completion...")
    edu = load_wcde_lsec_female()
    print(f"  WCDE lsec female: {len(edu)} country-year obs, "
          f"{edu['country'].nunique()} countries")

    # Restrict to overlapping years
    edu = edu[(edu["year"] >= 1950) & (edu["year"] <= 2020)]
    mac = mac[(mac["year"] >= 1950) & (mac["year"] <= 2020)]

    # ── 1. Cross-country distribution of MAC ────────────────────────────
    print("\n" + "=" * 70)
    print("1. CROSS-COUNTRY DISTRIBUTION OF MAC (1950-2020 pooled)")
    print("=" * 70)
    mac_all = mac["MAC"].values
    pct = np.percentile(mac_all, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    print(f"  n = {len(mac_all):,} country-year obs ({mac['country'].nunique()} countries)")
    print(f"  Mean: {mac_all.mean():.2f}, SD: {mac_all.std():.2f}")
    print(f"  Min:  {mac_all.min():.2f}, Max: {mac_all.max():.2f}")
    print(f"  Percentiles  p01={pct[0]:.1f}  p05={pct[1]:.1f}  "
          f"p10={pct[2]:.1f}  p25={pct[3]:.1f}")
    print(f"               p50={pct[4]:.1f}  p75={pct[5]:.1f}  "
          f"p90={pct[6]:.1f}  p95={pct[7]:.1f}  p99={pct[8]:.1f}")

    # ── 2. MAC by lsec completion bin ────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. MAC BY FEMALE LOWER-SEC COMPLETION BIN")
    print("=" * 70)
    merged = mac.merge(edu, on=["country", "year"], how="inner")
    print(f"  Merged n = {len(merged):,}, "
          f"{merged['country'].nunique()} countries")

    bins = [0, 10, 25, 50, 75, 90, 101]
    labels = ["<10%", "10-25%", "25-50%", "50-75%", "75-90%", "≥90%"]
    merged["bin"] = pd.cut(merged["lsec_female"], bins=bins, labels=labels,
                            include_lowest=True, right=False)
    bin_stats = []
    print(f"  {'Bin':>10}  {'n':>6}  {'mean':>6}  {'sd':>5}  "
          f"{'p10':>5}  {'p50':>5}  {'p90':>5}")
    print("  " + "-" * 50)
    for lab in labels:
        sub = merged[merged["bin"] == lab]
        if len(sub) == 0:
            continue
        v = sub["MAC"].values
        bin_stats.append({
            "bin": lab,
            "n": len(v),
            "mean": round(v.mean(), 2),
            "sd": round(v.std(), 2),
            "p10": round(np.percentile(v, 10), 2),
            "p50": round(np.percentile(v, 50), 2),
            "p90": round(np.percentile(v, 90), 2),
        })
        print(f"  {lab:>10}  {len(v):>6}  {v.mean():>6.2f}  "
              f"{v.std():>5.2f}  {np.percentile(v, 10):>5.2f}  "
              f"{np.percentile(v, 50):>5.2f}  {np.percentile(v, 90):>5.2f}")

    # ── 3. Within-country trajectory ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. WITHIN-COUNTRY TRAJECTORY: MAC shift as edu rises")
    print("=" * 70)
    countries_with_both = (
        merged.groupby("country")
              .agg(lsec_min=("lsec_female", "min"),
                   lsec_max=("lsec_female", "max"),
                   mac_min=("MAC", "min"),
                   mac_max=("MAC", "max"),
                   n_obs=("MAC", "count"))
              .reset_index()
    )
    # Countries that crossed at least 50 percentage points of lsec
    big_shift = countries_with_both[
        (countries_with_both["lsec_max"] - countries_with_both["lsec_min"]) >= 50
    ].copy()
    big_shift["mac_range"] = big_shift["mac_max"] - big_shift["mac_min"]
    print(f"  Countries with ≥50pp lsec expansion: {len(big_shift)}")
    print(f"  MAC range (max - min) within these countries:")
    if len(big_shift) > 0:
        mr = big_shift["mac_range"].values
        print(f"    Mean: {mr.mean():.2f}, SD: {mr.std():.2f}")
        print(f"    p10: {np.percentile(mr, 10):.2f}, "
              f"p50: {np.percentile(mr, 50):.2f}, "
              f"p90: {np.percentile(mr, 90):.2f}")
        print(f"    Max country range: {big_shift.nlargest(5, 'mac_range')[['country','mac_min','mac_max','mac_range']].to_string(index=False)}")

    # ── 4. Verdict ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4. VERDICT ON PRIOR")
    print("=" * 70)
    overall_p10p90 = pct[2], pct[6]
    span = overall_p10p90[1] - overall_p10p90[0]
    print(f"  Prior: MAC ≈ 25, invariant across populations.")
    print(f"  Actual: median = {pct[4]:.1f}, p10-p90 span = "
          f"{overall_p10p90[0]:.1f}-{overall_p10p90[1]:.1f} "
          f"({span:.1f} year spread).")
    if span <= 4:
        verdict = "PASS: invariance claim holds within ±2 years."
    elif span <= 8:
        verdict = "PARTIAL: MAC spread is ±3-4 years; anchor framing needs caveat."
    else:
        verdict = "FAIL: MAC varies more widely than ±4 years; reframe as range."
    print(f"  Verdict: {verdict}")

    # ── 5. Expansion-phase subset (10% <= lsec_both <= 90%) ───────────────
    # This is the subset the paper's panel regressions actually use; the §0
    # reference table in the_long_childhood.tex cites MAC stats from this
    # filter.
    print("\n" + "=" * 70)
    print("5. EXPANSION-PHASE SUBSET (10% <= lsec_both <= 90%)")
    print("=" * 70)
    edu_both = load_wcde_lsec_both()
    merged_both = mac.merge(edu_both, on=["country", "year"], how="inner")
    exp = merged_both[(merged_both["lsec_both"] >= 10) &
                      (merged_both["lsec_both"] <= 90)].copy()
    exp_pct = np.percentile(exp["MAC"].values, [10, 50, 90])
    print(f"  n = {len(exp):,}, countries = {exp['country'].nunique()}")
    print(f"  Mean MAC: {exp['MAC'].mean():.2f}")
    print(f"  Median MAC: {exp['MAC'].median():.2f}")
    print(f"  p10-p90: {exp_pct[0]:.2f} - {exp_pct[2]:.2f}")

    # By lsec_both bin (within expansion phase)
    bins_exp = [10, 25, 40, 55, 70, 90.01]
    labels_exp = ["10-25%", "25-40%", "40-55%", "55-70%", "70-90%"]
    exp["bin"] = pd.cut(exp["lsec_both"], bins=bins_exp, labels=labels_exp,
                        include_lowest=True, right=False)
    exp_bin_stats = []
    print(f"\n  {'Bin':>10}  {'n':>5}  {'ctry':>4}  {'mean':>5}  "
          f"{'TFR':>6}")
    for lab in labels_exp:
        s = exp[exp["bin"] == lab]
        if len(s) == 0:
            continue
        exp_bin_stats.append({
            "bin": lab,
            "n": len(s),
            "n_ctry": int(s["country"].nunique()),
            "mean": round(float(s["MAC"].mean()), 2),
            "p50": round(float(s["MAC"].median()), 2),
            "avg_tfr": round(float(s["TFR"].mean()), 2)
                if "TFR" in s.columns else None,
        })
        tfr_str = (f"{s['TFR'].mean():>6.2f}" if "TFR" in s.columns
                   and not s["TFR"].isna().all() else "    --")
        print(f"  {lab:>10}  {len(s):>5}  {s['country'].nunique():>4}  "
              f"{s['MAC'].mean():>5.2f}  {tfr_str}")

    # Range across bins (the "invariance" key statistic)
    bin_means = [b["mean"] for b in exp_bin_stats]
    exp_bin_range = max(bin_means) - min(bin_means)
    print(f"\n  Range of MAC across lsec bins (10-90% panel): "
          f"{exp_bin_range:.2f} years")

    # ── Write checkin ─────────────────────────────────────────────────────
    write_checkin("mean_age_first_birth_invariance.json", {
        "notes": (
            "UN WPP 2024 Medium variant, Mean Age at Childbearing (MAC) "
            "by country-year, 1950-2020. Joined with WCDE v3 female and "
            "both-sex lower-sec completion. Prior under test: MAC ≈ 25 "
            "invariant across populations and education levels. Paper "
            "§the-generational-lag cites the 'expansion phase' subset stats "
            "(exp_*) which match the panel sample filter."
        ),
        "numbers": {
            "n_country_years": len(mac_all),
            "n_countries": int(mac["country"].nunique()),
            "mac_mean": round(float(mac_all.mean()), 2),
            "mac_sd": round(float(mac_all.std()), 2),
            "mac_min": round(float(mac_all.min()), 2),
            "mac_max": round(float(mac_all.max()), 2),
            "mac_p01": round(float(pct[0]), 2),
            "mac_p05": round(float(pct[1]), 2),
            "mac_p10": round(float(pct[2]), 2),
            "mac_p25": round(float(pct[3]), 2),
            "mac_p50": round(float(pct[4]), 2),
            "mac_p75": round(float(pct[5]), 2),
            "mac_p90": round(float(pct[6]), 2),
            "mac_p95": round(float(pct[7]), 2),
            "mac_p99": round(float(pct[8]), 2),
            "mac_p10_p90_span": round(float(span), 2),
            "by_lsec_bin": bin_stats,
            # Expansion-phase panel (matches paper's sample filter)
            "exp_n": int(len(exp)),
            "exp_n_countries": int(exp["country"].nunique()),
            "exp_mac_mean": round(float(exp["MAC"].mean()), 2),
            "exp_mac_median": round(float(exp["MAC"].median()), 2),
            "exp_mac_p10": round(float(exp_pct[0]), 2),
            "exp_mac_p90": round(float(exp_pct[2]), 2),
            "exp_by_lsec_bin": exp_bin_stats,
            "exp_bin_range": round(float(exp_bin_range), 2),
            "verdict": verdict,
        },
    }, script_path="scripts/mean_age_first_birth_invariance.py")
    print("\nCheckin written.")


if __name__ == "__main__":
    main()
