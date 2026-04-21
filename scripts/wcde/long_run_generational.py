"""
long_run_generational.py (was wcde/scripts/04b_long_run_generational.py)
Long-run generational transmission using cohort reconstruction (back to ~1900).

Uses 02b_cohort_reconstruction.py output: education by cohort year (when they
were 20-24), reconstructed from older age groups at historical observation years.

Valid for: countries with reliable historical data and no colonial suppression.
  Japan, USA, UK, Germany, France, Australia, NZ, Canada, Argentina, Chile,
  Cuba, Uruguay, Costa Rica — "self-determination" countries.

Key question: does the T-25 parental multiplier hold over a 100-year horizon?
Does Japan's 1920s education investment explain its 1940s-1960s progress?

Outputs: wcde/output/long_run_generational.md, checkin/long_run_generational.json
"""

import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from _shared import REPO_ROOT, write_checkin

PROC = os.path.join(REPO_ROOT, "wcde", "data", "processed")
OUT  = os.path.join(REPO_ROOT, "wcde", "output")
os.makedirs(OUT, exist_ok=True)

# Countries with reliable pre-1960 cohort data (self-determining, good historical records)
RELIABLE_COUNTRIES = {
    # East Asia (post-Meiji Japan, Taiwan, Korea rapid investment)
    "Japan",
    "Taiwan Province of China",
    "Republic of Korea",
    # Western Europe
    "United Kingdom of Great Britain and Northern Ireland",
    "France",
    "Germany",
    "Sweden",
    "Norway",
    "Denmark",
    "Finland",
    "Netherlands",
    "Belgium",
    "Switzerland",
    "Austria",
    "Italy",
    "Spain",
    "Portugal",
    # New World
    "United States of America",
    "Canada",
    "Australia",
    "New Zealand",
    # Latin America (independent since 1820s, own policy choices)
    "Argentina",
    "Chile",
    "Uruguay",
    "Cuba",
    "Costa Rica",
    # Special colonial cases with active education investment
    "Sri Lanka",  # British Ceylon — active investment, known anomaly
    "Hong Kong Special Administrative Region of China",
}

print("Loading cohort data...")
long = pd.read_csv(os.path.join(PROC, "cohort_completion_both_long.csv"))
print(f"  Total: {len(long)} rows, {long['country'].nunique()} countries, "
      f"cohort years {long['cohort_year'].min()}–{long['cohort_year'].max()}")

# Filter to reliable countries
long_r = long[long["country"].isin(RELIABLE_COUNTRIES)].copy()
print(f"  Reliable countries: {long_r['country'].nunique()}")

# Pivot to wide for easy lookup
low_w = long.pivot(index="country", columns="cohort_year", values="lower_sec")
pri_w = long.pivot(index="country", columns="cohort_year", values="primary")

def v(df_w, country, year):
    try:
        val = float(df_w.loc[country, int(year)])
        return val if not np.isnan(val) else np.nan
    except (KeyError, ValueError, TypeError):
        return np.nan

# ── Long-run panel with T-25 lag ──────────────────────────────────────────────
CHILD_COHORTS_ALL = list(range(1900, 2016, 5))

panel_rows = []
for c in RELIABLE_COUNTRIES:
    if c not in low_w.index:
        continue
    for child_yr in CHILD_COHORTS_ALL:
        parent_yr  = child_yr - 25
        child_low  = v(low_w, c, child_yr)
        parent_low = v(low_w, c, parent_yr)
        child_pri  = v(pri_w, c, child_yr)

        if any(np.isnan(x) for x in [child_low, parent_low]):
            continue

        panel_rows.append({
            "country": c,
            "cohort_year": child_yr,
            "child_low": child_low,
            "parent_low": parent_low,
            "child_pri": child_pri,
            "pre_direct": child_yr < 1960,
        })

panel = pd.DataFrame(panel_rows)
print(f"\nLong-run panel: {len(panel)} obs, {panel['country'].nunique()} countries")
print(f"  Pre-1960 (reconstructed): {panel['pre_direct'].sum()} obs")
print(f"  Post-1960 (direct 20-24): {(~panel['pre_direct']).sum()} obs")

# ── Pooled OLS: full panel ─────────────────────────────────────────────────────
ok = ~panel["child_low"].isna() & ~panel["parent_low"].isna()
X = sm.add_constant(panel.loc[ok, ["parent_low"]])
y = panel.loc[ok, "child_low"]
reg_all = sm.OLS(y, X).fit()
r2_all = reg_all.rsquared
beta_all = reg_all.params.iloc[1]
print(f"\nPooled OLS (full, 1900-2015): β={beta_all:.3f}, R²={r2_all:.3f}")

# FE version
pan_fe = panel[ok].copy()
pan_fe["child_dm"]  = pan_fe["child_low"]  - pan_fe.groupby("country")["child_low"].transform("mean")
pan_fe["parent_dm"] = pan_fe["parent_low"] - pan_fe.groupby("country")["parent_low"].transform("mean")
ok_fe = ~pan_fe["child_dm"].isna() & ~pan_fe["parent_dm"].isna()
reg_fe = sm.OLS(pan_fe.loc[ok_fe, "child_dm"], pan_fe.loc[ok_fe, ["parent_dm"]]).fit()
r2_fe = reg_fe.rsquared
beta_fe = reg_fe.params.iloc[0]
print(f"Country FE (full, 1900-2015): β={beta_fe:.3f}, R²={r2_fe:.3f}")

# ── Key country trajectories ──────────────────────────────────────────────────
KEY = [
    "Japan",
    "Republic of Korea",
    "Taiwan Province of China",
    "United States of America",
    "United Kingdom of Great Britain and Northern Ireland",
    "Germany",
    "France",
    "Sri Lanka",
    "Argentina",
    "Chile",
]
KEY = [c for c in KEY if c in low_w.index]

SNAP_COHORTS = [1875,1890,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2015]

# ── Report ────────────────────────────────────────────────────────────────────
lines = []
def h(t=""): lines.append(t)
def pct(val): return f"{val:.1f}%" if not np.isnan(val) else "n/a"
def cn(name, maxlen=40): return str(name)[:maxlen]

def pipe_table(headers, rows_data, aligns=None):
    if aligns is None:
        aligns = ["left"] + ["right"] * (len(headers) - 1)
    def sep(a): return ":---" if a == "left" else "---:"
    h("| " + " | ".join(headers) + " |")
    h("| " + " | ".join(sep(a) for a in aligns) + " |")
    for r in rows_data:
        h("| " + " | ".join(str(x) for x in r) + " |")
    h()

h("# Long-Run Generational Transmission of Education — WCDE v3 Cohort Reconstruction")
h()
h("*A 100-year view of the intergenerational multiplier using cohort-based education reconstruction.*")
h()
h("## How This Works")
h()
h("WCDE v3 provides education attainment by **age group** at each observation year (1950-2015).")
h("By reading older age groups at historical observation years, we reconstruct what each")
h("birth cohort looked like when they were 20-24:")
h()
h("  cohort_year = obs_year - (midpoint_age - 22)")
h()
h("Examples:")
h("- Age 60-64 at obs_year=1950 -> cohort_year = 1910 (these people were 20-24 in 1910)")
h("- Age 80-84 at obs_year=1950 -> cohort_year = 1890")
h("- Age 95-99 at obs_year=1950 -> cohort_year = 1875")
h()
h("**Best estimate per cohort**: earliest available observation (youngest age at measurement)")
h("minimises survivorship bias.")
h()
h("---")
h()
h("## Regression Results")
h()
h(f"Panel: {len(panel)} obs across {panel['country'].nunique()} countries, cohort years 1900-2015.")
h()
h("| Model | beta (parental) | R² | Notes |")
h("|---|---|---|---|")
h(f"| Pooled OLS (1900-2015) | {beta_all:.3f} | {r2_all:.3f} | Every 1pp parental -> {beta_all:.2f}pp child |")
h(f"| Country FE (1900-2015) | {beta_fe:.3f} | {r2_fe:.3f} | Within-country, controls for all fixed traits |")
h()
h("---")
h()
h("## Table 1 — Lower Secondary Completion by Birth Cohort (Key Countries)")
h()

headers = ["Country"] + [str(c) for c in SNAP_COHORTS]
rows_data = []
for c in KEY:
    row = [cn(c, 28)]
    for cy in SNAP_COHORTS:
        val = v(low_w, c, cy)
        mark = "*" if cy < 1960 else ""
        row.append((pct(val) + mark) if not np.isnan(val) else "n/a")
    rows_data.append(row)
pipe_table(headers, rows_data, ["left"] + ["right"]*len(SNAP_COHORTS))

h("*\\* = reconstructed from older age group at 1950 observation year.*")
h()
h("---")
h()
h("## Table 2 — Primary Completion by Birth Cohort (Key Countries)")
h()
headers = ["Country"] + [str(c) for c in SNAP_COHORTS]
rows_data = []
for c in KEY:
    row = [cn(c, 28)]
    for cy in SNAP_COHORTS:
        val = v(pri_w, c, cy)
        mark = "*" if cy < 1960 else ""
        row.append((pct(val) + mark) if not np.isnan(val) else "n/a")
    rows_data.append(row)
pipe_table(headers, rows_data, ["left"] + ["right"]*len(SNAP_COHORTS))
h()
h("---")
h()
h("## Table 3 — The Generational Chain (T-25 Parent-Child Pairs for Key Countries)")
h()
for c in KEY[:6]:
    h(f"**{c}:**")
    h()
    chain_rows = []
    for child_yr in [1925,1940,1950,1960,1970,1980,1990,2000,2015]:
        parent_yr = child_yr - 25
        child_low  = v(low_w, c, child_yr)
        parent_low = v(low_w, c, parent_yr)
        if np.isnan(child_low) or np.isnan(parent_low): continue
        gain = child_low - parent_low
        mark_c = "*" if child_yr < 1960 else ""
        mark_p = "*" if parent_yr < 1960 else ""
        chain_rows.append([
            f"{child_yr}{mark_c}",
            f"{parent_yr}{mark_p}",
            pct(parent_low) + mark_p,
            pct(child_low) + mark_c,
            f"+{gain:.1f} pp" if gain >= 0 else f"{gain:.1f} pp"
        ])
    if chain_rows:
        pipe_table(["Child Cohort","Parent Cohort","Parent Low Sec","Child Low Sec","Gain"],
                   chain_rows,
                   ["right","right","right","right","right"])

h("*\\* = reconstructed estimate.*")
h()
h("---")
h()
h("*Method: cohort reconstruction from WCDE v3 age-period data.*")

OUT_MD = os.path.join(OUT, "long_run_generational.md")
with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved: {OUT_MD}")

# ── Write checkin JSON ───────────────────────────────────────────────────────
write_checkin("long_run_generational.json", {
    "numbers": {
        "LR-countries": panel["country"].nunique(),
        "LR-obs": len(panel),
        "LR-pooled-beta": round(float(beta_all), 3),
        "LR-pooled-R2": round(r2_all, 3),
        "LR-FE-beta": round(float(beta_fe), 3),
        "LR-FE-R2": round(r2_fe, 3),
    },
}, script_path="scripts/wcde/long_run_generational.py")
print("Done.")
