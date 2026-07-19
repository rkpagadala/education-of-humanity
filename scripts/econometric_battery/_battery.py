"""
_battery.py — shared utilities for the econometric test battery.

Builds the canonical panel used by every test in this directory:
  child_t = α_i + β * parent_{t-28} + ε_it
with optional GDP and outcome (LE, TFR, U5MR) columns.

Conventions:
  - 5-year cohort spacing (1975–2015 child years; 1950–1990 parent years)
  - Country fixed effects via within-transformation
  - Country-clustered SEs by default
  - lower_sec completion (both sexes, age 20-24, WCDE v3) for education
  - WDI constant-2015 USD for GDP (NY.GDP.PCAP.KD)
  - WDI for LE, TFR, U5MR

Each test script imports panel(), demean(), run_fe(), and write_battery_checkin().
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
REPO = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)

from _shared import (  # noqa: E402
    DATA,
    PROC,
    REGIONS,
    load_education,
    load_wb,
    write_checkin,
    interpolate_wide_to_annual,
    LAG_GENERATION,
    LAG_CHILDREARING,
    LAG_TFR,
)

PARENTAL_LAG = LAG_GENERATION
PERIOD = 5
CHILD_YEARS = list(range(1975, 2016, PERIOD))

# Outcome-specific forward lag. TFR is observed at parent_year + LAG_TFR
# (biological parent-of-newborn timing — the cohort aged 20-24 at the
# parent year is at reproductive peak ~5 years later). U5MR is observed at
# parent_year + LAG_CHILDREARING (childrearing window — the cohort's own
# children are at risk of under-5 mortality through ~T+10-15). LE is
# observed at parent_year + LAG_GENERATION (time-to-agency / cross-
# generation step). The panel reports each outcome at its canonical lag.
OUTCOME_LAG = {
    "le":   LAG_GENERATION,
    "tfr":  LAG_TFR,
    "u5mr": LAG_CHILDREARING,
}

# Entry-cohort window: include country-years where child-year
# lower-secondary completion is in [10%, 90%]. Outside this window the
# country is either pre-expansion (no schooling stock to transmit) or
# ceiling-saturated (no variance left to predict). The paper's claim
# is restricted to this active-transition window.
ENTRY_COHORT_LOWER = 10.0
ENTRY_COHORT_UPPER = 90.0

# Legacy alias for code that still asks for the one-sided cutoff used
# by Table 9 in earlier drafts. New code should use entry_cohort().
ACTIVE_EXPANSION_CUTOFF = 30.0

BATTERY_CHECKIN_DIR = os.path.join(REPO, "checkin", "econometric_battery")


def entry_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to country-years inside the entry-cohort window
    (child-year lower_sec in [ENTRY_COHORT_LOWER, ENTRY_COHORT_UPPER])."""
    return df[
        (df["lower_sec"] >= ENTRY_COHORT_LOWER)
        & (df["lower_sec"] <= ENTRY_COHORT_UPPER)
    ].reset_index(drop=True)


def _wb_long(filename: str, value_name: str) -> pd.DataFrame:
    """Reshape a WB wide indicator into long form (country, year, value)."""
    wide = load_wb(filename)
    long = wide.reset_index().melt(
        id_vars=["Country"], var_name="year", value_name=value_name,
    )
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long = long.rename(columns={"Country": "country"})
    long["country"] = long["country"].astype(str).str.lower()
    long = long.dropna(subset=["year", value_name])
    long["year"] = long["year"].astype(int)
    return long[["country", "year", value_name]]


def panel(
    include_outcomes: bool = True,
    include_gdp: bool = True,
    require_balanced: bool = False,
    child_year_min: int = 1975,
    child_year_max: int = 2015,
    entry_cohort_filter: bool = True,
) -> pd.DataFrame:
    """Build the canonical child/parent panel with optional GDP and outcomes.

    Columns returned (when available):
      country, year (=child year, = parent_year + LAG_GENERATION),
      parent_year, lower_sec (=child edu at year), parent_edu, log_gdp,
      le, tfr, u5mr.

    Outcome lags: LE is read at year (= parent_year + LAG_GENERATION);
    TFR is read at parent_year + LAG_TFR (= 5 — the biological
    parent-of-newborn timing); U5MR is read at parent_year +
    LAG_CHILDREARING (= 12 — the childrearing window). The "year" column
    reflects the child / LE observation year; the TFR and U5MR columns
    carry the parent_year + LAG_TFR and parent_year + LAG_CHILDREARING
    observations respectively.

    entry_cohort_filter: when True (default), restrict to country-years
      where child-year lower_sec is in [ENTRY_COHORT_LOWER, ENTRY_COHORT_UPPER].
      Set False only for sample-composition diagnostics or sensitivity checks
      that need the full pooled panel.

    Drops rows missing any requested column.
    """
    edu = load_education()
    edu["country"] = edu["country"].str.lower()
    edu = edu[~edu["country"].isin({r.lower() for r in REGIONS})].copy()

    # Child rows stay on the WCDE 5-year cohort grid.
    df = edu[["country", "year", "lower_sec"]].copy()
    df = df[(df["year"] >= child_year_min) & (df["year"] <= child_year_max)]
    df["parent_year"] = df["year"] - PARENTAL_LAG
    df["tfr_year"]    = df["parent_year"] + OUTCOME_LAG["tfr"]
    df["u5mr_year"]   = df["parent_year"] + OUTCOME_LAG["u5mr"]

    # Parent lookups are at year - LAG_GENERATION (28), off the WCDE 5-year
    # cohort grid; interpolate lower_sec to annual so the generational merge
    # resolves. Same pattern as tables/panel_expansion_fe.py and
    # grandparent_effect.py — only the parent anchor is interpolated; child
    # years stay grid-aligned, so cohort spacing and N are unchanged in kind.
    ls_wide = edu.pivot_table(index="country", columns="year", values="lower_sec")
    ls_wide.columns = ls_wide.columns.astype(int)
    ls_wide = interpolate_wide_to_annual(ls_wide)
    parent = ls_wide.reset_index().melt(
        id_vars="country", var_name="parent_year", value_name="parent_edu")
    parent["parent_year"] = parent["parent_year"].astype(int)
    parent = parent.dropna(subset=["parent_edu"])
    df = df.merge(parent, on=["country", "parent_year"], how="inner")
    df = df.dropna(subset=["lower_sec", "parent_edu"])

    if include_gdp:
        gdp = _wb_long("gdppercapita_us_inflation_adjusted.csv", "gdp")
        gdp["log_gdp"] = np.log(gdp["gdp"].clip(lower=1))
        df = df.merge(gdp[["country", "year", "log_gdp"]], on=["country", "year"], how="left")

    if include_outcomes:
        le = _wb_long("life_expectancy_years.csv", "le")
        tfr = _wb_long("children_per_woman_total_fertility.csv", "tfr")
        u5mr = _wb_long("child_mortality_u5.csv", "u5mr")
        # LE at child year (= parent_year + LAG_GENERATION).
        df = df.merge(le, on=["country", "year"], how="left")
        # U5MR at parent_year + LAG_CHILDREARING. Merge on u5mr_year to read
        # U5MR at its childrearing-window anchor.
        u5mr_keyed = u5mr.rename(columns={"year": "u5mr_year"})
        df = df.merge(u5mr_keyed, on=["country", "u5mr_year"], how="left")
        # TFR at parent_year + LAG_TFR. Merge on tfr_year to read TFR at
        # its biological-parent-timing anchor.
        tfr_keyed = tfr.rename(columns={"year": "tfr_year"})
        df = df.merge(tfr_keyed, on=["country", "tfr_year"], how="left")

    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    if entry_cohort_filter:
        df = entry_cohort(df)

    if require_balanced:
        counts = df.groupby("country").size()
        target = counts.max()
        keep = counts[counts == target].index
        df = df[df["country"].isin(keep)].reset_index(drop=True)

    return df


def demean(df: pd.DataFrame, cols, group: str = "country") -> pd.DataFrame:
    """Subtract within-group mean for each named column. Returns a copy."""
    out = df.copy()
    if isinstance(cols, str):
        cols = [cols]
    for c in cols:
        out[f"{c}_d"] = out[c] - out.groupby(group)[c].transform("mean")
    return out


@dataclass
class FEResult:
    beta: float
    se: float
    t: float
    p: float
    r2_within: float
    n_obs: int
    n_groups: int
    label: str = ""


def run_fe(
    df: pd.DataFrame,
    y: str,
    x: str | list[str],
    cluster: str = "country",
    label: str = "",
) -> FEResult:
    """One-way (country) FE with cluster-robust SE on the primary regressor."""
    cols = [x] if isinstance(x, str) else list(x)
    d = df.dropna(subset=cols + [y, cluster]).copy()
    d = demean(d, cols + [y], group=cluster)
    X = sm.add_constant(d[[f"{c}_d" for c in cols]], has_constant="add")
    m = sm.OLS(d[f"{y}_d"], X).fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})
    primary = f"{cols[0]}_d"
    return FEResult(
        beta=float(m.params[primary]),
        se=float(m.bse[primary]),
        t=float(m.tvalues[primary]),
        p=float(m.pvalues[primary]),
        r2_within=float(m.rsquared),
        n_obs=int(m.nobs),
        n_groups=int(d[cluster].nunique()),
        label=label,
    )


def fe_beta_only(df: pd.DataFrame, y: str, x: str) -> float:
    """Fast within-FE β without cluster SE — used in tight inner loops."""
    d = df.dropna(subset=[y, x]).copy()
    d["y_d"] = d[y] - d.groupby("country")[y].transform("mean")
    d["x_d"] = d[x] - d.groupby("country")[x].transform("mean")
    X = sm.add_constant(d[["x_d"]], has_constant="add")
    return float(sm.OLS(d["y_d"], X).fit().params["x_d"])


def write_battery_checkin(name: str, payload: dict, script_path: str | None = None):
    """Write a JSON output to checkin/econometric_battery/<name>.json."""
    os.makedirs(BATTERY_CHECKIN_DIR, exist_ok=True)
    if script_path:
        # Store a repo-relative path — never leak absolute local paths/usernames.
        payload["script"] = (
            os.path.relpath(script_path, REPO)
            if os.path.isabs(script_path) else script_path
        )
    path = os.path.join(BATTERY_CHECKIN_DIR, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    print(f"  → wrote {path}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    raise TypeError(f"unserializable: {type(o)}")


def banner(title: str, char: str = "="):
    print(char * 72)
    print(title)
    print(char * 72)


def section(title: str):
    print()
    print(f"── {title} " + "─" * max(0, 70 - len(title)))


def pass_fail(condition: bool, label: str = "") -> str:
    tag = "PASS" if condition else "FAIL"
    return f"[{tag}] {label}".rstrip()
