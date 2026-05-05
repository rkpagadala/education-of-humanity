"""
permutation_null.py

Permutation null distribution for the headline parental → child education
regression (Table 1, column 1: child lower-sec at T on parental lower-sec
at T-25, country fixed effects).

The test breaks the country-level parent-child match while preserving panel
structure, then refits and records the resulting β. Two schemes:

  1. Within-year shuffle: permute parental values among countries within
     the same year. Preserves the global temporal distribution of parental
     completion; breaks the country-level match. The conservative null —
     it asks whether the panel's common temporal trends could produce 0.483
     by chance.

  2. Full shuffle: permute parental values across all country-years. The
     aggressive null — it asks whether ANY systematic link between parent
     and child could produce 0.483 by chance.

A real relationship sits many standard deviations above both nulls. A
spurious one (driven by common trends, panel autocorrelation, or generic
serial-correlation patterns) overlaps the null distribution.

Sample matches Table~1 column 1: panel restricted to outcome years
1975-2015, after the lag merge drops country-years with missing
parent or child observations.

Output: checkin/permutation_null.json with the real β, null mean,
null SD, null range, z-score, and one-sided p-value for each scheme.
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from _shared import load_education, write_checkin

N_ITER = 200
SEED = 2026
LAG = 25
YEAR_MIN = 1975
YEAR_MAX = 2015


def fit_beta(df, parent_col="parent_edu"):
    """Within-country β of child lower-sec on parental lower-sec."""
    df = df.copy()
    df["child_d"] = df.groupby("country")["lower_sec"].transform(
        lambda x: x - x.mean()
    )
    df["parent_d"] = df.groupby("country")[parent_col].transform(
        lambda x: x - x.mean()
    )
    X = sm.add_constant(df[["parent_d"]], has_constant="add")
    return sm.OLS(df["child_d"], X).fit().params["parent_d"]


def summarize(perm_betas, real_beta, label):
    arr = np.asarray(perm_betas)
    null_mean = float(arr.mean())
    null_sd = float(arr.std())
    z = float((real_beta - null_mean) / null_sd) if null_sd > 0 else float("inf")
    p = float((arr >= real_beta).mean())
    print(f"\n{label}")
    print(f"  Null mean β: {null_mean:.6f}")
    print(f"  Null SD β:   {null_sd:.6f}")
    print(f"  Null range:  [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"  Real β:      {real_beta:.4f}")
    print(f"  Z-score:     {z:.1f} SDs")
    print(f"  p-value:     {p:.4f} ({int((arr >= real_beta).sum())}/{len(arr)})")
    return {
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_min": float(arr.min()),
        "null_max": float(arr.max()),
        "z": z,
        "p": p,
        "n_iter": int(len(arr)),
    }


def main():
    np.random.seed(SEED)

    edu = load_education()
    edu["country"] = edu["country"].str.lower()

    df = edu[["country", "year", "lower_sec"]].copy()
    df["parent_year"] = df["year"] - LAG
    parent = edu[["country", "year", "lower_sec"]].rename(
        columns={"year": "parent_year", "lower_sec": "parent_edu"}
    )
    df = df.merge(parent, on=["country", "parent_year"])
    df = df.dropna(subset=["lower_sec", "parent_edu"])
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]

    n_obs = len(df)
    n_countries = df["country"].nunique()
    print(f"Panel: {n_obs} obs, {n_countries} countries, "
          f"years {YEAR_MIN}-{YEAR_MAX}, lag {LAG}")

    real_beta = float(fit_beta(df))
    print(f"Real β = {real_beta:.4f}")

    # Permutation 1: within-year shuffle
    perm_within = []
    for _ in range(N_ITER):
        df_p = df.copy()
        df_p["parent_perm"] = df_p.groupby("year")["parent_edu"].transform(
            lambda x: np.random.permutation(x.values)
        )
        perm_within.append(fit_beta(df_p, parent_col="parent_perm"))
    within_summary = summarize(
        perm_within, real_beta,
        "PERMUTATION 1 — within-year shuffle (preserves global trend)",
    )

    # Permutation 2: full shuffle
    perm_full = []
    for _ in range(N_ITER):
        df_p = df.copy()
        df_p["parent_perm"] = np.random.permutation(df_p["parent_edu"].values)
        perm_full.append(fit_beta(df_p, parent_col="parent_perm"))
    full_summary = summarize(
        perm_full, real_beta,
        "PERMUTATION 2 — full shuffle (aggressive null)",
    )

    write_checkin("permutation_null.json", {
        "n_obs": n_obs,
        "n_countries": n_countries,
        "year_min": YEAR_MIN,
        "year_max": YEAR_MAX,
        "lag": LAG,
        "n_iter": N_ITER,
        "seed": SEED,
        "real_beta": real_beta,
        "within_year": within_summary,
        "full": full_summary,
    }, script_path="scripts/robustness/permutation_null.py")


if __name__ == "__main__":
    main()
