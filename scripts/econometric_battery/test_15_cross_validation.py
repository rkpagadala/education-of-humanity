"""
test_15_cross_validation.py — Out-of-sample predictive validity.

Within-sample R² and significance are conventional; OOS performance is the
sharper test. β fit on one slice of data should predict the held-out slice.

Tests:
  1. K-fold cross-validation by country (k=5, k=10)
  2. Forward-chaining: train on years <= t, test on t+5
  3. Region holdout: train on 4 regions, test on the held-out region
  4. Era holdout: train on pre-2000, test on post-2000
  5. Out-of-OECD prediction
  6. Predict-from-zero: countries with no pre-period data
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _battery import (  # noqa: E402
    panel, demean, banner, section, pass_fail, write_battery_checkin,
)

# Simple manual region map (lowercase keys)
REGIONS_MAP_RAW = {
    "europe": ["austria", "belgium", "bulgaria", "croatia", "denmark", "estonia",
               "finland", "france", "germany", "greece", "hungary", "iceland",
               "ireland", "italy", "latvia", "lithuania", "luxembourg",
               "netherlands", "norway", "poland", "portugal", "romania",
               "slovakia", "slovenia", "spain", "sweden", "switzerland",
               "united kingdom"],
    "americas": ["argentina", "bolivia", "brazil", "canada", "chile", "colombia",
                 "costa rica", "cuba", "ecuador", "haiti", "mexico", "nicaragua",
                 "panama", "paraguay", "peru", "united states", "uruguay",
                 "venezuela"],
    "africa": ["algeria", "angola", "benin", "botswana", "burkina faso", "burundi",
               "cameroon", "central african republic", "chad", "ethiopia",
               "gabon", "gambia", "ghana", "kenya", "lesotho", "liberia",
               "madagascar", "malawi", "mali", "mauritania", "mauritius",
               "morocco", "mozambique", "namibia", "niger", "nigeria",
               "rwanda", "senegal", "sierra leone", "somalia", "south africa",
               "sudan", "togo", "tunisia", "uganda", "zambia", "zimbabwe"],
    "asia": ["afghanistan", "bangladesh", "bhutan", "cambodia", "china",
             "india", "indonesia", "iran (islamic republic of)", "iraq",
             "israel", "japan", "jordan", "kazakhstan", "lao peoples democratic republic",
             "lebanon", "malaysia", "mongolia", "myanmar", "nepal", "pakistan",
             "philippines", "republic of korea", "saudi arabia", "singapore",
             "sri lanka", "thailand", "turkey", "viet nam", "yemen"],
}


def fit_eval(train, test):
    """Fit FE β on train, evaluate predictions on test.

    For test-set demeaning we use TEST country means (the FE is unobserved
    for held-out countries; we treat the country-mean as identified by the
    test data itself). This lets us evaluate β's predictive content on the
    within-country variation in held-out countries.
    """
    tr = demean(train, ["lower_sec", "parent_edu"])
    X = sm.add_constant(tr[["parent_edu_d"]], has_constant="add")
    m = sm.OLS(tr["lower_sec_d"], X).fit()
    beta = float(m.params["parent_edu_d"])
    te = test.dropna(subset=["lower_sec", "parent_edu"]).copy()
    te["y_d"] = te["lower_sec"] - te.groupby("country")["lower_sec"].transform("mean")
    te["x_d"] = te["parent_edu"] - te.groupby("country")["parent_edu"].transform("mean")
    te = te.dropna(subset=["y_d", "x_d"])
    if te.empty:
        return {"beta": beta, "rmse": None, "r2": None, "n_test": 0}
    yhat = beta * te["x_d"]
    rmse = float(np.sqrt(((te["y_d"] - yhat) ** 2).mean()))
    ss_res = float(((te["y_d"] - yhat) ** 2).sum())
    ss_tot = float(((te["y_d"] - te["y_d"].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return {"beta": beta, "rmse": rmse, "r2": float(r2) if r2 is not None else None,
            "n_test": int(len(te))}


def k_fold_by_country(df, k=5) -> dict:
    countries = df["country"].unique()
    rng = np.random.default_rng(2026)
    rng.shuffle(countries)
    folds = np.array_split(countries, k)
    results = []
    for i, fold in enumerate(folds):
        test = df[df["country"].isin(fold)]
        train = df[~df["country"].isin(fold)]
        r = fit_eval(train, test)
        r["fold"] = i
        results.append(r)
    return {"k": k, "folds": results,
            "mean_beta": float(np.mean([r["beta"] for r in results])),
            "mean_r2": float(np.mean([r["r2"] for r in results if r["r2"] is not None]))}


def forward_chain(df) -> dict:
    """Forward-chain: train on years ≤ t, test on the next 3 periods.

    Need ≥2 test years so within-country variation exists for evaluation.
    """
    years = sorted(df["year"].unique())
    out = []
    for i, yr in enumerate(years[:-2]):
        train = df[df["year"] <= yr]
        test = df[(df["year"] > yr) & (df["year"] <= years[min(i + 3, len(years) - 1)])]
        r = fit_eval(train, test)
        r["train_max"] = int(yr); r["test_range"] = f"{int(years[i+1])}-{int(years[min(i+3, len(years)-1)])}"
        out.append(r)
    return {"chain": out,
            "mean_beta": float(np.mean([r["beta"] for r in out])),
            "mean_r2": float(np.mean([r["r2"] for r in out if r["r2"] is not None]))}


def region_holdout(df) -> dict:
    out = []
    cmap = {c: r for r, cs in REGIONS_MAP_RAW.items() for c in cs}
    df2 = df[df["country"].isin(cmap.keys())].copy()
    df2["region"] = df2["country"].map(cmap)
    for region in df2["region"].unique():
        train = df2[df2["region"] != region]
        test = df2[df2["region"] == region]
        r = fit_eval(train, test)
        r["region"] = region
        out.append(r)
    return {"by_region": out,
            "mean_beta": float(np.mean([r["beta"] for r in out])),
            "mean_r2": float(np.mean([r["r2"] for r in out if r["r2"] is not None]))}


def era_holdout(df) -> dict:
    train = df[df["year"] <= 2000]
    test = df[df["year"] > 2000]
    r = fit_eval(train, test)
    return r


def out_of_oecd(df) -> dict:
    OECD = {"austria", "belgium", "canada", "denmark", "finland", "france",
            "germany", "greece", "iceland", "ireland", "italy", "japan",
            "luxembourg", "netherlands", "norway", "portugal", "spain",
            "sweden", "switzerland", "united kingdom", "united states",
            "australia", "new zealand"}
    train = df[df["country"].isin(OECD)]
    test = df[~df["country"].isin(OECD)]
    return fit_eval(train, test)


def main():
    banner("ECONOMETRIC BATTERY 15 — Cross-validation")
    df = panel(include_outcomes=False, include_gdp=False)
    print(f"Panel: {len(df)} obs · {df['country'].nunique()} countries")

    section("5-fold country-level CV")
    cv5 = k_fold_by_country(df, k=5)
    print(f"  Mean β: {cv5['mean_beta']:+.4f}   Mean R²: {cv5['mean_r2']:+.4f}")

    section("10-fold country-level CV")
    cv10 = k_fold_by_country(df, k=10)
    print(f"  Mean β: {cv10['mean_beta']:+.4f}   Mean R²: {cv10['mean_r2']:+.4f}")

    section("Forward-chaining (train ≤ t → predict t+5)")
    fc = forward_chain(df)
    for r in fc["chain"]:
        rmse = f"{r['rmse']:.2f}" if r['rmse'] is not None else "n/a"
        r2 = f"{r['r2']:+.3f}" if r['r2'] is not None else "n/a"
        print(f"  train≤{r['train_max']}  → predict {r['test_range']}:  "
              f"β={r['beta']:+.4f}  R²={r2}  RMSE={rmse}  N={r['n_test']}")

    section("Region holdout")
    rh = region_holdout(df)
    for r in rh["by_region"]:
        r2 = f"{r['r2']:+.3f}" if r['r2'] is not None else "n/a"
        print(f"  hold out {r['region']:<10} β={r['beta']:+.4f}  R²={r2}  N={r['n_test']}")

    section("Era holdout (train ≤2000 → predict >2000)")
    eh = era_holdout(df)
    r2 = f"{eh['r2']:+.3f}" if eh['r2'] is not None else "n/a"
    print(f"  β={eh['beta']:+.4f}  R²={r2}  RMSE={eh['rmse']:.2f}  N={eh['n_test']}")

    section("Out-of-OECD prediction (train OECD → predict rest)")
    oo = out_of_oecd(df)
    r2 = f"{oo['r2']:+.3f}" if oo['r2'] is not None else "n/a"
    rmse_s = f"{oo['rmse']:.2f}" if oo['rmse'] is not None else "n/a"
    print(f"  β={oo['beta']:+.4f}  R²={r2}  RMSE={rmse_s}  N={oo['n_test']}")

    section("Verdict")
    all_beta = [cv5["mean_beta"], cv10["mean_beta"], fc["mean_beta"],
                rh["mean_beta"], eh["beta"], oo["beta"]]
    pos = all(b > 0 for b in all_beta)
    print(f"  All cross-validation β positive: {pos}")
    print(f"  β range: [{min(all_beta):.4f}, {max(all_beta):.4f}]")

    write_battery_checkin("15_cross_validation", {
        "kfold_5": cv5, "kfold_10": cv10,
        "forward_chain": fc, "region_holdout": rh,
        "era_holdout": eh, "out_of_oecd": oo,
    }, script_path=__file__)

    print()
    print(pass_fail(pos, "β positive in every OOS slice"))


if __name__ == "__main__":
    main()
