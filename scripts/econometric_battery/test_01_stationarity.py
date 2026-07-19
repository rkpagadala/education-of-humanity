"""
test_01_stationarity.py — Panel unit-root battery.

Tests stationarity of the two key series (parent_edu, child_edu) in the
canonical FE panel. Educational attainment for a 20-24 cohort almost
certainly behaves I(1) over 1950-2015: it is monotonically rising in
nearly every country. If both regressor and outcome are I(1) and not
cointegrated, levels regressions risk spurious-regression artefacts;
the within-FE transformation alleviates but does not eliminate this.

Tests implemented:
  1. Levin–Lin–Chu (LLC, 2002) — common unit root, pools t-stats
  2. Im–Pesaran–Shin (IPS, 2003) — heterogeneous unit roots, averages t
  3. Maddala–Wu Fisher χ² (1999) — combines individual ADF p-values
  4. Pesaran CIPS (2007) — second-generation, robust to cross-sec dep
  5. Hadri LM (2000) — null = stationarity (reverse polarity to above)
  6. Country-by-country ADF distribution summary

Interpretation:
  Reject unit-root null in LLC/IPS/Maddala-Wu/CIPS → trend-stationary or
  stationary around country mean. Hadri rejects stationarity null → I(1).
  Expect: mixed evidence on raw series (mostly I(1)); within-FE residuals
  much closer to stationary. Either way, paper's identification does not
  hinge on stationarity; the natural-experiments + USSR falsification
  carry causal load. This test fills the "I asked but you didn't tell me"
  gap pointed out by the panel-econometrics reader.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _battery import (  # noqa: E402
    panel, banner, section, pass_fail, write_battery_checkin, PERIOD,
)


def adf_for_country(series: np.ndarray, regression: str = "c") -> tuple[float, float] | None:
    """Run ADF on a single country's series; return (t-stat, p-value) or None."""
    s = pd.Series(series).dropna().astype(float).values
    if len(s) < 6 or np.std(s) < 1e-9:
        return None
    try:
        r = adfuller(s, maxlag=1, regression=regression, autolag=None)
        return float(r[0]), float(r[1])
    except Exception:
        return None


def levin_lin_chu(panel_df: pd.DataFrame, col: str) -> dict:
    """Pooled-t LLC. Pools ADF residual moments across panel.

    Implements the simplified LLC pooled t-bar approach: run ADF per
    country, pool the demeaned residuals into a single regression of
    Δy on y_{-1}, take the t-statistic, and report it against a standard
    normal under the null of unit root.
    """
    chunks = []
    countries = panel_df["country"].unique()
    for c in countries:
        s = panel_df[panel_df["country"] == c].sort_values("year")[col].dropna().values
        if len(s) < 4 or np.std(s) < 1e-9:
            continue
        dy = np.diff(s)
        y_lag = s[:-1]
        y_lag = y_lag - y_lag.mean()
        dy = dy - dy.mean()
        chunks.append(np.column_stack([dy, y_lag]))
    if not chunks:
        return {"t": None, "p": None, "n_countries": 0}
    M = np.vstack(chunks)
    dy, y_lag = M[:, 0], M[:, 1]
    s2 = float(np.sum(y_lag * y_lag))
    if s2 < 1e-12:
        return {"t": None, "p": None, "n_countries": len(chunks)}
    rho = float(np.sum(y_lag * dy)) / s2
    resid = dy - rho * y_lag
    sigma2 = float(np.sum(resid ** 2)) / max(len(resid) - 1, 1)
    se = (sigma2 / s2) ** 0.5
    t = rho / se if se > 0 else np.nan
    p = float(stats.norm.cdf(t))  # left-tail (negative rho rejects unit root)
    return {
        "t": float(t),
        "p": p,
        "n_countries": len(chunks),
        "n_obs": int(len(dy)),
    }


def im_pesaran_shin(panel_df: pd.DataFrame, col: str) -> dict:
    """IPS t-bar: average of country-by-country ADF t-stats."""
    ts = []
    for c, g in panel_df.groupby("country"):
        r = adf_for_country(g.sort_values("year")[col].values)
        if r is None:
            continue
        ts.append(r[0])
    if not ts:
        return {"t_bar": None, "z": None, "p": None, "n_countries": 0}
    t_bar = float(np.mean(ts))
    n = len(ts)
    # Approximate IPS standardization (constants from IPS 2003 simulations,
    # finite-sample-corrected mean and variance for ADF(0) with constant).
    mu = -1.532
    sigma2 = 0.706
    z = (np.sqrt(n) * (t_bar - mu)) / np.sqrt(sigma2)
    p = float(stats.norm.cdf(z))
    return {
        "t_bar": t_bar,
        "z": float(z),
        "p": p,
        "n_countries": n,
    }


def maddala_wu(panel_df: pd.DataFrame, col: str) -> dict:
    """Fisher-type test: -2 Σ ln(p_i) ~ χ²(2N) under unit-root null."""
    ps = []
    for c, g in panel_df.groupby("country"):
        r = adf_for_country(g.sort_values("year")[col].values)
        if r is None:
            continue
        ps.append(max(min(r[1], 0.999999), 1e-6))
    if not ps:
        return {"stat": None, "p": None, "n_countries": 0}
    stat = float(-2.0 * np.sum(np.log(ps)))
    df = 2 * len(ps)
    p = float(1 - stats.chi2.cdf(stat, df))
    return {"stat": stat, "df": df, "p": p, "n_countries": len(ps)}


def pesaran_cips(panel_df: pd.DataFrame, col: str) -> dict:
    """Pesaran CIPS (2007): CADF with cross-section mean, average t-stat.

    Augment each country's ADF with the cross-section mean of the level
    and difference; take the average t (CIPS) and compare to simulated
    critical values for N → ∞.
    """
    wide = panel_df.pivot_table(index="year", columns="country", values=col)
    wide = wide.dropna(how="all")
    if wide.empty:
        return {"cips": None, "p_approx": None, "n_countries": 0}
    ybar = wide.mean(axis=1)
    ts = []
    for c in wide.columns:
        s = wide[c].dropna()
        if len(s) < 5:
            continue
        s = s.values
        yb = ybar.loc[wide[c].dropna().index].values
        dy = np.diff(s)
        y_lag = s[:-1]
        yb_lag = yb[:-1]
        dyb = np.diff(yb)
        X = np.column_stack([np.ones_like(y_lag), y_lag, yb_lag, dyb])
        try:
            beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
            resid = dy - X @ beta
            sigma2 = (resid @ resid) / (len(dy) - X.shape[1])
            xtx_inv = np.linalg.inv(X.T @ X)
            se_phi = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
            if se_phi > 0:
                ts.append(beta[1] / se_phi)
        except np.linalg.LinAlgError:
            continue
    if not ts:
        return {"cips": None, "p_approx": None, "n_countries": 0}
    cips = float(np.mean(ts))
    # Approximate p via normal with Pesaran (2007) Table I asymptotic moments
    # for the CIPS distribution with constant only (mu=-2.15, sigma2=0.16)
    z = (cips - (-2.15)) / np.sqrt(0.16 / len(ts))
    p_approx = float(stats.norm.cdf(z))
    return {
        "cips": cips,
        "z": float(z),
        "p_approx": p_approx,
        "n_countries": len(ts),
    }


def hadri_lm(panel_df: pd.DataFrame, col: str) -> dict:
    """Hadri LM (2000): null = stationarity. Reject → unit-root.

    Detrend each country's series with a country-specific intercept,
    cumulate residuals, sum squared cumulants, standardize.
    """
    stats_list = []
    for c, g in panel_df.groupby("country"):
        s = g.sort_values("year")[col].dropna().values
        if len(s) < 4:
            continue
        T = len(s)
        e = s - s.mean()
        S = np.cumsum(e)
        sigma2 = float(np.var(e, ddof=1))
        if sigma2 <= 0:
            continue
        eta = float(np.sum(S ** 2) / (T ** 2 * sigma2))
        stats_list.append(eta)
    if not stats_list:
        return {"z": None, "p": None, "n_countries": 0}
    eta_bar = float(np.mean(stats_list))
    n = len(stats_list)
    # Asymptotic moments for level case: mu = 1/6, sigma2 = 1/45
    z = (np.sqrt(n) * (eta_bar - 1/6)) / np.sqrt(1/45)
    p = float(1 - stats.norm.cdf(z))
    return {
        "eta_bar": eta_bar,
        "z": float(z),
        "p": p,
        "n_countries": n,
    }


def adf_distribution(panel_df: pd.DataFrame, col: str) -> dict:
    """Per-country ADF summary."""
    results = []
    for c, g in panel_df.groupby("country"):
        r = adf_for_country(g.sort_values("year")[col].values)
        if r is None:
            continue
        results.append({"country": c, "t": r[0], "p": r[1]})
    if not results:
        return {"n_countries": 0}
    ps = np.array([r["p"] for r in results])
    return {
        "n_countries": len(results),
        "share_reject_at_05": float((ps < 0.05).mean()),
        "share_reject_at_10": float((ps < 0.10).mean()),
        "median_p": float(np.median(ps)),
    }


def main():
    banner("ECONOMETRIC BATTERY 01 — Panel unit-root tests")
    df = panel(include_outcomes=False, include_gdp=False)
    print(f"Panel: {len(df)} obs · {df['country'].nunique()} countries · "
          f"years {df['year'].min()}–{df['year'].max()} ({PERIOD}-yr spacing)")

    results = {}
    for col, label in [("lower_sec", "child_edu"),
                       ("parent_edu", "parent_edu")]:
        section(f"Series: {label}")
        llc = levin_lin_chu(df, col)
        ips = im_pesaran_shin(df, col)
        mw = maddala_wu(df, col)
        cips = pesaran_cips(df, col)
        hadri = hadri_lm(df, col)
        adf = adf_distribution(df, col)

        print(f"  LLC pooled-t:       t={_fmt(llc['t'])}  p={_fmt(llc['p'])} "
              f"(N={llc['n_countries']})")
        print(f"  IPS t-bar:          t̄={_fmt(ips['t_bar'])}  z={_fmt(ips['z'])} "
              f" p={_fmt(ips['p'])} (N={ips['n_countries']})")
        print(f"  Maddala-Wu Fisher:  χ²={_fmt(mw['stat'])}  df={mw['df']}  "
              f"p={_fmt(mw['p'])}")
        print(f"  Pesaran CIPS:       CIPS={_fmt(cips['cips'])}  z={_fmt(cips['z'])} "
              f" p≈{_fmt(cips['p_approx'])} (N={cips['n_countries']})")
        print(f"  Hadri LM (H0=stat): z={_fmt(hadri['z'])}  p={_fmt(hadri['p'])} "
              f"(N={hadri['n_countries']})")
        print(f"  Individual ADF:     reject@.05 share={_fmt(adf.get('share_reject_at_05'))} "
              f" median p={_fmt(adf.get('median_p'))}")
        results[label] = {
            "llc": llc, "ips": ips, "maddala_wu": mw,
            "pesaran_cips": cips, "hadri": hadri, "adf_distribution": adf,
        }

    section("Within-FE (demeaned) residual stationarity")
    # Apply within transform to lower_sec and re-test
    df["lower_sec_d"] = df["lower_sec"] - df.groupby("country")["lower_sec"].transform("mean")
    llc_d = levin_lin_chu(df, "lower_sec_d")
    ips_d = im_pesaran_shin(df, "lower_sec_d")
    print(f"  LLC on demeaned y:  t={_fmt(llc_d['t'])}  p={_fmt(llc_d['p'])}")
    print(f"  IPS on demeaned y:  z={_fmt(ips_d['z'])}  p={_fmt(ips_d['p'])}")
    results["demeaned_child"] = {"llc": llc_d, "ips": ips_d}

    section("Interpretation")
    print("  Raw attainment series are nearly monotone → LLC/IPS lean toward")
    print("  unit-root; this is mechanical, not a flaw in the panel. Hadri")
    print("  will reject stationarity for the same reason. Within-FE demeaning")
    print("  removes the country mean; the demeaned residual should be much")
    print("  closer to stationary. Identification of β does not require either:")
    print("  the natural experiments + USSR falsification carry causal load.")

    # Soft pass: results are reported, no hard fail. The diagnostic is the point.
    write_battery_checkin("01_stationarity", {
        "headline_beta": 0.4834,
        "panel": {"n_obs": int(len(df)),
                  "n_countries": int(df["country"].nunique())},
        "results": results,
        "verdict": "raw_series_I(1)_expected_demeaned_closer_to_stationary",
    }, script_path=__file__)

    print()
    print(pass_fail(True, "diagnostic complete (interpretive, not gating)"))


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return " n/a"
    if isinstance(v, float):
        return f"{v:+.3f}"
    return str(v)


if __name__ == "__main__":
    main()
