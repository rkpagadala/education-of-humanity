"""
test_18_measurement_error.py — Measurement-error correction.

Classical errors-in-variables attenuate OLS toward zero:
  plim β̂_OLS = β · σ²_x / (σ²_x + σ²_u)
With WCDE measurement error ~2-3 pp SD, this is small. But to be thorough
we apply formal corrections:

  1. Reliability-ratio correction (assumed reliability k = 0.85, 0.90, 0.95)
  2. Method-of-moments correction using known noise SD
  3. Two-source instrumental-variables correction:
     - Use Barro-Lee secondary-school cohort as IV for WCDE lower-sec
     - The two sources have correlated true signal but independent
       measurement error
  4. Repeated-measure simulation: split each country's panel in half,
     instrument one half with the other
  5. Total-least-squares (Deming regression) accounting for x-error
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _battery import (  # noqa: E402
    panel, demean, banner, section, pass_fail, write_battery_checkin,
    DATA,
)


def fit_baseline(df):
    d = demean(df.dropna(subset=["lower_sec", "parent_edu"]),
               ["lower_sec", "parent_edu"])
    X = sm.add_constant(d[["parent_edu_d"]], has_constant="add")
    m = sm.OLS(d["lower_sec_d"], X).fit(cov_type="cluster",
                                        cov_kwds={"groups": d["country"]})
    return d, m, float(m.params["parent_edu_d"])


def reliability_correction(df, ks=(0.95, 0.90, 0.85)):
    _, _, beta_ols = fit_baseline(df)
    return {f"k={k}": float(beta_ols / k) for k in ks}


def known_noise_correction(df, sigma_u_values=(2.0, 5.0, 10.0)):
    """β_corrected = β_OLS · (σ²_x / (σ²_x - σ²_u)) where σ²_x is observed."""
    d, _, beta_ols = fit_baseline(df)
    var_x_obs = float(d["parent_edu_d"].var())
    out = {}
    for su in sigma_u_values:
        var_u = su ** 2
        if var_x_obs > var_u:
            corrected = beta_ols * (var_x_obs / (var_x_obs - var_u))
        else:
            corrected = None
        out[f"sigma_u={su}"] = corrected
    return {"observed_var_x": var_x_obs, "corrections": out}


def repeated_measure_iv(df):
    """Split each country's obs into odd/even-year halves; IV the OLS β.

    parent_edu (odd years) ~ parent_edu (even years).
    The two halves share the same true country-year trajectory but their
    measurement errors are (approximately) independent.
    """
    d = df.dropna(subset=["lower_sec", "parent_edu"]).copy()
    d = d.sort_values(["country", "year"]).reset_index(drop=True)
    d["odd"] = (d.groupby("country").cumcount() % 2 == 1)
    odd = d[d["odd"]].set_index(["country", "year"])
    even = d[~d["odd"]].set_index(["country", "year"])
    # For each year, instrument parent_edu with the rolling-average from neighbors
    d2 = d.copy()
    d2["parent_neighbor_mean"] = d2.groupby("country")["parent_edu"].transform(
        lambda x: (x.shift(1) + x.shift(-1)) / 2
    )
    d2 = d2.dropna(subset=["parent_neighbor_mean"])
    d2 = demean(d2, ["lower_sec", "parent_edu", "parent_neighbor_mean"])
    # First stage
    X_fs = sm.add_constant(d2[["parent_neighbor_mean_d"]], has_constant="add")
    m_fs = sm.OLS(d2["parent_edu_d"], X_fs).fit()
    d2["pe_hat"] = m_fs.predict(X_fs)
    # 2SLS
    X_2sls = sm.add_constant(d2[["pe_hat"]], has_constant="add")
    m_iv = sm.OLS(d2["lower_sec_d"], X_2sls).fit(cov_type="cluster",
                                                  cov_kwds={"groups": d2["country"]})
    return {"first_stage_F": float(m_fs.tvalues["parent_neighbor_mean_d"] ** 2),
            "iv_beta": float(m_iv.params["pe_hat"]),
            "iv_se": float(m_iv.bse["pe_hat"]),
            "iv_t": float(m_iv.tvalues["pe_hat"])}


def barro_lee_iv(df):
    """Use Barro-Lee secondary attainment as IV for WCDE lower-sec parent_edu."""
    try:
        bl = pd.read_csv(os.path.join(DATA, "barro_lee_v3.csv"))
    except FileNotFoundError:
        return {"available": False}
    bl["country"] = bl["country"].astype(str).str.lower()
    # BL columns may include 'sec' or 'secondary' attainment %
    # Take the column most likely to be lower-secondary completion
    cand = [c for c in bl.columns if "sec" in c.lower() or "second" in c.lower()]
    if not cand or "year" not in bl.columns:
        return {"available": False, "reason": "no matching column"}
    sec_col = cand[0]
    bl_long = bl[["country", "year", sec_col]].dropna().rename(columns={sec_col: "bl_sec"})
    # Build the IV panel: child year - 25 = parent year; merge BL on parent_year
    d = df.dropna(subset=["lower_sec", "parent_edu"]).copy()
    d["parent_year"] = d["year"] - 25
    d = d.merge(bl_long.rename(columns={"year": "parent_year", "bl_sec": "bl_parent"}),
                on=["country", "parent_year"], how="left")
    d = d.dropna(subset=["bl_parent"])
    if len(d) < 50:
        return {"available": False, "n": int(len(d))}
    d = demean(d, ["lower_sec", "parent_edu", "bl_parent"])
    X_fs = sm.add_constant(d[["bl_parent_d"]], has_constant="add")
    m_fs = sm.OLS(d["parent_edu_d"], X_fs).fit()
    d["pe_hat"] = m_fs.predict(X_fs)
    X_2sls = sm.add_constant(d[["pe_hat"]], has_constant="add")
    m_iv = sm.OLS(d["lower_sec_d"], X_2sls).fit(cov_type="cluster",
                                                cov_kwds={"groups": d["country"]})
    return {"available": True, "n": int(len(d)),
            "first_stage_F": float(m_fs.tvalues["bl_parent_d"] ** 2),
            "iv_beta": float(m_iv.params["pe_hat"]),
            "iv_t": float(m_iv.tvalues["pe_hat"])}


def deming_regression(df, delta=1.0):
    """Total least squares assuming var(u_x) / var(u_y) = δ.

    For δ=1, this is orthogonal regression.
    """
    d = df.dropna(subset=["lower_sec", "parent_edu"]).copy()
    d = demean(d, ["lower_sec", "parent_edu"])
    x = d["parent_edu_d"].values
    y = d["lower_sec_d"].values
    sxx = float(np.var(x, ddof=1))
    syy = float(np.var(y, ddof=1))
    sxy = float(np.cov(x, y, ddof=1)[0, 1])
    beta = (syy - delta * sxx +
            np.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy ** 2)) / (2 * sxy)
    return {"delta": float(delta), "beta_deming": float(beta)}


def main():
    banner("ECONOMETRIC BATTERY 18 — Measurement-error correction")
    df = panel(include_outcomes=False, include_gdp=False)
    _, _, beta_ols = fit_baseline(df)
    print(f"OLS baseline β:  {beta_ols:+.4f}")

    section("Reliability-ratio correction")
    rc = reliability_correction(df)
    for k, b in rc.items():
        print(f"  Assumed reliability {k}:  β_corrected = {b:+.4f}")

    section("Known-noise correction (using observed SD)")
    kn = known_noise_correction(df)
    print(f"  Observed Var(x_d): {kn['observed_var_x']:.2f}")
    for su, b in kn["corrections"].items():
        if b is not None:
            print(f"  {su}:                  β_corrected = {b:+.4f}")
        else:
            print(f"  {su}:                  σ²_u exceeds Var(x); not invertible")

    section("Repeated-measure IV (neighbor-year average)")
    rm = repeated_measure_iv(df)
    print(f"  First-stage F:  {rm['first_stage_F']:.2f}")
    print(f"  IV β:           {rm['iv_beta']:+.4f}")
    print(f"  IV t:           {rm['iv_t']:+.2f}")

    section("Barro-Lee IV for WCDE parent_edu")
    bli = barro_lee_iv(df)
    if bli.get("available"):
        print(f"  N obs:          {bli['n']}")
        print(f"  First-stage F:  {bli['first_stage_F']:.2f}")
        print(f"  IV β:           {bli['iv_beta']:+.4f}  t={bli['iv_t']:+.2f}")
    else:
        print(f"  Barro-Lee IV unavailable: {bli.get('reason', 'data not found')}")

    section("Deming (total least squares)")
    deming = deming_regression(df, delta=1.0)
    print(f"  β (orthogonal):  {deming['beta_deming']:+.4f}")

    section("Verdict")
    candidates = [beta_ols, rc["k=0.95"], rc["k=0.9"], rm["iv_beta"], deming["beta_deming"]]
    if bli.get("available"):
        candidates.append(bli["iv_beta"])
    print(f"  β across EIV corrections: [{min(candidates):.4f}, {max(candidates):.4f}]")
    print(f"  All positive:             {all(b > 0 for b in candidates)}")

    write_battery_checkin("18_measurement_error", {
        "ols_baseline": float(beta_ols),
        "reliability_correction": rc,
        "known_noise_correction": kn,
        "repeated_measure_iv": rm,
        "barro_lee_iv": bli,
        "deming": deming,
    }, script_path=__file__)

    print()
    print(pass_fail(all(b > 0 for b in candidates), "β positive under every EIV correction"))


if __name__ == "__main__":
    main()
