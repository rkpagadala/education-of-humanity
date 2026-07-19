"""
residualization/_shared.py
==========================
Panel construction, fixed-effects regression, and residualization utilities
for the education-vs-GDP analysis scripts.

Re-exports everything from the root _shared.py for convenience.
"""

import sys, os
import importlib.util

# Both this file and the root are named _shared.py. Scripts in residualization/
# add this directory to sys.path, so `from _shared import ...` would find THIS
# file and cause a circular import. Use importlib to load root _shared by path.
_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared.py")
_spec = importlib.util.spec_from_file_location("_shared_root", _root_path)
_root = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root)

# Re-export all public names from root _shared
for _n in dir(_root):
    if not _n.startswith("_"):
        globals()[_n] = getattr(_root, _n)

import numpy as np
import pandas as pd
import statsmodels.api as sm


def find_entry_year(series, threshold):
    """Find the first year where education >= threshold."""
    above = series[series >= threshold]
    return above.index[0] if len(above) > 0 else None


def build_panel(edu_annual, outcome_df, gdp_df, t_years, lag,
                outcome_name="outcome"):
    """
    Build a panel DataFrame with education, log GDP, and outcome at T+lag.
    Returns DataFrame with columns: country, t, edu_t, log_gdp_t, {outcome_name}.
    """
    rows = []
    for c in sorted(edu_annual.keys()):
        s = edu_annual[c]
        for t in t_years:
            if t not in s.index:
                continue
            edu_val = s[t]
            gdp_t = get_wb_val(gdp_df, c, t)
            out_val = get_wb_val(outcome_df, c, t + lag)
            if np.isnan(edu_val) or np.isnan(out_val):
                continue
            rows.append({
                "country": c, "t": t, "edu_t": edu_val,
                "log_gdp_t": np.log(gdp_t) if not np.isnan(gdp_t) and gdp_t > 0 else np.nan,
                outcome_name: out_val,
            })
    return pd.DataFrame(rows)


def build_child_edu_panel(edu_annual, gdp_df, t_years, lag):
    """
    Build a panel for intergenerational education transmission.
    Parent education at T → child education at T+lag (same country, same measure).
    Returns DataFrame with columns: country, t, edu_t, log_gdp_t, child_edu.
    """
    rows = []
    for c in sorted(edu_annual.keys()):
        s = edu_annual[c]
        for t in t_years:
            if t not in s.index or (t + lag) not in s.index:
                continue
            parent_edu = s[t]
            child_edu = s[t + lag]
            gdp_t = get_wb_val(gdp_df, c, t)
            if np.isnan(parent_edu) or np.isnan(child_edu):
                continue
            rows.append({
                "country": c, "t": t, "edu_t": parent_edu,
                "log_gdp_t": np.log(gdp_t) if not np.isnan(gdp_t) and gdp_t > 0 else np.nan,
                "child_edu": child_edu,
            })
    return pd.DataFrame(rows)


def precompute_entry_years(edu_annual, thresholds=range(10, 91)):
    """For each threshold, find entry year per country."""
    entry_years = {}
    for threshold in thresholds:
        cohort = {}
        for c, s in edu_annual.items():
            entry = find_entry_year(s, threshold)
            if entry is not None:
                cohort[c] = entry
        entry_years[threshold] = cohort
    return entry_years


def filter_panel(panel, cohort, ceiling, edu_col="edu_t"):
    """Filter panel to entry-cohort with ceiling."""
    entry_series = panel["country"].map(cohort)
    mask = entry_series.notna() & (panel["t"] >= entry_series) & (panel[edu_col] <= ceiling)
    return panel[mask]


def _demean_and_filter(data, cols):
    """Demean columns by country, drop NaN rows, require ≥2 obs per country.

    Returns (sub, demeaned_dict, n_countries) or None if insufficient data.

    Vectorized: replaces pandas `groupby().transform('mean')` (which runs
    the full group-machinery for every call) with numpy `bincount`. Matches
    pandas numerically to within float64 summation roundoff (~1e-14).
    """
    sub = data.dropna(subset=cols)
    country_col = sub["country"].to_numpy()
    # Factorize country → integer codes; bincount-based demean
    codes, uniq = pd.factorize(country_col, sort=False)
    n_u = len(uniq)
    counts = np.bincount(codes, minlength=n_u)
    keep_c = counts >= 2
    if keep_c.sum() < 3:
        return None
    keep = keep_c[codes]
    if keep.sum() < 10:
        return None
    sub = sub.iloc[keep].copy()
    codes = codes[keep]
    counts = np.bincount(codes, minlength=n_u).astype(float)
    n_countries = int(keep_c.sum())
    dm = {}
    for col in cols:
        v = sub[col].to_numpy()
        s = np.bincount(codes, weights=v, minlength=n_u)
        means = s / counts
        dm[col] = pd.Series(v - means[codes], index=sub.index)
    return sub, dm, n_countries


def fe_r2(x_col, y_col, data):
    """
    Country fixed-effects R² (within-R²).
    Demean by country, regress demeaned y on demeaned x via OLS (no intercept).
    Returns (r2, n_obs, n_countries).

    Closed-form: R² = sxy² / (sxx · syy) matches statsmodels' uncentered R²
    for a no-intercept 1-regressor fit.
    """
    result = _demean_and_filter(data, [x_col, y_col])
    if result is None:
        return np.nan, 0, 0
    sub, dm, n_countries = result
    X = dm[x_col].values
    y = dm[y_col].values
    ok = ~np.isnan(X) & ~np.isnan(y)
    n = int(ok.sum())
    if n < 10:
        return np.nan, 0, 0
    x = X[ok]
    yv = y[ok]
    sxx = float(np.dot(x, x))
    syy = float(np.dot(yv, yv))
    sxy = float(np.dot(x, yv))
    if sxx <= 0.0 or syy <= 0.0:
        return np.nan, n, n_countries
    return (sxy * sxy) / (sxx * syy), n, n_countries


def fe_twoway_r2(x_col, y_col, data, time_col="t"):
    """
    Two-way fixed effects R² (country + time).
    Demean by country AND by time period, then regress via OLS.
    Returns (r2, n_obs, n_countries).
    """
    result = _demean_and_filter(data, [x_col, y_col])
    if result is None:
        return np.nan, 0, 0
    sub, dm, n_countries = result
    # Second pass: demean by time period
    sub["_xdm"] = dm[x_col]
    sub["_ydm"] = dm[y_col]
    xdm2 = sub["_xdm"] - sub.groupby(time_col)["_xdm"].transform("mean")
    ydm2 = sub["_ydm"] - sub.groupby(time_col)["_ydm"].transform("mean")
    X = xdm2.values
    y = ydm2.values
    ok = ~np.isnan(X) & ~np.isnan(y)
    n = int(ok.sum())
    if n < 10:
        return np.nan, 0, 0
    x = X[ok]
    yv = y[ok]
    sxx = float(np.dot(x, x))
    syy = float(np.dot(yv, yv))
    sxy = float(np.dot(x, yv))
    if sxx <= 0.0 or syy <= 0.0:
        return np.nan, n, n_countries
    return (sxy * sxy) / (sxx * syy), n, n_countries


def fe_residualize_gdp(data, fe_func=fe_r2):
    """
    Residualize GDP against education with country FE (Frisch-Waugh-Lovell).
    Returns (sub_with_gdp_resid, edu_gdp_r2) or None.
    """
    result = _demean_and_filter(data, ["edu_t", "log_gdp_t"])
    if result is None:
        return None
    sub, dm, _ = result
    X = dm["edu_t"].values
    y = dm["log_gdp_t"].values
    ok = ~np.isnan(X) & ~np.isnan(y)
    if ok.sum() < 10:
        return None
    xo, yo = X[ok], y[ok]
    sxx = float(np.dot(xo, xo))
    syy = float(np.dot(yo, yo))
    sxy = float(np.dot(xo, yo))
    if sxx <= 0.0:
        return None
    beta = sxy / sxx
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0.0 else np.nan
    sub["gdp_resid"] = y - beta * X
    return sub, r2


def fe_residualize_gdp_twoway(data, time_col="t"):
    """
    Residualize GDP against education with two-way FE (country + time).
    Returns (sub_with_gdp_resid, edu_gdp_r2) or None.
    """
    result = _demean_and_filter(data, ["edu_t", "log_gdp_t"])
    if result is None:
        return None
    sub, dm, _ = result
    # Second pass: demean by time
    sub["_edu_dm"] = dm["edu_t"]
    sub["_gdp_dm"] = dm["log_gdp_t"]
    edu_dm2 = sub["_edu_dm"] - sub.groupby(time_col)["_edu_dm"].transform("mean")
    gdp_dm2 = sub["_gdp_dm"] - sub.groupby(time_col)["_gdp_dm"].transform("mean")
    X = edu_dm2.values
    y = gdp_dm2.values
    ok = ~np.isnan(X) & ~np.isnan(y)
    if ok.sum() < 10:
        return None
    xo, yo = X[ok], y[ok]
    sxx = float(np.dot(xo, xo))
    syy = float(np.dot(yo, yo))
    sxy = float(np.dot(xo, yo))
    if sxx <= 0.0:
        return None
    beta = sxy / sxx
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0.0 else np.nan
    sub["gdp_resid"] = y - beta * X
    return sub, r2


def run_residualized_sweep(panel, entry_years, outcome_col, ceilings,
                           fe_func=fe_r2, resid_func=fe_residualize_gdp,
                           label="outcome", print_every=10):
    """
    Run the full entry-threshold × ceiling sweep with residualization.
    Returns nested dict: results[ceiling][threshold] = {...}.
    """
    all_results = {}

    for ceiling in ceilings:
        print(f"\n  Ceiling = {ceiling}%")
        print(f"  {'Entry':<8} {'Edu R²':>7} {'GDP R²':>7} {'Resid R²':>9} {'Edu→GDP':>8} {'n':>5} {'Ctry':>5}")
        print(f"  {'-' * 60}")

        ceil_results = {}

        for threshold in range(10, 91):
            if threshold > ceiling:
                break

            cohort = entry_years.get(threshold, {})
            if len(cohort) < 3:
                continue

            sub = filter_panel(panel, cohort, ceiling)
            if len(sub) < 10:
                continue

            r2_e, n_e, c_e = fe_func("edu_t", outcome_col, sub)
            r2_g, n_g, c_g = fe_func("log_gdp_t", outcome_col, sub)

            resid_result = resid_func(sub)
            if resid_result is not None:
                sub_resid, edu_gdp_r2 = resid_result
                r2_resid, _, _ = fe_func("gdp_resid", outcome_col, sub_resid)
            else:
                r2_resid, edu_gdp_r2 = np.nan, np.nan

            if threshold % print_every == 0 or threshold == 10:
                print(f"  >= {threshold}%{'':<2} {fmt_r2(r2_e):>7} {fmt_r2(r2_g):>7} "
                      f"{fmt_r2(r2_resid):>9} {fmt_r2(edu_gdp_r2):>8} {n_e:>5} {c_e:>5}")

            ceil_results[str(threshold)] = {
                "edu_r2": round(r2_e, 3) if not np.isnan(r2_e) else None,
                "raw_gdp_r2": round(r2_g, 3) if not np.isnan(r2_g) else None,
                "resid_gdp_r2": round(r2_resid, 3) if not np.isnan(r2_resid) else None,
                "edu_gdp_r2": round(edu_gdp_r2, 3) if not np.isnan(edu_gdp_r2) else None,
                "edu_n": n_e, "edu_countries": c_e,
            }

        all_results[str(ceiling)] = ceil_results

    return all_results


def print_summary(results, ceilings, label="outcome"):
    """Print summary table for entry=10% across ceilings."""
    print(f"\n  SUMMARY: entry=10%")
    efmt = f"Edu→{label}"
    gfmt = f"GDP→{label}"
    rfmt = f"Resid→{label}"
    print(f"  {'Ceiling':<10} {efmt:>9} {gfmt:>9} {rfmt:>10} {'Edu→GDP':>8}")
    print(f"  {'-' * 50}")
    for ceiling in ceilings:
        r = results[str(ceiling)].get("10", {})
        if r and r.get("edu_r2") is not None:
            print(f"  <= {ceiling}%{'':<4} {fmt_r2(r.get('edu_r2')):>9} {fmt_r2(r.get('raw_gdp_r2')):>9} "
                  f"{fmt_r2(r.get('resid_gdp_r2')):>10} {fmt_r2(r.get('edu_gdp_r2')):>8}")


def fe_beta_r2(x_col, y_col, data):
    """
    Country fixed-effects regression returning beta AND R².
    Returns (beta, r2, n_obs, n_countries) or (nan, nan, 0, 0).
    """
    result = _demean_and_filter(data, [x_col, y_col])
    if result is None:
        return np.nan, np.nan, 0, 0
    sub, dm, n_countries = result
    X = dm[x_col].values
    y = dm[y_col].values
    ok = ~np.isnan(X) & ~np.isnan(y)
    n = int(ok.sum())
    if n < 10:
        return np.nan, np.nan, 0, 0
    x = X[ok]
    yv = y[ok]
    sxx = float(np.dot(x, x))
    syy = float(np.dot(yv, yv))
    sxy = float(np.dot(x, yv))
    if sxx <= 0.0:
        return np.nan, np.nan, n, n_countries
    beta = sxy / sxx
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0.0 else np.nan
    return beta, r2, n, n_countries


def clustered_fe(x_col, y_col, data):
    """
    Country FE regression with country-clustered standard errors.
    Uses statsmodels OLS with cov_type="cluster".
    Returns dict with beta, se, pval, r2, n, countries — or None.
    """
    result = _demean_and_filter(data, [x_col, y_col])
    if result is None:
        return None
    sub, dm, n_countries = result
    X = dm[x_col].values
    y = dm[y_col].values
    ok = ~np.isnan(X) & ~np.isnan(y)
    if ok.sum() < 10:
        return None
    model = sm.OLS(y[ok], X[ok]).fit(
        cov_type="cluster",
        cov_kwds={"groups": sub.loc[sub.index[ok], "country"].values},
    )
    return {
        "beta": model.params[0],
        "se": model.bse[0],
        "pval": model.pvalues[0],
        "r2": model.rsquared,
        "n": int(ok.sum()),
        "countries": n_countries,
    }


def compare_predictors(data, outcome_col, fe_func=fe_r2, resid_func=fe_residualize_gdp):
    """Compare education vs GDP vs residualized GDP for predicting an outcome.

    Runs the three-way comparison that most residualization scripts need:
      1. Education → outcome (FE R²)
      2. GDP → outcome (FE R²)
      3. Residualized GDP → outcome (after removing education's contribution)

    Returns dict with edu_r2, gdp_r2, resid_gdp_r2, edu_gdp_r2, n, countries.
    """
    r2_e, n, ctry = fe_func("edu_t", outcome_col, data)
    r2_g, _, _ = fe_func("log_gdp_t", outcome_col, data)
    r2_resid = np.nan
    edu_gdp_r2 = np.nan
    resid = resid_func(data)
    if resid is not None:
        sub_r, edu_gdp_r2 = resid
        r2_resid, _, _ = fe_func("gdp_resid", outcome_col, sub_r)
    return {
        "edu_r2": r2_e, "gdp_r2": r2_g, "resid_gdp_r2": r2_resid,
        "edu_gdp_r2": edu_gdp_r2, "n": n, "countries": ctry,
    }
