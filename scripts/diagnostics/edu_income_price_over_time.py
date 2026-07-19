"""
edu_income_price_over_time.py

The reductio the static scatter invites. Read ln(GDP) = a + b*E as "income
drives education" and it says: to reach a given schooling level you need a
particular income -- and to go higher you need exponentially more. If that were
the mechanism, the income required to attain a given education level would be a
stable structural price over time.

It is not. Ever-poorer countries have reached the same schooling levels decade
after decade (Korea, Cuba, Bangladesh, Sri Lanka schooled while poor). So the
"price of education in income" collapses over time -- the signature that
education was never bought with income in the first place.

TWO independent estimators of that price, reported together so the result does
not rest on one modelling choice:

  1. Model-free band median/mean. Among countries whose lower-sec completion is
     within +/-BAND of a target, the median (and mean) GDP per capita. Reported
     for BAND in {5,10,15} pp as a robustness sweep.
  2. Model-based fit inversion. Refit ln G = a_y + b_y*E on ALL countries each
     year and invert at the target: implied income = exp(a_y + b_y*E*). Uses
     every country, not just those in a band.

Both are given in absolute constant-2015 USD and relative to each year's
world-median income (which removes global growth -- the honest comparison, since
secular growth would otherwise push the price up, not down).

Three panels (single axis each; no dual-axis):
  A  The exponential fit refit per decade -- the curve drifts DOWN/RIGHT: any
     given completion is reached at lower income each decade.
  B  Income required for a given completion (30/50/70%), constant 2015 USD:
     fit-inversion line + band-median markers (the two estimators agree).
  C  The same relative to each year's world-median income.

Run modes (via env, matching the repo's figure convention):
  default   -> English figure to figures/ (gallery) + paper/figures_web/ (web sync)
  FOR_ZH=1  -> Simplified-Chinese figure to paper/figures_web/<name>_zh.png only
              (rides the same figures_web sync; _shared registers the CJK font)

Data:
  lower-sec completion, both sexes 20-24 .... wcde/data/processed/cohort_lower_sec_both.csv
  GDP per capita, constant 2015 USD ......... data/gdppercapita_us_inflation_adjusted.csv

Outputs:
  figures/edu_income_price_over_time.png              (English, local gallery)
  paper/figures_web/edu_income_price_over_time.png    (English, website sync)
  paper/figures_web/edu_income_price_over_time_zh.png (Chinese, website sync; FOR_ZH=1)
  checkin/edu_income_price_over_time.json

Diagnostic, not a registered paper number.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT)

from _shared import (  # noqa: E402
    PROC, REPO_ROOT, write_checkin, load_wide_indicator,
    standardize_country_name, FOR_ZH,
)

WEB_DIR = os.path.join(REPO_ROOT, "paper", "figures_web")
FIG_PATH = os.path.join(REPO_ROOT, "figures", "edu_income_price_over_time.png")
WEB_FIG_PATH = os.path.join(WEB_DIR, "edu_income_price_over_time.png")
WEB_FIG_PATH_ZH = os.path.join(WEB_DIR, "edu_income_price_over_time_zh.png")

GRID = list(range(1960, 2016, 5))
TARGETS = [30, 50, 70]                 # completion levels (%)
BANDS = [5, 10, 15]                    # +/- pp robustness sweep
PRIMARY_BAND = 10
DECADE_LINES = [1970, 1985, 2000, 2015]
MIN_IN_BAND = 6
MIN_FIT_N = 30

TARGET_COLORS = {30: "#C4441C", 50: "#4269D0", 70: "#3CA951"}
DECADE_COLORS = {1970: "#8A4FBF", 1985: "#4269D0",
                 2000: "#3CA951", 2015: "#C4441C"}
GREY = "#9aa0a6"

# In-image strings by language (paper convention: web/zh figures are titled).
LABELS = {
    "en": {
        "suptitle": "If income bought education, its price would be stable. It collapses.",
        "A_title": "A · The curve drifts down over time\n"
                   "same completion, reached at lower income each decade",
        "A_x": "lower-secondary completion, 20–24 (%)",
        "A_y": "GDP per capita (constant 2015 USD)",
        "A_legend": "fit year",
        "B_title": "B · Income needed for a given completion\n"
                   "line = fit inversion · dots = ±10 pp band median",
        "B_y": "income to reach that completion (constant 2015 USD)",
        "B_leg": "{t}% (fit)",
        "C_title": "C · …relative to each era’s world income\n"
                   "removes global growth (the honest comparison)",
        "C_y": "income ÷ world-median income that year",
        "C_leg": "{t}% completion",
        "world_median": "world median",
        "year": "year",
    },
    "zh": {
        "suptitle": "如果收入能买来教育，其价格本应稳定。事实是它在崩塌。",
        "A_title": "A · 曲线随时间下移\n同样的完成率，每十年以更低收入即可达到",
        "A_x": "初中完成率，20–24岁（%）",
        "A_y": "人均GDP（2015年不变价美元）",
        "A_legend": "拟合年份",
        "B_title": "B · 达到给定完成率所需的收入\n实线＝拟合反推 · 圆点＝±10个百分点区间中位数",
        "B_y": "达到该完成率所需收入（2015年不变价美元）",
        "B_leg": "{t}%（拟合）",
        "C_title": "C · ……相对于同期世界收入\n剔除全球增长（更诚实的比较）",
        "C_y": "收入 ÷ 当年世界收入中位数",
        "C_leg": "{t}%完成率",
        "world_median": "世界中位数",
        "year": "年份",
    },
}


def _style(ax):
    ax.grid(True, alpha=0.2, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load_grid():
    edu = pd.read_csv(os.path.join(PROC, "cohort_lower_sec_both.csv"),
                      index_col="country")
    edu.columns = [int(c) for c in edu.columns]
    edu.index = edu.index.str.lower().str.strip()

    gdp = load_wide_indicator("gdppercapita_us_inflation_adjusted.csv")
    gdp.columns = gdp.columns.astype(int)
    gdp.index = [standardize_country_name(c) or c for c in gdp.index]

    yrs = [y for y in GRID if y in edu.columns and y in gdp.columns]
    common = sorted(set(edu.index) & set(gdp.index))
    return edu.loc[common, yrs], gdp.loc[common, yrs], yrs


def world_median_by_year(edu, gdp, yrs):
    med = {}
    for y in yrs:
        e, g = edu[y], gdp[y]
        m = e.notna() & g.notna() & (g > 0)
        if m.sum() >= 10:
            med[y] = float(g[m].median())
    return med


def band_estimates(edu, gdp, yrs, band, world_med):
    """Model-free: median & mean GDP among countries within +/-band of each
    target, absolute and relative to world median, per year."""
    out = {t: {"abs_median": {}, "abs_mean": {}, "rel_median": {}}
           for t in TARGETS}
    for y in yrs:
        if y not in world_med:
            continue
        e, g = edu[y], gdp[y]
        m = e.notna() & g.notna() & (g > 0)
        e, g = e[m], g[m]
        for t in TARGETS:
            sel = g[(e >= t - band) & (e <= t + band)]
            if len(sel) >= MIN_IN_BAND:
                out[t]["abs_median"][y] = float(sel.median())
                out[t]["abs_mean"][y] = float(sel.mean())
                out[t]["rel_median"][y] = float(sel.median() / world_med[y])
    return out


def yearly_fits(edu, gdp, yrs):
    """Per-year exponential fit ln G = a + b E on ALL countries."""
    fits = {}
    for y in yrs:
        e, g = edu[y], gdp[y]
        m = e.notna() & g.notna() & (g > 0)
        e, g = e[m].values, g[m].values
        if len(e) >= MIN_FIT_N:
            b, a = np.polyfit(e, np.log(g), 1)
            fits[y] = {"a": float(a), "b": float(b), "n": int(len(e))}
    return fits


def fit_implied(fits, world_med, target):
    """Model-based: implied income at a target completion from each year's fit,
    absolute and relative to world median."""
    absolute, relative = {}, {}
    for y, f in fits.items():
        val = float(np.exp(f["a"] + f["b"] * target))
        absolute[y] = val
        if y in world_med:
            relative[y] = val / world_med[y]
    return absolute, relative


def ratio(series):
    ks = sorted(series)
    if len(ks) >= 2:
        return {"first_year": ks[0], "first": series[ks[0]],
                "last_year": ks[-1], "last": series[ks[-1]],
                "ratio_first_over_last": series[ks[0]] / series[ks[-1]]}
    return None


def build_figure(fits, primary, world_med, yrs, lang, out_path):
    L = LABELS[lang]
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 5))

    # A: drifting curve
    xs = np.linspace(0, 100, 200)
    for d in DECADE_LINES:
        if d in fits:
            f = fits[d]
            axA.plot(xs, np.exp(f["a"] + f["b"] * xs),
                     color=DECADE_COLORS[d], lw=2, label=str(d))
    axA.set_yscale("log")
    axA.set_xlabel(L["A_x"])
    axA.set_ylabel(L["A_y"])
    axA.set_title(L["A_title"], fontsize=10, loc="left")
    axA.legend(title=L["A_legend"], frameon=False, fontsize=9)
    _style(axA)

    # B: absolute price — fit line + band-median dots
    for t in TARGETS:
        fa, _ = fit_implied(fits, world_med, t)
        ks = sorted(fa)
        axB.plot(ks, [fa[k] for k in ks], color=TARGET_COLORS[t], lw=2,
                 label=L["B_leg"].format(t=t))
        bm = primary[t]["abs_median"]
        bk = sorted(bm)
        axB.scatter(bk, [bm[k] for k in bk], color=TARGET_COLORS[t], s=16,
                    zorder=3)
    axB.set_yscale("log")
    axB.set_xlabel(L["year"])
    axB.set_ylabel(L["B_y"])
    axB.set_title(L["B_title"], fontsize=10, loc="left")
    axB.legend(frameon=False, fontsize=8)
    _style(axB)

    # C: relative price
    for t in TARGETS:
        _, fr = fit_implied(fits, world_med, t)
        ks = sorted(fr)
        axC.plot(ks, [fr[k] for k in ks], color=TARGET_COLORS[t], lw=2,
                 label=L["C_leg"].format(t=t))
        rm = primary[t]["rel_median"]
        rk = sorted(rm)
        axC.scatter(rk, [rm[k] for k in rk], color=TARGET_COLORS[t], s=16,
                    zorder=3)
    axC.axhline(1.0, color=GREY, lw=1, ls="--")
    axC.text(yrs[0], 1.05, L["world_median"], color=GREY, fontsize=8)
    axC.set_xlabel(L["year"])
    axC.set_ylabel(L["C_y"])
    axC.set_title(L["C_title"], fontsize=10, loc="left")
    axC.legend(frameon=False, fontsize=9)
    _style(axC)

    fig.suptitle(L["suptitle"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    edu, gdp, yrs = load_grid()
    world_med = world_median_by_year(edu, gdp, yrs)
    fits = yearly_fits(edu, gdp, yrs)
    band_sweep = {b: band_estimates(edu, gdp, yrs, b, world_med) for b in BANDS}
    primary = band_sweep[PRIMARY_BAND]

    if FOR_ZH:
        # Chinese web figure only; English + JSON already produced by the
        # default run. Rides the paper/figures_web -> public figures sync.
        build_figure(fits, primary, world_med, yrs, "zh", WEB_FIG_PATH_ZH)
        print("zh figure:", WEB_FIG_PATH_ZH)
        return

    build_figure(fits, primary, world_med, yrs, "en", FIG_PATH)
    build_figure(fits, primary, world_med, yrs, "en", WEB_FIG_PATH)

    fa50, fr50 = fit_implied(fits, world_med, 50)
    robustness_50 = {
        f"band_{b}": {
            "abs_median_ratio": (ratio(band_sweep[b][50]["abs_median"]) or {})
            .get("ratio_first_over_last"),
            "abs_mean_ratio": (ratio(band_sweep[b][50]["abs_mean"]) or {})
            .get("ratio_first_over_last"),
            "rel_median_ratio": (ratio(band_sweep[b][50]["rel_median"]) or {})
            .get("ratio_first_over_last"),
        } for b in BANDS
    }

    result = {
        "description": "Price of education in income over time. If income drove "
                       "education the income required for a given completion would "
                       "be stable; it collapses in absolute terms and far more "
                       "relative to world income. Two estimators (band median/mean "
                       "and per-year fit inversion) agree. Diagnostic, not a paper number.",
        "targets_pct": TARGETS, "bands_pp": BANDS, "primary_band_pp": PRIMARY_BAND,
        "yearly_exponential_fits": fits,
        "world_median_gdp_by_year": world_med,
        "fit_inversion": {
            t: {"absolute": fit_implied(fits, world_med, t)[0],
                "relative": fit_implied(fits, world_med, t)[1]}
            for t in TARGETS
        },
        "band_median_primary": {t: primary[t] for t in TARGETS},
        "headline_50pct": {
            "fit_absolute": ratio(fa50),
            "fit_relative": ratio(fr50),
            "band10_absolute_median": ratio(primary[50]["abs_median"]),
            "band10_relative_median": ratio(primary[50]["rel_median"]),
        },
        "robustness_50pct_ratios": robustness_50,
        "figures": [os.path.relpath(FIG_PATH, REPO_ROOT),
                    os.path.relpath(WEB_FIG_PATH, REPO_ROOT)],
    }
    write_checkin("edu_income_price_over_time.json", result,
                  script_path="scripts/diagnostics/edu_income_price_over_time.py")

    print("E=50% price drop (first→last year):")
    print(f"  fit  ABS ÷{ratio(fa50)['ratio_first_over_last']:.2f}   "
          f"REL ÷{ratio(fr50)['ratio_first_over_last']:.2f}")
    print(f"  band10 ABS ÷{ratio(primary[50]['abs_median'])['ratio_first_over_last']:.2f}   "
          f"REL ÷{ratio(primary[50]['rel_median'])['ratio_first_over_last']:.2f}")
    print("robustness (abs_median ÷ across bands):",
          {b: round(robustness_50[f'band_{b}']['abs_median_ratio'], 2) for b in BANDS})
    print("robustness (rel_median ÷ across bands):",
          {b: round(robustness_50[f'band_{b}']['rel_median_ratio'], 2) for b in BANDS})
    print("figures:", FIG_PATH, "and", WEB_FIG_PATH)


if __name__ == "__main__":
    main()
