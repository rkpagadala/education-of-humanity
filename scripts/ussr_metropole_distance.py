"""
ussr_metropole_distance.py

Tests the social-distance / colonial-periphery hypothesis directly:
do USSR republics farther from Moscow show larger U5MR residuals
in the global education-vs-phenotype relationship?

Distance measured great-circle from Moscow (55.7558N, 37.6173E) to
the Soviet-era capital of each republic.

If the inflation pattern in Soviet education statistics is a
metropolitan-fiction artifact (the center claimed near-100% lower-secondary completion for
republics it did not have to answer to), distance to Moscow should
positively correlate with the residual. The republics the center
could lie about most easily — physically far, ethnically distant —
should show the largest gap between reported education and
phenotype.

Outputs: ranked table, correlation, scatter figure, checkin JSON.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from _shared import REPO_ROOT, CHECKIN, write_checkin

OUT_FIG = os.path.join(REPO_ROOT, "scripts", "figures",
                       "ussr_metropole_distance.png")

# Soviet-era capital cities. For Kazakhstan we use Alma-Ata (Almaty),
# the Soviet-era capital, not Astana. For Kyrgyzstan we use Frunze
# coords (= modern Bishkek).
CAPITALS = {
    "russian federation":   ("Moscow",       55.7558, 37.6173),
    "ukraine":              ("Kyiv",         50.4501, 30.5234),
    "belarus":              ("Minsk",        53.9006, 27.5590),
    "estonia":              ("Tallinn",      59.4370, 24.7536),
    "latvia":               ("Riga",         56.9496, 24.1052),
    "lithuania":            ("Vilnius",      54.6872, 25.2797),
    "republic of moldova":  ("Chisinau",     47.0105, 28.8638),
    "georgia":              ("Tbilisi",      41.7151, 44.8271),
    "armenia":              ("Yerevan",      40.1872, 44.5152),
    "azerbaijan":           ("Baku",         40.4093, 49.8671),
    "kazakhstan":           ("Alma-Ata",     43.2220, 76.8512),
    "uzbekistan":           ("Tashkent",     41.2995, 69.2401),
    "turkmenistan":         ("Ashgabat",     37.9601, 58.3261),
    "tajikistan":           ("Dushanbe",     38.5598, 68.7870),
    "kyrgyzstan":           ("Frunze",       42.8746, 74.5698),
}
MOSCOW = (55.7558, 37.6173)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    return float(2 * R * np.arcsin(np.sqrt(a)))


def main():
    ck = json.load(open(
        os.path.join(CHECKIN, "ussr_per_country_residuals.json")))
    wcde = ck["wcde"]

    rows = []
    for c, (city, lat, lon) in CAPITALS.items():
        d = haversine_km(MOSCOW[0], MOSCOW[1], lat, lon)
        v = wcde.get(c, {})
        rows.append({
            "country": c,
            "capital": city,
            "dist_km": d,
            "le_resid_sd": v.get("le_resid_sd"),
            "u5log_resid_sd": v.get("u5log_resid_sd"),
            "le_u5_sd": v.get("mean_abs_le_u5_sd"),
        })
    df = pd.DataFrame(rows).sort_values("dist_km")

    print("Distance from Moscow vs USSR education-anomaly residuals\n")
    print(f"  {'country':<22} {'capital':<12} {'dist_km':>9}  "
          f"{'LE σ':>7} {'U5MR σ':>8} {'LE+U5 |σ|':>10}")
    print("  " + "-" * 78)
    for _, r in df.iterrows():
        def f(v, w, p):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return f"{'  --':>{w}}"
            return f"{v:>{w}.{p}f}"
        print(f"  {r['country']:<22} {r['capital']:<12} "
              f"{r['dist_km']:>9.0f}  "
              f"{f(r['le_resid_sd'], 7, 2)} "
              f"{f(r['u5log_resid_sd'], 8, 2)} "
              f"{f(r['le_u5_sd'], 10, 2)}")

    # Correlations (exclude Moscow itself: 0 km, 0 residual is trivial)
    print()
    for ycol, ylabel in [
        ("u5log_resid_sd", "U5MR σ"),
        ("le_resid_sd", "LE σ  "),
        ("le_u5_sd", "LE+U5 |σ|"),
    ]:
        sub = df.dropna(subset=["dist_km", ycol])
        # Exclude Russia (zero distance to itself)
        sub_no_ru = sub[sub["country"] != "russian federation"]
        if len(sub_no_ru) >= 5:
            r_all = np.corrcoef(sub["dist_km"], sub[ycol])[0, 1]
            r_excl = np.corrcoef(sub_no_ru["dist_km"],
                                 sub_no_ru[ycol])[0, 1]
            print(f"  Pearson r(dist_km → {ylabel}): "
                  f"{r_all:+.3f} (n={len(sub)} incl. Russia), "
                  f"{r_excl:+.3f} (n={len(sub_no_ru)} excl. Russia)")

    # ----- scatter -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    def _scatter(ax, ycol, ylabel, title):
        sub = df.dropna(subset=["dist_km", ycol])
        # Color: Christian core (closer) vs Muslim periphery (farther)
        for _, r in sub.iterrows():
            color = "#1f4e79" if r["dist_km"] < 2500 else "#a05a2c"
            ax.scatter(r["dist_km"], r[ycol], s=80, color=color,
                       edgecolor="white", linewidth=0.5, zorder=3)
            label = (r["country"]
                     .replace(" federation", "")
                     .replace("republic of ", ""))
            ax.annotate(label,
                        xy=(r["dist_km"], r[ycol]),
                        xytext=(5, 3), textcoords="offset points",
                        fontsize=8, color="#222")
        sub_no_ru = sub[sub["country"] != "russian federation"]
        if len(sub_no_ru) >= 5:
            b1, b0 = np.polyfit(sub_no_ru["dist_km"],
                                sub_no_ru[ycol], 1)
            xx = np.linspace(0, sub["dist_km"].max() * 1.05, 50)
            ax.plot(xx, b0 + b1 * xx, "--", color="#222",
                    linewidth=1.0, alpha=0.6,
                    label=f"OLS (excl. Russia): "
                          f"y = {b0:.2f} + {b1*1000:.3f}·(dist/1000km)")
            r_val = np.corrcoef(sub_no_ru["dist_km"],
                                sub_no_ru[ycol])[0, 1]
            ax.set_title(f"{title}\nr = {r_val:+.2f}, "
                         f"n = {len(sub_no_ru)} (Russia excluded)",
                         fontsize=11, fontweight="bold")
        ax.axhline(0, color="#222", linewidth=0.6, linestyle=":",
                   alpha=0.4)
        ax.set_xlabel("Great-circle distance from Moscow (km)",
                      fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best", frameon=False)

    _scatter(ax1, "u5log_resid_sd",
             "log U5MR residual (σ from non-USSR fit)",
             "U5MR phenotype residual vs distance to Moscow")
    _scatter(ax2, "le_u5_sd",
             "LE + U5 mean |σ|",
             "Combined LE+U5 residual vs distance to Moscow")
    fig.suptitle(
        "Soviet education statistics: residual scales with distance "
        "to Moscow\n"
        "Republics the metropole could not check on closely show "
        "larger gaps between reported lsec and phenotype",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {OUT_FIG}")

    write_checkin("ussr_metropole_distance.json", {
        "rows": df.where(pd.notna(df), None).to_dict(orient="records"),
        "correlations_excl_russia": {
            "dist_to_u5log_sd": float(np.corrcoef(
                df[(df["country"] != "russian federation") &
                   df["u5log_resid_sd"].notna()]["dist_km"],
                df[(df["country"] != "russian federation") &
                   df["u5log_resid_sd"].notna()]["u5log_resid_sd"])[0, 1]),
            "dist_to_le_sd": float(np.corrcoef(
                df[(df["country"] != "russian federation") &
                   df["le_resid_sd"].notna()]["dist_km"],
                df[(df["country"] != "russian federation") &
                   df["le_resid_sd"].notna()]["le_resid_sd"])[0, 1]),
        },
    }, script_path="scripts/ussr_metropole_distance.py")


if __name__ == "__main__":
    main()
