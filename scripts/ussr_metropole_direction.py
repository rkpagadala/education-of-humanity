"""
ussr_metropole_direction.py

Refines the Moscow-distance test by splitting distance into westward
(toward Europe) and eastward (toward Asia) components.

If "distance from Moscow" is the right story, both directions should
predict the U5MR residual symmetrically — Riga at 842 km W and Tbilisi
at 1,646 km S should sit on the same regression line, just at different
points on it.

If "distance from the European core" is the right story (i.e., the
metropole was a Russian-European apparatus that lied harder about
populations farther into Asia/the Muslim periphery), then westward
distance should NOT predict the residual at all, and eastward distance
should carry the entire effect. The 6 westward republics would cluster
near zero residual regardless of how far west; the 8 eastward republics
would form the inflation gradient.

We test by:
  1. Splitting the 14 non-Russia republics into west (Δlon < 0) and
     east/south (Δlon ≥ 0) groups.
  2. Within each group, correlating great-circle distance with the
     U5MR residual.
  3. Comparing the slopes / R² between groups.
  4. Producing a map-style scatter showing the asymmetry.
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
                       "ussr_metropole_direction.png")

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
        # Signed eastward distance (km on E-W axis at Moscow latitude)
        # positive = east of Moscow, negative = west of Moscow
        east_km = haversine_km(MOSCOW[0], MOSCOW[1],
                               MOSCOW[0], lon)
        if lon < MOSCOW[1]:
            east_km = -east_km
        # Signed northward distance
        north_km = haversine_km(MOSCOW[0], MOSCOW[1],
                                lat, MOSCOW[1])
        if lat < MOSCOW[0]:
            north_km = -north_km
        v = wcde.get(c, {})
        rows.append({
            "country": c,
            "capital": city,
            "dist_km": d,
            "east_km": east_km,
            "north_km": north_km,
            "le_resid_sd": v.get("le_resid_sd"),
            "u5log_resid_sd": v.get("u5log_resid_sd"),
            "le_u5_sd": v.get("mean_abs_le_u5_sd"),
        })
    df = pd.DataFrame(rows)
    no_ru = df[df["country"] != "russian federation"].copy()
    no_ru["direction"] = np.where(no_ru["east_km"] < 0,
                                  "west (Europe)", "east (Asia)")

    print("USSR republics — directional decomposition of Moscow distance\n")
    print(f"  {'country':<22} {'capital':<11} {'dist_km':>8} "
          f"{'east_km':>9} {'dir':<14}  {'U5MR σ':>8} "
          f"{'LE+U5 |σ|':>10}")
    print("  " + "-" * 92)
    for _, r in no_ru.sort_values("east_km").iterrows():
        u5 = (f"{r['u5log_resid_sd']:>+8.2f}"
              if r['u5log_resid_sd'] is not None
              else f"{'  --':>8}")
        leu5 = (f"{r['le_u5_sd']:>10.2f}"
                if r['le_u5_sd'] is not None
                else f"{'  --':>10}")
        print(f"  {r['country']:<22} {r['capital']:<11} "
              f"{r['dist_km']:>8.0f} {r['east_km']:>+9.0f} "
              f"{r['direction']:<14}  {u5} {leu5}")

    # ----- correlations -----
    print("\nCorrelations (excl. Russia):\n")

    def _corr(sub, x, y, label):
        s = sub.dropna(subset=[x, y])
        if len(s) < 3:
            return None
        r = np.corrcoef(s[x], s[y])[0, 1]
        return f"  r({label}) = {r:+.3f}  (n={len(s)})"

    print("  All 14 republics:")
    for x in ["dist_km", "east_km"]:
        for y, ylabel in [("u5log_resid_sd", "U5MR σ"),
                          ("le_u5_sd", "LE+U5 |σ|")]:
            line = _corr(no_ru, x, y, f"{x} → {ylabel}")
            if line:
                print(line)

    print("\n  WESTWARD subgroup (Belarus, Ukraine, Baltics, Moldova):")
    west = no_ru[no_ru["east_km"] < 0]
    for y, ylabel in [("u5log_resid_sd", "U5MR σ"),
                      ("le_u5_sd", "LE+U5 |σ|")]:
        line = _corr(west, "dist_km", y, f"dist_km → {ylabel}")
        if line:
            print(line)
    if west["u5log_resid_sd"].notna().any():
        m = west["u5log_resid_sd"].dropna()
        print(f"  → mean U5MR σ in westward group: {m.mean():+.2f} "
              f"(min {m.min():+.2f}, max {m.max():+.2f}, n={len(m)})")

    print("\n  EASTWARD subgroup (Caucasus + Central Asia):")
    east = no_ru[no_ru["east_km"] >= 0]
    for y, ylabel in [("u5log_resid_sd", "U5MR σ"),
                      ("le_u5_sd", "LE+U5 |σ|")]:
        line = _corr(east, "dist_km", y, f"dist_km → {ylabel}")
        if line:
            print(line)
    if east["u5log_resid_sd"].notna().any():
        m = east["u5log_resid_sd"].dropna()
        print(f"  → mean U5MR σ in eastward group: {m.mean():+.2f} "
              f"(min {m.min():+.2f}, max {m.max():+.2f}, n={len(m)})")

    # Within-group slopes
    print()
    for grp_name, grp in [("WEST", west), ("EAST", east)]:
        s = grp.dropna(subset=["dist_km", "u5log_resid_sd"])
        if len(s) >= 3:
            b1, b0 = np.polyfit(s["dist_km"], s["u5log_resid_sd"], 1)
            print(f"  OLS within {grp_name}: U5MR σ = "
                  f"{b0:.2f} + {b1*1000:.3f}·(dist/1000km)  "
                  f"(n={len(s)})")

    # ----- map-style figure -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Panel 1: spatial map with U5MR residual encoded
    ax1.axhline(0, color="#888", linewidth=0.5, alpha=0.5)
    ax1.axvline(0, color="#888", linewidth=0.5, alpha=0.5)
    ax1.scatter([0], [0], s=200, marker="*", color="#c0392b",
                edgecolor="white", zorder=5, label="Moscow")
    for _, r in no_ru.iterrows():
        if r["u5log_resid_sd"] is None or pd.isna(r["u5log_resid_sd"]):
            color, size = "#888", 60
        else:
            v = r["u5log_resid_sd"]
            # color: blue (low/clean) to red (high/inflated)
            cmap = plt.cm.RdBu_r
            color = cmap(min(1, max(0, v / 4)))
            size = 80 + 30 * v
        ax1.scatter(r["east_km"], r["north_km"], s=size, color=color,
                    edgecolor="black", linewidth=0.5, zorder=4)
        label = (r["country"]
                 .replace(" federation", "")
                 .replace("republic of ", ""))
        u5_str = (f"\n{r['u5log_resid_sd']:+.1f}σ"
                  if r["u5log_resid_sd"] is not None and
                  not pd.isna(r["u5log_resid_sd"]) else "")
        ax1.annotate(f"{label}{u5_str}",
                     xy=(r["east_km"], r["north_km"]),
                     xytext=(7, 5), textcoords="offset points",
                     fontsize=8, color="#222")
    ax1.set_xlabel("East-west distance from Moscow (km, + = east)",
                   fontsize=10)
    ax1.set_ylabel("North-south distance from Moscow (km, + = north)",
                   fontsize=10)
    ax1.set_title("USSR republics positioned around Moscow,\n"
                  "U5MR residual encoded by color and size",
                  fontsize=11, fontweight="bold")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9, loc="lower left", frameon=False)

    # Panel 2: distance vs U5MR residual, broken by direction
    sub = no_ru.dropna(subset=["dist_km", "u5log_resid_sd"])
    for direction, color, marker in [
        ("west (Europe)", "#1f4e79", "o"),
        ("east (Asia)",   "#a05a2c", "s"),
    ]:
        s = sub[sub["direction"] == direction]
        ax2.scatter(s["dist_km"], s["u5log_resid_sd"], s=90,
                    color=color, marker=marker, edgecolor="white",
                    linewidth=0.5, zorder=3,
                    label=f"{direction} (n={len(s)})")
        for _, r in s.iterrows():
            label = (r["country"]
                     .replace(" federation", "")
                     .replace("republic of ", ""))
            ax2.annotate(label,
                         xy=(r["dist_km"], r["u5log_resid_sd"]),
                         xytext=(5, 3), textcoords="offset points",
                         fontsize=8, color="#222")
        if len(s) >= 3:
            b1, b0 = np.polyfit(s["dist_km"], s["u5log_resid_sd"], 1)
            xx = np.linspace(s["dist_km"].min() * 0.9,
                             s["dist_km"].max() * 1.05, 50)
            ax2.plot(xx, b0 + b1 * xx, "--", color=color,
                     linewidth=1.2, alpha=0.7,
                     label=f"  slope = {b1*1000:+.3f} σ / 1000km")
    ax2.axhline(0, color="#222", linewidth=0.6, linestyle=":",
                alpha=0.4)
    ax2.set_xlabel("Great-circle distance from Moscow (km)",
                   fontsize=10)
    ax2.set_ylabel("log U5MR residual (σ from non-USSR fit)",
                   fontsize=10)
    ax2.set_title("Distance does no work westward; "
                  "all the work is eastward",
                  fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, loc="upper left", frameon=False)
    fig.suptitle(
        "The USSR education-anomaly residual is asymmetric: "
        "distance from Moscow only predicts inflation toward Asia\n"
        "Westward republics (Baltics, Belarus, Ukraine, Moldova) "
        "are clean regardless of distance; "
        "eastward republics (Caucasus, Central Asia) form the gradient.",
        fontsize=12, fontweight="bold", y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {OUT_FIG}")

    # ----- checkin -----
    def _corr_val(sub, x, y):
        s = sub.dropna(subset=[x, y])
        if len(s) < 3:
            return None
        return float(np.corrcoef(s[x], s[y])[0, 1])

    write_checkin("ussr_metropole_direction.json", {
        "rows": no_ru.where(pd.notna(no_ru), None)
                     .to_dict(orient="records"),
        "correlations": {
            "all_dist_to_u5log": _corr_val(no_ru, "dist_km",
                                           "u5log_resid_sd"),
            "west_dist_to_u5log": _corr_val(west, "dist_km",
                                            "u5log_resid_sd"),
            "east_dist_to_u5log": _corr_val(east, "dist_km",
                                            "u5log_resid_sd"),
        },
        "group_means_u5log_sd": {
            "west": float(west["u5log_resid_sd"].dropna().mean()),
            "east": float(east["u5log_resid_sd"].dropna().mean()),
        },
    }, script_path="scripts/ussr_metropole_direction.py")


if __name__ == "__main__":
    main()
