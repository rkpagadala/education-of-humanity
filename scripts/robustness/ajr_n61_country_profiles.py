"""
robustness/ajr_n61_country_profiles.py
======================================
Reads data/ajr2001/ajr_n61_country_table.csv (produced by
ajr_n61_deep_dive.py) and writes a markdown report with one card per
country, grouped by cluster (Catholic crossers, Protestant non-crossers,
city-states, settler colonies, etc.). Each card carries 1-2 sentence
narrative tags tied to the paper's framework.

Output: checkin/ajr_n61_profiles.md
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _shared import DATA, CHECKIN, TFR_THRESHOLD, LE_THRESHOLD


CSV_PATH = os.path.join(DATA, "ajr2001", "ajr_n61_country_table.csv")
OUT_PATH = os.path.join(CHECKIN, "ajr_n61_profiles.md")

df = pd.read_csv(CSV_PATH)


# ── Cluster assignment ───────────────────────────────────────────────

SETTLER_COLONIES = {"United States of America", "Canada", "Australia", "New Zealand"}
CITY_STATES = {"Singapore", "Hong Kong, China", "Malta", "Bahamas",
               "Trinidad and Tobago"}


def cluster(row):
    if row["country"] in SETTLER_COLONIES:
        return "1_settler_protestant"
    if row["country"] in CITY_STATES:
        return "2_small_islands_or_city_states"
    crossed = not pd.isna(row["crossing_year"])
    prot = row["protestant"] == 1
    if prot and crossed:
        return "3_protestant_crossers"
    if prot and not crossed:
        return "4_protestant_non_crossers"
    if not prot and crossed:
        return "5_catholic_crossers"
    if not prot and not crossed:
        return "6_catholic_non_crossers"
    return "7_other"


df["cluster"] = df.apply(cluster, axis=1)


# ── Hand-curated narrative tags ──────────────────────────────────────
# One-sentence frame anchored to the paper's mechanism. Default fallback
# generated programmatically; overrides for the high-leverage countries.

NARRATIVE = {
    "United States of America": (
        "Settler-Protestant founding tradition; Knox/Luther literacy chain "
        "operating from arrival; first nation to reach near-universal "
        "primary in the 19th century."
    ),
    "Canada": (
        "Settler-Protestant + state-priority schooling; crossed early; "
        "education trajectory diverges sharply from Latin American "
        "Catholic-colonized peers despite similar continent-scale geography."
    ),
    "Australia": (
        "Settler-Protestant; deep state schooling commitment from federation; "
        "crossed alongside other Anglo settler colonies."
    ),
    "New Zealand": (
        "Settler-Protestant; among earliest universal-schooling regimes."
    ),
    "Singapore": (
        "Lee Kuan Yew's singular-priority education push 1965-onward; "
        "the paper-clean case that high avexpr is downstream of state-paced "
        "schooling, not inherited from colonial institutions."
    ),
    "Cuba": "Not in AJR-64 base sample; flagged for reference only.",
    "Viet Nam": (
        "Within-AJR-64 socialist-credentialing test; M5 finds LE_2020 "
        "+4.5 above the edu_2000-predicted line — over-performer, not the "
        "Soviet-style hollow-education signature."
    ),
    "Argentina": (
        "Sarmiento-era secular state schooling 1880s; Catholic colonizer "
        "but secular-state-priority — within-Catholic above-median crosser."
    ),
    "Costa Rica": (
        "Catholic colonizer + abolition of military 1948 redirected fiscal "
        "capacity to schooling and health; crossed despite low GDP at "
        "transition — a paper-style state-priority crosser."
    ),
    "Chile": (
        "Catholic colonizer with relatively early state schooling; "
        "post-1973 dictatorship preserved the schooling apparatus; crossed."
    ),
    "Uruguay": (
        "Catholic colonizer with strongest secular schooling tradition in "
        "Latin America (Vasco Battle 1900s); within-Catholic above-median crosser."
    ),
    "Mexico": (
        "Catholic colonizer; post-revolutionary state-led schooling 1920s+; "
        "crossed mid-20th century as the schooling expansion compounded."
    ),
    "Bangladesh": (
        "Protestant-coded (British India) but Britain did not settle and "
        "did not build mass schools in tropical East Bengal; the religion "
        "dummy is misleading here — the school-niche never flipped at "
        "scale before independence."
    ),
    "Pakistan": (
        "Protestant-coded; British administration built minimal schooling "
        "infrastructure outside Punjab/elite circles; convergence still "
        "incomplete — religion-of-colonizer is a noisy proxy for schools-built."
    ),
    "India": (
        "Protestant-coded; uneven regional schooling under British rule "
        "(Bengal high, Bihar low); state-level variance after 1947 carries "
        "the explanatory weight, not the colonial-religion dummy."
    ),
    "Sri Lanka": (
        "Protestant-coded; mission-school tradition + post-1948 free-education "
        "policy; crossed despite low GDP — paper-clean case for "
        "state-priority schooling outpacing income."
    ),
    "Malaysia": (
        "Protestant-coded; competing-priority schooling regime; crossed "
        "mid-trajectory as state expanded secondary."
    ),
    "Hong Kong, China": (
        "City-state, dropped in n=61 due to missing edu_1950 in WCDE "
        "panel; high avexpr ambiguous between colonial inheritance and "
        "post-1965 state-priority schooling."
    ),
    "Côte d'Ivoire": (
        "Catholic colonizer; dropped in n=61 due to missing edu_1950 in "
        "WCDE panel."
    ),
    "South Africa": (
        "Apartheid-era schooling was racially bifurcated — population mean "
        "edu_1950 hides the within-country structural fact; paper's "
        "mechanism predicts the post-1990 acceleration as the Black "
        "schooling expansion compounds."
    ),
    "Haiti": (
        "Catholic colonizer; post-1804 state never built mass schooling; "
        "lowest edu_1950 in the Americas; non-crosser predicted by mechanism."
    ),
    "Cambodia": "Not in AJR-64 base sample; flagged for reference only.",
    "Bolivia (Plurinational State of)": (
        "Catholic colonizer; Andean state with limited schooling reach "
        "outside La Paz; non-crosser at low-edu equilibrium."
    ),
    "Honduras": (
        "Catholic colonizer; state-fiscal capacity for schooling among the "
        "lowest in Latin America; non-crosser tracking edu_1950 fundamentals."
    ),
    "Egypt": (
        "Protestant-coded (British administration) but underlying Arab "
        "secular schooling state since Mehmet Ali; non-trivial cross-cutting "
        "case for the religion dummy."
    ),
    "Nigeria": (
        "Protestant-coded; British administration left mission-school "
        "asymmetry (Igbo high, North low); aggregate slowed by Northern "
        "low-edu-equilibrium."
    ),
    "Kenya": (
        "Protestant-coded; British settler+mission schools; modest "
        "expansion post-1963; recent crosser on the trajectory."
    ),
    "Ghana": (
        "Protestant-coded; mission-school tradition + Nkrumah-era state "
        "expansion; partial crosser on TFR."
    ),
}


def fmt(v, fmt_str=".1f", default="—"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    try:
        return format(v, fmt_str)
    except (TypeError, ValueError):
        return str(v)


def country_card(row):
    c = row["country"]
    cross_year = fmt(row["crossing_year"], ".0f", default="not crossed by 2022")
    edu_at_cross = fmt(row.get("edu_at_cross"), ".1f")
    narrative = NARRATIVE.get(c, "")
    if not narrative:
        # Generate a minimal default narrative
        if row["protestant"] == 1:
            religion = "Protestant"
        else:
            religion = "Catholic"
        if pd.isna(row["crossing_year"]):
            status = "non-crosser"
        else:
            status = f"crossed {int(row['crossing_year'])}"
        narrative = (f"{religion}-colonized; edu_1950={fmt(row['edu_1950'])}%; "
                     f"{status}.")

    lines = []
    lines.append(f"### {c}  *(colonizer: {row['colonizer']}, religion: {row['religion']})*")
    lines.append("")
    lines.append(f"- **Independence:** {fmt(row['independence_year'], '.0f')}    "
                 f"**avexpr (1985-95):** {fmt(row['avexpr'], '.2f')}    "
                 f"**logem4:** {fmt(row['logem4'], '.2f')}    "
                 f"**polity2 (2015):** {fmt(row['polity2_2015'], '.0f')}")
    lines.append(f"- **edu (lower-sec, 20-24, both sexes):** "
                 f"1900 cohort={fmt(row['edu_1900_cohort_lsec'])}, "
                 f"1925 cohort={fmt(row['edu_1925_cohort_lsec'])}, "
                 f"1950={fmt(row['edu_1950'])}, "
                 f"1975={fmt(row['edu_1975'])}, "
                 f"2000={fmt(row['edu_2000'])}, "
                 f"2020={fmt(row['edu_2020'])}")
    lines.append(f"- **GDP (log, const. 2015 USD):** "
                 f"1960={fmt(row['log_gdp_1960'], '.2f')}, "
                 f"2020={fmt(row['log_gdp_2020'], '.2f')}    "
                 f"**LE:** {fmt(row['le_1960'])}→{fmt(row['le_2020'])}    "
                 f"**TFR:** {fmt(row['tfr_1960'])}→{fmt(row['tfr_2020'])}    "
                 f"**U5MR:** {fmt(row['u5mr_1960'])}→{fmt(row['u5mr_2020'])}")
    lines.append(f"- **Convergence (TFR<{TFR_THRESHOLD} ∧ LE>{LE_THRESHOLD}):** "
                 f"{cross_year}"
                 + (f"; lower-sec at crossing = {edu_at_cross}%" if not pd.isna(row['crossing_year']) else ""))
    lines.append("")
    lines.append(f"*{narrative}*")
    lines.append("")
    return "\n".join(lines)


# ── Build the markdown report ─────────────────────────────────────────

CLUSTER_ORDER = [
    ("1_settler_protestant", "Settler-Protestant Crossers (USA, Canada, Australia, New Zealand)",
     "Founder-population schooling tradition arriving with the colonizer; "
     "all crossed early. The paper's framework predicts these as the "
     "earliest crossers because Protestant + state-priority schooling was "
     "embedded from settlement, not built post-independence."),
    ("3_protestant_crossers", "Protestant-Colonized Crossers (excluding settler colonies)",
     "Protestant colonial heritage + post-independence state schooling "
     "produced the convergence. Within this cluster, Sri Lanka is the "
     "paper-clean case where state-priority schooling pulled outcomes "
     "ahead of GDP."),
    ("5_catholic_crossers", "Catholic-Colonized Crossers",
     "These cases falsify any simple religion→outcome story. Each had a "
     "post-independence secular-state schooling push (Argentine Sarmiento, "
     "Cuban literacy campaign, Costa Rican demilitarization, Mexican "
     "post-revolutionary schooling). The paper's mechanism — state-paced "
     "schooling regardless of colonial religion — predicts these crossers."),
    ("4_protestant_non_crossers", "Protestant-Colonized Non-Crossers",
     "These cases break the religion-of-colonizer dummy as a proxy. "
     "Britain did not settle and did not build mass schools in tropical "
     "South Asia or Sub-Saharan Africa; the religion variable encodes a "
     "schools-built variable that wasn't actually delivered."),
    ("6_catholic_non_crossers", "Catholic-Colonized Non-Crossers",
     "The default trajectory predicted by the paper for Catholic-colonized + "
     "no secular-state-priority push: low-edu equilibrium, slow convergence."),
    ("2_small_islands_or_city_states", "Small Islands and City-States",
     "Anomalous in IV regressions due to small population, post-1965 "
     "trajectory dominated by state policy choices. avexpr_1985-95 here "
     "is post-development, not pre-determined."),
    ("7_other", "Other"),
]

doc = []
doc.append("# AJR n=61 Country Profiles")
doc.append("")
doc.append(f"Generated by `scripts/robustness/ajr_n61_country_profiles.py` "
           f"from `data/ajr2001/ajr_n61_country_table.csv`.")
doc.append("")
doc.append(f"**Sample:** {len(df)} of the 64 AJR (2001) base-sample colonies "
           f"(3 dropped on merge: Côte d'Ivoire, Ethiopia, Hong Kong). "
           f"Convergence threshold: TFR < {TFR_THRESHOLD} ∧ LE > {LE_THRESHOLD} "
           "(1960 US values).")
doc.append("")
n_cross = int(df["crossing_year"].notna().sum())
doc.append(f"**Crossed by 2022:** {n_cross} of {len(df)} "
           f"(Protestant: {int(((df.protestant==1) & df.crossing_year.notna()).sum())} of "
           f"{int((df.protestant==1).sum())}; "
           f"Catholic: {int(((df.protestant==0) & df.crossing_year.notna()).sum())} of "
           f"{int((df.protestant==0).sum())}).")
doc.append("")
doc.append("---")
doc.append("")

for cluster_id, *rest in CLUSTER_ORDER:
    title = rest[0]
    intro = rest[1] if len(rest) > 1 else ""
    sub = df[df["cluster"] == cluster_id].sort_values("country")
    if len(sub) == 0:
        continue
    doc.append(f"## {title}")
    doc.append("")
    if intro:
        doc.append(intro)
        doc.append("")
    doc.append(f"*n = {len(sub)} countries*")
    doc.append("")
    for _, row in sub.iterrows():
        doc.append(country_card(row))
    doc.append("---")
    doc.append("")

with open(OUT_PATH, "w") as f:
    f.write("\n".join(doc))

print(f"Wrote {OUT_PATH}")
print(f"Country count by cluster:")
for cid, *rest in CLUSTER_ORDER:
    n = (df["cluster"] == cid).sum()
    if n > 0:
        print(f"  {cid:<35}  n = {n}")
