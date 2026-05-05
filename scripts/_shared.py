"""
_shared.py
==========
Shared constants, country mapping, and data loading utilities.

Panel construction and regression utilities live in residualization/_shared.py.
"""

import hashlib
import json
import os
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PROC = os.path.join(REPO_ROOT, "wcde", "data", "processed")
DATA = os.path.join(REPO_ROOT, "data")
CHECKIN = os.path.join(REPO_ROOT, "checkin")
_CACHE_DIR = os.path.join(REPO_ROOT, ".cache", "panels")


def _cache_key(name, files):
    parts = [name]
    for f in files:
        try:
            st = os.stat(f)
            parts.append(f"{f}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append(f"{f}:missing")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _cached(name, files, compute):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{_cache_key(name, files)}.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass
    result = compute()
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return result

# ── Development thresholds (1960 USA benchmarks) ───────────────────
TFR_THRESHOLD = 3.65          # 1960 USA fertility rate
LE_THRESHOLD = 69.8           # 1960 USA life expectancy (years)

# ── Oil states (excluded from floor tests) ─────────────────────────
OIL_STATES = [
    "Qatar", "United Arab Emirates", "Kuwait",
    "Saudi Arabia", "Oman", "Bahrain",
]

WCDE_AGGREGATES = {
    "Africa", "Asia", "Europe", "World", "Oceania", "Caribbean",
    "Central America", "South America",
    "Latin America and the Caribbean",
    "Central Asia", "Eastern Africa", "Eastern Asia", "Eastern Europe",
    "Northern Africa", "Northern America", "Northern Europe",
    "Southern Africa", "Southern Asia", "Southern Europe",
    "Western Africa", "Western Asia", "Western Europe",
    "Middle Africa", "South-Eastern Asia",
    "Melanesia", "Micronesia", "Polynesia",
    "Less developed regions", "More developed regions",
    "Least developed countries",
    "Australia and New Zealand", "Channel Islands", "Sub-Saharan Africa",
    "South-Central Asia",
}

# Everything to exclude from country-level analysis (185 countries remain)
REGIONS = WCDE_AGGREGATES

# ── Canonical country standardization ──────────────────────────────────────
# Every country name appearing in any data file (WCDE v3, World Bank WDI,
# Gapminder, Polity, etc.) must standardize to exactly one canonical form.
# Use standardize_country_name() at every boundary where data is loaded or
# joined. Any name that does not standardize is a bug — fix it by adding an
# alias here, never by dropping the row silently.
#
# Canonical form: short lowercase common name (e.g. "south korea", "russia",
# "dr congo"). Matches the register used in the paper.

import re as _re
import unicodedata as _ud


def _normalize_key(name):
    """Lowercase, strip accents, collapse whitespace, drop trailing markers.

    Used only for alias lookup. Not the canonical form.
    """
    if name is None:
        return ""
    s = str(name)
    # Fix common mojibake (UTF-8 read as Latin-1) before normalizing.
    s = s.replace("\u00a0", " ")            # non-breaking space
    s = s.replace("¬†", " ")                # mojibake NBSP
    s = s.replace("√¥", "o").replace("√®", "e").replace("√©", "e")
    s = _ud.normalize("NFKD", s)
    s = "".join(c for c in s if not _ud.combining(c))
    s = s.lower().strip()
    s = s.rstrip("*").strip()
    s = _re.sub(r"\s+", " ", s)
    return s


# Canonical → list of every known variant (raw, pre-normalization).
# Keep this sorted alphabetically by canonical for reviewability.
# When adding a new data source, run verify_country_names.py; any name it
# reports as UNKNOWN goes in the appropriate entry here.
_CANONICAL_ALIASES = {
    "afghanistan": ["Afghanistan"],
    "albania": ["Albania"],
    "algeria": ["Algeria"],
    "andorra": ["Andorra"],
    "angola": ["Angola"],
    "antigua and barbuda": ["Antigua and Barbuda"],
    "argentina": ["Argentina"],
    "armenia": ["Armenia"],
    "australia": ["Australia"],
    "austria": ["Austria"],
    "azerbaijan": ["Azerbaijan"],
    "bahamas": ["Bahamas", "Bahamas, The", "The Bahamas"],
    "bahrain": ["Bahrain"],
    "bangladesh": ["Bangladesh"],
    "barbados": ["Barbados"],
    "belarus": ["Belarus"],
    "belgium": ["Belgium"],
    "belize": ["Belize"],
    "benin": ["Benin"],
    "bhutan": ["Bhutan"],
    "bolivia": ["Bolivia", "Bolivia (Plurinational State of)"],
    "bosnia and herzegovina": ["Bosnia and Herzegovina", "Bosnia & Herzegovina"],
    "botswana": ["Botswana"],
    "brazil": ["Brazil"],
    "brunei": ["Brunei", "Brunei Darussalam"],
    "bulgaria": ["Bulgaria"],
    "burkina faso": ["Burkina Faso"],
    "burundi": ["Burundi"],
    "cambodia": ["Cambodia"],
    "cameroon": ["Cameroon"],
    "canada": ["Canada"],
    "cape verde": ["Cape Verde", "Cabo Verde"],
    "central african republic": ["Central African Republic", "CAR"],
    "chad": ["Chad"],
    "chile": ["Chile"],
    "china": ["China"],
    "colombia": ["Colombia"],
    "comoros": ["Comoros"],
    "congo": ["Congo", "Congo, Rep.", "Congo (Brazzaville)", "Republic of the Congo"],
    "costa rica": ["Costa Rica"],
    "cote d'ivoire": ["Cote d'Ivoire", "Côte d'Ivoire", "Ivory Coast", "CoteDIvoire", "Cote dIvoire"],
    "croatia": ["Croatia"],
    "cuba": ["Cuba"],
    "cyprus": ["Cyprus"],
    "czech republic": ["Czech Republic", "Czechia"],
    "denmark": ["Denmark"],
    "djibouti": ["Djibouti"],
    "dominica": ["Dominica"],
    "dominican republic": ["Dominican Republic", "Dominican Rep."],
    "dr congo": [
        "Democratic Republic of the Congo", "Congo, Dem. Rep.",
        "Congo (Kinshasa)", "Congo, (Kinshasa)", "DRC", "Zaire",
        "DR Congo", "Dem. Rep. of the Congo",
    ],
    "ecuador": ["Ecuador"],
    "egypt": ["Egypt", "Egypt, Arab Rep.", "Arab Republic of Egypt"],
    "el salvador": ["El Salvador"],
    "equatorial guinea": ["Equatorial Guinea"],
    "eritrea": ["Eritrea"],
    "estonia": ["Estonia"],
    "ethiopia": ["Ethiopia"],
    "fiji": ["Fiji"],
    "finland": ["Finland"],
    "france": ["France"],
    "gabon": ["Gabon"],
    "gambia": ["Gambia", "Gambia, The", "The Gambia"],
    "georgia": ["Georgia"],
    "germany": ["Germany"],
    "ghana": ["Ghana"],
    "greece": ["Greece"],
    "grenada": ["Grenada"],
    "guatemala": ["Guatemala"],
    "guinea": ["Guinea"],
    "guinea-bissau": ["Guinea-Bissau", "Guinea Bissau"],
    "guyana": ["Guyana"],
    "haiti": ["Haiti"],
    "honduras": ["Honduras"],
    "hungary": ["Hungary"],
    "iceland": ["Iceland"],
    "india": ["India"],
    "indonesia": ["Indonesia"],
    "iran": [
        "Iran", "Iran (Islamic Republic of)", "Iran, Islamic Rep.",
        "Iran, Islamic Republic of", "Islamic Republic of Iran",
    ],
    "iraq": ["Iraq"],
    "ireland": ["Ireland"],
    "israel": ["Israel"],
    "italy": ["Italy"],
    "jamaica": ["Jamaica"],
    "japan": ["Japan"],
    "jordan": ["Jordan"],
    "kazakhstan": ["Kazakhstan"],
    "kenya": ["Kenya"],
    "kiribati": ["Kiribati"],
    "kuwait": ["Kuwait"],
    "kyrgyzstan": ["Kyrgyzstan", "Kyrgyz Republic"],
    "laos": [
        "Laos", "Lao", "Lao PDR", "Lao People's Democratic Republic",
    ],
    "latvia": ["Latvia"],
    "lebanon": ["Lebanon"],
    "lesotho": ["Lesotho"],
    "liberia": ["Liberia"],
    "libya": ["Libya", "Libyan Arab Jamahiriya"],
    "liechtenstein": ["Liechtenstein"],
    "lithuania": ["Lithuania"],
    "luxembourg": ["Luxembourg"],
    "macedonia": [
        "Macedonia", "Macedonia, Republic of",
        "North Macedonia", "The former Yugoslav Republic of Macedonia",
    ],
    "madagascar": ["Madagascar"],
    "malawi": ["Malawi"],
    "malaysia": ["Malaysia"],
    "maldives": ["Maldives"],
    "mali": ["Mali"],
    "malta": ["Malta"],
    "marshall islands": ["Marshall Islands"],
    "mauritania": ["Mauritania"],
    "mauritius": ["Mauritius"],
    "mexico": ["Mexico"],
    "micronesia": [
        "Micronesia", "Micronesia (Federated States of)",
        "Micronesia, Fed. Sts.", "Federated States of Micronesia",
    ],
    "moldova": ["Moldova", "Republic of Moldova"],
    "monaco": ["Monaco"],
    "mongolia": ["Mongolia"],
    "montenegro": ["Montenegro"],
    "morocco": ["Morocco"],
    "mozambique": ["Mozambique"],
    "myanmar": ["Myanmar", "Burma"],
    "namibia": ["Namibia"],
    "nauru": ["Nauru"],
    "nepal": ["Nepal"],
    "netherlands": ["Netherlands", "The Netherlands"],
    "new zealand": ["New Zealand"],
    "nicaragua": ["Nicaragua"],
    "niger": ["Niger"],
    "nigeria": ["Nigeria"],
    "north korea": [
        "Democratic People's Republic of Korea",
        "Korea, Dem. People's Rep.", "North Korea", "DPRK",
    ],
    "norway": ["Norway"],
    "oman": ["Oman"],
    "pakistan": ["Pakistan"],
    "palau": ["Palau"],
    "panama": ["Panama"],
    "papua new guinea": ["Papua New Guinea"],
    "paraguay": ["Paraguay"],
    "peru": ["Peru"],
    "philippines": ["Philippines"],
    "poland": ["Poland"],
    "portugal": ["Portugal"],
    "qatar": ["Qatar"],
    "romania": ["Romania"],
    "russia": ["Russia", "Russian Federation"],
    "rwanda": ["Rwanda"],
    "saint kitts and nevis": [
        "Saint Kitts and Nevis", "St. Kitts and Nevis", "St Kitts and Nevis",
    ],
    "saint lucia": ["Saint Lucia", "St. Lucia", "St Lucia"],
    "saint vincent and the grenadines": [
        "Saint Vincent and the Grenadines", "St. Vincent and the Grenadines",
        "St Vincent and the Grenadines",
    ],
    "samoa": ["Samoa"],
    "san marino": ["San Marino"],
    "sao tome and principe": [
        "Sao Tome and Principe", "São Tomé and Príncipe",
        "Sao Tome & Principe",
    ],
    "saudi arabia": ["Saudi Arabia"],
    "senegal": ["Senegal"],
    "serbia": ["Serbia"],
    "seychelles": ["Seychelles"],
    "sierra leone": ["Sierra Leone"],
    "singapore": ["Singapore"],
    "slovakia": ["Slovakia", "Slovak Republic"],
    "slovenia": ["Slovenia"],
    "solomon islands": ["Solomon Islands"],
    "somalia": ["Somalia", "Somalia, Fed. Rep."],
    "south africa": ["South Africa"],
    "south korea": [
        "Republic of Korea", "Korea, Rep.", "South Korea", "Korea (South)",
        "ROK",
    ],
    "south sudan": ["South Sudan"],
    "spain": ["Spain"],
    "sri lanka": ["Sri Lanka"],
    "sudan": ["Sudan"],
    "suriname": ["Suriname"],
    "swaziland": ["Swaziland", "Eswatini", "Eswatini, Kingdom of", "Kingdom of Eswatini"],
    "sweden": ["Sweden"],
    "switzerland": ["Switzerland"],
    "syria": [
        "Syria", "Syrian Arab Republic", "Syrian Arab Republic (Syria)",
    ],
    "taiwan": [
        "Taiwan", "Taiwan Province of China", "Taiwan, Republic of China",
        "Chinese Taipei", "Taiwan (ROC)",
    ],
    "tajikistan": ["Tajikistan"],
    "tanzania": [
        "Tanzania", "United Republic of Tanzania",
        "Tanzania, United Republic of",
    ],
    "thailand": ["Thailand"],
    "timor-leste": ["Timor-Leste", "East Timor", "Timor Leste"],
    "togo": ["Togo"],
    "tonga": ["Tonga"],
    "trinidad and tobago": ["Trinidad and Tobago", "Trinidad & Tobago"],
    "tunisia": ["Tunisia"],
    "turkey": ["Turkey", "Türkiye", "Turkiye"],
    "turkmenistan": ["Turkmenistan"],
    "tuvalu": ["Tuvalu"],
    "uganda": ["Uganda"],
    "ukraine": ["Ukraine"],
    "united arab emirates": ["United Arab Emirates", "UAE"],
    "united kingdom": [
        "United Kingdom",
        "United Kingdom of Great Britain and Northern Ireland",
        "UK", "Great Britain", "Britain",
    ],
    "united states": [
        "United States", "United States of America", "USA", "U.S.", "US",
    ],
    "uruguay": ["Uruguay"],
    "uzbekistan": ["Uzbekistan"],
    "vanuatu": ["Vanuatu"],
    "venezuela": [
        "Venezuela", "Venezuela (Bolivarian Republic of)", "Venezuela, RB",
        "Venezuela (Bolivarian Republic)",
    ],
    "vietnam": ["Vietnam", "Viet Nam"],
    "yemen": ["Yemen", "Yemen, Rep."],
    "zambia": ["Zambia"],
    "zimbabwe": ["Zimbabwe"],
}

# Non-sovereign territories and disputed regions. These appear in some data
# files but are excluded from the 185-country panel. Knowing them lets the
# verifier distinguish "legitimate territory" from "unknown name → bug".
_TERRITORY_ALIASES = {
    "american samoa": ["American Samoa"],
    "aruba": ["Aruba"],
    "bermuda": ["Bermuda"],
    "british virgin islands": ["British Virgin Islands"],
    "cayman islands": ["Cayman Islands"],
    "curacao": ["Curacao", "Curaçao"],
    "faroe islands": ["Faroe Islands"],
    "french guiana": ["French Guiana"],
    "french polynesia": ["French Polynesia"],
    "gibraltar": ["Gibraltar"],
    "greenland": ["Greenland"],
    "guadeloupe": ["Guadeloupe"],
    "guam": ["Guam"],
    "holy see": ["Holy See", "Vatican", "Vatican City"],
    "hong kong": [
        "Hong Kong", "Hong Kong SAR", "Hong Kong SAR, China",
        "Hong Kong S.A.R., China", "Hong Kong S.A.R. of China",
        "Hong Kong Special Administrative Region of China",
        "Hong Kong, SAR China", "Hong Kong, China",
        "China, Hong Kong Special Administrative Region",
    ],
    "isle of man": ["Isle of Man"],
    "kosovo": ["Kosovo"],
    "macao": [
        "Macao", "Macao SAR, China",
        "Macao Special Administrative Region of China",
        "Macao, SAR China", "Macau",
        "China, Macao Special Administrative Region",
    ],
    "martinique": ["Martinique"],
    "mayotte": ["Mayotte"],
    "new caledonia": ["New Caledonia"],
    "north cyprus": ["North Cyprus", "Northern Cyprus"],
    "northern mariana islands": ["Northern Mariana Islands"],
    "palestine": [
        "Palestine", "Palestinian Territories", "Palestinian Territory",
        "Occupied Palestinian Territory", "State of Palestine",
        "West Bank and Gaza",
    ],
    "puerto rico": ["Puerto Rico", "Puerto Rico (US)"],
    "reunion": ["Reunion", "Réunion"],
    "serbia and montenegro": ["Serbia and Montenegro"],
    "sint maarten": ["Sint Maarten (Dutch part)", "Sint Maarten"],
    "somaliland": ["Somaliland Region", "Somaliland region", "Somaliland"],
    "st. martin": ["St. Martin (French part)", "St. Martin", "Saint Martin"],
    "turks and caicos islands": ["Turks and Caicos Islands"],
    "us virgin islands": [
        "United States Virgin Islands", "Virgin Islands (U.S.)",
        "US Virgin Islands", "U.S. Virgin Islands",
    ],
    "western sahara": ["Western Sahara"],
}

# Regional / economic aggregates that appear in WB, WCDE, Gapminder, etc.
# These are legitimate but never belong in the 185-country panel.
_AGGREGATE_ALIASES = {
    "_agg_" + k.lower().replace(" ", "_"): [k] for k in (
        list(WCDE_AGGREGATES) + [
            "Africa Eastern and Southern", "Africa Western and Central",
            "Arab World", "Caribbean small states",
            "Central Europe and the Baltics", "Early-demographic dividend",
            "East Asia & Pacific", "East Asia & Pacific (IDA & IBRD countries)",
            "East Asia & Pacific (excluding high income)", "Euro area",
            "Europe & Central Asia",
            "Europe & Central Asia (IDA & IBRD countries)",
            "Europe & Central Asia (excluding high income)", "European Union",
            "Fragile and conflict affected situations",
            "Heavily indebted poor countries (HIPC)", "High income",
            "IBRD only", "IDA & IBRD total", "IDA blend", "IDA only",
            "IDA total", "Late-demographic dividend",
            "Latin America & Caribbean",
            "Latin America & Caribbean (excluding high income)",
            "Latin America & the Caribbean (IDA & IBRD countries)",
            "Least developed countries: UN classification",
            "North America",  # WDI aggregate + UN sub-region
            "Low & middle income", "Low income", "Lower middle income",
            "Middle East & North Africa",
            "Middle East & North Africa (IDA & IBRD countries)",
            "Middle East & North Africa (excluding high income)",
            "Middle East, North Africa, Afghanistan & Pakistan",
            "Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD)",
            "Middle East, North Africa, Afghanistan & Pakistan (excluding high income)",
            "Middle income", "Not classified", "OECD members",
            "Other small states", "Pacific island small states",
            "Post-demographic dividend", "Pre-demographic dividend",
            "Small states", "South Asia", "South Asia (IDA & IBRD)",
            "Sub-Saharan Africa (IDA & IBRD countries)",
            "Sub-Saharan Africa (excluding high income)",
            "Upper middle income",
        ]
    )
}


def _build_alias_lookup():
    """Flatten canonical → aliases into a single normalized-key → canonical map."""
    lookup = {}
    collisions = []
    for canonical, aliases in _CANONICAL_ALIASES.items():
        lookup[_normalize_key(canonical)] = canonical
        for a in aliases:
            k = _normalize_key(a)
            if k in lookup and lookup[k] != canonical:
                collisions.append((a, lookup[k], canonical))
            lookup[k] = canonical
    for canonical, aliases in _TERRITORY_ALIASES.items():
        lookup[_normalize_key(canonical)] = canonical
        for a in aliases:
            k = _normalize_key(a)
            if k in lookup and lookup[k] != canonical:
                collisions.append((a, lookup[k], canonical))
            lookup[k] = canonical
    for canonical, aliases in _AGGREGATE_ALIASES.items():
        for a in aliases:
            lookup[_normalize_key(a)] = canonical
    if collisions:
        raise RuntimeError(
            "Alias table has collisions (same name maps to multiple canonicals): "
            + "; ".join(f"{a!r} → {x!r} and {y!r}" for a, x, y in collisions)
        )
    return lookup


_ALIAS_LOOKUP = _build_alias_lookup()

# Public canonical sets. Import these instead of rebuilding locally.
CANONICAL_COUNTRIES = frozenset(_CANONICAL_ALIASES.keys())
CANONICAL_TERRITORIES = frozenset(_TERRITORY_ALIASES.keys())
CANONICAL_AGGREGATES = frozenset(_AGGREGATE_ALIASES.keys())


def standardize_country_name(name, strict=False):
    """Map any known country-name variant to its canonical form.

    Returns the canonical lowercase short form (e.g. "south korea"), or None
    if the name is not recognized. In strict mode, raises KeyError instead
    of returning None — use this when a mismatch should stop the pipeline.

    Accepts sovereign countries, non-sovereign territories, and regional
    aggregates. Use classify_country_name() to distinguish these categories.
    """
    k = _normalize_key(name)
    if not k:
        if strict:
            raise KeyError(f"empty country name: {name!r}")
        return None
    canonical = _ALIAS_LOOKUP.get(k)
    if canonical is None and strict:
        raise KeyError(
            f"unknown country name {name!r} (normalized {k!r}); "
            "add an alias in scripts/_shared.py _CANONICAL_ALIASES"
        )
    return canonical


def classify_country_name(name):
    """Return one of {'country', 'territory', 'aggregate', 'unknown'}."""
    canonical = standardize_country_name(name)
    if canonical is None:
        return "unknown"
    if canonical in CANONICAL_COUNTRIES:
        return "country"
    if canonical in CANONICAL_TERRITORIES:
        return "territory"
    if canonical in CANONICAL_AGGREGATES:
        return "aggregate"
    return "unknown"


def is_sovereign_country(name):
    """True iff name refers to a sovereign country in the canonical panel."""
    return classify_country_name(name) == "country"


def standardize_country_series(series, strict=False):
    """Apply standardize_country_name to a pandas Series. Returns Series."""
    return series.map(lambda x: standardize_country_name(x, strict=strict))

NAME_MAP = {
    "Viet Nam": "vietnam",
    "Iran (Islamic Republic of)": "iran",
    "Republic of Korea": "south korea",
    "United States of America": "united states",
    "United Kingdom of Great Britain and Northern Ireland": "united kingdom",
    "Russian Federation": "russia",
    "United Republic of Tanzania": "tanzania",
    "Democratic Republic of the Congo": "congo, dem. rep.",
    "Congo": "congo, rep.",
    "Bolivia (Plurinational State of)": "bolivia",
    "Venezuela (Bolivarian Republic of)": "venezuela",
    "Republic of Moldova": "moldova",
    "Syrian Arab Republic": "syria",
    "Taiwan Province of China": "taiwan",
    "Lao People's Democratic Republic": "laos",
    "Türkiye": "turkey",
    "Eswatini": "swaziland",
    "Cabo Verde": "cape verde",
    "Czechia": "czech republic",
    "North Macedonia": "macedonia",
    "Libyan Arab Jamahiriya": "libya",
    "Hong Kong Special Administrative Region of China": "hong kong",
    "Macao Special Administrative Region of China": "macao",
    "Côte d'Ivoire": "cote d'ivoire",
    "Democratic People's Republic of Korea": "north korea",
    "Myanmar": "myanmar",
    "Timor-Leste": "timor-leste",
}

# ── Case study countries ───────────────────────────────────────────
# WCDE names → display names for the six case-study countries
CASE_STUDY_NAMES = {
    "Taiwan Province of China": "Taiwan",
    "Republic of Korea":        "South Korea",
    "Cuba":                     "Cuba",
    "Bangladesh":               "Bangladesh",
    "Sri Lanka":                "Sri Lanka",
    "China":                    "China",
}

# Combined development crossing (TFR < 3.65 AND LE > 69.8) — from Table 4
# Taiwan: WCDE 5-year data shows both crossed by 1970 (TFR=3.47, LE=70.0)
CASE_CROSSING_COMBINED = {
    "Taiwan Province of China": 1970,
    "Republic of Korea":        1987,
    "Cuba":                     1974,
    "Bangladesh":               2014,
    "Sri Lanka":                1993,
    "China":                    1994,
}

# TFR-only crossing (TFR < 3.65) — from Table 4
CASE_CROSSING_TFR = {
    "Taiwan Province of China": 1970,
    "Republic of Korea":        1975,
    "Cuba":                     1972,
    "Bangladesh":               1995,
    "Sri Lanka":                1981,
    "China":                    1975,
}

# LE-only crossing (LE > 69.8) — same as combined for these countries
CASE_CROSSING_LE = dict(CASE_CROSSING_COMBINED)

# ── World Bank region names (for excluding from WDI CSVs) ─────────
WB_REGIONS = {
    "africa eastern and southern", "africa western and central", "arab world",
    "caribbean small states", "central europe and the baltics",
    "early-demographic dividend", "east asia & pacific",
    "east asia & pacific (excluding high income)",
    "east asia & pacific (ida & ibrd countries)", "euro area",
    "europe & central asia", "europe & central asia (excluding high income)",
    "europe & central asia (ida & ibrd countries)", "european union",
    "fragile and conflict affected situations",
    "heavily indebted poor countries (hipc)", "high income",
    "ida & ibrd total", "ida blend", "ida only", "ida total", "ibrd only",
    "late-demographic dividend", "latin america & caribbean",
    "latin america & caribbean (excluding high income)",
    "latin america & the caribbean (ida & ibrd countries)",
    "least developed countries: un classification", "low & middle income",
    "low income", "lower middle income", "middle east & north africa",
    "middle east & north africa (excluding high income)",
    "middle east & north africa (ida & ibrd countries)",
    "middle east, north africa, afghanistan & pakistan",
    "middle east, north africa, afghanistan & pakistan (ida & ibrd)",
    "middle east, north africa, afghanistan & pakistan (excluding high income)",
    "middle income", "north america", "not classified", "oecd members",
    "other small states", "pacific island small states",
    "post-demographic dividend", "pre-demographic dividend", "small states",
    "south asia", "south asia (ida & ibrd)", "sub-saharan africa",
    "sub-saharan africa (excluding high income)",
    "sub-saharan africa (ida & ibrd countries)", "upper middle income", "world",
}

# WB country names → WCDE country names (for population lookups)
WB_TO_WCDE = {
    "bahamas, the": "bahamas",
    "cabo verde": "cape verde",
    "congo, dem. rep.": "democratic republic of the congo",
    "congo, rep.": "congo",
    "czechia": "czech republic",
    "egypt, arab rep.": "egypt",
    "gambia, the": "gambia",
    "hong kong sar, china": "hong kong special administrative region of china",
    "iran, islamic rep.": "iran (islamic republic of)",
    "korea, dem. people's rep.": "democratic people's republic of korea",
    "korea, rep.": "republic of korea",
    "kyrgyz republic": "kyrgyzstan",
    "lao pdr": "lao people's democratic republic",
    "libya": "libyan arab jamahiriya",
    "macao sar, china": "macao special administrative region of china",
    "micronesia, fed. sts.": "micronesia (federated states of)",
    "moldova": "republic of moldova",
    "north macedonia": "the former yugoslav republic of macedonia",
    "puerto rico (us)": "puerto rico",
    "slovak republic": "slovakia",
    "somalia, fed. rep.": "somalia",
    "st. kitts and nevis": "saint kitts and nevis",
    "st. lucia": "saint lucia",
    "st. vincent and the grenadines": "saint vincent and the grenadines",
    "turkiye": "turkey",
    "united kingdom": "united kingdom of great britain and northern ireland",
    "united states": "united states of america",
    "venezuela, rb": "venezuela (bolivarian republic of)",
    "virgin islands (u.s.)": "united states virgin islands",
    "west bank and gaza": "state of palestine",
    "yemen, rep.": "yemen",
    "eswatini": "swaziland",
    "cote d'ivoire": "côte d'ivoire",
    "timor-leste": "timor-leste",
}


def load_wb(filename):
    """Load a World Bank WDI CSV (Country × Year wide format).

    Expands the index with every known alias for each country so that downstream
    lookups by any naming convention (WCDE long form, WB short form, common
    names) all resolve to the same row. Prevents silent data loss from join
    mismatches like "Republic of Korea" ↔ "Korea, Rep.".
    """
    csv_path = os.path.join(DATA, filename)

    def compute():
        df = pd.read_csv(csv_path)
        df["Country"] = df["Country"].str.lower()
        df = df.set_index("Country")
        df = df.apply(pd.to_numeric, errors="coerce")
        return add_canonical_aliases(df)

    return _cached(f"load_wb:{filename}", [csv_path], compute)


def add_canonical_aliases(df):
    """Add alias rows so every known name variant indexes the same data.

    For a DataFrame indexed by lowercased country name, ensures that a lookup
    by any known WCDE long form, WB short form, or common alias hits the same
    row. Call this on any country-indexed DataFrame loaded outside load_wb().
    """
    extras = {}
    for idx in list(df.index):
        canon = standardize_country_name(idx)
        if canon is None:
            continue
        variants = _CANONICAL_ALIASES.get(canon, []) + _TERRITORY_ALIASES.get(canon, [])
        for v in variants + [canon]:
            vl = v.lower()
            if vl == idx or vl in df.index or vl in extras:
                continue
            extras[vl] = df.loc[idx]
    if extras:
        extra_df = pd.DataFrame(list(extras.values()), index=list(extras.keys()))
        original_name = df.index.name
        df = pd.concat([df, extra_df])
        df.index.name = original_name
    return df


def load_wide_indicator(filename):
    """Load wide-format WB CSV, excluding WB region rows.

    Returns DataFrame: index=country (lowercase), columns=year strings.
    """
    df = pd.read_csv(os.path.join(DATA, filename))
    df["country"] = df["Country"].str.lower()
    df = df[~df["country"].isin(WB_REGIONS)]
    return df.set_index("country").drop(columns=["Country"])


def load_population_by_year():
    """Load WCDE population totals by country and year.

    Returns DataFrame: index=WB lowercase country name, columns=WCDE years,
    values=population in thousands.
    """
    pop_path = os.path.join(REPO_ROOT, "wcde", "data", "raw", "pop_both.csv")
    pop = pd.read_csv(pop_path)
    pop = pop[pop["scenario"] == 2]
    pop["country"] = pop["name"].str.lower()
    wcde_regions_lower = {r.lower() for r in REGIONS}
    pop = pop[~pop["country"].isin(wcde_regions_lower)]
    totals = pop.groupby(["country", "year"])["pop"].sum().unstack("year")
    wcde_to_wb = {v: k for k, v in WB_TO_WCDE.items()}
    totals.index = [wcde_to_wb.get(n, n) for n in totals.index]
    return totals


def load_education(filename="completion_both_long.csv"):
    """Load WCDE education data, excluding regional aggregates."""
    csv_path = os.path.join(PROC, filename)

    def compute():
        edu = pd.read_csv(csv_path)
        edu = edu[~edu["country"].isin(REGIONS)].copy()
        return edu

    return _cached(f"load_education:{filename}", [csv_path], compute)


def get_wb_val(df, wcde_name, year):
    """Look up a World Bank value for a WCDE country name.

    Resolution order:
      1. Raw lowercase (handles cases where WCDE and WB already agree).
      2. NAME_MAP lookup (legacy WCDE → WB-short mapping).
      3. Canonical standardization on both sides (catches every known variant
         — South Korea, Turkey, Iran, Slovakia, etc. that NAME_MAP misses).
    """
    for k in (wcde_name.lower(), NAME_MAP.get(wcde_name, wcde_name).lower()):
        if k in df.index:
            try:
                v = float(df.loc[k, str(year)])
                return v if not np.isnan(v) else np.nan
            except (KeyError, ValueError):
                pass

    canonical = standardize_country_name(wcde_name)
    if canonical is not None:
        if not hasattr(df, "_canonical_index"):
            idx_map = {}
            for raw in df.index:
                c = standardize_country_name(raw)
                if c is not None and c not in idx_map:
                    idx_map[c] = raw
            object.__setattr__(df, "_canonical_index", idx_map)
        raw = df._canonical_index.get(canonical)
        if raw is not None:
            try:
                v = float(df.loc[raw, str(year)])
                return v if not np.isnan(v) else np.nan
            except (KeyError, ValueError):
                pass
    return np.nan


def interpolate_to_annual(edu_df, col_name):
    """Interpolate 5-year WCDE data to annual values, per country.

    Vectorized pivot + row-wise linear interpolate. Preserves per-country
    year ranges: years outside a country's observed span remain NaN and
    are dropped from the returned Series, matching the old per-country loop.
    """
    wide = edu_df.pivot_table(
        index="country", columns="year", values=col_name, aggfunc="first"
    ).sort_index(axis=1)
    if wide.empty:
        return {}
    full_idx = range(int(wide.columns.min()), int(wide.columns.max()) + 1)
    wide = wide.reindex(columns=full_idx)
    wide = wide.interpolate(method="linear", axis=1, limit_area="inside")
    return {c: wide.loc[c].dropna() for c in wide.index}


# ── WCDE → WDI country name mapping ─────────────────────────────
# Maps WCDE v3 country names to World Bank WDI country names.
# Used by scripts that join education data (WCDE) with WDI indicators.
WCDE_TO_WDI = {
    "Bahamas": "Bahamas, The",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Cape Verde": "Cabo Verde",
    "Congo": "Congo, Rep.",
    "Curaçao": "Curacao",
    "Czech Republic": "Czechia",
    "Democratic People's Republic of Korea": "Korea, Dem. People's Rep.",
    "Democratic Republic of the Congo": "Congo, Dem. Rep.",
    "Egypt": "Egypt, Arab Rep.",
    "Gambia": "Gambia, The",
    "Hong Kong Special Administrative Region of China": "Hong Kong SAR, China",
    "Iran (Islamic Republic of)": "Iran, Islamic Rep.",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Lao People's Democratic Republic": "Lao PDR",
    "Libyan Arab Jamahiriya": "Libya",
    "Macao Special Administrative Region of China": "Macao SAR, China",
    "Micronesia (Federated States of)": "Micronesia, Fed. Sts.",
    "Occupied Palestinian Territory": "West Bank and Gaza",
    "Puerto Rico": "Puerto Rico (US)",
    "Republic of Korea": "Korea, Rep.",
    "Republic of Moldova": "Moldova",
    "Saint Lucia": "St. Lucia",
    "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",
    "Slovakia": "Slovak Republic",
    "Swaziland": "Eswatini",
    "The former Yugoslav Republic of Macedonia": "North Macedonia",
    "Turkey": "Turkiye",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "United Republic of Tanzania": "Tanzania",
    "United States of America": "United States",
    "United States Virgin Islands": "Virgin Islands (U.S.)",
    "Venezuela (Bolivarian Republic of)": "Venezuela, RB",
    "Yemen": "Yemen, Rep.",
}


def wcde_to_wdi(name):
    """Map a WCDE country name to its WDI equivalent; identity fallback."""
    return WCDE_TO_WDI.get(name, name)


def interpolate_wide_to_annual(df):
    """Interpolate a wide-format DataFrame from 5-year to annual resolution.

    Input:  index=country, columns=integer years (e.g. 1950, 1955, ...).
    Output: same shape but with every intervening year filled by linear
            interpolation.
    """
    dft = df.T
    dft.index = dft.index.astype(int)
    full_idx = range(dft.index.min(), dft.index.max() + 1)
    dft = dft.reindex(full_idx).interpolate(method="linear")
    return dft.T  # back to country × year


def completion_at_year(df_wide, country, year):
    """Interpolate a single value from a wide-format education CSV.

    Works with 5-year WCDE data: finds the bracketing years and linearly
    interpolates.  Returns np.nan if the country or year is out of range.
    """
    if country not in df_wide.index:
        return np.nan
    row = df_wide.loc[country].dropna()
    years = np.array([int(c) for c in row.index], dtype=float)
    vals = row.values.astype(float)
    if len(years) == 0 or year < years.min() or year > years.max():
        return np.nan
    return float(np.interp(year, years, vals))


def write_checkin(filename, data, script_path=None):
    """Write a checkin JSON file. Adds 'produced' date and 'script' path."""
    if script_path:
        data["script"] = script_path
    data["produced"] = str(pd.Timestamp.now().date())
    os.makedirs(CHECKIN, exist_ok=True)
    path = os.path.join(CHECKIN, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Checkin written to {path}")


# ── Formatting helpers ────────────────────────────────────────────

def fmt_r2(v):
    """Format an R² or float value for display: '0.456' or 'n/a'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.3f}"


# ── Standard data loading ─────────────────────────────────────────

def load_standard_outcomes():
    """Load GDP and the three standard WB outcome datasets.

    Returns dict with keys: gdp, le, tfr, u5mr (all wide-format DataFrames).
    """
    return {
        "gdp": load_wb("gdppercapita_us_inflation_adjusted.csv"),
        "le": load_wb("life_expectancy_years.csv"),
        "tfr": load_wb("children_per_woman_total_fertility.csv"),
        "u5mr": load_wb("child_mortality_u5.csv"),
    }


# ── FE regression with clustered standard errors ─────────────────

def fe_regression(df, x_cols, y_col, cluster_col="country"):
    """Country fixed effects via demeaning, with cluster-robust SEs.

    Parameters
    ----------
    df : DataFrame with columns for x_cols, y_col, and cluster_col.
    x_cols : list of predictor column names.
    y_col : outcome column name.
    cluster_col : column to cluster on (default 'country').

    Returns (statsmodels OLS result, n_obs, n_countries).
    """
    d = df.dropna(subset=x_cols + [y_col]).copy()
    for col in x_cols + [y_col]:
        d[col + "_dm"] = d.groupby(cluster_col)[col].transform(
            lambda x: x - x.mean()
        )
    X = d[[c + "_dm" for c in x_cols]]
    y = d[y_col + "_dm"]
    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": d[cluster_col]},
    )
    return model, len(d), d[cluster_col].nunique()

