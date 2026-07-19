# External Data Provenance (Pre-1950 Outcomes)

This document records what each pre-1950 outcome series is, where it
came from, when it was downloaded, and what its quality limitations are.
The longrun test reads these files. Every change to the inputs MUST
be reflected here.

## Tier A: HMD core (mortality.org)

Source: Human Mortality Database, https://www.mortality.org/
License: HMD Terms of Use; redistribution requires acknowledgment.
Native format: country life tables (1×1 cohort and period).

### File: `hmd/e0_1880_1950.csv`
- Indicator: e0 (life expectancy at birth, both sexes)
- Years: 1880–1950 (interpolated to 5-year grid 1880, 1885, ...)
- Countries: see header row
- Sha256: `5ad735b127f331caf2e08894cc313d8102b94ca98e8c98bc5b9122b86bf50f1d`
- Downloaded: `2026-05-29`

### File: `hmd/u5mr_1880_1950.csv`
- Indicator: under-5 mortality (sum_{0..4} q_x × 1000), both sexes
- Years: 1880–1950 (5-year grid)
- Countries: see header row
- Sha256: `3231d262f07b0370a9ebdc50085d104f475a853d09df510c628cf0595125eb57`
- Downloaded: `2026-05-29`

### Per-country quality tier (HMD)

Of the 21 first-mover countries we extracted, **12 have continuous
1880–1950 coverage** in HMD period life tables. These form the within-era
headline panel:

  Belgium, Denmark, England and Wales, Finland, France, Iceland,
  Italy, Netherlands, Norway, Scotland, Sweden, Switzerland.

The remaining 9 enter HMD later, typically because national vital
registration rolled out later:

  Australia (~1921), Austria (~1947), Canada (~1921), Germany (~1956),
  Hungary (~1950), Japan (~1947), New Zealand (~1948), Spain (~1908),
  USA (~1933).

These countries appear in the CSV with NaN for missing pre-coverage
years; the loader masks them out at within-mode (T+25 ≤ 1945) sample
construction. They contribute to the forward-mode test (training T as
late as 1940) where their HMD years overlap the training window.

## Tier B: OWID long-run series (republished Gapminder/Riley/Zijdeman/HMD/UN-WPP)

Source: Our World in Data, https://ourworldindata.org/grapher/
License: CC BY 4.0.
Native format: long CSV — `entity, code, year, <indicator>` (one row per country-year).
Underlying sources for OWID's long-run LE/CMR series: Riley (2005), Zijdeman et al.
(2015), HMD, UN WPP.

The directory is named `gapminder/` for historical reasons (Gapminder is the
best-known source for these long-run series); files are fetched from OWID
because OWID republishes the merged historical series with stable URLs.

### File: `gapminder/life_expectancy_1800_2020.csv`
- Indicator: life expectancy at birth (years), period
- Slug: `life-expectancy` (OWID grapher)
- Years: 1543–2023 where available (annual)
- Sha256: `6336e80067d72cd910e5b03a4150bbb203fb8f22c6a9c0c43e91aba07038720b`
- Downloaded: `2026-05-29`

### File: `gapminder/child_mortality_1800_2020.csv`
- Indicator: child mortality (under-5), probability of dying before age 5
- **Units: PERCENT** (per 100 live births). The WB / paper convention is per
  1000; the loader multiplies by 10 to convert.
- Slug: `child-mortality` (OWID grapher)
- Years: 1751–2023 where available (annual)
- Sha256: `d8ede251e65ea8325f9520d9223b8abe8c2c592adea5deb9f21dc18115092b07`
- Downloaded: `2026-05-29`

### File: `gapminder/tfr_1800_2020.csv`
- Indicator: total fertility rate (births per woman), complete Gapminder
- Slug: `fertility-rate-complete-gapminder` (OWID grapher)
- Years: 1891–2023 where available (annual)
- Sha256: `c4398f67298b5e78f1fab29f39dcaf8df371752c05ea072b97e568f41bf0c997`
- Downloaded: `2026-05-29`

## Reconstruction caveats

Gapminder values pre-1950 for non-European countries are heavily
reconstructed (regional averages projected onto countries, or
expert-interpolated from sparse sources). They are appropriate for
broad pattern detection but NOT for fine-grained country-level claims.

HMD pre-1900 values are also reconstructed (life tables filled from
incomplete vital registration in some country-years) but are derived
from country-specific data, not regional fill.

The longrun headline is reported on HMD core only. Gapminder-extended
results are reported as sensitivity, never as primary.
