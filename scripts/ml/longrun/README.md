# Longrun (Pre-Medicine) Test

Pre-1950 panel test of the education→development thesis. Severs the
medical-tech confound by testing the relationship in an era before
antibiotics (1940s), modern vaccines, and global health infrastructure.

## What this is

Twin findings:

- **Within-era (1885–1920):** Education at T predicts (LE, U5MR) at
  T+25 for outcomes that are all pre-1945. No modern medicine anywhere
  in the panel. Targets are the two MORTALITY outcomes only: pre-1945
  fertility (TFR) is too sparse in the historical sources to admit more
  than ~4 countries, and mortality is precisely what the medical-tech
  confound is about, so the test is strongest where the data is richest.
- **Forward (pre-1950 → 1960+):** Train on pre-1950, predict 1960–2015
  (LE, U5MR, TFR). Tests cross-regime extrapolation.

## Which number is the headline

The leakage-free, deterministic headline is the **country-FE OLS β** (the
paper's primary spec), produced by `panel_fe.py`:

- within-era LE β = +0.404 (cluster-SE 0.095); U5MR β = −2.977 (SE 0.636)
- forward LE β = +0.566; U5MR β = −3.464; TFR β ≈ 0

These are registered in `verify_the_long_childhood.py`
(`Longrun-Within/Forward-*`). They are closed-form, so bit-reproducible,
and carry no model-selection bias.

The **transformer R²** (in the checkin JSONs) is supporting evidence and
must be read with one caveat: like the canonical 1960+ panel runner
(`universal_transformer.train_one_fold` / `run_5fold`), the longrun
transformer selects each fold's best epoch on that fold's held-out rows
and reports the same rows' R². This is the repo's established convention —
using it here is deliberate, so the longrun transformer is procedurally
identical to the 1960+ panel and the two eras are directly comparable. The
mild in-sample optimism it introduces applies equally to both eras, so the
cross-era comparison is unaffected; only the absolute transformer R² should
be read as optimistic. Primary inference rests on the OLS β, not the
transformer R².

## What this is NOT

- Not a replacement for the 1960+ panel. The 1960+ measured panel
  remains the rigor backbone.
- Not a new universal-rigor claim. Pre-1950 outcomes are partly
  reconstructed and Europe-heavy.
- Not a wider-N study. The within-era HMD core panel that actually
  contributes samples is 15 first-mover countries (85 country-years);
  the Gapminder-extended tier adds more at lower data quality.

## Data tiers

- **Core (headline):** HMD direct downloads, ~22 first-mover countries
  with continuous e0 and q5 back to ~1880. Highest-quality pre-1950
  mortality data that exists.
- **Extended (sensitivity):** Gapminder fixed historical series, ~40
  countries. Documented per-country quality in `data/external/PROVENANCE.md`.

## Pipeline

1. `fetch_hmd.py` — manual one-time download from mortality.org; verifies checksums
2. `fetch_gapminder.py` — automated pinned-URL fetch
3. `data_loader_longrun.py` — assembles within / forward panels (parallel to `data_loader.py`)
4. `panel_fe.py` — OLS + GBM baseline runners
5. `run_job.py` dispatch — 4 new kinds: `transformer_longrun_within`,
   `transformer_longrun_forward`, `panel_fe_longrun_within`,
   `panel_fe_longrun_forward`
6. `aggregate_longrun.py` — produces `checkin/longrun_within.json` and
   `checkin/longrun_forward.json`
