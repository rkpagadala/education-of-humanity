# Econometric Battery — Comprehensive test suite

Twenty test files, each running a battery of related diagnostics on the
panel underlying *The Long Childhood*. Every test prints a numeric
result and a `[PASS]` or `[FAIL]` line to the console and writes a JSON
payload to `checkin/econometric_battery/<test_name>.json`.

The battery is comprehensive rather than minimal. It covers the standard
diagnostics any modern panel-econometrics handbook expects to see, and
adds those the paper does not run in the main text — Pesaran CD, panel
unit-root tests, cointegration, and Wooldridge xtserial for serial
correlation — so a reader who wants the full panel workup can find it
here.

## How to run

```bash
cd Human-Development-Prediction
python3 scripts/econometric_battery/run_all.py     # all 20
python3 scripts/econometric_battery/test_02_cross_section_dependence.py   # one
```

Each test takes ~5-60 seconds. `run_all` writes `REPORT.md` summarising
JSON outputs.

## Test inventory

| #  | Test                              | What it diagnoses                                     |
|----|-----------------------------------|-------------------------------------------------------|
| 01 | `stationarity.py`                 | Levin-Lin-Chu, IPS, Maddala-Wu, Pesaran CIPS, Hadri   |
| 02 | `cross_section_dependence.py`     | Pesaran CD, Friedman, Frees, BP-LM                    |
| 03 | `cointegration.py`                | Kao, Pedroni (panel/group), Westerlund ECM             |
| 04 | `serial_correlation.py`           | Wooldridge xtserial, Baltagi-Wu LBI, Born-Breitung    |
| 05 | `robust_se.py`                    | White, cluster, two-way cluster, Driscoll-Kraay, Conley|
| 06 | `exogeneity.py`                   | Wooldridge strict, Hausman, Dumitrescu-Hurlin, AR-IV   |
| 07 | `specification.py`                | RESET, polynomial, Hansen threshold, splines, terciles |
| 08 | `structural_breaks.py`            | Era β, Chow, Andrews-Quandt sup-F, Bai-Perron, CUSUM   |
| 09 | `influence.py`                    | Cook's D, DFBETAs, LOO, leverage, studentized residuals|
| 10 | `robust_regression.py`            | Huber, Tukey, LAD, Theil-Sen, trimmed, winsorized      |
| 11 | `multiple_testing.py`             | Bonferroni, Holm, BH, BY, Westfall-Young, Romano-Wolf  |
| 12 | `dynamic_panels.py`               | Anderson-Hsiao, Arellano-Bond, Hansen J, AR tests      |
| 13 | `modern_did.py`                   | Sun-Abraham, dCH, Borusyak-Spiess, Honest DiD          |
| 14 | `synthetic_control.py`            | Abadie-Diamond-Hainmueller for 7 canonical cases       |
| 15 | `cross_validation.py`             | k-fold, forward-chain, region/era/OECD holdout         |
| 16 | `functional_form.py`              | Log, sqrt, arcsine, logit, Box-Cox transformations     |
| 17 | `sample_composition.py`           | Balanced, by-era, by-region, by-GDP, by-start          |
| 18 | `measurement_error.py`            | Reliability, known-noise, IV (neighbor, Barro-Lee), Deming |
| 19 | `placebo.py`                      | CO₂ placebo, within-country shuffle, cohort mismatch   |
| 20 | `predictive_validity.py`          | OOS RMSE, Diebold-Mariano, Mizon-Richard encompassing  |

## What the battery is **not**

The battery is not the paper's identification strategy. The panel β is
descriptive — it documents the pattern across 185 countries over
1950–2015. Identification in *The Long Childhood* rests on three
sources of evidence the panel cannot supply on its own:

1. **Temporal ordering.** Parental education at T-25 precedes child
   education at T by biology, not by statistical assumption.
2. **Natural experiments.** Cambodia 1975–79, Sri Lanka 1945, Korea
   1953, the Asian Financial Crisis 1997 (paper §10).
3. **USSR falsification.** The negative case in which centralised
   schooling without literate-CT depth did not produce convergence
   (paper §11).

The twenty tests here are a panel-econometrics companion to those
three. They probe the panel; they do not carry the identification.

## Interpretation guide

- Tests labelled `[PASS]` indicate the headline result is robust to the
  specific concern the test addresses.
- Tests labelled `[FAIL]` mark specifications under which the headline
  result weakens — for example, out-of-sample Diebold–Mariano tests,
  or dynamic-panel GMM with a lagged dependent variable. Both passes
  and failures are reported in full so a reader can see the boundary
  of the result, not just its interior.
- Diagnostic tests (stationarity, cross-section dependence,
  cointegration, serial correlation) are descriptive: they characterise
  the structure of the panel. This panel is cross-sectionally dependent
  (CD rejects), non-stationary at the level of attainment series
  (LLC/Hadri reject one direction or the other), and serially
  correlated. These are structural features of long-horizon
  country-level macro panels, not specification defects.

## Outputs

- `checkin/econometric_battery/*.json` — one JSON per test
- `scripts/econometric_battery/REPORT.md` — auto-generated digest
- Console output — full diagnostic narrative

## Adding a new test

1. Create `test_NN_<name>.py` matching the existing pattern
2. Import shared utilities from `_battery`
3. Use `write_battery_checkin(name, payload)` to persist results
4. Add it to `TESTS` list in `run_all.py`
