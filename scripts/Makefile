# scripts/Makefile — build checkin JSONs from analysis scripts
#
# Each script reads source data and writes a JSON to ../checkin/.
# Only reruns if the script or its data dependencies changed.
#
# Usage:
#   make          — build all checkin JSONs
#   make clean    — remove all checkin JSONs
#   make <name>   — build one, e.g. make regime

PYTHON = python3
CHECKIN = ../checkin
DATA = ../data
WCDE = ../wcde/data/processed
RESID_SHARED = residualization/_shared.py

# Default to parallel builds when invoked directly (e.g. `cd scripts && make`).
# If MAKEFLAGS already contains -j (e.g. top-level recurse passes it), leave alone.
ifeq (,$(findstring -j,$(MAKEFLAGS)))
MAKEFLAGS += -j$(shell nproc 2>/dev/null || echo 4)
endif

WB_DATA = $(wildcard $(DATA)/*.csv)
WCDE_DATA = $(wildcard $(WCDE)/*.csv)
ALL_DATA = $(WB_DATA) $(WCDE_DATA)

# Recipe wrapper: redirect each script's chatty stdout to a per-script log
# under checkin/logs/, print a single-line status, and on failure dump the
# tail of the log so traceback context is visible without re-running.
# Set VERBOSE=1 to stream script output directly (old behavior).
ifeq ($(VERBOSE),1)
RUN = $(PYTHON) $<
else
RUN = @mkdir -p $(CHECKIN)/logs && \
      log=$(CHECKIN)/logs/$(notdir $(@:.json=.log)); \
      if $(PYTHON) $< > $$log 2>&1; then \
          echo "  [ok] $(notdir $@)"; \
      else \
          echo "  [FAIL] $(notdir $@) (tail $$log:)"; \
          tail -n 30 $$log; \
          exit 1; \
      fi
endif

.PHONY: all clean

all: verify analysis

# ── Verify scripts (produce checkin JSONs) ────────────────────────────

verify: \
	$(CHECKIN)/country_education.json \
	$(CHECKIN)/country_gdp.json \
	$(CHECKIN)/country_le_tfr.json \
	$(CHECKIN)/table4_crossings.json \
	$(CHECKIN)/china_cr.json \
	$(CHECKIN)/college_le_gradient.json \
	$(CHECKIN)/costa_rica_korea.json \
	$(CHECKIN)/figure2_betas.json \
	$(CHECKIN)/table_a1_cutoffs.json \
	$(CHECKIN)/kerala.json \
	$(CHECKIN)/edu_vs_gdp_predicts_le.json \
	$(CHECKIN)/regression_tables.json \
	$(CHECKIN)/u5mr_residual_by_year.json \
	$(CHECKIN)/regime_education_test.json \
	$(CHECKIN)/iv_2sls_colonial_icrg.json \
	$(CHECKIN)/korea_vs_biological_pace.json \
	$(CHECKIN)/sdg4_implied_pace.json

# ── Analysis scripts (produce checkin JSONs) ──────────────────────────

analysis: \
	$(CHECKIN)/beta_vs_baseline.json \
	$(CHECKIN)/beta_by_baseline_group.json \
	$(CHECKIN)/robustness_tests.json \
	$(CHECKIN)/education_outcomes.json \
	$(CHECKIN)/famine_education_test.json \
	$(CHECKIN)/long_run_generational.json \
	$(CHECKIN)/panel_full_fe.json \
	$(CHECKIN)/summary_stats.json \
	$(CHECKIN)/table_1_stepwise.json \
	$(CHECKIN)/table_1_subgroups.json \
	$(CHECKIN)/twfe_child_edu.json \
	$(CHECKIN)/education_vs_gdp_by_cutoff.json \
	$(CHECKIN)/beta_by_ceiling_cutoff.json \
	$(CHECKIN)/asian_financial_crisis.json \
	$(CHECKIN)/outcomes_r2_by_lag.json \
	$(CHECKIN)/edu_vs_gdp_residualized.json \
	$(CHECKIN)/edu_vs_gdp_tfr_residualized.json \
	$(CHECKIN)/edu_vs_gdp_entry_ceiling.json \
	$(CHECKIN)/edu_vs_gdp_entry_threshold.json \
	$(CHECKIN)/edu_vs_gdp_by_level.json \
	$(CHECKIN)/edu_vs_gdp_child_edu_residualized.json \
	$(CHECKIN)/female_education_residualized.json \
	$(CHECKIN)/twfe_all_outcomes.json \
	$(CHECKIN)/residual_by_outcome_year_all.json \
	$(CHECKIN)/u5mr_by_edu_level.json \
	$(CHECKIN)/lag_sensitivity.json \
	$(CHECKIN)/completion_vs_test_scores.json \
	$(CHECKIN)/threshold_robustness.json \
	$(CHECKIN)/happiness_education.json \
	$(CHECKIN)/singapore_continuation.json \
	$(CHECKIN)/grandparent_effect_all_outcomes.json \
	$(CHECKIN)/backfill_all_outcomes.json \
	$(CHECKIN)/lag_coefficients.json \
	$(CHECKIN)/edu_vs_resid_gdp_bars_beta.json \
	$(CHECKIN)/cutoff_all_outcomes.json \
	$(CHECKIN)/wooldridge_exogeneity.json \
	$(CHECKIN)/ppml_outcomes.json \
	$(CHECKIN)/log_outcomes.json \
	$(CHECKIN)/balanced_panel.json \
	$(CHECKIN)/period_length.json \
	$(CHECKIN)/soviet_inflation.json \
	$(CHECKIN)/edu_measure_horse_race.json \
	$(CHECKIN)/ussr_anomaly_barrolee.json \
	$(CHECKIN)/ussr_residual_by_year.json \
	$(CHECKIN)/ussr_per_country_residuals.json \
	$(CHECKIN)/ussr_metropole_distance.json \
	$(CHECKIN)/ussr_metropole_direction.json \
	$(CHECKIN)/ussr_politburo_voice.json \
	$(CHECKIN)/ussr_exclusion_panel.json \
	$(CHECKIN)/hanushek_horse_race.json \
	$(CHECKIN)/hlo_is_pt.json \
	$(CHECKIN)/hlo_lag_sweep.json \
	$(CHECKIN)/lsec_thresholds.json \
	$(CHECKIN)/cs_threshold_robustness.json \
	$(CHECKIN)/permutation_null.json \
	$(CHECKIN)/missing_by_period.json \
	$(CHECKIN)/table7_stepwise.json \
	$(CHECKIN)/ssa_education_tfr.json \
	$(CHECKIN)/post_socialist_lag.json

# ── Rules ─────────────────────────────────────────────────────────────

$(CHECKIN)/country_education.json: cases/country_education.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/country_gdp.json: cases/country_gdp.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/country_le_tfr.json: cases/country_le_tfr.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/table4_crossings.json: cases/threshold_crossings.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/china_cr.json: cases/china_cultural_revolution.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/college_le_gradient.json: cases/college_le_gradient.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/costa_rica_korea.json: cases/costa_rica_korea.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/korea_vs_biological_pace.json: korea_vs_biological_pace.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/sdg4_implied_pace.json: sdg4_implied_pace.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/figure2_betas.json: cases/figure2_betas.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/table_a1_cutoffs.json: cases/threshold_cutoffs.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/kerala.json: cases/kerala.py $(DATA)/kerala_srs.csv
	$(RUN)

$(CHECKIN)/edu_vs_gdp_predicts_le.json: residualization/education_predicts_le.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/regression_tables.json: tables/regression_tables.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(RUN)

$(CHECKIN)/u5mr_residual_by_year.json: robustness/u5mr_residual_by_year.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(RUN)

$(CHECKIN)/regime_education_test.json: robustness/regime_education.py $(ALL_DATA) $(DATA)/p5v2018.xls
	$(RUN)

$(CHECKIN)/iv_2sls_colonial_icrg.json: robustness/iv_2sls_colonial_icrg.py robustness/colonial_vs_institutions.py $(ALL_DATA) $(DATA)/ajr2001/maketable1.dta
	$(RUN)

$(CHECKIN)/panel_full_fe.json: tables/panel_full_fe.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/summary_stats.json: tables/summary_stats.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/table_1_stepwise.json: tables/table_1_stepwise.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/table_1_subgroups.json: tables/table_1_subgroups.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/twfe_child_edu.json: robustness/twfe_child_edu.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/education_vs_gdp_by_cutoff.json: residualization/by_gdp_cutoff.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(RUN)

$(CHECKIN)/beta_by_ceiling_cutoff.json: robustness/beta_by_ceiling_cutoff.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(RUN)

$(CHECKIN)/asian_financial_crisis.json: robustness/asian_financial_crisis.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/outcomes_r2_by_lag.json: figures/outcomes_r2_by_lag.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/edu_vs_gdp_residualized.json: residualization/education_vs_gdp.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/edu_vs_gdp_tfr_residualized.json: residualization/education_vs_tfr.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/ssa_education_tfr.json: ssa_education_tfr.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/post_socialist_lag.json: post_socialist_lag.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/edu_vs_gdp_entry_ceiling.json: residualization/by_entry_ceiling.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/edu_vs_gdp_entry_threshold.json: residualization/by_entry_threshold.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/edu_vs_gdp_by_level.json: residualization/by_education_level.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/edu_vs_gdp_child_edu_residualized.json: residualization/child_education.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(RUN)

$(CHECKIN)/female_education_residualized.json: residualization/female_education.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(RUN)

$(CHECKIN)/twfe_all_outcomes.json: robustness/twfe_all_outcomes.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(RUN)

$(CHECKIN)/residual_by_outcome_year_all.json: robustness/residual_by_outcome_year.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(RUN)

$(CHECKIN)/u5mr_by_edu_level.json: robustness/u5mr_by_edu_level.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(RUN)

$(CHECKIN)/lag_sensitivity.json: robustness/lag_sensitivity.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(RUN)

$(CHECKIN)/completion_vs_test_scores.json: robustness/completion_vs_test_scores.py _shared.py $(RESID_SHARED) $(ALL_DATA) $(DATA)/hlo_raw.csv
	$(RUN)

$(CHECKIN)/threshold_robustness.json: robustness/threshold_robustness.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/happiness_education.json: happiness_education.py $(DATA)/world_happiness_report.csv $(ALL_DATA)
	$(RUN)

$(CHECKIN)/singapore_continuation.json: singapore_continuation.py $(WCDE)/college_both.csv $(WCDE)/upper_sec_both.csv
	$(RUN)

$(CHECKIN)/grandparent_effect_all_outcomes.json: robustness/grandparent_effect_all_outcomes.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/backfill_all_outcomes.json: robustness/backfill_all_outcomes.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/lag_coefficients.json: robustness/lag_coefficients.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/permutation_null.json: robustness/permutation_null.py _shared.py $(WCDE)/completion_both_long.csv
	$(RUN)

$(CHECKIN)/edu_vs_resid_gdp_bars_beta.json: figures/edu_vs_resid_gdp_bars_beta.py _shared.py residualization/_shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/cutoff_all_outcomes.json: residualization/cutoff_all_outcomes.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(RUN)

$(CHECKIN)/famine_education_test.json: famine_education_test.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/wooldridge_exogeneity.json: robustness/wooldridge_exogeneity.py _shared.py $(WCDE)/cohort_completion_both_long.csv
	$(RUN)

$(CHECKIN)/ppml_outcomes.json: robustness/ppml_outcomes.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/log_outcomes.json: robustness/log_outcomes.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/balanced_panel.json: robustness/balanced_panel.py _shared.py $(WCDE)/cohort_completion_both_long.csv
	$(RUN)

$(CHECKIN)/period_length.json: robustness/period_length.py _shared.py $(WCDE)/cohort_completion_both_long.csv
	$(RUN)

$(CHECKIN)/beta_vs_baseline.json: figures/beta_vs_baseline.py $(WCDE_DATA)
	$(RUN)

$(CHECKIN)/beta_by_baseline_group.json: robustness/beta_by_baseline_group.py $(WCDE_DATA)
	$(RUN)

$(CHECKIN)/robustness_tests.json: robustness/robustness_tests.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(RUN)

$(CHECKIN)/education_outcomes.json: wcde/education_outcomes.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/long_run_generational.json: wcde/long_run_generational.py _shared.py $(WCDE_DATA)
	$(RUN)

# ── §9 Data Credibility: Goskomstat anomaly + Hanushek reconciliation ─

$(CHECKIN)/soviet_inflation.json: soviet_education_inflation_test.py _shared.py $(WCDE_DATA)
	$(RUN)

$(CHECKIN)/edu_measure_horse_race.json: which_edu_measure_is_correct.py _shared.py $(ALL_DATA) $(DATA)/barro_lee_v3.csv
	$(RUN)

$(CHECKIN)/ussr_anomaly_barrolee.json: ussr_anomaly_barrolee.py _shared.py $(ALL_DATA) $(DATA)/barro_lee_v3.csv
	$(RUN)

$(CHECKIN)/ussr_residual_by_year.json: ussr_residual_by_year.py _shared.py $(ALL_DATA) $(DATA)/barro_lee_v3.csv
	$(RUN)

$(CHECKIN)/ussr_per_country_residuals.json: ussr_per_country_residuals.py _shared.py $(ALL_DATA) $(DATA)/barro_lee_v3.csv
	$(RUN)

$(CHECKIN)/ussr_metropole_distance.json: ussr_metropole_distance.py _shared.py $(CHECKIN)/ussr_per_country_residuals.json
	$(RUN)

$(CHECKIN)/ussr_metropole_direction.json: ussr_metropole_direction.py _shared.py $(CHECKIN)/ussr_per_country_residuals.json
	$(RUN)

$(CHECKIN)/ussr_politburo_voice.json: ussr_politburo_voice.py _shared.py $(CHECKIN)/ussr_per_country_residuals.json
	$(RUN)

$(CHECKIN)/ussr_exclusion_panel.json: robustness/ussr_exclusion_panel.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/hanushek_horse_race.json: hanushek_horse_race_comprehensive.py _shared.py $(ALL_DATA) $(DATA)/hlo_raw.csv $(DATA)/barro_lee_v3.csv $(WCDE)/population_2015.csv
	$(RUN)

$(CHECKIN)/hlo_is_pt.json: hlo_is_parental_education.py _shared.py $(ALL_DATA) $(DATA)/hlo_raw.csv $(DATA)/barro_lee_v3.csv
	$(RUN)

$(CHECKIN)/hlo_lag_sweep.json: hlo_vs_education_lag_sweep.py _shared.py $(WCDE_DATA) $(DATA)/hlo_raw.csv
	$(RUN)

$(CHECKIN)/lsec_thresholds.json: lsec_threshold_forward_test.py _shared.py $(ALL_DATA)
	$(RUN)

$(CHECKIN)/cs_threshold_robustness.json: robustness/cs_threshold_robustness.py _shared.py $(WCDE_DATA) $(ALL_DATA)
	$(RUN)

$(CHECKIN)/missing_by_period.json: robustness/missing_by_period.py _shared.py $(WCDE_DATA) $(ALL_DATA)
	$(RUN)

$(CHECKIN)/table7_stepwise.json: wcde/table7_stepwise.py _shared.py $(WCDE_DATA) $(ALL_DATA)
	$(RUN)

# ── Clean ─────────────────────────────────────────────────────────────

clean:
	rm -f $(CHECKIN)/*.json
