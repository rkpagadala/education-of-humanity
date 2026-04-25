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
	$(CHECKIN)/colonial_education_vs_institutions.json \
	$(CHECKIN)/iv_2sls_colonial.json \

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
	$(CHECKIN)/colonial_all_outcomes.json \
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
	$(CHECKIN)/ussr_exclusion_panel.json \
	$(CHECKIN)/hanushek_horse_race.json \
	$(CHECKIN)/hlo_is_pt.json \
	$(CHECKIN)/hlo_lag_sweep.json \
	$(CHECKIN)/lsec_thresholds.json \
	$(CHECKIN)/cs_threshold_robustness.json \
	$(CHECKIN)/missing_by_period.json \
	$(CHECKIN)/table7_stepwise.json

# ── Rules ─────────────────────────────────────────────────────────────

$(CHECKIN)/country_education.json: cases/country_education.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/country_gdp.json: cases/country_gdp.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/country_le_tfr.json: cases/country_le_tfr.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/table4_crossings.json: cases/threshold_crossings.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/china_cr.json: cases/china_cultural_revolution.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/college_le_gradient.json: cases/college_le_gradient.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/costa_rica_korea.json: cases/costa_rica_korea.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/figure2_betas.json: cases/figure2_betas.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/table_a1_cutoffs.json: cases/threshold_cutoffs.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/kerala.json: cases/kerala.py $(DATA)/kerala_srs.csv
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_predicts_le.json: residualization/education_predicts_le.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/regression_tables.json: tables/regression_tables.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/u5mr_residual_by_year.json: robustness/u5mr_residual_by_year.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/regime_education_test.json: robustness/regime_education.py $(ALL_DATA) $(DATA)/p5v2018.xls
	$(PYTHON) $<

$(CHECKIN)/colonial_education_vs_institutions.json: robustness/colonial_vs_institutions.py $(ALL_DATA) $(DATA)/p5v2018.xls
	$(PYTHON) $<

$(CHECKIN)/iv_2sls_colonial.json: robustness/iv_2sls_colonial.py robustness/colonial_vs_institutions.py $(ALL_DATA) $(DATA)/p5v2018.xls
	$(PYTHON) $<

$(CHECKIN)/panel_full_fe.json: tables/panel_full_fe.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/summary_stats.json: tables/summary_stats.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/table_1_stepwise.json: tables/table_1_stepwise.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/table_1_subgroups.json: tables/table_1_subgroups.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/twfe_child_edu.json: robustness/twfe_child_edu.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/education_vs_gdp_by_cutoff.json: residualization/by_gdp_cutoff.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/beta_by_ceiling_cutoff.json: robustness/beta_by_ceiling_cutoff.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/asian_financial_crisis.json: robustness/asian_financial_crisis.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/outcomes_r2_by_lag.json: figures/outcomes_r2_by_lag.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_residualized.json: residualization/education_vs_gdp.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_tfr_residualized.json: residualization/education_vs_tfr.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_entry_ceiling.json: residualization/by_entry_ceiling.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_entry_threshold.json: residualization/by_entry_threshold.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_by_level.json: residualization/by_education_level.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_gdp_child_edu_residualized.json: residualization/child_education.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/female_education_residualized.json: residualization/female_education.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/twfe_all_outcomes.json: robustness/twfe_all_outcomes.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/residual_by_outcome_year_all.json: robustness/residual_by_outcome_year.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/u5mr_by_edu_level.json: robustness/u5mr_by_edu_level.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/lag_sensitivity.json: robustness/lag_sensitivity.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/completion_vs_test_scores.json: robustness/completion_vs_test_scores.py _shared.py $(RESID_SHARED) $(ALL_DATA) $(DATA)/hlo_raw.csv
	$(PYTHON) $<

$(CHECKIN)/threshold_robustness.json: robustness/threshold_robustness.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/happiness_education.json: happiness_education.py $(DATA)/world_happiness_report.csv $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/singapore_continuation.json: singapore_continuation.py $(WCDE)/college_both.csv $(WCDE)/upper_sec_both.csv
	$(PYTHON) $<

$(CHECKIN)/grandparent_effect_all_outcomes.json: robustness/grandparent_effect_all_outcomes.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/backfill_all_outcomes.json: robustness/backfill_all_outcomes.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/lag_coefficients.json: robustness/lag_coefficients.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_vs_resid_gdp_bars_beta.json: figures/edu_vs_resid_gdp_bars_beta.py _shared.py residualization/_shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/colonial_all_outcomes.json: robustness/colonial_all_outcomes.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/cutoff_all_outcomes.json: residualization/cutoff_all_outcomes.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/famine_education_test.json: famine_education_test.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/wooldridge_exogeneity.json: robustness/wooldridge_exogeneity.py _shared.py $(WCDE)/cohort_completion_both_long.csv
	$(PYTHON) $<

$(CHECKIN)/ppml_outcomes.json: robustness/ppml_outcomes.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/log_outcomes.json: robustness/log_outcomes.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/balanced_panel.json: robustness/balanced_panel.py _shared.py $(WCDE)/cohort_completion_both_long.csv
	$(PYTHON) $<

$(CHECKIN)/period_length.json: robustness/period_length.py _shared.py $(WCDE)/cohort_completion_both_long.csv
	$(PYTHON) $<

$(CHECKIN)/beta_vs_baseline.json: figures/beta_vs_baseline.py $(WCDE_DATA)
	$(PYTHON) $<

$(CHECKIN)/beta_by_baseline_group.json: robustness/beta_by_baseline_group.py $(WCDE_DATA)
	$(PYTHON) $<

$(CHECKIN)/robustness_tests.json: robustness/robustness_tests.py _shared.py $(RESID_SHARED) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/education_outcomes.json: wcde/education_outcomes.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/long_run_generational.json: wcde/long_run_generational.py _shared.py $(WCDE_DATA)
	$(PYTHON) $<

# ── §9 Data Credibility: Goskomstat anomaly + Hanushek reconciliation ─

$(CHECKIN)/soviet_inflation.json: soviet_education_inflation_test.py _shared.py $(WCDE_DATA)
	$(PYTHON) $<

$(CHECKIN)/edu_measure_horse_race.json: which_edu_measure_is_correct.py _shared.py $(ALL_DATA) $(DATA)/barro_lee_v3.csv
	$(PYTHON) $<

$(CHECKIN)/ussr_anomaly_barrolee.json: ussr_anomaly_barrolee.py _shared.py $(ALL_DATA) $(DATA)/barro_lee_v3.csv
	$(PYTHON) $<

$(CHECKIN)/ussr_residual_by_year.json: ussr_residual_by_year.py _shared.py $(ALL_DATA) $(DATA)/barro_lee_v3.csv
	$(PYTHON) $<

$(CHECKIN)/ussr_exclusion_panel.json: robustness/ussr_exclusion_panel.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/hanushek_horse_race.json: hanushek_horse_race_comprehensive.py _shared.py $(ALL_DATA) $(DATA)/hlo_raw.csv $(DATA)/barro_lee_v3.csv ../wcde/data/raw/pop_both.csv
	$(PYTHON) $<

$(CHECKIN)/hlo_is_pt.json: hlo_is_parental_education.py _shared.py $(ALL_DATA) $(DATA)/hlo_raw.csv $(DATA)/barro_lee_v3.csv
	$(PYTHON) $<

$(CHECKIN)/hlo_lag_sweep.json: hlo_vs_education_lag_sweep.py _shared.py $(WCDE_DATA) $(DATA)/hlo_raw.csv
	$(PYTHON) $<

$(CHECKIN)/lsec_thresholds.json: lsec_threshold_forward_test.py _shared.py $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/cs_threshold_robustness.json: robustness/cs_threshold_robustness.py _shared.py $(WCDE_DATA) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/missing_by_period.json: robustness/missing_by_period.py _shared.py $(WCDE_DATA) $(ALL_DATA)
	$(PYTHON) $<

$(CHECKIN)/table7_stepwise.json: wcde/table7_stepwise.py _shared.py $(WCDE_DATA) $(ALL_DATA)
	$(PYTHON) $<

# ── Clean ─────────────────────────────────────────────────────────────

clean:
	rm -f $(CHECKIN)/*.json
