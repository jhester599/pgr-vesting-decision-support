# Autoresearch Execution Log - 2026-04-14

This log records actual overnight/autonomous progress against
`docs/superpowers/plans/2026-04-13-autoresearch-execution-plan.md` using the
repo state that existed on 2026-04-14.

## Completed

### Target 1 - Per-benchmark feature-map evaluation re-scope

Status: completed as a live-state re-scope of the original plan.

Artifacts created:
- `results/research/v129_feature_map_eval.py`
- `tests/test_research_v129_feature_map_eval.py`
- `results/research/v129_candidate_map.csv`

Validation:
- `python -m pytest tests/test_research_v129_feature_map_eval.py tests/test_classification_shadow.py -q --tb=short`
- Passing as of 2026-04-14.

Observed baselines:
- `--strategy lean_baseline` -> `covered_ba=0.5000`, `coverage=0.8700`
- `--strategy v128_map` -> `covered_ba=0.5016`, `coverage=0.8891`
- `--strategy file:results/research/v129_candidate_map.csv` ->
  `covered_ba=0.6164`, `coverage=0.6564`

Interpretation:
- The plan's literal no-retrain replay path was impossible from current
  artifacts, so `v129` was implemented on top of the existing `v128`
  benchmark-WFO machinery.
- The file-backed candidate path materially outperformed the canonical saved
  pooled v128 rows on the re-scoped replay frame, but this is not a
  drop-in replacement for the original plan's proposed `v125`-detail replay.

### Target 5 - Path B temperature parameter tuning

Status: completed for the first bounded sweep.

Artifacts created:
- `results/research/v135_temp_param_search.py`
- `tests/test_research_v135_temp_param_search.py`
- `results/research/v135_temp_max_candidate.txt`
- `results/research/v135_warmup_candidate.txt`
- `results/research/v135_temp_param_autoresearch_log.jsonl`
- `results/research/v135_temp_param_search_summary.md`

Validation:
- `python -m pytest tests/test_research_v135_temp_param_search.py tests/test_research_v131_threshold_sweep_eval.py -q --tb=short`
- Passing as of 2026-04-14.

Observed default baseline:
- `(temp_max=3.0, warmup=24, low=0.15, high=0.70)` ->
  `covered_ba=0.6322`, `coverage=0.4524`

Observed winner from the first 80-point search:
- `(temp_max=2.5, warmup=42)` ->
  `covered_ba=0.6987`, `coverage=0.5476`, `brier=0.1589`,
  `log_loss=0.6195`, `ece=0.1339`

Interpretation:
- The selection-frame success gate was met (`BA >= 0.60`, `coverage >= 0.25`,
  `brier <= 0.20`).
- This is still a research result on the same historical evaluation frame; do
  not promote to `config/model.py` without temporal hold-out confirmation.

### Target 4 - FRED publication lag optimization

Status: completed for the first bounded sweep.

Artifacts created:
- `docs/data/fred_publication_lag_reference.md`
- `results/research/v134_fred_lag_sweep.py`
- `tests/test_research_v134_fred_lag_sweep.py`
- `results/research/v134_lag_candidate.json`
- `results/research/v134_fred_lag_autoresearch_log.jsonl`
- `results/research/v134_fred_lag_search_summary.md`

Validation:
- `python -m pytest tests/test_research_v134_fred_lag_sweep.py -q --tb=short`
- Passing as of 2026-04-14.

Observed default baseline:
- current all-ones candidate -> `pooled_oos_r2=-0.1578`,
  `pooled_ic=0.1261`, `pooled_hit_rate=0.6906`

Observed best candidate from the bounded sweep:
- `{"GS10": 1, "GS5": 1, "GS2": 1, "T10Y2Y": 1, "T10YIE": 0, "VIXCLS": 1,
  "BAA10Y": 1, "BAMLH0A0HYM2": 1, "MORTGAGE30US": 1}` ->
  `pooled_oos_r2=-0.1573`, `pooled_ic=0.1262`,
  `pooled_hit_rate=0.6983`

Interpretation:
- The plan's older v38 baseline (`-0.1310`) is stale for the current repo
  state. The new harness establishes `-0.1578` as the reproducible baseline on
  2026-04-14.
- Reducing daily/weekly series to `lag=0` does not produce a promotable
  improvement on this frame. Only `T10YIE -> 0` produced a tiny R2 uptick, and
  the magnitude is too small to justify a config promotion without further
  evidence.

### Target 3 - Ridge alpha re-baseline

Status: completed as a current-state re-baseline plus bounded high-alpha sweep.

Artifacts created:
- `config/model.py` constants: `RIDGE_ALPHA_MIN`, `RIDGE_ALPHA_MAX`,
  `RIDGE_ALPHA_N`
- `results/research/v133_ridge_alpha_sweep.py`
- `tests/test_research_v133_ridge_alpha_sweep.py`
- `results/research/v133_alpha_max_candidate.txt`
- `results/research/v133_ridge_alpha_autoresearch_log.jsonl`
- `results/research/v133_ridge_alpha_search_summary.md`

Validation:
- `python -m pytest tests/test_research_v133_ridge_alpha_sweep.py -q --tb=short`
- Passing as of 2026-04-14.

Observed baseline and bounded sweep:
- `alpha_max=1e2` -> `pooled_oos_r2=-0.5906`, `pooled_ic=0.1317`,
  `pooled_hit_rate=0.6868`
- best observed bounded candidate: `alpha_max=1e3` ->
  `pooled_oos_r2=-0.4548`, `pooled_ic=0.1181`,
  `pooled_hit_rate=0.6992`

Interpretation:
- The original Target 3 wording is stale. The modern ridge-only path is
  materially weaker than the historical v38 ensemble baseline.
- Higher alpha does help relative to the current low-alpha ridge-only frame,
  but not enough to justify promotion.

### Target 7 - Standalone GBT parameter sweep

Status: completed for the first bounded sweep.

Artifacts created:
- `results/research/v137_gbt_param_sweep.py`
- `tests/test_research_v137_gbt_param_sweep.py`
- `results/research/v137_gbt_params_candidate.json`
- `results/research/v137_gbt_param_autoresearch_log.jsonl`
- `results/research/v137_gbt_param_search_summary.md`

Validation:
- `python -m pytest tests/test_research_v137_gbt_param_sweep.py -q --tb=short`
- Passing as of 2026-04-14.

Observed baseline and bounded sweep:
- default config `(depth=2, trees=50, lr=0.1, subsample=0.8)` ->
  `pooled_oos_r2=-0.4629`, `pooled_ic=0.1040`,
  `pooled_hit_rate=0.6485`
- best observed bounded candidate `(depth=1, trees=25, lr=0.05,
  subsample=0.8)` ->
  `pooled_oos_r2=-0.2675`, `pooled_ic=0.0924`,
  `pooled_hit_rate=0.6830`

Interpretation:
- Very shallow trees help materially on the standalone GBT frame.
- Even the best bounded candidate remains below the original success
  threshold, so this is a research lead rather than a promotion case.

### Target 6 - Backlog prioritization artifact

Status: completed as a documented local predict-equivalent artifact.

Artifacts created:
- `docs/research/backlog.md`
- `docs/research/backlog_scoring_rubric.md`
- `results/research/v136_predict_output.json`

Validation:
- structural guard checked manually via Python
- Output contains 10 ranked items and 5 items with `consensus_score >= 7`

Observed top priorities:
- `DATA-01` - EDGAR filing lag review
- `REG-01` - ensemble-level clip/shrink recalibration revisit
- `BL-01` - Black-Litterman tau/view-confidence tuning

## Review Findings On The 2026-04-13 Plan

### Target 5 baseline note

The Target 5 text says the default `v135` config should reproduce
`covered_ba ~= 0.5725`, but that value corresponds to the old
`(low=0.30, high=0.70)` baseline from `v130`/`v131`.

For the actual Target 5 loop, which fixes `(low=0.15, high=0.70)`, the true
baseline is `covered_ba=0.6322`, not `0.5725`.

### Target 1 data-contract mismatch

The plan assumes `results/research/v125_portfolio_target_fold_detail.csv`
contains stored per-benchmark fold-level logits or coefficients that can be
re-routed without retraining.

Actual file contents as of 2026-04-14:
- `fold`
- `train_start`
- `train_end`
- `train_obs`
- `test_date`
- `y_true`
- `path_b_prob`
- `path_a_prob`
- `path_a_available_benchmarks`
- `path_a_weight_sum`

It does not contain per-benchmark logits, coefficient slices, or benchmark
feature-routing state. The proposed "fast no-retrain" `v129_feature_map_eval`
cannot be implemented literally from current artifacts.

Recommended re-scope for Target 1:
- Build the evaluator on top of the `v128_benchmark_feature_search.py`
  benchmark-WFO machinery and document that it re-runs the exact benchmark
  classifier family instead of replaying stored logits.

### Target 3 staleness

The plan says Ridge has never been extended above `alpha_max=100`, but the live
repo no longer matches that assumption:
- `src/models/regularized_models.py` currently defaults Ridge to
  `np.logspace(-4, 4, 50)`
- `results/research/v39_ridge_alpha_results.csv` already contains a prior
  extended high-alpha ridge study

Before executing Target 3 literally, re-baseline the current production ridge
path and decide whether a new `v133` loop is still novel or whether the plan
should be updated to reflect the post-v39 state.

### Target 4 baseline drift

The plan says the default lag configuration should be evaluated relative to the
v38 baseline of `pooled_oos_r2=-0.1310`, but the current production research
frame no longer reproduces that value.

Actual `v134` no-override baseline as of 2026-04-14:
- `pooled_oos_r2=-0.1578`
- `pooled_ic=0.1261`
- `pooled_hit_rate=0.6906`

This does not invalidate the harness; it means future Target 4 work must treat
the current baseline as the reference point rather than the original plan text.

## Recommended Next Step

The highest-value unfinished plan item is now Target 8
(Black-Litterman tau/view-confidence optimization). Target 2 remains open if
test-runtime reduction is still desired inside this same campaign.

## Resume Checklist

When resuming after a token reset:
1. Read this file first.
2. Read `results/research/v135_temp_param_search_summary.md`.
3. Read `results/research/v134_fred_lag_search_summary.md`.
4. Read `results/research/v133_ridge_alpha_search_summary.md`.
5. Read `results/research/v137_gbt_param_search_summary.md`.
6. Treat `v129`, `v133`, `v134`, `v135`, `v136`, and `v137` as completed
   bounded sweeps unless a wider search is explicitly desired.
7. The next unfinished empirical target is Target 8.
