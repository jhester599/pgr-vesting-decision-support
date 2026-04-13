# v126 Methodology Hardening And Artifact Refresh

Date: 2026-04-12
Status: implemented on `codex/v122-classifier-audit-note`

## Why v126 was inserted

The original `v125` Path B research drop surfaced a useful composite-target
classifier, but review found three reliability issues that made the published
conclusion too strong:

1. Path B was compared against a hardcoded `v92` baseline instead of a matched
   Path A baseline.
2. `run_wfo()` used an expanding window even though the project standard is the
   configured rolling window (`WFO_TRAIN_WINDOW_MONTHS = 60`).
3. Checked-in monthly artifacts still reflected the older 4-benchmark
   investable pool instead of the expanded 6-benchmark `{VOO, VGT, VIG, VXUS,
   VWO, BND}` configuration.

v126 exists to fix those issues before additional calibration or feature-set
research continues.

## What changed

### 1. Path B comparison is now matched and reproducible

File:
- `results/research/v125_portfolio_target_classifier.py`

Key changes:
- Removed the hardcoded `PATH_A_REFERENCE` baseline.
- Added `_resolve_n_splits()` so feasible rolling WFO split counts are derived
  from dataset length, train window, embargo/purge gap, and test size.
- Updated `run_wfo()` to use `TimeSeriesSplit(..., max_train_size=60)`.
- Added `build_path_a_matched_probability_frame()`:
  - rebuilds Path A benchmark-level OOS probabilities for all 6 investable
    benchmarks,
  - applies prequential logistic calibration benchmark-by-benchmark,
  - aggregates those calibrated probabilities with row-wise renormalized fixed
    portfolio weights.
- The fold-detail artifact now stores both `path_a_prob` and `path_b_prob` on
  the same matched `test_date` rows.

Resulting artifact outputs:
- `results/research/v125_portfolio_target_results.csv`
- `results/research/v125_portfolio_target_fold_detail.csv`
- `results/research/v125_portfolio_target_summary.md`

Current matched-v126 summary:
- Path A covered balanced accuracy: `0.5000`
- Path B covered balanced accuracy: `0.6450`
- Path B calibration is worse (`brier_score`, `log_loss`, `ece` all degrade)
- v126 verdict: keep Path B as a secondary research track until calibration is
  improved

### 2. Added dedicated v126 regression tests

File:
- `tests/test_research_v126_portfolio_target_classifier.py`

Coverage added:
- rolling split capacity resolves to 14 folds for the current 152-row dataset
- `run_wfo()` respects the max 60-month rolling window logic
- row-wise probability aggregation renormalizes correctly when a benchmark is
  missing
- verdict text no longer promotes Path B when balanced accuracy improves but
  calibration degrades materially

### 3. Refreshed monthly artifacts to match the 6-benchmark investable pool

Files refreshed:
- `results/monthly_decisions/2026-02/*`
- `results/monthly_decisions/2026-03/*`
- `results/monthly_decisions/2026-04/*`
- `results/monthly_decisions/classification_shadow_history.csv`
- `results/monthly_decisions/decision_log.md`

Important artifact-level effects:
- `classification_shadow.benchmark_count` now reflects the 10 modeled
  benchmarks (8 original + `VGT` + `VIG`)
- `classification_shadow.investable_benchmark_count` is now `6`
- `classification_shadow.probability_investable_pool` is now computed from the
  6-benchmark fixed-weight pool
- `classification_shadow.csv` now contains `VGT` and `VIG` rows and retains the
  `is_contextual` flag

Example refreshed values:
- `2026-03 monthly_summary.json`
  - `benchmark_count = 10`
  - `investable_benchmark_count = 6`
  - `probability_investable_pool = 42.4%`

## Commands used for verification

Research:
- `python results/research/v125_portfolio_target_classifier.py --as-of 2024-03-31`
- `python results/research/v122_classifier_audit.py`

Tests:
- `python -m pytest tests/test_research_v126_portfolio_target_classifier.py`
- `python -m pytest tests/test_research_v126_portfolio_target_classifier.py tests/test_classification_shadow.py tests/test_classification_artifacts.py tests/test_research_v122_classifier_audit.py tests/test_classification_config.py`

Artifact refresh:
- `python scripts/monthly_decision.py --as-of 2026-02-28 --dry-run --skip-fred`
- `python scripts/monthly_decision.py --as-of 2026-03-31 --dry-run --skip-fred`
- `python scripts/monthly_decision.py --as-of 2026-04-11 --dry-run --skip-fred`

## Handoff notes for Claude Code

If this branch is handed back to Claude Code later, the intended next work is:

1. Keep Path A as the primary reference architecture until Path B calibration is
   improved on the matched-v126 setup.
2. Treat the `v125_*` artifact filenames as continuity names only; their
   contents now reflect the v126 remediation logic.
3. Build any future Path B calibration work on top of the matched comparison
   frame already written to `v125_portfolio_target_fold_detail.csv`.
4. If historical monthly artifacts are regenerated again, use the same dry-run
   command pattern above so `classification_shadow` stays aligned with the
   investable-pool config.
