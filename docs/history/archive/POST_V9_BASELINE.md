# Post-v9 Baseline Reconciliation

This document defines the repository baseline after the v9 research program and
before any future production-model promotion.

## Baseline Summary

The repository now has two deliberate layers:

- A production operating layer that keeps the committed database, monthly
  decision artifacts, workflows, and user-facing recommendations current.
- A research layer introduced in v9 that adds evaluation harnesses, candidate
  model bakeoffs, policy studies, and committed research outputs used to inform
  future promotion decisions.

The repository baseline at v10.1 is therefore mixed:

- Production baseline: v8.13 behavior, plus v10.1 operational hardening.
- Research baseline: completed v9 evaluation and classifier-sidecar work.

## Inventory of Meaningful v9 Additions

### New research scripts

- `scripts/benchmark_suite.py`
- `scripts/feature_cost_report.py`
- `scripts/feature_experiments.py`
- `scripts/target_experiments.py`
- `scripts/benchmark_reduction.py`
- `scripts/regime_slice_backtest.py`
- `scripts/policy_evaluation.py`
- `scripts/pooled_benchmark_experiments.py`
- `scripts/candidate_model_bakeoff.py`
- `scripts/weekly_snapshot_experiments.py`
- `scripts/confirmatory_classifier_experiments.py`
- `scripts/classifier_feature_selection.py`

### New reusable modules

- `src/research/evaluation.py`
- `src/research/benchmark_sets.py`
- `src/research/policy_metrics.py`

### New result paths

- `results/v9/`

### New top-level research documents

- `docs/plans/codex-v9-plan.md`
- `docs/results/V9_RESULTS_SUMMARY.md`
- `docs/closeouts/V9_CLOSEOUT_AND_V91_NEXT.md`

## Classification of Current Repo Components

### Production

- `config.py`
- `src/database/`
- `src/ingestion/`
- `src/models/`
- `src/portfolio/`
- `src/processing/`
- `src/reporting/`
- `scripts/weekly_fetch.py`
- `scripts/peer_fetch.py`
- `scripts/edgar_8k_fetcher.py`
- `scripts/monthly_decision.py`
- `.github/workflows/weekly_data_fetch.yml`
- `.github/workflows/peer_data_fetch.yml`
- `.github/workflows/monthly_8k_fetch.yml`
- `.github/workflows/monthly_decision.yml`
- `results/monthly_decisions/`
- `data/pgr_financials.db`

### Research / evaluation

- `src/research/`
- all v9 research scripts listed above
- `results/v9/`
- v9 summary and closeout documents

### Provisional

- any v9 candidate model recommendation not yet promoted to the monthly
  workflow
- the tuned Ridge classifier-sidecar candidate

### Generated artifact only

- `results/monthly_decisions/<YYYY-MM>/`
- `results/v9/`

### Historical / superseded planning artifacts

- `ROADMAP.md`
- `DEVELOPMENT_PLAN.md`
- `docs/history/claude-v7-plan.md`
- `docs/plans/codex-v8-plan.md`
- older review documents under `docs/`

## What Production Currently Uses

The production monthly decision path currently remains the v8.13 4-model
ensemble:

- ElasticNet
- Ridge
- BayesianRidge
- GBT

It also includes:

- model-quality gating
- EDGAR parser breadth improvements
- tax scenario reporting
- HTML/plaintext decision email generation

## Treatment of `results/v9/`

`results/v9/` is retained in the repository as committed research evidence.

Its role is:

- preserve the diagnostics that informed the v9 conclusion
- support future promotion decisions and follow-on research
- make future review reproducible without re-running every experiment

It is not used by production workflows and should not be treated as live
operational output.

## Current Promotion Position

The current recommendation after v9 is:

- keep the existing production monthly workflow in place
- do not promote a v9 model replacement yet
- use v9 outputs as the evidence base for the next reduced-universe production
  promotion bakeoff

## What v10.1 Changes

v10.1 does not revisit the v9 modeling conclusion. It hardens the repository
around that conclusion by improving:

- baseline coherence
- workflow safety
- CI
- schema evolution
- documentation
- run traceability
- contributor and operator guidance

## Authoritative Rule

When there is a mismatch between:

- production workflows and reports
- historical planning documents
- research outputs

the production workflow code and current hardening docs in `README.md`,
`POST_V9_BASELINE.md`, and `docs/` should be treated as authoritative for the
repo's current state.
