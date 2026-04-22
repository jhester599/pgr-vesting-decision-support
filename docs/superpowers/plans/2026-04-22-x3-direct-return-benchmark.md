# x3 Direct Return Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the x3 research-only direct forward-return and log-return regression benchmark for absolute PGR forecasting.

**Architecture:** Reuse x1 target construction, the existing processed feature matrix, and x2's strict horizon-aware WFO split policy. Keep x3 isolated under `src/research/`, `scripts/research/`, `tests/`, and `results/research/`; do not touch production monthly decision paths.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn Ridge and shallow histogram GBT regressors, existing feature cache and SQLite price data. No K-Fold CV and no full-sample scaling.

---

## Scope

x3 tests whether direct forward-return or log-forward-return regression produces
useful implied future prices at +1m, +3m, +6m, and +12m.

The first x3 benchmark compares:

- `ridge_return`: strongly L2-regularized regression on simple forward return.
- `ridge_log_return`: strongly L2-regularized regression on log forward return.
- `hist_gbt_return`: shallow histogram GBT regression on simple forward return.
- `hist_gbt_log_return`: shallow histogram GBT regression on log forward return.
- `no_change`: zero-return / random-walk price baseline.
- `drift`: fold-local historical mean-return baseline.

x3 does not include BVPS x P/B decomposition, special-dividend modeling,
ensemble selection, TA expansion, production wiring, or monthly shadow outputs.

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/superpowers/plans/2026-04-22-x3-direct-return-benchmark.md` | Create | x3 plan |
| `src/research/x3_direct_return.py` | Create | Research-only direct-return regression utilities |
| `scripts/research/x3_direct_return.py` | Create | Artifact-producing x3 experiment runner |
| `tests/test_research_x3_direct_return.py` | Create | WFO, baseline, transform, and summary tests |
| `results/research/x3_direct_return_detail.csv` | Create | Per-horizon/model metrics |
| `results/research/x3_direct_return_summary.json` | Create | Ranked x3 summary |
| `results/research/x3_research_memo.md` | Create | Human-readable x3 memo |

## Task 1: Tests First

- [ ] Test that the evaluator emits chronological OOS predictions with the
      same horizon-aware WFO gap policy used by x2.
- [ ] Test that return-target and log-return-target models convert predictions
      back into implied future price consistently.
- [ ] Test that fold-local `drift` uses only each training fold's realized
      target history.
- [ ] Test that `no_change` predicts zero return and current price as implied
      future price.
- [ ] Test that summaries rank by lower implied price MAE, then lower return
      RMSE, then higher directional hit rate.

## Task 2: Regression Utilities

- [ ] Implement `build_direct_return_regressor`.
- [ ] Implement fold-local median imputation using training folds only.
- [ ] Implement `evaluate_direct_return_regressor`.
- [ ] Implement `evaluate_direct_return_baseline`.
- [ ] Implement `summarize_direct_return_results`.

## Task 3: x3 Runner

- [ ] Load the existing processed feature matrix without refreshing it.
- [ ] Load PGR prices from the checked-in SQLite database.
- [ ] Build x1 forward return and log-return targets for horizons 1, 3, 6, and
      12.
- [ ] Evaluate baseline and model rows for each horizon.
- [ ] Write deterministic detail CSV, summary JSON, and memo under
      `results/research/`.

## Task 4: Verification

- [ ] Run `python -m pytest tests/test_research_x3_direct_return.py -q --tb=short`.
- [ ] Run `python scripts/research/x3_direct_return.py`.
- [ ] Run focused x1/x2/x3 tests together.
- [ ] Run Python compile checks for the new x3 module, script, and tests.

## Production Boundary

x3 is research-only. It must not edit `scripts/monthly_decision.py`,
production configuration, monthly output artifacts, or existing `v###`
research-lane artifacts.
