# x4 BVPS Forecasting Leg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the x4 research-only BVPS forecasting leg for the eventual BVPS x P/B decomposition path.

**Architecture:** Reuse x1 decomposition targets and x2's horizon-aware WFO split policy. Load monthly BVPS from the checked-in PGR EDGAR table, apply the repo EDGAR filing lag, normalize it to the feature matrix's business-month-end availability calendar, and evaluate simple growth/log-growth models under `src/research/`, `scripts/research/`, `tests/`, and `results/research/` only.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn Ridge and shallow histogram GBT regressors, existing feature cache, existing SQLite PGR EDGAR/price data. No K-Fold CV and no full-sample scaling.

---

## Scope

x4 tests whether future BVPS or BVPS growth can be forecast at +1m, +3m,
+6m, and +12m. It is the BVPS leg only; x5 will forecast future P/B and
combine the two legs into an implied price.

The first x4 benchmark compares:

- `ridge_bvps_growth`: strongly L2-regularized regression on simple BVPS growth.
- `ridge_log_bvps_growth`: strongly L2-regularized regression on log BVPS growth.
- `hist_gbt_bvps_growth`: shallow histogram GBT regression on simple BVPS growth.
- `hist_gbt_log_bvps_growth`: shallow histogram GBT regression on log BVPS growth.
- `no_change_bvps`: zero-growth current-BVPS baseline.
- `drift_bvps_growth`: fold-local historical mean simple BVPS-growth baseline.

x4 does not forecast future P/B, recombine implied price, model special
dividends, expand TA features, or touch production monthly decision paths.

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/superpowers/plans/2026-04-22-x4-bvps-forecasting-leg.md` | Create | x4 plan |
| `src/research/x4_bvps_forecasting.py` | Create | Research-only BVPS leg WFO utilities |
| `scripts/research/x4_bvps_forecasting.py` | Create | Artifact-producing x4 runner |
| `tests/test_research_x4_bvps_forecasting.py` | Create | Calendar, WFO, transform, baseline, and ranking tests |
| `results/research/x4_bvps_forecasting_detail.csv` | Create | Per-horizon/model metrics |
| `results/research/x4_bvps_forecasting_summary.json` | Create | Ranked x4 summary |
| `results/research/x4_research_memo.md` | Create | Human-readable x4 memo |

## Task 1: Tests First

- [ ] Test that monthly BVPS snapshots normalize to lagged business month-end
      availability dates.
- [ ] Test that the evaluator emits chronological OOS predictions and respects
      horizon-aware WFO gaps.
- [ ] Test that growth and log-growth predictions convert back to implied
      future BVPS consistently.
- [ ] Test that fold-local drift uses only training-fold simple BVPS growth.
- [ ] Test that `no_change_bvps` predicts current BVPS.
- [ ] Test that summaries rank by lower future-BVPS MAE, then lower growth
      RMSE, then higher directional hit rate.

## Task 2: BVPS Utilities

- [ ] Implement business-month-end normalization for BVPS source rows.
- [ ] Implement `build_bvps_regressor`.
- [ ] Implement fold-local non-finite feature imputation.
- [ ] Implement `evaluate_bvps_regressor`.
- [ ] Implement `evaluate_bvps_baseline`.
- [ ] Implement `summarize_bvps_results`.

## Task 3: x4 Runner

- [ ] Load the existing processed feature matrix without refreshing it.
- [ ] Load PGR prices and PGR monthly EDGAR BVPS from the checked-in database.
- [ ] Build x1 decomposition targets for horizons 1, 3, 6, and 12.
- [ ] Evaluate baseline and model rows for each horizon.
- [ ] Write deterministic detail CSV, summary JSON, and memo under
      `results/research/`.

## Task 4: Verification

- [ ] Run `python -m pytest tests/test_research_x4_bvps_forecasting.py -q --tb=short`.
- [ ] Run `python scripts/research/x4_bvps_forecasting.py`.
- [ ] Run focused x1/x2/x4 tests together.
- [ ] Run Python compile checks for the new x4 module, script, and tests.

## Production Boundary

x4 is research-only. It must not edit `scripts/monthly_decision.py`,
production configuration, monthly output artifacts, or existing `v###`
research-lane artifacts.
