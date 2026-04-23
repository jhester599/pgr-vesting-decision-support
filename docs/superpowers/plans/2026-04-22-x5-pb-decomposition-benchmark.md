# x5 P/B Decomposition Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the x5 research-only P/B forecasting leg and recombined BVPS x P/B implied-price benchmark.

**Architecture:** Build on x1 decomposition targets, x2 WFO splitting, and the x4 BVPS forecasting utilities. Forecast future P/B or log(P/B), then join fold-aligned BVPS and P/B predictions to compute implied future price. Keep all outputs under x-series research paths.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn Ridge and shallow histogram GBT regressors, x4 BVPS utilities, existing feature cache and SQLite data. No K-Fold CV and no production wiring.

---

## Scope

x5 tests whether a structural decomposition can compete with direct-return x3:

`future_price ~= predicted_future_BVPS * predicted_future_P/B`

The first x5 benchmark compares:

- P/B baselines: `no_change_pb`, `drift_pb`
- P/B models: `ridge_pb`, `ridge_log_pb`, `hist_gbt_pb`, `hist_gbt_log_pb`
- Recombined pairs using a bounded set of BVPS-leg rows from x4 and P/B-leg rows
  from x5.

x5 does not model special dividends, expand TA features, tune ensembles, or wire
anything into monthly decision outputs.

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/superpowers/plans/2026-04-22-x5-pb-decomposition-benchmark.md` | Create | x5 plan |
| `src/research/x5_pb_decomposition.py` | Create | P/B leg and recombination utilities |
| `scripts/research/x5_pb_decomposition.py` | Create | Artifact-producing x5 runner |
| `tests/test_research_x5_pb_decomposition.py` | Create | P/B transform, baseline, recombination, and ranking tests |
| `results/research/x5_pb_leg_detail.csv` | Create | Per-horizon P/B leg metrics |
| `results/research/x5_decomposition_detail.csv` | Create | Per-horizon recombined price metrics |
| `results/research/x5_decomposition_summary.json` | Create | Ranked x5 summary |
| `results/research/x5_research_memo.md` | Create | Human-readable x5 memo |

## Task 1: Tests First

- [ ] Test P/B and log(P/B) predictions convert back to positive P/B.
- [ ] Test no-change P/B predicts current P/B.
- [ ] Test drift P/B uses fold-local training history only.
- [ ] Test recombination aligns BVPS and P/B predictions by date/fold.
- [ ] Test recombination summary ranks lower implied-price MAE first.

## Task 2: P/B Utilities

- [ ] Implement `build_pb_regressor`.
- [ ] Implement `evaluate_pb_regressor`.
- [ ] Implement `evaluate_pb_baseline`.
- [ ] Implement `combine_decomposition_predictions`.
- [ ] Implement `summarize_decomposition_results`.

## Task 3: x5 Runner

- [ ] Load existing feature cache, prices, and lagged BVPS source data.
- [ ] Build x1 decomposition targets.
- [ ] Produce BVPS-leg predictions with selected x4 methods.
- [ ] Produce P/B-leg predictions with selected x5 methods.
- [ ] Join selected pairs into recombined implied-price metrics.
- [ ] Write deterministic x5 artifacts and memo.

## Production Boundary

x5 is research-only. It must not edit `scripts/monthly_decision.py`,
production configuration, monthly output artifacts, or existing `v###`
research-lane artifacts.
