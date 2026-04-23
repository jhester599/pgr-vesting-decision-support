# x12 BVPS Target Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit whether raw BVPS targets are being distorted by capital-return
events and test dividend-adjusted BVPS targets under the x-series WFO setup.

**Architecture:** Build research-only monthly dividend-adjusted BVPS targets,
flag capital-return discontinuity months, and compare raw vs adjusted BVPS
forecasting using a bounded set of x9-style baselines and regularized models.
Keep all work inside `src/research/`, `scripts/research/`, `tests/`, and
`results/research/`.

**Tech Stack:** Python 3.10+, pandas, numpy, existing dividend/EDGAR loaders,
x9 BVPS bridge utilities, existing WFO evaluators, JSON/CSV memo artifacts.

---

## Scope

x12 exists because x9 improved 1m and 3m BVPS forecasting but did not improve
6m/12m versus x4, while the raw monthly BVPS series shows large December/January
drops that line up with large special-dividend months.

x12 will:

- quantify raw BVPS discontinuity months
- create adjusted future-BVPS targets that add back dividends paid during the
  forward window
- compare raw vs adjusted target predictability
- document whether future x13/x14 work should shift to adjusted BVPS targets
  and adjusted BVPS x P/B recombination

## Decision Log

- **Target definition:** Use ex-dividend dates and add cumulative dividends in
  the forward window back to future BVPS. Criterion: this preserves the
  operating capital-generation path while keeping raw BVPS available for audit.
- **Model scope:** Reuse bounded x9 models and baselines instead of opening a
  fresh model zoo. Criterion: the question is target construction first, not
  model novelty first.
- **Seasonality handling:** Explicitly audit December/January windows. Criterion:
  these months show the largest raw BVPS discontinuities in repo data.
- **Promotion boundary:** Even if adjusted targets help, keep results
  research-only. Criterion: target-definition changes should not feed reports or
  production until a later synthesis step.

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/superpowers/plans/2026-04-23-x12-bvps-target-audit.md` | Create | x12 plan |
| `src/research/x12_bvps_target_audit.py` | Create | x12 target and audit utilities |
| `scripts/research/x12_bvps_target_audit.py` | Create | x12 artifact writer |
| `tests/test_research_x12_bvps_target_audit.py` | Create | x12 target and audit tests |
| `results/research/x12_*` | Create | x12 detail, summary, discontinuity, memo |

## Task 1: Tests First

- [ ] Test monthly dividend aggregation to business month-end.
- [ ] Test adjusted future BVPS adds back forward-window dividends.
- [ ] Test discontinuity detection flags large negative BVPS changes with
      nearby dividend months.
- [ ] Test comparison summary distinguishes raw vs adjusted target variants.

## Task 2: Target Audit Utilities

- [ ] Implement `build_monthly_dividend_series`.
- [ ] Implement `build_adjusted_bvps_targets`.
- [ ] Implement `identify_bvps_discontinuities`.
- [ ] Implement `summarize_x12_results`.

## Task 3: x12 Runner

- [ ] Load existing feature matrix, PGR prices, PGR dividends, and PGR monthly
      EDGAR BVPS without refreshing caches.
- [ ] Build x9 bridge features once and reuse them for both raw and adjusted
      target variants.
- [ ] Evaluate a bounded set of baselines and x9 regularized models on raw and
      adjusted targets for 1m/3m/6m/12m.
- [ ] Write detail CSV, summary JSON, discontinuity CSV, and memo.

## Task 4: Verification

- [ ] Run `python -m pytest tests/test_research_x12_bvps_target_audit.py -q --tb=short`.
- [ ] Run `python scripts/research/x12_bvps_target_audit.py`.
- [ ] Run focused x9/x10/x11/x12 tests together.
- [ ] Run py_compile on the new x12 module, script, and tests.

## Production Boundary

Do not edit `scripts/monthly_decision.py`, production configuration, monthly
decision outputs, shadow artifacts, or any `v###` research-lane artifacts.
