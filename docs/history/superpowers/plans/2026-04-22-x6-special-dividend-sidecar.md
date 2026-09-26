# x6 Special-Dividend Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the x6 research-only two-stage Q1 special-dividend forecasting sidecar.

**Architecture:** Reuse x1 annual November snapshot targets and the existing processed feature matrix. Evaluate a conservative annual temporal framework: stage 1 predicts whether a Q1 special dividend occurs, stage 2 predicts excess dollars/share conditional on occurrence, and expected value combines probability times conditional size.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn LogisticRegression and Ridge, existing SQLite dividend/EDGAR/price data. Strict temporal annual splits only; no K-Fold CV and no production wiring.

---

## Scope

x6 treats special dividends as an annual capital-allocation problem, not another
monthly return target.

The first x6 benchmark compares:

- Stage 1 occurrence:
  - `historical_rate`
  - `logistic_l2_balanced`
- Stage 2 size conditional on occurrence:
  - `historical_positive_mean`
  - `ridge_positive_excess`
- Expected value:
  - `stage1_probability * stage2_conditional_size`

x6 does not wire predictions into monthly decisions, shadow artifacts, or the
main v-series line.

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/superpowers/plans/2026-04-22-x6-special-dividend-sidecar.md` | Create | x6 plan |
| `src/research/x6_special_dividend.py` | Create | Annual temporal two-stage utilities |
| `scripts/research/x6_special_dividend.py` | Create | Artifact-producing x6 runner |
| `tests/test_research_x6_special_dividend.py` | Create | Snapshot, stage, EV, and summary tests |
| `results/research/x6_special_dividend_detail.csv` | Create | Annual OOS prediction rows |
| `results/research/x6_special_dividend_summary.json` | Create | x6 metrics summary |
| `results/research/x6_research_memo.md` | Create | Human-readable x6 memo |

## Task 1: Tests First

- [ ] Test expanding annual splits use only prior years.
- [ ] Test stage-1 historical rate is fold-local.
- [ ] Test stage-2 historical positive mean is conditional on prior positives.
- [ ] Test expected value equals probability times conditional size.
- [ ] Test summary metrics report stage-1, stage-2, and expected-value errors.

## Task 2: Utilities

- [ ] Implement annual expanding split generation.
- [ ] Implement fold-local imputation for annual features.
- [ ] Implement `evaluate_special_dividend_two_stage`.
- [ ] Implement `summarize_special_dividend_results`.

## Task 3: x6 Runner

- [ ] Build November annual snapshots with x1 target utilities.
- [ ] Add conservative annual feature subset from the existing feature matrix.
- [ ] Evaluate baseline and model two-stage rows.
- [ ] Write deterministic CSV, JSON, and memo artifacts.

## Production Boundary

x6 is research-only. It must not edit `scripts/monthly_decision.py`,
production configuration, monthly output artifacts, or existing `v###`
research-lane artifacts.
