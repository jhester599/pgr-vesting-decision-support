# x2 Absolute Direction Classification Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first x-series modeling step: research-only multi-horizon
absolute PGR direction classification.

**Architecture:** Reuse x1 target construction, the existing processed feature
matrix, and existing TimeSeriesSplit-based WFO helpers. Keep all modeling under
`src/research/`, `scripts/research/`, `tests/`, and `results/research/`. Do not
wire x2 into production monthly decision outputs or shadow artifacts.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn logistic regression
and shallow histogram GBT, existing WFO split helpers. No K-Fold CV and no
full-sample scaling.

---

## Scope

x2 tests whether regularized logistic classification can predict whether PGR's
future absolute return is positive at +1m, +3m, +6m, and +12m.

The first x2 baseline compares:

- `logistic_l2_balanced`: L2 logistic with fold-local `StandardScaler`.
- `hist_gbt_depth2`: shallow histogram GBT challenger.
- `base_rate`: fold-local historical positive-rate probability.
- `always_up`: naive majority-style long-equity baseline.

This does not include x3 return regression, x4/x5 decomposition modeling, x6
special-dividend modeling, or x7 TA expansion.

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/superpowers/plans/2026-04-22-x2-absolute-classification-baseline.md` | Create | x2 plan |
| `src/research/x2_absolute_classification.py` | Create | Research-only classifier evaluation utilities |
| `scripts/research/x2_absolute_classification.py` | Create | Artifact-producing x2 experiment runner |
| `tests/test_research_x2_absolute_classification.py` | Create | WFO, baseline, and artifact-summary tests |
| `results/research/x2_absolute_classification_detail.csv` | Create | Per-horizon/model metrics |
| `results/research/x2_absolute_classification_summary.json` | Create | Ranked x2 baseline summary |
| `results/research/x2_research_memo.md` | Create | Human-readable x2 memo |

## Task 1: Tests First

- [ ] Test that the evaluator emits chronological OOS probabilities with no
      train/test overlap and respects the requested horizon gap through existing
      WFO helpers.
- [ ] Test that logistic classification uses a fold-local pipeline and returns
      bounded probabilities.
- [ ] Test fold-local base-rate and always-up baselines.
- [ ] Test the summary ranking prefers higher balanced accuracy and lower Brier
      score.

## Task 2: Classifier Utilities

- [ ] Implement `build_absolute_classifier_pipeline`.
- [ ] Implement fold-local median imputation using training folds only.
- [ ] Implement `evaluate_absolute_classifier`.
- [ ] Implement `evaluate_absolute_baseline`.
- [ ] Implement `summarize_absolute_classification_results`.

## Task 3: x2 Runner

- [ ] Load the existing processed feature matrix without refreshing it.
- [ ] Load PGR prices from the checked-in SQLite database.
- [ ] Build x1 forward direction targets for horizons 1, 3, 6, and 12.
- [ ] Evaluate baseline and model rows for each horizon.
- [ ] Write deterministic detail CSV, summary JSON, and memo under
      `results/research/`.

## Task 4: Verification

- [ ] Run `python -m pytest tests/test_research_x2_absolute_classification.py -q --tb=short`.
- [ ] Run `python scripts/research/x2_absolute_classification.py`.
- [ ] Run focused x1/x2 tests together.

## Production Boundary

x2 is research-only. It must not edit `scripts/monthly_decision.py`, production
configuration, monthly output artifacts, or existing `v###` research lane
artifacts.
