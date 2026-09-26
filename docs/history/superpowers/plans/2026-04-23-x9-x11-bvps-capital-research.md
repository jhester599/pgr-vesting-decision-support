# x9-x11 BVPS Capital Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the x-series BVPS/capital-generation research path before
returning to dividend and price recombination.

**Architecture:** Add research-only bridge features and pre-registered logical
interactions for BVPS forecasting, reuse the resulting capital features in the
annual special-dividend sidecar, then write an x11 synthesis memo. No model is
promoted to production, monthly decision output, or shadow artifacts.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn Ridge/ElasticNet,
existing x1/x4/x5/x6 WFO and annual-split utilities, checked-in SQLite/cache
data only.

---

## Scope

x9-x11 are deliberately focused on the most predictable economic component:
PGR book value per share and the capital-generation path that informs special
dividends.

- x9 improves BVPS forecasting with explicit BVPS lags, accounting bridge
  features, logical interactions, stronger baselines, and regularized feature
  stability summaries.
- x10 upgrades the annual special-dividend sidecar by adding x9-derived
  capital-generation and excess-capital features to November snapshots.
- x11 synthesizes x9/x10 against x4/x6/x8 and records whether the evidence is
  strong enough to proceed toward a later shadow plan.

## Decision Log

When a decision point appears, document the question and the chosen rule:

- **Feature count vs exploration:** Evaluate many candidates inside
  pre-registered feature blocks, but report a low-count stability-ranked
  feature set. Criterion: repeated WFO selection or clear baseline improvement.
- **Interactions:** Include only economically logical interactions, not all
  pairwise combinations. Criterion: each interaction must map to capital
  generation, underwriting scale, valuation-sensitive buybacks, investment
  capital buffer, or dividend-season mechanics.
- **Dividend vs BVPS priority:** Improve BVPS first. Criterion: dividend
  labels are annual and small-sample, while BVPS gives monthly capital-engine
  diagnostics that can become better dividend predictors.
- **Promotion boundary:** Any edge remains research-only. Criterion: no x-series
  result touches production/monthly/shadow files without a later explicit plan.

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/superpowers/plans/...x9-x11...md` | Create | Plan |
| `src/research/x9_bvps_bridge.py` | Create | x9 utilities |
| `scripts/research/x9_bvps_bridge.py` | Create | x9 runner |
| `tests/test_research_x9_bvps_bridge.py` | Create | x9 tests |
| `src/research/x10_dividend_capital.py` | Create | x10 utilities |
| `scripts/research/x10_dividend_capital.py` | Create | x10 runner |
| `tests/test_research_x10_dividend_capital.py` | Create | x10 tests |
| `src/research/x11_capital_synthesis.py` | Create | x11 utilities |
| `scripts/research/x11_capital_synthesis.py` | Create | x11 runner |
| `tests/test_research_x11_capital_synthesis.py` | Create | x11 tests |
| `results/research/x9_*` | Create | x9 artifacts |
| `results/research/x10_*` | Create | x10 artifacts |
| `results/research/x11_*` | Create | x11 artifacts |

## Task 1: x9 Tests First

- [x] Test BVPS bridge features include current BVPS, lagged growth, YTD
      growth, YoY dollar change, and month/season flags without future data.
- [x] Test interaction features are pre-registered and preserve low feature
      count.
- [x] Test trailing-growth and seasonal-drift baselines use only training-fold
      history.
- [x] Test ElasticNet feature stability counts non-zero coefficients by fold.

## Task 2: x9 Implementation

- [x] Implement `build_bvps_bridge_features`.
- [x] Implement `build_bvps_interactions`.
- [x] Implement `build_x9_feature_blocks`.
- [x] Implement x9 baselines: no-change, drift, trailing continuation,
      seasonal drift.
- [x] Implement ElasticNet/Ridge WFO evaluator with fold-local imputation and
      scaling.
- [x] Write x9 detail CSV, summary JSON, stability CSV, and memo.

## Task 3: x10 Tests First

- [x] Test November snapshots include x9 capital-generation features.
- [x] Test predicted capital-generation proxies are lagged to November only.
- [x] Test annual special-dividend evaluation still uses expanding temporal
      splits and no future Q1 data in features.

## Task 4: x10 Implementation

- [x] Build annual November capital frame from x6 targets plus x9 features.
- [x] Evaluate historical/logistic stage 1 and historical/ridge stage 2 rows.
- [x] Add x9 capital feature set as a challenger to the x6 feature set.
- [x] Write x10 detail CSV, summary JSON, and memo.

## Task 5: x11 Tests First

- [x] Test x11 recognizes whether x9 beats x4 by horizon.
- [x] Test x11 keeps dividend confidence low when annual observations remain
      below 30.
- [x] Test x11 recommendation remains research-only.

## Task 6: x11 Implementation

- [x] Read x4/x6/x8/x9/x10 summaries.
- [x] Compare x9 against x4 BVPS results by horizon.
- [x] Compare x10 against x6 special-dividend results.
- [x] Write x11 synthesis JSON and memo with decision questions and criteria.

## Verification

- [x] Run x9/x10/x11 unit tests.
- [x] Run all new scripts and validate JSON artifacts.
- [x] Run focused x-series tests covering x4, x6, x8, x9, x10, and x11.
- [x] Run py_compile on all new modules, scripts, and tests.

## Production Boundary

Do not edit `scripts/monthly_decision.py`, production configuration, monthly
decision outputs, shadow artifacts, or any `v###` research-lane artifacts.
