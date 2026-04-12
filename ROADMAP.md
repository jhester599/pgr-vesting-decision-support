# PGR Vesting Decision Support - Roadmap

For completed work and release history, see [CHANGELOG.md](CHANGELOG.md).

## Current State

**Master baseline: v86** - the live monthly workflow now uses the promoted
quality-weighted consensus on top of the `v38` shrinkage-calibrated prediction
stack, with the adoption-and-contract cleanup completed across workflow,
dashboard, email, and docs.

**Current operating posture**

- production recommendation path: quality-weighted consensus
- equal-weight comparison: retained in diagnostic artifacts only
- monthly artifacts now include `benchmark_quality.csv`,
  `consensus_shadow.csv`, `dashboard.html`, and `monthly_summary.json`
- deferred but still promising research branches:
  - `v70` per-benchmark shrinkage
  - `v46` classification
  - `v73` hybrid decision gating

## Near-Term Backlog

### Post-Promotion Monitoring

- run the next clean monthly report on current production
- keep checking that the equal-weight diagnostic path remains redundant before
  removing the underlying artifact entirely

### Structured Output Adoption

- continue moving secondary consumers toward `monthly_summary.json` instead of
  scraping markdown for top-level fields
- decide whether additional automation or notification surfaces should read the
  summary payload directly

### Classification-Led Decision Research

- deepen the `v46` classification branch rather than treating it as a single
  promising result
- compare per-benchmark classifiers, pooled panel classifiers, and hybrid
  classifier + regression policies
- use [2026-04-11-v87-v96-classification-hybrid-research.md](docs/superpowers/plans/2026-04-11-v87-v96-classification-hybrid-research.md)
  as the active planning reference for this work
- that plan now explicitly incorporates archived peer-review guidance on:
  - composite / basket targets
  - pooled vs separate benchmark structures
  - lean feature-set discipline
  - calibration-first decision gating
- current outcome from the completed `v87-v96` cycle:
  - no production promotion
  - most promising follow-on is a classifier-only benchmark-panel shadow path
    using the `actionable_sell_3pct` target and prequentially calibrated
    probabilities
- current outcome from the completed `v102-v117` cycle:
  - no classifier promotion
  - strongest constrained gate candidate is the Gemini-style veto overlay
  - next active phase is prospective shadow monitoring in
    [2026-04-11-v118-v121-prospective-shadow-monitoring.md](docs/superpowers/plans/2026-04-11-v118-v121-prospective-shadow-monitoring.md)

## Strategic Backlog

| Item | Description |
|---|---|
| Structured monthly schema follow-through | Expand `monthly_summary.json` usage into future automations and non-markdown consumers |
| Secondary calibration branch | Revisit `v70` if benchmark-specific variance control becomes more important than raw hit rate |
| Decision-layer follow-on | Revisit `v46` + `v73` as a policy-layer abstention or gating branch after the current production path settles |
| Cross-check retirement cleanup | Decide when the equal-weight diagnostic artifact itself can be de-emphasized further or archived |

## Development Principles

- Never finalize a module without a passing pytest suite.
- No K-Fold cross-validation - `TimeSeriesSplit` with purge/embargo only.
- No `StandardScaler` across the full dataset prior to temporal splitting.
- No `yfinance` for fundamentals or historical ratios.
- Python 3.10+, strict PEP 8, standard type hints.

## Monthly Decision Log

See [`results/monthly_decisions/decision_log.md`](results/monthly_decisions/decision_log.md)
for the persistent record of monthly recommendations.
