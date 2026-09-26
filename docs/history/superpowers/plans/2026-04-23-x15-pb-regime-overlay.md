# x15 P/B Regime Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a simple, bounded P/B regime overlay can improve the
current `no_change_pb` anchor at the medium horizons without expanding the
x-series into a large new model family.

**Architecture:** Reuse x2 WFO classifier mechanics, x5 P/B target framing,
and x13/x14 medium-horizon focus. Build binary up/down hurdle targets on
future P/B change, map fold-local class probabilities back to bounded P/B
shifts, and compare against the existing no-change P/B anchor.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn pipelines already in
the repo, JSON/CSV/memo artifacts only.

---

## Scope

x15 is not a general P/B forecasting reset. It tests one narrow question:
whether a low-complexity regime overlay can improve the 3m and 6m P/B anchor
that still bottlenecks the structural lane.

- Build medium-horizon P/B regime targets using a pre-registered hurdle band.
- Compare no-change P/B against bounded logistic / shallow GBT regime overlays.
- Keep features low-count and focused on valuation anchor, rates/spreads, and
  market-relative context.
- Stay research-only; no monthly report wiring, dashboard changes, production
  updates, or shadow outputs.

## Decision Log

- **Horizon scope:** Focus on 3m and 6m. Criterion: x13 medium-term evidence is
  where adjusted structural decomposition is most interesting.
- **Target design:** Use hurdle-based up/down P/B regime labels around current
  P/B. Criterion: a neutral band is more aligned with noisy multiple dynamics
  than forcing every case into up/down.
- **Overlay design:** Convert classifier probabilities into bounded fold-local
  median P/B shifts. Criterion: keep the overlay interpretable and anchored to
  training history rather than letting a model emit arbitrary P/B levels.
- **Complexity control:** Prefer one combined feature block plus one challenger
  block, not a broad feature sweep.

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/superpowers/plans/2026-04-23-x15-pb-regime-overlay.md` | Create | Plan |
| `src/research/x15_pb_regime_overlay.py` | Create | x15 utilities |
| `scripts/research/x15_pb_regime_overlay.py` | Create | x15 runner |
| `tests/test_research_x15_pb_regime_overlay.py` | Create | x15 tests |
| `results/research/x15_*` | Create | x15 artifacts |

## Task 1: Tests First

- [x] Test regime target construction for up, down, and neutral cases.
- [x] Test bounded overlay mapping from probabilities to predicted P/B.
- [x] Test x15 summary ranking and no-change comparison.

## Task 2: x15 Implementation

- [x] Build future-P/B regime targets for 3m and 6m.
- [x] Reuse x2-style WFO classifier machinery with fold-local preprocessing.
- [x] Map up/down classifier outputs into bounded P/B overlay predictions.
- [x] Write x15 detail CSV, summary JSON, and memo.

## Verification

- [x] Run x15 unit tests.
- [x] Run x15 script and validate JSON artifacts.
- [x] Run focused x-series tests covering x13/x14/x15.
- [x] Run py_compile on new modules, scripts, and tests.

## Production Boundary

Do not edit `scripts/monthly_decision.py`, production configuration, monthly
decision outputs, dashboard code, shadow artifacts, or any `v###` lane files.
