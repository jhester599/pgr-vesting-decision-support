# x13-x14 Adjusted Decomposition Indicator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether dividend-adjusted BVPS improves decomposition at the
promising 3m/6m horizons and, if so, turn that into a research-only indicator
candidate for the monthly report/dashboard.

**Architecture:** Reuse x12 adjusted BVPS targets, x5 decomposition mechanics,
and x9 bridge features. Keep the scope bounded: build an x13 adjusted
decomposition benchmark first, then write an x14 indicator-candidate synthesis
only from checked-in x-series artifacts.

**Tech Stack:** Python 3.10+, pandas, numpy, existing x5/x9/x12 research
utilities, JSON/CSV memo artifacts only.

---

## Scope

x13 exists because x12 showed dividend-adjusted BVPS helps at 3m and 6m but
hurts at 1m and 12m. The goal is not to replace raw BVPS universally; it is to
see whether adjusted BVPS improves the structural price path where x12 says it
might matter.

- x13: compare raw-vs-adjusted BVPS x no-change-P/B decomposition at 3m and 6m,
  plus one bounded adjusted-BVPS challenger set.
- x14: synthesize x3/x5/x8/x11/x12/x13 into a compact indicator-candidate memo,
  with a proposed monthly-report/dashboard readout if the evidence is strong
  enough.

## Decision Log

- **Horizon selection:** Focus x13 on 3m and 6m. Criterion: x12 improved those
  horizons and did not improve 1m/12m.
- **P/B treatment:** Keep `no_change_pb` as the structural anchor unless a
  bounded challenger clearly beats it. Criterion: x5 showed P/B no-change is
  still dominant.
- **Indicator threshold:** x14 only proposes a candidate indicator if the same
  path is directionally coherent with prior x-series evidence. Criterion:
  consistency across multiple research stages beats one-off metric wins.
- **Complexity control:** Prefer replacing weaker decomposition variants with
  adjusted-BVPS variants rather than adding a large new model zoo.

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/superpowers/plans/2026-04-23-x13-x14-adjusted-decomposition-indicator.md` | Create | Plan |
| `src/research/x13_adjusted_decomposition.py` | Create | x13 utilities |
| `scripts/research/x13_adjusted_decomposition.py` | Create | x13 runner |
| `tests/test_research_x13_adjusted_decomposition.py` | Create | x13 tests |
| `src/research/x14_indicator_synthesis.py` | Create | x14 utilities |
| `scripts/research/x14_indicator_synthesis.py` | Create | x14 runner |
| `tests/test_research_x14_indicator_synthesis.py` | Create | x14 tests |
| `results/research/x13_*` | Create | x13 artifacts |
| `results/research/x14_*` | Create | x14 artifacts |

## Task 1: x13 Tests First

- [x] Test adjusted decomposition combines adjusted BVPS with P/B predictions.
- [x] Test x13 comparison separates raw and adjusted decomposition paths.
- [x] Test x13 focuses on 3m and 6m horizons only.

## Task 2: x13 Implementation

- [x] Reuse x12 adjusted BVPS targets and x9 bridge models for 3m/6m.
- [x] Recombine adjusted BVPS predictions with `no_change_pb`.
- [x] Compare raw and adjusted decomposition rows in one summary.
- [x] Write x13 detail CSV, summary JSON, and memo.

## Task 3: x14 Tests First

- [x] Test x14 recommendation stays research-only.
- [x] Test x14 can nominate one indicator candidate with horizon, sign, and
      rationale.
- [x] Test x14 falls back to "no candidate" when evidence is inconsistent.

## Task 4: x14 Implementation

- [x] Read x3/x5/x8/x11/x12/x13 artifacts.
- [x] Compare adjusted decomposition evidence against prior raw paths.
- [x] Build a compact indicator-candidate payload for monthly report/dashboard
      discussion.
- [x] Write x14 summary JSON and memo.

## Verification

- [x] Run x13/x14 unit tests.
- [x] Run x13/x14 scripts and validate JSON artifacts.
- [x] Run focused x-series tests covering x12/x13/x14.
- [x] Run py_compile on all new modules, scripts, and tests.

## Production Boundary

Do not edit `scripts/monthly_decision.py`, production configuration, monthly
decision outputs, shadow artifacts, or any `v###` research-lane artifacts.
