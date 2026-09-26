# x8 Synthesis Ranking Memo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the x8 research-only synthesis layer that ranks x2-x7
evidence and records a shadow-readiness recommendation without changing
production behavior.

**Architecture:** Read checked-in x-series research summaries, normalize
the best per-horizon rows into a compact comparison table, compute simple
gate counts, and write deterministic JSON plus Markdown artifacts. x8 does
not train models, refresh source data, or wire anything into monthly
decisions.

**Tech Stack:** Python 3.10+, pandas, json, pathlib, existing
`results/research` artifacts.

---

## Scope

x8 answers three repo-grounded questions:

- Which modeling path currently has the best evidence by horizon?
- Which paths beat their own conservative baselines or gates?
- Is any x-series path ready for shadow wiring now?

The expected recommendation is conservative: continue research, do not create
shadow artifacts yet. BVPS forecasting is the strongest structural leg, x7
targeted TA is a bounded follow-up for 3m/6m classification, x3 direct returns
remain mostly baseline-dominated, x5 decomposition is promising only with
no-change P/B, and x6 special dividends remain annual small-sample research.

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/superpowers/plans/2026-04-23-x8-synthesis-ranking-memo.md` | Create | x8 plan |
| `src/research/x8_synthesis.py` | Create | Normalization and recommendation utilities |
| `scripts/research/x8_synthesis.py` | Create | Deterministic x8 artifact writer |
| `tests/test_research_x8_synthesis.py` | Create | Unit tests for x8 synthesis |
| `results/research/x8_synthesis_summary.json` | Create | Machine-readable synthesis |
| `results/research/x8_synthesis_memo.md` | Create | Human-readable closeout memo |

## Task 1: Tests First

- [x] Test that horizon leaders are selected by rank within each horizon.
- [x] Test that gate counts count true values and observed horizons separately.
- [x] Test that shadow readiness stays `not_ready` when evidence is mixed or
      annual sample size is very small.
- [x] Test that JSON records replace non-finite and missing values with nulls.

## Task 2: Utility Module

- [x] Implement `json_records`.
- [x] Implement `extract_horizon_leaders`.
- [x] Implement `count_gate_successes`.
- [x] Implement `build_shadow_readiness`.
- [x] Implement `build_x8_summary`.

## Task 3: Artifact Runner

- [x] Read x2-x7 checked-in summary JSON files.
- [x] Write `x8_synthesis_summary.json`.
- [x] Write `x8_synthesis_memo.md`.
- [x] Include explicit production/shadow boundary flags.

## Task 4: Verification

- [x] Run x8 unit tests.
- [x] Run the x8 artifact writer and validate JSON parsing.
- [x] Run focused x-series tests touching x7/x8 synthesis utilities.

## Production Boundary

x8 is research-only. It must not edit `scripts/monthly_decision.py`,
production configuration, monthly output artifacts, shadow artifacts, or
existing `v###` research-lane artifacts.
