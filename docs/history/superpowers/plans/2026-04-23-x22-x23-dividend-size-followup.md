# x22-x23 Dividend Size Follow-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the x21 dividend-size result is genuinely useful or
mostly a target-scaling artifact, then package the surviving dividend-lane
evidence for the broader x-series.

**Architecture:** x22 keeps the post-policy annual size-only frame from x21 but
adds stronger low-complexity baselines so we can separate target-scale gains
from feature-driven gains. x23 synthesizes x18-x22 into one research-only
recommendation and packages any surviving signal for later dashboard/monthly
indicator discussion.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn ridge, existing x21
size-target helpers, checked-in research artifacts only.

---

## Scope

- x22 asks whether `special_dividend_excess / current_bvps` still looks best
  after comparing it against:
  - prior-positive-year carry-forward
  - short trailing mean / median normalized baselines
  - the existing historical mean and ridge rows
- x23 records what the dividend lane now really is:
  - occurrence remains underidentified post-policy
  - size appears more learnable than occurrence
  - normalized-to-current-BVPS is the leading size target unless x22 disproves it

## Decision Log

- **Which targets survive into x22?** Keep only the best x21 scales plus raw
  dollars as anchor. Criterion: avoid spreading the tiny annual sample too thin.
- **How many baselines?** Add only low-complexity annual baselines with clear
  economic meaning. Criterion: each baseline should answer a distinct question,
  not duplicate another row.
- **What gets packaged in x23?** Only a research-only signal or memo, not a
  production contract. Criterion: occurrence is not identifiable enough yet for
  deployment-adjacent packaging.

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/superpowers/plans/2026-04-23-x22-x23-dividend-size-followup.md` | Create | Plan |
| `src/research/x22_dividend_size_baselines.py` | Create | x22 helpers |
| `scripts/research/x22_dividend_size_baselines.py` | Create | x22 runner |
| `tests/test_research_x22_dividend_size_baselines.py` | Create | x22 tests |
| `src/research/x23_dividend_lane_package.py` | Create | x23 helpers |
| `scripts/research/x23_dividend_lane_package.py` | Create | x23 runner |
| `tests/test_research_x23_dividend_lane_package.py` | Create | x23 tests |
| `results/research/x22_*` | Create | x22 artifacts |
| `results/research/x23_*` | Create | x23 artifacts |

## Task 1: x22 Tests First

- [ ] Test the prior-positive-year normalized baseline uses only earlier annual
      positives.
- [ ] Test x22 ranks rows by dollar MAE after back-transformation.
- [ ] Test x22 includes raw-dollar anchor rows alongside surviving normalized
      targets.

## Task 2: x22 Implementation

- [ ] Build bounded x22 candidate targets from x21 findings.
- [ ] Add carry-forward and trailing-normalized baselines.
- [ ] Write x22 detail CSV, summary JSON, and memo.

## Task 3: x23 Tests First

- [ ] Test x23 marks occurrence as underidentified when overlap positive rate is
      one-class.
- [ ] Test x23 promotes the dividend lane only as a research-size indicator, not
      as a production candidate.

## Task 4: x23 Implementation

- [ ] Read x18, x20, x21, and x22 artifacts.
- [ ] Synthesize the dividend lane into one recommendation payload.
- [ ] Write x23 package JSON, memo, and peer-review prompt.

## Verification

- [ ] Run x22/x23 unit tests.
- [ ] Run x22/x23 scripts and validate JSON artifacts.
- [ ] Run focused dividend-lane tests spanning x18-x23.
- [ ] Run `py_compile` on all new modules, scripts, and tests.
