# x7 Targeted TA Feature Follow-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an x-series research-only targeted TA follow-up for absolute PGR direction classification.

**Architecture:** Reuse the v160 TA feature factory, x1 absolute direction targets, and x2 WFO classifier utilities. Test only pre-registered replacement candidates from prior TA research; do not dump a broad TA matrix into x-series models.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn through x2 classifier utilities, existing SQLite price data. No K-Fold CV, no full-sample scaling, no production wiring.

---

## Scope

x7 evaluates whether targeted TA replacement variants improve x2-style
absolute-direction classification at +1m, +3m, +6m, and +12m.

Variants:

- `x2_core_baseline`: existing conservative x2 feature set.
- `ta_minimal_replacement`: replace `mom_12m` with `ta_pgr_obv_detrended` and
  `vol_63d` with `ta_pgr_natr_63d`.
- `ta_minimal_plus_vwo_pct_b`: the minimal replacement plus replacing `vix`
  with `ta_ratio_bb_pct_b_6m_vwo`.
- `ta_bollinger_width_probe`: the minimal replacement plus replacing `vix`
  with `ta_ratio_bb_width_6m_voo`.

x7 does not add TA features to the production feature matrix, monthly decision
outputs, shadow artifacts, or the v-series roadmap.

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/superpowers/plans/2026-04-23-x7-targeted-ta-feature-followup.md` | Create | x7 plan |
| `src/research/x7_targeted_ta.py` | Create | Variant, swap, and summary utilities |
| `scripts/research/x7_targeted_ta.py` | Create | Artifact-producing x7 runner |
| `tests/test_research_x7_targeted_ta.py` | Create | Replacement-only and summary tests |
| `results/research/x7_targeted_ta_detail.csv` | Create | Per-horizon variant metrics |
| `results/research/x7_targeted_ta_summary.json` | Create | Ranked x7 summary |
| `results/research/x7_research_memo.md` | Create | Human-readable x7 memo |

## Task 1: Tests First

- [x] Test replacement swaps preserve feature count and order.
- [x] Test candidate variants are replacement-only and bounded.
- [x] Test summary computes per-horizon deltas versus `x2_core_baseline`.
- [x] Test recommendation logic requires both better balanced accuracy and
      better Brier score before marking a TA variant as clearing the x7 gate.

## Task 2: Utilities

- [x] Implement `apply_feature_swaps`.
- [x] Implement `build_x7_ta_variants`.
- [x] Implement `attach_baseline_deltas`.
- [x] Implement `summarize_ta_variants`.

## Task 3: Runner

- [x] Load existing processed feature cache without refreshing it.
- [x] Load existing DB prices for PGR and selected TA benchmarks.
- [x] Build v160 TA features and join them to the x-series feature frame.
- [x] Evaluate x2 logistic classification for each variant and horizon.
- [x] Write deterministic detail CSV, summary JSON, and memo.

## Closeout Verification

- `python scripts/research/x7_targeted_ta.py`
- `python -m pytest tests\test_research_x1_targets.py tests\test_research_x2_absolute_classification.py tests\test_research_x7_targeted_ta.py tests\test_research_v160_ta_features.py -q --tb=short`
- `python -m py_compile src\research\x2_absolute_classification.py src\research\x7_targeted_ta.py scripts\research\x7_targeted_ta.py tests\test_research_x2_absolute_classification.py tests\test_research_x7_targeted_ta.py`

## Production Boundary

x7 is research-only. It must not edit `scripts/monthly_decision.py`,
production configuration, monthly output artifacts, shadow artifacts, or
existing `v###` research-lane artifacts.
