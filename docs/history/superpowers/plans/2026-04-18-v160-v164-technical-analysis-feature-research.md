# v160-v164 Technical Analysis Feature Research Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research-only broad technical-analysis feature screen for Alpha Vantage-style indicators and evaluate candidate features in both regression and classification models without changing production recommendations.

**Architecture:** Archive the source reports, pre-register the TA candidate inventory, add a pure pandas/numpy TA feature factory, run a broad but pruned screen, confirm capped survivors, and write a go/no-go synthesis. The implementation stays under `src/research/`, `results/research/`, tests, and docs.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn through existing WFO/evaluation utilities. No TA-Lib, pandas-ta, yfinance, K-Fold, or full-sample scaling.

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/archive/history/v160-ta-research-reports/` | Create | Archive external TA research reports with normalized names |
| `src/research/v160_ta_features.py` | Create | Research-only TA indicator feature factory |
| `results/research/v162_ta_broad_screen.py` | Create | Broad-screen inventory and add/replace harness |
| `results/research/v163_ta_survivor_confirm.py` | Create | Survivor cap, correlation pruning, regime labels, candidate payload |
| `results/research/v164_ta_synthesis_summary.md` | Create | Go/no-go synthesis placeholder for the arc |
| `results/research/v164_ta_candidate.json` | Create | Deterministic candidate outcome |
| `docs/closeouts/V164_CLOSEOUT_AND_HANDOFF.md` | Create | Closeout and next-session handoff |
| `tests/test_research_v160_ta_features.py` | Create | Mathematical feature tests |
| `tests/test_research_v162_ta_screen.py` | Create | Inventory/add-replace/delta tests |
| `tests/test_research_v163_ta_confirm.py` | Create | Survivor/pruning/payload tests |

## Task 1: Archive Reports And Pre-Register Direction

- [x] Create `docs/archive/history/v160-ta-research-reports/`.
- [x] Copy the three source reports into that directory as:
  - `v160_claude_ta_research_20260418.md`
  - `v160_gemini_ta_research_20260418.md`
  - `v160_chatgpt_ta_research_20260418.md`
- [x] Add a `README.md` summarizing consensus, disagreement, and repo integration.
- [x] Add roadmap/backlog/changelog documentation for `TA-01`.

## Task 2: TA Feature Factory

- [x] Write tests for RSI, EMA gap, Bollinger `%B`, NATR, detrended OBV, and month-end alignment.
- [x] Implement `src/research/v160_ta_features.py` with pure pandas/numpy indicator calculations.
- [x] Ensure ratio-series features are benchmark-specific and sampled only at completed month-ends.
- [x] Keep production `src/ingestion/technical_loader.py` unchanged because the research module computes `%B` locally from actual close.

## Task 3: Broad Screen Harness

- [x] Write tests for candidate inventory, excluded noisy families, add/replace specs, and baseline deltas.
- [x] Implement `results/research/v162_ta_broad_screen.py`.
- [x] Support regression rows for Ridge and shallow GBT through existing WFO evaluation.
- [x] Support classification rows through existing confirmatory classifier evaluation.
- [x] Include add-one and replacement experiments without fitting all TA features at once.

## Task 4: Survivor Confirmation

- [x] Write tests for survivor caps, correlation pruning, regime labels, and deterministic candidate JSON.
- [x] Implement `results/research/v163_ta_survivor_confirm.py`.
- [x] Enforce at most 6 survivor features and 3 survivor groups.
- [x] Add benchmark-family and regime-slice helper labels.

## Task 5: Synthesis And Closeout

- [x] Create `results/research/v164_ta_synthesis_summary.md`.
- [x] Create `results/research/v164_ta_candidate.json`.
- [x] Create `docs/closeouts/V164_CLOSEOUT_AND_HANDOFF.md`.
- [x] Run focused and required regression verification commands.

## Verification

Run:

```bash
python -m pytest tests/test_research_v160_ta_features.py -q --tb=short
python -m pytest tests/test_research_v162_ta_screen.py -q --tb=short
python -m pytest tests/test_research_v163_ta_confirm.py -q --tb=short
python -m pytest tests/test_feature_engineering.py tests/test_wfo_engine.py tests/test_path_b_classifier.py -q --tb=short
```

Expected: all tests pass. If the empirical v162/v163 harnesses are run later,
they should write deterministic artifacts under `results/research/` for the
current fixed database snapshot.

## Promotion Boundaries

- No production config changes in this arc.
- No live monthly decision-path changes in this arc.
- TA candidates can only become future shadow candidates through a later,
  separate plan after empirical v162/v163 artifacts exist.
