# v7 Enhancement Session Progress

## Overview

Enhancements defined in `claude-v7-plan.md` (uploaded 2026-04-02).  
Working branch: `claude/v7-plan-enhancements-WSf9L`  
Baseline: v6.5 (984 passed, 1 skipped)

**Note:** The plan file (`claude-v7-plan.md`) is truncated at line 508,
cutting off mid-description of v7.2.  v7.0 and v7.1 are fully specified.
v7.2–v7.4 are partially specified; implement conservatively from what is
described.

---

## Version Status

| Version | Theme | Status | Session |
|---------|-------|--------|---------|
| v7.0 | Feature Ablation Backtest | ✅ Complete | 1 |
| v7.1 | Three-Scenario Tax Framework | ✅ Complete | 1 |
| v7.2 | EDGAR 8-K Parser Hardening | ✅ Complete | 2 |
| v7.3 | Monthly Report Tax Section + Decision Log Fix | 🔲 Pending | — |
| v7.4 | CPCV Path Stability Guard + Obs/Feature Ratio | 🔲 Pending | — |

---

## Session 1 — 2026-04-02

### Completed

#### v7.0 — Feature Ablation Backtest

**Files created:**
- `scripts/feature_ablation.py` — Standalone ablation script.  Iterates
  five cumulative feature groups (A–E) across configurable ETF benchmarks
  and model types (elasticnet, gbt), outputs CSV to `results/backtests/`.
- `tests/test_feature_ablation.py` — 10 tests covering group structure,
  CSV output, CLI, and filtering logic.

**Key design decisions:**
- Feature groups are cumulative (each group = all columns from prior groups
  plus new additions).  The script filters to only columns present in the
  actual DataFrame, so missing columns are silently skipped rather than
  raising errors.
- The script uses `get_X_y_relative` (not `get_X_y`) because every WFO
  model in this project is trained on relative return vs. an ETF benchmark.
- GBT model is included alongside ElasticNet as specified.
- If a WFO call raises ValueError (insufficient data), that (group, bench,
  model) row is recorded with NaN metrics rather than crashing.

**Acceptance criteria met:**
- [x] `python scripts/feature_ablation.py --benchmarks VTI --horizons 6`
      runs to completion and produces a CSV.
- [x] All 10 tests pass.
- [x] Net-negative groups are reported with a recommendation to add to
      `config.FEATURES_TO_DROP`.

#### v7.1 — Three-Scenario Tax Framework

**Files modified:**
- `src/tax/capital_gains.py` — Added `TaxScenario`, `ThreeScenarioResult`
  dataclasses; `compute_stcg_ltcg_breakeven()` and `compute_three_scenarios()`
  functions.
- `src/portfolio/rebalancer.py` — Added `three_scenario` optional field
  to `VestingRecommendation`; added `ThreeScenarioResult` import.

**Files created:**
- `tests/test_three_scenario_tax.py` — 19 tests covering breakeven math,
  all three scenarios, edge cases, and recommendation logic.

**Key design decisions:**
- `TaxScenario.probability = 1.0` for Scenario A (sell now — certain outcome).
- Scenario C is degenerate (probability=0.0) when predicted_6m_return ≥ 0.
- Utility = probability × net_proceeds; max utility selects recommendation.
- `three_scenario` field on `VestingRecommendation` is optional (None by
  default) for full backward compatibility with existing code.

**Acceptance criteria met:**
- [x] Import works from `src.tax.capital_gains`.
- [x] All 19 tests pass.
- [x] `compute_stcg_ltcg_breakeven()` ≈ 0.2125 with default config.
- [x] `VestingRecommendation.three_scenario` is optional (backward compatible).

---

---

## Session 2 — 2026-04-02

### Completed

#### v7.2 — EDGAR 8-K Parser Hardening

**Files modified:**
- `scripts/edgar_8k_fetcher.py` — Three defensive layers added:

  1. **`_validate_parsed_record()`** (new module-level function, ~70 lines):
     - CR cross-check: nullifies `combined_ratio` when `|CR - (LR+ER)| > 5pp`
     - NPW segment check: warns when total < 90% of agency+direct+commercial+property sum
     - PIF floor: nullifies `pif_total < 100,000`
     - EPS range: nullifies `eps_basic` outside `[-5.0, 15.0]`
     - Called in `fetch_and_upsert()` immediately after `_parse_html_exhibit()`.
       If both CR and PIF are nullified, the filing is skipped entirely.

  2. **Most-complete-wins deduplication** (replaces last-wins):
     - Inline `_completeness()` counts non-None fields (excluding `month_end`).
     - On ties, last-seen still wins (preserves existing behavior).
     - Dedup condition: `>= _completeness(seen[me])`.

  3. **Zero-new-data alerting**:
     - After upsert, queries `COUNT(DISTINCT month_end)` from the DB.
     - Logs a WARNING when `n_upserted <= existing_months`, signaling a
       possible filing format change.

**Files created:**
- `tests/test_edgar_validation.py` — 14 tests: 11 for `_validate_parsed_record()`
  (CR consistent/inconsistent/missing, PIF floor, EPS range, NPW segment warning,
  boundary cases), 2 for dedup logic, 1 for both-nullified skip condition.

**Key design decisions:**
- `_validate_parsed_record` is a standalone function (not nested) so it can be
  imported and tested directly.
- Dedup uses `>=` not `>` to preserve last-wins-on-tie semantics matching the
  spec's "existing behavior for ties" requirement.
- The zero-new-data warning fires on `n <= existing_count`, not on `n == 0`,
  to catch the case where an existing month is merely re-upserted with no net
  new coverage.

**Acceptance criteria met:**
- [x] All 14 tests pass.
- [x] No new regressions (1002 passed / 30 pre-existing failures unchanged).

---

## Next Session — Pick Up at v7.3

### v7.3 — Monthly Report Tax Section + Decision Log Fix

Partially described in the plan.  Requires:
- Adding the three-scenario tax analysis output to `scripts/monthly_decision.py`
- Fixing `decision_log.md` (duplicate dry-run entries, malformed table rows)

### v7.4 — CPCV Path Stability Guard + Obs/Feature Ratio

Partially described.  Requires:
- Adding a guard when CPCV positive path fraction is borderline
- Enforcing minimum obs/feature ratio before adding new features to the pipeline
