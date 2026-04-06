# v34 Completion Pass — Peer-Review Remaining Items

**Branch:** `codex/v34-completion`
**Date:** 2026-04-06
**Predecessor:** v33 (code-quality split + mypy expansion, merged PR #61)

## Context

All Tier 1–4 core peer-review items are complete.  The v34 sequence
closes the remaining open items: BL fallback surfacing (Tier 1.4,
partial), exception-context sweep (Tier 3.3, partial), SQLite out of
git (Tier 5.1, not started), and research script archival (Tier 5.2,
not started).  The heavier Tier 4.5 Monte Carlo work and Tier 5.5
property testing are scoped to separate branches (v35, v36) because
they require independent review gates.

---

## v34.0 — Surface Black-Litterman Fallback in Monthly Report (Tier 1.4)

**Scope:** The `src/portfolio/black_litterman.py` module already sets
`BLDiagnostics.fallback_used` in every exit path, but the monthly
decision pipeline never calls `build_bl_weights`.  This version wires
BL into the pipeline as a diagnostic-only shadow call so that the report
can display optimizer health.

### Changes

**`scripts/monthly_decision.py`:**
- Add `from src.portfolio.black_litterman import BLDiagnostics,
  build_bl_weights` import
- Add `_load_etf_monthly_returns(conn, tickers, end_date)` helper that
  queries `daily_prices` for all ETF benchmark tickers, resamples to
  month-end close, and computes pct_change
- In `main()` between Steps 3 and 4: call `build_bl_weights(
  ensemble_results, etf_returns_df, return_diagnostics=True)` wrapped
  in a try/except; store result as `bl_diagnostics: BLDiagnostics | None`
- Pass `bl_diagnostics` to `_write_recommendation_md()`

**`scripts/monthly_decision.py` — `_write_recommendation_md()`:**
- Accept `bl_diagnostics: BLDiagnostics | None = None`
- Append a "## Portfolio Optimizer Status" section to recommendation.md
  below the redeploy portfolio section:
  - Shows ✅ and optimizer stats when `not fallback_used`
  - Shows ⚠️ and the `fallback_reason` when `fallback_used`
  - Shows "—" (not run) when `bl_diagnostics is None`

**`tests/test_bl_fallback_monthly.py` (new):**  Six tests:
1. Helper returns DataFrame with expected shape for mock tickers
2. Helper returns empty DataFrame when prices table is empty
3. Section appears in recommendation.md when `fallback_used=False`
4. Section appears with ⚠️ when `fallback_used=True`
5. Section absent (graceful skip) when `bl_diagnostics is None`
6. E2E stub still passes after new parameter added

---

## v34.1 — Exception Logging Sweep (Tier 3.3 Completion)

**Scope:** 13 remaining `except Exception` blocks across 6 files lack
traceback context.  Add `logger.exception()` or `exc_info=True` to
each so that failures are traceable without changing fallback behavior.

### Files

- `src/ingestion/edgar_8k_fetcher.py` (lines 155, 317)
- `src/processing/feature_engineering.py` (line 1789)
- `src/reporting/run_manifest.py` (line 40)
- `src/research/evaluation.py` (line 642)
- `src/visualization/plots.py` (line 66)
- `scripts/edgar_8k_fetcher.py` (lines 265, 411, 1368)
- `scripts/monthly_decision.py` (lines 280, 1370, 2655, 2706)

Each block will receive either `logger.exception(...)` (for ERROR-level)
or `logger.warning(..., exc_info=True)` (for WARNING-level graceful
fallback paths) as appropriate.

**Tests:** Run full test suite; no new tests required (the change is
pure logging context, not behavior).

---

## v34.2 — SQLite Database Out of Git (Tier 5.1)

**Scope:** Stop tracking `data/pgr_financials.db` in git.  The file is
4.5MB binary, grows with every monthly run, and is fully reconstructable
via `scripts/bootstrap.py`.

### Changes

- Add `data/pgr_financials.db` and `data/*.db` to `.gitignore`
- Remove the DB from git tracking: `git rm --cached data/pgr_financials.db`
- Update `README.md` quick-start section with a "Database setup" step:
  ```
  python scripts/bootstrap.py  # rebuilds DB from CSV + API on fresh clone
  ```
- Add a note in `docs/architecture.md` that the DB is gitignored and
  rebuilt via bootstrap

**Note:** The DB remains in git *history* (pre-v34.2 commits) but new
commits will no longer track it.  The CI `--dry-run` tests use an
in-memory SQLite DB so CI is unaffected.

---

## v34.3 — Archive Completed Research Scripts (Tier 5.2, Limited Scope)

**Scope:** The peer review recommends archiving completed one-time study
modules.  However, `src/research/v11.py`–`v24.py` are imported by
`scripts/monthly_decision.py` and other production scripts and cannot be
safely moved without a larger refactor.  This version archives only the
completed standalone *study scripts* (the `scripts/v11_*.py` through
`scripts/v24_*.py` family) which have no callers.

### Changes

- Create `archive/scripts/` directory
- Move `scripts/v11_autonomous_loop.py`, `scripts/v12_shadow_study.py`,
  `scripts/v14_prediction_layer_study.py`, `scripts/v15_execute.py`,
  `scripts/v15_feature_replacement_setup.py`,
  `scripts/v16_promotion_study.py`, `scripts/v17_shadow_gate.py`,
  `scripts/v18_bias_reduction_study.py`,
  `scripts/v19_feature_completion.py`, `scripts/v20_synthesis_study.py`,
  `scripts/v21_historical_comparison.py`,
  `scripts/v22_cross_check_promotion.py`,
  `scripts/v23_extended_history_proxy_study.py`,
  `scripts/v24_vti_replacement_study.py` to `archive/scripts/`
- Update the status document: note that the `src/research/v11-v24`
  modules remain in place pending a future dedicated refactor that
  promotes their utility functions to proper production modules

---

## v34.4 — Status Refresh

- Update `docs/history/repo-peer-reviews/2026-04-05/
  claude_opus_peer_review_status_20260405.md`:
  - Tier 1.4 → Completed
  - Tier 3.3 → Completed
  - Tier 5.1 → Completed
  - Tier 5.2 → Partial (study scripts archived; src/research/ refactor deferred)
  - Version mapping: add v34.0–v34.3
  - PR list: add PR #62

---

## Version Numbering

| Step | Label | Tier |
|------|-------|------|
| v34.0 | BL fallback surface in recommendation.md | 1.4 |
| v34.1 | Exception logging context sweep | 3.3 |
| v34.2 | SQLite database out of git | 5.1 |
| v34.3 | Archive completed study scripts | 5.2 |
| v34.4 | Status refresh | — |
