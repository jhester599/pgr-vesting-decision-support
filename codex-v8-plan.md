# Codex v8 Plan and Implementation Record

## Summary

This document records the v8 enhancement plan requested on 2026-04-02 and the
work completed against that plan. The starting point for v8 was a local checkout
that lagged GitHub `master`, a checked-in SQLite database that still reflected a
recent-only `pgr_edgar_monthly` window, and documentation that stopped at v6.5
even though v7 work had already been merged upstream.

GitHub `master` on 2026-04-02 was treated as the authoritative baseline. That
baseline includes `claude-v7-plan.md`, `SESSION_PROGRESS.md`, and the merged
v7.0-v7.4 implementation.

## Review Findings

- The local checkout was behind `origin/master` by the merged v6.5/v7.x work.
- `v7.0` through `v7.4` were already complete upstream and should be preserved,
  not reimplemented.
- The checked-in `data/pgr_financials.db` had only 22 recent `pgr_edgar_monthly`
  rows before v8 remediation, despite a committed 256-row historical CSV backfill.
- Full `pytest` failed locally because the AV loader tests required a real
  `AV_API_KEY` even when `requests.get` was mocked.
- Generated monthly artifacts were stale relative to the current codebase and
  still carried an old model-version label.

## v7 Status Verification

Verified from the merged upstream baseline and `SESSION_PROGRESS.md`:

- `v7.0` complete: feature ablation backtest.
- `v7.1` complete: three-scenario tax framework.
- `v7.2` complete: EDGAR 8-K parser hardening.
- `v7.3` complete: monthly report tax section and decision-log fix.
- `v7.4` complete: CPCV path stability guard and obs/feature ratio checks.

## Implemented v8 Work

### v8.0 — Baseline Reconciliation

- Fast-forwarded the local worktree to GitHub `origin/master`.
- Preserved the merged v7.0-v7.4 implementation as the new baseline.
- Added this `codex-v8-plan.md` record.

### v8.1 — Test and Ingestion Stability

- Updated `src/ingestion/multi_ticker_loader.py` and
  `src/ingestion/multi_dividend_loader.py` so mocked HTTP calls can execute
  without a real `AV_API_KEY`.
- Kept runtime validation for real network calls: a missing key still raises at
  execution time when `requests.get` is not mocked.
- Added/retained regression coverage through the existing loader test suite.

### v8.2 — Data Parity and Reproducibility

- Added `get_db_health_report()` and `warn_if_db_behind()` to
  `src/database/db_client.py`.
- Wired startup DB health warnings into `scripts/bootstrap.py` and
  `scripts/monthly_decision.py`.
- Loaded the committed historical CSV into the checked-in database via:
  `python scripts/edgar_8k_fetcher.py --load-from-csv`
- Verified the backfilled DB is now clean against the committed CSV baseline.
- Generated a fresh live monthly report:
  `python scripts/monthly_decision.py --as-of 2026-04-02 --skip-fred`

### v8.3 — Monthly Reporting Refresh

- Updated the recommendation metadata label in `scripts/monthly_decision.py` to
  reflect the current pipeline state (`v8.7` after the feature-set tuning work).
- Confirmed the v7.3 tax-context section remains active in fresh report output.
- Added a new committed monthly artifact set under `results/monthly_decisions/2026-04/`.

### v8.4 — Governance Baseline

- Verified the merged v7.4 guardrails remain the active implementation baseline.
- Left CPCV/obs-feature guard logic intact and documented as complete.

### v8.5 — Documentation and Workflow Refresh

- Removed the stale `FMP_API_KEY` export from `.github/workflows/monthly_decision.yml`.
- Refreshed top-level docs (`README.md`, `ROADMAP.md`, `DEVELOPMENT_PLAN.md`)
  so they point readers to the current v7/v8 state instead of stopping at v6.5.
- Replaced CP1252-hostile log arrows in `scripts/edgar_8k_fetcher.py` with ASCII.

### v8.6 — ElasticNet / GBT Feature-Set Tuning

- Completed the v7.0 ablation execution and saved outputs under
  `results/backtests/feature_ablation_20260402.csv`.
- Ran focused ElasticNet follow-up tests over Group E additions and selected a
  production ElasticNet feature set of Group B plus
  `investment_income_growth_yoy`, `roe_net_income_ttm`, and
  `underwriting_income`.
- Confirmed GBT performs best with the lean Group B macro regime set and should
  not inherit the larger EDGAR-heavy stacks.

### v8.7 — Full Ensemble Model-Specific Feature Sets

- Extended the final head-to-head tuning to Ridge and BayesianRidge.
- Ridge and BayesianRidge both performed best with a small Group C + Group E
  hybrid: Group B plus `combined_ratio_ttm`,
  `investment_income_growth_yoy`, and `roe_net_income_ttm`.
- Added `MODEL_FEATURE_OVERRIDES` in `config.py` and enforced those subsets in
  `run_wfo()` and `predict_current()` so backtests and live predictions use the
  same per-model feature definitions.
- Added regression coverage for the new model-specific selection behavior.

## Validation

- Targeted regression run:
  `python -m pytest -q tests/test_multi_ticker_loader.py tests/test_db_health.py`
- Full suite should be run after the final doc/code refresh to confirm that the
  entire repository remains green under the updated baseline.

## Current State After v8

- Local checkout matches the merged GitHub baseline from 2026-04-02 plus the
  v8 remediation described here.
- `pgr_edgar_monthly` in the checked-in DB now reflects the committed historical
  CSV backfill rather than only the recent live-fetch window.
- Startup scripts warn when the DB falls behind the documented schema/data baseline.
- The latest committed monthly output is `results/monthly_decisions/2026-04/`.
- The live ensemble now uses model-specific production feature sets rather than
  a single shared feature block across all four models.
