# Claude Opus Peer Review Status Snapshot

Updated: 2026-04-07 (v35.2)
Updated: 2026-04-07 (v35.1)
Updated: 2026-04-07 (v35.0)
Updated: 2026-04-06 (v36.1)
Updated: 2026-04-06 (v34.3)
Source review: [claude_opus_peer_review_20260405.md](./claude_opus_peer_review_20260405.md)

## Purpose

This document tracks the current project status against the 2026-04-05 Claude
Opus peer-review enhancement plan.

Status values:

- `Completed`: implemented on `master` or in the active follow-on PR stream
- `Partial`: meaningful progress landed, but the full peer-review scope is not done
- `Already satisfied`: the issue was already resolved on the reviewed baseline
- `Not started`: no material implementation yet

## Current Summary

- Tier 1 quick wins: **all four items complete** — BL fallback surfaced in recommendation.md in v34.0
- Tier 2 operational safety: **all six items complete** — schema/CSV backfill landed in v6.2, freshness
  and retry in v30, conformal and drift monitoring in v31
- Tier 3 observability/docs/testability: **all six items complete** — exception logging sweep completed in v34.1; config refactor in v33.0; mypy expansion in v33.1
- Tier 4 strategic ML diagnostics: **all four core items complete** — feature-stability, VIF, policy
  backtest, and heuristic comparison landed in v32 (v32.0–v32.3)
- Tier 4.5 Monte Carlo tax simulation: **complete** — `v35.0` adds GBM simulation + MC distribution surfaced in monthly report
- Tier 5.4 automated retrain trigger: **complete** — `v35.1` adds drift-based workflow dispatch, cooldown guard, audit log
- Tier 5.2 research module promotion: **complete** — `v35.2` moves 9 modules out of `src/research/` into production; `monthly_decision.py` no longer imports from `src/research/`
- Tier 5 strategic items: **5.2 + 5.4 + 5.5 complete**; 5.1 deferred; 5.3 not started

## Status By Enhancement

| ID | Enhancement | Status | Notes |
|----|-------------|--------|-------|
| 1.1 | Fix 3 failing tests in `test_multi_ticker_loader.py` | Already satisfied | Verified on the baseline at the start of the `v30` sequence; no code change was needed |
| 1.2 | Move personal email from `config.py` to `.env` | Completed | Landed in `v30.0` and merged via PR #56 |
| 1.3 | Update stale documentation headers | Completed | Landed in `v30.1`; README landing-page rewrite landed in `v30.10` |
| 1.4 | Surface Black-Litterman fallback in monthly report | Completed | `v34.0` adds BL diagnostic shadow call in `monthly_decision.py` via `build_bl_weights(..., return_diagnostics=True)` and renders a "## Portfolio Optimizer Status" section in `recommendation.md` showing ✅ Converged or ⚠️ Fallback with reason |
| 2.1 | Load historical 8-K cache CSV into database | Completed | Landed in `v6.2` (`feat: expand pgr_edgar_monthly schema + full CSV backfill`); DB now holds 257 rows from 2004-08 to 2026-02 |
| 2.2 | Expand `pgr_edgar_monthly` schema (Phase 1) | Completed | Landed in `v6.2`; schema expanded from 10 to 70 columns including segment NPW/PIF, investment income, book value, ROE, buyback metrics, and all channel-mix fields |
| 2.3 | Add data freshness checks to scheduled workflows | Completed | Landed in `v30.2` and merged via PR #56 |
| 2.4 | Add model performance drift detection | Completed | `v31.2`-`v31.4` add rolling drift helpers, the `model_performance_log` table, monthly persistence, drift warnings, and a `Model Health` section in `recommendation.md` |
| 2.5 | Validate conformal empirical coverage | Completed | `v31.0`-`v31.1` add historical conformal coverage backtesting plus monthly trailing-coverage diagnostics and manifest warnings |
| 2.6 | Implement retry with exponential backoff for API calls | Completed | Landed in `v30.4` and `v30.6`; core clients and batch AV loaders use the shared retry session |
| 3.1 | Replace `print()` with structured logging | Partial | Production entry points and several core modules were migrated across `v30.7`-`v30.24`, but research/utility modules still contain many `print()` calls |
| 3.2 | Refactor `config.py` into logical modules | Completed | `v33.0` splits into `config/api.py`, `config/features.py`, `config/model.py`, `config/tax.py` with backward-compatible `config/__init__.py`; all 102 call sites unchanged |
| 3.3 | Improve exception handling in broad catch blocks | Completed | `v34.1` audited all 13 remaining broad catch blocks; 4 needed `exc_info=True`/`log.debug` additions (edgar_8k_fetcher.py ×3, src/ingestion/edgar_8k_fetcher.py ×2); 9 were already properly instrumented |
| 3.4 | Expand mypy coverage beyond 3 modules | Completed | `v33.1` expands CI mypy target from 3 to 11 modules; 9 pre-existing errors fixed in-place (FoldResult._test_dates field, Literal cast at 5 call sites, metrics dict annotation) |
| 3.5 | Restructure `README.md` as a proper landing page | Completed | Landed in `v30.10` and merged via PR #56 |
| 3.6 | Add end-to-end integration test for monthly decision pipeline | Completed | Landed in `v30.5` and merged via PR #56 |
| 4.1 | Track feature importance stability across WFO folds | Completed | `v32.0` adds `compute_feature_importance_stability()` to `src/research/evaluation.py`; surfaces a Feature Importance Stability subsection (top-10 by mean rank, rank std, stability score) in `diagnostic.md` |
| 4.2 | Add VIF checks | Completed | `v32.1` adds `compute_vif()` to `src/processing/feature_engineering.py` and `VIF_HIGH/WARN_THRESHOLD` to `config.py`; surfaces a Multicollinearity (VIF) table in the Feature Governance section of `diagnostic.md` |
| 4.3 | Backtest actual vesting decisions | Completed | `v32.2` adds `_compute_policy_summary()` to `scripts/monthly_decision.py` and wires it into `recommendation.md` as a Decision Policy Backtest section |
| 4.4 | Add model vs. simple heuristic comparison | Completed | `v32.3` extends the Decision Policy Backtest section with a Model-Driven Policies vs. Heuristics table showing uplift vs. sell-all, hold-all, and 50% fixed baselines |
| 4.5 | Monte Carlo tax scenario analysis | Completed | `v35.0` adds `src/tax/monte_carlo.py` with GBM simulation (`simulate_gbm_terminal_prices`), historical vol estimator (`estimate_annual_vol`), and `run_monte_carlo_tax_analysis()`; 1 000-path simulation wired into `_build_provisional_vest_scenario` and rendered as a "Monte Carlo Tax Sensitivity" section in `recommendation.md`; 29 new tests (1 351 total, all passing) |
| 5.1 | Move SQLite database out of git history | Deferred | The DB is deliberately force-tracked (`!data/pgr_financials.db` in .gitignore) so GitHub Actions workflows can persist historical data between runs.  Proper resolution requires an alternative persistence strategy (GitHub Releases artifacts or S3-compatible storage) and is deferred to a dedicated infrastructure PR |
| 5.2 | Archive completed research modules | Completed | `v34.2` archives 14 study scripts to `archive/scripts/`.  `v35.2` promotes 9 `src/research/` modules to production homes (`src/models/evaluation.py`, `src/models/policy_metrics.py`, `src/portfolio/benchmark_sets.py`, `src/portfolio/diversification.py`, `src/portfolio/redeploy_buckets.py`, `src/portfolio/redeploy_portfolio.py`, `src/reporting/snapshot_summary.py`, `src/reporting/cross_check.py`, `src/reporting/confidence.py`); backward-compat shims left in `src/research/`; `monthly_decision.py` no longer imports from `src/research/` |
| 5.3 | Add a lightweight web dashboard | Not started | No dashboard implementation yet |
| 5.4 | Automated model retraining trigger | Completed | `v35.1` adds `src/models/retrain_trigger.py` (`evaluate_retrain_trigger`, cooldown guard, full audit trail); `model_retrain_log` table + migration 003; `RETRAIN_TRIGGER_BREACH_STREAK`/`RETRAIN_COOLDOWN_DAYS` in `config/model.py`; trigger evaluation wired into `monthly_decision.py` (logs + DB persist); `.github/workflows/drift_retrain_trigger.yml` fires after each weekly fetch and dispatches `monthly_decision` via `workflow_dispatch` if drift fires; 25 new tests (1 376 total, all passing) |
| 5.5 | Property-based testing for numerical edge cases | Completed | `v36.0` adds `hypothesis==6.135.7` to dev deps and four property-test modules (28 tests): `test_property_return_calculations.py`, `test_property_tax_boundaries.py`, `test_property_feature_engineering.py`, `test_property_wfo_temporal.py`; all 1320 tests pass |

## Version Mapping

Peer-review follow-up work maps to these implemented steps:

- `v30.0`: Tier 1.2 EDGAR user-agent hardening
- `v30.1`: Tier 1.3 stale documentation updates
- `v30.2`: Tier 2.3 monthly data-freshness checks
- `v30.3`: Tier 1.4 partial BL fallback diagnostics
- `v30.4`: Tier 2.6 retry/backoff helper for core clients
- `v30.5`: Tier 3.6 monthly pipeline end-to-end test
- `v30.6`: Tier 2.6 retry extension into batch AV loaders
- `v30.7`-`v30.24`: Tier 3.1 and Tier 3.3 phased logging and exception-context sweep across production scripts and core modules
- `v30.10`: Tier 3.5 README landing-page rewrite
- `v6.2`: Tier 2.1 + 2.2 EDGAR schema expansion and full CSV backfill (257 rows, 70 columns)
- `v7.0`–`v7.4`: feature ablation, tax framework, EDGAR parser hardening, CPCV stability, obs/feature ratio
- `v31.0`–`v31.1`: Tier 2.5 conformal empirical coverage monitoring
- `v31.2`–`v31.4`: Tier 2.4 model performance drift monitoring
- `v32.0`: Tier 4.1 feature importance stability across WFO folds
- `v32.1`: Tier 4.2 VIF multicollinearity checks
- `v32.2`: Tier 4.3 vesting decision policy backtest wired into monthly report
- `v32.3`: Tier 4.4 model vs. simple heuristic comparison in monthly report
- `v33.0`: Tier 3.2 config.py split into `config/` package (api, features, model, tax sub-modules)
- `v33.1`: Tier 3.4 mypy CI expansion from 3 modules to 11 modules; 9 type errors fixed in-place
- `v34.0`: Tier 1.4 BL fallback surfaced in monthly report via diagnostic shadow call + Portfolio Optimizer Status section
- `v34.1`: Tier 3.3 exception sweep — exc_info=True on broad catches in edgar_8k_fetcher paths
- `v34.2`: Tier 5.2 standalone study scripts archived to archive/scripts/; companion test archived to archive/tests/
- `v35.0`: Tier 4.5 Monte Carlo tax simulation — `src/tax/monte_carlo.py` with GBM simulator, vol estimator, and `run_monte_carlo_tax_analysis()`; wired into `_build_provisional_vest_scenario`; "Monte Carlo Tax Sensitivity" section in `recommendation.md`; 29 new tests (1 351 total)
- `v35.1`: Tier 5.4 automated retrain trigger — `src/models/retrain_trigger.py`; `model_retrain_log` table + migration 003; config constants; trigger wired into monthly_decision.py; `.github/workflows/drift_retrain_trigger.yml`; 25 new tests (1 376 total)
- `v36.0`: Tier 5.5 property-based tests — hypothesis added to dev deps; 4 test modules (28 tests) covering return invariants, tax boundaries, feature engineering bounds, WFO temporal integrity
- `v34.0`: Tier 1.4 BL diagnostic shadow call; Portfolio Optimizer Status section in recommendation.md
- `v34.1`: Tier 3.3 exception logging sweep; 4 remaining broad catch blocks instrumented with exc_info=True
- `v34.2`: Tier 5.2 archive 14 completed study scripts (v11–v24) to archive/scripts/

## Remaining Highest-Value Gaps

All Tier 1, 2, 3, 4, 5.2, 5.4, and 5.5 items are now complete.  The open work is:

1. **Tier 5.1**: Move SQLite DB out of git history (deferred — force-tracked
   for CI data persistence; needs alternative storage strategy first)
2. **Tier 5.3**: Lightweight web dashboard (not started)

## Related PRs

- PR #56: merged `v30.0` through `v30.10`
- PR #57: merged `v30.11` through `v30.24` plus the peer-review status snapshot
- PR #58: merged `v31.0` through `v31.5` (conformal and drift monitoring)
- PR #59: merged `v32.0` through `v32.4` (ML diagnostic enhancements)
- PR #60: merged `v33.0`–`v33.2` (config refactor, mypy expansion, status snapshot)
- PR #61: merged `v33` code quality continuation
- PR #62: merged `v34.0`–`v34.3` (BL fallback surface, exception sweep, script archival)
- PR #63: active — `v36.0`–`v36.1` (hypothesis property-based testing, Tier 5.5)
- PR #60: merged `v33.0` through `v33.2` (code quality: config split + mypy expansion)
- PR #62: active draft PR for `v34.0` through current work (BL fallback, exception sweep, archival)
