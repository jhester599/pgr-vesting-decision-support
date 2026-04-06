# Claude Opus Peer Review Status Snapshot

Updated: 2026-04-05
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

- Tier 1 quick wins: mostly complete
- Tier 2 operational safety: strong progress, with freshness, retry, conformal coverage, and drift-monitoring work now landed or in the active PR stream
- Tier 3 observability/docs/testability: strong progress, still incomplete overall
- Tier 4 and Tier 5 strategic items: mostly not started

## Status By Enhancement

| ID | Enhancement | Status | Notes |
|----|-------------|--------|-------|
| 1.1 | Fix 3 failing tests in `test_multi_ticker_loader.py` | Already satisfied | Verified on the baseline at the start of the `v30` sequence; no code change was needed |
| 1.2 | Move personal email from `config.py` to `.env` | Completed | Landed in `v30.0` and merged via PR #56 |
| 1.3 | Update stale documentation headers | Completed | Landed in `v30.1`; README landing-page rewrite landed in `v30.10` |
| 1.4 | Surface Black-Litterman fallback in monthly report | Partial | BL fallback diagnostics landed in `v30.3` and BL logging landed in `v30.17`, but the monthly report still does not surface a live BL fallback warning because the current monthly path is not fully driven by BL |
| 2.1 | Load historical 8-K cache CSV into database | Not started | No CSV bulk-load path has been implemented yet |
| 2.2 | Expand `pgr_edgar_monthly` schema (Phase 1) | Not started | No schema-expansion migration for the peer-review Phase 1 set has been added yet |
| 2.3 | Add data freshness checks to scheduled workflows | Completed | Landed in `v30.2` and merged via PR #56 |
| 2.4 | Add model performance drift detection | Completed | `v31.2`-`v31.4` add rolling drift helpers, the `model_performance_log` table, monthly persistence, drift warnings, and a `Model Health` section in `recommendation.md` |
| 2.5 | Validate conformal empirical coverage | Completed | `v31.0`-`v31.1` add historical conformal coverage backtesting plus monthly trailing-coverage diagnostics and manifest warnings |
| 2.6 | Implement retry with exponential backoff for API calls | Completed | Landed in `v30.4` and `v30.6`; core clients and batch AV loaders use the shared retry session |
| 3.1 | Replace `print()` with structured logging | Partial | Production entry points and several core modules were migrated across `v30.7`-`v30.24`, but research/utility modules still contain many `print()` calls |
| 3.2 | Refactor `config.py` into logical modules | Not started | `config.py` remains monolithic |
| 3.3 | Improve exception handling in broad catch blocks | Partial | A large observability pass landed across `v30.11`-`v30.24`, but the peer review identified more broad catches than have been covered so far |
| 3.4 | Expand mypy coverage beyond 3 modules | Not started | The CI/type-check target list has not been expanded yet, though fixes were kept mypy-clean in the existing scoped modules |
| 3.5 | Restructure `README.md` as a proper landing page | Completed | Landed in `v30.10` and merged via PR #56 |
| 3.6 | Add end-to-end integration test for monthly decision pipeline | Completed | Landed in `v30.5` and merged via PR #56 |
| 4.1 | Track feature importance stability across WFO folds | Not started | No fold-stability metric or diagnostic-report surfacing yet |
| 4.2 | Add VIF checks | Not started | No VIF monitoring has been added yet |
| 4.3 | Backtest actual vesting decisions | Not started | No decision-backtest module has been added yet |
| 4.4 | Add model vs. simple heuristic comparison | Not started | No monthly heuristic benchmark output yet |
| 4.5 | Monte Carlo tax scenario analysis | Not started | No Monte Carlo tax module yet |
| 5.1 | Move SQLite database out of git history | Not started | Current repo layout still includes the SQLite DB in the repo |
| 5.2 | Archive completed research modules | Not started | No research archive move has been performed yet |
| 5.3 | Add a lightweight web dashboard | Not started | No dashboard implementation yet |
| 5.4 | Automated model retraining trigger | Not started | No retraining trigger workflow yet |
| 5.5 | Property-based testing for numerical edge cases | Not started | `hypothesis`-style property tests have not been introduced yet |

## Version Mapping

Peer-review follow-up work currently maps to these implemented `v30` steps:

- `v30.0`: Tier 1.2 EDGAR user-agent hardening
- `v30.1`: Tier 1.3 stale documentation updates
- `v30.2`: Tier 2.3 monthly data-freshness checks
- `v30.3`: Tier 1.4 partial BL fallback diagnostics
- `v30.4`: Tier 2.6 retry/backoff helper for core clients
- `v30.5`: Tier 3.6 monthly pipeline end-to-end test
- `v30.6`: Tier 2.6 retry extension into batch AV loaders
- `v30.7`-`v30.24`: Tier 3.1 and Tier 3.3 phased logging and exception-context sweep across production scripts and core modules
- `v30.10`: Tier 3.5 README landing-page rewrite
- `v31.0`-`v31.1`: Tier 2.5 conformal empirical coverage monitoring
- `v31.2`-`v31.4`: Tier 2.4 model performance drift monitoring

## Remaining Highest-Value Gaps

If we continue working directly from the peer-review backlog, the biggest open
items are:

1. Tier 2.1 and Tier 2.2: historical 8-K CSV loading plus schema expansion
2. Tier 3.2 and Tier 3.4: config modularization and broader mypy coverage
3. Tier 4.3 and Tier 4.4: evaluating the actual decision policy against simpler baselines
4. Tier 4.1 and Tier 4.2: feature-stability and multicollinearity diagnostics

## Related PRs

- PR #56: merged `v30.0` through `v30.10`
- PR #57: merged `v30.11` through `v30.24` plus the peer-review status snapshot
- PR #58: active draft PR for `v31.0` through the current continuation work
