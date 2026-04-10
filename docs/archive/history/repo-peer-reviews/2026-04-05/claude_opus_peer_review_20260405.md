# PGR Vesting Decision Support - Comprehensive Peer Review & Enhancement Plan
**Reviewer:** Claude Opus 4.6 | **Date:** 2026-04-05 | **System Version:** v28-v29
**Scope:** Full codebase review post-v7 through v29 (~22 version iterations)

---

## Overall Assessment

This is a **high-quality, mature production ML system** with exceptional discipline:

- **55,157 lines** of Python across 184+ source files
- **1,218 tests** across 87 test files
- **28+ research/promotion cycles** with rigorous governance
- **9 GitHub Actions workflows** for automated data ingestion and reporting
- Rigorous ML methodology: TimeSeriesSplit only (no K-Fold), embargo/purge, publication lag enforcement
- Clean separation of production vs. research code
- Strong governance: model-governance.md, artifact-policy.md, promotion gates, shadow testing
- No security vulnerabilities: secrets via .env, no SQL injection, no eval/exec

The project demonstrates a level of methodological rigor rarely seen in personal finance tooling.

---

## Strengths

| Dimension | Rating | Notes |
|-----------|--------|-------|
| ML Methodology | 5/5 | Walk-forward only, embargo enforcement, CPCV, conformal intervals |
| Testing | 5/5 | 1,218 tests; edge cases; temporal integrity checks |
| Data Integrity | 5/5 | Publication lag guards, NaN per-fold isolation, scaler isolation |
| Governance | 5/5 | Promotion gates, shadow testing, run manifests |
| DevOps | 4/5 | CI/CD, scheduled workflows, dry-run modes |
| Documentation | 4/5 | Extensive but some staleness (see below) |
| Code Quality | 4/5 | Clean architecture; logging and typing could improve |
| Security | 5/5 | No hardcoded secrets, no eval/exec, parameterized SQL |

---

## Tier 1: Critical Fixes & High-Impact Quick Wins

### 1.1 Fix 3 Failing Tests in test_multi_ticker_loader.py
- **Impact:** High | **Complexity:** S
- **Issue:** 3 pre-existing test failures (noted in DEVELOPMENT_PLAN.md P1.3) undermine CI confidence and mask regressions
- **File:** `tests/test_multi_ticker_loader.py`
- **Action:** Update stale AV API mocks to match current multi-ticker loader interface

### 1.2 Move Personal Email from config.py to .env
- **Impact:** Medium | **Complexity:** S
- **Issue:** EDGAR User-Agent header contains personal email hardcoded in `config.py:35` (`"Jeff Hester jeffrey.r.hester@gmail.com"`)
- **Action:** Move to `EDGAR_USER_AGENT` env var in `.env`, load via `os.getenv()` with a generic fallback

### 1.3 Update Stale Documentation Headers
- **Impact:** Medium | **Complexity:** S
- **Issue:** `DEVELOPMENT_PLAN.md` header says "v6.1" but system is at v28-v29; `docs/architecture.md` references only `results/v9/` for research artifacts
- **Action:** Update DEVELOPMENT_PLAN.md header to reflect current version and add a deprecation/archival note. Update architecture.md to reference v9-v29 research artifacts

### 1.4 Surface Black-Litterman Fallback in Monthly Report
- **Impact:** High | **Complexity:** S
- **Issue:** `src/portfolio/black_litterman.py` falls back silently to equal-weight on optimization failure. User would not know the portfolio recommendation was degraded
- **Files:** `src/portfolio/black_litterman.py`, `src/reporting/decision_rendering.py`
- **Action:** Return a `fallback_used: bool` flag from BL optimization; display a prominent warning in `recommendation.md` when fallback is active

---

## Tier 2: High-Value Data & Model Enhancements

### 2.1 Load Historical 8-K Cache CSV into Database
- **Impact:** High | **Complexity:** M
- **Issue:** `data/processed/pgr_edgar_cache.csv` (256 rows, 2004-2026, 65 columns) sits unused. The model trains on only ~22 months of live-fetched 8-K data vs. 20+ years available
- **Files:** `scripts/edgar_8k_fetcher.py`, `src/database/db_client.py`
- **Action:** Implement `--load-from-csv` path that maps CSV columns to the expanded schema and upserts all rows. This is DEVELOPMENT_PLAN P1.1
- **Dependencies:** Tier 2.2 (schema expansion)

### 2.2 Expand pgr_edgar_monthly Schema (Phase 1)
- **Impact:** High | **Complexity:** M
- **Issue:** Only 10/65 available fields are captured. Missing: segment-level NPW/PIF, investment income, book value, ROE, shares repurchased
- **Files:** `src/database/migrations/`, `src/database/db_client.py`, `src/ingestion/edgar_8k_fetcher.py`
- **Action:** Add migration for Phase 1 high-signal columns (see DEVELOPMENT_PLAN P1.2 for full list). Update upsert logic and parser
- **Note:** This unlocks channel-mix features, underwriting income, ROE/book value, buyback signal

### 2.3 Add Data Freshness Checks to Scheduled Workflows
- **Impact:** High | **Complexity:** S
- **Issue:** No alerting when data goes stale. If a weekly fetch silently fails, the monthly decision runs on outdated data
- **Files:** `src/database/db_client.py` (existing `get_db_health_report()`), `.github/workflows/`
- **Action:** Add a `check_data_freshness()` function that verifies last price date is within 10 days, last FRED date within 45 days, last 8-K within 35 days. Wire into monthly_decision.py as a pre-flight check with prominent warnings in the report

### 2.4 Add Model Performance Drift Detection
- **Impact:** High | **Complexity:** M
- **Issue:** No automated monitoring of whether model signal quality is degrading over time. IC, hit rate, and calibration could drift without notice
- **Files:** `src/models/wfo_engine.py`, `scripts/monthly_decision.py`, new `src/models/drift_monitor.py`
- **Action:** Track rolling 12-month IC, hit rate, and ECE after each monthly run. Store in a `model_performance_log` DB table. Flag when rolling IC drops below `DIAG_MIN_IC` for 3+ consecutive months. Include a "Model Health" section in monthly report

### 2.5 Validate Conformal Prediction Empirical Coverage
- **Impact:** Medium | **Complexity:** S
- **Issue:** 80% nominal conformal coverage is set but empirical coverage in production is not tracked
- **Files:** `src/models/conformal.py`, `scripts/monthly_decision.py`
- **Action:** After each monthly run, compute empirical coverage over the trailing 12 months. Flag if coverage deviates from nominal by more than 10pp. Add to diagnostic report

### 2.6 Implement Retry with Exponential Backoff for API Calls
- **Impact:** Medium | **Complexity:** S
- **Issue:** API clients (AV, EDGAR, FRED) have no retry logic. Transient failures cause full workflow failure
- **Files:** `src/ingestion/av_client.py`, `src/ingestion/edgar_client.py`, `src/ingestion/fred_loader.py`
- **Action:** Add a shared `retry_with_backoff()` decorator (3 retries, exponential backoff) for HTTP calls. Use `requests.Session` with retry adapter

---

## Tier 3: Architecture & Code Quality Improvements

### 3.1 Replace print() Statements with Structured Logging
- **Impact:** Medium | **Complexity:** M
- **Issue:** 254 `print()` calls across 27 files vs. only 3 files using `import logging`. No structured log format, no log levels, no centralized config
- **Action:**
  1. Create `src/logging_config.py` with a standard format (`%(asctime)s %(name)s %(levelname)s %(message)s`)
  2. Migrate production scripts (`weekly_fetch.py`, `monthly_decision.py`, etc.) first
  3. Research scripts can remain with print() since they're interactive
- **Phased:** Start with 4 production entry points + core src/ modules

### 3.2 Refactor config.py into Logical Modules
- **Impact:** Medium | **Complexity:** M
- **Issue:** `config.py` is 528 lines mixing API credentials, model hyperparameters, tax rates, feature sets, benchmark lists, and UI constants
- **Action:** Split into:
  - `config/api.py` - credentials, URLs, rate limits
  - `config/model.py` - WFO params, ensemble settings, thresholds
  - `config/tax.py` - tax rates, lot rules, STCG boundaries
  - `config/features.py` - feature sets, FRED series, model-specific overrides
  - `config/__init__.py` - re-exports for backward compatibility

### 3.3 Improve Exception Handling in Broad Catch Blocks
- **Impact:** Medium | **Complexity:** S
- **Issue:** 33 `except Exception` blocks across 17 files. Many intentionally catch broadly for fallback behavior but lose error context
- **Action:** Add `logging.exception()` or `logging.warning(repr(e))` inside each broad catch block so failures are traceable without changing fallback behavior. Depends on Tier 3.1 logging setup

### 3.4 Expand mypy Coverage Beyond 3 Modules
- **Impact:** Medium | **Complexity:** M
- **Issue:** Only `migration_runner.py`, `provider_registry.py`, `run_manifest.py` are type-checked. ~60% type hint coverage elsewhere
- **Files:** `mypy.ini`, `.github/workflows/ci.yml`
- **Action:** Incrementally add modules to mypy strict checking. Priority order: `src/database/db_client.py`, `src/models/wfo_engine.py`, `src/portfolio/`, `src/processing/feature_engineering.py`

### 3.5 Restructure README.md as a Proper Landing Page
- **Impact:** Medium | **Complexity:** S
- **Issue:** README is 487 lines of version status history. No quick-start, no architecture overview, no "how to run locally"
- **Action:** Rewrite README with:
  1. One-paragraph project description
  2. Architecture diagram (text-based)
  3. Quick-start (setup, env vars, `monthly_decision.py --dry-run`)
  4. Link to `ROADMAP.md` for version history
  5. Link to `docs/` for detailed documentation
  - Move current README content to `docs/VERSION_HISTORY.md`

### 3.6 Add End-to-End Integration Test for Monthly Decision Pipeline
- **Impact:** High | **Complexity:** M
- **Issue:** No single test runs the full pipeline (DB read -> features -> WFO -> recommendation -> report) with a test database
- **Files:** `tests/test_monthly_pipeline_e2e.py` (new)
- **Action:** Create a fixture with a small synthetic SQLite DB (24 months of fake data). Run `monthly_decision.py` logic end-to-end in `--dry-run` mode. Assert: recommendation.md is generated, signals.csv has expected columns, no exceptions thrown

---

## Tier 4: Advanced ML Enhancements

### 4.1 Track Feature Importance Stability Across WFO Folds
- **Impact:** Medium | **Complexity:** S
- **Issue:** Feature coefficients/importances are computed per fold but stability across folds is not tracked. Unstable features may be noise
- **Files:** `src/models/wfo_engine.py`, `src/models/multi_benchmark_wfo.py`
- **Action:** Compute coefficient of variation (CV) for each feature's importance across folds. Flag features with CV > 1.5 as unstable. Include in diagnostic report

### 4.2 Add VIF (Variance Inflation Factor) Checks
- **Impact:** Medium | **Complexity:** S
- **Issue:** Known multicollinearity (momentum 3M/6M/12M correlated; yield curve measures overlap) but no VIF monitoring
- **Files:** `src/processing/feature_engineering.py` or `src/models/wfo_engine.py`
- **Action:** Compute VIF for the feature matrix at monthly build time. Log warnings for features with VIF > 10. Do NOT auto-remove (ElasticNet/Lasso already handles this), but surface for research cycles

### 4.3 Backtest Actual Vesting Decisions
- **Impact:** High | **Complexity:** M
- **Issue:** The system recommends sell percentages monthly but never evaluates whether following those recommendations would have produced better outcomes than alternatives (hold all, sell all, sell fixed %)
- **Files:** New `src/backtest/decision_backtest.py`, `scripts/decision_backtest.py`
- **Action:** Using historical recommendations from `results/monthly_decisions/`, simulate portfolio value under: (a) followed recommendations, (b) hold 100%, (c) sell 100% at vest, (d) sell 10% monthly. Report total return, Sharpe, max drawdown, tax efficiency

### 4.4 Add Model vs. Simple Heuristic Comparison
- **Impact:** Medium | **Complexity:** S
- **Issue:** IC >= 0.07, hit rate >= 55%, OOS R2 >= 2% are modest thresholds. Should continuously verify the model adds value vs. a naive "always diversify X%" heuristic
- **Files:** `scripts/monthly_decision.py`, `src/research/evaluation.py`
- **Action:** Each month, compute what a "constant 10% sell" or "sell when P/E > 20" heuristic would recommend. Include in diagnostic.md as a sanity check baseline

### 4.5 Monte Carlo Tax Scenario Analysis
- **Impact:** Medium | **Complexity:** L
- **Issue:** Tax calculations use point estimates. A Monte Carlo simulation over price paths would show the distribution of after-tax outcomes
- **Files:** New `src/tax/monte_carlo.py`
- **Action:** Generate 1000 price paths from the calibrated return distribution. For each path, simulate the tax-optimal vesting strategy. Report: expected after-tax value, 5th/95th percentile, probability of each lot crossing LTCG threshold before a significant price decline

---

## Tier 5: Long-Term Strategic Improvements

### 5.1 Move SQLite Database Out of Git History
- **Impact:** Medium | **Complexity:** M
- **Issue:** `data/pgr_financials.db` is a binary file committed to git. Every schema change or data update bloats the repo history. Currently 4.5MB data/ but will grow
- **Action:** Options: (a) Add DB to `.gitignore` and use `bootstrap.py` to rebuild from CSV + API on clone, (b) Use Git LFS for the DB file, (c) Host DB as a GitHub release artifact. Recommend (a) with a documented bootstrap procedure

### 5.2 Archive Completed Research Modules
- **Impact:** Low | **Complexity:** S
- **Issue:** 21 research modules in `src/research/` (v11-v29) totaling significant code. Most are completed one-off studies that will never run again
- **Action:** Move completed research modules (v11-v24) to `archive/research/`. Keep v25+ and utility modules (`evaluation.py`, `benchmark_sets.py`, `policy_metrics.py`) in `src/research/`

### 5.3 Add a Lightweight Web Dashboard
- **Impact:** Medium | **Complexity:** L
- **Issue:** All outputs are markdown files and email. A simple dashboard showing current recommendation, model health, data freshness, and historical decisions would improve usability
- **Action:** Streamlit single-page app reading from the SQLite DB and `results/monthly_decisions/`. Deploy via GitHub Pages or a simple hosting solution

### 5.4 Automated Model Retraining Trigger
- **Impact:** Medium | **Complexity:** L
- **Issue:** Model retraining is manual (research cycles). If drift detection (Tier 2.4) shows degradation, the system should suggest or auto-trigger a refit
- **Action:** When rolling IC drops below threshold for 3 months AND new data is available (12+ months since last promotion), automatically refit the ensemble on the expanded training window and run the promotion-gate evaluation. Require manual approval before promoting

### 5.5 Property-Based Testing for Numerical Edge Cases
- **Impact:** Low | **Complexity:** M
- **Issue:** No hypothesis/property-based tests for numerical functions (feature engineering, return calculations, tax computations)
- **Files:** `requirements-dev.txt`, new test files
- **Action:** Add `hypothesis` to dev dependencies. Write property tests for: return calculation invariants, feature engineering monotonicity, tax bracket boundary conditions

---

## Bugs & Potential Errors Found

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| B1 | 3 failing tests in test_multi_ticker_loader.py | Medium | `tests/test_multi_ticker_loader.py` |
| B2 | BLP requires 12mo OOS data - no graceful handling in early months | Low | `src/models/blp.py` |
| B3 | Inverse-variance weight could be extreme if MAE near epsilon | Low | `src/models/multi_benchmark_wfo.py:299` (guarded but edge case) |
| B4 | Broad exception catches suppress error details (33 instances) | Low | 17 files across src/ and scripts/ |
| B5 | DEVELOPMENT_PLAN.md version header stale (says v6.1) | Low | `DEVELOPMENT_PLAN.md:2` |

---

## Stale Documentation Summary

| Document | Issue | Recommendation |
|----------|-------|----------------|
| `DEVELOPMENT_PLAN.md` | Header says "v6.1", content is pre-v6.5 backlog | Add prominent archival banner; update header to v29 |
| `README.md` | 487 lines of version history, no quick-start | Rewrite as landing page; move history to `docs/VERSION_HISTORY.md` |
| `ROADMAP.md` | Mixes 15+ completed versions with planned ones | Split into `CHANGELOG.md` (completed) and `ROADMAP.md` (planned only) |
| `docs/architecture.md` | References only `results/v9/` | Update to reflect v9-v29 research artifacts |
| `docs/data-sources.md` | May reference deprecated FMP as active | Verify FMP is marked deprecated; update EDGAR as primary |

---

## ML Methodology Deep-Dive Notes

### What's Working Well
1. **Temporal integrity**: Rigorous embargo enforcement, publication lags, no future data leakage
2. **No K-Fold**: TimeSeriesSplit enforced system-wide with clear prohibitions in code
3. **Regularization**: ElasticNet + Ridge appropriate for high-p small-N (~180 obs, 15-25 features)
4. **Shallow GBT**: depth=2, 50 trees is appropriate for the sample size
5. **Per-fold NaN imputation**: Training fold median only, preventing test-specific leakage
6. **CPCV**: C(8,2) = 28 backtest paths for overfitting detection
7. **Uncertainty quantification**: Conformal + Bayesian posterior + Platt calibration

### Areas to Watch
1. **Small N per fold**: With ~180 obs and 5-year rolling windows, some ETF benchmarks may have only 2-3 WFO folds - limited statistical power
2. **Modest signal thresholds**: IC >= 0.07, hit rate >= 55%, OOS R2 >= 2% are realistic for 6-month stock prediction but warrant continuous "is the model adding value?" checks
3. **Feature multicollinearity**: Momentum (3M/6M/12M) and yield curve measures overlap significantly. ElasticNet mitigates but VIF monitoring would help research cycles
4. **Conformal coverage**: 80% nominal should be validated empirically in production

---

## Implementation Priority Order

For maximum impact with minimum disruption:
1. **Tier 1** (all 4 items) - immediate quality improvements
2. **Tier 2.3** (data freshness) + **Tier 2.6** (retry logic) - operational safety
3. **Tier 3.5** (README) + **Tier 1.3** (stale docs) - documentation cleanup
4. **Tier 2.1 + 2.2** (historical data + schema expansion) - biggest model improvement
5. **Tier 3.1** (logging) + **Tier 3.3** (exception handling) - observability
6. **Tier 2.4** (drift detection) + **Tier 2.5** (conformal validation) - model monitoring
7. **Tier 3.6** (e2e test) - pipeline confidence
8. **Tier 4** items as research cycles
9. **Tier 5** items as strategic initiatives
