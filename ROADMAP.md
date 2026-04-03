# PGR Vesting Decision Support — Version Roadmap

Day 1 = 2026-03-25 (initial price fetch). Day 2 = 2026-03-26 (dividend fetch +
afternoon bootstrap). Development starts Day 3.

Current status as of 2026-04-02:
- v7.0-v7.4 are complete and verified in `SESSION_PROGRESS.md`.
- v8.0-v8.5 are complete: repo baseline reconciled with GitHub `master`, the
  checked-in database is backfilled from the committed CSV, startup DB health
  checks are in place, and the monthly workflow/docs were refreshed.

## Version History

### v2.7 (complete)
**Released:** March 2026
**Theme:** Complete v2 Relative Return Engine

- 20 independent WFO models (one per ETF benchmark)
- Correct 6M/12M embargo periods (eliminates autocorrelation leakage)
- SQLite accumulation pipeline with weekly GitHub Actions fetch
- DRIP total return reconstruction with split-adjusted unadjusted prices
- Tax-aware lot selection (LTCG/STCG prioritization)
- 271 passing pytest tests covering all modules

---

### v3.0 (complete)
**Released:** March 2026
**Theme:** Macro Intelligence + Monthly Decision Engine

- `src/ingestion/fred_loader.py` — FRED public API client (8 macro series: yield curve, credit spreads, NFCI)
- `src/database/schema.sql` — New `fred_macro_monthly` table
- `src/processing/feature_engineering.py` — 6 derived FRED features in monthly feature matrix
- `src/models/regularized_models.py` — ElasticNetCV pipeline (l1_ratio grid [0.1–1.0])
- `src/models/wfo_engine.py` — Purge buffer fix: gap = horizon + buffer (6M→8, 12M→15)
- `src/backtest/vesting_events.py` — `enumerate_monthly_evaluation_dates()` → 120+ evaluation points
- `src/reporting/backtest_report.py` — Campbell-Thompson OOS R², BHY FDR correction, Newey-West IC
- `scripts/monthly_decision.py` — Automated monthly sell/hold recommendation script
- `.github/workflows/monthly_decision.yml` — Cron: 20th of each month (first business day)
- `results/` — Structured output folder with `monthly_decisions/decision_log.md`
- **New tests:** test_fred_loader (13), test_fred_db (10), test_fred_features (5),
  test_elasticnet (11), test_embargo_fix (9), test_monthly_backtest (11), test_oos_r2 (22)

---

### v3.1 (complete)
**Released:** March 2026
**Theme:** Ensemble Models + Kelly Sizing + Regime Diagnostics

- `src/models/regularized_models.py` — `UncertaintyPipeline` + `build_bayesian_ridge_pipeline()`
- `src/models/multi_benchmark_wfo.py` — `EnsembleWFOResult` + equal-weight 3-model ensemble
- `src/portfolio/rebalancer.py` — `_compute_sell_pct_kelly()` (0.25× Kelly, 30% cap)
- `src/ingestion/fred_loader.py` — PGR-specific: VMT (TRFVOLUSM227NFWA) — note:
  `CUSR0000SETC01` (motor vehicle insurance CPI) added here but removed in v4.1.1
  because the series does not exist in FRED's observations API
- `src/processing/feature_engineering.py` — `vmt_yoy`, `vix` features
- `src/reporting/backtest_report.py` — Rolling 24M IC series, 4-quadrant regime breakdown
- `config.py` — `KELLY_FRACTION=0.25`, `KELLY_MAX_POSITION=0.30`, VIXCLS in FRED series
- **New tests:** test_bayesian_ridge (14), test_kelly_sizing (16),
  test_pgr_fred_features (7), test_regime_breakdown (10)

---

### v4.0 (complete)
**Released:** March 2026
**Theme:** Production Validation + Portfolio Optimization + Tax-Loss Harvesting

- `src/models/wfo_engine.py` — `run_cpcv()` using skfolio `CombinatorialPurgedCV`
  (C(6,2)=15 splits, 5 backtest paths); `CPCVResult` dataclass
- `src/portfolio/black_litterman.py` — **NEW**: `build_bl_weights()` via PyPortfolioOpt
  `BlackLittermanModel`; Ledoit-Wolf shrunk covariance; view confidence = MAE²
- `src/tax/capital_gains.py` — `identify_tlh_candidates()`, `compute_after_tax_expected_return()`,
  `suggest_tlh_replacement()`, `wash_sale_clear_date()`
- `src/processing/feature_engineering.py` — `apply_fracdiff()` + `_fracdiff_weights()`;
  FFD stationarity transform implemented in numpy/scipy (fracdiff package Python <3.10 only)
- `src/portfolio/rebalancer.py` — `compute_benchmark_weights()` (IC × hit_rate normalized)
- `config.py` — `TLH_REPLACEMENT_MAP` (20 ETF pairs), CPCV/BL/TLH/fracdiff constants
- `requirements.txt` — Add `skfolio>=0.3.0`, `PyPortfolioOpt>=1.5.5`
- **New tests:** test_cpcv (17), test_black_litterman (14), test_tlh (23),
  test_fracdiff (13), test_benchmark_weights (11)
- **Total: 477 tests, all passing**

---

### v4.1 (complete)
**Released:** March 2026
**Theme:** Data Integrity + Look-Ahead Bias Guards

Critical fixes deployed before the 2026-03-25 initial ML training label bootstrap
(`post_initial_bootstrap.yml`). All changes prevent historical look-ahead bias from
contaminating the first training dataset.

- `config.py` — `FRED_DEFAULT_LAG_MONTHS = 1`, `FRED_SERIES_LAGS` dict
  (NFCI=2 months, VMT=2 months, all other series=1 month), `EDGAR_FILING_LAG_MONTHS = 2`,
  `KELLY_MAX_POSITION` reduced 0.30 → 0.20
- `src/ingestion/fred_loader.py` — `apply_publication_lags: bool = True` parameter
  added to `fetch_all_fred_macro()`; shifts each series by its configured lag
- `src/ingestion/pgr_monthly_loader.py` — `apply_filing_lag: bool = True` parameter
  added to `load()`; shifts EDGAR index forward by `EDGAR_FILING_LAG_MONTHS`
- `src/processing/feature_engineering.py` — `_apply_fred_lags()` and `_apply_edgar_lag()`
  module-level helpers; both called in `build_feature_matrix_from_db()` after reading
  raw DB values — **authoritative enforcement point** since the DB stores latest-vintage
  values and feature engineering re-reads them
- **Test updates:** test_fred_loader (+3 lag tests), test_feature_engineering (+EDGAR lag
  test, updated feature count), test_kelly_sizing (cap updated to 0.20),
  test_black_litterman (cap-related fixture updated)
- **Total: 482 tests, all passing**

**Rationale:**
- *FRED lag*: FRED serves latest-vintage data; NFCI and VMT undergo meaningful revisions
  post-release (McCracken & Ng 2015). Without lags, Jan features use Jan data — months
  before that data was actually published or finalized.
- *EDGAR lag*: PGR 10-Q for Q4 (period end Dec 31) is filed ~late February. Indexing on
  `report_period` makes Q4 combined_ratio appear in Jan features — 2 months too early.
- *Kelly cap*: Meulbroek (2005) shows 25% employer-stock concentration yields ~42%
  certainty-equivalent loss when human capital correlation is included. Cap at 20% is
  consistent with financial advisor consensus for employer stock specifically.

---

### v4.1.1 (hotfix — 2026-03-24)
**Released:** 2026-03-24
**Theme:** GitHub Actions permissions fix + FRED data bootstrap + schedule adjustment

Root cause analysis of two consecutive bootstrap failures:

1. **2026-03-23 failure** — AV rate limit hit at 14/22 tickers. Workflow was scheduled
   at 11:00 UTC; the free-tier 25 calls/day cap was exhausted mid-run.  Reschedule
   to a quieter slot (+1 day) mitigated runner contention but did not fix permissions.

2. **2026-03-24 failure** — AV price fetch succeeded (22/22 tickers, 22 AV calls,
   284 s), but `git push` exited 403.  **Root cause:** all 6 GitHub Actions workflows
   were missing `permissions: contents: write`.  The `GITHUB_TOKEN` defaults to
   read-only; without an explicit `contents: write` grant the `git push` step is
   rejected by the GitHub REST API.

Changes in this hotfix:

- **All 6 workflows** — Added `permissions: contents: write` at the job level.
- **Bootstrap schedule** — Day 1 moved to Wed 2026-03-25 14:00 UTC, Day 2 to
  Thu 2026-03-26 14:00 UTC, bootstrap moved to Thu 2026-03-26 18:00 UTC (same day
  as Day 2, 4 hours later — no AV calls needed, only DB reads).
- **Daily workflow** (`daily_data_fetch.yml`) — removed from repo in master prior
  to this hotfix; reference removed from documentation.
- **FRED bootstrap** — 12 FRED series (4 967 rows, 1990–2026-03) pre-populated
  locally and committed to `data/pgr_financials.db`.  Removed invalid series
  `CUSR0000SETC01` (motor vehicle insurance CPI) — FRED returns 400 for this ID;
  the BLS publishes it under series code SETE but FRED does not index it directly.
- **`scripts/bootstrap.py`** — `skip_fred` parameter added (default `True`);
  `_run_monthly_decision()` now skips live FRED fetch since data is pre-populated.
  Pass `--fetch-fred` to force a live refresh.

---

### v4.1.2 — FMP → SEC EDGAR XBRL replacement (complete)

---

### v4.1.3 (patch — 2026-03-25)
**Released:** 2026-03-25
**Theme:** Day 1 bootstrap confirmed successful; Day 2 timing adjusted for AV rate-limit safety

Day 1 results (2026-03-25):

- **`initial_fetch_prices.yml`** — ✅ SUCCESS. Ran at 15:01 UTC (61-min scheduler lag),
  completed in 5m 30s. All 22 AV calls succeeded (22/22 tickers). DB committed to master.
- **`monthly_8k_fetch.yml` Pass 2** — ✅ SUCCESS. Ran at 14:51 UTC (51-min scheduler lag),
  completed in 1m 4s. No new rows (idempotent — March data already present from Pass 1
  on the 20th).

Timing adjustment for Day 2 (2026-03-26):

- **Root cause concern:** GitHub Actions scheduler consistently fires 51–61 minutes late
  (free-tier runner queue). Day 1 prices completed at ~15:06 UTC. Day 2 dividends were
  scheduled at 14:00 UTC — only ~23 hours later. AV's 25 calls/day limit may reset on
  a rolling 24-hour window rather than at UTC midnight; if so, 22 new calls at 14:00 UTC
  tomorrow would land within the prior 24-hour window and exhaust the budget mid-run
  (repeating the 2026-03-23 failure).
- **Fix:** `initial_fetch_dividends.yml` cron shifted **14:00 → 15:00 UTC**;
  `post_initial_bootstrap.yml` shifted **18:00 → 19:00 UTC** (preserves 4-hour gap).
  With typical scheduler lag, Day 2 will actually execute ~16:00–16:15 UTC — well past
  the 24-hour mark from Day 1's ~15:06 UTC completion.

---

### v4.2 — 8-K Retry/Recheck + Historical Backfill (complete)
**Released:** 2026-03-24
**Theme:** Remove FMP dependency; free, no-key-required quarterly fundamentals

FMP deprecated all `/v3/` REST endpoints on 2025-08-31.  This sprint replaces
the FMP fundamentals pipeline with the SEC EDGAR XBRL Company-Concept API
(`data.sec.gov/api/xbrl/companyconcept/{cik}/us-gaap/{concept}.json`).

- `src/ingestion/edgar_client.py` — **NEW**: EDGAR XBRL client; fetches
  `EarningsPerShareDiluted`, `Revenues`, `NetIncomeLoss` from 10-Q/10-K filings
  for PGR (CIK 0000080661).  No API key required.  Quarterly cadence only —
  PGR monthly 8-K earnings supplements are PDF attachments not in XBRL.
  Returns records compatible with `db_client.upsert_pgr_fundamentals()`.
  `pe_ratio`, `pb_ratio`, `roe` are `None` (not available via XBRL).
- `src/ingestion/fmp_client.py` — retained for reference; `FMPEndpointDeprecatedError`
  already surfaces clean warnings; no further changes.
- `src/database/db_client.py` — `upsert_pgr_fundamentals` `source` column now
  stores `"edgar_xbrl"` (schema unchanged; `source` column already existed).
- `tests/test_edgar_client.py` — **NEW**: 20 passing tests (all mocked, no
  network calls): `_filter_quarterly` deduplication (6), full fetch shape/types
  (11), `fetch_pgr_latest_quarter` convenience wrapper (3).

**Data availability notes:**
- EDGAR XBRL provides quarterly data aligned with 10-Q/10-K filing dates.
  Publication lag of ~45 days (10-Q) / ~60 days (10-K) after period-end; the
  existing `EDGAR_FILING_LAG_MONTHS = 2` guard in feature engineering remains
  correct.
- `pe_ratio`, `pb_ratio`, `roe` will be `None` for all EDGAR-sourced rows.
  These columns were sparsely populated even under FMP; the WFO engine already
  handles `NaN` gracefully via `WFO_MIN_OBS` guards.
- PGR monthly combined-ratio and PIF data continue to come from the user-provided
  CSV cache (`pgr_monthly_loader.py`); EDGAR XBRL does not expose those metrics.

---

## Planned Versions

Day references below are relative to Day 1 = 2026-03-25 (first bootstrap day).
Week+ targets are only used where genuine data accumulation or external dependencies
require them (noted explicitly).

---

### v4.3 — Signal Quality + Confidence Layer
**Target:** Day 3 (2026-03-27)
**Theme:** Surface BayesianRidge uncertainty in reports; fix BL Ω; reduce feature redundancy

No data accumulation needed — all changes are pure code against the already-populated DB.

- `src/models/multi_benchmark_wfo.py` — `get_confidence_tier(y_hat, y_std)` via
  `norm.cdf(y_hat / y_std)`; `confidence_tier` and `prob_outperform` columns in
  `get_ensemble_signals()` output
- `scripts/monthly_decision.py` — Wire `run_ensemble_benchmarks` + `get_ensemble_signals`
  (currently uses single-model elasticnet path); confidence tier + P(outperform) in report
- `src/portfolio/black_litterman.py` — `prediction_variances: dict[str, float]` param;
  switch Ω diagonal from MAE² to BayesianRidge posterior variance (σ²_pred per benchmark);
  fallback uses `(MAE × √(π/2))²` as RMSE approximation
- `config.py` — `FEATURES_TO_DROP = ["vol_21d", "credit_spread_ig"]` (redundant features);
  `BL_USE_BAYESIAN_VARIANCE = True`
- `src/processing/feature_engineering.py` — Drop redundant features at end of matrix build;
  result: 15 features from 17, improving obs/feature ratio from ~3.5:1 to ~4:1

**Target monthly report format:**
```
COMPOSITE SIGNAL: OUTPERFORM (MODERATE CONFIDENCE)
  P(outperform): ~65% [90% CI: 50%–75%]
  Benchmarks favoring outperformance: 14/20 (70%)
  Expected PGR-SPY spread: +3.2%, 80% CI [-2.1%, +8.5%]
  Calibration status: Phase 1 (uncalibrated Bayesian posterior)
```

---

### v4.3 — Diagnostic OOS Evaluation Report
**Target:** Day 5 (2026-03-29)

Pure code addition — surfaces already-computed diagnostics into a sidecar report.

- `scripts/monthly_decision.py` — `_write_diagnostic_report()` calling existing
  `compute_oos_r_squared()`, `compute_newey_west_ic()`, `generate_regime_breakdown()`;
  writes `diagnostic.md` alongside `recommendation.md`
- `config.py` — `DIAG_MIN_OOS_R2 = 0.02`, `DIAG_MIN_IC = 0.07`,
  `DIAG_MIN_HIT_RATE = 0.55`, `DIAG_CPCV_MIN_POSITIVE_PATHS = 10`

**Benchmark thresholds (peer review):**

| Metric | Good | Marginal | Bad |
|--------|------|----------|-----|
| OOS R² | >2% | 0.5–2% | <0% |
| Mean IC | >0.07 | 0.03–0.07 | <0.03 |
| Hit Rate | >55% | 52–55% | <52% |
| CPCV positive paths | ≥13/15 | 10–12/15 | <10/15 |
| PBO | <15% | 15–40% | >40% |

---

### v4.4 — STCG Tax Boundary Guard
**Target:** Day 7 (2026-04-01)

Small focused addition; no data dependencies.

- `src/portfolio/rebalancer.py` — `_check_stcg_boundary()`: warns when lots in the
  6–12 month STCG zone and predicted alpha < STCG-to-LTCG penalty (~17–22pp)
- `config.py` — `STCG_BREAKEVEN_THRESHOLD = 0.18`
- `src/portfolio/rebalancer.py` — `stcg_warning: str | None` field added to
  `VestingRecommendation` dataclass

---

### v4.5 — New Predictor Variables
**Target:** Day 10 (2026-04-04)
*Note: KIE ticker and new FRED series (CUSR0000SETA02, CUSR0000SAM2) added in v4.5-prep
(2026-03-24). Remaining work is feature engineering and the `pgr_vs_kie_6m` signal.*

| Feature | FRED Series / Source | Mechanism |
|---------|----------------------|-----------|
| `used_car_cpi_yoy` | `CUSR0000SETA02` | Auto total-loss severity; 2021–22 spike was a major PGR headwind |
| `medical_cpi_yoy` | `CUSR0000SAM2` | Bodily injury / PIP claim severity |
| `cr_acceleration` | EDGAR (existing data) | 3-period diff of `combined_ratio_ttm`; second derivative of underwriting margins |
| `pgr_vs_kie_6m` | AV: KIE prices (already in DB) | PGR return minus KIE 6M return; insurance-sector idiosyncratic alpha |

- `src/processing/feature_engineering.py` — 4 new feature computations
- KIE prices already being fetched (v4.5-prep); no additional AV calls needed

---

### v5.0 — CPCV Upgrade + Ensemble Diversity (complete)
**Released:** March 2026

- **CPCV**: C(6,2)=15 → C(8,2)=28 paths (`CPCV_N_FOLDS = 8`)
- **Inverse-variance ensemble**: `1/MAE²`-weighted average in `get_ensemble_signals()`;
  GBT (MAE=0.156) and ElasticNet (MAE=0.165) receive ~75% of total weight
- **Shallow GBT**: `build_gbt_pipeline()` — `GradientBoostingRegressor(max_depth=2,
  n_estimators=50, learning_rate=0.1)`; mean IC +0.148 vs +0.081 ElasticNet across
  8 representative benchmarks; largest gains on VHT (+0.262), VNQ (+0.184), VPU (+0.192)
- **ETF descriptions**: `_ETF_DESCRIPTIONS` dict wired into per-benchmark signal
  table in `recommendation.md` and `diagnostic.md`
- **config.py**: `ENSEMBLE_MODELS` updated to 4 members; `DIAG_CPCV_MIN_POSITIVE_PATHS`
  updated to 19/28; `CPCV_N_FOLDS = 8`
- **459 new tests** in `test_v50_ensemble.py`; total: 675 passed, 1 skipped

---

### v5.1 — Per-Benchmark Platt Calibration (complete)
**Released:** March 2026

- **New file `src/models/calibration.py`**: `CalibrationResult` dataclass,
  `compute_ece()`, `block_bootstrap_ece_ci()` (circular block bootstrap, block_len
  = prediction horizon), `fit_calibration_model()` (Platt at n≥20, isotonic at
  n≥500), `calibrate_prediction()`
- **Per-benchmark design**: one Platt model fitted per ETF benchmark on that
  benchmark's own OOS fold history.  Global pooled calibration was evaluated and
  rejected — pooling 21 asset classes with different return scales caused isotonic
  regression to return a single constant for all benchmarks (plateau collapse).
  With n=78–260 OOS obs per benchmark (2026), isotonic threshold raised to
  `CALIBRATION_MIN_OBS_ISOTONIC = 500`; re-evaluate ~2028.
- **Calibration pipeline**: `_calibrate_signals()` reconstructs inverse-variance
  ensemble OOS fold predictions per benchmark, fits per-benchmark Platt, adds
  `calibrated_prob_outperform` column.  ECE = 2.1% (aggregate, block bootstrap)
  on 3,270 pooled OOS observations.
- **Report updates**: `recommendation.md` shows P(raw) and P(calibrated) side-by-side;
  calibration note replaced with live ECE and 95% CI; `diagnostic.md` calibration
  phase table is data-driven (reads `cal_result.method` at runtime)
- **`VestingRecommendation`**: `calibrated_prob_outperform: float | None` field added
- **config.py**: `CALIBRATION_MIN_OBS_PLATT=20`, `CALIBRATION_MIN_OBS_ISOTONIC=500`,
  `CALIBRATION_N_BINS=10`, `CALIBRATION_BOOTSTRAP_REPS=500`
- **33 new tests** in `test_calibration.py`; total: 747 passed, 1 skipped

---

### v5.2 — Conformal Prediction Intervals ✅ COMPLETE
**Delivered:** 2026-03-30

Distribution-free 80% prediction intervals for each benchmark's ensemble prediction,
with marginal coverage guarantees under time-series non-stationarity.

**Delivered:**
- `src/models/conformal.py` — Native split conformal + Adaptive Conformal Inference (ACI)
  implementation; no MAPIE refit latency in the monthly pipeline
  - `ConformalResult` dataclass: lower, upper, width, coverage_level, empirical_coverage,
    n_calibration, method
  - `split_conformal_interval()`: finite-sample corrected quantile of WFO OOS absolute
    residuals; P(y ∈ CI) ≥ 1-α guarantee (Vovk et al. 2005)
  - `aci_adjusted_interval()`: walk-forward α_t adaptation; update rule:
    α_{t+1} = clip(α_t + γ(α_nominal − err_t), 0.01, 0.99); γ=0.05 default;
    handles distribution shift in 6-month overlapping return windows (Gibbs & Candès 2021)
  - `conformal_interval_from_ensemble()`: main entry; computes residuals from WFO OOS
    y_true/y_hat; dispatches to split or ACI
- `config.py` — Added `CONFORMAL_COVERAGE=0.80`, `CONFORMAL_METHOD="aci"`,
  `CONFORMAL_ACI_GAMMA=0.05`
- `requirements.txt` — Added `mapie>=1.3.0` (used for TimeSeriesRegressor validation path;
  production pipeline uses native implementation)
- `scripts/monthly_decision.py` — `_compute_conformal_intervals()` per-benchmark ACI
  intervals added as Step 2.7; recommendation.md consensus table shows median 80% CI
  range; per-benchmark table adds CI Lower / CI Upper columns; diagnostic.md adds
  Conformal Prediction Intervals section with empirical vs nominal coverage per benchmark
- **46 new tests** in `tests/test_conformal.py`; total: 793 passed, 1 skipped

---

### v6.0 — Cross-Asset Signals + BLP Aggregation
**Target:** Day 42 (2026-05-06)
**Status (2026-03-30):** Feature engineering complete; two of four planned signals shipped.

**Peer data source decision (2026-03-30):** Peer price/dividend history for ALL,
TRV, CB, HIG is sourced from Alpha Vantage — NOT yfinance.  yfinance scrapes
Yahoo Finance's undocumented internal endpoints, has no API contract, and has
broken silently multiple times.  Using AV keeps the entire price/dividend stack
on a single source with consistent unadjusted price handling, known call budget,
and no new dependencies.

**AV budget solution:** Peer tickers (8 calls/run) are fetched on a dedicated
Sunday 04:00 UTC cron — exactly 30 hours after the main Friday 22:00 UTC
weekly_fetch.py cron.  Each day stays within the 25 calls/day free-tier limit:
- Friday 22:00 UTC: `weekly_fetch.py` — 24 AV calls (PGR + 22 ETFs)
- Sunday 04:00 UTC: `peer_data_fetch.py` — 8 AV calls (ALL, TRV, CB, HIG prices + dividends)

**Data already flowing (2026-03-30):** `peer_bootstrap.yml` ran manually; ALL,
TRV, CB, HIG full price and dividend history seeded into the DB.  `peer_data_fetch.yml`
runs weekly from now on.

BLP parameter fitting needs ~12 months of live OOS predictions — delay this
sub-feature to Week 8+ (2026-05-20) while the rest of v6.0 ships on Day 42.

**SHIPPED (2026-03-30):**
- **`high_52w`**: `current_price / 52-week_high` (George & Hwang 2004).
  Implemented in `build_feature_matrix()` as a price-derived feature using
  `daily_close.rolling(252, min_periods=126).max()`.
  IC=0.122 (p=0.041, n=281); 91.4% data coverage.
- **`pgr_vs_peers_6m`**: PGR 6M DRIP return minus equal-weight peer composite
  (ALL, TRV, CB, HIG) 6M return.  Pre-computed in `build_feature_matrix_from_db()`
  and injected as a synthetic FRED column.  IC=0.115 (p=0.045, n=304);
  98.7% data coverage.  Current value: −0.232 (PGR −23% vs peers over 6M).
- **`pgr_vs_vfh_6m`**: PGR 6M return minus VFH (Vanguard Financials ETF) 6M return.
  Broadens KIE benchmark to all US financials (banks, insurance, diversified).
  VFH already in ETF universe — no separate bootstrap needed.
  IC=0.088 (p=0.165, n=not significant independently); 82.1% data coverage.
  Current value: −0.073.  Lasso regularization will select or shrink based on
  marginal contribution in the WFO ensemble.
- **30 new/updated tests** in `tests/test_v60_features.py`; total: **849 passed, 1 skipped**

**REMAINING:**
- **Beta-Transformed Linear Pool (BLP)**: Replaces naive equal-weight ensemble
  averaging (Ranjan & Gneiting 2010: any linear pool of calibrated forecasts is
  necessarily uncalibrated); 5-parameter BLP fit via negative log-likelihood
  *(requires ~12 months of live OOS predictions; BLP sub-feature ships Week 8)*
- **Residual momentum**: Regress PGR returns on Fama-French 3-factor over trailing
  36M window; cumulate factor-neutral residuals from t-12 to t-1 (Blitz et al. 2011:
  2× alpha of raw momentum, greater consistency)
- *(Cross-asset signal infrastructure complete; all planned signals shipped)*

---

---

### v6.1 — Monthly Decision Email Notification (complete)
**Released:** 2026-03-30
**Theme:** Automated email delivery of monthly prediction report

Added `Send monthly decision email` step to `.github/workflows/monthly_decision.yml`
immediately after the `Commit results` step.

**Implementation:**
- Inline Python (smtplib) — no third-party actions, no new dependencies
- Subject: `PGR Monthly Decision — {Month YYYY}: {SIGNAL}` (signal parsed from `recommendation.md`)
- Body: full plain-text content of `results/monthly_decisions/YYYY-MM/recommendation.md`
- Port 465 → SMTP_SSL; port 587 → STARTTLS (auto-detected from `SMTP_PORT` secret)
- Skips gracefully if secrets are unconfigured or report file doesn't exist
- `continue-on-error: true` — email failure never blocks data collection or DB commit
- Skipped on `dry_run: true` dispatches

**Repository secrets required:**

| Secret | Purpose |
|--------|---------|
| `SMTP_SERVER` | Outbound SMTP hostname |
| `SMTP_PORT` | SMTP port (465 for SSL, 587 for STARTTLS) |
| `SMTP_USERNAME` | SMTP authentication username |
| `SMTP_PASSWORD` | SMTP authentication password |
| `PREDICTION_EMAIL_FROM` | Sender address shown in the From header |
| `PREDICTION_EMAIL_TO` | Recipient address |

---

### v6.2 — Historical Backfill + Expanded 8-K Schema (complete)
**Released:** 2026-04-01
**Theme:** Load 20+ years of PGR operating data; expand pgr_edgar_monthly to 44 columns

Unlocks `data/processed/pgr_edgar_cache.csv` (256 rows, 2004–2026, 65 columns) for
model training.  The primary data gap was that `pgr_edgar_monthly` had only 22 months
of live-fetched data capturing 7 of 65 available fields.

**P1.2 — Schema Expansion:**
- `src/database/schema.sql` — `pgr_edgar_monthly` extended from 7 to 44 columns:
  foundational P&L (NPW, NPE, net income, EPS diluted, loss/LAE ratio, expense ratio),
  segment channels (NPW/NPE/PIF by agency/direct/commercial/property),
  company-level operating metrics (investment income, total revenues, total assets,
  ROE, book equity, unearned premiums, buyback data), investment portfolio metrics
  (FTE return, book yield, unrealized gains, duration), and derived features
- `src/database/db_client.py` — `initialize_schema` applies 37 idempotent `ALTER TABLE`
  migrations; `upsert_pgr_edgar_monthly` and `get_pgr_edgar_monthly` updated for all 44 cols

**P1.1 — CSV Backfill (load_from_csv):**
- `scripts/edgar_8k_fetcher.py` — `load_from_csv` now maps all 65 CSV columns via
  `DIRECT_MAP`; computes derived features:
  `channel_mix_agency_pct = npw_agency / (npw_agency + npw_direct)`,
  `npw_growth_yoy` (12M pct_change), `underwriting_income = npe × (1 − CR/100)`,
  `unearned_premium_growth_yoy` (12M pct_change); `buyback_yield` remains NULL
  (requires market_cap not available in CSV)
- Coverage log after load: combined_ratio, NPW, npw_agency, investment_income,
  book_value_per_share, gainshare_estimate
- CSV column rename handled: `roe_net_income_trailing_12m` → `roe_net_income_ttm`

**Testing:**
- `tests/test_v62_schema_and_csv.py` — **29 new tests**:
  schema migration, round-trip upsert, backward compat, direct field mapping,
  derived field correctness, NaN/NULL handling, dry_run, error cases, edge cases
- **Total: 909 passed, 1 skipped**

```bash
# Bootstrap 20+ years of history (no network calls):
python scripts/edgar_8k_fetcher.py --load-from-csv
# Dry run to verify coverage before writing:
python scripts/edgar_8k_fetcher.py --load-from-csv --dry-run
```

---

### v6.3 — Channel-Mix Features in Monthly Decision Model (complete)
**Released:** 2026-04-01
**Theme:** Wire agency/direct channel-mix signals into the ML feature pipeline (P1.4)

Adds two new predictive features to `build_feature_matrix()`, consuming the
segment-level data loaded by v6.2's CSV backfill:

- **`channel_mix_agency_pct`**: `npw_agency / (npw_agency + npw_direct)`.
  Agency share trending down (direct gaining) is a leading indicator of
  improved unit economics and combined-ratio improvement — historically one
  of PGR's key competitive differentiation signals.
- **`npw_growth_yoy`**: companywide NPW 12-month YoY growth rate.  Strong
  growth (> 10%) signals rate adequacy and market-share gain.

**Implementation:**
- `src/processing/feature_engineering.py` — new channel-mix block in the
  `if pgr_monthly is not None` section; both features added to the sparsity-guard
  loop (`WFO_MIN_GAINSHARE_OBS` threshold, same as combined_ratio_ttm)
- Features read directly from `pgr_edgar_monthly` columns pre-computed at
  CSV load time; forward-filled to monthly feature matrix dates
- Absent when `pgr_monthly=None`, column missing, or all-NaN (full backward compat)

**Testing:** 12 new tests in `tests/test_v63_channel_mix_features.py`; total **921 passed, 1 skipped**

---

### v6.4 — P2.x Operational & Valuation Features (current)
**Released:** 2026-04-01
**Theme:** Wire underwriting income, unearned premium pipeline, ROE trend,
investment portfolio quality, and share repurchase signal into the ML feature
pipeline (P2.1–P2.5)

Adds eleven new predictive features to `build_feature_matrix()`, all sourced
from `pgr_edgar_monthly` (PGR monthly 8-K supplements).  Defaulting to monthly
8-K data throughout maximises observation count and ensures consistent sourcing.

**P2.2 — Underwriting income:**
- `underwriting_income` (DB pre-computed: `npe × (1 − CR/100)`)
- `underwriting_income_3m` (3-month trailing average)
- `underwriting_income_growth_yoy` (12M YoY pct_change)

**P2.3 — Unearned premium pipeline:**
- `unearned_premium_growth_yoy` (DB pre-computed 12M pct_change; leads earned premium ~6M)
- `unearned_premium_to_npw_ratio` (`unearned_premiums / net_premiums_written`)

**P2.4 — ROE trend:**
- `roe_net_income_ttm` (8-K monthly TTM ROE; 4× more obs than quarterly XBRL)
- `roe_trend` (current ROE − rolling 12M mean; positive = improving efficiency)

**P2.1 — Investment portfolio:**
- `investment_income_growth_yoy` (12M YoY growth; rate-environment proxy)
- `investment_book_yield` (fixed-income book yield; complements `yield_slope`)

**P2.5 — Share repurchase signal:**
- `buyback_yield` (annualised buyback spend / est. market cap via BVPS + equity)
- `buyback_acceleration` (current month / trailing 12M mean; > 1 = accelerating)

**Implementation:**
- `src/processing/feature_engineering.py` — v6.4 P2.x block added in the
  `if pgr_monthly is not None` section; all eleven features added to the
  sparsity-guard loop (`WFO_MIN_GAINSHARE_OBS` threshold)
- All features sourced from `pgr_edgar_monthly`; forward-filled to monthly dates
- Absent when `pgr_monthly=None`, column missing, or below sparsity threshold
  (full backward compat with pre-v6.2 databases)

**Testing:** 28 new tests in `tests/test_v64_p2x_features.py`; total **949 passed, 1 skipped**

---

### v6.5 — P2.6 / P2.7 / P2.8: HTML Parser Extension, Calibration Plot, Email Module (current)
**Released:** 2026-04-02
**Theme:** Live 8-K field capture (P2.6), calibration diagnostic (P2.7), testable email (P2.8)

**P2.6 — Extend 8-K HTML Parser:**
Extends `_parse_html_exhibit()` in `scripts/edgar_8k_fetcher.py` to capture
12 additional fields from the monthly 8-K exhibit HTML:
`net_premiums_written`, `net_premiums_earned`, `npw_agency`, `npw_direct`,
`npw_commercial`, `npw_property`, `investment_income`, `book_value_per_share`,
`eps_basic`, `shares_repurchased`, `avg_cost_per_share`, `investment_book_yield`.
New `_try_parse_dollar()` helper reduces boilerplate for range-guarded regex extraction.
`_compute_derived_fields()` extended to compute `channel_mix_agency_pct`,
`underwriting_income`, `npw_growth_yoy`, `unearned_premium_growth_yoy` from the
assembled time series (mirrors the CSV backfill path).

**P2.7 — Calibration Reliability Diagram:**
`_plot_calibration_curve()` added to `scripts/monthly_decision.py`.  Written to
`results/monthly_decisions/YYYY-MM/plots/calibration_curve.png` on each monthly run.
Shows binned predicted P(outperform) vs. actual fraction positive, with ECE annotation
and 95% bootstrap CI.  `_calibrate_signals()` return signature updated to expose
pooled probabilities and outcomes for the diagram.

**P2.8 — Testable Email Module:**
Email logic extracted from inline YAML into `src/reporting/email_sender.py`:
- `build_email_message()` — pure function; constructs MIMEMultipart from report body
- `send_monthly_email()` — env-var / kwarg config; SMTP_SSL (port 465) or STARTTLS (587);
  `dry_run=True` returns subject without network connection
Workflow YAML step updated to call the module.

**Testing:** 35 new tests in `tests/test_v65_p26_p27_p28.py`; total **984 passed, 1 skipped**

---

### Housekeeping — AV "Information" vs "Note" Response Handling
**Status:** Complete
**Theme:** Weekly fetch resilience against benign AV advisories

The AV free-tier API returns two distinct response types that the current code
treats identically:

| AV Key | Meaning | Correct action |
|--------|---------|----------------|
| `"Note"` | Hard daily quota (25 calls) exhausted | Stop immediately; defer remaining tickers |
| `"Information"` | Advisory nudge ("spread out requests") | Log warning; **continue fetching** |

**Current behaviour:** `multi_ticker_loader.py` and `multi_dividend_loader.py` now
raise `AVRateLimitAdvisory` on `"Information"` so the batch skips only the affected
ticker and continues.  Only `"Note"` raises `AVRateLimitError` and stops the batch.

**Root cause context:** The advisory fires when a free-tier session uses ~22–23 of its
25 daily calls in one run.  The 13-second inter-call delay already exceeds AV's stated
1-request/second limit; increasing the delay would not suppress the advisory and would
add ~2.5 minutes to every weekly run with no benefit.  As the DB matures and more
tickers return 0 new rows (already-fresh data), the total calls-per-run will decrease
naturally and the advisory will stop appearing.

**Shipped fix:**
- `src/ingestion/multi_ticker_loader.py` — `"Information"` continues the batch; `"Note"`
  remains the hard stop.
- `src/ingestion/multi_dividend_loader.py` — Same pattern.
- `src/ingestion/exceptions.py` — `AVRateLimitAdvisory` distinguishes soft advisories
  from hard quota exhaustion.
- `tests/test_multi_ticker_loader.py` — Regression coverage confirms mocked advisory
  responses no longer abort the full batch and do not require a real `AV_API_KEY`.

---

### v6.x Completed — Feature Engineering Enhancements (2026-04-01)

**pb_ratio and pe_ratio now use monthly EDGAR 8-K data throughout the stack.**

Previously `pe_ratio` used quarterly XBRL EPS from `pgr_fundamentals_quarterly`
(rolling 4-quarter sum, ~86 data points).  Now both ratios are computed entirely
from `pgr_edgar_monthly` — the same monthly 8-K supplements that supply
`combined_ratio` and `book_value_per_share`:

| Feature | Source (before) | Source (after) |
|---|---|---|
| `pe_ratio` | `pgr_fundamentals_quarterly.eps` (quarterly XBRL, 4-quarter sum) | `pgr_edgar_monthly.eps_basic` (monthly 8-K, 12-month rolling sum) |
| `pb_ratio` | `pgr_edgar_monthly.book_value_per_share` ✅ | unchanged ✅ |
| `roe` | `pgr_fundamentals_quarterly.roe` (quarterly XBRL) | unchanged (see candidate below) |

**Changes shipped:**
- `src/database/schema.sql` — Added `eps_basic REAL` to `pgr_edgar_monthly`
- `src/database/db_client.py` — Migration, upsert, and get updated for `eps_basic`
- `src/ingestion/pgr_monthly_loader.py` — Loads `eps_basic` from CSV with alias resolution
- `src/processing/feature_engineering.py` — `pe_ratio` uses `edgar_raw["eps_basic"].rolling(12).sum()`
- `docs/PGR_EDGAR_CACHE_DATA_DICTIONARY.md` — Full data dictionary for all 65 columns

**Bug fixes also shipped:**
- `src/ingestion/multi_ticker_loader.py` / `multi_dividend_loader.py` — Fixed UTC/local
  date mismatch in skip-if-fresh logic (`date.today()` → `datetime.now(tz=timezone.utc)`)
- `tests/test_multi_ticker_loader.py` — Fixed budget-exhaustion test to insert UTC dates
- `src/processing/feature_engineering.py` — Removed over-aggressive `cr_acceleration`
  gate (was dropping the feature in test fixtures with <60 obs; production data has 225+)
- `src/ingestion/exceptions.py` / `multi_ticker_loader.py` / `multi_dividend_loader.py` —
  AV `"Information"` (soft advisory) now raises `AVRateLimitAdvisory` and continues
  batch; only `"Note"` (hard quota) raises `AVRateLimitError` and stops

---

### Candidate Features — EDGAR Monthly 8-K (Future Sprint)

Full column documentation in `docs/PGR_EDGAR_CACHE_DATA_DICTIONARY.md`.
All features below are derivable from `data/processed/pgr_edgar_cache.csv`
with 256 monthly observations back to August 2004.

**Highest priority (strong theoretical prior + straightforward derivation):**

| Candidate Feature | Derivation | Rationale |
|---|---|---|
| `npw_growth_yoy` | `net_premiums_written.pct_change(12)` | Volume momentum; acceleration above peers signals pricing power |
| `channel_mix_direct_pct` | `pif_direct_auto / pif_total_personal_lines` | Rising direct share = structurally higher margin (no agent commission) |
| `unearned_premium_growth_yoy` | `unearned_premiums.pct_change(12)` | Forward revenue signal; converts to earned revenue over next 6–12 months |
| `underwriting_income` | `npe - losses_lae - policy_acquisition_costs - other_underwriting_expenses` | Core insurance profit stripped of investment income and taxes; cleaner signal than net income |
| `npw_per_pif` | `net_premiums_written / pif_total` | Average premium per policy — captures rate increases independent of volume |
| `roe` (monthly) | `roe_net_income_trailing_12m` from 8-K (pre-computed) | Switch from quarterly XBRL to monthly 8-K for consistency; same 256-obs depth as pe/pb |

**Medium priority (informative but more data-engineering work):**

| Candidate Feature | Derivation | Rationale |
|---|---|---|
| `reserve_to_npe_ratio` | `loss_lae_reserves / net_premiums_earned` | Reserve adequacy; rising ratio precedes adverse development |
| `realized_gain_to_ni_ratio` | `total_net_realized_gains / net_income` | Earnings quality flag; high ratio = income driven by portfolio sales, not underwriting |
| `price_to_npw` | `market_cap / (net_premiums_written * 12)` | Insurance-sector valuation multiple alongside P/B |
| `investment_income_growth_yoy` | `investment_income.pct_change(12)` | Interest rate reinvestment tailwind/headwind |
| `equity_per_share_growth_yoy` | `book_value_per_share.pct_change(12)` | Intrinsic value compounding rate on a per-share basis |
| `loss_ratio_ttm` | `rolling(12).mean()` of `loss_lae_ratio` | Separates loss deterioration from expense pressure within combined_ratio |
| `expense_ratio_ttm` | `rolling(12).mean()` of `expense_ratio` | Structural cost efficiency trend independent of loss activity |

**Lower priority / requires additional data:**

| Candidate Feature | Notes |
|---|---|
| `unrealized_gain_pct_equity` | `net_unrealized_gains_fixed / shareholders_equity` — OCI/rate risk proxy; high sensitivity to rate regime |
| `cr_vs_industry_spread` | Requires external peer CR data (Travelers, Allstate, etc.) |
| `combined_ratio_ex_cats` | Not directly reported; would need catastrophe loss estimates from external source |

**Implementation notes:**
- All features require the 2-month EDGAR filing lag (`config.EDGAR_FILING_LAG_MONTHS`)
- `pif_property`, `npw_property`, and `npe_property` are only available from ~2015; any feature using them will have reduced coverage
- Add to `pgr_monthly_loader.py` → `schema.sql` + `db_client.py` migration → `feature_engineering.py` (same pattern as `eps_basic` / `book_value_per_share`)
- Validate each feature with a `pytest` test confirming non-null coverage ≥ `WFO_MIN_GAINSHARE_OBS` before activating in WFO

---

## Development Principles

- **Never finalize a module without a passing pytest suite** (CLAUDE.md mandate)
- **No K-Fold cross-validation** — `TimeSeriesSplit` with embargo + purge buffer only
- **No StandardScaler across full dataset** — scaler isolated within each WFO fold Pipeline
- **No yfinance** — not for fundamentals, ratios, or price data; AV is the canonical price
  source for all tickers including v6.0 insurance peers (see v6.0 peer data source decision)
- **Python 3.10+**, strict PEP 8, full type hinting
- **Approved libraries:** pandas, numpy, scikit-learn, matplotlib, xgboost, requests,
  statsmodels (v3.0+), skfolio/PyPortfolioOpt (v4.0+)

## Monthly Decision Log

See [`results/monthly_decisions/decision_log.md`](results/monthly_decisions/decision_log.md) for the persistent record of all automated monthly recommendations generated by `scripts/monthly_decision.py`.
