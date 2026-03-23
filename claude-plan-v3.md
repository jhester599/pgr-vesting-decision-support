# PGR Vesting Decision Support: v3.0 → v4.0 Development Plan

## Context

The system is at v2.7 with a complete relative-return engine (20 ETF benchmarks, LassoCV/RidgeCV, Walk-Forward Optimization, DRIP reconstruction, SQLite pipeline, GitHub Actions weekly fetch). A Claude peer-review research report identified three high-value enhancement phases. This plan implements those recommendations, adds a monthly automated decision action, and creates a structured results logging system.

**Current version:** v2.7 (develop branch)
**Starting version for new work:** v3.0
**Branch strategy:** feature branches → develop → master per version

---

## Key Architectural Findings (pre-plan research)

- `wfo_engine.py` already passes `gap=target_horizon_months` to `TimeSeriesSplit`; the embargo equals exactly the horizon. The research report recommends adding a `purge_buffer` (default 2M for 6M targets, 3M for 12M) so total gap = horizon + buffer.
- FRED is a free public REST API; fetches don't consume AV/FMP budget. Requires a free `FRED_API_KEY` env var.
- `weekly_fetch.py` uses exactly 24 of 25 AV calls — FRED calls are additive with no budget impact.
- `build_feature_matrix_from_db()` in `backtest_engine.py` always calls `force_refresh=True`; FRED features join into the feature matrix as an additional DataFrame merge.
- `conftest.py` is nearly empty; each test file defines its own fixtures — follow this pattern for new test files.

---

## Results Folder Structure (create before v3.0 work begins)

```
results/
├── monthly_decisions/
│   ├── decision_log.md          # Persistent append-only log of all monthly runs
│   ├── 2026-03/
│   │   ├── recommendation.md   # Human-readable sell/hold recommendation report
│   │   ├── signals.csv          # Per-benchmark: ticker, IC, hit_rate, predicted_return, signal
│   │   ├── backtest_summary.csv # Monthly stability stats at this point in time
│   │   └── plots/               # Visualizations generated this month
│   └── YYYY-MM/
│       └── ...
└── backtests/
    └── historical_backtest_YYYY-MM-DD.csv   # Point-in-time backtest exports
```

`decision_log.md` schema (one row per monthly run):
```
| Date       | Run Date   | Signal     | Sell % | Predicted 6M Return | IC   | Hit Rate | Notes |
|------------|------------|------------|--------|---------------------|------|----------|-------|
| 2026-03-20 | 2026-03-20 | OUTPERFORM | 25%    | +8.3%               | 0.12 | 63%      |       |
```

---

## v3.0 — Macro Features + ElasticNet + Monthly Stability + Monthly Decision Action

**Goal:** Add FRED macro features, replace Lasso/Ridge with ElasticNet, fix embargo purge buffer, add statistical rigor, expand evaluation to 120+ monthly points, and deliver first automated monthly decision report.

### New Dependencies (add to `requirements.txt`)
- `statsmodels>=0.14.0` — BHY multiple testing correction, Newey-West standard errors

### New Config Keys (`config.py`)
```python
FRED_API_KEY: str | None = os.getenv("FRED_API_KEY")
FRED_BASE_URL: str = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES_MACRO: list[str] = [
    "T10Y2Y", "GS5", "GS2", "GS10", "T10YIE",   # yield curve
    "BAA10Y", "BAMLH0A0HYM2",                       # credit spreads
    "NFCI",                                          # financial conditions index
]
FRED_SERIES_PGR: list[str] = [                      # used in v3.1
    "CUSR0000SETC01",     # motor vehicle insurance CPI
    "TRFVOLUSM227NFWA",   # vehicle miles traveled
]
WFO_PURGE_BUFFER_6M: int = 2    # extra months of purge beyond 6M horizon → gap=8
WFO_PURGE_BUFFER_12M: int = 3   # extra months of purge beyond 12M horizon → gap=15
```

### Task 3.0.1 — Create `src/ingestion/fred_loader.py`
**New file.** FRED public REST API client (no budget impact).

Key functions:
```python
def fetch_fred_series(series_id: str, observation_start: str = "2008-01-01", dry_run: bool = False) -> pd.DataFrame
def fetch_all_fred_macro(series_ids: list[str], dry_run: bool = False) -> pd.DataFrame
# Returns: DatetimeIndex (month-end), one column per series_id; forward-filled; NaN for missing
```

### Task 3.0.2 — Add `fred_macro_monthly` table to `src/database/schema.sql`
```sql
CREATE TABLE IF NOT EXISTS fred_macro_monthly (
    series_id  TEXT NOT NULL,
    month_end  TEXT NOT NULL,  -- 'YYYY-MM-DD' last day of month
    value      REAL,
    PRIMARY KEY (series_id, month_end)
);
```
Add to `src/database/db_client.py`:
```python
def upsert_fred_macro(conn, records: list[dict]) -> int  # INSERT OR REPLACE
def get_fred_macro(conn, series_ids: list[str] | None = None) -> pd.DataFrame
```

### Task 3.0.3 — Integrate FRED into `src/processing/feature_engineering.py`
Add `fred_macro: pd.DataFrame | None = None` parameter to `build_feature_matrix()`. After existing Gainshare block, merge FRED columns on month-end date. Add derived features:
- `yield_slope` = T10Y2Y (direct)
- `yield_curvature` = 2×GS5 − GS2 − GS10
- `real_rate_10y` = GS10 − T10YIE
- `credit_spread_ig` = BAA10Y
- `credit_spread_hy` = BAMLH0A0HYM2
- `nfci` = NFCI (already a composite)

Update `build_feature_matrix_from_db()` to load `get_fred_macro(conn, FRED_SERIES_MACRO)` and pass it through.

### Task 3.0.4 — Fix embargo purge buffer in `src/models/wfo_engine.py`
Change `TimeSeriesSplit(gap=target_horizon_months)` to `TimeSeriesSplit(gap=target_horizon_months + purge_buffer)`.
Add `purge_buffer: int = 2` parameter to `run_wfo()`. Default behavior changes: 6M → gap=8, 12M → gap=15.
Update `src/models/multi_benchmark_wfo.py` to pass `purge_buffer` from config.

### Task 3.0.5 — Add ElasticNetCV to `src/models/regularized_models.py`
```python
def build_elasticnet_pipeline(
    l1_ratios: list[float] = [0.1, 0.5, 0.9, 0.95, 1.0],
    alphas: np.ndarray | None = None,  # default: 50 log-spaced from 1e-4 to 1e2
    cv_splits: int = 3,
) -> Pipeline
```
Update `run_wfo()` Literal to accept `"elasticnet"` as `model_type`. Update `multi_benchmark_wfo.py` to default to `"elasticnet"` for v3.0+ runs.

### Task 3.0.6 — Add statistical rigor to `src/reporting/backtest_report.py`
New functions:
```python
def compute_oos_r_squared(predicted: pd.Series, realized: pd.Series) -> float
# Campbell-Thompson OOS R²: 1 - MSE_model / MSE_historical_mean

def apply_bhy_correction(
    results: list[BacktestEventResult],
    alpha: float = 0.05,
) -> dict[str, bool]
# Benjamini-Hochberg-Yekutieli correction across benchmarks
# Returns: {benchmark_ticker: passes_bhy}

def compute_newey_west_ic(predictions: pd.Series, realized: pd.Series, lags: int) -> tuple[float, float]
# Returns (IC, p_value) with HAC standard errors
```

### Task 3.0.7 — Monthly Stability Backtesting in `src/backtest/vesting_events.py`
```python
def enumerate_monthly_evaluation_dates(
    start_year: int = 2014,
    end_year: int | None = None,
) -> list[VestingEvent]
# Month-end dates from start_year to end_year; not actual vesting events
# Returns VestingEvent-compatible objects for the backtest engine
```
**Key invariant to test:** `monthly_evaluation_dates ∩ vesting_event_dates` must produce identical predictions via both code paths.

Update `src/backtest/backtest_engine.py` to add:
```python
def run_monthly_stability_backtest(conn: sqlite3.Connection) -> list[BacktestEventResult]
# Evaluates 120+ month-end dates; uses same temporal slicing logic as run_full_backtest()
```

Add to `src/reporting/backtest_report.py`:
```python
def generate_rolling_ic_series(results: list[BacktestEventResult], window: int = 24) -> pd.DataFrame
def generate_regime_breakdown(results: list[BacktestEventResult]) -> pd.DataFrame
# Regime: bull/bear × low/high vol (4 quadrants); OOS R² and hit rate per quadrant
```

### Task 3.0.8 — Create `scripts/monthly_decision.py`
New script for automated monthly decision generation.

```python
def main(as_of_date: date | None = None, dry_run: bool = False) -> None
# as_of_date defaults to today; supports --as-of YYYY-MM-DD CLI arg for backtesting
# Steps:
# 1. Fetch latest FRED data (upsert to DB)
# 2. Build feature matrix as-of as_of_date (strict temporal cutoff)
# 3. Run multi-benchmark WFO (ElasticNet)
# 4. Generate VestingRecommendation (reuse rebalancer.py)
# 5. Write results to results/monthly_decisions/YYYY-MM/
# 6. Append summary row to results/monthly_decisions/decision_log.md
```

Output files per run:
- `results/monthly_decisions/YYYY-MM/recommendation.md` — Formatted sell/hold report
- `results/monthly_decisions/YYYY-MM/signals.csv` — Per-benchmark signal details
- `results/monthly_decisions/YYYY-MM/backtest_summary.csv` — Monthly stability stats
- `results/monthly_decisions/YYYY-MM/plots/` — IC time series, regime breakdown charts

### Task 3.0.9 — Create `results/monthly_decisions/decision_log.md` (initial file)
Initialize with markdown table header and one example/placeholder row explaining the schema.

### Task 3.0.10 — Create `.github/workflows/monthly_decision.yml`
```yaml
name: Monthly Decision Report
on:
  schedule:
    - cron: '0 14 20 * *'       # 20th of each month at 14:00 UTC (primary)
    - cron: '0 14 21 * *'       # 21st (if 20th falls on Saturday)
    - cron: '0 14 22 * *'       # 22nd (if 20th falls on Sunday)
  workflow_dispatch:
    inputs:
      as_of_date:
        description: 'Override date (YYYY-MM-DD); defaults to today'
        required: false
      dry_run:
        description: 'Dry run (no commit)'
        type: boolean
        default: false
```
**Note on weekend handling:** The cron fires on 20th, 21st, and 22nd. The script itself checks if today is the first run after the 20th and skips if a result for this month already exists (idempotent). This handles the "first business day after the 20th" requirement in pure Python without needing a custom cron expression.

Steps: checkout → setup Python 3.11 → pip install → run `scripts/monthly_decision.py` → git commit results (skip on dry_run) → push.

### Task 3.0.11 — Update `scripts/weekly_fetch.py` and `.github/workflows/weekly_data_fetch.yml`
`weekly_fetch.py`: Add a `fetch_fred_macro_step()` call at the end of `main()` that fetches all `FRED_SERIES_MACRO` series and upserts to `fred_macro_monthly`. Add `--skip-fred` CLI flag for budget management during testing.

`weekly_data_fetch.yml`: Add `FRED_API_KEY` to the env block alongside `AV_API_KEY` and `FMP_API_KEY`.

### Task 3.0.12 — Tests for v3.0
- `tests/test_fred_loader.py` — mock HTTP; verify monthly resampling, NaN handling, dry_run, missing key raises
- `tests/test_fred_db.py` — schema creation, upsert idempotence, `get_fred_macro()` column alignment
- `tests/test_fred_features.py` — verify 6 derived FRED features present in feature matrix; no future leakage
- `tests/test_elasticnet.py` — pipeline builds; l1_ratio grid; temporal isolation of StandardScaler
- `tests/test_embargo_fix.py` — assert gap=8 for 6M, gap=15 for 12M; verify old gap=6 behavior still accessible via `purge_buffer=0`
- `tests/test_monthly_backtest.py` — `enumerate_monthly_evaluation_dates()` count ≥ 120; vesting date intersection invariant; no-lookahead check
- `tests/test_oos_r2.py` — OOS R² = 0 when predicted = historical mean; BHY correction rejects < 5% FDR; Newey-West returns finite p-value
- `tests/test_monthly_decision_script.py` — dry_run produces output files without committing; idempotency (second run skips if result exists for month)

### v3.0 Verification
```bash
pytest tests/test_fred_loader.py tests/test_fred_db.py tests/test_fred_features.py \
       tests/test_elasticnet.py tests/test_embargo_fix.py tests/test_monthly_backtest.py \
       tests/test_oos_r2.py tests/test_monthly_decision_script.py -v
python scripts/monthly_decision.py --dry-run --as-of 2026-01-20
# Verify results/monthly_decisions/2026-01/ created; decision_log.md appended
```

---

## v3.1 — BayesianRidge Ensemble + Fractional Kelly + PGR-Specific FRED + Regime Diagnostics

**Goal:** Add uncertainty quantification via BayesianRidge, implement equal-weight forecast combination across 3 models, add PGR-specific insurance features, replace fixed sell-% tiers with fractional Kelly sizing, and deliver rolling IC / regime breakdown reports.

### New Config Keys (`config.py`)
```python
KELLY_FRACTION: float = 0.25         # quarter-Kelly for personal portfolio
KELLY_MAX_POSITION: float = 0.30     # cap single-stock allocation at 30%
ENSEMBLE_MODELS: list[str] = ["elasticnet", "ridge", "bayesian_ridge"]
```

### Task 3.1.1 — Add BayesianRidge to `src/models/regularized_models.py`
```python
def build_bayesian_ridge_pipeline(
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
) -> Pipeline
# Note: BayesianRidge.predict(X, return_std=True) returns (y_pred, y_std)
# The pipeline must expose this; wrap in a thin UncertaintyPipeline class
# that delegates predict_with_std() to the final estimator.
```

### Task 3.1.2 — Equal-Weight Ensemble in `src/models/multi_benchmark_wfo.py`
Add `run_ensemble_benchmarks()` that trains ElasticNet + Ridge + BayesianRidge independently per benchmark, then averages predictions (equal weight). Returns an `EnsembleWFOResult` with:
- `point_prediction`: mean of 3 model predictions
- `prediction_std`: std dev from BayesianRidge `return_std`
- `signal_to_noise`: `|point_prediction| / prediction_std`
- All existing IC/hit_rate/MAE metrics from the ensemble

### Task 3.1.3 — Fractional Kelly Sizing in `src/portfolio/rebalancer.py`
Replace the fixed-tier `_compute_sell_pct()` with Kelly-based sizing:
```python
def _compute_sell_pct_kelly(
    predicted_excess_return: float,
    prediction_variance: float,
    kelly_fraction: float = KELLY_FRACTION,
    max_position: float = KELLY_MAX_POSITION,
) -> float:
    # f* = kelly_fraction × predicted_excess_return / prediction_variance
    # Clamp to [0.0, 1.0]; translate to sell % as (1 - f*)
    # When signal is OUTPERFORM with high conviction → hold more (sell less)
```
Keep old `_compute_sell_pct()` as `_compute_sell_pct_legacy()` for backward compatibility in backtests.

Update `VestingRecommendation` dataclass to add `prediction_std: float` and `kelly_fraction_used: float`.

### Task 3.1.4 — Add PGR-Specific FRED Features (`src/ingestion/fred_loader.py`)
Extend `fetch_all_fred_macro()` to also fetch `FRED_SERIES_PGR` (CUSR0000SETC01, TRFVOLUSM227NFWA). Trigger this in `weekly_fetch.py` via existing FRED step (no code change to workflow needed).

Add to `feature_engineering.py`:
- `insurance_cpi_mom3m`: 3-month momentum of CUSR0000SETC01
- `vmt_yoy`: year-over-year change in TRFVOLUSM227NFWA

### Task 3.1.5 — Regime Diagnostic Reports in `src/reporting/backtest_report.py`
```python
def generate_rolling_ic_series(
    results: list[BacktestEventResult],
    window_months: int = 24,
) -> pd.DataFrame
# Returns: DatetimeIndex, columns=[ic_rolling, hit_rate_rolling, oos_r2_rolling]

def generate_regime_breakdown(
    results: list[BacktestEventResult],
    sp500_returns: pd.Series,   # for bull/bear classification
    vix_series: pd.Series,      # for low/high vol classification
) -> pd.DataFrame
# 4-quadrant table: bull/bear × low/high vol → OOS R², hit rate, n_obs per cell
```

Add SP500 (SPY) and VIX proxy to the backtest engine's data loading (SPY already in ETF universe; VIX via FRED series `VIXCLS`). Add `VIXCLS` to `FRED_SERIES_MACRO` in config.

Update `scripts/monthly_decision.py` to generate rolling IC chart and regime table in `results/monthly_decisions/YYYY-MM/plots/`.

### Task 3.1.6 — Tests for v3.1
- `tests/test_bayesian_ridge.py` — `predict_with_std()` returns finite std; temporal isolation; ensemble averaging correct
- `tests/test_kelly_sizing.py` — Kelly formula math; max position cap; zero-prediction → 100% sell
- `tests/test_pgr_fred_features.py` — insurance CPI momentum and VMT YoY present in feature matrix post-v3.1
- `tests/test_regime_breakdown.py` — 4 quadrants populated; OOS R² between -1 and 1; rolling IC window = 24

### v3.1 Verification
```bash
pytest tests/test_bayesian_ridge.py tests/test_kelly_sizing.py \
       tests/test_pgr_fred_features.py tests/test_regime_breakdown.py -v
python scripts/monthly_decision.py --dry-run
# Verify recommendation.md shows Kelly-based sell%, prediction_std populated
# Verify regime_breakdown table in backtest_summary.csv
```

---

## v4.0 — CPCV + Black-Litterman + Tax-Loss Harvesting + Fractional Differentiation

**Goal:** Production-grade validation (CPCV), optimal portfolio construction (Black-Litterman), proactive tax management (TLH with wash-sale), and signal quality improvements (fractional differentiation).

### New Dependencies (add to `requirements.txt`)
```
skfolio>=0.3.0          # CPCV via CombinatorialPurgedCV
PyPortfolioOpt>=1.5.5   # BlackLittermanModel
fracdiff>=0.1.0         # FracdiffStat for fractional differentiation
```

### Task 4.0.1 — Combinatorial Purged CV in `src/models/wfo_engine.py`
Add `run_cpcv()` alongside `run_wfo()`:
```python
def run_cpcv(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: Literal["elasticnet", "ridge", "bayesian_ridge"],
    target_horizon_months: int,
    n_folds: int = 6,
    n_test_folds: int = 2,
) -> CPCVResult
# Uses skfolio.model_selection.CombinatorialPurgedCV
# Returns: distribution of IC across 15 train-test splits; 5 backtest paths
# CPCVResult.path_ics: list of per-path IC sequences (for overfitting detection)
```
CPCV is a validation/diagnostic tool, not a replacement for WFO — keep `run_wfo()` as the primary production path.

### Task 4.0.2 — Black-Litterman Portfolio Construction
New file: `src/portfolio/black_litterman.py`
```python
def build_bl_weights(
    ensemble_signals: dict[str, EnsembleWFOResult],  # from run_ensemble_benchmarks()
    covariance_matrix: pd.DataFrame,                  # Ledoit-Wolf shrunk
    risk_aversion: float = 2.5,
    view_confidence_scalar: float = 1.0,             # scales Ω = RMSE² × scalar
) -> dict[str, float]
# Uses PyPortfolioOpt BlackLittermanModel
# Views: predicted excess return per ETF benchmark
# Ω_ii = model's cross-validation RMSE² (uncertainty-scaled view confidence)
# Returns: optimal portfolio weights (ETF allocation targets)
```
Integrate into `rebalancer.py`: when BL weights available, use them as `reallocation_targets` instead of the existing drift-based targets.

### Task 4.0.3 — Tax-Loss Harvesting in `src/tax/capital_gains.py`
```python
def identify_tlh_candidates(
    tax_lots: list[TaxLot],
    current_price: float,
    loss_threshold: float = -0.10,  # -10% return triggers harvest
) -> list[TaxLot]
# Returns lots where unrealized return < loss_threshold

def compute_after_tax_expected_return(
    predicted_return: float,
    unrealized_gain_fraction: float,
    tax_rate: float,
) -> float:
    # after_tax = predicted_return - max(0, unrealized_gain_fraction × tax_rate)
    # Creates natural asymmetry: losses increase sell incentive, gains raise hurdle

def suggest_tlh_replacement(harvested_ticker: str) -> str | None
# Returns a correlated-but-not-substantially-identical replacement ETF
# Wash-sale rule: 31+ day separation required
# PGR → no direct substitute; ETF pairs defined in config
```
Add `TLH_REPLACEMENT_MAP: dict[str, str]` to `config.py` (e.g., `{"VTI": "ITOT", "VGT": "QQQ"}`).

### Task 4.0.4 — Fractional Differentiation in `src/processing/feature_engineering.py`
```python
def apply_fracdiff(series: pd.Series, max_d: float = 0.5) -> tuple[pd.Series, float]:
    # Uses fracdiff.FracdiffStat to find minimum d* achieving stationarity
    # while preserving ≥ 90% correlation with original series
    # Returns: (differenced_series, d_star)
    # Apply to: raw log prices (not returns), any non-stationary feature series
    # Do NOT apply to returns (already d=1 differenced)
```
Add `use_fracdiff: bool = False` parameter to `build_feature_matrix()`. Default False for backward compatibility; set True in v4.0 production runs.

### Task 4.0.5 — Per-Benchmark Weighting in Recommendation Layer
Update `rebalancer.py` to weight benchmark signals by their historical stability:
```python
def compute_benchmark_weights(
    monthly_stability_results: list[BacktestEventResult],
) -> dict[str, float]:
    # Weight = normalized (rolling 24M IC × hit_rate) per benchmark
    # Benchmarks with negative IC get weight = 0
    # Weights sum to 1.0 across active benchmarks
```
The final sell recommendation uses the weighted average signal across benchmarks instead of a simple majority vote.

### Task 4.0.6 — Tests for v4.0
- `tests/test_cpcv.py` — 15 splits from n_folds=6, n_test_folds=2; path count = 5; temporal ordering preserved
- `tests/test_black_litterman.py` — weights sum to 1.0; higher confidence → closer to view; Ledoit-Wolf shrinkage applied
- `tests/test_tlh.py` — harvest triggered at -10% threshold; wash-sale date gap ≥ 31 days; after-tax return formula correct
- `tests/test_fracdiff.py` — d* in [0.0, 0.5]; correlation ≥ 0.90 with original; stationarity ADF p < 0.05
- `tests/test_benchmark_weights.py` — weights sum to 1.0; zero-IC benchmarks get zero weight; normalization correct

### v4.0 Verification
```bash
pytest tests/test_cpcv.py tests/test_black_litterman.py tests/test_tlh.py \
       tests/test_fracdiff.py tests/test_benchmark_weights.py -v
python scripts/monthly_decision.py --dry-run
# Verify BL weights in recommendation.md; TLH candidates section present
# Verify per-benchmark weights in signals.csv
```

---

## Critical Files Modified Per Version

### v3.0
| File | Change Type |
|------|-------------|
| `requirements.txt` | Add statsmodels |
| `config.py` | FRED keys, series lists, purge buffer constants |
| `src/ingestion/fred_loader.py` | **NEW** |
| `src/database/schema.sql` | Add fred_macro_monthly table |
| `src/database/db_client.py` | Add upsert_fred_macro(), get_fred_macro() |
| `src/processing/feature_engineering.py` | Add FRED feature merge, 6 derived features |
| `src/models/regularized_models.py` | Add build_elasticnet_pipeline() |
| `src/models/wfo_engine.py` | Add purge_buffer param, fix gap calculation |
| `src/models/multi_benchmark_wfo.py` | Default to elasticnet, pass purge_buffer |
| `src/backtest/vesting_events.py` | Add enumerate_monthly_evaluation_dates() |
| `src/backtest/backtest_engine.py` | Add run_monthly_stability_backtest() |
| `src/reporting/backtest_report.py` | Add OOS R², BHY correction, Newey-West IC, regime breakdown |
| `scripts/weekly_fetch.py` | Add FRED fetch step |
| `scripts/monthly_decision.py` | **NEW** |
| `.github/workflows/weekly_data_fetch.yml` | Add FRED_API_KEY to env |
| `.github/workflows/monthly_decision.yml` | **NEW** |
| `results/monthly_decisions/decision_log.md` | **NEW** (initialized with header) |

### v3.1
| File | Change Type |
|------|-------------|
| `config.py` | KELLY_FRACTION, KELLY_MAX_POSITION, ENSEMBLE_MODELS, add VIXCLS to FRED_SERIES_MACRO |
| `src/models/regularized_models.py` | Add build_bayesian_ridge_pipeline(), UncertaintyPipeline wrapper |
| `src/models/multi_benchmark_wfo.py` | Add run_ensemble_benchmarks(), EnsembleWFOResult |
| `src/portfolio/rebalancer.py` | Add _compute_sell_pct_kelly(), update VestingRecommendation |
| `src/ingestion/fred_loader.py` | Add PGR-specific series to fetch |
| `src/processing/feature_engineering.py` | Add insurance_cpi_mom3m, vmt_yoy features |
| `src/reporting/backtest_report.py` | Add generate_rolling_ic_series(), generate_regime_breakdown() |
| `scripts/monthly_decision.py` | Add rolling IC and regime charts to output |

### v4.0
| File | Change Type |
|------|-------------|
| `requirements.txt` | Add skfolio, PyPortfolioOpt, fracdiff |
| `src/models/wfo_engine.py` | Add run_cpcv() |
| `src/portfolio/black_litterman.py` | **NEW** |
| `src/portfolio/rebalancer.py` | Integrate BL weights, benchmark weighting |
| `src/tax/capital_gains.py` | Add TLH functions |
| `src/processing/feature_engineering.py` | Add apply_fracdiff(), use_fracdiff param |
| `config.py` | Add TLH_REPLACEMENT_MAP |

---

## Implementation Order Within Each Version

Each version should be implemented in this order to minimize rework:
1. Config changes (constants, new env vars)
2. Schema / DB layer (tables, upsert helpers)
3. Ingestion / data fetching (new loaders)
4. Processing / feature engineering (integrate new data)
5. Model layer (new algorithms)
6. Backtest / validation (new evaluation methods)
7. Reporting (new metrics and outputs)
8. Portfolio / recommendation layer (new sizing logic)
9. Scripts and GitHub Actions (automation)
10. Tests (write + run for each module before moving to next)

**Rule:** Never finalize a module without a passing pytest suite for it (CLAUDE.md mandate).
