# PGR Vesting Decision Support — v2 Development Plan

## Context

The v1 engine (fully built, 69 tests passing) predicts whether to hold or sell PGR RSUs at each vesting event using Walk-Forward Optimization (WFO) with Lasso regression. It only models PGR in isolation — there is no relative return comparison against alternative investments, and historical vesting events have never been backtested.

**v2 goal:** Build a relative return prediction engine that answers: *"At this vesting event, is PGR statistically likely to outperform each of the 22 alternative funds over the next 6 or 12 months?"* Back-test this question against all mid-January and mid-July events going back ≥10 years. All shares are in a taxable brokerage — NUA/retirement logic remains out of scope.

**Primary drivers:**
- Gemini peer review recommendations (SQLite accumulation pipeline, relative return target variable, fixed embargo, multi-benchmark modeling)
- User requirement: 6M and 12M prediction horizons (not the 90-day in the peer review)
- User requirement: backtest all 22 benchmark ETFs, not just SPY

---

## Known v1 Bug: Embargo Too Short

`config.WFO_EMBARGO_MONTHS = 1` is incorrect. With a 6-month forward return target computed on monthly observations, consecutive observations share 5 months of overlapping return window. This is autocorrelation leakage. The embargo must equal the target horizon:
- 6M target → embargo = 6 months
- 12M target → embargo = 12 months

This fix reduces the number of WFO folds but is methodologically non-negotiable.

---

## Known Data Gaps: Short-History and Non-US ETFs

| Ticker | Issue | Proxy |
|--------|-------|-------|
| FZROX, FNILX | Launched 2018-08-02 | VTI (total market, identical exposure) |
| FZILX | Launched 2018-08-02 | VXUS (total international) |
| DGRO | Launched 2014-06-11 | VIG (dividend appreciation) |
| VBTLX | Mutual fund (variable AV availability) | BND (same index) |
| VEE | Canadian TSX ETF (limited AV data) | IEMG (MSCI emerging markets) |

Pre-launch periods are filled from proxy tickers, flagged `proxy_fill=1` in the DB. A 10-year backtest is achievable for all 22 benchmarks using this approach.

---

## Architecture Decision: 22 Separate WFO Models

One LassoCV/RidgeCV model per ETF benchmark. PGR-vs-BND has a fundamentally different statistical character than PGR-vs-IEMG; a multi-output model would apply the same regularization to both, which is methodologically incorrect. Computational cost is negligible (each fold < 2s on ~180 rows × 15 features; 22 models × 41 folds ≈ 60 seconds total).

---

## Phase 1: Database Infrastructure
**New files:** `src/database/schema.sql`, `src/database/db_client.py`, `scripts/migrate_v1_to_v2.py`, `tests/test_db_client.py`
**Modified files:** `config.py`

### SQLite Schema (`pgr_financials.db`)

```sql
daily_prices(ticker, date PK, open, high, low, close, volume, source, proxy_fill)
daily_dividends(ticker, ex_date PK, amount, source)
split_history(ticker, split_date PK, split_ratio, numerator, denominator)
pgr_fundamentals_quarterly(period_end PK, pe_ratio, pb_ratio, roe, eps, revenue, net_income)
pgr_edgar_monthly(month_end PK, combined_ratio, pif_total, pif_growth_yoy, gainshare_estimate)
monthly_relative_returns(date, benchmark, target_horizon PK, pgr_return, benchmark_return, relative_return, proxy_fill)
api_request_log(api, date PK, endpoint, count)
ingestion_metadata(ticker, data_type PK, last_fetched, rows_stored)
```

### `db_client.py` Key Functions
- `get_connection(db_path) -> sqlite3.Connection` — WAL mode, FK enforcement
- `initialize_schema(conn)` — idempotent CREATE TABLE IF NOT EXISTS
- `upsert_prices(conn, records)` / `upsert_dividends(conn, records)` — bulk INSERT OR REPLACE
- `get_prices(conn, ticker, start_date, end_date) -> pd.DataFrame`
- `log_api_request(conn, api, endpoint)` — raises RuntimeError if daily limit exceeded
- `get_api_request_count(conn, api, date) -> int`

### `config.py` Additions
```python
DB_PATH: str = "data/pgr_financials.db"
WFO_EMBARGO_MONTHS_6M: int = 6      # fixes v1 bug (was 1)
WFO_EMBARGO_MONTHS_12M: int = 12
WFO_TARGET_HORIZONS: list[int] = [6, 12]
ETF_BENCHMARK_UNIVERSE: list[str]   # 22 ETFs (moved from drift_analyzer hardcode)
ETF_PROXY_MAP: dict[str, str]       # {"FZROX": "VTI", "FNILX": "VTI", ...}
ETF_LAUNCH_DATES: dict[str, str]    # {"FZROX": "2018-08-02", ...}
AV_FETCH_GROUPS: dict[str, list[str]]  # ETF groups for day-of-week scheduling
```

### Migration Script
`scripts/migrate_v1_to_v2.py` — one-time: reads existing JSON cache + Parquet + `pgr_edgar_cache.csv`, inserts into `pgr_financials.db`. Does not delete v1 files.

### Tests (`tests/test_db_client.py`)
- Schema creation is idempotent (run twice, no error)
- Duplicate upsert replaces, does not duplicate
- `log_api_request` raises RuntimeError at limit
- `get_prices` respects date filters
- `proxy_fill=1` round-trips correctly

---

## Phase 2: Multi-Ticker Data Accumulation Pipeline
**New files:** `src/ingestion/multi_ticker_loader.py`, `src/ingestion/multi_dividend_loader.py`, `src/ingestion/fetch_scheduler.py`, `scripts/daily_fetch.py`, `.github/workflows/daily_data_fetch.yml`, `tests/test_multi_ticker_loader.py`
**Modified files:** `src/ingestion/av_client.py`, `src/ingestion/fmp_client.py`

### `MultiTickerLoader` Class
- `fetch_ticker_prices(ticker, force_refresh) -> int` — AV `TIME_SERIES_DAILY` with `outputsize=full`; upserts to DB
- `fetch_all_prices(tickers, dry_run) -> dict[str, int]` — batch with rate limit enforcement
- `fill_proxy_history(ticker, proxy_ticker, cutoff_date) -> int` — copies proxy rows flagged `proxy_fill=1`

### `MultiDividendLoader` Class
- `fetch_dividends(ticker) -> int` — AV `DIVIDENDS` endpoint, upserts to `daily_dividends`

### Fetch Scheduler (`fetch_scheduler.py`)
- `get_tickers_for_today(today) -> list[str]` — PGR always included; 22 ETFs split across weekday groups (11/day) to stay under 25 AV calls/day
- Schedule: PGR price + PGR dividends (2 calls) + 11 ETF prices = 13 calls/day max

### GitHub Actions Cron (`.github/workflows/daily_data_fetch.yml`)
```yaml
schedule: '30 16 * * 1-5'   # 4:30 PM UTC, weekdays (after US market close)
```
Steps: checkout → install deps → run `scripts/daily_fetch.py` (reads AV/FMP keys from secrets) → git commit `data/pgr_financials.db` → push

Note: `data/pgr_financials.db` must be tracked in git (not in `.gitignore`). Update `.gitignore` to un-ignore it.

### `av_client.py` / `fmp_client.py` Modifications
Replace `_request_counts.json` JSON counter with calls to `db_client.log_api_request()`. All existing cache-first logic preserved.

### Tests
- Mocked AV response produces correct DB row count
- `fill_proxy_history` sets `proxy_fill=1` on all copied rows
- `get_tickers_for_today` always contains PGR
- Different weekdays return different ETF groups
- Budget enforcement: 25+ logged AV calls → RuntimeError
- Re-running same fetch does not duplicate rows

---

## Phase 3: Relative Return & Feature Engineering Extensions
**New files:** `src/processing/multi_total_return.py`, `tests/test_multi_total_return.py`
**Modified files:** `src/processing/feature_engineering.py`

### `multi_total_return.py` Key Functions
- `build_etf_monthly_returns(conn, ticker, forward_months) -> pd.Series` — generalizes v1's `build_monthly_returns()` for any ticker; calls existing `build_position_series()` (no changes to `total_return.py`)
- `build_relative_return_targets(conn, forward_months) -> pd.DataFrame` — for each of 22 ETFs: compute PGR return minus ETF return; upsert to `monthly_relative_returns` table; return DataFrame (index=date, columns=ETF tickers)
- `load_relative_return_matrix(conn, benchmark, forward_months) -> pd.Series` — load pre-computed series from DB for one benchmark/horizon

### `feature_engineering.py` Additions
- `build_feature_matrix_from_db(conn, force_refresh) -> pd.DataFrame` — DB-backed entry point; calls existing `build_feature_matrix()` unchanged (backward compatible)
- `get_X_y_relative(df, relative_returns, drop_na_target) -> tuple[pd.DataFrame, pd.Series]` — aligns feature matrix to one ETF's relative return series; drops NaN rows

Optional new features (added to existing `build_feature_matrix()`):
- `pgr_vs_spy_52w_rel`: PGR 52-week return minus SPY 52-week return (mean-reversion signal)
- `pgr_beta_rolling_12m`: rolling 12-month beta vs VTI

### Tests
- `build_etf_monthly_returns` produces Series with correct length and name
- `build_relative_return_targets` returns DataFrame with 22 columns
- Relative return = PGR minus ETF (sign and magnitude verified with synthetic data)
- `proxy_fill=1` rows excluded when `exclude_proxy=True`
- `get_X_y_relative` raises ValueError on zero-overlap index

---

## Phase 4: WFO Engine Upgrade
**Modified files:** `src/models/wfo_engine.py`, `src/portfolio/rebalancer.py`, `tests/test_wfo_engine.py`
**New files:** `src/models/multi_benchmark_wfo.py`, `tests/test_multi_benchmark_wfo.py`

### `wfo_engine.py` Changes
**1. Embargo fix** — new `target_horizon_months` parameter drives embargo:
```python
def run_wfo(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: Literal["lasso", "ridge"] = "lasso",
    target_horizon_months: int = 6,   # NEW: embargo = this value
    train_window_months: int | None = None,
    test_window_months: int | None = None,
) -> WFOResult:
```

**2. WFOResult dataclass extension:**
```python
benchmark: str = ""          # which ETF this was run against
target_horizon: int = 6
model_type: str = "lasso"
```

**3. `predict_current()` refit fix** — replace v1 placeholder (last-fold mean) with actual refit on most-recent `train_window_months` of data before predicting on current observation.

### `multi_benchmark_wfo.py` Key Functions
- `run_all_benchmarks(X, relative_return_matrix, model_type, target_horizon_months) -> dict[str, WFOResult]` — loop over 22 ETF columns; one `run_wfo()` per benchmark
- `get_current_signals(X_full, relative_return_matrix, ...) -> pd.DataFrame` — columns: `predicted_relative_return`, `ic`, `hit_rate`, `signal` (HOLD/SELL/NEUTRAL)

### `rebalancer.py` Changes
- `VestingRecommendation` gains `benchmark_signals: dict[str, dict]`, `target_horizon: int`, `recommended_sell_pct_6m: float`, `recommended_sell_pct_12m: float`
- `generate_recommendation()` accepts `multi_benchmark_results: dict[str, WFOResult]` and aggregates signals (weighted vote: sell if majority of benchmarks predict PGR underperformance)

### Tests
- `test_embargo_equals_target_horizon_6m`: gap between train end and test start ≥ 6 months (this test WILL FAIL on v1, confirming the bug)
- `test_embargo_equals_target_horizon_12m`: same for 12M
- `test_benchmark_stored_in_wfo_result`
- `test_predict_current_refit_uses_recent_data` (not mean of last fold)
- `test_run_all_benchmarks_returns_22_results`
- `test_no_benchmark_shares_train_data_with_test`

---

## Phase 5: Historical Vesting Event Backtest Engine
**New files:** `src/backtest/vesting_events.py`, `src/backtest/backtest_engine.py`, `tests/test_backtest_engine.py`

### Vesting Event Enumeration (`vesting_events.py`)
```python
@dataclass
class VestingEvent:
    event_date: date
    rsu_type: Literal["time", "performance"]
    horizon_6m_end: date
    horizon_12m_end: date

def enumerate_vesting_events(start_year: int = 2014, end_year: int | None = None) -> list[VestingEvent]:
```
- January vesting: nearest business day to Jan 19 each year
- July vesting: nearest business day to Jul 17 each year
- `end_year` defaults to last year with a fully realized 12M return (current_year - 2 to be safe; so up to July 2024 for 12M, January 2025 for 12M)
- Expected output: ~22–24 events for a 2014–2024 backtest window

### Backtest Engine (`backtest_engine.py`)
```python
@dataclass
class BacktestEventResult:
    event: VestingEvent
    benchmark: str
    target_horizon: int
    predicted_relative_return: float
    realized_relative_return: float
    signal_direction: Literal["OUTPERFORM", "UNDERPERFORM"]
    correct_direction: bool
    predicted_sell_pct: float
    ic_at_event: float
    hit_rate_at_event: float
    n_train_observations: int
    proxy_fill_fraction: float   # fraction of training obs using proxy data

def run_historical_backtest(conn, model_type, target_horizon_months) -> list[BacktestEventResult]:
def run_full_backtest(conn, model_type: str = "lasso") -> pd.DataFrame:
```

**Temporal integrity:** For each vesting event date `t`:
1. Feature matrix sliced to all rows ≤ `t`
2. Relative return target sliced to all rows ≤ `t - embargo` (never exposes future overlapping returns during training)
3. `predict_current()` generates signal as of date `t`
4. Realized outcome loaded from pre-computed `monthly_relative_returns` table

### Tests
- January events are within 5 business days of Jan 19
- July events are within 5 business days of Jul 17
- No training observation postdates event_date
- Realized return matches DB value (spot check)
- `correct_direction` logic verified with synthetic data
- Row count: N_events × 22 benchmarks × 2 horizons = expected total (e.g., 22 × 22 × 2 = 968 rows)

---

## Phase 6: Reporting & Visualization
**New files:** `src/reporting/backtest_report.py`, `tests/test_reporting.py`
**Modified files:** `src/visualization/plots.py`

### `backtest_report.py` Key Functions
- `generate_backtest_table(results, horizon) -> pd.DataFrame` — rows=event dates, columns=ETF tickers, values=realized relative return; separate table for predictions and correct_direction flags
- `print_backtest_summary(results)` — overall hit rate, by horizon, by RSU type, top/bottom 5 benchmarks, avg IC
- `export_backtest_to_csv(results, path)` — full detail dump for offline analysis

### New Plot Functions (added to `plots.py`)
- `plot_backtest_heatmap(results, horizon) -> str` — rows=vesting events, columns=ETF benchmarks, color=realized outperformance, markers=correct/incorrect prediction
- `plot_hit_rate_by_benchmark(results) -> str` — horizontal bar chart of directional hit rate per ETF (two bars: 6M and 12M)
- `plot_predicted_vs_realized_scatter(results, benchmark) -> str` — scatter per event, colored by RSU type
- `plot_multi_benchmark_signals(signals_df) -> str` — current prediction bar chart: green=hold, red=sell PGR

### Tests
- Table dimensions correct
- Summary prints without error
- CSV has expected columns
- Plots write PNG files (tmp_path fixture)

---

## Phase 7: Test Hardening
**Modified files:** `tests/test_wfo_engine.py`
**New files:** `tests/test_integration.py`, `pytest.ini`

### Fix Existing Test
`test_wfo_engine.py::test_embargo_gap_enforced` — update assertion from `≥ 28 days` to `≥ 168 days` (6 months). Initially fails on v1, passes after Phase 4 fix.

### Integration Smoke Test (`test_integration.py`)
End-to-end: temp SQLite DB → synthetic prices for PGR + 3 ETFs → feature matrix → relative returns → WFO (3 ETF models) → backtest (4 synthetic events) → report table. Verifies composition across all phase boundaries.

### `pytest.ini`
```ini
[pytest]
testpaths = tests
addopts = --tb=short -q
markers =
    integration: marks tests as integration (slow, real I/O)
    unit: fast, no I/O
```

---

## File Map Summary

| File | Action | Phase |
|------|--------|-------|
| `config.py` | Modify | 1 |
| `src/database/schema.sql` | New | 1 |
| `src/database/db_client.py` | New | 1 |
| `scripts/migrate_v1_to_v2.py` | New | 1 |
| `tests/test_db_client.py` | New | 1 |
| `src/ingestion/multi_ticker_loader.py` | New | 2 |
| `src/ingestion/multi_dividend_loader.py` | New | 2 |
| `src/ingestion/fetch_scheduler.py` | New | 2 |
| `scripts/daily_fetch.py` | New | 2 |
| `.github/workflows/daily_data_fetch.yml` | New | 2 |
| `src/ingestion/av_client.py` | Modify | 2 |
| `src/ingestion/fmp_client.py` | Modify | 2 |
| `tests/test_multi_ticker_loader.py` | New | 2 |
| `src/processing/multi_total_return.py` | New | 3 |
| `src/processing/feature_engineering.py` | Modify | 3 |
| `tests/test_multi_total_return.py` | New | 3 |
| `src/models/wfo_engine.py` | Modify | 4 |
| `src/models/multi_benchmark_wfo.py` | New | 4 |
| `src/portfolio/rebalancer.py` | Modify | 4 |
| `tests/test_wfo_engine.py` | Modify | 4 |
| `tests/test_multi_benchmark_wfo.py` | New | 4 |
| `src/backtest/vesting_events.py` | New | 5 |
| `src/backtest/backtest_engine.py` | New | 5 |
| `tests/test_backtest_engine.py` | New | 5 |
| `src/reporting/backtest_report.py` | New | 6 |
| `src/visualization/plots.py` | Modify | 6 |
| `tests/test_reporting.py` | New | 6 |
| `tests/test_integration.py` | New | 7 |
| `pytest.ini` | New | 7 |

**New files: 20 | Modified files: 9 | Total tests after v2: ~120 (69 existing + ~51 new)**

---

## Verification Plan (per phase)

Each phase ends with:
1. `pytest tests/test_<phase_module>.py -v` passes fully before proceeding
2. Any modified existing test file runs its full original suite + new tests
3. Phase 7 integration test confirms the full pipeline composes correctly end-to-end

Final end-to-end verification:
- Run `scripts/migrate_v1_to_v2.py` → confirm row counts match v1 data
- Run `scripts/daily_fetch.py --dry-run` → confirm scheduler logic, no actual API calls
- Run `python -m src.backtest.backtest_engine` (or a driver script) → produce backtest report for 2014–2024
- Inspect `plots/backtest_heatmap_6m.png` and `plots/hit_rate_by_benchmark.png` visually

---

## Notes & Open Questions

1. **VEE (Canadian ETF):** VEE trades on the Toronto Stock Exchange. Alpha Vantage free tier may not carry it. The plan proposes substituting IEMG as a US-listed equivalent. If the user wants to keep VEE, this will need a paid AV tier or a different data source.

2. **VBTLX (mutual fund):** AV coverage is uncertain. BND tracks the same Bloomberg US Aggregate index. Plan treats VBTLX and BND as interchangeable for price/return purposes with `proxy_fill=1` flagging.

3. **Vesting dates:** The plan uses the specific dates from config (`Jan 19`, `Jul 17`) and snaps to the nearest business day. If the actual vesting dates vary year-to-year (e.g., third Friday of the month), the user should confirm the historical schedule so `enumerate_vesting_events()` can replicate it accurately.

4. **GitHub Actions DB commit:** Committing a binary SQLite file daily will grow git history. If this becomes a storage concern, use `git lfs` for the DB file. For now, the file starts at ~0 bytes and grows slowly (daily price rows for 23 tickers ≈ ~50KB/year).
