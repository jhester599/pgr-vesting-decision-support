# PGR Vesting Decision Support · v2.7

A quantitative decision-support engine for systematically unwinding a concentrated
Progressive Corporation (PGR) RSU position held in a taxable brokerage account.

The engine combines Walk-Forward Optimized machine learning with after-tax lot
selection to produce a structured sell/hold recommendation for each vesting event.

---

## Overview

Two RSU programs vest on fixed annual schedules:

| Program | Vest Date | Holding Period at Vest |
|---------|-----------|------------------------|
| Performance RSU | July 17, 2026 | ~4 years (LTCG eligible) |
| Time RSU | January 19, 2027 | ~4 years (LTCG eligible) |

For each event the engine outputs:

- Recommended sale percentage (0–100%), driven by the WFO model signal
- Estimated gross proceeds, tax liability, and net after-tax proceeds
- Tax lot selection strategy (loss harvesting → LTCG → STCG priority)
- ETF reallocation targets for the proceeds
- Per-benchmark signal grid (OUTPERFORM / UNDERPERFORM / NEUTRAL vs. all 20 ETFs)

---

## Versions

### v1 — PGR Absolute Return Prediction (complete)

Models PGR price performance in isolation.  Inputs are PGR-only features
(momentum, volatility, combined ratio, PIF growth, Gainshare estimate).
Predicts PGR 6-month DRIP total return; recommendation is sell/hold/partial
based on model IC and predicted magnitude.

### v2 — Relative Return Prediction Engine (v2.7 — complete)

Answers the question: *"At this vesting event, is PGR statistically likely to
outperform each of the 20 alternative diversified funds over the next 6 or 12
months?"*  One separate LassoCV/RidgeCV WFO model per benchmark ETF.

Key improvements over v1:

- **Relative return target**: PGR DRIP total return minus ETF DRIP total return
- **20 benchmark ETFs**: Vanguard-preferred, sector coverage, all pre-2014 history
- **6M and 12M horizons**: separate models for each prediction window
- **Correct embargo**: 6M target → 6-month embargo; 12M target → 12-month embargo
  (v1 used `gap=1` — a bug that allowed autocorrelation leakage)
- **SQLite accumulation pipeline**: price + dividend + fundamentals history
  auto-updated weekly via GitHub Actions; committed back to the repository
- **Multi-benchmark WFO**: 20 parallel models with per-event temporal slicing;
  `predict_current()` genuinely refits on the most recent training window
- **Historical vesting backtest**: all mid-January and mid-July events since 2014
  backtested end-to-end; hit rate and IC reported per benchmark
- **Reporting & visualization**: pivot tables, CSV export, heatmap, hit-rate bars,
  predicted-vs-realized scatter, live multi-benchmark signal chart

---

## Architecture

### v1 modules (stable)

```
pgr-vesting-decision-support/
├── config.py                         # API keys, WFO params, tax rates, ETF universe
├── requirements.txt
│
├── src/
│   ├── ingestion/
│   │   ├── price_loader.py           # AV TIME_SERIES_WEEKLY → weekly OHLCV (PGR)
│   │   ├── dividend_loader.py        # AV DIVIDENDS → ex-div dates + amounts (PGR)
│   │   ├── split_loader.py           # Hardcoded from config (3 known PGR splits)
│   │   ├── pgr_monthly_loader.py     # EDGAR cache CSV → combined ratio, PIF
│   │   ├── fundamentals_loader.py    # FMP quarterly key metrics (optional)
│   │   ├── technical_loader.py       # AV SMA/RSI/MACD/BBANDS (optional)
│   │   ├── fmp_client.py             # Cache-first FMP REST wrapper
│   │   └── av_client.py              # Cache-first Alpha Vantage wrapper
│   │
│   ├── processing/
│   │   ├── corporate_actions.py      # Forward-applies splits to share count
│   │   ├── total_return.py           # DRIP total return reconstruction (PGR)
│   │   └── feature_engineering.py   # Rolling monthly feature matrix + target
│   │
│   ├── models/
│   │   ├── wfo_engine.py             # Walk-Forward Optimization orchestrator
│   │   └── regularized_models.py    # LassoCV / RidgeCV inside sklearn Pipeline
│   │
│   ├── tax/
│   │   └── capital_gains.py          # LTCG/STCG lot-level optimizer
│   │
│   ├── portfolio/
│   │   ├── drift_analyzer.py         # Sector deviation vs. MSCI ACWI equilibrium
│   │   └── rebalancer.py             # Unified recommendation output
│   │
│   └── visualization/
│       └── plots.py                  # WFO curve, feature importance, drift chart
│
├── data/
│   ├── raw/                          # Cached API JSON (gitignored)
│   └── processed/
│       └── pgr_edgar_cache.csv       # EDGAR monthly fundamentals (committed)
│
└── tests/                            # 69 pytest tests (v1), all passing
```

### v2 modules (v2.7 — all phases complete)

```
├── src/
│   ├── database/
│   │   ├── schema.sql                # 8-table SQLite schema (prices, dividends,
│   │   │                             #   splits, fundamentals, EDGAR, relative
│   │   │                             #   returns, API log, ingestion metadata)
│   │   └── db_client.py              # Connection, schema init, upsert/get helpers,
│   │                                 #   API budget enforcement
│   │
│   ├── ingestion/
│   │   ├── multi_ticker_loader.py    # AV TIME_SERIES_WEEKLY → DB (23 tickers)
│   │   ├── multi_dividend_loader.py  # AV DIVIDENDS → DB (23 tickers)
│   │   └── fetch_scheduler.py        # get_all_price_tickers() / get_all_dividend_tickers()
│   │
│   ├── processing/
│   │   ├── multi_total_return.py     # DRIP total return for all 20 ETFs;
│   │   │                             #   build_relative_return_targets() upserts to DB
│   │   └── feature_engineering.py   # + build_feature_matrix_from_db()
│   │                                 #   + get_X_y_relative() (inner-join alignment)
│   │
│   ├── models/
│   │   ├── wfo_engine.py             # v2: embargo=target_horizon; WFOResult extended
│   │   │                             #   with benchmark/target_horizon/model_type;
│   │   │                             #   predict_current() genuinely refits + predicts
│   │   └── multi_benchmark_wfo.py    # run_all_benchmarks() — 20 parallel WFO models
│   │                                 #   get_current_signals() — live signal grid
│   │
│   ├── portfolio/
│   │   └── rebalancer.py             # + benchmark_signals / target_horizon fields;
│   │                                 #   generate_recommendation() accepts
│   │                                 #   multi_benchmark_results kwarg
│   │
│   ├── backtest/
│   │   ├── vesting_events.py         # enumerate_vesting_events() — Jan/Jul since 2014
│   │   │                             #   nearest-business-day snap, forward windows
│   │   └── backtest_engine.py        # run_historical_backtest() — strict temporal
│   │                                 #   slicing per event; run_full_backtest()
│   │
│   ├── reporting/
│   │   └── backtest_report.py        # generate_backtest_table() / prediction_table()
│   │                                 #   / correct_direction_table();
│   │                                 #   print_backtest_summary(); export_backtest_to_csv()
│   │
│   └── visualization/
│       └── plots.py                  # + plot_backtest_heatmap()
│                                     # + plot_hit_rate_by_benchmark()
│                                     # + plot_predicted_vs_realized_scatter()
│                                     # + plot_multi_benchmark_signals()
│
├── scripts/
│   ├── migrate_v1_to_v2.py          # One-time: v1 JSON/Parquet → SQLite
│   ├── initial_fetch.py             # One-time full history bootstrap
│   │                                #   Day 1: --prices  (23 AV calls)
│   │                                #   Day 2: --dividends (23 AV calls)
│   └── weekly_fetch.py              # Weekly cron: 23 prices + PGR dividends
│                                    #   + PGR FMP fundamentals (24 AV, 2 FMP)
│
├── data/
│   └── pgr_financials.db            # SQLite accumulation DB (committed to git;
│                                    #   updated weekly by GitHub Actions)
│
├── pytest.ini                        # testpaths=tests; integration / unit markers
│
├── .github/workflows/
│   └── weekly_data_fetch.yml        # Cron: Friday 10 PM UTC; commits DB update
│
└── tests/
    ├── test_db_client.py             # 37 tests: schema, upserts, budget enforcement
    ├── test_multi_ticker_loader.py  # 34 tests: parsers, loaders, scheduler, proxy fill
    ├── test_multi_total_return.py   # 36 tests: DRIP computation, relative targets,
    │                                 #   DatetimeIndex guard, get_X_y_relative
    ├── test_wfo_engine.py            # 24 tests: temporal integrity, embargo (6M/12M),
    │                                 #   v2 metadata, predict_current refit
    ├── test_multi_benchmark_wfo.py  # 16 tests: run_all_benchmarks, get_current_signals,
    │                                 #   skip-on-no-overlap, signal classification
    ├── test_backtest_engine.py       # 32 tests: date enumeration, business-day snap,
    │                                 #   forward windows, signal/sell-pct logic
    ├── test_reporting.py             # 27 tests: pivot tables, print summary, CSV export,
    │                                 #   all 4 new plot functions
    └── test_integration.py           # 11 tests: end-to-end smoke test, synthetic DB,
                                      #   full pipeline from schema to signal generation
```

---

## Data Sources

| Source | Data | Tier |
|--------|------|------|
| Alpha Vantage `TIME_SERIES_WEEKLY` | Weekly OHLCV — PGR + 20 ETF benchmarks (~25 years) | Free |
| Alpha Vantage `DIVIDENDS` | Ex-dividend history — PGR + 20 ETF benchmarks | Free (25 req/day) |
| FMP `/v3/key-metrics` + `/v3/income-statement` | PGR quarterly PE, PB, ROE, EPS, revenue | Free |
| EDGAR cache CSV | 256 months of combined ratio, PIF, EPS | User-provided |
| `config.PGR_KNOWN_SPLITS` | 3 historical splits (1992, 2002, 2006) | Hardcoded |

### v2 ETF Benchmark Universe (20 ETFs)

Vanguard is the preferred provider.  Sector ETFs are included so the engine
can evaluate PGR against meaningfully different industry exposures (e.g.
technology, energy) rather than only broad-market alternatives.

| Category | Tickers | Notes |
|----------|---------|-------|
| US Broad Market | VTI, VOO | Total market + S&P 500 |
| US Sectors | VGT, VHT, VFH, VIS, VDE, VPU | Tech, Health, Financials, Industrials, Energy, Utilities |
| International | VXUS, VEA, VWO | Total intl, Developed ex-US, Emerging |
| Dividend | VIG, SCHD | Dividend growth (Vanguard) vs high yield (Schwab) |
| Fixed Income | BND, BNDX, VCIT, VMBS | Total bond, Intl bond, Corporate, Mortgage-backed |
| Real Assets | VNQ, GLD, DBC | REIT, Gold, Commodities |

All 20 ETFs have history predating January 2014 (the earliest backtested
vesting event).  No proxy backfill is required.

### API Budget

| Script | AV calls | FMP calls | Notes |
|--------|----------|-----------|-------|
| `initial_fetch.py --prices` | 23 | 0 | Day 1 of initial bootstrap |
| `initial_fetch.py --dividends` | 23 | 0 | Day 2 of initial bootstrap |
| `weekly_fetch.py` | 24 | 2 | Weekly cron (limit: 25 AV, 250 FMP) |

---

## Methodology

### DRIP Total Return Reconstruction

Unadjusted prices are used throughout. Share count is tracked explicitly:

```
Split:    shares *= split_ratio           (forward-applied on split date)
Dividend: shares += (shares * div) / price_on_exdiv_date   (fractional DRIP)
Value:    V[t] = shares[t] * close[t]
```

This produces an accurate total-return series that correctly models fractional
share accumulation — required for the forward return training labels.

### Feature Matrix

Features are computed at month-end frequency, expanding the effective dataset
from ~30 semi-annual observations to 300+ monthly samples.

| Feature | Description |
|---------|-------------|
| `mom_3m` | 3-month price momentum (63 trading days) |
| `mom_6m` | 6-month price momentum (126 trading days) |
| `mom_12m` | 12-month price momentum (252 trading days) |
| `vol_21d` | 21-day realized volatility (annualized) |
| `vol_63d` | 63-day realized volatility (annualized) |
| `combined_ratio_ttm` | Trailing 12M combined ratio (from EDGAR cache) |
| `pif_growth_yoy` | Policies in Force YoY growth |
| `gainshare_est` | Estimated Gainshare multiplier (0–2 scale) |
| `target_6m_return` | Forward 6-month DRIP total return **(v1 label)** |
| `target_Nm_relative` | Forward N-month PGR return minus ETF return **(v2 label)** |

Gainshare features are retained only if ≥ 60 non-NaN observations exist.

### Walk-Forward Optimization (WFO)

**No K-Fold cross-validation.** All validation uses strict temporal splits.

```python
TimeSeriesSplit(
    n_splits       = (total_months - train_window - target_horizon) // test_window,
    max_train_size = 60,             # 5-year rolling window
    test_size      = 6,              # 6-month out-of-sample test
    gap            = target_horizon, # embargo = target horizon (6 or 12 months)
)
```

**v1 embargo bug:** `gap=1` was used, which is insufficient for a 6-month
forward return target — consecutive monthly observations share 5 months of
overlapping return window (autocorrelation leakage).  v2 sets `gap` equal to
the target horizon: 6 for the 6M model, 12 for the 12M model.

**20 separate models (v2):** One LassoCV/RidgeCV model per ETF benchmark.
PGR-vs-BND has a fundamentally different statistical character than PGR-vs-VGT;
a multi-output model would impose the same regularization on both.

**`predict_current()` refit (v2):** v1 returned `last_fold.y_hat.mean()` — a
constant regardless of the current observation.  v2 refits on the most recent
`train_window_months` rows and calls `model.predict(X_current)`.

### Signal Classification (v2)

```
IC < 0.05  OR  |predicted return| < 1%  →  NEUTRAL
IC ≥ 0.05  AND  predicted return > +1%  →  OUTPERFORM  (favor holding PGR)
IC ≥ 0.05  AND  predicted return < −1%  →  UNDERPERFORM (favor selling PGR)
```

### Recommendation Logic

```
IC < 0.05                           → 50% sale (model below confidence threshold)
IC ≥ 0.05, predicted return > +15%  → 25% sale (hold majority)
IC ≥ 0.05, predicted return > +5%   → 50% sale (balanced)
IC ≥ 0.05, predicted return ≤ +5%   → 100% sale (full diversification)
sell_pct_override provided          → use override directly
```

### Tax Lot Selection

Lots are selected to minimize tax liability on the sold tranche:

1. **Embedded losses first** — harvest the loss, reduce taxable gain elsewhere
2. **LTCG-eligible lots** (held > 365 days) — highest basis first to minimize gain
3. **STCG lots last** — most expensive tax treatment, sold only when necessary

```python
LTCG_RATE = 0.20   # Federal maximum (configurable via .env)
STCG_RATE = 0.37   # Federal maximum ordinary income rate
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root (never committed):

```
FMP_API_KEY=your_fmp_key
AV_API_KEY=your_alphavantage_key
LTCG_RATE=0.20
STCG_RATE=0.37
```

Place your position data at `data/processed/position_lots.csv` (gitignored):

```csv
vest_date,rsu_type,shares,cost_basis_per_share
2026-07-17,performance,500,116.08
2027-01-19,time,500,133.65
```

### v2 Database Bootstrap

The SQLite database is committed to the repository and updated weekly by GitHub
Actions.  To populate it locally for the first time:

```bash
# Day 1: fetch full price history for all 23 tickers (23 AV calls)
python scripts/initial_fetch.py --prices

# Day 2: fetch full dividend history for all 23 tickers (23 AV calls)
python scripts/initial_fetch.py --dividends

# Optional: migrate existing v1 JSON/Parquet cache to the DB
python scripts/migrate_v1_to_v2.py
```

To verify without consuming API budget:

```bash
python scripts/weekly_fetch.py --dry-run
```

### GitHub Actions (v2)

The weekly cron (`weekly_data_fetch.yml`) runs every Friday at 10 PM UTC.
It requires two repository secrets:

| Secret | Description |
|--------|-------------|
| `AV_API_KEY` | Alpha Vantage free-tier key |
| `FMP_API_KEY` | Financial Modeling Prep free-tier key |

The workflow commits the updated `data/pgr_financials.db` back to the branch
with `[skip ci]` to avoid triggering a recursive run.

---

## Running the Engine

### v2 (recommended)

```python
import sqlite3
import config
from src.database.db_client import get_connection, initialize_schema
from src.processing.feature_engineering import build_feature_matrix_from_db
from src.processing.multi_total_return import build_relative_return_targets, load_relative_return_matrix
from src.models.multi_benchmark_wfo import run_all_benchmarks, get_current_signals
from src.backtest.backtest_engine import run_full_backtest
from src.reporting.backtest_report import print_backtest_summary, export_backtest_to_csv
from src.visualization.plots import (
    plot_backtest_heatmap, plot_hit_rate_by_benchmark,
    plot_multi_benchmark_signals
)

conn = get_connection()
initialize_schema(conn)

# Build feature matrix and relative return targets
df = build_feature_matrix_from_db(conn)
feature_cols = [c for c in df.columns if c != "target_6m_return"]
X = df[feature_cols]

# Pre-compute relative returns for all benchmarks (upserts to DB)
build_relative_return_targets(conn, forward_months=6)
build_relative_return_targets(conn, forward_months=12)

# Build relative return matrix
import pandas as pd
rel = pd.DataFrame({
    etf: load_relative_return_matrix(conn, etf, 6).rename(etf)
    for etf in config.ETF_BENCHMARK_UNIVERSE
})

# Train 20 WFO models
wfo_results = run_all_benchmarks(X, rel, model_type="lasso", target_horizon_months=6)

# Generate live signals for the current observation
signals = get_current_signals(X, rel, wfo_results, X.iloc[[-1]])
print(signals[["predicted_relative_return", "signal", "ic"]])

# Run historical backtest
backtest_df = run_full_backtest(conn)
print_backtest_summary(list_of_results)
export_backtest_to_csv(list_of_results, "data/backtest_results.csv")

# Save plots
plot_backtest_heatmap(list_of_results, horizon=6)
plot_hit_rate_by_benchmark(list_of_results)
plot_multi_benchmark_signals(signals)
```

### v1 (legacy)

```python
from src.ingestion import price_loader, dividend_loader, split_loader, pgr_monthly_loader
from src.processing.feature_engineering import build_feature_matrix, get_X_y
from src.models.wfo_engine import run_wfo
from src.tax.capital_gains import load_position_lots
from src.portfolio.drift_analyzer import PortfolioState
from src.portfolio.rebalancer import generate_recommendation, print_recommendation

prices      = price_loader.load()
divs        = dividend_loader.load()
splits      = split_loader.load()
pgr_monthly = pgr_monthly_loader.load()

fm = build_feature_matrix(prices, divs, splits, pgr_monthly=pgr_monthly)
X, y = get_X_y(fm, drop_na_target=True)
wfo = run_wfo(X, y, model_type="lasso")

current_price = 206.00
lots = load_position_lots("data/processed/position_lots.csv")
portfolio = PortfolioState(pgr_value=1000 * current_price, etf_holdings={})

from datetime import date
for vest_date, rsu_type in [(date(2026, 7, 17), "performance"), (date(2027, 1, 19), "time")]:
    rec = generate_recommendation(vest_date, rsu_type, current_price, lots, wfo, portfolio)
    print_recommendation(rec)
```

---

## Tests

```bash
pytest                     # all tests
pytest -m integration      # integration smoke tests only
pytest -m "not integration" # fast unit tests only
```

**271 tests across 14 modules, all passing** (as of v2.7):

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_corporate_actions.py` | 17 | Split application, known split validation, cumulative multiplier |
| `test_total_return.py` | 13 | DRIP share accumulation, portfolio value, no negative prices |
| `test_feature_engineering.py` | 11 | No-leakage guarantee, target NaN in final 6M, Gainshare threshold |
| `test_capital_gains.py` | 13 | LTCG/STCG rate selection, lot priority, oversell validation |
| `test_db_client.py` | 37 | Schema idempotency, upsert/replace semantics, budget enforcement, proxy_fill |
| `test_multi_ticker_loader.py` | 34 | Price/dividend parsers, MultiTickerLoader, MultiDividendLoader, scheduler |
| `test_multi_total_return.py` | 36 | DRIP computation, relative targets, DatetimeIndex guard, get_X_y_relative |
| `test_wfo_engine.py` | 24 | Temporal integrity, embargo (6M & 12M), v2 metadata, predict_current refit |
| `test_multi_benchmark_wfo.py` | 16 | run_all_benchmarks, get_current_signals, skip-on-no-overlap, signal classification |
| `test_backtest_engine.py` | 32 | Date enumeration, business-day snap, forward windows, signal/sell-pct logic |
| `test_reporting.py` | 27 | Pivot tables, print summary, CSV export, all 4 new plot functions |
| `test_integration.py` | 11 | End-to-end smoke test: synthetic DB → schema → features → WFO → signals |

Critical invariants enforced by tests:

- `max(train_idx) < min(test_idx)` for every WFO fold
- Embargo gap ≥ `target_horizon × 28` days (168 days for 6M, 336 days for 12M)
- `StandardScaler` is a named step inside `Pipeline` — never fit on full dataset
- `target_6m_return` is NaN for the final 6 months (no look-ahead leakage)
- `predict_current()` produces different outputs for different `X_current` inputs
  (verifies it is a live model, not the v1 constant placeholder)
- LTCG boundary: exactly 365 days does NOT qualify (must be > 365)
- API budget enforcement: 26th AV call raises `RuntimeError` (limit: 25/day)
- Upsert is idempotent: duplicate inserts replace, never duplicate rows
- `proxy_fill=1` rows are flagged and round-trip correctly through the DB
- All 20 vesting event dates (2014–2023) fall on weekdays

---

## Known Limitations

- **Weekly price resolution**: Alpha Vantage free tier provides weekly (not daily)
  OHLCV. Sufficient for monthly feature engineering and DRIP calculations; ex-dividend
  dates are matched to the nearest weekly close.
- **v1 IC below threshold**: With ~25 years of monthly data and a rolling 5-year
  training window, the v1 Lasso model does not achieve IC ≥ 0.05 on current data.
  The engine falls back to the 50% diversification-sale rule. v2's relative return
  framing and correct embargo are expected to produce more stable IC estimates.
- **Both lots are STCG at vest**: Shares vest on grant date with zero holding
  period, so both tranches are classified as short-term at the moment of vest-day
  sale. LTCG status accrues if shares are held at least 366 days post-vest.
- **BNDX history**: Vanguard Total International Bond (BNDX) launched June 2013,
  roughly 7 months before the earliest backtested vesting event (Jan 2014).  The
  BNDX relative-return model will have slightly fewer early training observations
  than the other 19 benchmarks; no proxy backfill is applied.
- **`scipy` optional dependency**: `plot_predicted_vs_realized_scatter()` uses
  `scipy.stats.spearmanr` to annotate the IC on scatter plots.  The function
  degrades gracefully if fewer than 3 events are available.

---

## Dependencies

```
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
xgboost>=2.0.0
requests>=2.31.0
pytest>=8.0.0
python-dotenv>=1.0.0
pyarrow>=15.0.0
scipy>=1.12.0
```

---

## License

Private — personal financial decision support tool.
