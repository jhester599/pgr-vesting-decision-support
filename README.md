# PGR Vesting Decision Support · v4.0

A quantitative decision-support engine for systematically unwinding a concentrated
Progressive Corporation (PGR) RSU position held in a taxable brokerage account.

The engine combines Walk-Forward Optimized machine learning, Black-Litterman portfolio
construction, and proactive tax-loss harvesting to produce a structured sell/hold
recommendation for each vesting event.

---

## Overview

Two RSU programs vest on fixed annual schedules:

| Program | Vest Date | Holding Period at Vest |
|---------|-----------|------------------------|
| Performance RSU | July 17, 2026 | ~4 years (LTCG eligible) |
| Time RSU | January 19, 2027 | ~4 years (LTCG eligible) |

For each event the engine outputs:

- Recommended sale percentage (0–100%), driven by fractional Kelly sizing
- Estimated gross proceeds, tax liability, and net after-tax proceeds
- Tax lot selection strategy (TLH candidates → LTCG → STCG priority)
- Black-Litterman optimal ETF reallocation targets (Ledoit-Wolf shrunk covariance)
- Per-benchmark signal grid (OUTPERFORM / UNDERPERFORM / NEUTRAL vs. all 20 ETFs)
- CPCV validation distribution (15 train-test paths, overfitting detection)
- Monthly automated decision report (committed to `results/` on the 20th)

---

## Versions

### v1 — PGR Absolute Return Prediction (complete)

Models PGR price performance in isolation. Inputs are PGR-only features
(momentum, volatility, combined ratio, PIF growth, Gainshare estimate).
Predicts PGR 6-month DRIP total return; recommendation is sell/hold/partial
based on model IC and predicted magnitude.

### v2 — Relative Return Prediction Engine (v2.7 — complete)

Answers: *"Is PGR statistically likely to outperform each of 20 diversified ETFs
over the next 6 or 12 months?"* One separate LassoCV/RidgeCV WFO model per
benchmark ETF, with correct horizon-matched embargo.

### v3.0 — Macro Intelligence + Monthly Decision Engine (complete)

FRED macro features (yield curve, credit spreads, NFCI), ElasticNetCV model,
purge buffer fix (gap = horizon + buffer), Campbell-Thompson OOS R², BHY multiple
testing correction, Newey-West HAC IC, 120+ monthly evaluation points,
automated `scripts/monthly_decision.py` cron on the 20th of each month.

### v3.1 — Ensemble Models + Kelly Sizing + Regime Diagnostics (complete)

Equal-weight BayesianRidge + ElasticNet + Ridge ensemble with uncertainty
quantification, fractional Kelly position sizing (0.25×, 30% cap), PGR-specific
FRED features (motor vehicle insurance CPI, vehicle miles traveled), and
4-quadrant bull/bear × low/high-vol regime breakdown reports.

### v4.0 — Production Validation + Portfolio Optimization + TLH (current)

Combinatorial Purged Cross-Validation (CPCV) for overfitting detection,
Black-Litterman portfolio construction with view confidence scaled by CV RMSE²,
tax-loss harvesting with wash-sale replacement ETF suggestions, fractional
differentiation for stationarity-preserving feature transforms, and
per-benchmark signal weighting by historical IC × hit rate.

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

### v2 modules (v2.7 — complete)

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
│   │   ├── wfo_engine.py             # embargo = target_horizon; WFOResult extended
│   │   │                             #   with benchmark/target_horizon/model_type;
│   │   │                             #   predict_current() genuinely refits + predicts
│   │   └── multi_benchmark_wfo.py    # run_all_benchmarks() — 20 parallel WFO models
│   │                                 #   get_current_signals() — live signal grid
│   │
│   ├── backtest/
│   │   ├── vesting_events.py         # enumerate_vesting_events() — Jan/Jul since 2014
│   │   └── backtest_engine.py        # run_full_backtest() — strict temporal slicing
│   │
│   └── reporting/
│       └── backtest_report.py        # generate_backtest_table(), print_backtest_summary()
│
├── scripts/
│   ├── initial_fetch.py             # One-time full history bootstrap
│   └── weekly_fetch.py              # Weekly cron: prices + dividends + FMP fundamentals
│
├── data/
│   └── pgr_financials.db            # SQLite accumulation DB (committed; auto-updated)
│
└── .github/workflows/
    └── weekly_data_fetch.yml        # Cron: Friday 10 PM UTC; commits DB update
```

### v3.0 modules (complete)

```
├── src/
│   ├── database/
│   │   └── schema.sql               # + fred_macro_monthly table (series_id, month_end, value)
│   │   └── db_client.py             # + upsert_fred_macro(), get_fred_macro()
│   │
│   ├── ingestion/
│   │   └── fred_loader.py           # NEW: FRED public REST API client
│   │                                #   fetch_fred_series(), fetch_all_fred_macro()
│   │                                #   Monthly resampling, forward-fill, NaN handling
│   │
│   ├── processing/
│   │   └── feature_engineering.py  # + 6 FRED macro features (yield_slope,
│   │                                #   yield_curvature, real_rate_10y,
│   │                                #   credit_spread_ig, credit_spread_hy, nfci)
│   │
│   ├── models/
│   │   ├── regularized_models.py    # + build_elasticnet_pipeline() (l1_ratio grid)
│   │   └── wfo_engine.py            # + purge_buffer param; gap = horizon + buffer
│   │                                #   (6M→8, 12M→15); default ElasticNet
│   │
│   ├── backtest/
│   │   ├── vesting_events.py        # + enumerate_monthly_evaluation_dates()
│   │   │                            #   120+ month-end evaluation points (2014–present)
│   │   └── backtest_engine.py       # + run_monthly_stability_backtest()
│   │
│   └── reporting/
│       └── backtest_report.py       # + compute_oos_r_squared() (Campbell-Thompson)
│                                    #   + apply_bhy_correction() (FDR control)
│                                    #   + compute_newey_west_ic() (HAC std errors)
│
├── scripts/
│   ├── weekly_fetch.py              # + FRED macro fetch step (--skip-fred flag)
│   └── monthly_decision.py          # NEW: automated monthly sell/hold report
│
├── results/
│   └── monthly_decisions/
│       ├── decision_log.md          # Persistent append-only decision log
│       └── YYYY-MM/
│           ├── recommendation.md    # Human-readable report
│           ├── signals.csv          # Per-benchmark signal details
│           └── backtest_summary.csv # Monthly stability stats
│
└── .github/workflows/
    └── monthly_decision.yml         # NEW: Cron 20th of each month; commits results
```

### v3.1 modules (complete)

```
├── src/
│   ├── models/
│   │   ├── regularized_models.py    # + UncertaintyPipeline (Pipeline subclass)
│   │   │                            #   + build_bayesian_ridge_pipeline()
│   │   │                            #   + predict_with_std() → (y_pred, y_std)
│   │   └── multi_benchmark_wfo.py   # + EnsembleWFOResult dataclass
│   │                                #   + run_ensemble_benchmarks() — 3-model ensemble
│   │                                #   + get_ensemble_signals() — live uncertainty
│   │
│   ├── processing/
│   │   └── feature_engineering.py  # + vix (VIXCLS)
│   │                                #   + insurance_cpi_mom3m (CUSR0000SETC01 3M mom)
│   │                                #   + vmt_yoy (TRFVOLUSM227NFWA YoY)
│   │
│   ├── portfolio/
│   │   └── rebalancer.py            # + _compute_sell_pct_kelly()
│   │                                #   + VestingRecommendation.prediction_std
│   │                                #   + VestingRecommendation.kelly_fraction_used
│   │
│   └── reporting/
│       └── backtest_report.py       # + generate_rolling_ic_series() (24M window)
│                                    #   + generate_regime_breakdown() (4-quadrant)
```

### v4.0 modules (current)

```
├── src/
│   ├── models/
│   │   └── wfo_engine.py            # + run_cpcv() via skfolio CombinatorialPurgedCV
│   │                                #   + CPCVResult (n_splits, n_paths, path_ics,
│   │                                #     mean_ic, ic_std)
│   │
│   ├── portfolio/
│   │   ├── black_litterman.py       # NEW: build_bl_weights() via PyPortfolioOpt
│   │   │                            #   _ledoit_wolf_covariance() (sklearn LedoitWolf)
│   │   │                            #   compute_equilibrium_returns() (π = δΣw)
│   │   │                            #   Views: predicted excess return per ETF
│   │   │                            #   Ω_ii = CV RMSE² × confidence scalar
│   │   └── rebalancer.py            # + compute_benchmark_weights()
│   │                                #   weight = IC × hit_rate, normalized; IC≤0 → 0
│   │
│   ├── tax/
│   │   └── capital_gains.py         # + identify_tlh_candidates() (−10% threshold)
│   │                                #   + compute_after_tax_expected_return()
│   │                                #   + suggest_tlh_replacement() (TLH_REPLACEMENT_MAP)
│   │                                #   + wash_sale_clear_date() (+31 days)
│   │
│   └── processing/
│       └── feature_engineering.py  # + apply_fracdiff() — FFD stationarity transform
│                                    #   + _fracdiff_weights() — numpy/scipy FFD weights
│                                    #   Finds min d* preserving ≥90% memory correlation
│                                    #   while achieving ADF stationarity
```

---

## Data Sources

| Source | Data | Tier |
|--------|------|------|
| Alpha Vantage `TIME_SERIES_WEEKLY` | Weekly OHLCV — PGR + 20 ETF benchmarks (~25 years) | Free |
| Alpha Vantage `DIVIDENDS` | Ex-dividend history — PGR + 20 ETF benchmarks | Free (25 req/day) |
| FMP `/v3/key-metrics` + `/v3/income-statement` | PGR quarterly PE, PB, ROE, EPS, revenue | Free |
| FRED public REST API | 8 macro series + 2 PGR-specific series (no budget impact) | Free |
| EDGAR cache CSV | 256 months of combined ratio, PIF, EPS | User-provided |
| `config.PGR_KNOWN_SPLITS` | 3 historical splits (1992, 2002, 2006) | Hardcoded |

### FRED Series

| Category | Series | Feature |
|----------|--------|---------|
| Yield curve | T10Y2Y | `yield_slope` |
| Yield curve | GS2, GS5, GS10 | `yield_curvature`, `real_rate_10y` |
| Inflation | T10YIE | `real_rate_10y` |
| Credit spreads | BAA10Y, BAMLH0A0HYM2 | `credit_spread_ig`, `credit_spread_hy` |
| Financial conditions | NFCI | `nfci` |
| Volatility | VIXCLS | `vix` |
| PGR-specific | CUSR0000SETC01 | `insurance_cpi_mom3m` (3M momentum) |
| PGR-specific | TRFVOLUSM227NFWA | `vmt_yoy` (YoY change) |

### v2 ETF Benchmark Universe (20 ETFs)

| Category | Tickers | Notes |
|----------|---------|-------|
| US Broad Market | VTI, VOO | Total market + S&P 500 |
| US Sectors | VGT, VHT, VFH, VIS, VDE, VPU | Tech, Health, Financials, Industrials, Energy, Utilities |
| International | VXUS, VEA, VWO | Total intl, Developed ex-US, Emerging |
| Dividend | VIG, SCHD | Dividend growth (Vanguard) vs high yield (Schwab) |
| Fixed Income | BND, BNDX, VCIT, VMBS | Total bond, Intl bond, Corporate, Mortgage-backed |
| Real Assets | VNQ, GLD, DBC | REIT, Gold, Commodities |

### API Budget

| Script | AV calls | FMP calls | FRED calls | Notes |
|--------|----------|-----------|------------|-------|
| `initial_fetch.py --prices` | 23 | 0 | 0 | Day 1 bootstrap |
| `initial_fetch.py --dividends` | 23 | 0 | 0 | Day 2 bootstrap |
| `weekly_fetch.py` | 24 | 2 | ~10 | Weekly cron (FRED is free, unlimited) |

---

## Methodology

### DRIP Total Return Reconstruction

Unadjusted prices are used throughout. Share count is tracked explicitly:

```
Split:    shares *= split_ratio           (forward-applied on split date)
Dividend: shares += (shares * div) / price_on_exdiv_date   (fractional DRIP)
Value:    V[t] = shares[t] * close[t]
```

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
| `yield_slope` | T10Y2Y (10Y minus 2Y Treasury spread) |
| `yield_curvature` | 2×GS5 − GS2 − GS10 |
| `real_rate_10y` | GS10 − T10YIE |
| `credit_spread_ig` | BAA10Y (investment-grade credit spread) |
| `credit_spread_hy` | BAMLH0A0HYM2 (high-yield credit spread) |
| `nfci` | Chicago Fed National Financial Conditions Index |
| `vix` | CBOE VIX (VIXCLS from FRED) |
| `insurance_cpi_mom3m` | 3-month momentum of motor vehicle insurance CPI |
| `vmt_yoy` | Vehicle miles traveled YoY change |

### Walk-Forward Optimization (WFO)

**No K-Fold cross-validation.** All validation uses strict temporal splits.

```python
TimeSeriesSplit(
    n_splits       = (total_months - train_window - target_horizon) // test_window,
    max_train_size = 60,                          # 5-year rolling window
    test_size      = 6,                           # 6-month out-of-sample test
    gap            = target_horizon + purge_buffer, # 6M → gap=8; 12M → gap=15
)
```

The `purge_buffer` (default 2M for 6M targets, 3M for 12M) prevents autocorrelation
leakage from overlapping return windows in consecutive training observations.

**Default model (v3.0+):** ElasticNetCV with l1_ratio grid `[0.1, 0.5, 0.9, 0.95, 1.0]`.
**Ensemble (v3.1+):** Equal-weight combination of ElasticNet, Ridge, and BayesianRidge.

### Combinatorial Purged Cross-Validation (CPCV)

CPCV is used as a validation diagnostic alongside WFO — it is not the production
training path.

```python
CombinatorialPurgedCV(n_folds=6, n_test_folds=2)
# C(6,2) = 15 train-test splits → 5 independent backtest paths
# Each path covers the full dataset length; detects combinatorial overfitting
```

`CPCVResult.path_ics` contains per-path IC sequences. High variance across paths
signals overfitting even when mean IC appears strong.

### Black-Litterman Portfolio Construction

```
π = δ × Σ × w_mkt          # equilibrium returns (reverse optimization)
Q = ensemble predicted excess returns per ETF
Ω_ii = CV_RMSE²_i × scalar # diagonal uncertainty matrix
→ PyPortfolioOpt BlackLittermanModel → EfficientFrontier.max_sharpe
→ weights clipped at KELLY_MAX_POSITION = 0.30
```

Ledoit-Wolf shrinkage is applied to the sample covariance matrix to ensure
positive semi-definiteness with limited history.

### Fractional Kelly Sizing

```python
f* = KELLY_FRACTION × predicted_excess_return / prediction_variance
position_fraction = min(max(f*, 0.0), KELLY_MAX_POSITION)
sell_pct = 1.0 - position_fraction
# KELLY_FRACTION = 0.25 (quarter-Kelly for personal portfolio risk tolerance)
# KELLY_MAX_POSITION = 0.30 (maximum single position)
```

### Fractional Differentiation

The Fixed-Width Window (FFD) method finds the minimum differencing order d*
that achieves ADF stationarity while retaining ≥ 90% Pearson correlation with
the original (undifferenced) series — preserving maximum memory.

```
w_k = -(d - k + 1) / k × w_{k-1}  (FFD weights, truncated at threshold 1e-5)
Grid search: d ∈ [0.0, max_d] in 11 steps; ADF at α=0.05; corr ≥ 0.90
```

### Tax-Loss Harvesting

```python
# Harvest candidates: unrealized return < -10% (TLH_LOSS_THRESHOLD)
loss_pct = (current_price - cost_basis) / cost_basis
candidates = sorted by loss_pct ascending (largest loss first)

# After-tax expected return (creates asymmetry favoring sell in gain positions)
after_tax = predicted_return - max(0, unrealized_gain_fraction × tax_rate)

# Wash-sale compliance: replacement ETF after 31-day window
# e.g., VTI → ITOT; VOO → IVV; VGT → QQQ; BND → AGG
```

### Signal Classification

```
IC < 0.05  OR  |predicted return| < 1%  →  NEUTRAL
IC ≥ 0.05  AND  predicted return > +1%  →  OUTPERFORM  (favor holding PGR)
IC ≥ 0.05  AND  predicted return < −1%  →  UNDERPERFORM (favor selling PGR)
```

### Tax Lot Selection

Lots are selected to minimize tax liability on the sold tranche:

1. **TLH candidates first** — unrealized loss > 10%; harvest the loss
2. **LTCG-eligible lots** (held > 365 days) — highest basis first to minimize gain
3. **STCG lots last** — most expensive tax treatment, sold only when necessary

```python
LTCG_RATE = 0.20   # Federal maximum (configurable via .env)
STCG_RATE = 0.37   # Federal maximum ordinary income rate
```

### Statistical Rigor (v3.0+)

- **OOS R²** (Campbell-Thompson): `1 - MSE_model / MSE_naive`, where naive = expanding
  historical mean. Negative OOS R² means the model underperforms a simple mean forecast.
- **BHY correction**: Benjamini-Hochberg-Yekutieli FDR control across 20 benchmark
  p-values; controls false discovery rate under arbitrary correlation structure.
- **Newey-West IC**: HAC standard errors for the IC statistic to account for
  autocorrelation from overlapping 6M/12M return windows.

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
FRED_API_KEY=your_fred_key      # free at fred.stlouisfed.org/docs/api/api_key.html
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
Actions. To populate it locally for the first time:

```bash
# Day 1: fetch full price history for all 23 tickers (23 AV calls)
python scripts/initial_fetch.py --prices

# Day 2: fetch full dividend history for all 23 tickers (23 AV calls)
python scripts/initial_fetch.py --dividends
```

To verify without consuming API budget:

```bash
python scripts/weekly_fetch.py --dry-run
python scripts/weekly_fetch.py --dry-run --skip-fred
```

### GitHub Actions

The weekly cron (`weekly_data_fetch.yml`) runs every Friday at 10 PM UTC.
The monthly decision cron (`monthly_decision.yml`) runs on the 20th of each month.

Both workflows require repository secrets:

| Secret | Description |
|--------|-------------|
| `AV_API_KEY` | Alpha Vantage free-tier key |
| `FMP_API_KEY` | Financial Modeling Prep free-tier key |
| `FRED_API_KEY` | FRED public API key (free) |

---

## Running the Engine

### v4.0 (current — ensemble + BL + CPCV)

```python
import sqlite3
import config
from src.database.db_client import get_connection, initialize_schema
from src.models.multi_benchmark_wfo import run_ensemble_benchmarks, get_ensemble_signals
from src.backtest.backtest_engine import run_full_backtest, run_monthly_stability_backtest
from src.reporting.backtest_report import (
    print_backtest_summary, export_backtest_to_csv,
    compute_oos_r_squared, apply_bhy_correction, generate_regime_breakdown,
)
from src.portfolio.black_litterman import build_bl_weights
from src.portfolio.rebalancer import compute_benchmark_weights
from src.models.wfo_engine import run_cpcv

conn = get_connection()
initialize_schema(conn)

# Build 3-model ensemble for all 20 ETF benchmarks
ensemble_results = run_ensemble_benchmarks(conn, target_horizon_months=6)

# Get live signals with BayesianRidge uncertainty
signals = get_ensemble_signals(conn, ensemble_results)

# Black-Litterman optimal ETF allocation
import pandas as pd
from src.processing.multi_total_return import load_relative_return_matrix
returns = pd.DataFrame({
    etf: load_relative_return_matrix(conn, etf, 6).rename(etf)
    for etf in config.ETF_BENCHMARK_UNIVERSE
})
bl_weights = build_bl_weights(ensemble_results, returns)
print(pd.Series(bl_weights).sort_values(ascending=False))

# CPCV validation for a single benchmark
from src.processing.feature_engineering import build_feature_matrix_from_db, get_X_y_relative
df = build_feature_matrix_from_db(conn)
y_vti = load_relative_return_matrix(conn, "VTI", 6)
X, y = get_X_y_relative(df, y_vti)
cpcv_result = run_cpcv(X, y, n_folds=6, n_test_folds=2)
print(f"CPCV mean IC: {cpcv_result.mean_ic:.4f} ± {cpcv_result.ic_std:.4f}")
print(f"Path ICs: {cpcv_result.path_ics}")

# Benchmark weights by historical IC × hit rate
backtest_results = run_full_backtest(conn)
bm_weights = compute_benchmark_weights(backtest_results)
```

### Monthly Decision Script

```bash
# Generate decision for the current month
python scripts/monthly_decision.py

# Backtest: generate decision as-of a past date
python scripts/monthly_decision.py --as-of 2025-07-20

# Dry run (writes files but does not commit)
python scripts/monthly_decision.py --dry-run
```

Outputs `results/monthly_decisions/YYYY-MM/recommendation.md` with:
- Fractional Kelly sell percentage recommendation
- Black-Litterman ETF reallocation targets
- TLH candidates from current lot positions
- CPCV validation summary
- Regime breakdown table

### v2 (legacy — LassoCV/RidgeCV)

```python
from src.models.multi_benchmark_wfo import run_all_benchmarks, get_current_signals
wfo_results = run_all_benchmarks(X, rel, model_type="lasso", target_horizon_months=6)
signals = get_current_signals(X, rel, wfo_results, X.iloc[[-1]])
```

---

## Tests

```bash
pytest                     # all tests
pytest -m integration      # integration smoke tests only
pytest -m "not integration" # fast unit tests only
```

**477 tests across 28 modules, all passing** (as of v4.0):

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_corporate_actions.py` | 17 | Split application, known split validation, cumulative multiplier |
| `test_total_return.py` | 13 | DRIP share accumulation, portfolio value, no negative prices |
| `test_feature_engineering.py` | 10 | No-leakage guarantee, target NaN in final 6M, Gainshare threshold |
| `test_capital_gains.py` | 14 | LTCG/STCG rate selection, lot priority, oversell validation |
| `test_db_client.py` | 37 | Schema idempotency, upsert/replace semantics, budget enforcement, proxy_fill |
| `test_multi_ticker_loader.py` | 34 | Price/dividend parsers, MultiTickerLoader, MultiDividendLoader, scheduler |
| `test_multi_total_return.py` | 36 | DRIP computation, relative targets, DatetimeIndex guard, get_X_y_relative |
| `test_wfo_engine.py` | 24 | Temporal integrity, embargo (6M/12M), v2 metadata, predict_current refit |
| `test_multi_benchmark_wfo.py` | 16 | run_all_benchmarks, get_current_signals, skip-on-no-overlap, signal classification |
| `test_backtest_engine.py` | 32 | Date enumeration, business-day snap, forward windows, signal/sell-pct logic |
| `test_reporting.py` | 27 | Pivot tables, print summary, CSV export, plot functions |
| `test_integration.py` | 11 | End-to-end smoke test: synthetic DB → schema → features → WFO → signals |
| `test_fred_loader.py` | 13 | Mock HTTP, monthly resampling, NaN handling, dry_run, missing key raises |
| `test_fred_db.py` | 10 | Schema creation, upsert idempotence, get_fred_macro() column alignment |
| `test_fred_features.py` | 5 | 6 derived FRED features present in feature matrix; no future leakage |
| `test_elasticnet.py` | 11 | Pipeline builds, l1_ratio grid, StandardScaler temporal isolation |
| `test_embargo_fix.py` | 9 | gap=8 for 6M; gap=15 for 12M; purge_buffer=0 reverts to old behavior |
| `test_monthly_backtest.py` | 11 | 120+ evaluation dates; vesting-date intersection invariant; no-lookahead |
| `test_oos_r2.py` | 22 | OOS R²=0 when predicted=historical mean; BHY FDR ≤ 5%; Newey-West finite p |
| `test_bayesian_ridge.py` | 14 | predict_with_std() finite; temporal isolation; ensemble averaging correct |
| `test_kelly_sizing.py` | 16 | Kelly formula; max position cap; zero-prediction → 100% sell |
| `test_pgr_fred_features.py` | 7 | insurance_cpi_mom3m, vmt_yoy, vix present in feature matrix |
| `test_regime_breakdown.py` | 10 | 4 quadrants populated; OOS R² in valid range; rolling IC window=24 |
| `test_cpcv.py` | 17 | C(6,2)=15 splits; 5 paths; train/test disjoint; all obs appear in test sets |
| `test_black_litterman.py` | 14 | Weights sum to 1; non-negative; ≤KELLY_MAX_POSITION; LW covariance PSD |
| `test_tlh.py` | 23 | Harvest at −10%; largest loss first; wash-sale +31d; after-tax formula |
| `test_fracdiff.py` | 13 | d* in [0,0.5]; Pearson corr ≥ 0.90; ADF stationarity; burn-in NaN |
| `test_benchmark_weights.py` | 11 | Weights sum to 1; IC≤0 → zero weight; IC×HR normalization correct |

Critical invariants enforced by tests:

- `max(train_idx) < min(test_idx)` for every WFO fold
- Embargo gap ≥ `target_horizon + purge_buffer` months
- `StandardScaler` is a named step inside `Pipeline` — never fit on full dataset
- `target_6m_return` is NaN for the final 6 months (no look-ahead leakage)
- `predict_current()` produces different outputs for different `X_current` inputs
- LTCG boundary: exactly 365 days does NOT qualify (must be > 365)
- API budget enforcement: 26th AV call raises `RuntimeError` (limit: 25/day)
- Upsert is idempotent: duplicate inserts replace, never duplicate rows
- CPCV: within every split, train and test index sets are disjoint
- BL covariance matrix is positive semi-definite (all eigenvalues ≥ −1e-10)
- TLH wash-sale clear date is exactly `harvest_date + TLH_WASH_SALE_DAYS`

---

## Known Limitations

- **Weekly price resolution**: Alpha Vantage free tier provides weekly (not daily)
  OHLCV. Sufficient for monthly feature engineering and DRIP calculations; ex-dividend
  dates are matched to the nearest weekly close.
- **Both lots are STCG at vest**: Shares vest with zero holding period, so both
  tranches are classified as short-term at the moment of vest-day sale. LTCG status
  accrues if shares are held at least 366 days post-vest.
- **BNDX history**: Vanguard Total International Bond (BNDX) launched June 2013,
  roughly 7 months before the earliest backtested vesting event (Jan 2014). The
  BNDX model will have slightly fewer early training observations than other benchmarks.
- **`fracdiff` package**: Incompatible with Python ≥ 3.10. The FFD algorithm is
  implemented internally in `feature_engineering.py` using numpy/scipy with equivalent
  mathematical behavior.
- **CPCV compute cost**: C(6,2)=15 splits × 20 benchmarks = 300 model fits per
  CPCV run. Used as a periodic diagnostic, not in the weekly cron.
- **Black-Litterman views require live predictions**: `build_bl_weights()` derives
  ETF views from ensemble model y_hat. If fewer than 12 months of returns data are
  available for a benchmark, that benchmark falls back to equal-weight allocation.

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
statsmodels>=0.14.0
skfolio>=0.3.0
PyPortfolioOpt>=1.5.5
```

---

## License

Private — personal financial decision support tool.
