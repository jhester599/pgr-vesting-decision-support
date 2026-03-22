# PGR Vesting Decision Support

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

---

## Architecture

```
pgr-vesting-decision-support/
├── config.py                         # API keys, WFO params, tax rates
├── requirements.txt
│
├── src/
│   ├── ingestion/
│   │   ├── price_loader.py           # AV TIME_SERIES_WEEKLY → weekly OHLCV
│   │   ├── dividend_loader.py        # AV DIVIDENDS → ex-div dates + amounts
│   │   ├── split_loader.py           # Hardcoded from config (3 known PGR splits)
│   │   ├── pgr_monthly_loader.py     # EDGAR cache CSV → combined ratio, PIF
│   │   ├── fundamentals_loader.py    # FMP quarterly key metrics (optional)
│   │   ├── technical_loader.py       # AV SMA/RSI/MACD/BBANDS (optional)
│   │   ├── fmp_client.py             # Cache-first FMP REST wrapper
│   │   └── av_client.py              # Cache-first Alpha Vantage wrapper
│   │
│   ├── processing/
│   │   ├── corporate_actions.py      # Forward-applies splits to share count
│   │   ├── total_return.py           # DRIP total return reconstruction
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
├── plots/                            # Output PNG charts (gitignored)
│
└── tests/                            # 69 pytest tests, all passing
```

---

## Methodology

### Data Sources

| Source | Data | Tier |
|--------|------|------|
| Alpha Vantage `TIME_SERIES_WEEKLY` | Weekly OHLCV (1,376 weeks, 1999–present) | Free |
| Alpha Vantage `DIVIDENDS` | 76 ex-dividend records | Free |
| Alpha Vantage `SMA/RSI/MACD/BBANDS` | Monthly technical indicators | Free (25 req/day) |
| EDGAR cache CSV | 256 months of combined ratio, PIF, EPS | User-provided |
| `config.PGR_KNOWN_SPLITS` | 3 historical splits (1992, 2002, 2006) | Hardcoded |

All API responses are cached to `data/raw/` as JSON. After initial population,
subsequent runs consume zero API calls.

### DRIP Total Return Reconstruction

Unadjusted prices are used throughout. Share count is tracked explicitly:

```
Split:    shares *= split_ratio           (forward-applied on split date)
Dividend: shares += (shares * div) / price_on_exdiv_date   (fractional DRIP)
Value:    V[t] = shares[t] * close[t]
```

This produces an accurate total-return series that correctly models fractional
share accumulation — required for the 6-month forward return training label.

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
| `target_6m_return` | Forward 6-month DRIP total return **(label)** |

Gainshare features are retained only if ≥ 60 non-NaN observations exist;
otherwise dropped before WFO training. The final 6 months of `target_6m_return`
are always NaN (no look-ahead leakage).

### Walk-Forward Optimization (WFO)

**No K-Fold cross-validation.** All validation uses strict temporal splits.

```python
TimeSeriesSplit(
    n_splits      = (total_months - 60) // 6,   # ~41 folds on current data
    max_train_size = 60,   # 5-year rolling window
    test_size      = 6,    # 6-month out-of-sample test
    gap            = 1,    # 1-month embargo to prevent leakage
)
```

Per-fold procedure:
1. Slice `X_train / X_test` by fold indices
2. Impute NaN with **training-fold column medians** (no leakage; all-NaN columns → 0.0)
3. Fit `StandardScaler` only on `X_train` (inside sklearn `Pipeline`)
4. Tune `alpha` via nested `TimeSeriesSplit(n_splits=3)` inner CV on `X_train`
5. Fit `LassoCV` on full `X_train`, predict `X_test`
6. Store fold metrics and non-zero Lasso coefficients

**Aggregate metrics (41 out-of-sample folds, current data):**

| Metric | Value |
|--------|-------|
| Information Coefficient (Spearman) | -0.056 |
| Hit Rate (sign accuracy) | 66.7% |
| Mean Absolute Error | 0.135 |

The IC is below the 0.05 confidence threshold, so the engine defaults to the
diversification-first rule (50% sale) rather than acting on directional signal.
The dominant pattern identified by Lasso is **mean-reversion** — all three
momentum features carry negative coefficients.

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

### Portfolio Reallocation

Net proceeds are distributed across the top-5 ETFs most underweight relative to
MSCI ACWI equilibrium sector weights. 22 low-expense ETFs are considered across
5 categories (US broad, international, sector tilts, dividend, fixed income).

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

---

## Running the Engine

```python
import sys
sys.path.insert(0, ".")

from datetime import date
from src.ingestion import price_loader, dividend_loader, split_loader, pgr_monthly_loader
from src.processing.feature_engineering import build_feature_matrix, get_X_y
from src.models.wfo_engine import run_wfo
from src.tax.capital_gains import load_position_lots
from src.portfolio.drift_analyzer import PortfolioState
from src.portfolio.rebalancer import generate_recommendation, print_recommendation
from src.visualization.plots import (
    plot_wfo_equity_curve, plot_feature_importance, plot_portfolio_drift
)

# Load data
prices      = price_loader.load()
divs        = dividend_loader.load()
splits      = split_loader.load()
pgr_monthly = pgr_monthly_loader.load()

# Build features and train WFO
fm = build_feature_matrix(prices, divs, splits, pgr_monthly=pgr_monthly)
X, y = get_X_y(fm, drop_na_target=True)
wfo = run_wfo(X, y, model_type="lasso")

# Generate recommendations
current_price = 206.00
lots = load_position_lots("data/processed/position_lots.csv")
portfolio = PortfolioState(pgr_value=1000 * current_price, etf_holdings={})

for vest_date, rsu_type in [(date(2026, 7, 17), "performance"), (date(2027, 1, 19), "time")]:
    rec = generate_recommendation(vest_date, rsu_type, current_price, lots, wfo, portfolio)
    print_recommendation(rec)

# Save plots
plot_wfo_equity_curve(wfo)
plot_feature_importance(wfo)
plot_portfolio_drift(portfolio)
```

---

## Tests

```bash
pytest tests/ -v
```

69 tests across 5 modules, all passing:

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_corporate_actions.py` | 17 | Split application, known split validation, cumulative multiplier |
| `test_total_return.py` | 13 | DRIP share accumulation, portfolio value, no negative prices |
| `test_feature_engineering.py` | 11 | No-leakage guarantee, target NaN in final 6M, Gainshare threshold |
| `test_wfo_engine.py` | 15 | Temporal integrity, embargo gap, scaler isolation, independent folds |
| `test_capital_gains.py` | 13 | LTCG/STCG rate selection, lot priority, oversell validation |

Critical invariants enforced by tests:
- `max(train_idx) < min(test_idx)` for every WFO fold
- `min(test_idx) - max(train_idx) >= EMBARGO_MONTHS`
- `StandardScaler` is a named step inside `Pipeline` — never fit on full dataset
- `target_6m_return` is NaN for the final 6 months (no look-ahead leakage)
- LTCG boundary: exactly 365 days does NOT qualify (must be > 365)

---

## Known Limitations

- **Weekly price resolution**: Alpha Vantage free tier provides weekly (not daily)
  OHLCV. This is sufficient for grant-date cost basis lookups and monthly feature
  engineering but limits intraday precision.
- **IC below threshold**: With ~25 years of monthly data and a rolling 5-year
  training window, the Lasso model does not achieve IC ≥ 0.05 on current data.
  The engine falls back to the 50% diversification-sale rule. IC may improve with
  additional fundamental features or a longer EDGAR cache history.
- **Both lots are STCG at vest**: Shares vest on grant date with zero holding
  period, so both tranches are classified as short-term at the moment of vest-day
  sale. LTCG status accrues if shares are held at least 366 days post-vest.

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
