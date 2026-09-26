# x1 Research Memo

## Scope

x1 sets up the separate x-series research lane for absolute PGR
forecasting and Q1 special-dividend forecasting. It does not fit
models and does not alter monthly decision outputs.

## Available History

- Feature matrix: 316 rows x 77 columns, 2000-01-31 to 2026-04-30.
- Feature matrix source: existing processed cache, read without refresh.
- PGR monthly EDGAR: 258 rows, 2004-08-31 to 2026-03-31.
- PGR dividends: 76 rows, 1999-03-10 to 2026-04-02.

## Candidate Target Depth

- 1m forward return: 317 non-null observations.
- 3m forward return: 315 non-null observations.
- 6m forward return: 312 non-null observations.
- 12m forward return: 306 non-null observations.
- Special-dividend annual snapshots: 22 labeled observations.

## Feature Inventory Takeaways

- price_momentum_volatility: 5 features.
- technical_indicators_existing: 0 features.
- book_value_related: 2 features.
- buyback_capital_return: 2 features.
- gainshare: 1 features.
- peer_market_relative: 6 features.
- valuation: 5 features.
- underwriting_profitability: 9 features.
- growth_pif_premium: 10 features.
- capital_balance_sheet: 2 features.
- macro_rates_spreads: 19 features.
- uncategorized: 15 features.

## Recommended Ordering

1. x2 multi-horizon classification baseline.
2. x3 direct forward-return/log-return implied-price benchmark.
3. x4/x5 BVPS and P/B decomposition benchmark.
4. x6 special-dividend two-stage annual sidecar.
5. x7 targeted feature expansion, including bounded TA follow-up.

## Caveats

- Annual special-dividend labels are sparse and should be treated as
  high-uncertainty capital-allocation research.
- Multi-month absolute targets overlap by construction, so x2+ must use
  horizon-specific purge/embargo logic.
- The normal quarterly dividend baseline is inferred from repo dividend
  history rather than hardcoded.
