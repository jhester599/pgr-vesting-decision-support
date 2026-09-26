# v19 Feature Completion Plan

## Goal

Finish the original v15 research inventory so every externally proposed feature
idea has an explicit outcome:

- tested through the one-feature-at-a-time swap framework
- or closed out as blocked by a concrete source gap

## Scope

v19 focuses on the 23 feature ideas that were still unresolved after v18.

Workstreams:

1. backfill missing public macro and valuation series needed to make the
   remaining benchmark-side / valuation ideas testable
2. implement the remaining EDGAR-derived feature constructions that can be
   built from the broadened live parser output
3. rerun the fixed-budget feature-replacement cycle on the full now-available
   inventory
4. produce a final traceability matrix covering all 46 original features

## Public-Series Backfill

Use public endpoints only:

- FRED graph CSV:
  - `DTWEXBGS`
  - `DCOILWTICO`
  - `MORTGAGE30US`
  - `WPU45110101`
  - `PPIACO`
  - `MRTSSM447USN`
  - `THREEFYTP10`
- BLS public API:
  - `CUSR0000SETE`
- Multpl monthly tables:
  - `SP500_PE_RATIO_MULTPL`
  - `SP500_EARNINGS_YIELD_MULTPL`
  - `SP500_PRICE_TO_BOOK_MULTPL`

Store the fetched monthly observations in `fred_macro_monthly` so the feature
builder can reuse them without introducing a separate research-only table.

## Remaining EDGAR / Derived Features

Implement and evaluate these where the data exists:

- `reserve_to_npe_ratio`
- `direct_channel_pif_share_ttm`
- `channel_mix_direct_pct_yoy`
- `realized_gain_to_net_income_ratio`
- `unrealized_gain_pct_equity`
- `loss_ratio_ttm`
- `expense_ratio_ttm`
- `pgr_premium_to_surplus`
- `auto_pricing_power_spread`
- `usd_broad_return_3m`
- `usd_momentum_6m`
- `wti_return_3m`
- `mortgage_spread_30y_10y`
- `credit_spread_ratio`
- `term_premium_10y`
- `legal_services_ppi_relative`
- `gasoline_retail_sales_delta`
- `pgr_price_to_book_relative`
- `pgr_pe_vs_market_pe`
- `equity_risk_premium`
- `excess_bond_premium_proxy`

## Explicit Blocker Review

Close out as blocked if still not testable after the source audit:

- `pgr_cr_vs_peer_cr`
- `pgr_fcf_yield`

These should not silently remain in `queued_*` status once v19 finishes.

## Research-Only Gate Relaxation

Some broadened live-parser EDGAR fields only have about 25 monthly
observations. For v19 evaluation only, relax the EDGAR breadth gate from the
production-oriented `WFO_MIN_GAINSHARE_OBS=60` to `24` so those fields can be
screened one feature at a time.

This relaxation is for research coverage, not production promotion.

## Acceptance Criteria

- all 46 original inventory rows have one of:
  - `tested`
  - `blocked`
- a traceability CSV exists under `results/v19/`
- the repo has a v19 summary and closeout memo
- full regression remains green
