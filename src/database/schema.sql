-- PGR Vesting Decision Support — v2 SQLite Schema
-- All date columns store ISO 8601 strings ('YYYY-MM-DD').
-- Primary keys enforce uniqueness and enable efficient upserts.
-- WAL mode and foreign_keys are enabled at connection time by db_client.py.

-- ---------------------------------------------------------------------------
-- Applied schema migrations
-- Tracks ordered migration files applied by migration_runner.py.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_id  TEXT PRIMARY KEY,
    applied_at    TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Daily OHLCV prices for PGR and all 22 ETF benchmarks
-- source: 'av' (Alpha Vantage) or 'fmp' (Financial Modeling Prep)
-- proxy_fill: 1 if this row was copied from a proxy ticker (e.g. VTI for FZROX
--             before its 2018 launch date). Used to weight or exclude proxy rows
--             in backtests that require only true history.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS daily_prices (
    ticker      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL    NOT NULL,
    volume      INTEGER,
    source      TEXT,
    proxy_fill  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (ticker, date)
);

-- ---------------------------------------------------------------------------
-- Dividend payments (ex-dividend dates) for PGR and all ETF benchmarks
-- Required for DRIP total-return reconstruction.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS daily_dividends (
    ticker      TEXT    NOT NULL,
    ex_date     TEXT    NOT NULL,
    amount      REAL    NOT NULL,
    source      TEXT,
    PRIMARY KEY (ticker, ex_date)
);

-- ---------------------------------------------------------------------------
-- Historical stock splits (currently only PGR has known splits)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS split_history (
    ticker      TEXT    NOT NULL,
    split_date  TEXT    NOT NULL,
    split_ratio REAL    NOT NULL,
    numerator   REAL,
    denominator REAL,
    PRIMARY KEY (ticker, split_date)
);

-- ---------------------------------------------------------------------------
-- PGR quarterly fundamental metrics from SEC EDGAR XBRL (via edgar_client.py)
-- Sourced from 10-Q and 10-K filings; free, authoritative, no API key needed.
-- Previously sourced from FMP (deprecated 2025-08-31 for free-tier accounts).
-- pe_ratio, pb_ratio: NULL (require market price data; not available from XBRL)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pgr_fundamentals_quarterly (
    period_end  TEXT    NOT NULL,
    pe_ratio    REAL,
    pb_ratio    REAL,
    roe         REAL,
    eps         REAL,
    revenue     REAL,
    net_income  REAL,
    source      TEXT,
    PRIMARY KEY (period_end)
);

-- ---------------------------------------------------------------------------
-- PGR monthly operating metrics from SEC EDGAR 8-K filings
-- Source: user-provided pgr_edgar_cache.csv (migrated at v2 init); live fetch
--         via scripts/edgar_8k_fetcher.py for recent months.
--
-- Core metrics (v2 baseline):
--   combined_ratio:      GAAP combined ratio (loss + expense); below 96% = target
--   pif_total:           policies in force — total count (thousands)
--   pif_growth_yoy:      year-over-year PIF growth (decimal, e.g. 0.12 = 12%)
--   gainshare_estimate:  estimated annual multiplier [0.0, 2.0]
--
-- v6.x additions:
--   book_value_per_share: BVPS from monthly 8-K; used to derive pb_ratio
--   eps_basic:            monthly basic EPS; TTM sum used for pe_ratio
--
-- v6.2 Phase 1 — foundational P&L / balance sheet fields (all in CSV):
--   net_premiums_written, net_premiums_earned, net_income, eps_diluted,
--   loss_lae_ratio, expense_ratio — foundational companywide P&L
--
-- v6.2 Phase 1 — segment-level channel metrics:
--   npw_*/npe_*: net premiums written/earned by segment (agency/direct/commercial/property)
--   pif_agency_auto, pif_direct_auto, pif_commercial_lines, pif_total_personal_lines
--
-- v6.2 Phase 1 — company-level operating metrics:
--   investment_income, total_revenues, total_expenses, income_before_income_taxes,
--   roe_net_income_ttm, shareholders_equity, total_assets,
--   unearned_premiums, shares_repurchased, avg_cost_per_share
--
-- v6.2 Phase 2 — investment portfolio metrics:
--   fte_return_total_portfolio, investment_book_yield,
--   net_unrealized_gains_fixed, fixed_income_duration
--
-- v6.2 derived fields (computed at CSV load time):
--   channel_mix_agency_pct: npw_agency / (npw_agency + npw_direct)
--   npw_growth_yoy:         net_premiums_written YoY % change
--   underwriting_income:    net_premiums_earned * (1 - combined_ratio/100)
--   unearned_premium_growth_yoy: unearned_premiums YoY % change
--   buyback_yield:          (shares_repurchased * avg_cost_per_share) / market_cap
--                           NULL for CSV-load path (market_cap requires price data)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pgr_edgar_monthly (
    month_end                    TEXT    NOT NULL,
    filing_date                  TEXT,
    filing_type                  TEXT,
    accession_number             TEXT,
    -- Core v2 fields
    combined_ratio               REAL,
    pif_total                    REAL,
    pif_growth_yoy               REAL,
    gainshare_estimate           REAL,
    -- v6.x additions
    book_value_per_share         REAL,
    eps_basic                    REAL,
    avg_diluted_equivalent_shares REAL,
    -- v6.2: foundational P&L
    net_premiums_written         REAL,
    net_premiums_earned          REAL,
    net_income                   REAL,
    eps_diluted                  REAL,
    total_net_realized_gains     REAL,
    service_revenues             REAL,
    fees_and_other_revenues      REAL,
    losses_lae                   REAL,
    policy_acquisition_costs     REAL,
    other_underwriting_expenses  REAL,
    interest_expense             REAL,
    provision_for_income_taxes   REAL,
    total_comprehensive_income   REAL,
    comprehensive_eps_diluted    REAL,
    avg_shares_basic             REAL,
    avg_shares_diluted           REAL,
    loss_lae_ratio               REAL,
    expense_ratio                REAL,
    -- v6.2: segment-level channel metrics
    npw_agency                   REAL,
    npw_direct                   REAL,
    npw_commercial               REAL,
    npw_property                 REAL,
    npe_agency                   REAL,
    npe_direct                   REAL,
    npe_commercial               REAL,
    npe_property                 REAL,
    pif_agency_auto              REAL,
    pif_direct_auto              REAL,
    pif_special_lines            REAL,
    pif_property                 REAL,
    pif_commercial_lines         REAL,
    pif_total_personal_lines     REAL,
    -- v6.2: company-level operating metrics
    investment_income            REAL,
    total_revenues               REAL,
    total_expenses               REAL,
    income_before_income_taxes   REAL,
    roe_net_income_ttm           REAL,
    roe_comprehensive_trailing_12m REAL,
    shareholders_equity          REAL,
    total_assets                 REAL,
    total_investments            REAL,
    loss_lae_reserves            REAL,
    unearned_premiums            REAL,
    debt                         REAL,
    total_liabilities            REAL,
    common_shares_outstanding    REAL,
    shares_repurchased           REAL,
    avg_cost_per_share           REAL,
    -- v6.2: investment portfolio metrics
    fte_return_fixed_income      REAL,
    fte_return_common_stocks     REAL,
    fte_return_total_portfolio   REAL,
    investment_book_yield        REAL,
    net_unrealized_gains_fixed   REAL,
    fixed_income_duration        REAL,
    debt_to_total_capital        REAL,
    weighted_avg_credit_quality  TEXT,
    -- v6.2: derived fields
    channel_mix_agency_pct       REAL,
    npw_growth_yoy               REAL,
    underwriting_income          REAL,
    unearned_premium_growth_yoy  REAL,
    buyback_yield                REAL,
    PRIMARY KEY (month_end)
);

-- ---------------------------------------------------------------------------
-- Pre-computed monthly relative returns for each benchmark / horizon combo
-- pgr_return:        PGR DRIP total return over the horizon
-- benchmark_return:  ETF DRIP total return over the same horizon
-- relative_return:   pgr_return - benchmark_return (the ML target variable)
-- proxy_fill:        1 if the benchmark's return was derived from proxy data
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS monthly_relative_returns (
    date             TEXT    NOT NULL,
    benchmark        TEXT    NOT NULL,
    target_horizon   INTEGER NOT NULL,   -- 6 or 12 (months)
    pgr_return       REAL,
    benchmark_return REAL,
    relative_return  REAL,
    proxy_fill       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (date, benchmark, target_horizon)
);

-- ---------------------------------------------------------------------------
-- Daily API request log — used by db_client.log_api_request() to enforce
-- free-tier daily limits (AV: 25/day, FMP: 250/day).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS api_request_log (
    api      TEXT    NOT NULL,
    date     TEXT    NOT NULL,   -- UTC date 'YYYY-MM-DD'
    endpoint TEXT,
    count    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (api, date, endpoint)
);

-- ---------------------------------------------------------------------------
-- Tracks the last successful fetch time and row count per ticker/data_type.
-- Used by the fetch scheduler to skip already-fresh data.
-- data_type: 'prices', 'dividends', or 'fundamentals'
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_metadata (
    ticker       TEXT NOT NULL,
    data_type    TEXT NOT NULL,
    last_fetched TEXT,           -- ISO 8601 datetime of last successful fetch
    rows_stored  INTEGER,
    PRIMARY KEY (ticker, data_type)
);

-- ---------------------------------------------------------------------------
-- FRED macro and insurance-specific monthly series (v3.0+)
-- series_id: FRED series identifier (e.g. 'T10Y2Y', 'BAMLH0A0HYM2')
-- month_end:  ISO date of the last calendar day of the month ('YYYY-MM-DD')
-- value:      Raw FRED observation value (numeric); NULL if FRED reports '.'
--
-- Populated by src/ingestion/fred_loader.py via scripts/weekly_fetch.py.
-- Used by src/processing/feature_engineering.py as macro regime features.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fred_macro_monthly (
    series_id  TEXT NOT NULL,
    month_end  TEXT NOT NULL,
    value      REAL,
    PRIMARY KEY (series_id, month_end)
);
