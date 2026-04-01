-- PGR Vesting Decision Support — v2 SQLite Schema
-- All date columns store ISO 8601 strings ('YYYY-MM-DD').
-- Primary keys enforce uniqueness and enable efficient upserts.
-- WAL mode and foreign_keys are enabled at connection time by db_client.py.

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
-- Source: user-provided pgr_edgar_cache.csv (migrated at v2 init)
-- combined_ratio:      GAAP combined ratio (loss + expense); below 96% = target
-- pif_total:           policies in force (total count)
-- pif_growth_yoy:      year-over-year PIF growth (decimal, e.g. 0.12 = 12%)
-- gainshare_estimate:  estimated annual multiplier [0.0, 2.0]
-- book_value_per_share: BVPS from monthly 8-K (v6.x); used to derive pb_ratio
-- eps_basic:           monthly basic EPS from 8-K (v6.x); TTM sum used for pe_ratio
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pgr_edgar_monthly (
    month_end             TEXT    NOT NULL,
    combined_ratio        REAL,
    pif_total             REAL,
    pif_growth_yoy        REAL,
    gainshare_estimate    REAL,
    book_value_per_share  REAL,
    eps_basic             REAL,
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
