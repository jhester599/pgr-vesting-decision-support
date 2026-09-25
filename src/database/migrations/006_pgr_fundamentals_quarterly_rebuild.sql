-- Migration 006: pgr_fundamentals_quarterly without the always-NULL columns.
--
-- Review 2026-09-25, finding F09. pe_ratio and pb_ratio were never populated
-- (0 of 74 rows): XBRL has no prices, and P/E and P/B are computed from the
-- monthly 8-K data in feature_engineering. They are dropped. filing_date (the
-- filing date of the net-income value, the 10-K for Q4) is added so that each
-- row says when it became public.
--
-- Existing rows are carried over unchanged; they still hold the old
-- definitions (Q4 = full year, ROE = 4 x quarterly NI / ending equity) until
-- the table is refreshed from EDGAR (scripts/repair_edgar_history.py, or the
-- weekly fetch).

CREATE TABLE IF NOT EXISTS pgr_fundamentals_quarterly_v6 (
    period_end   TEXT    NOT NULL,
    roe          REAL,
    eps          REAL,
    revenue      REAL,
    net_income   REAL,
    filing_date  TEXT,
    source       TEXT,
    PRIMARY KEY (period_end)
);

INSERT OR IGNORE INTO pgr_fundamentals_quarterly_v6
    (period_end, roe, eps, revenue, net_income, source)
SELECT period_end, roe, eps, revenue, net_income, source
FROM pgr_fundamentals_quarterly;

DROP TABLE pgr_fundamentals_quarterly;

ALTER TABLE pgr_fundamentals_quarterly_v6 RENAME TO pgr_fundamentals_quarterly;
