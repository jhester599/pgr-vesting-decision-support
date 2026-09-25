-- Migration 006: append-only provenance for the PGR monthly 8-K table.
--
-- Review 2026-09-25, finding F33. pgr_edgar_monthly had no point-in-time
-- history: every monthly run re-parsed 24 months, overwrote filing_date and
-- accession_number, and COALESCEd each value column, so one row could mix
-- values from the CSV seed and from several live parses.
--
-- Every parse of a filing is kept, and nothing is ever updated or deleted
-- (the triggers below abort both). A parser fix adds a new parse under a new
-- parser_version instead of rewriting old values.
--
--   pgr_edgar_filing_parses  one row per (accession, parser_version): which
--                            filing and exhibit were read, when, by which
--                            parser version
--   pgr_edgar_monthly_raw    one row per (parse, field): the value as read
--
--   method = 'parsed'  read directly from the filing's exhibit
--            'derived' computed by the parser from other fields of the same
--                      filing (e.g. equity = BVPS x shares when no equity line)
--            'csv'     loaded by load_from_csv from pgr_edgar_cache.csv into a
--                      month the table did not have; fetched_at is NULL
--
-- Views:
--   pgr_edgar_monthly_raw_values     flat form: one row per
--                                    (accession, field, parser_version) with
--                                    fetched_at and the source URL
--   pgr_edgar_monthly_raw_current    the latest parse's value per
--                                    (accession, field)
--   pgr_edgar_monthly_first_reported the value from the earliest filing per
--                                    (month_end, field): what was known first

CREATE TABLE IF NOT EXISTS pgr_edgar_filing_parses (
    parse_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    accession_number TEXT    NOT NULL,
    parser_version   TEXT    NOT NULL,
    month_end        TEXT    NOT NULL,
    filing_date      TEXT,
    source_url       TEXT,
    fetched_at       TEXT,
    recorded_at      TEXT    NOT NULL,
    UNIQUE (accession_number, parser_version)
);

CREATE TABLE IF NOT EXISTS pgr_edgar_monthly_raw (
    parse_id    INTEGER NOT NULL REFERENCES pgr_edgar_filing_parses (parse_id),
    field       TEXT    NOT NULL,
    value_real  REAL,
    value_text  TEXT,
    method      TEXT    NOT NULL DEFAULT 'parsed',
    PRIMARY KEY (parse_id, field)
) WITHOUT ROWID;

CREATE TRIGGER IF NOT EXISTS pgr_edgar_filing_parses_no_update
BEFORE UPDATE ON pgr_edgar_filing_parses
BEGIN
    SELECT RAISE(ABORT, 'pgr_edgar_filing_parses is append-only');
END;

CREATE TRIGGER IF NOT EXISTS pgr_edgar_filing_parses_no_delete
BEFORE DELETE ON pgr_edgar_filing_parses
BEGIN
    SELECT RAISE(ABORT, 'pgr_edgar_filing_parses is append-only');
END;

CREATE TRIGGER IF NOT EXISTS pgr_edgar_monthly_raw_no_update
BEFORE UPDATE ON pgr_edgar_monthly_raw
BEGIN
    SELECT RAISE(ABORT, 'pgr_edgar_monthly_raw is append-only');
END;

CREATE TRIGGER IF NOT EXISTS pgr_edgar_monthly_raw_no_delete
BEFORE DELETE ON pgr_edgar_monthly_raw
BEGIN
    SELECT RAISE(ABORT, 'pgr_edgar_monthly_raw is append-only');
END;

CREATE VIEW IF NOT EXISTS pgr_edgar_monthly_raw_values AS
SELECT
    p.accession_number,
    r.field,
    p.parser_version,
    p.month_end,
    p.filing_date,
    r.value_real,
    r.value_text,
    r.method,
    p.source_url,
    p.fetched_at,
    p.recorded_at,
    p.parse_id
FROM pgr_edgar_monthly_raw AS r
JOIN pgr_edgar_filing_parses AS p ON p.parse_id = r.parse_id;

CREATE VIEW IF NOT EXISTS pgr_edgar_monthly_raw_current AS
SELECT
    accession_number, field, parser_version, month_end, filing_date,
    value_real, value_text, method, source_url, fetched_at, recorded_at, parse_id
FROM (
    SELECT
        v.*,
        ROW_NUMBER() OVER (
            PARTITION BY v.accession_number, v.field
            ORDER BY v.parse_id DESC
        ) AS rn
    FROM pgr_edgar_monthly_raw_values AS v
)
WHERE rn = 1;

CREATE VIEW IF NOT EXISTS pgr_edgar_monthly_first_reported AS
SELECT
    month_end, field, value_real, value_text, accession_number, filing_date,
    parser_version, method, fetched_at
FROM (
    SELECT
        v.*,
        ROW_NUMBER() OVER (
            PARTITION BY v.month_end, v.field
            ORDER BY COALESCE(v.filing_date, '9999'), v.accession_number, v.parse_id DESC
        ) AS rn
    FROM pgr_edgar_monthly_raw_values AS v
)
WHERE rn = 1;
