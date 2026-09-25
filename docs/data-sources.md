# Data Sources

## Alpha Vantage

Used for:

- daily/weekly prices
- dividends

Operational notes:

- free-tier daily limit applies
- tracked in the database request log
- production workflows should verify row growth and latest dates after runs
- prices come from `TIME_SERIES_WEEKLY` and are **unadjusted**; the store
  keeps one bar per ticker per ISO week (the latest-dated one), because the
  in-progress week is labelled with its latest trading day
- `DIVIDENDS` amounts are raw per-share amounts (not split-adjusted); the
  loader sleeps before its first call and retries "Information" advisories
  with exponential backoff
- splits are not ingested automatically: `config/splits.py` is the canonical
  list, and `scripts/detect_splits.py` checks it against split coefficients
  recovered from `TIME_SERIES_WEEKLY_ADJUSTED`

Relative-return targets (`monthly_relative_returns`) are DRIP total returns
on unadjusted prices from the last bar on or before business month-end `t` to
the last bar on or before `BMonthEnd(t + h)`.

## FRED

Used for:

- macro regime features
- insurance-relevant CPI/PPI and miles-traveled inputs

Operational notes:

- publication lags are applied in feature engineering
- production monthly workflow can skip live fetch if the environment lacks
  `FRED_API_KEY`

## SEC EDGAR

Used for:

- quarterly companyfacts fundamentals
- monthly PGR 8-K supplement parsing

Operational notes:

- every monthly release since August 2004 is parsed by the repository's
  parser (`scripts/edgar_8k_fetcher.py`): items 7.01 and 2.02, and 9.01-only
  filings with an EX-99 exhibit, found through the primary submissions file
  and the flat pagination files
- monthly results are stored in `pgr_edgar_monthly`; each row's values all
  come from the filing in its `accession_number`
- every parsed value is also kept, append-only, in `pgr_edgar_monthly_raw`
  (with `pgr_edgar_filing_parses`: accession, parser version, exhibit URL,
  fetched-at); `pgr_edgar_monthly_first_reported` gives the first-reported
  value of each field. A later filing for a month never overwrites the row.
- derived fields use one definition each (`src/processing/pgr_edgar_derived.py`):
  calendar-month YoY (NaN when the base month is missing), `pif_total` =
  agency + direct + special lines + commercial (property excluded), one
  Gainshare formula
- `pgr_fundamentals_quarterly` holds discrete quarters from XBRL (Q4 = FY −
  9M), earliest-filed values, ROE = TTM net income / average equity, and the
  filing date; the always-NULL `pe_ratio` / `pb_ratio` columns were dropped
- the committed CSV is exported from the DB and seeds only missing months
- freshness checks are calendar-aware: prior-month PGR monthly 8-K data is
  required only after the configured filing grace window

## Local CSV Inputs

- `data/processed/pgr_edgar_cache.csv`
  - PGR monthly operating metrics, one row per month since 2004-08, exported
    from `pgr_edgar_monthly` (`scripts/repair_edgar_history.py --export-csv`);
    columns are described in `docs/PGR_EDGAR_CACHE_DATA_DICTIONARY.md`
    (regenerate with `python scripts/generate_edgar_data_dictionary.py`)
- `data/processed/pgr_valuation_monthly.csv`
  - derived monthly P/B and trailing P/E (split-consistent TTM EPS), one row
    per calendar month; regenerate with
    `python scripts/export_pgr_valuation_multiples.py`
  - a missing month's EPS would be filled as quarterly XBRL EPS less the
    other two reported months, and missing book value interpolated by rolling
    forward monthly EPS; since the step 3b repair no month is missing, so
    every row is reported. `*_source` columns flag any fill and
    `data_available_date` gives the first date all inputs behind a row were
    public
- `data/processed/position_lots.csv`
  - lot-level position input for tax-aware reporting

## Source of Truth

- The committed SQLite database is the operational source of truth for
  workflows and monthly reporting.
- Research outputs are not sources of truth for production behavior.
- Historical plan documents are informational, not operational authorities.
