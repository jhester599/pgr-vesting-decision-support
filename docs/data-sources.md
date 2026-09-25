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

- `fred_macro_monthly` stores raw, unlagged observations: one row per series
  and calendar month (the month's last observation), labelled with the
  month's last business day. A unique index on (series, month) prevents a
  second label for the same month (migration 005).
- publication lags (`config.FRED_SERIES_LAGS`, default 1 month) are applied
  exactly once, by calendar month, in
  `feature_engineering.build_feature_matrix_from_db`: the feature row for
  month M uses the observation for month M − lag. Interior gaps (e.g. the
  October 2025 CPI BLS never published) are forward-filled for up to
  `FRED_MAX_GAP_FILL_MONTHS`; a series is never carried past its latest
  observation.
- the weekly and monthly jobs refresh the macro and PGR-specific series
  (`production_fred_series()`).
- freshness is checked per FRED series behind a live Ridge/GBT feature
  (`config.FRED_FEATURE_SOURCES`) against the month the decision row needs,
  and per ticker for PGR and the 8 forecast benchmarks.
- FRED serves only the last three years of ICE BofA series
  (`BAMLH0A0HYM2`); older months in the table were recovered from the
  pre-rebuild store and must not be deleted. Values are the current vintage,
  not point-in-time (NFCI, CPIs, PPIs and VMT are revised).
- `scripts/rebuild_fred_macro.py --db COPY` rebuilds the table from FRED
  (API with `FRED_API_KEY`, else the public `fredgraph.csv` endpoint).
- production monthly workflow can skip live fetch if the environment lacks
  `FRED_API_KEY`

## SEC EDGAR

Used for:

- quarterly companyfacts fundamentals
- monthly PGR 8-K supplement parsing

Operational notes:

- the live HTML parser now covers a broad slice of the historical CSV shape
- monthly results are stored in `pgr_edgar_monthly`
- the committed CSV remains the historical baseline / backfill source
- freshness checks are calendar-aware: prior-month PGR monthly 8-K data is
  required only after the configured filing grace window

## Local CSV Inputs

- `data/processed/pgr_edgar_cache.csv`
  - historical PGR monthly operating metrics
- `data/processed/pgr_valuation_monthly.csv`
  - derived monthly P/B and trailing P/E (split-consistent TTM EPS), one row
    per calendar month; regenerate with
    `python scripts/export_pgr_valuation_multiples.py`
  - missing EPS (2015-05, 2019-04) is filled as quarterly XBRL EPS less the
    other two reported months; missing book value (2005-02, 2005-03,
    2007-08, 2015-05, 2019-04) is interpolated by rolling forward monthly
    EPS; `*_source` columns flag every fill and `data_available_date` gives
    the first date all inputs behind a row were public
- `data/processed/position_lots.csv`
  - lot-level position input for tax-aware reporting

## Source of Truth

- The committed SQLite database is the operational source of truth for
  workflows and monthly reporting.
- Research outputs are not sources of truth for production behavior.
- Historical plan documents are informational, not operational authorities.
