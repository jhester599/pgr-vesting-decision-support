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

Price-derived features never use raw closes across dates. Every price feature
goes through `src/processing/price_adjustment.split_adjusted_close`
(`close × share_basis_factor(date) / latest factor`, from `split_history`),
and windows are calendar-based on weekly bars:

- `mom_3m`, `mom_6m`, `mom_12m`: calendar-month returns between month-end
  closes (last bar on or before each business month-end);
- `vol_63d` (and the dropped `vol_21d`): standard deviation of the last 13
  (4) weekly log returns × √52. The names keep the old trading-day labels:
  13 weeks span the same quarter that 63 trading days did;
- `high_52w`: month-end close over the highest close of the last 52 weekly
  bars;
- the synthetic spreads (`pgr_vs_kie_6m`, `pgr_vs_peers_6m`, `pgr_vs_vfh_6m`,
  `vwo_vxus_spread_6m`, `gold_vs_treasury_6m`,
  `commodity_equity_momentum`): 6-month calendar returns on each ticker's
  split-adjusted closes;
- P/B, P/E, BVPS growth and buyback yield: EDGAR per-share values are
  restated to the latest share basis on their report period, before they
  are placed on the first month-end on or after their filing date, and
  divided by the split-adjusted price;
- TA shadow features (`src/research/v160_ta_features.py`): split-adjusted
  OHLCV (volume scaled inversely) on weekly bars, with 13/26/52-bar windows;
- Monte Carlo tax volatility: the last 52 split-adjusted weekly log returns
  × √52.

`build_feature_matrix` and `build_ta_feature_matrix` collapse daily input to
weekly bars and raise `ValueError` for anything coarser than weekly. Raw
closes are used only for DRIP total returns, which apply splits to the share
count.

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
- timing (review 2026-09-25, F23): each monthly and quarterly EDGAR row
  enters the feature matrix on the first business month-end on or after its
  `filing_date` (`feature_engineering.edgar_availability_dates`). Monthly
  8-Ks are filed 9-29 days after the month, so they enter one month after it
  (the fixed 2-month lag used before was a month late; for some 10-Ks, such
  as FY2024 filed 2025-03-03, it was early). The lag remains only as the
  fallback for a row without a filing date
- every request sends `EDGAR_USER_AGENT` (a name and contact e-mail); calls
  fail when it is unset, blank or the old placeholder

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
