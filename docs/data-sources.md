# Data Sources

## Alpha Vantage

Used for:

- daily/weekly prices
- dividends

Operational notes:

- free-tier daily limit applies
- tracked in the database request log
- production workflows should verify row growth and latest dates after runs

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

- the live HTML parser now covers a broad slice of the historical CSV shape
- monthly results are stored in `pgr_edgar_monthly`
- the committed CSV remains the historical baseline / backfill source

## Local CSV Inputs

- `data/processed/pgr_edgar_cache.csv`
  - historical PGR monthly operating metrics
- `data/processed/position_lots.csv`
  - lot-level position input for tax-aware reporting

## Source of Truth

- The committed SQLite database is the operational source of truth for
  workflows and monthly reporting.
- Research outputs are not sources of truth for production behavior.
- Historical plan documents are informational, not operational authorities.
