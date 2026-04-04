# Workflows

## Production Workflows

### `weekly_data_fetch.yml`

Purpose:

- refresh main benchmark price data
- refresh PGR dividends
- refresh PGR quarterly fundamentals
- refresh FRED macro data
- rebuild monthly relative-return targets

Outputs:

- `data/pgr_financials.db`
- job summary with key counts and latest dates

### `peer_data_fetch.yml`

Purpose:

- refresh peer price and dividend history for `ALL`, `TRV`, `CB`, and `HIG`

Outputs:

- `data/pgr_financials.db`
- job summary with peer freshness verification

### `monthly_8k_fetch.yml`

Purpose:

- refresh monthly PGR 8-K operating metrics
- support a primary and fallback schedule each month

Outputs:

- `data/pgr_financials.db`
- job summary with latest EDGAR row metadata

### `monthly_decision.yml`

Purpose:

- generate the monthly recommendation and diagnostic artifacts
- update the decision log
- optionally send the decision email

Outputs:

- `results/monthly_decisions/<YYYY-MM>/recommendation.md`
- `results/monthly_decisions/<YYYY-MM>/diagnostic.md`
- `results/monthly_decisions/<YYYY-MM>/signals.csv`
- `results/monthly_decisions/<YYYY-MM>/run_manifest.json`
- `results/monthly_decisions/decision_log.md`

## Historical / Manual Workflows

The repository also retains one-off bootstrap workflows such as:

- `initial_fetch_prices.yml`
- `initial_fetch_dividends.yml`
- `peer_bootstrap.yml`
- `post_initial_bootstrap.yml`

These are retained for historical recovery and manual bootstrap scenarios, but
they are not part of the normal steady-state operating loop.

## CI Workflow

`ci.yml` runs:

- lint checks
- unit/integration test suite
- smoke runs for major production entrypoints
- migration/fresh-temp-DB checks

## Concurrency Policy

Production workflows that can mutate the committed database or monthly artifacts
use workflow-level concurrency groups so overlapping runs do not step on each
other.
