# Workflows

## Production Workflows

### `weekly_data_fetch.yml`

Two schedules (review 2026-09-25, WP1/WP4):

- **Friday 22:00 UTC** (`weekly_fetch.py`): refresh main benchmark prices
  (one bar per ticker per ISO week), PGR dividends, PGR quarterly
  fundamentals and FRED macro data; seed `split_history` from
  `config/splits.py`; rebuild monthly relative-return targets.
- **Wednesday 16:00 UTC** (`weekly_fetch.py --dividend-refresh`, also
  `workflow_dispatch` input `dividend_refresh`): budget-aware DIVIDENDS
  refresh for PGR and the ETF benchmarks. A ticker is re-fetched once its
  last dividend fetch is 27+ days old (6+ days for monthly payers), oldest
  first, within the AV calls left that day minus 2. Then targets are rebuilt.

Both runs skip the target rebuild if PGR or a benchmark has a weekly close
ratio outside [0.6, 1.7] with no split row within 7 days (a split missing
from `config/splits.py`).

Outputs:

- `data/pgr_financials.db`
- job summary with key counts and latest dates
- a final "Check price, split and dividend integrity" step
  (`scripts/check_data_integrity.py`), run after the DB commit. It fails the
  run on an unexplained price jump or a duplicate ticker-week bar and, after a
  dividend refresh, on any stale dividend feed.

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
- `results/monthly_decisions/<YYYY-MM>/benchmark_quality.csv`
- `results/monthly_decisions/<YYYY-MM>/consensus_shadow.csv`
- `results/monthly_decisions/<YYYY-MM>/classification_shadow.csv`
- `results/monthly_decisions/<YYYY-MM>/decision_overlays.csv`
- `results/monthly_decisions/<YYYY-MM>/dashboard.html`
- `results/monthly_decisions/<YYYY-MM>/monthly_summary.json`
- `results/monthly_decisions/<YYYY-MM>/run_manifest.json`
- `results/monthly_decisions/decision_log.md`
- `results/monthly_decisions/classification_shadow_history.csv`
- `results/monthly_decisions/ta_shadow_variant_history.csv`

Notes:

- the dashboard is now represented both as a committed static monthly snapshot
  and as a richer local Streamlit viewer over the same artifacts
- the static dashboard snapshot is the primary shareable UI surface; the
  Streamlit app remains local/operator-facing
- classifier overlays remain shadow-only in this workflow and do not alter the
  live recommendation mode or sell percentage
- reporting-only TA shadow variants remain monitoring-only and are checked by
  `scripts/verify_monthly_outputs.py`
- monthly postconditions are verified by
  `python scripts/verify_monthly_outputs.py --summary-path workflow_summary.md`
- the email step is non-fatal by design and should not block report generation

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
- unit and integration tests
- smoke runs for major production entrypoints
- migration and fresh-temp-DB checks

## Concurrency Policy

Production workflows that can mutate the committed database or monthly artifacts
use workflow-level concurrency groups so overlapping runs do not step on each
other.
