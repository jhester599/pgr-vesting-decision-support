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
- scheduled on the 20th (primary) and 25th (fallback) at 14:00 UTC; its
  completion triggers `monthly_decision.yml`
- sets `EDGAR_USER_AGENT` (name and contact e-mail); the fetcher fails if it
  is unset or the old placeholder (review 2026-09-25, F26)

Outputs:

- `data/pgr_financials.db`
- job summary with latest EDGAR row metadata

### `monthly_decision.yml`

Purpose:

- generate the monthly recommendation and diagnostic artifacts
- update the decision log
- optionally send the decision email

Triggers (review 2026-09-25, F26):

- `workflow_run` on completion of `monthly_8k_fetch.yml`, so the decision
  runs after that month's 8-K data is in the DB;
- crons on the 21st and 22nd (15:00 UTC) as fallbacks;
- `workflow_dispatch` (and the drift retrain trigger).

`scripts/monthly_decision.py` writes `generated=true` to the step output only
when it produced a new production report. Verify, charts, commit and email
run only then, so the fallback runs (which find the month's report and exit
with `generated=false`) no longer re-send the email or recommit the charts.
From the 20th the as-of date is the last business day on or before the 20th,
never later than the run date; an unknown `RECOMMENDATION_LAYER_MODE` fails
the run.

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
- the 12 recurring capital-return charts named in the workflow's
  `RESEARCH_CHARTS` list, under `results/research/pgr_*.png`, from
  `scripts/repurchase_timeseries_charts.py` and
  `scripts/capital_return_charts.py`

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
- the chart step fails if either chart script errors or any chart in
  `RESEARCH_CHARTS` is not rewritten. The decision artifacts, DB and email
  still go out (their steps run after a chart failure), but the charts are
  not committed and the job ends red. Both scripts open the DB read-only and
  take `--db-path` / `--out-dir`

## Historical / Manual Workflows

The repository also retains one-off bootstrap workflows:

- `initial_fetch_prices.yml`
- `initial_fetch_dividends.yml`
- `peer_bootstrap.yml`
- `post_initial_bootstrap.yml`

They are dispatch-only (review 2026-09-25, F26): their old yearly crons would
have fired again every March. They are retained for historical recovery and
manual bootstrap scenarios, not the steady-state operating loop.

## CI Workflow

`ci.yml` runs:

- lint checks
- unit and integration tests
- smoke runs for major production entrypoints, each through
  `scripts/ci_offline_smoke.py`: every socket connection is refused and the
  SEC submissions index is served from a canned empty response, so no smoke
  run reaches Alpha Vantage, FRED or EDGAR
- migration and fresh-temp-DB checks

## Concurrency Policy

Every workflow that commits `data/pgr_financials.db` uses one concurrency
group, `db-writer`, with `cancel-in-progress: false` (review 2026-09-25, F26):
`weekly_data_fetch`, `peer_data_fetch`, `monthly_8k_fetch`,
`monthly_decision` and the four bootstrap workflows. Runs queue instead of
racing to push a binary DB. `tests/test_ops_wp8.py` checks the list.

GitHub keeps at most one *pending* run per group: if a third writer queues
while one runs and one waits, the waiting run is cancelled. The schedules are
spread to keep that rare (weekly Friday 22:00 and Wednesday 16:00 UTC, peers
Sunday 04:00 UTC, 8-K on the 20th/25th at 14:00 UTC, the decision after the
8-K run). Re-run a cancelled job by hand.
