# V168 Closeout And Handoff

Created: 2026-04-18

## Completed Block

`v168` fixes the PGR monthly EDGAR freshness warning observed in the April 18
monthly dry run.

## Root Cause

The old freshness check treated `pgr_edgar_monthly.month_end` as a simple
maximum-age feed. That made February 28 data look stale on April 18, even
though March 31 monthly 8-K data is not required until PGR has had its normal
filing window and the scheduled primary/fallback 8-K workflows have run.

A dry-run of `scripts/edgar_8k_fetcher.py --backfill-years 1 --dry-run`
confirmed the parser still works and no March filing was available as of
April 18.

## What Changed

- Added `DATA_FRESHNESS_PGR_EDGAR_FILING_GRACE_DAYS = 25`.
- `check_data_freshness` now computes the expected PGR monthly EDGAR
  `month_end` from the reference date and the 25-day filing grace window.
- The freshness report includes:
  - `expected_month_end`
  - `limit_label`
- `recommendation.md` renders the richer limit label, such as
  `25-day filing grace`.
- The monthly 8-K workflow now enforces the same calendar-aware freshness rule
  after fetch attempts and includes expected month/status in the workflow
  summary.

## Verification

Validated with:

```bash
python -m pytest tests/test_data_freshness.py -q --tb=short
python -m pytest tests/test_workflow_contracts.py tests/test_data_freshness.py -q --tb=short
python -m ruff check src/database/db_client.py src/reporting/decision_rendering.py tests/test_data_freshness.py tests/test_workflow_contracts.py config/api.py
python scripts/edgar_8k_fetcher.py --backfill-years 1 --dry-run
python scripts/monthly_decision.py --as-of 2026-04-18 --dry-run --skip-fred
```

The final monthly dry run no longer emits the
`PGR monthly EDGAR is stale` warning. Generated dry-run database and monthly
artifact churn was restored.

## Production Boundary

No model, feature, recommendation, sell percentage, or classifier gate behavior
changed.

## Next Direction

Let the April 20/25 monthly 8-K workflow run normally. On and after the grace
deadline, the workflow will fail if the expected prior-month PGR monthly row is
still missing, which is the right time to investigate parser or filing changes.
