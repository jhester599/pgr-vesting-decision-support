# Operations Runbook

## Normal Workflow Expectations

- `weekly_data_fetch.yml`
  - Friday: should update main prices, PGR dividends, fundamentals, macro
    data, and relative-return targets
  - Wednesday (`--dividend-refresh`): should update ETF and PGR dividends that
    are due, within the AV budget
  - its last step (`scripts/check_data_integrity.py`) must be green: no
    unexplained price jumps, one bar per ticker-week, fresh dividends
- `peer_data_fetch.yml`
  - should update peer prices and dividends
- `monthly_8k_fetch.yml`
  - should update `pgr_edgar_monthly`
- `monthly_decision.yml`
  - should write a new monthly folder under `results/monthly_decisions/`
  - should append exactly one row to `decision_log.md`
  - should produce a `run_manifest.json`
  - should produce `benchmark_quality.csv`, `consensus_shadow.csv`,
    `dashboard.html`, and `monthly_summary.json`
  - should rewrite every chart in its `RESEARCH_CHARTS` list; a red
    "Regenerate research charts" step means the charts were not committed.
    Reproduce locally against a DB copy with
    `python scripts/repurchase_timeseries_charts.py --db-path <copy> --out-dir <dir>`
    (and the same for `scripts/capital_return_charts.py`)

## Local Dry-Run Commands

```bash
python scripts/weekly_fetch.py --dry-run --skip-fred
python scripts/peer_fetch.py --dry-run
python scripts/edgar_8k_fetcher.py --dry-run
python scripts/monthly_decision.py --as-of 2026-04-11 --dry-run --skip-fred
```

`weekly_fetch.py --dry-run` and `monthly_decision.py --dry-run` are read-only:
they open the DB with `mode=ro`, skip migrations, API-log rows, split seeding,
the model-health snapshot, the retrain log, `decision_log.md` and the shadow
ledgers, and write monthly artifacts to the gitignored
`results/dry_run/monthly_decisions/YYYY-MM/` (manifest `dry_run: true`).
`edgar_8k_fetcher.py --dry-run` also opens the DB read-only and skips
migrations, but it calls SEC EDGAR (set `EDGAR_USER_AGENT`; add
`--cache-dir data/raw/edgar_8k_cache` to reuse cached filings).
`peer_fetch.py --dry-run` is not yet read-only; run it against a DB copy.

Optional local dashboard check:

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

## Rebuilding / Validating the Database

Changes to the committed DB go through ordered migration files (`.sql`, or
`.py` with an `upgrade(conn)` function) in `src/database/migrations/`. Apply them to a copy first, check the diff, then
commit the file the script produced:

```bash
cp data/pgr_financials.db /tmp/pgr_copy.db
python scripts/apply_db_migrations.py --db /tmp/pgr_copy.db   # applies pending, then finalizes
```

Before `git add data/pgr_financials.db`, every workflow runs
`python scripts/finalize_db.py`, which checkpoints the WAL
(`PRAGMA wal_checkpoint(TRUNCATE)`) and sets `journal_mode=DELETE` so the
committed file is self-contained. `*.db-wal` / `*.db-shm` are gitignored.

### Splits and relative-return targets

`config/splits.py` (`KNOWN_SPLITS`) is the only split list. When the
integrity check reports an unexplained price jump:

1. Verify the split from the issuer notice. Optionally confirm it with
   `python scripts/detect_splits.py --tickers <T> --db /tmp/pgr_copy.db`
   (one AV call per ticker, cached for the day, budget-capped).
2. Add the row to `KNOWN_SPLITS` with its `evidence`.
3. Rebuild the targets on a copy, review the diff, then commit the copy:

```bash
cp data/pgr_financials.db /tmp/pgr_copy.db
python scripts/rebuild_relative_returns.py --db /tmp/pgr_copy.db \
    --report /tmp/rebuild.md --rows-csv /tmp/rebuild_rows.csv --attribute
python scripts/check_data_integrity.py --db /tmp/pgr_copy.db
```

The weekly job also re-seeds splits and rebuilds (replaces) every target row
each run, so a registry fix reaches the committed DB on the next Friday.

Fresh schema init:

```bash
python - <<'PY'
import sqlite3
from src.database import db_client
conn = sqlite3.connect(':memory:')
db_client.initialize_schema(conn)
print('schema ok')
PY
```

Historical CSV backfill (seeds an empty or partial DB):

```bash
python scripts/edgar_8k_fetcher.py --load-from-csv
```

It inserts only months that `pgr_edgar_monthly` does not have, so it never
overwrites rows written by the monthly fetch; re-running it changes nothing.
Derived fields (YoY growth, Gainshare, PIF totals) are then recomputed over the
whole table.

Rebuilding the EDGAR tables from the filings (review 2026-09-25 step 3b; only
when the parser changes, and always on a copy first):

```bash
cp data/pgr_financials.db /tmp/repair.db
EDGAR_USER_AGENT="Name email@example.com" python scripts/repair_edgar_history.py \
    --db /tmp/repair.db --diff-csv /tmp/edgar_cell_diff.csv \
    --export-csv data/processed/pgr_edgar_cache.csv
python scripts/generate_edgar_data_dictionary.py --db /tmp/repair.db
PGR_EDGAR_INTEGRITY_DB=/tmp/repair.db python -m pytest tests/test_pgr_edgar_integrity.py
```

The script re-fetches every monthly release since August 2004 (cached in
`data/raw/edgar_8k_cache`, at most 4 requests per second), records every value
in the append-only `pgr_edgar_monthly_raw` table under the current
`PARSER_VERSION`, rebuilds `pgr_edgar_monthly` and `pgr_fundamentals_quarterly`,
writes a cell-level diff and prints the row-level validation results. Review the
diff, then finalize the copy (`scripts/finalize_db.py --db /tmp/repair.db`) and
copy it over the committed DB. Bump `PARSER_VERSION` in
`scripts/edgar_8k_fetcher.py` whenever a parser change can change a value.

Health check:

```bash
python - <<'PY'
import config
from src.database import db_client
conn = db_client.get_connection(config.DB_PATH)
print(db_client.build_db_health_report(conn))
conn.close()
PY
```

## Monthly Report Validation

After a monthly run, use the reusable verifier:

```bash
python scripts/verify_monthly_outputs.py --summary-path workflow_summary.md
```

For a specific as-of date:

```bash
python scripts/verify_monthly_outputs.py --as-of 2026-04-19 --summary-path workflow_summary.md
```

The verifier checks:

- monthly folder exists
- all required monthly files exist
- `monthly_summary.json` and `run_manifest.json` are readable
- data freshness is `OK`
- reporting-only TA shadow variants are present in `classification_shadow.csv`
- matching TA rows exist in `ta_shadow_variant_history.csv`

Recommended spot checks:

- `run_manifest.json` lists all expected outputs
- `monthly_summary.json` matches the top-level recommendation shown in
  `recommendation.md`
- `diagnostic.md` includes Clark-West and per-benchmark quality sections
- `decision_log.md` contains one row for the month

## Recovery Guidance

- If a fetch workflow fails, re-run the workflow manually with the same inputs.
- If the DB is stale, inspect DB health and ingestion metadata before forcing
  any backfill.
- If a migration fails, stop and inspect the failing SQL rather than manually
  editing the DB in place.
- If the monthly report is missing the new CSV artifacts, treat that as a
  production regression in the reporting path rather than a harmless omission.
- If `scripts/verify_monthly_outputs.py` fails, investigate the named missing
  artifact, stale feed, or TA ledger row before trusting the monthly package.
