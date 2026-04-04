# Operations Runbook

## Normal Workflow Expectations

- `weekly_data_fetch.yml`
  - should update main prices, dividends, fundamentals, macro data, and
    relative-return targets
- `peer_data_fetch.yml`
  - should update peer prices and dividends
- `monthly_8k_fetch.yml`
  - should update `pgr_edgar_monthly`
- `monthly_decision.yml`
  - should write a new monthly folder under `results/monthly_decisions/`
  - should append exactly one row to `decision_log.md`
  - should produce `run_manifest.json`

## Local Dry-Run Commands

```bash
python scripts/weekly_fetch.py --dry-run --skip-fred
python scripts/peer_fetch.py --dry-run
python scripts/edgar_8k_fetcher.py --dry-run
python scripts/monthly_decision.py --as-of 2026-04-02 --dry-run --skip-fred
```

## Rebuilding / Validating the Database

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

Historical CSV backfill:

```bash
python scripts/edgar_8k_fetcher.py --load-from-csv
```

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

After a monthly run, verify:

- monthly folder exists
- `recommendation.md` exists
- `diagnostic.md` exists
- `signals.csv` exists
- `run_manifest.json` exists
- `decision_log.md` contains one row for the month

## Recovery Guidance

- If a fetch workflow fails, re-run the workflow manually with the same inputs.
- If the DB is stale, inspect DB health and ingestion metadata before forcing
  any backfill.
- If a migration fails, stop and inspect the failing SQL rather than manually
  editing the DB in place.
