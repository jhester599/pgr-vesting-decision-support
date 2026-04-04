# Troubleshooting

## Missing Secrets

Symptoms:

- workflow fails during fetch or email steps

Check:

- `AV_API_KEY`
- `FRED_API_KEY`
- `SMTP_SERVER`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `EMAIL_FROM`
- `EMAIL_TO`

## Stale Database

Symptoms:

- monthly report warns that the DB is behind
- latest dates in summaries do not advance

Check:

- DB health report
- ingestion metadata
- latest table dates in workflow summaries

## Migration Failure

Symptoms:

- workflow or local run fails during schema initialization

Action:

- inspect `schema_migrations`
- inspect the failing migration file
- do not manually patch tables unless you are explicitly performing a recovery

## EDGAR Fetch Problems

Symptoms:

- no new monthly 8-K row
- parser warnings

Action:

- confirm the filing exists on SEC EDGAR
- run the fetcher locally in `--dry-run`
- compare live parser output to the committed CSV baseline

## Monthly Decision Generation Failure

Symptoms:

- recommendation folder missing
- missing `run_manifest.json`
- `decision_log.md` not updated

Action:

- run `scripts/monthly_decision.py --dry-run --skip-fred`
- inspect model-quality gate output and DB health
- confirm the target month folder does not already exist unless intentionally rerunning

## Workflow Push / Commit Issues

Symptoms:

- workflow finishes logic but does not push changes

Action:

- confirm `permissions: contents: write`
- confirm the workflow staged the expected files only
- inspect the no-op commit guard in the workflow logs
