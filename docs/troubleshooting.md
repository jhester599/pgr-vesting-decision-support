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
- `benchmark_quality.csv`, `consensus_shadow.csv`, or `monthly_summary.json`
  missing

Action:

- run `scripts/monthly_decision.py --dry-run --skip-fred`
- inspect model-quality gate output and DB health
- confirm the target month folder does not already exist unless intentionally rerunning
- treat missing CSV artifacts as a reporting-path regression, not as optional output
- treat a missing `monthly_summary.json` as a contract regression for email,
  dashboard, and future automation consumers

## Email Rendering Drift

Symptoms:

- email sends, but the rendered HTML does not reflect the latest report sections
- current report includes new sections, but the email still uses older wording or layout

Action:

- compare `recommendation.md` to `src/reporting/email_sender.py`
- regenerate a dry-run email locally
- update parser and email tests together whenever report headings or table
  structures change

## Dashboard Drift

Symptoms:

- dashboard opens, but metrics or sections do not match the live monthly report
- new monthly artifacts are present, but the dashboard does not surface them

Action:

- compare `dashboard/app.py` against the current monthly artifact schema
- prefer structured CSV/manifest inputs over regex-only parsing from markdown
- add or refresh dashboard smoke tests if the dashboard becomes a supported surface

## Workflow Push / Commit Issues

Symptoms:

- workflow finishes logic but does not push changes

Action:

- confirm `permissions: contents: write`
- confirm the workflow staged the expected files only
- inspect the no-op commit guard in the workflow logs
