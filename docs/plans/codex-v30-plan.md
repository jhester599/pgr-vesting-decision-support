# v30 Enhancement Sequence

Created: 2026-04-05

## Goal

Execute the 2026-04-05 peer-review enhancements in small, low-risk increments
with frequent commits.

## Versioning Approach

The current repo closes out at `v29`, so this sequence uses compact follow-on
version tags:

- `v30.0` - EDGAR identity and config hardening
- `v30.1` - stale documentation header cleanup
- `v30.2` - data-freshness preflight checks
- `v30.3` - Black-Litterman fallback visibility, only if BL is promoted into the live monthly path
- `v30.4` - shared retry/backoff helper for core provider clients
- `v30.5` - lightweight monthly workflow integration test
- `v30.6` - extend retry helper into batch AV loaders
- `v30.7` - add production logging scaffold for weekly fetch

This keeps the work aligned with the existing closeout cadence while avoiding a
single oversized `v30` batch.

## v30.0 Scope

`v30.0` handles the smallest production-facing Tier 1 fix that still changes
runtime behavior:

- move the EDGAR `User-Agent` identity out of hardcoded source strings
- source the value from `EDGAR_USER_AGENT` in `.env`
- keep a generic fallback so tests and clean environments still import safely
- route both EDGAR ingestion paths through the same header builder
- add focused pytest coverage for the env override and request header usage

## Notes

- Tier 1.1 from the peer review is already satisfied on the cloud baseline:
  `tests/test_multi_ticker_loader.py` passes unchanged on `origin/master`.
- The local virtualenv is currently missing `lxml`, which causes
  `tests/test_edgar_8k_fetcher.py` to fail for environment reasons unrelated to
  `v30.0`. Handle that separately from this config-hardening step.
- The peer-review Black-Litterman fallback item is real in
  `src/portfolio/black_litterman.py`, but the current monthly workflow still
  renders redeploy guidance from the v27 heuristic portfolio builder rather
  than a live BL optimizer. That makes BL fallback visibility a larger workflow
  change than a quick report tweak on this baseline.

## v30.2 Scope

`v30.2` adds operational freshness checks around the existing DB snapshot:

- evaluate latest price, FRED, and monthly EDGAR dates against explicit
  thresholds
- print preflight freshness warnings at monthly runtime
- surface the same warnings in `recommendation.md`
- carry freshness warnings into the run manifest

## v30.4 Scope

`v30.4` adds a small shared retry-enabled HTTP session for transient failures:

- central retry/backoff helper in `src/ingestion/http_utils.py`
- wire the helper into the core AV, FRED, and EDGAR clients
- add focused regression tests so the new session-based request path is covered

## v30.5 Scope

`v30.5` adds one thin end-to-end monthly pipeline test:

- call `scripts.monthly_decision.main()` directly
- stub the expensive modeling and plotting steps
- verify the workflow still produces `recommendation.md`, `signals.csv`,
  `diagnostic.md`, and `run_manifest.json`

## v30.6 Scope

`v30.6` extends the shared retry helper into the Alpha Vantage batch loaders:

- `src/ingestion/multi_ticker_loader.py`
- `src/ingestion/multi_dividend_loader.py`
- refresh the existing batch-loader tests to mock the retry-session path

## PR Checkpoint

After `v30.6`, the branch is large enough to open a draft PR without waiting
for the entire peer-review backlog:

- the changes form a coherent reliability slice around ingestion and monthly
  operations
- the diff is still compact enough for review before riskier workflow changes
- subsequent enhancements can continue as small follow-up commits on the same
  branch or a successor branch if the scope shifts

## v30.7 Scope

`v30.7` starts the logging migration with one production entry point:

- add `src/logging_config.py` with a shared production log format
- migrate `scripts/weekly_fetch.py` from `print()` to logging calls
- add exception-context coverage for the weekly FRED fallback path
