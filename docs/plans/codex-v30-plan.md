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
- `v30.2` - Black-Litterman fallback visibility in the monthly report
- `v30.3` - data-freshness preflight checks

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
