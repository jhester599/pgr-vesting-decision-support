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
- `v30.8` - add operational logging to monthly decision fallbacks
- `v30.9` - migrate peer fetch entrypoint to structured logging
- `v30.10` - rewrite README as a landing page
- `v30.11` - align script-level EDGAR fetcher logging and headers
- `v30.12` - add logging fallbacks to initial fetch
- `v30.13` - add logging fallbacks to bootstrap
- `v30.14` - add logging fallback to v1 migration script
- `v30.15` - add exception-context logging to fred loader
- `v30.16` - add logging for silent WFO benchmark/model skips
- `v30.17` - add exception-context logging to BL fallback paths
- `v30.18` - log run-manifest git metadata fallback
- `v30.19` - add backtest fallback logging
- `v30.20` - log optional synthetic feature fallback paths
- `v30.21` - log research evaluation NW fallback
- `v30.22` - log CPCV recombination fallback
- `v30.23` - log fracdiff candidate fallback paths
- `v30.24` - log plot date fallback

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

## v30.8 Scope

`v30.8` extends the logging scaffold into the monthly workflow's operational
paths:

- configure shared logging in `scripts/monthly_decision.py`
- route FRED, CPCV, cross-check, and freshness fallback messaging through
  structured logging
- add focused tests that assert exception context is preserved

## v30.9 Scope

`v30.9` keeps the production entry points aligned by migrating
`scripts/peer_fetch.py` to the shared logging scaffold and refreshing its
dry-run tests to assert on logged output instead of captured stdout.

## v30.10 Scope

`v30.10` turns the repository root back into an onboarding page:

- replace the version-history-heavy `README.md` with a concise landing page
- link out to `docs/architecture.md`, `docs/operations-runbook.md`,
  `docs/changelog.md`, and `ROADMAP.md`
- keep detailed version history in dedicated docs instead of the repo root

## v30.11 Scope

`v30.11` finishes a small production-script consistency pass:

- route `scripts/edgar_8k_fetcher.py` through the shared logging setup
- reuse the central env-backed EDGAR header builder for script-side HTTP calls
- add focused tests around the script-level `_get()` helper

## v30.12 Scope

`v30.12` hardens the one-time bootstrap script's fallback paths:

- migrate `scripts/initial_fetch.py` to the shared logging scaffold
- preserve exception context for FRED and loader-level broad catches
- add focused tests for those fallback logs

## v30.13 Scope

`v30.13` continues the Tier 3.3 fallback sweep in `scripts/bootstrap.py`:

- migrate bootstrap orchestration output to the shared logging scaffold
- preserve exception context when the delegated monthly decision run fails
- add focused tests for the bootstrap fallback path

## v30.14 Scope

`v30.14` keeps the fallback sweep moving in the one-time migration helper:

- migrate `scripts/migrate_v1_to_v2.py` to the shared logging scaffold
- preserve exception context when the legacy EDGAR loader import/run fails
- add focused tests for that migration fallback

## v30.15 Scope

`v30.15` extends the same fallback traceability into core ingestion:

- add module-level logging to `src/ingestion/fred_loader.py`
- preserve exception context when one FRED series fails during a multi-series fetch
- add focused tests that the failed series is logged and the remaining series still load

## v30.16 Scope

`v30.16` makes silent benchmark/model failures visible in the core WFO runner:

- add module-level logging to `src/models/multi_benchmark_wfo.py`
- log skipped ensemble members when one model fails for a benchmark
- log live prediction failures while preserving the existing continue-on-error behavior

## v30.17 Scope

`v30.17` extends the same observability into the BL allocator:

- add module-level logging to `src/portfolio/black_litterman.py`
- log view-prediction extraction failures while still building remaining views
- log optimization failures before falling back to equal weights

## v30.18 Scope

`v30.18` closes a small operational blind spot in manifest generation:

- add module-level logging to `src/reporting/run_manifest.py`
- log git metadata lookup failures before falling back to `"unknown"`
- add focused tests for the fallback path so the manifest contract stays stable

## v30.19 Scope

`v30.19` extends the observability pass into historical backtesting:

- add module-level logging to `src/backtest/backtest_engine.py`
- log live-prediction failures before skipping a backtest cell
- log proxy-fill estimation failures before defaulting to `0.0`

## v30.20 Scope

`v30.20` carries the same pattern into DB-backed feature construction:

- add fallback logging to the optional synthetic relative-feature blocks in `src/processing/feature_engineering.py`
- preserve the existing fail-closed behavior when auxiliary benchmark prices are missing or invalid
- add focused pytest coverage for a logged synthetic-feature failure

## v30.21 Scope

`v30.21` tightens one remaining silent fallback in research evaluation:

- add module-level logging to `src/research/evaluation.py`
- log Newey-West summary failures before returning `NaN` diagnostics
- extend the benchmark-suite tests to cover the logged fallback path

## v30.22 Scope

`v30.22` makes one remaining CPCV diagnostic fallback visible:

- add module-level logging to `src/models/wfo_engine.py`
- log recombined-path failures before returning empty CPCV path diagnostics
- extend the CPCV tests to cover the logged fallback path

## v30.23 Scope

`v30.23` makes fracdiff candidate skips observable during research runs:

- log ADF-evaluation failures in `src/processing/feature_engineering.py`
- log correlation-evaluation failures while preserving the existing continue-on-error grid search
- extend the fracdiff tests to cover a logged candidate skip

## v30.24 Scope

`v30.24` closes one last silent plotting fallback:

- add module-level logging to `src/visualization/plots.py`
- log fold-date construction failures before falling back to positional plotting
- extend the reporting tests to cover the logged fallback path
