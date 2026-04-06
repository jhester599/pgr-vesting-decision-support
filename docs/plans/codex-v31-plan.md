# v31 Enhancement Sequence

Created: 2026-04-05

## Goal

Continue the 2026-04-05 peer-review follow-up with a tighter focus on
conformal-coverage monitoring and adjacent monthly-diagnostic improvements.

## Versioning Approach

The `v30` sequence landed the first large reliability and observability pass.
This follow-on sequence starts at `v31.0` for the next chunk of production and
diagnostic enhancements.

- `v31.0` - add historical conformal coverage backtest helper
- `v31.1` - surface trailing conformal coverage in monthly diagnostics
- `v31.2` - add rolling drift-monitor helper for monthly model health
- `v31.3` - add model performance log schema and DB helpers
- `v31.4` - persist monthly model health and surface drift in the report

## v31.0 Scope

`v31.0` establishes the reusable metric needed for peer-review Tier 2.5:

- add a sequential OOS conformal-coverage backtest helper in `src/models/conformal.py`
- report both full-history and trailing-window empirical coverage
- add focused pytest coverage for the helper contract and edge cases

## v31.1 Scope

`v31.1` wires the new conformal monitoring metric into the monthly workflow:

- add a monthly trailing-coverage summary derived from per-benchmark conformal diagnostics
- surface trailing conformal coverage in `diagnostic.md`
- add a run-manifest warning when trailing realized coverage drifts materially from nominal
- extend report and monthly end-to-end pytest coverage for the new diagnostic path

## v31.2 Scope

`v31.2` starts the peer-review Tier 2.4 work with reusable monitoring logic:

- add a drift-monitor helper for rolling IC, hit-rate, and ECE tracking
- compute the latest IC-breach streak against `DIAG_MIN_IC`
- add focused pytest coverage for rolling windows, sorting, validation, and drift-flag behavior

## v31.3 Scope

`v31.3` starts persisting the monthly model-health history needed for Tier 2.4:

- add a `model_performance_log` migration/table for aggregate OOS, calibration, and conformal diagnostics
- add DB upsert/query helpers for the monthly model-health log
- include the new table in operational snapshots and schema tests

## v31.4 Scope

`v31.4` turns the Tier 2.4 scaffolding into a live monthly workflow path:

- persist the current monthly model-health snapshot after each decision run
- derive the latest rolling drift summary from the DB-backed history
- surface a `Model Health` section in `recommendation.md`
- extend the monthly end-to-end test to assert both report rendering and DB persistence
