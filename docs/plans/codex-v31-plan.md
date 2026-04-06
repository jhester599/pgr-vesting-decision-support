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
