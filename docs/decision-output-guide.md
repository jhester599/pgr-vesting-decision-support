# Decision Output Guide

## Monthly Output Files

Each monthly run writes a folder under `results/monthly_decisions/<YYYY-MM>/`.

Expected files:

- `recommendation.md`
- `diagnostic.md`
- `signals.csv`
- `run_manifest.json`

## Recommendation Modes

The monthly recommendation can be one of:

- `ACTIONABLE`
- `MONITORING-ONLY`
- `DEFER-TO-TAX-DEFAULT`

Interpretation:

- `ACTIONABLE`: model quality is strong enough to support a prediction-led
  recommendation.
- `MONITORING-ONLY`: signal may be informative, but not strong enough to drive
  a vest action.
- `DEFER-TO-TAX-DEFAULT`: follow the default diversification / tax-discipline
  rule rather than a prediction-led deviation.

## Next Vest Section

The recommendation report surfaces:

- the next relevant vest date
- RSU tranche type
- suggested action for the new vest
- provisional tax-scenario comparison

## Existing-Holdings Guidance

The email and report can also summarize how to think about already-held shares
using the lot file in `data/processed/position_lots.csv`.

The logic is intended as guidance, not an automated trade instruction.

## Diagnostic Report

The diagnostic report is the technical appendix. It includes:

- aggregate model health
- per-benchmark diagnostics
- calibration notes
- CPCV / observation-to-feature context

Use the diagnostic report to understand why the recommendation mode landed where
it did.
