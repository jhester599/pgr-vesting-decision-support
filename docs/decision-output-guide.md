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

The preferred order remains:

- loss lots first
- LTCG gain lots next
- avoid STCG gains unless the model edge is unusually strong

## Diversification-Aware Redeploy Guidance

v11 research adds a second layer of recommendation logic:

- some funds are useful as forecast benchmarks
- not all of them are good destinations for capital leaving a concentrated PGR
  position

When the project discusses redeployment, it should prefer buckets that reduce
single-stock concentration:

- broad US equity
- international equity
- fixed income
- real assets

Funds that remain too correlated with PGR, such as `VFH` or `KIE`, may still
appear as contextual benchmarks, but they should not normally be presented as
preferred redeployment destinations.

## v12 Shadow Memos

The v12 study adds side-by-side shadow memos under `results/v12/dry_runs/`.

Each memo compares:

- the live production monthly stack
- the simpler diversification-first baseline selected in v11

The shadow memos are meant to answer a practical question:

- would the simpler baseline produce steadier and more useful monthly guidance,
  even before any model-stack promotion?

They are review artifacts only. They do not change the live workflow on their
own.

## v13 Recommendation-Layer Pilot

v13 brings part of that shadow-study output into the production-facing monthly
communication.

The report and email can now include:

- `Existing Holdings Guidance`
- `Redeploy Guidance`
- `Simple-Baseline Cross-Check`
- `Suggested Redeploy Portfolio`

Under the current `shadow_promoted` default:

- the simpler diversification-first baseline determines the official recommendation layer
- the live model stack is shown as a cross-check
- the report explicitly distinguishes:
  - what to do with the next vest
  - what to do with already-held shares
  - where sold exposure should go if redeployed

## v27 Suggested Redeploy Portfolio

v27 adds a concrete buy list to the monthly workflow.

The intent is to answer:

- if the user sells some PGR shares, what should the proceeds be used for?

The live monthly process now:

- starts from a stable high-equity base portfolio
- keeps stock exposure above `90%` in normal months
- uses bounded monthly tilts from current projected relative returns and
  confidence
- limits the recommendation to a small investable set

The live investable redeploy universe is:

- `VOO`
- `VGT`
- `SCHD`
- `VXUS`
- `VWO`
- `BND`

Important distinction:

- this is the monthly buy recommendation universe
- it is narrower than the broader forecast benchmark universe

That split allows the project to keep contextual forecasting funds such as
`VFH` or `KIE` without presenting them as preferred destinations for capital
leaving a concentrated PGR position.

## Diagnostic Report

The diagnostic report is the technical appendix. It includes:

- aggregate model health
- per-benchmark diagnostics
- calibration notes
- CPCV / observation-to-feature context

Use the diagnostic report to understand why the recommendation mode landed where
it did.
