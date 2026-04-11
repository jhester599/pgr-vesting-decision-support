# Decision Output Guide

## Monthly Output Files

Each monthly run writes a folder under `results/monthly_decisions/<YYYY-MM>/`.

Expected files:

- `recommendation.md`
- `diagnostic.md`
- `signals.csv`
- `benchmark_quality.csv`
- `consensus_shadow.csv`
- `dashboard.html`
- `monthly_summary.json`
- `run_manifest.json`

## Current Recommendation Surface

The live monthly recommendation currently combines:

- the promoted quality-weighted consensus
- the recommendation mode gate:
  - `ACTIONABLE`
  - `MONITORING-ONLY`
  - `DEFER-TO-TAX-DEFAULT`

Interpretation:

- `ACTIONABLE`: model quality is strong enough to support a prediction-led
  recommendation
- `MONITORING-ONLY`: signal may be informative, but not strong enough to drive
  a vest action
- `DEFER-TO-TAX-DEFAULT`: follow the default diversification and tax-discipline
  rule rather than a prediction-led deviation

## Structured Monthly Summary

`monthly_summary.json` is now the machine-readable top-level summary artifact.

It exists so that:

- the email
- the dashboard
- future automation or notification surfaces

can consume the current monthly decision state without scraping markdown for
headline fields.

## Consensus Shadow Diagnostic

`consensus_shadow.csv` still preserves the live-vs-equal-weight comparison.

That comparison is now diagnostic-only. It remains useful for:

- promotion auditing
- stability review
- governance checks when the live path changes

It is no longer rendered as a primary recommendation section in the main
monthly memo.

## Next Vest Section

The recommendation report surfaces:

- the next relevant vest date
- RSU tranche type
- suggested action for the new vest
- provisional tax-scenario comparison
- Monte Carlo tax sensitivity for the LTCG-vs-sell-now choice

## Existing-Holdings Guidance

The report and email can also summarize how to think about already-held shares
using the lot file in `data/processed/position_lots.csv`.

The preferred order remains:

- loss lots first
- LTCG gain lots next
- avoid STCG gains unless the model edge is unusually strong

## Diversification-Aware Redeploy Guidance

The monthly output separates:

- the broader forecast benchmark universe
- the narrower investable redeploy universe

When the project discusses redeployment, it should prefer buckets that reduce
single-stock concentration:

- broad US equity
- international equity
- fixed income
- real assets

Funds that remain too correlated with PGR may still appear as contextual or
forecast-only benchmarks, but they should not normally be presented as preferred
destinations for sold exposure.

## Benchmark Quality Diagnostics

`benchmark_quality.csv` is the monthly per-benchmark quality export.

It currently contains metrics such as:

- `oos_r2`
- `nw_ic`
- `hit_rate`
- `cw_t_stat`
- `cw_p_value`

This file is intended for:

- operator review
- later weighting and gating research
- consistency checks between the report and the underlying benchmark-level data

## Diagnostic Report

The diagnostic report is the technical appendix. It includes:

- aggregate model health
- pooled Clark-West results
- per-benchmark diagnostics
- calibration notes
- CPCV and observation-to-feature context

Use the diagnostic report to understand why the recommendation mode landed where
it did.

## Local Dashboard

The repo also includes:

- a static monthly dashboard snapshot at `results/monthly_decisions/<YYYY-MM>/dashboard.html`
- a local Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The static HTML snapshot is the linkable lightweight surface.

The Streamlit app remains a richer local viewer over the same committed monthly
artifacts.
