# Decision Output Guide

## Monthly Output Files

Each monthly run writes a folder under `artifacts/monthly_decisions/<YYYY-MM>/`.

Expected files:

- `recommendation.md`
- `diagnostic.md`
- `signals.csv`
- `benchmark_quality.csv`
- `consensus_shadow.csv`
- `classification_shadow.csv`
- `decision_overlays.csv`
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

### Recommendation-Mode Gates

Since the pre-v200 remediation R3 (gate contract
`chronological-readiness-2026-09-27`, recorded in `run_manifest.json` and
`monthly_summary.json`), the mode comes from five gates, evaluated in
`src/reporting/decision_rendering.py::evaluate_quality_gates`.
`ACTIONABLE` needs all five to pass; any failure gives
`DEFER-TO-TAX-DEFAULT` at 50 %; otherwise the mode is `MONITORING-ONLY`. A
missing, non-finite or unknown input fails its gate.

| Gate | Pass | Fail |
|------|------|------|
| `oos_r2`: aggregate OOS R² against each benchmark's prevailing mean of the targets realised by each forecast date (training history included) | ≥ 2 % | < 0 % |
| `mean_ic`: equal-weight mean of the per-benchmark OOS ICs | ≥ 0.07 | < 0.03 |
| `directional_skill`: one-sided Pesaran–Timmermann test that the up/down calls track the realised sign, clustered by date | p < 0.05 | p ≥ 0.10 or undefined |
| `wfo_completed`: walk-forward results for every required model (Ridge, GBT) and benchmark (the eight of `PRIMARY_FORECAST_UNIVERSE`), each with non-empty folds, finite OOS predictions, the production gap, labels realised before each test fold and outcomes realised by the as-of date, plus a finite live forecast | exactly `true` | anything else |
| `data_ready`: every live model feature finite before imputation; prices, the FRED series behind live features, PGR monthly EDGAR and dividends of PGR and the eight benchmarks fresh at the as-of date (GLD is an audited non-payer) | `true`, with no missing feature and no stale feed | anything else |

- The plain hit rate is reported but not gated: PGR beat the benchmarks in
  about 68 % of windows, so a 55 % hit-rate gate passed without skill.
- The quality-weighted consensus still sets the direction and the mean
  forecast, but the IC gate uses equal weights: the quality weights are fitted
  on the same OOS record, which inflates a weighted IC.
- Validation is walk-forward only (`TimeSeriesSplit`, 60-month window, 6-month
  test folds, gap = horizon + purge buffer = 8 months, fold-local scaling and
  imputation). The representative CPCV diagnostic was a combinatorial K-fold,
  which AGENTS.md prohibits; R3 retired it and its completeness gate.
- Readiness is judged at the as-of date: rows dated or filed later never make
  a back-dated run ready. A back-dated run's readiness is a reconstruction from
  today's DB (`readiness_basis = backdated_reconstruction`), not evidence of
  what the original decision had.
- Every surface names why a month is not `ACTIONABLE`: the executive summary
  and Confidence Snapshot in `recommendation.md` (and so the e-mail),
  `recommendation.failed_gates` / `deferral_reasons` in
  `monthly_summary.json`, `decision_gates` and a warning in
  `run_manifest.json`, the dashboard warnings and the decision-log notes.
- The `ACTIONABLE` mapping (step 6) never sells more than the 50 % default on
  an OUTPERFORM consensus: 25 % above a 15 % forecast, otherwise 50 %.
  UNDERPERFORM sells 100 %. A missing IC maps to 50 %.
- The Tax Context section reports the absolute PGR return at which holding a
  lot to its LTCG date ties selling now, `-g × (STCG − LTCG) / (1 − LTCG)`
  (−21.25 % for a lot that is all gain). The model forecast is relative to the
  benchmarks and is not compared with it. The scenario table and Monte Carlo
  assume `TAX_SCENARIO_PGR_ANNUAL_RETURN` (0 %) for PGR's price, count vested
  lots only, and move a loss sale out of the 30-day wash-sale window of every
  vest.

The portfolio rebalancer's STCG boundary warning counts days until the first
calendar LTCG day (one-year anniversary plus one day), rather than subtracting
holding days from 365. For a March 1, 2023 acquisition, February 29, 2024
shows two days remaining, March 1 shows one, and March 2 has no STCG warning.
For February 29 acquisitions the following anniversary is February 28. The
alert's age/wait settings do not change eligibility or tax rates. R5 does not
change the monthly e-mail renderer or its content/format.

All health numbers are realised-only: every OOS month's ensemble weights,
shrinkage, calibrator and conformal interval use only targets whose 6-month
window had ended by that month (`src/models/prequential.py`).

## Structured Monthly Summary

`monthly_summary.json` is now the machine-readable top-level summary artifact.

It exists so that:

- the email
- the dashboard
- future automation or notification surfaces

can consume the current monthly decision state without scraping markdown for
headline fields.

The structured summary now carries top-level values such as:

- decision headline
- hold-vs-sell label
- actionability label
- classifier shadow summary
- shadow gate overlay summary
- `model_health`: each gate's status and value, the equal-weight and
  quality-weighted mean IC, OOS R², the date-clustered pooled IC p-value, the
  hit rate against the base rate, the Pesaran–Timmermann p-value, the
  prequential shrinkage alpha, the prequential ECE, trailing conformal
  coverage, `gate_contract_version` and `readiness` (the inputs of the
  `wfo_completed` and `data_ready` gates)
- `recommendation.failed_gates` and `recommendation.deferral_reasons`

`recommendation.prob_outperform_raw` is null since step 5: it came from the
retired BayesianRidge posterior and was a constant 50 %. Use
`recommendation.prob_outperform_calibrated`.

## Consensus Shadow Diagnostic

`consensus_shadow.csv` still preserves the live-vs-equal-weight comparison.

That comparison is now diagnostic-only. It remains useful for:

- promotion auditing
- stability review
- governance checks when the live path changes

It is no longer rendered as a primary recommendation section in the main
monthly memo.

## Classification Shadow Diagnostic

`classification_shadow.csv` is the per-benchmark classifier detail export.

It is intended for:

- probability inspection by benchmark
- dashboard/email/report support
- later calibration and drift monitoring

The monthly memo, email, and dashboard surface an aggregated interpretation from
this artifact:

- `P(Actionable Sell)`
- a low / moderate / high confidence tier
- classifier stance
- agreement versus the live recommendation

This classifier layer is shadow-only today.

Recent monthly reports may also include reporting-only TA replacement variants:

- `ta_minimal_replacement`
- `ta_minimal_plus_vwo_pct_b`

These rows are monitoring-only. They do not affect the live recommendation,
sell percentage, or shadow gate overlay. Their prospective probabilities are
tracked in `artifacts/monthly_decisions/ta_shadow_variant_history.csv` for later
matured-outcome review.

## Shadow Gate Overlay

`decision_overlays.csv` records the live policy and the currently selected
shadow gate candidate side by side.

It is intended for:

- conservative classifier-gate monitoring
- promotion-readiness review
- disagreement analysis without changing live behavior

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

- `oos_r2` (against the benchmark's prevailing mean of realised targets)
- `nw_ic` (Spearman IC of the prequential ensemble score) and `nw_p_value`
- `hit_rate`, `base_rate` and `hit_rate_excess` (hit rate minus the better
  constant-sign rule)
- `pt_p_value` (Pesaran–Timmermann directional test)
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
- calibration notes (prequential ECE)
- trailing conformal coverage (each point's interval calibrated on residuals
  realised by then)
- shadow gate overlay status
- matured classifier-monitoring summary when available
- the walk-forward completion and input-readiness gates, and observation-to-feature context

Use the diagnostic report to understand why the recommendation mode landed where
it did.

## Local Dashboard

The repo also includes:

- a static monthly dashboard snapshot at `artifacts/monthly_decisions/<YYYY-MM>/dashboard.html`
- a local Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The static HTML snapshot is the linkable lightweight surface.

The Streamlit app remains a richer local viewer over the same committed monthly
artifacts.
