# Model Governance

## Purpose

This document defines the boundary between:

- production modeling behavior
- research and promotion work
- operational monitoring and stabilization

## Current Production Baseline

The live monthly workflow currently uses:

- the `v11.1` lean 2-model prediction stack (`Ridge + GBT`, v18 feature sets)
- post-ensemble shrinkage chosen by the `v38` rule, applied prequentially: the
  grid alpha with the lowest squared error over the realised OOS record,
  re-chosen every month (the fixed 0.50 is research-only since review
  2026-09-25, step 5)
- the promoted quality-weighted consensus as the live recommendation path for
  direction and forecast
- the recommendation-mode gates of review 2026-09-25, step 5, gated on the
  equal-weight IC, plus the `wfo_completed` and `data_ready` readiness gates of
  the pre-v200 remediation R3 (below); validation is walk-forward only
- the equal-weight consensus retained only as a diagnostic comparison in
  `consensus_shadow.csv`

The monthly workflow is the only place where a model or consensus stack becomes
operational.

## Current Monitoring Baseline

The production monthly output now tracks:

- aggregate OOS R^2 against each benchmark's prevailing mean of the targets
  realised by each forecast date
- pooled IC with a Driscoll-Kraay p-value clustered by date; per-benchmark
  Newey-West IC
- hit rate against the base rate, with the Pesaran-Timmermann directional test
- prequential ECE and trailing conformal coverage
- walk-forward completion for every required model/benchmark pair and
  required-input readiness at the as-of date (both gates; R3)
- pooled and per-benchmark Clark-West diagnostics
- benchmark-level quality exports in `benchmark_quality.csv`
- live-vs-equal-weight comparison in `consensus_shadow.csv`
- per-benchmark classifier detail in `classification_shadow.csv`
- shadow gate comparison in `decision_overlays.csv`
- append-only classifier history in `artifacts/monthly_decisions/classification_shadow_history.csv`
- reporting-only TA replacement variant history in
  `artifacts/monthly_decisions/ta_shadow_variant_history.csv`
- machine-readable top-level state in `monthly_summary.json`

## Decision Record

Each promotion decision (what became live behaviour, or was deliberately kept
out of it, and why) is one file in [`docs/decisions/`](decisions/README.md).
The context, evidence and consequences live there; this table is the summary.

| # | Decision | Date | Versions | Status |
|---|---|---|---|---|
| [0001](decisions/0001-v38-post-ensemble-shrinkage.md) | Post-ensemble shrinkage (v38) is the baseline every candidate must beat | 2026-04-10 | v37–v60 | Live; alpha prequential since 0006 |
| [0002](decisions/0002-quality-weighted-consensus.md) | Quality-weighted consensus (v72) sets the live direction and forecast; equal weight kept as a diagnostic | 2026-04-10 | v66–v78 | Live |
| [0003](decisions/0003-monthly-summary-contract.md) | Post-promotion stabilisation; `monthly_summary.json` contract; visible equal-weight cross-check retired | 2026-04-11 | v79–v86 | Live |
| [0004](decisions/0004-classifier-stays-shadow-only.md) | The directional classifier and its gate overlays stay shadow-only | 2026-04-11 | v102–v117 | Shadow only |
| [0005](decisions/0005-ta-variants-reporting-only.md) | Two technical-analysis classifier variants are monitored as reporting-only shadow rows | 2026-04-18 | v160–v169 | Reporting only |
| [0006](decisions/0006-validation-gates-and-cpcv-diagnostic.md) | Four recommendation-mode gates on realised-only health; CPCV is a diagnostic, not a gate | 2026-09-26 | v178 (review step 5) | Live; CPCV retired by 0008 |
| [0007](decisions/0007-actionable-sell-mapping.md) | ACTIONABLE sell-% mapping: a bullish consensus never sells more than the 50 % default | 2026-09-26 | v179 (review step 6) | Live |
| [0008](decisions/0008-chronological-validation-and-readiness-gates.md) | Chronological validation only; `wfo_completed` and `data_ready` gates replace the CPCV completeness gate | 2026-09-27 | v188 (R3) | Live |

### Gates and mapping in force

Gate contract `chronological-readiness-2026-09-27`
([0008](decisions/0008-chronological-validation-and-readiness-gates.md)),
recorded in every `run_manifest.json` and `monthly_summary.json`. ACTIONABLE
needs all five gates to pass:

- OOS R² ≥ 2 % against the prevailing mean of realised targets;
- equal-weight IC ≥ 0.07;
- one-sided Pesaran–Timmermann p < 0.05, Driscoll–Kraay by date;
- `wfo_completed`: walk-forward results for every required model (Ridge, GBT)
  and benchmark (the eight of `PRIMARY_FORECAST_UNIVERSE`), with non-empty
  folds, finite OOS predictions, the production gap, labels realised before
  each test fold, outcomes realised by the as-of date and a finite live
  forecast;
- `data_ready`: every live model feature finite before imputation, and
  prices, the FRED series behind live features, PGR monthly EDGAR and
  dividends (PGR and the eight benchmarks; GLD an audited non-payer) fresh at
  the as-of date.

Any failure gives DEFER-TO-TAX-DEFAULT at 50 %; otherwise MONITORING-ONLY. A
missing, non-finite or unknown input fails its gate; the reason is named in
every output surface. When every gate passes,
[0007](decisions/0007-actionable-sell-mapping.md) sets the sell percentage:
OUTPERFORM > 15 % → 25 %; OUTPERFORM ≤ 15 % → 50 %; UNDERPERFORM → 100 %;
NEUTRAL, or IC < 0.05 / missing → 50 %. Validation is walk-forward only; the
representative CPCV (a combinatorial K-fold) no longer runs and no longer
gates.

### Health baseline at 2026-09-21

Replayed on the corrected pipeline (review 2026-09-25, step 5; month-by-month
replay in [0006](decisions/0006-validation-gates-and-cpcv-diagnostic.md)). Later
months are compared against it.

| Metric | Value | Gate |
|---|---|---|
| Aggregate OOS R² (vs prevailing mean) | +5.08 % (was −2.16 % against the leaky naive) | PASS |
| Equal-weight mean IC | 0.0706 (quality-weighted 0.0855) | PASS, by 0.0006 |
| Pooled rank IC, Driscoll–Kraay p | 0.130, p 0.030 | — |
| Hit rate vs base rate, Pesaran–Timmermann p | 62.8 % vs 68.1 %, p 0.255 | FAIL |
| Prequential ECE | 14.2 % (in-sample 0.6 %) | — |
| Trailing conformal coverage (target 80 %) | 40.6 % | — |
| CPCV positive paths (diagnostic) | 5/7, GOOD | ran |
| Shrinkage alpha (prequential) | 0.50 | — |

`model_performance_log` stores these corrected metrics from the next
production run, tagged `metrics_version = prequential-2026-09-25`.

## Research Candidates Still Worth Tracking

The following remain promising but are not live:

- `v70`
  - per-benchmark shrinkage calibration
- `v46`
  - classification / directional sidecar
- `v73`
  - hybrid decision-gating design
- `v110-v113`
  - constrained classifier overlay candidates, with Gemini-style veto gating
    currently the strongest shadow candidate
- `v165-v169`
  - TA replacement classifier variants, monitored only through monthly shadow
    artifacts and the TA ledger until enough 6M horizons mature

These remain research-only until they clear a later promotion study.

## Promotion Rule

Research results do not become production behavior automatically.

A candidate should only be promoted when it demonstrates:

- better policy-level utility than the current production baseline
- acceptable aggregate model health
- stable behavior across the repo's existing time-series validation framework
- acceptable operational complexity and maintainability
- clear documentation of the change and its reporting consequences
- acceptable agreement and churn versus the current live recommendation path
- no material calibration drift once sufficient matured classifier history exists

Record every promotion, and every deliberate decision not to promote a
candidate that reached a promotion check, as the next numbered file in
[`docs/decisions/`](decisions/README.md) and add a row to the Decision Record
table above, in the same PR as the change.

## Research vs. Production Labels

- Production:
  - scheduled workflow code
  - monthly decision generation
  - committed monthly output artifacts
- Research:
  - `src/research/`
  - `research/studies/` (registry: `research/registry.yaml`)
  - versioned plan, closeout and summary documents (now under `docs/history/`)
- Provisional:
  - live-vs-shadow observability paths kept temporarily after a promotion

## Current Governance Conclusion

The current production path is the quality-weighted consensus.

Since review 2026-09-25, step 5, the corrected gates hold every month from
2026-02 to 2026-09 at the 50 % tax default, and directional skill is the gate
that binds. Step 6 changed the ACTIONABLE sell-percentage mapping
([0007](decisions/0007-actionable-sell-mapping.md)).

The most immediate governance questions are now:

- whether `monthly_summary.json` should become the default contract for future
  automation and notification surfaces
- when the diagnostic-only equal-weight comparison can be further de-emphasized
  or archived
- whether the shadow-only classifier overlay remains stable enough to justify a
  later limited production gate candidate
