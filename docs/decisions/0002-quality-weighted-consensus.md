# 0002 — Quality-weighted consensus (v72 → v76)

| | |
|---|---|
| **Status** | Accepted, live |
| **Date** | 2026-04-10 (v66–v73 research; v74–v78 promotion cycle) |
| **Where it lives** | `config.model.CONSENSUS_WEIGHTING_MODE`; `src/models/consensus_shadow.py` |
| **Studies** | `research/studies/v72_quality_weighted_consensus/`, `research/studies/v75_holdout_shadow_replay/` (both `promoted`) |

## Context

The monthly signal combines per-benchmark forecasts into one consensus.
Until v76 every benchmark had equal weight, although some benchmarks'
models forecast much better than others.

## Decision

Weight each benchmark by its model quality in the consensus that sets the
live direction and forecast. Keep the equal-weight consensus as a diagnostic
comparison (`consensus_shadow.csv`).

## Evidence

- **v72 (research).** The strongest result of the v66–v73 cycle: pooled OOS R²
  improved to −0.0445 (v38 baseline −0.1310) and Newey–West IC to 0.3620,
  while keeping the policy uplift over the diversification baseline.
- **v74 (shadow).** Monthly runs wrote the quality-weighted path next to the
  live one in `consensus_shadow.csv`, without changing the recommendation.
- **v75 (hold-out replay, 2024-04-30 → 2026-03-31).** Against the live
  equal-weight path: mean IC 0.1359 vs 0.1265; 4 signal changes vs 6;
  100 % agreement on recommendation mode and on sell percentage; largest
  benchmark weight 18.36 %. Gate result: `advance_to_promotion_check`.
- **v76.** Promoted into the live monthly path, with equal weight kept as a
  visible cross-check for the first release cycle.

## Consequences

- Promotion changed neither recommendation mode nor sell percentage over the
  replay window; it changed the forecast and the IC.
- The visible cross-check was retired in v86
  ([0003](0003-monthly-summary-contract.md)).
- Since review 2026-09-25, step 5, the recommendation-mode IC gate uses the
  equal-weight IC, so the weighting cannot help the model pass its own gate
  ([0006](0006-validation-gates-and-cpcv-diagnostic.md)).

## Sources

- [v66–v73 plan and results](../history/superpowers/plans/2026-04-10-v66-v73-calibration-and-decision-layer.md)
- [v74–v78 promotion cycle](../history/superpowers/plans/2026-04-10-v74-v78-quality-weighted-promotion.md)
- [v72 study README](../../research/studies/v72_quality_weighted_consensus/README.md),
  [v75 study README](../../research/studies/v75_holdout_shadow_replay/README.md)
