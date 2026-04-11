# v66-v73 - Calibration and Decision-Layer Refinement

> Post-`v38` follow-on cycle. `v38` production shrinkage promotion is already complete and remains the fixed starting baseline for both production evaluation and research comparisons.

## Goal

Run a narrow, evidence-driven cycle that:

- aligns all monthly diagnostics and policy backtests to the deployed ensemble forecast
- adds Clark-West as a first-class forecast-value diagnostic
- exports benchmark-quality diagnostics for later weighting and gating work
- tests only conservative calibration and decision-layer extensions of the `v38` baseline

## Source Inputs

- prior consolidated summary: `docs/superpowers/plans/2026-04-10-v37-v60-results-summary.md`
- external repo review: `docs/archive/history/repo-peer-reviews/2026-04-10/chatgpt_repo_peerreview_20260410.md`

## Version Map

### v66 - Documentation and archive framing

- archive the 2026-04-10 external repo review under `docs/archive/history/repo-peer-reviews/2026-04-10/`
- create this master plan document and explicitly frame the cycle as post-`v38`

### v67 - Ensemble-aligned monthly diagnostics

- replace remaining component-model fallback OOS paths in `scripts/monthly_decision.py`
- align aggregate health, policy backtests, and diagnostic reporting to `reconstruct_ensemble_oos_predictions(...)`
- keep CLI behavior and recommendation-layer behavior unchanged

### v68 - Clark-West production diagnostics

- add reusable Clark-West helpers outside the report script
- extend aggregate monthly health payloads with `cw_t_stat` and `cw_p_value`
- render pooled Clark-West results in `diagnostic.md`

### v69 - Benchmark-quality export

- compute benchmark-level ensemble OOS diagnostics
- export `benchmark_quality.csv` in monthly outputs
- surface per-benchmark OOS R², Newey-West IC, hit rate, and Clark-West diagnostics in `diagnostic.md`

### v70 - Per-benchmark shrinkage research

- test prequential benchmark-specific shrinkage calibration against the fixed `v38` global alpha baseline
- keep all variants research-only

### v71 - Affine recalibration research

- test conservative prequential `a + b * y_hat` recalibration with bounded coefficients
- preserve strict as-of fitting discipline

### v72 - Quality-weighted consensus research

- test shrinkage-to-equal benchmark weighting using `v69` quality diagnostics
- compare consensus-level forecast and policy metrics against equal weighting

### v73 - Hybrid decision-gate research

- combine `v38` regression magnitude with `v46` directional probability as a gating layer
- evaluate simple thresholded decision rules with policy metrics as the primary outcome

## Outputs

- monthly diagnostics now include Clark-West and benchmark-quality exports
- research outputs:
  - `results/research/v70_benchmark_shrinkage.py`
  - `results/research/v71_affine_recalibration.py`
  - `results/research/v72_quality_weighted_consensus.py`
  - `results/research/v73_hybrid_decision_gate.py`
- matching CSVs under `results/research/`
- matching tests under `tests/`

## Execution Snapshot

- `v67-v69` completed: monthly reporting now reconstructs ensemble OOS paths for
  aggregate health, diagnostic markdown, policy summaries, and
  `benchmark_quality.csv`
- `v68` completed: Clark-West is now exposed through shared diagnostics code and
  rendered in both pooled and per-benchmark monthly outputs
- `v70` completed as research-only: best variant `A_prior12` improved pooled OOS
  R2 to `-0.1041` versus the `v38` baseline at `-0.1310`, but with weaker hit
  rate
- `v71` completed as research-only: conservative affine recalibration degraded
  pooled OOS R2 to roughly `-0.1530` to `-0.1592`
- `v72` completed as research-only: quality-weighted consensus was the strongest
  research result in this cycle, improving pooled OOS R2 to `-0.0445` and
  Newey-West IC to `0.3620` while preserving policy uplift versus the
  diversification baseline
- `v73` completed as research-only: hybrid decision gating did not improve
  pooled forecast accuracy, but it remains useful as a recommendation-layer
  design pattern for later policy work

## Recommended Follow-Up

- keep the current production `v38` baseline live while monitoring the new
  Clark-West and benchmark-quality diagnostics in monthly runs
- treat `v72` as the leading candidate for the next promotion-oriented research
  cycle, with stability and policy diagnostics predeclared before any rollout
- retain `v70` as a secondary calibration branch worth revisiting if
  per-benchmark variance control becomes more important than raw hit rate
- do not promote `v71` or `v73` without a stronger decision-policy benefit on a
  reserved holdout check

## Promotion Defaults

- `v38` remains the production baseline until a later candidate beats it on predeclared metrics
- `v69-v73` outputs are diagnostic or research-only in this cycle
- no reserved holdout window is consumed during `v66-v73`
