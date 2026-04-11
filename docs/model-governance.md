# Model Governance

## Purpose

This document defines the boundary between:

- production modeling behavior
- research and promotion work
- operational monitoring and stabilization

## Current Production Baseline

The live monthly workflow currently uses:

- the `v11.1` lean 2-model prediction stack (`Ridge + GBT`, v18 feature sets)
- `v38` post-ensemble shrinkage as the prediction-layer calibration baseline
- the promoted quality-weighted consensus as the live recommendation path
- the equal-weight consensus retained only as a diagnostic comparison in
  `consensus_shadow.csv`

The monthly workflow is the only place where a model or consensus stack becomes
operational.

## Current Monitoring Baseline

The production monthly output now tracks:

- aggregate OOS R^2
- Newey-West IC
- hit rate
- pooled and per-benchmark Clark-West diagnostics
- benchmark-level quality exports in `benchmark_quality.csv`
- live-vs-equal-weight comparison in `consensus_shadow.csv`
- machine-readable top-level state in `monthly_summary.json`

## Recent Promotion Record

Recent completed cycles:

- `v37-v60`
  - established `v38` as the best conservative calibration baseline
- `v66-v73`
  - aligned monthly diagnostics to ensemble reconstruction
  - added Clark-West and benchmark-quality exports
  - identified `v72` quality-weighted consensus as the strongest next candidate
- `v74-v78`
  - promoted the quality-weighted consensus into production after shadow and
    holdout-style review
- `v79-v80`
  - restored post-promotion monthly artifact wiring and validated the promoted
    path on a real monthly rerun
- `v81-v86`
  - aligned workflow, email, dashboard, and docs to the promoted baseline
  - added `monthly_summary.json`
  - retired the visible equal-weight cross-check from primary surfaces while
    keeping the diagnostic artifact

Supporting plan documents:

- `docs/superpowers/plans/2026-04-10-v37-v60-results-summary.md`
- `docs/superpowers/plans/2026-04-10-v66-v73-calibration-and-decision-layer.md`
- `docs/superpowers/plans/2026-04-10-v74-v78-quality-weighted-promotion.md`
- `docs/superpowers/plans/2026-04-11-v79-v80-post-promotion-stabilization.md`

## Research Candidates Still Worth Tracking

The following remain promising but are not live:

- `v70`
  - per-benchmark shrinkage calibration
- `v46`
  - classification / directional sidecar
- `v73`
  - hybrid decision-gating design

These remain research-only until they clear a later promotion study.

## Promotion Rule

Research results do not become production behavior automatically.

A candidate should only be promoted when it demonstrates:

- better policy-level utility than the current production baseline
- acceptable aggregate model health
- stable behavior across the repo's existing time-series validation framework
- acceptable operational complexity and maintainability
- clear documentation of the change and its reporting consequences

## Research vs. Production Labels

- Production:
  - scheduled workflow code
  - monthly decision generation
  - committed monthly output artifacts
- Research:
  - `src/research/`
  - `results/research/`
  - versioned plan and summary documents for candidate studies
- Provisional:
  - live-vs-shadow observability paths kept temporarily after a promotion

## Current Governance Conclusion

The current production path is the quality-weighted consensus.

The most immediate governance questions are now:

- whether `monthly_summary.json` should become the default contract for future
  automation and notification surfaces
- when the diagnostic-only equal-weight comparison can be further de-emphasized
  or archived
