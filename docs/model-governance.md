# Model Governance

## Purpose

This document defines the boundary between:

- production modeling behavior
- research/evaluation work
- promotion decisions

## Current Production Baseline

The current production monthly decision workflow remains the v8.13 4-model
ensemble with model-quality gating and tax-aware reporting.

The monthly workflow is the only place where a model stack becomes operational.

## Current Research Baseline

v9 research is complete and documented in:

- `codex-v9-plan.md`
- `V9_RESULTS_SUMMARY.md`
- `V9_CLOSEOUT_AND_V91_NEXT.md`

It introduced:

- benchmark harnesses
- feature-cost and feature-selection studies
- target reformulation experiments
- benchmark-reduction studies
- policy evaluation
- weekly snapshot experiments
- a tuned Ridge classifier-sidecar candidate

v11 research is now also complete and documented in:

- `codex-v11-plan.md`
- `V11_RESULTS_SUMMARY.md`
- `V11_CLOSEOUT_AND_V12_NEXT.md`

It added:

- diversification-aware benchmark scoring relative to PGR
- explicit separation between forecast benchmarks and redeployment destinations
- diversification-first universe reduction
- reduced-universe candidate bakeoffs
- policy redesign around simpler sell / hold / neutral logic
- production-like dry-run recommendation memos with lot-level guidance

## Promotion Rule

Research results do not become production behavior automatically.

A candidate should only be promoted when it demonstrates:

- better policy-level utility than the current production baseline or the agreed
  naive baseline
- acceptable aggregate model health
- acceptable obs/feature discipline
- stable behavior across the validation framework already used by the repo
- clear documentation of the change and its operational consequences
- a recommendation that improves diversification usefulness, not just local
  benchmark prediction quality

## Research vs. Production Labels

- Production:
  - scheduled workflow code
  - monthly decision generation
  - committed monthly output artifacts
- Research:
  - `src/research/`
  - v9 and v11 scripts and outputs
- Provisional:
  - candidates under consideration for future production use
  - diversification-aware redeploy policies not yet promoted

## Classifier Sidecar Policy

The tuned Ridge classifier from v9 is currently a sidecar confidence candidate.

It may be used for:

- additional confidence diagnostics
- abstention research
- future policy experiments

v11 kept the classifier in that role. Even when the abstain-only overlay looked
useful in research, it was still not promoted to primary-engine status.

It is not the primary production decision engine.
