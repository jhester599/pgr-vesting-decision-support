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

## Promotion Rule

Research results do not become production behavior automatically.

A candidate should only be promoted when it demonstrates:

- better policy-level utility than the current production baseline or the agreed
  naive baseline
- acceptable aggregate model health
- acceptable obs/feature discipline
- stable behavior across the validation framework already used by the repo
- clear documentation of the change and its operational consequences

## Research vs. Production Labels

- Production:
  - scheduled workflow code
  - monthly decision generation
  - committed monthly output artifacts
- Research:
  - `src/research/`
  - v9 scripts and outputs
- Provisional:
  - candidates under consideration for future production use

## Classifier Sidecar Policy

The tuned Ridge classifier from v9 is currently a sidecar confidence candidate.

It may be used for:

- additional confidence diagnostics
- abstention research
- future policy experiments

It is not the primary production decision engine.
