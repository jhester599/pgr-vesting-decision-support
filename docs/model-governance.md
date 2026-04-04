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

v12 shadow-promotion work is now also complete and documented in:

- `codex-v12-plan.md`
- `V12_RESULTS_SUMMARY.md`
- `V12_CLOSEOUT_AND_V13_NEXT.md`

It added:

- a rolling 12-month shadow comparison between the live monthly stack and the
  simpler diversification-first baseline
- side-by-side dry-run memos under `results/v12/dry_runs/`
- an explicit check for whether recommendation-layer simplification should
  happen before any new model stack promotion

v13 production-facing recommendation-layer work is now also documented in:

- `codex-v13-plan.md`
- `V13_RESULTS_SUMMARY.md`

It adds:

- a recommendation-layer mode switch with `live_only`, `live_with_shadow`, and
  `shadow_promoted`
- production report and email support for a simpler-baseline cross-check
- explicit existing-holdings lot guidance
- explicit diversification-first redeploy guidance

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
  - v12 shadow-baseline recommendation behavior

## Current Recommendation-Layer Conclusion

After v12, the repo still does not recommend promoting a new live model stack.

The most promising next production change is narrower:

- consider promoting the simpler diversification-first recommendation layer
- keep model-stack promotion separate until a reduced-universe candidate
  clearly beats the baseline

## Current v13 Default

The current production-facing default is:

- `RECOMMENDATION_LAYER_MODE=live_with_shadow`

This means:

- the live 4-model monthly stack still drives the official recommendation
- the simpler v12 baseline is surfaced as a cross-check
- usefulness improvements from v11-v12 are now part of the production report
  and email path

## Classifier Sidecar Policy

The tuned Ridge classifier from v9 is currently a sidecar confidence candidate.

It may be used for:

- additional confidence diagnostics
- abstention research
- future policy experiments

v11 kept the classifier in that role. Even when the abstain-only overlay looked
useful in research, it was still not promoted to primary-engine status.

It is not the primary production decision engine.
