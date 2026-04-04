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

- `docs/plans/codex-v9-plan.md`
- `docs/results/V9_RESULTS_SUMMARY.md`
- `docs/closeouts/V9_CLOSEOUT_AND_V91_NEXT.md`

It introduced:

- benchmark harnesses
- feature-cost and feature-selection studies
- target reformulation experiments
- benchmark-reduction studies
- policy evaluation
- weekly snapshot experiments
- a tuned Ridge classifier-sidecar candidate

v11 research is now also complete and documented in:

- `docs/plans/codex-v11-plan.md`
- `docs/results/V11_RESULTS_SUMMARY.md`
- `docs/closeouts/V11_CLOSEOUT_AND_V12_NEXT.md`

It added:

- diversification-aware benchmark scoring relative to PGR
- explicit separation between forecast benchmarks and redeployment destinations
- diversification-first universe reduction
- reduced-universe candidate bakeoffs
- policy redesign around simpler sell / hold / neutral logic
- production-like dry-run recommendation memos with lot-level guidance

v12 shadow-promotion work is now also complete and documented in:

- `docs/plans/codex-v12-plan.md`
- `docs/results/V12_RESULTS_SUMMARY.md`
- `docs/closeouts/V12_CLOSEOUT_AND_V13_NEXT.md`

It added:

- a rolling 12-month shadow comparison between the live monthly stack and the
  simpler diversification-first baseline
- side-by-side dry-run memos under `results/v12/dry_runs/`
- an explicit check for whether recommendation-layer simplification should
  happen before any new model stack promotion

v13 production-facing recommendation-layer work is now also documented in:

- `docs/plans/codex-v13-plan.md`
- `docs/results/V13_RESULTS_SUMMARY.md`

It adds:

- a recommendation-layer mode switch with `live_only`, `live_with_shadow`, and
  `shadow_promoted`
- production report and email support for a simpler-baseline cross-check
- explicit existing-holdings lot guidance
- explicit diversification-first redeploy guidance
- a promoted v13.1 default that uses the simpler diversification-first
  recommendation layer while keeping the live stack visible as a cross-check

v14 reduced-universe prediction-layer work is now also documented in:

- `docs/plans/codex-v14-plan.md`
- `docs/results/V14_RESULTS_SUMMARY.md`
- `docs/closeouts/V14_CLOSEOUT_AND_V15_NEXT.md`

It adds:

- a post-v13 baseline freeze after the real April production run
- a narrow reduced-universe bakeoff between the live 4-model stack and lean
  Ridge/GBT-centered candidates
- minimal one-feature-at-a-time surgery on the surviving lean candidates
- production-like shadow review memos for the leading replacement candidate

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
  - v14 reduced-universe replacement candidate shadowing

## Current Recommendation-Layer Conclusion

After v12, the repo still does not recommend promoting a new live model stack.

The most promising next production change is narrower:

- consider promoting the simpler diversification-first recommendation layer
- keep model-stack promotion separate until a reduced-universe candidate
  clearly beats the baseline

## Current v13 Default

The current production-facing default is:

- `RECOMMENDATION_LAYER_MODE=shadow_promoted`

This means:

- the simpler v12 baseline now drives the official recommendation layer
- the live 4-model monthly stack is still surfaced as a cross-check
- usefulness improvements from v11-v12 are now part of the production report
  and email path

## Current Prediction-Layer Conclusion

After v14, the repo still does not recommend replacing the live prediction
stack immediately.

The current narrowed conclusion is:

- `ensemble_ridge_gbt` is the leading reduced-universe replacement candidate
- it improves on the reduced-universe live 4-model stack
- it remains close to, but not clearly ahead of, the `historical_mean`
  baseline
- the next step should be fixed-budget feature replacement in v15 rather than
  broader methodology expansion

## Classifier Sidecar Policy

The tuned Ridge classifier from v9 is currently a sidecar confidence candidate.

It may be used for:

- additional confidence diagnostics
- abstention research
- future policy experiments

v11 kept the classifier in that role. Even when the abstain-only overlay looked
useful in research, it was still not promoted to primary-engine status.

It is not the primary production decision engine.
