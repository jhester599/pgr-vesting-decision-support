# v74-v78 - Quality-Weighted Consensus Promotion Cycle

> Follow-on cycle to the completed `v66-v73` work. This track starts from the
> current `v38` production baseline and the `v72` research result, which is the
> strongest remaining promotion candidate.

## Goal

Move from research evidence to promotion discipline without changing the live
recommendation layer prematurely.

## Version Map

### v74 - Shadow consensus prototype

- add a production-safe, evaluation-only monthly shadow path for the
  `v72`-style quality-weighted consensus
- write a `consensus_shadow.csv` artifact and surface the comparison in
  `recommendation.md`
- keep the live recommendation mapping unchanged

### v75 - Holdout-ready promotion gate

- define the reserved holdout procedure for any future consensus promotion
- lock the scorecard to forecast, policy, and stability diagnostics before
  promotion

### v76 - Stability and concentration guardrails

- test whether the quality-weighted consensus remains stable under benchmark
  exclusions and score perturbations
- add concentration caps only if they materially improve robustness

### v77 - Recommendation-layer shadow tracking

- monitor whether the quality-weighted path would have changed the monthly
  recommendation mode or sell percentage versus the live path
- require repeated, non-noisy improvement before any production switch

### v78 - Promotion decision

- either promote the quality-weighted consensus into production after the
  predeclared checks pass, or document a no-promotion decision and keep `v38`
  live

## Current Position

- `v74` is implemented as a shadow-only monthly comparison
- live outputs still use the existing production consensus and recommendation
  mapping
- the new monthly artifact is intended to build operational confidence before a
  holdout-based promotion decision

## Execution Snapshot

- `v74` completed: monthly runs now export `consensus_shadow.csv` and render a
  live-vs-cross-check consensus section in `recommendation.md`
- `v75` completed: holdout-era replay run over the monthly review window from
  `2024-04-30` through `2026-03-31`
- replay outcome:
  - live equal-weight path: mean predicted return `-0.0152`, mean IC `0.1265`,
    signal changes `6`
  - `v74` quality-weighted path: mean predicted return `-0.0146`, mean IC
    `0.1359`, signal changes `4`
  - mode agreement with live: `100%`
  - sell agreement with live: `100%`
  - max top-benchmark weight: `18.36%`
- current gate result: `advance_to_promotion_check`
- `v76` completed on this branch: the quality-weighted consensus is now the
  live monthly recommendation path, with equal-weight retained as the visible
  production cross-check

## Implication

- the quality-weighted path has now cleared the promotion gate and been wired
  into production on this branch
- the next remaining decision is whether to leave the equal-weight cross-check
  visible for one release cycle or retire it after the first clean post-merge
  monthly run
