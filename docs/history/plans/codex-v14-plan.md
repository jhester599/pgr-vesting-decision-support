# codex-v14-plan.md

## Goal

Run a narrow post-v13 study to determine whether the underlying prediction
layer can be simplified or replaced without disturbing the newly promoted
recommendation layer.

v14 is intentionally not a broad methodology cycle. It assumes:

- the v13.1 recommendation layer stays in place
- the live monthly workflow remains the operational path
- the next question is whether a leaner prediction stack can beat the current
  live 4-model stack on a reduced, diversification-aware benchmark universe

## Why v14 Exists

v9-v12 established four important facts:

1. more model complexity was not the bottleneck
2. reduced benchmark universes were more promising than the full 21-benchmark stack
3. the simplest recommendation policies were steadier and easier to explain
4. the best immediate production improvement was the recommendation layer, not a
   model-stack promotion

v13.1 implemented that recommendation-layer promotion. v14 now isolates the
remaining open question:

> Can a leaner, reduced-universe prediction stack materially outperform the
> current live stack without making the system less stable or less useful?

## Scope

v14 should evaluate only a very small candidate set:

- current live 4-model production stack
- `ridge_lean_v1`
- `gbt_lean_plus_two`
- optional `ridge_lean_v1 + gbt_lean_plus_two` ensemble
- `historical_mean` baseline for context

The recommendation layer should stay fixed at the v13.1 default throughout the
study.

## Non-Goals

Do not spend v14 on:

- new model families
- more feature-family expansion
- daily/interpolated higher-frequency experiments
- classifier promotion
- recommendation-layer redesign
- new data vendors

## Core Hypothesis

A reduced benchmark universe and a leaner prediction stack may improve the
truthfulness of the model layer enough to justify replacing the current live
stack, while leaving the recommendation layer unchanged.

## Workstreams

### v14.0 - Freeze the Post-v13 Baseline

- Capture one clean baseline snapshot after the real April production run.
- Record:
  - current recommendation output
  - current live prediction metrics
  - current recommendation-layer setting
  - whether the live stack and simplified baseline agreed
- Write the baseline record under `results/v14/`.

### v14.1 - Define the Reduced Forecast Universe

- Start with the strongest diversification-aware reduced sets already surfaced
  in v11/v12.
- Reconfirm which forecast universe should be used for a production-candidate
  study.
- Keep contextual PGR-like funds small in number.
- Prefer funds that still support diversification usefulness rather than simply
  mirroring PGR.

Expected output:

- one chosen v14 forecast universe
- one markdown memo explaining why it was selected

### v14.2 - Direct Candidate Bakeoff

- Run the candidate stack head-to-head on the selected reduced universe:
  - live 4-model production stack
  - `ridge_lean_v1`
  - `gbt_lean_plus_two`
  - optional 2-model lean ensemble
  - `historical_mean`
- Score each candidate on:
  - aggregate OOS R²
  - IC
  - hit rate
  - policy utility
  - month-to-month stability

Advancement rule:

- only candidates that clearly improve on the live stack and remain within
  reach of the `historical_mean` baseline proceed

### v14.3 - Minimal Feature Surgery

- For the surviving prediction candidates only, run one-feature-at-a-time
  add/drop validation.
- Preserve a lean feature budget.
- Reject any feature change that improves local metrics but worsens policy
  utility or coverage burden.

This stage should stay intentionally small. It is for tightening a candidate,
not reopening a broad feature-discovery program.

### v14.4 - Production-Like Shadow Runs

- Plug the best v14 candidate prediction layer into the monthly decision flow
  while keeping the v13.1 recommendation layer fixed.
- Generate shadow outputs for several recent monthly dates.
- Compare:
  - recommendation stability
  - agreement/disagreement with the current live stack
  - whether the new prediction layer creates clearer or noisier guidance

### v14.5 - Promotion Decision

Produce one explicit outcome:

- promote a new prediction layer candidate next
- continue shadowing, do not promote yet
- reject replacement and keep the current live stack

## Acceptance Gates

A v14 candidate should only be recommended if all are true:

- it materially improves on the current live stack
- it does not create a more confusing recommendation story
- it preserves or improves diversification usefulness
- it does not rely on a heavier feature burden than necessary
- it remains stable enough across recent monthly snapshots

## Deliverables

- `results/v14/` scorecards
- `docs/results/V14_RESULTS_SUMMARY.md`
- `docs/closeouts/V14_CLOSEOUT_AND_V15_NEXT.md`
- candidate comparison CSVs
- shadow monthly output comparisons

## Best-Case Outcome

v14 ends with a narrow, production-ready proposal to replace the current live
prediction stack with a simpler reduced-universe candidate while keeping the
v13.1 recommendation layer intact.

## Acceptable Outcome

v14 ends with a well-supported decision not to replace the live prediction
stack yet, but with better evidence about what must improve next.
