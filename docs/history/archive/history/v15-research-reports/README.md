# v15 External Research Reports

Created: 2026-04-04

## Purpose

These files archive the external deep-research inputs used to build the v15 fixed-budget feature-replacement plan.

They are kept in-repo so future feature work can trace:

- which feature ideas came from which external model
- where the overlap and disagreement came from
- which concepts were already considered before adding new features

## Archived Reports

- `v15_chatgptdeepresearch_20260404.md`
- `v15_geminideepresearch_20260404.md`
- `v15_geminipro_20260404.md`
- `v15_claudeopusresearch_20260404.md`

## High-Level Takeaways

- All four reports supported a fixed-budget replacement approach rather than feature-count expansion.
- All four reports recommended adding benchmark-predictive features because the target is relative return, not absolute PGR return.
- The strongest cross-report overlap was:
  - insurance pricing power / rate adequacy
  - underwriting-quality decomposition
  - benchmark-side USD / inflation / rates / credit drivers
  - leaner relative / peer-aware price features
- The main disagreement was not directionally important. The reports mostly differed on:
  - exact horizon choice (`3m` vs `6m`)
  - whether to prefer level vs change vs acceleration versions
  - whether a concept belongs first in Ridge or GBT

## Repo Integration

The synthesized execution plan built from these reports is:

- `docs/plans/codex-v15-feature-test-plan.md`

The normalized candidate inventory and generated swap queue are stored in:

- `results/v15/`
