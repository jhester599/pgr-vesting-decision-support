# v28 Forecast Universe Review Plan

Created: 2026-04-05

## Goal

Test whether the broader forecast benchmark universe should be pruned to better
match realistic alternative buys after the v27 redeploy-portfolio work.

The v28 question is narrower than v27:

- should the prediction layer use a more buyable-first benchmark list?

## Why This Matters

v27 already pruned the *monthly buy recommendation* universe. It did not
prune the *forecast benchmark* universe.

That split was intentional, but it creates an open follow-up question:

- are some of the remaining forecast benchmarks still pulling their weight, or
  are they now just non-buyable complexity?

## Compared Universes

v28 evaluates three universe candidates:

1. `current_reduced`
   - `VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE`
2. `buyable_only`
   - `VOO, VGT, SCHD, VXUS, VWO, BND`
3. `buyable_plus_context`
   - `VOO, VGT, SCHD, VXUS, VWO, BND, VFH, KIE`

## Fixed Elements

v28 holds these constant:

- the promoted v13.1 recommendation layer
- the promoted visible cross-check candidate:
  - `ensemble_ridge_gbt_v18`
- the historical-mean shadow baseline
- the corrected v25-v26 evaluation foundation

## Comparison Criteria

Each universe is evaluated on:

- mean sign-policy utility
- mean OOS R^2
- historical agreement with the promoted simpler baseline
- comparison against the live reduced 4-model stack in the same universe
- review-window coverage and buyable-share metadata

## Decision Rule

Prune the forecast universe only if a narrower buyable-first universe:

- preserves or beats the live stack inside that universe
- remains close enough to the current promoted universe on policy utility
- remains close enough to the current promoted universe on OOS fit
- remains close enough to the current promoted universe on historical signal
  agreement with the simpler baseline

Otherwise:

- keep the current reduced forecast universe
- keep the v27 redeploy universe narrower than the forecast universe
