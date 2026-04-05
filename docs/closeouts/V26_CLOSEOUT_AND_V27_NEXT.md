# V26 Closeout And V27 Next

Created: 2026-04-05

## v26 Closeout

`v26` is complete.

It was intentionally narrow:

- no new prediction research
- no new feature expansion
- no benchmark-universe redesign

It finished the productionization work around the `v25` correction cycle by:

- cleaning the remaining historical-comparison warning noise
- confirming the promoted visible cross-check still renders correctly
- confirming the monthly decision flow still behaves as expected on the
  corrected foundation

## Current Governing Conclusion

The repo state after `v26` is:

- keep the simpler diversification-first recommendation layer active
- keep `ensemble_ridge_gbt_v18` as the visible promoted cross-check
- keep `VOO`
- avoid reopening a broad model / feature search until there is a clearly
  scoped new question

## Recommended Next Step: v27

`v27` should be an operational / UX refinement cycle, not a new prediction
research loop.

Best targets:

1. clean up the monthly email / report presentation using the latest April run
   feedback
2. tighten any remaining stale documentation that still describes pre-`v21`
   promotion conclusions as current
3. optionally add a small diagnostics surface that shows:
   - active recommendation layer
   - visible promoted cross-check
   - whether they agree on mode / sell %

## Non-Goals For v27

- no new broad feature inventory
- no new benchmark replacement study
- no restart of the v15-v19 style sweep unless a fresh bounded hypothesis is
  defined first
