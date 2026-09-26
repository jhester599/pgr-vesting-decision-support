# V28 Results Summary

Created: 2026-04-05

## Objective

v28 tested whether the project should prune the forecast benchmark universe to
better align it with the realistic buyable alternatives introduced in v27.

This was a distinct question from the v27 redeploy work:

- v27 pruned the monthly *buy answer*
- v28 tests whether the *prediction universe* should also be pruned

## Universes Tested

1. `current_reduced`
   - `VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE`
2. `buyable_only`
   - `VOO, VGT, SCHD, VXUS, VWO, BND`
3. `buyable_plus_context`
   - `VOO, VGT, SCHD, VXUS, VWO, BND, VFH, KIE`

## Fixed Paths

v28 held the following fixed:

- promoted recommendation layer:
  - v13.1 simpler diversification-first layer
- promoted visible cross-check candidate:
  - `ensemble_ridge_gbt_v18`
- live reduced comparison path:
  - `live_production_ensemble_reduced`
- shadow baseline:
  - `historical_mean`

## Result

v28 decision:

- `keep_current_forecast_universe`

Recommended universe:

- `current_reduced`

## Why

The narrower buyable-first universes were cleaner conceptually, but they did
not preserve enough of the promoted candidate's edge.

### Current Reduced Universe

For `ensemble_ridge_gbt_v18`:

- mean sign-policy return: `0.0798`
- mean OOS R^2: `-0.1424`
- signal agreement with shadow baseline: `84.3%`
- review window: `2016-10-31` to `2025-09-30` (`108` months)

### Buyable-Only Universe

For `ensemble_ridge_gbt_v18`:

- mean sign-policy return: `0.0679`
- mean OOS R^2: `-0.1539`
- signal agreement with shadow baseline: `67.7%`
- review window: `2017-10-31` to `2025-09-30` (`96` months)

### Buyable-Plus-Context Universe

For `ensemble_ridge_gbt_v18`:

- mean sign-policy return: `0.0606`
- mean OOS R^2: `-0.1840`
- signal agreement with shadow baseline: `71.9%`
- review window: `2017-10-31` to `2025-09-30` (`96` months)

## Interpretation

The buyable-first universes lost too much relative to the current promoted
forecast universe on the metrics that matter most:

- policy utility
- historical agreement with the promoted simpler baseline
- comparable OOS fit

This means the non-buyable funds still appear to be doing useful work as
forecast benchmarks, even if they should not be shown as preferred destinations
for sold PGR proceeds.

## Important Distinction Preserved

v28 confirms the project should keep two different concepts:

1. forecast benchmark universe
2. investable monthly redeploy universe

So the current project state should be:

- keep the broader `current_reduced` forecast universe
- keep the narrower v27 redeploy universe for the user-facing buy answer

## Practical Conclusion

Do **not** prune the forecast benchmark universe yet.

Do continue to prune the user-facing redeploy answer.

That is now the most defensible setup:

- richer forecast context in the prediction layer
- realistic buy recommendations in the monthly output
