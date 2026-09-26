# V13 Results Summary

Created: 2026-04-04

## Goal

Implement the safest production-ready improvements from v11 and v12 without promoting a new model stack, then promote the simpler recommendation layer if it remains steadier and equally actionable.

## What Changed

- The monthly report now includes:
  - `Recommendation Layer`
  - `Existing Holdings Guidance`
  - `Redeploy Guidance`
  - `Simple-Baseline Cross-Check`
- The email now includes:
  - the same lot-trimming order used in the report
  - diversification-first redeploy guidance
  - a simpler-baseline cross-check near the top of the message
- The recommendation-layer mode is now configurable through:
  - `live_only`
  - `live_with_shadow`
  - `shadow_promoted`

## Current Default

- `RECOMMENDATION_LAYER_MODE=shadow_promoted`

This keeps the live production model stack in place for prediction generation,
but promotes the steadier diversification-first baseline as the active
recommendation layer while retaining the live stack as a cross-check.

## April Dry-Run Result

- Live production:
  - signal `OUTPERFORM`
  - mode `DEFER-TO-TAX-DEFAULT`
  - sell `50%`
- Simpler baseline:
  - signal `OUTPERFORM`
  - mode `DEFER-TO-TAX-DEFAULT`
  - sell `50%`

The action is unchanged, but the report and email are more useful because they
now explicitly answer:

- what to do with the next vest
- what held shares to trim first
- where sold exposure should go if redeployed
- whether the simpler baseline agrees with the live recommendation layer

## Conclusion

v13.1 improves usefulness now, without promoting a new model stack.

The promoted recommendation-layer default is now:

- simpler diversification-first baseline is the active recommendation layer
- live 4-model production stack remains visible as a diagnostic cross-check
- existing-holdings lot guidance and redeploy guidance remain part of the
  production report and email
