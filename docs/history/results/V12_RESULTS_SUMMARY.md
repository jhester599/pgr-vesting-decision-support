# V12 Results Summary

Created: 2026-04-04

## Shadow Study Goal

Test whether a simpler diversification-first baseline should replace the live monthly decision engine in practice, even before any new model stack is promoted.

## Selected Universes

- Forecast universe: `VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE, VFH`
- Recommended diversification universe: `VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE`

## Candidate Scoreboard

- Best candidate by diversification-aware utility: `ensemble_ridge_gbt`
- Best policy row: `baseline_historical_mean` with `neutral_band_3pct`

## Review Window Findings

- Review months evaluated: `12`
- Average live sell percentage: `50%`
- Average shadow sell percentage: `50%`
- Live signal changes: `5`
- Shadow signal changes: `0`
- Live recommendation-mode changes: `0`
- Shadow recommendation-mode changes: `0`
- Shadow redeploy universe mean diversification score: `0.459`

## Interpretation

- v12 is intentionally testing a simpler path, not a more complex one.
- The shadow baseline inherits the diversification-first redeploy logic and the clearer lot-trimming order from v11.
- The live stack changed its directional signal repeatedly across the review window, but the action never moved off the 50% default because the quality gate still failed.
- The shadow baseline produced a steadier directional story while preserving the same diversification-first action and adding clearer redeploy guidance.

Detailed artifacts are stored in `results\v12`.
