# codex-v21-plan

Created: 2026-04-05

## Goal

- Re-evaluate the leading post-v19 candidate stacks over the full historically evaluable period instead of the recent 12-month window.

## Paths Compared

- `shadow_baseline`
- `live_production_ensemble_reduced`
- `ensemble_ridge_gbt_v16`
- `ensemble_ridge_gbt_v18`
- `ensemble_ridge_gbt_v20_value`
- `ensemble_ridge_gbt_v20_best`

## Gate

- Prefer a candidate only if it improves metrics and matches or exceeds the live path's historical agreement with the promoted simpler baseline.
