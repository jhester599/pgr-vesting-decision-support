# codex-v20-plan

Created: 2026-04-05

## Goal

- Build one best-of-v19 candidate stack from the strongest confirmed swaps and judge whether it is promotion-ready.

## Candidate Stacks

- `ensemble_ridge_gbt_v16`
- `ensemble_ridge_gbt_v18`
- `ensemble_ridge_gbt_v20_value`
- `ensemble_ridge_gbt_v20_best`
- `ensemble_ridge_gbt_v20_usd`
- `ensemble_ridge_gbt_v20_pricing`

## Comparators

- `live_production_ensemble_reduced`
- `baseline_historical_mean`
- promoted simpler shadow baseline via monthly review

## Gate

- Promote only if the best assembled stack improves metrics and behaves at least as cleanly as the current cross-check versus the promoted simpler baseline.
