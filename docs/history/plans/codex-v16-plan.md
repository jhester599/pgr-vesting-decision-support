# codex-v16-plan

Created: 2026-04-04

## Goal

- Run a narrow promotion study on the best v15 feature-replacement candidate rather than reopening the feature search.

## Candidate Stack

- Base candidate: `ensemble_ridge_gbt_v14`
- Modified candidate: `ensemble_ridge_gbt_v16`
- Modified swaps:
  - Ridge: `book_value_per_share_growth_yoy` replacing `roe_net_income_ttm`
  - GBT: `rate_adequacy_gap_yoy` replacing `vmt_yoy`

## Comparators

- `live_production_ensemble_reduced`
- `baseline_historical_mean`
- individual modified and unmodified Ridge / GBT rows

## Promotion Gate

- Promote only if the modified pair clearly beats the reduced live stack and also separates enough from the historical-mean baseline.
- Otherwise keep the v13.1 recommendation layer and current live prediction layer in production.
