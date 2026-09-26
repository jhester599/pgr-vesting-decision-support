# V16 Results Summary

Created: 2026-04-04

## Scope

- v16 is a narrow promotion study, not a new feature search.
- It tests the two confirmed v15 replacements inside the leading Ridge+GBT candidate stack.
- The recommendation layer remains fixed at the promoted v13.1 path.

## Forecast Universe

- Reduced universe: `VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE`

## Promotion Decision

- Status: `shadow_for_v17`
- Candidate: `ensemble_ridge_gbt_v16`
- Rationale: The modified Ridge+GBT pair improved on the reduced live stack and stayed close to the historical-mean baseline, but it did not clear a strong enough edge for direct promotion.

## Top Row

- Candidate: `ensemble_ridge_gbt_v16`
- Type: `ensemble`
- Mean sign-policy return: `0.0748`
- Mean neutral-band return: `0.0733`
- Mean OOS R^2: `-0.1950`
- Mean IC: `0.2018`

## Output Artifacts

- `results/v16/v16_candidate_bakeoff_detail_20260404.csv`
- `results/v16/v16_candidate_bakeoff_summary_20260404.csv`
- `results/v16/v16_promotion_decision_20260404.csv`
