# V18 Results Summary

Created: 2026-04-04

## Scope

- v18 focuses on reducing the modified candidate's directional bias against the promoted simpler baseline.
- It only tests narrow benchmark-side and peer-relative one-for-one swaps on the v16 Ridge+GBT pair.

## Best Swaps

- `gbt_lean_plus_two__v16`: `vwo_vxus_spread_6m` for `real_rate_10y` (policy delta `+0.0041`, OOS R^2 delta `+0.0350`)
- `ridge_lean_v1__v16`: `real_yield_change_6m` for `yield_curvature` (policy delta `+0.0023`, OOS R^2 delta `-0.1507`)

## Final Decision

- Status: `keep_v16_as_research_only`
- Recommended candidate: `ensemble_ridge_gbt_v16`
- Rationale: The benchmark-side and peer-relative swaps did not reduce the candidate's directional bias against the promoted simpler baseline enough to justify another promotion attempt.

## Output Artifacts

- `results/v18/v18_core_swap_summary_20260404.csv`
- `results/v18/v18_best_swaps_20260404.csv`
- `results/v18/v18_candidate_metric_summary_20260404.csv`
- `results/v18/v18_shadow_review_summary_20260404.csv`
- `results/v18/v18_decision_20260404.csv`
