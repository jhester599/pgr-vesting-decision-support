# V16 Closeout And V17 Next

Created: 2026-04-04

## Closeout

- v16 completed the narrow promotion study recommended at the end of v15.
- The study compared the modified Ridge+GBT pair against the reduced-universe live stack and the historical-mean baseline.

## Result

- Promotion status: `shadow_for_v17`
- Candidate reviewed: `ensemble_ridge_gbt_v16`
- Decision rationale: The modified Ridge+GBT pair improved on the reduced live stack and stayed close to the historical-mean baseline, but it did not clear a strong enough edge for direct promotion.

## Recommended V17 Scope

- If v16 does not promote, keep the live prediction layer unchanged and continue with a narrow v17 feature phase focused only on the highest-value deferred families.
- If v16 does promote, implement the Ridge and GBT feature swaps without changing the recommendation layer.
- In either case, keep the v13.1 recommendation layer in place until a later study proves that a new prediction stack improves real usefulness as well as metrics.
