# V18 Closeout And V19 Next

Created: 2026-04-04

## Closeout

- v18 tested narrow benchmark-side and peer-relative swaps on the modified Ridge+GBT candidate stack.

## Result

- Status: `keep_v16_as_research_only`
- Recommended candidate: `ensemble_ridge_gbt_v16`
- Rationale: The benchmark-side and peer-relative swaps did not reduce the candidate's directional bias against the promoted simpler baseline enough to justify another promotion attempt.

## Recommended V19 Scope

- If v18 advances, run one more narrow promotion gate on the v18 candidate.
- If v18 does not advance, keep the current production paths unchanged and return only to the highest-value deferred families that remain executable with existing data.
