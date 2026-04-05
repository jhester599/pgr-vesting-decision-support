# V20 Results Summary

Created: 2026-04-04

## Scope

- v20 is a narrow synthesis and promotion-readiness study.
- It assembles the strongest confirmed v16-v19 swaps into a small set of Ridge+GBT replacement stacks.
- It compares those stacks against the reduced live production cross-check, the historical-mean baseline, and the promoted simpler baseline in a monthly shadow review.

## Forecast Universe

- Reduced universe: `VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE`

## Promotion Decision

- Status: `continue_research_keep_current_cross_check`
- Recommended candidate: `ensemble_ridge_gbt_v18`
- Rationale: The best assembled v20 stack improved reduced-universe metrics, but it still diverged too much from the promoted simpler baseline to replace the current live cross-check.

## Top Metric Row

- Candidate: `ensemble_ridge_gbt_v18`
- Mean sign-policy return: `0.0771`
- Mean neutral-band return: `0.0761`
- Mean OOS R^2: `-0.1991`
- Mean IC: `0.2232`
- Mean hit rate: `0.6606`

## Best-Candidate Review Behavior

- Path: `ensemble_ridge_gbt_v18`
- Signal agreement with shadow baseline: `0.0%`
- Mode agreement with shadow baseline: `100.0%`
- Signal changes: `0`
- Underperform share: `100.0%`
- Outperform share: `0.0%`
- Neutral share: `0.0%`

## Output Artifacts

- `results/v20/v20_candidate_metric_detail_20260404.csv`
- `results/v20/v20_candidate_metric_summary_20260404.csv`
- `results/v20/v20_shadow_review_detail_20260404.csv`
- `results/v20/v20_shadow_review_summary_20260404.csv`
- `results/v20/v20_promotion_decision_20260404.csv`
- `results/v20/v20_candidate_manifest_20260404.csv`
