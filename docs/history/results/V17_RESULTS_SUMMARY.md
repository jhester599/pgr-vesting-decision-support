# V17 Results Summary

Created: 2026-04-04

## Scope

- v17 tests whether the modified Ridge+GBT pair should replace the current live production stack as the visible cross-check under the promoted v13.1 recommendation layer.
- The active recommendation layer remains the simpler diversification-first baseline.

## Review Window

- Monthly snapshots reviewed: `12`
- End as-of date: `2026-04-04`

## Promotion Decision

- Status: `keep_current_live_cross_check`
- Recommended path: `live_production`
- Rationale: The modified Ridge+GBT candidate improved the reduced-universe metrics, but it did not behave clearly enough versus the current live cross-check over recent monthly snapshots to justify replacing the current cross-check path.

## Review Summary

### shadow_baseline

- Signal agreement with shadow baseline: `100.0%`
- Recommendation-mode agreement with shadow baseline: `100.0%`
- Sell agreement with shadow baseline: `100.0%`
- Signal changes: `0`
- Mode changes: `0`
- Mean aggregate OOS R^2: `-0.1565`

### live_production

- Signal agreement with shadow baseline: `16.7%`
- Recommendation-mode agreement with shadow baseline: `100.0%`
- Sell agreement with shadow baseline: `100.0%`
- Signal changes: `5`
- Mode changes: `0`
- Mean aggregate OOS R^2: `-1.1015`

### candidate_v16

- Signal agreement with shadow baseline: `0.0%`
- Recommendation-mode agreement with shadow baseline: `100.0%`
- Sell agreement with shadow baseline: `100.0%`
- Signal changes: `0`
- Mode changes: `0`
- Mean aggregate OOS R^2: `-0.1980`

## Output Artifacts

- `results/v17/v17_shadow_review_detail_20260404.csv`
- `results/v17/v17_shadow_review_summary_20260404.csv`
- `results/v17/v17_promotion_decision_20260404.csv`
