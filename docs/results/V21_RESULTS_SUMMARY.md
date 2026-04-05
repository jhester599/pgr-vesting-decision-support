# V21 Results Summary

Created: 2026-04-04

## Scope

- v21 replaces the recent-window shadow gate with a point-in-time historical comparison over the full evaluable period.
- It compares the current live reduced cross-check and the leading v16-v20 assembled candidates against the promoted simpler baseline.

## Historical Window

- Common evaluable monthly dates: `108`
- First common date: `2016-10-31`
- Last common date: `2025-09-30`

## Decision

- Status: `promote_candidate_cross_check`
- Recommended path: `ensemble_ridge_gbt_v18`
- Rationale: The best v21 candidate improved on the reduced live stack and matched or exceeded the live cross-check's agreement with the promoted simpler baseline over the full historical window.

## Top Metric Row

- Candidate: `ensemble_ridge_gbt_v18`
- Mean sign-policy return: `0.0771`
- Mean OOS R^2: `-0.1991`
- Mean IC: `0.2232`
- Mean hit rate: `0.6606`

## Current Live Historical Behavior

- Signal agreement with shadow baseline: `64.8%`
- Signal changes: `18`
- Outperform / neutral / underperform mix: `64.8%` / `25.9%` / `9.3%`

## Recommended-Path Historical Behavior

- Path: `ensemble_ridge_gbt_v18`
- Signal agreement with shadow baseline: `81.5%`
- Signal agreement with live cross-check: `70.4%`
- Signal changes: `16`
- Outperform / neutral / underperform mix: `81.5%` / `13.0%` / `5.6%`

## Output Artifacts

- `results/v21/v21_candidate_metric_summary_20260404.csv`
- `results/v21/v21_historical_review_detail_20260404.csv`
- `results/v21/v21_historical_review_summary_20260404.csv`
- `results/v21/v21_slice_summary_20260404.csv`
- `results/v21/v21_promotion_decision_20260404.csv`
- `results/v21/v21_model_manifest_20260404.csv`
