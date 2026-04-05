# V21 Results Summary

Created: 2026-04-05

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
- Mean sign-policy return: `0.0798`
- Mean OOS R^2: `-0.1424`
- Mean IC: `0.1911`
- Mean hit rate: `0.6857`

## Current Live Historical Behavior

- Signal agreement with shadow baseline: `63.0%`
- Signal changes: `16`
- Outperform / neutral / underperform mix: `63.0%` / `25.0%` / `12.0%`

## Recommended-Path Historical Behavior

- Path: `ensemble_ridge_gbt_v18`
- Signal agreement with shadow baseline: `84.3%`
- Signal agreement with live cross-check: `73.1%`
- Signal changes: `15`
- Outperform / neutral / underperform mix: `84.3%` / `8.3%` / `7.4%`

## Output Artifacts

- `results/v21/v21_candidate_metric_summary_20260405.csv`
- `results/v21/v21_historical_review_detail_20260405.csv`
- `results/v21/v21_historical_review_summary_20260405.csv`
- `results/v21/v21_slice_summary_20260405.csv`
- `results/v21/v21_promotion_decision_20260405.csv`
- `results/v21/v21_model_manifest_20260405.csv`
