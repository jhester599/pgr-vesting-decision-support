# codex-v15-plan.md

Created: 2026-04-04

## Goal

Run a fixed-budget feature-replacement cycle on the v14 leading replacement candidate without expanding model complexity or feature count materially.

## Starting Point

- Research status: `phase_planned`
- Forecast universe: `VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE`
- Leading prediction-layer candidate from v14: `ensemble_ridge_gbt`
- Working model baselines:
  - ridge: `mom_12m, vol_63d, yield_slope, yield_curvature, real_rate_10y, credit_spread_hy, nfci, vix, combined_ratio_ttm, investment_income_growth_yoy, roe_net_income_ttm, npw_growth_yoy`
  - gbt: `mom_3m, mom_6m, mom_12m, vol_63d, yield_slope, yield_curvature, real_rate_10y, credit_spread_hy, nfci, vix, vmt_yoy, pif_growth_yoy, investment_book_yield, underwriting_income_growth_yoy`

## Rules

- keep the v13.1 recommendation layer fixed
- test one replacement at a time
- prefer one-for-one swaps
- only consider broader changes after the one-for-one queue is exhausted
- include both PGR-specific features and benchmark-predictive/shared-regime features

## Execution Stages

- `v15.0`
  - exhaustive one-feature-at-a-time screening on `ridge_lean_v1` and `gbt_lean_plus_two`
- `v15.1`
  - retest the v15.0 winners across all deployed model types:
    - `elasticnet`
    - `ridge`
    - `bayesian_ridge`
    - `gbt`
- `v15.2`
  - final cross-model bakeoff comparing:
    - baseline deployed models
    - best-confirmed modified models
    - `baseline_historical_mean`

## Current Status

- setup: complete
- external report review: complete
- canonical inventory + swap queue: complete
- `v15.0`: complete
- `v15.1`: complete
- `v15.2`: complete

## Setup Artifacts

- candidate inventory template: `results\v15\feature_candidate_inventory_template.csv`
- existing baseline feature inventory: `results/v15/v15_existing_feature_inventory_20260404.csv`
- currently available feature coverage: `results/v15/v15_available_feature_coverage_20260404.csv`
- generated swap queue: `results\v15\v15_swap_queue_20260404.csv`
