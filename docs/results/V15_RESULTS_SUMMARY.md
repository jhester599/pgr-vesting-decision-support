# V15 Results Summary

Created: 2026-04-04

## Status

v15 is no longer just setup. The report review, candidate normalization, queue generation, and the full `v15.0` through `v15.2` execution cycle are complete for the features that are available in the repo today.

## What Was Reviewed

External reports archived in:

- `docs/history/v15-research-reports/v15_chatgptdeepresearch_20260404.md`
- `docs/history/v15-research-reports/v15_geminideepresearch_20260404.md`
- `docs/history/v15-research-reports/v15_geminipro_20260404.md`
- `docs/history/v15-research-reports/v15_claudeopusresearch_20260404.md`

Their synthesis is in:

- `docs/plans/codex-v15-feature-test-plan.md`

## What Was Added

### New v15 feature-engineering coverage

The repo now computes several new v15 candidate features from already-available DB inputs, including:

- `rate_adequacy_gap_yoy`
- `severity_index_yoy`
- `monthly_combined_ratio_delta`
- `pif_growth_acceleration`
- `npw_per_pif_yoy`
- `npw_vs_npe_spread_pct`
- `underwriting_margin_ttm`
- `book_value_per_share_growth_yoy`
- `duration_rate_shock_3m`
- `baa10y_spread`
- `breakeven_inflation_10y`
- `breakeven_momentum_3m`
- `real_yield_change_6m`

### New v15 execution artifacts

- `results/v15/v15_feature_candidate_inventory_from_reports_20260404.csv`
- `results/v15/v15_feature_inventory_normalized_20260404.csv`
- `results/v15/v15_swap_queue_20260404.csv`
- `results/v15/v15_0_core_summary_20260404.csv`
- `results/v15/v15_1_confirmation_summary_20260404.csv`
- `results/v15/v15_2_bakeoff_summary_20260404.csv`

### New v15 execution code

- `src/research/v15.py`
- `scripts/v15_feature_replacement_setup.py`
- `scripts/v15_execute.py`

## Stage Results

### v15.0 — Core Screening On Ridge And GBT

Best GBT results:

- `rate_adequacy_gap_yoy` replacing `vmt_yoy`
  - mean sign-policy uplift: `+0.0057`
  - mean OOS R² delta: `+0.1174`
- `breakeven_momentum_3m` replacing `vol_63d`
  - mean sign-policy uplift: `+0.0034`
  - mean OOS R² delta: `+0.0132`

Best Ridge results:

- `book_value_per_share_growth_yoy` replacing `roe_net_income_ttm`
  - mean sign-policy uplift: `+0.0008`
  - mean OOS R² delta: `+0.1272`
- `npw_vs_npe_spread_pct` replacing `npw_growth_yoy`
  - mean sign-policy uplift: `+0.0002`
  - mean OOS R² delta: `+0.2275`
- `underwriting_margin_ttm` replacing `combined_ratio_ttm`
  - near-flat policy uplift, but positive OOS R² delta

Main takeaway:

- the strongest single new insurance-economic feature was `rate_adequacy_gap_yoy`
- the strongest balance-sheet replacement was `book_value_per_share_growth_yoy`
- the reports were directionally right that benchmark-aware and insurer-native replacements can beat several older generic features

### v15.1 — Cross-Model Confirmation

The most stable cross-model winner was:

- `book_value_per_share_growth_yoy` replacing `roe_net_income_ttm`
  - best for `elasticnet`
  - best for `bayesian_ridge`
  - also positive for `ridge`

The clearest GBT-specific winner remained:

- `rate_adequacy_gap_yoy` replacing `vmt_yoy`

Other confirmed positives:

- `breakeven_momentum_3m` improved `bayesian_ridge` and GBT
- `medical_cpi_yoy` modestly improved `elasticnet`

Main takeaway:

- one replacement did not win everywhere
- the best pattern was model-specific:
  - linear models liked `book_value_per_share_growth_yoy`
  - the tree model liked `rate_adequacy_gap_yoy`

### v15.2 — Final Cross-Model Bakeoff

Top results:

1. `gbt_lean_plus_two__v15_best`
   - replacement: `rate_adequacy_gap_yoy` for `vmt_yoy`
   - mean sign-policy return: `0.0741`
   - mean OOS R²: `-0.2355`
   - mean IC: `0.1672`

2. `ridge_lean_v1__v15_best`
   - replacement: `book_value_per_share_growth_yoy` for `roe_net_income_ttm`
   - mean sign-policy return: `0.0728`
   - mean OOS R²: `-0.5203`
   - mean IC: `0.1927`

3. baseline `ridge_lean_v1`
   - mean sign-policy return: `0.0720`

4. `baseline_historical_mean`
   - mean sign-policy return: `0.0704`
   - mean OOS R²: `-0.2047`

## Key Conclusions

1. v15 found real feature improvements without increasing feature count materially.
2. The best new GBT feature is `rate_adequacy_gap_yoy`.
3. The best new linear-model feature is `book_value_per_share_growth_yoy`.
4. The v15 best GBT variant now beats the `historical_mean` policy baseline on mean sign-policy return.
5. OOS R² is still negative, so this is progress, not a final production-promotion verdict.
6. The feature problem is not solved, but the feature set was part of the problem.

## Recommended Next Step

Do not change the recommendation layer.

The best next research step is:

- test a modified `ridge + gbt` ensemble using the v15-confirmed replacements
- compare that combined stack directly against:
  - current live prediction stack
  - current v13.1 recommendation-layer output
  - `historical_mean`

That is the natural follow-on promotion study after v15.
