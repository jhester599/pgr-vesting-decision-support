# V9 Results Summary

Created: 2026-04-03  
Author: Codex  
Scope: Summary of the completed v9 research workstreams, results, and recommended next steps

## Overview

v9 focused on the underlying predictive weakness of the project rather than on
adding more methodology complexity. The goal was to answer a simpler question:

> What is the simplest target, benchmark universe, and feature stack that
> produces the most stable and decision-useful out-of-sample behavior?

That required building a dedicated research harness and then using it to test:

- current production models against naive baselines
- individual features by model type
- alternate target formulations
- reduced benchmark universes
- regime slices
- policy-level decision utility
- pooled benchmark-family targets
- lean model candidates derived from the per-feature results

The short answer is:

- the current weakness still appears structural, not architectural
- reducing benchmark breadth helps
- smaller, deliberate feature sets help some models materially
- pooled targets do not fix the problem
- sign-based decision rules outperform the more conservative tiered action map
- the best near-term path is a simpler production candidate on a reduced
  benchmark universe, not a deeper model family

## What Was Implemented

### Research infrastructure

New shared helpers:

- `src/research/evaluation.py`
- `src/research/benchmark_sets.py`
- `src/research/policy_metrics.py`

New scripts:

- `scripts/benchmark_suite.py`
- `scripts/feature_cost_report.py`
- `scripts/feature_experiments.py`
- `scripts/target_experiments.py`
- `scripts/benchmark_reduction.py`
- `scripts/regime_slice_backtest.py`
- `scripts/policy_evaluation.py`
- `scripts/pooled_benchmark_experiments.py`
- `scripts/candidate_model_bakeoff.py`

Production-safe research hook:

- `src/models/wfo_engine.py`
  - added explicit `feature_columns` support so research experiments can test
    custom feature subsets without changing production defaults

New tests:

- `tests/test_benchmark_suite.py`
- `tests/test_feature_cost_report.py`
- `tests/test_feature_experiments.py`
- `tests/test_target_experiments.py`
- `tests/test_benchmark_reduction.py`
- `tests/test_regime_slice_backtest.py`
- `tests/test_policy_evaluation.py`
- `tests/test_pooled_benchmark_experiments.py`
- `tests/test_candidate_model_bakeoff.py`
- `tests/test_wfo_engine.py`

### Outputs generated

The v9 outputs were written to `results/v9/`, including:

- `benchmark_suite_summary_20260403.csv`
- `feature_cost_report_20260403.csv`
- `feature_set_health_20260403.csv`
- `feature_experiments_summary_20260403.csv`
- `target_experiments_summary_20260403.csv`
- `benchmark_reduction_scorecard_20260403.csv`
- `benchmark_reduction_candidate_summary_20260403.csv`
- `regime_slice_summary_20260403.csv`
- `policy_evaluation_summary_full21_20260403.csv`
- `policy_evaluation_summary_balanced_core7_20260403.csv`
- `pooled_benchmark_experiments_summary_20260403.csv`
- `candidate_model_bakeoff_summary_20260403.csv`
- `weekly_snapshot_experiments_detail_20260403.csv`
- `weekly_snapshot_experiments_summary_20260403.csv`

## Key Findings

## 1. The main problem is still target fit / calibration, not lack of model complexity

From `benchmark_suite_summary_20260403.csv`:

- `bayesian_ridge` had the highest mean IC at roughly `0.1885`
- but it also had catastrophic mean OOS R² near `-56.47%`
- the current ensemble had better balance, but still failed badly:
  - mean IC about `0.1296`
  - mean hit rate about `57.7%`
  - mean OOS R² about `-0.708`
- the simple `historical_mean` baseline had worse IC, but better mean OOS R²
  than the current production ensemble at about `-0.169`

Interpretation:

- the project seems to retain some ranking signal
- but the magnitude forecasts remain badly misspecified
- adding more expressive model classes would likely amplify variance before it
  solves the real problem

## 2. Feature cost still matters materially

From `feature_set_health_20260403.csv`:

- `gbt` current feature set:
  - `11` features
  - `259` fully observed rows
  - strongest completeness profile among the production models
- `elasticnet` and `ridge`:
  - `14` features
  - `217` fully observed rows
- `bayesian_ridge`:
  - `16` features
  - `208` fully observed rows

Weak-coverage features included:

- `investment_book_yield`
- `roe`
- `roe_trend`
- `pif_growth_yoy`
- `roe_net_income_ttm`
- `unearned_premium_growth_yoy`
- `npw_growth_yoy`

Interpretation:

- some production candidates are still paying a real sample-size tax
- this helps explain why simpler GBT variants often remain more stable

## 3. Individual feature tests were useful and pointed to concrete add/drop moves

The per-feature experiments were run deliberately by model type.

From `feature_experiments_summary_20260403.csv`:

### ElasticNet

Best add-one features:

- `cr_acceleration`
- `medical_cpi_yoy`
- `roe`
- `npw_growth_yoy`

Features that looked weaker inside the current baseline:

- `vol_63d`
- `real_rate_10y`
- `vmt_yoy`

### Ridge

Best add-one features:

- `npw_growth_yoy`
- `high_52w`

Most useful simplification signals:

- dropping `mom_3m`
- dropping `mom_6m`
- dropping `vmt_yoy`

### GBT

Best add-one features:

- `pif_growth_yoy`
- `investment_book_yield`
- `underwriting_income_growth_yoy`
- `investment_income_growth_yoy`

### Bayesian Ridge

Some add-one IC improvements appeared:

- `npw_growth_yoy`
- `roe`

But the model remained too unstable in OOS R² terms to trust as a production
candidate without a much stronger simplification step.

## 4. Binary targets did not solve the problem

From `target_experiments_summary_20260403.csv`:

- `binary_outperform` and `binary_outperform_3pct` did not rescue the model stack
- the ensemble still failed badly on OOS R² under the binary formulations
- the `historical_mean` baseline often had better practical error behavior than
  the ML models even when the ML models had stronger IC

Interpretation:

- the problem is not simply “make it classification”
- target reformulation alone is not enough

## 5. Benchmark reduction helps meaningfully, but not enough by itself

From `benchmark_reduction_candidate_summary_20260403.csv`:

Full 21-benchmark ensemble:

- mean IC: `0.1296`
- mean OOS R²: `-0.7079`

Best reduced candidate, `balanced_core7`:

- benchmarks: `VXUS, VEA, VHT, VPU, BNDX, BND, VNQ`
- ensemble mean IC: `0.1596`
- ensemble mean hit rate: `62.3%`
- ensemble mean OOS R²: `-0.4065`

Interpretation:

- reducing benchmark noise helped materially
- but the reduced universe still failed the research gate
- benchmark breadth is part of the problem, not the whole problem

## 6. Signal quality is regime-sensitive

From `regime_slice_summary_20260403.csv`:

Ensemble behavior was materially better in calmer regimes than in stressed ones:

- `low_vix`:
  - mean IC about `0.267`
  - hit rate about `60.2%`
  - OOS R² about `-0.468`
- `high_vix`:
  - mean IC about `0.080`
  - hit rate about `50.0%`
  - OOS R² about `-1.829`

Trailing 36 months were especially weak:

- ensemble trailing-36m IC only about `0.059`
- hit rate about `50.3%`
- OOS R² about `-0.799`

Interpretation:

- the signal is real enough to show a regime shape
- but recent / stressed-market usefulness is not strong enough
- any next production candidate should likely be more conservative in weak
  regimes rather than trying to predict through them aggressively

## 7. Policy-level evaluation changed the story in a useful way

From `policy_evaluation_summary_full21_20260403.csv`:

On the full universe, the best average decision utility came from:

- `baseline_historical_mean` with `sign_hold_vs_sell`
  - mean policy return about `0.0601`
- then `gbt` with `sign_hold_vs_sell`
  - mean policy return about `0.0586`
- then the ensemble with `sign_hold_vs_sell`
  - mean policy return about `0.0535`

The tiered action rule underperformed the simple sign-based rule across the
same predictors.

On the reduced `balanced_core7` universe from
`policy_evaluation_summary_balanced_core7_20260403.csv`:

- `baseline_historical_mean` with `sign_hold_vs_sell`
  - mean policy return about `0.0718`
- `gbt` with `sign_hold_vs_sell`
  - mean policy return about `0.0633`
- ensemble with `sign_hold_vs_sell`
  - mean policy return about `0.0627`

Interpretation:

- policy evaluation confirmed that benchmark reduction improves practical
  decision utility
- it also showed that the current tiered policy map is too conservative /
  overcomplicated relative to the actual signal quality
- the best decisions so far come from very simple “positive vs negative” rules

## 8. Pooled benchmark-family targets did not create a clean breakthrough

From `pooled_benchmark_experiments_summary_20260403.csv`:

Better pooled slices did exist:

- `fixed_income`
- `sector`
- `defensive_assets`
- `real_asset`

But they still failed the gate:

- good IC in some cases
- still negative OOS R² almost everywhere
- Bayesian Ridge still unstable even when pooled

Interpretation:

- pooled targets are informative research diagnostics
- but they are not the main unlock

## 9. Lean candidate bakeoff produced the most actionable result of v9

From `candidate_model_bakeoff_summary_20260403.csv` on `balanced_core7`:

Top non-baseline candidates:

1. `ridge_lean_v1`
   - `12` features
   - mean IC: `0.1980`
   - mean hit rate: `66.4%`
   - mean OOS R²: `-0.4282`
   - sign-policy mean return: `0.0679`

2. `gbt_lean_plus_two`
   - `13` features
   - mean IC: `0.1770`
   - mean OOS R²: `-0.3545`
   - sign-policy mean return: `0.0654`

3. `elasticnet_lean_v1`
   - `13` features
   - mean IC: `0.1733`
   - mean OOS R²: `-0.4541`
   - sign-policy mean return: `0.0612`

Compared with current production versions:

- `ridge_current`
  - sign-policy mean return: `0.0528`
- `gbt_current`
  - sign-policy mean return: `0.0633`
- `elasticnet_current`
  - sign-policy mean return: `0.0611`

Interpretation:

- Ridge improved the most from simplification
- GBT improved modestly but consistently from adding `pif_growth_yoy` and
  `investment_book_yield`
- ElasticNet improved strongly on predictive metrics and modestly on policy utility
- Bayesian Ridge improved when simplified, but still remained too unstable to
  justify promotion

## 10. Weekly snapshot carry-forward experiments were feasible, but not a breakthrough

To test whether observation count was a major bottleneck, v9 also added a
point-in-time safe weekly snapshot pilot in
`weekly_snapshot_experiments_summary_20260403.csv`.

Important design choice:

- this did not linearly interpolate EDGAR or FRED data
- monthly state was carried forward as-of each weekly snapshot
- price-sensitive features were recomputed at weekly cadence
- forward targets remained 6M relative total returns versus `balanced_core7`

Best weekly results:

- `elasticnet_lean_v1`
  - sign-policy mean return: `0.0705`
  - mean IC: `0.1571`
  - mean hit rate: `64.9%`
  - mean OOS R2: `-0.5347`
- `ridge_current`
  - sign-policy mean return: `0.0663`
  - mean IC: `0.1652`
  - mean hit rate: `64.4%`
  - mean OOS R2: `-0.4170`
- `baseline_historical_mean`
  - sign-policy mean return: `0.0662`
  - mean OOS R2: `-0.2904`

Compared with the monthly reduced-universe bakeoff:

- the weekly pilot did not beat the monthly `baseline_historical_mean`
  sign-policy return of `0.0718`
- weekly OOS R2 generally became worse, not better
- only three sparse candidate / benchmark pairs were skipped, so this is not a
  harness failure story

Interpretation:

- increasing row count via higher-frequency snapshots is technically possible
- but more rows did not translate into cleaner magnitude fit
- the project still appears constrained more by target quality / calibration
  than by raw monthly sample count alone
- weekly carry-forward snapshots are therefore not the next best promotion path
  right now

## What Changed Most During v9

The most important shifts from the start of v9 to the current state are:

1. The repo now has a proper research harness instead of one-off ablation work.
2. Policy utility is now measured directly, not inferred from IC alone.
3. Benchmark reduction is now evidence-based rather than intuitive.
4. Feature simplification is now tied to deliberate add/drop results per model.
5. The strongest near-term candidate is no longer “current production ensemble”.
6. The evidence now supports a simpler next-step production candidate rather than
   a more complex architecture.
7. Higher-frequency carry-forward snapshots did not provide enough incremental
   value to justify moving away from the monthly cadence yet.

## What Did Not Work

The following did not solve the core problem:

- switching to binary targets
- pooled family targets as a standalone fix
- Bayesian Ridge as a production-leading model
- using the full 21-benchmark universe
- relying on the current tiered action map as the best policy layer
- increasing effective row count via weekly carry-forward snapshots

## Recommended Next Steps

## 1. Promote a v9.1 candidate research branch around `balanced_core7`

The strongest evidence now points to:

- benchmark universe: `balanced_core7`
- action policy: sign-based rather than current tiered research policy
- model shortlist:
  - `ridge_lean_v1`
  - `gbt_lean_plus_two`
  - optionally `elasticnet_lean_v1`

These should become the next formal promotion candidates.

## 2. Keep Bayesian Ridge as research-only for now

It still shows attractive IC, but the OOS R² instability remains too large.
It should not lead the next production candidate.

## 3. Rebuild the ensemble candidate from only the winners

The next ensemble candidate should not simply inherit all four current models.
It should instead test:

- `ridge_lean_v1`
- `gbt_lean_plus_two`
- optionally `elasticnet_lean_v1`

with the reduced benchmark universe.

## 4. Revisit decision mapping before revisiting model class

The current evidence says:

- sign-based policies are stronger than the tiered research mapping
- the monthly decision layer should probably become more binary when research
  quality is only moderate

That is a better next step than TFT or RL.

## 5. Only consider a new architecture after the lean candidate is tested

If the reduced-universe lean-candidate ensemble still fails:

- do not jump to deep RL
- do not jump directly to TFT
- first confirm whether the simplified candidate still beats the historical-mean
  baseline on policy utility and OOS R²

Only if the lean candidate clearly stabilizes should a more complex model be
considered as a controlled research branch.

## 6. Do not prioritize daily or every-third-day interpolation work yet

The weekly pilot was the lowest-risk version of the higher-frequency idea:

- point-in-time safe carry-forward of low-frequency state
- recomputed market features
- reduced benchmark universe

It still did not beat the best monthly baseline.

That means the next best step is not:

- daily interpolation of monthly fundamentals
- every-third-day live scheduling
- additional data-cadence complexity

Those ideas can remain in reserve, but they should not take priority over the
lean reduced-universe production-candidate tests.

## Recommended Promotion Decision

Current recommendation:

- Do not promote a v9 candidate to production yet.
- Do promote the following to “next production-candidate research” status:
  - `balanced_core7`
  - `ridge_lean_v1`
  - `gbt_lean_plus_two`
  - optional `elasticnet_lean_v1`
- Treat `baseline_historical_mean` sign-policy performance as the benchmark to beat.

## Residual Cleanup Items

The v9 runs surfaced a few technical cleanup items:

- `feature_engineering.py`
  - explicit `fill_method=None` should be passed to `pct_change()` to remove the
    deprecation warning
- research runs with sparse features can still trigger fold-level all-NaN median
  warnings in `wfo_engine.py`
- some ElasticNet runs still emit convergence warnings during exhaustive sweeps

These do not invalidate the research results, but they are worth tightening in
the next pass.

## Bottom Line

v9 strongly supports a simplification-first path.

The best next candidate is not:

- a bigger ensemble
- a deeper model
- a more complex target

The best next candidate is:

- a reduced benchmark universe
- a leaner Ridge / GBT-centered stack
- a simpler sign-based decision policy
- explicit comparison against the historical-mean baseline

That is now the highest-probability path to improving actual decision usefulness.
