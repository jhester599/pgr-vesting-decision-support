# Model Governance

## Purpose

This document defines the boundary between:

- production modeling behavior
- research/evaluation work
- promotion decisions

## Current Production Baseline

The current production monthly decision workflow remains the v8.13 4-model
ensemble with model-quality gating and tax-aware reporting.

The monthly workflow is the only place where a model stack becomes operational.

## Current Research Baseline

v9 research is complete and documented in:

- `docs/plans/codex-v9-plan.md`
- `docs/results/V9_RESULTS_SUMMARY.md`
- `docs/closeouts/V9_CLOSEOUT_AND_V91_NEXT.md`

It introduced:

- benchmark harnesses
- feature-cost and feature-selection studies
- target reformulation experiments
- benchmark-reduction studies
- policy evaluation
- weekly snapshot experiments
- a tuned Ridge classifier-sidecar candidate

v11 research is now also complete and documented in:

- `docs/plans/codex-v11-plan.md`
- `docs/results/V11_RESULTS_SUMMARY.md`
- `docs/closeouts/V11_CLOSEOUT_AND_V12_NEXT.md`

It added:

- diversification-aware benchmark scoring relative to PGR
- explicit separation between forecast benchmarks and redeployment destinations
- diversification-first universe reduction
- reduced-universe candidate bakeoffs
- policy redesign around simpler sell / hold / neutral logic
- production-like dry-run recommendation memos with lot-level guidance

v12 shadow-promotion work is now also complete and documented in:

- `docs/plans/codex-v12-plan.md`
- `docs/results/V12_RESULTS_SUMMARY.md`
- `docs/closeouts/V12_CLOSEOUT_AND_V13_NEXT.md`

It added:

- a rolling 12-month shadow comparison between the live monthly stack and the
  simpler diversification-first baseline
- side-by-side dry-run memos under `results/v12/dry_runs/`
- an explicit check for whether recommendation-layer simplification should
  happen before any new model stack promotion

v13 production-facing recommendation-layer work is now also documented in:

- `docs/plans/codex-v13-plan.md`
- `docs/results/V13_RESULTS_SUMMARY.md`

It adds:

- a recommendation-layer mode switch with `live_only`, `live_with_shadow`, and
  `shadow_promoted`
- production report and email support for a simpler-baseline cross-check
- explicit existing-holdings lot guidance
- explicit diversification-first redeploy guidance
- a promoted v13.1 default that uses the simpler diversification-first
  recommendation layer while keeping a visible cross-check in the monthly
  output

v14 reduced-universe prediction-layer work is now also documented in:

- `docs/plans/codex-v14-plan.md`
- `docs/results/V14_RESULTS_SUMMARY.md`
- `docs/closeouts/V14_CLOSEOUT_AND_V15_NEXT.md`

It adds:

- a post-v13 baseline freeze after the real April production run
- a narrow reduced-universe bakeoff between the live 4-model stack and lean
  Ridge/GBT-centered candidates
- minimal one-feature-at-a-time surgery on the surviving lean candidates
- production-like shadow review memos for the leading replacement candidate

v15 feature-replacement work is now also documented in:

- `docs/plans/codex-v15-plan.md`
- `docs/plans/codex-v15-feature-test-plan.md`
- `docs/results/V15_RESULTS_SUMMARY.md`
- `docs/closeouts/V15_CLOSEOUT_AND_V16_NEXT.md`

It adds:

- a fixed-budget feature-replacement inventory template
- archived external research reports under `docs/history/v15-research-reports/`
- baseline feature inventories for the post-v14 Ridge / GBT candidate pair
- current feature coverage reports under `results/v15/`
- a canonical one-for-one swap queue
- `v15.0` exhaustive screening on Ridge / GBT
- `v15.1` cross-model confirmation on all deployed model types
- `v15.2` final cross-model bakeoff

v16 promotion-study work is now also documented in:

- `docs/plans/codex-v16-plan.md`
- `docs/results/V16_RESULTS_SUMMARY.md`
- `docs/closeouts/V16_CLOSEOUT_AND_V17_NEXT.md`

It adds:

- a narrow promotion gate for the confirmed v15 Ridge / GBT feature swaps
- direct comparison against the reduced-universe live 4-model stack
- direct comparison against the `historical_mean` baseline
- a clear `promote / shadow / do not promote` decision record

v17 shadow-gate work is now also documented in:

- `docs/plans/codex-v17-plan.md`
- `docs/results/V17_RESULTS_SUMMARY.md`
- `docs/closeouts/V17_CLOSEOUT_AND_V18_NEXT.md`

It adds:

- a production-style monthly review window for the modified v16 Ridge+GBT pair
- direct comparison against the current live production cross-check under the promoted v13.1 recommendation layer
- a gate for whether the modified pair should replace the visible cross-check path

v18 directional-bias work is now also documented in:

- `docs/plans/codex-v18-plan.md`
- `docs/results/V18_RESULTS_SUMMARY.md`
- `docs/closeouts/V18_CLOSEOUT_AND_V19_NEXT.md`

It adds:

- narrow benchmark-side and peer-relative swaps on the modified v16 Ridge+GBT pair
- a direct check for whether those swaps reduce the pair's directional disagreement with the promoted simpler baseline
- a final v18 gate on whether the resulting pair should advance at all

v19 feature-completion work is now also documented in:

- `docs/plans/codex-v19-plan.md`
- `docs/results/V19_RESULTS_SUMMARY.md`
- `docs/closeouts/V19_CLOSEOUT_AND_V20_NEXT.md`

It adds:

- public-macro backfill for the deferred benchmark-side and valuation ideas
- the remaining EDGAR-derived insurance-quality features
- a complete tested / blocked traceability matrix for all 46 original v15
  inventory rows
- an explicit closeout of the features that remain impossible without new
  source classes

v20 synthesis / promotion-readiness work is now also documented in:

- `docs/plans/codex-v20-plan.md`
- `docs/results/V20_RESULTS_SUMMARY.md`
- `docs/closeouts/V20_CLOSEOUT_AND_V21_NEXT.md`

It adds:

- one assembled set of best-of-confirmed Ridge / GBT replacement stacks
- direct comparison against the reduced live production cross-check
- direct comparison against the `historical_mean` baseline
- a 12-month monthly shadow review that explicitly checks directional agreement
  with the promoted simpler baseline
- a final v20 gate on whether any assembled replacement stack is actually
  promotion-ready

v21 historical-comparison work is now also documented in:

- `docs/plans/codex-v21-plan.md`
- `docs/results/V21_RESULTS_SUMMARY.md`
- `docs/closeouts/V21_CLOSEOUT_AND_V22_NEXT.md`

It adds:

- a point-in-time historical comparison over the full common evaluable window
- a correction to the narrow recent-window shadow-gate interpretation used in
  v17-v20
- a direct comparison of the live reduced cross-check and the leading v16-v20
  candidates against the promoted simpler baseline across history
- a final v21 gate on whether any candidate is actually cleaner historically
  than the current live cross-check

v22-v24 promotion-validation work is now also documented in:

- `docs/plans/codex-v22-plan.md`
- `docs/plans/codex-v23-plan.md`
- `docs/plans/codex-v24-plan.md`
- `docs/results/V22_RESULTS_SUMMARY.md`
- `docs/results/V23_RESULTS_SUMMARY.md`
- `docs/results/V24_RESULTS_SUMMARY.md`
- `docs/closeouts/V23_CLOSEOUT_AND_V24_NEXT.md`
- `docs/closeouts/V24_CLOSEOUT_AND_V25_NEXT.md`

It adds:

- the narrow implementation path that promotes `ensemble_ridge_gbt_v18` as the
  visible cross-check while leaving the simpler recommendation layer unchanged
- a stitched-history validation that extends the common evaluable window back
  to `2013-04-30`
- a direct test of `VOO` versus `VTI`, which still ends with `keep_voo`

v25 correctness-validation work is now also documented in:

- `docs/results/V25_PEER_REVIEW_SYNTHESIS.md`
- `docs/plans/codex-v25-plan.md`
- `docs/results/V25_RESULTS_SUMMARY.md`
- `docs/closeouts/V25_CLOSEOUT_AND_V26_NEXT.md`

It adds:

- canonical last-business-day month-end handling across the active production
  feature paths
- gap-aware inner CV for the regularized models
- a corrected WFO minimum-length guard using `TRAIN + GAP + TEST`
- corrected CPCV recombined-path provenance
- reruns of the promotion-sensitive `v20-v24` studies on the corrected
  foundation

v26 productionization-cleanup work is now also documented in:

- `docs/plans/codex-v26-plan.md`
- `docs/results/V26_RESULTS_SUMMARY.md`
- `docs/closeouts/V26_CLOSEOUT_AND_V27_NEXT.md`

It adds:

- cleanup of the remaining invalid-divide warning noise in the historical
  comparison layer
- a fresh monthly dry run on the corrected `v25` foundation
- explicit confirmation that the active recommendation layer and the promoted
  visible cross-check remain unchanged

v27 redeploy-portfolio work is now also documented in:

- `docs/plans/codex-v27-plan.md`
- `docs/results/V27_RESULTS_SUMMARY.md`
- `docs/closeouts/V27_CLOSEOUT_AND_V28_NEXT.md`

It adds:

- archived external redeploy-portfolio research reports under
  `docs/history/redeploy-portfolio-reports/`
- a repeatable, backtested sell-proceeds portfolio recommendation
- explicit separation between:
  - the broader forecast benchmark universe
  - the narrower monthly investable redeploy universe
- a live monthly `Suggested Redeploy Portfolio` section for the report and email
- a benchmark-pruning review for which funds should remain contextual only in
  the buy recommendation

v28 forecast-universe review is now also documented in:

- `docs/plans/codex-v28-plan.md`
- `docs/results/V28_RESULTS_SUMMARY.md`
- `docs/closeouts/V28_CLOSEOUT_AND_V29_NEXT.md`

It adds:

- a direct test of whether the forecast benchmark universe should be pruned to
  align more closely with the v27 buyable redeploy universe
- explicit comparison between:
  - the current reduced forecast universe
  - a buyable-only forecast universe
  - a buyable-plus-context forecast universe
- a governance conclusion that forecast and redeploy universes should remain
  separate for now because the broader forecast set still performs materially
  better

## Promotion Rule

Research results do not become production behavior automatically.

A candidate should only be promoted when it demonstrates:

- better policy-level utility than the current production baseline or the agreed
  naive baseline
- acceptable aggregate model health
- acceptable obs/feature discipline
- stable behavior across the validation framework already used by the repo
- clear documentation of the change and its operational consequences
- a recommendation that improves diversification usefulness, not just local
  benchmark prediction quality

## Research vs. Production Labels

- Production:
  - scheduled workflow code
  - monthly decision generation
  - committed monthly output artifacts
- Research:
  - `src/research/`
  - v9 and v11 scripts and outputs
- Provisional:
  - candidates under consideration for future production use
  - diversification-aware redeploy policies not yet promoted
  - v14 reduced-universe replacement candidate shadowing

## Current Recommendation-Layer Conclusion

After v12, the repo still does not recommend promoting a new live model stack.

The most promising next production change is narrower:

- consider promoting the simpler diversification-first recommendation layer
- keep model-stack promotion separate until a reduced-universe candidate
  clearly beats the baseline

## Current v13 Default

The current production-facing default is:

- `RECOMMENDATION_LAYER_MODE=shadow_promoted`

This means:

- the simpler v12 baseline now drives the official recommendation layer
- the live 4-model monthly stack is still surfaced as a cross-check
- usefulness improvements from v11-v12 are now part of the production report
  and email path

## Current Prediction-Layer Conclusion

After v14, the repo still does not recommend replacing the live prediction
stack immediately.

The current narrowed conclusion is:

- `ensemble_ridge_gbt` is the leading reduced-universe replacement candidate
- it improves on the reduced-universe live 4-model stack
- it remains close to, but not clearly ahead of, the `historical_mean`
  baseline
- the next step should be fixed-budget feature replacement in v15 rather than
  broader methodology expansion

## Current v20 Conclusion

v20 improves the replacement-candidate picture again, but still does not
justify replacing the current live production cross-check.

What v20 proved:

- `ensemble_ridge_gbt_v18` is the strongest assembled reduced-universe metric
  row
- `ensemble_ridge_gbt_v20_best` is the strongest best-of-confirmed stack built
  from the v16-v19 winning swaps
- the assembled candidates materially improve on the reduced live production
  cross-check in research metrics

What v20 still blocks on:

- the leading assembled candidates remain directionally misaligned with the
  promoted simpler baseline
- the best v20 metric row had `0.0%` signal agreement with the simpler
  baseline over the 12-month review window
- the best v20 metric row stayed `UNDERPERFORM` in every reviewed month, which
  is not a cleaner user-facing cross-check than the current live path

So the current next-step recommendation is:

- keep the v13.1 recommendation layer unchanged
- keep the current live production cross-check unchanged
- avoid another generic feature sweep
- focus next on blocked-source work or narrow calibration / sign-bias
  diagnostics if further prediction-layer work is needed

## Current v21 Conclusion

v21 changes the promotion picture materially.

What v21 proved:

- the narrow recent-window result from v20 was not the right basis for the
  final promotion decision
- over the full common evaluable history, `ensemble_ridge_gbt_v18` is cleaner
  than the current live reduced cross-check versus the promoted simpler
  baseline
- `ensemble_ridge_gbt_v18` achieved stronger historical agreement with the
  simpler baseline than the current live reduced cross-check while also
  preserving the reduced-universe metric improvement

What v21 recommends now:

- keep the v13.1 recommendation layer unchanged
- treat `ensemble_ridge_gbt_v18` as the leading candidate for replacing the
  current live production cross-check
- move next to a narrow promotion / implementation step rather than another
  generic feature sweep

## Current v22 Conclusion

v22 implements that promotion recommendation narrowly.

What v22 changed:

- the active recommendation layer is still the simpler diversification-first
  baseline
- the visible monthly cross-check is now `ensemble_ridge_gbt_v18`
- the underlying 4-model production signal path remains unchanged

So the current conclusion is:

- keep the v13.1 recommendation layer unchanged
- use `ensemble_ridge_gbt_v18` as the visible production cross-check
- keep the broader 4-model signal path unchanged unless a later cycle
  explicitly promotes a full replacement

## Current v23 Conclusion

v23 extends the benchmark-side historical validation materially.

What v23 proved:

- the v21 promotion result survives a longer research-only stitched-history
  window
- with pre-inception proxies for `VOO`, `VXUS`, and `VMBS`, the common
  evaluable window extends back to `2013-04-30`
- `ensemble_ridge_gbt_v18` still behaves much more like the promoted simpler
  baseline than the reduced live production cross-check does

Important proxy-quality caveat:

- `VXUS <- VEA + VWO` and `VMBS <- BND` are strong research proxies
- `VOO <- VTI` is usable for longer-window validation, but it is a looser
  proxy than ideal and should not be silently promoted into production

So the current conclusion is:

- keep the v13.1 recommendation layer unchanged
- keep `ensemble_ridge_gbt_v18` as the visible production cross-check
- treat the v21/v22 promotion path as confirmed on a longer stitched-history
  window
- if further benchmark-history work is pursued, prioritize a cleaner S&P 500
  pre-inception proxy before another broad model sweep

## Current v24 Conclusion

v24 answers a narrower benchmark-definition question directly.

What v24 proved:

- simply replacing `VOO` with `VTI` does not improve the reduced forecast
  universe enough to justify changing it
- the raw-history advantage of `VTI` is real, but it does not translate into a
  better leading candidate under the current modeling and policy setup
- even the stitched-history `VTI` scenario, while longer, was weaker than the
  current `VOO`-based universe on policy return and agreement with the simpler
  baseline

So the current conclusion is:

- keep `VOO` in the reduced forecast universe
- keep the v13.1 recommendation layer unchanged
- keep `ensemble_ridge_gbt_v18` as the visible production cross-check
- if more history is pursued later, use explicit research-only proxy studies
  rather than silently redefining the benchmark universe

## Current v15 Conclusion

v15 is executed, but it still does not justify an immediate production
prediction-stack promotion.

What v15 did prove:

- the feature set was part of the prediction-quality problem
- `rate_adequacy_gap_yoy` is a real upgrade for the GBT path
- `book_value_per_share_growth_yoy` is a real upgrade for the linear-model path

What v15 did not yet prove:

- that a modified prediction stack should replace the live one immediately

So the current next-step recommendation is:

- keep the v13.1 recommendation layer unchanged
- keep the live production prediction stack unchanged for now
- move to a narrow promotion study using the v15-confirmed Ridge / GBT
  replacements

## Current v16 Conclusion

v16 improves the replacement candidate, but still does not justify an
immediate production prediction-stack promotion.

What v16 proved:

- `ensemble_ridge_gbt_v16` is now the best reduced-universe row
- the v15 feature swaps improved the Ridge+GBT pair versus the reduced live
  production stack
- the modified pair materially improved OOS R^2 versus the reduced live stack

What v16 did not yet prove:

- that the modified pair clears a strong enough edge versus the
  `historical_mean` baseline to become the new live prediction layer

So the current conclusion is:

- keep the v13.1 recommendation layer unchanged
- keep the current live prediction stack unchanged for now
- treat `ensemble_ridge_gbt_v16` as the lead shadow candidate for the next
  narrow promotion gate

## Current v17 Conclusion

v17 still does not justify a production prediction-layer change.

What v17 proved:

- the modified Ridge+GBT pair is steadier than the current live production cross-check
- the modified pair keeps the improved reduced-universe metrics from v16

What v17 did not prove:

- that the modified pair behaves as a better user-facing cross-check under the
  promoted simpler baseline

In fact, the main blocker is now clearer:

- the modified pair disagrees with the simpler baseline too consistently on
  direction, so promoting it would likely make the report more confusing even
  though its reduced-universe metrics are better

So the current conclusion is:

- keep the v13.1 recommendation layer unchanged
- keep the current live production cross-check unchanged
- keep `ensemble_ridge_gbt_v16` in research only
- target directional-bias reduction and deferred benchmark-side / peer-relative
  feature families next rather than broader model complexity

## Current v18 Conclusion

v18 improved the candidate's reduced-universe metrics again, but still did not
improve the user-facing promotion case.

What v18 proved:

- narrow benchmark-side and peer-relative swaps are feasible with the existing data
- `vwo_vxus_spread_6m` and `real_yield_change_6m` were the best v18 one-for-one swaps
- the resulting `ensemble_ridge_gbt_v18` improved metrics versus `ensemble_ridge_gbt_v16`

What v18 did not prove:

- that those swaps reduce the candidate's directional disagreement with the
  promoted simpler baseline

In fact, the main blocker remains:

- both the v16 and v18 candidate pairs still disagree with the simpler
  baseline on direction in every reviewed month

So the current conclusion is:

- keep the v13.1 recommendation layer unchanged
- keep the current live production cross-check unchanged
- keep `ensemble_ridge_gbt_v16` as the leading research candidate
- do not advance the v18 swaps to another promotion gate

## Current v19 Conclusion

v19 closes the original feature-inventory question, but it still does not
justify a production prediction-layer change.

What v19 proved:

- the repo has now evaluated essentially the full original feature queue
- `44 / 46` original v15 feature ideas were tested
- only `2 / 46` remain blocked, and both are blocked for clear source reasons
- the best newly resolved positive additions were:
  - `pgr_pe_vs_market_pe`
  - `usd_broad_return_3m`
  - `auto_pricing_power_spread`

What v19 did not prove:

- that the newly completed feature inventory is enough to replace the current
  live production cross-check

So the current conclusion is:

- keep the v13.1 recommendation layer unchanged
- keep the current live production cross-check unchanged
- treat the feature-inventory question as substantially closed
- move next to a narrow synthesis / promotion-readiness study rather than
  another broad feature sweep

## Classifier Sidecar Policy

The tuned Ridge classifier from v9 is currently a sidecar confidence candidate.

It may be used for:

- additional confidence diagnostics
- abstention research
- future policy experiments

v11 kept the classifier in that role. Even when the abstain-only overlay looked
useful in research, it was still not promoted to primary-engine status.

It is not the primary production decision engine.

## Current v28 Conclusion

v28 confirms that the project should keep separate universes for:

- forecast benchmarking
- user-facing redeploy recommendations

The narrower buyable-first universes were conceptually cleaner, but they gave
up too much on policy utility, agreement with the promoted simpler baseline,
and review-window coverage.

So the current governance rule is:

- keep the broader reduced forecast universe
- keep the narrower v27 redeploy universe for the buy answer

## Current v29 Conclusion

v29 does not change the model stack or recommendation policy.

It changes how the monthly output explains the current system:

- benchmark roles are explicit
- the confidence gate is summarized near the top
- forecast-only context is easier to distinguish from realistic buy candidates

This is a presentation improvement, not a new promotion decision.
