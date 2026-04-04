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
  recommendation layer while keeping the live stack visible as a cross-check

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
