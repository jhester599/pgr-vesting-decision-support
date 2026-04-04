# v9 Closeout and v9.1 Recommendation

Created: 2026-04-03  
Author: Codex  
Purpose: Final closeout memo for the completed v9 research program and a concrete recommendation for the next implementation phase

## Status

The v9 research program is complete.

The repo now has a full research harness, committed outputs, and enough
evidence to make a clear recommendation about what should happen next.

What is complete:

- benchmark harness and baseline comparisons
- feature cost / missingness audit
- deliberate per-feature experiments by model type
- target-formulation experiments
- benchmark-universe reduction experiments
- regime-slice backtests
- policy-level evaluation
- pooled benchmark-family experiments
- weekly carry-forward snapshot experiments
- confirmatory classifier experiments
- classifier-specific Ridge feature selection
- summary documentation and committed result artifacts

What is not complete:

- a promoted v9 production replacement for the monthly workflow
- a final production-candidate bakeoff that directly tests a narrowed v9 stack
  inside the live decision path

So the right conclusion is:

- `v9 research`: complete
- `v9.1 production-candidate promotion`: next

## What v9 Proved

### 1. The main weakness is structural, not architectural

The project retains some ranking signal, but return-magnitude fit remains weak.
The evidence does not support moving next to TFT, RL, or broader model
complexity.

### 2. Benchmark reduction matters

The full 21-benchmark universe is too noisy. The best reduced universe found in
v9 was:

- `balanced_core7`
- `VXUS, VEA, VHT, VPU, BNDX, BND, VNQ`

This reduced set improved IC, hit rate, and decision utility relative to the
full-universe setup, even though it did not fully solve negative OOS R².

### 3. Leaner feature sets help more than broader feature sets

The best practical candidates were simpler, model-specific sets:

- `ridge_lean_v1`
- `gbt_lean_plus_two`
- optional `elasticnet_lean_v1`

Bayesian Ridge remained too unstable to lead the next production candidate.

### 4. Policy usefulness is the real scorecard

The most important v9 shift was evaluating the models as vesting-decision
policies rather than only as forecasters.

This changed the ranking materially:

- simple sign-based actions were better than the more complex tiered policy map
- the practical baseline to beat remained `historical_mean` with a sign-based
  rule

### 5. Higher-frequency carry-forward snapshots did not unlock the problem

Weekly snapshots were feasible and point-in-time safe, but they did not beat
the best monthly baseline and did not improve OOS fit enough to justify a move
away from monthly cadence.

### 6. The classifier idea was worth testing, but only as a sidecar

The first confirmatory classifier did not improve policy results.

After model-specific tuning, the Ridge classifier became much more credible:

- best single feature: `investment_book_yield`
- best tuned classifier: 12-feature Ridge set
- best tuned metrics:
  - balanced accuracy `0.6736`
  - Brier `0.2332`
  - hybrid policy return `0.0688`
  - uplift vs regression sign `-0.0011`

That is good enough to keep as a sidecar confidence / abstention candidate, but
not good enough to replace the lean regression decision stack.

## What v9 Does Not Recommend

The v9 evidence does not support the following as the next step:

- broader feature expansion
- a larger benchmark universe
- daily or every-third-day interpolation work
- a classifier-led replacement of the regression stack
- Bayesian Ridge as the lead model
- TFT, deep sequence models, or RL
- promoting any candidate on IC alone

## Final v9 Recommendation

Do not promote a broad v9 replacement yet.

Do proceed to a narrower `v9.1` production-candidate bakeoff centered on the
best v9 findings.

## Recommended v9.1 Scope

### Objective

Test the strongest lean reduced-universe candidate directly against the current
practical benchmark in a production-candidate frame.

### Universe

Use:

- `balanced_core7`

Do not use the full 21-benchmark universe for the first promotion pass.

### Primary model shortlist

Use:

- `ridge_lean_v1`
- `gbt_lean_plus_two`
- optional `elasticnet_lean_v1`

Do not include:

- `bayesian_ridge`

### Decision policy

Default research policy for the bakeoff:

- sign-based hold vs sell logic

Do not lead with the older tiered policy map in `v9.1`.

### Baseline to beat

The real baseline is:

- `historical_mean`
- sign-based decision rule

This is more important than beating the old full production ensemble.

### Classifier role

Keep the tuned Ridge classifier as:

- a sidecar confidence model
- an abstention / diagnostic candidate
- a possible future tie-breaker

Do not use it as:

- the primary decision model
- a mandatory hard confirmation gate

For `v9.1`, the classifier should be reported, not promoted.

## Concrete v9.1 Success Criteria

Treat `v9.1` as successful only if the reduced-universe candidate:

1. beats or materially matches `historical_mean` sign-policy utility
2. improves or materially stabilizes OOS R² relative to the current practical
   candidate set
3. preserves the leaner feature burden
4. remains reproducible under the existing research harness
5. produces a clearer monthly recommendation behavior than the current broader
   setup

If those criteria are not met, the right outcome is not to force promotion.

## Suggested v9.1 Implementation Order

1. Build a reduced-universe candidate ensemble runner using:
   - `ridge_lean_v1`
   - `gbt_lean_plus_two`
   - optional `elasticnet_lean_v1`
2. Compare it directly against:
   - `historical_mean` sign-policy baseline
   - the current production-style candidate where needed for context
3. Add the tuned Ridge classifier as a reported sidecar diagnostic only.
4. Run the candidate through the monthly-decision/reporting path in a
   production-like dry run.
5. Decide:
   - promote
   - do not promote
   - or keep as research-only

## Canonical v9 Files

Primary plan and summary:

- [codex-v9-plan.md](C:/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/docs/plans/codex-v9-plan.md)
- [V9_RESULTS_SUMMARY.md](C:/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/docs/results/V9_RESULTS_SUMMARY.md)

Most important result artifacts:

- [candidate_model_bakeoff_summary_20260403.csv](C:/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/results/v9/candidate_model_bakeoff_summary_20260403.csv)
- [policy_evaluation_summary_balanced_core7_20260403.csv](C:/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/results/v9/policy_evaluation_summary_balanced_core7_20260403.csv)
- [weekly_snapshot_experiments_summary_20260403.csv](C:/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/results/v9/weekly_snapshot_experiments_summary_20260403.csv)
- [confirmatory_classifier_summary_20260403.csv](C:/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/results/v9/confirmatory_classifier_summary_20260403.csv)
- [classifier_feature_selection_recommendation_20260403.md](C:/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/results/v9/classifier_feature_selection_recommendation_20260403.md)
- [confirmatory_classifier_feature_selection_summary_20260403.md](C:/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/results/v9/confirmatory_classifier_feature_selection_summary_20260403.md)

## Bottom Line

v9 accomplished what it needed to accomplish.

It did not prove that a more complex model is needed.
It did prove that the next serious candidate should be:

- smaller
- benchmark-pruned
- Ridge / GBT-centered
- compared against simple policy baselines
- optionally accompanied by a tuned classifier sidecar for confidence

That is the correct starting point for `v9.1`.
