# Codex v9.x Development Plan

## Status

Created: 2026-04-03  
Author: Codex  
Scope: Final consolidated v9.x plan for improving predictive usefulness and model stability

This document supersedes ad hoc v9.x recommendation notes by turning them into
one implementation-oriented plan. It is based on:

- the current repo state after v8.13
- current production diagnostics and backtest artifacts
- the Gemini peer review (`gemini-peer-review-20260329.md`)
- the attached ChatGPT/Codex execution draft (`v_9_x_codex_execution_plan.md`)

The central conclusion is unchanged:

> v9.x should focus on improving predictive usefulness through simplification,
> measurement, benchmark/target redesign, and policy-level evaluation before
> attempting any major architectural expansion.

---

## Executive Summary

The project is already sophisticated enough that adding more modeling machinery
is unlikely to be the highest-value next step. The current problems are more
consistent with:

- insufficient effective sample size relative to feature count
- unstable behavior across benchmarks and validation paths
- a target formulation that may preserve some direction but poor magnitude
- benchmark aggregation that may be diluting stronger local signals
- promotion criteria that need to prioritize decision usefulness over isolated
  predictive metrics

Current evidence from the repo supports that diagnosis:

- aggregate OOS R² is strongly negative
- representative CPCV is failing
- obs/feature ratio is failing
- production has already moved toward leaner, model-specific feature sets
- monthly decision logic is correctly falling back to `DEFER-TO-TAX-DEFAULT`
  when predictive quality is weak

That means v9.x should not begin with TFT, reinforcement learning, or other
high-capacity modeling approaches. It should begin by answering:

1. Are the current targets and benchmark universe well chosen?
2. Is the feature set still too large, sparse, or redundant?
3. Is the ensemble diluting useful signals?
4. Does the modeling stack improve policy outcomes relative to simple rules?
5. Is the signal broad, or only present in narrow benchmark or regime slices?

---

## Comparison of Recommendations

## What Codex Recommended Before

The earlier Codex recommendation prioritized:

- benchmark pruning and smaller benchmark universes
- smaller feature sets and explicit feature-cost accounting
- testing classification and thresholded targets against regression
- policy-level evaluation against simple heuristics
- regime-slice backtests
- only one moderate architecture step after simpler tests

That remains the correct framing for v9.x.

## What the Gemini Peer Review Got Right

The Gemini review is strongest in domain feature suggestions, especially around
insurance economics. The most useful ideas to carry forward are:

- cost-side macro features, not just pricing-power features
- explicit underwriting spread features
- deeper use of monthly segment operating data
- frequency/severity-aware proxies
- a more explicit connection between model outputs and user utility

These are good additions, but only in a disciplined low-complexity form.

## What the Gemini Peer Review Overreaches On

The Gemini report moves too quickly toward:

- Temporal Fusion Transformer
- deep sequence models
- deep reinforcement learning
- a general assumption that more sophisticated architecture is the natural next step

Given the current repo diagnostics, these are not the right v9.x starting point.
They risk adding variance, development cost, and interpretability burden before
the project has demonstrated stable edge with simpler setups.

## What the Attached v9.x Execution Draft Got Right

The attached `v_9_x_codex_execution_plan.md` is directionally very strong and
aligned with the earlier Codex recommendation. Its strongest features are:

- explicit non-goals for over-complex modeling
- workstream ordering that begins with falsification and baselines
- a feature-cost audit before feature expansion
- target-formulation experiments
- policy-level evaluation
- pooled benchmark-family experiments as a moderate next complexity step

## What This Final Plan Changes

This final plan keeps the structure of the attached v9.x draft, but adds:

- a clearer promotion gate for deciding what can become production candidate
- a stronger requirement to compare against simple non-ML policies
- a narrower initial set of new feature additions derived from the Gemini review
- a more explicit distinction between research-only outputs and production candidates
- versioned milestones so the work can be implemented incrementally

---

## v9.x Primary Objective

Build a reproducible evaluation framework that determines:

> What is the simplest target, benchmark universe, and feature stack that
> generates the most stable and decision-useful out-of-sample behavior?

The output of v9.x should be either:

- a leaner and more stable production candidate
- a decision that the current signal is too unstable for promotion
- or a narrowly justified case for one moderate next-step modeling extension

---

## v9.x Non-Goals

The following are out of scope unless earlier v9.x work produces strong,
specific evidence that they are needed:

- Temporal Fusion Transformer
- deep learning sequence models
- deep reinforcement learning
- full architecture rewrites
- new paid data vendors
- new uncertainty layers beyond the current calibration/conformal framework
- turning the project into a generic market-prediction platform

These are explicitly deferred until after the core v9.x simplification and
falsification work.

---

## Guiding Principles

1. Prefer smaller experiments over broad rewrites.
2. Treat policy usefulness as the real scorecard.
3. Do not promote on IC alone.
4. Track effective sample size explicitly in every experiment.
5. Treat feature count as a cost, not just an opportunity.
6. Preserve point-in-time integrity, purge, and embargo behavior.
7. Keep production monthly reporting stable while research runs in parallel.
8. Favor CLI-driven scripts, CSV outputs, and markdown summaries over notebooks.
9. Require side-by-side comparison versus simple baselines.
10. Promote only if results improve stability and decision utility together.

---

## Current v8.13 Baseline To Beat

Any v9.x candidate must be judged relative to the current v8.13 baseline:

- 4-model ensemble: ElasticNet, Ridge, BayesianRidge, GBT
- model-specific feature overrides
- Platt calibration
- ACI conformal intervals
- representative CPCV
- model-quality gating
- current monthly recommendation and email workflow

Known weaknesses of the current baseline:

- negative aggregate OOS R²
- failing representative CPCV
- failing obs/feature ratio
- unstable per-benchmark behavior
- positive directional signal in places, but insufficient robustness for
  prediction-led vest decisions

v9.x is successful only if it improves on that baseline in a defensible way.

---

## Promotion Gate For v9.x

No candidate may be promoted to “production candidate” unless it passes all of
the following:

1. beats or materially matches the current baseline on policy-level utility
2. does not materially worsen OOS R²
3. improves or materially stabilizes CPCV behavior
4. improves or materially reduces feature burden / obs-feature stress
5. is reproducible via committed script + committed output summary
6. outperforms at least one simple heuristic baseline for a meaningful decision metric

No candidate may be promoted if it:

- only improves IC while worsening OOS R² or stability
- improves calibration but not decision usefulness
- requires substantially more complexity without clear robustness gain

---

## Deliverables

v9.x should produce:

### New scripts

1. `scripts/benchmark_suite.py`
2. `scripts/feature_cost_report.py`
3. `scripts/target_experiments.py`
4. `scripts/regime_slice_backtest.py`
5. `scripts/policy_evaluation.py`
6. `scripts/pooled_benchmark_experiments.py`

### Optional shared helpers

If reusable logic naturally emerges, place it under a stable helper module, for
example:

- `src/research/v9_utils.py`
- `src/research/policy_metrics.py`
- `src/research/benchmark_sets.py`

### Results directories

- `results/v9/benchmark_suite/`
- `results/v9/feature_cost/`
- `results/v9/target_experiments/`
- `results/v9/regime_slices/`
- `results/v9/policy_eval/`
- `results/v9/pooled_models/`

### Summary docs

- `V9_RESULTS_SUMMARY.md`

That final summary must explicitly document:

- what improved
- what failed
- what was not worth the complexity
- what should become the next production candidate
- what should not be done next

---

## Versioned Roadmap

## v9.0 - Benchmark Harness and Acceptance Gate

### Objective

Create one repeatable benchmark harness that compares:

- current production
- lean baselines
- naive statistical baselines
- simple policy heuristics

### Why

The project needs a single place to answer:

- does the production system beat a naive mean forecast?
- does it beat a simple price-only model?
- does it beat “always sell 50%”?
- which metrics move together and which are misleading?

### Required implementation

Create `scripts/benchmark_suite.py`.

### Required candidates

1. expanding historical mean baseline
2. price-only ElasticNet
3. price-only GBT
4. lean macro baseline
5. current v8.13 production stack
6. policy heuristics:
   - always sell 50%
   - always sell 100%
   - always defer to tax default
   - optional tax-breakeven hold heuristic

### Required metrics

- candidate name
- benchmark universe
- horizon
- n_obs
- n_features
- full obs/feature ratio
- per-fold obs/feature ratio
- OOS R²
- IC
- hit rate
- MAE
- CPCV mean IC
- CPCV std
- CPCV positive paths
- policy utility metrics

### Acceptance criteria

- one CLI run produces a unified comparison output
- the project can clearly answer whether the current system beats simple baselines

---

## v9.1 - Feature Cost, Missingness, and Redundancy Audit

### Objective

Quantify which features damage effective sample size and which features are
low-value or redundant.

### Why

The current obs/feature ratio is a likely root cause. Before adding more
features, we need to understand which current features are most expensive.

### Required implementation

Create `scripts/feature_cost_report.py`.

### Required per-feature outputs

- feature name
- source family
- first valid date
- last valid date
- valid observation count
- percent missing
- current production usage flag
- fully populated rows lost if required
- correlation versus currently selected features
- approximate incremental obs/feature burden

### Required summary outputs

- top sample-expensive features
- top high-collinearity features
- candidate drop list for v9.2

### Acceptance criteria

- effective sample shrinkage becomes visible and measurable
- produces a prioritized list of features to test for removal

---

## v9.2 - Feature Diet Experiments

### Objective

Find the smallest feature sets that improve stability and decision usefulness.

### Required experiment modes

1. leave-one-feature-out
2. leave-one-group-out
3. forward selection under stability constraints
4. backward elimination under stability constraints

### Required candidate tiers

- ultra-lean
- lean
- moderate
- current-production

### Required metrics

- n_features
- n_obs
- obs/feature ratio
- OOS R²
- IC
- hit rate
- MAE
- CPCV positive-path count
- policy utility

### Promotion rule

A larger feature set must improve at least two of:

- OOS R²
- CPCV stability
- policy utility

without materially worsening effective sample size.

### Additional v9.2 requirement from Gemini review

Test only a small, high-conviction subset of new domain features first:

1. insurance pricing minus repair inflation
2. insurance pricing minus used-vehicle inflation
3. insurance pricing minus parts inflation
4. vehicle miles traveled YoY
5. combined-ratio gap versus management target proxy
6. a small number of segment spread features such as direct vs agency growth

Do not add broad feature families all at once.

### Acceptance criteria

- produces a recommended lean feature stack per model family
- determines whether current production features are still too broad

---

## v9.3 - Target Formulation Experiments

### Objective

Test whether the project should predict direction/class instead of precise return magnitude.

### Why

Current evidence suggests the project may retain some directional information
but weak magnitude accuracy.

### Required target types

1. current regression target: 6M relative return
2. binary classification: outperform vs not outperform
3. ternary classification:
   - underperform
   - neutral
   - outperform
   with configurable neutral band
4. optional thresholded outperform target:
   - for example outperform by more than +3%

### Additional v9.3 requirement

Test binary classification in two distinct roles:

1. as a standalone replacement target
2. as a confirmatory sidecar model that can gate a positive regression signal

The confirmatory use case should be evaluated explicitly rather than assumed to
help by default.

If the confirmatory classifier shows promise, continue with a model-specific
feature-selection pass rather than assuming the regression feature set is also
best for classification.

### Required metrics

For regression:

- OOS R²
- IC
- MAE
- policy utility

For classification:

- Brier score
- log loss
- calibration / ECE
- AUC where applicable
- hit rate
- policy utility

### Key question

Is a calibrated classifier with a simple decision rule more useful than a noisy
regression model with unstable magnitude predictions?

Secondary question:

Can a binary outperform-probability model improve policy outcomes when used only
to confirm a positive regression signal?

### Acceptance criteria

- provides a direct recommendation on regression vs classification vs thresholded targets
- explicitly states whether a confirmatory classifier should be:
  - promoted
  - retained as research-only
  - or dropped

---

## v9.4 - Benchmark Universe Reduction

### Objective

Determine whether the current 21-benchmark universe is diluting useful signals.

### Required benchmark sets

At minimum compare:

1. full benchmark universe
2. core benchmark universe
3. optional sector-focused subset
4. optional defensive subset

Recommended core default:

- `VTI`
- `KIE`
- `VFH`
- `BND`
- `GLD`

This set can be revised if the benchmark suite shows a better core.

### Required metrics

- OOS R²
- IC
- hit rate
- CPCV stability
- policy utility
- recommendation consistency

### Acceptance criteria

- clearly recommends full vs reduced benchmark universe

---

## v9.5 - Regime Slice Backtests

### Objective

Determine whether the signal is broad or concentrated in specific regimes.

### Required slices

- pre-2020
- 2020-2021
- 2022-present
- trailing 36 months
- optional low-vol vs high-vol
- optional inflation/rate regimes
- optional insurance-pricing spread regimes

### Required outputs

- n_obs
- OOS R²
- IC
- hit rate
- CPCV where feasible
- policy utility

### Key question

Is the signal broad, or only useful in specific underwriting / inflation /
insurance-pricing environments?

### Acceptance criteria

- determines whether regime gating is justified

---

## v9.6 - Policy-Level Evaluation

### Objective

Evaluate models as vesting-decision policies, not just predictors.

### Required policy comparisons

1. sell 50% at vest
2. sell 100% at vest
3. current gated ML policy
4. confidence-gated ML policy
5. tax-breakeven heuristic policy
6. best v9.x lean candidate

### Required outputs

- after-tax proceeds
- average after-tax utility proxy
- downside / drawdown proxy if meaningful
- certainty-equivalent style metric if feasible
- frequency of action changes
- stability of decisions over time

### Hard rule

A candidate that does not beat a simple baseline policy should not be promoted,
regardless of IC or hit rate.

### Acceptance criteria

- makes it explicit whether the ML stack adds real decision value

---

## v9.7 - Pooled Benchmark-Family Experiments

### Objective

Test a moderate next-complexity step that is still much simpler than deep sequence models.

### Why

If one-model-per-benchmark is too noisy, but a fully pooled model is too crude,
family-level pooling may stabilize learning without a major architecture jump.

### Candidate families

- broad equities: `VTI`, `VOO`, `VXUS`, `VEA`, `VWO`
- sectors: `VGT`, `VHT`, `VFH`, `VIS`, `VDE`, `VPU`, `KIE`
- defensive/income: `BND`, `BNDX`, `VCIT`, `VMBS`, `VIG`, `SCHD`
- real assets: `VNQ`, `GLD`, `DBC`

### Compare against

- current per-benchmark architecture
- reduced core benchmark architecture

### Acceptance criteria

- determines whether family pooling is superior enough to justify adoption
- remains the largest acceptable modeling step inside v9.x

---

## v9.8 - Final Promotion Decision

### Objective

At the end of v9.x, recommend one of:

1. promote a leaner v9.x candidate
2. keep v8.13 production behavior but retain the new research diagnostics
3. conclude the signal is too unstable for predictive promotion
4. justify one moderate next-step modeling extension

### Required final outputs

Create `V9_RESULTS_SUMMARY.md` with sections:

1. baseline findings
2. feature-cost findings
3. feature-diet findings
4. target-formulation findings
5. benchmark-universe findings
6. regime findings
7. policy-evaluation findings
8. pooled-model findings
9. recommendation
10. what should not be done next

### Required final recommendation format

- `Promote`
- `Do not promote`
- `Needs more evidence`

And specify:

- proposed feature set
- proposed target formulation
- proposed benchmark universe
- whether pooled models should be used
- whether current gating should remain unchanged

---

## Cross-Workstream Requirements

### Temporal integrity

All new scripts must preserve:

- point-in-time feature availability
- lag discipline
- purge / embargo behavior
- no leakage in scaling, selection, or imputation

### Reuse existing infrastructure

Prefer reusing:

- `build_feature_matrix_from_db`
- `get_feature_columns`
- `get_X_y_relative`
- `run_wfo`
- `run_cpcv`
- existing tax framework and position-lot handling
- current benchmark and feature config structures
- current ablation outputs where compatible

### Reproducibility

Every major experiment must:

- run from repo root
- write machine-readable CSV output
- write human-readable markdown summary
- avoid notebook-only logic

### Testing

Add tests for:

- CLI argument parsing
- summary metric calculations
- feature-cost accounting helpers
- target transformation helpers
- policy evaluation helpers
- benchmark family mapping helpers

---

## What Should Not Be Done In v9.x

Unless earlier work produces strong evidence, do not:

- add TFT
- add RL
- add a new vendor data dependency
- expand the feature set broadly before the feature-cost audit
- promote on IC alone
- replace the current validation framework
- introduce a large production refactor before research outputs are conclusive

These restrictions are part of the plan, not temporary suggestions.

---

## Suggested Branch Strategy

Preferred umbrella branch:

- `codex/v9x-evaluation`

Suggested staged branches if needed:

1. `codex/v9x-benchmark-suite`
2. `codex/v9x-feature-cost`
3. `codex/v9x-feature-diet`
4. `codex/v9x-target-experiments`
5. `codex/v9x-benchmark-universe`
6. `codex/v9x-regime-slices`
7. `codex/v9x-policy-eval`
8. `codex/v9x-pooled-models`
9. `codex/v9x-final-summary`

If one umbrella branch is more practical, that is acceptable as long as commits
remain logically grouped.

---

## Suggested Commit Sequence

1. add benchmark suite skeleton and outputs
2. add feature-cost report
3. extend ablation / feature-diet metrics
4. add target experiments
5. add reduced benchmark-universe experiments
6. add regime-slice backtests
7. add policy evaluation
8. add pooled benchmark experiments
9. add final summary doc
10. clean up and ensure tests pass

---

## Minimum CLI Surface

The final implementation should support commands approximately like:

```bash
python scripts/benchmark_suite.py --horizon 6 --as-of 2026-04-02
python scripts/feature_cost_report.py
python scripts/feature_ablation.py --benchmarks VTI,VFH,BND,GLD,KIE --horizons 6
python scripts/target_experiments.py --horizon 6
python scripts/regime_slice_backtest.py --horizon 6
python scripts/policy_evaluation.py --horizon 6
python scripts/pooled_benchmark_experiments.py --horizon 6
```

Exact flags may differ, but the equivalent research surface must exist.

---

## Final Checklist

Before closing v9.x, confirm:

- [ ] benchmark suite exists and runs
- [ ] feature-cost report exists and runs
- [ ] feature-diet outputs include stability and sample-size metrics
- [ ] target formulation experiments completed
- [ ] confirmatory-classifier experiments completed
- [ ] reduced benchmark-universe experiments completed
- [ ] regime-slice backtests completed
- [ ] policy evaluation completed
- [ ] pooled benchmark experiments completed
- [ ] final v9.x summary document created
- [ ] tests pass
- [ ] no production workflow unintentionally broken

---

## Expected Outcome Philosophy

The ideal v9.x result is not “a more advanced model.”

The ideal v9.x result is one of:

- a smaller, more stable model stack that beats the current one
- a disciplined proof that the signal is only useful as a gated directional aid
- a conclusion that the current predictive edge is too unstable for promotion
- a narrowly justified case for one moderate next-step experiment

If v9.x concludes that the best outcome is:

- fewer features,
- fewer benchmarks,
- classification instead of regression,
- or stronger abstention / no-call behavior,

that should be treated as a successful outcome.

That is the standard for v9.x.
