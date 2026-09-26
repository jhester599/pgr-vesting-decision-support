# v87-v96 Classification and Hybrid Decision-Layer Research Plan

## Purpose

This document defines the next research direction after the `v76` quality-
weighted consensus promotion and the `v81-v86` adoption/contract cleanup.

Status update after execution:

- `v87-v96` completed on this branch
- no classification-led path is ready for production promotion
- the most promising result is a classifier-only benchmark-panel shadow path
  built around the `actionable_sell_3pct` target, lean baseline features, and
  pooled/separate linear probabilities with prequential calibration
- the final `v96` recommendation is `continue_research_no_promotion`
- the supporting execution artifacts live under `results/research/`

The central question is:

- should the monthly recommendation problem be modeled primarily as a
  classification problem, with regression retained as a supporting magnitude
  signal?

Current view:

- yes, the decision problem should be treated as both classification and
  regression
- classification should become the primary action gate
- regression should remain the magnitude/sizing/context layer

This plan turns that idea into a structured research program rather than a
single new experiment.

---

## Execution Outcome

The completed cycle ended up being more conservative than the original
hypothesis.

Main findings:

- `v87`: the decision-aligned `actionable_sell_3pct` target slightly edged the
  simpler underperformance label in the per-benchmark pooled binary setup, so
  it became the forward research target
- `v88`: the lean baseline feature set remained best; broader feature-family
  expansion generally hurt balanced accuracy and calibration
- `v89`: among separate linear classifiers, balanced logistic remained the best
  overall decision-oriented baseline
- `v90`: pooled shared logistic improved balanced accuracy over separate
  benchmark models, making pooled linear structure the strongest pure-model
  result in the cycle
- `v91`: shallow nonlinear models did not improve on the pooled linear
  reference
- `v92`: prequential probability calibration materially improved ECE, and the
  best abstention path came from the calibrated separate logistic branch
- `v93`: when translated into monthly-policy usefulness, the benchmark-panel
  probability path still beat the direct basket and breadth target formulations
- `v94-v95`: the best policy result was actually classifier-only benchmark-panel
  gating rather than a regression-plus-classifier hybrid
- `v96`: despite that improvement, the agreement/stability tradeoff versus the
  current production regression path was not strong enough to justify immediate
  promotion

Working interpretation:

- classification is useful enough to remain an active research branch
- it is not yet strong enough to replace or directly override the production
  quality-weighted regression path
- the best near-term use is as a shadow diagnostic or future decision-surface
  companion, not as a production switch

---

## Prior Peer-Review Guidance Integrated

This plan is not based only on the `v46` and `v73` outcomes.

It explicitly incorporates the recurring recommendations from the archived
deep-review set:

- `docs/archive/history/peer-reviews/2026-04-08/`
- `docs/archive/history/repo-peer-reviews/2026-04-05/`
- `docs/archive/history/repo-peer-reviews/2026-04-10/`

The strongest themes that carry forward into this cycle are:

- classification is closer to the real sell-vs-hold decision than raw return
  regression
- regression should still be retained as a sizing and context layer
- feature expansion should be stepwise and interpretable, not combinatorial
- pooled or panel structures deserve direct comparison with separate
  benchmark models
- calibration and abstention behavior matter more than headline accuracy alone
- hard regime splits, broad interaction expansion, and high-variance model
  families should remain low priority
- multiple-testing discipline needs to be explicit, not implicit

This plan therefore treats classification as a decision-layer research program,
not as a wholesale replacement of the current production regression stack.

---

## Why This Direction

The monthly recommendation is fundamentally not just a forecast problem.

What matters most operationally is:

- should the next vest be treated as actionable or not?
- should the recommendation stay at `DEFER-TO-TAX-DEFAULT`?
- if actionable, how aggressive should the sell percentage be?

The current production stack is strong on:

- relative-return forecasting
- cross-benchmark consensus construction
- calibration and quality diagnostics

But the final user decision is still closer to:

- `sell vs. hold`
- `actionable vs. non-actionable`
- `override default diversification vs. follow default`

The earlier `v46` results suggest classification has promise:

- pooled accuracy above chance
- pooled balanced accuracy above chance
- Brier score and log loss in a useful range

The earlier `v73` results suggest a hybrid path is plausible:

- use regression to estimate edge size
- use classification probability to decide whether the edge is reliable enough
  to act on

That makes classification the right next research branch.

---

## Strategic Goal

Design a decision-layer architecture in which:

- a classification model answers whether the recommendation should become
  action-oriented
- a regression model answers how strong the expected edge is
- policy logic combines the two conservatively under strict walk-forward
  validation

The target outcome is not "best forecast metric in isolation."

The target outcome is:

- better actionability classification
- more stable monthly recommendation modes
- better policy-level utility
- improved interpretability for the user-facing recommendation

---

## Working Hypotheses

### H1 - Classification is closer to the real objective

A properly calibrated classifier should map more directly to the actual monthly
decision than a pure relative-return regression target.

### H2 - Classification-only is not enough

A classifier can identify direction or actionability without providing useful
information about effect size, so regression should not be discarded.

### H3 - Hybrid should outperform either alone

The best decision layer will likely:

- use classification probabilities as an action gate
- use regression outputs to size or qualify the recommendation

### H4 - Model structure may differ by benchmark

Per-benchmark classifiers may outperform a pooled panel classifier for some
benchmarks, but a pooled model may generalize better and use scarce history
more efficiently. This is an empirical question and should be studied directly.

### H5 - Feature usefulness for classification will differ from regression

Some features that help rank relative returns may not be the same features that
help classify actionable underperformance or outperformance.

---

## Research Questions

The `v87-v96` cycle should answer the following:

1. What classification target best matches the real recommendation problem?
2. Which features are most useful for classification, and does that differ by
   benchmark or by model family?
3. Are per-benchmark classifiers better than pooled panel classifiers?
4. Which classifier types are best suited to the small-sample, time-series
   regime in this repo?
5. How much does probability calibration improve the usefulness of a
   classification model for decision gating?
6. Does a hybrid classifier + regression policy outperform regression-only and
   classifier-only policy variants?
7. Can a classification-led decision layer improve recommendation stability
   without degrading governance metrics?

---

## Target Definitions To Explore

We should not assume one classification target up front.

The plan should evaluate several target formulations.

### Target Family A - Simple direction

Binary target:

- `1` if `PGR relative return > 0`
- `0` otherwise

This is the direct extension of `v46`.

### Target Family B - Actionability

Binary target:

- `1` if realized outcome would have justified deviating from the default
  diversification / tax policy
- `0` otherwise

This is closer to the actual decision problem than raw relative direction.

### Target Family C - Ternary recommendation class

Multiclass target:

- `ACTIONABLE-SELL`
- `MONITORING-ONLY`
- `DEFER-TO-TAX-DEFAULT`

This is likely data-hungry, so it may be better as a later-stage experiment
after binary targets are understood.

### Target Family D - Benchmark agreement breadth

Binary or ordinal target:

- whether benchmark consensus breadth exceeds a threshold
- whether underperformance is broad enough across benchmarks to matter

This could be useful for hybrid gating because it reflects cross-benchmark
robustness instead of only average sign.

---

## Validation Rules

All experiments in this plan must follow existing project rules:

- strict walk-forward or time-series split only
- no K-Fold cross-validation
- no leakage across time
- all feature selection and model tuning must be performed using only training
  history available at that point in time
- any calibration must be prequential / expanding-window or fold-local

Additional classification-specific rules:

- probability calibration must never be fit on future data
- threshold tuning must be treated as part of model selection and validated
  inside the time-series framework
- pooled panel models must preserve an as-of discipline and never learn from
  future benchmark rows when generating current predictions

Additional repo-review carry-forward rules:

- expanded feature sets should pass a feature-availability and missingness
  sanity check before entering the research loop
- no broad feature search over the full inventory without family-level
  precommitment
- no hard regime-split models unless a later cycle demonstrates enough sample
  to support them
- no holdout consumption during exploratory research

---

## Data and Governance Preconditions

The April 5 repo reviews were a useful reminder that deeper modeling work is
only worth as much as the correctness of the feature matrix entering it.

For this classification cycle, assume:

- current business-month-end alignment fixes remain the canonical indexing
  convention
- existing all-null / feature-availability protections remain active
- any newly promoted feature family must be checked for:
  - usable history length
  - missingness concentration by date
  - whether the signal is genuinely available as-of

This cycle should not reopen a broad ingestion refactor, but it should treat
feature health and publication-lag correctness as entry criteria for any
expanded feature family.

---

## Version Map

### v87 - Problem framing and target taxonomy

Purpose:

- formalize the decision problem as classification + regression rather than
  regression alone

Deliverables:

- one research note comparing the candidate target families
- one baseline artifact describing what each target means operationally
- explicit metric definitions for forecast, classification, and policy layers

Implementation notes:

- create shared helper utilities for classification research under
  `src/research/`
- define a canonical evaluation schema for:
  - accuracy
  - balanced accuracy
  - precision / recall
  - Brier score
  - log loss
  - calibration error
  - abstention rate
  - policy uplift

### v88 - Feature audit and stepwise classification sweep

Purpose:

- explore the full existing feature library for classification usefulness

Research scope:

- baseline lean feature set from `v46`
- incremental feature families added stepwise:
  - price and momentum
  - volatility
  - valuation
  - PGR operating features
  - macro / rates / spreads
  - inflation / insurance-cost proxies
  - benchmark-relative and breadth features
  - only a very small number of interaction or regime-indicator features if
    earlier families prove useful

Feature families to explicitly consider from the archived reviews:

- PGR operating and underwriting:
  - combined ratio
  - loss / underwriting proxies
  - NPW growth
  - PIF growth
  - investment income and book-yield measures
- macro and rate structure:
  - yield slope
  - real yields
  - credit spreads
  - NFCI
  - VIX
- inflation / insurance-cost proxies where already available:
  - auto insurance PPI
  - medical CPI
  - used-car CPI
  - related FRED-derived momentum or level measures
- benchmark-context features:
  - cross-benchmark breadth
  - composite / basket-relative features
  - selected benchmark spreads already present in the repo

Approach:

- run stepwise ablation / addition experiments
- record pooled and per-benchmark effects
- avoid global feature selection outside the training window
- keep candidate retained feature sets lean, with a strong prior toward
  5-8 effective features in the final shortlisted classifiers

Expected outputs:

- ranked feature-family table
- stepwise gain/loss table
- shortlist of classification feature sets for later versions

### v89 - Per-benchmark linear classifiers

Purpose:

- deepen the `v46` path with benchmark-specific classifiers

Model families:

- logistic regression with regularization
- linear SVM with probability calibration if practical
- possibly ridge-style classification equivalents if useful
- keep the first pass squarely in low-variance linear families

Research questions:

- do benchmark-specific models outperform the current pooled `v46` setup?
- should different benchmarks use different feature subsets?

Expected outputs:

- per-benchmark classifier results
- pooled summary rows
- feature-set comparison rows
- stability diagnostics by benchmark

### v90 - Pooled panel classifiers

Purpose:

- compare per-benchmark models against one combined panel model

Design options:

- panel with benchmark identifier dummies
- benchmark-cluster panels
- pooled model with benchmark-specific intercept handling where feasible
- stacked panel with common coefficients and benchmark fixed effects as the
  default pooled reference design

Research questions:

- does pooling improve sample efficiency?
- does pooling degrade benchmark specificity?
- does a pooled model calibrate more cleanly than separate models?

Expected outputs:

- pooled-vs-separate comparison table
- benchmark dispersion analysis
- policy-level comparison using the same downstream threshold rules

### v91 - Nonlinear classifier family sweep

Purpose:

- explore whether shallow nonlinear classifiers improve decision usefulness

Candidate models:

- shallow gradient-boosted trees
- shallow random forest only if it can be justified under the sample regime
- extremely conservative XGBoost classification with strong regularization
- monotonic-constrained gradient boosting only for a tiny set of features with
  strong sign priors and only after linear baselines are understood

Constraints:

- prioritize low-variance settings
- keep tree depth shallow
- avoid wide hyperparameter searches that exceed the sample size

Expected outputs:

- model-family leaderboard
- calibration diagnostics by model family
- recommendation-stability comparison

### v92 - Probability calibration and abstention design

Purpose:

- make classification probabilities decision-usable rather than merely ranked

Research scope:

- Platt / logistic calibration
- isotonic only if sample size is sufficient
- neutral-band / abstention threshold design

Questions to answer:

- what probability thresholds best support:
  - `ACTIONABLE`
  - `MONITORING-ONLY`
  - `DEFER-TO-TAX-DEFAULT`
- how stable are those thresholds over time?
- how much does calibration improve policy usefulness versus raw probabilities?

Expected outputs:

- calibrated-vs-uncalibrated comparison
- threshold sweep table
- abstention-rate and stability diagnostics

### v93 - Basket target formulation and consensus labeling

Purpose:

- test whether the classifier should be trained against:
  - each benchmark separately
  - a pooled benchmark panel
  - the aggregate ETF basket / consensus target

Research scope:

- benchmark-specific target labels
- basket-level label derived from average or weighted relative return
- breadth-of-underperformance target

Questions:

- is the decision problem better captured by benchmark-specific labels or
  basket-level labels?
- does the monthly recommendation need a basket classifier more than individual
  benchmark classifiers?
- does a composite or breadth-style target outperform pure per-benchmark
  direction labels for policy stability?

Expected outputs:

- target-definition comparison matrix
- benchmark-vs-basket policy outcomes

### v94 - Hybrid gate architecture

Purpose:

- define the actual hybrid decision-layer logic

Core design:

- classifier probability determines whether action is allowed
- regression magnitude determines how strong or large the action should be

Variants to test:

- classifier-only gate, fixed sell percentage
- classifier gate + regression sizing
- classifier gate + regression consensus breadth filter
- classifier gate + regression neutral-band override

Decision outputs to study:

- mode agreement / disagreement with current production
- sell percentage changes
- stability of month-to-month recommendations

### v95 - Policy replay and promotion-style evaluation

Purpose:

- judge candidate classification/hybrid variants on recommendation usefulness,
  not just statistical metrics

Evaluation criteria:

- policy uplift versus current production baseline
- stability of recommendation mode
- sell percentage volatility
- abstention quality
- consistency with tax-aware default policy

Holdout discipline:

- preserve the repo's reserved holdout / promotion-check philosophy
- do not consume any final promotion holdout casually during model exploration

Expected outputs:

- classification-only vs regression-only vs hybrid policy table
- promotion gate memo

### v96 - Production candidacy decision

Purpose:

- decide whether any classification-led architecture should proceed to a
  production shadow or promotion cycle

Possible outcomes:

- no candidate is strong enough; keep current production unchanged
- classifier becomes a shadow-only monthly diagnostic
- hybrid gate becomes the next promotion candidate

Promotion criteria:

- policy-level improvement over current production
- no material stability regression
- understandable operator-facing behavior
- maintainable implementation complexity

---

## Model Families To Cover

The first full pass should cover:

- logistic regression
- regularized linear classifiers
- shallow tree boosting classifiers
- pooled panel classifiers
- benchmark-specific classifiers
- basket or breadth-target classifiers

The first pass should not prioritize:

- very deep tree models
- large neural networks
- opaque high-variance ensembles
- broad Bayesian or latent-factor structures unless a later cycle clearly
  justifies them
- large interaction expansions
- hard regime-split classifiers

This repo still lives in a small-sample, time-series setting. Simpler models
are more likely to be robust.

---

## Feature Exploration Strategy

The user explicitly wants deeper feature exploration. This plan should treat
that as first-class work, not an afterthought.

### Feature study structure

1. Start with the current `v46` lean feature set.
2. Define feature families from the existing engineered features.
3. Add families one at a time.
4. Remove families one at a time from the best candidate.
5. Evaluate:
   - pooled metrics
   - per-benchmark metrics
   - calibration quality
   - policy-level effect

### Why stepwise instead of one-shot feature search

- it is more interpretable
- it better fits the small-sample setting
- it helps identify which existing feature families actually move the decision
  problem
- it reduces the risk of accidental overfitting from large combinatorial
  searches

### Deliverable expectation

Each feature sweep should produce:

- a results CSV
- a concise summary markdown note
- a clearly named retained feature-set candidate for later versions

### What to de-prioritize during feature work

The archived reports were unusually consistent on this point. We should not
spend the early classification cycle on:

- broad interaction generation
- wide feature crosses across the full inventory
- full-inventory multivariate imputation by default
- regime splits that fragment the sample
- latent-factor compression before we understand the direct family-level signal

Those can be revisited later only if the lean, stepwise family study produces a
clear reason to do so.

---

## Per-Benchmark vs Combined-Panel Design

This is one of the most important questions in the classification program.

### Per-benchmark models may help because:

- each ETF has a different return distribution
- the signal may genuinely differ by benchmark type
- classification thresholds may be benchmark-specific

### Combined panel models may help because:

- data is limited
- pooling may improve sample efficiency
- one classifier may generalize more smoothly than many separate ones

### Plan stance

Do not assume one answer.

The `v89-v90` stages should compare:

- separate benchmark models
- one pooled panel model
- hybrid pooled structures with benchmark identity controls

This comparison should be treated as a major research axis, not a minor detail.

---

## Peer-Review Carry-Forward Map

The earlier review set already suggested several ideas that belong in this
program. This table keeps that provenance visible.

| Prior guidance | Carry-forward in this plan |
|---|---|
| Classification should be the primary prediction task for the business question | `v87`, `v92`, `v94-v96` |
| Composite benchmark or basket target should be tested explicitly | `v93` |
| Pooled panel estimation with benchmark fixed effects deserves direct study | `v90` |
| Keep final models lean and favor 5-8 effective features | `v88-v90` |
| Use operating, underwriting, macro, rates, and inflation-proxy features stepwise | `v88` |
| Prefer regime indicators over hard regime splits | `v88`, `v91` |
| Treat calibration and abstention as core, not optional | `v92`, `v94-v96` |
| Use classification as an action gate, not a wholesale regression replacement | `v94-v96` |
| Avoid high-dimensional search and repeated unstructured experimentation | governance sections below |

---

## Evaluation Hierarchy

The evaluation hierarchy for this cycle should be:

1. Policy and recommendation usefulness
2. Recommendation stability
3. Probability calibration quality
4. Classification discrimination metrics
5. Regression context metrics

That means a candidate with slightly weaker accuracy but materially better:

- abstention behavior
- mode stability
- policy uplift

may still be the better production candidate.

---

## Multiple-Testing Discipline

The archived reviews were also right to push on research hygiene.

This cycle should behave like a precommitted program, not an open-ended model
search.

Rules:

- the `v87-v96` sequence is the declared search boundary for this direction
- each version should have a written hypothesis, comparison set, and primary
  success criteria before execution
- feature-family additions should be evaluated at the family level before
  individual-feature tinkering
- reserve final promotion-style holdout checking for the end of the cycle
- if the cycle fails to produce a clearly better hybrid candidate, default to
  no promotion rather than expanding the search space casually

This is especially important here because classification makes it easy to
"improve" one surface metric while quietly degrading calibration or policy
stability.

---

## Artifact Framework

Planned artifact pattern:

- scripts in `results/research/v87_*.py` through `results/research/v96_*.py`
- outputs in matching `results/research/*_results.csv`
- tests in `tests/test_research_v87_*.py` through `tests/test_research_v96_*.py`
- shared helpers in `src/research/`

Likely shared helper areas:

- classification target builders
- feature-family registries
- pooled vs per-benchmark splitter helpers
- threshold sweep evaluators
- hybrid policy evaluators

---

## Test Plan

Each version should include flat pytest coverage validating:

- script execution shape
- required output columns
- no use of non-time-series validation
- correct benchmark / pooled row presence where applicable
- calibration metric presence where applicable
- policy comparison fields where applicable

Additional tests to add during this cycle:

- feature-family registry tests
- pooled panel alignment tests
- threshold sweep monotonicity / sanity tests
- hybrid gate behavior tests on synthetic data

---

## Recommended Execution Order

Recommended order:

1. `v87`
2. `v88`
3. `v89`
4. `v90`
5. `v92`
6. `v93`
7. `v91`
8. `v94`
9. `v95`
10. `v96`

Rationale:

- define the problem first
- understand features before broad model-family expansion
- compare separate vs pooled linear baselines before nonlinear models
- calibrate probabilities before making final hybrid policy decisions
- evaluate nonlinear models after a strong linear baseline exists

---

## Recommended Immediate Next Step

Start with `v87-v88` on this branch.

That means:

- formalize classification target definitions
- build the shared classification helper layer
- run the first stepwise feature sweep on top of the existing `v46` baseline

This is the cleanest way to deepen the classification program without jumping
too early into model-family sprawl.
