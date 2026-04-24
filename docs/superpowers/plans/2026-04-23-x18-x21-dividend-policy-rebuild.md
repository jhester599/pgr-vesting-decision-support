# x18-x21 Dividend Policy Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the x-series special-dividend lane around policy-aware labels,
post-2018 modeling, and normalized target comparisons so we can judge whether
the dividend path is economically coherent enough to keep.

**Architecture:** Split the work into four research-only steps. x18 rebuilds
the dividend labels around the December 2018 policy change and the December to
February payout window. x19 reruns the annual two-stage model on post-policy
snapshots only, using persistent-BVPS and capital-generation features. x20
synthesizes the new evidence against x10 so the decision criteria stay
explicit. x21 compares raw-dollar and normalized size targets while keeping the
same annual split logic and low-complexity models.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn logistic/ridge,
existing x6 annual WFO helpers, x9/x10 capital features, x17 persistent BVPS,
checked-in database snapshot only.

---

## Scope

- x18 formalizes the dividend-policy regime split:
  - pre-December-2018 quantitative annual policy
  - post-December-2018 regular quarterly plus periodic specials
- x19 tests whether the dividend lane improves when we:
  - keep only post-policy annual observations
  - use persistent-BVPS/book-value-creation features
  - preserve the existing two-stage annual evaluation
- x20 answers whether the new lane actually changed the recommendation or only
  moved errors around.
- x21 tests whether stage-2 size is better modeled as:
  - raw excess dollars per share
  - excess as a share of BVPS / persistent BVPS
  - excess as a share of price

## Decision Log

Document the question and rule whenever the path forks:

- **Policy-break date:** Use December 2018 unless repo data contradicts it.
  Criterion: the date must align with the start of recurring quarterly
  dividends in the checked-in dividend history.
- **Payment window:** Use December through February for annual labels.
  Criterion: the window must cover the old January/February behavior and the
  newer December/January behavior without peeking beyond the immediate Q1
  payout season.
- **Training sample:** Prefer post-policy-only modeling for x19/x21.
  Criterion: the practical model should match the current policy regime even if
  that reduces sample size.
- **Target scale:** Keep a normalized target only if it improves expected-value
  dollar error after back-transformation. Criterion: normalized elegance alone
  is not enough.
- **Complexity:** Keep stage 1 and stage 2 conservative.
  Criterion: no target-scale experiment justifies a more complex learner unless
  it beats the low-bias baselines out of sample.

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/superpowers/plans/2026-04-23-x18-x21-dividend-policy-rebuild.md` | Create | Plan |
| `src/research/x18_dividend_policy_regime.py` | Create | Policy split and label construction |
| `scripts/research/x18_dividend_policy_regime.py` | Create | x18 audit runner |
| `tests/test_research_x18_dividend_policy_regime.py` | Create | x18 tests |
| `src/research/x19_post_policy_dividend_model.py` | Create | Post-policy feature-set helpers |
| `scripts/research/x19_post_policy_dividend_model.py` | Create | x19 runner |
| `tests/test_research_x19_post_policy_dividend_model.py` | Create | x19 tests |
| `src/research/x20_dividend_policy_synthesis.py` | Create | x20 synthesis helpers |
| `scripts/research/x20_dividend_policy_synthesis.py` | Create | x20 runner |
| `tests/test_research_x20_dividend_policy_synthesis.py` | Create | x20 tests |
| `src/research/x21_dividend_target_scales.py` | Create | Normalized target helpers |
| `scripts/research/x21_dividend_target_scales.py` | Create | x21 runner |
| `tests/test_research_x21_dividend_target_scales.py` | Create | x21 tests |
| `results/research/x18_*` | Create | x18 artifacts |
| `results/research/x19_*` | Create | x19 artifacts |
| `results/research/x20_*` | Create | x20 artifacts |
| `results/research/x21_*` | Create | x21 artifacts |

## Task 1: x18 Tests First

- [ ] Test the dividend audit labels the pre-policy and post-policy eras
      correctly around December 2018.
- [ ] Test the annual target window spans December through February.
- [ ] Test post-policy annual labels compute special-dividend excess relative
      to the inferred regular baseline.

## Task 2: x18 Implementation

- [ ] Implement regime classification.
- [ ] Implement November snapshot target construction with a December-February
      payout window.
- [ ] Write x18 audit CSV, target CSV, summary JSON, and memo.

## Task 3: x19 Tests First

- [ ] Test post-policy filtering keeps only eligible annual rows.
- [ ] Test x19 feature sets include a persistent-BVPS/capital block.
- [ ] Run x19 unit tests before touching the annual runner.

## Task 4: x19 Implementation

- [ ] Build the post-policy annual frame from x18 labels and x17 persistent
      BVPS.
- [ ] Reuse the x6 two-stage evaluation with low-count feature sets only.
- [ ] Write x19 detail CSV, summary JSON, and memo.

## Task 5: x20 Tests First

- [ ] Test x20 recognizes whether x19 beats x10 on expected-value MAE.
- [ ] Test x20 keeps confidence low when post-policy annual OOS observations
      remain below 12.
- [ ] Test x20 recommendation stays research-only.

## Task 6: x20 Implementation

- [ ] Read x10, x18, and x19 artifacts.
- [ ] Compare x10 and x19 leaders, sample sizes, and policy scope.
- [ ] Write x20 synthesis JSON and memo with explicit decision criteria.

## Task 7: x21 Tests First

- [ ] Test normalized targets divide by the right scale and recover the raw
      dollar target after back-transformation.
- [ ] Test scale comparison ranks rows by dollar expected-value MAE, not only
      normalized error.
- [ ] Test invalid zero or missing scales are handled fold-locally and
      conservatively.

## Task 8: x21 Implementation

- [ ] Build raw-dollar, BVPS-normalized, persistent-BVPS-normalized, and
      price-normalized annual targets from the post-policy frame.
- [ ] Reuse the x19 stage-1 path and compare conservative stage-2 scale
      regressions after converting back to dollar expected values.
- [ ] Write x21 detail CSV, summary JSON, and memo.

## Verification

- [ ] Run x18-x21 unit tests.
- [ ] Run x18-x21 scripts and validate JSON artifacts.
- [ ] Run focused dividend-lane tests spanning x6, x10, x17, x18, x19, x20,
      and x21.
- [ ] Run `py_compile` on all new modules, scripts, and tests.

## Production Boundary

Do not edit `scripts/monthly_decision.py`, monthly output artifacts, dashboard
plumbing, production configs, or any `v###` lane files. This entire package is
research-only.
