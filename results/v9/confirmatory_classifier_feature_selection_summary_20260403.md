# Confirmatory Classifier Follow-Up Summary

Created: 2026-04-03  
Scope: Ridge confirmatory classifier follow-up after the initial v9 classification experiments

## Why this follow-up was run

The first confirmatory-classifier pass showed that a binary outperform model was
directionally reasonable, but it did not improve policy outcomes when used as a
standalone rule or as a hard confirmation gate for the lean regression models.

Because the Ridge classifier had the best initial classification metrics, this
follow-up tested whether classification needed its own feature set rather than
reusing the regression features.

## What was tested

The follow-up kept the regression side fixed at `ridge_lean_v1` on the reduced
`balanced_core7` benchmark universe, then tuned the classifier itself.

Work completed:

1. Tested every available feature one at a time as a Ridge classifier.
2. Measured both:
   - classification quality
   - decision utility when used as a hybrid confirmation gate
3. Built the classifier forward one feature at a time.
4. Chose the smallest near-best set based on:
   - hybrid uplift versus the regression sign policy
   - balanced accuracy
   - Brier score
   - model size

## Key outputs

- `confirmatory_classifier_detail_20260403.csv`
- `confirmatory_classifier_summary_20260403.csv`
- `classifier_feature_selection_detail_20260403.csv`
- `classifier_feature_selection_summary_20260403.csv`
- `classifier_forward_selection_trace_20260403.csv`

## Best single feature

`investment_book_yield`

Metrics:

- mean balanced accuracy: `0.6135`
- mean Brier score: `0.2608`
- mean hybrid policy return: `0.0675`
- hybrid uplift vs Ridge regression sign policy: `-0.0024`

Interpretation:

- this was the best one-feature tradeoff
- it materially improved classifier quality relative to the untuned version
- it came close to matching the Ridge regression sign policy while adding only
  one classifier feature

## Best tuned classifier set

Recommended 12-feature Ridge classifier:

- `investment_book_yield`
- `roe`
- `npw_growth_yoy`
- `buyback_yield`
- `credit_spread_hy`
- `gainshare_est`
- `unearned_premium_growth_yoy`
- `roe_net_income_ttm`
- `nfci`
- `combined_ratio_ttm`
- `investment_income_growth_yoy`
- `yield_slope`

Best threshold from the sweep:

- `0.50`

Metrics:

- mean balanced accuracy: `0.6736`
- mean Brier score: `0.2332`
- mean hybrid policy return: `0.0688`
- hybrid uplift vs Ridge regression sign policy: `-0.0011`

Interpretation:

- this is the strongest classifier-specific feature set found in the sweep
- it meaningfully improves classification metrics
- it still does not clearly beat the simpler regression sign policy on decision
  utility

## Recommendation

Do not replace the lean regression candidate with the classifier.

Do keep the Ridge confirmatory classifier as a sidecar research candidate for:

- confidence diagnostics
- abstention research
- future “only act when both agree” tests if the regression stack improves

Current preferred use:

- primary decision signal: lean regression candidate
- secondary diagnostic: tuned Ridge classifier probability

## Practical takeaway

The classifier idea was worth pursuing.

The right conclusion is not:

- “classification solves the problem”

The right conclusion is:

- “a tuned classifier can become a credible confidence sidecar, but it is not
  yet strong enough to be the main decision engine”
