# v172 VGT Selector-Agreement Gate Summary

## Objective

Complete the VGT governance review by adding the L1 and elastic-net
selector-agreement gate to the existing v129 BA audit results.
Governance rule: adopt only if forward-stepwise and regularized selectors
agree on the same signal cluster (rate_adequacy_gap_yoy, severity_index_yoy).

## v129 BA Results (reproduced)

| As-of Date | VGT 2-feat BA | Lean BA | Delta | n_covered | Fwd-stepwise top-3 |
|------------|--------------|---------|-------|-----------|-------------------|
| 2022-03-31 | 0.6667 | 0.5789 | +0.0878 | 5 | ['buyback_acceleration', 'pb_ratio', 'excess_bond_premium_proxy'] |
| 2023-03-31 | 0.8571 | 0.5789 | +0.2782 | 9 | ['yield_curvature', 'rate_adequacy_gap_yoy', 'buyback_acceleration'] |
| 2024-03-31 | 0.9474 | 0.5789 | +0.3685 | 21 | ['rate_adequacy_gap_yoy', 'pgr_price_to_book_relative', 'buyback_acceleration'] |

## Regularized Selector Gate Results

| As-of Date | L1 selects rate_adequacy? | L1 selects severity? | EN selects rate_adequacy? | EN selects severity? | Gate passed? |
|------------|--------------------------|---------------------|--------------------------|---------------------|-------------|
| 2022-03-31 | NO | NO | NO | YES | NO |
| 2023-03-31 | NO | NO | NO | NO | NO |
| 2024-03-31 | NO | NO | NO | NO | NO |

## Synthesis

- BA advantage >= +5% at all 3 dates: YES
- Regularized gate passed at: 0/3 dates (required >= 2 for CONDITIONAL_SHADOW)
- Single-feature top-3 agreement: rate_adequacy_gap_yoy appears at 2022 NO / 2023 YES / 2024 YES

## Verdict: REJECT

The regularized gate did not provide sufficient agreement to justify
even shadow-only adoption of the VGT 2-feature subset.

**Recommended action:** Retain VGT on lean_baseline.  The UNSTABLE
verdict from v129 stands.  Revisit if/when additional pre-holdout
data becomes available or a broader feature set is evaluated.