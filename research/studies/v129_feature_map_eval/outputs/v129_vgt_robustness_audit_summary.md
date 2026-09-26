# v129 VGT Robustness Audit Summary

## Objective

Audit the v128 VGT forward-stepwise 2-feature finding
(`rate_adequacy_gap_yoy`, `severity_index_yoy`) for temporal stability.
The v128 search reported covered balanced accuracy of 0.947 (n_covered=21)
vs lean_baseline BA of 0.579 -- a +0.368 advantage that demands scrutiny.

## Methodology

- Re-run the WFO evaluation at 3 as-of dates: 2022-03-31, 2023-03-31, 2024-03-31
- Same WFO geometry: max_train=60, test_size=6, gap=8 (TimeSeriesSplit)
- Same model: LogisticRegression(C=0.5, class_weight='balanced', penalty='l2')
- Same calibration: prequential logistic calibration
- Covered BA computed on observations where P(sell) <= 0.30 or >= 0.70

## Results

| As-of Date | Model | n_obs | n_covered | BA_covered | Brier |
|------------|-------|-------|-----------|------------|-------|
| 2022-03-31 | vgt_2feature | 150 | 5 | 0.6667 | 0.2519 |
| 2022-03-31 | lean_baseline | 150 | 38 | 0.5789 | 0.2748 |
| 2023-03-31 | vgt_2feature | 162 | 9 | 0.8571 | 0.2460 |
| 2023-03-31 | lean_baseline | 162 | 38 | 0.5789 | 0.2744 |
| 2024-03-31 | vgt_2feature | 174 | 21 | 0.9474 | 0.2308 |
| 2024-03-31 | lean_baseline | 174 | 38 | 0.5789 | 0.2648 |

## BA Advantage (2-feature minus lean_baseline)

- **2022-03-31**: delta_BA = +0.0877, n_covered = 5
- **2023-03-31**: delta_BA = +0.2782, n_covered = 9
- **2024-03-31**: delta_BA = +0.3684, n_covered = 21

## Single-Feature Stability Check

- 2022-03-31: top-3 = ['buyback_acceleration', 'pb_ratio', 'excess_bond_premium_proxy']; VGT 2-feature in top-3: []
- 2023-03-31: top-3 = ['yield_curvature', 'rate_adequacy_gap_yoy', 'buyback_acceleration']; VGT 2-feature in top-3: ['rate_adequacy_gap_yoy']
- 2024-03-31: top-3 = ['rate_adequacy_gap_yoy', 'pgr_price_to_book_relative', 'buyback_acceleration']; VGT 2-feature in top-3: ['rate_adequacy_gap_yoy']

## Economic Plausibility

PGR (Progressive Corp) is a P&C insurance company. VGT is a technology
sector ETF. The two winning features are insurance-specific:

- `rate_adequacy_gap_yoy`: measures how much PGR's pricing power is
  changing year-over-year. When PGR raises rates faster than loss costs
  grow, underwriting margins expand, boosting PGR relative to non-insurance
  benchmarks like VGT.
- `severity_index_yoy`: captures claims severity inflation. Rising severity
  compresses margins unless offset by rate adequacy, creating a direct
  headwind to PGR vs tech-sector returns.

These features are economically sensible for predicting PGR-vs-VGT relative
returns because VGT has essentially zero exposure to underwriting cycle
dynamics, making insurance-specific features orthogonal to VGT's drivers.

## Verdict

**UNSTABLE**

Adoption criteria: BA advantage >= +0.05 at all 3 as-of dates AND n_covered >= 10 at every date.

Recommendation: do NOT adopt. The v128 VGT result does not satisfy the temporal stability criteria. Retain the lean_baseline for VGT.