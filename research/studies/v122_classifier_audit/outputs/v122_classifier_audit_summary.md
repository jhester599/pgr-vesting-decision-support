# v122 Classifier Audit

As-of monthly run: `2026-04-11`.
Feature anchor date: `2026-03-31`.

## Implemented Shadow Classifier

- monthly shadow model family: `separate_benchmark_logistic_balanced`
- target: `actionable_sell_3pct`
- feature set: `lean_baseline`
- calibration: `oos_logistic_calibration`
- aggregation: benchmark-quality weighted probability pool

Lean baseline features:

- `mom_12m`
- `vol_63d`
- `yield_slope`
- `real_yield_change_6m`
- `real_rate_10y`
- `credit_spread_hy`
- `nfci`
- `vix`
- `combined_ratio_ttm`
- `investment_income_growth_yoy`
- `book_value_per_share_growth_yoy`
- `npw_growth_yoy`

## Training Scope

- benchmark coverage: BND, DBC, GLD, VDE, VMBS, VOO, VWO, VXUS
- benchmarks covered: `8`
- pooled benchmark-month sample used in the selected `v90` pooled reference: `1292`
- current monthly shadow implementation fits one logistic model per benchmark on all history available up to the as-of month, with explicit target truncation to prevent backdated leakage

Per-benchmark training rows used in the current audit fit:

- `BND` (`train_n=222`): `combined_ratio_ttm` (1.088), `book_value_per_share_growth_yoy` (0.397), `yield_slope` (0.367)
- `DBC` (`train_n=236`): `combined_ratio_ttm` (1.677), `mom_12m` (0.720), `vix` (0.653)
- `GLD` (`train_n=251`): `combined_ratio_ttm` (1.208), `real_rate_10y` (0.512), `nfci` (0.393)
- `VDE` (`train_n=252`): `combined_ratio_ttm` (1.229), `real_yield_change_6m` (0.487), `real_rate_10y` (0.484)
- `VMBS` (`train_n=190`): `combined_ratio_ttm` (1.197), `yield_slope` (0.723), `credit_spread_hy` (0.597)
- `VOO` (`train_n=181`): `combined_ratio_ttm` (1.151), `credit_spread_hy` (0.850), `real_yield_change_6m` (0.770)
- `VWO` (`train_n=247`): `credit_spread_hy` (1.768), `combined_ratio_ttm` (1.527), `real_yield_change_6m` (0.674)
- `VXUS` (`train_n=176`): `combined_ratio_ttm` (0.883), `credit_spread_hy` (0.749), `real_yield_change_6m` (0.665)

## Accuracy And Calibration

- best pooled classifier family from `v90`: `pooled_shared_logistic_balanced`
- pooled accuracy: `0.6889`
- pooled balanced accuracy: `0.5827`
- pooled Brier score: `0.2278`
- pooled log loss: `0.7471`

- best calibrated shadow-style path from `v92`: `separate_logistic_balanced__prequential_logistic__0.30_0.70`
- calibrated accuracy: `0.7538`
- calibrated balanced accuracy: `0.5132`
- calibrated Brier score: `0.1852`
- calibrated log loss: `0.5985`
- calibrated ECE: `0.0813`

## Current-Month Probability Snapshot

| benchmark   |   classifier_prob_actionable_sell |   classifier_weight |   classifier_weighted_contribution | classifier_shadow_tier   |
|:------------|----------------------------------:|--------------------:|-----------------------------------:|:-------------------------|
| BND         |                              25.7 |                13.4 |                                3.5 | HIGH                     |
| DBC         |                              40.2 |                12.4 |                                5   | LOW                      |
| GLD         |                              43.7 |                12   |                                5.2 | LOW                      |
| VDE         |                              33.4 |                11.7 |                                3.9 | MODERATE                 |
| VMBS        |                              26.8 |                13.3 |                                3.6 | HIGH                     |
| VOO         |                              41.9 |                10.6 |                                4.5 | LOW                      |
| VWO         |                              35.1 |                13.4 |                                4.7 | MODERATE                 |
| VXUS        |                              37.4 |                13.2 |                                4.9 | MODERATE                 |

Aggregated `P(Actionable Sell)`: `35.2%`.
Confidence tier: `MODERATE`.
Stance: `NEUTRAL`.
Top supporting benchmark: `GLD`.

## Approximate Feature Importance

This audit uses standardized absolute logistic coefficients within each benchmark as an approximate importance measure. That is directionally useful, but it is not a causal importance ranking.

Top weighted features across the eight-benchmark monthly pool:

- `combined_ratio_ttm`: weighted importance 1.245
- `credit_spread_hy`: weighted importance 0.653
- `real_yield_change_6m`: weighted importance 0.507
- `mom_12m`: weighted importance 0.378
- `yield_slope`: weighted importance 0.349
- `real_rate_10y`: weighted importance 0.310

Interpretation:

- `combined_ratio_ttm` dominates the current audit, which means underwriting quality is the strongest repeated driver of the actionable-sell classifier
- `credit_spread_hy`, `real_yield_change_6m`, `yield_slope`, and `real_rate_10y` show that macro / rates / credit conditions are the next most important layer
- `mom_12m` matters, but it is not the leading driver in the current audit
- the current classifier is therefore not just a momentum rule; it is leaning heavily on insurance fundamentals plus macro stress context

Detailed coefficients: `C:/Users/Jeff/Documents/pgr-vesting-decision-support/results/research/v122_classifier_audit_coefficients.csv`
Aggregated feature totals: `C:/Users/Jeff/Documents/pgr-vesting-decision-support/results/research/v122_classifier_audit_feature_totals.csv`
