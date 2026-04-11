# v94 Hybrid Gate Summary

Best non-regression policy variant: `classifier_only_benchmark_panel`.
Benchmark probability source: `separate_logistic_balanced`.
Basket probability source: `breadth_underperform_majority`.

| variant                          |   mean_policy_return |   uplift_vs_sell_50 |   capture_ratio |   hold_fraction_changes | selected_next   |
|:---------------------------------|---------------------:|--------------------:|----------------:|------------------------:|:----------------|
| classifier_only_benchmark_panel  |            0.0820759 |           0.041038  |        0.837291 |                       1 | True            |
| classifier_only_best_basket      |            0.0780288 |           0.0369909 |        0.796005 |                      15 | False           |
| hybrid_benchmark_panel_30_70     |            0.0683909 |           0.0273529 |        0.697684 |                      17 | False           |
| hybrid_benchmark_panel_35_65     |            0.0680893 |           0.0270513 |        0.694607 |                      17 | False           |
| hybrid_best_basket_35_65         |            0.0679074 |           0.0268694 |        0.692752 |                      23 | False           |
| regression_only_quality_weighted |            0.067588  |           0.0265501 |        0.689494 |                      21 | False           |
