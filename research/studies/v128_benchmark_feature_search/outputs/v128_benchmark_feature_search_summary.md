# v128 Benchmark-Specific Feature Search Summary

v128 performs a full benchmark-specific feature search across the 72-feature non-target research matrix while preserving the current benchmark-specific balanced-logistic classifier family, rolling WFO geometry, and prequential logistic calibration path.

## Pooled Comparison

| benchmark   | method            | selector_penalty   |   n_features | features   |   n_obs |   n_covered |   coverage |   accuracy |   balanced_accuracy |   balanced_accuracy_covered |   brier_score |   log_loss |   precision |    recall |   base_rate |   predicted_positive_rate |    ece_10 | production_eligible   |   benchmark_count |
|:------------|:------------------|:-------------------|-------------:|:-----------|--------:|------------:|-----------:|-----------:|--------------------:|----------------------------:|--------------:|-----------:|------------:|----------:|------------:|--------------------------:|----------:|:----------------------|------------------:|
| POOLED      | lean_baseline     |                    |            0 |            |    1362 |        1185 |   0.870044 |   0.760646 |            0.506759 |                    0.5      |      0.181347 |   0.5548   |    0.545455 | 0.0183486 |    0.240088 |                0.00807636 | 0.0488489 | True                  |                10 |
| POOLED      | final_feature_map |                    |            0 |            |    1362 |        1211 |   0.889134 |   0.749633 |            0.498467 |                    0.501596 |      0.181945 |   0.551626 |    0.208333 | 0.0152905 |    0.240088 |                0.0176211  | 0.0386939 | True                  |                10 |

## Benchmark Winners

| benchmark   | selected_method       |   n_features |   balanced_accuracy_covered |   delta_balanced_accuracy_covered |   delta_ece_10 |   delta_brier_score | switched_from_baseline   |
|:------------|:----------------------|-------------:|----------------------------:|----------------------------------:|---------------:|--------------------:|:-------------------------|
| BND         | elastic_net_consensus |           12 |                    0.552379 |                         0.0722587 |    -0.0102955  |         -0.0193014  | True                     |
| DBC         | elastic_net_consensus |           12 |                    0.56079  |                         0.0656446 |     0.00169892 |         -0.0291001  | True                     |
| GLD         | lean_baseline         |           12 |                    0.565615 |                         0         |     0          |          0          | False                    |
| VDE         | lean_baseline         |           12 |                    0.5      |                         0         |     0          |          0          | False                    |
| VGT         | forward_stepwise      |            2 |                    0.947368 |                         0.368421  |    -0.0718079  |         -0.0340321  | True                     |
| VIG         | elastic_net_consensus |           12 |                    0.574359 |                         0.0666384 |     0.00835594 |         -0.00792547 | True                     |
| VMBS        | lean_baseline         |           12 |                    0.549242 |                         0         |     0          |          0          | False                    |
| VOO         | lean_baseline         |           12 |                    0.5      |                         0         |     0          |          0          | False                    |
| VWO         | lean_baseline         |           12 |                    0.5      |                         0         |     0          |          0          | False                    |
| VXUS        | lean_baseline         |           12 |                    0.491379 |                         0         |     0          |          0          | False                    |

Benchmarks switching away from the incumbent baseline: `4`.

## Ridge Diagnostic Control

| benchmark   | method                  | selector_penalty   |   n_features | features   |   n_obs |   n_covered |   coverage |   accuracy |   balanced_accuracy |   balanced_accuracy_covered |   brier_score |   log_loss |   precision |   recall |   base_rate |   predicted_positive_rate |    ece_10 | production_eligible   |   benchmark_count |
|:------------|:------------------------|:-------------------|-------------:|:-----------|--------:|------------:|-----------:|-----------:|--------------------:|----------------------------:|--------------:|-----------:|------------:|---------:|------------:|--------------------------:|----------:|:----------------------|------------------:|
| POOLED      | ridge_full_pool_control |                    |            0 |            |    1362 |        1321 |   0.969897 |   0.744493 |            0.489855 |                    0.494006 |      0.192725 |   0.579287 |           0 |        0 |    0.240088 |                 0.0154185 | 0.0503022 | True                  |                10 |

## Artifact Notes

- `v128_feature_inventory.csv`: benchmark-by-feature availability and eligibility
- `v128_single_feature_results.csv`: full single-feature leaderboard
- `v128_forward_stepwise_trace.csv`: every considered forward-stepwise addition
- `v128_regularized_selection_detail.csv`: fold-level L1 / elastic-net feature selections
- `v128_regularized_comparison.csv`: evaluated L1, elastic-net, and ridge candidates
- `v128_benchmark_feature_map.csv`: final benchmark-specific recommendation map
