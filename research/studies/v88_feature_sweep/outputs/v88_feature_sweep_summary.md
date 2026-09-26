# v88 Feature Sweep Summary

Forward feature set for later versions: `lean_baseline`.

| feature_set                     |   balanced_accuracy |   brier_score |   ece_10 |   n_features | selected_next   |
|:--------------------------------|--------------------:|--------------:|---------:|-------------:|:----------------|
| lean_baseline                   |            0.555353 |      0.240905 | 0.221274 |           12 | True            |
| lean_plus_inflation             |            0.553709 |      0.241347 | 0.223348 |           22 | False           |
| lean_plus_benchmark_context     |            0.549993 |      0.239972 | 0.21996  |           16 | False           |
| lean_plus_inflation_and_context |            0.548169 |      0.240725 | 0.222256 |           26 | False           |
| lean_plus_investment            |            0.545982 |      0.249265 | 0.250038 |           17 | False           |
| lean_plus_extended_ops          |            0.51062  |      0.275722 | 0.258446 |           29 | False           |
| lean_plus_ops_and_context       |            0.508187 |      0.274658 | 0.258997 |           33 | False           |
| lean_plus_all_curated           |            0.482475 |      0.387523 | 0.402365 |           56 | False           |
| lean_plus_valuation             |            0.455185 |      0.452397 | 0.4803   |           20 | False           |
