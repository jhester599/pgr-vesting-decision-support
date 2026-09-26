# v92 Calibration and Abstention Summary

Selected probability path: `separate_logistic_balanced__prequential_logistic__0.30_0.70`.

| candidate_name                         | calibration          |   lower_threshold |   upper_threshold |   balanced_accuracy |    ece_10 |   abstention_rate |   mean_policy_return | selected_next   |
|:---------------------------------------|:---------------------|------------------:|------------------:|--------------------:|----------:|------------------:|---------------------:|:----------------|
| separate_logistic_balanced             | prequential_logistic |              0.3  |              0.7  |            0.513234 | 0.0812762 |         0.0432099 |            0.0826672 | True            |
| separate_logistic_balanced             | raw                  |              0.4  |              0.6  |            0.555353 | 0.221274  |         0.0802469 |            0.0824819 | False           |
| separate_logistic_balanced             | prequential_logistic |              0.35 |              0.65 |            0.513234 | 0.0812762 |         0         |            0.0820759 | False           |
| separate_logistic_balanced             | prequential_logistic |              0.4  |              0.6  |            0.513234 | 0.0812762 |         0         |            0.0820759 | False           |
| separate_logistic_balanced             | raw                  |              0.35 |              0.65 |            0.555353 | 0.221274  |         0.160494  |            0.080245  | False           |
| pooled_fixed_effects_logistic_balanced | prequential_logistic |              0.35 |              0.65 |            0.539879 | 0.0730457 |         0.135802  |            0.0801348 | False           |
| pooled_fixed_effects_histgb_depth2     | prequential_logistic |              0.4  |              0.6  |            0.502754 | 0.0944872 |         0.0740741 |            0.0797734 | False           |
| pooled_fixed_effects_logistic_balanced | prequential_logistic |              0.4  |              0.6  |            0.539879 | 0.0730457 |         0.0679012 |            0.0795928 | False           |
| pooled_fixed_effects_logistic_balanced | prequential_logistic |              0.3  |              0.7  |            0.539879 | 0.0730457 |         0.228395  |            0.0789274 | False           |
| pooled_fixed_effects_histgb_depth2     | prequential_logistic |              0.35 |              0.65 |            0.502754 | 0.0944872 |         0.104938  |            0.0777157 | False           |
| separate_logistic_balanced             | raw                  |              0.3  |              0.7  |            0.555353 | 0.221274  |         0.234568  |            0.0771893 | False           |
| pooled_fixed_effects_logistic_balanced | raw                  |              0.4  |              0.6  |            0.572754 | 0.197745  |         0.0802469 |            0.0757965 | False           |
