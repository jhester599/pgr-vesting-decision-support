# v87 Target Taxonomy Summary

Recommended forward binary target: `actionable_sell_3pct`.

Top pooled / basket rows:
| benchmark   | target                        |   balanced_accuracy |   brier_score |   ece_10 |   base_rate | recommended_next   |
|:------------|:------------------------------|--------------------:|--------------:|---------:|------------:|:-------------------|
| POOLED      | benchmark_underperform_0pct   |            0.554114 |      0.25459  | 0.209337 |    0.272031 | False              |
| POOLED      | actionable_sell_3pct          |            0.555353 |      0.240905 | 0.221274 |    0.212644 | True               |
| BASKET      | basket_underperform_0pct      |            0.632675 |      0.222408 | 0.175965 |    0.296296 | False              |
| BASKET      | breadth_underperform_majority |            0.6375   |      0.245069 | 0.25097  |    0.259259 | False              |
