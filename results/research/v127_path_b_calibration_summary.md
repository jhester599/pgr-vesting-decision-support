# v127 Path B Calibration Sweep

Run date: `2026-04-12`
Input fold frame: `C:/Users/Jeff/Documents/pgr-vesting-decision-support/results/research/v125_portfolio_target_fold_detail.csv`
Matched OOS observations: `84`
Warmup before activating prequential calibrators: `24` observations

## Candidate Comparison

| model               |   balanced_accuracy_covered |   brier_score |   log_loss |    ece |   coverage | best_calibrated_candidate   | selected_next   |
|:--------------------|----------------------------:|--------------:|-----------:|-------:|-----------:|:----------------------------|:----------------|
| path_a_matched_v126 |                      0.5    |        0.2058 |     0.6132 | 0.1269 |     0.7619 | False                       | False           |
| path_b_raw_v126     |                      0.645  |        0.2188 |     0.8207 | 0.2234 |     0.8333 | False                       | False           |
| path_b_platt_v127   |                      0.5    |        0.2158 |     0.6638 | 0.1387 |     0.7738 | True                        | False           |
| path_b_temp_v127    |                      0.5725 |        0.1917 |     0.601  | 0.157  |     0.75   | False                       | False           |

## Selected Candidate

- best calibrated candidate: `path_b_platt_v127`
- adoption candidate selected_next: `none`
- delta vs raw covered balanced accuracy: `-0.1450`
- delta vs raw Brier score: `-0.0030`
- delta vs raw log loss: `-0.1569`
- delta vs raw ECE: `-0.0847`

## Temperature Notes

- average fitted temperature after warmup: `2.351`
- min / max fitted temperature after warmup: `0.850` / `3.000`

## Verdict

The best v127 calibration candidate improves Path B's reliability metrics, but no candidate clears the adoption gate because covered balanced accuracy falls too much versus raw v126. Continue calibration research before replacing the raw Path B probabilities.
