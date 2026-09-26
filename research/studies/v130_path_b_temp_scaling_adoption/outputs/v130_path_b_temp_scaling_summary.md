# v130 Path B Temperature Scaling Revised Adoption Analysis

Run date: `2026-04-12`
Input fold frame: `C:/Users/Jeff/Documents/pgr-vesting-decision-support/results/research/v125_portfolio_target_fold_detail.csv`
Matched OOS observations: `84`
Warmup (prequential calibration): `24` observations
Coverage abstain window: `[0.3, 0.7]`

## Background

v127 rejected temperature scaling because it compared against raw Path B.
The correct baseline is **Path A matched** (the incumbent signal).
This script re-evaluates with the corrected adoption criterion.

## Candidate Comparison

| model              |   balanced_accuracy_covered |   brier_score |   log_loss |   ece_10 |   coverage |
|:-------------------|----------------------------:|--------------:|-----------:|---------:|-----------:|
| path_a_matched     |                      0.5    |        0.2058 |     0.6132 |   0.1269 |     0.7619 |
| path_b_raw         |                      0.645  |        0.2188 |     0.8207 |   0.2234 |     0.8333 |
| path_b_temp_scaled |                      0.5725 |        0.1917 |     0.601  |   0.157  |     0.75   |

## Adoption Criteria (vs Path A matched)

| Criterion | Threshold | Observed | Pass |
|---|---|---|---|
| A: BA delta | >= 0.03 | +0.0725 | YES |
| B: Brier excess | <= 0.02 | -0.0141 | YES |
| C: ECE ratio | <= 1.5x | 1.2372x | YES |

## Verdict

ADOPT temperature scaling for Path B shadow signal. Replace raw Path B probability with temperature-scaled probability in the investable-pool aggregate computation.
