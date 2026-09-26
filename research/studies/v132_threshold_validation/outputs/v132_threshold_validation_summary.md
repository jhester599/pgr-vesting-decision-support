# v132 Threshold Validation Summary

**Holdout cutoff:** 2021-12-31
**Selection set:** 63 rows
**Hold-out set:** 21 rows
**Candidate pairs evaluated:** 58

## Selection Set -- Top 5 Pairs

| low | high | covered_ba | coverage |
|-----|------|-----------|---------|
| 0.10 | 0.60 | 0.7750 | 0.4762 |
| 0.10 | 0.65 | 0.7599 | 0.4286 |
| 0.10 | 0.55 | 0.7548 | 0.4921 |
| 0.35 | 0.60 | 0.7071 | 0.7778 |
| 0.15 | 0.60 | 0.7067 | 0.5873 |

**Selection winner:** low=0.10, high=0.60

## Hold-out Set -- Candidate Evaluation

| pair | low | high | covered_ba | coverage | delta_vs_baseline |
|------|-----|------|-----------|---------|-----------------|
| baseline | 0.30 | 0.70 | 0.5000 | 1.0000 | +0.0000 |
| a_priori_v131 | 0.15 | 0.70 | 0.5000 | 0.2857 | +0.0000 |
| selection_winner | 0.10 | 0.60 | 0.5000 | 0.0000 | +0.0000 |

## Verdict

> DO NOT ADOPT: BA delta +0.0000 < 0.03.
