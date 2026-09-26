# v148 Class Weight Search Summary

Date: 2026-04-16

## Baseline

- incumbent candidate: `positive_weight=1.0`
- `covered_ba=0.6987`
- `coverage=0.5476`

## Sweep

Tested weights: `0.75`, `1.00`, `1.25`, `1.50`, `2.00`

## Decision

Keep `results/research/v148_class_weight_candidate.txt` unchanged at `1.0`.

None of the tested bounded weights beat the incumbent on covered balanced
accuracy.
