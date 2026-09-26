# v146 Threshold Search Summary

Date: 2026-04-16

## Baseline

- incumbent candidate: `{"low": 0.15, "high": 0.70}`
- `covered_ba=0.6987`
- `coverage=0.5476`

## Sweep

Tested pairs:

- `(0.10, 0.65)`
- `(0.10, 0.70)`
- `(0.15, 0.65)`
- `(0.15, 0.70)`
- `(0.20, 0.70)`
- `(0.20, 0.75)`

## Decision

Keep `results/research/v146_threshold_candidate.json` unchanged.

No tested bounded pair beat the incumbent threshold combination on covered
balanced accuracy while offering a clearly better overall tradeoff.
