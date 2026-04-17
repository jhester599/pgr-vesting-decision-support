# v149 Kelly Search Summary

Date: 2026-04-16

## Baseline

- incumbent candidate: `{"fraction": 0.25, "cap": 0.20}`
- `utility_score=0.0010`
- `coverage=0.2531`
- `success_rate=0.8293`

## Sweep

Tested pairs:

- `(0.10, 0.10)`
- `(0.15, 0.15)`
- `(0.25, 0.20)`
- `(0.35, 0.20)`
- `(0.35, 0.25)`
- `(0.50, 0.25)`

Best bounded candidate:

- `{"fraction": 0.50, "cap": 0.25}`
- `utility_score=0.0021`
- `coverage=0.4506`
- `success_rate=0.7671`

## Decision

Update `results/research/v149_kelly_candidate.json` to
`{"fraction": 0.50, "cap": 0.25}`.

This is the strongest bounded utility-score result on the replay-proxy frame,
with the caveat that it is materially more aggressive than the incumbent.
