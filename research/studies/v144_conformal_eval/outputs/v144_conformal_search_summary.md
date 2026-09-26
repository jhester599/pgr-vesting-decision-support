# v144 Conformal Search Summary

Date: 2026-04-16

## Baseline

- incumbent candidate: `{"coverage": 0.80, "aci_gamma": 0.05}`
- realized replay metrics:
  - `coverage=0.7962`
  - `target_coverage=0.8000`
  - `coverage_gap=-0.0038`

## Sweep

Tested pairs:

- `(0.75, 0.03)`
- `(0.75, 0.05)`
- `(0.80, 0.05)`
- `(0.85, 0.05)`
- `(0.85, 0.10)`

Best bounded candidate:

- `{"coverage": 0.75, "aci_gamma": 0.03}`
- `coverage=0.7490`
- `target_coverage=0.7500`
- `coverage_gap=-0.0010`

## Decision

Update `results/research/v144_conformal_candidate.json` to
`{"coverage": 0.75, "aci_gamma": 0.03}` as the cleanest bounded replay result.
