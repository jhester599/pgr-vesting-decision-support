# v147 Aggregate Search Summary

Date: 2026-04-16

## Baseline

- incumbent candidate: `path_b_multiplier=1.0`
- `covered_ba=0.5000`
- `coverage=0.4405`

## Sweep

Tested multipliers: `0.50`, `0.75`, `1.00`, `1.25`, `1.50`, `2.00`

## Decision

Keep `results/research/v147_path_b_multiplier_candidate.txt` unchanged at
`1.0`.

None of the tested bounded multipliers improved covered balanced accuracy above
the baseline proxy.
