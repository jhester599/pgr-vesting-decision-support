# v150 Neutral Band Search Summary

Date: 2026-04-16

## Baseline

- incumbent candidate: `neutral_band=0.015`
- evaluated on the updated Kelly candidate:
  `{"fraction": 0.50, "cap": 0.25}`
- `success_rate=0.7671`
- `coverage=0.4506`
- `utility_score=0.0021`

## Sweep

Tested bands: `0.000`, `0.010`, `0.015`, `0.020`, `0.030`, `0.050`

Utility score remained flat at `0.0021` across the tested grid.

## Decision

Keep `results/research/v150_neutral_band_candidate.txt` unchanged at `0.015`.

Wider bands increased selectivity and sometimes success rate, but they did not
produce a clearly better stable tradeoff once coverage loss was considered.
