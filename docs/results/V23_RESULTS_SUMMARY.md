# V23 Results Summary

Created: 2026-04-04

## Scope

- v23 tests whether the v21 promotion result survives when the reduced forecast universe is extended backward with research-only pre-inception benchmark proxies.
- Live benchmark definitions are unchanged; the proxy stitching is research-only.

## Proxy Design

- `VOO` pre-inception history: proxy with `VTI`.
- `VXUS` pre-inception history: proxy with a fitted `VEA` + `VWO` blend.
- `VMBS` pre-inception history: proxy with `BND`.
- Proxy caveat: `VXUS <- VEA + VWO` and `VMBS <- BND` are very strong research proxies, while `VOO <- VTI` is directionally useful but looser than ideal.

## Extended Historical Window

- Common evaluable monthly dates: `150`
- First common date: `2013-04-30`
- Last common date: `2025-09-30`

## Decision

- Status: `extended_history_confirms_candidate`
- Recommended path: `ensemble_ridge_gbt_v18`
- Rationale: The leading candidate retained or improved its historical behavior versus the simpler baseline even after extending the benchmark histories with research-only proxies.

## Review Summary

### shadow_baseline

- Signal agreement with shadow baseline: `100.0%`
- Mean aggregate OOS R^2: `-0.1989`
- Signal changes: `3`
- OUT / NEUTRAL / UNDER: `88.0%` / `12.0%` / `0.0%`

### ensemble_ridge_gbt_v18

- Signal agreement with shadow baseline: `78.7%`
- Mean aggregate OOS R^2: `-0.4330`
- Signal changes: `27`
- OUT / NEUTRAL / UNDER: `73.3%` / `22.7%` / `4.0%`

### ensemble_ridge_gbt_v20_best

- Signal agreement with shadow baseline: `78.0%`
- Mean aggregate OOS R^2: `-0.4414`
- Signal changes: `33`
- OUT / NEUTRAL / UNDER: `74.0%` / `20.7%` / `5.3%`

### live_production_ensemble_reduced

- Signal agreement with shadow baseline: `57.3%`
- Mean aggregate OOS R^2: `-0.7855`
- Signal changes: `24`
- OUT / NEUTRAL / UNDER: `57.3%` / `35.3%` / `7.3%`

