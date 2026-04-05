# V23 Results Summary

Created: 2026-04-05

## Scope

- v23 tests whether the v21 promotion result survives when the reduced forecast universe is extended backward with research-only pre-inception benchmark proxies.
- Live benchmark definitions are unchanged; the proxy stitching is research-only.

## Proxy Design

- `VOO` pre-inception history: proxy with `VTI`.
- `VXUS` pre-inception history: proxy with a fitted `VEA` + `VWO` blend.
- `VMBS` pre-inception history: proxy with `BND`.

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

- Signal agreement with shadow baseline: `80.7%`
- Mean aggregate OOS R^2: `-0.4599`
- Signal changes: `22`
- OUT / NEUTRAL / UNDER: `76.7%` / `18.0%` / `5.3%`

### ensemble_ridge_gbt_v20_best

- Signal agreement with shadow baseline: `78.0%`
- Mean aggregate OOS R^2: `-0.7188`
- Signal changes: `22`
- OUT / NEUTRAL / UNDER: `74.0%` / `20.7%` / `5.3%`

### live_production_ensemble_reduced

- Signal agreement with shadow baseline: `58.7%`
- Mean aggregate OOS R^2: `-1.4126`
- Signal changes: `23`
- OUT / NEUTRAL / UNDER: `56.0%` / `35.3%` / `8.7%`

