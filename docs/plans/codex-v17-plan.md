# codex-v17-plan

Created: 2026-04-04

## Goal

- Decide whether the modified Ridge+GBT pair should replace the current live stack as the cross-check path inside the promoted v13.1 recommendation layer.

## Paths Compared

- `shadow_baseline`
- `live_production`
- `candidate_v16`

## Gate

- Favor the candidate only if it agrees with the simpler baseline at least as well as the current live cross-check, while also carrying forward the v16 metric improvement over the reduced live stack.
