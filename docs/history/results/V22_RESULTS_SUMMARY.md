# V22 Results Summary

Created: 2026-04-04

## Scope

- v22 is a narrow implementation step: keep the v13.1 simpler diversification-first recommendation layer active, but replace the visible production cross-check with the v21 historical winner.

## Implemented Path

- Promoted cross-check candidate: `ensemble_ridge_gbt_v18`
- Members: `ridge_lean_v1__v18, gbt_lean_plus_two__v18`
- As-of validation date: `2026-04-04`

## Current Snapshot

- Visible cross-check signal: `UNDERPERFORM`
- Visible cross-check recommendation mode: `DEFER-TO-TAX-DEFAULT`
- Visible cross-check sell %: `50%`
- Visible cross-check predicted 6M relative return: `-16.25%`
- Signal agrees with simpler baseline: `False`
- Recommendation mode agrees with simpler baseline: `True`
- Sell % agrees with simpler baseline: `True`

## Decision

- Status: `implemented_visible_cross_check_promotion`
- The active recommendation layer is unchanged; only the displayed cross-check candidate changed.
