# V19 Results Summary

Created: 2026-04-04

## Scope

- Backfilled the remaining public macro / valuation series needed to finish the v15 inventory.
- Relaxed the research-only EDGAR breadth gate to 24 non-null rows so newer live-parser fields could be evaluated.
- Re-ran the fixed-budget feature replacement cycle on the full now-available inventory.
- Produced a final tested/blocked traceability matrix for all original 46 features.

## Coverage

- available in feature matrix: `44`
- tested through v19 swap phase: `44`
- blocked after source audit: `2`

## Blocked Features

- `pgr_cr_vs_peer_cr`: Requires point-in-time peer combined-ratio history for ALL/TRV/CB/HIG, which the repo does not currently ingest.
- `pgr_fcf_yield`: Requires quarterly operating-cash-flow and capex ingestion from EDGAR, which is not present in the current fundamentals schema.

## Current Leader

- candidate: `ridge_lean_v1__v15_best`
- model type: `ridge`
- mean IC: `0.2381`
- mean hit rate: `0.6655`
- mean OOS R^2: `-0.6364`
- mean sign-policy return: `0.0756`

## Artifacts

- `results\v19/v19_public_macro_backfill_20260404.csv`
- `results\v19/v19_0_core_summary_20260404.csv`
- `results\v19/v19_1_confirmation_summary_20260404.csv`
- `results\v19/v19_2_bakeoff_summary_20260404.csv`
- `results\v19/v19_feature_traceability_20260404.csv`
