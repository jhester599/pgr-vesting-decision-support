# Architecture

## System Overview

The repository is organized around a scheduled decision-support pipeline for a
concentrated `PGR` RSU position.

High-level flow:

1. Ingestion workflows update the committed SQLite database.
2. Processing code builds monthly features and benchmark-relative targets.
3. Modeling code runs walk-forward training and produces per-benchmark signals.
4. Portfolio and tax logic translate those signals into vest guidance.
5. Reporting code writes monthly artifacts and sends the decision email.
6. Research code evaluates alternative model/feature/benchmark designs without
   changing production behavior.

## Core Directories

- `src/database/`
  - schema initialization
  - migrations
  - DB health and metadata helpers
- `src/ingestion/`
  - provider clients
  - fetch scheduling
  - live EDGAR parsing
- `src/processing/`
  - monthly feature engineering
  - relative-return target construction
- `src/models/`
  - walk-forward optimization
  - CPCV
  - calibration
  - conformal intervals
- `src/portfolio/`
  - recommendation construction
  - tax-lot-aware portfolio logic
- `src/reporting/`
  - markdown report generation
  - email rendering
  - run manifest support
- `src/research/`
  - v9 evaluation harnesses
  - policy scoring
  - benchmark-set helpers

## Production Entry Points

- `scripts/weekly_fetch.py`
- `scripts/peer_fetch.py`
- `scripts/edgar_8k_fetcher.py`
- `scripts/monthly_decision.py`

These should remain thin orchestration entrypoints. Reusable business logic
belongs in `src/`.

## Data Stores

- SQLite database: `data/pgr_financials.db`
- Cached raw provider payloads: `data/raw/`
- Processed local inputs: `data/processed/`
- Production artifacts: `results/monthly_decisions/`
- Versioned research artifacts: `results/v9/` through `results/v29/`

## Research vs. Production Boundary

Production code:

- supports scheduled workflows
- writes user-facing monthly outputs
- must stay stable and testable

Research code:

- explores alternative benchmarks, targets, policies, and model subsets
- writes versioned outputs under `results/v9/` through `results/v29/`
- should not silently modify production recommendations

See [model-governance.md](model-governance.md) and
[artifact-policy.md](artifact-policy.md) for the promotion boundary.
