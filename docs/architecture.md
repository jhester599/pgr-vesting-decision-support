# Architecture

## System Overview

The repository is organized around a scheduled decision-support pipeline for a
concentrated `PGR` RSU position.

High-level flow:

1. Ingestion workflows update the committed SQLite database.
2. Processing code builds monthly features and benchmark-relative targets.
3. Modeling code runs walk-forward training and produces per-benchmark signals.
4. Consensus, policy, tax, and portfolio logic translate those signals into
   vest guidance.
5. Reporting code writes monthly artifacts, sends the decision email, and feeds
   the local dashboard surface.
6. Research code evaluates alternative calibration, consensus, and
   decision-layer designs without changing production behavior automatically.

## Core Directories

- `src/database/`
  - schema initialization
  - migrations
  - DB metadata helpers
- `src/ingestion/`
  - provider clients
  - fetch scheduling
  - live EDGAR parsing
- `src/processing/`
  - monthly feature engineering
  - relative-return target construction
- `src/models/`
  - walk-forward optimization
  - calibration
  - conformal intervals
  - forecast diagnostics
  - consensus helpers
- `src/portfolio/`
  - recommendation construction
  - tax-lot-aware portfolio logic
  - redeploy portfolio helpers
- `src/reporting/`
  - markdown report generation
  - email rendering
  - run manifest support
- `src/research/`
  - research helper modules and promotion-gate tooling
- `dashboard/`
  - local Streamlit dashboard for viewing current outputs

## Production Entry Points

- `scripts/weekly_fetch.py`
- `scripts/peer_fetch.py`
- `scripts/edgar_8k_fetcher.py`
- `scripts/monthly_decision.py`

These should remain thin orchestration entrypoints. Reusable business logic
belongs in `src/`.

## Current Production Output Surface

Each monthly run writes a folder under `results/monthly_decisions/<YYYY-MM>/`
containing:

- `recommendation.md`
- `diagnostic.md`
- `signals.csv`
- `benchmark_quality.csv`
- `consensus_shadow.csv`
- `dashboard.html`
- `monthly_summary.json`
- `run_manifest.json`

The workflow also updates:

- `results/monthly_decisions/decision_log.md`

The same monthly output set feeds:

- the email renderer in `src/reporting/email_sender.py`
- the local dashboard in `dashboard/app.py`

## Data Stores

- SQLite database: `data/pgr_financials.db`
- Cached raw provider payloads: `data/raw/`
- Processed local inputs: `data/processed/`
- Production artifacts: `results/monthly_decisions/`
- Versioned research artifacts: `results/research/`

## Research vs. Production Boundary

Production code:

- supports scheduled workflows
- writes user-facing monthly outputs
- must stay stable and testable

Research code:

- explores calibration, weighting, benchmark, and decision-layer alternatives
- writes versioned outputs under `results/research/`
- should not silently modify production recommendations

See [model-governance.md](model-governance.md) and
[artifact-policy.md](artifact-policy.md) for the promotion boundary.
