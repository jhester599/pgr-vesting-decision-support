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

- `src/pgr_vds/` (installed as the top-level package `pgr_vds`; it does not
  import as `src.pgr_vds`)
  - `decision/`: the monthly decision, one module per step: `schedule`,
    `refresh` (FRED), `signal_generation`, `health` (OOS health, gates,
    drift, policy backtest, manifest warnings), `tax_lots`, `portfolio`,
    `rendering` / `recommendation_report` / `diagnostic_report`, `artifacts`
    (CSVs, decision log, shadow ledgers, run manifest) and `pipeline` (`main`)
  - `ingestion/edgar_monthly/`: PGR's monthly 8-K releases: `fetch` (SEC
    EDGAR HTTP), `parse` (Exhibit 99 parser), `derive` (derived fields),
    `load` (fetch-and-upsert, the CSV loader and export)
- `cli/`
  - `monthly_decision.py`, `edgar_monthly_fetch.py`: command-line wrappers
- `src/database/`
  - schema initialization
  - migrations
  - DB metadata helpers
- `src/ingestion/`
  - provider clients
  - fetch scheduling
  - EDGAR XBRL and 10-Q clients (the monthly 8-K parser is in
    `src/pgr_vds/ingestion/edgar_monthly/`)
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
- `tests/`
  - `unit/<package>/` mirrors `src/` and `src/pgr_vds/` (`decision/`;
    `ingestion/` holds the `edgar_monthly` tests), plus `config`, `dashboard`,
    `scripts`
  - `integration/` (`pipeline/`, `data/`, `repo/`) for cross-layer,
    committed-DB and repository-contract tests
  - `research/` for study tests; CI runs them in a separate job
- `docs/`
  - operator docs at the top level, `decisions/` (one file per promotion
    decision), `reviews/` (audits) and `history/` (finished plans,
    closeouts, result summaries, peer reviews, retired-code index)

## Production Entry Points

- `scripts/weekly_fetch.py`
- `scripts/peer_fetch.py`
- `cli/edgar_monthly_fetch.py` (logic in `src/pgr_vds/ingestion/edgar_monthly/`)
- `cli/monthly_decision.py` (logic in `src/pgr_vds/decision/`)

These should remain thin orchestration entrypoints. Reusable business logic
belongs in `src/`. The files in `cli/` only parse the command line; the
monthly decision is `pgr_vds.decision.pipeline.main` and the EDGAR job is
`pgr_vds.ingestion.edgar_monthly` (review 2026-09-25, section 5, phase 5).

## Current Production Output Surface

Each monthly run writes a folder under `artifacts/monthly_decisions/<YYYY-MM>/`
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

- `artifacts/monthly_decisions/decision_log.md`

The same monthly output set feeds:

- the email renderer in `src/reporting/email_sender.py`
- the local dashboard in `dashboard/app.py`

## Data Stores

- SQLite database: `data/pgr_financials.db`
- Cached raw provider payloads: `data/raw/`
- Processed local inputs: `data/processed/`
- Production artifacts: `artifacts/` (`monthly_decisions/`, `charts/`,
  `ops/`, `shadow_reviews/`); paths in `config/paths.py`
- Research studies: `research/studies/<id>_<slug>/` (script, README,
  `outputs/`), indexed by `research/registry.yaml` → `research/README.md`;
  v9–v28 result folders in `research/legacy/`

## Research vs. Production Boundary

Production code:

- supports scheduled workflows
- writes user-facing monthly outputs
- must stay stable and testable

Research code:

- explores calibration, weighting, benchmark, and decision-layer alternatives
- lives in `research/studies/<id>_<slug>/` and writes to that folder's
  `outputs/` (large detail files to the gitignored `outputs/detail/`)
- should not silently modify production recommendations

See [model-governance.md](model-governance.md) and
[artifact-policy.md](artifact-policy.md) for the promotion boundary.
