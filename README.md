# PGR Vesting Decision Support

PGR Vesting Decision Support is a tax-aware decision-support system for
managing a concentrated Progressive Corporation (`PGR`) RSU position in a
taxable account. The project combines scheduled market and fundamentals
ingestion, monthly feature engineering, walk-forward model evaluation,
tax-lot-aware recommendation logic, and user-facing decision reports.

## What The Repo Does

- refreshes prices, dividends, macro data, and EDGAR fundamentals on scheduled workflows
- builds monthly benchmark-relative features and targets with time-series-safe validation
- produces a monthly recommendation package with `recommendation.md`,
  `diagnostic.md`, `signals.csv`, and `run_manifest.json`
- separates production workflows from research cycles and archived promotion studies

## Architecture

```text
GitHub Actions / local scripts
        |
        v
Provider ingestion -> SQLite database -> Feature engineering -> WFO modeling
        |                                              |
        +-> run metadata / freshness checks            v
                                           tax + portfolio recommendation layer
                                                          |
                                                          v
                                           monthly markdown, CSV, plots, manifest
```

More detail: [docs/architecture.md](docs/architecture.md)

## Production Entry Points

- `scripts/weekly_fetch.py`
- `scripts/peer_fetch.py`
- `scripts/edgar_8k_fetcher.py`
- `scripts/monthly_decision.py`

These scripts are the operational surface area. Reusable logic belongs under
`src/`.

## Quick Start

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies from `requirements.txt` and any dev requirements you use locally.
3. Configure `.env` with the required API keys and EDGAR user-agent value.
4. Run the dry-run workflow checks below.

Recommended local smoke checks:

```bash
python scripts/weekly_fetch.py --dry-run --skip-fred
python scripts/peer_fetch.py --dry-run
python scripts/edgar_8k_fetcher.py --dry-run
python scripts/monthly_decision.py --as-of 2026-04-02 --dry-run --skip-fred
```

Operational runbook: [docs/operations-runbook.md](docs/operations-runbook.md)

## Documentation Map

- [docs/architecture.md](docs/architecture.md): system layout and production boundary
- [docs/operations-runbook.md](docs/operations-runbook.md): dry runs, validation, recovery
- [docs/workflows.md](docs/workflows.md): GitHub Actions behavior and expectations
- [docs/model-governance.md](docs/model-governance.md): promotion rules and evaluation discipline
- [docs/artifact-policy.md](docs/artifact-policy.md): research vs. production artifact handling
- [docs/data-sources.md](docs/data-sources.md): external provider inventory

## Version History

Full version history: [CHANGELOG.md](CHANGELOG.md)

Active development direction: [ROADMAP.md](ROADMAP.md)

## Current Baseline

The repo reflects a mature post-`v33` baseline. Recent work has added:

- config package modularization (`config/` replaces monolithic `config.py`)
- expanded mypy CI coverage to 11 modules
- walk-forward diagnostic tooling (VIF checks, feature importance stability)
- conformal prediction coverage monitoring and rolling drift detection
- EDGAR data expansion: segment-level NPW/PIF, valuation features, channel-mix signals
- vesting policy backtest and heuristic comparison in monthly reports
- structured Hypothesis property-based tests (v36, active branch)

## Project Principles

- Python 3.10+
- strict time-series validation only; no K-Fold cross-validation
- preference for simpler, regularized models under small-sample constraints
- test-first verification for production-facing changes
- explicit separation between research experiments and promoted behavior
