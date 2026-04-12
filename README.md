# PGR Vesting Decision Support

PGR Vesting Decision Support is a tax-aware decision-support system for
managing a concentrated Progressive Corporation (`PGR`) RSU position in a
taxable account. The project combines scheduled data ingestion, monthly
feature engineering, strict walk-forward modeling, recommendation-layer
governance, and user-facing reporting.

## What The Repo Does

- refreshes prices, dividends, macro data, and EDGAR fundamentals on scheduled
  workflows
- engineers benchmark-relative monthly features with time-series-safe
  validation only
- produces a monthly recommendation package with:
  - `recommendation.md`
  - `diagnostic.md`
  - `signals.csv`
  - `benchmark_quality.csv`
  - `consensus_shadow.csv`
  - `classification_shadow.csv`
  - `decision_overlays.csv`
  - `dashboard.html`
  - `monthly_summary.json`
  - `run_manifest.json`
- keeps production behavior separate from research and promotion studies

## Current Production Baseline

The live monthly workflow currently uses:

- the `v11.1` lean 2-model prediction stack (`Ridge + GBT`, v18 feature sets)
- `v38` post-ensemble shrinkage as the prediction-layer calibration baseline
- the `v76` quality-weighted cross-benchmark consensus as the live
  recommendation path
- the equal-weight consensus retained in diagnostic artifacts only
- a shadow-only classifier interpretation layer and shadow gate overlay for
  confidence and future promotion monitoring

Recent production and research context is documented in:

- [2026-04-10-v66-v73-calibration-and-decision-layer.md](docs/superpowers/plans/2026-04-10-v66-v73-calibration-and-decision-layer.md)
- [2026-04-10-v74-v78-quality-weighted-promotion.md](docs/superpowers/plans/2026-04-10-v74-v78-quality-weighted-promotion.md)
- [2026-04-11-v79-v80-post-promotion-stabilization.md](docs/superpowers/plans/2026-04-11-v79-v80-post-promotion-stabilization.md)
- [2026-04-11-v81-v88-repo-review-and-adoption-plan.md](docs/superpowers/plans/2026-04-11-v81-v88-repo-review-and-adoption-plan.md)
- [2026-04-11-v102-v117-post-review-enhancement-plan.md](docs/superpowers/plans/2026-04-11-v102-v117-post-review-enhancement-plan.md)

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
  monthly markdown, CSV artifacts, static dashboard HTML, email, local dashboard
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
2. Install `requirements.txt`.
3. Configure `.env` with required API keys, SMTP settings, and EDGAR user-agent
   values.
4. Run the dry-run checks below.

Recommended local smoke checks:

```bash
python scripts/weekly_fetch.py --dry-run --skip-fred
python scripts/peer_fetch.py --dry-run
python scripts/edgar_8k_fetcher.py --dry-run
python scripts/monthly_decision.py --as-of 2026-04-11 --dry-run --skip-fred
```

Optional local dashboard:

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

Operational runbook: [docs/operations-runbook.md](docs/operations-runbook.md)

## Documentation Map

- [docs/architecture.md](docs/architecture.md): system layout and production boundary
- [docs/operations-runbook.md](docs/operations-runbook.md): dry runs, validation, recovery
- [docs/workflows.md](docs/workflows.md): GitHub Actions behavior and artifact expectations
- [docs/model-governance.md](docs/model-governance.md): current live baseline and promotion rules
- [docs/decision-output-guide.md](docs/decision-output-guide.md): how to read monthly outputs
- [docs/artifact-policy.md](docs/artifact-policy.md): production vs. research artifact handling
- [docs/data-sources.md](docs/data-sources.md): external provider inventory

## Version History

Full version history: [CHANGELOG.md](CHANGELOG.md)

Forward-looking backlog: [ROADMAP.md](ROADMAP.md)

## Project Principles

- Python 3.10+
- strict time-series validation only; no K-Fold cross-validation
- preference for simpler, regularized models under small-sample constraints
- test-first verification for production-facing changes
- explicit separation between research experiments and promoted behavior
