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
- post-ensemble shrinkage chosen prequentially with the `v38` rule (re-chosen
  every month from the realised OOS record; the fixed 0.50 is research-only)
- the `v76` quality-weighted cross-benchmark consensus as the live
  recommendation path, gated on look-ahead-free health metrics (OOS R² against
  the prevailing mean, equal-weight IC, Pesaran–Timmermann directional skill)
  and on two readiness gates: completed walk-forward validation for every
  required model and benchmark, and finite, fresh required inputs at the
  as-of date. Validation is walk-forward only; the CPCV diagnostic (a
  combinatorial K-fold) was retired in R3
- the equal-weight consensus retained in diagnostic artifacts only
- a shadow-only classifier interpretation layer and shadow gate overlay for
  confidence and future promotion monitoring

Why each part is live is recorded one decision per file in
[docs/decisions/](docs/decisions/README.md) (summary table in
[docs/model-governance.md](docs/model-governance.md#decision-record)); the
plans and closeouts behind them are in [docs/history/](docs/history/README.md).

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

Documentation map: [docs/README.md](docs/README.md)

## Production Entry Points

- `scripts/weekly_fetch.py`
- `scripts/peer_fetch.py`
- `cli/edgar_monthly_fetch.py` (logic in `src/pgr_vds/ingestion/edgar_monthly/`)
- `cli/monthly_decision.py` (logic in `src/pgr_vds/decision/`)

These scripts are the operational surface area. Reusable logic belongs under
`src/`; the two monthly jobs are thin command-line wrappers in `cli/` around
the installable `pgr_vds` package (`src/pgr_vds/`, review 2026-09-25 phase 5).

## Quick Start

1. Create and activate a Python 3.11+ virtual environment (pandas 3 needs
   3.11).
2. Install the package and its dependencies with `pip install -e .`, or
   `pip install -e ".[dev]"` to also get the pinned test and lint tools.
   `pyproject.toml` (package `pgr_vds`) is the only dependency list; the
   editable install lets `src`, `config` and `pgr_vds` import without
   `sys.path` edits.
3. Configure `.env` with required API keys, SMTP settings, and EDGAR user-agent
   values.
4. Run the dry-run checks below.

Recommended local smoke checks:

```bash
python scripts/weekly_fetch.py --dry-run --skip-fred
python scripts/peer_fetch.py --dry-run
python cli/edgar_monthly_fetch.py --dry-run
python cli/monthly_decision.py --as-of 2026-04-11 --dry-run --skip-fred
```

`weekly_fetch.py --dry-run` and `cli/monthly_decision.py --dry-run` are read-only:
they open the DB with `mode=ro`, skip migrations, API-log rows, split seeding,
the model-health snapshot, the retrain log, `decision_log.md` and the shadow
ledgers, and write monthly artifacts to the gitignored
`results/dry_run/monthly_decisions/YYYY-MM/` (manifest `dry_run: true`).
`cli/edgar_monthly_fetch.py --dry-run` also opens the DB read-only and skips
migrations, but it calls SEC EDGAR (set `EDGAR_USER_AGENT`; add
`--cache-dir data/raw/edgar_8k_cache` to reuse cached filings).
`peer_fetch.py --dry-run` is read-only too. CI runs all four dry runs through
`scripts/ci_offline_smoke.py`, which blocks the network. Every EDGAR call
needs `EDGAR_USER_AGENT` (a name and contact e-mail); without it the call fails.

Optional local dashboard:

```bash
pip install -e ".[dashboard]"
streamlit run dashboard/app.py
```

Operational runbook: [docs/operations-runbook.md](docs/operations-runbook.md)

## Documentation Map

- [docs/architecture.md](docs/architecture.md): system layout and production boundary
- [docs/operations-runbook.md](docs/operations-runbook.md): dry runs, validation, recovery
- [docs/workflows.md](docs/workflows.md): GitHub Actions behavior and artifact expectations
- [docs/model-governance.md](docs/model-governance.md): current live baseline and promotion rules
- [docs/decisions/](docs/decisions/README.md): one record per promotion decision
- [docs/decision-output-guide.md](docs/decision-output-guide.md): how to read monthly outputs
- [docs/artifact-policy.md](docs/artifact-policy.md): production vs. research artifact handling
- [docs/data-sources.md](docs/data-sources.md): external provider inventory
- [docs/README.md](docs/README.md): active, historical, and legacy documentation map
- [docs/history/](docs/history/README.md): plans, closeouts, result summaries, peer reviews and retired code
- [CONTRIBUTING.md](CONTRIBUTING.md): local checks, test layout (`tests/unit`, `tests/integration`, `tests/research`) and CI jobs

## Version History

Full version history: [CHANGELOG.md](CHANGELOG.md)

Forward-looking backlog: [ROADMAP.md](ROADMAP.md)

## Project Principles

- Python 3.11+
- strict time-series validation only; no K-Fold cross-validation
- preference for simpler, regularized models under small-sample constraints
- test-first verification for production-facing changes
- explicit separation between research experiments and promoted behavior
