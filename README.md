# PGR Vesting Decision Support - v14 research + v13.1 production baseline

PGR Vesting Decision Support is a tax-aware decision-support system for unwinding
a concentrated Progressive Corporation (`PGR`) RSU position in a taxable
account. The repository combines scheduled data ingestion, feature engineering,
walk-forward modeling, tax-lot analysis, and monthly reporting so the user can
make a more disciplined decision at each vest.

## Current Status

Status as of 2026-04-04:

- `v7.x` is fully implemented in runtime code: the ablation work, tax scenario
  framework, EDGAR parser hardening, monthly report cleanup, and
  observation-per-feature / CPCV governance are all live.
- `v8.x` is fully implemented in production operations: the repo baseline was
  reconciled, the checked-in database was backfilled from the committed CSV,
  the monthly workflow and email were refreshed, and the user-facing decision
  output now uses recommendation modes rather than overconfident prediction
  language.
- `v9.x` is a completed research layer, not a promoted production change. It
  added research harnesses, benchmark-reduction studies, target experiments,
  policy evaluation, a weekly snapshot experiment, and a tuned Ridge
  classifier-sidecar candidate. The v9 conclusion was to avoid promoting a
  production model change yet.
- `v10.1` hardens the repo around that post-v9 state: it clarifies research vs.
  production boundaries, improves workflow safety and CI, introduces explicit
  schema migrations, adds run manifests, documents artifact policy, and makes
  the codebase easier to operate and extend safely.
- `v11.x` is an implemented research loop focused on accuracy and usefulness
  under a diversification-first objective. It adds a canonical `results/v11/`
  scoreboard, diversification-aware universe selection, reduced-universe
  bakeoffs, policy redesign, sidecar classifier review, and production-like
  dry-run recommendation memos. The current v11 conclusion is still
  `do not promote a live model change yet`.
- `v12.x` is a shadow-promotion study layered on top of v11. It compares the
  live production monthly stack against the simpler diversification-first
  baseline over a rolling 12-month review window, writes side-by-side dry-run
  memos under `results/v12/`, and tests whether the recommendation layer
  should be simplified before any new model stack is promoted.
- `v13.x` is the production-facing follow-through from that result. It keeps
  the live model stack intact, promotes the steadier diversification-first
  recommendation layer, and adds explicit existing-holdings guidance,
  diversification-first redeploy guidance, and a live-stack cross-check to the
  monthly report and email.
- `v14.x` is a narrow post-v13 prediction-layer study. It keeps the promoted
  v13.1 recommendation layer fixed, retests reduced benchmark universes, and
  compares the live 4-model stack against lean Ridge/GBT-centered replacement
  candidates. The current v14 conclusion is `continue shadowing, do not
  promote yet`, with `ensemble_ridge_gbt` as the leading v15 candidate.

## Production vs. Research

The repo now has explicit operating boundaries:

- Production:
  - `scripts/weekly_fetch.py`
  - `scripts/peer_fetch.py`
  - `scripts/edgar_8k_fetcher.py`
  - `scripts/monthly_decision.py`
  - scheduled GitHub workflows under `.github/workflows/`
  - committed database `data/pgr_financials.db`
  - monthly decision artifacts under `results/monthly_decisions/`
- Research / evaluation:
  - `src/research/`
  - v9 research scripts under `scripts/` such as
    `benchmark_suite.py`, `target_experiments.py`,
    `candidate_model_bakeoff.py`, and classifier experiments
  - research outputs under `results/v9/`
- Provisional:
  - v9 candidate production recommendations that have not yet been promoted
  - classifier-sidecar confidence work from v9
  - v11 diversification-first candidates and policy recommendations
  - v12 shadow-baseline recommendation-layer study
  - v14 reduced-universe replacement candidate shadowing
- Active recommendation-layer default:
  - v13.1 `shadow_promoted` recommendation-layer mode
- Active prediction-layer conclusion:
  - keep the live 4-model stack for now; carry `ensemble_ridge_gbt` into v15
    fixed-budget feature replacement work
- Historical planning and review artifacts:
  - `docs/history/claude-v7-plan.md`
  - `docs/plans/codex-v8-plan.md`
  - `docs/plans/codex-v9-plan.md`
  - older roadmap / research review files in `docs/`

See [POST_V9_BASELINE.md](docs/baselines/POST_V9_BASELINE.md) and
[docs/model-governance.md](docs/model-governance.md) for the authoritative
boundary and promotion policy.

## Quickstart

### 1. Install runtime dependencies

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

### 2. Optional developer tooling

```bash
pip install -r requirements-dev.txt -c constraints-dev.txt
```

### 3. Configure environment

Set the environment variables used by your workflow or local run:

- `AV_API_KEY`
- `FRED_API_KEY`
- `SMTP_SERVER`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `EMAIL_FROM`
- `EMAIL_TO`

### 4. Run the main production entrypoints

```bash
python scripts/weekly_fetch.py --dry-run --skip-fred
python scripts/peer_fetch.py --dry-run
python scripts/edgar_8k_fetcher.py --dry-run
python scripts/monthly_decision.py --as-of 2026-04-02 --dry-run --skip-fred
```

### 5. Run the local CI-equivalent checks

```bash
ruff check .
python -m pytest -q
```

## Repository Layout

- `config.py`: central configuration and model / workflow constants
- `src/database/`: DB schema, migrations, and DB helpers
- `src/ingestion/`: data-provider clients and fetch helpers
- `src/models/`: walk-forward training, calibration, CPCV, and model pipelines
- `src/portfolio/`: vesting recommendation and portfolio-allocation logic
- `src/reporting/`: report and email rendering helpers
- `src/research/`: v9 research harnesses and policy evaluation utilities
- `results/v11/`: diversification-first research outputs from the v11 loop
- `results/v12/`: shadow-promotion comparisons between the live stack and the
  simpler diversification-first baseline
- `results/v14/`: reduced-universe prediction-layer bakeoffs, minimal feature
  surgery, and shadow review outputs
- `scripts/`: runnable CLI entrypoints for production and research tasks
- `results/monthly_decisions/`: committed production monthly decision outputs
- `results/v9/`: committed research outputs from the v9 program
- `docs/`: current operator, architecture, workflow, governance, and historical
  documentation

## Docs Map

- Baseline / current state:
  - [POST_V9_BASELINE.md](docs/baselines/POST_V9_BASELINE.md)
  - [V10_1_RESULTS_SUMMARY.md](docs/results/V10_1_RESULTS_SUMMARY.md)
  - [V11_RESULTS_SUMMARY.md](docs/results/V11_RESULTS_SUMMARY.md)
  - [V11_CLOSEOUT_AND_V12_NEXT.md](docs/closeouts/V11_CLOSEOUT_AND_V12_NEXT.md)
  - [V12_RESULTS_SUMMARY.md](docs/results/V12_RESULTS_SUMMARY.md)
  - [V12_CLOSEOUT_AND_V13_NEXT.md](docs/closeouts/V12_CLOSEOUT_AND_V13_NEXT.md)
  - [V13_RESULTS_SUMMARY.md](docs/results/V13_RESULTS_SUMMARY.md)
  - [V14_RESULTS_SUMMARY.md](docs/results/V14_RESULTS_SUMMARY.md)
  - [V14_CLOSEOUT_AND_V15_NEXT.md](docs/closeouts/V14_CLOSEOUT_AND_V15_NEXT.md)
- Architecture and operations:
  - [docs/architecture.md](docs/architecture.md)
  - [docs/workflows.md](docs/workflows.md)
  - [docs/operations-runbook.md](docs/operations-runbook.md)
  - [docs/troubleshooting.md](docs/troubleshooting.md)
- Data, artifacts, and governance:
  - [docs/data-sources.md](docs/data-sources.md)
  - [docs/model-governance.md](docs/model-governance.md)
  - [docs/decision-output-guide.md](docs/decision-output-guide.md)
  - [docs/artifact-policy.md](docs/artifact-policy.md)
- Process:
  - [CONTRIBUTING.md](CONTRIBUTING.md)
  - [docs/changelog.md](docs/changelog.md)
- Historical context:
  - [ROADMAP.md](ROADMAP.md)
  - [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)
  - older reviews and plans in [docs/](docs/)

## Key Operational Outputs

- `results/monthly_decisions/<YYYY-MM>/recommendation.md`
- `results/monthly_decisions/<YYYY-MM>/diagnostic.md`
- `results/monthly_decisions/<YYYY-MM>/signals.csv`
- `results/monthly_decisions/<YYYY-MM>/run_manifest.json`
- `results/monthly_decisions/decision_log.md`

## Notes on v9

v9 did not promote a new production model stack. It established the research
baseline, showed that benchmark reduction and leaner feature sets help, and
concluded that the next production promotion test should focus on a reduced
benchmark universe and leaner Ridge/GBT-centered candidates rather than model
class expansion.

The v9 closeout is documented in
[V9_CLOSEOUT_AND_V91_NEXT.md](docs/closeouts/V9_CLOSEOUT_AND_V91_NEXT.md).

## Notes on v12

v12 did not promote a new production model stack either. It shadow-tested the
best v11 policy row, `baseline_historical_mean` with `neutral_band_3pct`,
against the live monthly stack over 12 recent monthly snapshots. The main
finding is that the live stack changed directional signals several times while
still landing on the same `DEFER-TO-TAX-DEFAULT` 50% sell action, whereas the
shadow baseline was steadier and easier to explain. That makes recommendation-
layer simplification the most plausible next promotion candidate.

## Notes on v13

v13 does not promote a new model stack either. Instead, it promotes the best
usefulness improvements from v11-v12 into the production-facing report and
email, and now also promotes the steadier recommendation layer:

- simpler-baseline cross-checks
- existing-holdings lot guidance
- diversification-first redeploy guidance
- simpler diversification-first recommendation layer as the active default

That makes the monthly output more useful immediately while still keeping the
live model stack unchanged.

## Notes on v14

v14 kept the promoted v13.1 recommendation layer fixed and retested the
underlying prediction layer on a reduced, diversification-aware benchmark
universe. The best replacement candidate was `ensemble_ridge_gbt`, which
improved on the reduced-universe live stack and stayed close to the
`historical_mean` baseline, but still did not earn immediate promotion. The
main v14 recommendation is to use fixed-budget feature replacement in v15
rather than reopening broader methodology expansion.
