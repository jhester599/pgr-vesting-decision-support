# Artifact Policy

## Purpose

This repository intentionally commits some generated state. This document
explains which artifacts are production, which are research, and how to treat
them.

## Production Artifacts

Everything a scheduled workflow writes and commits, other than the DB, lives
under `artifacts/` (review 2026-09-25, section 5, phase 1; moved there in
v180 with `git mv`). The paths are constants in `config/paths.py`, and each
folder has a README naming the script and workflow that write it.

| Path | Was | Written by |
|---|---|---|
| `data/pgr_financials.db` | — | every DB-writing workflow |
| `artifacts/monthly_decisions/` | `results/monthly_decisions/` | `scripts/monthly_decision.py` |
| `artifacts/charts/pgr_*.png` | `results/research/pgr_*.png` | `scripts/repurchase_timeseries_charts.py`, `scripts/capital_return_charts.py` |
| `artifacts/ops/fetch_status.md` | `data/fetch_status.md` | `scripts/initial_fetch.py` |
| `artifacts/shadow_reviews/` | `results/v14/shadow_reviews/` | nothing today (v14 study record) |

These are committed because the operating model relies on them as durable state
and human-readable history. Workflow commit steps stage these exact paths
only; no workflow runs `git add results/`. Dry runs write to the gitignored
`results/dry_run/`, never to `artifacts/`.

Current monthly production artifacts include:

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
- `decision_log.md`

Shared longitudinal monitoring artifacts include:

- `artifacts/monthly_decisions/classification_shadow_history.csv`
- `artifacts/monthly_decisions/ta_shadow_variant_history.csv`

Monthly workflow postconditions are verified by:

- `scripts/verify_monthly_outputs.py`

That verifier checks required monthly files, data freshness, reporting-only TA
shadow variants, and matching TA ledger rows before the workflow commits
monthly artifacts.

## Research Artifacts

- `research/studies/<id>_<slug>/outputs/` (one folder per study; index in
  `research/README.md`, generated from `research/registry.yaml`)
- `research/legacy/` (the v9–v28 result folders, unchanged)

These are committed as reproducible evidence from versioned research and
promotion studies. They are not consumed directly by production workflows
unless a later promotion explicitly wires them in. No production or library
module imports Python code from `research/studies/` or `results/` (the last
one, `v46_classification.py`, moved to `src/research/binary_classification.py`
in v180; `results/` holds no code since v183). Three production modules still
read committed study outputs, from paths named in code:
`config.V128_BENCHMARK_FEATURE_MAP_PATH` (v128),
`classification_gate_overlay.DEFAULT_OVERLAY_RESULTS_PATH` (v113) and
`shadow_followon.FOLLOWON_CANDIDATE_PATHS` (v141–v150).
`tests/test_restructure_phase3.py` checks that those files exist.

Per-fold `*_detail.csv` files over 1 MB are not committed: study scripts
write them to `outputs/detail/`, which is gitignored, and the study README
says how to regenerate them. A test fails if one over 1 MB is committed
outside `research/legacy/`.

The monthly capital-return charts used to live here as
`results/research/pgr_*.png`; they are production outputs and now live in
`artifacts/charts/`. `monthly_decision.yml` regenerates and commits them by
name (`MONTHLY_CHARTS`), and the email embeds four of them.

## Provenance Rule

Major generated artifacts should be traceable to:

- run date
- git SHA
- schema version
- workflow or script name

Run manifests strengthen this provenance for production monthly artifacts.

`monthly_summary.json` is now the preferred machine-readable contract for top-
level decision surfaces. The email renderer, static dashboard snapshot, and
local dashboard should prefer structured values from that file before falling
back to markdown parsing.

## Source Of Truth

- Production source of truth for runtime behavior:
  - current code
  - production workflows
  - current docs
- Research source of truth for promotion evidence:
  - `research/studies/` (and `research/registry.yaml`)
  - current plan and summary documents under `docs/superpowers/plans/`

## What Should Not Be Committed

Do not commit ad hoc local-only helpers or previews such as:

- temporary PR body files
- local email previews
- one-off local dashboard scratch output
- smoke-run scratch files not intended as permanent evidence
