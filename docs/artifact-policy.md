# Artifact Policy

## Purpose

This repository intentionally commits some generated state. This document
explains which artifacts are production, which are research, and how to treat
them.

## Production Artifacts

- `data/pgr_financials.db`
- `results/monthly_decisions/`

These are committed because the operating model relies on them as durable state
and human-readable history.

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

- `results/monthly_decisions/classification_shadow_history.csv`

## Research Artifacts

- `results/research/`

These are committed as reproducible evidence from versioned research and
promotion studies. They are not consumed directly by production workflows
unless a later promotion explicitly wires them in.

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
  - `results/research/`
  - current plan and summary documents under `docs/superpowers/plans/`

## What Should Not Be Committed

Do not commit ad hoc local-only helpers or previews such as:

- temporary PR body files
- local email previews
- one-off local dashboard scratch output
- smoke-run scratch files not intended as permanent evidence
