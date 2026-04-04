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

## Research Artifacts

- `results/v9/`

These are committed as reproducible evidence from the v9 research program. They
are not consumed by production workflows.

## Provenance Rule

Major generated artifacts should be traceable to:

- run date
- git SHA
- schema version
- workflow or script name

v10.1 adds run manifests to strengthen this provenance for production monthly
artifacts.

## Source of Truth

- Production source of truth for runtime behavior:
  - current code
  - production workflows
  - current docs
- Research source of truth for promotion evidence:
  - `results/v9/`
  - v9 summary documents

## What Should Not Be Committed

Do not commit ad hoc local-only helpers or previews such as:

- temporary PR body files
- local email previews
- smoke-run scratch output not intended as permanent evidence
