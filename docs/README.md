# Documentation Map

This directory separates active operator documentation from historical research
and implementation records. When adding new docs, prefer the smallest location
that matches the document's job.

## Active Operator Docs

These are the docs to read first when operating or changing the live system:

- `architecture.md` - production architecture and major module boundaries
- `artifact-policy.md` - which generated artifacts are intentionally committed
- `data-sources.md` - provider inventory and source-of-truth guidance
- `decision-output-guide.md` - how to interpret monthly decision artifacts
- `model-governance.md` - production/research boundary and promotion rules
- `operations-runbook.md` - local dry runs, recovery, and validation commands
- `troubleshooting.md` - common failure modes and response steps
- `workflows.md` - GitHub Actions schedule and artifact expectations
- `PGR_EDGAR_CACHE_DATA_DICTIONARY.md` - PGR monthly EDGAR CSV/DB fields

Root-level docs are also active:

- `README.md` - project overview
- `ROADMAP.md` - current state and next direction
- `CHANGELOG.md` - version history

## Historical And Research Docs

- `docs/superpowers/plans/` - active-era implementation plans and execution
  logs from v37 onward. New versioned plans should go here.
- `docs/superpowers/specs/` - design specs that fed implementation plans.
- `docs/closeouts/` - per-version closeout and handoff notes.
- `docs/research/` - current research backlog and scoring rubric.
- `docs/archive/` - external reports, peer reviews, old session docs, and
  other historical source material.
- `repo-hygiene-review-2026-04-19.md` - latest documentation and archive audit.

## Legacy Version Docs

- `docs/plans/` - legacy v8-v34 execution plans. These are kept because older
  closeouts and archived docs reference them, but they are not the active plan
  location.
- `docs/results/` - legacy v9-v29 result summaries. Current research artifacts
  live under `results/research/`; this folder is retained for historical
  continuity.

## Archive Guidance

Do not delete historical docs just because they are old. Prefer one of these:

- add a short README that labels a directory as legacy
- move clearly stale one-off material under `docs/archive/history/`
- keep active operational docs short and point to closeouts or research
  artifacts for detail

Large generated artifacts should follow `artifact-policy.md`.

Root `archive/` is separate from `docs/archive/`: it stores retired code/tests,
not documentation.
