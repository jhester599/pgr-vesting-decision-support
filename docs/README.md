# Documentation Map

This directory has three kinds of document: operator docs that describe the
live system, decision and review records that explain why it is the way it
is, and history. When adding a doc, put it in the smallest location that
matches its job.

## Active Operator Docs

Read these first when operating or changing the live system:

- [`architecture.md`](architecture.md) - production architecture and major module boundaries
- [`artifact-policy.md`](artifact-policy.md) - which generated artifacts are intentionally committed
- [`data-sources.md`](data-sources.md) - provider inventory and source-of-truth guidance
- [`decision-output-guide.md`](decision-output-guide.md) - how to interpret monthly decision artifacts
- [`model-governance.md`](model-governance.md) - live baseline, gates, health baseline, promotion rule, decision summary table
- [`operations-runbook.md`](operations-runbook.md) - local dry runs, recovery, and validation commands
- [`troubleshooting.md`](troubleshooting.md) - common failure modes and response steps
- [`workflows.md`](workflows.md) - GitHub Actions schedule and artifact expectations
- [`PGR_EDGAR_CACHE_DATA_DICTIONARY.md`](PGR_EDGAR_CACHE_DATA_DICTIONARY.md) - PGR monthly EDGAR CSV/DB fields
- [`data/fred_publication_lag_reference.md`](data/fred_publication_lag_reference.md) - FRED publication lags

Root-level docs are also active:

- `README.md` - project overview and quick start
- `ROADMAP.md` - current state and next direction
- `CHANGELOG.md` - version history
- `CONTRIBUTING.md` - local checks, test layout, CI jobs, research studies
- `AGENTS.md` - project directives for coding agents (`CLAUDE.md` points to it)

## Decisions, Reviews And History

- [`decisions/`](decisions/README.md) - one file per promotion decision
  (what became live behaviour, or was kept out of it, and why). The summary
  table is in `model-governance.md`. Add a file here whenever a promotion
  decision is made.
- [`reviews/`](reviews/) - repository audits and their follow-up reports
  ([`REPO_REVIEW_2026-09-25.md`](reviews/REPO_REVIEW_2026-09-25.md) and the
  `2026-09-25_step*` reports).
- [`research/`](research/) - the current research backlog and scoring rubric.
  The studies themselves are in `research/studies/` at the repository root
  (index: [`research/README.md`](../research/README.md), generated from
  `research/registry.yaml`); `research/legacy/` holds the v9-v28 result
  folders.
- [`history/`](history/README.md) - finished records behind one index:
  legacy v8-v34 plans (`history/plans/`), v37+ plans and specs
  (`history/superpowers/`), version closeouts (`history/closeouts/`), v9-v29
  result summaries (`history/results/`), early plans and external peer
  reviews (`history/archive/`), and the index of retired code
  (`history/retired-code/`).

## Where A New Document Goes

| Document | Location |
|---|---|
| How the live system works or is operated | an operator doc above (edit it; do not add a parallel one) |
| A promotion decision | `decisions/NNNN-slug.md` + a row in `model-governance.md` |
| An audit, or the report of a review step | `reviews/` |
| A research study's question, command and conclusion | `research/studies/<id>_<slug>/README.md` |
| A plan or closeout for a multi-version workstream | `history/superpowers/plans/` or `history/closeouts/` |
| What changed in a version | `CHANGELOG.md` |

## Archive Guidance

Do not delete historical docs just because they are old, and do not edit them
to match today's code. Link to them from the doc that supersedes them. Retired
code is not kept in the tree: delete it and list it in
`history/retired-code/README.md` with the commit that last had it.

`scripts/checks/check_doc_links.py` checks every relative link in the docs
above, including all of `history/` (CI runs it). Large generated artifacts
follow `artifact-policy.md`.
