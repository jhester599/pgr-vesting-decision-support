---
title: Documentation Restructure Design
date: 2026-04-10
status: approved
---

# Documentation Restructure Design

## Goal

Consolidate fragmented version history into a single source of truth (`CHANGELOG.md`),
create a lean forward-looking `ROADMAP.md`, archive stale/redundant docs that are not
the active operational baseline, and commit untracked housekeeping files as part of
the same PR.

## Problem Statement

Version history is currently spread across five locations:
- `ROADMAP.md` (744 lines, v2.7–v33, authoritative but named misleadingly)
- `docs/changelog.md` (v8–v10.1 only, redundant subset)
- `docs/plans/` (28 individual plan files — kept, not redundant)
- `docs/results/` (26 results summaries — kept, not redundant)
- `docs/closeouts/` (20 closeout files — kept, good structure)

Additional stale planning and AI-session artifacts under `docs/history/`,
`docs/baselines/`, and `docs/` root distract from the active operational docs.

## What Changes

### 1. `ROADMAP.md` → `CHANGELOG.md` (rename via `git mv`)

`ROADMAP.md` contains the complete, detailed per-version history (v2.7–v33).
Renaming it `CHANGELOG.md` aligns the filename with its actual content and
follows the GitHub convention (`CHANGELOG` = what happened, `ROADMAP` = what's next).

Content changes: add a one-line forward reference at the top:
> For active development direction, see [ROADMAP.md](ROADMAP.md).

All existing version entries, granular file-level detail, and test counts are
preserved exactly as written.

### 2. New `ROADMAP.md` (forward-looking only, ~50 lines)

Covers:
- **Current state**: v33 is the landed master baseline; v36 property-based
  tests are on the active branch (`codex/v36-property-testing`)
- **Active branch**: v36 Hypothesis property tests (Tier 5.5) — pending merge
- **Near-term backlog themes**: data expansion (EDGAR schema phase 2), model
  promotion readiness (12-month OOS validation window), operational hardening
- **Strategic backlog**: conformal interval dashboard, peer comparison expansion,
  BLP validation (target Q1 2027)
- Link to `CHANGELOG.md` for full version history

### 3. Files archived to `docs/archive/`

Moved (using `git mv`) from their current location:

| From | To |
|---|---|
| `docs/changelog.md` | `docs/archive/changelog.md` |
| `DEVELOPMENT_PLAN.md` | `docs/archive/DEVELOPMENT_PLAN.md` |
| `docs/history/SESSION_PROGRESS.md` | `docs/archive/SESSION_PROGRESS.md` |
| `docs/history/claude-v7-plan.md` | `docs/archive/claude-v7-plan.md` |
| `docs/claude-plan-v2.md` | `docs/archive/claude-plan-v2.md` |
| `docs/claude-plan-v3.md` | `docs/archive/claude-plan-v3.md` |
| `docs/claude-research-report-20260322.md` | `docs/archive/claude-research-report-20260322.md` |
| `docs/gemini-peer-review-20260322.md` | `docs/archive/gemini-peer-review-20260322.md` |
| `docs/gemini-peer-review-20260322v1.md` | `docs/archive/gemini-peer-review-20260322v1.md` |
| `docs/baselines/POST_V9_BASELINE.md` | `docs/archive/POST_V9_BASELINE.md` |

`docs/history/` and `docs/baselines/` become empty and are removed.

`docs/closeouts/` is **not** archived — it has good structure and is isolated.

### 4. `README.md` targeted updates

Two sections updated:

**"Current Baseline"** — replace the stale v30 description with the v33/v36
current state (matches what is on master + active branch).

**"Version History"** — replace the three-bullet list pointing to
`docs/changelog.md`, `ROADMAP.md`, and `docs/history/` with a single line:
> Full version history: [CHANGELOG.md](CHANGELOG.md)

### 5. Housekeeping for untracked files

| File | Action |
|---|---|
| `AGENTS.md` | Commit (Codex agent directives, mirrors `CLAUDE.md`) |
| `claude-v7-plan.md` (root) | Delete (stale planning artifact, not committed) |
| `.hypothesis/` | Add to `.gitignore` (test cache, not committed) |

## What Does Not Change

- `docs/plans/` — 28 version plan files, kept as-is
- `docs/results/` — 26 results summaries, kept as-is
- `docs/closeouts/` — 20 closeout files, kept as-is
- All operational docs: `docs/architecture.md`, `docs/operations-runbook.md`,
  `docs/workflows.md`, `docs/model-governance.md`, `docs/artifact-policy.md`,
  `docs/data-sources.md`, `docs/troubleshooting.md`, `docs/decision-output-guide.md`,
  `docs/PGR_EDGAR_CACHE_DATA_DICTIONARY.md`
- `CONTRIBUTING.md` — accurate and clean, no changes needed
- `CLAUDE.md` / `AGENTS.md` — project directives, kept

## No Test Changes Required

`grep` confirms no test files reference any of the files being renamed, moved,
or deleted. The v36 Hypothesis tests (`894f505`) are already committed on the
active branch and will merge with the docs work in the same PR.

## Resulting Active Documentation Map

After the restructure, the root-level and `docs/` surface area is:

```
CHANGELOG.md          ← full version history (v2.7–present)
ROADMAP.md            ← forward direction only (~50 lines)
README.md             ← project overview + doc pointers
CONTRIBUTING.md       ← contributor guide
AGENTS.md             ← Codex agent directives
CLAUDE.md             ← Claude agent directives

docs/
  architecture.md
  artifact-policy.md
  data-sources.md
  decision-output-guide.md
  model-governance.md
  operations-runbook.md
  PGR_EDGAR_CACHE_DATA_DICTIONARY.md
  troubleshooting.md
  workflows.md
  archive/              ← stale/historical, not actively reviewed
  closeouts/            ← per-version closeout notes (v9–v29)
  plans/                ← per-version plan files (v8–v33)
  results/              ← per-version results summaries (v9–v29)
```
