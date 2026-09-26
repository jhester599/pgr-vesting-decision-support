# Documentation Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate version history into a single `CHANGELOG.md`, create a lean forward-looking `ROADMAP.md`, archive stale redundant docs, and commit housekeeping files — all in one PR on the existing `codex/v36-property-testing` branch.

**Architecture:** `ROADMAP.md` is renamed to `CHANGELOG.md` via `git mv` (preserving blame history). A new short `ROADMAP.md` is written from scratch. Ten stale files are moved to `docs/archive/` via `git mv`. `README.md` gets two targeted edits. Three untracked housekeeping items are resolved.

**Tech Stack:** Git, Markdown. No Python code changes.

**Spec:** `docs/superpowers/specs/2026-04-10-docs-restructure-design.md`

---

## File Map

| Action | Path |
|---|---|
| Rename (git mv) | `ROADMAP.md` → `CHANGELOG.md` |
| Create | `ROADMAP.md` (new, ~50 lines) |
| Modify | `CHANGELOG.md` (add one forward-reference line at top) |
| Modify | `README.md` (two sections updated) |
| Modify | `.gitignore` (add `.hypothesis/`) |
| Delete | `claude-v7-plan.md` (untracked, rm only) |
| git mv × 10 | see Task 3 |
| git add | `AGENTS.md` (untracked, commit as-is) |
| Remove (empty after mv) | `docs/history/`, `docs/baselines/` |

---

### Task 1: Housekeeping — .gitignore and untracked cleanup

**Files:**
- Modify: `.gitignore`
- Delete: `claude-v7-plan.md` (working-tree only, never tracked)

- [ ] **Step 1: Add `.hypothesis/` to `.gitignore`**

Open `.gitignore` and append after the existing `__pycache__/` block:

```
# Hypothesis property-based test cache
.hypothesis/
```

- [ ] **Step 2: Delete the stale root-level plan file**

```bash
rm claude-v7-plan.md
```

This file was never committed, so no `git` command needed.

- [ ] **Step 3: Commit `.gitignore` change**

```bash
git add .gitignore
git commit -m "chore: ignore hypothesis test cache directory"
```

Expected output: `1 file changed, 2 insertions(+)`

---

### Task 2: Commit `AGENTS.md`

**Files:**
- Add: `AGENTS.md` (Codex agent directives, mirrors `CLAUDE.md`)

- [ ] **Step 1: Verify content matches `CLAUDE.md`**

```bash
diff CLAUDE.md AGENTS.md
```

Expected: no meaningful differences (both contain the same project directives).
If there are differences, keep `CLAUDE.md` as authoritative and do not edit either file.

- [ ] **Step 2: Commit `AGENTS.md`**

```bash
git add AGENTS.md
git commit -m "chore: commit AGENTS.md for Codex agent directives"
```

Expected output: `1 file changed, N insertions(+)`

---

### Task 3: Archive redundant docs

**Files:**
- Create dir: `docs/archive/`
- git mv × 10 (see steps below)
- Remove empty dirs: `docs/history/`, `docs/baselines/`

- [ ] **Step 1: Create the archive directory**

```bash
mkdir docs/archive
```

- [ ] **Step 2: Move all ten files**

```bash
git mv docs/changelog.md docs/archive/changelog.md
git mv DEVELOPMENT_PLAN.md docs/archive/DEVELOPMENT_PLAN.md
git mv docs/history/SESSION_PROGRESS.md docs/archive/SESSION_PROGRESS.md
git mv docs/history/claude-v7-plan.md docs/archive/claude-v7-plan.md
git mv docs/claude-plan-v2.md docs/archive/claude-plan-v2.md
git mv docs/claude-plan-v3.md docs/archive/claude-plan-v3.md
git mv docs/claude-research-report-20260322.md docs/archive/claude-research-report-20260322.md
git mv docs/gemini-peer-review-20260322.md docs/archive/gemini-peer-review-20260322.md
git mv docs/gemini-peer-review-20260322v1.md docs/archive/gemini-peer-review-20260322v1.md
git mv docs/baselines/POST_V9_BASELINE.md docs/archive/POST_V9_BASELINE.md
```

- [ ] **Step 3: Remove now-empty directories**

```bash
rmdir docs/history
rmdir docs/baselines
```

- [ ] **Step 4: Verify archive contents**

```bash
ls docs/archive/
```

Expected: 10 files listed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: archive redundant and stale documentation files"
```

Expected output: `10 files changed` (renames counted as delete+add by git).

---

### Task 4: Rename `ROADMAP.md` → `CHANGELOG.md`

**Files:**
- Rename: `ROADMAP.md` → `CHANGELOG.md`

- [ ] **Step 1: Rename via git mv**

```bash
git mv ROADMAP.md CHANGELOG.md
```

- [ ] **Step 2: Add forward-reference line at top of `CHANGELOG.md`**

Open `CHANGELOG.md`. The current first line is:

```
# PGR Vesting Decision Support — Version Roadmap
```

Replace just the title and the preamble block (lines 1–13) with:

```markdown
# PGR Vesting Decision Support — Changelog

> For active development direction, see [ROADMAP.md](ROADMAP.md).

Day 1 = 2026-03-25 (initial price fetch). Day 2 = 2026-03-26 (dividend fetch +
afternoon bootstrap). Development starts Day 3.
```

Remove the stale "Current status as of 2026-04-02" paragraph (lines 6–13 of the
original) — that content will live in the new `ROADMAP.md` instead.

- [ ] **Step 3: Verify the rename and edit look correct**

```bash
head -10 CHANGELOG.md
```

Expected first lines:
```
# PGR Vesting Decision Support — Changelog

> For active development direction, see [ROADMAP.md](ROADMAP.md).

Day 1 = 2026-03-25 ...
```

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: rename ROADMAP.md to CHANGELOG.md (version history)"
```

---

### Task 5: Create new `ROADMAP.md`

**Files:**
- Create: `ROADMAP.md`

- [ ] **Step 1: Write `ROADMAP.md`**

Create `ROADMAP.md` at the repo root with this exact content:

```markdown
# PGR Vesting Decision Support — Roadmap

For full version history see [CHANGELOG.md](CHANGELOG.md).

## Current State

**Master baseline: v33** — config package modularization, expanded mypy CI
coverage, walk-forward diagnostics, conformal prediction monitoring, EDGAR
data expansion, and channel/valuation feature engineering are complete and
passing in CI.

**Active branch: `codex/v36-property-testing`** — Hypothesis property-based
tests (Tier 5.5 from the 2026-04-05 peer review) are added and pending merge.

## Near-Term Backlog

### Data Expansion (EDGAR Phase 2)
- Extend `pgr_edgar_monthly` schema with investment portfolio and capital
  allocation fields (`investment_book_yield`, `net_unrealized_gains_fixed`,
  `fixed_income_duration`, `fte_return_total_portfolio`)
- Extend HTML parser in `scripts/edgar_8k_fetcher.py` to capture all Phase 2
  fields from new live 8-K filings
- Historical backfill for expanded fields once schema is in place

### Model Promotion Readiness
- Accumulate 12 months of live OOS predictions (target: Q1 2027 validation)
- Formal BLP validation: Diebold-Mariano test vs. ensemble baseline
- Conditional coverage tests on conformal intervals

### Operational Hardening
- Monte Carlo tax scenario modeling (Tier 4.5 from 2026-04-05 peer review)
- Calibration diagnostic in monthly report (P2.7)
- Automated email delivery test coverage (P2.8)

## Strategic Backlog

| Item | Description |
|---|---|
| Conformal interval dashboard | Rolling empirical coverage plot in monthly diagnostic; alert if coverage < 70% |
| Peer comparison expansion | EDGAR 8-K fetchers for ALL, TRV, CB, HIG to add peer combined-ratio spread features |
| Property segment CAT tracking | Cross `npw_property` / `npe_property` with FRED property insurance CPI as a CAT-exposure signal |
| BLP formal OOS validation | Diebold-Mariano + conditional coverage tests once 12M of live predictions accumulate (Q1 2027) |

## Development Principles

- Never finalize a module without a passing pytest suite (CLAUDE.md mandate)
- No K-Fold cross-validation — `TimeSeriesSplit` with embargo + purge buffer only
- No `StandardScaler` across full dataset — scaler isolated within each WFO fold pipeline
- No `yfinance` — AV is the canonical price source; FMP/EDGAR for fundamentals
- Python 3.10+, strict PEP 8, full type hinting
- Approved libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `xgboost`,
  `requests`, `statsmodels`, `skfolio`, `PyPortfolioOpt`

## Monthly Decision Log

See [`results/monthly_decisions/decision_log.md`](results/monthly_decisions/decision_log.md)
for the persistent record of all automated monthly recommendations.
```

- [ ] **Step 2: Verify the file was created**

```bash
wc -l ROADMAP.md
```

Expected: ~60 lines.

- [ ] **Step 3: Commit**

```bash
git add ROADMAP.md
git commit -m "docs: add lean forward-looking ROADMAP.md"
```

---

### Task 6: Update `README.md`

**Files:**
- Modify: `README.md`

Two targeted edits only. Do not change any other section.

- [ ] **Step 1: Update the "Version History" section**

Locate this block in `README.md` (lines 71–78):

```markdown
## Version History

The repository's detailed version history is intentionally kept out of the
landing page.

- High-level completed changes: [docs/changelog.md](docs/changelog.md)
- Broader roadmap and historical version notes: [ROADMAP.md](ROADMAP.md)
- Peer reviews and session history: `docs/history/`
```

Replace with:

```markdown
## Version History

Full version history: [CHANGELOG.md](CHANGELOG.md)

Active development direction: [ROADMAP.md](ROADMAP.md)
```

- [ ] **Step 2: Update the "Current Baseline" section**

Locate this block in `README.md` (lines 82–97):

```markdown
## Current Baseline

The repo currently reflects a mature post-`v29` baseline with an active `v30`
enhancement sequence focused on operational hardening, observability, and docs
cleanup. Recent work has added:

- env-backed EDGAR identity configuration
- monthly data-freshness warnings
- shared HTTP retry/backoff helpers
- lightweight monthly pipeline integration coverage
- structured logging for the main production entry points
```

Replace with:

```markdown
## Current Baseline

The repo reflects a mature post-`v33` baseline. Recent work has added:

- config package modularization (`config/` replaces monolithic `config.py`)
- expanded mypy CI coverage to 11 modules
- walk-forward diagnostic tooling (VIF checks, feature importance stability)
- conformal prediction coverage monitoring and rolling drift detection
- EDGAR data expansion: segment-level NPW/PIF, valuation features, channel-mix signals
- vesting policy backtest and heuristic comparison in monthly reports
- structured Hypothesis property-based tests (v36, active branch)
```

- [ ] **Step 3: Verify the two edits and nothing else changed**

```bash
git diff README.md
```

Confirm only the two sections changed. No other lines modified.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README baseline and version history pointers"
```

---

### Task 7: Open PR

- [ ] **Step 1: Verify branch is clean**

```bash
git status
```

Expected: `nothing to commit, working tree clean`

- [ ] **Step 2: Push branch**

```bash
git push origin codex/v36-property-testing
```

- [ ] **Step 3: Open PR**

```bash
gh pr create \
  --base master \
  --title "docs: restructure documentation — CHANGELOG, lean ROADMAP, archive stale files" \
  --body "$(cat <<'EOF'
## Summary

- Renames `ROADMAP.md` → `CHANGELOG.md` (full v2.7–v33 version history, preserved via `git mv`)
- Creates lean forward-looking `ROADMAP.md` (~60 lines: current state, near-term backlog, strategic backlog)
- Archives 10 stale/redundant docs to `docs/archive/` (changelog, development plan, session history, AI review artifacts, v2/v3 plans)
- Updates `README.md`: current baseline (v29→v33) and version history pointer (3 bullets → 2 lines)
- Commits `AGENTS.md` (Codex agent directives, previously untracked)
- Adds `.hypothesis/` to `.gitignore`
- Includes v36 Hypothesis property-based tests (`894f505`) already on this branch

## What is NOT changed
- `docs/plans/` (28 version plan files)
- `docs/results/` (26 results summaries)
- `docs/closeouts/` (20 closeout files)
- All operational docs (`architecture.md`, `operations-runbook.md`, etc.)
- `CONTRIBUTING.md`

## Test plan
- [ ] `git log --follow CHANGELOG.md` shows full history from original `ROADMAP.md`
- [ ] `ROADMAP.md` renders cleanly and contains no version history detail
- [ ] All links in `README.md` resolve to existing files
- [ ] `docs/archive/` contains exactly 10 files
- [ ] `docs/history/` and `docs/baselines/` no longer exist

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Note the PR URL from the output**

---

## Self-Review Against Spec

| Spec requirement | Task |
|---|---|
| `git mv ROADMAP.md → CHANGELOG.md` | Task 4 |
| Add forward-reference line to CHANGELOG.md | Task 4, Step 2 |
| Create lean ROADMAP.md (~50 lines) | Task 5 |
| Archive 10 files to `docs/archive/` | Task 3 |
| Remove `docs/history/` and `docs/baselines/` | Task 3, Step 3 |
| Update README "Current Baseline" (v29→v33) | Task 6, Step 2 |
| Update README "Version History" to single pointer | Task 6, Step 1 |
| Commit `AGENTS.md` | Task 2 |
| Delete `claude-v7-plan.md` (untracked) | Task 1, Step 2 |
| Add `.hypothesis/` to `.gitignore` | Task 1, Step 1 |
| Open PR on `codex/v36-property-testing` | Task 7 |
| `docs/plans/`, `docs/results/`, `docs/closeouts/` untouched | — (no task modifies them) |
