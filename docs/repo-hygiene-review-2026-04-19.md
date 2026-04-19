# Repo Hygiene Review - 2026-04-19

## Scope

Reviewed repository documentation layout, operational docs, legacy research
artifacts, archive folders, and obvious local-orphan surfaces after v169.

## Changes Made

- Added `docs/README.md` as the documentation map.
- Labeled `docs/plans/` and `docs/results/` as legacy directories with README
  files rather than moving them and breaking older references.
- Updated active docs for v167-v169:
  - TA shadow ledger
  - monthly output verifier
  - calendar-aware PGR monthly EDGAR freshness
- Removed duplicate `.hypothesis/` entry from `.gitignore`.
- Promoted the previously untracked BL-01 and v159 plan drafts into tracked
  historical plan docs with status notes.

## Findings

### Documentation Layout

The current split is workable once labeled:

- Active operator docs live directly under `docs/`.
- Current-era plans live under `docs/superpowers/plans/`.
- Legacy v8-v34 plans live under `docs/plans/`.
- Legacy v9-v29 result summaries live under `docs/results/`.
- External reports and peer reviews live under `docs/archive/`.
- Old executable research scripts live under root `archive/`.

Moving `docs/plans/` or `docs/results/` now would create unnecessary churn,
because older closeouts and archived review notes still reference those paths.

### Stale Or Orphan Candidates

- `docs/plans/` and `docs/results/` are legacy, not active. Keep for now.
- Root `archive/` is separate from `docs/archive/` because it stores archived
  code/tests rather than documents. Keep for now, but do not add new docs there.
- `docs/gemini-prompts.txt` is a root docs file without an index entry. It is
  harmless but could move under `docs/archive/history/` in a future cleanup if
  it is no longer used.
- Absolute local paths remain inside some archived legacy docs. They are
  historical artifacts; fixing every old link would add churn without improving
  active operations.
- Several root `scripts/` entries are manual research/diagnostic tools rather
  than workflow entrypoints, for example `feature_ablation.py`,
  `feature_experiments.py`, `repurchase_timeseries_charts.py`, and
  `weekly_snapshot_experiments.py`. Keep them for now because tests and
  research docs still cover this style of harness, but consider a future
  `scripts/research/` split if the operator surface becomes noisy.
- `src/research/v11.py` through `src/research/v29.py` remain in place because
  older production/reporting code and tests still import selected utilities.
  Do not archive them until import checks prove they are unused.

## Recommended Next Cleanup

1. Add a lightweight markdown-link checker that skips `docs/archive/` by
   default and checks active docs only.
2. Consider moving `docs/gemini-prompts.txt` to
   `docs/archive/history/prompts/` if no current workflow references it.
3. Add a short `archive/README.md` at the root explaining the distinction
   between archived code and `docs/archive/` historical documents. Completed
   in v170.
4. Consider adding `scripts/README.md` or splitting manual research scripts
   under `scripts/research/` after a reference/import audit.
5. In a later larger pass, consider consolidating very old closeouts/results
   into an index, but avoid path-moving unless link checks are automated first.
