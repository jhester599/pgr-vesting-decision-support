# V170 Closeout And Handoff

Created: 2026-04-19

## Completed Block

`v170` completes a repository documentation and hygiene pass after the TA and
monthly-verifier work.

## What Changed

- Added `docs/README.md` as the active documentation map.
- Added README labels for legacy directories:
  - `docs/plans/`
  - `docs/results/`
- Added `archive/README.md` to distinguish root archived code/tests from
  `docs/archive/` historical documents.
- Updated operator-facing docs for:
  - monthly output verifier
  - TA shadow variant ledger
  - calendar-aware PGR EDGAR freshness
  - legacy/current documentation locations
- Added `docs/repo-hygiene-review-2026-04-19.md` with findings and future
  cleanup candidates.
- Removed the duplicate `.hypothesis/` entry from `.gitignore`.
- Added the previously local BL-01 and v159 plan drafts to
  `docs/superpowers/plans/` with historical status notes.

## What Stayed Put

- `docs/plans/` and `docs/results/` were not moved. Older closeouts and
  archived review notes still reference those paths.
- Root `archive/` was not merged into `docs/archive/` because it stores retired
  executable code/tests rather than documents.
- Absolute paths inside archived legacy docs were not rewritten; those files
  are historical records, not active operator docs.

## Verification

Validated with:

```bash
python -m pytest tests/test_docs_hygiene.py -q --tb=short
python -m pytest tests/test_docs_hygiene.py tests/test_workflow_contracts.py tests/test_verify_monthly_outputs.py -q --tb=short
python -m ruff check tests/test_docs_hygiene.py
```

## Next Direction

The next hygiene step should be a lightweight active-doc markdown link checker
that skips `docs/archive/` by default. That would make any future path moves
safer.
