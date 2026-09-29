# v201 attempt 2 implementation plan

The owner authorized this continuation after merging PR 148. Its design is
the same six frozen candidates and campaign slots 1Ã¢â‚¬â€œ6, with no extra search.

Goal: execute v201 against the exact accepted v200 comparator after repairing
historical registry verification. Keep v200 code, DB and lock unchanged.

Architecture: a new installed snapshot verifier checks the original registry
at its full execution commit, enforces unchanged original entries, and checks
every other pin in the checkout. The runner reuses tested v200 causal adapters,
targets, folds, controls, grids and calibration. Attempt 1 is archived byte for
byte in outputs/attempt1; attempt 2 artifacts go in outputs/attempt2.

Tech stack: the exact v200 Python/dependency lock, pandas/numpy/sklearn.

- [x] Branch from merged master, preserve unrelated work, verify baseline pins.
- [x] Archive attempt 1 with its original manifest and provenance.
- [x] Test registry growth, tampering, duplicate IDs and wrong historical hash
  red before implementing research_lib/snapshot.py; then verify green.
- [x] Add an independent split/relative-price/future-perturbation fixture
  before implementing the missing relative-price adapter.
- [x] Complete run.py preparation/execution modes and synthetic orchestration
  fixtures; retain the identical six blocks and lag rules.
- [x] Commit catalog, grids, source/input hashes and development features before
  fitting; independently review the frozen procedure.
- [x] Verify exact support/folds/naive against v200, execute only development
  forecasts for all six candidates at h6 and secondary h12, with no retries
  or feature combinations after metrics.
- [x] Emit examples, availability/fold/access ledgers, uncertainty, campaign
  Holm results, closeout, complete provenance and hashes.
- [x] Run required full pytest and compare tracked DB SHA256 before/after;
  register the continuation and regenerate the research index.
- [x] Prepare the single follow-up research PR after final artifact verification.

No provider calls, fetchers, email, data writes or live promotion are authorized.
Failing preflight stops fitting and produces a blocked closeout. Quarantine
metrics remain sealed until v207. Any winner is only a possible v207 finalist.
