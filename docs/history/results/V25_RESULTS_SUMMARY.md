# V25 Results Summary

Created: 2026-04-05

## Scope

- `v25` executed the correctness-first cycle prompted by the external ChatGPT
  and Claude repo peer reviews.
- It focused on monthly index integrity, WFO / CPCV evaluation integrity, and
  silent-failure guards before rerunning the promotion-sensitive `v20-v24`
  studies.

## Code Fixes Landed

- Canonical monthly indexing now uses last-business-day month-end in the active
  production feature paths.
  - `src/processing/feature_engineering.py`
  - `src/ingestion/fred_loader.py`
  - `src/ingestion/pgr_monthly_loader.py`
- Inner CV for the regularized models now uses a gap-aware adaptive
  time-series splitter instead of an ungapped fixed `TimeSeriesSplit`.
  - `src/models/regularized_models.py`
- The WFO minimum-length guard now requires `TRAIN + GAP + TEST`, not only
  `TRAIN + TEST`.
  - `src/models/wfo_engine.py`
  - `src/research/evaluation.py`
- CPCV recombined paths now use split-specific stored predictions instead of
  rebuilding paths from a shared overwritten prediction buffer.
  - `src/models/wfo_engine.py`
- The EDGAR ROE compatibility path is now normalized at feature-matrix build
  time, and the feature builder warns when key EDGAR / FRED feature groups are
  structurally present but all-null.
  - `src/processing/feature_engineering.py`

## Validation

- Focused correctness tests:
  - `python -m pytest tests/test_wfo_engine.py tests/test_cpcv.py tests/test_feature_engineering.py tests/test_fred_loader.py tests/test_embargo_fix.py -q`
- Full suite:
  - `python -m pytest -q`
- Lint:
  - `python -m ruff check .`

## Rerun Outcomes

### v20

- Status: `continue_research_keep_current_cross_check`
- Interpretation: the narrow recent-window synthesis still does not clear a
  direct promotion gate on its own.
- Practical meaning: `v20` remains an intermediate diagnostic, not the final
  governance basis.

### v21

- Status: `promote_candidate_cross_check`
- Recommended path: `ensemble_ridge_gbt_v18`
- Full common evaluable window: `2016-10-31` to `2025-09-30`
- Common monthly dates: `108`
- Historical signal agreement with the promoted simpler baseline:
  - `ensemble_ridge_gbt_v18`: `84.3%`
  - current live reduced cross-check: `63.0%`

### v23

- Status: `extended_history_confirms_candidate`
- Recommended path: `ensemble_ridge_gbt_v18`
- Extended stitched-history window: `2013-04-30` to `2025-09-30`
- Common monthly dates: `150`
- Signal agreement with the promoted simpler baseline:
  - `ensemble_ridge_gbt_v18`: `80.7%`
  - live reduced cross-check: `58.7%`

### v24

- Status: `keep_voo`
- Interpretation: the corrected rerun still does not support replacing `VOO`
  with `VTI` just to gain more raw history.

## Main Conclusion

`v25` strengthens the post-`v21` promotion case rather than overturning it.

The most important result is:

- the correctness fixes did **not** invalidate the key historical promotion
  finding
- `ensemble_ridge_gbt_v18` remains the right visible cross-check candidate
- `VOO` remains the right benchmark definition among the tested alternatives

So the repo should treat `v25` as a confidence-building correction cycle, not a
reason to reopen another broad search.

## Output Artifacts

- `results/v20/*_20260405.csv`
- `results/v21/*_20260405.csv`
- `results/v23/*_20260405.csv`
- `results/v24/*_20260405.csv`
- `docs/results/V25_PEER_REVIEW_SYNTHESIS.md`
- `docs/plans/codex-v25-plan.md`
