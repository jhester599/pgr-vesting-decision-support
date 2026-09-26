# v139-v152 Autoresearch Follow-On Plan

## Summary

This plan continues the autonomous research program from the actual `v138`
GitHub `master` baseline rather than the older local `v75` snapshot that was
present before the fast-forward. The cycle remains research-only: build or run
bounded harnesses, update candidate files and handoff docs, but do not promote
new settings into live config without a later explicit promotion cycle.

## Current Progress

- `v139` complete: branch setup, archive, scaffolding, tests, and restart docs
- `v140` complete: bounded shrinkage sweep was flat; candidate remains `0.50`
- `v141` complete: bounded fixed blend-weight sweep updated the research-only
  candidate to `ridge_weight=0.60`
- `v142` complete: bounded EDGAR lag review retained incumbent `lag=2`
- `v143` complete: bounded correlation-pruning sweep updated the candidate to
  `rho=0.80`
- `v144` complete: bounded conformal replay tuning updated the candidate to
  `{"coverage": 0.75, "aci_gamma": 0.03}`
- `v145` complete: bounded WFO review kept the incumbent candidate after a
  tradeoff-heavy `(48, 6)` result
- `v146` complete: threshold follow-through kept the incumbent threshold pair
- `v147` complete: coverage-weighted aggregate proxy did not beat baseline
- `v148` complete: class-weight proxy kept the incumbent weight
- `v149` complete: Kelly replay-proxy sweep updated the candidate to
  `{"fraction": 0.50, "cap": 0.25}`
- `v150` complete: neutral-band replay review kept the incumbent `0.015`
  setting on stability grounds
- `v151` complete: the surviving winners now appear side-by-side in one
  reporting-only shadow lane named `autoresearch_followon_v150`
- `v152` complete: final synthesis, ranking, and handoff documentation are now
  recorded
- current handoff target: `docs/closeouts/V152_CLOSEOUT_AND_HANDOFF.md`

## Already Complete On Master

- `v129` benchmark feature-map evaluation
- `v131` threshold-wired Path B shadow integration
- `v133` ridge alpha re-baseline
- `v134` FRED lag sweep
- `v135` Path B temperature search
- `v136` backlog prioritization artifacts
- `v137` standalone GBT parameter sweep
- `v138` Black-Litterman replay proxy
- runtime optimization pass (`scripts/measure_test_time.sh`, `--fast`, slow markers)

## Remaining Open Targets

None inside this plan. The `v139-v152` cycle is complete.

Recommended next backlog after this handoff:

1. `BL-01` Black-Litterman tau/view tuning
2. `CLS-02` Firth logistic for short-history benchmarks
3. `FEAT-01` DTWEXBGS post-v128 feature search
4. `FEAT-02` WTI 3M momentum for DBC/VDE

## Implemented Scaffolding In This Branch

- Shared helper module:
  - `src/research/v139_utils.py`
- New regression-side harnesses:
  - `results/research/v140_shrinkage_eval.py`
  - `results/research/v141_blend_eval.py`
  - `results/research/v142_edgar_lag_eval.py`
  - `results/research/v143_corr_prune_eval.py`
  - `results/research/v144_conformal_eval.py`
  - `results/research/v145_wfo_window_sweep.py`
- New classifier / decision-layer follow-on proxies:
  - `results/research/v146_threshold_sweep.py`
  - `results/research/v147_coverage_weighted_aggregate.py`
  - `results/research/v148_class_weight_eval.py`
  - `results/research/v149_kelly_eval.py`
  - `results/research/v150_neutral_band_eval.py`
- Matching candidate files and pytest coverage for `v140` through `v150`

## Restart Procedure

1. Read the latest closeout note, currently
   `docs/closeouts/V150_CLOSEOUT_AND_V151_NEXT.md`.
2. Confirm current branch is `codex/v139-v152-autoresearch-followon`.
3. Run the targeted tests for the next intended version block before editing:
   - regression block:
     `python -m pytest tests/test_research_v140_shrinkage_eval.py tests/test_research_v141_blend_eval.py tests/test_research_v142_edgar_lag_eval.py tests/test_research_v143_corr_prune_eval.py tests/test_research_v144_conformal_eval.py tests/test_research_v145_wfo_window_sweep.py -q --tb=short`
   - classifier / decision block:
     `python -m pytest tests/test_research_v146_threshold_sweep.py tests/test_research_v147_coverage_weighted_aggregate.py tests/test_research_v148_class_weight_eval.py tests/test_research_v149_kelly_eval.py tests/test_research_v150_neutral_band_eval.py -q --tb=short`
4. Run the next harness with its candidate file, record the metric, and only
   then update the candidate or write a summary.
5. After each completed block, update:
   - `CHANGELOG.md`
   - `ROADMAP.md` or `docs/research/backlog.md`
   - the next closeout note

## Next Commands

Start the next autonomous cycle from the ranked backlog, not from this plan:

```bash
python -m pytest tests/test_shadow_followon.py tests/test_monthly_pipeline_e2e.py -q --tb=short
# then author the next plan from docs/research/backlog.md, starting with BL-01 unless priorities change
```
