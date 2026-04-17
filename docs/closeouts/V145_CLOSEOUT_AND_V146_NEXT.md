# V145 Closeout And v146 Next

Created: 2026-04-16

## Completed Block

`v143` through `v145` are now complete on branch
`codex/v139-v152-autoresearch-followon`.

These runs stayed research-only and did not promote any live production config.

## Results Recorded

- `v143` correlation pruning:
  - tested `rho=0.80`, `0.85`, `0.90`, `0.95`, `0.99`
  - best bounded candidate:
    `rho=0.80` ->
    `pooled_oos_r2=-0.1569`, `pooled_ic=0.1411`,
    `pooled_hit_rate=0.6944`
  - candidate updated in `results/research/v143_corr_prune_candidate.txt`
- `v144` conformal coverage replay:
  - tested bounded pairs:
    `(0.75, 0.03)`, `(0.75, 0.05)`, `(0.80, 0.05)`,
    `(0.85, 0.05)`, `(0.85, 0.10)`
  - best bounded candidate by coverage-gap closeness:
    `coverage=0.75`, `aci_gamma=0.03` ->
    `coverage=0.7490`, `target_coverage=0.7500`,
    `coverage_gap=-0.0010`
  - candidate updated in `results/research/v144_conformal_candidate.json`
- `v145` WFO train/test windows:
  - tested `(48, 6)`, `(54, 6)`, `(60, 6)`, `(72, 6)`, `(60, 3)`, `(60, 9)`
  - `(48, 6)` improved pooled OOS R^2 and IC, but reduced hit rate to `0.6535`
  - incumbent candidate remains
    `results/research/v145_wfo_candidate.json = {"train": 60, "test": 6}`

## Artifacts Produced

- `results/research/v143_corr_prune_autoresearch_log.jsonl`
- `results/research/v143_corr_prune_search_summary.md`
- `results/research/v144_conformal_autoresearch_log.jsonl`
- `results/research/v144_conformal_search_summary.md`
- `results/research/v145_wfo_autoresearch_log.jsonl`
- `results/research/v145_wfo_search_summary.md`

## Verification

- `python -m pytest tests/test_research_v143_corr_prune_eval.py tests/test_research_v144_conformal_eval.py tests/test_research_v145_wfo_window_sweep.py -q --tb=short`

Known warning:

- `src/models/wfo_engine.py:294` still emits repeated
  `RuntimeWarning: All-NaN slice encountered` warnings during the regression
  harnesses, but the targeted tests and sweep commands complete successfully.

## v146 Recommendation

Run the classifier-side follow-on block next:

```bash
python -m pytest tests/test_research_v146_threshold_sweep.py tests/test_research_v147_coverage_weighted_aggregate.py tests/test_research_v148_class_weight_eval.py -q --tb=short
python results/research/v146_threshold_sweep.py --candidate-file results/research/v146_threshold_candidate.json
python results/research/v147_coverage_weighted_aggregate.py --candidate-file results/research/v147_path_b_multiplier_candidate.txt
python results/research/v148_class_weight_eval.py --candidate-file results/research/v148_class_weight_candidate.txt
```

If the next session is interrupted mid-block, restart from this note, re-run the
targeted tests above, and only then continue into `v146`.
