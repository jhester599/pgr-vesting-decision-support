# V142 Closeout And v143 Next

Created: 2026-04-16

## Completed Block

`v140` through `v142` are now complete on branch
`codex/v139-v152-autoresearch-followon`.

These runs stayed research-only and did not promote any live production config.

## Results Recorded

- `v140` shrinkage sweep:
  - tested `0.35`, `0.40`, `0.45`, `0.50`, `0.55`, `0.60`, `0.65`
  - all tested values returned the same pooled headline metrics on the current
    frame:
    `pooled_oos_r2=-0.1578`, `pooled_ic=0.1261`,
    `pooled_hit_rate=0.6906`
  - candidate remains `results/research/v140_shrinkage_candidate.txt = 0.50`
- `v141` fixed blend-weight sweep:
  - baseline `ridge_weight=0.50`:
    `pooled_oos_r2=-0.1634`, `pooled_ic=0.1250`,
    `pooled_hit_rate=0.6906`
  - best bounded candidate:
    `ridge_weight=0.60` ->
    `pooled_oos_r2=-0.1624`, `pooled_ic=0.1263`,
    `pooled_hit_rate=0.6935`
  - candidate updated in
    `results/research/v141_blend_weight_candidate.txt`
- `v142` EDGAR lag review:
  - incumbent `lag=2` remains the best balanced choice:
    `pooled_oos_r2=-0.1578`, `pooled_ic=0.1261`,
    `pooled_hit_rate=0.6906`
  - `lag=1` improved IC to `0.1492`, but lost enough OOS R^2 and hit rate that
    it is not the preferred research candidate

## Artifacts Produced

- `results/research/v140_shrinkage_autoresearch_log.jsonl`
- `results/research/v140_shrinkage_search_summary.md`
- `results/research/v141_blend_weight_autoresearch_log.jsonl`
- `results/research/v141_blend_weight_search_summary.md`
- `results/research/v142_edgar_lag_autoresearch_log.jsonl`
- `results/research/v142_edgar_lag_search_summary.md`

## Verification

- `python -m pytest tests/test_research_v140_shrinkage_eval.py tests/test_research_v141_blend_eval.py tests/test_research_v142_edgar_lag_eval.py -q --tb=short`

Known warning:

- `src/models/wfo_engine.py:294` still emits repeated
  `RuntimeWarning: All-NaN slice encountered` warnings during these harnesses,
  but the targeted tests and sweep commands complete successfully.

## v143 Recommendation

Run the next bounded regression block:

```bash
python -m pytest tests/test_research_v143_corr_prune_eval.py tests/test_research_v144_conformal_eval.py tests/test_research_v145_wfo_window_sweep.py -q --tb=short
python results/research/v143_corr_prune_eval.py --candidate-file results/research/v143_corr_prune_candidate.txt
python results/research/v144_conformal_eval.py --candidate-file results/research/v144_conformal_candidate.json
python results/research/v145_wfo_window_sweep.py --candidate-file results/research/v145_wfo_candidate.json
```

If the next session is interrupted mid-block, restart from this note, re-run the
targeted tests above, and only then continue into `v143`.
