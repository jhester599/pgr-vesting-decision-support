# V150 Closeout And v151 Next

Created: 2026-04-16

## Completed Block

`v146` through `v150` are now complete on branch
`codex/v139-v152-autoresearch-followon`.

These runs stayed research-only and did not promote any live production config.

## Results Recorded

- `v146` threshold follow-through:
  - tested `(0.10, 0.65)`, `(0.10, 0.70)`, `(0.15, 0.65)`, `(0.15, 0.70)`,
    `(0.20, 0.70)`, `(0.20, 0.75)`
  - incumbent candidate remains
    `results/research/v146_threshold_candidate.json = {"low": 0.15, "high": 0.70}`
- `v147` coverage-weighted aggregate proxy:
  - tested multipliers `0.50`, `0.75`, `1.00`, `1.25`, `1.50`, `2.00`
  - none improved `covered_ba` above the baseline `0.5000`
  - incumbent candidate remains
    `results/research/v147_path_b_multiplier_candidate.txt = 1.0`
- `v148` class-weight replay proxy:
  - tested `0.75`, `1.00`, `1.25`, `1.50`, `2.00`
  - incumbent candidate remains
    `results/research/v148_class_weight_candidate.txt = 1.0`
- `v149` Kelly replay proxy:
  - tested `(0.10, 0.10)`, `(0.15, 0.15)`, `(0.25, 0.20)`, `(0.35, 0.20)`,
    `(0.35, 0.25)`, `(0.50, 0.25)`
  - best bounded utility-score candidate:
    `{"fraction": 0.50, "cap": 0.25}` ->
    `utility_score=0.0021`, `coverage=0.4506`, `success_rate=0.7671`
  - candidate updated in `results/research/v149_kelly_candidate.json`
- `v150` neutral-band replay review:
  - re-ran after the `v149` Kelly update with bands
    `0.00`, `0.01`, `0.015`, `0.02`, `0.03`, `0.05`
  - utility remained flat at `0.0021` across the tested grid
  - wider bands increased selectivity and sometimes success rate, but did not
    produce a clearly better stable tradeoff
  - incumbent candidate remains
    `results/research/v150_neutral_band_candidate.txt = 0.015`

## Artifacts Produced

- `results/research/v146_threshold_autoresearch_log.jsonl`
- `results/research/v146_threshold_search_summary.md`
- `results/research/v147_aggregate_autoresearch_log.jsonl`
- `results/research/v147_aggregate_search_summary.md`
- `results/research/v148_class_weight_autoresearch_log.jsonl`
- `results/research/v148_class_weight_search_summary.md`
- `results/research/v149_kelly_autoresearch_log.jsonl`
- `results/research/v149_kelly_search_summary.md`
- `results/research/v150_neutral_band_autoresearch_log.jsonl`
- `results/research/v150_neutral_band_search_summary.md`

## Verification

- `python -m pytest tests/test_research_v146_threshold_sweep.py tests/test_research_v147_coverage_weighted_aggregate.py tests/test_research_v148_class_weight_eval.py tests/test_research_v149_kelly_eval.py tests/test_research_v150_neutral_band_eval.py -q --tb=short`

## v151 Recommendation

Start the final documentation/polish block next:

```bash
python -m pytest tests/test_research_v149_kelly_eval.py tests/test_research_v150_neutral_band_eval.py -q --tb=short
# then refresh v151 reporting/docs artifacts using the final v140-v150 candidate files
```

If the next session is interrupted mid-block, restart from this note, re-run the
targeted tests above, and then continue into `v151`.
