# v145 — wfo window sweep

<!-- Registry entry: research/registry.yaml (id: v145). -->

**Question.** WFO train/test window sweep on the current ensemble frame.

**Date.** 2026-04-16 (first commit).

**Status.** `retained`, feeds `config.model.WFO_TRAIN_WINDOW_MONTHS / WFO_TEST_WINDOW_MONTHS`.

**Notes.** Kept (60, 6).

**Closeout / record.** [V145_CLOSEOUT_AND_V146_NEXT.md](../../../docs/closeouts/V145_CLOSEOUT_AND_V146_NEXT.md)

## Run

From the repository root:

```bash
python research/studies/v145_wfo_window_sweep/v145_wfo_window_sweep.py
```

## Outputs

`outputs/` (3 files, committed): `v145_wfo_autoresearch_log.jsonl`, `v145_wfo_candidate.json`, `v145_wfo_search_summary.md`

## Tests

- [`tests/research/test_research_v145_wfo_window_sweep.py`](../../../tests/research/test_research_v145_wfo_window_sweep.py)
