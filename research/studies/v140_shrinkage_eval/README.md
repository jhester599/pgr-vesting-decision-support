# v140 — shrinkage eval

<!-- Registry entry: research/registry.yaml (id: v140). -->

**Question.** Standalone shrinkage evaluation on the current ensemble frame.

**Date.** 2026-04-16 (first commit).

**Status.** `retained`, feeds `config.model.ENSEMBLE_PREDICTION_SHRINKAGE_ALPHA`.

**Notes.** Flat across the bounded range; 0.50 kept.

**Closeout / record.** [V152_CLOSEOUT_AND_HANDOFF.md](../../../docs/closeouts/V152_CLOSEOUT_AND_HANDOFF.md)

## Run

From the repository root:

```bash
python research/studies/v140_shrinkage_eval/v140_shrinkage_eval.py
```

## Outputs

`outputs/` (3 files, committed): `v140_shrinkage_autoresearch_log.jsonl`, `v140_shrinkage_candidate.txt`, `v140_shrinkage_search_summary.md`

## Tests

- [`tests/research/test_research_v140_shrinkage_eval.py`](../../../tests/research/test_research_v140_shrinkage_eval.py)
