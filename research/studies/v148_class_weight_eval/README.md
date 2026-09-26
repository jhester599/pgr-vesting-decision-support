# v148 — class weight eval

<!-- Registry entry: research/registry.yaml (id: v148). -->

**Question.** Positive-class weight replay proxy on preserved Path B probabilities.

**Date.** 2026-04-16 (first commit).

**Status.** `closed`.

**Closeout / record.** [V152_CLOSEOUT_AND_HANDOFF.md](../../../docs/closeouts/V152_CLOSEOUT_AND_HANDOFF.md)

## Run

From the repository root:

```bash
python research/studies/v148_class_weight_eval/v148_class_weight_eval.py
```

## Outputs

`outputs/` (3 files, committed): `v148_class_weight_autoresearch_log.jsonl`, `v148_class_weight_candidate.txt`, `v148_class_weight_search_summary.md`

## Tests

- [`tests/research/test_research_v148_class_weight_eval.py`](../../../tests/research/test_research_v148_class_weight_eval.py)
