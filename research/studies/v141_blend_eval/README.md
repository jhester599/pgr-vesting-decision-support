# v141 — blend eval

<!-- Registry entry: research/registry.yaml (id: v141). -->

**Question.** Fixed Ridge-vs-GBT ensemble blend evaluation.

**Date.** 2026-04-16 (first commit).

**Status.** `shadow`, feeds `src/reporting/shadow_followon.py`.

**Notes.** Candidate ridge weight feeds the autoresearch_followon_v150 shadow lane.

**Closeout / record.** [V152_CLOSEOUT_AND_HANDOFF.md](../../../docs/closeouts/V152_CLOSEOUT_AND_HANDOFF.md)

## Run

From the repository root:

```bash
python research/studies/v141_blend_eval/v141_blend_eval.py
```

## Outputs

`outputs/` (3 files, committed): `v141_blend_weight_autoresearch_log.jsonl`, `v141_blend_weight_candidate.txt`, `v141_blend_weight_search_summary.md`

## Tests

- [`tests/research/test_research_v141_blend_eval.py`](../../../tests/research/test_research_v141_blend_eval.py)
