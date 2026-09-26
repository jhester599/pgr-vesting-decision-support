# v143 — corr prune eval

<!-- Registry entry: research/registry.yaml (id: v143). -->

**Question.** Correlation-pruned feature-set evaluation on the current frame.

**Date.** 2026-04-16 (first commit).

**Status.** `shadow`, feeds `src/reporting/shadow_followon.py`.

**Notes.** Candidate correlation-prune threshold feeds the autoresearch_followon_v150 shadow lane.

**Closeout / record.** [V152_CLOSEOUT_AND_HANDOFF.md](../../../docs/closeouts/V152_CLOSEOUT_AND_HANDOFF.md)

## Run

From the repository root:

```bash
python research/studies/v143_corr_prune_eval/v143_corr_prune_eval.py
```

## Outputs

`outputs/` (3 files, committed): `v143_corr_prune_autoresearch_log.jsonl`, `v143_corr_prune_candidate.txt`, `v143_corr_prune_search_summary.md`

## Tests

- [`tests/research/test_research_v143_corr_prune_eval.py`](../../../tests/research/test_research_v143_corr_prune_eval.py)
