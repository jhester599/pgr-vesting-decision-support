# v144 — conformal eval

<!-- Registry entry: research/registry.yaml (id: v144). -->

**Question.** Conformal coverage backtest on the current pooled ensemble frame.

**Date.** 2026-04-16 (first commit).

**Status.** `shadow`, feeds `src/reporting/shadow_followon.py`.

**Notes.** Candidate conformal settings feed the autoresearch_followon_v150 shadow lane.

**Closeout / record.** [V152_CLOSEOUT_AND_HANDOFF.md](../../../docs/closeouts/V152_CLOSEOUT_AND_HANDOFF.md)

## Run

From the repository root:

```bash
python research/studies/v144_conformal_eval/v144_conformal_eval.py
```

## Outputs

`outputs/` (3 files, committed): `v144_conformal_autoresearch_log.jsonl`, `v144_conformal_candidate.json`, `v144_conformal_search_summary.md`

## Tests

- [`tests/research/test_research_v144_conformal_eval.py`](../../../tests/research/test_research_v144_conformal_eval.py)
