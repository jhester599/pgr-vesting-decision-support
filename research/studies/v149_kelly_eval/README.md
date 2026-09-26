# v149 — kelly eval

<!-- Registry entry: research/registry.yaml (id: v149). -->

**Question.** Kelly fraction / cap replay proxy on the v138 BL posterior frame.

**Date.** 2026-04-16 (first commit).

**Status.** `shadow`, feeds `src/reporting/shadow_followon.py`.

**Notes.** Candidate Kelly fraction and cap feed the autoresearch_followon_v150 shadow lane.

**Closeout / record.** [V152_CLOSEOUT_AND_HANDOFF.md](../../../docs/closeouts/V152_CLOSEOUT_AND_HANDOFF.md)

## Run

From the repository root:

```bash
python research/studies/v149_kelly_eval/v149_kelly_eval.py
```

## Outputs

`outputs/` (3 files, committed): `v149_kelly_autoresearch_log.jsonl`, `v149_kelly_candidate.json`, `v149_kelly_search_summary.md`

## Tests

- [`tests/research/test_research_v149_kelly_eval.py`](../../../tests/research/test_research_v149_kelly_eval.py)
