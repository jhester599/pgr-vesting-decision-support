# v146 — threshold sweep

<!-- Registry entry: research/registry.yaml (id: v146). -->

**Question.** Threshold sweep on top of the current v135 temperature baseline.

**Date.** 2026-04-16 (first commit).

**Status.** `closed`.

**Closeout / record.** [V152_CLOSEOUT_AND_HANDOFF.md](../../../docs/closeouts/V152_CLOSEOUT_AND_HANDOFF.md)

## Run

From the repository root:

```bash
python research/studies/v146_threshold_sweep/v146_threshold_sweep.py
```

## Outputs

`outputs/` (3 files, committed): `v146_threshold_autoresearch_log.jsonl`, `v146_threshold_candidate.json`, `v146_threshold_search_summary.md`

## Tests

- [`tests/research/test_research_v146_threshold_sweep.py`](../../../tests/research/test_research_v146_threshold_sweep.py)
