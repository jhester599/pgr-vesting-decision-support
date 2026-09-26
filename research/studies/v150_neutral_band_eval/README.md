# v150 — neutral band eval

<!-- Registry entry: research/registry.yaml (id: v150). -->

**Question.** Neutral-band replay proxy on top of the v149 Kelly baseline.

**Date.** 2026-04-16 (first commit).

**Status.** `shadow`, feeds `src/reporting/shadow_followon.py`.

**Notes.** Candidate neutral band feeds the autoresearch_followon_v150 shadow lane.

**Closeout / record.** [V150_CLOSEOUT_AND_V151_NEXT.md](../../../docs/closeouts/V150_CLOSEOUT_AND_V151_NEXT.md)

## Run

From the repository root:

```bash
python research/studies/v150_neutral_band_eval/v150_neutral_band_eval.py
```

## Outputs

`outputs/` (3 files, committed): `v150_neutral_band_autoresearch_log.jsonl`, `v150_neutral_band_candidate.txt`, `v150_neutral_band_search_summary.md`

## Tests

- [`tests/research/test_research_v150_neutral_band_eval.py`](../../../tests/research/test_research_v150_neutral_band_eval.py)
