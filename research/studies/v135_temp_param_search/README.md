# v135 — temp param search

<!-- Registry entry: research/registry.yaml (id: v135). -->

**Question.** Path B temperature-parameter search harness.

**Date.** 2026-04-14 (first commit).

**Status.** `closed`.

## Run

From the repository root:

```bash
python research/studies/v135_temp_param_search/v135_temp_param_search.py
```

## Outputs

`outputs/` (4 files, committed): `v135_temp_max_candidate.txt`, `v135_temp_param_autoresearch_log.jsonl`, `v135_temp_param_search_summary.md`, `v135_warmup_candidate.txt`

## Tests

- [`tests/research/test_research_v135_temp_param_search.py`](../../../tests/research/test_research_v135_temp_param_search.py)
