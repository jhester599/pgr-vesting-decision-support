# v142 — edgar lag eval

<!-- Registry entry: research/registry.yaml (id: v142). -->

**Question.** EDGAR filing-lag evaluation on the current ensemble frame.

**Date.** 2026-04-16 (first commit).

**Status.** `retained`, feeds `config.features.EDGAR_FILING_LAG_MONTHS`.

**Notes.** Kept lag 2, chosen by in-sample R2 among lags 0-3 (review F23); step 6 placed EDGAR rows by filing date. Re-run pending (WP11).

**Closeout / record.** [V142_CLOSEOUT_AND_V143_NEXT.md](../../../docs/closeouts/V142_CLOSEOUT_AND_V143_NEXT.md)

## Run

From the repository root:

```bash
python research/studies/v142_edgar_lag_eval/v142_edgar_lag_eval.py
```

## Outputs

`outputs/` (3 files, committed): `v142_edgar_lag_autoresearch_log.jsonl`, `v142_edgar_lag_candidate.txt`, `v142_edgar_lag_search_summary.md`

## Tests

- [`tests/research/test_research_v142_edgar_lag_eval.py`](../../../tests/research/test_research_v142_edgar_lag_eval.py)
