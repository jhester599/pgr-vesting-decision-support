# v162 — ta broad screen

<!-- Registry entry: research/registry.yaml (id: v162). -->

**Question.** Broad technical-analysis screen helpers and harness.

**Date.** 2026-04-18 (first commit).

**Status.** `closed`.

**Closeout / record.** [V164_CLOSEOUT_AND_HANDOFF.md](../../../docs/closeouts/V164_CLOSEOUT_AND_HANDOFF.md)

## Run

From the repository root:

```bash
python research/studies/v162_ta_broad_screen/v162_ta_broad_screen.py
```

## Outputs

`outputs/` (2 files, committed): `v162_ta_broad_screen_inventory.json`, `v162_ta_broad_screen_summary.csv`

### Detail file (not committed)

`v162_ta_broad_screen_detail.csv` is over 1 MB, so it is written to `outputs/detail/`, which is
gitignored. To regenerate it, run:

```bash
python research/studies/v162_ta_broad_screen/v162_ta_broad_screen.py
```

The last committed copy is in history as `results/research/v162_ta_broad_screen_detail.csv`:

```bash
git show c0aa1d6:results/research/v162_ta_broad_screen_detail.csv > v162_ta_broad_screen_detail.csv
```

## Tests

- [`tests/research/test_research_v162_ta_screen.py`](../../../tests/research/test_research_v162_ta_screen.py)
