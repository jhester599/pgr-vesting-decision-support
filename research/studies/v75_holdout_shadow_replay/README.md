# v75 — holdout shadow replay

<!-- Registry entry: research/registry.yaml (id: v75). -->

**Question.** Holdout-era monthly replay for the v74 quality-weighted shadow path.

**Date.** 2026-04-10 (first commit).

**Status.** `promoted`, feeds `config.model.CONSENSUS_WEIGHTING_MODE`.

**Notes.** Hold-out replay that supported the v76 promotion of the v72 quality-weighted consensus.

**Closeout / record.** [2026-04-10-v74-v78-quality-weighted-promotion.md](../../../docs/superpowers/plans/2026-04-10-v74-v78-quality-weighted-promotion.md)

## Run

From the repository root:

```bash
python research/studies/v75_holdout_shadow_replay/v75_holdout_shadow_replay.py
```

## Outputs

`outputs/` (3 files, committed): `v75_holdout_shadow_replay_decision.md`, `v75_holdout_shadow_replay_detail.csv`, `v75_holdout_shadow_replay_summary.csv`

## Tests

- [`tests/research/test_research_v75_shadow_replay.py`](../../../tests/research/test_research_v75_shadow_replay.py)
