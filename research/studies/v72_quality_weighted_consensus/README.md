# v72 — quality weighted consensus

<!-- Registry entry: research/registry.yaml (id: v72). -->

**Question.** Benchmark-quality-weighted consensus on top of the v38 baseline.

**Date.** 2026-04-10 (first commit).

**Status.** `promoted`, feeds `config.model.CONSENSUS_WEIGHTING_MODE`.

**Notes.** Quality-weighted consensus, promoted to the live path in v76 after the v74 shadow period and the v75 hold-out replay.

**Closeout / record.** [2026-04-10-v66-v73-calibration-and-decision-layer.md](../../../docs/superpowers/plans/2026-04-10-v66-v73-calibration-and-decision-layer.md)

## Run

From the repository root:

```bash
python research/studies/v72_quality_weighted_consensus/v72_quality_weighted_consensus.py
```

## Outputs

`outputs/` (1 files, committed): `v72_quality_weighted_consensus_results.csv`

## Tests

- [`tests/research/test_research_v72_quality_weighted_consensus.py`](../../../tests/research/test_research_v72_quality_weighted_consensus.py)
