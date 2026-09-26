# v129 — feature map eval

<!-- Registry entry: research/registry.yaml (id: v129). -->

**Question.** Benchmark feature-map evaluation harness.

**Date.** 2026-04-12 (first commit).

**Status.** `shadow`, feeds `config.features.DUAL_TRACK_LEAN_BASELINE_OVERRIDES`.

**Notes.** Dual-track shadow integration; the VGT robustness audit put VGT back on the lean baseline.

## Run

From the repository root:

```bash
python research/studies/v129_feature_map_eval/v129_feature_map_eval.py
python research/studies/v129_feature_map_eval/v129_vgt_robustness_audit.py
```

## Outputs

`outputs/` (3 files, committed): `v129_candidate_map.csv`, `v129_vgt_robustness_audit_results.csv`, `v129_vgt_robustness_audit_summary.md`

## Tests

- [`tests/research/test_research_v129_feature_map_eval.py`](../../../tests/research/test_research_v129_feature_map_eval.py)
- [`tests/research/test_research_v129_vgt_robustness_audit.py`](../../../tests/research/test_research_v129_vgt_robustness_audit.py)
