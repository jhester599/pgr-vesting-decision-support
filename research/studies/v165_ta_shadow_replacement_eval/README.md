# v165 — ta shadow replacement eval

<!-- Registry entry: research/registry.yaml (id: v165). -->

**Question.** TA classification replacement shadow evaluation.

**Date.** 2026-04-18 (first commit).

**Status.** `shadow`, feeds `src/reporting/classification_artifacts.py`.

**Notes.** Reporting-only TA replacement variants (ta_shadow_variant_history.csv).

**Closeout / record.** [V165_CLOSEOUT_AND_HANDOFF.md](../../../docs/closeouts/V165_CLOSEOUT_AND_HANDOFF.md)

## Run

From the repository root:

```bash
python research/studies/v165_ta_shadow_replacement_eval/v165_ta_shadow_replacement_eval.py
```

## Outputs

`outputs/` (7 files, committed): `v165_ta_shadow_candidate.json`, `v165_ta_shadow_current.csv`, `v165_ta_shadow_current_summary.csv`, `v165_ta_shadow_replacement_detail.csv`, `v165_ta_shadow_replacement_predictions.csv`, `v165_ta_shadow_replacement_regime_slices.csv`, `v165_ta_shadow_replacement_summary.csv`

## Tests

- [`tests/research/test_research_v165_ta_shadow_replacement.py`](../../../tests/research/test_research_v165_ta_shadow_replacement.py)
