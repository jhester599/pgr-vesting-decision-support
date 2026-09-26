# v113 — constrained candidate selection

<!-- Registry entry: research/registry.yaml (id: v113). -->

**Question.** Constrained selection of promotable policy candidates.

**Date.** 2026-04-11 (first commit).

**Status.** `shadow`, feeds `src/models/classification_gate_overlay.py`.

**Notes.** The monthly run reads v113_constrained_candidate_selection_results.csv to pick the shadow gate overlay.

**Closeout / record.** [2026-04-11-v102-v117-post-review-enhancement-plan.md](../../../docs/superpowers/plans/2026-04-11-v102-v117-post-review-enhancement-plan.md)

## Run

From the repository root:

```bash
python research/studies/v113_constrained_candidate_selection/v113_constrained_candidate_selection.py
```

## Outputs

`outputs/` (2 files, committed): `v113_constrained_candidate_selection_results.csv`, `v113_constrained_candidate_selection_summary.md`

## Tests

- [`tests/research/test_research_v110_v117.py`](../../../tests/research/test_research_v110_v117.py)
