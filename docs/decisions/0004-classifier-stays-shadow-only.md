# 0004 — Classifier stays shadow-only (v102–v117)

| | |
|---|---|
| **Status** | Accepted. The classifier informs no live decision |
| **Date** | 2026-04-11 |
| **Where it lives** | `src/models/classification_shadow.py`, `src/models/classification_gate_overlay.py`; artifacts `classification_shadow.csv`, `decision_overlays.csv`, `classification_shadow_history.csv` |
| **Studies** | `research/studies/v110_gemini_veto_gate/` … `v117_primary_mode_selector_evaluation/` (all `closed`) |

## Context

After the April 11 repository peer reviews, the v102–v117 cycle built a
directional classifier next to the regression ensemble, with monthly shadow
artifacts and a history ledger, and tested bounded ways for it to gate the
recommendation (a veto on ACTIONABLE months, or a permission gate).

## Decision

Keep classification shadow-only. It is computed and logged every month, and
a shadow gate overlay shows what it would have changed, but it does not set
the recommendation mode or the sell percentage. Promotion needs longer shadow
evidence and matured monitoring.

## Evidence

- v110–v113: bounded veto-style and permission-style gates; a Gemini-style
  veto was the strongest shadow candidate.
- v116–v117: the selected variant (`gemini_veto_0.50`) was ready as a limited
  gate candidate, and v117 concluded: "defer classifier-led recommendation-mode promotion pending longer
  shadow evidence and matured monitoring."
- The same cycle hardened backdated `--as-of` target truncation so simulated
  months cannot see realised targets.

## Consequences

- Later classifier work (v125–v158 Path B, thresholds, Firth; v160–v169 TA,
  [0005](0005-ta-variants-reporting-only.md)) also runs in the shadow lane.
- The shadow-classifier thresholds stay at (0.30, 0.70): v131 found
  (0.15, 0.70), and the v132 temporal hold-out rejected it.

## Sources

- [v102–v117 post-review enhancement plan](../history/superpowers/plans/2026-04-11-v102-v117-post-review-enhancement-plan.md)
- [v117 study README](../../research/studies/v117_primary_mode_selector_evaluation/README.md)
- [April 11 repo peer reviews](../history/archive/history/repo-peer-reviews/2026-04-11/)
