# 0005 — Technical-analysis classifier variants are reporting-only (v160–v169)

| | |
|---|---|
| **Status** | Accepted. Reporting-only shadow monitoring |
| **Date** | 2026-04-18 → 2026-04-19 |
| **Where it lives** | `classification_shadow.csv` (`reporting_only` rows); `artifacts/monthly_decisions/ta_shadow_variant_history.csv`; `scripts/verify_monthly_outputs.py` |
| **Studies** | `research/studies/v162_ta_broad_screen/`, `v163_ta_survivor_confirm/`, `v164_ta_synthesis/`, `v165_ta_shadow_replacement_eval/` |

## Context

Three external reports proposed Alpha Vantage-style technical-analysis
features for the classifier. The v160–v164 arc screened them under
walk-forward validation only, with no change to the live path.

## Decision

Monitor two TA replacement variants of the lean 12-feature classifier as
reporting-only shadow rows: `ta_minimal_replacement` and
`ta_minimal_plus_vwo_pct_b`. They appear in `classification_shadow.csv` and
`monthly_summary.json` and are logged to a TA ledger, but the gate overlay
reads only the baseline shadow probability.

## Evidence

- v164: `replacement_candidate` for a later shadow-only replacement plan.
- v165: `shadow_monitor`; the strongest historical variant,
  `ta_minimal_plus_vwo_pct_b`, improved mean balanced accuracy by +0.0584 and
  mean Brier by −0.0656, positive on 8 of 8 benchmarks.

## Consequences

- v166 wired the variants into the monthly artifacts; v167 added the ledger
  (forecast anchor, 6-month maturity date, probability, stance, and
  placeholders for realised outcomes); v169 added the output verifier.
- No new Alpha Vantage calls: the variants use the PGR and VWO bars the weekly
  fetch already stores.
- Promotion needs enough matured 6-month horizons to judge calibration,
  Brier score, balanced accuracy and stability.

## Sources

- [v160–v164 technical-analysis plan](../history/superpowers/plans/2026-04-18-v160-v164-technical-analysis-feature-research.md)
- [V165](../history/closeouts/V165_CLOSEOUT_AND_HANDOFF.md),
  [V166](../history/closeouts/V166_CLOSEOUT_AND_HANDOFF.md),
  [V167](../history/closeouts/V167_CLOSEOUT_AND_HANDOFF.md),
  [V169](../history/closeouts/V169_CLOSEOUT_AND_HANDOFF.md) closeouts
- [TA research reports](../history/archive/history/v160-ta-research-reports/)
