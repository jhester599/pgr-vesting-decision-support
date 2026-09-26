# 0003 — Post-promotion stabilisation, `monthly_summary.json`, cross-check retired (v79–v86)

| | |
|---|---|
| **Status** | Accepted, live |
| **Date** | 2026-04-11 |
| **Where it lives** | `src/reporting/monthly_summary.py`; `artifacts/monthly_decisions/<YYYY-MM>/monthly_summary.json` |

## Context

After the v76 promotion ([0002](0002-quality-weighted-consensus.md)) the first
merged monthly rerun had lost part of the promoted reporting and evaluation
wiring. The workflow, e-mail, dashboard and docs had also drifted from the
promoted baseline.

## Decision

- **v79–v80.** Restore the promoted reporting path and validate it on a real
  monthly rerun before anything else changes.
- **v81–v84.** Align the workflow, e-mail, dashboard and docs to the promoted
  baseline; write a static dashboard snapshot every month.
- **v85.** Add `monthly_summary.json` as the machine-readable contract for each
  monthly run.
- **v86.** Retire the visible equal-weight cross-check from the report and
  e-mail; keep `consensus_shadow.csv` as the diagnostic artifact.

## Evidence

The first post-merge rerun had not written `benchmark_quality.csv` or
`consensus_shadow.csv`, and the run manifest did not list the new artifacts.
The stabilisation rerun as of 2026-04-11, after the restore:

| Path | Mean forecast | Mean IC | Hit rate | Mode | Sell % |
|---|---:|---:|---:|---|---:|
| Quality-weighted (live) | −2.34 % | 0.1744 | 66.8 % | DEFER-TO-TAX-DEFAULT | 50 % |
| Equal-weight (cross-check) | −2.45 % | 0.1703 | 66.3 % | DEFER-TO-TAX-DEFAULT | 50 % |

The two paths reached the same recommendation, so the visible cross-check
added no decision information; the diagnostic CSV keeps the comparison.
These metrics predate review 2026-09-25 and carry its look-ahead (F04, F13).

## Consequences

- Automation reads `monthly_summary.json`, not the Markdown report. Whether it
  becomes the only contract for notification surfaces is still open
  (`docs/model-governance.md`, Current Governance Conclusion).
- The equal-weight path remains computable and auditable month by month.

## Sources

- [v79–v80 post-promotion stabilisation](../history/superpowers/plans/2026-04-11-v79-v80-post-promotion-stabilization.md)
- [v81–v88 repo review and adoption plan](../history/superpowers/plans/2026-04-11-v81-v88-repo-review-and-adoption-plan.md)
