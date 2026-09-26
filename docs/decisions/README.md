# Decision Records

One file per promotion decision: a change to what the live monthly decision
does, or a deliberate choice to keep a candidate out of it. Each file says
what was decided, the evidence, and what followed. The summary table, the
gates in force and the current health baseline are in
[`docs/model-governance.md`](../model-governance.md#decision-record).

These were split out of `docs/model-governance.md` in review 2026-09-25,
section 5, phase 4. Records 0001–0005 are reconstructed from the plans and
closeouts of their cycle (linked in each file), now under
[`docs/history/`](../history/README.md).

| # | Decision | Date | Status |
|---|---|---|---|
| [0001](0001-v38-post-ensemble-shrinkage.md) | Post-ensemble shrinkage (v38) | 2026-04-10 | Live; alpha prequential since 0006 |
| [0002](0002-quality-weighted-consensus.md) | Quality-weighted consensus (v72 → v76) | 2026-04-10 | Live |
| [0003](0003-monthly-summary-contract.md) | Post-promotion stabilisation, `monthly_summary.json`, cross-check retired (v79–v86) | 2026-04-11 | Live |
| [0004](0004-classifier-stays-shadow-only.md) | Classifier stays shadow-only (v102–v117) | 2026-04-11 | Shadow only |
| [0005](0005-ta-variants-reporting-only.md) | Technical-analysis classifier variants are reporting-only (v160–v169) | 2026-04-18 | Reporting only |
| [0006](0006-validation-gates-and-cpcv-diagnostic.md) | Recommendation-mode gates, realised-only health, CPCV as a diagnostic | 2026-09-26 | Live |
| [0007](0007-actionable-sell-mapping.md) | ACTIONABLE sell-percentage mapping | 2026-09-26 | Live |

## Adding a decision

1. Copy the layout of the newest file: a header table (status, date, where
   it lives in code and config, study or findings, report), then Context,
   Decision, Evidence, Consequences, Sources.
2. Name it `NNNN-short-slug.md` with the next number. Numbers are never
   reused.
3. Add a row here and to the Decision Record table in
   `docs/model-governance.md`, in the same PR as the change.
4. When a later decision changes an earlier one, set the earlier file's
   status to "Amended by NNNN" or "Superseded by NNNN" and link it. Leave
   the rest of the earlier file as it was.
5. If the change came from a study, set its `status` and `promoted_to` in
   `research/registry.yaml`.
