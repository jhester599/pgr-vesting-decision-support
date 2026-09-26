# v79-v80 Post-Promotion Stabilization

## Purpose

This note captures the first post-merge stabilization pass after the quality-weighted consensus promotion landed in production.

The goal of this cycle was narrow:

- verify that the merged monthly reporting path still uses the promoted ensemble logic
- confirm that the new diagnostic artifacts are produced in the live monthly workflow
- decide whether the equal-weight cross-check should remain visible for one more cycle

---

## What Was Fixed

During the first April 2026 rerun after the production merge, the monthly output folder was missing:

- `benchmark_quality.csv`
- `consensus_shadow.csv`

The root cause was that `master` had an older `scripts/monthly_decision.py` path that no longer fully reflected the `v66-v75` diagnostic and consensus changes. In particular:

- aggregate health still fell back to a component-model OOS series in some paths
- benchmark-quality export was not written
- consensus shadow export was not written
- the run manifest did not include the new artifacts

This stabilization cycle restored the promoted reporting/evaluation wiring and revalidated the monthly flow.

---

## Verified April 2026 Rerun

As-of date used for the stabilization rerun:

- `2026-04-11`

Observed live monthly outputs after the restore:

- consensus: `NEUTRAL`
- confidence tier: `LOW`
- live mean predicted return: `-2.34%`
- live mean IC: `0.1744`
- live mean hit rate: `66.8%`
- aggregate OOS R^2: `-2.75%`
- recommendation mode: `DEFER-TO-TAX-DEFAULT`
- recommended sell percentage: `50%`

Equal-weight production cross-check:

- mean predicted return: `-2.45%`
- mean IC: `0.1703`
- mean hit rate: `66.3%`
- recommendation mode: `DEFER-TO-TAX-DEFAULT`
- recommended sell percentage: `50%`

Key stabilization takeaway:

- the live quality-weighted path and the equal-weight cross-check reached the same recommendation outcome
- the restored promoted path produced materially better aggregate diagnostics than the initial broken rerun
- the new monthly artifacts are now present and listed in the manifest

---

## Artifact Check

The April 2026 monthly folder now correctly includes:

- `recommendation.md`
- `diagnostic.md`
- `signals.csv`
- `benchmark_quality.csv`
- `consensus_shadow.csv`

The run manifest also records both new CSV artifacts in its `outputs` section.

---

## Validation

Targeted verification completed successfully:

- `python -m pytest tests\test_consensus_shadow.py tests\test_monthly_ensemble_alignment.py tests\test_monthly_pipeline_e2e.py tests\test_monthly_logging.py tests\test_diagnostic_report.py -v`
- `python -m mypy src\models\forecast_diagnostics.py src\models\consensus_shadow.py src\models\evaluation.py config\model.py`

Result:

- `47` targeted tests passed
- mypy passed with no issues in the focused stabilization modules

---

## Decision

Keep the equal-weight cross-check visible for one more release cycle.

Reasoning:

- this was the first post-merge stabilization rerun after promotion
- the cross-check is low-cost and now renders correctly in monthly reporting
- live and shadow outputs currently agree on recommendation mode and sell percentage
- one more clean monthly cycle will provide a better basis for retiring the cross-check without reducing observability too early

---

## Recommended Next Step

After this stabilization fix merges:

1. run the next monthly report on the promoted path
2. confirm that live and cross-check outputs remain stable
3. retire the equal-weight cross-check if it continues to add no decision-changing information
