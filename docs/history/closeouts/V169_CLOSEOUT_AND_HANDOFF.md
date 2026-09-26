# V169 Closeout And Handoff

Created: 2026-04-19

## Completed Block

`v169` moves monthly decision postcondition checks out of inline GitHub Actions
glue and into a reusable, testable verifier script.

## What Changed

- Added `scripts/verify_monthly_outputs.py`.
- The verifier checks:
  - required monthly decision artifacts
  - `monthly_summary.json` and `run_manifest.json`
  - calendar-aware data freshness through `db_client.check_data_freshness`
  - reporting-only TA variants in `classification_shadow.csv`
  - matching rows in `results/monthly_decisions/ta_shadow_variant_history.csv`
- The monthly decision workflow now calls:

```bash
python scripts/verify_monthly_outputs.py --summary-path workflow_summary.md
```

and appends that summary to `$GITHUB_STEP_SUMMARY`.

## Verification

Validated with:

```bash
python -m pytest tests/test_verify_monthly_outputs.py -q --tb=short
python -m pytest tests/test_workflow_contracts.py tests/test_verify_monthly_outputs.py -q --tb=short
python -m ruff check scripts/verify_monthly_outputs.py tests/test_verify_monthly_outputs.py tests/test_workflow_contracts.py
python scripts/monthly_decision.py --as-of 2026-04-19 --dry-run --skip-fred
python scripts/verify_monthly_outputs.py --as-of 2026-04-19 --summary-path workflow_summary.md
```

The real-output verifier run reported data freshness `OK` and two TA shadow
variants. Generated dry-run monthly artifacts and the temporary
`workflow_summary.md` file were restored afterward.

## Production Boundary

No model, feature, recommendation, sell percentage, classifier gate, or live
decision behavior changed.

## Next Direction

After this merges, let the monthly workflows run. If a future run fails the
postcondition verifier, investigate the specific failed artifact or TA ledger
row before trusting the monthly report output.
