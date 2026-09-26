# V166 Closeout And Handoff

Created: 2026-04-18

## Completed Block

`v166` completes TA-03 by wiring the TA replacement candidates into the monthly
shadow classifier artifact path as reporting-only variants.

## What Changed

- `classification_shadow.csv` now supports TA replacement rows with:
  - `variant_label`
  - `feature_set`
  - `reporting_only`
- Monthly decision generation appends these TA variants when the baseline
  classifier shadow is available:
  - `ta_minimal_replacement`
  - `ta_minimal_plus_vwo_pct_b`
- `monthly_summary.json` includes TA replacement payloads in
  `classification_shadow_variants`.
- The classifier gate overlay still reads only the existing baseline shadow
  probability.

## Data And Workflow

No new Alpha Vantage schedule was added. TA-03 uses existing daily OHLCV data
for `PGR` and `VWO`, which are already fetched by the Friday weekly data
accumulation workflow through `get_all_price_tickers()`.

The weekly workflow now checks that required TA shadow price tickers have data
after the fetch and prints their latest dates in the workflow summary.

## Verification

Validated with:

```bash
python -m pytest tests/test_classification_shadow.py tests/test_classification_artifacts.py -q --tb=short
python -m ruff check src/models/classification_shadow.py src/reporting/classification_artifacts.py scripts/monthly_decision.py tests/test_classification_shadow.py tests/test_classification_artifacts.py
python scripts/monthly_decision.py --as-of 2026-04-18 --dry-run --skip-fred
```

The dry run confirmed `classification_shadow.csv` includes:

- `baseline_shadow`
- `autoresearch_followon_v150`
- `ta_minimal_replacement`
- `ta_minimal_plus_vwo_pct_b`

and `monthly_summary.json` carries reporting-only TA variant payloads. Generated
dry-run monthly artifacts were restored after verification to avoid committing
timestamp/noise churn.

## Next Direction

Monitor the TA variants prospectively through monthly artifacts. Do not promote
to production or gate behavior until enough matured prospective observations
exist to evaluate calibration, balanced accuracy, Brier score, and stability.
