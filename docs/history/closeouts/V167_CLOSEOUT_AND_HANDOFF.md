# V167 Closeout And Handoff

Created: 2026-04-18

## Completed Block

`v167` completes the immediate follow-through after TA-03 by adding a durable
prospective ledger for reporting-only TA shadow classifier variants.

## What Changed

- Added `results/monthly_decisions/ta_shadow_variant_history.csv` as the TA
  shadow monitoring ledger.
- Monthly decision generation now builds TA history rows from
  `classification_shadow_variants` and appends/upserts them by:
  - `as_of_date`
  - `variant`
- Ledger rows preserve:
  - forecast feature anchor date
  - 6M maturity date
  - P(Actionable Sell)
  - stance and confidence tier
  - benchmark count
  - future realized-outcome placeholders
- Only reporting-only variants whose names start with `ta_` are written.

## Production Boundary

No production recommendation, live monthly decision, sell-percentage, or
classifier gate behavior changed. The TA replacement rows remain monitoring
only.

## Verification

Validated with:

```bash
python -m pytest tests/test_classification_artifacts.py -q --tb=short
python -m pytest tests/test_classification_artifacts.py tests/test_classification_shadow.py -q --tb=short
python -m ruff check src/reporting/classification_artifacts.py scripts/monthly_decision.py tests/test_classification_artifacts.py
python scripts/monthly_decision.py --as-of 2026-04-18 --dry-run --skip-fred
```

The dry run appended two TA rows:

- `ta_minimal_replacement`
- `ta_minimal_plus_vwo_pct_b`

Both rows use feature anchor `2026-03-31` and mature on `2026-09-30`.
Incidental dry-run churn in the database and monthly artifacts was restored
after verification.

## Next Direction

Let the monthly workflow accumulate prospective TA shadow observations. The
next useful research block is a matured-outcome evaluator once enough rows have
crossed their 6M horizon; until then, do not promote TA into live gates or
production recommendations.
