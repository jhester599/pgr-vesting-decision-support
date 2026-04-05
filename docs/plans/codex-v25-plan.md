# v25 Accuracy-Through-Correctness Plan

Created: 2026-04-05

## Goal

Use the two external peer reviews to drive the next enhancement cycle without
reopening another broad model / feature search.

The v25 objective is:

- improve forecast reliability by fixing correctness and leakage issues first
- reduce silent failure risk in production
- rerun the key historical comparisons on the corrected foundation
- then make an explicit stop / continue decision on further prediction research

## Inputs

Archived reports:

- [chatgpt_repo_peerreview_20260404-1.md](/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/docs/history/repo-peer-reviews/2026-04-05/chatgpt_repo_peerreview_20260404-1.md)
- [claude_repo_peerreview_20260404.md](/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/docs/history/repo-peer-reviews/2026-04-05/claude_repo_peerreview_20260404.md)

Current repo context:

- active recommendation layer remains the simpler diversification-first path
- visible cross-check is `ensemble_ridge_gbt_v18`
- `v24` concluded `keep_voo`
- the remaining most likely technical accuracy drags are correctness issues, not
  missing model complexity

## v25.0 - Canonical Monthly Index Normalization

### Objective

Eliminate silent `ME` / `BME` mismatches across feature engineering, EDGAR
monthly data, synthetic benchmark-side series, and target alignment.

### Deliverables

- one canonical helper module for monthly-index normalization
- one canonical repo rule:
  - either `BME` everywhere or `ME` everywhere
- weekend-month-end regression tests
- updated architecture / data-source docs with the canonical convention

### Expected File Scope

- `src/processing/feature_engineering.py`
- `src/database/db_client.py`
- `src/ingestion/edgar_8k_fetcher.py`
- optionally a new helper such as `src/utils/time_index.py`
- tests in `tests/`

### Acceptance Criteria

- no mixed monthly index conventions remain in active feature-building paths
- a February-2026-style weekend month-end test passes
- aligned feature / target date counts do not silently shrink

## v25.1 - WFO And CPCV Integrity Sweep

### Objective

Fix the remaining evaluation-integrity issues that can distort both model
selection and stability diagnostics.

### Required Fixes

- add an inner CV `gap` to the regularized-model tuning paths
- update the WFO minimum-length guard to require `TRAIN + GAP + TEST`
- fix CPCV recombined-path prediction provenance so path ICs use split-specific
  predictions rather than a shared overwritten array

### Expected File Scope

- `src/models/regularized_models.py`
- `src/models/wfo_engine.py`
- new or expanded tests in `tests/`

### Acceptance Criteria

- inner CV splitters explicitly include a non-zero `gap`
- the WFO guard fails early on too-short datasets using the true total gap
- CPCV gets a provenance regression test that would have failed before the fix

## v25.2 - Silent-Failure Guards

### Objective

Convert silent missingness, stale data, and schema drift into explicit
warnings or failures.

### Required Fixes

- audit and close the `roe_net_income_trailing_12m` vs `roe_net_income_ttm`
  mapping path
- add all-null checks for expected EDGAR / macro feature groups
- add a data-freshness and schema-version guard at the monthly decision entry
  point
- clean up hardcoded benchmark-count / ticker-count doc drift

### Expected File Scope

- `scripts/edgar_8k_fetcher.py`
- `src/database/db_client.py`
- `src/processing/feature_engineering.py`
- `scripts/monthly_decision.py`
- selected docs / docstrings

### Acceptance Criteria

- ROE naming is provably normalized in the live path
- monthly runs cannot silently proceed with empty expected feature groups
- operational docs no longer hardcode stale universe counts

## v25.3 - Rerun The Studies That These Fixes Can Affect

### Objective

Re-evaluate the conclusions that are sensitive to the corrected indexing and
validation logic.

### Scope

At minimum rerun:

- `v20`
- `v21`
- `v23`
- `v24`

Why those:

- `v20-v21` depend directly on historical comparison integrity
- `v23-v24` depend on target alignment and benchmark-series construction

### Acceptance Criteria

- new artifacts are written under the existing version folders or a clearly
  marked `v25` rerun folder
- any changed promotion conclusion is documented explicitly
- unchanged conclusions are also documented explicitly, so the rerun has value

## v25.4 - Governance Closeout

### Objective

Decide whether the repo should continue prediction-layer research after the
correctness fixes land.

### Required Outputs

- compute a concise stop / continue memo that addresses the Claude review's
  central question:
  - is there still enough evidence to justify continued ML prediction research?
- if feasible, add one of:
  - Probability of Backtest Overfitting style summary across the v9-v24 family
  - a simpler multiple-testing-aware scorecard across major research candidates

### Decision Rule

- If the corrected reruns materially improve stability or the visible
  cross-check logic, continue narrowly.
- If the corrected reruns still leave the prediction layer weak and
  non-actionable, freeze the recommendation layer as the main product and
  demote further prediction work to optional research only.

## Explicit Non-Goals For v25

- no new broad feature search
- no new model families
- no benchmark-universe redefinition beyond what is necessary to support the
  correctness fixes
- no `VOO -> VTI` production swap
- no major config-system rewrite yet

## Sequence Recommendation

Execute strictly in this order:

1. `v25.0`
2. `v25.1`
3. `v25.2`
4. `v25.3`
5. `v25.4`

Do not start another broad predictive-accuracy experiment before `v25.3` is
complete.
