# v32 Enhancement Sequence

Created: 2026-04-06

## Goal

Continue the 2026-04-05 peer-review follow-up with the four remaining Tier 4
strategic ML diagnostics.  All Tier 1 and Tier 2 items are complete.  This
sequence surfaces feature-stability, multicollinearity, vesting-policy, and
heuristic-comparison diagnostics in the monthly workflow outputs.

## Versioning Approach

The `v31` sequence closed out conformal and drift monitoring.  This follow-on
sequence starts at `v32.0` for ML diagnostic enhancements.

- `v32.0` - feature importance stability across WFO folds (Tier 4.1)
- `v32.1` - VIF multicollinearity checks (Tier 4.2)
- `v32.2` - wire vesting decision policy backtest into monthly report (Tier 4.3)
- `v32.3` - model vs. simple heuristic comparison in monthly report (Tier 4.4)
- `v32.4` - refresh peer-review status against landed `v32` work

## v32.0 Scope

`v32.0` addresses peer-review Tier 4.1 with a reusable stability metric:

- add `compute_feature_importance_stability()` to `src/research/evaluation.py`
  - input: a `WFOResult` (list of `FoldResult` each with `feature_importances`)
  - output: `FeatureImportanceStability` dataclass with per-feature mean rank,
    rank std deviation, and an overall stability score (mean pairwise Spearman
    rank-correlation between consecutive folds)
- add a `FeatureImportanceStability` dataclass to the same module
- wire `compute_feature_importance_stability()` into `_write_diagnostic_report`
  in `scripts/monthly_decision.py` as a new **Feature Importance Stability**
  subsection under the Feature Governance section
- add focused pytest coverage for the helper contract, consecutive-fold
  rank correlation, degenerate (single-fold) edge case

## v32.1 Scope

`v32.1` addresses peer-review Tier 4.2:

- add `compute_vif()` to `src/processing/feature_engineering.py`
  - input: `pd.DataFrame` of feature values (rows = observations)
  - output: `pd.Series` keyed by feature name with VIF values; rows with
    perfect collinearity (VIF = ∞ / NaN) are excluded with a logged warning
  - uses `statsmodels.stats.outliers_influence.variance_inflation_factor` which
    is already pulled in transitively; add explicit import guard
- add `VIF_HIGH_THRESHOLD` and `VIF_WARN_THRESHOLD` to `config.py` (defaults
  10.0 and 5.0 respectively)
- wire VIF computation into `_write_diagnostic_report` via a `vif_series`
  parameter and surface a **Multicollinearity (VIF)** table in the Feature
  Governance section with per-feature flags
- add focused pytest coverage for the VIF helper and the flag thresholds

## v32.2 Scope

`v32.2` addresses peer-review Tier 4.3 by wiring the existing policy machinery
into the monthly report:

- add `_compute_monthly_policy_summary()` helper in `scripts/monthly_decision.py`
  - collects the aggregated OOS predictions and realized returns already built
    during `_write_diagnostic_report`
  - calls `evaluate_policy_series()` from `src/research/policy_metrics.py` with
    the model's OOS signal for a representative fixed policy set
    (`sell_all`, `hold_all`, `sell_50pct`, `signal_quartile`)
  - returns a list of `PolicySummary` objects, one per policy
- surface a compact **Decision Policy Backtest** section in `recommendation.md`
  showing per-policy mean return, hit rate, and regret vs. oracle over the full
  OOS history (not just trailing window)
- extend the monthly end-to-end test to assert the section is present when
  sufficient OOS data exists

## v32.3 Scope

`v32.3` addresses peer-review Tier 4.4 by extending the policy summary:

- extend `_compute_monthly_policy_summary()` from `v32.2` to also compute uplift
  of the model-driven policy vs. each fixed heuristic (`sell_all`, `hold_all`,
  `sell_50pct`) using the `uplift_vs_*` fields already in `PolicySummary`
- add a **Model vs. Heuristics** comparison table to the `recommendation.md`
  section added in `v32.2`; columns: policy, mean return, cumulative return,
  uplift vs. sell-all, uplift vs. hold-all, uplift vs. 50 %
- extend the end-to-end test to assert the uplift column is present

## v32.4 Scope

`v32.4` keeps the planning layer accurate after the new diagnostic work:

- update the 2026-04-05 peer-review status snapshot to mark Tier 4.1–4.4
  progress accurately
- map the completed `v32` steps back to the peer-review enhancement list
- refresh the "highest-value remaining gaps" section to point at Tier 3.2/3.4
  and Tier 5 items

## PR Checkpoint

After `v32.4`, open a draft PR for the v32 sequence:

- the changes form a coherent diagnostics-enhancement slice
- Tier 4.1–4.4 are now surfaced in the monthly outputs
- subsequent v33 code-quality work can continue on the same branch or a
  successor branch

## Notes

- `statsmodels` is not currently in `requirements.txt`; `v32.1` adds it.
- The `policy_evaluation.py` standalone script is a research tool; `v32.2`
  does NOT modify it — instead it reuses the underlying
  `src/research/policy_metrics.py` helpers directly.
- VIF computation is expensive on wide feature matrices; limit computation to
  the reduced feature set returned by `get_feature_columns()` and compute on
  the fully-populated subset of the training data.
