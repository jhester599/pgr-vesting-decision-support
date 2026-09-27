# 0008 — Chronological validation only; walk-forward completion and input readiness gates (pre-v200 remediation R3)

| | |
|---|---|
| **Status** | Accepted, live since v188. Amends [0006](0006-validation-gates-and-cpcv-diagnostic.md) |
| **Date** | 2026-09-27 |
| **Where it lives** | `src/reporting/decision_rendering.py` (gates); `src/pgr_vds/decision/health.py` (`assess_wfo_completion`, `assess_data_readiness`, `build_readiness`); `src/database/db_client.py` (`check_required_feed_readiness`); `config/model.py` (`DECISION_GATE_CONTRACT_VERSION`, `WFO_OPTIONAL_BENCHMARKS`); `config/api.py` (`AUDITED_NON_DIVIDEND_PAYERS`) |
| **Findings** | F02, F07, F20, F26 of [`REPO_REVIEW_2026-09-25.md`](../reviews/REPO_REVIEW_2026-09-25.md); V03/N1 and V08 of the [verification](../reviews/VERIFICATION_2026-09-26.md) |
| **Report** | [`R3_validation_closeout.md`](../reviews/R3_validation_closeout.md) |

## Context

After [0006](0006-validation-gates-and-cpcv-diagnostic.md) the monthly
decision still ran the representative CPCV. CPCV trains on folds after its
test folds, i.e. it is a combinatorial K-fold, which AGENTS.md prohibits
(F02). Its only remaining role was a completeness gate: a run where it did
not produce paths could not be ACTIONABLE (F20). That gate:

- tied a valid walk-forward result to a prohibited method;
- said nothing about whether the walk-forward validation itself was
  complete: `run_ensemble_benchmarks` silently skips a model or benchmark
  whose WFO fails, so one successful model looked like a finished run;
- said nothing about the live inputs: a NaN live feature was median-imputed
  with only a warning (F07), and six of the eight benchmarks' dividends were
  stale while the report showed every feed OK (N1);
- judged freshness at the run date, so a back-dated run was judged by, and
  could be made fresh by, data stored later.

## Decision

**Validation is walk-forward only.** The CPCV implementation, its result type,
thresholds and config are removed. `wfo_engine.run_cpcv` is a stub that raises
`UnsupportedValidationMethodError`. The production protocol is unchanged:
outer `TimeSeriesSplit` (60-row window, 6-row test folds, gap = horizon +
purge buffer = 8 rows for 6-month targets), inner Ridge alpha
`TimeSeriesSplit` with the same gap, scaling and imputation fitted per fold,
targets hidden until their window ends.

**Gates** (contract `chronological-readiness-2026-09-27`). ACTIONABLE needs
all five to pass; any FAIL gives DEFER-TO-TAX-DEFAULT at 50 %; otherwise
MONITORING-ONLY.

| Gate | Pass | Fail |
|------|------|------|
| OOS R² against the prevailing mean of realised targets | ≥ 2 % | < 0 % |
| Equal-weight mean of the per-benchmark OOS ICs | ≥ 0.07 | < 0.03 |
| Directional skill: one-sided Pesaran–Timmermann p, Driscoll–Kraay by date | < 0.05 | ≥ 0.10 or undefined |
| `wfo_completed` | exactly `true` | anything else |
| `data_ready` | exactly `true`, no missing feature, no stale feed | anything else |

`wfo_completed` is true only when, for every required model (Ridge, GBT) and
every required benchmark (the eight of `PRIMARY_FORECAST_UNIVERSE`; none is
optional), the WFO result exists and every fold is non-empty, has finite
predictions and outcomes, tests after the previous fold, trains on rows that
end at least the production gap before its test start with the last label
window realised by then, and tests only rows whose outcome was realised by
the as-of date; and each benchmark has a finite live forecast.

`data_ready` is true only when every live model feature is finite before
imputation, the decision row is the as-of date's decision month, and prices
(PGR and the eight), the FRED series behind live features, PGR monthly EDGAR
and dividends (PGR and the eight) are fresh at the as-of date. GLD is the one
audited non-payer; any other ticker without dividend history is not ready, so
a failed or empty dividend request is never read as "pays none".

Every check is bounded by the as-of date (EDGAR by filing date). A back-dated
run's readiness is labelled `backdated_reconstruction`: the DB does not record
when each value was fetched, so it is not evidence of what the original
decision had.

Every output surface names the failed gates and their reasons:
`recommendation.md` (executive summary, Confidence Snapshot; the e-mail
carries both), `monthly_summary.json`, `run_manifest.json` (`decision_gates`),
the dashboard warnings and the decision-log Notes.

Unchanged: the three quality gates and their thresholds, the metric
definitions (`MODEL_HEALTH_METRICS_VERSION`), the sell mapping
([0007](0007-actionable-sell-mapping.md)), features, model parameters,
targets and consensus weighting. The v13 simpler-baseline and v22 cross-check
paths (not live: `RECOMMENDATION_LAYER_MODE = live_only`) carry no readiness
contract and stay non-ACTIONABLE, as they were under the CPCV gate.

## Evidence

Tests: `tests/integration/pipeline/test_production_validation_contract.py`
(41 of 50 fail before the change, 50 pass after) and
`tests/unit/models/test_cpcv_retired.py`. A spy on the synthetic production
path counts one `run_cpcv` call and one `CombinatorialPurgedCV` construction
before, zero after.

Pre-refresh smoke replay of the eight committed months on one DB copy, old
and new code: see the [closeout](../reviews/R3_validation_closeout.md#smoke-replay-2026-02--2026-09).
It is smoke verification of already-inspected history, not promotion evidence
and not the new baseline.

## Consequences

- No month can be ACTIONABLE while a required dividend feed is stale. On the
  DB before the step-0 dividend refresh, the recent months therefore defer on
  `data_ready` as well as on directional skill.
- The replay on the refreshed DB and the new current health baseline follow
  once R2-lite has merged; `docs/model-governance.md` keeps the step-5
  baseline as dated history until then.
