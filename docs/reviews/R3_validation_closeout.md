# R3 closeout — chronological validation, readiness gates and current state

Session R3 of [the pre-v200 fix prompts](PRE_V200_FIX_PROMPTS_codex.md), with
the 2026-09-27 amendments (D3, D4, N8). Builder: Claude (Linux, cloud).
Reviewer: Codex. CHANGELOG **v188**. Decision record:
[0008](../decisions/0008-chronological-validation-and-readiness-gates.md).

Findings addressed: F02 (production CPCV/K-fold), F07/N1 (warning-only live
inputs, invisible stale dividends), F20 (completeness gate), F26 (as-of
safety of back-dated runs) and V08 (stale governance baseline). F04, F13 and
F21 are unchanged: their gates and metrics are kept as they are.

**Status.** The code and gate contract are complete. The replay on the
refreshed DB and the new current baseline are **pending R2-lite** (D3: "the
replay and the new baseline must use the refreshed DB, after R2-lite
merges"). On 2026-09-27 the step-0 dividend refresh had not run yet: it is
scheduled for 2026-09-28. This session therefore records a **pre-refresh smoke
replay** that isolates the code change. Data changes stay attributable to
step 0 / R2-lite.

## Pins

| | |
|---|---|
| Base | `master` `449fcdecee9abfaa8114f9cdee63ad2c7ffb5cc4` (PR #137) |
| R3 code used for the replay and counterfactual | `1b94bf0` on `claude/remediation-session-0btise`; the later wording-only commit changes the deferral summary text, not any gate, metric or mode |
| Tracked DB | sha256 `f453ab9817ffbc5d176bcca03ca7c64a86493db652a150012ac852d93811f51d` (after the 2026-09-27 peer update `e215be9`; before the step-0 dividend refresh). The verification's pin `7c68efbd…` predates the peer update. |
| Runtime | Python 3.11.15; pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1, scipy 1.17.1, statsmodels 0.15.0, xgboost 3.2.0; Linux 6.18 |
| Metric version | `prequential-2026-09-25` (unchanged: no metric definition changed) |
| Gate contract | `chronological-readiness-2026-09-27` (new) |

No provider call, fetcher run or e-mail was made. The tracked DB was only
read, and only through `mode=ro&immutable=1` URIs (dividend and EDGAR
inspection) or through copies. Subagents used: none.

## What changed

- **CPCV retired from active execution.** `signal_generation.generate_signals`
  no longer calls `run_cpcv`. The CPCV implementation, `CPCVResult`,
  `cpcv_path_thresholds`, `_recombined_path_members` and the `CPCV_*` /
  `DIAG_CPCV_*` config are removed. `wfo_engine.run_cpcv` is a stub that raises
  `UnsupportedValidationMethodError` before any splitter is built. No
  historical CPCV result needs deserialising: old payloads are JSON, and the
  replay script reads them as `historical_*` columns.
- **Gates** (`src/reporting/decision_rendering.py`): `oos_r2`, `mean_ic`,
  `directional_skill` (unchanged thresholds), `wfo_completed`, `data_ready`.
  Only exactly `True` passes. Unknown, missing or truthy values fail. Any FAIL
  gives DEFER-TO-TAX-DEFAULT at 50 %; the sell mapping is unchanged.
- **`wfo_completed`** (`health.assess_wfo_completion`) requires:
  - every required pair: `PRIMARY_FORECAST_UNIVERSE` (8) × `ENSEMBLE_MODELS`
    (ridge, gbt) = 16; `WFO_OPTIONAL_BENCHMARKS = ()`;
  - for every fold of each pair:
    - non-empty, with finite `y_hat` and `y_true`;
    - test dates in order, after the previous fold;
    - training ends ≥ 8 months (horizon + purge) before the test start;
    - the last training label window ends by the test start;
    - the test rows' outcomes are realised by the as-of date;
  - a finite live forecast for each benchmark.

  Failures are listed per pair.
- **`data_ready`** (`health.assess_data_readiness`,
  `db_client.check_required_feed_readiness`) requires:
  - every configured live feature is finite before imputation; a configured
    feature absent from the frame also counts as missing;
  - the decision row is the as-of date's decision month;
  - the required feeds are fresh at the as-of date:
    - prices of PGR and the eight benchmarks;
    - the FRED series behind live features;
    - PGR monthly EDGAR, judged by filing date (a row without one counts once
      its month has ended, as before);
    - dividends of PGR and the eight benchmarks. GLD is the one audited
      non-payer (`AUDITED_NON_DIVIDEND_PAYERS`); any other missing history
      fails.

  `check_dividend_freshness` gains `as_of`. The monthly freshness report is
  now evaluated at the as-of date, not the run date, and includes one row per
  required dividend feed.
- **As-of.** Back-dated runs are labelled
  `readiness_basis = backdated_reconstruction`, with a note that the DB does
  not record fetch times. A repaired value can therefore pass there although
  the original decision did not have it.
- **Surfaces.** The reason is named in all of these:
  - `recommendation.md`: the executive summary "How trustworthy" line and the
    Confidence Snapshot rows. The e-mail reuses both; `email_sender.py` is
    unchanged (D4).
  - `monthly_summary.json`: `recommendation.failed_gates` and
    `deferral_reasons`, plus `model_health.gate_contract_version` and
    `model_health.readiness`.
  - `run_manifest.json`: the `decision_gates` block and a warning.
  - the dashboard warnings;
  - `decision_log.md` Notes.

  The only e-mail lines that change are the removed CPCV text, the new gate
  rows and the named deferral reasons. When only quality gates fail, the
  pre-R3 summary sentence is unchanged, with the reasons appended.
- **Replay script** records the contract version, gate statuses, readiness
  and `db_sha256` per row. Pre-R3 payloads are labelled
  `historical (pre-R3)` and never given readiness values.
- **Not changed:** features, model parameters, targets, consensus weighting,
  thresholds, the WFO protocol, `MODEL_HEALTH_METRICS_VERSION`, the old
  performance log and the committed monthly artifacts. The v13 shadow and v22
  cross-check are not live (`live_only`). They carry no readiness contract,
  so they stay non-ACTIONABLE, exactly as under the CPCV gate.

## WFO protocol audit

| Item | Production value | Evidence |
|---|---|---|
| Outer split | `TimeSeriesSplit(n_splits=(n−60−8)//6, max_train_size=60, test_size=6, gap=8)` for 6M targets | `test_wfo_gap_is_the_horizon_plus_the_purge_buffer` (spy on the constructor) |
| Coverage | rows 72..239 of 240 scored once each, in order | `test_wfo_scores_every_row_after_the_first_window_once_in_order` |
| Label end | test start is 9 months after training end; the last training label (6M) ends 3 months before it | `test_wfo_protocol_is_unchanged_and_trains_only_on_realised_labels` |
| No future training | perturbing targets after fold k's test end leaves folds 0..k unchanged (atol 1e-12) | same test |
| Inner tuning | `RidgeCV(cv=AdaptiveGapTimeSeriesSplit(n_splits=3, gap=8))`; GBT fixed shallow | `src/models/regularized_models.py`; AST scan in `test_cpcv_retired.py` |
| Scaling/imputation | `StandardScaler` inside the per-fold pipeline; training-fold medians | `wfo_engine.run_wfo` |
| Maturity | targets hidden until their BME window end ≤ as-of | `truncate_relative_target_for_asof` |

No leakage was found, so the production protocol is unchanged. The stricter
v200 research settings are not adopted here.

## Red / green

New tests: `tests/integration/pipeline/test_production_validation_contract.py`
(50 tests), `tests/unit/models/test_cpcv_retired.py` (13 tests).

**Red, source checkout, unmodified `master` code with the new contract file**
(`python -m pytest -o addopts="--tb=line" -q tests/integration/pipeline/test_production_validation_contract.py`):

```text
41 failed, 9 passed in 34.06s
EXIT=1
```

Representative failures (reasons, count):

```text
 1  test_live_decision_does_not_invoke_cpcv:   assert {'CombinatorialPurgedCV': 1, 'run_cpcv': 1} == {... 0, ... 0}
 1  test_missing_live_feature_is_found_before_imputation: same spy counts 1 and 1
 1  test_healthy_wfo_is_independent_of_retired_cpcv: ['oos_r2','mean_ic','directional_skill','cpcv_completed'] != [..., 'wfo_completed', 'data_ready']
 1  test_healthy_run_is_actionable_on_every_surface: 'DEFER-TO-TAX-DEFAULT' == 'ACTIONABLE'
 5  test_incomplete_wfo_blocks_actionable[*]:   gate 'wfo_completed' absent (None == 'FAIL'); the old code returned ACTIONABLE
 8  test_unknown_data_readiness_blocks_actionable[*]: gate 'data_ready' absent (None == 'FAIL')
 6  test_deferral_reason_is_named_in_every_output_surface[*]: KeyError 'decision_gates'
10  test_wfo_completed_*: no health.assess_wfo_completion
 3  test_stale_required_dividend / test_backdated_readiness / test_later_data: no db_client.check_required_feed_readiness
```

The 9 passes on the old code are the already-green missing/non-finite metric
safety tests (7), `test_no_health_at_all_blocks_actionable` and the protocol
audit. They are kept, and they show the protocol did not change.

**Counterfactual reversal in an external scratch clone.** Clone at `1b94bf0`,
with `src/` and `config/` checked out from `449fcde`; tests kept. Run with
`PYTHONPATH=<clone>:<clone>/src` so that the clone's own code is imported (the
editable install points at the source checkout).

```text
$ git checkout 449fcde -- src config
$ python -m pytest -q tests/integration/pipeline/test_production_validation_contract.py tests/unit/models/test_cpcv_retired.py
46 failed, 17 passed in 34.40s        EXIT=1
$ git checkout HEAD -- src config
$ python -m pytest -q (same files)
63 passed in 22.14s                   EXIT=0
```

The 17 passes against the reverted code are the 9 above plus the AST scan's
own counterfactual probes (7 forbidden patterns caught, the production inner
splitter accepted). `test_no_active_module_uses_k_fold_validation`,
`test_run_cpcv_is_an_unsupported_method[*]` and
`test_cpcv_result_types_are_gone` fail against the reverted code.

**Mutations** (`scripts/checks/mutation_study.py <clone> M09… M14…`; M09 and
M14 targeted the CPCV and are retargeted):

```text
M09_wfo_gap_dropped KILLED | FAILED tests/unit/models/test_wfo_engine.py::TestWFOTemporalIntegrity::test_embargo_gap_enforced
M14_wfo_completed_gate_dropped KILLED | FAILED tests/integration/pipeline/test_validation_gating.py::test_missing_or_unknown_validation_does_not_permit_actionable[missing]
F28 survivors: 0 of 2
```

## Smoke replay 2026-02 → 2026-09

Pre-v200 smoke verification of already-inspected history: **not promotion
evidence and not the new baseline.** Two external scratch clones, each with its
own copy of the tracked DB (`f453ab98…`):

- old code: `…/scratchpad/r3_old`, at `449fcde`;
- new code: `…/scratchpad/r3_new`, at `1b94bf0`.

The session scratch directory is outside the source repository. Run
single-threaded (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`;
with several processes, BLAS thread oversubscription made the TA shadow
step take over 25 minutes a date). Per clone, with cwd = that clone:

```text
PYTHONPATH=<clone>:<clone>/src python scripts/replay_monthly_decisions.py --committed-dates --out <scratch>/replay_{old,new}_code.csv
```

Both exited 0. Rows: [`R3_smoke_replay_rows.csv`](R3_smoke_replay_rows.csv).

| As-of | Old code: mode / sell | Old gates not passing | R3: mode / sell | R3 failed gates | Consensus |
|---|---|---|---|---|---|
| 2026-02-28 | DEFER / 50 % | directional_skill FAIL | DEFER / 50 % | directional_skill | UNDERPERFORM |
| 2026-03-31 | DEFER / 50 % | directional_skill FAIL | DEFER / 50 % | directional_skill | NEUTRAL |
| 2026-04-22 | DEFER / 50 % | directional_skill FAIL | DEFER / 50 % | directional_skill; data_ready | NEUTRAL |
| 2026-05-22 | DEFER / 50 % | directional_skill FAIL | DEFER / 50 % | directional_skill; data_ready | UNDERPERFORM |
| 2026-06-22 | DEFER / 50 % | directional_skill FAIL | DEFER / 50 % | directional_skill; data_ready | UNDERPERFORM |
| 2026-07-22 | DEFER / 50 % | mean_ic MARGINAL, directional_skill FAIL | DEFER / 50 % | directional_skill; data_ready | NEUTRAL |
| 2026-08-20 | DEFER / 50 % | oos_r2 MARGINAL, mean_ic MARGINAL, directional_skill FAIL | DEFER / 50 % | directional_skill; data_ready | NEUTRAL |
| 2026-09-21 | DEFER / 50 % | directional_skill FAIL | DEFER / 50 % | directional_skill; data_ready | UNDERPERFORM |

R3 metrics (identical to the old code's: the maximum absolute difference is
exactly 0 for every metric, forecast and probability column; the old
`cpcv_completed` passed every month):

| As-of | OOS R² | EW IC | Pooled IC (DK p) | Hit vs base | PT p | ECE | Coverage | Old CPCV | `wfo_completed` | Stale required feeds at the as-of date |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-02-28 | +3.03 % | 0.1007 | 0.1263 (0.049) | 64.1 % vs 70.0 % | 0.364 | 16.2 % | 49.0 % | 4/7 MARGINAL | true | none |
| 2026-03-31 | +5.65 % | 0.1058 | 0.1320 (0.083) | 63.8 % vs 69.6 % | 0.383 | 17.4 % | 44.8 % | 4/7 MARGINAL | true | none |
| 2026-04-22 | +5.65 % | 0.1058 | 0.1320 (0.083) | 63.8 % vs 69.6 % | 0.383 | 17.4 % | 44.8 % | 4/7 MARGINAL | true | Dividends VMBS, BND |
| 2026-05-22 | +3.05 % | 0.0879 | 0.1076 (0.159) | 63.1 % vs 69.4 % | 0.460 | 19.3 % | 42.7 % | 4/7 MARGINAL | true | Dividends VOO, VWO, VMBS, BND |
| 2026-06-22 | +3.52 % | 0.0730 | 0.0873 (0.245) | 61.6 % vs 68.8 % | 0.576 | 17.0 % | 40.6 % | 4/7 MARGINAL | true | Dividends VOO, VWO, VMBS, BND |
| 2026-07-22 | +5.13 % | 0.0696 | 0.0899 (0.240) | 61.7 % vs 68.3 % | 0.502 | 16.9 % | 42.7 % | 4/7 MARGINAL | true | Dividends VOO, VWO, VMBS, BND |
| 2026-08-20 | +1.21 % | 0.0487 | 0.0730 (0.292) | 60.5 % vs 68.2 % | 0.615 | 14.4 % | 44.8 % | 4/7 MARGINAL | true | Dividends VOO, VXUS, VWO, VMBS, BND, VDE |
| 2026-09-21 | +2.86 % | 0.0762 | 0.1019 (0.125) | 62.8 % vs 68.1 % | 0.391 | 15.4 % | 42.7 % | 4/7 MARGINAL | true | Dividends VOO, VXUS, VWO, VMBS, BND, VDE |

No month has a missing live feature.

**Attribution:**
- **Code change (this PR).** It changes no metric, no forecast, no
  consensus, no mode and no sell percentage in any month. The
  only differences are:
  - the CPCV no longer runs, so its 4/7 MARGINAL line disappears;
  - two readiness gates are added. `wfo_completed` passes in every month:
    all 16 pairs complete. `data_ready` fails from 2026-04-22 on, because
    required benchmark dividends are stale at each as-of date: the
    monthly payers VMBS and BND first, the quarterly payers from May.
    Every month already deferred on directional skill, so the added
    failure does not change a decision. Were the skill gate to pass, it
    would block an ACTIONABLE month computed from targets missing those
    dividends.
- **Data.** The metrics equal the verification's current-code replay
  (VERIFICATION_2026-09-26, same eight months; e.g. September R² +2.86 %,
  EW IC 0.0762, pooled 0.1019, hit 62.83 % vs 68.14 %), although that ran on
  DB `7c68efbd…`. The 2026-09-27 peer update touched only peer data. The
  change against the published step-5 table (September +5.08 %, 5/7 GOOD) is
  the step-6 filing-date timing repair, which the verification had already
  attributed (V08); R3 does not change it.
- **R2-lite (pending).** The refreshed dividends will change
  `benchmark_return` and `relative_return` for windows containing the
  missing ex-dates, and so the metrics. They should also clear `data_ready`.
  That replay must be recorded separately, against this table.

## Hashes (dry runs)

```text
before replay                                   after replay
source_db       f453ab98…51d                    f453ab98…51d
source_ledgers  013aa7e8…84a                    013aa7e8…84a   (decision_log, classifier and TA ledgers)
r3_old_db       f453ab98…51d                    f453ab98…51d
r3_new_db       f453ab98…51d                    f453ab98…51d
r3_old artifacts tree 4235d078…42d              4235d078…42d
```

`git status` in both clones was clean after the replay: no tracked artifact
or ledger was written. The `r3_new` artifacts-tree hash was taken before that
clone was moved to `1b94bf0`, whose N8 commit changes `decision_log.md`.
After the move it equals the source tree. The tracked DB's sha256 was
`f453ab98…51d` before and after every full-suite run below.

## Commands and results

| Command | Result | Exit |
|---|---|---|
| `python -m pytest -o addopts="--tb=short" -q` (before any change, `449fcde`) | `2494 passed, 1 skipped, 105 warnings in 592.01s (0:09:52)` | 0 |
| same, after the code change (first pass) | `7 failed, 2518 passed, 1 skipped, 102 warnings in 1603.84s (0:26:43)`: freshness tests that seed EDGAR rows without a filing date; fixed by `1b94bf0` | 1 |
| focused R3 command (`test_production_validation_contract.py`, `test_validation_gating.py`, `test_monthly_pipeline_e2e.py`, `test_dry_run_read_only.py`) | `88 passed, 1 warning in 131.76s (0:02:11)` | 0 |
| `python -m pytest -o addopts="--tb=short" -q` (final, at `1ee54e2`) | `2525 passed, 1 skipped, 104 warnings in 571.10s (0:09:31)` | 0 |
| `python -m pytest -o addopts="--tb=short" -q` after merging `master` `dc12291` (R1, v186) | `2547 passed, 1 skipped, 127 warnings in 580.94s (0:09:40)` | 0 |
| same, after merging `master` `981a60b` (R4, v189; `artifacts.py` auto-merged, CHANGELOG ordered v189 → v188) | `2563 passed, 1 skipped, 99 warnings in 588.93s (0:09:48)` | 0 |
| `python scripts/checks/check_doc_links.py` | `[doc-links] 348 files, 0 broken links` | 0 |
| `python scripts/checks/check_sys_path_edits.py` | `[sys-path] 0 new edits, 0 stale allowlist entries` | 0 |
| `ruff check .` | `All checks passed!` | 0 |
| `mypy` (CI's 11 hardened modules) / `mypy --follow-imports=silent src/pgr_vds cli` | `Success: no issues found in 11 source files` / `… in 21 source files` | 0 |

Import smoke: `python -c "import pgr_vds.decision.pipeline"` succeeds, and
`cli/monthly_decision.py` runs end to end in the 16 replayed dry runs.

## What is left

1. **After R2-lite merges:** replay 2026-02 → 2026-09 on the refreshed DB with
   this code. Record the data change against the table above, and replace the
   "current state" in `docs/model-governance.md` with the pinned
   refreshed-DB baseline. That replay is the v200 starting point; v200 creates
   its own research-protocol lock.
2. `scripts/verify_monthly_outputs.py` still checks `check_data_freshness`
   without dividends. The gate itself covers them. Not changed here (out of
   scope).
3. A back-dated readiness check cannot prove what an original decision had,
   because the DB has no per-row fetch timestamps. It is labelled, not solved.
4. R4 (v189) merged before this PR; its TA-maturity change to
   `src/pgr_vds/decision/artifacts.py` merged cleanly with R3's decision-log
   and manifest changes.

Passing tests do not establish an investment-performance improvement.
