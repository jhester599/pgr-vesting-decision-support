# Review 2026-09-25, step 5 — validation and gating (WP7)

Findings F02, F04, F13, the missing-CPCV part of F20, and F21 in
[`REPO_REVIEW_2026-09-25.md`](REPO_REVIEW_2026-09-25.md). The ACTIONABLE
sell-percentage mapping is not changed here (step 6).

## What changed

- **Gate.** ACTIONABLE needs all four gates to pass: OOS R² ≥ 2 % against a
  look-ahead-free naive, equal-weight mean IC ≥ 0.07, one-sided
  Pesaran–Timmermann p < 0.05, and a CPCV diagnostic that ran. Any failure
  gives DEFER-TO-TAX-DEFAULT; otherwise the mode is MONITORING-ONLY. A missing
  input fails its gate.
- **CPCV is diagnostic only (F02).** It trains on folds after its test folds,
  so it is a combinatorial K-fold, which AGENTS.md prohibits for validation.
  - The 28 splits are now recombined into the 7 real paths, each scoring
    every row once. The old loop produced one "path" per fold.
  - Thresholds scale to the path count (GOOD ≥ 5/7, MARGINAL ≥ 3/7).
  - `embargo_size` = 2 after the 6-row purge.
  - A missing or UNKNOWN result fails closed (F20); the verdict itself never
    gates.
- **OOS R² (F04).** The naive at t is the mean of that benchmark's targets whose
  6-month window had ended by t, training history included. The old naive was
  the expanding mean of the pooled OOS series *including the target it was
  scored against*. Pooled R² sums each benchmark's errors against its own
  naive. The corrected value is what `model_performance_log.aggregate_oos_r2`
  stores from the next run.
- **Realised-only OOS record (F13).** Every OOS month uses only targets
  realised by then:
  - Ridge/GBT inverse-MAE weights;
  - the shrinkage alpha (v38's grid rule; the fixed 0.50 is research-only);
  - the Platt calibrator behind the reported ECE;
  - the conformal interval behind the trailing coverage.
  The IC gate uses equal weights, the hit rate is tested against the base
  rate (Pesaran–Timmermann), and pooled significance is Driscoll–Kraay by date.
- **Confidence tiers (F21)** come from the calibrated P(outperform), read in
  the signal's direction. They were LOW for every benchmark and month (the
  retired BayesianRidge posterior made P = 0.5).
- **Health log.** Rows carry a `metrics_version` (migration 008); the drift
  monitor compares only one version and no longer lets a back-dated run see
  later months.

## Tests: failing before the fix, passing after

`tests/test_validation_gating_wp7.py` holds the tests the review names plus
the unit tests written with the fix. The same file was run against unfixed
`master` (c089124) in a clean checkout: **29 failed, 1 passed**. The one pass
is a positive control (a genuinely skilled predictor must be able to reach
ACTIONABLE). On this branch all 30 pass.

The named tests fail on unfixed code by assertion, i.e. they catch the bug:

| Finding | Test | Unfixed code |
|---|---|---|
| F02 | `test_cpcv_recombines_seven_paths_each_covering_every_row_once` | 8 path ICs, one per fold |
| F02 | `test_cpcv_perfect_predictor_is_good` | perfect predictor: 8 of 7 "paths" positive, verdict FAIL |
| F02 | `test_cpcv_thresholds_scale_to_the_path_count` | 5/7 positive paths gives FAIL |
| F02 | `test_cpcv_purges_the_horizon_and_embargoes_at_least_two_rows` | `embargo_size` 0 |
| F02 | `test_cpcv_verdict_is_diagnostic_only` | a FAIL verdict forces DEFER |
| F04 | `test_oracle_forecaster_gets_positive_pooled_oos_r2` | oracle forecaster scores R² −0.100 |
| F04 | `test_compute_oos_r_squared_does_not_score_against_the_current_target` | the prevailing-mean forecaster scores −1.98 instead of 0 |
| F13 | `test_reported_ensemble_predictions_use_no_later_fold_statistics` | early OOS predictions change when later folds are added |
| F13 | `test_calibration_probabilities_use_no_later_fold_statistics` | early calibrated probabilities change when later folds are added |
| F13 | `test_always_positive_predictor_fails_hit_rate_gate` | an always-positive predictor on a 68 % base rate is ACTIONABLE |
| F13 | `test_gate_uses_the_equal_weight_ic` | a 0.095 quality-weighted IC rescues a 0.06 equal-weight IC |
| F13 | `test_pooled_ic_significance_is_clustered_by_date` | pooled p = 3.3e-20 by row against 4.5e-10 by date |
| F13 | `test_monthly_signals_no_longer_report_in_sample_conformal_coverage` | in-sample coverage column present |
| F20 | `test_missing_or_unknown_cpcv_does_not_permit_actionable` (2 cases) | ACTIONABLE with CPCV missing or UNKNOWN |
| F21 | `test_confidence_tiers_are_not_all_identical` | every tier LOW |
| — | `test_backdated_health_snapshot_ignores_later_months` | a 2026-02 run summarises drift up to 2026-09 |

The other failures on unfixed code are missing APIs (import or keyword
errors): the naive-benchmark and conformal realised-only tests, migration 008
and the metrics-version tests, and the unit tests of the grid shrinkage, the
PT edge cases, the prequential Platt fit and the date-block bootstrap.

Existing tests changed because they encoded the old behaviour:

- `test_oos_r2.py::test_historical_mean_prediction_gives_zero` treated the
  expanding mean *including the current target* as the naive; it now uses the
  realised-only prevailing mean and scores exactly 0. Three tests were added
  (a peeking forecast, 6-month overlap, training history).
- `test_v50_ensemble.py`: the reconstruction test asserted full-history
  weights and the fixed alpha; it now checks equal weights before any error
  is realised, weights from realised errors only, and the fixed-alpha
  override. The weighting tests pass an explicit alpha. The "C(8,2) = 28
  paths" config test asserted the F02 mistake; it now checks 28 splits and
  7 paths against skfolio.
- `test_v813_recommendation_mode.py`: ACTIONABLE now also needs the
  directional-skill gate; a test checks that 58 % hit rate with PT p = 0.40
  defers.
- `test_v29_research.py`: the confidence snapshot rows are the new gates.
- `test_cpcv.py`: the fake CV objects gained skfolio's `test_set_index` and
  `recombined_paths` shapes; the tests' intent is unchanged.
- `test_migration_runner.py`: expects migration 008.
- `test_monthly_logging.py`, `test_monthly_pipeline_e2e.py`: stubs accept the
  new `panel` keyword.

## How the months were replayed

`scripts/replay_monthly_decisions.py --committed-dates` runs
`monthly_decision.py --dry-run --as-of <date> --skip-fred` for the as-of date
of every committed production month and collects the dry-run artifacts. It
checks that the DB's sha256 is unchanged after each run.

- **Master replay:** a clean export of `master` at c089124 (steps 1–4
  merged; old validation).
- **Corrected:** this branch.

Each checkout had its own copy of the committed DB (sha256 `cadca467…3df5`,
before migration 008; the migration touches only `model_performance_log`).
Every sha256 was unchanged after its 8 dry runs.

The replays use today's data as far as each as-of date allows: features up to
that date, and targets whose window ended by it. They are not the data the
original runs saw. Steps 2–4 rebuilt targets, FRED and EDGAR, so the
"committed" column differs from both replays for data reasons alone. The
2026-03-31 and 2026-04-22 replays are identical for the same reason: both use
the 2026-03-31 feature row, and no target window ends between the two dates.
The committed runs differ because they ran on the data of the time.

## Re-baseline: 2026-02 → 2026-09

On the corrected pipeline the recommendation would have been **sell 50 % in
every month**: DEFER-TO-TAX-DEFAULT in six months and MONITORING-ONLY in March
and April. No month reaches ACTIONABLE. The committed runs and the master
replay were DEFER / 50 % in every month.

| As-of | Committed run (at the time) | Master replay (today's data, old validation) | Corrected (this step) | Corrected consensus | Gates not passing |
|---|---|---|---|---|---|
| 2026-02-28 | DEFER / 50 % (UNDERPERFORM) | DEFER / 50 % (NEUTRAL) | **DEFER / 50 %** | NEUTRAL (LOW) | direction (PT) FAIL |
| 2026-03-31 | DEFER / 50 % (NEUTRAL) | DEFER / 50 % (UNDERPERFORM) | **MONITORING / 50 %** | UNDERPERFORM (LOW) | direction (PT) MARGINAL |
| 2026-04-22 | DEFER / 50 % (NEUTRAL) | DEFER / 50 % (UNDERPERFORM) | **MONITORING / 50 %** | UNDERPERFORM (LOW) | direction (PT) MARGINAL |
| 2026-05-22 | DEFER / 50 % (NEUTRAL) | DEFER / 50 % (UNDERPERFORM) | **DEFER / 50 %** | UNDERPERFORM (LOW) | direction (PT) FAIL |
| 2026-06-22 | DEFER / 50 % (NEUTRAL) | DEFER / 50 % (UNDERPERFORM) | **DEFER / 50 %** | UNDERPERFORM (LOW) | direction (PT) FAIL |
| 2026-07-22 | DEFER / 50 % (NEUTRAL) | DEFER / 50 % (NEUTRAL) | **DEFER / 50 %** | NEUTRAL (LOW) | direction (PT) FAIL |
| 2026-08-20 | DEFER / 50 % (UNDERPERFORM) | DEFER / 50 % (UNDERPERFORM) | **DEFER / 50 %** | UNDERPERFORM (LOW) | direction (PT) FAIL |
| 2026-09-21 | DEFER / 50 % (NEUTRAL) | DEFER / 50 % (NEUTRAL) | **DEFER / 50 %** | NEUTRAL (LOW) | direction (PT) FAIL |

Directional skill is the gate that binds. Every month clears the other three:
- OOS R² is +3.8 % to +7.6 %.
- The equal-weight IC is 0.071–0.114; September passes by 0.0006.
- CPCV ran every month.

The ensemble calls the sign right 62.7–65.4 % of the time. That is below the
68.1–70.0 % scored by always calling "PGR outperforms", and the
Pesaran–Timmermann p-value is 0.095 (marginal: March, April) to 0.350.
Under the old gate, the 55 % hit-rate threshold passed on the base rate
alone, while the leaky R² and the always-FAIL CPCV forced DEFER.

### Health metrics (master replay → corrected)

| As-of | OOS R² old → new | EW IC (gate) | QW IC | Pooled IC (p by row) → (p by date) | Hit rate vs base rate | PT p | ECE in-sample → prequential | Trailing coverage | CPCV old → new | α |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-02-28 | -3.26 % → +6.81 % | 0.1046 | 0.1081 | 0.174 (p < 1e-4) → 0.161 (p 0.003) | 64.3 % vs 70.0 % | 0.179 | 1.1 % → 14.6 % | 44.8 % | 6/7 (75.0%) ❌ → 4/7 MARGINAL | 0.50 |
| 2026-03-31 | -1.79 % → +7.61 % | 0.1141 | 0.1193 | 0.178 (p < 1e-4) → 0.164 (p 0.008) | 65.4 % vs 69.6 % | 0.095 | 2.1 % → 16.7 % | 44.8 % | 7/7 (87.5%) ❌ → 4/7 MARGINAL | 0.50 |
| 2026-04-22 | -1.79 % → +7.61 % | 0.1141 | 0.1193 | 0.178 (p < 1e-4) → 0.164 (p 0.008) | 65.4 % vs 69.6 % | 0.095 | 2.1 % → 16.7 % | 44.8 % | 7/7 (87.5%) ❌ → 4/7 MARGINAL | 0.50 |
| 2026-05-22 | -2.93 % → +6.39 % | 0.0923 | 0.0998 | 0.145 (p 0.0007) → 0.130 (p 0.060) | 63.9 % vs 69.4 % | 0.213 | 1.2 % → 18.7 % | 35.4 % | 6/7 (75.0%) ❌ → 4/7 MARGINAL | 0.50 |
| 2026-06-22 | -1.95 % → +3.76 % | 0.0907 | 0.0977 | 0.163 (p 0.0001) → 0.145 (p 0.028) | 62.9 % vs 68.8 % | 0.350 | 0.8 % → 16.5 % | 36.5 % | 7/7 (87.5%) ❌ → 4/7 MARGINAL | 0.50 |
| 2026-07-22 | -1.78 % → +3.97 % | 0.0762 | 0.0859 | 0.147 (p 0.0003) → 0.137 (p 0.042) | 63.4 % vs 68.3 % | 0.224 | 0.9 % → 17.5 % | 36.5 % | 7/7 (87.5%) ❌ → 5/7 GOOD | 0.50 |
| 2026-08-20 | -1.85 % → +4.96 % | 0.0876 | 0.0966 | 0.158 (p < 1e-4) → 0.146 (p 0.019) | 62.7 % vs 68.2 % | 0.345 | 3.7 % → 15.4 % | 31.2 % | 7/7 (87.5%) ❌ → 6/7 GOOD | 0.50 |
| 2026-09-21 | -2.16 % → +5.08 % | 0.0706 | 0.0855 | 0.142 (p 0.0002) → 0.130 (p 0.030) | 62.8 % vs 68.1 % | 0.255 | 0.6 % → 14.2 % | 40.6 % | 7/7 (87.5%) ❌ → 5/7 GOOD | 0.50 |

- **OOS R² changes sign.** Against a look-ahead-free naive the model beats the
  prevailing mean in every month (+3.8 % to +7.6 %). The review's +10.8 % kept
  the in-sample ensemble weights and alpha; choosing both prequentially costs
  a few points.
- **Pooled IC significance is weaker.** Clustering by date gives p =
  0.003–0.060; by row it was below 1e-4. The pooled IC is ranked on the
  pre-shrinkage score (0.130–0.164); the old reconstruction's 0.142–0.178
  used full-history weights.
- **ECE is about 15 %, not 1–4 %.** The old ECE was measured on the rows each
  calibrator was fitted to.
- **Trailing conformal coverage is 31–45 % against an 80 % target.** Every
  run carries the manifest warning. The live intervals are calibrated on the
  whole OOS history and are too narrow for recent errors.
- **CPCV.**
  - Correct recombination gives 4/7 positive paths (MARGINAL) from February to
    June and 5–6/7 (GOOD) from July.
  - The old report printed "7/7 (87.5 %) ❌": 7 of 8 per-fold ICs, shown
    against 7 paths and failed against the unscaled 19/28 threshold.
- **Shrinkage.** The prequential alpha is 0.50 in every replayed month.
  September's continuous least-squares value is 0.54, and 0.50 is the nearest
  grid point. Along September's OOS history it was 0.05–0.15 in 2013–14, when
  the early realised record had little value, and 0.50–1.00 from 2016.
- **Confidence tiers vary.** Across the 64 benchmark-months, 49 are LOW,
  14 HIGH and 1 MODERATE (they were all LOW). The consensus tier is LOW in
  every month: the UNDERPERFORM consensus months have a calibrated
  P(outperform) of about 60 %, which does not support the call.

## Judgement calls

These are choices the brief left open, and why.

- **Hit-rate gate: Pesaran–Timmermann, not a margin over the base rate.** Both
  were allowed. A fixed margin over max(p, 1 − p) needs an arbitrary number
  of points, and it penalises a model that calls both directions with real
  skill. PT asks whether the up/down calls are associated with the outcome
  beyond what the two marginal rates imply, so an always-positive predictor
  scores exactly zero. The regression form with Driscoll–Kraay errors
  (Pesaran and Timmermann 2009) handles the overlap and the shared PGR leg.
  The bands (p < 0.05 pass, < 0.10 marginal) are the report's existing
  significance bands. The hit rate and the base rate are still reported.
- **A missing CPCV gives DEFER, not MONITORING.** The brief requires only that
  it cannot give ACTIONABLE; the review's fix says "treat UNKNOWN as DEFER".
  The mode's summary then says the validation run is incomplete rather than
  that the model is weak. Both modes sell 50 %.
- **Shrinkage: v38's rule, prequentially.** v38 picked alpha from a grid
  (0.05 … 1.00) by pooled R², i.e. by squared error. The prequential version
  applies the same rule to the rows realised at each date. A first attempt
  used the continuous least-squares scale clipped to [0, 1]. It set alpha to 0
  for 99 of 1,224 rows in 2012–14, when the early realised record had
  sum(z·y) ≤ 0, so those months would have issued no forecast at all. v38
  never allowed 0.
- **The rank IC uses the ensemble score before shrinkage.** Rescaling
  forecasts date by date with a time-varying alpha changes their pooled ranks
  without changing the model's ordering. On the September panel the pooled
  IC is 0.142 with the old in-sample weights and fixed alpha, 0.130 with
  prequential weights, and 0.078 when ranked on the shrunk forecasts. OOS R²,
  Clark–West and the conformal intervals use the issued (shrunk) forecast.
  v38 itself relied on IC being invariant to the scale.
- **Hit rate over rows with a call.** A zero forecast makes no call, so it is
  left out of the hit rate and the base rate as well as the PT test. With the
  grid there are no zero forecasts; the rule only guards the edge case.
- **Confidence tiers read in the signal's direction.** A calibrated P(outperform)
  of 62 % supports OUTPERFORM (MODERATE) but not UNDERPERFORM (LOW). The old
  symmetric thresholds would have labelled "UNDERPERFORM (MODERATE)" when the
  model's own calibrated probability favoured outperformance. NEUTRAL signals
  are LOW.
- **History is tagged, not rewritten.** The 8 stored `model_performance_log`
  rows keep what those runs reported and are tagged `pre-2026-09-25`
  (migration 008). The drift monitor compares only rows of one version, so its
  history restarts with the next production run. The corrected values for
  those months are in this report and the rows CSV, not in the DB.

## What is left

- **Step 6 (live mapping).** The ACTIONABLE sell-% mapping is unchanged and
  still unvalidated (F20): it sells 75 % on an OUTPERFORM consensus with a
  forecast ≤ 5 %, and it lost to "always sell 50 %" in the review's backtest.
  It matters again as soon as a month passes every gate.
- **Shrinkage target.** v38 shrinks toward zero, while R² is now scored
  against the prevailing mean, which is positive (PGR beat the benchmarks in
  about 68 % of windows). Shrinking toward the prevailing mean is a model
  change and belongs to a research step, not this fix.
- **Research re-baseline (WP11).** Two live settings were chosen on in-sample
  OOS statistics: the fixed v38 alpha, and the v72–v78 promotion of
  quality-weighted consensus over equal weights. Research harnesses calling
  `reconstruct_ensemble_oos_predictions` now get realised-only weights and a
  per-benchmark prequential alpha by default, so re-running them will move
  their numbers.
- **Drift monitoring restarts.** The drift monitor has no corrected history
  until the next production run; its 3-month breach rule needs three new
  rows.
