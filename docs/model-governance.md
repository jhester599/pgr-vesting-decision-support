# Model Governance

## Purpose

This document defines the boundary between:

- production modeling behavior
- research and promotion work
- operational monitoring and stabilization

## Current Production Baseline

The live monthly workflow currently uses:

- the `v11.1` lean 2-model prediction stack (`Ridge + GBT`, v18 feature sets)
- post-ensemble shrinkage chosen by the `v38` rule, applied prequentially: the
  grid alpha with the lowest squared error over the realised OOS record,
  re-chosen every month (the fixed 0.50 is research-only since review
  2026-09-25, step 5)
- the promoted quality-weighted consensus as the live recommendation path for
  direction and forecast
- the recommendation-mode gates of review 2026-09-25, step 5 (below), gated on
  the equal-weight IC
- the equal-weight consensus retained only as a diagnostic comparison in
  `consensus_shadow.csv`

The monthly workflow is the only place where a model or consensus stack becomes
operational.

## Current Monitoring Baseline

The production monthly output now tracks:

- aggregate OOS R^2 against each benchmark's prevailing mean of the targets
  realised by each forecast date
- pooled IC with a Driscoll-Kraay p-value clustered by date; per-benchmark
  Newey-West IC
- hit rate against the base rate, with the Pesaran-Timmermann directional test
- prequential ECE and trailing conformal coverage
- the representative CPCV as a stability diagnostic (not a gate)
- pooled and per-benchmark Clark-West diagnostics
- benchmark-level quality exports in `benchmark_quality.csv`
- live-vs-equal-weight comparison in `consensus_shadow.csv`
- per-benchmark classifier detail in `classification_shadow.csv`
- shadow gate comparison in `decision_overlays.csv`
- append-only classifier history in `results/monthly_decisions/classification_shadow_history.csv`
- reporting-only TA replacement variant history in
  `results/monthly_decisions/ta_shadow_variant_history.csv`
- machine-readable top-level state in `monthly_summary.json`

## Recent Promotion Record

Recent completed cycles:

- `v37-v60`
  - established `v38` as the best conservative calibration baseline
- `v66-v73`
  - aligned monthly diagnostics to ensemble reconstruction
  - added Clark-West and benchmark-quality exports
  - identified `v72` quality-weighted consensus as the strongest next candidate
- `v74-v78`
  - promoted the quality-weighted consensus into production after shadow and
    holdout-style review
- `v79-v80`
  - restored post-promotion monthly artifact wiring and validated the promoted
    path on a real monthly rerun
- `v81-v86`
  - aligned workflow, email, dashboard, and docs to the promoted baseline
  - added `monthly_summary.json`
  - retired the visible equal-weight cross-check from primary surfaces while
    keeping the diagnostic artifact
- `v102-v117`
  - archived the April 11 repo-level peer reviews
  - added classifier shadow artifacts, history logging, and shadow gate overlays
  - hardened backdated `--as-of` target truncation for monthly simulations
  - ran bounded veto-style and permission-style classifier gate studies
  - kept classification as shadow-only pending longer monitoring and stricter promotion gates
- `v160-v169`
  - screened Alpha Vantage-style technical-analysis features under WFO-only
    research constraints
  - selected two reporting-only TA replacement classifier variants for
    prospective shadow monitoring
  - added a durable TA shadow history ledger and reusable monthly output
    verifier

Supporting plan documents:

- `docs/superpowers/plans/2026-04-10-v37-v60-results-summary.md`
- `docs/superpowers/plans/2026-04-10-v66-v73-calibration-and-decision-layer.md`
- `docs/superpowers/plans/2026-04-10-v74-v78-quality-weighted-promotion.md`
- `docs/superpowers/plans/2026-04-11-v79-v80-post-promotion-stabilization.md`

## Validation and Gating (review 2026-09-25, step 5)

Findings F02, F04, F13, F20 (missing CPCV) and F21 of
`docs/reviews/REPO_REVIEW_2026-09-25.md`; full report in
`docs/reviews/2026-09-25_step5_validation_gating.md`.

**Recommendation-mode gates** (`src/reporting/decision_rendering.py`).
ACTIONABLE needs all four to pass; any failure gives DEFER-TO-TAX-DEFAULT;
otherwise MONITORING-ONLY. A missing input fails its gate.

| Gate | Pass | Fail |
|------|------|------|
| OOS R² against the prevailing mean of realised targets (per benchmark, training history included) | ≥ 2 % | < 0 % |
| Equal-weight mean of the per-benchmark OOS ICs | ≥ 0.07 | < 0.03 |
| Directional skill: one-sided Pesaran–Timmermann p, Driscoll–Kraay by date | < 0.05 | ≥ 0.10 or undefined |
| Representative CPCV diagnostic ran | ran | missing or UNKNOWN |

**CPCV is diagnostic only.** CombinatorialPurgedCV trains on folds after its
test folds: it is a combinatorial K-fold, which AGENTS.md prohibits for
validation. It is reported in `diagnostic.md` (7 recombined paths, purge
6 rows, embargo 2 rows, GOOD ≥ 5/7) and never gates. A run in which it fails
to produce paths is held back from ACTIONABLE (fail closed).

**Every reported health number is realised-only.** For each OOS month, the
Ridge/GBT weights, the shrinkage alpha, the Platt calibrator (ECE) and the
conformal interval (trailing coverage) use only targets whose 6-month window
had ended by that month (`src/models/prequential.py`). `model_performance_log`
rows carry `metrics_version`; rows before this step are `pre-2026-09-25`, and
the drift monitor compares only rows of one version.

**Unchanged in step 5:** the ACTIONABLE sell-percentage mapping (changed in
step 6), the feature sets and the model family.

### Baseline after step 5: 2026-02 → 2026-09 replayed

Each committed production month was replayed on the corrected pipeline with
`scripts/replay_monthly_decisions.py --committed-dates`: read-only dry runs on
a copy of the DB, using today's data as far as each as-of date allows. Rows:
`docs/reviews/2026-09-25_step5_rebaseline_rows.csv`.

| As-of | Committed run | Corrected mode | Sell % | Consensus (tier) | Gate not passing |
|---|---|---|---|---|---|
| 2026-02-28 | DEFER / 50 % | DEFER-TO-TAX-DEFAULT | 50 % | NEUTRAL (LOW) | directional skill FAIL, PT p 0.179 |
| 2026-03-31 | DEFER / 50 % | MONITORING-ONLY | 50 % | UNDERPERFORM (LOW) | directional skill MARGINAL, PT p 0.095 |
| 2026-04-22 | DEFER / 50 % | MONITORING-ONLY | 50 % | UNDERPERFORM (LOW) | directional skill MARGINAL, PT p 0.095 |
| 2026-05-22 | DEFER / 50 % | DEFER-TO-TAX-DEFAULT | 50 % | UNDERPERFORM (LOW) | directional skill FAIL, PT p 0.213 |
| 2026-06-22 | DEFER / 50 % | DEFER-TO-TAX-DEFAULT | 50 % | UNDERPERFORM (LOW) | directional skill FAIL, PT p 0.350 |
| 2026-07-22 | DEFER / 50 % | DEFER-TO-TAX-DEFAULT | 50 % | NEUTRAL (LOW) | directional skill FAIL, PT p 0.224 |
| 2026-08-20 | DEFER / 50 % | DEFER-TO-TAX-DEFAULT | 50 % | UNDERPERFORM (LOW) | directional skill FAIL, PT p 0.345 |
| 2026-09-21 | DEFER / 50 % | DEFER-TO-TAX-DEFAULT | 50 % | NEUTRAL (LOW) | directional skill FAIL, PT p 0.255 |

No month reaches ACTIONABLE, and every month sells the 50 % tax default.
Directional skill is the only gate that does not pass: the ensemble calls the
sign right 62.7–65.4 % of the time, below the 68.1–70.0 % of always calling
"PGR outperforms". Every month passes the other three gates (OOS R² +3.8 % to
+7.6 %, equal-weight IC 0.071–0.114, CPCV ran).

Health baseline at 2026-09-21, for later months to be compared against:

| Metric | Value | Gate |
|---|---|---|
| Aggregate OOS R² (vs prevailing mean) | +5.08 % (was −2.16 % against the leaky naive) | PASS |
| Equal-weight mean IC | 0.0706 (quality-weighted 0.0855) | PASS, by 0.0006 |
| Pooled rank IC, Driscoll–Kraay p | 0.130, p 0.030 | — |
| Hit rate vs base rate, Pesaran–Timmermann p | 62.8 % vs 68.1 %, p 0.255 | FAIL |
| Prequential ECE | 14.2 % (in-sample 0.6 %) | — |
| Trailing conformal coverage (target 80 %) | 40.6 % | — |
| CPCV positive paths (diagnostic) | 5/7, GOOD | ran |
| Shrinkage alpha (prequential) | 0.50 | — |

The replays were dry runs and wrote nothing to the DB. From the next
production run, `model_performance_log` stores these corrected metrics, tagged
`metrics_version = prequential-2026-09-25`. The full comparison with the
committed runs and with a replay of `master` is in
`docs/reviews/2026-09-25_step5_validation_gating.md`.

## Research Candidates Still Worth Tracking

The following remain promising but are not live:

- `v70`
  - per-benchmark shrinkage calibration
- `v46`
  - classification / directional sidecar
- `v73`
  - hybrid decision-gating design
- `v110-v113`
  - constrained classifier overlay candidates, with Gemini-style veto gating
    currently the strongest shadow candidate
- `v165-v169`
  - TA replacement classifier variants, monitored only through monthly shadow
    artifacts and the TA ledger until enough 6M horizons mature

These remain research-only until they clear a later promotion study.

## Promotion Rule

Research results do not become production behavior automatically.

A candidate should only be promoted when it demonstrates:

- better policy-level utility than the current production baseline
- acceptable aggregate model health
- stable behavior across the repo's existing time-series validation framework
- acceptable operational complexity and maintainability
- clear documentation of the change and its reporting consequences
- acceptable agreement and churn versus the current live recommendation path
- no material calibration drift once sufficient matured classifier history exists

## Research vs. Production Labels

- Production:
  - scheduled workflow code
  - monthly decision generation
  - committed monthly output artifacts
- Research:
  - `src/research/`
  - `results/research/`
  - versioned plan and summary documents for candidate studies
- Provisional:
  - live-vs-shadow observability paths kept temporarily after a promotion

## Current Governance Conclusion

The current production path is the quality-weighted consensus.

Since review 2026-09-25, step 5, the corrected gates hold every month from
2026-02 to 2026-09 at the 50 % tax default, and directional skill is the gate
that binds. Step 6 changed the
ACTIONABLE sell-percentage mapping (below).

### ACTIONABLE sell-% mapping (review 2026-09-25, step 6)

`decision_rendering.sell_pct_from_consensus`, used only when every gate
passes:

| Consensus | Mean forecast | Sell % |
|---|---|---|
| OUTPERFORM | > 15 % | 25 % |
| OUTPERFORM | ≤ 15 % | 50 % (was 75 % at ≤ 5 %) |
| UNDERPERFORM | any | 100 % |
| NEUTRAL, or IC < 0.05 / missing | any | 50 % |

A bullish consensus never sells more than the 50 % default. The live
consensus and mapping were replayed at 186 OOS dates (2010-09 → 2026-02) of
the realised-only record as of 2026-09-21
(`src/models/live_policy_backtest.py`, fixture
`tests/fixtures/live_mapping_oos_panel_2026-09-21.csv`), as if the gate had
passed every month. Mean relative return kept per decision: old mapping
+3.53 %, new +3.81 %, always-50 % +3.64 %. The uplift over always-50 % is
+0.17 pp per decision (was −0.11 pp). `tests/test_live_mapping_wp8.py`
requires it to stay ≥ 0; the monthly "Decision Policy Backtest" now scores
this mapping too. The UNDERPERFORM → 100 % cell rests on one historical
decision (it lost 7.2 pp); it is kept, and the regression test will catch it
if it starts to cost money.

The most immediate governance questions are now:

- whether `monthly_summary.json` should become the default contract for future
  automation and notification surfaces
- when the diagnostic-only equal-weight comparison can be further de-emphasized
  or archived
- whether the shadow-only classifier overlay remains stable enough to justify a
  later limited production gate candidate
