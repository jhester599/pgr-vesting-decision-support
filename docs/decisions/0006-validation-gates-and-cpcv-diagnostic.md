# 0006 — Recommendation-mode gates, realised-only health, CPCV as a diagnostic (review 2026-09-25, step 5)

| | |
|---|---|
| **Status** | Accepted, live since v178 |
| **Date** | 2026-09-26 |
| **Where it lives** | `src/reporting/decision_rendering.py` (gates); `src/models/prequential.py` (realised-only weights, alpha, calibration, conformal) |
| **Findings** | F02, F04, F13, F20 (missing CPCV), F21 of [`REPO_REVIEW_2026-09-25.md`](../reviews/REPO_REVIEW_2026-09-25.md) |
| **Report** | [`2026-09-25_step5_validation_gating.md`](../reviews/2026-09-25_step5_validation_gating.md) |

## Context

The review found that the gates deciding whether a month is ACTIONABLE were
measured with look-ahead or could not fail as intended:

- the OOS R² naive benchmark contained the target it was compared with (F04);
- CPCV could never pass, and is a combinatorial K-fold, which AGENTS.md
  prohibits for validation (F02);
- the reported health used weights, alpha, calibration and conformal
  intervals fitted in-sample (F13);
- a missing CPCV result failed open (F20), and confidence was a constant (F21).

## Decision

**Gates.** ACTIONABLE needs all four to pass; any failure gives
DEFER-TO-TAX-DEFAULT; otherwise MONITORING-ONLY. A missing input fails its
gate.

| Gate | Pass | Fail |
|------|------|------|
| OOS R² against the prevailing mean of realised targets (per benchmark, training history included) | ≥ 2 % | < 0 % |
| Equal-weight mean of the per-benchmark OOS ICs | ≥ 0.07 | < 0.03 |
| Directional skill: one-sided Pesaran–Timmermann p, Driscoll–Kraay by date | < 0.05 | ≥ 0.10 or undefined |
| Representative CPCV diagnostic ran | ran | missing or UNKNOWN |

**CPCV is diagnostic only.** It is reported in `diagnostic.md` (7 recombined
paths, purge 6 rows, embargo 2 rows, GOOD ≥ 5/7) and never gates. A run in
which it fails to produce paths is held back from ACTIONABLE (fail closed).

**Every reported health number is realised-only.** For each OOS month, the
Ridge/GBT weights, the shrinkage alpha ([0001](0001-v38-post-ensemble-shrinkage.md)),
the Platt calibrator (ECE) and the conformal interval (trailing coverage) use
only targets whose 6-month window had ended by that month.
`model_performance_log` rows carry `metrics_version`; rows before this change
are `pre-2026-09-25`, and the drift monitor compares only rows of one version.

Unchanged by this decision: the ACTIONABLE sell-percentage mapping (changed
next, [0007](0007-actionable-sell-mapping.md)), the feature sets and the model
family.

## Evidence: 2026-02 → 2026-09 replayed

Each committed production month was replayed on the corrected pipeline with
`scripts/replay_monthly_decisions.py --committed-dates`: read-only dry runs on
a copy of the DB, using today's data as far as each as-of date allows. Rows:
[`2026-09-25_step5_rebaseline_rows.csv`](../reviews/2026-09-25_step5_rebaseline_rows.csv).

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

The replays were dry runs and wrote nothing to the DB. The health baseline at
2026-09-21 that later months are compared against is in
[`docs/model-governance.md`](../model-governance.md#health-baseline-at-2026-09-21).

## Consequences

- The recommendation did not change (50 % every month), but the reported
  health did: aggregate OOS R² +5.08 % against the corrected naive (−2.16 %
  against the leaky one), prequential ECE 14.2 % (in-sample 0.6 %), trailing
  conformal coverage 40.6 % against an 80 % target.
- The equal-weight IC passes by 0.0006; directional skill is the gate that
  binds.
- From the next production run `model_performance_log` stores these metrics,
  tagged `metrics_version = prequential-2026-09-25`.
