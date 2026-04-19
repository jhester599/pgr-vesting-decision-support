# PGR Vesting Decision Support — Changelog

> For active development direction, see [ROADMAP.md](ROADMAP.md).

Day 1 = 2026-03-25 (initial price fetch). Day 2 = 2026-03-26 (dividend fetch +
afternoon bootstrap). Development starts Day 3.

## v170 (2026-04-19)

- Added `docs/README.md` as the active documentation map
- Labeled `docs/plans/` and `docs/results/` as legacy directories with README
  files while preserving old references
- Updated operator docs for the monthly output verifier, TA shadow ledger, and
  calendar-aware PGR EDGAR freshness
- Added `docs/repo-hygiene-review-2026-04-19.md` and `archive/README.md`
- Promoted the BL-01 and v159 local plan drafts into tracked historical plan
  docs with explicit closeout references
- Removed duplicate `.hypothesis/` ignore entry

## v169 (2026-04-19)

- Added reusable monthly postcondition verifier:
  `scripts/verify_monthly_outputs.py`
- The verifier checks required monthly artifacts, data freshness, TA shadow
  variants in `classification_shadow.csv`, and matching TA ledger rows
- Replaced inline monthly-decision workflow verification glue with the reusable
  script and GitHub step-summary output
- No model, feature, recommendation, or live decision behavior changed

## v168 (2026-04-18)

- Made `check_data_freshness` calendar-aware for PGR monthly EDGAR 8-K data:
  prior-month data is required only after a 25-day filing grace window
- Fixed false stale warnings before the monthly 8-K primary/fallback jobs have
  had a fair chance to collect the new period
- Updated the monthly 8-K workflow postcondition to enforce the same
  calendar-aware freshness rule and report expected month-end/status
- No model, feature, recommendation, or live decision behavior changed

## v167 (2026-04-18)

- Added durable TA shadow variant history ledger:
  `results/monthly_decisions/ta_shadow_variant_history.csv`
- Monthly runs now append/upsert reporting-only TA replacement variants by
  `as_of_date` and `variant`
- Ledger rows track forecast anchor, 6M maturity date, probability, stance,
  confidence tier, benchmark count, and future realized-outcome placeholders
- No production recommendation, live monthly decision, or classifier gate
  behavior changes

## v166 (2026-04-18)

- TA-03 monthly shadow artifact lane: `classification_shadow.csv` now carries
  reporting-only TA replacement variants alongside baseline and follow-on rows
- Added monthly TA payloads to `classification_shadow_variants` in
  `monthly_summary.json`
- Added `variant_label`, `feature_set`, and `reporting_only` columns to the
  classifier shadow CSV schema
- Weekly data workflow now verifies required TA shadow price coverage for
  `PGR` and `VWO`; no new Alpha Vantage schedule or API calls were needed
- No production recommendation, live monthly decision, or classifier gate
  behavior changes

## v165 (2026-04-18)

- TA-02 follow-through: added a research-only classification replacement shadow
  harness in `results/research/v165_ta_shadow_replacement_eval.py`
- Tested replacement-only variants that keep the lean classifier at 12 features
- Wrote benchmark, prediction-level, regime-slice, current-shadow, and candidate
  artifacts under `results/research/`
- Candidate outcome: `shadow_monitor`; strongest historical variant is
  `ta_minimal_plus_vwo_pct_b` (+0.0584 mean BA, -0.0656 mean Brier, 8/8
  positive benchmarks)
- No production recommendation, live monthly decision, or classifier gate
  behavior changes

## v164 (2026-04-18)

- TA-01 technical-analysis research scaffold: archived three Alpha Vantage TA
  reports and created `v160-v164` implementation plan
- New research-only feature factory: `src/research/v160_ta_features.py`
- New harnesses: `results/research/v162_ta_broad_screen.py` and
  `results/research/v163_ta_survivor_confirm.py`
- Ran v162/v163 empirical artifacts on the current database snapshot; no extra
  Alpha Vantage workflow scheduling was required
- Fixed v162 artifact shaping so duplicated baseline rows no longer inflate
  baseline-delta detail outputs
- Candidate outcome: `replacement_candidate` for a later shadow-only
  classification replacement plan
- No production config, live monthly decision, or shadow-reporting changes

## BL-01 (2026-04-18)

- Black-Litterman tau/risk_aversion Monte Carlo sweep: 5×5 grid × 50 scenarios
- Harness: `results/research/bl01_tau_sweep_eval.py`
- Candidate: `results/research/bl01_tau_candidate.json`
- Recommendation: keep_incumbent — tau=0.05, risk_aversion=2.5 retained (delta +0.009, threshold 0.05)
- Incumbent rank_corr=0.8643 already near-optimal; no config change warranted
- Side effect: added `risk_free_rate` parameter to `build_bl_weights` (production default unchanged)
- Next: CLS-03 (time-locked), CLS-01

## v158 (2026-04-17)

- Synthesis of v153-v157 classification and feature research cycle
- Sole winner: CLS-02 Firth logistic for VMBS (+0.0412) and BND (+0.0704)
- No-benefit: FEAT-02 WTI momentum, FEAT-01 USD momentum, FEAT-03 term premium diff
- See `results/research/v158_synthesis_summary.md` for full outcome table
- Next queue: v159 (Firth shadow integration) → BL-01 → CLS-03

## v157 (2026-04-17)

- FEAT-03: Term premium 3M differential evaluation across all 8 benchmarks
- Derived feature: `term_premium_diff_3m` (computed as `term_premium_10y.diff(3)`)
- Harness: `results/research/v157_term_premium_eval.py`
- Candidate: `results/research/v157_term_premium_candidate.json`
  - Source available: True
  - Term premium winners: [] (VOO delta=-0.0003, VXUS delta=+0.0000, VWO delta=+0.0006, VMBS delta=+0.0064, BND delta=-0.0875, GLD delta=-0.0151, DBC delta=+0.0051, VDE delta=+0.0169 — all below 0.02 threshold)
  - Recommendation: no_benefit
- No production config changes; derived feature computed in harness only

## v156 (2026-04-17)

- FEAT-01: USD index momentum evaluation for BND/VXUS/VWO classifiers
- Features tested: `usd_broad_return_3m`, `usd_momentum_6m`
- Harness: `results/research/v156_usd_momentum_eval.py`
- Candidate: `results/research/v156_usd_candidate.json`
  - USD features available: ['usd_broad_return_3m', 'usd_momentum_6m']
  - USD winners: [] (BND delta=-0.0767, VXUS delta=+0.0000, VWO delta=+0.0087 — all below 0.03 threshold)
  - Recommendation: no_benefit
- No production config changes

## v155 (2026-04-17)

- FEAT-02: WTI 3M momentum evaluation for DBC/VDE classifiers
- Harness: `results/research/v155_wti_momentum_eval.py`
- Feature tested: `wti_return_3m` (macro_rates_spreads family)
- Candidate: `results/research/v155_wti_candidate.json`
  - WTI winners: [] (DBC delta=+0.0051, VDE delta=+0.0206 — both below 0.04 threshold)
  - Recommendation: no_benefit
- No production config changes

## v154 (2026-04-17)

- CLS-02: Firth-penalized logistic research harness for short-history benchmarks
- Implemented IRLS Firth logistic in `src/research/v154_utils.py`
- Harness: `results/research/v154_firth_logistic_eval.py`
- Candidate: `results/research/v154_firth_candidate.json`
  - Firth winners: VMBS (+0.0412 BA), BND (+0.0704 BA)
  - Recommendation: adopt_firth_for_thin_benchmarks
- No production config changes

## v153 (2026-04-17)

- Archived 2026-04-17 ChatGPT repo peer review under
  `docs/archive/history/repo-peer-reviews/2026-04-17/`
- Added FEAT-03 (term premium 3M diff) to backlog
- Updated backlog queue: CLS-02 → FEAT-02 → FEAT-01 → FEAT-03 → BL-01
- Noted priority shift: classification-first per 2026-04-17 peer review
  (BL-01 deferred until after v158 synthesis)
- Created plan: `docs/superpowers/plans/2026-04-17-v153-v158-classification-feature-research.md`

## Version History

### v152 (docs complete)
**Released:** 2026-04-17
**Theme:** Final Synthesis And Handoff For The v139-v152 Follow-On Cycle

- Closed the `v139-v152` follow-on arc with one final synthesis pass across the
  bounded research harnesses, the candidate files, and the `v151` side-by-side
  shadow promotion
- Recorded the durable winners from the cycle as:
  `v141_blend_weight_candidate.txt = 0.60`,
  `v143_corr_prune_candidate.txt = 0.80`,
  `v144_conformal_candidate.json = {"coverage": 0.75, "aci_gamma": 0.03}`,
  and `v149_kelly_candidate.json = {"fraction": 0.50, "cap": 0.25}`
- Confirmed that the `v151` reporting-only shadow lane
  `autoresearch_followon_v150` is the only promotion outcome from this cycle;
  the live monthly recommendation path is still unchanged
- Marked the bounded no-change confirmations explicitly for `v140`, `v142`,
  `v145`, `v146`, `v147`, `v148`, and `v150`, so the next session does not
  waste time re-running settled follow-through work
- Added the final handoff note in
  `docs/closeouts/V152_CLOSEOUT_AND_HANDOFF.md` and refreshed the roadmap,
  backlog, and follow-on plan so the next autonomous session can start from the
  ranked post-merge backlog instead of reopening this cycle

---

### v151 (shadow/docs complete)
**Released:** 2026-04-17
**Theme:** Side-By-Side Shadow Promotion For The v139-v150 Follow-On Winners

- Added `src/reporting/shadow_followon.py` to load the surviving
  `v139-v150` candidate files and build one reporting-only
  `autoresearch_followon_v150` shadow payload
- `scripts/monthly_decision.py` now threads one additive follow-on shadow lane
  through the monthly reporting pipeline while leaving live recommendation
  behavior unchanged
- `src/reporting/monthly_summary.py` now preserves
  `classification_shadow_variants` and `decision_overlay_variants` alongside
  the existing baseline-compatible fields
- `src/reporting/dashboard_snapshot.py` now renders a “Shadow Variant
  Comparison” section so the current shadow lane and the follow-on lane appear
  side-by-side in `dashboard.html`
- `src/reporting/classification_artifacts.py` now preserves a `variant` column
  in `classification_shadow.csv` and `decision_overlays.csv`
- Added focused pytest coverage for the follow-on helper plus the new variant
  reporting path

---

### v150 (research complete)
**Released:** 2026-04-16
**Theme:** Neutral-Band Replay Proxy Review On Top Of The Updated Kelly Baseline

- `results/research/v150_neutral_band_autoresearch_log.jsonl` and
  `results/research/v150_neutral_band_search_summary.md` now record the first
  bounded neutral-band review after the `v149` Kelly candidate update
- Re-ran the replay proxy against the updated Kelly baseline
  `{"fraction": 0.50, "cap": 0.25}` and tested neutral-band values
  `0.00`, `0.01`, `0.015`, `0.02`, `0.03`, `0.05`
- Utility score stayed flat at `0.0021` across the tested band values, so the
  decision was driven by the coverage/success-rate tradeoff rather than raw
  objective movement
- Wider bands such as `0.03` improved success rate to `0.8293`, but they cut
  coverage to `0.2531`; because the gains were mostly a selectivity tradeoff,
  `results/research/v150_neutral_band_candidate.txt` remains unchanged at
  `0.015` as the more stable research-only setting

---

### v149 (research complete)
**Released:** 2026-04-16
**Theme:** Kelly Fraction / Cap Replay-Proxy Bounded Sweep

- `results/research/v149_kelly_autoresearch_log.jsonl` and
  `results/research/v149_kelly_search_summary.md` now record the first bounded
  Kelly fraction/cap sweep on the preserved `v138` BL replay-proxy frame
- Established the incumbent candidate baseline at
  `{"fraction": 0.25, "cap": 0.20}` with
  `utility_score=0.0010`, `coverage=0.2531`, `success_rate=0.8293`
- Tested bounded alternatives:
  `(0.10, 0.10)`, `(0.15, 0.15)`, `(0.25, 0.20)`, `(0.35, 0.20)`,
  `(0.35, 0.25)`, `(0.50, 0.25)`
- `results/research/v149_kelly_candidate.json` is updated to
  `{"fraction": 0.50, "cap": 0.25}`, which delivered the best bounded utility
  score observed in the grid:
  `utility_score=0.0021`, `coverage=0.4506`, `success_rate=0.7671`
- This is a more aggressive research-only candidate and remains subject to a
  later promotion decision rather than automatic adoption

---

### v148 (research complete)
**Released:** 2026-04-16
**Theme:** Positive-Class Weight Replay-Proxy Review

- `results/research/v148_class_weight_autoresearch_log.jsonl` and
  `results/research/v148_class_weight_search_summary.md` now record the first
  bounded positive-class weighting review on the preserved Path B frame
- Re-established the incumbent candidate baseline at `positive_weight=1.00`
  with `covered_ba=0.6987`, `coverage=0.5476`
- Tested bounded alternatives `0.75`, `1.25`, `1.50`, and `2.00`
- None of the tested alternatives beat the incumbent on covered balanced
  accuracy, so `results/research/v148_class_weight_candidate.txt` remains
  unchanged at `1.0`

---

### v147 (research complete)
**Released:** 2026-04-16
**Theme:** Coverage-Weighted Path A / Path B Aggregate Proxy Review

- `results/research/v147_aggregate_autoresearch_log.jsonl` and
  `results/research/v147_aggregate_search_summary.md` now record the first
  bounded coverage-weighted Path A / Path B aggregation review
- Re-established the incumbent multiplier baseline at `1.0`, which produced
  `covered_ba=0.5000`, `coverage=0.4405`
- Tested bounded multipliers `0.50`, `0.75`, `1.25`, `1.50`, and `2.00`
- None of the tested multipliers improved covered balanced accuracy above the
  baseline value, so `results/research/v147_path_b_multiplier_candidate.txt`
  remains unchanged at `1.0`

---

### v146 (research complete)
**Released:** 2026-04-16
**Theme:** Threshold Follow-Through Review On The Tuned v135 Baseline

- `results/research/v146_threshold_autoresearch_log.jsonl` and
  `results/research/v146_threshold_search_summary.md` now record the first
  bounded threshold follow-through review on top of the tuned `v135` Path B
  temperature baseline
- Re-established the incumbent candidate baseline at
  `{"low": 0.15, "high": 0.70}` with
  `covered_ba=0.6987`, `coverage=0.5476`
- Tested bounded threshold pairs:
  `(0.10, 0.65)`, `(0.10, 0.70)`, `(0.15, 0.65)`, `(0.15, 0.70)`,
  `(0.20, 0.70)`, `(0.20, 0.75)`
- None of the tested pairs improved covered balanced accuracy over the current
  candidate while maintaining an obviously better tradeoff, so
  `results/research/v146_threshold_candidate.json` remains unchanged at
  `{"low": 0.15, "high": 0.70}`

---

### v145 (research complete)
**Released:** 2026-04-16
**Theme:** Bounded WFO Train/Test Window Review On The Current Ensemble Frame

- `results/research/v145_wfo_autoresearch_log.jsonl` and
  `results/research/v145_wfo_search_summary.md` now record the first bounded
  WFO train/test window review on the post-v138 ensemble frame
- Re-established the incumbent candidate baseline at
  `train=60`, `test=6` with
  `pooled_oos_r2=-0.1578`, `pooled_ic=0.1261`, `pooled_hit_rate=0.6906`
- Tested bounded alternatives:
  `(48,6)`, `(54,6)`, `(72,6)`, `(60,3)`, `(60,9)`
- `(48,6)` improved pooled OOS R^2 to `-0.1570` and pooled IC to `0.1856`,
  but it reduced pooled hit rate to `0.6535`; because no tested window pair
  improved the full headline set cleanly, `results/research/v145_wfo_candidate.json`
  remains unchanged at `{"train": 60, "test": 6}`

---

### v144 (research complete)
**Released:** 2026-04-16
**Theme:** Conformal Coverage / ACI Gamma Bounded Replay Tuning

- `results/research/v144_conformal_autoresearch_log.jsonl` and
  `results/research/v144_conformal_search_summary.md` now record the first
  bounded conformal replay grid on the pooled ensemble OOS frame
- Established the initial candidate baseline at
  `coverage=0.80`, `aci_gamma=0.05` with
  `coverage=0.7962`, `target_coverage=0.8000`, `coverage_gap=-0.0038`
- Best bounded candidate from the tested grid:
  `coverage=0.75`, `aci_gamma=0.03` ->
  `coverage=0.7490`, `target_coverage=0.7500`, `coverage_gap=-0.0010`
- `results/research/v144_conformal_candidate.json` is updated to
  `{"coverage": 0.75, "aci_gamma": 0.03}` as the cleanest bounded
  coverage-gap result so far on this replay harness

---

### v143 (research complete)
**Released:** 2026-04-16
**Theme:** Correlation-Pruned Feature-Override Bounded Sweep

- `results/research/v143_corr_prune_autoresearch_log.jsonl` and
  `results/research/v143_corr_prune_search_summary.md` now record the first
  bounded greedy correlation-pruning sweep on the current regression frame
- Re-established the incumbent pruning-threshold baseline at `rho=0.95`, which
  matched the no-prune pooled baseline:
  `pooled_oos_r2=-0.1578`, `pooled_ic=0.1261`, `pooled_hit_rate=0.6906`
- Best bounded candidate from the tested grid:
  `rho=0.80` ->
  `pooled_oos_r2=-0.1569`, `pooled_ic=0.1411`, `pooled_hit_rate=0.6944`
- `results/research/v143_corr_prune_candidate.txt` is updated to `0.80` as the
  first observed threshold that improved all three pooled headline metrics on
  the tested frame

---

### v142 (research complete)
**Released:** 2026-04-16
**Theme:** EDGAR Filing-Lag Bounded Review On The Post-v138 Regression Frame

- `results/research/v142_edgar_lag_autoresearch_log.jsonl` and
  `results/research/v142_edgar_lag_search_summary.md` now record the first
  bounded four-point lag review on the current 8-benchmark ensemble frame
- Re-established the post-v138 EDGAR baseline at
  `pooled_oos_r2=-0.1578`, `pooled_ic=0.1261`, `pooled_hit_rate=0.6906`
  with `lag=2`
- Sweep result:
  `lag=0 -> pooled_oos_r2=-0.1882`,
  `lag=1 -> pooled_oos_r2=-0.1638`,
  `lag=2 -> pooled_oos_r2=-0.1578`,
  `lag=3 -> pooled_oos_r2=-0.1808`
- `lag=1` improved IC to `0.1492`, but it weakened both pooled OOS R^2 and hit
  rate versus the current setting, so the research-only winner remains the
  existing `lag=2` candidate

---

### v141 (research complete)
**Released:** 2026-04-16
**Theme:** Fixed Ridge-vs-GBT Blend-Weight Sweep On The Current Ensemble Frame

- `results/research/v141_blend_weight_autoresearch_log.jsonl` and
  `results/research/v141_blend_weight_search_summary.md` now record the first
  bounded fixed-weight search around the current inverse-variance ensemble
  frame
- Established the fixed-weight midpoint baseline at
  `pooled_oos_r2=-0.1634`, `pooled_ic=0.1250`, `pooled_hit_rate=0.6906`
  with `ridge_weight=0.50`
- First bounded sweep tested `ridge_weight` values
  `0.30`, `0.40`, `0.50`, `0.55`, `0.60`, `0.65`, and `0.70`
- `results/research/v141_blend_weight_candidate.txt` is updated to `0.60`,
  which produced the best balanced improvement on the tested frame:
  `pooled_oos_r2=-0.1624`, `pooled_ic=0.1263`, `pooled_hit_rate=0.6935`
- The nearby `0.65` and `0.70` settings slightly improved hit rate and IC, but
  they gave back enough OOS R^2 that `0.60` remains the cleanest research-only
  winner for any later promotion decision

---

### v140 (research complete)
**Released:** 2026-04-16
**Theme:** Ensemble Shrinkage Alpha Re-Check On The Current Production Frame

- `results/research/v140_shrinkage_autoresearch_log.jsonl` and
  `results/research/v140_shrinkage_search_summary.md` now record the first
  bounded shrinkage re-check on the current 8-benchmark ensemble frame
- Re-established the post-v138 baseline at
  `pooled_oos_r2=-0.1578`, `pooled_ic=0.1261`, `pooled_hit_rate=0.6906`
  with `shrinkage=0.50`
- Sweep result: `shrinkage` values from `0.35` through `0.65` were
  operationally flat on the tested frame, producing no meaningful pooled metric
  separation
- Because the metric surface is flat,
  `results/research/v140_shrinkage_candidate.txt` stays at `0.50` and the
  block closes as a no-change confirmation rather than a promotion candidate

---

### v139 (scaffold complete)
**Released:** 2026-04-16
**Theme:** Autoresearch Follow-On Scaffolding For The Post-v138 Queue

- Created `docs/superpowers/plans/2026-04-16-v139-v152-autoresearch-followon.md`
  as the restart-safe follow-on plan for the remaining April 2026 research queue
- Added `src/research/v139_utils.py` to centralize benchmark loading, temporary
  config patching, pooled ensemble reconstruction, and feature-pruning helpers
  for the new follow-on research harnesses
- Added research-only follow-on harness scaffolding and candidate files:
  `v140_shrinkage_eval.py`, `v141_blend_eval.py`, `v142_edgar_lag_eval.py`,
  `v143_corr_prune_eval.py`, `v144_conformal_eval.py`,
  `v145_wfo_window_sweep.py`, `v146_threshold_sweep.py`,
  `v147_coverage_weighted_aggregate.py`, `v148_class_weight_eval.py`,
  `v149_kelly_eval.py`, and `v150_neutral_band_eval.py`
- Added matching pytest coverage for `v140` through `v150`, covering candidate
  file schemas, range guards, CLI-like failure behavior, and broad metric sanity
- Refreshed `ROADMAP.md` and `docs/research/backlog.md` so the documented active
  direction now starts from the actual `v138` repo state rather than the older
  pre-autoresearch classification arc

---

### v138 (research complete)
**Released:** 2026-04-14
**Theme:** Black-Litterman Tau / View-Confidence Replay Proxy + First Bounded Sweep

- `results/research/v138_bl_param_eval.py` - new Target 8 research harness
  that evaluates a bounded BL replay proxy on the matured
  `v118_prospective_shadow_replay_results.csv` frame
- `tests/test_research_v138_bl_param_eval.py` - new tests covering baseline
  sanity, parameter-range guards, and candidate-file parsing
- `results/research/v138_bl_params_candidate.json` - bounded-sweep winner
  candidate file updated to `{"tau": 0.05, "view_confidence_scalar": 0.75}`
- `results/research/v138_bl_param_autoresearch_log.jsonl` and
  `results/research/v138_bl_param_search_summary.md` - machine-readable sweep
  log and handoff summary
- Established replay-proxy baseline at `recommendation_accuracy=0.8500`,
  `coverage=0.1235`, `mean_kelly_fraction=0.0151`
- Best success-gate candidate from the bounded sweep:
  `tau=0.05`, `view_confidence_scalar=0.75` ->
  `recommendation_accuracy=0.8293`, `coverage=0.2531`,
  `mean_kelly_fraction=0.0200`, `policy_uplift=0.0010`
- Important caveat: the historical repo does not retain the original monthly BL
  covariance/view tensors, so `v138` is a replay proxy rather than a literal
  reconstruction of historical BL calls

---

### Runtime Optimization Pass (research complete)
**Released:** 2026-04-14
**Theme:** Test Suite Timing Harness + First Full-Suite Reduction Pass

- `scripts/measure_test_time.sh` - added the timing wrapper requested by
  Target 2; compatible with the repo's Git Bash path on Windows
- `tests/conftest.py` - added `--fast` support that skips tests marked
  `@pytest.mark.slow`
- `pytest.ini` - registered the `slow` marker
- Marked the 5 originally slowest tests as `slow` and narrowed several
  research smoke tests to smaller benchmark subsets while preserving their
  assertion contract
- `results/research/v_test_runtime_autoresearch_log.jsonl` and
  `results/research/v_test_runtime_summary.md` - recorded the baseline and
  post-pass timing measurements
- Measured full-suite runtime reduction from `131.0s` to `85.2s`
  (`35.0%` faster than baseline); `--fast` local iteration lane measures
  `70.2s`
- `docs/superpowers/plans/2026-04-13-autoresearch-execution-plan.md` now
  carries the initial slowest-10 test profile at the top of the plan for
  restart-safe follow-on work

---

### v137 (research complete)
**Released:** 2026-04-14
**Theme:** Standalone GBT Parameter Sweep Harness + First Bounded Search

- `results/research/v137_gbt_param_sweep.py` - new standalone GBT research
  harness for evaluating tree depth, boosting rounds, learning rate, and
  subsample on the production research frame
- `tests/test_research_v137_gbt_param_sweep.py` - new guardrail tests for the
  GBT parameter ranges and baseline smoke run
- `results/research/v137_gbt_params_candidate.json` - candidate params file
  updated to the best observed bounded-search configuration
- `results/research/v137_gbt_param_autoresearch_log.jsonl` and
  `results/research/v137_gbt_param_search_summary.md` - bounded-search log and
  summary
- Established standalone GBT baseline at `pooled_oos_r2=-0.4629`,
  `pooled_ic=0.1040`, `pooled_hit_rate=0.6485`
- Best bounded candidate so far:
  `max_depth=1`, `n_estimators=25`, `learning_rate=0.05`, `subsample=0.8`
  with `pooled_oos_r2=-0.2675`

---

### v136 (research complete)
**Released:** 2026-04-14
**Theme:** Backlog Prioritization Artifacts Refreshed After The Bounded Sweeps

- `docs/research/backlog.md` - new structured backlog seeded with the open
  research items referenced by the 2026-04-13 execution plan and refreshed
  with the current post-v129/v133/v134/v135/v137 state
- `docs/research/backlog_scoring_rubric.md` - scoring rubric used for the local
  predict-equivalent prioritization pass
- `results/research/v136_predict_output.json` - ranked backlog output with 10
  entries and 5 items at `consensus_score >= 7`
- Current top priorities after the bounded sweep set:
  `DATA-01`, `REG-01`, and `BL-01`

---

### v133 (research complete)
**Released:** 2026-04-14
**Theme:** Ridge Alpha Re-Baseline Harness + Bounded High-Alpha Sweep

- `config/model.py` - added `RIDGE_ALPHA_MIN`, `RIDGE_ALPHA_MAX`, and
  `RIDGE_ALPHA_N` constants for research harnesses
- `results/research/v133_ridge_alpha_sweep.py` - new ridge-only research
  harness for injecting explicit alpha grids into the production research frame
- `tests/test_research_v133_ridge_alpha_sweep.py` - new tests covering the
  default smoke path and alpha-grid validation
- `results/research/v133_alpha_max_candidate.txt` - current best bounded-sweep
  `alpha_max` candidate (`1000.0`)
- `results/research/v133_ridge_alpha_autoresearch_log.jsonl` and
  `results/research/v133_ridge_alpha_search_summary.md` - bounded-sweep log and
  summary
- Established current ridge-only baseline at `pooled_oos_r2=-0.5906`
  (`alpha_max=1e2`, 60-grid sweep); best bounded candidate improved this to
  `-0.4548` at `alpha_max=1e3`, still far below the historical v38 ensemble
  reference and therefore not a promotion candidate

---

### v134 (research complete)
**Released:** 2026-04-14
**Theme:** FRED Publication Lag Sweep Harness + First Bounded Search

- `docs/data/fred_publication_lag_reference.md` - new internal reference table
  documenting the intended publication-lag floor for each configured FRED
  series, separating the nine daily/weekly candidates from the slower monthly
  series that should not be reduced in autonomous search
- `results/research/v134_fred_lag_sweep.py` - new production-frame research
  harness for testing JSON lag overrides against the current 8-benchmark
  Ridge+GBT ensemble path and reporting pooled OOS R^2, IC, and hit rate
- `tests/test_research_v134_fred_lag_sweep.py` - new tests covering baseline
  recovery, invalid lag rejection, unknown-series rejection, and candidate JSON
  parsing
- `results/research/v134_lag_candidate.json` - autoresearch candidate file for
  the nine eligible daily/weekly series; updated after the first bounded sweep
  to the best observed candidate (`T10YIE -> 0`, all others remain `1`)
- `results/research/v134_fred_lag_autoresearch_log.jsonl` - machine-readable
  log of the bounded sweep across the baseline, all-zero candidate, and all
  single-toggle candidates
- `results/research/v134_fred_lag_search_summary.md` - summary artifact for the
  first completed sweep
- Established the live-state baseline for this harness at
  `pooled_oos_r2=-0.1578`, `pooled_ic=0.1261`, `pooled_hit_rate=0.6906`, which
  is weaker than the older plan's v38-era reference and therefore becomes the
  correct comparison point for future Target 4 work
- First bounded sweep result: only `T10YIE -> 0` improved R^2, and only
  marginally (`-0.1578` -> `-0.1573`), so there is no current evidence that
  broad lag reductions should be promoted into live config

---

### v129 (research complete)
**Released:** 2026-04-14
**Theme:** Re-scoped Benchmark Feature-Map Evaluation Harness

- `results/research/v129_feature_map_eval.py` - new re-scoped Target 1 harness
  built on the existing `v128` benchmark-WFO machinery because the current
  `v125` fold-detail artifact does not store the per-benchmark logits or
  coefficients that the original plan assumed
- `tests/test_research_v129_feature_map_eval.py` - new tests for the canonical
  v128 pooled baselines, candidate-map validation, and coverage-floor CLI
  behavior
- `results/research/v129_candidate_map.csv` - file-backed candidate map in the
  target schema used by the new harness
- Verified canonical built-in baselines:
  `lean_baseline -> covered_ba=0.5000, coverage=0.8700`,
  `v128_map -> covered_ba=0.5016, coverage=0.8891`
- Verified file-backed candidate replay on the re-scoped frame:
  `covered_ba=0.6164`, `coverage=0.6564`

---

### v135 (research complete)
**Released:** 2026-04-14
**Theme:** Path B Temperature Parameter Search Harness + First 80-Iteration Sweep

- `results/research/v135_temp_param_search.py` - new research harness for
  tuning the Path B prequential temperature grid upper bound and warmup window
  while holding the v131 asymmetric abstention pair fixed
  (`low=0.15`, `high=0.70`)
- `tests/test_research_v135_temp_param_search.py` - new math/guardrail tests
  covering baseline recovery, temperature-bound validation, warmup validation,
  log-grid construction, and CLI low-coverage exit behaviour
- `results/research/v135_temp_max_candidate.txt` and
  `results/research/v135_warmup_candidate.txt` - autoresearch candidate files
  updated to the current best observed configuration from the first bounded
  sweep: `temp_max=2.5`, `warmup=42`
- `results/research/v135_temp_param_autoresearch_log.jsonl` - machine-readable
  log of the first 80-point search grid
  (`10 temp_max candidates x 8 warmup candidates`, plus baseline row)
- `results/research/v135_temp_param_search_summary.md` - summary artifact for
  the first completed sweep
- Verified default baseline for the new harness:
  `covered_ba=0.6322`, `coverage=0.4524` at `(temp_max=3.0, warmup=24)`,
  which reflects the existing `v131` threshold pair rather than the older
  `v130` `(0.30, 0.70)` baseline of `0.5725`
- First bounded sweep result: best configuration
  `(temp_max=2.5, warmup=42)` achieved `covered_ba=0.6987`,
  `coverage=0.5476`, `brier=0.1589`; this clears the Target 5 success gate on
  the selection frame and requires future temporal hold-out validation before
  any config promotion

---

### v131 (complete)
**Released:** 2026-04-13
**Theme:** Temperature-Scaled Path B Composite Classifier Wired Into Production Shadow

- `src/models/path_b_classifier.py` — new module: composite return builder, Path B logistic
  classifier, and prequential temperature scaling utilities extracted from v125/v127/v130
  research scripts
- `src/models/classification_shadow.py` — `ClassificationShadowSummary` gains 4 new fields:
  `probability_path_b_temp_scaled`, `probability_path_b_temp_scaled_label`,
  `confidence_tier_path_b`, `stance_path_b`; `build_classification_shadow_summary()` computes
  Path B live using the adopted temperature scaling from v130
- `scripts/monthly_decision.py` — monthly `recommendation.md` now shows Path B
  `P(Actionable Sell)` alongside the portfolio-aligned investable-pool signal
- `src/reporting/dashboard_snapshot.py` — Classification Confidence Check section adds
  Portfolio-aligned and Path B probability cards
- Monthly artifacts (2026-02, 2026-03, 2026-04) refreshed with new fields
- `tests/test_path_b_classifier.py` — unit tests for the new classifier module
- `tests/test_classification_shadow.py` — extended with Path B field presence tests
- Adoption basis: v130 proved temperature-scaled Path B achieves BA delta +0.0725 vs Path A
  matched (≥ 0.03 threshold), Brier 0.1917 < Path A 0.2058, ECE ratio 1.24× < 1.5× ceiling

---

### v128 (complete)
**Released:** 2026-04-12
**Theme:** Benchmark-Specific Full Feature Search Across The 72-Feature Universe

- `results/research/v128_benchmark_feature_search.py` â€” added a full
  benchmark-specific feature-search harness that reproduces the incumbent
  benchmark-specific balanced-logistic path, screens all eligible single
  features, runs forward stepwise search, builds `L1` / elastic-net consensus
  subsets, and evaluates a ridge full-pool control
- `results/research/v128_feature_inventory.csv` â€” benchmark-by-feature
  availability and eligibility inventory for the 72-column non-target matrix
- `results/research/v128_single_feature_results.csv` â€” full single-feature
  leaderboard across all 10 benchmark-specific classifiers
- `results/research/v128_forward_stepwise_trace.csv` â€” complete forward-search
  trace for every considered feature addition
- `results/research/v128_regularized_selection_detail.csv` â€” fold-level
  `L1` / elastic-net selection detail and consensus-subset membership flags
- `results/research/v128_regularized_comparison.csv` â€” evaluated
  `l1_consensus`, `elastic_net_consensus`, and `ridge_full_pool_control`
  candidates for each benchmark
- `results/research/v128_benchmark_feature_map.csv` â€” final benchmark-specific
  winner map; 4 benchmarks switch away from `lean_baseline`
  (`BND`, `DBC`, `VGT`, `VIG`)
- `results/research/v128_benchmark_feature_search_summary.md` â€” pooled result:
  covered balanced accuracy improves slightly (`0.5000` -> `0.5016`) and
  `ece_10` improves materially (`0.0488` -> `0.0387`), while `brier_score`
  is roughly flat (`0.1813` -> `0.1819`)
- `tests/test_research_v128_benchmark_feature_search.py` â€” dedicated tests for
  candidate-universe filtering, stepwise gate logic, consensus subset capping,
  winner fallback behavior, and reduced end-to-end smoke execution
- `docs/superpowers/plans/2026-04-12-v128-benchmark-specific-feature-search.md`
  â€” implementation and handoff note for future Claude Code continuation

---

### v127 (complete)
**Released:** 2026-04-12
**Theme:** Path B Calibration Sweep On The Matched v126 Fold Frame

- `results/research/v127_path_b_calibration.py` — added a strictly prequential
  calibration sweep for Path B over the matched v126 OOS fold table
- calibration candidates evaluated: raw Path B, prequential Platt scaling, and
  prequential temperature scaling
- `results/research/v127_path_b_calibration_results.csv` — candidate comparison
  table with adoption-gate flags
- `results/research/v127_path_b_calibration_detail.csv` — per-month calibrated
  probability series and fitted temperature trace
- `results/research/v127_path_b_calibration_summary.md` — conclusion: both
  calibrators improve reliability, but neither preserves enough covered balanced
  accuracy to replace raw Path B
- `tests/test_research_v127_path_b_calibration.py` — mathematical and ranking
  tests for the new calibration utilities
- `docs/superpowers/plans/2026-04-12-v127-path-b-calibration.md` — detailed
  handoff note for future Claude Code continuation

---

### v126 (complete)
**Released:** 2026-04-12
**Theme:** Methodology Hardening For Path B And Portfolio-Aligned Artifact Refresh

- `results/research/v125_portfolio_target_classifier.py` — removed the hardcoded
  `v92` baseline, rebuilt Path A on matched dates, and enforced rolling WFO with
  `max_train_size=WFO_TRAIN_WINDOW_MONTHS`
- `results/research/v125_portfolio_target_fold_detail.csv` — now stores matched
  `path_a_prob` and `path_b_prob` on the same evaluation rows
- `results/research/v125_portfolio_target_summary.md` — downgraded Path B from
  promotion candidate to secondary research track because matched-v126 results
  show improved covered balanced accuracy but worse calibration
- `tests/test_research_v126_portfolio_target_classifier.py` — new regression
  tests for split geometry, row-wise probability renormalization, and verdict
  guardrails
- `results/monthly_decisions/2026-02/`, `2026-03/`, `2026-04/` — refreshed
  monthly artifacts so `classification_shadow` reflects the 6-benchmark
  investable pool `{VOO, VGT, VIG, VXUS, VWO, BND}`
- `docs/superpowers/plans/2026-04-12-v126-methodology-hardening.md` —
  implementation and handoff note for future Claude Code continuation

---

### v2.7 (complete)
**Released:** March 2026
**Theme:** Complete v2 Relative Return Engine

- 20 independent WFO models (one per ETF benchmark)
- Correct 6M/12M embargo periods (eliminates autocorrelation leakage)
- SQLite accumulation pipeline with weekly GitHub Actions fetch
- DRIP total return reconstruction with split-adjusted unadjusted prices
- Tax-aware lot selection (LTCG/STCG prioritization)
- 271 passing pytest tests covering all modules

---

### v3.0 (complete)
**Released:** March 2026
**Theme:** Macro Intelligence + Monthly Decision Engine

- `src/ingestion/fred_loader.py` — FRED public API client (8 macro series: yield curve, credit spreads, NFCI)
- `src/database/schema.sql` — New `fred_macro_monthly` table
- `src/processing/feature_engineering.py` — 6 derived FRED features in monthly feature matrix
- `src/models/regularized_models.py` — ElasticNetCV pipeline (l1_ratio grid [0.1–1.0])
- `src/models/wfo_engine.py` — Purge buffer fix: gap = horizon + buffer (6M→8, 12M→15)
- `src/backtest/vesting_events.py` — `enumerate_monthly_evaluation_dates()` → 120+ evaluation points
- `src/reporting/backtest_report.py` — Campbell-Thompson OOS R², BHY FDR correction, Newey-West IC
- `scripts/monthly_decision.py` — Automated monthly sell/hold recommendation script
- `.github/workflows/monthly_decision.yml` — Cron: 20th of each month (first business day)
- `results/` — Structured output folder with `monthly_decisions/decision_log.md`
- **New tests:** test_fred_loader (13), test_fred_db (10), test_fred_features (5),
  test_elasticnet (11), test_embargo_fix (9), test_monthly_backtest (11), test_oos_r2 (22)

---

### v3.1 (complete)
**Released:** March 2026
**Theme:** Ensemble Models + Kelly Sizing + Regime Diagnostics

- `src/models/regularized_models.py` — `UncertaintyPipeline` + `build_bayesian_ridge_pipeline()`
- `src/models/multi_benchmark_wfo.py` — `EnsembleWFOResult` + equal-weight 3-model ensemble
- `src/portfolio/rebalancer.py` — `_compute_sell_pct_kelly()` (0.25× Kelly, 30% cap)
- `src/ingestion/fred_loader.py` — PGR-specific: VMT (TRFVOLUSM227NFWA) — note:
  `CUSR0000SETC01` (motor vehicle insurance CPI) added here but removed in v4.1.1
  because the series does not exist in FRED's observations API
- `src/processing/feature_engineering.py` — `vmt_yoy`, `vix` features
- `src/reporting/backtest_report.py` — Rolling 24M IC series, 4-quadrant regime breakdown
- `config.py` — `KELLY_FRACTION=0.25`, `KELLY_MAX_POSITION=0.30`, VIXCLS in FRED series
- **New tests:** test_bayesian_ridge (14), test_kelly_sizing (16),
  test_pgr_fred_features (7), test_regime_breakdown (10)

---

### v4.0 (complete)
**Released:** March 2026
**Theme:** Production Validation + Portfolio Optimization + Tax-Loss Harvesting

- `src/models/wfo_engine.py` — `run_cpcv()` using skfolio `CombinatorialPurgedCV`
  (C(6,2)=15 splits, 5 backtest paths); `CPCVResult` dataclass
- `src/portfolio/black_litterman.py` — **NEW**: `build_bl_weights()` via PyPortfolioOpt
  `BlackLittermanModel`; Ledoit-Wolf shrunk covariance; view confidence = MAE²
- `src/tax/capital_gains.py` — `identify_tlh_candidates()`, `compute_after_tax_expected_return()`,
  `suggest_tlh_replacement()`, `wash_sale_clear_date()`
- `src/processing/feature_engineering.py` — `apply_fracdiff()` + `_fracdiff_weights()`;
  FFD stationarity transform implemented in numpy/scipy (fracdiff package Python <3.10 only)
- `src/portfolio/rebalancer.py` — `compute_benchmark_weights()` (IC × hit_rate normalized)
- `config.py` — `TLH_REPLACEMENT_MAP` (20 ETF pairs), CPCV/BL/TLH/fracdiff constants
- `requirements.txt` — Add `skfolio>=0.3.0`, `PyPortfolioOpt>=1.5.5`
- **New tests:** test_cpcv (17), test_black_litterman (14), test_tlh (23),
  test_fracdiff (13), test_benchmark_weights (11)
- **Total: 477 tests, all passing**

---

### v4.1 (complete)
**Released:** March 2026
**Theme:** Data Integrity + Look-Ahead Bias Guards

Critical fixes deployed before the 2026-03-25 initial ML training label bootstrap
(`post_initial_bootstrap.yml`). All changes prevent historical look-ahead bias from
contaminating the first training dataset.

- `config.py` — `FRED_DEFAULT_LAG_MONTHS = 1`, `FRED_SERIES_LAGS` dict
  (NFCI=2 months, VMT=2 months, all other series=1 month), `EDGAR_FILING_LAG_MONTHS = 2`,
  `KELLY_MAX_POSITION` reduced 0.30 → 0.20
- `src/ingestion/fred_loader.py` — `apply_publication_lags: bool = True` parameter
  added to `fetch_all_fred_macro()`; shifts each series by its configured lag
- `src/ingestion/pgr_monthly_loader.py` — `apply_filing_lag: bool = True` parameter
  added to `load()`; shifts EDGAR index forward by `EDGAR_FILING_LAG_MONTHS`
- `src/processing/feature_engineering.py` — `_apply_fred_lags()` and `_apply_edgar_lag()`
  module-level helpers; both called in `build_feature_matrix_from_db()` after reading
  raw DB values — **authoritative enforcement point** since the DB stores latest-vintage
  values and feature engineering re-reads them
- **Test updates:** test_fred_loader (+3 lag tests), test_feature_engineering (+EDGAR lag
  test, updated feature count), test_kelly_sizing (cap updated to 0.20),
  test_black_litterman (cap-related fixture updated)
- **Total: 482 tests, all passing**

**Rationale:**
- *FRED lag*: FRED serves latest-vintage data; NFCI and VMT undergo meaningful revisions
  post-release (McCracken & Ng 2015). Without lags, Jan features use Jan data — months
  before that data was actually published or finalized.
- *EDGAR lag*: PGR 10-Q for Q4 (period end Dec 31) is filed ~late February. Indexing on
  `report_period` makes Q4 combined_ratio appear in Jan features — 2 months too early.
- *Kelly cap*: Meulbroek (2005) shows 25% employer-stock concentration yields ~42%
  certainty-equivalent loss when human capital correlation is included. Cap at 20% is
  consistent with financial advisor consensus for employer stock specifically.

---

### v4.1.1 (hotfix — 2026-03-24)
**Released:** 2026-03-24
**Theme:** GitHub Actions permissions fix + FRED data bootstrap + schedule adjustment

Root cause analysis of two consecutive bootstrap failures:

1. **2026-03-23 failure** — AV rate limit hit at 14/22 tickers. Workflow was scheduled
   at 11:00 UTC; the free-tier 25 calls/day cap was exhausted mid-run.  Reschedule
   to a quieter slot (+1 day) mitigated runner contention but did not fix permissions.

2. **2026-03-24 failure** — AV price fetch succeeded (22/22 tickers, 22 AV calls,
   284 s), but `git push` exited 403.  **Root cause:** all 6 GitHub Actions workflows
   were missing `permissions: contents: write`.  The `GITHUB_TOKEN` defaults to
   read-only; without an explicit `contents: write` grant the `git push` step is
   rejected by the GitHub REST API.

Changes in this hotfix:

- **All 6 workflows** — Added `permissions: contents: write` at the job level.
- **Bootstrap schedule** — Day 1 moved to Wed 2026-03-25 14:00 UTC, Day 2 to
  Thu 2026-03-26 14:00 UTC, bootstrap moved to Thu 2026-03-26 18:00 UTC (same day
  as Day 2, 4 hours later — no AV calls needed, only DB reads).
- **Daily workflow** (`daily_data_fetch.yml`) — removed from repo in master prior
  to this hotfix; reference removed from documentation.
- **FRED bootstrap** — 12 FRED series (4 967 rows, 1990–2026-03) pre-populated
  locally and committed to `data/pgr_financials.db`.  Removed invalid series
  `CUSR0000SETC01` (motor vehicle insurance CPI) — FRED returns 400 for this ID;
  the BLS publishes it under series code SETE but FRED does not index it directly.
- **`scripts/bootstrap.py`** — `skip_fred` parameter added (default `True`);
  `_run_monthly_decision()` now skips live FRED fetch since data is pre-populated.
  Pass `--fetch-fred` to force a live refresh.

---

### v4.1.2 — FMP → SEC EDGAR XBRL replacement (complete)

---

### v4.1.3 (patch — 2026-03-25)
**Released:** 2026-03-25
**Theme:** Day 1 bootstrap confirmed successful; Day 2 timing adjusted for AV rate-limit safety

Day 1 results (2026-03-25):

- **`initial_fetch_prices.yml`** — ✅ SUCCESS. Ran at 15:01 UTC (61-min scheduler lag),
  completed in 5m 30s. All 22 AV calls succeeded (22/22 tickers). DB committed to master.
- **`monthly_8k_fetch.yml` Pass 2** — ✅ SUCCESS. Ran at 14:51 UTC (51-min scheduler lag),
  completed in 1m 4s. No new rows (idempotent — March data already present from Pass 1
  on the 20th).

Timing adjustment for Day 2 (2026-03-26):

- **Root cause concern:** GitHub Actions scheduler consistently fires 51–61 minutes late
  (free-tier runner queue). Day 1 prices completed at ~15:06 UTC. Day 2 dividends were
  scheduled at 14:00 UTC — only ~23 hours later. AV's 25 calls/day limit may reset on
  a rolling 24-hour window rather than at UTC midnight; if so, 22 new calls at 14:00 UTC
  tomorrow would land within the prior 24-hour window and exhaust the budget mid-run
  (repeating the 2026-03-23 failure).
- **Fix:** `initial_fetch_dividends.yml` cron shifted **14:00 → 15:00 UTC**;
  `post_initial_bootstrap.yml` shifted **18:00 → 19:00 UTC** (preserves 4-hour gap).
  With typical scheduler lag, Day 2 will actually execute ~16:00–16:15 UTC — well past
  the 24-hour mark from Day 1's ~15:06 UTC completion.

---

### v4.2 — 8-K Retry/Recheck + Historical Backfill (complete)
**Released:** 2026-03-24
**Theme:** Remove FMP dependency; free, no-key-required quarterly fundamentals

FMP deprecated all `/v3/` REST endpoints on 2025-08-31.  This sprint replaces
the FMP fundamentals pipeline with the SEC EDGAR XBRL Company-Concept API
(`data.sec.gov/api/xbrl/companyconcept/{cik}/us-gaap/{concept}.json`).

- `src/ingestion/edgar_client.py` — **NEW**: EDGAR XBRL client; fetches
  `EarningsPerShareDiluted`, `Revenues`, `NetIncomeLoss` from 10-Q/10-K filings
  for PGR (CIK 0000080661).  No API key required.  Quarterly cadence only —
  PGR monthly 8-K earnings supplements are PDF attachments not in XBRL.
  Returns records compatible with `db_client.upsert_pgr_fundamentals()`.
  `pe_ratio`, `pb_ratio`, `roe` are `None` (not available via XBRL).
- `src/ingestion/fmp_client.py` — retained for reference; `FMPEndpointDeprecatedError`
  already surfaces clean warnings; no further changes.
- `src/database/db_client.py` — `upsert_pgr_fundamentals` `source` column now
  stores `"edgar_xbrl"` (schema unchanged; `source` column already existed).
- `tests/test_edgar_client.py` — **NEW**: 20 passing tests (all mocked, no
  network calls): `_filter_quarterly` deduplication (6), full fetch shape/types
  (11), `fetch_pgr_latest_quarter` convenience wrapper (3).

**Data availability notes:**
- EDGAR XBRL provides quarterly data aligned with 10-Q/10-K filing dates.
  Publication lag of ~45 days (10-Q) / ~60 days (10-K) after period-end; the
  existing `EDGAR_FILING_LAG_MONTHS = 2` guard in feature engineering remains
  correct.
- `pe_ratio`, `pb_ratio`, `roe` will be `None` for all EDGAR-sourced rows.
  These columns were sparsely populated even under FMP; the WFO engine already
  handles `NaN` gracefully via `WFO_MIN_OBS` guards.
- PGR monthly combined-ratio and PIF data continue to come from the user-provided
  CSV cache (`pgr_monthly_loader.py`); EDGAR XBRL does not expose those metrics.

---

## Planned Versions

Day references below are relative to Day 1 = 2026-03-25 (first bootstrap day).
Week+ targets are only used where genuine data accumulation or external dependencies
require them (noted explicitly).

---

### v4.3 — Signal Quality + Confidence Layer
**Target:** Day 3 (2026-03-27)
**Theme:** Surface BayesianRidge uncertainty in reports; fix BL Ω; reduce feature redundancy

No data accumulation needed — all changes are pure code against the already-populated DB.

- `src/models/multi_benchmark_wfo.py` — `get_confidence_tier(y_hat, y_std)` via
  `norm.cdf(y_hat / y_std)`; `confidence_tier` and `prob_outperform` columns in
  `get_ensemble_signals()` output
- `scripts/monthly_decision.py` — Wire `run_ensemble_benchmarks` + `get_ensemble_signals`
  (currently uses single-model elasticnet path); confidence tier + P(outperform) in report
- `src/portfolio/black_litterman.py` — `prediction_variances: dict[str, float]` param;
  switch Ω diagonal from MAE² to BayesianRidge posterior variance (σ²_pred per benchmark);
  fallback uses `(MAE × √(π/2))²` as RMSE approximation
- `config.py` — `FEATURES_TO_DROP = ["vol_21d", "credit_spread_ig"]` (redundant features);
  `BL_USE_BAYESIAN_VARIANCE = True`
- `src/processing/feature_engineering.py` — Drop redundant features at end of matrix build;
  result: 15 features from 17, improving obs/feature ratio from ~3.5:1 to ~4:1

**Target monthly report format:**
```
COMPOSITE SIGNAL: OUTPERFORM (MODERATE CONFIDENCE)
  P(outperform): ~65% [90% CI: 50%–75%]
  Benchmarks favoring outperformance: 14/20 (70%)
  Expected PGR-SPY spread: +3.2%, 80% CI [-2.1%, +8.5%]
  Calibration status: Phase 1 (uncalibrated Bayesian posterior)
```

---

### v4.3 — Diagnostic OOS Evaluation Report
**Target:** Day 5 (2026-03-29)

Pure code addition — surfaces already-computed diagnostics into a sidecar report.

- `scripts/monthly_decision.py` — `_write_diagnostic_report()` calling existing
  `compute_oos_r_squared()`, `compute_newey_west_ic()`, `generate_regime_breakdown()`;
  writes `diagnostic.md` alongside `recommendation.md`
- `config.py` — `DIAG_MIN_OOS_R2 = 0.02`, `DIAG_MIN_IC = 0.07`,
  `DIAG_MIN_HIT_RATE = 0.55`, `DIAG_CPCV_MIN_POSITIVE_PATHS = 10`

**Benchmark thresholds (peer review):**

| Metric | Good | Marginal | Bad |
|--------|------|----------|-----|
| OOS R² | >2% | 0.5–2% | <0% |
| Mean IC | >0.07 | 0.03–0.07 | <0.03 |
| Hit Rate | >55% | 52–55% | <52% |
| CPCV positive paths | ≥13/15 | 10–12/15 | <10/15 |
| PBO | <15% | 15–40% | >40% |

---

### v4.4 — STCG Tax Boundary Guard
**Target:** Day 7 (2026-04-01)

Small focused addition; no data dependencies.

- `src/portfolio/rebalancer.py` — `_check_stcg_boundary()`: warns when lots in the
  6–12 month STCG zone and predicted alpha < STCG-to-LTCG penalty (~17–22pp)
- `config.py` — `STCG_BREAKEVEN_THRESHOLD = 0.18`
- `src/portfolio/rebalancer.py` — `stcg_warning: str | None` field added to
  `VestingRecommendation` dataclass

---

### v4.5 — New Predictor Variables
**Target:** Day 10 (2026-04-04)
*Note: KIE ticker and new FRED series (CUSR0000SETA02, CUSR0000SAM2) added in v4.5-prep
(2026-03-24). Remaining work is feature engineering and the `pgr_vs_kie_6m` signal.*

| Feature | FRED Series / Source | Mechanism |
|---------|----------------------|-----------|
| `used_car_cpi_yoy` | `CUSR0000SETA02` | Auto total-loss severity; 2021–22 spike was a major PGR headwind |
| `medical_cpi_yoy` | `CUSR0000SAM2` | Bodily injury / PIP claim severity |
| `cr_acceleration` | EDGAR (existing data) | 3-period diff of `combined_ratio_ttm`; second derivative of underwriting margins |
| `pgr_vs_kie_6m` | AV: KIE prices (already in DB) | PGR return minus KIE 6M return; insurance-sector idiosyncratic alpha |

- `src/processing/feature_engineering.py` — 4 new feature computations
- KIE prices already being fetched (v4.5-prep); no additional AV calls needed

---

### v5.0 — CPCV Upgrade + Ensemble Diversity (complete)
**Released:** March 2026

- **CPCV**: C(6,2)=15 → C(8,2)=28 paths (`CPCV_N_FOLDS = 8`)
- **Inverse-variance ensemble**: `1/MAE²`-weighted average in `get_ensemble_signals()`;
  GBT (MAE=0.156) and ElasticNet (MAE=0.165) receive ~75% of total weight
- **Shallow GBT**: `build_gbt_pipeline()` — `GradientBoostingRegressor(max_depth=2,
  n_estimators=50, learning_rate=0.1)`; mean IC +0.148 vs +0.081 ElasticNet across
  8 representative benchmarks; largest gains on VHT (+0.262), VNQ (+0.184), VPU (+0.192)
- **ETF descriptions**: `_ETF_DESCRIPTIONS` dict wired into per-benchmark signal
  table in `recommendation.md` and `diagnostic.md`
- **config.py**: `ENSEMBLE_MODELS` updated to 4 members; `DIAG_CPCV_MIN_POSITIVE_PATHS`
  updated to 19/28; `CPCV_N_FOLDS = 8`
- **459 new tests** in `test_v50_ensemble.py`; total: 675 passed, 1 skipped

---

### v5.1 — Per-Benchmark Platt Calibration (complete)
**Released:** March 2026

- **New file `src/models/calibration.py`**: `CalibrationResult` dataclass,
  `compute_ece()`, `block_bootstrap_ece_ci()` (circular block bootstrap, block_len
  = prediction horizon), `fit_calibration_model()` (Platt at n≥20, isotonic at
  n≥500), `calibrate_prediction()`
- **Per-benchmark design**: one Platt model fitted per ETF benchmark on that
  benchmark's own OOS fold history.  Global pooled calibration was evaluated and
  rejected — pooling 21 asset classes with different return scales caused isotonic
  regression to return a single constant for all benchmarks (plateau collapse).
  With n=78–260 OOS obs per benchmark (2026), isotonic threshold raised to
  `CALIBRATION_MIN_OBS_ISOTONIC = 500`; re-evaluate ~2028.
- **Calibration pipeline**: `_calibrate_signals()` reconstructs inverse-variance
  ensemble OOS fold predictions per benchmark, fits per-benchmark Platt, adds
  `calibrated_prob_outperform` column.  ECE = 2.1% (aggregate, block bootstrap)
  on 3,270 pooled OOS observations.
- **Report updates**: `recommendation.md` shows P(raw) and P(calibrated) side-by-side;
  calibration note replaced with live ECE and 95% CI; `diagnostic.md` calibration
  phase table is data-driven (reads `cal_result.method` at runtime)
- **`VestingRecommendation`**: `calibrated_prob_outperform: float | None` field added
- **config.py**: `CALIBRATION_MIN_OBS_PLATT=20`, `CALIBRATION_MIN_OBS_ISOTONIC=500`,
  `CALIBRATION_N_BINS=10`, `CALIBRATION_BOOTSTRAP_REPS=500`
- **33 new tests** in `test_calibration.py`; total: 747 passed, 1 skipped

---

### v5.2 — Conformal Prediction Intervals ✅ COMPLETE
**Delivered:** 2026-03-30

Distribution-free 80% prediction intervals for each benchmark's ensemble prediction,
with marginal coverage guarantees under time-series non-stationarity.

**Delivered:**
- `src/models/conformal.py` — Native split conformal + Adaptive Conformal Inference (ACI)
  implementation; no MAPIE refit latency in the monthly pipeline
  - `ConformalResult` dataclass: lower, upper, width, coverage_level, empirical_coverage,
    n_calibration, method
  - `split_conformal_interval()`: finite-sample corrected quantile of WFO OOS absolute
    residuals; P(y ∈ CI) ≥ 1-α guarantee (Vovk et al. 2005)
  - `aci_adjusted_interval()`: walk-forward α_t adaptation; update rule:
    α_{t+1} = clip(α_t + γ(α_nominal − err_t), 0.01, 0.99); γ=0.05 default;
    handles distribution shift in 6-month overlapping return windows (Gibbs & Candès 2021)
  - `conformal_interval_from_ensemble()`: main entry; computes residuals from WFO OOS
    y_true/y_hat; dispatches to split or ACI
- `config.py` — Added `CONFORMAL_COVERAGE=0.80`, `CONFORMAL_METHOD="aci"`,
  `CONFORMAL_ACI_GAMMA=0.05`
- `requirements.txt` — Added `mapie>=1.3.0` (used for TimeSeriesRegressor validation path;
  production pipeline uses native implementation)
- `scripts/monthly_decision.py` — `_compute_conformal_intervals()` per-benchmark ACI
  intervals added as Step 2.7; recommendation.md consensus table shows median 80% CI
  range; per-benchmark table adds CI Lower / CI Upper columns; diagnostic.md adds
  Conformal Prediction Intervals section with empirical vs nominal coverage per benchmark
- **46 new tests** in `tests/test_conformal.py`; total: 793 passed, 1 skipped

---

### v6.0 — Cross-Asset Signals + BLP Aggregation
**Target:** Day 42 (2026-05-06)
**Status (2026-03-30):** Feature engineering complete; two of four planned signals shipped.

**Peer data source decision (2026-03-30):** Peer price/dividend history for ALL,
TRV, CB, HIG is sourced from Alpha Vantage — NOT yfinance.  yfinance scrapes
Yahoo Finance's undocumented internal endpoints, has no API contract, and has
broken silently multiple times.  Using AV keeps the entire price/dividend stack
on a single source with consistent unadjusted price handling, known call budget,
and no new dependencies.

**AV budget solution:** Peer tickers (8 calls/run) are fetched on a dedicated
Sunday 04:00 UTC cron — exactly 30 hours after the main Friday 22:00 UTC
weekly_fetch.py cron.  Each day stays within the 25 calls/day free-tier limit:
- Friday 22:00 UTC: `weekly_fetch.py` — 24 AV calls (PGR + 22 ETFs)
- Sunday 04:00 UTC: `peer_data_fetch.py` — 8 AV calls (ALL, TRV, CB, HIG prices + dividends)

**Data already flowing (2026-03-30):** `peer_bootstrap.yml` ran manually; ALL,
TRV, CB, HIG full price and dividend history seeded into the DB.  `peer_data_fetch.yml`
runs weekly from now on.

BLP parameter fitting needs ~12 months of live OOS predictions — delay this
sub-feature to Week 8+ (2026-05-20) while the rest of v6.0 ships on Day 42.

**SHIPPED (2026-03-30):**
- **`high_52w`**: `current_price / 52-week_high` (George & Hwang 2004).
  Implemented in `build_feature_matrix()` as a price-derived feature using
  `daily_close.rolling(252, min_periods=126).max()`.
  IC=0.122 (p=0.041, n=281); 91.4% data coverage.
- **`pgr_vs_peers_6m`**: PGR 6M DRIP return minus equal-weight peer composite
  (ALL, TRV, CB, HIG) 6M return.  Pre-computed in `build_feature_matrix_from_db()`
  and injected as a synthetic FRED column.  IC=0.115 (p=0.045, n=304);
  98.7% data coverage.  Current value: −0.232 (PGR −23% vs peers over 6M).
- **`pgr_vs_vfh_6m`**: PGR 6M return minus VFH (Vanguard Financials ETF) 6M return.
  Broadens KIE benchmark to all US financials (banks, insurance, diversified).
  VFH already in ETF universe — no separate bootstrap needed.
  IC=0.088 (p=0.165, n=not significant independently); 82.1% data coverage.
  Current value: −0.073.  Lasso regularization will select or shrink based on
  marginal contribution in the WFO ensemble.
- **30 new/updated tests** in `tests/test_v60_features.py`; total: **849 passed, 1 skipped**

**REMAINING:**
- **Beta-Transformed Linear Pool (BLP)**: Replaces naive equal-weight ensemble
  averaging (Ranjan & Gneiting 2010: any linear pool of calibrated forecasts is
  necessarily uncalibrated); 5-parameter BLP fit via negative log-likelihood
  *(requires ~12 months of live OOS predictions; BLP sub-feature ships Week 8)*
- **Residual momentum**: Regress PGR returns on Fama-French 3-factor over trailing
  36M window; cumulate factor-neutral residuals from t-12 to t-1 (Blitz et al. 2011:
  2× alpha of raw momentum, greater consistency)
- *(Cross-asset signal infrastructure complete; all planned signals shipped)*

---

---

### v6.1 — Monthly Decision Email Notification (complete)
**Released:** 2026-03-30
**Theme:** Automated email delivery of monthly prediction report

Added `Send monthly decision email` step to `.github/workflows/monthly_decision.yml`
immediately after the `Commit results` step.

**Implementation:**
- Inline Python (smtplib) — no third-party actions, no new dependencies
- Subject: `PGR Monthly Decision — {Month YYYY}: {SIGNAL}` (signal parsed from `recommendation.md`)
- Body: full plain-text content of `results/monthly_decisions/YYYY-MM/recommendation.md`
- Port 465 → SMTP_SSL; port 587 → STARTTLS (auto-detected from `SMTP_PORT` secret)
- Skips gracefully if secrets are unconfigured or report file doesn't exist
- `continue-on-error: true` — email failure never blocks data collection or DB commit
- Skipped on `dry_run: true` dispatches

**Repository secrets required:**

| Secret | Purpose |
|--------|---------|
| `SMTP_SERVER` | Outbound SMTP hostname |
| `SMTP_PORT` | SMTP port (465 for SSL, 587 for STARTTLS) |
| `SMTP_USERNAME` | SMTP authentication username |
| `SMTP_PASSWORD` | SMTP authentication password |
| `PREDICTION_EMAIL_FROM` | Sender address shown in the From header |
| `PREDICTION_EMAIL_TO` | Recipient address |

---

### v6.2 — Historical Backfill + Expanded 8-K Schema (complete)
**Released:** 2026-04-01
**Theme:** Load 20+ years of PGR operating data; expand pgr_edgar_monthly to 44 columns

Unlocks `data/processed/pgr_edgar_cache.csv` (256 rows, 2004–2026, 65 columns) for
model training.  The primary data gap was that `pgr_edgar_monthly` had only 22 months
of live-fetched data capturing 7 of 65 available fields.

**P1.2 — Schema Expansion:**
- `src/database/schema.sql` — `pgr_edgar_monthly` extended from 7 to 44 columns:
  foundational P&L (NPW, NPE, net income, EPS diluted, loss/LAE ratio, expense ratio),
  segment channels (NPW/NPE/PIF by agency/direct/commercial/property),
  company-level operating metrics (investment income, total revenues, total assets,
  ROE, book equity, unearned premiums, buyback data), investment portfolio metrics
  (FTE return, book yield, unrealized gains, duration), and derived features
- `src/database/db_client.py` — `initialize_schema` applies 37 idempotent `ALTER TABLE`
  migrations; `upsert_pgr_edgar_monthly` and `get_pgr_edgar_monthly` updated for all 44 cols

**P1.1 — CSV Backfill (load_from_csv):**
- `scripts/edgar_8k_fetcher.py` — `load_from_csv` now maps all 65 CSV columns via
  `DIRECT_MAP`; computes derived features:
  `channel_mix_agency_pct = npw_agency / (npw_agency + npw_direct)`,
  `npw_growth_yoy` (12M pct_change), `underwriting_income = npe × (1 − CR/100)`,
  `unearned_premium_growth_yoy` (12M pct_change); `buyback_yield` remains NULL
  (requires market_cap not available in CSV)
- Coverage log after load: combined_ratio, NPW, npw_agency, investment_income,
  book_value_per_share, gainshare_estimate
- CSV column rename handled: `roe_net_income_trailing_12m` → `roe_net_income_ttm`

**Testing:**
- `tests/test_v62_schema_and_csv.py` — **29 new tests**:
  schema migration, round-trip upsert, backward compat, direct field mapping,
  derived field correctness, NaN/NULL handling, dry_run, error cases, edge cases
- **Total: 909 passed, 1 skipped**

```bash
# Bootstrap 20+ years of history (no network calls):
python scripts/edgar_8k_fetcher.py --load-from-csv
# Dry run to verify coverage before writing:
python scripts/edgar_8k_fetcher.py --load-from-csv --dry-run
```

---

### v6.3 — Channel-Mix Features in Monthly Decision Model (complete)
**Released:** 2026-04-01
**Theme:** Wire agency/direct channel-mix signals into the ML feature pipeline (P1.4)

Adds two new predictive features to `build_feature_matrix()`, consuming the
segment-level data loaded by v6.2's CSV backfill:

- **`channel_mix_agency_pct`**: `npw_agency / (npw_agency + npw_direct)`.
  Agency share trending down (direct gaining) is a leading indicator of
  improved unit economics and combined-ratio improvement — historically one
  of PGR's key competitive differentiation signals.
- **`npw_growth_yoy`**: companywide NPW 12-month YoY growth rate.  Strong
  growth (> 10%) signals rate adequacy and market-share gain.

**Implementation:**
- `src/processing/feature_engineering.py` — new channel-mix block in the
  `if pgr_monthly is not None` section; both features added to the sparsity-guard
  loop (`WFO_MIN_GAINSHARE_OBS` threshold, same as combined_ratio_ttm)
- Features read directly from `pgr_edgar_monthly` columns pre-computed at
  CSV load time; forward-filled to monthly feature matrix dates
- Absent when `pgr_monthly=None`, column missing, or all-NaN (full backward compat)

**Testing:** 12 new tests in `tests/test_v63_channel_mix_features.py`; total **921 passed, 1 skipped**

---

### v6.4 — P2.x Operational & Valuation Features (current)
**Released:** 2026-04-01
**Theme:** Wire underwriting income, unearned premium pipeline, ROE trend,
investment portfolio quality, and share repurchase signal into the ML feature
pipeline (P2.1–P2.5)

Adds eleven new predictive features to `build_feature_matrix()`, all sourced
from `pgr_edgar_monthly` (PGR monthly 8-K supplements).  Defaulting to monthly
8-K data throughout maximises observation count and ensures consistent sourcing.

**P2.2 — Underwriting income:**
- `underwriting_income` (DB pre-computed: `npe × (1 − CR/100)`)
- `underwriting_income_3m` (3-month trailing average)
- `underwriting_income_growth_yoy` (12M YoY pct_change)

**P2.3 — Unearned premium pipeline:**
- `unearned_premium_growth_yoy` (DB pre-computed 12M pct_change; leads earned premium ~6M)
- `unearned_premium_to_npw_ratio` (`unearned_premiums / net_premiums_written`)

**P2.4 — ROE trend:**
- `roe_net_income_ttm` (8-K monthly TTM ROE; 4× more obs than quarterly XBRL)
- `roe_trend` (current ROE − rolling 12M mean; positive = improving efficiency)

**P2.1 — Investment portfolio:**
- `investment_income_growth_yoy` (12M YoY growth; rate-environment proxy)
- `investment_book_yield` (fixed-income book yield; complements `yield_slope`)

**P2.5 — Share repurchase signal:**
- `buyback_yield` (annualised buyback spend / est. market cap via BVPS + equity)
- `buyback_acceleration` (current month / trailing 12M mean; > 1 = accelerating)

**Implementation:**
- `src/processing/feature_engineering.py` — v6.4 P2.x block added in the
  `if pgr_monthly is not None` section; all eleven features added to the
  sparsity-guard loop (`WFO_MIN_GAINSHARE_OBS` threshold)
- All features sourced from `pgr_edgar_monthly`; forward-filled to monthly dates
- Absent when `pgr_monthly=None`, column missing, or below sparsity threshold
  (full backward compat with pre-v6.2 databases)

**Testing:** 28 new tests in `tests/test_v64_p2x_features.py`; total **949 passed, 1 skipped**

---

### v6.5 — P2.6 / P2.7 / P2.8: HTML Parser Extension, Calibration Plot, Email Module (current)
**Released:** 2026-04-02
**Theme:** Live 8-K field capture (P2.6), calibration diagnostic (P2.7), testable email (P2.8)

**P2.6 — Extend 8-K HTML Parser:**
Extends `_parse_html_exhibit()` in `scripts/edgar_8k_fetcher.py` to capture
12 additional fields from the monthly 8-K exhibit HTML:
`net_premiums_written`, `net_premiums_earned`, `npw_agency`, `npw_direct`,
`npw_commercial`, `npw_property`, `investment_income`, `book_value_per_share`,
`eps_basic`, `shares_repurchased`, `avg_cost_per_share`, `investment_book_yield`.
New `_try_parse_dollar()` helper reduces boilerplate for range-guarded regex extraction.
`_compute_derived_fields()` extended to compute `channel_mix_agency_pct`,
`underwriting_income`, `npw_growth_yoy`, `unearned_premium_growth_yoy` from the
assembled time series (mirrors the CSV backfill path).

**P2.7 — Calibration Reliability Diagram:**
`_plot_calibration_curve()` added to `scripts/monthly_decision.py`.  Written to
`results/monthly_decisions/YYYY-MM/plots/calibration_curve.png` on each monthly run.
Shows binned predicted P(outperform) vs. actual fraction positive, with ECE annotation
and 95% bootstrap CI.  `_calibrate_signals()` return signature updated to expose
pooled probabilities and outcomes for the diagram.

**P2.8 — Testable Email Module:**
Email logic extracted from inline YAML into `src/reporting/email_sender.py`:
- `build_email_message()` — pure function; constructs MIMEMultipart from report body
- `send_monthly_email()` — env-var / kwarg config; SMTP_SSL (port 465) or STARTTLS (587);
  `dry_run=True` returns subject without network connection
Workflow YAML step updated to call the module.

**Testing:** 35 new tests in `tests/test_v65_p26_p27_p28.py`; total **984 passed, 1 skipped**

---

### Housekeeping — AV "Information" vs "Note" Response Handling
**Status:** Complete
**Theme:** Weekly fetch resilience against benign AV advisories

The AV free-tier API returns two distinct response types that the current code
treats identically:

| AV Key | Meaning | Correct action |
|--------|---------|----------------|
| `"Note"` | Hard daily quota (25 calls) exhausted | Stop immediately; defer remaining tickers |
| `"Information"` | Advisory nudge ("spread out requests") | Log warning; **continue fetching** |

**Current behaviour:** `multi_ticker_loader.py` and `multi_dividend_loader.py` now
raise `AVRateLimitAdvisory` on `"Information"` so the batch skips only the affected
ticker and continues.  Only `"Note"` raises `AVRateLimitError` and stops the batch.

**Root cause context:** The advisory fires when a free-tier session uses ~22–23 of its
25 daily calls in one run.  The 13-second inter-call delay already exceeds AV's stated
1-request/second limit; increasing the delay would not suppress the advisory and would
add ~2.5 minutes to every weekly run with no benefit.  As the DB matures and more
tickers return 0 new rows (already-fresh data), the total calls-per-run will decrease
naturally and the advisory will stop appearing.

**Shipped fix:**
- `src/ingestion/multi_ticker_loader.py` — `"Information"` continues the batch; `"Note"`
  remains the hard stop.
- `src/ingestion/multi_dividend_loader.py` — Same pattern.
- `src/ingestion/exceptions.py` — `AVRateLimitAdvisory` distinguishes soft advisories
  from hard quota exhaustion.
- `tests/test_multi_ticker_loader.py` — Regression coverage confirms mocked advisory
  responses no longer abort the full batch and do not require a real `AV_API_KEY`.

---

### v6.x Completed — Feature Engineering Enhancements (2026-04-01)

**pb_ratio and pe_ratio now use monthly EDGAR 8-K data throughout the stack.**

Previously `pe_ratio` used quarterly XBRL EPS from `pgr_fundamentals_quarterly`
(rolling 4-quarter sum, ~86 data points).  Now both ratios are computed entirely
from `pgr_edgar_monthly` — the same monthly 8-K supplements that supply
`combined_ratio` and `book_value_per_share`:

| Feature | Source (before) | Source (after) |
|---|---|---|
| `pe_ratio` | `pgr_fundamentals_quarterly.eps` (quarterly XBRL, 4-quarter sum) | `pgr_edgar_monthly.eps_basic` (monthly 8-K, 12-month rolling sum) |
| `pb_ratio` | `pgr_edgar_monthly.book_value_per_share` ✅ | unchanged ✅ |
| `roe` | `pgr_fundamentals_quarterly.roe` (quarterly XBRL) | unchanged (see candidate below) |

**Changes shipped:**
- `src/database/schema.sql` — Added `eps_basic REAL` to `pgr_edgar_monthly`
- `src/database/db_client.py` — Migration, upsert, and get updated for `eps_basic`
- `src/ingestion/pgr_monthly_loader.py` — Loads `eps_basic` from CSV with alias resolution
- `src/processing/feature_engineering.py` — `pe_ratio` uses `edgar_raw["eps_basic"].rolling(12).sum()`
- `docs/PGR_EDGAR_CACHE_DATA_DICTIONARY.md` — Full data dictionary for all 65 columns

**Bug fixes also shipped:**
- `src/ingestion/multi_ticker_loader.py` / `multi_dividend_loader.py` — Fixed UTC/local
  date mismatch in skip-if-fresh logic (`date.today()` → `datetime.now(tz=timezone.utc)`)
- `tests/test_multi_ticker_loader.py` — Fixed budget-exhaustion test to insert UTC dates
- `src/processing/feature_engineering.py` — Removed over-aggressive `cr_acceleration`
  gate (was dropping the feature in test fixtures with <60 obs; production data has 225+)
- `src/ingestion/exceptions.py` / `multi_ticker_loader.py` / `multi_dividend_loader.py` —
  AV `"Information"` (soft advisory) now raises `AVRateLimitAdvisory` and continues
  batch; only `"Note"` (hard quota) raises `AVRateLimitError` and stops

---

### v122 — Classifier Audit + Peer Reviews (2026-04-12)

**Theme:** Classification layer audit, portfolio alignment gap identified,
dual peer-review research cycle completed.

**Classifier audit (v122):**
- Monthly shadow classifier audit run as-of 2026-04-11 (feature anchor 2026-03-31)
- Shadow model: `separate_benchmark_logistic_balanced`, lean 12-feature baseline,
  `oos_logistic_calibration`, benchmark-quality weighted aggregation
- Pooled metrics: accuracy 68.89%, balanced accuracy 58.27%, Brier 0.2278
- Calibrated shadow path: accuracy 75.38%, balanced accuracy 51.32%, Brier 0.1852, ECE 0.0813
- April 2026 snapshot: P(Actionable Sell) = 35.2%, MODERATE confidence, NEUTRAL stance
- Top feature: `combined_ratio_ttm` (standardized importance 1.245) — underwriting quality
  dominates across all benchmarks; macro/rates layer (credit spreads, real yields) secondary
- Results archived: `results/research/v122_classifier_audit_summary.md`,
  `v122_classifier_audit_coefficients.csv`, `v122_classifier_audit_feature_totals.csv`

**Portfolio alignment gap identified:**
- v27 established two separate ETF universes: forecast benchmarks vs investable redeploy
- Current classifier uses regression quality weights to aggregate 8 benchmark probabilities,
  but the user's redeploy portfolio is {VOO, VGT, SCHD, VXUS, VWO, BND}
- VGT and SCHD are entirely absent from the classifier; DBC, GLD, VMBS, VDE are in the
  classifier but not investable — the aggregated signal answers the wrong question

**Peer review cycle:**
- Deep research prompt drafted: `docs/superpowers/plans/2026-04-12-v123-classification-enhancement-research-prompt.md`
- Two independent reviews commissioned and archived:
  - `docs/archive/history/peer-reviews/2026-04-12/claude_opus_peerreview_20260412.md`
  - `docs/archive/history/peer-reviews/2026-04-12/chatgpt_peerreview_20260412.md`
- Both reports converge on: portfolio-weighted aggregation (fixed redeploy weights), VGT
  addition now, SCHD deferred, replace prequential calibration with temperature/Platt
  scaling, benchmark-specific feature subsetting (not expansion), veto gate as first
  production role requiring ≥ 24 matured prospective months

**Synthesis and planning:**
- Synthesis plan: `docs/superpowers/plans/2026-04-12-v123-v128-classification-enhancement-plan.md`
- Key architectural decision: Path A (per-benchmark → portfolio-weighted aggregation) and
  Path B (single composite portfolio-target classifier) will run in parallel from v125;
  architecture selection made empirically based on balanced accuracy and calibration results

**Active next cycle:** v123–v128 classification enhancement (see ROADMAP.md)

---

### Candidate Features — EDGAR Monthly 8-K (Future Sprint)

Full column documentation in `docs/PGR_EDGAR_CACHE_DATA_DICTIONARY.md`.
All features below are derivable from `data/processed/pgr_edgar_cache.csv`
with 256 monthly observations back to August 2004.

**Highest priority (strong theoretical prior + straightforward derivation):**

| Candidate Feature | Derivation | Rationale |
|---|---|---|
| `npw_growth_yoy` | `net_premiums_written.pct_change(12)` | Volume momentum; acceleration above peers signals pricing power |
| `channel_mix_direct_pct` | `pif_direct_auto / pif_total_personal_lines` | Rising direct share = structurally higher margin (no agent commission) |
| `unearned_premium_growth_yoy` | `unearned_premiums.pct_change(12)` | Forward revenue signal; converts to earned revenue over next 6–12 months |
| `underwriting_income` | `npe - losses_lae - policy_acquisition_costs - other_underwriting_expenses` | Core insurance profit stripped of investment income and taxes; cleaner signal than net income |
| `npw_per_pif` | `net_premiums_written / pif_total` | Average premium per policy — captures rate increases independent of volume |
| `roe` (monthly) | `roe_net_income_trailing_12m` from 8-K (pre-computed) | Switch from quarterly XBRL to monthly 8-K for consistency; same 256-obs depth as pe/pb |

**Medium priority (informative but more data-engineering work):**

| Candidate Feature | Derivation | Rationale |
|---|---|---|
| `reserve_to_npe_ratio` | `loss_lae_reserves / net_premiums_earned` | Reserve adequacy; rising ratio precedes adverse development |
| `realized_gain_to_ni_ratio` | `total_net_realized_gains / net_income` | Earnings quality flag; high ratio = income driven by portfolio sales, not underwriting |
| `price_to_npw` | `market_cap / (net_premiums_written * 12)` | Insurance-sector valuation multiple alongside P/B |
| `investment_income_growth_yoy` | `investment_income.pct_change(12)` | Interest rate reinvestment tailwind/headwind |
| `equity_per_share_growth_yoy` | `book_value_per_share.pct_change(12)` | Intrinsic value compounding rate on a per-share basis |
| `loss_ratio_ttm` | `rolling(12).mean()` of `loss_lae_ratio` | Separates loss deterioration from expense pressure within combined_ratio |
| `expense_ratio_ttm` | `rolling(12).mean()` of `expense_ratio` | Structural cost efficiency trend independent of loss activity |

**Lower priority / requires additional data:**

| Candidate Feature | Notes |
|---|---|
| `unrealized_gain_pct_equity` | `net_unrealized_gains_fixed / shareholders_equity` — OCI/rate risk proxy; high sensitivity to rate regime |
| `cr_vs_industry_spread` | Requires external peer CR data (Travelers, Allstate, etc.) |
| `combined_ratio_ex_cats` | Not directly reported; would need catastrophe loss estimates from external source |

**Implementation notes:**
- All features require the 2-month EDGAR filing lag (`config.EDGAR_FILING_LAG_MONTHS`)
- `pif_property`, `npw_property`, and `npe_property` are only available from ~2015; any feature using them will have reduced coverage
- Add to `pgr_monthly_loader.py` → `schema.sql` + `db_client.py` migration → `feature_engineering.py` (same pattern as `eps_basic` / `book_value_per_share`)
- Validate each feature with a `pytest` test confirming non-null coverage ≥ `WFO_MIN_GAINSHARE_OBS` before activating in WFO

---

## Development Principles

- **Never finalize a module without a passing pytest suite** (CLAUDE.md mandate)
- **No K-Fold cross-validation** — `TimeSeriesSplit` with embargo + purge buffer only
- **No StandardScaler across full dataset** — scaler isolated within each WFO fold Pipeline
- **No yfinance** — not for fundamentals, ratios, or price data; AV is the canonical price
  source for all tickers including v6.0 insurance peers (see v6.0 peer data source decision)
- **Python 3.10+**, strict PEP 8, full type hinting
- **Approved libraries:** pandas, numpy, scikit-learn, matplotlib, xgboost, requests,
  statsmodels (v3.0+), skfolio/PyPortfolioOpt (v4.0+)

## Monthly Decision Log

See [`results/monthly_decisions/decision_log.md`](results/monthly_decisions/decision_log.md) for the persistent record of all automated monthly recommendations generated by `scripts/monthly_decision.py`.
