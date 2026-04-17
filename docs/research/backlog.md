## v139-v152 Follow-On Status

- `v140` through `v145` are complete, with machine-readable logs and summary
  artifacts now recorded under `results/research/`
- `v141` produced the first new research-only winner in this follow-on cycle:
  `results/research/v141_blend_weight_candidate.txt = 0.60`
- `v143` and `v144` added two more research-only winners:
  `v143_corr_prune_candidate.txt = 0.80` and
  `v144_conformal_candidate.json = {"coverage": 0.75, "aci_gamma": 0.03}`
- `v149` added the latest decision-layer winner:
  `v149_kelly_candidate.json = {"fraction": 0.50, "cap": 0.25}`
- `v140` and `v142` both closed as no-change confirmations of the incumbent
  candidates on the current post-v138 frame
- `v146`, `v147`, `v148`, and `v150` all closed as no-change confirmations on
  the bounded follow-through grids
- `v151` now surfaces the promoted winners side-by-side in monthly shadow
  artifacts under the reporting-only variant `autoresearch_followon_v150`
- `v145` recorded a promising but tradeoff-heavy `(48, 6)` WFO window result,
  so the incumbent `{"train": 60, "test": 6}` candidate remains in place
- `v152` is now complete, so this follow-on cycle is closed
- The durable shadow-only promotion outcome is the additive
  `autoresearch_followon_v150` reporting lane built from the `v141`, `v143`,
  `v144`, and `v149` winners
- The next autonomous session should start from the ranked backlog below rather
  than reopening any `v140-v150` follow-through sweeps

## Ranked Next Queue

1. `BL-01` - Black-Litterman tau/view tuning
   Reason: highest-value remaining decision-layer calibration task now that the
   replay-proxy and Kelly follow-through work are complete.
2. `CLS-02` - Firth logistic for short-history benchmarks
   Reason: most direct remaining classifier-stability improvement that is not
   blocked on prospective shadow time.
3. `FEAT-01` - DTWEXBGS post-v128 feature search
   Reason: small-scope benchmark-specific feature follow-up with a plausible
   upside and low integration risk.
4. `FEAT-02` - WTI 3M momentum for DBC/VDE
   Reason: still a credible domain feature addition, but it depends on a clean
   external-series verification pass first.

Leave `CLS-03` blocked on the matured-month gate, and leave `REG-02` deferred
until a future ensemble-level plan justifies reopening the standalone GBT line.

### BL-01 â€” Black-Litterman Tau/View Tuning
**Status:** open
**Priority:** medium
**Rationale:** The decision layer still uses untuned BL priors even though the regression and classifier research stack has moved materially since the original defaults.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** recommendation accuracy up, policy uplift up modestly
**Last touched:** 2026-04-13 plan

### CLS-01 â€” SCHD Per-Benchmark Classifier Addition
**Status:** deferred
**Priority:** medium
**Rationale:** SCHD remains outside the current benchmark-specific classifier path because of history depth, but it is still investable in the real redeploy portfolio.
**Estimated effort:** M
**Depends on:** CLS-03
**Expected metric impact:** better portfolio alignment, unknown BA impact
**Last touched:** v124-v128 cycle

### CLS-02 â€” Firth Logistic For Short-History Benchmarks
**Status:** open
**Priority:** medium
**Rationale:** Short-history benchmarks still risk unstable logistic fits under small-sample class imbalance.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** classifier calibration up, covered BA up slightly
**Last touched:** 2026-04-13 plan

### CLS-03 â€” Path A vs Path B Production Decision
**Status:** blocked
**Priority:** high
**Rationale:** Path B now has stronger historical evidence, but the 24 matured prospective-month gate remains time-locked.
**Estimated effort:** M
**Depends on:** 24 matured months of shadow monitoring
**Expected metric impact:** governance milestone, not immediate backtest uplift
**Last touched:** v135

### REG-01 â€” Ensemble-Level Clip/Shrink Blend Revisit
**Status:** complete
**Priority:** medium
**Rationale:** The bounded `v140` sweep on 2026-04-16 found the current frame effectively flat across the tested shrinkage range, so the incumbent `0.50` setting remains the research-only winner for now.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** no current uplift observed
**Last touched:** v140

### REG-02 â€” Rank-Normalized GBT With New Search Region
**Status:** deferred
**Priority:** low
**Rationale:** The fresh v137 sweep improved standalone GBT with shallow trees, but it still fell short of the old success bar and should not be promoted before ensemble-level validation.
**Estimated effort:** M
**Depends on:** v137 ensemble follow-up
**Expected metric impact:** low-confidence OOS R2 improvement
**Last touched:** v137

### FEAT-01 â€” DTWEXBGS Post-v128 Feature Search
**Status:** open
**Priority:** medium
**Rationale:** Currency momentum remains a plausible benchmark-specific feature addition after the v128 feature-map work.
**Estimated effort:** S
**Depends on:** none
**Expected metric impact:** per-benchmark BA and ECE up slightly
**Last touched:** v128

### FEAT-02 â€” WTI 3M Momentum For DBC/VDE
**Status:** open
**Priority:** medium
**Rationale:** Energy/commodity benchmarks likely still underuse oil-specific state information.
**Estimated effort:** M
**Depends on:** external series verification
**Expected metric impact:** DBC/VDE benchmark R2 and BA up slightly
**Last touched:** 2026-04-13 plan

### DATA-01 â€” EDGAR Filing Lag Review
**Status:** complete
**Priority:** high
**Rationale:** `v142` re-validated the EDGAR lag guard on the post-v138 frame and found that the incumbent `lag=2` still gives the best balanced pooled result even though `lag=1` improves IC in isolation.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** reliability evidence refreshed; no promotion candidate
**Last touched:** v142

### INFRA-01 â€” Conformal Coverage Calibration
**Status:** complete
**Priority:** medium
**Rationale:** The bounded `v144` replay grid found a cleaner realized-coverage candidate at `{"coverage": 0.75, "aci_gamma": 0.03}` than the original `0.80 / 0.05` starting point.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** interval coverage quality up, recommendation confidence quality up
**Last touched:** v144

### REG-03 â€” Fixed Ridge-vs-GBT Blend Weight Sweep
**Status:** complete
**Priority:** medium
**Rationale:** The first bounded `v141` sweep found a better fixed research candidate at `ridge_weight=0.60`, improving pooled OOS R2, IC, and hit rate versus the midpoint baseline.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** pooled OOS R2 up modestly, calibration up
**Last touched:** v141

### REG-04 â€” Correlation-Pruned Feature Overrides
**Status:** complete
**Priority:** medium
**Rationale:** The bounded `v143` sweep found that `rho=0.80` improved pooled OOS R2, IC, and hit rate relative to the incumbent pruning threshold on the tested frame.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** pooled OOS R2 up slightly, sigma ratio down
**Last touched:** v143

### CLS-04 â€” Threshold Follow-Through On Tuned Path B Baseline
**Status:** complete
**Priority:** high
**Rationale:** `v146` confirmed that the incumbent threshold pair remains the strongest bounded candidate on top of the tuned `v135` temperature baseline.
**Estimated effort:** S
**Depends on:** v135 complete
**Expected metric impact:** covered BA up, coverage quality up
**Last touched:** v146

### CLS-05 â€” Coverage-Weighted Path A / Path B Aggregate Proxy
**Status:** complete
**Priority:** medium
**Rationale:** `v147` tested the preserved aggregate probability frame, but none of the bounded multipliers improved covered balanced accuracy above the baseline proxy.
**Estimated effort:** S
**Depends on:** CLS-04
**Expected metric impact:** covered BA up modestly at acceptable coverage
**Last touched:** v147
