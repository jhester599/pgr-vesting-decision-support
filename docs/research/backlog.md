### BL-01 — Black-Litterman Tau/View Tuning
**Status:** open
**Priority:** medium
**Rationale:** The decision layer still uses untuned BL priors even though the regression and classifier research stack has moved materially since the original defaults.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** recommendation accuracy up, policy uplift up modestly
**Last touched:** 2026-04-13 plan

### CLS-01 — SCHD Per-Benchmark Classifier Addition
**Status:** deferred
**Priority:** medium
**Rationale:** SCHD remains outside the current benchmark-specific classifier path because of history depth, but it is still investable in the real redeploy portfolio.
**Estimated effort:** M
**Depends on:** CLS-03
**Expected metric impact:** better portfolio alignment, unknown BA impact
**Last touched:** v124-v128 cycle

### CLS-02 — Firth Logistic For Short-History Benchmarks
**Status:** open
**Priority:** medium
**Rationale:** Short-history benchmarks still risk unstable logistic fits under small-sample class imbalance.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** classifier calibration up, covered BA up slightly
**Last touched:** 2026-04-13 plan

### CLS-03 — Path A vs Path B Production Decision
**Status:** blocked
**Priority:** high
**Rationale:** Path B now has stronger historical evidence, but the 24 matured prospective-month gate remains time-locked.
**Estimated effort:** M
**Depends on:** 24 matured months of shadow monitoring
**Expected metric impact:** governance milestone, not immediate backtest uplift
**Last touched:** v135

### REG-01 — Ensemble-Level Clip/Shrink Blend Revisit
**Status:** open
**Priority:** medium
**Rationale:** v38 shrinkage was strong historically, but the live regression frame has drifted and may benefit from a refreshed calibration layer.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** pooled OOS R2 up modestly, calibration up
**Last touched:** v38-v39

### REG-02 — Rank-Normalized GBT With New Search Region
**Status:** deferred
**Priority:** low
**Rationale:** The fresh v137 sweep improved standalone GBT with shallow trees, but it still fell short of the old success bar and should not be promoted before ensemble-level validation.
**Estimated effort:** M
**Depends on:** v137 ensemble follow-up
**Expected metric impact:** low-confidence OOS R2 improvement
**Last touched:** v137

### FEAT-01 — DTWEXBGS Post-v128 Feature Search
**Status:** open
**Priority:** medium
**Rationale:** Currency momentum remains a plausible benchmark-specific feature addition after the v128 feature-map work.
**Estimated effort:** S
**Depends on:** none
**Expected metric impact:** per-benchmark BA and ECE up slightly
**Last touched:** v128

### FEAT-02 — WTI 3M Momentum For DBC/VDE
**Status:** open
**Priority:** medium
**Rationale:** Energy/commodity benchmarks likely still underuse oil-specific state information.
**Estimated effort:** M
**Depends on:** external series verification
**Expected metric impact:** DBC/VDE benchmark R2 and BA up slightly
**Last touched:** 2026-04-13 plan

### DATA-01 — EDGAR Filing Lag Review
**Status:** open
**Priority:** high
**Rationale:** The FRED lag sweep showed that data-timing assumptions matter; EDGAR filing lag is another high-leverage leakage guard worth re-validating.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** reliability up, possible regression metric changes
**Last touched:** 2026-04-13 plan

### INFRA-01 — Conformal Coverage Calibration
**Status:** open
**Priority:** medium
**Rationale:** Conformal intervals are still configured statically and have not been re-calibrated against the newer ensemble paths.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** interval coverage quality up, recommendation confidence quality up
**Last touched:** v5.x
