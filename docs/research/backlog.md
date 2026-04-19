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

1. `CLS-03` â€” Path A vs Path B production decision â€” blocked (24 matured months)
2. `v159` — Wire Firth logistic for VMBS/BND into shadow classification lane — **complete (2026-04-18)**
3. `BL-01` - Black-Litterman tau/view tuning — **complete (2026-04-18)**
4. `CLS-03` — Path A vs Path B production decision — blocked (24 matured months)

Leave `CLS-03` blocked on the matured-month gate, and leave `REG-02` deferred
until a future ensemble-level plan justifies reopening the standalone GBT line.

### TA-01 — Alpha Vantage Technical-Analysis Broad Feature Research
**Status:** complete
**Priority:** medium
**Rationale:** Three external TA reports converged on a low-prior but defensible
one-cycle broad screen if redundant Alpha Vantage indicator families are pruned
before modeling and all tests remain WFO-only.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** classification replacement candidates identified
**Last touched:** v164 (2026-04-18)
**Outcome:** replacement_candidate. No production or shadow change, but the
v162/v163 artifacts justify a separate shadow-only replacement plan.

### TA-02 - TA Classification Replacement Shadow Plan
**Status:** complete
**Priority:** medium
**Rationale:** TA-01 found the strongest signal in replacement-style
classification candidates rather than additive feature expansion.
**Estimated effort:** M
**Depends on:** TA-01
**Expected metric impact:** balanced accuracy and Brier improvement in
historical shadow diagnostics
**Last touched:** v165 (2026-04-18)
**Candidate scope:** test `mom_12m -> ta_pgr_obv_detrended`,
`vol_63d -> ta_pgr_natr_63d`, and one representative ratio Bollinger feature
under reporting-only shadow constraints.
**Outcome:** shadow_monitor. `ta_minimal_plus_vwo_pct_b` was strongest
historically (+0.0584 mean BA, -0.0656 mean Brier, 8/8 positive benchmarks),
while `ta_minimal_replacement` remains the simpler comparison.

### TA-03 - Monthly TA Shadow Artifact Lane
**Status:** complete
**Priority:** medium
**Rationale:** v165 produced promising historical replacement diagnostics, but
the signal needs prospective monthly monitoring before any promotion discussion.
**Estimated effort:** M
**Depends on:** TA-02
**Expected metric impact:** governance and monitoring evidence; no immediate
production impact
**Last touched:** v166 (2026-04-18)
**Candidate scope:** write reporting-only monthly artifacts for
`ta_minimal_plus_vwo_pct_b` and `ta_minimal_replacement`. Append prospective
probabilities to history without changing production recommendations, sell
percentages, or classifier gate overlays.
**Outcome:** complete. TA replacement variants are included in
`classification_shadow.csv` and `monthly_summary.json` as reporting-only rows.
The weekly data workflow verifies required `PGR` and `VWO` price coverage.

### TA-04 - TA Shadow Monitoring Ledger
**Status:** complete
**Priority:** medium
**Rationale:** Monthly TA shadow rows need a durable prospective ledger so the
project can evaluate matured calibration and accuracy later without scraping
point-in-time monthly JSON artifacts.
**Estimated effort:** S
**Depends on:** TA-03
**Expected metric impact:** governance and monitoring evidence; no immediate
production impact
**Last touched:** v167 (2026-04-18)
**Candidate scope:** append/upsert reporting-only TA variant probabilities by
`as_of_date` and `variant`, preserve 6M maturity dates, and reserve realized
outcome fields for future evaluation.
**Outcome:** complete. Monthly runs write
`results/monthly_decisions/ta_shadow_variant_history.csv`; production
recommendations and classifier gate overlays remain unchanged.

### OPS-01 - Calendar-Aware PGR Monthly EDGAR Freshness
**Status:** complete
**Priority:** high
**Rationale:** The monthly report warned that PGR EDGAR data was stale before
the normal prior-month 8-K filing window and scheduled fetch fallback had
elapsed.
**Estimated effort:** S
**Depends on:** none
**Expected metric impact:** operational reliability; fewer false stale warnings
**Last touched:** v168 (2026-04-18)
**Outcome:** complete. `check_data_freshness` now expects the prior month only
after a 25-day filing grace window, and the monthly 8-K workflow enforces the
same rule after fetch attempts.

### OPS-02 - Monthly Postcondition Verifier
**Status:** complete
**Priority:** medium
**Rationale:** Monthly artifact, freshness, and TA shadow ledger postconditions
should be testable Python rather than repeated inline GitHub Actions glue.
**Estimated effort:** S
**Depends on:** OPS-01, TA-04
**Expected metric impact:** operational reliability; faster detection of
missing reporting artifacts
**Last touched:** v169 (2026-04-19)
**Outcome:** complete. `scripts/verify_monthly_outputs.py` verifies required
monthly artifacts, calendar-aware freshness, reporting-only TA variants, and
matching TA ledger rows.

### OPS-03 - Documentation Hygiene And Archive Map
**Status:** complete
**Priority:** medium
**Rationale:** The docs tree had active, legacy, and archive material without a
single navigation map, making old plans/results look current.
**Estimated effort:** S
**Depends on:** OPS-02
**Expected metric impact:** operational clarity; lower onboarding and review
cost
**Last touched:** v170 (2026-04-19)
**Outcome:** complete. Added `docs/README.md`, legacy directory labels,
current archive guidance, and `docs/repo-hygiene-review-2026-04-19.md`.

### BL-01 — Black-Litterman Tau/View Tuning
**Status:** complete
**Priority:** medium
**Rationale:** The decision layer still uses untuned BL priors even though the regression and classifier research stack has moved materially since the original defaults.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** recommendation accuracy up, policy uplift up modestly
**Last touched:** 2026-04-18
**Outcome:** keep_incumbent — tau=0.05, risk_aversion=2.5 are already near-optimal (rank_corr=0.8643, best alternative delta=+0.009 below 0.05 threshold). No config change. Side effect: `risk_free_rate` parameter added to `build_bl_weights`.

### CLS-01 â€” SCHD Per-Benchmark Classifier Addition
**Status:** deferred
**Priority:** medium
**Rationale:** SCHD remains outside the current benchmark-specific classifier path because of history depth, but it is still investable in the real redeploy portfolio.
**Estimated effort:** M
**Depends on:** CLS-03
**Expected metric impact:** better portfolio alignment, unknown BA impact
**Last touched:** v124-v128 cycle

### CLS-02 — Firth Logistic For Short-History Benchmarks
**Status:** complete
**Priority:** medium
**Rationale:** Short-history benchmarks still risk unstable logistic fits under small-sample class imbalance.
**Estimated effort:** M
**Depends on:** none
**Expected metric impact:** VMBS BA_cov +0.0412, BND BA_cov +0.0704; shadow adoption pending v159
**Last touched:** v154 (2026-04-17)

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

### FEAT-01 — DTWEXBGS Post-v128 Feature Search
**Status:** complete
**Priority:** medium
**Rationale:** Currency momentum remains a plausible benchmark-specific feature addition after the v128 feature-map work.
**Estimated effort:** S
**Depends on:** none
**Expected metric impact:** no benefit observed (BND -0.077, VXUS flat, VWO +0.009)
**Last touched:** v156 (2026-04-17)

### FEAT-02 — WTI 3M Momentum For DBC/VDE
**Status:** complete
**Priority:** medium
**Rationale:** Energy/commodity benchmarks likely still underuse oil-specific state information.
**Estimated effort:** M
**Depends on:** external series verification
**Expected metric impact:** no benefit at 0.04 threshold (DBC +0.005, VDE +0.021)
**Last touched:** v155 (2026-04-17)

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

### FEAT-03 — Term Premium 3M Differential Signal
**Status:** complete
**Priority:** medium
**Rationale:** The 10Y term premium is already in the feature matrix as `term_premium_10y`, but a 3-month change (`term_premium_diff_3m`) has not been tested. Sudden jumps in term premium historically signal equity headwinds; the differenced series may improve classifier timing without requiring new data.
**Estimated effort:** S
**Depends on:** none
**Expected metric impact:** no benefit at 0.02 threshold (best VDE +0.017)
**Last touched:** v157 (2026-04-17)
