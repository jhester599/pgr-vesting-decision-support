# v123–v128 Classification Enhancement Plan

Created: 2026-04-12

## Source Materials

This plan synthesizes two independent deep research reports commissioned on 2026-04-12:

- [`docs/archive/history/peer-reviews/2026-04-12/claude_opus_peerreview_20260412.md`](../archive/history/peer-reviews/2026-04-12/claude_opus_peerreview_20260412.md)
- [`docs/archive/history/peer-reviews/2026-04-12/chatgpt_peerreview_20260412.md`](../archive/history/peer-reviews/2026-04-12/chatgpt_peerreview_20260412.md)

The deep research prompt used is archived at:

- [`docs/superpowers/plans/2026-04-12-v123-classification-enhancement-research-prompt.md`](2026-04-12-v123-classification-enhancement-research-prompt.md)

---

## Where The Reports Agree

Both reviewers reached the same conclusions on every major architectural question
except primary research path priority (see disagreement section below).

**Portfolio alignment gap is the highest-value fix:**
The current classifier aggregates per-benchmark probabilities using regression-layer
quality weights. Both reports confirm this is methodologically misaligned — the user's
actual decision is "will I be better off selling PGR and buying my redeploy portfolio?"
not "will PGR underperform a generic forecast-benchmark set?" Closing this gap is the
single highest-priority change.

**Portfolio-weighted aggregation is statistically valid (with recalibration):**
A linear opinion pool `P_composite = Σ(w_i × P_i)` using fixed redeploy weights is
valid per Geweke & Amisano (2011). However, both reports note that a weighted average
of individually calibrated classifiers is not automatically calibrated at the portfolio
level — a second-stage portfolio-level recalibration step is required.

**Fixed redeploy weights are correct:**
Both reports recommend using fixed `balanced_pref_95_5` base weights (renormalized to
sum to 1 across the investable benchmark set). Dynamic monthly signal-tilted weights
would create circular dependency and amplify estimation noise. The forecast combination
puzzle (DeMiguel et al. 2009) confirms that estimated optimal weights require far more
observations than are available.

**Non-investable benchmarks belong in diagnostics, not the primary signal:**
DBC, GLD, VMBS, and VDE should be excluded from the primary `P(Actionable Sell)` and
retained as a regime-conditioning diagnostic overlay. Including non-investable assets
in the portfolio-level probability makes the composite answer a different question than
the user is actually asking.

**VGT should be added now:**
VGT inception is January 2004 (~264 monthly observations pre-holdout), comparable to
existing benchmarks. The EPV concern is real but manageable with reduced feature sets
(6–8 features) and strong regularization. Both reports confirm inclusion is appropriate
now.

**SCHD timing diverges slightly (see below), but both agree caution is warranted:**
SCHD has ~168 observations with ~50 positive events at 30% base rate, yielding EPV ≈ 4.2
with 12 features — below the defensible threshold. Both reports flag this risk.

**The calibration–balanced accuracy conflict has a clear cause and fix:**
Post-hoc prequential logistic calibration improves Brier/ECE by pulling predictions
toward the base rate (~0.30), but moves borderline positives below the abstention band's
lower threshold, reducing sensitivity. This is the Murphy (1973) reliability–resolution
tradeoff. Both reports recommend replacing the current calibration approach. Temperature
scaling (Claude) and Platt scaling on logits (ChatGPT) both preserve prediction ranking
and cannot degrade balanced accuracy — they are equivalent approaches for logistic
regression and should be tested jointly.

**Benchmark-specific feature subsetting from the lean baseline is appropriate:**
Feature expansion (adding new features) was correctly ruled out by v88. Feature
subsetting (using the L1→L2 two-step to select 6–8 of the 12 lean features per
benchmark) is a different and appropriate intervention. The Hughes peaking phenomenon
predicts a 5–8 feature optimum at N ≈ 200. L1 selection within each WFO training fold
is methodologically valid and does not introduce leakage.

**Promotion remains premature:**
Current matured prospective monitoring is effectively zero months. Both reports agree
the minimum threshold is 24 matured prospective months with stable calibration, covered
balanced accuracy ≥ 0.60, and actionable-sell precision ≥ 0.70. The veto gate is the
right first production role given the asymmetric cost structure (false positive =
unnecessary tax event >> false negative = missed diversification).

---

## Where The Reports Disagree

**Primary architecture: Path A vs Path B**

- **Claude report**: Path A (per-benchmark classifiers → portfolio-weighted aggregation)
  as primary. Rationale: disaggregate forecasting outperforms direct aggregate forecasting
  when components are genuinely heterogeneous (Marcellino, Stock & Watson 2003; Lütkepohl
  1984). Benchmark heterogeneity is real (bond vs. commodity vs. equity drivers differ
  substantially). Per-benchmark structure preserves sleeve-level interpretability.

- **ChatGPT report**: Path B (single classifier on composite portfolio-return target)
  as primary. Rationale: Path B directly answers the user's decision question with no
  aggregation design choices. Composite target has lower idiosyncratic variance via
  diversification. Operationally simpler (one model, one calibration, one abstention
  policy).

**Resolution:** Run both paths in parallel from v125. This is not additional complexity
— it is the correct way to settle the empirical question. Path A preserves the existing
shadow monitoring infrastructure and interpretability. Path B tests the composite target
hypothesis directly. Both are research-only phases and produce comparable metrics. The
architecture decision can be made after v125–v126 results are in hand, with the winner
selected based on balanced accuracy, Brier score, and calibration on the pre-holdout
period.

**SCHD timing:**

- **Claude report**: Defer to v135–v140 when history reaches ~185 observations.
- **ChatGPT report**: Include SCHD now in Path B by construction (it is part of the
  portfolio return composite), but do not give it Path A per-benchmark weight until
  stability criteria are met.

**Resolution:** ChatGPT's framing is more precise. SCHD enters the portfolio target
(Path B) in v125 automatically — no special decision needed, since the composite return
already includes it. For Path A, treat SCHD as a future per-benchmark addition (v135+).
Begin constructing SCHD target and proxy validation immediately as a background task.

---

## Recommended Architecture for v123+

**Primary signal (production candidate):** Path A with portfolio-weighted aggregation,
restricted to investable benchmarks, with portfolio-level recalibration. This is the
evolution of the existing shadow infrastructure and has the most interpretability value.

**Secondary signal (parallel research track):** Path B composite portfolio target
classifier. Runs in parallel from v125. Evaluated head-to-head against Path A on
all standard metrics. Promoted to co-primary only if it demonstrates ≥ 3% balanced
accuracy improvement with comparable calibration.

**Investable benchmark set:** VOO, VGT, VXUS, VWO, BND (5 funds with sufficient
history now). SCHD enters Path B by construction; Path A SCHD deferred to v135+.

**Contextual benchmarks (diagnostics only):** DBC, GLD, VMBS, VDE. Continue running
per-benchmark classifiers for these; output appears in a "regime diagnostics" table
separate from the primary `P(Actionable Sell)` signal.

**Calibration:** Replace prequential logistic calibration with temperature scaling /
Platt scaling on logits (test both, keep better). Apply class_weight tuning during
training to improve balanced accuracy before any post-hoc calibration step.

**Thresholds:** Replace uniform 0.30–0.70 abstention band with asymmetric cost-sensitive
thresholds selected within WFO validation folds. Sell threshold should be higher than
hold threshold given false-positive cost asymmetry. Pooled thresholds (not per-benchmark)
for stability at N ≈ 200.

**Feature approach:** Benchmark-specific subsetting from lean baseline (L1→L2 two-step)
in v127. New feature candidates (USD momentum, term premium, oil momentum, equity risk
premium) deferred to v128+ after subsetting is validated.

---

## Implementation Plan

### v123 — Portfolio alignment plumbing (research + shadow additions)

**Objective:** Add portfolio-weighted investable aggregate as a new shadow column
alongside the existing quality-weighted aggregate. No model changes. No production
changes.

**Research tasks:**
- Implement investable-only portfolio-weighted aggregation over {VOO, VXUS, VWO, BND}
  (the 4 benchmarks currently in both the classifier and investable universe)
- Fixed weights: renormalize `balanced_pref_95_5` base weights to sum to 1 over
  this subset
- Add second-stage Platt/logistic portfolio-level recalibration for the new aggregate
- Reclassify DBC, GLD, VMBS, VDE outputs as "contextual regime diagnostics" in artifacts
- Begin SCHD relative-return series construction and target labeling (background task)

**Shadow additions:**
- New columns in `classification_shadow.csv`:
  - `classifier_prob_investable_pool` (unrecalibrated portfolio-weighted)
  - `classifier_prob_investable_pool_calibrated` (after portfolio-level recalibration)
  - `classifier_tier_investable_pool`, `classifier_stance_investable_pool`
- New contextual diagnostics table (separate from primary signal):
  - per-benchmark probabilities for DBC, GLD, VMBS, VDE with explicit "contextual only" label

**Validation:**
- Unit test: investable weights sum to 1; contextual benchmark weights are zero in the
  primary aggregate
- Unit test: as-of truncation applies before portfolio-level recalibration
- Comparison report: quality-weighted vs investable-pool aggregate on full shadow history

**Files likely to change:**
- `src/models/classification_shadow.py`
- `src/reporting/classification_artifacts.py`
- `scripts/monthly_decision.py`

---

### v124 — VGT benchmark classifier addition (shadow expansion)

**Objective:** Add VGT to the per-benchmark classifier suite so the investable pool
covers {VOO, VGT, VXUS, VWO, BND}.

**Research tasks:**
- Verify VGT relative-return series exists in DB and passes as-of truncation
- Train per-benchmark logistic classifier for PGR-vs-VGT (lean 12-feature baseline)
- Report: OOS balanced accuracy, Brier, calibration curve vs. existing benchmarks
- Validate no coefficient explosion or complete separation
- Update investable-pool aggregation to include VGT with its `balanced_pref_95_5`
  portfolio weight (renormalized over 5-fund set)

**Validation:**
- VGT classifier balanced accuracy > 50% and Brier < 0.25 required for inclusion
- Coefficient signs stable across ≥ 60% of WFO folds
- EPV check: with ~264 observations and 12 features, EPV ≈ 6.6 — document this
  and flag as "borderline; reduce to 8 features if instability observed"

**Files likely to change:**
- `src/models/classification_shadow.py` (benchmark list, aggregation weights)
- `config.py` (investable benchmark set constant)
- `scripts/monthly_decision.py`

---

### v125 — Path B prototype: composite portfolio-target classifier (parallel research)

**Objective:** Train and evaluate a single logistic classifier on the composite
`balanced_pref_95_5`-weighted portfolio return target. Run in parallel with the
existing Path A shadow.

**Research script:** `results/research/v125_portfolio_target_classifier.py`

- Build portfolio relative-return series: `rr_port = r_pgr - Σ(w_i × r_i)`
  using fixed `balanced_pref_95_5` weights over {VOO, VGT, SCHD, VXUS, VWO, BND}
- Apply `truncate_relative_target_for_asof` for 6-month horizon
- Binary label: `actionable_sell_3pct` on `rr_port`
- WFO: `TimeSeriesSplit` with `max_train=60`, `test_size=6`, `gap=8`
- Model: L2 logistic regression, `class_weight="balanced"`, lean 12-feature baseline
- Metrics: balanced accuracy, precision/recall, Brier, log loss, ECE, coverage
- Comparison table vs Path A quality-weighted baseline on matched fold dates

**Shadow addition:**
- New monthly column: `classifier_prob_portfolio_target` with tier and stance
- Explicitly labeled "Path B — composite target (research shadow)"

**Validation:**
- Portfolio target computed from data available at time t only; no future dynamic weights
- Path B Brier and balanced accuracy compared to Path A investable pool; Path A remains
  primary unless Path B shows ≥ 3% balanced accuracy improvement

---

### v126 — Calibration overhaul and asymmetric threshold optimization

**Objective:** Resolve the calibrated balanced accuracy degradation by replacing
prequential logistic calibration with a rank-preserving approach and replacing the
uniform abstention band with cost-sensitive thresholds.

**Research tasks:**
- Implement and compare two calibrators at portfolio signal level (not per-benchmark):
  - Temperature scaling (single parameter T on logits)
  - Platt scaling on logits (two-parameter, logit-space)
  - Baseline: current prequential logistic on probabilities
- Implement asymmetric threshold band sweep within WFO:
  - Symmetric candidates: (0.25, 0.75), (0.30, 0.70), (0.35, 0.65)
  - Asymmetric / tax-averse: (0.25, 0.80), (0.20, 0.80), (0.20, 0.75)
- Selection rule inside WFO training fold:
  - Maximize balanced accuracy on covered months
  - Subject to: actionable-sell precision ≥ 0.70 and coverage ≥ 0.25
- Apply selected calibrator + thresholds to both Path A investable pool and Path B

**Outputs:** `results/research/v126_calibration_threshold_results.csv` + summary md

**Validation:**
- Temperature/Platt scaling must not reduce balanced accuracy (by construction, but verify)
- If neither calibration method improves Brier by > 0.005 over no post-hoc calibration,
  prefer no post-hoc calibration and rely on class weighting alone
- Selected thresholds must be stable: if per-fold variation exceeds ± 0.10, use pooled
  default rather than estimated thresholds

---

### v127 — Benchmark-specific feature subsetting (L1→L2 within WFO)

**Objective:** Reduce per-benchmark feature sets from 12 to 6–8 using L1→L2 two-step
selection, targeting the Hughes peaking optimum for N ≈ 200 with weak financial signals.

**Research tasks:**
- Implement stability selection: run L1 across 50 bootstrap resamples within each WFO
  training fold; retain features appearing in > 60% of resamples
- Conservative λ for L1 step: select λ that retains 6–8 features; do not tune λ via
  nested CV (too much variance at N ≈ 200)
- Two-step procedure: LASSO identifies feature subset, Ridge fits final model on subset
- Report: selected features per benchmark, fold-to-fold consistency, balanced accuracy
  vs. full 12-feature baseline

**Acceptance gate:** Benchmark-specific subsetting must show balanced accuracy ≥ baseline
on average across benchmarks. If subsetting degrades overall balanced accuracy, revert to
lean 12-feature baseline.

**Outputs:** `results/research/v127_feature_subsetting_results.csv` + summary md

**Note on new features:** Do not add features beyond the lean baseline in this version.
New feature candidates (USD momentum, term premium, WTI oil momentum, equity risk premium)
are staged for v128+ after subsetting is validated and their value can be measured in
isolation.

---

### v128 — New feature candidates and promotion governance

**Objective:** (a) Test the highest-priority benchmark-specific new features identified
by both peer reviews. (b) Implement the promotion gate infrastructure.

**Research tasks (feature candidates — one per asset class, no-new-dependency only):**
- USD momentum 6m (`DTWEXBGS` FRED series): for VXUS, VWO, and GLD
  - DTWEXBGS is already in the FRED series list based on feature builder references
- ACM 10Y term premium (`THREEFYTP10` FRED series): for BND, VMBS, and VGT
  - Already referenced in the feature builder
- Equity risk premium proxy (earnings yield − GS10): for VOO and VGT
  - Earnings yield available via Multpl series already in ingestion
- WTI oil price momentum 3m: for DBC and VDE
  - Requires verifying WTI series availability (EIA or FRED)

**Acceptance gate per candidate:** ≥ +0.02 absolute balanced accuracy lift, ECE not worse
by > 0.01, coefficient sign stability ≥ 70% across WFO folds. Reject otherwise.

**Promotion governance:**
- Implement `classification_gate_overlay.py` with optional limited veto rule
- `monthly_decision.py`: include gate check output as diagnostic; do not change live sell
  logic until `promotion_approved = True` flag is set explicitly
- Document promotion criteria (below) in `docs/model-governance.md`

**Promotion gate criteria (hard requirements, all must be met):**
1. ≥ 24 matured prospective months (real-time predictions with elapsed 6-month horizon)
2. Calibrated Brier ≤ 0.18 and ECE ≤ 0.08 on matured months, no upward trend
3. Covered-month balanced accuracy ≥ 0.60 at coverage ≥ 0.25
4. Actionable-sell precision ≥ 0.70 (limit false positives / unnecessary tax events)
5. Agreement rate with regression baseline in range [0.80, 0.97]
6. Non-negative cumulative policy uplift on disagreement months
7. No single disagreement month with > 5% adverse portfolio impact vs. baseline

**First production role (once gate passes):** Limited veto gate. Regression sell signal
requires classifier confirmation (portfolio-target probability > 0.30) before execution.
This primarily reduces false positives at the cost of some false negatives — the correct
asymmetry for a taxable account.

---

### v130+ — SCHD preparation and per-benchmark inclusion (conditional)

- Continue SCHD target series construction and proxy validation (Dow Jones US Select
  Dividend Index as proxy for longer history)
- Evaluate Firth's penalized regression as alternative estimator for short-history benchmarks
- Add SCHD as Path A per-benchmark classifier when history reaches ≥ 185 observations
  (~late 2027 based on current trajectory)
- Promotion gate for SCHD same as VGT: balanced accuracy > 50%, Brier < 0.25, coefficient
  stability ≥ 60% of folds

---

## Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Portfolio-weight aggregation over fewer benchmarks (4→5) loses informational breadth | Retain contextual benchmarks as regime diagnostics; monitor regime-signal agreement |
| Temperature/Platt scaling provides < 0.005 Brier improvement for logistic regression | Fall back to class weighting only (no post-hoc calibration); both reviewers agree this is acceptable |
| Benchmark-specific feature subsetting produces unstable selections across WFO folds | Stability selection (> 60% bootstrap consistency) as hard gate; revert to lean baseline if gate fails |
| VGT EPV borderline (≈ 6.6 with 12 features) produces unstable coefficients | Reduce to 8 features in v127 subsetting step; monitor coefficient stability across folds |
| Path B composite target label noise degrades classification at N ≈ 200 | Parallel evaluation in v125; only promote if ≥ 3% balanced accuracy improvement |
| High regression-classifier agreement rate (93.83%) means promotion adds minimal decision value | Decompose the 10 disagreement months; if uplift is concentrated in 2–3 regime-change events, the classifier has tail-event value that justifies the veto gate architecture |

---

## Documentation Updates Required

As each version completes, the following documents should be updated:

- `ROADMAP.md`: update active research direction and backlog
- `CHANGELOG.md`: add version entry with theme and key deliverables
- `docs/model-governance.md`: add promotion gate criteria once v128 is complete
- `results/research/`: add per-version summary markdown + CSV
- Monthly artifacts: add new shadow columns per v123 and v124 changes

## v125 Empirical Result (Path A vs Path B)

**Evaluation date:** 2026-04-12
**As-of cutoff:** 2024-03-31

**Architecture Verdict:** Path B shows >= 3% balanced accuracy improvement over Path A reference. Elevate Path B to co-primary research track.

**Metrics:**
- Path B balanced accuracy (covered): 0.6331
- Path A reference (v92): 0.5132
- Delta: +0.1199
- Path B Brier score: 0.2393 vs Path A reference: 0.1852
