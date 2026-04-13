# Deep Research Prompt: Classification Enhancement — v123 Cycle

Use the prompt below with a deep research model (Claude Opus, ChatGPT o3, Gemini Advanced).
Focused on the next classification improvement cycle. Does **not** repeat the full-repo
scope of the v97 prompt — assumes familiarity with the existing project structure and
addresses specific open questions from the v87–v122 research arc.

---

## Copy-Paste Prompt

You are conducting a targeted deep research review of a binary classification decision-support
system embedded in the GitHub repository `jhester599/pgr-vesting-decision-support`.

This is not a general repo review. It is focused on a specific set of open design questions
about how the classification layer should be improved in the next research cycle.

---

### Project Context

This repository supports a monthly `hold vs. sell` decision for a user holding a concentrated
position in Progressive Corporation (`PGR`) equity in a taxable brokerage account. RSUs vest
periodically. At each vest event, the user decides whether to hold vested shares or sell them
and redeploy proceeds into a diversified ETF portfolio.

The system has two layers:

**Regression layer (production):**
A quality-weighted consensus of per-benchmark Walk-Forward Optimization (WFO) regression
models (Ridge + shallow GBT ensemble) that forecast PGR's relative return vs. each ETF
benchmark. The consensus uses OOS R², Information Coefficient, and hit-rate to weight
benchmark forecasts into a final recommendation mode (ACTIONABLE / MONITORING-ONLY /
DEFER-TO-TAX-DEFAULT).

**Classification layer (shadow only, v87–v122):**
A binary classifier trained per benchmark on the `actionable_sell_3pct` target — whether
the realized outcome for that month justified a sell action (PGR underperformed the benchmark
by more than a 3% cost-of-carry threshold). One logistic model per benchmark, trained on
all available history up to the as-of month with explicit target truncation to prevent
leakage. Probabilities are calibrated using a prequential logistic calibration step and
aggregated into a single `P(Actionable Sell)` using benchmark-quality weights derived from
the regression layer.

The classification layer is currently shadow-only. The v96 conclusion was
`continue_research_no_promotion` — promising but not stable enough to override production.

---

### Current Classifier State (v122 Audit — April 2026)

**Feature set:** `lean_baseline` (12 features, same for all 8 benchmarks):
- `mom_12m`, `vol_63d`
- `yield_slope`, `real_yield_change_6m`, `real_rate_10y`, `credit_spread_hy`, `nfci`, `vix`
- `combined_ratio_ttm`, `investment_income_growth_yoy`, `book_value_per_share_growth_yoy`, `npw_growth_yoy`

**Benchmark coverage (8):** BND, DBC, GLD, VDE, VMBS, VOO, VWO, VXUS

**Per-benchmark training size:** 176–252 rows (monthly observations)

**Pooled classifier performance:**
- Accuracy: 68.89%, Balanced accuracy: 58.27%, Brier: 0.2278

**Calibrated shadow path performance:**
- Accuracy: 75.38%, Balanced accuracy: 51.32%, Brier: 0.1852, ECE: 0.0813

**Aggregation method:** benchmark-quality weights derived from regression OOS metrics
(not from classifier-specific quality, and not from user portfolio allocation)

**Top feature importances (standardized logistic coefficients, quality-weighted across benchmarks):**
1. `combined_ratio_ttm`: 1.245 — dominant across all benchmarks
2. `credit_spread_hy`: 0.653
3. `real_yield_change_6m`: 0.507
4. `mom_12m`: 0.378
5. `yield_slope`: 0.349
6. `real_rate_10y`: 0.310

**Per-benchmark top-3 drivers from latest audit:**
- BND: `combined_ratio_ttm` (1.088), `book_value_per_share_growth_yoy` (0.397), `yield_slope` (0.367)
- DBC: `combined_ratio_ttm` (1.677), `mom_12m` (0.720), `vix` (0.653)
- GLD: `combined_ratio_ttm` (1.208), `real_rate_10y` (0.512), `nfci` (0.393)
- VDE: `combined_ratio_ttm` (1.229), `real_yield_change_6m` (0.487), `real_rate_10y` (0.484)
- VMBS: `combined_ratio_ttm` (1.197), `yield_slope` (0.723), `credit_spread_hy` (0.597)
- VOO: `combined_ratio_ttm` (1.151), `credit_spread_hy` (0.850), `real_yield_change_6m` (0.770)
- VWO: `credit_spread_hy` (1.768), `combined_ratio_ttm` (1.527), `real_yield_change_6m` (0.674)
- VXUS: `combined_ratio_ttm` (0.883), `credit_spread_hy` (0.749), `real_yield_change_6m` (0.665)

---

### The Portfolio Alignment Gap

A critical structural issue has been identified that the v87–v122 cycle did not address.

The repository makes a deliberate distinction between two ETF universes (established in v27):

**Forecast benchmark universe (used for modeling):**
BND, DBC, GLD, VDE, VMBS, VOO, VWO, VXUS

**Investable redeploy universe (what the user actually buys with sell proceeds):**
VOO, VGT, SCHD, VXUS, VWO, BND

The winning v27 redeploy portfolio is `balanced_pref_95_5` — a bounded signal-tilted
portfolio with approximate base weights:
- VOO ~35–40% (US large-cap core)
- VXUS ~20–25% (international developed + EM)
- VGT ~15–20% (technology tilt)
- SCHD ~10–15% (dividend/value tilt)
- VWO ~5–10% (emerging markets satellite)
- BND ~5% (bond ballast)

The exact monthly weights shift modestly based on signal confidence but stay within
bounded ranges and maintain >90% equity exposure.

**The problem:**

1. The current classifier aggregation uses regression-layer quality weights — not the
   user's redeploy portfolio weights. This means the composite `P(Actionable Sell)` signal
   is not directly answering "will I be better off selling PGR and buying my target portfolio?"

2. VGT and SCHD are completely absent from the classifier benchmark suite. These are the
   two most differentiated sleeves of the user's intended portfolio (tech tilt and
   value/dividend tilt), yet they contribute zero weight to the classifier signal.

3. DBC, GLD, VMBS, and VDE are in the classifier but are NOT in the investable redeploy
   portfolio. They are retained as contextual benchmarks for model diagnostics, but their
   inclusion in the aggregated `P(Actionable Sell)` signal means the signal reflects
   comparisons to assets the user would never actually purchase.

---

### Established Constraints

Respect these constraints throughout your recommendations:

- **No K-Fold validation.** All time-series models must use Walk-Forward Optimization with
  strict purge/embargo periods (`sklearn.model_selection.TimeSeriesSplit`).
- **No temporal leakage.** Targets that require future prices must be explicitly truncated
  at the as-of date. Calibration must only use pre-holdout OOS folds.
- **Lean features preferred.** v88 research found that expanding beyond the 12-feature lean
  baseline consistently hurt balanced accuracy and calibration. Any feature additions must
  demonstrate improvement on balanced accuracy (not just raw accuracy) with holdout
  validation.
- **High-bias / low-variance model families only.** L1/L2 regularized logistic regression
  is the default. Shallow nonlinear models (v91) did not improve on the linear baseline.
  Deep models or large ensembles are out of scope.
- **Small N constraint.** Per-benchmark training sets are 176–252 monthly observations.
  Recommendations must be appropriate for this regime.
- **GitHub Actions as primary deployment surface.** No heavy infrastructure. Monthly
  artifact-based workflow must be preserved.
- **Interpretability matters.** Coefficient-level attribution is a first-class requirement.
  The user reads a monthly markdown report and needs to understand *why* the classifier
  said what it said.
- **Production promotion requires evidence.** Agreement rate, policy uplift, and calibration
  stability across multiple live shadow months are required before any classifier path
  overrides the production regression consensus.

---

### Research Questions

Please explicitly address each of the following questions. Where relevant, make a specific,
quantifiable recommendation rather than a general observation.

#### Question 1: Portfolio-Weighted Aggregation Design

The most immediately actionable improvement is aligning the classifier aggregation weights
with the user's actual redeploy portfolio rather than with regression quality proxies.

Please address:

1a. What is the methodologically correct way to construct a portfolio-weighted composite
    `P(Actionable Sell)` from a set of per-benchmark binary classifiers?

    Specifically: is it valid to compute `Σ(redeploy_weight_i × P_i)` across the 4 benchmarks
    that overlap between the classifier universe and the investable universe (VOO, VXUS, VWO, BND)?
    What are the statistical properties of this estimator vs. the current quality-weighted scheme?

1b. How should the user handle the 4 contextual benchmarks (DBC, GLD, VMBS, VDE) that are
    in the classifier but not in the redeploy portfolio?
    - Option A: exclude them from the primary aggregated signal entirely, retain for diagnostics only
    - Option B: retain them in the primary signal with zero or near-zero weight
    - Option C: include them at a small non-zero weight as insurance against concentration in the
      investable set
    What does the literature or quantitative practice suggest for this class of "informative but
    non-investable" reference assets?

1c. Should the redeploy portfolio weights used for aggregation be:
    - Fixed (e.g., the base `balanced_pref_95_5` weights)
    - Dynamic (the actual monthly signal-tilted weights from the redeploy allocation output)
    - Some hybrid (base weights updated annually)
    What are the tradeoffs between fixed and dynamic weighting in the context of a monthly
    binary decision with small training samples?

#### Question 2: Expanding the Classifier to Cover VGT and SCHD

VGT (Vanguard Information Technology ETF) and SCHD (Schwab US Dividend Equity ETF) are the
two funds in the redeploy portfolio that have no classifier representation.

Please address:

2a. For VGT (tech/growth-tilted equity ETF): what macro and fundamental signals have the
    strongest documented historical relationship with tech-sector relative performance vs.
    the broad market? Focus on signals constructable from the following already-available
    data sources: FRED, Alpha Vantage fundamentals (EPS, revenue, book value), and derived
    momentum/volatility features. Which features from the current lean baseline are most
    and least likely to carry predictive signal for a PGR-vs-VGT comparison?

2b. For SCHD (dividend / value ETF): what signals most predict the dividend/value factor's
    relative performance? Which of the 12 lean features are likely to be relevant or
    irrelevant for PGR vs. SCHD? Are there one or two supplemental features worth testing
    (e.g., dividend yield spread, value vs. growth spread, payout ratio momentum)?

2c. VGT's available ETF history is shorter than most current benchmarks (~20 years vs. ~25+).
    At an expected training N of ~140–180 monthly observations, what are the risks of including
    VGT in a per-benchmark separate logistic classifier, and what mitigation strategies are
    appropriate?

2d. SCHD has an even shorter live history (~14 years, ~168 months). Same question: what are
    the risks, and should SCHD be included now or deferred until more history is available?

#### Question 3: Benchmark-Specific Feature Selection

The current design uses identical feature sets for all 8 benchmarks. The v122 audit shows that
coefficient patterns differ substantially by benchmark. The question is whether this divergence
is strong enough to justify benchmark-specific feature sets, and if so, how to select them.

Please address:

3a. Given the "lean features win" finding from v88 (adding features consistently hurt balanced
    accuracy), under what conditions would benchmark-specific feature selection be expected to
    improve rather than degrade performance in a sample of 176–252 training rows?

3b. For each asset class below, identify one or two candidate features (not in the current lean
    baseline) that have the strongest theoretical and empirical case for predicting PGR's relative
    return vs. that asset class. Cite specific evidence where available. Practical constraint:
    features must be constructable from FRED, Alpha Vantage, or PGR EDGAR 8-K filings.

    - Bond benchmarks (BND, VMBS): duration-sensitive credit/rate signals
    - Commodities (DBC): commodity-cycle or inflation-breakeven signals
    - Gold (GLD): USD index momentum, real-rate regime signals
    - Energy equity (VDE): oil/gas price momentum, energy-sector relative momentum
    - US large-cap equity (VOO, VGT): earnings revision signals, sector rotation
    - International / EM equity (VXUS, VWO): USD strength, EM/DM growth differential
    - Dividend/value equity (SCHD): value spread, dividend yield spread

3c. What is the recommended approach for selecting benchmark-specific features while
    maintaining strict time-series discipline? Specifically: is it valid to select features
    per benchmark using L1 logistic regularization within a WFO walk, or does feature
    selection need to occur on a fully held-out pre-training sample to avoid leakage?

#### Question 4: Composite Portfolio Target as an Alternative Architecture

An alternative to aggregating per-benchmark classifiers is to train a single classifier
directly on whether PGR outperforms the user's weighted composite portfolio return.

Please address:

4a. What are the theoretical and practical tradeoffs between:
    - **Path A:** per-benchmark separate classifiers → portfolio-weighted aggregation
    - **Path B:** single classifier on composite portfolio return target
    for a decision problem of this type (hold vs. sell a concentrated equity position)?

4b. In Path B, the composite return target is `Σ(redeploy_weight_i × benchmark_return_i)`.
    How does the additional noise in this composite target (vs. individual benchmark returns)
    affect classifier stability at N ≈ 200 training samples? Is there a literature precedent
    for this construction in the active vs. passive equity management context?

4c. Which path would you recommend as the primary research direction for the next cycle,
    and which should be run as a secondary comparison? Justify your recommendation in terms
    of expected improvement in balanced accuracy, interpretability, and operational stability.

#### Question 5: Calibration and Balanced Accuracy

The current shadow path achieves Brier score 0.1852 and ECE 0.0813 — good calibration
relative to raw probability accuracy — but balanced accuracy drops from 58.27% (uncalibrated)
to 51.32% (calibrated with 0.30–0.70 abstention band). This post-calibration degradation in
balanced accuracy is a concern.

Please address:

5a. Is it common for prequential logistic calibration to reduce balanced accuracy even while
    improving Brier score and ECE? What is the mechanistic explanation for this tradeoff?

5b. For a small-sample monthly binary classification problem with class imbalance (approximately
    30% positive class), what calibration approaches are most likely to preserve or improve
    balanced accuracy while maintaining good probability calibration?
    Candidates to evaluate:
    - Prequential logistic (current)
    - Isotonic regression calibration
    - Platt scaling (logistic on logit outputs)
    - No post-hoc calibration with improved in-training class weighting
    - Temperature scaling

5c. Are there threshold optimization strategies for the abstention band (currently 0.30–0.70
    uniform across all benchmarks) that could improve the tradeoff between abstention rate and
    balanced accuracy on non-abstained predictions? Is per-benchmark threshold optimization
    valid under strict time-series discipline, or does it introduce leakage risk?

#### Question 6: Production Promotion Path

The v96 conclusion was `continue_research_no_promotion`. The shadow monitoring program
(v118–v121) found 93.83% agreement with the live baseline, 10 disagreement months over 162,
and cumulative uplift of 0.0812 on the shadow path.

Please address:

6a. What is a principled set of criteria for promoting the classification layer from shadow
    to a production gate role? Specifically, across how many prospective (not backtested)
    monthly shadow observations should the classifier demonstrate stability before promotion
    is justified? What metrics should the promotion gate check?

6b. Should the first production role for the classifier be:
    - A **veto gate** (regression sell requires classifier confirmation before acting)
    - A **permission-to-deviate gate** (classifier signal can override a regression hold
      recommendation in specific conditions)
    - A **co-equal primary signal** (classifier and regression both contribute to recommendation mode)
    - A **confidence tier modifier** (classifier probability shifts the recommendation confidence
      tier without changing the base recommendation)
    Justify your recommendation in terms of the asymmetric error costs for this specific
    decision (false positives = unnecessary tax event; false negatives = missed diversification).

6c. What is the strongest argument for keeping the classifier shadow-only indefinitely, and
    under what project conditions (data accumulation, model stability, metric thresholds) would
    that argument lose force?

---

### Constraints and Preferences for Recommendations

- Prefer interpretable models and feature sets. Coefficient-level attribution is required.
- Avoid recommending model families that are not stable in N ≈ 200 time-series samples.
- Do not recommend K-Fold, full-sample normalization, or approaches that violate the
  temporal split discipline in CLAUDE.md.
- Prefer additions that are backward-compatible with the existing monthly artifact workflow.
- Distinguish clearly between:
  - **Research-only** next steps (scripts, results summaries, no production changes)
  - **Shadow-only** additions (new diagnostic columns in monthly artifacts)
  - **Production promotion candidates** (require explicit promotion gate criteria)
- If a recommendation increases complexity, justify why the expected value exceeds that cost.
- If something was tried and did not work (shallow nonlinear models in v91, broad feature
  expansion in v88), do not re-propose it without a specific new angle.

---

### Required Output Format

#### 1. Executive Summary
- Top 3–5 most important conclusions from your research
- Recommended primary research direction for the next cycle

#### 2. Answers to Research Questions
- Address each numbered question (1a–6c) explicitly
- Cite literature, documented empirical results, or quantitative reasoning
- Do not give generic advice — be specific to the project context

#### 3. Recommended Feature Candidates
- For each benchmark that warrants benchmark-specific features, provide a prioritized
  short list (1–3 candidates) with justification and expected data source
- Flag any candidate that introduces new data-source dependencies or FRED/AV API changes

#### 4. Recommended Architecture for Next Cycle
- State clearly: portfolio-weighted aggregation vs. composite target vs. both in parallel
- State clearly: include VGT/SCHD now vs. defer, and why
- State clearly: benchmark-specific features now vs. lean-baseline-only, and why

#### 5. Implementation Plan
Produce a concrete implementation plan for the next research cycle (v123+) that:
- Uses version numbers consistent with the existing sequence
- Separates research scripts, shadow additions, and any production changes
- Identifies files likely to change (`src/models/classification_shadow.py`,
  `src/processing/`, `scripts/monthly_decision.py`, etc.)
- Includes specific validation checks for each phase (metric thresholds, leakage checks)
- Includes promotion gate criteria for any shadow → production candidates

#### 6. Final Deliverable Block

End with:

```text
<proposed_plan>
[full implementation plan here, in enough detail for Codex or Claude Code to execute directly]
</proposed_plan>
```

---

### Additional Context for Reviewer

The project uses:
- Python 3.10+, pandas, numpy, scikit-learn, xgboost (shallow only)
- Alpha Vantage free tier for fundamental data; FRED for macro series
- SQLite as the local database
- GitHub Actions as the monthly orchestration surface
- One technically capable user, maintained with LLM coding assistance

The user's primary goal is **decision quality**, not forecasting accuracy per se. A system
that correctly identifies 65% of truly actionable sell months while keeping false positives
low (to avoid unnecessary tax events) is more valuable than one that maximizes log-likelihood
on a held-out set.

Calibrated uncertainty — knowing *when* the model does not know — is as important as
raw accuracy. The 0.30–0.70 abstention band in the current shadow path is a feature, not
a bug. The goal is not to force a recommendation every month; it is to recommend
confidently when the evidence is strong and stay neutral when it is not.
