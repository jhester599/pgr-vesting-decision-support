# Deep research review of the classification layer for the PGR vesting decision-support system

## Executive summary

The biggest actionable design fix for the next cycle is to **make the classifier answer the same question the user is actually asking**: “Will I be better off selling PGR and buying my target redeploy portfolio?” Today’s shadow classifier aggregates per-benchmark probabilities using **regression-layer quality weights**, which is methodologically defensible as a *forecast-combination heuristic* but is **not portfolio-aligned** and is explicitly in tension with the repo’s own v27 “two universes” separation (forecast benchmarks vs. investable redeploy set). fileciteturn26file0L1-L1 fileciteturn44file0L1-L1

The most defensible portfolio-aligned path is to introduce a **portfolio-level target (fixed base weights)** and train/calibrate a **single binary classifier** on that target (Path B), while keeping per-benchmark models as diagnostics and as a secondary ensemble comparator (Path A). This directly resolves the “VGT + SCHD absent” problem and eliminates the need to give non-investable contextual benchmarks any weight in the primary “sell probability” signal. fileciteturn26file0L1-L1 fileciteturn44file0L1-L1

The calibration finding in v87–v96/v122—**better Brier/ECE but worse balanced accuracy**—is consistent with known theory: improving reliability (calibration) can reduce sharpness/resolution, and any fixed abstention band will change the effective confusion matrix after calibration. This is not a bug; it means the project should **separate probability calibration evaluation from threshold/abstention optimization**, and (critically) treat thresholds as *policy parameters* tuned under strict walk-forward discipline, not as constants carried over without re-optimization. fileciteturn30file0L1-L1 citeturn4search1turn4search5turn9search48

Promotion remains premature. The repo’s own prospective replay shows good agreement and positive uplift in a simulated setting, but also explicitly notes that this is not a substitute for matured real-time monitoring; and the current matured monitoring count is effectively zero. A promotion gate should therefore be framed around **matured prospective months** (not replay months), with explicit minimum sample sizes and stability checks. fileciteturn35file0L1-L1 fileciteturn36file0L1-L1 fileciteturn40file0L1-L1

Recommended primary research direction for v123+: **Path B (single classifier on a fixed-weight redeploy portfolio target)** as the primary line, with **Path A (portfolio-weighted linear pool of per-benchmark classifiers, followed by portfolio-level recalibration)** as a secondary comparator for interpretability and continuity.

## Answers to research questions

### Portfolio-weighted aggregation design

**Question 1a — What is the methodologically correct portfolio-weighted composite probability?**

A portfolio-weighted sum of per-benchmark probabilities,
\[
p_{\text{pool}}=\sum_i w_i\,p_i,
\]
is best understood as a **linear opinion pool** (LOP): a weighted average of probabilistic forecasts. In forecast-combination theory, the linear pool is widely used, but it has a key known property: **any non-trivial weighted average of distinct calibrated forecasts is generally not calibrated and tends to lose sharpness**, meaning it often requires *recalibration* even when the components are individually calibrated. citeturn3search1turn3search0

So: **yes, it is valid as a probabilistic *combiner*** (and often improves Brier/log loss empirically), but **no, it is not automatically a consistent estimator of**  
\[
P\!\left(r_{\text{PGR}}-\sum_i w_i r_i < -0.03\right),
\]
because that portfolio event is defined on a *linear combination of returns*, while each classifier predicts a *different event* \(I(r_{\text{PGR}}-r_i<-0.03)\). Those events are related but not identical.

In your repo’s current implementation, the composite is:
- per-benchmark calibrated probabilities, then
- a weighted sum using **benchmark_quality_weights** derived from regression metrics (nw_ic), with shrinkage toward equal weights. fileciteturn44file0L1-L1 fileciteturn6file0L1-L1

That is a valid “expert weighting” heuristic, but it is **not portfolio-aligned**.

**Portfolio-aligned recommendation (quantifiable):**
- Construct an **investable-only linear pool**:
  - benchmarks: {VOO, VGT, SCHD, VXUS, VWO, BND} (the v27 investable set). fileciteturn26file0L1-L1
  - weights: base redeploy weights (fixed; details in 1c below). fileciteturn7file0L1-L1
- Then perform **a second-stage portfolio-level recalibration** of \(p_{\text{pool}}\) against the **portfolio-level realized actionable-sell label** (defined on the weighted portfolio return), using strictly prequential / walk-forward data.

This structure is directly motivated by the forecast-combination literature: linear pooling is common but requires recalibration for reliability. citeturn3search1turn3search0

A practical calibrator that stays in-scope (high-bias/low-variance, interpretable):
- **Platt-style logistic recalibration on the logit of \(p_{\text{pool}}\)** (or equivalently logistic on a 1D score), fit only on past matured months.

This preserves your interpretability goal (one extra monotone mapping) and eliminates the “portfolio alignment gap” without needing to model joint return distributions.

**Statistical properties vs. the current quality-weight scheme:**
- Portfolio-weighted pooling is **unbiased with respect to the decision objective** only after portfolio-level recalibration, because the raw pool is not a coherent probability for the portfolio event. citeturn3search1turn3search0
- Quality-weight pooling can reduce variance by emphasizing historically predictive benchmarks, but it targets “best forecasting benchmarks,” not “best redeploy destinations,” and therefore can be systematically mis-specified for the user’s utility function. fileciteturn44file0L1-L1 fileciteturn26file0L1-L1
- In small samples, any weighting scheme that depends on estimated quality metrics is itself noisy; your implementation already mitigates this with shrinkage toward equal weights (lambda_mix). fileciteturn6file0L1-L1

**Question 1b — What to do with contextual non-investable benchmarks (DBC, GLD, VMBS, VDE)?**

The repo explicitly treats “forecast benchmark universe” and “investable redeploy universe” as distinct, and recommends keeping certain assets (e.g., GLD, DBC, VMBS, VDE) as contextual diagnostics even when they are not buy destinations. fileciteturn26file0L1-L1

Given that framing, the strongest recommendation is:

- **Option A (exclude from the primary aggregated signal; diagnostics only)**.

Rationale:
- Including non-investable assets with non-zero weight makes the composite probability answer a different question: “Will PGR underperform a partly hypothetical portfolio I won’t buy?” That is a classic benchmark/objective mismatch.
- If these assets contain regime information, the more coherent way to use them is as **features** (risk regime indicators) or as **diagnostic side panels**, not as part of the objective function.

A technically equivalent operational variant is “Option B with explicit zero-weight,” which can be useful for reporting consistency (still show their probabilities, but clearly not part of the primary portfolio signal). In practice, Option A and “B with zero weight” are the same for decisions.

**Option C (small non-zero weight as insurance)** is not recommended here. If you want “insurance against concentration in the investable set,” that insurance should be implemented in the **portfolio construction layer** (where you already constrain and diversify weights) rather than by smuggling non-investable benchmarks into a probability meant to represent the investable redeploy decision. fileciteturn26file0L1-L1

**Question 1c — Fixed vs dynamic redeploy weights for aggregation?**

Your redeploy portfolio is bounded and modestly signal-tilted, with a stable high-equity shape (the v27 “balanced_pref_95_5” family). fileciteturn26file0L1-L1 fileciteturn7file0L1-L1

In a monthly binary decision with small N (~200), the trade-offs are:

- **Fixed weights (recommended for the primary classifier target and primary aggregation)**
  - Pros: stationary target definition, stable interpretation, cleaner evaluation, less multiplicative noise (probability + weights), lower leakage risk.
  - Cons: slightly less “exact” alignment if the user truly redeploys with time-varying weights.

- **Dynamic monthly weights**
  - Pros: closer to what the user actually buys that month.
  - Cons: target becomes time-varying and potentially endogenous (weights depend on model outputs); this increases variance and complicates clean walk-forward evaluation; it can also create feedback loops where regression signals indirectly affect classifier outcomes through weights.

**Quantified recommendation:**
- Use **fixed base weights** for:
  - the *primary portfolio target* (Path B), and
  - the *primary portfolio-weighted aggregation* (Path A comparator).
- Compute the **dynamic-weight version as a shadow diagnostic only**, and report it side-by-side, but do not treat it as the primary probability without strong evidence.

A “hybrid annual update” (base weights reviewed annually / when the user changes preferences) can capture long-run preference drift without injecting month-to-month noise.

### Expanding the classifier to cover VGT and SCHD

**Question 2a — VGT signals and lean-feature relevance**

VGT is a tech-sector equity ETF with inception January 26, 2004. citeturn13search23  
Two documented, high-level relationships matter most for tech/growth-relative behavior:

1) **Discount-rate / duration sensitivity of growth-like equities.** Asset-pricing work models growth stocks as having cash flows weighted further in the future (longer “equity duration”), making them more sensitive to discount-rate changes; this is a core mechanism behind value/growth dynamics. citeturn8search4turn8search8turn8search0

2) **Interest-rate cycle linkage for growth vs value exposure.** Practitioner research explicitly studying interest-rate cycles finds systematic differences in rate sensitivity between growth and value indices and links that sensitivity to growth-related attributes (earnings growth, earnings changes, book-to-price). citeturn8search0

Given those, the most plausible signals (all constructable with your existing data plumbing) are:

- **Most likely to help for PGR vs VGT**
  - real-rate level and changes: `real_rate_10y`, `real_yield_change_6m` (already in lean baseline). fileciteturn43file0L1-L1
  - term premium: `term_premium_10y` (already available in your feature builder when the v19 series is present). fileciteturn20file0L1-L1
  - risk appetite / liquidity proxies: `credit_spread_hy`, `nfci`, `vix` (lean baseline). fileciteturn43file0L1-L1 citeturn7search0

- **Most likely to be *weaker* for PGR vs VGT**
  - pure insurer operating metrics *may* be less directly linked to tech-relative moves, but v122 shows `combined_ratio_ttm` dominates across many benchmarks, implying it is strongly predictive of PGR’s relative regime broadly; it is therefore not safe to assume it is irrelevant for VGT without testing. fileciteturn24file0L1-L1

**Concrete feature recommendation for VGT coverage (minimal additions):**
- Add **one** benchmark-relative momentum feature:
  - `vgt_voo_spread_6m` (VGT 6M return − VOO 6M return), constructed from ETF total-return series in your DB the same way other benchmark-relative spreads are already constructed (e.g., `vwo_vxus_spread_6m`). fileciteturn20file0L1-L1  
This directly captures tech rotation vs broad US equity and is both interpretable and cheap.

**Question 2b — SCHD signals and lean-feature relevance**

SCHD is a dividend/value-oriented ETF with inception October 20, 2011. citeturn12search3  
Dividend/value relative performance is often framed through (a) **value vs growth duration** and (b) **discount-rate vs cash-flow news** in present-value models; the dividend–price ratio literature also links valuation ratios to expected returns via discount-rate expectations. citeturn8search4turn11search1turn11search0

For PGR vs SCHD, the most plausible signals are:

- **Likely relevant lean features**
  - `yield_slope`, `real_rate_10y`, `real_yield_change_6m` (rate regime). fileciteturn43file0L1-L1
  - `credit_spread_hy` (risk/credit regime). fileciteturn43file0L1-L1 citeturn7search0
  - `combined_ratio_ttm` and PGR underwriting/financial strength drivers (PGR-specific alpha), which v122 indicates is consistently important. fileciteturn24file0L1-L1

- **Potentially less relevant lean features**
  - `mom_12m` may remain useful (momentum is broad), but you should not expect it to dominate a value/dividend-relative signal since the decision is “PGR vs diversified dividend basket,” not “PGR trend-following.” This is consistent with v122’s finding that momentum matters but is not leading. fileciteturn24file0L1-L1

**One or two supplemental features worth testing for SCHD (bounded, interpretable):**
1) `equity_risk_premium` proxy (earnings yield minus GS10), which your feature builder can compute from Multpl earnings yield and Treasury yields. fileciteturn20file0L1-L1 citeturn11search7turn6search2  
2) A dividend-yield spread proxy would be conceptually ideal (dividend yield minus Treasury yield), but it adds a new dependency unless you already ingest a dividend-yield series; treat this as optional and only if you can source it within existing ingestion. (Your current v19 Multpl series list includes PE/PB/earnings yield but not dividend yield.) fileciteturn20file0L1-L1

**Question 2c — VGT shorter history risks and mitigations**

At ~20+ years of ETF history (monthly N roughly 240+ pre-holdout, less after applying holdout and horizon truncation), VGT is somewhat shorter than older benchmark ETFs but still within the rough regime used by your per-benchmark models (monthly samples on the order of ~200). citeturn13search23 fileciteturn43file0L1-L1

Risks at N ≈ 140–180 (if you restrict to certain windows or post-inception periods):
- coefficient instability and sign flips,
- calibration overfitting (especially with flexible calibrators),
- threshold tuning instability.

Mitigations consistent with your constraints:
- Run VGT initially as **portfolio-target Path B** (single classifier) or as part of a **pooled-panel classifier** across investable benchmarks (shared coefficients + benchmark intercept) to increase effective sample size; v90 already found pooled panel structure can improve balanced accuracy relative to separate models. fileciteturn25file0L1-L1
- If maintaining separate per-benchmark models, enforce:
  - stronger regularization (lower C),
  - strictly limited feature additions (≤1 new feature for VGT initially),
  - calibrate at the **portfolio level** rather than per-benchmark when VGT is a small-N benchmark. citeturn3search1turn3search0

**Question 2d — SCHD shorter history risks and include vs defer**

SCHD has ~14+ years of history; at monthly frequency this is roughly ~168 observations in total, which is right around your stated “~168 months” expectation. citeturn12search3

Risks are more acute than VGT:
- less regime coverage,
- higher sensitivity to any single period (e.g., post-2020 rate/inflation regime),
- greater multiple-testing risk if you add features.

Recommendation:
- **Include SCHD now**, but **only in portfolio-aligned modeling** (Path B primary) and as a shadow diagnostic in Path A, not as a benchmark that can drive production gating yet.
- Require explicit stability checks before giving SCHD any material weight under Path A:
  - coefficient sign stability for top drivers ≥70% across WFO folds,
  - calibrated Brier not worse than a constant-base-rate predictor by more than 0.01,
  - and no large calibration drift in the last ~24 months of pre-holdout folds.

This lets the project gain signal coverage for a core sleeve without pretending the short history is as robust as long-lived benchmarks.

### Benchmark-specific feature selection

**Question 3a — When does benchmark-specific feature selection help in N ≈ 200?**

Benchmark-specific feature sets can help when **bias dominates variance** because the same feature set is clearly mis-specified for certain asset classes. Your own audit shows benchmark-to-benchmark coefficient patterns differ materially (e.g., VWO emphasizes `credit_spread_hy` strongly; some benchmarks emphasize curve/rate variables more). fileciteturn24file0L1-L1

However, v88 found that broad feature expansion hurt balanced accuracy and calibration, which is consistent with the general “small sample + many features = overfit” dynamic (and is one reason you explicitly guard obs/feature ratio). fileciteturn25file0L1-L1 fileciteturn20file0L1-L1

So benchmark-specific features are only likely to improve outcomes if all of the following hold:

1) **Precommitted, small candidate sets** (≤1–3 features per asset class) rather than open-ended feature search.  
2) **Strong prior plausibility + interpretability** (features map to the asset class’ return drivers).  
3) **Nested time-series evaluation** where feature inclusion is validated walk-forward, not chosen globally.  
4) **Stability criterion**: new features must improve balanced accuracy *and* show stable coefficient signs across folds, not only a one-off lift.

**Question 3b — Asset-class-specific candidate features not in lean baseline**

Below are the best “1–2 feature” candidates per asset class that fit your data-source constraints and are already compatible with your ingestion patterns (FRED / DB prices / EDGAR), with brief evidence/logic.

- **Bond benchmarks (BND, VMBS)**  
  Candidate features:
  - `term_premium_10y` (ACM 10Y term premium series), capturing rate-risk compensation regime. fileciteturn20file0L1-L1  
  - `mortgage_spread_30y_10y` (30Y mortgage rate − 10Y Treasury), directly relevant to mortgage-backed and bond spread conditions. fileciteturn20file0L1-L1  
  Why: these are duration/spread regime signals beyond `yield_slope` and `real_rate_10y`.

- **Commodities (DBC)**  
  Candidate features:
  - `breakeven_inflation_10y` or `breakeven_momentum_3m` (inflation expectations proxy), since commodities are commonly treated as inflation-sensitive real assets. fileciteturn20file0L1-L1  
  - `wti_return_3m` (oil momentum), as a commodity-cycle proxy. fileciteturn20file0L1-L1  
  Why: directly targets “inflation/commodity cycle” state instead of relying only on generic credit/rates.

- **Gold (GLD)**  
  Candidate features:
  - `usd_momentum_6m` or `usd_broad_return_3m` (broad USD index momentum). fileciteturn20file0L1-L1  
  - (Optional) keep `real_rate_10y` but add the term premium as a regime interaction proxy if you remain linear. fileciteturn20file0L1-L1  
  Why: gold is widely modeled as sensitive to real rates and USD strength; your current lean set covers real rates but not USD. fileciteturn24file0L1-L1

- **Energy equity (VDE)**  
  Candidate features:
  - `wti_return_3m` (oil price momentum). fileciteturn20file0L1-L1  
  - an energy-relative momentum feature such as `vde_voo_spread_6m` (analogous to `commodity_equity_momentum`), built from DB prices. fileciteturn20file0L1-L1  
  Why: captures sector-specific driver (energy beta to oil) that general macro variables may not represent cleanly.

- **US large-cap equity (VOO, VGT)**  
  Candidate features:
  - `equity_risk_premium` proxy (earnings yield − GS10), tying valuation/discount rate to expected returns; valuation ratios are central in the stock return predictability literature. fileciteturn20file0L1-L1 citeturn11search1turn6search2  
  - for VGT specifically: `vgt_voo_spread_6m` (sector rotation). fileciteturn20file0L1-L1 citeturn13search23

- **International / EM equity (VXUS, VWO)**  
  Candidate features:
  - `usd_momentum_6m` (broad USD strength). fileciteturn20file0L1-L1  
  - `vwo_vxus_spread_6m` (already implemented), capturing EM vs DM relative cycle. fileciteturn20file0L1-L1  

- **Dividend/value equity (SCHD)**  
  Candidate features:
  - `equity_risk_premium` proxy (earnings yield − GS10). fileciteturn20file0L1-L1 citeturn11search7turn6search2  
  - (Optional, new dependency) dividend-yield spread if you can ingest a dividend yield series without adding fragile scraping. fileciteturn20file0L1-L1

**Question 3c — Is L1 feature selection within WFO valid, or does it leak?**

Using L1-regularized logistic regression as embedded feature selection is valid **if and only if** it is treated as part of the model fit inside each walk-forward training window and is evaluated strictly on the subsequent test window (i.e., selection happens using training data only). This aligns with the repo’s strict time-series split discipline based on `TimeSeriesSplit` with a `gap` to avoid leakage. fileciteturn43file0L1-L1 citeturn6search0

What is *not* valid is selecting features once on the full history (or on pooled folds that include later periods) and then claiming OOS performance—this would be classic look-ahead selection bias.

**Recommended approach (quantifiable and governance-friendly):**
- Keep the **default feature set fixed** (lean baseline) for the first pass.
- If testing benchmark-specific additions:
  - predefine ≤3 candidate features per asset class,
  - evaluate inclusion via WFO,
  - require:
    - pooled balanced accuracy lift ≥ +0.02 (absolute),
    - ECE not worse by >0.01,
    - coefficient sign stability for the new feature ≥70% across folds,
  - otherwise revert to lean baseline.

### Composite portfolio target as an alternative architecture

**Question 4a — Tradeoffs Path A vs Path B**

- **Path A: per-benchmark classifiers → portfolio-weighted aggregation**
  - Strengths:
    - per-benchmark interpretability (which sleeve is “supporting” the sell regime), consistent with current reporting. fileciteturn44file0L1-L1  
    - can keep contextual benchmarks as diagnostics without affecting the investable signal. fileciteturn26file0L1-L1
  - Weaknesses:
    - portfolio alignment requires additional design (choice of weights + portfolio-level recalibration), and the raw linear pool is not inherently calibrated. citeturn3search1turn3search0  
    - adding VGT/SCHD increases “many models” complexity and can worsen variance in small samples.

- **Path B: single classifier on composite portfolio-return target**
  - Strengths:
    - directly answers the user’s decision question (highest methodological alignment).
    - automatically incorporates VGT/SCHD via the portfolio return definition.
    - operationally simpler: one model, one calibration, one abstention policy.
  - Weaknesses:
    - less benchmark-sleeve attribution (you can still attribute via scenario analysis, but it’s not as direct as per-benchmark probabilities).
    - the target depends on the portfolio definition; changes in portfolio weights require re-baselining.

**Question 4b — How does composite-target noise affect stability at N ≈ 200? Any precedent?**

A weighted composite of benchmark returns often has **lower idiosyncratic variance** than individual components (diversification effect), but it can also dampen “sharp” benchmark-specific signals. In probabilistic forecasting terms, you are trading some resolution on single benchmarks for a more decision-aligned target. This is analogous to the broader literature on predicting equity returns where predictable components are small and OOS performance is fragile; imposing structure and focusing on economically meaningful targets can improve usefulness even when statistical R² is small. citeturn6search2turn6search1

In forecast evaluation, the key is not whether the target is “noisier” in absolute terms, but whether the model can produce **calibrated probabilities** and useful abstention behavior on that target. Proper scoring rule frameworks explicitly separate calibration from sharpness. citeturn4search4turn4search5turn4search1

**Question 4c — Which path should be primary in v123+?**

Primary recommendation: **Path B** as the main research direction, with **Path A as a secondary comparison**.

Justification:
- Expected balanced-accuracy improvement: Path B avoids mixing non-investable signals and avoids the “wrong benchmark weighting” problem; this should reduce systematic error in the label definition relative to the user’s objective. fileciteturn26file0L1-L1
- Interpretability: Path B remains interpretable at coefficient level; Path A remains available as a diagnostic panel for “which sleeve is driving.” fileciteturn44file0L1-L1
- Operational stability: one model + one calibrator is materially less complex than maintaining (6–10) separate benchmark models plus aggregation logic.

### Calibration and balanced accuracy

**Question 5a — Is it common that calibration improves Brier/ECE while reducing balanced accuracy? Why?**

Yes. Brier score and ECE are **probability calibration** measures, while balanced accuracy is a **thresholded classification** measure. Improving calibration can change the distribution of predicted probabilities (often shrinking extremes toward the base rate), which can reduce “sharpness” and change how many predictions fall outside an abstention band—altering the confusion matrix and therefore balanced accuracy. This calibration–sharpness framing is central in probabilistic forecast evaluation. citeturn4search5turn4search1

Mechanistically in your setting:
- the shadow path uses a **0.30–0.70 abstention band** (neutral region). fileciteturn30file0L1-L1  
- prequential/logistic calibration tends to move probabilities toward reliability; if that moves more months into 0.30–0.70, you will see:
  - higher accuracy on the majority outcome (because you abstain or default more),
  - but potentially poor class recall on the minority class, which drags balanced accuracy.  
This is consistent with known distortions and data needs of calibration methods in small samples. citeturn9search48

**Question 5b — Calibration approaches likely to preserve or improve balanced accuracy in small-N, imbalanced monthly classification**

Given your candidate list and constraints, the best-prioritized approaches are:

1) **Platt scaling on logits (recommended baseline alternative)**  
Platt scaling is explicitly designed as a parametric monotone calibration map; it is often more stable than isotonic in limited data. citeturn9search48

2) **Current “logistic on probability” calibrator**  
Your current monthly code fits a 1D logistic calibrator on OOS probabilities per benchmark and applies it to the current month. fileciteturn44file0L1-L1  
This is reasonable but should be compared against logit-space Platt scaling; logit-space often behaves better near 0/1.

3) **Isotonic regression calibration (only with minimum-history gating)**  
Isotonic is flexible but data-hungry and can overfit; the calibration literature explicitly cautions that isotonic’s extra power needs more data to be effective. citeturn9search48  
If tested, gate it with a minimum calibration-history requirement (e.g., ≥60 matured observations) and prefer portfolio-level isotonic on a pooled signal rather than per-benchmark isotonic.

4) **No post-hoc calibration, improve in-training weighting / shrinkage**  
If calibration is causing “probability mass collapse into the abstention band,” it may be better to:
- tighten/loosen regularization and class weights,
- then recalibrate thresholds, rather than adding a post-hoc calibrator.  
(This is a policy-driven choice, not a universally correct one.) citeturn4search5turn4search1

5) **Temperature scaling**
Temperature scaling is essentially a one-parameter logit rescaling; it can be stable, but because your base model is logistic regression (already a logit-linear model), temperature scaling mostly behaves like a controlled shrink/expand of confidence. It is worth testing only at the portfolio-level aggregation stage to avoid excessive plumbing.

**Question 5c — Threshold optimization for the abstention band; is per-benchmark optimization valid?**

Threshold optimization can materially improve the tradeoff between abstention rate and balanced accuracy—*but only if treated as a walk-forward policy parameter*.

There is classic “reject option” theory showing that (under calibrated posteriors and a specified reject cost) the optimal abstention region is where posterior probabilities lie between two cutoffs. citeturn4search2turn4search0  
Your current symmetric 0.30–0.70 band is a reasonable default, but it is not necessarily optimal given asymmetric costs (taxable false positives are expensive). fileciteturn30file0L1-L1

**Quantified recommendation:**
- Move from a single fixed (0.30, 0.70) to a **small grid** of candidate bands tested under WFO:
  - symmetric: (0.25, 0.75), (0.30, 0.70), (0.35, 0.65)
  - asymmetric (tax-averse): (0.25, 0.80), (0.20, 0.80)
- Choose the band by optimizing a policy score such as:
  - maximize balanced accuracy on *covered* months subject to:
    - actionable-sell precision ≥ 0.70, and
    - coverage ≥ 0.25 (or another governance minimum).
- Do this selection within the walk-forward framework (train-fold-only selection; test-fold evaluation) to avoid leakage. citeturn6search0

Per-benchmark threshold optimization is valid under strict time-series discipline, but it is much more likely to be unstable at N ≈ 200. The recommended compromise is:
- choose thresholds at the **portfolio signal level** (Path B primary), and
- only allow per-benchmark thresholds for diagnostics, not for production gating.

### Production promotion path

**Question 6a — Principled promotion criteria and how many prospective months?**

Your repo already distinguishes simulated prospective replay from true monitoring and requires matured live history before any real promotion. fileciteturn36file0L1-L1  
It also tracks that current matured monitoring is effectively zero so far. fileciteturn40file0L1-L1

A principled promotion gate for a 6-month-horizon classifier should be framed around **matured prospective predictions**. Concretely:

- Minimum sample size:
  - **≥24 matured prospective months** (i.e., months where the classifier output was produced in real-time and the 6-month horizon has elapsed), with an aspirational target of 36.  
  This is slow, but monthly high-stakes systems need actual prospective evidence; finance backtests are especially prone to overfitting and selection bias. citeturn5search0turn7search1

- Promotion gate checks (portfolio-aligned):
  1) **Calibration stability**
     - Brier ≤ 0.18 (or ≥0.01 improvement vs. baseline proxy) and ECE ≤ 0.08 on matured months, plus no single rolling-12m ECE spike > 0.12.
  2) **Decision discrimination on covered months**
     - covered-month balanced accuracy ≥ 0.60 with coverage ≥ 0.25.
  3) **Tax-aware error control**
     - actionable-sell precision ≥ 0.70 (limit false positives).
  4) **Operational stability**
     - agreement rate with baseline ≥ 0.90 and max consecutive “would-change” months ≤ 3 (consistent with your existing governance heuristics). fileciteturn41file0L1-L1
  5) **Policy uplift**
     - cumulative shadow-minus-live ≥ 0 over matured months, with a disagreement-month scorecard not negative (again consistent with your existing assessment logic). fileciteturn41file0L1-L1

**Question 6b — Best first production role: veto, permission-to-deviate, coequal, or confidence modifier?**

Given asymmetric costs (false positives trigger a taxable sale; false negatives merely delay diversification), the most defensible initial production role is:

- **A conservative veto gate**, but only in **high-confidence non-actionable** regimes (e.g., portfolio-level \(P(\text{actionable sell}) \le 0.20\)).  
This means: the regression layer may recommend selling, but the classifier can block it only when the classifier is strongly confident the month is not truly actionable.

Why:
- This structure primarily reduces false positives (unnecessary tax events) at the expense of some false negatives, which aligns with the user’s asymmetry.
- It also matches the repo’s existing overlay concept of “veto_regression_sell.” fileciteturn14file0L1-L1

A more incremental alternative (even safer) is:
- **confidence tier modifier first** (no decision overrides; just adjust the confidence narrative and monitoring flags),
followed by the limited veto gate once matured evidence exists.

**Question 6c — Strongest argument for keeping the classifier shadow-only indefinitely; when does it lose force?**

Strongest argument to keep shadow-only:
- the system operates in a **small-N, non-stationary financial environment**, where backtest selection bias is common and where even statistically significant predictors can fail out-of-sample for extended periods. citeturn5search0turn6search2turn6search1  
- the classifier’s current calibration/abstention behavior shows that probability improvements do not automatically translate into improved balanced accuracy and actionable performance, so “shadow-only as interpretation layer” may remain the highest expected value rather than risking governance drift. fileciteturn24file0L1-L1 fileciteturn44file0L1-L1

When this argument loses force:
- once you have **≥24–36 matured prospective months** showing stable calibration and a demonstrable reduction in costly false positives without negative policy impact, and once the portfolio-aligned architecture (Path B) proves consistently better than the misaligned benchmark-quality blend. fileciteturn36file0L1-L1 citeturn4search5turn3search1

## Recommended feature candidates

The following prioritized candidates are intentionally “small deltas” designed to respect the v88 finding that broad feature expansion harms balanced accuracy, and to stay compatible with your existing data sources and feature-engineering patterns. fileciteturn25file0L1-L1

**Bonds (BND, VMBS)**
- `term_premium_10y` (FRED/NY Fed ACM term premium; already supported in feature builder). fileciteturn20file0L1-L1  
- `mortgage_spread_30y_10y` (rate/spread regime; already supported). fileciteturn20file0L1-L1  

**Commodities (DBC)**
- `breakeven_inflation_10y` or `breakeven_momentum_3m` (inflation expectations). fileciteturn20file0L1-L1  
- `wti_return_3m` (commodity cycle proxy). fileciteturn20file0L1-L1  

**Gold (GLD)**
- `usd_momentum_6m` (USD regime). fileciteturn20file0L1-L1  

**Energy equity (VDE)**
- `wti_return_3m` (oil momentum). fileciteturn20file0L1-L1  

**US equity core (VOO)**
- `equity_risk_premium` (earnings yield − GS10). fileciteturn20file0L1-L1 citeturn11search7turn6search2  

**Tech equity (VGT)**
- `vgt_voo_spread_6m` (tech rotation vs broad). fileciteturn20file0L1-L1  
- (Optional second) `term_premium_10y` if not already used for equities in the portfolio model. fileciteturn20file0L1-L1 citeturn8search4turn8search0  

**International / EM (VXUS, VWO)**
- `usd_momentum_6m` (USD strength). fileciteturn20file0L1-L1  
- keep `vwo_vxus_spread_6m` (already available). fileciteturn20file0L1-L1  

**Dividend/value (SCHD)**
- `equity_risk_premium` (earnings yield − GS10). fileciteturn20file0L1-L1 citeturn11search7turn6search2  
- (Flagged new dependency) dividend-yield spread if you add a dividend yield series to ingestion; optional only. fileciteturn20file0L1-L1

New data-source dependencies introduced: **none** for the features listed above *except* the optional dividend-yield spread if you choose to ingest dividend yields (not currently in the v19 Multpl series list). fileciteturn20file0L1-L1

## Recommended architecture for the next cycle

**Primary architecture: composite portfolio target (Path B)**  
Train a single monthly logistic classifier on:
- target: `actionable_sell_3pct` vs **fixed-weight redeploy portfolio** return:
  \[
  y_t = I\!\left(r_{\text{PGR},t} - \sum_i w_i r_{i,t} < -0.03\right)
  \]
  where \(w_i\) are the base `balanced_pref_95_5` weights over {VOO, VGT, SCHD, VXUS, VWO, BND}. fileciteturn26file0L1-L1
- model: L2 logistic regression with class_weight balanced (consistent with current family). fileciteturn44file0L1-L1
- calibration: portfolio-level Platt/logistic recalibration; abstention band optimized under WFO.

**Secondary architecture: portfolio-weighted aggregation (Path A comparator)**  
Keep per-benchmark classifiers (including VGT/SCHD), but compute:
- investable-only portfolio-weighted linear pool
- portfolio-level recalibration
- keep contextual benchmarks as diagnostics only (zero weight)

**Include VGT and SCHD now:** yes, but treat them as:
- fully included in Path B by construction,
- included in Path A as diagnostic and comparator,
- not eligible for production gating weight until stability criteria are met. citeturn13search23turn12search3

**Benchmark-specific features:** defer full benchmark-specific feature sets, but allow a **small, precommitted “1-feature per asset class” test** after the portfolio-target baseline is established, using the candidate list above. fileciteturn25file0L1-L1

## Implementation plan

This plan is designed to be executable within the repo’s existing GitHub Actions + monthly artifact workflow, preserve your strict time-series discipline, and clearly separate research, shadow additions, and any production candidates.

### Research-only phases

- **v123 — Portfolio-aligned signal plumbing (no model change)**
  - Implement investable-only portfolio-weighted aggregation as *additional columns* (shadow-only), leaving the existing quality-weighted aggregation unchanged.
  - Add portfolio-level realized target computation for monitoring (fixed weights), with strict `truncate_relative_target_for_asof` behavior.
  - Validation:
    - unit test that weights sum to 1 and that contextual benchmarks have zero weight in the investable signal.
    - leakage test: as-of truncation holds for portfolio target (extend existing truncation tests). fileciteturn20file0L1-L1

- **v124 — Add VGT + SCHD benchmark coverage (shadow-only)**
  - Extend the benchmark list used by classification shadow detail to include VGT and SCHD for per-benchmark diagnostics and investable aggregation.
  - Validation:
    - verify relative return series exists in DB for VGT/SCHD and passes as-of truncation.
    - verify per-benchmark training does not proceed if class degeneracy occurs (already handled). fileciteturn44file0L1-L1

- **v125 — Path B prototype: portfolio-target classifier (research script + shadow column)**
  - Add a research script that trains/evaluates a portfolio-target classifier using WFO (`TimeSeriesSplit` + gap).
  - Add a shadow-only monthly column: `classifier_prob_actionable_sell_portfolio` with calibrated probability + confidence tier.
  - Validation:
    - pooled metrics reported: balanced accuracy, Brier, ECE, coverage.
    - compare to existing v92 baseline on matched dates (pre-holdout). fileciteturn43file0L1-L1 fileciteturn30file0L1-L1

- **v126 — Calibration and abstention tuning under strict WFO**
  - Implement and compare:
    - Platt scaling on logits vs current logistic-on-prob calibrator,
    - portfolio-level threshold band sweep.
  - Validation:
    - ensure calibration is fit only on training-period OOS predictions,
    - select thresholds only on training folds.

- **v127 — Minimal benchmark-specific feature tests (precommitted)**
  - Test only the top 1–2 candidate features per asset class, with strict acceptance criteria:
    - balanced accuracy lift ≥ +0.02 and no ECE regression > 0.01,
    - sign stability ≥70% across WFO folds,
    - otherwise reject.
  - Validation:
    - add a “feature inclusion registry” file so changes are explicit and reviewable.

### Shadow-only additions

- Add monthly report sections/columns:
  - existing: quality-weighted `P(Actionable Sell)` (unchanged)
  - new:
    - `P(Actionable Sell | investable_pool_fixed_weights)` (Path A, recalibrated)
    - `P(Actionable Sell | portfolio_target_fixed_weights)` (Path B)
  - add explicit “contextual benchmark table” that is not used in the investable composite.

- Update matured monitoring to compute realized outcomes against the **portfolio target** (fixed weights), not the current equal-weight average across `PRIMARY_FORECAST_UNIVERSE`. fileciteturn23file0L1-L1

### Production promotion candidates (gated, not automatic)

- Earliest eligible production role (once metrics achieved): **limited veto gate**:
  - only veto regression sells when portfolio-target classifier probability ≤ 0.20 and regression sell confidence is not high.
- Promotion gate requires ≥24 matured prospective months and the metric bundle described in 6a.

### Files likely to change

- `src/models/classification_shadow.py` (add investable aggregation + Path B option; new columns). fileciteturn44file0L1-L1  
- `src/models/classification_monitoring.py` (portfolio-target matured outcomes). fileciteturn23file0L1-L1  
- `src/reporting/classification_artifacts.py` (new artifact columns). fileciteturn15file0L1-L1  
- `src/portfolio/redeploy_portfolio.py` (export helper for base weights; ensure consistent normalization). fileciteturn7file0L1-L1  
- `scripts/monthly_decision.py` (wire new summary fields into monthly artifact outputs). fileciteturn18file0L1-L1  
- new: `results/research/v123_*`, `v125_*`, etc scripts for evaluation tables.

## Final deliverable block

```text
<proposed_plan>
v123 (research + shadow-only plumbing): portfolio-aligned aggregation foundations
- Goal: Add portfolio-weighted “investable” P(Actionable Sell) without changing existing shadow signal.
- Code changes:
  1) src/models/classification_shadow.py
     - Add an “aggregation_mode” concept:
       - "quality_weighted" (current behavior; default)
       - "investable_portfolio_fixed" (new)
     - Implement investable benchmark set:
       - investable = ["VOO","VGT","SCHD","VXUS","VWO","BND"]
       - contextual = remaining benchmarks currently produced (e.g., GLD/DBC/VMBS/VDE)
     - Compute two aggregates:
       - p_quality = sum(w_quality_b * p_b)  (unchanged)
       - p_investable_fixed = sum(w_portfolio_i * p_i) over investable only, with:
         - weights = base balanced_pref_95_5 weights normalized to sum to 1
         - contextual benchmarks excluded (or explicit 0 weight)
     - Add a portfolio-level recalibration step for p_investable_fixed:
       - fit 1D Platt-style calibrator on historical OOS pooled predictions vs realized portfolio-actionable label
       - apply to current month p_investable_fixed
     - Emit new summary fields:
       - probability_actionable_sell_investable_fixed
       - confidence_tier_investable_fixed
       - stance_investable_fixed
  2) src/reporting/classification_artifacts.py
     - Add columns to classification_shadow.csv and monthly_summary.json:
       - investable_fixed_prob, investable_fixed_prob_calibrated, investable_fixed_tier, investable_fixed_stance
       - keep existing columns unchanged
  3) scripts/monthly_decision.py
     - Include the new investable-fixed summary fields in monthly markdown + json outputs.
- Validation:
  - Add tests:
    - weights sum to 1; contextual weights == 0; investable list contains VGT/SCHD
    - as-of truncation applied before any target labeling (reuse/extend tests/test_asof_target_truncation.py)

v124 (shadow expansion): add VGT and SCHD per-benchmark classifier rows
- Goal: Ensure classification_shadow.csv includes benchmark-specific probabilities for VGT and SCHD.
- Code changes:
  - src/models/classification_shadow.py
    - benchmark list for per-benchmark scoring should include investable set even if regression quality df excludes them.
    - If benchmark_quality_df is provided and lacks VGT/SCHD, still score them but set quality weights only for legacy set; investable aggregation uses portfolio weights anyway.
- Validation:
  - Test that VGT and SCHD rel-return series load functions return non-empty when DB has data; otherwise, graceful skip.

v125 (research primary): Path B portfolio-target classifier
- Goal: Train and evaluate a single portfolio-target logistic model under strict WFO; compare vs Path A.
- Add new research script:
  - results/research/v125_portfolio_target_classifier.py
    - Build portfolio relative return series:
      rr_port = r_pgr - sum(w_i * r_i) (fixed base weights)
    - Apply truncate_relative_target_for_asof for 6-month horizon
    - Build binary label actionable_sell_3pct on rr_port
    - Run WFO:
      - TimeSeriesSplit with max_train=60, test_size=6, gap=8
      - Model: logistic regression (L2), class_weight="balanced"
      - Feature set: lean_baseline (RIDGE_FEATURES_12)
    - Produce results CSV including:
      - balanced_accuracy, precision/recall, Brier, log_loss, ECE, coverage under abstention bands
    - Produce comparison table vs current “quality-weighted benchmark panel” path on the same fold dates.
- Shadow addition:
  - Add portfolio_target_prob + tier + stance to monthly summary (shadow only).
- Validation:
  - Ensure the portfolio target is computed only from data available at time t; no use of future dynamic weights.

v126 (calibration + thresholds): stabilize balanced accuracy with calibrated probabilities
- Goal: Resolve “Brier/ECE improves but balanced accuracy collapses” by jointly tuning calibration + abstention.
- Implement calibrators (portfolio-level, not per-benchmark first):
  - Platt scaling on logits
  - current logistic-on-prob (baseline)
  - isotonic only if calibration-history >= 60
- Implement threshold sweep:
  - symmetric: (0.25,0.75), (0.30,0.70), (0.35,0.65)
  - asymmetric (tax-averse): (0.25,0.80), (0.20,0.80)
- Selection rule inside WFO:
  - maximize balanced_accuracy_on_covered subject to:
    - actionable_sell_precision >= 0.70
    - coverage >= 0.25
- Outputs:
  - v126_calibration_threshold_results.csv + summary md

v127 (optional, tightly scoped): benchmark-specific feature micro-tests
- Goal: test only 1–2 precommitted features per asset class (no broad search).
- Candidate adds:
  - term_premium_10y (rates)
  - usd_momentum_6m (intl/gold)
  - wti_return_3m (energy/commodities)
  - equity_risk_premium (equity valuation proxy)
- Acceptance gate:
  - pooled balanced accuracy lift >= +0.02
  - ECE not worse by > 0.01
  - coefficient sign stability >= 70% across folds
- If gate fails: revert to lean baseline.

v128 (promotion governance memo + implementation toggle)
- Goal: define and implement a “promotion gate check” (still off by default).
- Add:
  - src/models/classification_gate_overlay.py: allow an optional limited veto rule driven by portfolio-target classifier
  - scripts/monthly_decision.py: include gate summary but do not change live sell logic until promotion_approved flag is set
- Promotion criteria (hard requirements):
  - >= 24 matured prospective (real-time) months
  - Brier <= 0.18 and ECE <= 0.08 on matured months
  - covered-month balanced accuracy >= 0.60 with coverage >= 0.25
  - actionable-sell precision >= 0.70
  - agreement with baseline >= 0.90 and max consecutive would-change months <= 3
  - non-negative cumulative shadow-minus-live over matured months, including disagreement-month subset
</proposed_plan>
```