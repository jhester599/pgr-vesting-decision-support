# PGR classifier layer: design review and v123+ research plan

The classification layer should be promoted to a **veto-gate role** using **portfolio-weighted aggregation** with fixed redeploy weights, after switching from prequential logistic calibration to temperature scaling and reducing per-benchmark feature sets to 6–8 features. The system's 162-month shadow track record far exceeds standard promotion thresholds, and the ~8% cumulative uplift on disagreement months provides economically meaningful signal — but the architecture must first close the portfolio alignment gap by adding VGT (now) and SCHD (deferred 12–18 months). Path A (per-benchmark classifiers with portfolio-weighted aggregation) should remain the primary architecture; Path B (composite target) should be investigated only as a secondary diagnostic. The single highest-impact change for v123 is replacing the current quality-weighted aggregation with redeploy-portfolio-weighted aggregation over the 4 overlapping benchmarks, which directly aligns the classifier's output with the economic decision it supports.

---

## Question 1: Portfolio-weighted aggregation is statistically valid and economically superior

**1a. Constructing the composite P(Actionable Sell).** Computing P_composite = Σ(w_i × P_i) where w_i are redeploy portfolio weights and P_i are per-benchmark calibrated probabilities is a **linear opinion pool** — a well-studied construct in the forecast combination literature. Geweke & Amisano (2011) established that linear prediction pools produce proper probabilities when constituent classifiers are individually calibrated and weights are non-negative and sum to one. Portfolio allocation weights satisfy both constraints by construction.

The statistical properties of this estimator differ from the current quality-weighted scheme in an important way. Quality weights (OOS R², IC, hit-rate) optimize for **forecast accuracy** — they upweight benchmarks where the classifier discriminates best. Portfolio weights optimize for **economic relevance** — they upweight benchmarks representing the largest share of the user's actual redeploy allocation. The forecast combination literature, particularly Granger & Machina (2006) and the decision-based model combination framework of Billio et al. (2013), establishes that when forecasts serve a specific economic decision, weighting by economic criteria can outperform pure statistical weighting. Since the classifier exists solely to inform a hold/sell decision against the user's actual redeploy portfolio, **portfolio weights are the methodologically correct choice** for the primary signal.

A hybrid approach is most defensible in practice: w_i_adjusted = (portfolio_weight_i × quality_score_i) / Σ(portfolio_weight_j × quality_score_j). This preserves economic alignment while downweighting benchmarks where the classifier has poor discrimination. However, given the forecast combination puzzle — the robust finding from Claeskens et al. (2016), Stock & Watson (2004), and Smith & Wallis (2009) that estimation error in optimized weights overwhelms theoretical gains at small sample sizes — the quality adjustment should be minimal. A simple binary quality filter (include/exclude based on minimum OOS hit-rate threshold) is preferable to continuous quality weighting at N≈200.

**1b. Handling non-investable benchmarks (DBC, GLD, VMBS, VDE).** The recommended approach is **Option A with a diagnostic overlay**: exclude from the primary P(Actionable Sell) computation, but retain as regime-conditioning indicators. This two-layer architecture is directly supported by Hendry & Hubrich (2011), who showed that including disaggregate information as conditioning variables in the aggregate model often outperforms both pure aggregation and pure exclusion.

Concretely, the non-investable benchmark classifiers should continue running in shadow mode. Their outputs can serve as a **regime confidence modifier** — if 3 of 4 non-investable benchmarks agree with the primary signal direction, confidence in the primary signal increases; if they strongly disagree, it suggests the primary signal may be regime-specific rather than broad-based. This preserves the informational content (commodity cycles, gold/real-rate regimes, mortgage credit conditions) without contaminating the portfolio-weighted signal with non-actionable components.

Option C (small non-zero insurance weight) is theoretically appealing but practically dangerous at N≈200. The tactical asset allocation literature (Ang & Bekaert, 2004; Guidolin & Timmermann, 2007) uses non-investable signals as conditional regime indicators rather than direct allocation inputs, which supports the diagnostic-overlay approach.

**1c. Fixed vs. dynamic aggregation weights.** **Fixed weights are strongly recommended.** The forecast combination puzzle is the most important empirical regularity here. DeMiguel, Garlappi & Uppal (2009) demonstrated that approximately **3,000 monthly observations** are needed for mean-variance optimized portfolio weights to outperform naive 1/N allocation with 25 assets. By direct analogy, estimated optimal combination weights for 8 classifiers would require data far exceeding the available ~200 observations per benchmark. Dynamic weights compound this problem by requiring estimation of time-varying parameters, which Del Negro et al. (2016) found beneficial only with hundreds of quarterly observations.

The recommended implementation uses the **base `balanced_pref_95_5` weights**, renormalized to sum to one across the overlapping benchmarks (VOO, VXUS, VWO, BND). These weights should be updated only when the user's target allocation changes — not monthly, not based on signal-tilted weights. Using the dynamic monthly signal-tilted weights would introduce circular dependency (the allocation system's output feeding back into the classification system's input) and would amplify estimation noise. Annual review of base weights is sufficient and operationally clean.

---

## Question 2: Add VGT now, defer SCHD, reduce features for both

**2a. VGT feature analysis.** Technology stocks behave as **long-duration equity assets** — their valuation is dominated by distant future cash flows, making them disproportionately sensitive to real interest rates. The 2022 episode (10-year Treasury yield rising from 1.6% to 4.2% while the Nasdaq fell ~35%) is the canonical illustration. For a PGR-vs-VGT comparison, the key dynamic is whether PGR's insurance fundamentals or VGT's growth-duration profile will dominate relative returns.

From the current lean baseline, the most relevant features for PGR-vs-VGT are: **`real_rate_10y`** and **`real_yield_change_6m`** (capturing VGT's duration sensitivity), **`vix`** (growth stocks are high-beta to volatility regimes), and **`yield_slope`** (yield curve inversions historically compress growth premiums). The least relevant are **`combined_ratio_ttm`** and **`npw_growth_yoy`** — PGR-specific insurance fundamentals that tell the model about PGR's earnings quality but provide no signal about VGT's return dynamics. For a relative-return target, these insurance features serve only the PGR side of the comparison; while still informative, their marginal value is lower for VGT than for bond or commodity benchmarks where PGR's fundamentals drive most of the relative return variance.

The strongest candidate supplemental feature is the **change in 10-year TIPS real yield (DFII10)**, which is already proxied by `real_yield_change_6m` in the baseline — confirming the lean baseline captures the dominant VGT signal channel. A second candidate is **ISM Manufacturing New Orders (inverse)**, reflecting Macrosynergy research showing that tech sector relative performance is counter-cyclically correlated with manufacturing sentiment (tech's revenue stability creates outperformance when cyclical sectors weaken). This is available from FRED.

**2b. SCHD feature analysis.** The dividend/value factor's relative performance is most strongly predicted by the **value spread** — the valuation gap between cheap and expensive stocks. Asness et al. (2000), Cohen et al. (2003), and Baba Yara et al. (2017) all document that wider value spreads predict higher subsequent HML returns, with the signal working across equities, commodities, currencies, and bonds. For PGR-vs-SCHD, the relevant question is whether PGR (a specific growth-quality insurer) will outperform a diversified dividend-value basket.

From the lean baseline, the most relevant features are **`yield_slope`** (value stocks tend to outperform in steepening environments as economic recovery favors cyclicals), **`credit_spread_hy`** (value stocks carry more financial leverage and are credit-sensitive), and **`real_rate_10y`** (value's shorter duration benefits from rising rates). The least relevant is **`investment_income_growth_yoy`** — a PGR-specific fundamental with no connection to the value factor.

The strongest supplemental candidates are: (1) **5-year breakeven inflation rate (T5YIE)** — value/dividend stocks have documented positive inflation beta (AllianceBernstein), available from FRED; (2) **value-growth spread** — constructable from Fama-French HML portfolio data (publicly available from Ken French's website) or from Alpha Vantage sector P/E ratios. The value spread is the single most theoretically and empirically supported predictor of value factor returns, but constructing it from Alpha Vantage data introduces a new data pipeline dependency.

**2c. VGT inclusion risks and mitigation.** VGT's actual ETF history extends from January 2004 — roughly **264 monthly observations** (not 140–180 as initially estimated). This is comparable to the existing benchmark suite's 176–252 range and adequate for regularized logistic regression with a reduced feature set. The MSCI US IMI IT 25/50 Index data extends further via back-tested index returns, but these are explicitly hypothetical and should not be used for classifier training.

The primary risk is **events per variable (EPV)**. With ~264 observations, ~30% positive rate (~79 events), and 12 features, EPV is approximately **6.6** — below the traditional 10 EPV threshold and well below Harrell's recommended 20 EPV. However, van Smeden et al. (2019) showed EPV alone is not a strong predictor of model performance, and regularization (Ridge/LASSO) provides meaningful benefits in this regime. The recommended mitigation is **reducing the feature set to 6–8 features** for VGT specifically, which raises EPV to ~10–13 and enters a defensible range. L2 regularization should be the default; Firth's penalized regression is a strong alternative showing "remarkable resilience under small sample conditions" even at n=20.

**2d. SCHD: defer inclusion.** SCHD has only ~**168 monthly observations** (inception October 2011), yielding approximately **50 events** at 30% positive rate. With 12 features, EPV is **4.2** — firmly in the zone where logistic regression produces biased, unstable coefficients even with regularization. The Dow Jones US Dividend 100 Index was launched August 2011, essentially the same date as the ETF, so all pre-inception index data is back-tested and hypothetical.

SCHD should be **deferred 12–18 months** (to ~v135–v140) until its history reaches ~185+ observations. In the interim, two actions can prepare for eventual inclusion: (1) begin constructing SCHD return series and the `actionable_sell_3pct` target now so the target is ready when training begins; (2) evaluate whether the **Dow Jones US Select Dividend Index** (inception November 2003, ~270 months of live data) serves as an adequate proxy for SCHD's return dynamics, which would enable classifier development on a longer-lived proxy while monitoring SCHD's actual track record.

---

## Question 3: Benchmark-specific features can help, but only with strict guardrails

**3a. When benchmark-specific selection improves performance.** The v88 finding that expanding beyond 12 features consistently hurt balanced accuracy is a textbook manifestation of the **Hughes peaking phenomenon** — at fixed sample sizes, classifier accuracy increases with features up to a peak, then declines as estimation noise overwhelms signal. With 176–252 training samples and ~53–76 events, the system operates at an EPV of **4.4–6.3** with 12 features. The peaking literature (Jain & Waller, 1978; Zollanvari et al., 2020) predicts that the optimal feature count for weak financial signals (Mahalanobis distance Δ² < 1) at this sample size is likely **5–8 features**.

Benchmark-specific feature *selection* (choosing different subsets of the existing 12 features per benchmark) is therefore likely to **improve** performance, because it reduces the effective feature count while preserving benchmark-relevant signal. This is categorically different from benchmark-specific feature *expansion* (adding new features beyond the 12), which v88 correctly found harmful. The key condition is that different benchmarks must have genuinely different feature relevance — which they clearly do (insurance fundamentals matter for PGR-vs-BND but not for PGR-vs-VGT's growth-duration dynamics).

**3b. Candidate features per asset class.** All features below are constructable from FRED unless noted. Features marked with † introduce a new data-source dependency.

- **Bonds (BND, VMBS):** ACM 10-Year Term Premium (FRED: THREEFYTP10) — directly measures duration risk compensation. Yield curve slope change (DGS10 – FEDFUNDS) is partially captured by existing `yield_slope`.
- **Commodities (DBC):** 5-Year Breakeven Inflation change (FRED: T5YIE) — commodities have strong positive correlation with inflation expectations. ISM Manufacturing New Orders momentum captures industrial demand cycle.
- **Gold (GLD):** Trade-Weighted US Dollar Index change (FRED: DTWEXBGS) — gold has a historical correlation of approximately –0.75 with USD. The TIPS real yield (DFII10) has been the dominant gold predictor historically (R² of 84% from 2005–2021), though this relationship weakened significantly post-2022 due to central bank buying.
- **Energy equity (VDE):** Crude oil price momentum† (Alpha Vantage or EIA) and breakeven inflation change (T5YIE). Oil price momentum is the single strongest VDE predictor.
- **US large-cap (VOO, VGT):** Earnings revision breadth† (Alpha Vantage forward EPS estimates) — requires constructing an up/down revision ratio. ISM Manufacturing New Orders (inverse, for VGT specifically).
- **International/EM (VXUS, VWO):** Trade-Weighted USD change (DTWEXBGS) — strongest documented predictor of international vs. US equity relative performance. EM-DM growth differential† would require IMF/World Bank data not available on FRED.
- **Dividend/value (SCHD):** Value spread† (Fama-French HML data or Alpha Vantage P/E quintiles) and 5-Year Breakeven Inflation (T5YIE).

**Recommendation for v123:** Do not add new features to the pipeline yet. Instead, implement benchmark-specific feature *subsetting* from the existing 12 features using L1 selection within WFO (see 3c). New feature candidates should be staged for v125+ after the subsetting approach is validated.

**3c. Feature selection methodology.** L1 logistic regularization within each WFO training fold is **valid and does not introduce leakage**, provided all selection happens within the training window. This is an embedded feature selection method — regularization and model fitting occur simultaneously, using only past data at each WFO step. Andrew Ng's ICML 2004 analysis showed L1 makes logistic regression "extremely insensitive to irrelevant features" even with 100 examples.

However, L1 feature selection is **unstable at N≈200** — different WFO folds may select different features (Nogueira et al., 2017). Two mitigation strategies are recommended: (1) **Stability selection** (Meinshausen & Bühlmann, 2010): run L1 across multiple bootstrap resamples within each training window and select features appearing in >60% of resamples. (2) **Fix λ conservatively** rather than tuning via nested CV within each fold — nested CV with inner folds of ~30–50 observations introduces excessive variance. A conservative λ that retains 6–8 features is safer than data-driven λ selection.

The two-step L1→L2 procedure (L1 for selection, L2 for final model) is supported by the literature for small samples and should be adopted: use LASSO to identify 6–8 features per benchmark, then fit the final classifier with Ridge on those features.

---

## Question 4: Per-benchmark aggregation beats composite target at N≈200

**4a. Theoretical and practical tradeoffs.** The forecast aggregation literature provides a clear hierarchy. Lütkepohl (1984) proved that in population (known DGPs, infinite data), disaggregate-then-aggregate is at least as efficient as direct aggregate forecasting, because disaggregate models exploit cross-component dynamics lost in aggregation. Hendry & Hubrich (2011) extended this to show that including disaggregate information in the aggregate model is theoretically best of all.

However, **estimation uncertainty reverses these rankings in finite samples.** When DGPs must be estimated from limited data, the parameter burden of separate models can overwhelm the information advantage. Bermingham & D'Agostino (2014) found that papers unsupportive of disaggregate forecasting "often have short spans of data" — directly relevant to N≈200. The Marcellino, Stock & Watson (2003) Euro-area study confirms the flip side: when components are genuinely heterogeneous (as equity, bond, commodity, and gold benchmarks clearly are), disaggregate approaches outperform.

For the PGR system, **Path A (per-benchmark classifiers → portfolio-weighted aggregation) is recommended as the primary architecture** because: (1) benchmarks span fundamentally different asset classes with heterogeneous return dynamics, making a single composite model poorly specified; (2) the per-benchmark approach preserves interpretability and allows attribution of which benchmark is driving the signal; (3) with portfolio-weight aggregation and no weight estimation, there is zero estimation overhead in the combination step.

**4b. Composite target noise.** In Path B, the composite return r_portfolio = Σ(w_i × r_i) conflates benchmark-specific regime information before binarization. A bear market in bonds masked by a bull market in equities creates **label noise** — months that should be "actionable sell" against bonds are labeled "hold" because the portfolio-weighted composite exceeded the threshold. At N≈200, this label noise is particularly damaging because the classifier has limited capacity to learn the underlying conditional relationships. Bermingham & D'Agostino (2014) found empirically that "the aggregate forecast often has the least satisfactory performance" across their test datasets.

There is no strong literature precedent for composite-return-target binary classification in active equity management specifically. The closest parallel is the multi-benchmark tracking error literature, which universally treats benchmarks separately and aggregates tracking decisions rather than constructing composite return targets.

**4c. Recommended research priority.** Path A as primary, Path B as secondary diagnostic only. Path A should receive **~80% of research effort** in v123–v125. Path B can be investigated as a v126+ experiment: train a single classifier on the composite `balanced_pref_95_5`-weighted return target and compare its Brier score, balanced accuracy, and calibration to the portfolio-weighted Path A aggregate. This comparison will definitively answer whether the composite target's label noise problem outweighs its estimation efficiency advantage at the specific N≈200 regime of this system.

---

## Question 5: Temperature scaling resolves the calibration–balanced accuracy conflict

**5a. Mechanistic explanation.** Yes, it is well-documented that prequential logistic calibration can reduce balanced accuracy while improving Brier score and ECE. The Murphy (1973) Brier score decomposition into **BS = Reliability – Resolution + Uncertainty** reveals the mechanism. Post-hoc calibration directly reduces Reliability (calibration error), which improves Brier score. But some calibration methods also reduce Resolution (the model's ability to separate classes) by shrinking predictions toward the base rate.

With ~30% positive class rate, the mechanism is concrete: prequential logistic calibration learns an intercept reflecting the base rate (~0.3), which pulls uncertain predictions below the 0.5 decision threshold. Cases with true probabilities in the 0.3–0.5 range — which are genuine positives under uncertainty — get shifted below the 0.5 decision boundary, reducing sensitivity (true positive rate) without proportionally improving specificity. Since balanced accuracy = (sensitivity + specificity) / 2, losing sensitivity hurts balanced accuracy even as calibration metrics improve. The Jiang et al. (2012) analysis in PLoS ONE formally demonstrated that "perfect calibration may harm discrimination."

**5b. Calibration method recommendations.** Evaluating six approaches at N≈200 with 30% positive rate:

**Temperature scaling is the recommended default.** It uses a single parameter T that divides logits before the sigmoid: p = σ(z/T). Guo et al. (2017) demonstrated it is **accuracy-preserving** — it does not change prediction rankings and therefore cannot degrade AUC, accuracy, or balanced accuracy at any fixed threshold. With only 1 parameter, overfitting risk at N≈200 is negligible. The limitation is expressive power: for logistic regression (which already uses a canonical link function), the improvement may be modest. But critically, it **cannot make things worse**, unlike prequential logistic calibration which provably shifts the decision boundary.

**Class weighting during training is the best approach if balanced accuracy is the primary metric.** Setting `class_weight` to 'balanced' or tuning the minority class weight (search over {1.5, 2.0, 2.33, 3.0}) directly adjusts the loss function to penalize minority class errors, shifting the learned decision boundary rather than post-hoc adjusting probabilities. Van den Goorbergh et al. (2022, JAMIA) showed this achieves comparable sensitivity/specificity balance to resampling approaches without the calibration distortion. The tradeoff is less reliable probability estimates — but if the system's primary output is a binary decision (hold/sell) rather than a precise probability, this is acceptable.

**Isotonic regression is not recommended** at N≈200. Niculescu-Mizil & Caruana (2005) explicitly recommend it only "when there is enough data (greater than ~1000 samples)." **Prequential logistic calibration** (current approach) should be replaced — it is the likely source of the observed balanced accuracy degradation. **Platt scaling** shares the same mechanistic problem (intercept encodes base rate). **Beta calibration** (Kull et al., 2017) has 3 parameters offering asymmetric flexibility useful for imbalanced data, but at N≈200 the additional parameters may slightly overfit versus temperature scaling's single parameter.

The recommended pipeline for v123: (1) train with class_weight tuned on WFO validation fold, (2) apply temperature scaling for calibration, (3) optimize asymmetric thresholds (see 5c).

**5c. Threshold and abstention optimization.** Per-benchmark threshold optimization within WFO validation folds is **valid and does not introduce leakage** — it is analogous to hyperparameter tuning on the validation set of each walk-forward step. The threshold is optimized on past data and applied to the test period, strictly respecting temporal ordering.

The current uniform abstention band (0.30–0.70) should be replaced with **asymmetric, cost-sensitive thresholds.** The literature strongly supports this. Gandouz et al. (2021, BMC Medical Informatics) showed that asymmetric abstention intervals are "better suited for imbalanced data" and "reject as many or fewer samples compared to symmetric abstention." Fumera et al. (2000) demonstrated that Chow's optimal reject rule performs poorly when posterior probabilities are noisy, advocating for separate rejection thresholds per class.

Given the asymmetric error costs (false positive = unnecessary tax event >> false negative = missed diversification), the **sell threshold should be higher** than the hold threshold. The cost-sensitive threshold formula is: threshold = C_FP / (C_FP + C_FN). If the tax-event cost is approximately 2× the missed-diversification cost, the sell threshold should be ~0.67 rather than 0.50. A concrete implementation: abstain when P(Actionable Sell) ∈ [0.25, 0.67], hold when P < 0.25, sell-signal when P > 0.67. These thresholds should be optimized on the WFO validation fold using a cost-weighted variant of Youden's J statistic: maximize (w₁ × sensitivity + w₂ × specificity) where w₁ < w₂ reflects the asymmetric costs.

Per-benchmark thresholds are theoretically valid but practically risky at N≈200 — **pooled thresholds across benchmarks** are more stable. If per-benchmark thresholds are tested, check stability across WFO folds. If selected thresholds vary by more than ±0.10 across folds, the per-benchmark approach is overfitting and pooled thresholds should be used.

---

## Question 6: Promote to veto gate after architectural improvements land

**6a. Promotion criteria.** The system's **162-month shadow track record** far exceeds any reasonable minimum observation period. Regulatory guidance (SR 11-7, OCC 2011-12) requires validation "at least annually" and prospective testing against actuals at a frequency matching the forecast horizon. Industry practice for monthly-frequency financial classifiers typically requires **12–24 prospective months** as a minimum parallel run. The 162-month track record is exceptional and is not the bottleneck for promotion.

The promotion gate should check five metrics, measured over a trailing 24-month window with no degradation trend:

- **Calibration stability:** ECE < 0.10 and no statistically significant drift (Hosmer-Lemeshow or similar); the current calibrated path ECE of 0.0813 is acceptable but should be monitored for upward trends.
- **Balanced accuracy on non-abstained predictions:** ≥55% (above random chance adjusted for class imbalance); the current 51.32% after calibration is below this threshold, which is precisely why the calibration approach must change before promotion.
- **Agreement rate with regression baseline:** >85% (the current 93.83% is well above this floor; paradoxically, very high agreement reduces the value of promotion but also reduces its risk).
- **Cumulative policy uplift on disagreement months:** Positive and statistically distinguishable from zero; the current 0.0812 across 10 disagreement months should be evaluated per-disagreement to assess whether the uplift is concentrated in a few months or broadly distributed.
- **No catastrophic disagreements:** No single disagreement month where the classifier's recommendation would have caused >5% portfolio loss relative to the regression baseline.

**6b. Recommended production role: Option A (veto gate).** The asymmetric error cost structure — where false positives (unnecessary tax events in a taxable account) produce concrete, irreversible harm while false negatives (missed diversification) produce uncertain opportunity cost — strongly favors a conservative architecture. A veto gate means the regression layer's sell signal requires classifier confirmation before execution. If the classifier says "hold," the sell is blocked.

This maps directly to the **Neyman-Pearson paradigm** (Li et al., 2020): when one error type is much costlier, the decision rule should control that error type with high probability. A veto gate achieves this by requiring model agreement for the costly action (sell/tax event). The expected effects: false positive rate **significantly reduced** (both models must agree on sell); false negative rate **moderately increased** (some valid sell signals vetoed). Given that the classifier agrees with the regression 93.83% of the time, the veto gate would block approximately **1–2 sell signals per year** that the regression recommends but the classifier doesn't confirm.

Options B (permission-to-deviate) and C (co-equal) are inappropriate because they create new pathways for sell signals, increasing the false positive rate. Option D (confidence tier modifier) is the natural graduation path — after 12+ months of veto-gate operation demonstrating that blocked sells were indeed correct holds, the classifier can transition to modulating the regression signal's confidence level rather than binary gating.

**6c. Strongest argument for staying shadow-only.** The high agreement rate (93.83%) means promotion **adds minimal decision value** while introducing real operational complexity, governance burden, and automation bias risk. With only 10 disagreement months in 162, the classifier is an independent voice on just 6.17% of decisions. The governance overhead of SR 11-7 compliance (independent validation, ongoing monitoring infrastructure, documentation) may exceed the expected benefit.

This argument also has a deeper behavioral dimension: Sele & Chugunova (2022) demonstrated that human-in-the-loop systems increase decision uptake but *decrease* accuracy — operators become less likely to intervene on the largest errors when an automated system is providing recommendations. Keeping the classifier shadow-only preserves a natural "human on the loop" safety layer.

**This argument loses force under three conditions:** (1) if the 10 disagreement months show large, consistent uplift (not just marginally positive) — particularly if the classifier correctly identified regime changes the regression missed; (2) if the regression baseline begins degrading (regime change rendering the regression's macro features less predictive), making the classifier's independent signal more urgent; (3) if the disagreement rate increases above ~10% of months, indicating the classifier is capturing genuinely different information rather than redundant signal. The current cumulative uplift of 0.0812 over 10 months averages ~0.008 per disagreement — this needs to be decomposed: if 2–3 months drive most of the uplift, the classifier is capturing occasional tail events (valuable for veto gate); if uplift is evenly distributed, the signal is more noise-like.

---

## Recommended feature candidates

| Benchmark | Feature | Source | Priority | New dependency? |
|-----------|---------|--------|----------|-----------------|
| VGT | TIPS real yield change (DFII10) | FRED | High | No (proxied by `real_yield_change_6m`) |
| VGT | ISM Mfg New Orders (inverse) | FRED | Medium | Yes — ISM data feed |
| SCHD | 5Y Breakeven Inflation (T5YIE) | FRED | Medium | No (FRED already used) |
| SCHD | Value spread (HML quintile P/E gap) | Ken French / Alpha Vantage | Medium | Yes — new data pipeline |
| BND/VMBS | ACM 10Y Term Premium (THREEFYTP10) | FRED | Medium | No |
| GLD | Trade-Weighted USD change (DTWEXBGS) | FRED | Medium | No |
| VDE | Crude oil price momentum | Alpha Vantage / EIA | Medium | Partially — oil price series |
| VXUS/VWO | Trade-Weighted USD change (DTWEXBGS) | FRED | High | No |

**Do not add new features in v123.** First implement benchmark-specific feature subsetting from the existing 12-feature lean baseline using the L1→L2 two-step procedure within WFO. New feature candidates should be staged for v125+ after subsetting is validated. The highest-priority additions (DTWEXBGS for international benchmarks, DFII10 for VGT) are already proxied by existing baseline features (`real_yield_change_6m`, `real_rate_10y`), confirming the lean baseline's design quality.

---

## Recommended architecture for v123+

**Aggregation:** Portfolio-weighted aggregation (Path A) using fixed `balanced_pref_95_5` base weights, renormalized across overlapping benchmarks. Non-investable benchmarks (DBC, GLD, VMBS, VDE) excluded from primary signal, retained as regime diagnostics.

**Benchmark expansion:** Add VGT to classifier suite in v123. Defer SCHD to v135–v140 (~late 2027) when history reaches ~185 observations. Begin SCHD target construction and proxy validation immediately.

**Feature approach:** Implement benchmark-specific feature subsetting (L1→L2 two-step within WFO) in v124. Retain lean 12-feature baseline as the full candidate pool. Target 6–8 features per benchmark. Defer new feature additions to v125+.

**Calibration:** Replace prequential logistic calibration with temperature scaling. Implement cost-sensitive class weighting during training. Optimize asymmetric abstention thresholds on WFO validation folds.

**Composite target (Path B):** Investigate as a secondary diagnostic experiment in v126. Compare composite-target classifier to Path A aggregate on all standard metrics. Do not plan for production use unless it demonstrates >3% balanced accuracy improvement.

---

## Implementation plan for v123+

### v123: Portfolio alignment and VGT addition
**Research scripts:**
- `v123_portfolio_weighted_aggregation.py`: Implement renormalized portfolio-weight aggregation over {VOO, VXUS, VWO, BND}. Compare against current quality-weighted aggregation on full shadow history. Report: agreement rate, policy uplift, calibration metrics under both schemes.
- `v123_vgt_classifier.py`: Train per-benchmark logistic classifier for PGR-vs-VGT. Use full lean_baseline (12 features). Report OOS balanced accuracy, Brier, calibration curve. Compare against existing benchmark classifiers.
- `v123_non_investable_diagnostic.py`: Compute regime-conditioning signal from non-investable benchmarks (DBC, GLD, VMBS, VDE). Assess whether agreement/disagreement with primary signal predicts primary signal accuracy.

**Validation checks:** Portfolio-weighted aggregate must show non-negative policy uplift relative to quality-weighted aggregate on held-out shadow period. VGT classifier must achieve balanced accuracy >55% and Brier <0.25. Regression diagnostics on VGT classifier must show no sign of complete separation or coefficient explosion.

### v124: Calibration overhaul and feature subsetting
**Research scripts:**
- `v124_temperature_scaling.py`: Replace prequential logistic calibration with temperature scaling. Report balanced accuracy, Brier, ECE before/after. Confirm balanced accuracy is preserved or improved.
- `v124_class_weighting.py`: Tune class_weight parameter per benchmark on WFO validation folds. Report optimal weights and balanced accuracy impact.
- `v124_feature_subsetting.py`: Implement L1→L2 two-step feature selection per benchmark within WFO. Report selected features per benchmark, stability across folds, balanced accuracy vs. full 12-feature baseline.
- `v124_asymmetric_thresholds.py`: Optimize cost-sensitive abstention thresholds per benchmark. Report abstention rate and balanced accuracy on non-abstained predictions.

**Validation checks:** Temperature scaling must not reduce balanced accuracy (by construction, but verify). Feature subsetting must show stable feature selection (>60% consistency across WFO folds). Asymmetric thresholds must improve cost-weighted balanced accuracy relative to uniform 0.30–0.70 band.

### v125: Integration and shadow refresh
**Shadow additions:**
- Deploy updated classifier (portfolio-weighted aggregation, temperature scaling, benchmark-specific feature subsets, VGT added, asymmetric thresholds) in shadow mode alongside current v122 shadow.
- Run dual-shadow for minimum 6 months comparing v122 architecture vs. v125 architecture.

**Files likely to change:** `classifier_config.yaml` (benchmark list, feature sets, aggregation weights), `calibration.py` (temperature scaling replacing prequential logistic), `aggregation.py` (portfolio-weighted replacing quality-weighted), `thresholds.py` (asymmetric cost-sensitive), `shadow_monitor.py` (dual-shadow comparison logic), GitHub Actions workflow YAML (VGT data ingestion).

### v126: Composite target experiment (secondary)
- `v126_composite_target.py`: Train single classifier on `balanced_pref_95_5`-weighted composite return target. Compare Brier, balanced accuracy, calibration against Path A aggregate.
- Decision gate: If composite target shows >3% balanced accuracy improvement with comparable calibration, elevate to co-primary research track. Otherwise, archive.

### v130+: SCHD preparation
- Begin constructing SCHD target series and proxy analysis (Dow Jones US Select Dividend Index).
- Evaluate Firth's penalized regression as alternative estimator for short-history benchmarks.

### v135–v140: SCHD inclusion (conditional)
- Add SCHD classifier with reduced feature set (6–8 features).
- Promotion gate: same criteria as VGT in v123.

### Promotion gate criteria for production (v125+ shadow period)
- Minimum 12 months of v125-architecture shadow operation
- Balanced accuracy on non-abstained predictions ≥55% over trailing 24 months
- ECE <0.10 with no upward trend
- Positive cumulative policy uplift on disagreement months
- No single disagreement month with >5% adverse portfolio impact
- Agreement rate with regression baseline 80–97% (too-high agreement suggests redundancy; too-low suggests instability)

---

## Proposed plan

```
proposed_plan:
  primary_research_direction: "Portfolio-weighted per-benchmark aggregation (Path A) with VGT expansion, temperature scaling, and benchmark-specific feature subsetting"
  
  v123_objectives:
    - "Implement portfolio-weighted aggregation over {VOO, VXUS, VWO, BND} using fixed balanced_pref_95_5 weights"
    - "Add VGT per-benchmark classifier with full lean_baseline features"
    - "Reclassify DBC, GLD, VMBS, VDE as regime-diagnostic (non-primary) signals"
    - "Begin SCHD target construction and proxy validation"
  
  v124_objectives:
    - "Replace prequential logistic calibration with temperature scaling"
    - "Implement cost-sensitive class weighting tuned per benchmark on WFO validation folds"
    - "Run L1→L2 two-step benchmark-specific feature subsetting (target 6-8 features per benchmark)"
    - "Optimize asymmetric abstention thresholds with cost-sensitive Youden's J"
  
  v125_objectives:
    - "Deploy updated architecture in dual-shadow alongside v122"
    - "Begin 12-month minimum shadow observation for promotion gate"
  
  v126_objectives:
    - "Composite target (Path B) experiment as secondary research track"
    - "Decision gate: promote to co-primary only if >3% balanced accuracy improvement"
  
  v135_v140_objectives:
    - "SCHD classifier inclusion (conditional on 185+ observations)"
    - "Evaluate Firth's penalized regression for short-history benchmarks"
  
  architecture_decisions:
    aggregation: "Portfolio-weighted (fixed redeploy weights), NOT quality-weighted"
    composite_target: "Secondary experiment only (v126)"
    vgt: "Include now (v123)"
    schd: "Defer to v135-v140"
    feature_approach: "Benchmark-specific subsetting from lean baseline first (v124); new features deferred to v125+"
    calibration: "Temperature scaling (replaces prequential logistic)"
    thresholds: "Asymmetric cost-sensitive, pooled across benchmarks initially"
    production_role: "Veto gate (regression sell requires classifier confirmation)"
  
  promotion_criteria:
    min_shadow_months: 12
    balanced_accuracy_floor: 0.55
    ece_ceiling: 0.10
    policy_uplift: "positive and statistically distinguishable from zero"
    catastrophic_disagreement_ceiling: "0.05 single-month adverse impact"
    agreement_rate_band: [0.80, 0.97]
  
  key_risks:
    - "VGT feature relevance may differ substantially from existing benchmarks; monitor coefficient stability"
    - "Temperature scaling may provide minimal improvement over no calibration for logistic regression (already canonical link)"
    - "Benchmark-specific feature subsetting may produce unstable selections across WFO folds"
    - "Portfolio-weight aggregation over 4 benchmarks loses diversification benefit of 8-benchmark suite"
  
  mitigation_strategies:
    - "Stability selection (>60% bootstrap consistency) as guard on feature subsetting"
    - "Retain non-investable benchmarks as regime diagnostics to preserve informational breadth"
    - "Dual-shadow comparison (v122 vs v125) provides controlled evaluation"
    - "Conservative λ for L1 step; avoid nested CV within WFO folds"
    - "If temperature scaling provides <0.005 Brier improvement, fall back to class weighting only (no post-hoc calibration)"
```