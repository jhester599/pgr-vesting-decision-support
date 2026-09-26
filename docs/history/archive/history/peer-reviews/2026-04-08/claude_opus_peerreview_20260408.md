# Fixing Negative OOS R² When Your Model Already Has Signal

## Prediction Calibration for PGR Relative Returns

**Your model's directional signal is real — the problem is prediction calibration, not predictive power.** The coexistence of IC=0.19 and hit rate=69% with OOS R²=−13% is a well-documented phenomenon in financial ML: the model correctly ranks outcomes but generates predictions with far too much variance. Mathematically, a perfectly calibrated model with IC=0.19 would achieve R²=IC²=**3.6%** — exceeding your +2% threshold. The entire **16.6 percentage point gap** between that theoretical ceiling and the observed −13% is attributable to prediction scale mismatch and bias drift across rolling windows. This diagnosis reshapes the problem from "find more signal" to "stop amplifying noise," and the highest-leverage fix is dramatically simpler than the sophisticated methods already attempted.

---

## 1. The Central Puzzle: Why Positive IC Coexists with Negative R²

The disconnect between rank correlation and mean squared error has a precise mathematical explanation. **Information Coefficient measures whether higher predictions correspond to higher actuals (ordinal relationship), while R² measures whether prediction magnitudes match actual magnitudes (cardinal accuracy).** A model can rank perfectly yet produce R²=−∞ if its predictions are scaled 10× too aggressively.

MSE decomposes into three components: bias² (mean prediction error), variance (prediction volatility), and irreducible noise. For the PGR model, the Ridge+GBT ensemble almost certainly suffers from **variance dominance** — particularly the GBT component, which with only 30–60 training observations per fold is fitting noise and producing extreme predictions. When the model is wrong on magnitude, the squared errors from those misses overwhelm the signal from correct directional calls. A few catastrophic predictions in a 120-observation OOS sample can drive R² deeply negative.

This is not exotic. Goyal and Welch (2008) documented that most equity premium predictors produce negative OOS R² even with centuries of data. Campbell and Thompson (2008) showed that the **expected OOS R² under the null of no predictability is itself negative** because parameter estimation consumes degrees of freedom. Clark and West (2006) developed a test specifically for this situation — their MSFE-adjusted statistic can detect genuine predictability even when R² is negative. The Grinold and Kahn Fundamental Law of Active Management uses IC, not R², precisely because portfolio construction depends on prediction ranking, not calibration. With IC=0.19 and breadth of ~16 semi-independent forecasts per year, the implied information ratio is **IR ≈ 0.76** — a genuinely strong result by quantitative finance standards.

Kelly, Pruitt, and Xiu (2024, *Journal of Finance*) proved that shrinking predictions toward zero reduces forecast variance faster than it degrades signal, mechanically improving R². This is the James-Stein phenomenon applied to return prediction: in high-noise settings, shrinkage toward the null hypothesis always reduces total MSE when estimating three or more parameters simultaneously.

---

## 2. Prediction Shrinkage: The Single Highest-Leverage Intervention

The simplest fix requires three lines of code. Multiply all raw predictions by a shrinkage factor α ∈ [0.10, 0.20], selected via inner cross-validation within each training window. If raw predictions have standard deviation σ\_pred and the true conditional signal has standard deviation σ\_signal ≪ σ\_pred, then R² is approximately 2ρ(σ\_signal/σ\_pred) − (σ\_pred/σ\_signal)². When σ\_pred is too large, R² goes deeply negative despite positive ρ. Rescaling predictions by α reduces σ\_pred proportionally, converting R² from negative to positive while preserving the correlation structure — hit rate stays at ~69%.

**Implementation is trivial**: `y_pred_final = alpha * y_pred_raw` where alpha is tuned via leave-one-out CV within the training window. Campbell and Thompson's (2008) sign-constraint approach — which converted many negative-R² predictors to positive — is a discrete version of this same principle. Swanson et al. (2024) found that shorter windows and nonlinear methods particularly require high levels of forecast winsorization, directly applicable to a 60-month rolling GBT.

Three complementary shrinkage mechanisms should be applied simultaneously:

### 2a. Increase Ridge α by 10–100×

The current regularization strength is almost certainly too low for 13 features with 60 observations. Use `RidgeCV(alphas=np.logspace(2, 6, 100), cv=None)` — the `cv=None` default uses efficient LOOCV via the hat matrix, which is ideal for small N. Apply the one-standard-error rule: pick the largest α within one standard error of the minimum CV error.

### 2b. Constrain GBT Aggressively or Drop It Entirely

With N=60, tree-based methods are generally dominated by regularized linear models. Research consistently shows linear regression outperforms trees below ~100 observations. If keeping GBT, use `max_depth=2, n_estimators=50, learning_rate=0.01, min_samples_leaf=10, subsample=0.7` and reduce its ensemble weight to 20–30%.

### 2c. Winsorize Both Targets and Predictions at the 5th/95th Percentiles

Jose and Winkler (2008) found winsorized forecasts improve predictability by **15–45%**. Leone et al. found 53% of financial studies winsorize at 1%/99%; for small-N regression, the 5%/95% level is more appropriate.

---

## 3. Feature Engineering: Less Is Dramatically More at N=60

The feature-to-observation ratio is the binding constraint. With 12–13 features and 60 training observations, the ratio is **4.6:1** — well below the traditional "one-in-ten" rule requiring 10 observations per parameter, and even below the relaxed 5:1 minimum for heavily regularized models. Reducing to **5–7 features** brings the ratio to 8.5–12:1, a meaningful improvement in estimation stability.

### PCA for Dimensionality Reduction

PCA offers the most principled dimensionality reduction. Fit PCA only on each training window's standardized features, extracting 3–5 components explaining 80–90% of variance. Stock and Watson (2002) showed that 3–5 factors from large macro panels capture most predictive information. For the Ridge component specifically, PCA→Ridge eliminates multicollinearity and improves matrix conditioning. For GBT, pre-selection of 5–8 features via univariate correlation screening (within training window only) is simpler and equally effective.

### Handling Extreme Missingness

For the extreme missingness problem (7/316 observations fully populated across 72 features), the current approach of using only 12–13 well-populated features is correct. Expanding beyond this should be conservative: forward-fill quarterly fundamental data, then use `sklearn.impute.IterativeImputer` fitted exclusively on each training window to recover 2–3 additional high-value features. Never fit the imputer on future data. Adding binary missingness indicators via `MissingIndicator` can capture informative patterns in data availability.

### Feature Transformations

Feature transformations should follow a hierarchy: log-transform skewed features first (premiums, volumes), then apply rolling-window z-scoring for Ridge (fit StandardScaler on training window only), and use rank-normalization or raw features for GBT (tree models are invariant to monotone transforms).

### Domain-Specific Insurance Features

Features with theoretical predictive power that may be underutilized or missing:

- **PGR monthly operating results** (combined ratio, loss ratio, net premiums written, policies in force) — Progressive uniquely reports monthly, making this the highest-value proprietary data source
- **Auto insurance PPI** (FRED: PCU9241269241261) — direct pricing proxy for PGR's core product
- **CPI motor vehicle parts and repair** (FRED: CUSR0000SETC, CUSR0000SETD) — loss cost inflation drivers
- **CPI medical care services** (FRED: CPIMEDSL) — bodily injury claims cost driver
- **Vehicle miles traveled** (FRED: TRFVOLUSM227NFWA) — accident frequency proxy
- **10-year Treasury rate** (FRED: GS10) — investment income driver for the float portfolio

---

## 4. Model Architecture: Bayesian Methods and Multi-Task Learning for Very Small N

### BayesianRidge as Primary Model

`sklearn.linear_model.BayesianRidge` is the single best model replacement for this problem. It uses empirical Bayes to automatically tune regularization via marginal likelihood maximization — no cross-validation needed, which eliminates a major source of instability at N=60. Priors centered at zero encode the efficient-market null hypothesis that most predictability is spurious. The posterior naturally implements James-Stein shrinkage, and prediction uncertainty estimates enable confidence-weighted position sizing. Implementation requires changing one import and one line of code.

### ARDRegression for Feature Selection

`sklearn.linear_model.ARDRegression` (Automatic Relevance Determination) extends BayesianRidge by learning individual precision per feature, performing automatic feature selection — particularly valuable when starting with 12–13 features and wanting to identify which 5–7 matter.

### Multi-Task Learning Across 8 Benchmarks

With 8 PGR-vs-ETF prediction tasks sharing the same predictive features, `sklearn.linear_model.MultiTaskElasticNetCV` enforces joint feature selection via the L2,1 norm penalty, effectively multiplying the sample size for shared parameters from 60 to ~480. This is appropriate because PGR-specific factors (combined ratio, premium growth) affect relative returns against all benchmarks similarly. The alternative — training 8 independent models — wastes the shared structure.

### Gaussian Process Regression

`sklearn.gaussian_process.GaussianProcessRegressor` is a strong candidate for the nonlinear component, replacing GBT. GPR's O(N³) computation is trivial at N=60, it provides built-in uncertainty quantification via the posterior predictive distribution, and its marginal likelihood automatically controls complexity. Use a Matérn 5/2 kernel with ARD (per-feature length scales) plus WhiteKernel for observation noise. The key advantage over GBT: GPR predictions naturally revert to the prior mean far from training data — a desirable property that implements implicit shrinkage toward zero.

### Regime-Conditional Models

Regime-conditional models using observable insurance metrics (combined ratio > 100 as hard-market threshold) are theoretically appealing but practically dangerous at N=60. Splitting into two regimes yields ~30 observations each — far too few for reliable estimation with 12 features. If attempted, limit to 2–3 features per regime with strong regularization, or use the regime indicator as a single additional feature in the unified model rather than fitting separate models.

---

## 5. Walk-Forward Design: Expanding Windows and the CPCV Diagnosis

### Switch from Rolling to Expanding Windows

The 60-month rolling window discards data that expanding windows retain. Later folds in an expanding-window scheme train on 100–150+ observations, dramatically improving the feature-to-observation ratio. The cost — potential non-stationarity from including older data — is manageable for macro predictors and mitigatable via sample weighting (give recent observations higher weight in Ridge via `sample_weight`). Feng, Zhang, and Wang (2024) proposed a "Momentum of Predictability" strategy that dynamically switches between rolling and expanding based on recent performance, outperforming either fixed approach.

### CPCV Failure Diagnosis

The CPCV failure (0/7 positive paths) is both diagnostic and amplified by the small sample. CPCV with N=6 groups, k=2 test groups, and 8-month purge/embargo creates training sets of roughly 70–90 effective observations after boundary purging — better than the rolling 60 but still thin. The 0/7 result likely reflects the combination of a genuinely weak signal (true R² perhaps 1–3%) and CPCV's conservative multi-path testing overwhelming it. A simpler purged K-fold (N=5, k=1 — standard K-fold with purging) would give larger training sets per fold (~130 observations after purging) and may detect signal that CPCV cannot.

### Embargo Gap Adequacy

The 8-month embargo (6-month horizon + 2-month purge) is adequate. Lopez de Prado recommends embargo h ≈ 0.01T ≈ 1.8 months for 180 observations, so 2 months is appropriate. The 6-month purge correctly handles overlapping forward returns. Verify implementation: any training observation whose 6-month forward return period overlaps with any test observation's forward return period must be excluded.

### Systematic Window Testing

Test window lengths systematically using `TimeSeriesSplit(n_splits=5, test_size=6, gap=8)` with expanding (no `max_train_size`) and compare against rolling windows of 48, 60, 72, 84, and 96 months. The optimal window depends on the bias-variance tradeoff specific to your features and market regime — Pesaran and Timmermann (2007) showed this varies over time and cannot be determined analytically.

---

## 6. Target Reformulation and Alternative Data Strategies

### Binary Classification

Binary classification deserves serious consideration. When OOS R² is negative but hit rate is 69%, the model is solving a classification problem while being evaluated on a regression metric. Wolff (2024, *Journal of Forecasting*) found that regularized logistic regression for stock outperformance classification generates substantial and significant outperformance. Classification eliminates the magnitude calibration problem entirely — a correct directional call counts equally whether the magnitude was 1% or 10%. For the RSU vesting decision, the actionable question is "will PGR outperform benchmark X over 6 months?" — a binary classification problem. Switching to logistic regression preserves the existing signal while aligning the loss function with the actual decision.

### Composite Benchmark

Predicting against a single composite benchmark (equal-weighted or volatility-weighted average of 8 ETFs) reduces the problem from 8 targets to 1, increasing effective sample size per target and eliminating multiple-testing burden across benchmarks.

### Rank-Based Targets

Rank-based targets (ordinal position of PGR return among benchmarks, scaled 1–9) naturally bound the target, eliminate extreme values, and focus the model on relative ordering — directly aligned with the existing positive IC. Implementable via ordinal regression or simply by rank-transforming the target before fitting Ridge.

### Peer Company Pooling

Peer company pooling is the most promising data augmentation strategy. Pool training data from ALL, TRV, CB, and HIG (P&C insurance peers), adding company indicators as features. This quintuples the effective sample size for shared parameters. A two-stage approach — train a sector model on peers, then use sector predictions as a feature for PGR — avoids architectural changes to the existing ensemble while incorporating sector-level information as a Bayesian prior.

### Synthetic Data

Synthetic data generation (GANs, TimeGAN) is not recommended — with only 180 observations, the generative model cannot learn a reliable distribution, risking injection of artificial patterns. Block bootstrap is primarily useful for inference (confidence intervals) rather than data augmentation, though bootstrap aggregating within model training can reduce prediction variance.

---

## 7. Ranked Implementation Roadmap

The interventions below are ordered by expected impact per unit of implementation effort. **The realistic ceiling for single-stock 6-month relative return prediction with ~180 observations is OOS R² of 0.5–3%.** Gu, Kelly, and Xiu (2020) achieved 0.09–0.39% monthly R² pooling 30,000 stocks; Campbell and Thompson (2008) found 0.5% monthly R² economically significant. Your +2% target for 6-month relative returns is ambitious but achievable given the demonstrated IC=0.19 — the entire gap is calibration, not signal.

| Rank | Intervention | Complexity | Expected R² Impact | Key Risk |
|------|-------------|-----------|-------------------|----------|
| 1 | Aggressive prediction shrinkage (α=0.10–0.15) | S | +10 to +16pp | Over-shrinkage kills weak signals in specific periods |
| 2 | Increase Ridge α by 10–100× with LOOCV | S | +3 to +8pp | Compounds with #1; diminishing returns |
| 3 | Reduce features to 5–7 (PCA or pre-committed selection) | M | +2 to +5pp | May drop features with real but inconsistent signal |
| 4 | Switch to expanding window | S | +1 to +3pp | Old data may dilute if regime has shifted |
| 5 | Drop GBT or constrain to max_depth=2, weight=20% | S | +1 to +3pp | Loses nonlinear signal capture (likely minimal at N=60) |

### Experimental Sequence

Execute interventions 1–2 as **Phase 1** (one inner-CV experiment). Add #3–5 as **Phase 2** (one experiment). Reserve the most recent 24 months as a never-touched holdout. Run the final specification on this holdout exactly once. This preserves statistical validity despite prior experimentation.

### Multiple Testing Discipline

Multiple testing discipline is critical. Harvey, Liu, and Zhu (2016) showed that with hundreds of tested specifications, conventional t>2.0 is insufficient — t>3.0 is needed. Prior experiments (feature ablation, model selection, benchmark reduction, Platt calibration, conformal prediction, Black-Litterman, shadow baseline) each count as a test. Pre-commit to a single specification before the holdout evaluation.

### Should the +2% R² Gate Be Relaxed?

There is a strong case for replacing R² as the primary gate with IC + hit rate + economic value. The Fundamental Law of Active Management uses IC exclusively. For a personal RSU hold/sell decision, directional accuracy of 69% already provides substantial economic value. Consider computing the **certainty equivalent return gain** — the annualized return a risk-averse investor would sacrifice to have access to the model versus a naive strategy. If positive, the model adds value regardless of R².

---

## Conclusion

The path from −13% to +2% OOS R² does not require new features, new models, or new data. It requires **making the existing signal quieter**. The model already identifies PGR's relative performance direction correctly 69% of the time — a genuinely strong result that most quantitative finance practitioners would deploy without hesitation. The negative R² reflects overconfident predictions, not absent signal. Aggressive shrinkage (multiply all predictions by 0.10–0.15), heavier Ridge regularization (α in the thousands, not tens), ruthless feature reduction (5–7 features, not 12–13), and expanding windows collectively address the variance-dominated error structure. The theoretical ceiling of IC²=3.6% makes the +2% target achievable without any structural model changes.

The deeper strategic insight is that for a binary RSU vesting decision, reframing the problem as classification — where the 69% hit rate already exceeds any reasonable threshold — may be more valuable than continuing to optimize a regression metric that the financial econometrics literature considers secondary to rank-based measures.
