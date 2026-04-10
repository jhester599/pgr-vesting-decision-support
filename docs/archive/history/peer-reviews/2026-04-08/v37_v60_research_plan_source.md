# v37–v60 Research Plan — Complete OOS R² Improvement Experiments

## Purpose

This document contains (1) a structured research plan implementing **every recommendation** from two independent deep-research reports on improving OOS R², and (2) a ready-to-paste Claude Code prompt that creates each experiment as a standalone research script.

The core diagnosis: **the model has real directional signal (IC=0.19, hit rate=69%) but overconfident prediction magnitudes.** The theoretical ceiling with IC=0.19 is IC²=3.6%, so the entire gap from −13.26% is prediction variance, not absent signal.

---

## Version Context

- **Latest research version:** v36
- **These experiments:** v37 through v60 (24 standalone scripts, one per version)
- **Production model:** v11.0 (lean Ridge+GBT ensemble on 8 benchmarks)
- **If promoted to production:** The successful test version number carries forward

---

## Experiment Inventory — All Phases

### Phase 1 — Calibration (v37–v43)
Quick, low-risk interventions targeting prediction variance within existing Ridge+GBT architecture.

| Version | Experiment | Source | Expected Impact |
|---------|-----------|--------|----------------|
| v37 | Baseline measurement | — | Establishes comparison point |
| v38 | Prediction shrinkage | Both reports | HIGH — primary fix |
| v39 | Ridge alpha extension | Both reports | HIGH — reduces coefficient magnitude |
| v40 | GBT constraint/removal | Both reports | MEDIUM — reduces ensemble variance |
| v41 | Target winsorization | Report 1 | MEDIUM — removes outlier influence |
| v42 | Expanding window | Both reports | MEDIUM — increases training N |
| v43 | Feature reduction | Both reports | MEDIUM — improves obs/feature ratio |

### Phase 2 — Feature & Model Architecture (v44–v46)
Medium-complexity changes to feature construction or model type.

| Version | Experiment | Source | Expected Impact |
|---------|-----------|--------|----------------|
| v44 | Blockwise PCA | Report 1 | MEDIUM — eliminates multicollinearity |
| v45 | BayesianRidge replacement | Both reports | MEDIUM — automatic regularization |
| v46 | Binary classification | Both reports | MEDIUM — aligns loss with decision |

### Phase 3 — Target & Structure Reforms (v47–v49)
Fundamental changes to prediction target or cross-benchmark structure.

| Version | Experiment | Source | Expected Impact |
|---------|-----------|--------|----------------|
| v47 | Composite benchmark target | Both reports | HIGH — reduces label noise |
| v48 | Panel pooling across benchmarks | Both reports | HIGH — multiplies effective N |
| v49 | Regime indicator features | Both reports | LOW-MEDIUM — adds market context |

### Phase 4 — Additional Model & Data (v50–v55)
Alternative models, data augmentation, and calibration refinements.

| Version | Experiment | Source | Expected Impact |
|---------|-----------|--------|----------------|
| v50 | Prediction winsorization | Report 1 | MEDIUM — clips extreme predictions |
| v51 | Peer company pooling | Report 1 | MEDIUM — quintuples shared N |
| v52 | Shorter WFO test windows | Report 2 | LOW-MEDIUM — more calibration data |
| v53 | ARDRegression | Report 1 | MEDIUM — automatic feature selection |
| v54 | Gaussian Process Regression | Report 1 | MEDIUM — implicit shrinkage to prior |
| v55 | Rank-based targets | Report 1 | LOW-MEDIUM — bounds target range |

### Phase 5 — Data & Diagnostic Enhancements (v56–v60)
Feature engineering, alternative horizons, and evaluation framework improvements.

| Version | Experiment | Source | Expected Impact |
|---------|-----------|--------|----------------|
| v56 | 12-month horizon branch | Both reports | LOW — more embargo cost |
| v57 | Feature transformations + lags | Report 1 | LOW-MEDIUM — reduces skew/staleness |
| v58 | Domain-specific FRED features | Report 1 | LOW-MEDIUM — new signal sources |
| v59 | Imputation strategies | Report 1 | LOW — limited value with current features |
| v60 | Clark-West test + eval framework | Report 1 | DIAGNOSTIC — better metrics |

---

## Recommended Run Order

**Week 1:** v37 → v38 → v39 → v40 (baseline + highest-leverage calibration)
**Week 2:** v41 → v42 → v43 → v50 (remaining calibration)
**Week 3:** v47 → v48 → v49 (target/structure reforms — conditional on Phase 1)
**Week 4:** v44 → v45 → v46 (feature/model architecture — conditional on Phase 1)
**Week 5:** v51 → v52 → v53 → v54 → v55 (additional models — conditional on Phases 1-3)
**Week 6:** v56 → v57 → v58 → v59 → v60 (lower priority — conditional on all above)

After each phase, evaluate results and skip subsequent experiments that are superseded by earlier wins.

---

## Key Codebase References

### File Locations (relative to repo root)
```
config/features.py              → PRIMARY_FORECAST_UNIVERSE, MODEL_FEATURE_OVERRIDES
config/model.py                 → WFO_TRAIN_WINDOW_MONTHS, ENSEMBLE_MODELS, DIAG_* thresholds
src/models/regularized_models.py → build_ridge_pipeline(), build_gbt_pipeline(), build_bayesian_ridge_pipeline()
src/models/wfo_engine.py         → run_wfo(), WFOResult, predict_current()
src/models/multi_benchmark_wfo.py → run_ensemble_benchmarks(), EnsembleWFOResult
src/models/evaluation.py         → evaluate_wfo_model(), summarize_predictions(), reconstruct_ensemble_oos_predictions()
src/processing/feature_engineering.py → build_feature_matrix_from_db(), get_X_y_relative()
src/processing/multi_total_return.py  → load_relative_return_matrix()
src/reporting/backtest_report.py      → compute_oos_r_squared(), compute_newey_west_ic()
data/pgr_financials.db           → SQLite database with all historical data
```

### Current Configurations
```python
# Ridge alpha grid (regularized_models.py line 143)
alphas = np.logspace(-4, 4, 50)  # Range: 0.0001 to 10,000

# GBT (regularized_models.py line 235)
GradientBoostingRegressor(max_depth=2, n_estimators=50, learning_rate=0.1, subsample=0.8, random_state=42)

# WFO (model.py)
WFO_TRAIN_WINDOW_MONTHS = 60; WFO_TEST_WINDOW_MONTHS = 6; WFO_PURGE_BUFFER_6M = 2  # gap = 8

# Ridge features (12)
["mom_12m", "vol_63d", "yield_slope", "real_yield_change_6m", "real_rate_10y",
 "credit_spread_hy", "nfci", "vix", "combined_ratio_ttm",
 "investment_income_growth_yoy", "book_value_per_share_growth_yoy", "npw_growth_yoy"]

# GBT features (13)
["mom_3m", "mom_6m", "mom_12m", "vol_63d", "yield_slope", "yield_curvature",
 "vwo_vxus_spread_6m", "credit_spread_hy", "nfci", "vix",
 "rate_adequacy_gap_yoy", "pif_growth_yoy", "investment_book_yield"]

# 8 benchmarks
["VOO", "VXUS", "VWO", "VMBS", "BND", "GLD", "DBC", "VDE"]

# Peer tickers (already fetched weekly)
["ALL", "TRV", "CB", "HIG"]

# Diagnostic thresholds
DIAG_MIN_OOS_R2 = 0.02; DIAG_MIN_IC = 0.07; DIAG_MIN_HIT_RATE = 0.55
```

---

## Experiment Details — All Phases

### Phase 1: Calibration (v37–v43)

#### v37 — Baseline Measurement
Measure current v11.0 lean pipeline. Per benchmark: N OOS, OOS R², IC, NW IC/p-value, hit rate, MAE, std(y_hat), std(y_true), σ_pred/σ_true ratio. Aggregate: pool all OOS predictions. Output: `v37_baseline_results.csv`.

#### v38 — Prediction Shrinkage
Post-hoc: y_hat_shrunk = α × y_hat for α ∈ [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]. No retraining. IC and hit rate must stay invariant (sanity check). Find optimal α maximizing aggregate OOS R². Output: `v38_shrinkage_results.csv`.

#### v39 — Ridge Alpha Extension
Custom WFO loop with three alpha grids: current logspace(-4,4), extended logspace(0,6), aggressive logspace(2,6). Ridge only, all 8 benchmarks. Use RidgeCV(cv=None) for LOOCV. Output: `v39_ridge_alpha_results.csv`.

#### v40 — GBT Constraint/Removal
Variant A: Ridge only (no GBT). Variant B: Constrained GBT (lr=0.01, min_samples_leaf=10, subsample=0.7). Variant C: Post-hoc reweight 80% Ridge / 20% GBT. Output: `v40_gbt_results.csv`.

#### v41 — Target Winsorization
Custom WFO loop winsorizing y_train at 5th/95th and 10th/90th within each fold. Evaluate against original y_test. Both Ridge and GBT. Output: `v41_winsorize_results.csv`.

#### v42 — Expanding Window
Variant A: Pure expanding (no max_train_size). Variant B: Expanding with exponential decay weighting. Variant C: Expanding capped at 120 months. Output: `v42_expanding_results.csv`.

#### v43 — Feature Reduction
Variant A: Ridge→7 features [yield_slope, npw_growth_yoy, investment_income_growth_yoy, real_yield_change_6m, combined_ratio_ttm, mom_12m, credit_spread_hy]. Variant B: GBT→7 [mom_12m, vol_63d, yield_slope, credit_spread_hy, vix, pif_growth_yoy, investment_book_yield]. Variant C: Shared 7 [mom_12m, vol_63d, yield_slope, credit_spread_hy, vix, combined_ratio_ttm, npw_growth_yoy]. Use run_wfo(..., feature_columns=list). Output: `v43_feature_results.csv`.

### Phase 2: Feature & Model Architecture (v44–v46)

#### v44 — Blockwise PCA
Macro block (6 features: yield_slope, real_rate_10y, credit_spread_hy, nfci, vix, real_yield_change_6m) → PCA. Insurance block (4 features: combined_ratio_ttm, npw_growth_yoy, investment_income_growth_yoy, pif_growth_yoy) → PCA. Combine with raw mom_12m + vol_63d. PCA fit within each fold. Variant A: 2 components/block (6 total). Variant B: 1 component/block (4 total). Ridge with extended alpha grid. Output: `v44_pca_results.csv`.

#### v45 — BayesianRidge Replacement
Variant A: Default BayesianRidge via run_wfo(model_type="bayesian_ridge"). Variant B: Tight prior (alpha_1=1e-5, alpha_2=1e-5, lambda_1=1e-4, lambda_2=1e-4). Variant C: BayesianRidge+GBT ensemble. Output: `v45_bayesian_ridge_results.csv`.

#### v46 — Binary Classification
Target: y_binary = (y > 0).astype(int). LogisticRegressionCV(Cs=logspace(-4,4,20), cv=3, penalty="l2"). Metrics: accuracy, balanced accuracy, Brier score, log-loss. Output: `v46_classification_results.csv`.

### Phase 3: Target & Structure Reforms (v47–v49)

#### v47 — Composite Benchmark Target
**Goal:** Replace 8 separate per-ETF models with one model predicting PGR vs. a blended composite benchmark, reducing label noise and multiple-testing burden.

**Method:**
1. Construct a composite benchmark return as the equal-weighted average of the 8 ETF returns:
   ```python
   composite_return = rel_matrix.mean(axis=1)  # Average PGR-minus-ETF across 8 ETFs
   ```
   Also test a volatility-weighted composite: weight each ETF inversely by its return std.
2. Run Ridge+GBT ensemble on this single composite target.
3. Compare OOS R², IC, hit rate vs. the current per-benchmark aggregate.

**Why this helps:** A single composite target averages out idiosyncratic benchmark noise. PGR-specific signals (combined ratio, NPW growth) should predict relative performance vs. a diversified composite more stably than vs. any single ETF.

**Variant A:** Equal-weighted composite.
**Variant B:** Inverse-volatility-weighted composite.
**Variant C:** Equity-only composite (VOO, VXUS, VWO, VDE — excludes bonds/commodities/gold for a cleaner equity relative return).

Output: `v47_composite_benchmark_results.csv`.

---

#### v48 — Panel Pooling Across Benchmarks
**Goal:** Stack all 8 benchmark-month observations into one regression with benchmark fixed effects, sharing coefficients across benchmarks. This multiplies effective N from ~60 to ~480 for shared parameters.

**Method:**
1. Stack the data: for each month t and benchmark b, create a row with features X_t and target y_{t,b} (PGR minus ETF_b relative return).
2. Add 7 benchmark dummy variables (one-hot, dropping one for intercept).
3. Fit Ridge on the stacked panel within each WFO fold.

```python
# Stack data for panel regression
panel_rows = []
for etf in benchmarks:
    X_aligned, y_aligned = get_X_y_relative(df, rel_series_dict[etf], drop_na_target=True)
    for i in range(len(X_aligned)):
        row = X_aligned.iloc[i].to_dict()
        row["benchmark"] = etf
        row["target"] = y_aligned.iloc[i]
        panel_rows.append(row)
panel_df = pd.DataFrame(panel_rows)
# Add benchmark dummies
panel_dummies = pd.get_dummies(panel_df["benchmark"], drop_first=True)
X_panel = pd.concat([panel_df[feature_cols], panel_dummies], axis=1)
y_panel = panel_df["target"]
```

4. **Critical WFO constraint:** The temporal split must respect time — all benchmarks for month t are either ALL in train or ALL in test. Split by month, not by row.

**Variant A:** Full panel with benchmark fixed effects.
**Variant B:** Panel with shared coefficients only (no benchmark dummies) — tests whether one model works for all benchmarks.

Also test `sklearn.linear_model.MultiTaskElasticNetCV` which enforces joint feature selection via the L2,1 norm penalty:
```python
from sklearn.linear_model import MultiTaskElasticNetCV
# Reshape y to (n_months, n_benchmarks) matrix
# MultiTaskElasticNet fits all 8 targets simultaneously with shared sparsity
```

Output: `v48_panel_pooling_results.csv`.

---

#### v49 — Regime Indicator Features
**Goal:** Add 1–2 regime indicators as features rather than splitting into separate regime models (insufficient N for splits).

**Method:**
1. Add the following binary/continuous regime indicators to the existing feature set:
   - `hard_market` = 1 if combined_ratio_ttm > 100, else 0
   - `high_vol` = 1 if vix > 20 (long-run median), else 0
   - `inverted_curve` = 1 if yield_slope < 0, else 0
2. Run Ridge+GBT ensemble with the augmented feature set (original 12/13 + 2–3 regime indicators).
3. Compare against baseline.

**Variant A:** Add hard_market + high_vol (2 indicators).
**Variant B:** Add all 3 indicators.
**Variant C:** Add interactions: hard_market × yield_slope, high_vol × credit_spread_hy (only 2 economically motivated interactions — more would overfit at N=60).

Output: `v49_regime_results.csv`.

---

### Phase 4: Additional Model & Data (v50–v55)

#### v50 — Prediction Winsorization
**Goal:** Clip OOS predictions at training-window percentiles, complementary to shrinkage (v38).

**Method:** Within each WFO fold, after generating y_hat on the test set:
```python
# Compute percentiles from TRAINING predictions (fit model on train, predict train)
y_hat_train = pipeline.predict(X_train)
p5 = np.percentile(y_hat_train, 5)
p95 = np.percentile(y_hat_train, 95)
y_hat_test_clipped = np.clip(y_hat_test, p5, p95)
```

**Variant A:** 5th/95th percentile clip.
**Variant B:** 10th/90th percentile clip.
**Variant C:** Combined with shrinkage — clip first, then apply optimal α from v38.

Output: `v50_pred_winsorize_results.csv`.

---

#### v51 — Peer Company Pooling
**Goal:** Use P&C insurance peer data (ALL, TRV, CB, HIG) to increase effective sample size.

**Method — two-stage approach:**
1. **Stage 1:** Train a "sector model" predicting peer-composite vs. each ETF benchmark using the same macro features (no PGR-specific features). This uses peer return data already in the DB.
2. **Stage 2:** Use the sector model's OOS prediction as an additional feature for the PGR model — effectively a Bayesian prior from sector-level information.

```python
# Stage 1: Build peer composite return
peer_tickers = ["ALL", "TRV", "CB", "HIG"]
# Load peer prices from DB, compute 6M returns, average across peers
# Train Ridge on macro features predicting peer_composite minus ETF
# Generate OOS predictions = "sector_signal"

# Stage 2: Add sector_signal as feature #13 in the PGR Ridge model
ridge_features_augmented = ridge_features + ["sector_signal"]
```

**Alternative simpler approach:** Just add `pgr_vs_peers_6m` (already computed in feature_engineering.py) as a feature if not already in the model.

Output: `v51_peer_pooling_results.csv`.

---

#### v52 — Shorter WFO Test Windows
**Goal:** Reduce test_size from 6 months to 1 month, creating more OOS prediction points for post-model calibration.

**Method:**
```python
tscv = TimeSeriesSplit(
    n_splits=n_splits,
    max_train_size=60,
    test_size=1,   # Changed from 6 to 1
    gap=total_gap,  # Still 8 months
)
```

**Variant A:** test_size=1 with rolling 60M window.
**Variant B:** test_size=3 with rolling 60M window (intermediate).
**Variant C:** test_size=1 with expanding window (combines with v42).

**Note:** More test points means more OOS predictions per fold, which improves the reliability of calibration metrics. The model itself is unchanged.

Output: `v52_test_window_results.csv`.

---

#### v53 — ARDRegression (Automatic Relevance Determination)
**Goal:** Test ARD as an alternative to Ridge/BayesianRidge that performs automatic per-feature selection via individual precision priors.

**Method:**
```python
from sklearn.linear_model import ARDRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ARDRegression(
        alpha_1=1e-6, alpha_2=1e-6,
        lambda_1=1e-6, lambda_2=1e-6,
        threshold_lambda=10000,  # Features with precision > this are pruned
        fit_intercept=True,
    ))
])
```

Run on all 8 benchmarks with Ridge feature set. Compare feature selection behavior across folds — does ARD consistently zero out the same features?

**Variant A:** Default ARD.
**Variant B:** ARD + GBT ensemble.
**Variant C:** ARD with extended feature set (all 12 Ridge features + 3–4 extras) — let ARD select the best subset automatically.

Output: `v53_ard_results.csv`.

---

#### v54 — Gaussian Process Regression
**Goal:** Replace GBT as the nonlinear ensemble component with GPR, which naturally reverts to prior mean far from training data (implicit shrinkage).

**Method:**
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,
    normalize_y=True,
    random_state=42,
)
pipe = Pipeline([("scaler", StandardScaler()), ("model", gpr)])
```

**Note:** GPR is O(N³) — at N=60, this is trivial (~0.001 seconds per fit). GPR also provides posterior predictive variance, which can replace BayesianRidge for confidence estimation.

**Variant A:** GPR with Matérn 5/2 kernel (smooth nonlinearity).
**Variant B:** GPR with RBF kernel (smoother, may be too flexible).
**Variant C:** Ridge + GPR ensemble (replacing Ridge + GBT).

Use a reduced feature set (7 features from v43) to avoid kernel curse of dimensionality.

Output: `v54_gpr_results.csv`.

---

#### v55 — Rank-Based Targets
**Goal:** Replace raw relative return targets with rank-based targets that are naturally bounded, reducing extreme value influence.

**Method:**
1. For each month, rank PGR's return among the 8 benchmarks (rank 1 = worst, rank 9 = best including PGR itself). Use: `rank = (y > 0).astype(int)` is too simple. Instead:
   ```python
   # For each month, compute PGR 6M return and all 8 ETF 6M returns
   # Rank PGR among the 9 assets (PGR + 8 ETFs)
   # Normalize rank to [0, 1]: rank_normalized = (rank - 1) / 8
   ```
2. Alternatively, just rank-transform the existing per-benchmark relative returns within a rolling window:
   ```python
   from scipy.stats import rankdata
   # Within each training fold, rank-transform y_train
   y_train_ranked = rankdata(y_train) / len(y_train)  # Uniform [0,1]
   ```
3. Fit Ridge on rank-transformed targets.
4. Evaluate OOS using both rank-based metrics (Spearman IC) and the original R² (by converting ranked predictions back to return space via the training quantile mapping).

**Variant A:** Rank-transform targets within each fold.
**Variant B:** Percentile-transform targets (similar to rank but continuous).

Output: `v55_rank_target_results.csv`.

---

### Phase 5: Data & Diagnostic Enhancements (v56–v60)

#### v56 — 12-Month Horizon Branch
**Goal:** Test whether the 12-month prediction horizon has better signal-to-noise despite larger embargo cost.

**Method:**
1. Load 12M relative return data: `load_relative_return_matrix(conn, etf, 12)`
2. Run ensemble with `target_horizon_months=12`, which automatically sets gap = 12 + 3 = 15 months.
3. Compare 12M OOS R², IC, hit rate vs. the 6M baseline.

**Key tradeoff:** 12M horizon halves the number of non-overlapping observations and requires 15-month gap vs. 8-month, reducing training data. But the signal may be stronger because insurance underwriting cycles operate on 12–24 month rhythms.

**Variant A:** 12M horizon with rolling 60M window.
**Variant B:** 12M horizon with expanding window.

Output: `v56_12m_horizon_results.csv`.

---

#### v57 — Feature Transformations and Lags
**Goal:** Test log transforms for skewed features, rank-normalization, and 1–3 month lags for stale fundamental data.

**Method — three independent sub-experiments:**

**v57-A: Log transforms.** Apply log(1+x) to skewed features within each fold:
```python
skewed_features = ["npw_growth_yoy", "investment_income_growth_yoy",
                   "pif_growth_yoy", "book_value_per_share_growth_yoy"]
X_train[:, skewed_idx] = np.sign(X_train[:, skewed_idx]) * np.log1p(np.abs(X_train[:, skewed_idx]))
```

**v57-B: Rank normalization for GBT.** Within each fold, rank-transform all features to uniform [0,1]:
```python
from scipy.stats import rankdata
for col in range(X_train.shape[1]):
    X_train[:, col] = rankdata(X_train[:, col]) / len(X_train)
    # Apply same quantile mapping to X_test using training distribution
```

**v57-C: Lagged features.** Add 1-month and 2-month lags of fundamental features to handle stale EDGAR data:
```python
lagged_features = ["combined_ratio_ttm", "npw_growth_yoy", "investment_income_growth_yoy"]
for feat in lagged_features:
    df[f"{feat}_lag1"] = df[feat].shift(1)
    df[f"{feat}_lag2"] = df[feat].shift(2)
```

Output: `v57_transformations_results.csv`.

---

#### v58 — Domain-Specific FRED Features
**Goal:** Test additional FRED series with theoretical predictive power for insurance stock performance.

**New features to construct (data may already be in DB from v19 research series):**

1. **auto_insurance_ppi_mom3m** — 3-month momentum of PCU5241265241261 (auto insurance PPI). Already partially available as `rate_adequacy_gap_yoy` but testing raw momentum form.

2. **medical_cpi_mom3m** — 3-month momentum of CUSR0000SAM2 (medical care CPI). Proxy for bodily injury claims cost acceleration.

3. **motor_parts_cpi_mom3m** — 3-month momentum of CUSR0000SETA02 (used car CPI). Proxy for total-loss severity trends.

4. **mortgage_rate_delta_3m** — 3-month change in MORTGAGE30US. Affects housing turnover → auto insurance demand.

5. **term_premium_10y** — THREEFYTP10 (ACM 10-year term premium). Captures risk compensation in long rates beyond level.

**Method:** Check which series are already in `fred_macro_monthly` table. For those present, compute the 3-month momentum or level and add as features. For those absent, note the gap and skip.

Test by adding 2–3 of these to the Ridge feature set and running WFO.

Output: `v58_fred_features_results.csv`.

---

#### v59 — Imputation Strategies
**Goal:** Test whether principled imputation of partially-missing features enables using more of the 72-feature inventory.

**Method:**
1. Expand from 12 to 18 features by adding 6 partially-filled features:
   ```python
   additional_features = [
       "pe_ratio",             # 67% fill
       "pb_ratio",             # 67% fill
       "roe",                  # 67% fill
       "buyback_yield",        # 69% fill
       "term_premium_10y",     # 69% fill
       "breakeven_inflation_10y",  # 69% fill
   ]
   ```
2. Test three imputation strategies within each fold:

**v59-A: Forward-fill then training-median imputation.** Apply ffill() on the raw feature DataFrame before entering the WFO loop (no lookahead — ffill only uses past values). For remaining NaN, use training-fold median.

**v59-B: IterativeImputer.** Fit sklearn's `IterativeImputer` on the training fold only:
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=42)
X_train_imputed = imputer.fit_transform(X_train)  # Fit on train only
X_test_imputed = imputer.transform(X_test)          # Apply to test
```

**v59-C: MissingIndicator.** Add binary columns indicating whether each feature was originally missing:
```python
from sklearn.impute import MissingIndicator
indicator = MissingIndicator().fit(X_train)
X_train_indicators = indicator.transform(X_train)
X_train_augmented = np.column_stack([X_train_imputed, X_train_indicators])
```

Output: `v59_imputation_results.csv`.

---

#### v60 — Clark-West Test + Evaluation Framework
**Goal:** Implement the Clark-West (2006) MSFE-adjusted test statistic, which can detect genuine predictability even when OOS R² is negative. Also add certainty-equivalent return gain as an economic evaluation metric.

**Method — this is a diagnostic script, not a model experiment:**

1. **Clark-West test:** For each benchmark, compute:
   ```python
   # f_hat = model prediction, y_bar = expanding historical mean
   # e1 = y - y_bar (naive forecast error)
   # e2 = y - f_hat (model forecast error)
   # CW_t = e1² - (e2² - (f_hat - y_bar)²)
   # Test: mean(CW_t) > 0 via t-test with Newey-West HAC
   ```
   If the CW test is significant (p < 0.05) even with negative R², the model has genuine predictive power that R² fails to detect.

2. **Certainty-equivalent return gain:** For a risk-averse investor (CRRA utility with γ=2), compute:
   ```python
   # CE gain = annualized return difference between model-informed strategy
   # and naive 50/50 strategy, adjusting for risk
   # Positive CE gain = model adds economic value regardless of R²
   ```

3. **MSE decomposition:** Decompose OOS MSE into bias² + variance + covariance term to identify whether the problem is systematic bias, prediction volatility, or both:
   ```python
   bias_sq = (y_hat.mean() - y_true.mean()) ** 2
   var_component = y_hat.var()
   mse = np.mean((y_hat - y_true) ** 2)
   # If var_component >> bias_sq, variance dominance confirmed
   ```

Run on the v37 baseline predictions and report all three diagnostics per benchmark.

Output: `v60_diagnostics_results.csv`.

---

## Success Criteria

| Metric | Baseline (v37) | Pass Threshold | Stretch Goal |
|--------|---------------|---------------|-------------|
| Aggregate OOS R² | TBD | ≥ 0% | ≥ +2% |
| Mean IC | TBD | No degradation > 0.02 | Improvement |
| Mean Hit Rate | TBD | No degradation > 2pp | Improvement |
| Mean MAE | TBD | No increase > 10% | Decrease |
| Clark-West p-value | TBD | < 0.05 | < 0.01 |

---

## Production Promotion Rules

1. **The successful test version number becomes the production version.**
2. **If a combination experiment (v61+) is promoted,** it gets its own version number.
3. **Gate requirements:** OOS R² ≥ 0% (minimum) or ≥ +2% (full pass), IC ≥ 0.07, hit rate ≥ 55%, reproducible across 2 runs.
4. **Pre-commit to one specification before holdout evaluation.**

---

## CLAUDE CODE PROMPT

Copy everything below this line and paste it into Claude Code as a single prompt.

---

```
# v37–v60 OOS R² Improvement Experiments — Complete Implementation

## Context

You are working on the pgr-vesting-decision-support repository. This is a personal RSU vesting decision support system for Progressive Corporation (PGR) stock. The ML pipeline predicts 6-month relative returns (PGR minus ETF) using a Walk-Forward Optimization ensemble of Ridge regression + Gradient Boosted Trees across 8 ETF benchmarks.

**The central problem:** The model has positive IC (0.19, statistically significant) and high hit rate (69%), but deeply negative OOS R² (−13.26%). Two independent research reports diagnosed this as a prediction calibration problem — the model predicts the right direction but with far too much variance in magnitude.

## Your Task

Create 24 standalone research scripts in `research/v37/` — one per version (v37 through v60). Each script must:
1. Import from the existing codebase (do NOT modify any production code)
2. Run against the real SQLite DB at `data/pgr_financials.db`
3. Output a comparison table to stdout AND save results to CSV in `research/v37/`
4. Complete in <15 minutes
5. Include a `__main__` block: `python research/v37/v37_baseline.py`

Create `research/v37/` and `research/__init__.py` if they don't exist.

## Important Codebase Details

### Database and Data Access
```python
import sqlite3
conn = sqlite3.connect("data/pgr_financials.db")

from src.processing.feature_engineering import build_feature_matrix_from_db, get_X_y_relative
from src.processing.multi_total_return import load_relative_return_matrix

df = build_feature_matrix_from_db(conn, force_refresh=True)
rel_series = load_relative_return_matrix(conn, "VOO", 6)
X, y = get_X_y_relative(df, rel_series, drop_na_target=True)
```

### Running WFO
```python
from src.models.wfo_engine import run_wfo, WFOResult
result = run_wfo(X, y, model_type="ridge", target_horizon_months=6, benchmark="VOO")
# result.y_true_all, result.y_hat_all → numpy arrays
```

### Running the Full Ensemble
```python
from src.models.multi_benchmark_wfo import run_ensemble_benchmarks
from src.models.evaluation import reconstruct_ensemble_oos_predictions, summarize_predictions
from config.features import MODEL_FEATURE_OVERRIDES

benchmarks = ["VOO", "VXUS", "VWO", "VMBS", "BND", "GLD", "DBC", "VDE"]
rel_matrix = pd.DataFrame()
for etf in benchmarks:
    s = load_relative_return_matrix(conn, etf, 6)
    if not s.empty:
        rel_matrix[etf] = s

ensemble_results = run_ensemble_benchmarks(
    df, rel_matrix, target_horizon_months=6,
    model_feature_overrides=MODEL_FEATURE_OVERRIDES,
)
ens_result = ensemble_results["VOO"]
y_hat, y_true = reconstruct_ensemble_oos_predictions(ens_result)
summary = summarize_predictions(y_hat, y_true, target_horizon_months=6)
```

### Computing Metrics
```python
from src.reporting.backtest_report import compute_oos_r_squared, compute_newey_west_ic
r2 = compute_oos_r_squared(y_hat_series, y_true_series)
nw_ic, nw_p = compute_newey_west_ic(y_hat_series, y_true_series, lags=5)
```

### Custom WFO Loop Template
For experiments that need non-standard training (custom alpha grids, target transforms, etc.):
```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

def custom_wfo(X_sub, y, pipeline_factory, max_train=60, test_size=6, gap=8):
    n = len(X_sub)
    available = n - max_train - gap
    n_splits = max(1, available // test_size)
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train, test_size=test_size, gap=gap)

    all_y_true, all_y_hat = [], []
    for train_idx, test_idx in tscv.split(X_sub):
        X_train, X_test = X_sub[train_idx].copy(), X_sub[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]

        # Impute NaN with training medians
        medians = np.nanmedian(X_train, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for c in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, c]), c] = medians[c]
            X_test[np.isnan(X_test[:, c]), c] = medians[c]

        pipe = pipeline_factory()
        pipe.fit(X_train, y_train)
        y_hat = pipe.predict(X_test)

        all_y_true.extend(y_test.tolist())
        all_y_hat.extend(y_hat.tolist())

    return np.array(all_y_true), np.array(all_y_hat)
```

### Current Configurations
- **Ridge alpha:** np.logspace(-4, 4, 50), max = 10,000
- **GBT:** max_depth=2, n_estimators=50, learning_rate=0.1, subsample=0.8
- **WFO:** 60M rolling train, 6M test, 8M gap
- **Ensemble:** Ridge+GBT, 1/MAE² weighting
- **Ridge features (12):** mom_12m, vol_63d, yield_slope, real_yield_change_6m, real_rate_10y, credit_spread_hy, nfci, vix, combined_ratio_ttm, investment_income_growth_yoy, book_value_per_share_growth_yoy, npw_growth_yoy
- **GBT features (13):** mom_3m, mom_6m, mom_12m, vol_63d, yield_slope, yield_curvature, vwo_vxus_spread_6m, credit_spread_hy, nfci, vix, rate_adequacy_gap_yoy, pif_growth_yoy, investment_book_yield
- **8 benchmarks:** VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE
- **Peer tickers (in DB):** ALL, TRV, CB, HIG

### Config imports
```python
import config  # config/ dir has __init__.py re-exporting from model.py and features.py
```

## Scripts to Create

Create ALL 24 scripts below. Each is a standalone experiment.

### Phase 1: Calibration (v37–v43)

**v37_baseline.py** — Measure current v11.0 lean pipeline. Per benchmark: N OOS, OOS R², IC, NW IC/p, hit rate, MAE, std(y_hat), std(y_true), σ_pred/σ_true. Aggregate: pool all OOS predictions. Save v37_baseline_results.csv.

**v38_shrinkage.py** — Post-hoc: y_hat_shrunk = α × y_hat for α ∈ [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]. No retraining. Verify IC/hit_rate invariance. Find optimal α. Save v38_shrinkage_results.csv.

**v39_ridge_alpha.py** — Custom WFO with three Ridge alpha grids: logspace(-4,4), logspace(0,6), logspace(2,6). RidgeCV(cv=None) for LOOCV. Ridge only, 8 benchmarks. Save v39_ridge_alpha_results.csv.

**v40_gbt_constraint.py** — A: Ridge only. B: Constrained GBT (lr=0.01, min_samples_leaf=10, subsample=0.7). C: Post-hoc 80/20 Ridge/GBT reweight. Save v40_gbt_results.csv.

**v41_winsorize.py** — Winsorize y_train at 5/95 and 10/90 within each fold. Evaluate vs original y_test. Both models. Save v41_winsorize_results.csv.

**v42_expanding.py** — A: Pure expanding (no max_train_size). B: Expanding + exponential decay weights. C: Expanding capped 120M. Save v42_expanding_results.csv.

**v43_features.py** — A: Ridge→7 [yield_slope, npw_growth_yoy, investment_income_growth_yoy, real_yield_change_6m, combined_ratio_ttm, mom_12m, credit_spread_hy]. B: GBT→7 [mom_12m, vol_63d, yield_slope, credit_spread_hy, vix, pif_growth_yoy, investment_book_yield]. C: Shared 7 [mom_12m, vol_63d, yield_slope, credit_spread_hy, vix, combined_ratio_ttm, npw_growth_yoy]. Use run_wfo(feature_columns=list). Save v43_feature_results.csv.

### Phase 2: Feature & Model Architecture (v44–v46)

**v44_pca.py** — Blockwise PCA within each fold. Macro block (yield_slope, real_rate_10y, credit_spread_hy, nfci, vix, real_yield_change_6m)→PCA. Insurance block (combined_ratio_ttm, npw_growth_yoy, investment_income_growth_yoy, pif_growth_yoy)→PCA. Combine with raw mom_12m, vol_63d. A: 2 components/block (6 total). B: 1/block (4 total). Ridge with logspace(0,6,50). Save v44_pca_results.csv.

**v45_bayesian_ridge.py** — A: Default BayesianRidge via run_wfo(model_type="bayesian_ridge"). B: Tight prior (alpha_1=1e-5, alpha_2=1e-5, lambda_1=1e-4, lambda_2=1e-4). C: BayesianRidge+GBT ensemble. Save v45_bayesian_ridge_results.csv.

**v46_classification.py** — y_binary = (y > 0).astype(int). LogisticRegressionCV(Cs=logspace(-4,4,20), cv=3, penalty="l2"). Metrics: accuracy, balanced_accuracy, Brier, log-loss. Save v46_classification_results.csv.

### Phase 3: Target & Structure Reforms (v47–v49)

**v47_composite_benchmark.py** — A: Equal-weighted composite target = mean of 8 per-ETF relative returns. B: Inverse-vol-weighted composite. C: Equity-only composite (VOO, VXUS, VWO, VDE). Run Ridge+GBT on single composite target. Save v47_composite_results.csv.

**v48_panel_pooling.py** — Stack all 8 benchmark-months into one panel. Add benchmark dummies. A: Panel with fixed effects. B: Panel shared coefficients only. Temporal split: all benchmarks for month t in same fold. Also test MultiTaskElasticNetCV with y reshaped to (n_months, n_benchmarks). Save v48_panel_results.csv.

**v49_regime_features.py** — Add regime indicators: hard_market=(combined_ratio>100), high_vol=(vix>20), inverted_curve=(yield_slope<0). A: +2 indicators. B: +3 indicators. C: +2 interactions (hard_market×yield_slope, high_vol×credit_spread_hy). Save v49_regime_results.csv.

### Phase 4: Additional Model & Data (v50–v55)

**v50_pred_winsorize.py** — Clip OOS predictions at training-window prediction percentiles. A: 5/95. B: 10/90. C: Combined with optimal α from v38. Save v50_pred_winsorize_results.csv.

**v51_peer_pooling.py** — Two-stage: (1) Train sector model on peer composite (ALL,TRV,CB,HIG) returns vs ETFs using macro features. (2) Add sector model OOS prediction as feature #13 for PGR Ridge. Alternative: just add pgr_vs_peers_6m if available. Save v51_peer_pooling_results.csv.

**v52_test_window.py** — A: test_size=1 (monthly). B: test_size=3. C: test_size=1 + expanding window. All with gap=8. Save v52_test_window_results.csv.

**v53_ard.py** — ARDRegression(threshold_lambda=10000). A: Default ARD. B: ARD+GBT ensemble. C: ARD with extended features (12+4 extras). Save v53_ard_results.csv.

**v54_gpr.py** — GaussianProcessRegressor with Matérn 5/2 kernel + WhiteKernel. A: GPR standalone. B: RBF kernel. C: Ridge+GPR ensemble (replacing GBT). Use 7-feature reduced set. Save v54_gpr_results.csv.

**v55_rank_target.py** — A: Rank-transform y_train within each fold (uniform [0,1]). B: Percentile-transform. Fit Ridge. Evaluate on original scale. Save v55_rank_target_results.csv.

### Phase 5: Data & Diagnostic Enhancements (v56–v60)

**v56_12m_horizon.py** — load_relative_return_matrix(conn, etf, 12). target_horizon_months=12, gap=15. A: Rolling 60M. B: Expanding. Save v56_12m_results.csv.

**v57_transforms.py** — A: Log-transform skewed features (npw_growth_yoy, investment_income_growth_yoy, etc.) via sign(x)*log1p(|x|). B: Rank-normalize all features within each fold for GBT. C: Add 1M and 2M lags of fundamental features (combined_ratio_ttm, npw_growth_yoy, investment_income_growth_yoy). Save v57_transforms_results.csv.

**v58_fred_features.py** — Check which of these are in fred_macro_monthly: PCU5241265241261 (auto ins PPI), CUSR0000SAM2 (medical CPI), CUSR0000SETA02 (used car CPI), MORTGAGE30US, THREEFYTP10. Compute 3M momentum for each. Add 2-3 to Ridge features. Save v58_fred_results.csv.

**v59_imputation.py** — Expand from 12 to 18 features by adding pe_ratio, pb_ratio, roe, buyback_yield, term_premium_10y, breakeven_inflation_10y. A: Forward-fill + training median. B: IterativeImputer within each fold. C: Add MissingIndicator columns. Save v59_imputation_results.csv.

**v60_diagnostics.py** — NOT a model experiment. Run on v37 baseline predictions. (1) Clark-West MSFE-adjusted test per benchmark. (2) MSE decomposition: bias² + variance. (3) Certainty-equivalent return gain (CRRA γ=2). Save v60_diagnostics_results.csv.

## Output Format

Each script prints:
```
============================================================
v3X: [Experiment Name] — Results Summary
============================================================

Per-Benchmark Results:
  Benchmark  N_OOS  OOS_R2    IC     Hit_Rate  MAE
  VOO        108    -0.0523   0.142  0.685     0.198
  ...

Aggregate (pooled):
  OOS_R2: -0.0234   IC: 0.187   Hit_Rate: 0.691   MAE: 0.205

vs. Baseline (from v37_baseline_results.csv):
  OOS_R2 delta: +0.1092 (+10.92pp)
  IC delta:     -0.003
  Hit rate delta: +0.005 (+0.5pp)
============================================================
```

For v37 (baseline), skip the delta section. For v60 (diagnostics), use a custom format showing Clark-West stats and MSE decomposition.

## Important Constraints

1. Do NOT modify any file outside `research/v37/`. All production code stays untouched.
2. All scripts runnable from repo root: `python research/v37/v37_baseline.py`
3. Add `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))` at top.
4. Handle missing benchmarks gracefully (skip with warning).
5. `warnings.filterwarnings("ignore", category=ConvergenceWarning)` to suppress sklearn noise.
6. random_state=42 everywhere.
7. Print progress: "Running benchmark VOO..."
8. Target <15 min per script. For complex scripts (v48, v51), limit to 4 benchmarks if needed.

## Verification

After creating all scripts, run v37_baseline.py first. If it fails, debug and fix. Then run v38_shrinkage.py to verify the expected shrinkage pattern (R² improves as α decreases, IC stays constant). Then run v60_diagnostics.py to get the MSE decomposition that confirms variance dominance.
```
