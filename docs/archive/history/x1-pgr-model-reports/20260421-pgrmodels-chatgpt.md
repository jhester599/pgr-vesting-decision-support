# Executive Recommendation

Begin with a **multi-horizon direction (“above/below”) classification** model as the first priority.  This path plays to the repo’s strengths in strict time-series WFO and small-sample modeling: binary outcomes are less noisy than raw price levels, align with the existing *classification_shadow* infrastructure, and directly answer the fundamental hold-vs-sell question.  Our recommended *baseline* will be a simple logistic regression (or penalized linear classifier) predicting whether PGR’s future price at each horizon will exceed today’s price.  As a benchmark, we’ll also run a direct *regression* model on forward *return* (e.g.  `price_{t+h}/price_t – 1`) using ridge/elastic-net, to produce an implied price forecast, and a naive random-walk/drift model for comparison.  

The **secondary path** will be a **BVPS×P/B decomposition** model: separately forecast *book-value-per-share (BVPS)* and *price-to-book (P/B)* components, then multiply to get an implied future price.  This leverages the existing EDGAR monthly fundamentals (book value) and reflects PGR’s insurance economics (book grows by retained earnings; P/B may mean-revert).  We *defer* full direct price regression (especially raw price-level) to later, because absolute-price trends are dominated by drift and very noisy; direct price models risk overfitting and are poor under strict WFO with ~150–200 monthly points.  Classification and decomposed valuation are more robust to small-N and easier to interpret (e.g. forecast “will PGR be above current price” or forecast fundamental drivers).  

This approach fits the repo’s philosophy and architecture: it uses **only time-series WFO**, respects the research/production boundary (development paths remain in `src/research`/results without altering live output), and favors simple regularized models on limited data【2†L121-L124】【28†L9-L17】.  Classification models can be slotted into the existing recommendation consensus framework (the repo already computes a relative-outperformance classifier for benchmarks).  The BVPS×P/B decomposition reuses the existing EDGAR data and book-value features with minimal restructuring.  Both paths preserve “research-only” artifacts (writing results to `results/research/`) until they are fully vetted.  In short, classification and valuation-decomposition align with the project’s strict no-leakage, small-sample rules【2†L121-L124】【28†L9-L17】 and can build on the repo’s utilities (WFO engine, calibration pipeline, etc.) with only additive changes. 

# Why This Fits the Repo 

- **Architecture & Data:**  The system already ingests price, dividend and EDGAR data into a SQLite DB and builds a rich monthly feature matrix for walk-forward modeling【28†L9-L17】【75†L12-L20】. We will reuse these pipelines. All new targets (price or classification) can be built from the committed DB tables (monthly PGR prices, dividends, splits, and EDGAR fundamentals).  No new data source or global-scope change is needed.  Ingestion and feature-engineering remain unchanged, and the new research outputs will live under `results/research/` per the artifact policy【36†L43-L49】.  Thus we maintain a clear production/research boundary.

- **Modeling Philosophy:**  The repo’s core principles forbid K‑fold CV and emphasize simple, regularized models on small data【2†L121-L124】.  Binary classification and low-dimensional decomposition adhere to that.  For classification we would use penalized logistic regression (akin to the ridge/elastic-net already used for return regression) and possibly a shallow tree ensemble as a challenger.  For decomposition we would again use simple linear models on BVPS and on log(P/B) (or even rule-based mean reversion) rather than heavy trees, given N≈150–200.  These respect the “large P, small N” caution (the Gemini peer review warns against over-parameterized models on ~150 samples【44†L25-L33】).  

- **Temporal Validation:**  Our evaluation will use the same strict walk-forward procedure in `src/models/wfo_engine.py` that enforces no leakage and an embargo (e.g. ≥horizon-month gap)【79†L73-L82】【79†L88-L97】.  Each horizon (+1m,+3m,+6m,+12m) will be treated separately in WFO, as in existing code with `target_horizon_months`.  This exactly fits the repo’s existing design, which already runs multi-model WFO per month.

- **Governance:**  All development is on a **research-only path**.  We will not alter the live monthly workflow (`scripts/monthly_decision.py`) or the production model versions.  Instead we create a parallel pipeline (under `src/research/` or new modules) whose results go to `results/research/`.  This respects the repo’s “no silent change” rule【28†L92-L100】【36†L43-L49】.  If a new modeling path shows merit, it would first be “shadow gated” (no change to recommendations) and only promoted after review.

In summary, classification and BVPS-decomposition are methodologically conservative fits: they can reuse the existing WFO engine, feature utils and calibration steps, and require only additive research code.  They also respect the explicit small-sample, no-leakage, research-boundary constraints set by this repo【2†L121-L124】【28†L9-L17】. 

# Target Design

We will construct separate targets for each horizon T = 1,3,6,12 months.  For each horizon, consider multiple formulations:

- **Raw Price-Level vs Transforms:**  We *avoid* modeling raw future price directly, since stock price is non-stationary and dominated by trend.  Instead, we focus on one of: (a) *forward return* – e.g. `target_return_T = (Price_{t+T}/Price_t - 1)`, or equivalently log-return; or (b) *log price* (which is effectively baseline plus return).  Forecasting returns (or log price) aligns with the current workflow (which targets 6-month DRIP total return). It also has more stationary distribution, making metrics (MAE, RMSE) meaningful. 

- **Binary Classification (Up/Down):**  The primary target in the new path will be a binary indicator:  
  \[
    y^T_{\text{class}} = 
    \begin{cases}
      1 & \text{if } Price_{t+T} \ge Price_t \text{ (up or flat)}\\
      0 & \text{if } Price_{t+T} < Price_t \text{ (down)}
    \end{cases}
  \]  
  at each month *t*.  This treats “above current price” vs “below current price” at horizon T.  (We could also test a small **threshold** edge case for “flat” if desired, but binary is simplest.)  This target is directly interpretable for hold-vs-sell decisions.  

- **Decomposed Targets:**  For the BVPS×P/B model, we define two sub-targets: the future book value per share and the future P/B multiple.  Since EDGAR provides monthly book_value_per_share, let  
  \[
    BV_{t+T} = \text{book\_value\_per\_share at }t+T,\quad PB_{t+T} = \frac{Price_{t+T}}{BV_{t+T}}.
  \]  
  We may model either the level of BV and PB, or their changes (e.g. T-month growth rates).  A *direct-level* forecast is simplest (predict $BV_{t+T}$ and $PB_{t+T}$, then compute $\hat P_{t+T} = \hat BV_{t+T}\times \hat PB_{t+T}$).  An alternative is to forecast *log-levels* (or $T$-month log-growth) to stabilize variance.  We will likely test both approaches.  

- **Regression vs Classification Trade-offs:**  For each horizon, we will compare:  
  - *Forward return regression:* target = percent return or log return.  This yields a numeric forecast $\hat P_{t+T} = P_t\,(1+\hat r_{t,T})$.  We transform back to price only after modeling.  
  - *Log-price regression:* target = $\log(P_{t+T})$ (with $\log(P_t)$ as implicit baseline).  This is equivalent to predicting log-return.  
  - *Price-level regression:* in practice we will not model $P_{t+T}$ directly, since it contains the baseline $P_t$.  (If we did, errors are huge on the level scale.) Instead always use returns or logs as above.  
  - *Binary classification:* target = up/down.  This will be evaluated with classification metrics.  

  Empirically, return-based or log-price regression often outperforms naive price-level regression on time-series data.  We expect raw price forecasts to be inferior because they require modeling a large drifting mean; return-based targets will give more stable learning and shrinkage.  

- **Horizon-specific vs Shared Models:**  Given the small sample size, we will *train separate models for each horizon*.  This avoids forcing one model to fit all horizons simultaneously, which would be complex and risk interfering cross-signals.  The WFO engine can take a `target_horizon_months` parameter, so we will run four separate WFO experiments (T=1,3,6,12) for each modeling family.  

# Feature Plan

We will begin with the repo’s existing feature matrix (from `src/processing/feature_engineering.py`) and categorize them for this new task.  These include (citing the code docstring in [75]):

- **Price/Momentum/Volatility:** lag-based momentum and volatility features (3m, 6m, 12m momentum; 21d and 63d vol)【75†L12-L20】.  These are directly predictive of future price moves (momentum, mean-reversion signals) and require no change.  We may also **add** short-term momentum (e.g. 1-month or 21-day momentum) for the +1m horizon if missing, since that uses only past data.  The repo already includes high-52-week ratio (`high_52w`)【75†L52-L53】 which can serve as an extreme-momentum feature.  

- **Technical Indicators:** existing Alpha Vantage–derived indicators (12-month SMA, 14-day RSI, MACD hist, Bollinger %b)【75†L12-L20】 are already in the matrix.  Based on the recent TA study, we will revisit a handful of those or similar ones.  In particular, the TA research identified *detrended OBV* and *normalized ATR* as promising【64†L38-L46】.  If volume data is available, we can add a “detrended OBV” feature (on daily volume) and a short-term ATR-like volatility.  We will treat these as optional additions under `src/research/` (so they don’t break production).   

- **Valuation/Macro:** fundamental ratios are already in the feature set: P/E, P/B, ROE【75†L12-L20】.  Additionally, the pipeline derives book_value_per_share (BVPS) monthly.  For decomposition models, BVPS itself (or its log) will become an input *and* a target.  Macro features from FRED (yield curve slope, curvature, real 10y rate, IG/HY credit spreads, NFCI, VIX)【75†L55-L63】 capture the market regime.  We will reuse these directly; no new macro data is needed.  If warranted, we might include the recent *change* in rates (e.g. month-on-month 10y rate change) as features, but initial focus is on existing aggregates (slope, spreads, etc.), which the repo already computes.  

- **Underwriting/Insurance Fundamentals:** these are perhaps the richest novel signals for PGR.  The existing feature set includes combined ratio (TTM), premium growth, gainshare estimates, underwriting income, unearned premium pipeline, ROE trend, investment yield, and buyback metrics【75†L23-L31】【75†L33-L43】.  We will reuse all of these, since they directly affect PGR’s future profitability and thus valuation.  For example, low combined ratio or rising underwriting income likely presage price appreciation.  We might normalize some on a “change from trend” basis (the code already has trailing YOY growth rates for many), which suits regression targets.  

- **Peer/Market Relative:** Synthetic “6m ahead” features exist (PGR’s 6m return minus peer ETF returns)【75†L68-L72】.  These measure how PGR has done relative to peers, and may be predictive of mean-reversion or momentum.  We will keep `pgr_vs_peers_6m` and `pgr_vs_vfh_6m`.  If we extend beyond 6m target, we may add analogous 1m or 3m peer spreads from history (these can be computed easily from the same price series).   

- **Additional Features:**  We will consider adding a few horizon-specific features:
  - **Short-term momentum/vol:** e.g. 1m (≈21-business-day) momentum for the +1m target, or 3m features for the +3m target, etc.  The repo does not have 1m momentum by default (it starts at 3m), so adding `mom_1m` could improve short-horizon forecasts.  - **Dividend yield / Changes:** If dividends are material, the drop on ex-dividend dates can affect prices.  We have dividend history; we might include the upcoming dividend yield or change as a feature. (However, at monthly granularity this may be low-frequency.)  
  - **Technical from TA study:** If volume data is available, implement the identified features (`ta_pgr_obv_detrended`, `ta_pgr_natr_63d`) in research code as additional regressors, to see if they improve classification.  
  - **Exclude:** We will **exclude any features that leak forward information** (none of the above do, by design).  We will also avoid overly redundant features: e.g. if P/B and price/BV are equivalent, we might drop one for parsimony.  The config file already has `FEATURES_TO_DROP` (v4.3) to remove any known redundancy【77†L1036-L1044】; we can add to that if needed after initial tests.  

All existing features are “time-indexed” and implement look-back only, so they fit strict no-leakage rules.  We will reuse them wholesale for the new models.  Any new features will be implemented in the same module (or research feature scripts) and subject to the same WFO and embargo logic.  

# Model Plan

We will evaluate **multiple model families** as baselines and challengers:

- **Baselines:**  
  - *Naive Random Walk:*  Predict no change (future price = today’s price, or return=0).  This corresponds to “always hold as is”.  Its MAE/RMSE is the baseline regression error.  
  - *Drift/Mean-Growth:*  Forecast based on the historical average return (e.g. $\hat r = \bar r_{past}$).  For example,  annualize the mean monthly return as a constant forecast.  
  - *Valuation Reversion:*  For decomposition, a simple rule: P/B regresses to its long-term mean.  E.g. forecast $\hat PB_{t+T} = \mu_{PB}$ (long-run mean P/B) and $\hat BV = BV_t$ (assuming book doesn’t jump).  This “mean reversion” baseline is naive but domain-informed.

- **Ridge / Elastic Net Regression:**  Direct regression on forward return or log-price, as used in production for relative returns.  We will use `sklearn` pipelines with `StandardScaler` + `RidgeCV` or `ElasticNetCV`.  These satisfy the “simple, regularized” mandate and are already heavily tested.  Hyperparameters (alpha, L1/L2 mix) will be tuned via inner split on each WFO fold, respecting embargo.  We will build one model per horizon (each with its own CV).  

- **Tree Ensembles (Gradient Boosting):**  A shallow Gradient Boosting model (e.g. lightGBM or sklearn’s HistGradientBoosting) as a nonlinear challenger.  Because of small N, we will keep tree depth shallow (e.g. depth ≤3–5).  This can capture simple interactions or nonlinearity among features.  We will compare GBT to ridge to see if any extra gain justifies the complexity.  (The repo’s production uses GBT as well, so this is consistent.)  

- **Logistic Regression / Classification:**  For the binary targets, use `LogisticRegression` (with L2 penalty) in a pipeline.  We will train a separate classifier per horizon.  As with regression, we must scale features within each fold to avoid leakage.  We will measure balanced accuracy, Brier score, log-loss, precision/recall, etc.  We will also try a simple decision-tree or gradient-boosted classifier as a challenger to see if linear or nonlinear classification performs better.

- **Ordinal or Multi-Class (if relevant):**  If initial binary models perform poorly, we may experiment with an *ordinal classifier* by bucketizing the percent-return (e.g. “up big”, “flat”, “down big”).  However, given limited data, we will likely stick to binary for clarity.  

- **Ensembles:**  Ultimately, we may ensemble across these families.  For example, a weighted average of ridge and GBT predictions can sometimes reduce error.  Or for classification, an ensemble voting or average of logistic and tree classifiers.  But ensemble blending would come *after* we identify strong individual models.  

All models will be wrapped in the existing WFO engine (`run_wfo`) with temporal splits.  We will run each model independently and record fold-by-fold forecasts and metrics.  We will compare their MAE/RMSE (for regression tasks) and accuracy/Brier (for classification) to select the top approach for each horizon.  

# Evaluation Plan

We will use **walk-forward evaluation** exactly as the repo does for relative-return models.  In practice, for each horizon T, we will do:  

1. **Feature Matrix Alignment:**  Build the feature matrix up to the latest available date.  For regression models we will compute the forward return target series (dropping last T months).  For classification we compute the up/down label for each month (dropping last T months similarly).  If we use decomposition, we align the BVPS and P/B targets to the same index.  

2. **WFO Splits:**  Call `run_wfo(X, y, target_horizon_months=T)` as in `src/models/wfo_engine.py`.  This creates sequential train/test folds with a fixed-size rolling window and an embargo gap of T months (to avoid overlapping forward windows)【79†L88-L97】.  We will use the same `WFO_TRAIN_WINDOW_MONTHS` and `WFO_TEST_WINDOW_MONTHS` from `config`, as in the production stack (e.g. ~120m train, 24m test by default).  

3. **Metrics (Regression):**  For continuous models (forecast price or return), we will compute on the test folds: **MAE, RMSE, MAPE or sMAPE** (to account for price scale), plus **Directional Hit Rate** (fraction of months where the sign of return is predicted correctly) and possibly **Rank IC** (correlation between predicted scores and actual outcomes) if we want a finance-style metric.  These echo the repo’s typical diagnostics for signal quality.  

4. **Metrics (Classification):**  For up/down, we compute per-fold: **balanced accuracy**, **Brier score** (squared error of predicted probability), **log loss**, and **precision/recall (especially recall on the “down” class)**.  We will also produce a **calibration plot** (predicted probability vs actual frequency) to check model calibration.  Since the repo uses a “shadow gate” on classification, good calibration is important for the gate.  

5. **Decomposition Error:**  For BVPS×P/B, we will measure error on each leg.  For example, take log-errors:  
   \[
     e_{BV} = \log(\hat BV_{t+T}) - \log(BV_{t+T}), \quad
     e_{PB} = \log(\hat PB_{t+T}) - \log(PB_{t+T}). 
   \]  
   We will report RMSE on $e_{BV}$ and $e_{PB}$ separately (or the variance explained), as well as the combined price RMSE from $\hat P = \hat BV \times \hat PB$.  This isolates which component is driving most of the error.  

6. **Economic Usefulness:**  Beyond standard metrics, we will estimate **practical decision impact**.  For classification, we can simulate a simple strategy: e.g. “if predicted down at 6m, sell X shares now; if predicted up, hold.”  We can compare cumulative return of this strategy vs a naive buy-and-hold.  For regression, we could test if using the price forecast to decide sell thresholds would have improved performance.  (This is exploratory, since the repo’s focus is more statistical; but we should comment on it qualitatively.)  

7. **Walk-Forward Diagnostics:**  We will produce the same WFO fold reports (fold indices, date ranges, train/test sizes) to verify no leakage【79†L73-L82】【79†L88-L97】.  Additionally, we will examine **regime slices** (e.g. boom/bust periods) to check stability.  The final outputs will include monthly signals (for classification, the predicted probability) stored under `results/research/vXYZ_classification.csv` and similar for regression (predicted returns, intervals).  

**Success criteria:**  We will look for a model that significantly beats the baselines on out-of-sample accuracy.  For regression, an RMSE below the drift-baseline RMSE and a direction hit rate substantially above 50% would be promising.  For classification, balanced accuracy well above 50% and sharp calibration would justify further work.  Improvement needs to be consistent (not a one-fold fluke).  We would require at least, say, 60–70% accuracy on direction and a substantial reduction in regression error before considering a shadow trial.  

If classification or decomposition yields minimal gain over naive (e.g. accuracy ~50% or RMSE ~equal to drift), we will halt further development.  The most likely false confidence trap is overfitting noise: small changes in train/test splits could yield seemingly good metrics by chance.  We will guard against this by robust WFO testing and by focusing on simple models.  

# Implementation Plan 

We propose breaking this into several incremental development steps (each PR-sized):

1. **Target Construction Module:**  
   - *Files:* Create `src/processing/target_constructor.py` (or under `src/research/`) with functions to compute new targets.  For example, `build_horizon_targets(price_df, horizon)` that returns forward returns, binary up/down, log price.  Also implement `build_bvps_pb_targets(edgar_df, horizon)`.  
   - *Tests:* Add tests (in `tests/test_target_constructor.py`) to verify alignment: e.g. a known price series should yield correct return and binary labels.  Confirm that first N-T rows align to `price[t+h]`.  
   - *Artifacts:* Unit tests, and possibly dump of a small feature matrix including new target columns.  

2. **Feature Matrix Extension (Research Only):**  
   - *Files:* Copy or extend the existing `build_feature_matrix()` (in `src/processing/feature_engineering.py`) to optionally include multiple target columns (e.g. `target_1m_return`, `target_1m_up`, etc.).  Alternatively, write a research helper `src/research/build_feature_matrix_ext.py` that reads the DB and uses the new target functions.  We should not modify the production builder; instead add a research entrypoint.  
   - *Tests:* Ensure no data leakage: e.g. simulate a fake DB and test that targets align with only past info.  
   - *Artifacts:* A sample extended feature matrix (parquet or CSV) in `results/research/` with targets for a few months.  

3. **Logistic Regression Pipeline:**  
   - *Files:* In `src/models/regularized_models.py`, add a function like `build_logistic_pipeline()` that returns an `sklearn` `Pipeline` with `StandardScaler` and `LogisticRegression(C=…regularization…)`.  
   - *Tests:* Add `tests/test_model_pipelines.py` to ensure the pipeline can `fit` on simple data and `predict_proba`.  Also verify that feature scaling is inside the pipeline (reuse pattern from existing tests).  
   - *Artifacts:* none yet.  

4. **WFO Runner for Classification:**  
   - *Files:* We will use the existing `run_wfo`. It can already handle any `y` series (the test shows using `target_horizon_months`).  We just supply the binary `y`.  We will likely wrap it in a script, e.g. `scripts/run_price_classification.py`.  
   - *Tests:* Similar to `test_wfo_engine.py`, but use a synthetic dataset where the “target” is 0/1.  Confirm WFO runs without error and that folds respect time order.  
   - *Artifacts:* A folder `results/research/classification` with CSV outputs of fold-level predictions for each horizon.  

5. **Regression Models for Return/Log-Price:**  
   - *Files:* Use existing `build_ridge_pipeline()` etc. Possibly update `run_wfo` calls to allow different model types.  Create a driver script `scripts/run_price_regression.py` that loops over horizons and model types (ridge, GBT).  
   - *Tests:* Add synthetic tests for WFO with regression (e.g. use linear combination data as in `test_wfo_engine`).  Check that metrics make sense.  
   - *Artifacts:* CSV results for regression predictions and actual vs predicted (for diagnostics).  

6. **BVPS×P/B Model:**  
   - *Files:* Under `src/research/`, implement `src/research/decomposition_model.py`.  It will: split data into two targets (BVPS and P/B), train two models (one for BVPS growth or level, one for log(P/B)), then multiply forecasts.  We may reuse ridge on each leg.  
   - *Tests:* On synthetic data where true price = BV*PB, test that errors decompose.  Also test a trivial case (constant BV, PB) for sanity.  
   - *Artifacts:* Summary of BVPS and P/B prediction errors, final implied price errors.  

7. **Evaluation and Comparison:**  
   - *Files:* A consolidated notebook or script `scripts/evaluate_price_models.py` that reads the WFO results from all methods and computes the metrics (MAE, RMSE, acc, etc.), and produces tables/plots.  
   - *Tests:* Verify that metrics code correctly computes known values (can reuse metrics from `sklearn`).  
   - *Artifacts:* A markdown report or notebook in `results/research/` summarizing which models won on each metric/horizon.  

8. **Technical-Indicator Additions (Optional):**  
   - *Files:* If pursuing TA features, implement them in `src/research/ta_features.py`.  For example, compute `OBV_detrended` from daily data and aggregate to monthly.  
   - *Tests:* Validate on known series (e.g. OBV on simple up/down sequence).  
   - *Artifacts:* Rerun WFO with these features, and compare performance.  

Each PR should include new tests and documentation as appropriate, and commit intermediate research outputs only under `results/research/` (per [36]).  

# Final Recommendation

**Order of Work:**  First implement and test the classification pipeline for +1m and +3m horizons (simpler horizons give more data points for WFO).  Compare to ridge regression on return as a sanity check.  If classification shows promise (say >60% balanced accuracy), extend to +6m and +12m.  In parallel, start the BVPS×P/B code, at least for +12m (longer horizon may rely more on fundamentals), and gradually for shorter if needed.  Only after those are stable would we entertain direct regression ensembling.  

**Expected Failure Modes:**  Be wary of *overfitting*: with ~150 samples, a model can look good on a particular partition but fail on another.  We must rely on the WFO gaps (embargo) to mitigate autocorrelation leakage【79†L88-L97】.  Also check that calibration does not degrade (especially for classification).  The most likely false confidence is a high in-sample accuracy that drops out-of-sample; hence we will be conservative in promoting any model.  

In summary, we recommend **multi-horizon classification** as the first research path, with a **BVPS×P/B decomposition** as a backup path.  We avoid raw price regression at the outset.  This plan uses only existing data, aligns with the repo’s validation rules, and reuses its feature and modeling framework (features like momentum, volatility, underwriting metrics【75†L12-L20】【75†L33-L43】).  It preserves the clean separation of research and production【28†L92-L100】【36†L43-L49】, and positions us to rigorously evaluate predictive value before any adoption into the live decision process.  

**Sources:** The above draws on the repo’s architecture and feature-engineering code【28†L9-L17】【75†L12-L20】, its project principles【2†L121-L124】, and recent TA research findings【64†L38-L46】 to ground the recommendations.