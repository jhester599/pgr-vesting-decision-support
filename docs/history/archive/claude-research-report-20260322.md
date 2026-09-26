# Upgrading a walk-forward stock prediction system

**The highest-impact improvements to a LassoCV/RidgeCV walk-forward system operating on ~300 monthly observations center on three areas: adding freely available macro and credit features from FRED, switching to ElasticNetCV with Bayesian uncertainty quantification, and implementing proper purge/embargo gaps calibrated to your 6–12 month prediction horizon.** These changes require minimal additional infrastructure while addressing the system's core constraints—small sample size, long horizons, and the need for calibrated confidence. The broader set of enhancements below spans feature engineering, model architecture, validation methodology, confidence calibration, insurance-specific data, portfolio construction, and backtesting rigor.

---

## Feature engineering that actually works at 6–12 month horizons

The strongest incremental features for medium-horizon equity return prediction are macro regime indicators and credit spread signals—not the NLP sentiment data that dominates short-horizon alpha research. With only ~300 observations, you need a disciplined approach: **keep total features under n/10 ≈ 30**, use PCA on correlated feature groups, and favor rank-transformed or z-scored features for robustness to outliers.

**Yield curve features** have the strongest empirical backing for your horizons. The 10Y-2Y spread (FRED: `T10Y2Y`) has preceded every US recession since 1970 with 6–18 month lead time, and for insurance stocks specifically, the yield *level* and *slope change* drive float income directly. Add three features: slope (`T10Y2Y`), curvature (`2×GS5 - GS2 - GS10`), and real rate (`GS10 - T10YIE`). Credit spreads complement these as risk-regime indicators—the Baa-Treasury spread (`BAA10Y`) and high-yield OAS (`BAMLH0A0HYM2`) capture risk premium variation that predicts low-frequency equity return fluctuations, a finding documented by Fama and French (1989) and Gilchrist and Zakrajšek (2012).

Cross-sectional features map directly to your relative-return prediction task. Computing PGR's percentile rank within a P&C insurance peer group (ALL, TRV, CB, HIG) for momentum, volatility, and valuation metrics is free via `yfinance` and captures the relative positioning that drives sector-adjusted returns. A single macro regime composite—PC1 extracted from NFCI, CFNAI, yield curve slope, and credit spread—condenses 10+ macro series into one feature while preserving the business-cycle information relevant to 6–12 month predictions.

For insurance-specific features, **motor vehicle insurance CPI** (FRED: `CUSR0000SETC01`) serves as a real-time rate adequacy proxy, while vehicle miles traveled (`TRFVOLUSM227NFWA`) proxies claims frequency. Insider trading data from SEC Form 4 filings—accessible free via the `edgartools` Python library—adds a well-documented alpha signal: insider purchases predict 3–8% abnormal returns over 6–12 months, with cluster buying by multiple insiders being particularly predictive.

Sentiment NLP features (FinBERT, VADER) and options-implied volatility spreads are lower priority. Sentiment alpha concentrates at daily-to-weekly horizons, requiring aggregation into "sentiment momentum" to be useful at your timeframe. The variance risk premium (IV minus realized vol) has solid academic support from Bollerslev et al. but requires building a historical IV series over time since free options data from `yfinance` lacks historical depth. Both are medium-priority additions after the macro and cross-sectional features are in place.

Critical implementation rules: lag all features by at least one month for reporting delays, use FRED's ALFRED vintage dates to ensure point-in-time accuracy for macro data, winsorize at 1st/99th percentiles before standardization, and standardize features using only in-sample data within each walk-forward window.

---

## ElasticNet and Bayesian regression outperform at small sample sizes

For ~300 observations, the model upgrade path follows a clear priority order. **ElasticNetCV is the single best replacement** for separate LassoCV/RidgeCV models because it blends L1 feature selection with L2 coefficient shrinkage, handling correlated predictors (e.g., multiple momentum variants) far more gracefully than pure Lasso. Tune the `l1_ratio` mixing parameter over `[0.1, 0.5, 0.9, 0.95, 1.0]` using `TimeSeriesSplit`—optimal values typically fall between 0.5 and 0.9 for financial features.

BayesianRidge from scikit-learn is the most important *complementary* model because it provides native uncertainty quantification via `predict(X_test, return_std=True)`, returning both point predictions and posterior predictive standard deviations. This directly enables confidence-based position sizing without requiring separate bootstrap procedures. ARD Regression (Automatic Relevance Determination) extends this by automatically pruning irrelevant features to exactly zero—a Bayesian analog of Lasso that's particularly valuable when you suspect only 2–3 features carry the signal.

The literature's most actionable finding for your sample size is the **forecast combination puzzle**: simple equal-weight averaging of diverse model predictions consistently outperforms learned stacking weights when N is small and targets are noisy. Stock and Watson (2004) and Claeskens et al. (2016) demonstrate this rigorously. The practical implication is powerful—train Ridge, ElasticNet, BayesianRidge, and a shallow Random Forest independently, then average their predictions with equal weights rather than using a learned meta-learner.

Gradient boosting (XGBoost/LightGBM) can capture nonlinear interactions missed by linear models, but at N~300 it requires extreme constraint: **max_depth=2, min_samples_leaf=20** (~7% of data per leaf), subsample=0.7, and strong L2 regularization (`reg_lambda=5.0`). These settings force gradient boosting to behave almost like a shallow ensemble, capturing only the strongest interaction effects. Neural networks, including TabNet, are generally not recommended at this sample size—the Gu-Kelly-Xiu (2020) results showing neural network superiority used ~30,000 stocks over ~60 years, a fundamentally different data regime.

Gaussian Process Regression deserves consideration for a secondary model. It provides excellent calibrated uncertainty estimates and scales as O(N³)—perfectly tractable at N=300. The limitation is feature dimensionality: GPs work best with ≤5–10 features after PCA reduction.

---

## Walk-forward validation needs proper purge gaps and CPCV

The most consequential WFO improvement is **calibrating your purge and embargo gaps to your prediction horizon**. If your label is a 6-month forward return, any training observation whose 6-month label window overlaps with the test period must be purged. For 6-month horizons, set purge = 6 months plus an embargo of 2 months for serial autocorrelation (8 months total). For 12-month horizons, use **15 months total** (12 + 3). Without proper purging, your walk-forward results contain information leakage that inflates apparent performance.

Combinatorial Purged Cross-Validation (CPCV), introduced by López de Prado, generates a *distribution* of backtest performance rather than a single path. With N=6 groups and k=2 test folds, CPCV produces 15 train-test splits and 5 distinct backtest paths from your data. The `skfolio` library provides the cleanest scikit-learn-compatible implementation via `CombinatorialPurgedCV(n_folds=6, n_test_folds=2, purged_size=6, embargo_size=6)`. This is significantly more informative than standard WFO for detecting overfitting—if performance varies widely across CPCV paths, the model is fragile.

Your 5-year rolling window is well-calibrated for the task. Academic consensus and empirical testing (including comparisons by The Alpha Scientist) favor rolling windows over expanding windows for financial ML because financial relationships are demonstrably non-stationary. A **hybrid approach**—expanding window with exponential decay weighting—offers a practical compromise that uses older observations while emphasizing recent data. Test both: if your features capture stable relationships (e.g., valuation ratios), expanding windows may win; if they're momentum-based, rolling will dominate.

**Fractional differentiation** addresses a subtle but important problem. Integer differencing (computing returns from prices) removes too much memory from the series, potentially destroying long-range predictive information. The `fracdiff` library's `FracdiffStat` class automatically finds the minimum fractional differencing order d* (typically 0.2–0.4 for stock prices) that achieves stationarity while preserving **90%+ correlation** with the original series. Apply this to raw log prices and any non-stationary feature series—not to returns, which are already d=1.

For structural breaks like COVID, consider adaptive window lengths: when the CUSUM filter (available in `mlfinlab` or easily implemented manually) detects a regime shift, temporarily shorten the training window to use only post-break data, then gradually expand back to the full 5-year window.

---

## Calibrating prediction confidence with practical tools

Translating regression predictions into position sizes requires quantifying *how much you should trust each prediction*. The most practical approach combines Bayesian uncertainty with fractional Kelly sizing.

BayesianRidge's `return_std` provides immediate prediction-level uncertainty. For any test observation, you get both ŷ (predicted return) and σ̂ (posterior standard deviation). The signal-to-noise ratio |ŷ|/σ̂ directly measures conviction—predictions where the expected return is large relative to uncertainty deserve larger positions. This is computationally free and requires no additional infrastructure.

For model-agnostic uncertainty, **block bootstrap** preserves time-series structure while generating prediction distributions. Use blocks of ~12 months for monthly data to capture annual cyclical patterns. The `arch` package provides `StationaryBootstrap` and `MovingBlockBootstrap` implementations. With 500–1000 bootstrap replications, you get percentile-based confidence intervals that account for both parameter uncertainty and model specification error.

**Conformal prediction** via the `MAPIE` library offers distribution-free coverage guarantees—a property no other method provides. Wrap any scikit-learn model with `MapieRegressor(method='plus')` to get prediction intervals with guaranteed finite-sample coverage. The caveat for financial data is that standard conformal prediction assumes exchangeability, violated by time series. Adaptive conformal prediction methods (temporal conformal prediction) adjust for this, though implementations are still maturing.

For position sizing, the **continuous Kelly criterion** (`f* = μ/σ²`) converts predicted excess returns and volatility into optimal capital allocation. Full Kelly is far too aggressive given parameter estimation error—**use 0.25× Kelly** for personal portfolios, which retains ~50% of optimal growth with 1/16th the variance. A practical implementation: `position = 0.25 × predicted_excess_return / estimated_volatility²`, capped at 30% for single stocks.

Realistic IC benchmarks for your system: at 6–12 month horizons for single-stock relative returns, an information coefficient of **0.03–0.08 is realistically achievable and economically meaningful**. Campbell and Thompson (2008) showed that even OOS R² of 0.5% is economically significant for return prediction. Track ICIR (mean IC / std of IC) above 0.5 as a minimum viability threshold, and monitor rolling IC for regime-dependent signal decay.

---

## Progressive-specific alternative data from free sources

PGR is uniquely transparent among insurers—it publishes **monthly** combined ratios, net premiums, and policy counts at `investors.progressive.com`, while peers report only quarterly. This data advantage should anchor your feature set.

For catastrophe exposure, NOAA's Storm Events Database (`ncei.noaa.gov/stormevents/`) provides county-level storm data with property damage estimates going back to 1950, downloadable as bulk CSV files for free. The NOAA Billion-Dollar Disasters dataset tracks major events exceeding $1B in insured losses. Swiss Re's annual sigma reports (freely available at `swissre.com/institute`) provide the authoritative global insured catastrophe loss figures. Construct quarterly aggregate cat loss features and focus on PGR-relevant states (Florida, Texas, California).

The most predictive insurance-specific leading indicators accessible via FRED include: motor vehicle insurance CPI (`CUSR0000SETC01`) for rate adequacy, vehicle miles traveled (`TRFVOLUSM227NFWA`) for claims frequency, used car prices CPI component for claims severity, and medical CPI (`CPIMEDSL`) for bodily injury severity. **Interest rate sensitivity is critical**: PGR's ~$65B investment portfolio is heavily fixed income with ~2–4 year duration, making the 2Y–5Y Treasury yield level and 6-month change directly relevant to float income projections.

Competitor analysis is achievable through quarterly earnings filings on SEC EDGAR (free via `edgartools`), where ALL, TRV, CB, and HIG all report combined ratios. State insurance department rate filings are publicly searchable—California's DOI (`interactive.web.insurance.ca.gov`) is the most transparent. Computing PGR's combined ratio rank versus peers and tracking the ratio of insurance CPI growth to overall CPI provides a systematic measure of industry pricing power.

---

## Black-Litterman and tax-aware rebalancing for the sell decision

The Black-Litterman model is the natural framework for converting ML predictions into portfolio weights because it directly handles relative views—"PGR will outperform XLF by X%"—which is exactly your prediction target. Use `PyPortfolioOpt`'s `BlackLittermanModel` with your ML predictions as views and calibrate the confidence matrix Ω using your model's cross-validation RMSE: `Ω_ii = RMSE²` from walk-forward residuals. When prediction confidence is low, the portfolio stays close to market equilibrium—a sensible default that prevents overtrading on weak signals.

For covariance estimation, **Ledoit-Wolf shrinkage** is essential even with a small number of assets, though for expected returns, shrinkage matters ~10× more than for covariances (Chopra and Ziemba, 1993). Black-Litterman implicitly provides return shrinkage by blending views with the market prior. The `skfolio` library integrates everything—walk-forward CV, shrunk covariance estimation, and Black-Litterman optimization—in a unified scikit-learn pipeline.

Tax-loss harvesting adds **0.5–1.3% annual alpha** according to Vanguard (2024) and Chaudhuri, Burnham, and Lo (2020). The optimal approach for monthly cadence: scan for positions with returns below -10%, sell to harvest the loss, replace with a correlated but not "substantially identical" security for 31+ days, and reinvest the tax savings. Integrate tax impact directly into sell decisions: `after_tax_expected_return = ML_prediction - max(0, unrealized_gain × tax_rate) / position_value`. This creates a natural asymmetry where unrealized losses increase the incentive to sell (tax benefit) while embedded gains raise the hurdle rate.

Rebalance quarterly with a **5% absolute deviation threshold**—trade only when portfolio weights drift beyond this band from target. For 6–12 month prediction horizons, more frequent rebalancing is generally not justified and amplifies estimation error through transaction costs.

---

## Avoiding backtesting self-deception

The most dangerous pitfall in your system is **double-dipping in feature selection**: if any feature screening (correlation filtering, variance thresholds) uses data from test folds, performance estimates are upward-biased. With LassoCV this is partially handled—Lasso's internal feature selection uses only training data—but ensure *all* pre-filtering steps are nested inside each walk-forward fold.

Apply **Benjamini-Hochberg (BHY) correction** to control the false discovery rate when testing multiple features or model configurations. Available via `statsmodels.stats.multitest.multipletests(p_values, method='fdr_bh')`, BHY is more powerful than Bonferroni while still controlling false positives at 5%. If you test 20 feature variants, a raw p-value of 0.01 might not survive correction—this is important information.

The **Deflated Sharpe Ratio** (Bailey and López de Prado, 2014) corrects for both multiple testing and non-normal returns. A system showing SR=0.75 from 240 monthly observations after exploring 200 configurations deflates to approximately **SR=0.32** under BHY correction—a 57% haircut. Harvey, Liu, and Zhu (2016) recommend requiring t-statistics above 3.0 (not the traditional 2.0) for new factors, reflecting the accumulated multiple testing burden across the field.

The **Campbell-Thompson OOS R²** is the most appropriate single metric for your system. Compute an expanding historical mean return at each step as the benchmark forecast. OOS R² above zero means your model beats naive prediction. Even OOS R² of **0.5–2% is economically significant** for return forecasting at 6–12 month horizons—don't be discouraged by apparently small numbers.

Regime-conditional performance analysis reveals whether your model works everywhere or only in specific environments. Classify months into four regimes (bull/bear × low/high volatility using rolling 12-month returns and volatility medians) and compute OOS R² separately for each. A robust model should show positive performance in at least 3 of 4 regimes. If performance concentrates in one regime, the model may be capturing a regime artifact rather than genuine predictive power.

---

## Monthly stability backtesting: proving the model works beyond vesting dates

The current backtest framework evaluates predictions exclusively at mid-January and mid-July vesting events — roughly 20 data points across 10 backtestable years. This is insufficient to distinguish genuine predictive skill from luck. A binomial test at n=20 and k=13 correct predictions yields p≈0.13 — not even marginally significant. Monthly backtesting expands the evaluation set to 120+ out-of-sample predictions, making a 60% hit rate statistically meaningful at p<0.01 and enabling diagnostic analyses that are impossible with semi-annual observations alone.

### Why monthly evaluation is essential

Semi-annual-only backtesting creates three blind spots that monthly evaluation eliminates.

First, **statistical power is too low to trust any performance metric**. With 20 observations, a model that gets lucky on 3–4 events can swing the hit rate from 50% to 65%. The confidence interval around a 65% hit rate at n=20 spans roughly 41%–85% (Wilson interval) — the model might be excellent or worthless, and you genuinely cannot tell. At n=120, the same 65% hit rate produces a confidence interval of 56%–73%, which excludes 50% convincingly.

Second, **you cannot diagnose why the model fails** at a given vesting event when you have only one data point per six-month window. If July 2022 was wrong, was it because the model was already failing in April? Was it a sudden regime shift in June? Monthly predictions create a continuous trail that reveals whether failures are abrupt or gradual, systematic or idiosyncratic.

Third, **seasonal and regime effects are invisible**. Does the model systematically perform better when predicting from Q4 observation dates (capturing January effects and year-end rebalancing flows) versus Q2? Does signal strength collapse during high-VIX environments or yield curve inversions? These questions require evaluation across the full calendar, not at two fixed points.

### Handling the overlapping-return problem

Consecutive monthly predictions with a 6-month forward target share 5 of their 6 months of return window. This serial overlap inflates naive significance tests because adjacent predictions are mechanically correlated — even a random model would show positive autocorrelation in its hit/miss sequence. Ignoring this problem overstates the effective sample size and produces artificially narrow confidence intervals.

Three complementary approaches address this. First, compute IC and hit rate on the full monthly series for power, but test significance using **Newey-West standard errors with lag = target_horizon - 1** (5 lags for 6M targets, 11 for 12M). Newey-West is designed exactly for this: it adjusts the variance-covariance matrix for serial autocorrelation up to the specified lag, producing valid t-statistics. Alternatively, Hansen-Hodrick standard errors provide equivalent corrections and are standard in the asset pricing literature.

Second, as a robustness check, **subsample to non-overlapping windows**: take every 6th monthly prediction for the 6M model and every 12th for the 12M model. This produces ~20 independent observations — the same count as vesting-only evaluation — but distributed across the full calendar rather than concentrated at two fixed dates. If performance holds on this non-overlapping subset, the signal is robust.

Third, **report both the full monthly and vesting-month-only results side by side**. The monthly series proves stability and enables diagnostics; the vesting-month subset confirms the signal is present at the actual decision points. Divergence between the two (strong monthly performance but weak at vesting dates, or vice versa) would itself be diagnostically informative and suggest seasonal dependencies in the feature set.

### What monthly backtesting reveals

With 120+ evaluation points, several analyses become feasible that are impossible at n=20.

**Rolling IC time series.** Compute a trailing 24-month rolling IC (Spearman correlation between predicted and realized relative returns). This reveals whether predictive power is stable, trending upward as data accumulates, decaying as relationships shift, or episodic — concentrated in specific market environments. Plot this as a time series alongside VIX or the yield curve slope to identify regime-dependent performance.

**Regime-conditional performance.** Classify each monthly evaluation into one of four regimes: bull/bear (rolling 12-month S&P 500 return above/below median) crossed with low/high volatility (trailing 63-day VIX above/below median). Compute OOS R² and hit rate separately for each quadrant. A robust model should show positive performance in at least 3 of 4 regimes. If performance concentrates entirely in one regime (e.g., bull/low-vol), the model may be capturing a regime artifact rather than a durable predictive relationship.

**Signal decay analysis.** If predictions issued in January are accurate for January–June but predictions issued in March are wrong for March–August, the features have short half-lives relative to the prediction horizon. This suggests either shortening the horizon or refreshing the model more frequently. Monthly evaluation makes this visible; semi-annual evaluation cannot detect it.

**Per-benchmark stability.** With 120 monthly evaluations × 20 benchmarks = 2,400 prediction cells, you can identify which ETF comparisons have stable predictive power and which are noisy. Perhaps PGR-vs-VFH (financials sector) is consistently predictable while PGR-vs-GLD (gold) is essentially random. This information should feed back into the recommendation layer: give more weight to benchmarks with stable historical performance and less to those where the model has no demonstrated edge.

### Implementation design

The backtest engine change is relatively contained. The `enumerate_vesting_events()` function currently generates only mid-January and mid-July dates. A parallel function — `enumerate_monthly_evaluation_dates()` — generates month-end dates from 2014 onward, producing the full evaluation calendar. The temporal slicing logic in `run_historical_backtest()` already works for arbitrary dates (it slices `X_full` to rows ≤ event_date), so the main change is feeding it more evaluation points.

```python
def enumerate_monthly_evaluation_dates(
    start_year: int = 2014,
    end_year: int | None = None,
) -> list[VestingEvent]:
    """Generate month-end evaluation dates for stability analysis.

    Returns VestingEvent-compatible objects for every month-end between
    start_year and end_year, snapped to the last business day of each month.
    These are not actual vesting events — they exist purely for backtesting
    the model's predictive stability across the full calendar.
    """
```

The reporting layer then presents two views: the **full monthly stability analysis** (IC time series, regime breakdowns, rolling hit rate, signal decay curves) and the **vesting-event-specific decision output** (the existing report with sell/hold recommendations). The monthly predictions are a validation tool, not a decision tool — the sell/hold recommendation logic in `_sell_pct_from_signal()` should continue to fire only at actual vesting dates. Monthly backtesting exists to answer the question: "is this model reliably capturing a predictive relationship, or did it get lucky on 20 coin flips?"

The key invariant to enforce in tests: `monthly_evaluation_dates ∩ vesting_event_dates` should produce identical predictions whether evaluated through the monthly or the vesting-specific code path. Any divergence indicates a bug in the temporal slicing logic.

### Statistical tests for monthly results

With monthly evaluation data, apply the following significance framework.

**Campbell-Thompson OOS R².** At each monthly evaluation point, compute an expanding historical mean return as the benchmark forecast. OOS R² above zero means the model beats naive historical-average prediction. Even values of 0.5–2.0% are economically significant for 6–12 month return forecasting. Report this alongside IC as the primary performance metric.

**Diebold-Mariano test with Newey-West correction.** Test whether the model's forecast errors are significantly smaller than those of the historical-mean benchmark. Use `statsmodels.stats.diagnostic` with HAC (heteroskedasticity and autocorrelation consistent) standard errors. This provides a proper p-value for model superiority that accounts for the overlapping-return autocorrelation.

**Hit rate significance via block permutation.** Rather than a naive binomial test, permute the prediction–realization pairs in blocks of 6 months (preserving serial dependence structure) and compute the hit rate on 1,000+ permuted samples. The empirical p-value — the fraction of permuted hit rates exceeding the observed rate — accounts for autocorrelation without requiring parametric assumptions.

---

## Conclusion: a prioritized implementation roadmap

The implementation sequence that maximizes value per unit of effort:

**Phase 1 (immediate, high impact):** Add yield curve, credit spread, and NFCI features from FRED (~5 new features). Fix purge/embargo gaps to match prediction horizons. Switch from separate LassoCV/RidgeCV to ElasticNetCV. Apply BHY multiple testing correction. Compute Campbell-Thompson OOS R². **Implement monthly stability backtesting** to establish baseline model reliability across 120+ evaluation points.

**Phase 2 (next iteration):** Add BayesianRidge as a parallel model for uncertainty quantification. Implement simple equal-weight forecast combination across ElasticNet, BayesianRidge, and Ridge. Add PGR-specific features (insurance CPI, VMT, cat loss proxy). Integrate fractional Kelly position sizing using Bayesian prediction intervals. Build regime-conditional and rolling-IC diagnostic reports from the monthly backtest data.

**Phase 3 (refinement):** Implement CPCV via `skfolio` for robust validation. Add Black-Litterman portfolio construction with ML views. Implement tax-loss harvesting logic with wash-sale-rule awareness. Test fractional differentiation on price-based features. Add per-benchmark weighting to the recommendation layer based on monthly stability results.

The overarching principle: with 300 observations, **simplicity and proper validation beat model complexity**. ElasticNet with well-chosen features, proper purging, and calibrated uncertainty will outperform sophisticated deep learning approaches that lack the data to generalize. The forecast combination puzzle confirms this—your best ensemble is likely just the average of three simple regularized models, each bringing a slightly different inductive bias to a fundamentally noisy prediction problem. And monthly stability backtesting is the mechanism that lets you *prove* this to yourself with statistical rigor rather than taking it on faith from 20 semi-annual coin flips.
