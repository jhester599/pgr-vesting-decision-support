# Peer review of pgr-vesting-decision-support

**The system's most important finding is also its most uncomfortable one: 15+ research cycles producing persistently negative OOS R² is not a calibration problem—it is strong evidence that the predictive signal does not exist at this horizon.** The DEFER-TO-TAX-DEFAULT fallback is doing the right thing. The engineering is thoughtful and the architecture reflects serious quantitative sophistication, but the project has crossed from productive research into research overfitting territory. The clearest path forward is to accept the null, simplify radically, and let the tax-aware recommendation layer—which is the system's genuine value—stand on its own without a prediction stack that actively undermines confidence.

This review is based on the detailed system description provided (the repository appears to be private and not publicly indexed). All findings are grounded in the described architecture, known issues, and research on best practices for each component.

---

## The ensemble architecture is sound but internally redundant

The 4-model ensemble (ElasticNet, Ridge, BayesianRidge, GBT) pairs three regularized linear models with one nonlinear learner. This is a defensible choice in principle—regularization is essential for financial data's punishing noise floor, and GBT captures interactions that linear models miss. In practice, however, **Ridge, ElasticNet, and BayesianRidge produce highly correlated predictions** because they share the same linear functional form and differ only in penalty structure. Ensemble diversity—the primary reason ensembles outperform individual models—is substantially undermined.

A more effective ensemble would replace one or two linear models with fundamentally different approaches. A historical-mean forecast (the benchmark any model must beat), a simple momentum or mean-reversion signal, or a rank-based model would each contribute genuinely independent information. Alternatively, replacing BayesianRidge with a BayesianRidge that outputs full posterior distributions—and propagating that uncertainty into Black-Litterman view confidence—would extract more value from the Bayesian framework than merely averaging its point prediction with near-identical competitors.

The observation-to-feature ratio being "borderline in some configurations" is a critical red flag. Academic consensus puts the minimum at **10:1 for stable generalization**, with **20:1+ preferred for noisy financial data**. At borderline ratios, Ridge and ElasticNet aggressively shrink all coefficients toward zero (producing mean-like predictions), while GBT overfits to training-set noise. This directly explains both the negative OOS R² and the frequent DEFER-TO-TAX-DEFAULT outcome. Before any further model experimentation, the feature count should be cut aggressively—preferably to 5–8 features maximum given likely training set sizes after CPCV purging.

---

## Walk-forward validation needs aggressive purging for 6-month horizons

CPCV (Combinatorial Purged Cross-Validation) is the right choice over standard walk-forward—it produces a distribution of performance metrics rather than a single path-dependent estimate, enabling direct estimation of the Probability of Backtest Overfitting (PBO). However, correct implementation for **6-month prediction horizons demands purge windows of ~126 trading days** plus an embargo of at least 1–2% of total observations beyond the purge boundary. Common implementation errors include off-by-one index mistakes in purge boundaries and insufficient purge/embargo sizes calibrated for shorter horizons.

The critical concern is that aggressive purging for 6-month horizons can **consume a large fraction of available training data**, shrinking effective sample sizes below the threshold where any model can learn reliably. With, say, 15 years of monthly data (~180 observations), a 6-month purge window removes roughly 6 observations per test group boundary. If CPCV uses N=6 groups with k=2, each of the 15 test splits loses 12+ training observations to purging—potentially reducing training sets to ~150 observations. Combined with even 15–20 features, the effective observation-to-feature ratio drops to **7:1 to 10:1**, firmly in the danger zone.

The system should verify that: (a) purge windows span the full 6-month label horizon on both sides of each test group boundary, (b) the embargo extends beyond the purge, (c) feature engineering statistics (means, standard deviations for z-scoring) are computed only on purged training data, and (d) feature selection happens inside the CV loop, never on the full dataset. A single leaked feature—such as a z-score computed with full-sample statistics—can inflate in-sample R² enough to mask the absence of genuine signal.

---

## Calibration and conformal prediction are misapplied in this context

**Platt scaling and isotonic calibration are classification methods** designed to transform classifier scores into probabilities via sigmoid or piecewise-constant mappings. Applying them to continuous regression outputs (predicted stock returns) is a category error. If the system converts regression to classification (predicting "outperform" vs. "underperform"), it discards magnitude information critical for portfolio construction. If it applies these methods directly to return predictions, the sigmoid/isotonic mapping has no meaningful interpretation.

For regression uncertainty quantification, the correct approach is to calibrate predicted confidence intervals—for example, by adjusting predicted distributions so that stated X% intervals achieve X% empirical coverage on held-out temporal data. The BayesianRidge model already provides posterior variance estimates that could serve this purpose directly, without the conceptual mismatch of classification calibration.

Conformal prediction intervals face a more fundamental obstacle: **standard conformal prediction requires exchangeability, which financial time series violate** due to serial correlation, volatility clustering, non-stationarity, and heavy tails. Naively applied CP intervals will be too wide in calm markets and dangerously narrow during volatility spikes—precisely when accurate intervals matter most. Time-series-adapted methods exist (ACI by Gibbs & Candès 2021, EnbPI by Xu & Xie 2021), but these provide only marginal coverage guarantees, not the conditional coverage that financial decisions require.

For 6-month horizons, the calibration set size becomes a binding constraint. Non-overlapping 6-month calibration observations accumulate slowly—10 years of data yields only ~20 calibration points, far too few for isotonic regression (which needs ~1,000+ points per Caruana et al. 2005) and marginal even for Platt scaling's two-parameter fit. **The recommendation is to replace the current calibration stack entirely** with direct use of BayesianRidge posterior uncertainty for view confidence and a simple historical-residual-based interval that adapts to recent volatility.

---

## The Black-Litterman and Kelly stack needs different inputs

The Black-Litterman framework is well-chosen for this problem—it starts from equilibrium (market-cap-weighted benchmark) and tilts based on views, which maps naturally to the concentrated-position unwinding decision. The key question is how ML signals feed into view formation and confidence.

**When OOS R² is negative, Black-Litterman should receive no views.** The model predicts worse than the historical mean, meaning any ML-derived view actively degrades the portfolio. With no views, BL correctly defaults to equilibrium weights—exactly the diversified benchmark allocation that a concentrated-position holder should target. The system appears to handle this correctly via the DEFER-TO-TAX-DEFAULT mechanism, but the logic should be explicit: negative OOS R² → empty view vector → equilibrium allocation → tax-default sell rule.

The Kelly criterion has an even sharper implication. **Kelly with zero or negative edge returns f* ≤ 0**, meaning no bet should be placed. In the RSU context, this translates to: the investor has no informational edge justifying concentration, so the optimal action is to sell and diversify fully. Even fractional Kelly (half-Kelly, quarter-Kelly) requires a positive edge to start from—it reduces the fraction of a positive edge, but cannot create an edge from nothing. If the system uses Kelly sizing with negative-R² model outputs, it is computing portfolio weights that are dominated by estimation error rather than signal.

The benchmark universe (VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE) is reasonable but has two issues. First, **VWO is largely a subset of VXUS**, creating collinearity that destabilizes covariance estimation. Second, BND and VMBS both carry interest-rate sensitivity correlated with PGR's insurance investment portfolio, slightly reducing diversification benefit. Consider replacing VWO with VB (small-cap) or VNQ (REITs) for more independent factor exposure, and replacing one of BND/VMBS with VTIP (inflation-linked bonds).

---

## Fifteen research cycles confirm the null hypothesis

The v9-through-v24 research progression—testing benchmark reduction, feature replacement, target reformulation, and promotion gates—represents at least 15 independent trials on the same dataset. Bailey & López de Prado's work on the Deflated Sharpe Ratio demonstrates that **after only 45 variations, a strategy will appear to have SR ≥ 1.0 purely by chance**. With 15+ cycles each testing 3–5 variations, the effective trial count easily exceeds 50–100, virtually guaranteeing discovery of spurious patterns.

The Quantopian study of 888 strategies found that "the more backtesting a quant has done for a strategy, the larger the discrepancy between backtest and out-of-sample performance." This is exactly the pattern observed here: models may show glimmers of in-sample promise that collapse out of sample. Each research cycle's "improvement"—a different benchmark subset, a different feature set, a different target formulation—is another draw from the null distribution, inflating the expected maximum IS performance without improving true predictive power.

The productive response is not v25. It is to:

- **Compute the Deflated Sharpe Ratio** across all trials conducted, which will almost certainly show that no trial survives correction for multiple testing
- **Estimate the Probability of Backtest Overfitting** using CSCV on the full set of configurations tested
- **Accept the null hypothesis**: predicting PGR's 6-month relative return versus a small ETF universe, using publicly available data, does not produce exploitable signal—this is consistent with semi-strong market efficiency for a well-covered large-cap stock
- **Retain the tax-aware recommendation layer** as a standalone decision tool that does not depend on ML predictions

Academic state-of-the-art for stock return prediction achieves OOS R² of **0.5% to 1.5%** (Campbell & Thompson 2008, Rapach & Zhou 2020). These results use the broadest possible cross-sections, decades of data, and carefully constrained models. Predicting a single stock's relative return against 8 ETFs is a much harder problem with far less data. **Negative OOS R² is the expected outcome, not a failure of engineering.**

---

## Software architecture and technical debt recommendations

**Config management**: If config.py uses hard-coded Python values rather than externalized YAML/JSON with typed validation, this should be refactored. Industry-standard tools (Hydra, OmegaConf, Pydantic BaseModels) separate hyperparameter search spaces from selected values, enforce type safety, and produce diffable, version-controlled configuration artifacts. Each research cycle should have its own named config file, not a code modification.

**Data ingestion risks**: Alpha Vantage's free tier allows only **25 requests/day** (reduced from 500 in recent years), and rate-limit responses return a JSON `"Note"` field rather than HTTP errors—a silent failure mode that can corrupt data if not explicitly checked. SEC EDGAR enforces **10 requests/second** and requires a User-Agent header. FRED data undergoes retroactive revisions (vintages), meaning the same series ID can return different values at different times—a subtle look-ahead bias if the pipeline uses revised data that wasn't available at the original timestamp.

**Test suite priorities**: The most critical tests for this system are:

- Temporal ordering assertions: every training observation precedes every test observation in every fold
- Purge/embargo verification: no label horizon overlap between train and test sets
- Feature leakage canary tests: inject a known future-only signal and verify it produces zero coefficient
- Tax logic boundary tests: verify correct behavior at short-term/long-term capital gains thresholds, NIIT thresholds, and supplemental wage withholding breakpoints
- Recommendation fallback tests: verify DEFER-TO-TAX-DEFAULT triggers correctly when model confidence is low

**GitHub Actions**: Workflows should mock external API calls (Alpha Vantage, FRED, EDGAR) rather than hitting live endpoints, pin action versions to commit SHAs rather than mutable tags (supply chain risk), and use fixed random seeds for reproducible CI runs. For a pipeline this complex, a scheduled workflow running the full walk-forward backtest weekly or monthly—with performance regression alerts—would catch data drift or API changes early.

**Technical debt from v9–v24**: Each completed research cycle likely left behind code paths, feature engineering functions, and configuration variants that are no longer active. A systematic audit should identify and remove dead code, consolidate utility functions, and archive experiment configs into a clearly separated `experiments/` directory. The ratio of active production code to legacy research code is probably unfavorable—ML repositories carry **2× the technical debt density** of traditional software (Bhatia et al. 2023) and remove it at rates below 5%.

---

## The tax-default recommendation is the system's real value

The 50% sell default is squarely within the range of sound financial planning advice. Most advisors recommend selling enough to diversify below 10–20% single-stock concentration, and **69% of individual U.S. stocks experience a catastrophic loss (>70% decline)** over their lifetime (Eaton Vance research). A staged selling approach—which the 50% sell at each vesting event implements—balances tax efficiency against concentration risk without requiring predictive accuracy.

The v13.1 diversification-first recommendation layer, by prioritizing concentration reduction over model-derived alpha, is the most defensible component of the system. Rather than trying to time the sell decision based on ML-predicted relative returns (which the model cannot reliably produce), it applies the overwhelming actuarial evidence that concentrated single-stock positions destroy wealth more often than they create it. This layer should be promoted to the primary and only recommendation path, with the ML pipeline demoted to an optional research/monitoring tool.

## Concrete next steps to improve the system

**If the goal is a better decision tool** (not a better prediction model), the path is straightforward: strip the ML prediction stack from the production recommendation path entirely. Use Black-Litterman with no views (equilibrium allocation) to determine target diversification weights. Apply the tax-aware sell logic to determine how much to sell at each vesting event. Display historical PGR performance versus the benchmark for context, but do not condition recommendations on predicted future returns.

**If the goal is continued ML research**, the approach should change fundamentally. Reduce features to 5–8 maximum. Use a single model (Ridge with strong regularization) rather than a 4-model ensemble. Target a binary classification (outperform/underperform) rather than continuous returns to reduce the dimensionality of the prediction problem. Apply the Deflated Sharpe Ratio to all results. Maintain a true holdout set (the most recent 3+ years) that is never touched during development. Pre-register each experiment's hypothesis, configuration, and success criterion before running it. **If OOS R² remains negative after 3 pre-registered experiments, terminate the ML research program.**

The most impactful single change would be computing and prominently displaying the **Probability of Backtest Overfitting** across all v9–v24 configurations. This number will communicate, more clearly than any code review, whether the prediction stack has information value or whether the system's honest best recommendation is exactly what it already defaults to: sell 50% and diversify.