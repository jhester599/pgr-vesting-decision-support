# Deep Research Prompt: Improving PGR Relative Return Prediction Accuracy

---

## Repository & Access

**GitHub:** https://github.com/jhester599/pgr-vesting-decision-support

If you cannot access GitHub directly, attach the following files (paths relative to repo root):

**Core model files:**
- `config/features.py` — feature definitions, benchmark universe, MODEL_FEATURE_OVERRIDES
- `config/model.py` — WFO parameters, ensemble config, calibration settings
- `src/models/wfo_engine.py` — Walk-Forward Optimization engine
- `src/models/multi_benchmark_wfo.py` — multi-benchmark ensemble runner
- `src/processing/feature_engineering.py` — all 72 features computed here
- `src/reporting/cross_check.py` — the promoted v18 lean model spec (inlined constants)

**Current outputs:**
- `results/monthly_decisions/2026-04/diagnostic.md` — full model health diagnostics
- `results/monthly_decisions/2026-04/signals.csv` — per-benchmark signals (April 2026)
- `results/monthly_decisions/2026-04/recommendation.md` — current recommendation report

**Research history:**
- `docs/history/repo-peer-reviews/2026-04-05/claude_opus_peer_review_20260405.md` — full peer review with research history and prior experiment results

---

## Project Overview

This is a personal RSU vesting decision support system for **Progressive Corporation (PGR)** stock. The owner receives semi-annual RSU vesting events (January and July) and must decide what fraction of each tranche to sell immediately versus hold. The system provides a monthly data-driven recommendation using ML models trained on historical data.

**Business question:** Over the next 6 months, will PGR outperform a diversified benchmark portfolio? If yes → hold more shares; if no → sell more for diversification.

**Stakes:** Purely personal financial decision-making. No external users. The recommendation is informational — final decisions are always made by the human owner. There is no latency requirement (monthly batch).

---

## Current Model Architecture (v11.0, as of April 2026)

**Model type:** Walk-Forward Optimization (WFO) ensemble  
**Ensemble members:** Ridge regression + Gradient Boosted Trees (GBT)  
**Benchmark universe:** 8 ETF benchmarks — VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE  
**Prediction target:** PGR 6-month total return minus ETF 6-month total return (one independent model per benchmark)  
**WFO structure:** 60-month rolling training window, 8-fold CPCV (C(8,2)=28 paths for diagnostics), 8-month total embargo gap (6M horizon + 2M purge buffer)  
**Feature sets:**
- Ridge (12 features): `mom_12m, vol_63d, yield_slope, real_yield_change_6m, real_rate_10y, credit_spread_hy, nfci, vix, combined_ratio_ttm, investment_income_growth_yoy, book_value_per_share_growth_yoy, npw_growth_yoy`
- GBT (13 features): `mom_3m, mom_6m, mom_12m, vol_63d, yield_slope, yield_curvature, vwo_vxus_spread_6m, credit_spread_hy, nfci, vix, rate_adequacy_gap_yoy, pif_growth_yoy, investment_book_yield`

**Ensemble aggregation:** Inverse-variance weighting (1/MAE²) across the two models  
**Signal generation:** One sell% recommendation via Kelly criterion applied to calibrated P(outperform)  
**Calibration:** Platt scaling (logistic regression on OOS scores; activates at n≥100 OOS observations)

---

## Current Performance (April 2026 Run)

| Metric | Value | Pass Threshold | Status |
|--------|-------|----------------|--------|
| Aggregate OOS R² | −13.26% | ≥ 2% | FAIL |
| Mean IC (Newey-West HAC) | 0.19 | ≥ 0.07 | PASS |
| IC p-value | < 0.001 | < 0.05 | PASS |
| Hit Rate | 68.6% | ≥ 55% | PASS |
| CPCV positive paths | 0/7 | ≥ 5/7 | FAIL |
| Obs/feature ratio (per-fold) | ~5.0 | ≥ 4.0 | OK (improved from 1.53) |
| Calibration ECE | 1.8% | < 5% | PASS |
| Conformal interval coverage | 92.8% | ≥ 80% | PASS |

**Prior architecture (4-model, 21 benchmarks):** OOS R² = −118.70%, IC = 0.13, Hit Rate = 56.3%  
**Current architecture improvement:** OOS R² 9× less negative, IC +42%, hit rate +12pp — but still failing the primary OOS R² gate.

**Key data facts:**
- Monthly observations: 316 total (2000-01 to 2026-04), but only ~181 usable for VOO (limited by relative return data availability)
- Fully-populated observations across all 72 features: only **7** (severe missingness in newer/alternative features)
- Relative return data range: varies by benchmark, roughly 2005–2025
- PGR vs. VOO 6M relative return: mean = −0.003, std = 0.266 (very noisy target)
- The signal-to-noise ratio is genuinely low: PGR's relative performance vs. broad equity ETFs is difficult to predict from monthly data

**Top feature correlations with VOO 6M relative return (Ridge v18 features):**

| Feature | Correlation | Economic interpretation |
|---------|-------------|------------------------|
| yield_slope | −0.224 | Flatter curve → worse PGR relative performance |
| npw_growth_yoy | +0.171 | Faster premium growth → relative outperformance |
| investment_income_growth_yoy | +0.154 | Better investment returns → outperformance |
| real_yield_change_6m | −0.138 | Rising real rates → underperformance |
| combined_ratio_ttm | +0.122 | Higher CR → outperformance? (counter-intuitive; may be lagged) |

---

## Full Feature Inventory (72 features available)

Features are grouped by category. Fill rate indicates the fraction of the 316 monthly observations where the feature is non-null.

**Price/momentum (100% fill):** `mom_3m, mom_6m, mom_12m, vol_63d, high_52w`

**PGR fundamentals — valuation (partial fill):** `pe_ratio (67%), pb_ratio (67%), roe (67%), pgr_pe_vs_market_pe (67%), pgr_price_to_book_relative (23%)`

**PGR insurance operations (high fill):** `combined_ratio_ttm, monthly_combined_ratio_delta, pif_growth_yoy, gainshare_est, cr_acceleration, pif_growth_acceleration, channel_mix_agency_pct, npw_growth_yoy, npw_per_pif_yoy, npw_vs_npe_spread_pct`

**PGR underwriting/income (high fill):** `underwriting_income, underwriting_income_3m, underwriting_income_growth_yoy, underwriting_margin_ttm, roe_net_income_ttm`

**PGR balance sheet/capital:** `book_value_per_share_growth_yoy (69%), unearned_premium_growth_yoy, unearned_premium_to_npw_ratio, pgr_premium_to_surplus, direct_channel_pif_share_ttm, channel_mix_direct_pct_yoy, unrealized_gain_pct_equity, buyback_yield (69%), buyback_acceleration (69%)`

**PGR investment portfolio:** `investment_income_growth_yoy, investment_book_yield (42%), duration_rate_shock_3m`

**Macro — rates (69–100% fill):** `yield_slope, yield_curvature, real_rate_10y, real_yield_change_6m, breakeven_inflation_10y, breakeven_momentum_3m, baa10y_spread, credit_spread_hy, term_premium_10y (69%)`

**Macro — conditions (69–100% fill):** `nfci, vix, vmt_yoy, usd_broad_return_3m (68%), usd_momentum_6m (67%), wti_return_3m (68%), mortgage_spread_30y_10y`

**Insurance industry pricing (34–65% fill):** `ppi_auto_ins_yoy, motor_vehicle_ins_cpi_yoy (34%), rate_adequacy_gap_yoy, auto_pricing_power_spread (34%), legal_services_ppi_relative (61%), used_car_cpi_yoy (34%), medical_cpi_yoy (34%), severity_index_yoy (57%), gasoline_retail_sales_delta (65%)`

**Relative/cross-asset signals (55–100%):** `pgr_vs_kie_6m, pgr_vs_peers_6m, pgr_vs_vfh_6m, vwo_vxus_spread_6m (55%), gold_vs_treasury_6m (69%), commodity_equity_momentum (57%), credit_spread_ratio, excess_bond_premium_proxy, equity_risk_premium`

---

## What Has Already Been Tried

This is critical context — please do not recommend approaches already exhausted:

1. **Feature ablation (v7/v8 research, ~2 years of experiments):** Extensive ablation comparing Group A (full kitchen sink), Group B (lean macro), Group C (macro + underwriting), Group D (macro + insurance pricing), Group E (all features). Group B (lean macro) plus selective fundamentals consistently outperformed.

2. **Model selection:** Tested ElasticNet, Ridge, BayesianRidge, GBT, and combinations. The lean Ridge+GBT ensemble on 8 benchmarks outperformed the 4-model stack on 21 benchmarks on IC and hit rate.

3. **Benchmark universe reduction (v20/v21):** Reduced from 21 to 8 core benchmarks — eliminated noisy sector ETFs (VTI, VGT, VHT, VFH, VIS, VDE, VPU, KIE, VEA, VIG, SCHD, BNDX, VCIT, VNQ) that added variance without signal. The current 8 (VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE) were retained.

4. **Feature swap experiments (v15/v16/v18/v20):** Tested replacing specific features in the GBT and Ridge models one at a time. Best confirmed swaps became the v18 lean spec now in production.

5. **Calibration:** Platt scaling active. Isotonic regression awaiting sufficient data (n≥500 OOS observations).

6. **Conformal prediction:** ACI (Adaptive Conformal Inference) implemented for prediction intervals.

7. **Black-Litterman portfolio overlay:** Implemented as a shadow layer for the redeploy-portfolio recommendation; not used for the primary vesting sell% decision.

8. **Shadow baseline (historical mean):** Tested "just predict the historical mean relative return" as the v13 baseline. This had negative IC (−0.31 for VOO) and was retired.

---

## Core Constraints & Requirements

1. **Data scarcity is the fundamental problem.** With 60-month rolling training windows and an 8-month embargo, early WFO folds may have as few as 30 training observations. New data arrives at one observation per month.

2. **No lookahead allowed.** All features must be computable from data available on or before the as-of date. The 6-month return target is realized 6 months later — this is why the relative return dataset only runs through September 2025 as of April 2026.

3. **Monthly cadence only.** Data is monthly (end-of-month). No intra-month signals. No daily features unless aggregated to monthly.

4. **Infrastructure is Python + SQLite + GitHub Actions.** No GPU. No external APIs beyond Alpha Vantage and FRED (both free-tier). Must run in <15 minutes on a standard GitHub Actions runner.

5. **Interpretability matters.** The owner needs to trust and understand the recommendation. Black-box approaches that can't be sanity-checked are undesirable.

6. **The target is noisy by construction.** PGR's 6-month relative return vs. VOO has std=0.266. Predicting 26.6% annual volatility relative returns from 60 monthly observations is inherently difficult — the research question is whether any signal exists at all, and if so, how to extract it more reliably.

---

## Research Questions — Please Address All of the Following

### 1. Alternative Target Variable Formulations

The current target is the raw 6-month relative return (PGR minus ETF). Is there a better-behaved target?
- Should we predict **directional sign** (binary classification) rather than magnitude?
- Should we predict **cumulative excess return** vs. a composite benchmark rather than per-ETF?
- Would a **12-month horizon** improve signal-to-noise despite requiring 12 more months of embargo?
- Could **rank-based targets** (e.g., percentile rank of PGR return among all 8 benchmarks) reduce the influence of outlier return months?
- What does the ML literature say about optimal target construction for financial relative return prediction with small datasets?

### 2. Feature Engineering Improvements

Given the 72 features available (many with partial fill), what are the highest-leverage improvements?
- **Imputation strategies:** The 7 fully-populated observations vs. 316 total reveals extreme missingness in newer features. Are there principled imputation approaches (forward-fill, EM, multiple imputation, matrix factorization) that preserve the time-series structure and don't introduce lookahead bias?
- **Feature transformations:** Should raw features be transformed (log, rank-normalize, standardize within rolling window) before entering the model? Which transformations are most appropriate for financial time series?
- **Lagged features:** Should features be lagged 1–3 months to create a buffer against stale fundamental data?
- **Interaction features:** Given the small sample, are interaction terms ever worthwhile, or do they exacerbate overfitting?
- **Dimensionality reduction:** PCA or factor models applied to the 72 features — does reducing to 3–5 orthogonal factors improve OOS performance in small-N settings?
- **Domain-specific features:** What insurance-industry metrics are most theoretically predictive of relative stock performance that may be missing? Consider: reserve development, catastrophe loss exposure, rate adequacy cycle position, competitive dynamics.

### 3. Model Architecture

- **Bayesian approaches:** Given the small dataset, would a fully Bayesian model (e.g., Bayesian linear regression with informative priors, Gaussian Process regression) outperform frequentist regularization? What priors would be appropriate for insurance-stock relative returns?
- **Shrinkage toward the mean:** Should the ensemble prediction be shrunk toward zero (the null hypothesis of no predictability) based on the current OOS R² being negative? What is the James-Stein estimator analog here?
- **Panel models:** We have 8 benchmark-specific models trained independently. Could a hierarchical/panel model that shares statistical strength across benchmarks improve performance?
- **Time-varying parameters:** Is there evidence that the predictive relationship changes over insurance market cycles (hard vs. soft market)? Should coefficients be allowed to change?
- **Regime-conditional models:** Train separate models for "hard market" vs. "soft market" regimes (using combined ratio threshold or industry pricing signals). Would regime-conditioned models have better within-regime fit?
- **Extremely small N approaches:** What ML methods are specifically designed for tabular regression with n≈30–60 training observations? (e.g., regularized regression with cross-validated λ, Lasso with stability selection, random forests with very deep subsampling, Gaussian processes)

### 4. Walk-Forward Optimization Design

- **Training window length:** Is 60 months optimal? The literature on financial WFO suggests shorter windows may adapt faster to structural breaks while longer windows provide more stable estimates. What is the optimal window for a monthly relative-return model with ~15 features?
- **Expanding vs. rolling window:** Should the model use an expanding window from a fixed start date (more data, but older data may be less relevant) or the current rolling window?
- **Embargo gap:** Is 8 months (6M horizon + 2M purge buffer) sufficient to prevent information leakage with 6-month overlapping return windows? The Newey-West correction uses 5 lags — does this adequately address the autocorrelation?
- **WFO fold structure:** Given that CPCV consistently fails (0/7 positive paths), is the current fold structure appropriate? Are there alternative validation schemes (walk-forward with growing window, time-series split with gap) that better match the data generating process?

### 5. The OOS R² Failure Despite Positive IC

This is the central puzzle: IC is positive (0.19, statistically significant) and hit rate is 69%, yet OOS R² is −13%. This suggests the model has directional skill but poor magnitude calibration. Specifically:
- What mechanisms cause positive IC + high hit rate to coexist with negative OOS R²?
- Is this a known phenomenon in financial ML? What is the standard explanation?
- What are the standard fixes? (e.g., targeting rank correlation directly via Spearman loss, monotone transformations of predictions, trimming extreme predictions)
- Should we abandon OOS R² as the primary gate and rely on IC + hit rate instead? What does the academic literature recommend for evaluating financial return predictions?
- Could the negative OOS R² be purely a calibration artifact (models predict the right direction but wrong magnitude) that would be fixed by isotonic regression once n≥500?

### 6. Data Augmentation & Alternative Data Sources

- **Synthetic data:** Are there principled ways to augment a 180-observation monthly financial time series (e.g., block bootstrap, stationary bootstrap, synthetic minority oversampling for financial time series)?
- **Alternative FRED series:** What additional FRED macroeconomic series might predict insurance stock relative performance that aren't currently in the feature set? (Consider: construction spending, used car prices, medical costs, weather/catastrophe proxies, casualty insurance pricing indices)
- **Industry data:** What publicly available insurance industry data (AM Best, NAIC, Insurance Information Institute) could improve prediction? Is any of it freely downloadable?
- **Transfer learning from peers:** Could training a model on insurance sector peer returns (ALL, TRV, CB, HIG) and using the model weights as a prior for PGR improve sample efficiency?

### 7. Practical Recommendations

Given all of the above, please provide:
- A **ranked list of the 3–5 highest-leverage interventions** most likely to improve OOS R² from −13% toward the +2% threshold, given the constraint of ~180 monthly observations
- For each intervention: estimated implementation complexity (S/M/L), expected impact on OOS R², key risks/downsides
- A suggested **experimental sequence** — what to try first, how to validate it without incurring multiple-comparison bias, and what constitutes sufficient evidence to promote a new specification to production
- Your honest assessment of whether **+2% OOS R² is achievable** with the current data volume, and if not, what the realistic ceiling is and why

---

## What a Successful Research Output Looks Like

The goal is not a literature survey — it is a concrete, prioritized action plan. Please:
- Be specific about which existing features to modify, which to add, which to drop
- Reference specific ML methods with enough detail to implement them in scikit-learn or statsmodels
- Acknowledge the fundamental tension between model complexity and sample size throughout
- Flag any recommendations that would require data sources not currently available (so the owner can decide whether to pursue them)
- Do not recommend neural networks, LLMs, or GPU-dependent methods — the infrastructure is CPU-only GitHub Actions runners

---

## Appendix: Key File Descriptions (if attaching)

| File | Purpose |
|------|---------|
| `config/features.py` | Canonical list of all 72 features by name; `MODEL_FEATURE_OVERRIDES` shows the v18 lean specs for Ridge and GBT; `PRIMARY_FORECAST_UNIVERSE` shows the 8 active benchmarks |
| `config/model.py` | WFO window (60M), embargo (8M), CPCV folds (8), ensemble model list |
| `src/processing/feature_engineering.py` | Full feature computation logic — how each of the 72 features is calculated from raw data |
| `src/models/wfo_engine.py` | The WFO training loop, fold construction, embargo enforcement |
| `src/models/multi_benchmark_wfo.py` | Outer loop over 8 benchmarks; inverse-variance ensemble aggregation |
| `src/reporting/cross_check.py` | The v18 lean model specs (feature lists per model type) with notes on what each swap replaced |
| `results/monthly_decisions/2026-04/diagnostic.md` | Current model health report with all diagnostic metrics |
| `results/monthly_decisions/2026-04/signals.csv` | Per-benchmark predictions, IC, hit rate, confidence tiers |
| `docs/history/repo-peer-reviews/2026-04-05/claude_opus_peer_review_20260405.md` | Full peer review — includes description of all prior research cycles (v7–v22) and what was tried |
