# PGR Monthly Decision Report — November 2026

**As-of Date:** 2026-11-20  
**Run Date:** 2026-11-20  
**Model Version:** v11.1 (lean 2-model ensemble: Ridge + GBT, v18 feature sets, 8-benchmark PRIMARY_FORECAST_UNIVERSE, inverse-variance weighting, prequential post-ensemble shrinkage and realised-only health metrics (review 2026-09-25 WP7); walk-forward validation only, with explicit completion and input-readiness gates (R3); ElasticNet+BayesianRidge retired after v18/v20 research showed Ridge+GBT outperforms on IC, hit rate, and obs/feature ratio)  
**Recommendation Layer:** Live production recommendation layer (quality-weighted consensus)  

---

## Executive Summary

- What changed since last month: Previous logged month (2026-10-20) was NEUTRAL at +0.16% with mean IC 0.0875.
- Current model view: Consensus signal is NEUTRAL, but the average relative-return forecast is +0.99% across benchmarks over the next 6 months. Recommendation mode remains DEFER-TO-TAX-DEFAULT.
- How trustworthy it is: Model quality is too weak to justify a prediction-led vesting action. Deferral reasons: directional_skill FAIL: p=0.462; hit 62.3% vs base rate 68.3% (needs PT p < 0.05). Aggregate health: OOS R^2 3.52%, IC 0.1111, hit rate 62.3%.
- What to do at the next vest: Next vest guidance unavailable because the lot file or latest PGR price is missing.
- What would change the recommendation: A more aggressive recommendation would require aggregate OOS R^2 >= 2% against the prevailing-mean benchmark, equal-weight mean IC >= 0.07, significant directional skill (Pesaran-Timmermann p < 0.05), completed walk-forward validation for every required benchmark and model, and fresh, finite required inputs.

---

## Data Freshness

> All monitored feeds are within freshness thresholds for this run.

| Feed | Latest Date | Age | Limit | Status |
|------|-------------|-----|-------|--------|
| Prices PGR | 2026-11-13 | 7 days | 10 days | **OK** |
| Prices VOO | 2026-11-13 | 7 days | 10 days | **OK** |
| Prices VXUS | 2026-11-13 | 7 days | 10 days | **OK** |
| Prices VWO | 2026-11-13 | 7 days | 10 days | **OK** |
| Prices VMBS | 2026-11-13 | 7 days | 10 days | **OK** |
| Prices BND | 2026-11-13 | 7 days | 10 days | **OK** |
| Prices GLD | 2026-11-13 | 7 days | 10 days | **OK** |
| Prices DBC | 2026-11-13 | 7 days | 10 days | **OK** |
| Prices VDE | 2026-11-13 | 7 days | 10 days | **OK** |
| FRED T10Y2Y | 2026-11 | 0 mo behind | lag 1 mo; needs 2026-09 | **OK** |
| FRED GS10 | 2026-10 | 0 mo behind | lag 1 mo; needs 2026-09 | **OK** |
| FRED T10YIE | 2026-11 | 0 mo behind | lag 1 mo; needs 2026-09 | **OK** |
| FRED BAMLH0A0HYM2 | 2026-11 | 0 mo behind | lag 1 mo; needs 2026-09 | **OK** |
| FRED NFCI | 2026-11 | 0 mo behind | lag 2 mo; needs 2026-08 | **OK** |
| FRED VIXCLS | 2026-11 | 0 mo behind | lag 1 mo; needs 2026-09 | **OK** |
| FRED GS5 | 2026-10 | 0 mo behind | lag 1 mo; needs 2026-09 | **OK** |
| FRED GS2 | 2026-10 | 0 mo behind | lag 1 mo; needs 2026-09 | **OK** |
| FRED PCU5241265241261 | 2026-10 | 0 mo behind | lag 1 mo; needs 2026-09 | **OK** |
| FRED CUSR0000SETA02 | 2026-10 | 0 mo behind | lag 1 mo; needs 2026-09 | **OK** |
| FRED CUSR0000SAM2 | 2026-10 | 0 mo behind | lag 1 mo; needs 2026-09 | **OK** |
| PGR monthly EDGAR | 2026-10-31 | 20 days | 25-day filing grace | **OK** |
| Dividends PGR | 2026-10-01 | interval 91 d | due by 2026-06-30 | **OK** |
| Dividends VOO | 2026-09-28 | interval 92 d | due by 2026-06-28 | **OK** |
| Dividends VXUS | 2026-09-18 | interval 91 d | due by 2026-06-30 | **OK** |
| Dividends VWO | 2026-09-18 | interval 91 d | due by 2026-06-30 | **OK** |
| Dividends VMBS | 2026-11-02 | interval 30 d | due by 2026-09-29 | **OK** |
| Dividends BND | 2026-11-02 | interval 30 d | due by 2026-09-29 | **OK** |
| Dividends GLD | none (audited non-payer) | n/a | history required | **OK** |
| Dividends DBC | 2025-12-22 | interval 364 d | due by 2025-05-16 | **OK** |
| Dividends VDE | 2026-09-23 | interval 92 d | due by 2026-06-29 | **OK** |

---

## Decision At A Glance

- Hold vs Sell: **Hold 50% / Sell 50% of the next vest tranche**
- Is this month actionable? **No — follow the default tax/diversification rule.**
- Top-line decision: **Hold 50% / Sell 50% of the next vest tranche. No — follow the default tax/diversification rule.**
- Shadow classifier probability: **49.2%** (LOW)
- **Portfolio-aligned P(Actionable Sell):** 59.6% [NEUTRAL] _(investable pool, fixed weights)_
- **Path B P(Actionable Sell):** 75.6% [ACTIONABLE-SELL] _(composite portfolio target, temp-scaled)_

## Agreement Panel

- Live recommendation: **DEFER-TO-TAX-DEFAULT / sell 50%**
- Consensus cross-check: **Aligned**
- Classifier shadow: **Aligned**
- Shadow gate overlay: **DEFER-TO-TAX-DEFAULT / sell 50%** (no live change)

---

## Consensus Signal

| Field | Value |
|-------|-------|
| Signal | **NEUTRAL (LOW CONFIDENCE)** |
| Recommendation Mode | **DEFER-TO-TAX-DEFAULT** |
| Recommended Sell % | **50%** |
| Predicted 6M Relative Return | +0.99% |
| P(Outperform, calibrated) | 67.1% |
| 80% Prediction Interval (median) | -27.37% to +30.25% |
| Mean IC (equal-weight across benchmarks) | 0.0827 |
| Mean Hit Rate (equal-weight) | 62.2% |
| Directional skill (hit rate vs base rate) | 62.3% vs 68.3%; Pesaran–Timmermann p = 0.462 |
| Aggregate OOS R^2 (vs prevailing mean) | 3.52% |

> **Note:** The sell % recommendation is used only at actual vesting events
> (January and July).  Monthly reports are monitoring tools, not trade signals.
>
> **Shrinkage:** the ensemble forecast is scaled by α = 0.50, the v38 grid value with the lowest squared error over the realised OOS record, re-chosen every month. Before review 2026-09-25 it was fixed at 0.50, a value chosen on the same history it was then scored on.
>
> **Calibration:** Phase 2 — Platt scaling per benchmark.  Prequential ECE = 19.5% [95% CI: 13.3%–26.5%] over 1,038 OOS benchmark-months, each scored by a calibrator fitted only on outcomes known at the time.

---

## Classification Confidence Check

> Shadow-only interpretation layer from the v87-v96 classifier research.
> It does not change the live recommendation or sell percentage.

| Field | Value |
|-------|-------|
| Target | actionable_sell_3pct |
| Construction | Separate benchmark logistic + quality-weighted aggregate |
| P(Actionable Sell) | 49.2% |
| Confidence Tier | LOW |
| Classifier Stance | NEUTRAL |
| Portfolio-aligned P(Actionable Sell) | 59.6% [NEUTRAL] |
| Investable Pool Confidence Tier | LOW |
| Path B P(Actionable Sell) | 75.6% [ACTIONABLE-SELL] |
| Path B Confidence Tier | HIGH |
| Agreement with Live Recommendation | Aligned |
| Interpretation | Shadow classifier is near its neutral band (49.2%); use it as a low-confidence interpretation layer rather than a decision override. |

---

## Confidence Snapshot

- 4/5 core gates pass and 1 fail, so the quality gate is too weak for a prediction-led vest action.

| Check | Current | Threshold | Status | Meaning |
|-------|---------|-----------|--------|---------|
| Aggregate OOS R^2 | 3.52% | >= 2.00% | **PASS** | Beats the prevailing mean of the targets realised by each forecast date. |
| Mean IC (equal-weight) | 0.0827 | >= 0.0700 | **PASS** | Equal-weight mean of the per-benchmark OOS rank ICs. |
| Directional skill | p=0.462; hit 62.3% vs base rate 68.3% | PT p < 0.05 | **FAIL** | Up/down calls beat chance given the base rate (Pesaran-Timmermann, clustered by date). |
| Walk-forward validation complete | complete (16 required model/benchmark pairs) | every required pair complete | **PASS** | Walk-forward (TimeSeriesSplit, purge and embargo) results exist for every required model and benchmark, with finite OOS and live forecasts and realised outcomes. |
| Required inputs ready | ready | all required inputs finite and fresh at the as-of date | **PASS** | Every live model feature is finite before imputation and the required price, FRED, EDGAR and dividend feeds are fresh on the decision's as-of date. |
| Mean hit rate | 62.2% | not gated | **INFO** | Directional accuracy versus zero; compare with the base rate above. |

---

## Model Health

- Latest tracked month: **2026-11-30**
- Rolling 12M IC: **0.1138**
- Rolling 12M Hit Rate: **62.7%**
- Rolling 12M ECE: **17.6%**
- IC breach streak: **0** month(s)
- Status: **Stable: no sustained rolling-IC drift alert is active.**

---

## Decision Policy Backtest

> OOS performance of each decision policy applied to all historical model predictions.  "Mean Return" is the portfolio-weighted realized relative return per vesting event.  "Cumulative" is the sum across all events.  "Capture Ratio" is the fraction of oracle (always hold when positive) gains captured.  N = number of OOS events.  The live ACTIONABLE mapping is the policy used when every gate passes; it is replayed on the realised-only record and scored once per date (the others once per benchmark and date).

### Fixed Heuristic Baselines

| Policy | N | Mean Return | Cumulative | Capture Ratio |
|--------|---|-------------|------------|---------------|
| Sell 100% (always) | 1248 | +0.00% | +0.00% | 0.0% |
| Sell 50% (always) | 1248 | +3.77% | +4699.33% | 32.0% |
| Hold 100% (always) | 1248 | +7.53% | +9398.67% | 64.1% |

### Model-Driven Policies vs. Heuristics

| Policy | N | Mean Return | Cumul. Return | Uplift vs Sell-All | Uplift vs Hold-All | Uplift vs 50% | Capture |
|--------|---|-------------|---------------|--------------------|--------------------|---------------|---------|
| Model: sign (hold if pred > 0) | 1248 | +6.51% | +8129.29% | +6.51% | -1.02% | +2.75% | 55.4% |
| Model: tiered 25/50/100 | 1248 | +2.01% | +2502.75% | +2.01% | -5.53% | -1.76% | 17.1% |
| Model: neutral band ±2% | 1248 | +6.22% | +7761.09% | +6.22% | -1.31% | +2.45% | 52.9% |
| Model: neutral band ±3% | 1248 | +6.03% | +7519.80% | +6.03% | -1.51% | +2.26% | 51.3% |
| **Live ACTIONABLE mapping** (consensus, per date) | 186 | +3.60% | +669.44% | +3.60% | -3.94% | -0.17% | 33.4% |


---

## Portfolio Optimizer Status

> ⚠️ **Optimizer fallback active** — Black-Litterman optimization could not converge (`optimization_failure`).  Portfolio weights fall back to equal-weight allocation.  This does not affect the primary recommendation; it is a diagnostic indicator.

| Parameter | Value |
|-----------|-------|
| Optimizer | Black-Litterman (PyPortfolioOpt / Ledoit-Wolf) |
| Status | ⚠️ Fallback — optimization_failure |
| Active benchmarks | 8 |
| View tickers incorporated | 8 |


---

## Interpretation

The point forecast leans neutral, and 3/8 (38%) benchmarks favour outperformance, but the broader quality gate is failing.

Recommended action at next vesting event: **DEFAULT 50% SALE** for diversification and tax discipline, not because the prediction is high-confidence.

---

## Redeploy Guidance

- Broad US Equity: VOO. Broad US equity diversifies away from single-stock risk without concentrating further in insurance.
- International Equity: VXUS, VWO. International equity lowers home-market and insurance concentration.
- Fixed Income: BND. Fixed income is the cleanest concentration-reduction bucket when model confidence is weak.
- Sector Context: VGT, SCHD. Sector funds are context-only unless no stronger diversifying destination is available.

## Suggested Redeploy Portfolio

- Default posture: `95%` equities / `5%` bonds across the curated investable universe.
- Monthly tilts use a `25%` signal overlay around the base weights, so the recommendation can adapt without becoming a full tactical allocation model.
- Investable universe used in the monthly workflow: `VOO, VGT, SCHD, VXUS, VWO, BND`.
- Constraint note: The current project universe does not yet include a dedicated small-cap ETF, so the value sleeve uses SCHD and the broad-market sleeve stays in VOO.

| Fund | Allocation | Sleeve | Why it is included | PGR Correlation | Relative Signal | P(Benchmark Beats PGR) |
|------|------------|--------|--------------------|-----------------|-----------------|------------------------|
| VOO | 40% | Broad US equity core | Core US beta sleeve that keeps the portfolio equity-heavy without recreating single-stock PGR risk. | 0.34 | Keep near base (+0.5%) | 36.7% |
| VGT | 20% | Technology tilt | Growth engine and explicit tech tilt when the relative signal supports owning more innovation exposure than a pure core index. | 0.30 | Base-weight only (n/a) | n/a |
| SCHD | 15% | Value / dividend tilt | Closest current project proxy for a value sleeve; adds a cheaper, income-oriented counterweight to the tech allocation. | 0.39 | Base-weight only (n/a) | n/a |
| VXUS | 10% | International core | Primary geographic diversifier away from a US employer-stock concentration. | 0.25 | Only keep at floor weight (+3.8%) | 31.8% |
| VWO | 10% | Emerging-markets satellite | Higher-growth international sleeve kept modest because it is more volatile than the core international allocation. | 0.26 | Keep near base (+1.7%) | 30.9% |
| BND | 5% | Bond ballast | Small stabilizer sleeve kept intentionally light so the redeploy portfolio stays above 90% equities in normal months. | 0.08 | Only keep at floor weight (+4.8%) | 24.6% |

## Per-Benchmark Signals

- Predicted Return is from the perspective of PGR versus each fund. Positive means PGR is expected to outperform that fund; negative means the fund is expected to outperform PGR.
- Benchmark Role distinguishes realistic buy candidates from contextual or forecast-only comparison funds.

| Benchmark | Benchmark Role | Description | Predicted Return | CI Lower | CI Upper | IC | Hit Rate | P(cal) | Confidence | Signal |
|-----------|----------------|-------------|----------------|----------|----------|----|----------|--------|------------|--------|
| VOO | Buy candidate | S&P 500 | +0.49% | -28.07% | +29.04% | 0.1549 | 60.0% | 63.3% | LOW | NEUTRAL |
| VXUS | Buy candidate | Total International Stock | +3.84% | -26.66% | +34.33% | 0.0888 | 63.6% | 68.2% | MODERATE | OUTPERFORM |
| VWO | Buy candidate | Emerging Markets | +1.74% | -26.44% | +29.92% | 0.0975 | 60.2% | 69.1% | MODERATE | OUTPERFORM |
| VMBS | Forecast only | Mortgage-Backed Securities | +3.71% | -23.08% | +30.51% | 0.0315 | 69.4% | 73.8% | LOW | NEUTRAL |
| BND | Buy candidate | Total Bond Market | +4.82% | -21.37% | +31.01% | 0.1133 | 66.7% | 75.4% | HIGH | OUTPERFORM |
| GLD | Forecast only | Gold Shares | +0.70% | -34.57% | +35.98% | 0.0734 | 61.3% | 67.0% | LOW | NEUTRAL |
| DBC | Forecast only | DB Commodity Index | -2.09% | -34.17% | +29.99% | 0.0997 | 62.9% | 55.6% | LOW | UNDERPERFORM |
| VDE | Forecast only | Energy | -5.13% | -37.27% | +27.02% | 0.0026 | 53.8% | 64.5% | LOW | NEUTRAL |

---

## Tax Context

| Parameter | Value |
|-----------|-------|
| STCG Rate (federal) | 37% |
| LTCG Rate (federal) | 20% |
| Tax-rate differential | 17% |
| **LTCG breakeven PGR return (lot all gain)** | **-21.25%** |
| LTCG breakeven PGR return (gain = half the price) | -10.62% |
| Model forecast, PGR vs benchmarks (6M, relative) | +0.99% |
| P(outperform) | 67.1% |
| Next time-based vest | 2027-01-19 |
| Next performance vest | 2027-07-17 |

**Holding a vested lot to its LTCG date gives more after-tax cash than selling it now at STCG unless PGR's own price falls by more than 21.25% (a lot that is all gain) or 10.62% (a lot whose gain is half its price) before that date.** At the vest itself the shares carry no gain, so the tax rate does not change the proceeds of an immediate sale.

The model expects PGR to beat the benchmarks by 1.0% over 6 months. That is a relative forecast, not a PGR price forecast, so it is not compared with the breakeven above.

> **Breakeven formula:** `-g × (STCG − LTCG) / (1 − LTCG)`, where `g` is the
> lot's unrealised gain as a fraction of the current price. Below this absolute
> PGR return, selling now at STCG beats holding to the LTCG date (the day after
> the one-year anniversary of the vest); above it, holding wins. It compares
> cash at the LTCG date and ignores what sale proceeds would earn meanwhile.

---

*Generated by `scripts/monthly_decision.py`*