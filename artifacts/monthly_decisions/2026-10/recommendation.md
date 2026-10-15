# PGR Monthly Decision Report — October 2026

**As-of Date:** 2026-10-15  
**Run Date:** 2026-10-15  
**Model Version:** v11.1 (lean 2-model ensemble: Ridge + GBT, v18 feature sets, 8-benchmark PRIMARY_FORECAST_UNIVERSE, inverse-variance weighting, prequential post-ensemble shrinkage and realised-only health metrics (review 2026-09-25 WP7); walk-forward validation only, with explicit completion and input-readiness gates (R3); ElasticNet+BayesianRidge retired after v18/v20 research showed Ridge+GBT outperforms on IC, hit rate, and obs/feature ratio)  
**Recommendation Layer:** Live production recommendation layer (quality-weighted consensus)  

---

## Executive Summary

- What changed since last month: Previous logged month (2026-09-21) was NEUTRAL at -3.71% with mean IC 0.1133.
- Current model view: Consensus signal is NEUTRAL, but the average relative-return forecast is +0.18% across benchmarks over the next 6 months. Recommendation mode remains DEFER-TO-TAX-DEFAULT.
- How trustworthy it is: Model quality is too weak to justify a prediction-led vesting action. Deferral reasons: directional_skill FAIL: p=0.311; hit 63.3% vs base rate 68.1% (needs PT p < 0.05). Aggregate health: OOS R^2 4.94%, IC 0.1165, hit rate 63.3%.
- What to do at the next vest: Next vest guidance unavailable because the lot file or latest PGR price is missing.
- What would change the recommendation: A more aggressive recommendation would require aggregate OOS R^2 >= 2% against the prevailing-mean benchmark, equal-weight mean IC >= 0.07, significant directional skill (Pesaran-Timmermann p < 0.05), completed walk-forward validation for every required benchmark and model, and fresh, finite required inputs.

---

## Data Freshness

> All monitored feeds are within freshness thresholds for this run.

| Feed | Latest Date | Age | Limit | Status |
|------|-------------|-----|-------|--------|
| Prices PGR | 2026-10-14 | 1 days | 10 days | **OK** |
| Prices VOO | 2026-10-14 | 1 days | 10 days | **OK** |
| Prices VXUS | 2026-10-14 | 1 days | 10 days | **OK** |
| Prices VWO | 2026-10-14 | 1 days | 10 days | **OK** |
| Prices VMBS | 2026-10-14 | 1 days | 10 days | **OK** |
| Prices BND | 2026-10-14 | 1 days | 10 days | **OK** |
| Prices GLD | 2026-10-14 | 1 days | 10 days | **OK** |
| Prices DBC | 2026-10-14 | 1 days | 10 days | **OK** |
| Prices VDE | 2026-10-14 | 1 days | 10 days | **OK** |
| FRED T10Y2Y | 2026-10 | 0 mo behind | lag 1 mo; needs 2026-08 | **OK** |
| FRED GS10 | 2026-09 | 0 mo behind | lag 1 mo; needs 2026-08 | **OK** |
| FRED T10YIE | 2026-10 | 0 mo behind | lag 1 mo; needs 2026-08 | **OK** |
| FRED BAMLH0A0HYM2 | 2026-10 | 0 mo behind | lag 1 mo; needs 2026-08 | **OK** |
| FRED NFCI | 2026-10 | 0 mo behind | lag 2 mo; needs 2026-07 | **OK** |
| FRED VIXCLS | 2026-10 | 0 mo behind | lag 1 mo; needs 2026-08 | **OK** |
| FRED GS5 | 2026-09 | 0 mo behind | lag 1 mo; needs 2026-08 | **OK** |
| FRED GS2 | 2026-09 | 0 mo behind | lag 1 mo; needs 2026-08 | **OK** |
| FRED PCU5241265241261 | 2026-08 | 0 mo behind | lag 1 mo; needs 2026-08 | **OK** |
| FRED CUSR0000SETA02 | 2026-09 | 0 mo behind | lag 1 mo; needs 2026-08 | **OK** |
| FRED CUSR0000SAM2 | 2026-09 | 0 mo behind | lag 1 mo; needs 2026-08 | **OK** |
| PGR monthly EDGAR | 2026-09-30 | 15 days | 25-day filing grace | **OK** |
| Dividends PGR | 2026-10-01 | interval 91 d | due by 2026-05-31 | **OK** |
| Dividends VOO | 2026-09-28 | interval 92 d | due by 2026-05-29 | **OK** |
| Dividends VXUS | 2026-09-18 | interval 91 d | due by 2026-05-31 | **OK** |
| Dividends VWO | 2026-09-18 | interval 91 d | due by 2026-05-31 | **OK** |
| Dividends VMBS | 2026-10-01 | interval 30 d | due by 2026-08-30 | **OK** |
| Dividends BND | 2026-10-01 | interval 30 d | due by 2026-08-30 | **OK** |
| Dividends GLD | none (audited non-payer) | n/a | history required | **OK** |
| Dividends DBC | 2025-12-22 | interval 364 d | due by 2025-04-16 | **OK** |
| Dividends VDE | 2026-09-23 | interval 92 d | due by 2026-05-30 | **OK** |

---

## Decision At A Glance

- Hold vs Sell: **Hold 50% / Sell 50% of the next vest tranche**
- Is this month actionable? **No — follow the default tax/diversification rule.**
- Top-line decision: **Hold 50% / Sell 50% of the next vest tranche. No — follow the default tax/diversification rule.**
- Shadow classifier probability: **45.2%** (LOW)
- **Portfolio-aligned P(Actionable Sell):** 56.7% [NEUTRAL] _(investable pool, fixed weights)_
- **Path B P(Actionable Sell):** 70.5% [ACTIONABLE-SELL] _(composite portfolio target, temp-scaled)_

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
| Predicted 6M Relative Return | +0.18% |
| P(Outperform, calibrated) | 65.6% |
| 80% Prediction Interval (median) | -31.06% to +29.75% |
| Mean IC (equal-weight across benchmarks) | 0.0873 |
| Mean Hit Rate (equal-weight) | 64.3% |
| Directional skill (hit rate vs base rate) | 63.3% vs 68.1%; Pesaran–Timmermann p = 0.311 |
| Aggregate OOS R^2 (vs prevailing mean) | 4.94% |

> **Note:** The sell % recommendation is used only at actual vesting events
> (January and July).  Monthly reports are monitoring tools, not trade signals.
>
> **Shrinkage:** the ensemble forecast is scaled by α = 0.50, the v38 grid value with the lowest squared error over the realised OOS record, re-chosen every month. Before review 2026-09-25 it was fixed at 0.50, a value chosen on the same history it was then scored on.
>
> **Calibration:** Phase 2 — Platt scaling per benchmark.  Prequential ECE = 15.9% [95% CI: 10.7%–24.9%] over 1,025 OOS benchmark-months, each scored by a calibrator fitted only on outcomes known at the time.

---

## Classification Confidence Check

> Shadow-only interpretation layer from the v87-v96 classifier research.
> It does not change the live recommendation or sell percentage.

| Field | Value |
|-------|-------|
| Target | actionable_sell_3pct |
| Construction | Separate benchmark logistic + quality-weighted aggregate |
| P(Actionable Sell) | 45.2% |
| Confidence Tier | LOW |
| Classifier Stance | NEUTRAL |
| Portfolio-aligned P(Actionable Sell) | 56.7% [NEUTRAL] |
| Investable Pool Confidence Tier | LOW |
| Path B P(Actionable Sell) | 70.5% [ACTIONABLE-SELL] |
| Path B Confidence Tier | HIGH |
| Agreement with Live Recommendation | Aligned |
| Interpretation | Shadow classifier is near its neutral band (45.2%); use it as a low-confidence interpretation layer rather than a decision override. |

---

## Confidence Snapshot

- 4/5 core gates pass and 1 fail, so the quality gate is too weak for a prediction-led vest action.

| Check | Current | Threshold | Status | Meaning |
|-------|---------|-----------|--------|---------|
| Aggregate OOS R^2 | 4.94% | >= 2.00% | **PASS** | Beats the prevailing mean of the targets realised by each forecast date. |
| Mean IC (equal-weight) | 0.0873 | >= 0.0700 | **PASS** | Equal-weight mean of the per-benchmark OOS rank ICs. |
| Directional skill | p=0.311; hit 63.3% vs base rate 68.1% | PT p < 0.05 | **FAIL** | Up/down calls beat chance given the base rate (Pesaran-Timmermann, clustered by date). |
| Walk-forward validation complete | complete (16 required model/benchmark pairs) | every required pair complete | **PASS** | Walk-forward (TimeSeriesSplit, purge and embargo) results exist for every required model and benchmark, with finite OOS and live forecasts and realised outcomes. |
| Required inputs ready | ready | all required inputs finite and fresh at the as-of date | **PASS** | Every live model feature is finite before imputation and the required price, FRED, EDGAR and dividend feeds are fresh on the decision's as-of date. |
| Mean hit rate | 64.3% | not gated | **INFO** | Directional accuracy versus zero; compare with the base rate above. |

---

## Model Health

- Latest tracked month: **2026-10-31**
- Rolling 12M IC: **0.1165**
- Rolling 12M Hit Rate: **63.3%**
- Rolling 12M ECE: **15.9%**
- IC breach streak: **0** month(s)
- Status: **Stable: no sustained rolling-IC drift alert is active.**

---

## Decision Policy Backtest

> OOS performance of each decision policy applied to all historical model predictions.  "Mean Return" is the portfolio-weighted realized relative return per vesting event.  "Cumulative" is the sum across all events.  "Capture Ratio" is the fraction of oracle (always hold when positive) gains captured.  N = number of OOS events.  The live ACTIONABLE mapping is the policy used when every gate passes; it is replayed on the realised-only record and scored once per date (the others once per benchmark and date).

### Fixed Heuristic Baselines

| Policy | N | Mean Return | Cumulative | Capture Ratio |
|--------|---|-------------|------------|---------------|
| Sell 100% (always) | 1236 | +0.00% | +0.00% | 0.0% |
| Sell 50% (always) | 1236 | +3.74% | +4623.33% | 31.8% |
| Hold 100% (always) | 1236 | +7.48% | +9246.67% | 63.7% |

### Model-Driven Policies vs. Heuristics

| Policy | N | Mean Return | Cumul. Return | Uplift vs Sell-All | Uplift vs Hold-All | Uplift vs 50% | Capture |
|--------|---|-------------|---------------|--------------------|--------------------|---------------|---------|
| Model: sign (hold if pred > 0) | 1236 | +6.76% | +8352.54% | +6.76% | -0.72% | +3.02% | 57.5% |
| Model: tiered 25/50/100 | 1236 | +2.33% | +2878.11% | +2.33% | -5.15% | -1.41% | 19.8% |
| Model: neutral band ±2% | 1236 | +6.37% | +7878.44% | +6.37% | -1.11% | +2.63% | 54.2% |
| Model: neutral band ±3% | 1236 | +6.12% | +7567.70% | +6.12% | -1.36% | +2.38% | 52.1% |
| **Live ACTIONABLE mapping** (consensus, per date) | 186 | +3.71% | +690.66% | +3.71% | -3.59% | +0.06% | 34.6% |


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

- Default posture: `96%` equities / `5%` bonds across the curated investable universe.
- Monthly tilts use a `25%` signal overlay around the base weights, so the recommendation can adapt without becoming a full tactical allocation model.
- Investable universe used in the monthly workflow: `VOO, VGT, SCHD, VXUS, VWO, BND`.
- Constraint note: The current project universe does not yet include a dedicated small-cap ETF, so the value sleeve uses SCHD and the broad-market sleeve stays in VOO.

| Fund | Allocation | Sleeve | Why it is included | PGR Correlation | Relative Signal | P(Benchmark Beats PGR) |
|------|------------|--------|--------------------|-----------------|-----------------|------------------------|
| VOO | 50% | Broad US equity core | Core US beta sleeve that keeps the portfolio equity-heavy without recreating single-stock PGR risk. | 0.35 | Supportive (-0.7%) | n/a |
| VGT | 18% | Technology tilt | Growth engine and explicit tech tilt when the relative signal supports owning more innovation exposure than a pure core index. | 0.32 | Base-weight only (n/a) | n/a |
| SCHD | 13% | Value / dividend tilt | Closest current project proxy for a value sleeve; adds a cheaper, income-oriented counterweight to the tech allocation. | 0.38 | Base-weight only (n/a) | n/a |
| VWO | 9% | Emerging-markets satellite | Higher-growth international sleeve kept modest because it is more volatile than the core international allocation. | 0.28 | Keep near base (+2.5%) | 28.5% |
| VXUS | 5% | International core | Primary geographic diversifier away from a US employer-stock concentration. | 0.26 | Keep near base (+1.2%) | 26.2% |
| BND | 5% | Bond ballast | Small stabilizer sleeve kept intentionally light so the redeploy portfolio stays above 90% equities in normal months. | 0.07 | Keep near base (+2.9%) | 27.0% |

## Per-Benchmark Signals

- Predicted Return is from the perspective of PGR versus each fund. Positive means PGR is expected to outperform that fund; negative means the fund is expected to outperform PGR.
- Benchmark Role distinguishes realistic buy candidates from contextual or forecast-only comparison funds.

| Benchmark | Benchmark Role | Description | Predicted Return | CI Lower | CI Upper | IC | Hit Rate | P(cal) | Confidence | Signal |
|-----------|----------------|-------------|----------------|----------|----------|----|----------|--------|------------|--------|
| VOO | Buy candidate | S&P 500 | -0.71% | -29.56% | +28.14% | 0.1395 | 58.3% | 63.7% | LOW | NEUTRAL |
| VXUS | Buy candidate | Total International Stock | +1.23% | -33.97% | +36.43% | 0.0496 | 66.2% | 73.8% | LOW | NEUTRAL |
| VWO | Buy candidate | Emerging Markets | +2.46% | -31.05% | +35.97% | 0.0652 | 61.4% | 71.5% | HIGH | OUTPERFORM |
| VMBS | Forecast only | Mortgage-Backed Securities | +3.50% | -23.12% | +30.12% | 0.0166 | 67.9% | 73.1% | LOW | NEUTRAL |
| BND | Buy candidate | Total Bond Market | +2.89% | -22.86% | +28.64% | 0.1261 | 69.6% | 73.0% | HIGH | OUTPERFORM |
| GLD | Forecast only | Gold Shares | +2.43% | -31.06% | +35.93% | 0.1287 | 63.4% | 67.1% | MODERATE | OUTPERFORM |
| DBC | Forecast only | DB Commodity Index | -2.06% | -32.88% | +28.75% | 0.1643 | 67.8% | 43.5% | LOW | UNDERPERFORM |
| VDE | Forecast only | Energy | -8.36% | -46.10% | +29.38% | 0.0088 | 59.4% | 59.3% | LOW | NEUTRAL |

---

## Tax Context

| Parameter | Value |
|-----------|-------|
| STCG Rate (federal) | 37% |
| LTCG Rate (federal) | 20% |
| Tax-rate differential | 17% |
| **LTCG breakeven PGR return (lot all gain)** | **-21.25%** |
| LTCG breakeven PGR return (gain = half the price) | -10.62% |
| Model forecast, PGR vs benchmarks (6M, relative) | +0.18% |
| P(outperform) | 65.6% |
| Next time-based vest | 2027-01-19 |
| Next performance vest | 2027-07-17 |

**Holding a vested lot to its LTCG date gives more after-tax cash than selling it now at STCG unless PGR's own price falls by more than 21.25% (a lot that is all gain) or 10.62% (a lot whose gain is half its price) before that date.** At the vest itself the shares carry no gain, so the tax rate does not change the proceeds of an immediate sale.

The model expects PGR to beat the benchmarks by 0.2% over 6 months. That is a relative forecast, not a PGR price forecast, so it is not compared with the breakeven above.

> **Breakeven formula:** `-g × (STCG − LTCG) / (1 − LTCG)`, where `g` is the
> lot's unrealised gain as a fraction of the current price. Below this absolute
> PGR return, selling now at STCG beats holding to the LTCG date (the day after
> the one-year anniversary of the vest); above it, holding wins. It compares
> cash at the LTCG date and ignores what sale proceeds would earn meanwhile.

---

*Generated by `scripts/monthly_decision.py`*