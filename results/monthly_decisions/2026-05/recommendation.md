# PGR Monthly Decision Report — May 2026

**As-of Date:** 2026-05-20  
**Run Date:** 2026-05-20  
**Model Version:** v11.1 (lean 2-model ensemble: Ridge + GBT, v18 feature sets, 8-benchmark PRIMARY_FORECAST_UNIVERSE, inverse-variance weighting, v38 post-ensemble shrinkage alpha=0.50, C(8,2)=28 CPCV paths; ElasticNet+BayesianRidge retired after v18/v20 research showed Ridge+GBT outperforms on IC, hit rate, and obs/feature ratio)  
**Recommendation Layer:** Live production recommendation layer (quality-weighted consensus)  

---

## Executive Summary

- What changed since last month: Previous logged month (2026-04-22) was NEUTRAL at -2.50% with mean IC 0.1427.
- Current model view: Consensus signal is NEUTRAL, but the average relative-return forecast is -2.69% across benchmarks over the next 6 months. Recommendation mode remains DEFER-TO-TAX-DEFAULT.
- How trustworthy it is: Model quality is too weak to justify a prediction-led vesting action. Aggregate health: OOS R^2 -3.51%, IC 0.0979, hit rate 64.2%.
- What to do at the next vest: Next vest guidance unavailable because the lot file or latest PGR price is missing.
- What would change the recommendation: A more aggressive recommendation would require aggregate OOS R^2 >= 2%, mean IC >= 0.07, hit rate >= 55%, and a non-failing representative CPCV check.

---

## Data Freshness

> All monitored feeds are within freshness thresholds for this run.

| Feed | Latest Date | Age | Limit | Status |
|------|-------------|-----|-------|--------|
| Daily prices | 2026-05-15 | 5 days | 10 days | **OK** |
| FRED macro | 2026-05-29 | 0 days | 45 days | **OK** |
| PGR monthly EDGAR | 2026-04-30 | 20 days | 25-day filing grace | **OK** |

---

## Decision At A Glance

- Hold vs Sell: **Hold 50% / Sell 50% of the next vest tranche**
- Is this month actionable? **No — follow the default tax/diversification rule.**
- Top-line decision: **Hold 50% / Sell 50% of the next vest tranche. No — follow the default tax/diversification rule.**
- Shadow classifier probability: **40.9%** (LOW)
- **Portfolio-aligned P(Actionable Sell):** 48.5% [NEUTRAL] _(investable pool, fixed weights)_
- **Path B P(Actionable Sell):** 52.7% [NEUTRAL] _(composite portfolio target, temp-scaled)_

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
| Predicted 6M Relative Return | -2.69% |
| P(Outperform, raw) | 50.0% |
| P(Outperform, calibrated) | 63.3% |
| 80% Prediction Interval (median) | -34.37% to +25.34% |
| Mean IC (across benchmarks) | 0.1026 |
| Mean Hit Rate | 63.9% |
| Aggregate OOS R^2 | -3.51% |

> **Note:** The sell % recommendation is used only at actual vesting events
> (January and July).  Monthly reports are monitoring tools, not trade signals.
>
> **Calibration:** Phase 2 — Platt scaling active (n=1,200 OOS obs).  ECE = 1.0% [95% CI: 1.1%–5.7%].

---

## Classification Confidence Check

> Shadow-only interpretation layer from the v87-v96 classifier research.
> It does not change the live recommendation or sell percentage.

| Field | Value |
|-------|-------|
| Target | actionable_sell_3pct |
| Construction | Separate benchmark logistic + quality-weighted aggregate |
| P(Actionable Sell) | 40.9% |
| Confidence Tier | LOW |
| Classifier Stance | NEUTRAL |
| Portfolio-aligned P(Actionable Sell) | 48.5% [NEUTRAL] |
| Investable Pool Confidence Tier | LOW |
| Path B P(Actionable Sell) | 52.7% [NEUTRAL] |
| Path B Confidence Tier | LOW |
| Agreement with Live Recommendation | Aligned |
| Interpretation | Shadow classifier is near its neutral band (40.9%); use it as a low-confidence interpretation layer rather than a decision override. |

---

## Confidence Snapshot

- 2/4 core gates pass. The signal may still be directionally interesting, but the quality gate remains too weak for a prediction-led vest action.

| Check | Current | Threshold | Status | Meaning |
|-------|---------|-----------|--------|---------|
| Mean IC | 0.1026 | >= 0.0700 | **PASS** | Cross-benchmark ranking signal. |
| Mean hit rate | 63.9% | >= 55.0% | **PASS** | Directional accuracy versus zero. |
| Aggregate OOS R^2 | -3.51% | >= 2.00% | **FAIL** | Calibration / fit versus a naive benchmark. |
| Representative CPCV | FAIL | not FAIL | **FAIL** | Stability across purged cross-validation paths. |

---

## Model Health

- Latest tracked month: **2026-05-31**
- Rolling 12M IC: **0.1550**
- Rolling 12M Hit Rate: **66.4%**
- Rolling 12M ECE: **1.9%**
- IC breach streak: **0** month(s)
- Status: **Stable: no sustained rolling-IC drift alert is active.**

---

## Decision Policy Backtest

> OOS performance of each decision policy applied to all historical model predictions.  "Mean Return" is the portfolio-weighted realized relative return per vesting event.  "Cumulative" is the sum across all events.  "Capture Ratio" is the fraction of oracle (always hold when positive) gains captured.  N = number of OOS events.

### Fixed Heuristic Baselines

| Policy | N | Mean Return | Cumulative | Capture Ratio |
|--------|---|-------------|------------|---------------|
| Sell 100% (always) | 1200 | +0.00% | +0.00% | 0.0% |
| Sell 50% (always) | 1200 | +3.98% | +4774.87% | 33.2% |
| Hold 100% (always) | 1200 | +7.96% | +9549.73% | 66.4% |

### Model-Driven Policies vs. Heuristics

| Policy | N | Mean Return | Cumul. Return | Uplift vs Sell-All | Uplift vs Hold-All | Uplift vs 50% | Capture |
|--------|---|-------------|---------------|--------------------|--------------------|---------------|---------|
| Model: sign (hold if pred > 0) | 1200 | +7.31% | +8773.57% | +7.31% | -0.65% | +3.33% | 61.0% |
| Model: tiered 25/50/100 | 1200 | +1.80% | +2161.91% | +1.80% | -6.16% | -2.18% | 15.0% |
| Model: neutral band ±2% | 1200 | +6.95% | +8342.36% | +6.95% | -1.01% | +2.97% | 58.0% |
| Model: neutral band ±3% | 1200 | +6.70% | +8036.72% | +6.70% | -1.26% | +2.72% | 55.9% |


---

## Portfolio Optimizer Status

> ⚠️ **Optimizer fallback active** — Black-Litterman optimization could not converge (`optimization_failure`).  Portfolio weights fall back to equal-weight allocation.  This does not affect the primary recommendation; it is a diagnostic indicator.

| Parameter | Value |
|-----------|-------|
| Optimizer | Black-Litterman (PyPortfolioOpt / Ledoit-Wolf) |
| Status | ⚠️ Fallback — optimization_failure |
| Active benchmarks | 8 |
| View tickers incorporated | 6 |


---

## Interpretation

The point forecast leans neutral, and 2/8 (25%) benchmarks favour outperformance, but the broader quality gate is failing.

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
| VOO | 28% | Broad US equity core | Core US beta sleeve that keeps the portfolio equity-heavy without recreating single-stock PGR risk. | 0.14 | Keep near base (+1.4%) | 32.3% |
| VGT | 22% | Technology tilt | Growth engine and explicit tech tilt when the relative signal supports owning more innovation exposure than a pure core index. | 0.29 | Base-weight only (n/a) | n/a |
| VXUS | 18% | International core | Primary geographic diversifier away from a US employer-stock concentration. | 0.28 | Supportive (-1.5%) | n/a |
| SCHD | 16% | Value / dividend tilt | Closest current project proxy for a value sleeve; adds a cheaper, income-oriented counterweight to the tech allocation. | 0.39 | Base-weight only (n/a) | n/a |
| VWO | 11% | Emerging-markets satellite | Higher-growth international sleeve kept modest because it is more volatile than the core international allocation. | 0.34 | Keep near base (+0.8%) | 28.0% |
| BND | 5% | Bond ballast | Small stabilizer sleeve kept intentionally light so the redeploy portfolio stays above 90% equities in normal months. | 0.04 | Keep near base (+2.0%) | 27.9% |

## Per-Benchmark Signals

- Predicted Return is from the perspective of PGR versus each fund. Positive means PGR is expected to outperform that fund; negative means the fund is expected to outperform PGR.
- Benchmark Role distinguishes realistic buy candidates from contextual or forecast-only comparison funds.

| Benchmark | Benchmark Role | Description | Predicted Return | CI Lower | CI Upper | IC | Hit Rate | P(raw) | P(cal) | Confidence | Signal |
|-----------|----------------|-------------|----------------|----------|----------|----|----------|--------|--------|------------|--------|
| VOO | Buy candidate | S&P 500 | +1.43% | -25.21% | +28.06% | -0.0330 | 53.9% | 50.0% | 67.7% | LOW | NEUTRAL |
| VXUS | Buy candidate | Total International Stock | -1.50% | -37.11% | +34.10% | 0.1203 | 65.7% | 50.0% | 70.4% | LOW | UNDERPERFORM |
| VWO | Buy candidate | Emerging Markets | +0.75% | -31.64% | +33.15% | 0.1423 | 61.9% | 50.0% | 72.0% | LOW | NEUTRAL |
| VMBS | Forecast only | Mortgage-Backed Securities | +2.27% | -19.96% | +24.50% | 0.0734 | 72.9% | 50.0% | 70.5% | LOW | OUTPERFORM |
| BND | Buy candidate | Total Bond Market | +2.00% | -19.65% | +23.66% | 0.1947 | 69.7% | 50.0% | 72.1% | LOW | OUTPERFORM |
| GLD | Forecast only | Gold Shares | -6.60% | -39.37% | +26.17% | 0.1268 | 58.1% | 50.0% | 60.7% | LOW | UNDERPERFORM |
| DBC | Forecast only | DB Commodity Index | -11.93% | -41.02% | +17.16% | 0.1068 | 66.1% | 50.0% | 41.0% | LOW | UNDERPERFORM |
| VDE | Forecast only | Energy | -9.59% | -41.07% | +21.89% | -0.0055 | 58.6% | 50.0% | 52.5% | LOW | NEUTRAL |

---

## Tax Context

| Parameter | Value |
|-----------|-------|
| STCG Rate (federal) | 37% |
| LTCG Rate (federal) | 20% |
| Tax-rate differential | 17% |
| **LTCG breakeven return** | **21.25%** |
| Current model prediction (6M) | -2.69% |
| P(outperform) | 63.3% |
| Next time-based vest | 2027-01-19 |
| Next performance vest | 2026-07-17 |

⚠️ **Model predicts negative return (-2.7%).**  Consider capital-loss harvesting scenario — a tax loss at 37% STCG rate can offset other gains.  See three-scenario analysis at vesting.

> **Breakeven formula:** `(STCG − LTCG) / (1 − LTCG)` — the minimum
> return needed on RSUs held to LTCG eligibility (366 days post-vest) to
> produce higher after-tax proceeds than selling immediately at STCG.
> Run `compute_three_scenarios()` at each vesting event for lot-specific analysis.

---

*Generated by `scripts/monthly_decision.py`*