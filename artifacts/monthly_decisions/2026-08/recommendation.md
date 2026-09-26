# PGR Monthly Decision Report — August 2026

**As-of Date:** 2026-08-20  
**Run Date:** 2026-08-20  
**Model Version:** v11.1 (lean 2-model ensemble: Ridge + GBT, v18 feature sets, 8-benchmark PRIMARY_FORECAST_UNIVERSE, inverse-variance weighting, v38 post-ensemble shrinkage alpha=0.50, C(8,2)=28 CPCV paths; ElasticNet+BayesianRidge retired after v18/v20 research showed Ridge+GBT outperforms on IC, hit rate, and obs/feature ratio)  
**Recommendation Layer:** Live production recommendation layer (quality-weighted consensus)  

---

## Executive Summary

- What changed since last month: Previous logged month (2026-07-22) was NEUTRAL at -4.66% with mean IC 0.0993.
- Current model view: PGR is projected to lag the benchmark set by -4.57% over the next 6 months. Recommendation mode remains DEFER-TO-TAX-DEFAULT.
- How trustworthy it is: Model quality is too weak to justify a prediction-led vesting action. Aggregate health: OOS R^2 -0.14%, IC 0.1806, hit rate 64.8%.
- What to do at the next vest: Next vest guidance unavailable because the lot file or latest PGR price is missing.
- What would change the recommendation: A more aggressive recommendation would require aggregate OOS R^2 >= 2%, mean IC >= 0.07, hit rate >= 55%, and a non-failing representative CPCV check.

---

## Data Freshness

> All monitored feeds are within freshness thresholds for this run.

| Feed | Latest Date | Age | Limit | Status |
|------|-------------|-----|-------|--------|
| Daily prices | 2026-08-14 | 6 days | 10 days | **OK** |
| FRED macro | 2026-08-31 | 0 days | 45 days | **OK** |
| PGR monthly EDGAR | 2026-07-31 | 20 days | 25-day filing grace | **OK** |

---

## Decision At A Glance

- Hold vs Sell: **Hold 50% / Sell 50% of the next vest tranche**
- Is this month actionable? **No — follow the default tax/diversification rule.**
- Top-line decision: **Hold 50% / Sell 50% of the next vest tranche. No — follow the default tax/diversification rule.**
- Shadow classifier probability: **41.9%** (LOW)
- **Portfolio-aligned P(Actionable Sell):** 51.6% [NEUTRAL] _(investable pool, fixed weights)_
- **Path B P(Actionable Sell):** 51.0% [NEUTRAL] _(composite portfolio target, temp-scaled)_

## Agreement Panel

- Live recommendation: **DEFER-TO-TAX-DEFAULT / sell 50%**
- Consensus cross-check: **Aligned**
- Classifier shadow: **Aligned**
- Shadow gate overlay: **DEFER-TO-TAX-DEFAULT / sell 50%** (no live change)

---

## Consensus Signal

| Field | Value |
|-------|-------|
| Signal | **UNDERPERFORM (LOW CONFIDENCE)** |
| Recommendation Mode | **DEFER-TO-TAX-DEFAULT** |
| Recommended Sell % | **50%** |
| Predicted 6M Relative Return | -4.57% |
| P(Outperform, raw) | 50.0% |
| P(Outperform, calibrated) | 58.4% |
| 80% Prediction Interval (median) | -31.34% to +24.08% |
| Mean IC (across benchmarks) | 0.1213 |
| Mean Hit Rate | 65.4% |
| Aggregate OOS R^2 | -0.14% |

> **Note:** The sell % recommendation is used only at actual vesting events
> (January and July).  Monthly reports are monitoring tools, not trade signals.
>
> **Calibration:** Phase 2 — Platt scaling active (n=1,224 OOS obs).  ECE = 2.5% [95% CI: 1.8%–7.0%].

---

## Classification Confidence Check

> Shadow-only interpretation layer from the v87-v96 classifier research.
> It does not change the live recommendation or sell percentage.

| Field | Value |
|-------|-------|
| Target | actionable_sell_3pct |
| Construction | Separate benchmark logistic + quality-weighted aggregate |
| P(Actionable Sell) | 41.9% |
| Confidence Tier | LOW |
| Classifier Stance | NEUTRAL |
| Portfolio-aligned P(Actionable Sell) | 51.6% [NEUTRAL] |
| Investable Pool Confidence Tier | LOW |
| Path B P(Actionable Sell) | 51.0% [NEUTRAL] |
| Path B Confidence Tier | LOW |
| Agreement with Live Recommendation | Aligned |
| Interpretation | Shadow classifier is near its neutral band (41.9%); use it as a low-confidence interpretation layer rather than a decision override. |

---

## Confidence Snapshot

- 2/4 core gates pass. The signal may still be directionally interesting, but the quality gate remains too weak for a prediction-led vest action.

| Check | Current | Threshold | Status | Meaning |
|-------|---------|-----------|--------|---------|
| Mean IC | 0.1213 | >= 0.0700 | **PASS** | Cross-benchmark ranking signal. |
| Mean hit rate | 65.4% | >= 55.0% | **PASS** | Directional accuracy versus zero. |
| Aggregate OOS R^2 | -0.14% | >= 2.00% | **FAIL** | Calibration / fit versus a naive benchmark. |
| Representative CPCV | FAIL | not FAIL | **FAIL** | Stability across purged cross-validation paths. |

---

## Model Health

- Latest tracked month: **2026-08-31**
- Rolling 12M IC: **0.1498**
- Rolling 12M Hit Rate: **65.4%**
- Rolling 12M ECE: **2.1%**
- IC breach streak: **0** month(s)
- Status: **Stable: no sustained rolling-IC drift alert is active.**

---

## Decision Policy Backtest

> OOS performance of each decision policy applied to all historical model predictions.  "Mean Return" is the portfolio-weighted realized relative return per vesting event.  "Cumulative" is the sum across all events.  "Capture Ratio" is the fraction of oracle (always hold when positive) gains captured.  N = number of OOS events.

### Fixed Heuristic Baselines

| Policy | N | Mean Return | Cumulative | Capture Ratio |
|--------|---|-------------|------------|---------------|
| Sell 100% (always) | 1224 | +0.00% | +0.00% | 0.0% |
| Sell 50% (always) | 1224 | +3.76% | +4596.30% | 32.0% |
| Hold 100% (always) | 1224 | +7.51% | +9192.59% | 63.9% |

### Model-Driven Policies vs. Heuristics

| Policy | N | Mean Return | Cumul. Return | Uplift vs Sell-All | Uplift vs Hold-All | Uplift vs 50% | Capture |
|--------|---|-------------|---------------|--------------------|--------------------|---------------|---------|
| Model: sign (hold if pred > 0) | 1224 | +7.19% | +8806.66% | +7.19% | -0.32% | +3.44% | 61.2% |
| Model: tiered 25/50/100 | 1224 | +2.25% | +2758.65% | +2.25% | -5.26% | -1.50% | 19.2% |
| Model: neutral band ±2% | 1224 | +7.07% | +8656.40% | +7.07% | -0.44% | +3.32% | 60.2% |
| Model: neutral band ±3% | 1224 | +6.98% | +8544.98% | +6.98% | -0.53% | +3.23% | 59.4% |


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

The point forecast leans underperform, and 2/8 (25%) benchmarks favour outperformance, but the broader quality gate is failing.

Recommended action at next vesting event: **DEFAULT 50% SALE** for diversification and tax discipline, not because the prediction is high-confidence.

---

## Redeploy Guidance

- Broad US Equity: VOO. Broad US equity diversifies away from single-stock risk without concentrating further in insurance.
- International Equity: VXUS, VWO. International equity lowers home-market and insurance concentration.
- Fixed Income: BND. Fixed income is the cleanest concentration-reduction bucket when model confidence is weak.
- Sector Context: VGT, SCHD. Sector funds are context-only unless no stronger diversifying destination is available.

## Suggested Redeploy Portfolio

- Default posture: `92%` equities / `8%` bonds across the curated investable universe.
- Monthly tilts use a `25%` signal overlay around the base weights, so the recommendation can adapt without becoming a full tactical allocation model.
- Investable universe used in the monthly workflow: `VOO, VGT, SCHD, VXUS, VWO, BND`.
- Constraint note: The current project universe does not yet include a dedicated small-cap ETF, so the value sleeve uses SCHD and the broad-market sleeve stays in VOO.

| Fund | Allocation | Sleeve | Why it is included | PGR Correlation | Relative Signal | P(Benchmark Beats PGR) |
|------|------------|--------|--------------------|-----------------|-----------------|------------------------|
| VOO | 28% | Broad US equity core | Core US beta sleeve that keeps the portfolio equity-heavy without recreating single-stock PGR risk. | 0.12 | Supportive (-0.6%) | n/a |
| SCHD | 22% | Value / dividend tilt | Closest current project proxy for a value sleeve; adds a cheaper, income-oriented counterweight to the tech allocation. | 0.36 | Base-weight only (n/a) | n/a |
| VWO | 16% | Emerging-markets satellite | Higher-growth international sleeve kept modest because it is more volatile than the core international allocation. | 0.33 | Preferred this month (-6.3%) | n/a |
| VXUS | 16% | International core | Primary geographic diversifier away from a US employer-stock concentration. | 0.26 | Keep near base (+2.5%) | 30.1% |
| VGT | 10% | Technology tilt | Growth engine and explicit tech tilt when the relative signal supports owning more innovation exposure than a pure core index. | 0.27 | Base-weight only (n/a) | n/a |
| BND | 8% | Bond ballast | Small stabilizer sleeve kept intentionally light so the redeploy portfolio stays above 90% equities in normal months. | 0.05 | Only keep at floor weight (+3.3%) | 25.5% |

## Per-Benchmark Signals

- Predicted Return is from the perspective of PGR versus each fund. Positive means PGR is expected to outperform that fund; negative means the fund is expected to outperform PGR.
- Benchmark Role distinguishes realistic buy candidates from contextual or forecast-only comparison funds.

| Benchmark | Benchmark Role | Description | Predicted Return | CI Lower | CI Upper | IC | Hit Rate | P(raw) | P(cal) | Confidence | Signal |
|-----------|----------------|-------------|----------------|----------|----------|----|----------|--------|--------|------------|--------|
| VOO | Buy candidate | S&P 500 | -0.60% | -26.12% | +24.92% | -0.1265 | 53.1% | 50.0% | 67.8% | LOW | NEUTRAL |
| VXUS | Buy candidate | Total International Stock | +2.48% | -27.20% | +32.17% | -0.0387 | 63.9% | 50.0% | 69.9% | LOW | NEUTRAL |
| VWO | Buy candidate | Emerging Markets | -6.30% | -35.48% | +22.89% | 0.0554 | 63.9% | 50.0% | 80.9% | LOW | UNDERPERFORM |
| VMBS | Forecast only | Mortgage-Backed Securities | +2.98% | -17.90% | +23.86% | 0.1714 | 73.0% | 50.0% | 71.1% | LOW | OUTPERFORM |
| BND | Buy candidate | Total Bond Market | +3.33% | -17.63% | +24.30% | 0.2274 | 67.9% | 50.0% | 74.5% | LOW | OUTPERFORM |
| GLD | Forecast only | Gold Shares | -6.08% | -40.79% | +28.62% | 0.1479 | 60.2% | 50.0% | 55.6% | LOW | UNDERPERFORM |
| DBC | Forecast only | DB Commodity Index | -13.71% | -47.97% | +20.54% | 0.2532 | 70.2% | 50.0% | 16.4% | LOW | UNDERPERFORM |
| VDE | Forecast only | Energy | -14.41% | -47.08% | +18.25% | 0.1112 | 64.8% | 50.0% | 31.0% | LOW | UNDERPERFORM |

---

## Tax Context

| Parameter | Value |
|-----------|-------|
| STCG Rate (federal) | 37% |
| LTCG Rate (federal) | 20% |
| Tax-rate differential | 17% |
| **LTCG breakeven return** | **21.25%** |
| Current model prediction (6M) | -4.57% |
| P(outperform) | 58.4% |
| Next time-based vest | 2027-01-19 |
| Next performance vest | 2027-07-17 |

⚠️ **Model predicts negative return (-4.6%).**  Consider capital-loss harvesting scenario — a tax loss at 37% STCG rate can offset other gains.  See three-scenario analysis at vesting.

> **Breakeven formula:** `(STCG − LTCG) / (1 − LTCG)` — the minimum
> return needed on RSUs held to LTCG eligibility (366 days post-vest) to
> produce higher after-tax proceeds than selling immediately at STCG.
> Run `compute_three_scenarios()` at each vesting event for lot-specific analysis.

---

*Generated by `scripts/monthly_decision.py`*