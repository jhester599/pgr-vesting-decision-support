# PGR Monthly Decision Report — September 2026

**As-of Date:** 2026-09-21  
**Run Date:** 2026-09-20  
**Model Version:** v11.1 (lean 2-model ensemble: Ridge + GBT, v18 feature sets, 8-benchmark PRIMARY_FORECAST_UNIVERSE, inverse-variance weighting, v38 post-ensemble shrinkage alpha=0.50, C(8,2)=28 CPCV paths; ElasticNet+BayesianRidge retired after v18/v20 research showed Ridge+GBT outperforms on IC, hit rate, and obs/feature ratio)  
**Recommendation Layer:** Live production recommendation layer (quality-weighted consensus)  

---

## Executive Summary

- What changed since last month: Previous logged month (2026-08-20) was UNDERPERFORM at -4.57% with mean IC 0.1213.
- Current model view: Consensus signal is NEUTRAL, but the average relative-return forecast is -3.71% across benchmarks over the next 6 months. Recommendation mode remains DEFER-TO-TAX-DEFAULT.
- How trustworthy it is: Model quality is too weak to justify a prediction-led vesting action. Aggregate health: OOS R^2 -1.13%, IC 0.1649, hit rate 66.6%.
- What to do at the next vest: Next vest guidance unavailable because the lot file or latest PGR price is missing.
- What would change the recommendation: A more aggressive recommendation would require aggregate OOS R^2 >= 2%, mean IC >= 0.07, hit rate >= 55%, and a non-failing representative CPCV check.

---

## Data Freshness

> All monitored feeds are within freshness thresholds for this run.

| Feed | Latest Date | Age | Limit | Status |
|------|-------------|-----|-------|--------|
| Daily prices | 2026-09-18 | 2 days | 10 days | **OK** |
| FRED macro | 2026-09-30 | 0 days | 45 days | **OK** |
| PGR monthly EDGAR | 2026-08-31 | 20 days | 25-day filing grace | **OK** |

---

## Decision At A Glance

- Hold vs Sell: **Hold 50% / Sell 50% of the next vest tranche**
- Is this month actionable? **No — follow the default tax/diversification rule.**
- Top-line decision: **Hold 50% / Sell 50% of the next vest tranche. No — follow the default tax/diversification rule.**
- Shadow classifier probability: **41.8%** (LOW)
- **Portfolio-aligned P(Actionable Sell):** 52.2% [NEUTRAL] _(investable pool, fixed weights)_
- **Path B P(Actionable Sell):** 49.4% [NEUTRAL] _(composite portfolio target, temp-scaled)_

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
| Predicted 6M Relative Return | -3.71% |
| P(Outperform, raw) | 50.0% |
| P(Outperform, calibrated) | 60.6% |
| 80% Prediction Interval (median) | -30.63% to +24.11% |
| Mean IC (across benchmarks) | 0.1133 |
| Mean Hit Rate | 66.2% |
| Aggregate OOS R^2 | -1.13% |

> **Note:** The sell % recommendation is used only at actual vesting events
> (January and July).  Monthly reports are monitoring tools, not trade signals.
>
> **Calibration:** Phase 2 — Platt scaling active (n=1,224 OOS obs).  ECE = 3.4% [95% CI: 2.1%–7.2%].

---

## Classification Confidence Check

> Shadow-only interpretation layer from the v87-v96 classifier research.
> It does not change the live recommendation or sell percentage.

| Field | Value |
|-------|-------|
| Target | actionable_sell_3pct |
| Construction | Separate benchmark logistic + quality-weighted aggregate |
| P(Actionable Sell) | 41.8% |
| Confidence Tier | LOW |
| Classifier Stance | NEUTRAL |
| Portfolio-aligned P(Actionable Sell) | 52.2% [NEUTRAL] |
| Investable Pool Confidence Tier | LOW |
| Path B P(Actionable Sell) | 49.4% [NEUTRAL] |
| Path B Confidence Tier | LOW |
| Agreement with Live Recommendation | Aligned |
| Interpretation | Shadow classifier is near its neutral band (41.8%); use it as a low-confidence interpretation layer rather than a decision override. |

---

## Confidence Snapshot

- 2/4 core gates pass. The signal may still be directionally interesting, but the quality gate remains too weak for a prediction-led vest action.

| Check | Current | Threshold | Status | Meaning |
|-------|---------|-----------|--------|---------|
| Mean IC | 0.1133 | >= 0.0700 | **PASS** | Cross-benchmark ranking signal. |
| Mean hit rate | 66.2% | >= 55.0% | **PASS** | Directional accuracy versus zero. |
| Aggregate OOS R^2 | -1.13% | >= 2.00% | **FAIL** | Calibration / fit versus a naive benchmark. |
| Representative CPCV | FAIL | not FAIL | **FAIL** | Stability across purged cross-validation paths. |

---

## Model Health

- Latest tracked month: **2026-09-30**
- Rolling 12M IC: **0.1517**
- Rolling 12M Hit Rate: **65.6%**
- Rolling 12M ECE: **2.2%**
- IC breach streak: **0** month(s)
- Status: **Stable: no sustained rolling-IC drift alert is active.**

---

## Decision Policy Backtest

> OOS performance of each decision policy applied to all historical model predictions.  "Mean Return" is the portfolio-weighted realized relative return per vesting event.  "Cumulative" is the sum across all events.  "Capture Ratio" is the fraction of oracle (always hold when positive) gains captured.  N = number of OOS events.

### Fixed Heuristic Baselines

| Policy | N | Mean Return | Cumulative | Capture Ratio |
|--------|---|-------------|------------|---------------|
| Sell 100% (always) | 1224 | +0.00% | +0.00% | 0.0% |
| Sell 50% (always) | 1224 | +3.75% | +4592.69% | 32.0% |
| Hold 100% (always) | 1224 | +7.50% | +9185.38% | 63.9% |

### Model-Driven Policies vs. Heuristics

| Policy | N | Mean Return | Cumul. Return | Uplift vs Sell-All | Uplift vs Hold-All | Uplift vs 50% | Capture |
|--------|---|-------------|---------------|--------------------|--------------------|---------------|---------|
| Model: sign (hold if pred > 0) | 1224 | +7.35% | +8991.60% | +7.35% | -0.16% | +3.59% | 62.6% |
| Model: tiered 25/50/100 | 1224 | +2.26% | +2763.60% | +2.26% | -5.25% | -1.49% | 19.2% |
| Model: neutral band ±2% | 1224 | +7.11% | +8702.53% | +7.11% | -0.39% | +3.36% | 60.6% |
| Model: neutral band ±3% | 1224 | +6.87% | +8410.31% | +6.87% | -0.63% | +3.12% | 58.5% |


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

- Default posture: `89%` equities / `11%` bonds across the curated investable universe.
- Monthly tilts use a `25%` signal overlay around the base weights, so the recommendation can adapt without becoming a full tactical allocation model.
- Investable universe used in the monthly workflow: `VOO, VGT, SCHD, VXUS, VWO, BND`.
- Constraint note: The current project universe does not yet include a dedicated small-cap ETF, so the value sleeve uses SCHD and the broad-market sleeve stays in VOO.

| Fund | Allocation | Sleeve | Why it is included | PGR Correlation | Relative Signal | P(Benchmark Beats PGR) |
|------|------------|--------|--------------------|-----------------|-----------------|------------------------|
| VOO | 31% | Broad US equity core | Core US beta sleeve that keeps the portfolio equity-heavy without recreating single-stock PGR risk. | 0.12 | Only keep at floor weight (+3.0%) | 35.2% |
| VXUS | 20% | International core | Primary geographic diversifier away from a US employer-stock concentration. | 0.26 | Preferred this month (-3.5%) | n/a |
| VWO | 18% | Emerging-markets satellite | Higher-growth international sleeve kept modest because it is more volatile than the core international allocation. | 0.33 | Preferred this month (-4.0%) | n/a |
| VGT | 11% | Technology tilt | Growth engine and explicit tech tilt when the relative signal supports owning more innovation exposure than a pure core index. | 0.27 | Base-weight only (n/a) | n/a |
| BND | 11% | Bond ballast | Small stabilizer sleeve kept intentionally light so the redeploy portfolio stays above 90% equities in normal months. | 0.05 | Only keep at floor weight (+3.4%) | 26.1% |
| SCHD | 9% | Value / dividend tilt | Closest current project proxy for a value sleeve; adds a cheaper, income-oriented counterweight to the tech allocation. | 0.36 | Base-weight only (n/a) | n/a |

## Per-Benchmark Signals

- Predicted Return is from the perspective of PGR versus each fund. Positive means PGR is expected to outperform that fund; negative means the fund is expected to outperform PGR.
- Benchmark Role distinguishes realistic buy candidates from contextual or forecast-only comparison funds.

| Benchmark | Benchmark Role | Description | Predicted Return | CI Lower | CI Upper | IC | Hit Rate | P(raw) | P(cal) | Confidence | Signal |
|-----------|----------------|-------------|----------------|----------|----------|----|----------|--------|--------|------------|--------|
| VOO | Buy candidate | S&P 500 | +3.04% | -21.84% | +27.93% | -0.0970 | 51.8% | 50.0% | 64.8% | LOW | NEUTRAL |
| VXUS | Buy candidate | Total International Stock | -3.51% | -32.57% | +25.55% | -0.0458 | 65.7% | 50.0% | 76.0% | LOW | NEUTRAL |
| VWO | Buy candidate | Emerging Markets | -3.98% | -28.69% | +20.73% | 0.0365 | 63.1% | 50.0% | 85.6% | LOW | NEUTRAL |
| VMBS | Forecast only | Mortgage-Backed Securities | +2.40% | -18.70% | +23.50% | 0.1321 | 76.2% | 50.0% | 69.4% | LOW | OUTPERFORM |
| BND | Buy candidate | Total Bond Market | +3.37% | -17.63% | +24.36% | 0.1864 | 72.4% | 50.0% | 73.9% | LOW | OUTPERFORM |
| GLD | Forecast only | Gold Shares | -2.54% | -38.92% | +33.85% | 0.1816 | 60.2% | 50.0% | 60.7% | LOW | UNDERPERFORM |
| DBC | Forecast only | DB Commodity Index | -11.22% | -46.28% | +23.85% | 0.2601 | 72.3% | 50.0% | 19.1% | LOW | UNDERPERFORM |
| VDE | Forecast only | Energy | -13.62% | -45.72% | +18.48% | 0.0697 | 61.0% | 50.0% | 35.3% | LOW | UNDERPERFORM |

---

## Tax Context

| Parameter | Value |
|-----------|-------|
| STCG Rate (federal) | 37% |
| LTCG Rate (federal) | 20% |
| Tax-rate differential | 17% |
| **LTCG breakeven return** | **21.25%** |
| Current model prediction (6M) | -3.71% |
| P(outperform) | 60.6% |
| Next time-based vest | 2027-01-19 |
| Next performance vest | 2027-07-17 |

⚠️ **Model predicts negative return (-3.7%).**  Consider capital-loss harvesting scenario — a tax loss at 37% STCG rate can offset other gains.  See three-scenario analysis at vesting.

> **Breakeven formula:** `(STCG − LTCG) / (1 − LTCG)` — the minimum
> return needed on RSUs held to LTCG eligibility (366 days post-vest) to
> produce higher after-tax proceeds than selling immediately at STCG.
> Run `compute_three_scenarios()` at each vesting event for lot-specific analysis.

---

*Generated by `scripts/monthly_decision.py`*