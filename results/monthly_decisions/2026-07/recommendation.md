# PGR Monthly Decision Report — July 2026

**As-of Date:** 2026-07-20  
**Run Date:** 2026-07-20  
**Model Version:** v11.1 (lean 2-model ensemble: Ridge + GBT, v18 feature sets, 8-benchmark PRIMARY_FORECAST_UNIVERSE, inverse-variance weighting, v38 post-ensemble shrinkage alpha=0.50, C(8,2)=28 CPCV paths; ElasticNet+BayesianRidge retired after v18/v20 research showed Ridge+GBT outperforms on IC, hit rate, and obs/feature ratio)  
**Recommendation Layer:** Live production recommendation layer (quality-weighted consensus)  

---

## Executive Summary

- What changed since last month: Previous logged month (2026-06-22) was NEUTRAL at -3.95% with mean IC 0.0984.
- Current model view: Consensus signal is NEUTRAL, but the average relative-return forecast is -4.66% across benchmarks over the next 6 months. Recommendation mode remains DEFER-TO-TAX-DEFAULT.
- How trustworthy it is: Model quality is too weak to justify a prediction-led vesting action. Aggregate health: OOS R^2 -2.80%, IC 0.1187, hit rate 64.2%.
- What to do at the next vest: Next vest guidance unavailable because the lot file or latest PGR price is missing.
- What would change the recommendation: A more aggressive recommendation would require aggregate OOS R^2 >= 2%, mean IC >= 0.07, hit rate >= 55%, and a non-failing representative CPCV check.

---

## Data Freshness

> All monitored feeds are within freshness thresholds for this run.

| Feed | Latest Date | Age | Limit | Status |
|------|-------------|-----|-------|--------|
| Daily prices | 2026-07-17 | 3 days | 10 days | **OK** |
| FRED macro | 2026-07-31 | 0 days | 45 days | **OK** |
| PGR monthly EDGAR | 2026-06-30 | 20 days | 25-day filing grace | **OK** |

---

## Decision At A Glance

- Hold vs Sell: **Hold 50% / Sell 50% of the next vest tranche**
- Is this month actionable? **No — follow the default tax/diversification rule.**
- Top-line decision: **Hold 50% / Sell 50% of the next vest tranche. No — follow the default tax/diversification rule.**
- Shadow classifier probability: **41.2%** (LOW)
- **Portfolio-aligned P(Actionable Sell):** 47.1% [NEUTRAL] _(investable pool, fixed weights)_
- **Path B P(Actionable Sell):** 55.2% [NEUTRAL] _(composite portfolio target, temp-scaled)_

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
| Predicted 6M Relative Return | -4.66% |
| P(Outperform, raw) | 50.0% |
| P(Outperform, calibrated) | 61.5% |
| 80% Prediction Interval (median) | -33.40% to +24.09% |
| Mean IC (across benchmarks) | 0.0980 |
| Mean Hit Rate | 64.9% |
| Aggregate OOS R^2 | -2.80% |

> **Note:** The sell % recommendation is used only at actual vesting events
> (January and July).  Monthly reports are monitoring tools, not trade signals.
>
> **Calibration:** Phase 2 — Platt scaling active (n=1,218 OOS obs).  ECE = 3.2% [95% CI: 2.3%–7.8%].

---

## Classification Confidence Check

> Shadow-only interpretation layer from the v87-v96 classifier research.
> It does not change the live recommendation or sell percentage.

| Field | Value |
|-------|-------|
| Target | actionable_sell_3pct |
| Construction | Separate benchmark logistic + quality-weighted aggregate |
| P(Actionable Sell) | 41.2% |
| Confidence Tier | LOW |
| Classifier Stance | NEUTRAL |
| Portfolio-aligned P(Actionable Sell) | 47.1% [NEUTRAL] |
| Investable Pool Confidence Tier | LOW |
| Path B P(Actionable Sell) | 55.2% [NEUTRAL] |
| Path B Confidence Tier | LOW |
| Agreement with Live Recommendation | Aligned |
| Interpretation | Shadow classifier is near its neutral band (41.2%); use it as a low-confidence interpretation layer rather than a decision override. |

---

## Confidence Snapshot

- 2/4 core gates pass. The signal may still be directionally interesting, but the quality gate remains too weak for a prediction-led vest action.

| Check | Current | Threshold | Status | Meaning |
|-------|---------|-----------|--------|---------|
| Mean IC | 0.0980 | >= 0.0700 | **PASS** | Cross-benchmark ranking signal. |
| Mean hit rate | 64.9% | >= 55.0% | **PASS** | Directional accuracy versus zero. |
| Aggregate OOS R^2 | -2.80% | >= 2.00% | **FAIL** | Calibration / fit versus a naive benchmark. |
| Representative CPCV | FAIL | not FAIL | **FAIL** | Stability across purged cross-validation paths. |

---

## Model Health

- Latest tracked month: **2026-07-31**
- Rolling 12M IC: **0.1445**
- Rolling 12M Hit Rate: **65.5%**
- Rolling 12M ECE: **2.0%**
- IC breach streak: **0** month(s)
- Status: **Stable: no sustained rolling-IC drift alert is active.**

---

## Decision Policy Backtest

> OOS performance of each decision policy applied to all historical model predictions.  "Mean Return" is the portfolio-weighted realized relative return per vesting event.  "Cumulative" is the sum across all events.  "Capture Ratio" is the fraction of oracle (always hold when positive) gains captured.  N = number of OOS events.

### Fixed Heuristic Baselines

| Policy | N | Mean Return | Cumulative | Capture Ratio |
|--------|---|-------------|------------|---------------|
| Sell 100% (always) | 1218 | +0.00% | +0.00% | 0.0% |
| Sell 50% (always) | 1218 | +3.77% | +4588.51% | 32.0% |
| Hold 100% (always) | 1218 | +7.53% | +9177.03% | 63.9% |

### Model-Driven Policies vs. Heuristics

| Policy | N | Mean Return | Cumul. Return | Uplift vs Sell-All | Uplift vs Hold-All | Uplift vs 50% | Capture |
|--------|---|-------------|---------------|--------------------|--------------------|---------------|---------|
| Model: sign (hold if pred > 0) | 1218 | +7.15% | +8703.41% | +7.15% | -0.39% | +3.38% | 60.6% |
| Model: tiered 25/50/100 | 1218 | +1.89% | +2298.84% | +1.89% | -5.65% | -1.88% | 16.0% |
| Model: neutral band ±2% | 1218 | +7.07% | +8614.03% | +7.07% | -0.46% | +3.31% | 60.0% |
| Model: neutral band ±3% | 1218 | +6.63% | +8079.03% | +6.63% | -0.90% | +2.87% | 56.3% |


---

## Portfolio Optimizer Status

> ⚠️ **Optimizer fallback active** — Black-Litterman optimization could not converge (`optimization_failure`).  Portfolio weights fall back to equal-weight allocation.  This does not affect the primary recommendation; it is a diagnostic indicator.

| Parameter | Value |
|-----------|-------|
| Optimizer | Black-Litterman (PyPortfolioOpt / Ledoit-Wolf) |
| Status | ⚠️ Fallback — optimization_failure |
| Active benchmarks | 8 |
| View tickers incorporated | 5 |


---

## Interpretation

The point forecast leans neutral, and 1/8 (12%) benchmarks favour outperformance, but the broader quality gate is failing.

Recommended action at next vesting event: **DEFAULT 50% SALE** for diversification and tax discipline, not because the prediction is high-confidence.

---

## Redeploy Guidance

- Broad US Equity: VOO. Broad US equity diversifies away from single-stock risk without concentrating further in insurance.
- International Equity: VXUS, VWO. International equity lowers home-market and insurance concentration.
- Fixed Income: BND. Fixed income is the cleanest concentration-reduction bucket when model confidence is weak.
- Sector Context: VGT, SCHD. Sector funds are context-only unless no stronger diversifying destination is available.

## Suggested Redeploy Portfolio

- Default posture: `96%` equities / `4%` bonds across the curated investable universe.
- Monthly tilts use a `25%` signal overlay around the base weights, so the recommendation can adapt without becoming a full tactical allocation model.
- Investable universe used in the monthly workflow: `VOO, VGT, SCHD, VXUS, VWO, BND`.
- Constraint note: The current project universe does not yet include a dedicated small-cap ETF, so the value sleeve uses SCHD and the broad-market sleeve stays in VOO.

| Fund | Allocation | Sleeve | Why it is included | PGR Correlation | Relative Signal | P(Benchmark Beats PGR) |
|------|------------|--------|--------------------|-----------------|-----------------|------------------------|
| VOO | 40% | Broad US equity core | Core US beta sleeve that keeps the portfolio equity-heavy without recreating single-stock PGR risk. | 0.12 | Supportive (-1.1%) | n/a |
| VGT | 17% | Technology tilt | Growth engine and explicit tech tilt when the relative signal supports owning more innovation exposure than a pure core index. | 0.27 | Base-weight only (n/a) | n/a |
| VWO | 16% | Emerging-markets satellite | Higher-growth international sleeve kept modest because it is more volatile than the core international allocation. | 0.33 | Preferred this month (-5.1%) | n/a |
| SCHD | 13% | Value / dividend tilt | Closest current project proxy for a value sleeve; adds a cheaper, income-oriented counterweight to the tech allocation. | 0.37 | Base-weight only (n/a) | n/a |
| VXUS | 9% | International core | Primary geographic diversifier away from a US employer-stock concentration. | 0.26 | Only keep at floor weight (+3.3%) | 30.3% |
| BND | 4% | Bond ballast | Small stabilizer sleeve kept intentionally light so the redeploy portfolio stays above 90% equities in normal months. | 0.04 | Keep near base (+2.0%) | 28.6% |

## Per-Benchmark Signals

- Predicted Return is from the perspective of PGR versus each fund. Positive means PGR is expected to outperform that fund; negative means the fund is expected to outperform PGR.
- Benchmark Role distinguishes realistic buy candidates from contextual or forecast-only comparison funds.

| Benchmark | Benchmark Role | Description | Predicted Return | CI Lower | CI Upper | IC | Hit Rate | P(raw) | P(cal) | Confidence | Signal |
|-----------|----------------|-------------|----------------|----------|----------|----|----------|--------|--------|------------|--------|
| VOO | Buy candidate | S&P 500 | -1.14% | -26.39% | +24.12% | -0.0954 | 54.4% | 50.0% | 67.7% | LOW | NEUTRAL |
| VXUS | Buy candidate | Total International Stock | +3.32% | -32.55% | +39.19% | -0.1483 | 60.6% | 50.0% | 69.7% | LOW | NEUTRAL |
| VWO | Buy candidate | Emerging Markets | -5.09% | -34.25% | +24.07% | -0.0661 | 62.2% | 50.0% | 89.1% | LOW | NEUTRAL |
| VMBS | Forecast only | Mortgage-Backed Securities | +0.13% | -21.86% | +22.12% | 0.1891 | 77.1% | 50.0% | 58.8% | LOW | NEUTRAL |
| BND | Buy candidate | Total Bond Market | +1.96% | -19.04% | +22.96% | 0.2911 | 67.6% | 50.0% | 71.4% | LOW | OUTPERFORM |
| GLD | Forecast only | Gold Shares | -7.47% | -44.27% | +29.32% | 0.1711 | 60.2% | 50.0% | 49.9% | LOW | UNDERPERFORM |
| DBC | Forecast only | DB Commodity Index | -14.19% | -58.22% | +29.84% | 0.1437 | 69.6% | 50.0% | 36.9% | LOW | UNDERPERFORM |
| VDE | Forecast only | Energy | -13.47% | -46.23% | +19.29% | 0.0514 | 60.2% | 50.0% | 48.0% | LOW | UNDERPERFORM |

---

## Tax Context

| Parameter | Value |
|-----------|-------|
| STCG Rate (federal) | 37% |
| LTCG Rate (federal) | 20% |
| Tax-rate differential | 17% |
| **LTCG breakeven return** | **21.25%** |
| Current model prediction (6M) | -4.66% |
| P(outperform) | 61.5% |
| Next time-based vest | 2027-01-19 |
| Next performance vest | 2027-07-17 |

⚠️ **Model predicts negative return (-4.7%).**  Consider capital-loss harvesting scenario — a tax loss at 37% STCG rate can offset other gains.  See three-scenario analysis at vesting.

> **Breakeven formula:** `(STCG − LTCG) / (1 − LTCG)` — the minimum
> return needed on RSUs held to LTCG eligibility (366 days post-vest) to
> produce higher after-tax proceeds than selling immediately at STCG.
> Run `compute_three_scenarios()` at each vesting event for lot-specific analysis.

---

*Generated by `scripts/monthly_decision.py`*