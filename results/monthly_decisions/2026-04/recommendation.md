# PGR Monthly Decision Report — April 2026

**As-of Date:** 2026-04-11  
**Run Date:** 2026-04-11  
**Model Version:** v11.1 (lean 2-model ensemble: Ridge + GBT, v18 feature sets, 8-benchmark PRIMARY_FORECAST_UNIVERSE, inverse-variance weighting, v38 post-ensemble shrinkage alpha=0.50, C(8,2)=28 CPCV paths; ElasticNet+BayesianRidge retired after v18/v20 research showed Ridge+GBT outperforms on IC, hit rate, and obs/feature ratio)  
**Recommendation Layer:** Live production recommendation layer (quality-weighted consensus)  

---

## Executive Summary

- What changed since last month: Previous logged month (2026-04-02) was OUTPERFORM at +2.41% with mean IC 0.1197.
- Current model view: Consensus signal is NEUTRAL, but the average relative-return forecast is -2.34% across benchmarks over the next 6 months. Recommendation mode remains DEFER-TO-TAX-DEFAULT.
- How trustworthy it is: Model quality is too weak to justify a prediction-led vesting action. Aggregate health: OOS R^2 -2.75%, IC 0.2006, hit rate 68.1%.
- What to do at the next vest: Next vest is 2026-07-17 (performance). Default action today: sell 50% at vest unless model quality improves.
- What would change the recommendation: A more aggressive recommendation would require aggregate OOS R^2 >= 2%, mean IC >= 0.07, hit rate >= 55%, and a non-failing representative CPCV check.

---

## Data Freshness

> Some upstream data is stale or missing. Treat this run with extra caution until the feeds refresh.

| Feed | Latest Date | Age | Limit | Status |
|------|-------------|-----|-------|--------|
| Daily prices | 2026-04-02 | 9 days | 10 days | **OK** |
| FRED macro | 2026-04-30 | 0 days | 45 days | **OK** |
| PGR monthly EDGAR | 2026-02-28 | 42 days | 35 days | **STALE** |

Warnings:
- PGR monthly EDGAR is stale: latest 2026-02-28 (42 days old, limit 35).

---

## Decision At A Glance

- Hold vs Sell: **Hold 50% / Sell 50% of the next vest tranche**
- Is this month actionable? **No — follow the default tax/diversification rule.**
- Top-line decision: **Hold 50% / Sell 50% of the next vest tranche. No — follow the default tax/diversification rule.**
- Shadow classifier probability: **35.2%** (MODERATE)

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
| Predicted 6M Relative Return | -2.34% |
| P(Outperform, raw) | 50.0% |
| P(Outperform, calibrated) | 61.9% |
| 80% Prediction Interval (median) | -28.67% to +23.78% |
| Mean IC (across benchmarks) | 0.1744 |
| Mean Hit Rate | 66.8% |
| Aggregate OOS R^2 | -2.75% |

> **Note:** The sell % recommendation is used only at actual vesting events
> (January and July).  Monthly reports are monitoring tools, not trade signals.
>
> **Calibration:** Phase 2 — Platt scaling active (n=1,188 OOS obs).  ECE = 1.1% [95% CI: 1.4%–6.0%].

---

## Classification Confidence Check

> Shadow-only interpretation layer from the v87-v96 classifier research.
> It does not change the live recommendation or sell percentage.

| Field | Value |
|-------|-------|
| Target | actionable_sell_3pct |
| Construction | Separate benchmark logistic + quality-weighted aggregate |
| P(Actionable Sell) | 35.2% |
| Confidence Tier | MODERATE |
| Classifier Stance | NEUTRAL |
| Agreement with Live Recommendation | Aligned |
| Interpretation | Shadow classifier is near its neutral band (35.2%); use it as a low-confidence interpretation layer rather than a decision override. |

---

## Confidence Snapshot

- 2/4 core gates pass. The signal may still be directionally interesting, but the quality gate remains too weak for a prediction-led vest action.

| Check | Current | Threshold | Status | Meaning |
|-------|---------|-----------|--------|---------|
| Mean IC | 0.1744 | >= 0.0700 | **PASS** | Cross-benchmark ranking signal. |
| Mean hit rate | 66.8% | >= 55.0% | **PASS** | Directional accuracy versus zero. |
| Aggregate OOS R^2 | -2.75% | >= 2.00% | **FAIL** | Calibration / fit versus a naive benchmark. |
| Representative CPCV | FAIL | not FAIL | **FAIL** | Stability across purged cross-validation paths. |

---

## Model Health

- Latest tracked month: **2026-04-30**
- Rolling 12M IC: **0.2006**
- Rolling 12M Hit Rate: **68.1%**
- Rolling 12M ECE: **1.1%**
- IC breach streak: **0** month(s)
- Status: **Stable: no sustained rolling-IC drift alert is active.**

---

## Decision Policy Backtest

> OOS performance of each decision policy applied to all historical model predictions.  "Mean Return" is the portfolio-weighted realized relative return per vesting event.  "Cumulative" is the sum across all events.  "Capture Ratio" is the fraction of oracle (always hold when positive) gains captured.  N = number of OOS events.

### Fixed Heuristic Baselines

| Policy | N | Mean Return | Cumulative | Capture Ratio |
|--------|---|-------------|------------|---------------|
| Sell 100% (always) | 1188 | +0.00% | +0.00% | 0.0% |
| Sell 50% (always) | 1188 | +4.03% | +4792.31% | 33.4% |
| Hold 100% (always) | 1188 | +8.07% | +9584.62% | 66.8% |

### Model-Driven Policies vs. Heuristics

| Policy | N | Mean Return | Cumul. Return | Uplift vs Sell-All | Uplift vs Hold-All | Uplift vs 50% | Capture |
|--------|---|-------------|---------------|--------------------|--------------------|---------------|---------|
| Model: sign (hold if pred > 0) | 1188 | +7.88% | +9365.04% | +7.88% | -0.18% | +3.85% | 65.3% |
| Model: tiered 25/50/100 | 1188 | +2.25% | +2676.50% | +2.25% | -5.81% | -1.78% | 18.7% |
| Model: neutral band ±2% | 1188 | +7.55% | +8972.58% | +7.55% | -0.52% | +3.52% | 62.5% |
| Model: neutral band ±3% | 1188 | +7.12% | +8457.78% | +7.12% | -0.95% | +3.09% | 59.0% |


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

The point forecast leans neutral, and 1/8 (12%) benchmarks favour outperformance, but the broader quality gate is failing.

Recommended action at next vesting event: **DEFAULT 50% SALE** for diversification and tax discipline, not because the prediction is high-confidence.

---

## Next Vest Decision

| Field | Value |
|-------|-------|
| Recommendation mode | **DEFER-TO-TAX-DEFAULT** |
| Next vest date | 2026-07-17 |
| RSU type | performance |
| Current PGR price | $195.25 |
| Current in-scope shares | 1000.00 |
| Average cost basis used | $124.86 |
| Suggested default vest action | Sell 50% of the vesting tranche |

> Use the default diversification and tax-discipline rule rather than the point forecast.
> The scenario table below is provisional and uses the current lot file as a proxy for the next vesting decision.

### Tax timing scenarios (informational)

| Scenario | Timing | Tax Rate | Predicted Return | Probability | Use when |
|----------|--------|----------|------------------|-------------|----------|
| Sell at vest (STCG) | 2026-07-17 | 37% | +0.00% | 100.0% | Use the default diversification / tax-discipline rule or when the model edge is weak. |
| Hold to LTCG date | 2027-07-18 | 20% | -4.68% | 61.9% | Use only when the edge is strong enough to justify waiting for lower long-term tax treatment. |
| Hold for downside / loss case | 2027-01-13 | 37% | -2.34% | 38.1% | Use only when you are intentionally waiting for a downside or tax-loss outcome. |

> Tax-engine scenario ranking (informational only): **SELL_NOW_STCG**.
> Because recommendation mode is not ACTIONABLE, do not treat the tax-engine ranking below as a standalone trading instruction.
> STCG/LTCG breakeven from the tax engine: 21.25%.

### Monte Carlo Tax Sensitivity (HOLD_TO_LTCG vs. Sell Now)

> **1,000 GBM paths** | drift -4.7%/yr | vol 95.3%/yr | horizon 366 days

| Metric | Value |
|--------|-------|
| Sell-now reference (STCG net) | $169,208 |
| HOLD_TO_LTCG — P10 net | $52,735 |
| HOLD_TO_LTCG — median net | $120,029 |
| HOLD_TO_LTCG — mean net | $167,924 |
| HOLD_TO_LTCG — P90 net | $337,809 |
| P(HOLD_TO_LTCG beats Sell Now) | 30.2% |
| P(terminal price > cost basis) | 47.7% |

> At 95.3% annualised volatility and a -4.7%/yr drift, 30.2% of simulated paths produce higher after-tax net proceeds from holding to LTCG eligibility than from selling immediately at STCG rates.

---

## Existing Holdings Guidance

- STCG: 2027-01-19 @ $133.65 (500.00 share(s)). Avoid STCG gain lots unless the signal is unusually strong or concentration risk is urgent.
- STCG: 2026-07-17 @ $116.08 (500.00 share(s)). Avoid STCG gain lots unless the signal is unusually strong or concentration risk is urgent.

## Redeploy Guidance

- Broad US Equity: VOO. Broad US equity diversifies away from single-stock risk without concentrating further in insurance.
- International Equity: VXUS, VWO. International equity lowers home-market and insurance concentration.
- Fixed Income: BND. Fixed income is the cleanest concentration-reduction bucket when model confidence is weak.
- Sector Context: SCHD, VGT. Sector funds are context-only unless no stronger diversifying destination is available.

## Suggested Redeploy Portfolio

- Default posture: `95%` equities / `5%` bonds across the curated investable universe.
- Monthly tilts use a `25%` signal overlay around the base weights, so the recommendation can adapt without becoming a full tactical allocation model.
- Investable universe used in the monthly workflow: `VOO, VGT, SCHD, VXUS, VWO, BND`.
- Constraint note: The current project universe does not yet include a dedicated small-cap ETF, so the value sleeve uses SCHD and the broad-market sleeve stays in VOO.

| Fund | Allocation | Sleeve | Why it is included | PGR Correlation | Relative Signal | P(Benchmark Beats PGR) |
|------|------------|--------|--------------------|-----------------|-----------------|------------------------|
| VOO | 37% | Broad US equity core | Core US beta sleeve that keeps the portfolio equity-heavy without recreating single-stock PGR risk. | 0.14 | Keep near base (+0.5%) | 32.6% |
| VGT | 19% | Technology tilt | Growth engine and explicit tech tilt when the relative signal supports owning more innovation exposure than a pure core index. | 0.36 | Base-weight only (n/a) | n/a |
| VWO | 16% | Emerging-markets satellite | Higher-growth international sleeve kept modest because it is more volatile than the core international allocation. | 0.30 | Supportive (-2.4%) | n/a |
| SCHD | 14% | Value / dividend tilt | Closest current project proxy for a value sleeve; adds a cheaper, income-oriented counterweight to the tech allocation. | 0.29 | Base-weight only (n/a) | n/a |
| VXUS | 9% | International core | Primary geographic diversifier away from a US employer-stock concentration. | 0.28 | Only keep at floor weight (+3.6%) | 30.0% |
| BND | 5% | Bond ballast | Small stabilizer sleeve kept intentionally light so the redeploy portfolio stays above 90% equities in normal months. | 0.04 | Keep near base (+0.5%) | 30.6% |

## Per-Benchmark Signals

- Predicted Return is from the perspective of PGR versus each fund. Positive means PGR is expected to outperform that fund; negative means the fund is expected to outperform PGR.
- Benchmark Role distinguishes realistic buy candidates from contextual or forecast-only comparison funds.

| Benchmark | Benchmark Role | Description | Predicted Return | CI Lower | CI Upper | IC | Hit Rate | P(raw) | P(cal) | Confidence | Signal |
|-----------|----------------|-------------|----------------|----------|----------|----|----------|--------|--------|------------|--------|
| VOO | Buy candidate | S&P 500 | +0.54% | -26.57% | +27.65% | 0.0714 | 51.4% | 50.0% | 67.4% | LOW | NEUTRAL |
| VXUS | Buy candidate | Total International Stock | +3.63% | -32.70% | +39.95% | 0.1889 | 69.9% | 50.0% | 70.0% | LOW | OUTPERFORM |
| VWO | Buy candidate | Emerging Markets | -2.37% | -30.76% | +26.02% | 0.2111 | 62.9% | 50.0% | 69.8% | LOW | UNDERPERFORM |
| VMBS | Forecast only | Mortgage-Backed Securities | +0.33% | -20.20% | +20.86% | 0.2286 | 78.8% | 50.0% | 64.5% | LOW | NEUTRAL |
| BND | Buy candidate | Total Bond Market | +0.54% | -20.48% | +21.55% | 0.2472 | 71.0% | 50.0% | 69.4% | LOW | NEUTRAL |
| GLD | Forecast only | Gold Shares | -0.02% | -26.45% | +26.42% | 0.1474 | 60.0% | 50.0% | 66.3% | LOW | NEUTRAL |
| DBC | Forecast only | DB Commodity Index | -9.53% | -36.29% | +17.24% | 0.1448 | 73.8% | 50.0% | 46.8% | LOW | UNDERPERFORM |
| VDE | Forecast only | Energy | -12.72% | -43.24% | +17.79% | 0.1232 | 62.8% | 50.0% | 41.2% | LOW | UNDERPERFORM |

---

## Tax Context

| Parameter | Value |
|-----------|-------|
| STCG Rate (federal) | 37% |
| LTCG Rate (federal) | 20% |
| Tax-rate differential | 17% |
| **LTCG breakeven return** | **21.25%** |
| Current model prediction (6M) | -2.34% |
| P(outperform) | 61.9% |
| Next time-based vest | 2027-01-19 |
| Next performance vest | 2026-07-17 |

⚠️ **Model predicts negative return (-2.3%).**  Consider capital-loss harvesting scenario — a tax loss at 37% STCG rate can offset other gains.  See three-scenario analysis at vesting.

> **Breakeven formula:** `(STCG − LTCG) / (1 − LTCG)` — the minimum
> return needed on RSUs held to LTCG eligibility (366 days post-vest) to
> produce higher after-tax proceeds than selling immediately at STCG.
> Run `compute_three_scenarios()` at each vesting event for lot-specific analysis.

---

*Generated by `scripts/monthly_decision.py`*  [DRY RUN]