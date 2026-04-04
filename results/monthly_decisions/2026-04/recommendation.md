# PGR Monthly Decision Report — April 2026

**As-of Date:** 2026-04-02  
**Run Date:** 2026-04-03  
**Model Version:** v10.1 (4-model ensemble: ElasticNet + Ridge + BayesianRidge + GBT, inverse-variance weighting, C(8,2)=28 CPCV paths, post-v9 baseline reconciliation, migrations, CI/workflow hardening, and run-manifest support)  

---

## Executive Summary

- What changed since last month: Previous logged month (2026-03-27) was NEUTRAL at +1.75% with mean IC 0.0421.
- Current model view: OUTPERFORM with low confidence and a 6M relative-return estimate of +2.41%.
- How trustworthy it is: Model quality is too weak to justify a prediction-led vesting action. Aggregate health: OOS R^2 -118.70%, IC 0.1348, hit rate 56.6%.
- What to do at the next vest: Next vest is 2026-07-17 (performance). Default action today: sell 50% at vest unless model quality improves.
- What would change the recommendation: A more aggressive recommendation would require aggregate OOS R^2 >= 2%, mean IC >= 0.07, hit rate >= 55%, and a non-failing representative CPCV check.

---

## Consensus Signal

| Field | Value |
|-------|-------|
| Signal | **OUTPERFORM (LOW CONFIDENCE)** |
| Recommendation Mode | **DEFER-TO-TAX-DEFAULT** |
| Recommended Sell % | **50%** |
| Predicted 6M Relative Return | +2.41% |
| P(Outperform, raw) | 59.6% |
| P(Outperform, calibrated) | 66.4% |
| 80% Prediction Interval (median) | -30.07% to +37.66% |
| Mean IC (across benchmarks) | 0.1197 |
| Mean Hit Rate | 56.3% |

> **Note:** The sell % recommendation is used only at actual vesting events
> (January and July).  Monthly reports are monitoring tools, not trade signals.
>
> **Calibration:** Phase 2 — Platt scaling active (n=3,312 OOS obs).  ECE = 1.8% [95% CI: 1.0%–4.5%].

---

## Interpretation

The point forecast leans outperform, and 14/21 (67%) benchmarks favour outperformance, but the broader quality gate is failing.

Recommended action at next vesting event: **DEFAULT 50% SALE** for diversification and tax discipline, not because the prediction is high-confidence.

---

## Per-Benchmark Signals

## Next Vest Decision

| Field | Value |
|-------|-------|
| Recommendation mode | **DEFER-TO-TAX-DEFAULT** |
| Next vest date | 2026-07-17 |
| RSU type | performance |
| Current PGR price | $195.25 |
| Current in-scope shares | 8.00 |
| Average cost basis used | $133.38 |
| Suggested default vest action | Sell 50% of the vesting tranche |

> Use the default diversification and tax-discipline rule rather than the point forecast.
> The scenario table below is provisional and uses the current lot file as a proxy for the next vesting decision.

| Scenario | Sell Date | Tax Rate | Predicted Return | Net Proceeds | Probability |
|----------|-----------|----------|------------------|--------------|-------------|
| SELL_NOW_STCG | 2026-07-17 | 37% | +0.00% | $1,378.85 | 100.0% |
| HOLD_TO_LTCG | 2027-07-18 | 20% | +4.83% | $1,523.30 | 66.4% |
| HOLD_FOR_LOSS | 2027-01-13 | 37% | +2.41% | $1,562.00 | 0.0% |

> Tax-engine scenario ranking (informational only): **SELL_NOW_STCG**.
> Because recommendation mode is not ACTIONABLE, do not treat the tax-engine ranking below as a standalone trading instruction.
> STCG/LTCG breakeven from the tax engine: 21.25%.

---

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | IC | Hit Rate | P(raw) | P(cal) | Confidence | Signal |
|-----------|-------------|----------------|----------|----------|----|----------|--------|--------|------------|--------|
| VTI | Total Stock Market | +4.61% | -25.39% | +34.61% | 0.0353 | 49.9% | 64.5% | 59.5% | MODERATE | NEUTRAL |
| VOO | S&P 500 | +1.85% | -30.07% | +33.78% | 0.0850 | 44.2% | 55.9% | 67.1% | LOW | OUTPERFORM |
| VGT | Information Technology | -0.56% | -37.36% | +36.23% | 0.1341 | 53.6% | 48.7% | 55.6% | LOW | NEUTRAL |
| VHT | Health Care | +3.50% | -31.18% | +38.17% | 0.1327 | 56.4% | 61.2% | 63.9% | MODERATE | OUTPERFORM |
| VFH | Financials | +6.36% | -20.38% | +33.11% | 0.1995 | 56.1% | 73.4% | 62.3% | HIGH | OUTPERFORM |
| VIS | Industrials | +3.97% | -40.66% | +48.60% | 0.1080 | 51.5% | 62.7% | 63.7% | MODERATE | OUTPERFORM |
| VDE | Energy | -10.61% | -58.87% | +37.66% | -0.0246 | 54.4% | 26.7% | 65.8% | HIGH | NEUTRAL |
| VPU | Utilities | -0.37% | -28.08% | +27.35% | 0.2346 | 56.1% | 48.8% | 64.6% | LOW | NEUTRAL |
| KIE | S&P Insurance | +6.62% | -17.29% | +30.52% | 0.1605 | 57.7% | 77.1% | 61.8% | HIGH | OUTPERFORM |
| VXUS | Total International Stock | +4.13% | -39.83% | +48.10% | 0.0738 | 61.6% | 64.5% | 70.1% | MODERATE | OUTPERFORM |
| VEA | Developed Markets ex-US | +2.50% | -41.89% | +46.89% | 0.1067 | 59.7% | 59.0% | 65.2% | LOW | OUTPERFORM |
| VWO | Emerging Markets | +6.22% | -32.81% | +45.26% | 0.0927 | 56.0% | 69.8% | 69.5% | MODERATE | OUTPERFORM |
| VIG | Dividend Appreciation | +2.99% | -24.31% | +30.30% | 0.1008 | 50.6% | 60.3% | 64.9% | MODERATE | OUTPERFORM |
| SCHD | US Dividend Equity | +0.12% | -60.16% | +60.40% | -0.0734 | 51.8% | 50.2% | 71.6% | LOW | NEUTRAL |
| BND | Total Bond Market | +7.11% | -23.72% | +37.93% | 0.1666 | 61.7% | 74.3% | 75.3% | HIGH | OUTPERFORM |
| BNDX | Total International Bond | +9.00% | -21.39% | +39.38% | 0.2340 | 59.6% | 79.6% | 71.9% | HIGH | OUTPERFORM |
| VCIT | Intermediate-Term Corporate Bond | +9.22% | -22.69% | +41.13% | 0.1238 | 59.6% | 80.3% | 75.2% | HIGH | OUTPERFORM |
| VMBS | Mortgage-Backed Securities | +8.27% | -21.84% | +38.37% | 0.1686 | 65.4% | 77.8% | 75.5% | HIGH | OUTPERFORM |
| VNQ | Real Estate | +5.47% | -24.33% | +35.26% | 0.2295 | 57.2% | 67.7% | 62.5% | MODERATE | OUTPERFORM |
| GLD | Gold Shares | -15.08% | -65.89% | +35.73% | 0.1799 | 58.9% | 10.8% | 56.4% | HIGH | UNDERPERFORM |
| DBC | DB Commodity Index | -4.65% | -39.64% | +30.33% | 0.0450 | 61.0% | 37.8% | 72.2% | MODERATE | NEUTRAL |

---

## Tax Context

| Parameter | Value |
|-----------|-------|
| STCG Rate (federal) | 37% |
| LTCG Rate (federal) | 20% |
| Tax-rate differential | 17% |
| **LTCG breakeven return** | **21.25%** |
| Current model prediction (6M) | +2.41% |
| P(outperform) | 66.4% |
| Next time-based vest | 2027-01-19 |
| Next performance vest | 2026-07-17 |

✓ **Model prediction (+2.4%) is below the LTCG breakeven (21.2%) by 18.8%.**  Holding RSUs for 366 days post-vest to qualify for LTCG treatment is likely the higher after-tax outcome.

> **Breakeven formula:** `(STCG − LTCG) / (1 − LTCG)` — the minimum
> return needed on RSUs held to LTCG eligibility (366 days post-vest) to
> produce higher after-tax proceeds than selling immediately at STCG.
> Run `compute_three_scenarios()` at each vesting event for lot-specific analysis.

---

*Generated by `scripts/monthly_decision.py`*  [DRY RUN]