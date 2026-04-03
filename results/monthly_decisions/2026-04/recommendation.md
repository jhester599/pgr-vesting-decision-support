# PGR Monthly Decision Report — April 2026

**As-of Date:** 2026-04-02  
**Run Date:** 2026-04-03  
**Model Version:** v8.13 (4-model ensemble: ElasticNet + Ridge + BayesianRidge + GBT, inverse-variance weighting, C(8,2)=28 CPCV paths, v8 reliability, communication, and model-quality gating refresh)  

---

## Executive Summary

- What changed since last month: Previous logged month (2026-03-27) was NEUTRAL at +1.75% with mean IC 0.0421.
- Current model view: OUTPERFORM with moderate confidence and a 6M relative-return estimate of +4.18%.
- How trustworthy it is: Model quality is too weak to justify a prediction-led vesting action. Aggregate health: OOS R^2 -123.80%, IC 0.1255, hit rate 56.8%.
- What to do at the next vest: Next vest is 2026-07-17 (performance). Default action today: sell 50% at vest unless model quality improves.
- What would change the recommendation: A more aggressive recommendation would require aggregate OOS R^2 >= 2%, mean IC >= 0.07, hit rate >= 55%, and a non-failing representative CPCV check.

---

## Consensus Signal

| Field | Value |
|-------|-------|
| Signal | **OUTPERFORM (MODERATE CONFIDENCE)** |
| Recommendation Mode | **DEFER-TO-TAX-DEFAULT** |
| Recommended Sell % | **50%** |
| Predicted 6M Relative Return | +4.18% |
| P(Outperform, raw) | 64.0% |
| P(Outperform, calibrated) | 67.4% |
| 80% Prediction Interval (median) | -29.14% to +40.41% |
| Mean IC (across benchmarks) | 0.1065 |
| Mean Hit Rate | 56.4% |

> **Note:** The sell % recommendation is used only at actual vesting events
> (January and July).  Monthly reports are monitoring tools, not trade signals.
>
> **Calibration:** Phase 2 — Platt scaling active (n=3,270 OOS obs).  ECE = 1.0% [95% CI: 1.0%–4.2%].

---

## Interpretation

The point forecast leans outperform, and 11/21 (52%) benchmarks favour outperformance, but the broader quality gate is failing.

Recommended action at next vesting event: **DEFAULT 50% SALE** for diversification and tax discipline, not because the prediction is high-confidence.

---

## Per-Benchmark Signals

## Next Vest Decision

| Field | Value |
|-------|-------|
| Recommendation mode | **DEFER-TO-TAX-DEFAULT** |
| Next vest date | 2026-07-17 |
| RSU type | performance |
| Current PGR price | $198.84 |
| Current in-scope shares | 8.00 |
| Average cost basis used | $133.38 |
| Suggested default vest action | Sell 50% of the vesting tranche |

> Use the default diversification and tax-discipline rule rather than the point forecast.
> The scenario table below is provisional and uses the current lot file as a proxy for the next vesting decision.

| Scenario | Sell Date | Tax Rate | Predicted Return | Net Proceeds | Probability |
|----------|-----------|----------|------------------|--------------|-------------|
| SELL_NOW_STCG | 2026-07-17 | 37% | +0.00% | $1,396.94 | 100.0% |
| HOLD_TO_LTCG | 2027-07-18 | 20% | +8.36% | $1,592.33 | 67.4% |
| HOLD_FOR_LOSS | 2027-01-13 | 37% | +4.18% | $1,590.72 | 0.0% |

> Tax-engine scenario ranking (informational only): **SELL_NOW_STCG**.
> Because recommendation mode is not ACTIONABLE, do not treat the tax-engine ranking below as a standalone trading instruction.
> STCG/LTCG breakeven from the tax engine: 21.25%.

---

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | IC | Hit Rate | P(raw) | P(cal) | Confidence | Signal |
|-----------|-------------|----------------|----------|----------|----|----------|--------|--------|------------|--------|
| VTI | Total Stock Market | +6.95% | -29.14% | +43.05% | 0.0407 | 51.6% | 70.9% | 59.3% | HIGH | NEUTRAL |
| VOO | S&P 500 | +0.65% | -34.51% | +35.82% | 0.1574 | 47.5% | 52.1% | 68.5% | LOW | NEUTRAL |
| VGT | Information Technology | -2.42% | -41.10% | +36.25% | 0.0802 | 52.6% | 44.3% | 54.0% | LOW | UNDERPERFORM |
| VHT | Health Care | +6.03% | -28.35% | +40.41% | 0.1665 | 58.5% | 68.3% | 66.1% | MODERATE | OUTPERFORM |
| VFH | Financials | +8.80% | -24.67% | +42.26% | 0.1511 | 53.6% | 80.1% | 62.0% | HIGH | OUTPERFORM |
| VIS | Industrials | +3.02% | -34.34% | +40.38% | 0.1192 | 51.5% | 59.6% | 63.5% | LOW | OUTPERFORM |
| VDE | Energy | -6.93% | -54.08% | +40.21% | -0.0259 | 53.5% | 34.1% | 66.3% | MODERATE | NEUTRAL |
| VPU | Utilities | +3.43% | -27.14% | +34.00% | 0.2303 | 57.3% | 60.5% | 65.4% | MODERATE | OUTPERFORM |
| KIE | S&P Insurance | +5.85% | -16.76% | +28.47% | 0.1463 | 56.4% | 73.8% | 61.7% | HIGH | OUTPERFORM |
| VXUS | Total International Stock | +6.34% | -38.55% | +51.24% | 0.0349 | 58.8% | 71.2% | 69.7% | HIGH | NEUTRAL |
| VEA | Developed Markets ex-US | +4.04% | -40.84% | +48.92% | 0.0477 | 58.5% | 64.0% | 68.6% | MODERATE | NEUTRAL |
| VWO | Emerging Markets | +6.24% | -32.58% | +45.06% | 0.0389 | 55.3% | 69.6% | 69.9% | MODERATE | NEUTRAL |
| VIG | Dividend Appreciation | +4.09% | -25.43% | +33.61% | 0.1390 | 54.3% | 63.7% | 65.3% | MODERATE | OUTPERFORM |
| SCHD | US Dividend Equity | -1.53% | -78.06% | +75.01% | -0.0871 | 52.1% | 47.4% | 71.7% | LOW | NEUTRAL |
| BND | Total Bond Market | +11.26% | -20.35% | +42.87% | 0.1452 | 63.2% | 84.3% | 75.9% | HIGH | OUTPERFORM |
| BNDX | Total International Bond | +11.47% | -18.16% | +41.09% | 0.1751 | 62.2% | 84.9% | 73.7% | HIGH | OUTPERFORM |
| VCIT | Intermediate-Term Corporate Bond | +10.75% | -21.69% | +43.19% | 0.0751 | 61.2% | 83.4% | 75.6% | HIGH | OUTPERFORM |
| VMBS | Mortgage-Backed Securities | +10.17% | -21.89% | +42.23% | 0.1147 | 64.2% | 82.2% | 76.0% | HIGH | OUTPERFORM |
| VNQ | Real Estate | +4.00% | -24.13% | +32.12% | 0.2237 | 57.1% | 62.8% | 61.5% | MODERATE | OUTPERFORM |
| GLD | Gold Shares | -2.19% | -44.30% | +39.93% | 0.1631 | 55.7% | 42.9% | 66.4% | LOW | UNDERPERFORM |
| DBC | DB Commodity Index | -2.29% | -37.45% | +32.87% | 0.0996 | 60.5% | 43.9% | 74.6% | LOW | UNDERPERFORM |

---

## Tax Context

| Parameter | Value |
|-----------|-------|
| STCG Rate (federal) | 37% |
| LTCG Rate (federal) | 20% |
| Tax-rate differential | 17% |
| **LTCG breakeven return** | **21.25%** |
| Current model prediction (6M) | +4.18% |
| P(outperform) | 67.4% |
| Next time-based vest | 2027-01-19 |
| Next performance vest | 2026-07-17 |

✓ **Model prediction (+4.2%) is below the LTCG breakeven (21.2%) by 17.1%.**  Holding RSUs for 366 days post-vest to qualify for LTCG treatment is likely the higher after-tax outcome.

> **Breakeven formula:** `(STCG − LTCG) / (1 − LTCG)` — the minimum
> return needed on RSUs held to LTCG eligibility (366 days post-vest) to
> produce higher after-tax proceeds than selling immediately at STCG.
> Run `compute_three_scenarios()` at each vesting event for lot-specific analysis.

---

*Generated by `scripts/monthly_decision.py`*