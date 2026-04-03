# PGR Diagnostic Report — April 2026

**As-of Date:** 2026-04-02  
**Horizon:** 6M  
**OOS observations (aggregate):** 3270  
**Newey-West lags:** 5 (accounts for 5-month return-window overlap)  

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² (Campbell-Thompson) | -1.2380 (-123.80%) | ❌ | ≥ 2.00% |
| IC (Newey-West HAC) | 0.1255 | ✅ | ≥ 0.07 |
| IC significance | 0.0000 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 56.8% | ✅ | ≥ 55.0% |
| CPCV Positive Paths | 0/7 (0.0%) | ❌ | ≥ 5/7 |

> **Representative CPCV:** benchmark=VTI, model=elasticnet, paths=7, mean IC=-0.2946, IC std=0.0868.
> Stability verdict: FAIL. Scaled monthly threshold: ≥ 5/7 (maps from the full C(8,2) standard of ≥ 19/28 positive paths).

---


## Feature Governance

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Full obs/feature ratio | 1.53 | ❌ | ≥ 4.0 |
| Per-fold obs/feature ratio | 1.58 | ❌ | ≥ 4.0 |
| Features in monthly run | 38 | — | — |
| Fully populated observations | 58 | — | — |

> obs/feature ratio: 1.5 (full matrix), 1.6 (per WFO fold, 60M window).  n_obs=58, n_features=38.  Verdict: FAIL.

---

## Calibration Phase

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Raw BayesianRidge posterior (uncalibrated) | ⬛ Superseded |
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (n=3,270  ECE=1.0% [1.0%–4.2%]) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

**Mean empirical coverage:** 93.0% (target ≥ 80%) ✅  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|---------------|-------|
| VTI | Total Stock Market | +6.95% | -29.14% | +43.05% | 72.20% | 89.6% ✅ | 222 |
| VOO | S&P 500 | +0.65% | -34.51% | +35.82% | 70.33% | 87.0% ✅ | 108 |
| VGT | Information Technology | -2.42% | -41.10% | +36.25% | 77.35% | 94.6% ✅ | 186 |
| VHT | Health Care | +6.03% | -28.35% | +40.41% | 68.76% | 95.7% ✅ | 186 |
| VFH | Financials | +8.80% | -24.67% | +42.26% | 66.93% | 93.5% ✅ | 186 |
| VIS | Industrials | +3.02% | -34.34% | +40.38% | 74.72% | 95.0% ✅ | 180 |
| VDE | Energy | -6.93% | -54.08% | +40.21% | 94.29% | 91.7% ✅ | 180 |
| VPU | Utilities | +3.43% | -27.14% | +34.00% | 61.14% | 90.9% ✅ | 186 |
| KIE | S&P Insurance | +5.85% | -16.76% | +28.47% | 45.24% | 76.8% ⚠️ | 168 |
| VXUS | Total International Stock | +6.34% | -38.55% | +51.24% | 89.78% | 99.0% ✅ | 102 |
| VEA | Developed Markets ex-US | +4.04% | -40.84% | +48.92% | 89.76% | 99.3% ✅ | 144 |
| VWO | Emerging Markets | +6.24% | -32.58% | +45.06% | 77.63% | 97.7% ✅ | 174 |
| VIG | Dividend Appreciation | +4.09% | -25.43% | +33.61% | 59.04% | 95.7% ✅ | 162 |
| SCHD | US Dividend Equity | -1.53% | -78.06% | +75.01% | 153.07% | 94.8% ✅ | 96 |
| BND | Total Bond Market | +11.26% | -20.35% | +42.87% | 63.22% | 96.7% ✅ | 150 |
| BNDX | Total International Bond | +11.47% | -18.16% | +41.09% | 59.26% | 94.9% ✅ | 78 |
| VCIT | Intermediate-Term Corporate Bond | +10.75% | -21.69% | +43.19% | 64.88% | 95.0% ✅ | 120 |
| VMBS | Mortgage-Backed Securities | +10.17% | -21.89% | +42.23% | 64.12% | 97.5% ✅ | 120 |
| VNQ | Real Estate | +4.00% | -24.13% | +32.12% | 56.25% | 86.7% ✅ | 180 |
| GLD | Gold Shares | -2.19% | -44.30% | +39.93% | 84.24% | 91.7% ✅ | 180 |
| DBC | DB Commodity Index | -2.29% | -37.45% | +32.87% | 70.32% | 90.1% ✅ | 162 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | IC | IC | Hit Rate | HR |
|-----------|-------------|-------|----|----|-----------|----|
| BND | Total Bond Market | 150 | 0.1169 | ✅ | 66.0% | ✅ |
| BNDX | Total International Bond | 78 | 0.0388 | ⚠️ | 56.4% | ✅ |
| DBC | DB Commodity Index | 162 | 0.1238 | ✅ | 59.9% | ✅ |
| GLD | Gold Shares | 180 | 0.1947 | ✅ | 60.0% | ✅ |
| KIE | S&P Insurance | 168 | 0.1776 | ✅ | 53.0% | ⚠️ |
| SCHD | US Dividend Equity | 96 | -0.0794 | ❌ | 56.2% | ✅ |
| VCIT | Intermediate-Term Corporate Bond | 120 | 0.0399 | ⚠️ | 60.0% | ✅ |
| VDE | Energy | 180 | -0.0492 | ❌ | 54.4% | ⚠️ |
| VEA | Developed Markets ex-US | 144 | 0.0099 | ❌ | 65.3% | ✅ |
| VFH | Financials | 186 | 0.0125 | ❌ | 52.2% | ⚠️ |
| VGT | Information Technology | 186 | -0.0284 | ❌ | 48.9% | ❌ |
| VHT | Health Care | 186 | 0.1752 | ✅ | 57.5% | ✅ |
| VIG | Dividend Appreciation | 162 | 0.0747 | ✅ | 56.2% | ✅ |
| VIS | Industrials | 180 | 0.0568 | ⚠️ | 48.9% | ❌ |
| VMBS | Mortgage-Backed Securities | 120 | 0.1157 | ✅ | 65.0% | ✅ |
| VNQ | Real Estate | 180 | 0.2511 | ✅ | 59.4% | ✅ |
| VOO | S&P 500 | 108 | 0.0839 | ✅ | 40.7% | ❌ |
| VPU | Utilities | 186 | 0.2719 | ✅ | 61.3% | ✅ |
| VTI | Total Stock Market | 222 | 0.0640 | ⚠️ | 49.5% | ❌ |
| VWO | Emerging Markets | 174 | -0.0224 | ❌ | 60.9% | ✅ |
| VXUS | Total International Stock | 102 | -0.0977 | ❌ | 67.6% | ✅ |

**IC summary:** 10 ✅  4 ⚠️  7 ❌  (of 21 benchmarks)  
**Hit rate ✅:** 14/21 benchmarks above 55% threshold  

---

## Threshold Reference

| Metric | Good | Marginal | Failing | Source |
|--------|------|----------|---------|--------|
| OOS R² | > 2% | 0.5–2% | < 0% | Campbell & Thompson (2008) |
| Mean IC | > 0.07 | 0.03–0.07 | < 0.03 | Harvey et al. (2016) |
| Hit Rate | > 55% | 52–55% | < 52% | Industry consensus |
| CPCV +paths | ≥ 19/28 | 14–18/28 | < 14/28 | López de Prado (2018) |
| PBO | < 15% | 15–40% | > 40% | Bailey et al. (2014) |

---

*Generated by `scripts/monthly_decision.py`*