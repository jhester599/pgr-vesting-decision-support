# PGR Diagnostic Report — April 2026

**As-of Date:** 2026-04-02  
**Horizon:** 6M  
**OOS observations (aggregate):** 3312  
**Newey-West lags:** 5 (accounts for 5-month return-window overlap)  

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² (Campbell-Thompson) | -1.1870 (-118.70%) | ❌ | ≥ 2.00% |
| IC (Newey-West HAC) | 0.1348 | ✅ | ≥ 0.07 |
| IC significance | 0.0000 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 56.6% | ✅ | ≥ 55.0% |
| CPCV Positive Paths | 0/7 (0.0%) | ❌ | ≥ 5/7 |

> **Representative CPCV:** benchmark=VTI, model=elasticnet, paths=7, mean IC=-0.2929, IC std=0.0877.
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
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (n=3,312  ECE=1.8% [1.0%–4.5%]) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

**Mean empirical coverage:** 92.8% (target ≥ 80%) ✅  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|---------------|-------|
| VTI | Total Stock Market | +4.61% | -25.39% | +34.61% | 60.00% | 89.6% ✅ | 222 |
| VOO | S&P 500 | +1.85% | -30.07% | +33.78% | 63.85% | 86.1% ✅ | 108 |
| VGT | Information Technology | -0.56% | -37.36% | +36.23% | 73.59% | 94.8% ✅ | 192 |
| VHT | Health Care | +3.50% | -31.18% | +38.17% | 69.35% | 95.8% ✅ | 192 |
| VFH | Financials | +6.36% | -20.38% | +33.11% | 53.49% | 90.6% ✅ | 192 |
| VIS | Industrials | +3.97% | -40.66% | +48.60% | 89.26% | 97.8% ✅ | 180 |
| VDE | Energy | -10.61% | -58.87% | +37.66% | 96.53% | 91.7% ✅ | 180 |
| VPU | Utilities | -0.37% | -28.08% | +27.35% | 55.44% | 88.5% ✅ | 192 |
| KIE | S&P Insurance | +6.62% | -17.29% | +30.52% | 47.81% | 81.5% ✅ | 168 |
| VXUS | Total International Stock | +4.13% | -39.83% | +48.10% | 87.94% | 98.1% ✅ | 108 |
| VEA | Developed Markets ex-US | +2.50% | -41.89% | +46.89% | 88.78% | 98.7% ✅ | 150 |
| VWO | Emerging Markets | +6.22% | -32.81% | +45.26% | 78.07% | 97.7% ✅ | 174 |
| VIG | Dividend Appreciation | +2.99% | -24.31% | +30.30% | 54.61% | 95.1% ✅ | 162 |
| SCHD | US Dividend Equity | +0.12% | -60.16% | +60.40% | 120.56% | 92.7% ✅ | 96 |
| BND | Total Bond Market | +7.11% | -23.72% | +37.93% | 61.65% | 96.0% ✅ | 150 |
| BNDX | Total International Bond | +9.00% | -21.39% | +39.38% | 60.77% | 94.9% ✅ | 78 |
| VCIT | Intermediate-Term Corporate Bond | +9.22% | -22.69% | +41.13% | 63.82% | 95.8% ✅ | 120 |
| VMBS | Mortgage-Backed Securities | +8.27% | -21.84% | +38.37% | 60.21% | 95.0% ✅ | 120 |
| VNQ | Real Estate | +5.47% | -24.33% | +35.26% | 59.58% | 90.6% ✅ | 180 |
| GLD | Gold Shares | -15.08% | -65.89% | +35.73% | 101.62% | 93.9% ✅ | 180 |
| DBC | DB Commodity Index | -4.65% | -39.64% | +30.33% | 69.97% | 83.9% ✅ | 168 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | IC | IC | Hit Rate | HR |
|-----------|-------------|-------|----|----|-----------|----|
| BND | Total Bond Market | 150 | 0.1073 | ✅ | 62.0% | ✅ |
| BNDX | Total International Bond | 78 | 0.0518 | ⚠️ | 59.0% | ✅ |
| DBC | DB Commodity Index | 168 | 0.0461 | ⚠️ | 66.1% | ✅ |
| GLD | Gold Shares | 180 | 0.2319 | ✅ | 63.3% | ✅ |
| KIE | S&P Insurance | 168 | 0.1396 | ✅ | 53.0% | ⚠️ |
| SCHD | US Dividend Equity | 96 | 0.0961 | ✅ | 54.2% | ⚠️ |
| VCIT | Intermediate-Term Corporate Bond | 120 | 0.0646 | ⚠️ | 59.2% | ✅ |
| VDE | Energy | 180 | 0.0422 | ⚠️ | 56.7% | ✅ |
| VEA | Developed Markets ex-US | 150 | 0.0116 | ❌ | 62.7% | ✅ |
| VFH | Financials | 192 | 0.1022 | ✅ | 54.2% | ⚠️ |
| VGT | Information Technology | 192 | 0.0362 | ⚠️ | 46.4% | ❌ |
| VHT | Health Care | 192 | 0.1022 | ✅ | 55.7% | ✅ |
| VIG | Dividend Appreciation | 162 | 0.0745 | ✅ | 51.2% | ❌ |
| VIS | Industrials | 180 | 0.0951 | ✅ | 50.0% | ❌ |
| VMBS | Mortgage-Backed Securities | 120 | 0.1090 | ✅ | 65.0% | ✅ |
| VNQ | Real Estate | 180 | 0.2918 | ✅ | 61.7% | ✅ |
| VOO | S&P 500 | 108 | -0.0731 | ❌ | 36.1% | ❌ |
| VPU | Utilities | 192 | 0.2521 | ✅ | 58.3% | ✅ |
| VTI | Total Stock Market | 222 | 0.0233 | ❌ | 49.1% | ❌ |
| VWO | Emerging Markets | 174 | 0.0660 | ⚠️ | 61.5% | ✅ |
| VXUS | Total International Stock | 108 | -0.0741 | ❌ | 66.7% | ✅ |

**IC summary:** 11 ✅  6 ⚠️  4 ❌  (of 21 benchmarks)  
**Hit rate ✅:** 13/21 benchmarks above 55% threshold  

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