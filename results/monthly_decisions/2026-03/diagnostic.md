# PGR Diagnostic Report — March 2026

**As-of Date:** 2026-03-20  
**Horizon:** 6M  
**OOS observations (aggregate):** 3270  
**Newey-West lags:** 5 (accounts for 5-month return-window overlap)  

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² (Campbell-Thompson) | -0.8472 (-84.72%) | ❌ | ≥ 2.00% |
| IC (Newey-West HAC) | 0.0779 | ✅ | ≥ 0.07 |
| IC significance | 0.0117 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 54.6% | ⚠️ | ≥ 55.0% |
| CPCV Positive Paths | N/A (Phase 1) | — | ≥ 19/28 |

> **CPCV note (v5.0):** C(8,2)=28 paths are configured but not run inside the monthly
> workflow — 4 models × 28 splits × 20 benchmarks = 2,240 fits per run.
> CPCV diagnostics are available on demand via `run_cpcv()` in `wfo_engine.py`.
> The DIAG_CPCV_MIN_POSITIVE_PATHS threshold (≥ 19/28) is defined in `config.py`.

---

## Calibration Phase

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Raw BayesianRidge posterior (uncalibrated) | ⬛ Superseded |
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (n=3,270  ECE=2.1% [1.1%–4.9%]) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

**Mean empirical coverage:** 92.0% (target ≥ 80%) ✅  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|---------------|-------|
| VTI | Total Stock Market | -1.72% | -45.42% | +41.98% | 87.40% | 94.6% ✅ | 222 |
| VOO | S&P 500 | +0.82% | -30.51% | +32.15% | 62.66% | 84.3% ✅ | 108 |
| VGT | Information Technology | -3.10% | -41.15% | +34.96% | 76.11% | 95.7% ✅ | 186 |
| VHT | Health Care | +4.54% | -30.37% | +39.45% | 69.82% | 95.7% ✅ | 186 |
| VFH | Financials | +5.58% | -19.76% | +30.91% | 50.67% | 83.9% ✅ | 186 |
| VIS | Industrials | +0.81% | -36.37% | +38.00% | 74.37% | 96.7% ✅ | 180 |
| VDE | Energy | +7.61% | -31.76% | +46.98% | 78.74% | 87.8% ✅ | 180 |
| VPU | Utilities | +6.07% | -28.97% | +41.10% | 70.07% | 95.7% ✅ | 186 |
| KIE | S&P Insurance | +2.55% | -22.88% | +27.98% | 50.87% | 81.5% ✅ | 168 |
| VXUS | Total International Stock | -2.96% | -43.13% | +37.21% | 80.34% | 95.1% ✅ | 102 |
| VEA | Developed Markets ex-US | -1.03% | -43.06% | +41.00% | 84.06% | 95.1% ✅ | 144 |
| VWO | Emerging Markets | -0.73% | -37.69% | +36.24% | 73.93% | 92.5% ✅ | 174 |
| VIG | Dividend Appreciation | +1.04% | -30.24% | +32.31% | 62.54% | 96.9% ✅ | 162 |
| SCHD | US Dividend Equity | +10.61% | -35.13% | +56.36% | 91.49% | 91.7% ✅ | 96 |
| BND | Total Bond Market | +6.36% | -24.40% | +37.12% | 61.52% | 94.7% ✅ | 150 |
| BNDX | Total International Bond | +6.92% | -22.93% | +36.78% | 59.71% | 94.9% ✅ | 78 |
| VCIT | Intermediate-Term Corporate Bond | +4.48% | -29.06% | +38.02% | 67.08% | 95.0% ✅ | 120 |
| VMBS | Mortgage-Backed Securities | +6.60% | -26.63% | +39.83% | 66.46% | 95.0% ✅ | 120 |
| VNQ | Real Estate | +0.88% | -31.47% | +33.24% | 64.71% | 90.0% ✅ | 180 |
| GLD | Gold Shares | -17.25% | -52.81% | +18.30% | 71.11% | 86.7% ✅ | 180 |
| DBC | DB Commodity Index | -1.43% | -35.02% | +32.16% | 67.18% | 88.9% ✅ | 162 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | IC | IC | Hit Rate | HR |
|-----------|-------------|-------|----|----|-----------|----|
| BND | Total Bond Market | 150 | 0.1248 | ✅ | 63.3% | ✅ |
| BNDX | Total International Bond | 78 | 0.1756 | ✅ | 59.0% | ✅ |
| DBC | DB Commodity Index | 162 | 0.1637 | ✅ | 63.6% | ✅ |
| GLD | Gold Shares | 180 | 0.2296 | ✅ | 61.7% | ✅ |
| KIE | S&P Insurance | 168 | 0.1437 | ✅ | 50.0% | ❌ |
| SCHD | US Dividend Equity | 96 | -0.1625 | ❌ | 50.0% | ❌ |
| VCIT | Intermediate-Term Corporate Bond | 120 | 0.0713 | ✅ | 61.7% | ✅ |
| VDE | Energy | 180 | -0.0384 | ❌ | 49.4% | ❌ |
| VEA | Developed Markets ex-US | 144 | -0.1571 | ❌ | 61.1% | ✅ |
| VFH | Financials | 186 | -0.1297 | ❌ | 44.1% | ❌ |
| VGT | Information Technology | 186 | 0.0180 | ❌ | 53.2% | ⚠️ |
| VHT | Health Care | 186 | 0.0009 | ❌ | 52.7% | ⚠️ |
| VIG | Dividend Appreciation | 162 | 0.0489 | ⚠️ | 51.2% | ❌ |
| VIS | Industrials | 180 | -0.0298 | ❌ | 47.8% | ❌ |
| VMBS | Mortgage-Backed Securities | 120 | 0.1244 | ✅ | 61.7% | ✅ |
| VNQ | Real Estate | 180 | 0.1147 | ✅ | 57.2% | ✅ |
| VOO | S&P 500 | 108 | -0.2127 | ❌ | 34.3% | ❌ |
| VPU | Utilities | 186 | 0.1693 | ✅ | 57.0% | ✅ |
| VTI | Total Stock Market | 222 | -0.0078 | ❌ | 50.0% | ❌ |
| VWO | Emerging Markets | 174 | -0.0387 | ❌ | 58.6% | ✅ |
| VXUS | Total International Stock | 102 | -0.2023 | ❌ | 66.7% | ✅ |

**IC summary:** 9 ✅  1 ⚠️  11 ❌  (of 21 benchmarks)  
**Hit rate ✅:** 11/21 benchmarks above 55% threshold  

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