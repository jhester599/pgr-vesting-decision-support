# PGR Diagnostic Report — April 2026

**As-of Date:** 2026-04-02  
**Horizon:** 6M  
**OOS observations (aggregate):** 3270  
**Newey-West lags:** 5 (accounts for 5-month return-window overlap)  

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² (Campbell-Thompson) | -1.4132 (-141.32%) | ❌ | ≥ 2.00% |
| IC (Newey-West HAC) | 0.0317 | ⚠️ | ≥ 0.07 |
| IC significance | 0.2820 | ❌ not sig. | p < 0.05 |
| Hit Rate | 52.1% | ⚠️ | ≥ 55.0% |
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
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (n=3,270  ECE=1.8% [1.1%–4.7%]) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

**Mean empirical coverage:** 89.7% (target ≥ 80%) ✅  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|---------------|-------|
| VTI | Total Stock Market | -11.75% | -45.91% | +22.41% | 68.32% | 89.6% ✅ | 222 |
| VOO | S&P 500 | -8.38% | -43.61% | +26.86% | 70.47% | 86.1% ✅ | 108 |
| VGT | Information Technology | -15.49% | -55.49% | +24.51% | 80.01% | 95.7% ✅ | 186 |
| VHT | Health Care | -19.71% | -58.86% | +19.45% | 78.31% | 95.7% ✅ | 186 |
| VFH | Financials | -5.01% | -28.30% | +18.29% | 46.58% | 68.8% ⚠️ | 186 |
| VIS | Industrials | -3.36% | -41.03% | +34.32% | 75.34% | 96.7% ✅ | 180 |
| VDE | Energy | -17.52% | -54.41% | +19.36% | 73.77% | 81.7% ✅ | 180 |
| VPU | Utilities | -16.67% | -43.14% | +9.81% | 52.95% | 85.5% ✅ | 186 |
| KIE | S&P Insurance | -8.36% | -33.45% | +16.72% | 50.17% | 73.8% ⚠️ | 168 |
| VXUS | Total International Stock | -2.16% | -46.21% | +41.89% | 88.11% | 99.0% ✅ | 102 |
| VEA | Developed Markets ex-US | -3.63% | -50.12% | +42.87% | 92.99% | 100.0% ✅ | 144 |
| VWO | Emerging Markets | -2.66% | -44.60% | +39.28% | 83.89% | 97.7% ✅ | 174 |
| VIG | Dividend Appreciation | -7.98% | -39.35% | +23.38% | 62.73% | 96.9% ✅ | 162 |
| SCHD | US Dividend Equity | -4.01% | -83.68% | +75.65% | 159.33% | 94.8% ✅ | 96 |
| BND | Total Bond Market | +0.65% | -33.06% | +34.36% | 67.42% | 96.7% ✅ | 150 |
| BNDX | Total International Bond | -0.29% | -28.75% | +28.16% | 56.91% | 89.7% ✅ | 78 |
| VCIT | Intermediate-Term Corporate Bond | +0.01% | -29.00% | +29.02% | 58.02% | 87.5% ✅ | 120 |
| VMBS | Mortgage-Backed Securities | +1.55% | -32.04% | +35.13% | 67.17% | 97.5% ✅ | 120 |
| VNQ | Real Estate | -7.56% | -33.78% | +18.66% | 52.44% | 70.6% ⚠️ | 180 |
| GLD | Gold Shares | -12.10% | -55.01% | +30.82% | 85.83% | 91.7% ✅ | 180 |
| DBC | DB Commodity Index | -17.31% | -50.63% | +16.01% | 66.65% | 87.7% ✅ | 162 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | IC | IC | Hit Rate | HR |
|-----------|-------------|-------|----|----|-----------|----|
| BND | Total Bond Market | 150 | 0.0889 | ✅ | 57.3% | ✅ |
| BNDX | Total International Bond | 78 | 0.0119 | ❌ | 44.9% | ❌ |
| DBC | DB Commodity Index | 162 | 0.0691 | ⚠️ | 66.7% | ✅ |
| GLD | Gold Shares | 180 | 0.0720 | ✅ | 55.6% | ✅ |
| KIE | S&P Insurance | 168 | 0.0174 | ❌ | 47.6% | ❌ |
| SCHD | US Dividend Equity | 96 | -0.0492 | ❌ | 52.1% | ⚠️ |
| VCIT | Intermediate-Term Corporate Bond | 120 | 0.0100 | ❌ | 57.5% | ✅ |
| VDE | Energy | 180 | -0.0282 | ❌ | 58.9% | ✅ |
| VEA | Developed Markets ex-US | 144 | -0.1347 | ❌ | 54.2% | ⚠️ |
| VFH | Financials | 186 | -0.1235 | ❌ | 46.8% | ❌ |
| VGT | Information Technology | 186 | -0.0168 | ❌ | 44.6% | ❌ |
| VHT | Health Care | 186 | 0.0286 | ❌ | 51.6% | ❌ |
| VIG | Dividend Appreciation | 162 | 0.0041 | ❌ | 52.5% | ⚠️ |
| VIS | Industrials | 180 | 0.0724 | ✅ | 54.4% | ⚠️ |
| VMBS | Mortgage-Backed Securities | 120 | 0.0593 | ⚠️ | 57.5% | ✅ |
| VNQ | Real Estate | 180 | 0.0472 | ⚠️ | 49.4% | ❌ |
| VOO | S&P 500 | 108 | -0.1984 | ❌ | 40.7% | ❌ |
| VPU | Utilities | 186 | 0.1089 | ✅ | 48.9% | ❌ |
| VTI | Total Stock Market | 222 | -0.0381 | ❌ | 44.6% | ❌ |
| VWO | Emerging Markets | 174 | -0.0461 | ❌ | 51.1% | ❌ |
| VXUS | Total International Stock | 102 | -0.1329 | ❌ | 59.8% | ✅ |

**IC summary:** 4 ✅  3 ⚠️  14 ❌  (of 21 benchmarks)  
**Hit rate ✅:** 7/21 benchmarks above 55% threshold  

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