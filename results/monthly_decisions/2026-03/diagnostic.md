# PGR Diagnostic Report — March 2026

**As-of Date:** 2026-03-20  
**Horizon:** 6M  
**OOS observations (aggregate):** 3270  
**Newey-West lags:** 5 (accounts for 5-month return-window overlap)  

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² (Campbell-Thompson) | -1.3930 (-139.30%) | ❌ | ≥ 2.00% |
| IC (Newey-West HAC) | 0.0417 | ⚠️ | ≥ 0.07 |
| IC significance | 0.1627 | ❌ not sig. | p < 0.05 |
| Hit Rate | 53.5% | ⚠️ | ≥ 55.0% |
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
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (n=3,270  ECE=1.1% [1.0%–4.5%]) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

**Mean empirical coverage:** 91.2% (target ≥ 80%) ✅  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|---------------|-------|
| VTI | Total Stock Market | -2.64% | -46.05% | +40.77% | 86.81% | 89.6% ✅ | 222 |
| VOO | S&P 500 | +1.88% | -30.40% | +34.15% | 64.55% | 88.9% ✅ | 108 |
| VGT | Information Technology | -4.70% | -43.64% | +34.24% | 77.87% | 95.7% ✅ | 186 |
| VHT | Health Care | +6.24% | -35.86% | +48.33% | 84.19% | 95.7% ✅ | 186 |
| VFH | Financials | +5.30% | -20.18% | +30.78% | 50.96% | 75.8% ⚠️ | 186 |
| VIS | Industrials | +1.51% | -37.88% | +40.89% | 78.77% | 91.7% ✅ | 180 |
| VDE | Energy | +10.09% | -37.84% | +58.01% | 95.85% | 86.7% ✅ | 180 |
| VPU | Utilities | -0.39% | -37.36% | +36.58% | 73.94% | 93.0% ✅ | 186 |
| KIE | S&P Insurance | +0.65% | -25.94% | +27.25% | 53.19% | 78.0% ⚠️ | 168 |
| VXUS | Total International Stock | +2.61% | -40.62% | +45.83% | 86.45% | 99.0% ✅ | 102 |
| VEA | Developed Markets ex-US | +0.29% | -44.37% | +44.96% | 89.33% | 99.3% ✅ | 144 |
| VWO | Emerging Markets | +2.85% | -39.53% | +45.22% | 84.76% | 92.5% ✅ | 174 |
| VIG | Dividend Appreciation | +1.35% | -29.90% | +32.61% | 62.51% | 96.9% ✅ | 162 |
| SCHD | US Dividend Equity | +11.39% | -67.02% | +89.80% | 156.81% | 94.8% ✅ | 96 |
| BND | Total Bond Market | +7.10% | -24.61% | +38.81% | 63.42% | 96.7% ✅ | 150 |
| BNDX | Total International Bond | +7.18% | -20.85% | +35.21% | 56.05% | 89.7% ✅ | 78 |
| VCIT | Intermediate-Term Corporate Bond | +5.96% | -22.64% | +34.56% | 57.19% | 92.5% ✅ | 120 |
| VMBS | Mortgage-Backed Securities | +3.78% | -30.62% | +38.18% | 68.80% | 97.5% ✅ | 120 |
| VNQ | Real Estate | -8.01% | -46.49% | +30.48% | 76.97% | 87.8% ✅ | 180 |
| GLD | Gold Shares | -25.25% | -54.07% | +3.57% | 57.64% | 77.8% ⚠️ | 180 |
| DBC | DB Commodity Index | +0.13% | -39.45% | +39.70% | 79.14% | 95.1% ✅ | 162 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | IC | IC | Hit Rate | HR |
|-----------|-------------|-------|----|----|-----------|----|
| BND | Total Bond Market | 150 | 0.1485 | ✅ | 65.3% | ✅ |
| BNDX | Total International Bond | 78 | 0.1106 | ✅ | 52.6% | ⚠️ |
| DBC | DB Commodity Index | 162 | -0.0315 | ❌ | 64.2% | ✅ |
| GLD | Gold Shares | 180 | 0.2424 | ✅ | 58.9% | ✅ |
| KIE | S&P Insurance | 168 | 0.0759 | ✅ | 50.0% | ❌ |
| SCHD | US Dividend Equity | 96 | -0.2241 | ❌ | 39.6% | ❌ |
| VCIT | Intermediate-Term Corporate Bond | 120 | 0.0838 | ✅ | 60.8% | ✅ |
| VDE | Energy | 180 | -0.1271 | ❌ | 48.9% | ❌ |
| VEA | Developed Markets ex-US | 144 | -0.2149 | ❌ | 56.9% | ✅ |
| VFH | Financials | 186 | -0.1351 | ❌ | 39.8% | ❌ |
| VGT | Information Technology | 186 | -0.0280 | ❌ | 51.1% | ❌ |
| VHT | Health Care | 186 | 0.0159 | ❌ | 53.8% | ⚠️ |
| VIG | Dividend Appreciation | 162 | 0.0422 | ⚠️ | 53.7% | ⚠️ |
| VIS | Industrials | 180 | 0.0171 | ❌ | 51.1% | ❌ |
| VMBS | Mortgage-Backed Securities | 120 | 0.1226 | ✅ | 62.5% | ✅ |
| VNQ | Real Estate | 180 | 0.0928 | ✅ | 55.6% | ✅ |
| VOO | S&P 500 | 108 | -0.2045 | ❌ | 32.4% | ❌ |
| VPU | Utilities | 186 | 0.0682 | ⚠️ | 51.1% | ❌ |
| VTI | Total Stock Market | 222 | 0.0488 | ⚠️ | 56.3% | ✅ |
| VWO | Emerging Markets | 174 | -0.0865 | ❌ | 54.6% | ⚠️ |
| VXUS | Total International Stock | 102 | -0.2808 | ❌ | 60.8% | ✅ |

**IC summary:** 7 ✅  3 ⚠️  11 ❌  (of 21 benchmarks)  
**Hit rate ✅:** 9/21 benchmarks above 55% threshold  

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