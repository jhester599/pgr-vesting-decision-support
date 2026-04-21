# PGR Diagnostic Report — April 2026

**As-of Date:** 2026-04-21  
**Horizon:** 6M  
**OOS observations (aggregate):** 1188  
**Newey-West lags:** 5 (accounts for 5-month return-window overlap)  

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² (Campbell-Thompson) | -0.0229 (-2.29%) | ❌ | ≥ 2.00% |
| IC (Newey-West HAC) | 0.1609 | ✅ | ≥ 0.07 |
| IC significance | 0.0003 | ✅ p < 0.05 | p < 0.05 |
| Clark-West t-stat | 3.4697 | ✅ p < 0.05 | p < 0.05 |
| Clark-West p-value | 0.0003 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 66.8% | ✅ | ≥ 55.0% |
| CPCV Positive Paths | 1/7 (12.5%) | ❌ | ≥ 5/7 |

> **Representative CPCV:** benchmark=VOO, model=ridge, paths=7, mean IC=-0.0443, IC std=0.1030.
> Stability verdict: FAIL. Scaled monthly threshold: ≥ 5/7 (maps from the full C(8,2) standard of ≥ 19/28 positive paths).

---


## Feature Governance

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Full obs/feature ratio | 18.42 | ✅ | ≥ 4.0 |
| Per-fold obs/feature ratio | 5.00 | ✅ | ≥ 4.0 |
| Features in monthly run | 12 | — | — |
| Fully populated observations | 221 | — | — |

> obs/feature ratio: 18.4 (full matrix), 5.0 (per WFO fold, 60M window).  n_obs=221, n_features=12.  Verdict: OK.

### Feature Importance Stability

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Mean consecutive-fold Spearman ρ | 0.5023 | ⚠️ MARGINAL | ≥ 0.70 |
| Folds included | 18 | — | — |

> Stability score measures mean pairwise Spearman rank-correlation between consecutive WFO fold importance rankings. A score < 0.40 indicates unstable feature rankings; model predictions may be driven by different features each period.

**Top 10 features by mean WFO rank:**

| Rank | Feature | Mean Rank | Rank Std | Mean |Importance| |
|------|---------|-----------|----------|----------------|
| 4 | combined_ratio_ttm | 4.0 | 3.0 | 0.0181 |
| 4 | mom_12m | 4.7 | 4.1 | 0.0221 |
| 5 | nfci | 5.7 | 1.6 | 0.0125 |
| 5 | real_yield_change_6m | 5.8 | 3.8 | 0.0164 |
| 6 | real_rate_10y | 6.1 | 2.7 | 0.0127 |
| 6 | credit_spread_hy | 6.2 | 3.5 | 0.0153 |
| 6 | book_value_per_share_growth_yoy | 6.5 | 3.7 | 0.0123 |
| 7 | vol_63d | 7.2 | 4.0 | 0.0129 |
| 7 | investment_income_growth_yoy | 7.2 | 3.2 | 0.0083 |
| 7 | yield_slope | 7.4 | 3.5 | 0.0071 |

---

## Calibration Phase

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Raw BayesianRidge posterior (uncalibrated) | ⬛ Superseded |
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (n=1,188  ECE=3.3% [1.9%–7.4%]) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

**Mean empirical coverage:** 90.2% (target ≥ 80%) ✅  

**Mean trailing 12-point empirical coverage:** 56.2% (gap -23.8% vs nominal) ❌  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | Trailing 12 Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|---------------|----------------------|-------|
| VOO | S&P 500 | -1.71% | -29.73% | +26.32% | 56.05% | 94.4% ✅ | 58.3% | 108 |
| VXUS | Total International Stock | +0.72% | -35.46% | +36.90% | 72.36% | 98.1% ✅ | 50.0% | 108 |
| VWO | Emerging Markets | -2.49% | -37.79% | +32.82% | 70.61% | 97.7% ✅ | 50.0% | 174 |
| VMBS | Mortgage-Backed Securities | -0.13% | -21.48% | +21.22% | 42.70% | 83.3% ✅ | 66.7% | 120 |
| BND | Total Bond Market | +0.08% | -21.40% | +21.57% | 42.97% | 86.7% ✅ | 75.0% | 150 |
| GLD | Gold Shares | -12.13% | -37.80% | +13.54% | 51.33% | 86.7% ✅ | 41.7% | 180 |
| DBC | DB Commodity Index | -9.69% | -37.98% | +18.60% | 56.58% | 88.7% ✅ | 58.3% | 168 |
| VDE | Energy | -10.69% | -40.99% | +19.62% | 60.60% | 85.6% ✅ | 50.0% | 180 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | OOS R² | NW IC | Hit Rate | CW t | CW p |
|-----------|-------------|-------|--------|-------|----------|------|------|
| DBC | DB Commodity Index | 168 | 1.54% | 0.2205 | 72.0% | 2.1384 | 0.0170 |
| VOO | S&P 500 | 108 | -9.19% | -0.0057 | 51.9% | 1.9178 | 0.0289 |
| VDE | Energy | 180 | -0.66% | 0.0753 | 65.6% | 1.7877 | 0.0378 |
| VXUS | Total International Stock | 108 | -5.29% | 0.1450 | 71.3% | 1.4578 | 0.0739 |
| VMBS | Mortgage-Backed Securities | 120 | -12.58% | 0.2379 | 80.8% | 1.2330 | 0.1100 |
| GLD | Gold Shares | 180 | -8.79% | 0.1300 | 55.6% | 1.0462 | 0.1484 |
| VWO | Emerging Markets | 174 | -5.67% | 0.1323 | 64.4% | 0.9463 | 0.1727 |
| BND | Total Bond Market | 150 | -12.54% | 0.2961 | 74.7% | 0.7960 | 0.2137 |

**IC summary:** 7 ✅  0 ⚠️  1 ❌  (of 8 benchmarks)  
**Hit rate ✅:** 7/8 benchmarks above 55% threshold  
**Clark-West ✅:** 3/8 benchmarks with p < 0.05  

---

## Shadow Gate Overlay

| Field | Value |
|-------|-------|
| Variant | gemini_veto_0.50 |
| Recommendation Mode | DEFER-TO-TAX-DEFAULT |
| Recommended Sell % | 50% |
| Would Change Live Output | No |
| Reason | no regression sell to veto |
| P(Actionable Sell) | 38.5% |

---

## Classifier Monitoring

| Metric | Value |
|--------|-------|
| Matured observations | 0 |
| Brier score | n/a |
| Log loss | n/a |
| ECE (10-bin) | n/a |

> Matured-horizon diagnostics are computed only once the forecast horizon has elapsed.

---

## Threshold Reference

| Metric | Good | Marginal | Failing | Source |
|--------|------|----------|---------|--------|
| OOS R² | > 2% | 0.5–2% | < 0% | Campbell & Thompson (2008) |
| Mean IC | > 0.07 | 0.03–0.07 | < 0.03 | Harvey et al. (2016) |
| Clark-West | p < 0.05 | p < 0.10 | ≥ 0.10 | Clark & West (2007) |
| Hit Rate | > 55% | 52–55% | < 52% | Industry consensus |
| CPCV +paths | ≥ 19/28 | 14–18/28 | < 14/28 | López de Prado (2018) |
| PBO | < 15% | 15–40% | > 40% | Bailey et al. (2014) |

---

*Generated by `scripts/monthly_decision.py`*