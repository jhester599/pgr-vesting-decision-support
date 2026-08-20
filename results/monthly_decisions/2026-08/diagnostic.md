# PGR Diagnostic Report — August 2026

**As-of Date:** 2026-08-20  
**Horizon:** 6M  
**OOS observations (aggregate):** 1224  
**Newey-West lags:** 5 (accounts for 5-month return-window overlap)  

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² (Campbell-Thompson) | -0.0014 (-0.14%) | ❌ | ≥ 2.00% |
| IC (Newey-West HAC) | 0.1806 | ✅ | ≥ 0.07 |
| IC significance | 0.0000 | ✅ p < 0.05 | p < 0.05 |
| Clark-West t-stat | 4.1504 | ✅ p < 0.05 | p < 0.05 |
| Clark-West p-value | 0.0000 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 64.8% | ✅ | ≥ 55.0% |
| CPCV Positive Paths | 1/7 (12.5%) | ❌ | ≥ 5/7 |

> **Representative CPCV:** benchmark=VOO, model=ridge, paths=7, mean IC=-0.0601, IC std=0.1262.
> Stability verdict: FAIL. Scaled monthly threshold: ≥ 5/7 (maps from the full C(8,2) standard of ≥ 19/28 positive paths).

---


## Feature Governance

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Full obs/feature ratio | 20.50 | ✅ | ≥ 4.0 |
| Per-fold obs/feature ratio | 5.00 | ✅ | ≥ 4.0 |
| Features in monthly run | 12 | — | — |
| Fully populated observations | 246 | — | — |

> obs/feature ratio: 20.5 (full matrix), 5.0 (per WFO fold, 60M window).  n_obs=246, n_features=12.  Verdict: OK.

### Feature Importance Stability

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Mean consecutive-fold Spearman ρ | 0.5730 | ⚠️ MARGINAL | ≥ 0.70 |
| Folds included | 19 | — | — |

> Stability score measures mean pairwise Spearman rank-correlation between consecutive WFO fold importance rankings. A score < 0.40 indicates unstable feature rankings; model predictions may be driven by different features each period.

**Top 10 features by mean WFO rank:**

| Rank | Feature | Mean Rank | Rank Std | Mean |Importance| |
|------|---------|-----------|----------|----------------|
| 3 | combined_ratio_ttm | 3.9 | 2.8 | 0.0214 |
| 4 | mom_12m | 4.8 | 4.0 | 0.0245 |
| 5 | credit_spread_hy | 5.4 | 2.9 | 0.0178 |
| 5 | nfci | 5.7 | 2.1 | 0.0139 |
| 5 | real_rate_10y | 5.9 | 3.0 | 0.0193 |
| 6 | book_value_per_share_growth_yoy | 6.1 | 3.1 | 0.0161 |
| 6 | real_yield_change_6m | 6.3 | 4.0 | 0.0226 |
| 6 | vol_63d | 6.4 | 3.9 | 0.0162 |
| 7 | investment_income_growth_yoy | 7.4 | 3.2 | 0.0103 |
| 8 | vix | 8.1 | 2.6 | 0.0087 |

---

## Calibration Phase

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Raw BayesianRidge posterior (uncalibrated) | ⬛ Superseded |
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (n=1,224  ECE=2.5% [1.8%–7.0%]) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

**Mean empirical coverage:** 89.4% (target ≥ 80%) ✅  

**Mean trailing 12-point empirical coverage:** 55.2% (gap -24.8% vs nominal) ❌  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | Trailing 12 Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|---------------|----------------------|-------|
| VOO | S&P 500 | -0.60% | -26.12% | +24.92% | 51.04% | 88.6% ✅ | 66.7% | 114 |
| VXUS | Total International Stock | +2.48% | -27.20% | +32.17% | 59.37% | 88.9% ✅ | 58.3% | 108 |
| VWO | Emerging Markets | -6.30% | -35.48% | +22.89% | 58.37% | 91.7% ✅ | 50.0% | 180 |
| VMBS | Mortgage-Backed Securities | +2.98% | -17.90% | +23.86% | 41.76% | 81.0% ✅ | 66.7% | 126 |
| BND | Total Bond Market | +3.33% | -17.63% | +24.30% | 41.94% | 85.9% ✅ | 75.0% | 156 |
| GLD | Gold Shares | -6.08% | -40.79% | +28.62% | 69.41% | 93.0% ✅ | 41.7% | 186 |
| DBC | DB Commodity Index | -13.71% | -47.97% | +20.54% | 68.51% | 97.6% ✅ | 41.7% | 168 |
| VDE | Energy | -14.41% | -47.08% | +18.25% | 65.33% | 88.7% ✅ | 41.7% | 186 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | OOS R² | NW IC | Hit Rate | CW t | CW p |
|-----------|-------------|-------|--------|-------|----------|------|------|
| VDE | Energy | 186 | 3.02% | 0.2152 | 61.8% | 2.5591 | 0.0056 |
| DBC | DB Commodity Index | 168 | 7.17% | 0.3572 | 72.6% | 2.3479 | 0.0100 |
| GLD | Gold Shares | 186 | -1.21% | 0.1931 | 56.5% | 1.3464 | 0.0899 |
| VOO | S&P 500 | 114 | -13.18% | -0.0978 | 57.0% | 1.2704 | 0.1033 |
| VMBS | Mortgage-Backed Securities | 126 | -10.68% | 0.2220 | 74.6% | 1.1800 | 0.1201 |
| VXUS | Total International Stock | 108 | -11.25% | -0.0395 | 59.3% | 0.8913 | 0.1874 |
| BND | Total Bond Market | 156 | -12.25% | 0.2558 | 68.6% | 0.6489 | 0.2587 |
| VWO | Emerging Markets | 180 | -8.54% | 0.0720 | 67.2% | 0.6317 | 0.2642 |

**IC summary:** 6 ✅  0 ⚠️  2 ❌  (of 8 benchmarks)  
**Hit rate ✅:** 8/8 benchmarks above 55% threshold  
**Clark-West ✅:** 2/8 benchmarks with p < 0.05  

---

## Shadow Gate Overlay

| Field | Value |
|-------|-------|
| Variant | gemini_veto_0.50 |
| Recommendation Mode | DEFER-TO-TAX-DEFAULT |
| Recommended Sell % | 50% |
| Would Change Live Output | No |
| Reason | no regression sell to veto |
| P(Actionable Sell) | 41.9% |

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