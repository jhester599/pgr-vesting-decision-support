# PGR Diagnostic Report — April 2026

**As-of Date:** 2026-04-11  
**Horizon:** 6M  
**OOS observations (aggregate):** 1188  
**Newey-West lags:** 5 (accounts for 5-month return-window overlap)  

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² (Campbell-Thompson) | -0.0453 (-4.53%) | ❌ | ≥ 2.00% |
| IC (Newey-West HAC) | 0.1701 | ✅ | ≥ 0.07 |
| IC significance | 0.0002 | ✅ p < 0.05 | p < 0.05 |
| Clark-West t-stat | 3.1080 | ✅ p < 0.05 | p < 0.05 |
| Clark-West p-value | 0.0010 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 67.1% | ✅ | ≥ 55.0% |
| CPCV Positive Paths | 1/7 (12.5%) | ❌ | ≥ 5/7 |

> **Representative CPCV:** benchmark=VOO, model=ridge, paths=7, mean IC=-0.0479, IC std=0.1028.
> Stability verdict: FAIL. Scaled monthly threshold: ≥ 5/7 (maps from the full C(8,2) standard of ≥ 19/28 positive paths).

---


## Feature Governance

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Full obs/feature ratio | 19.33 | ✅ | ≥ 4.0 |
| Per-fold obs/feature ratio | 5.00 | ✅ | ≥ 4.0 |
| Features in monthly run | 12 | — | — |
| Fully populated observations | 232 | — | — |

> obs/feature ratio: 19.3 (full matrix), 5.0 (per WFO fold, 60M window).  n_obs=232, n_features=12.  Verdict: OK.

### Feature Importance Stability

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Mean consecutive-fold Spearman ρ | 0.5014 | ⚠️ MARGINAL | ≥ 0.70 |
| Folds included | 18 | — | — |

> Stability score measures mean pairwise Spearman rank-correlation between consecutive WFO fold importance rankings. A score < 0.40 indicates unstable feature rankings; model predictions may be driven by different features each period.

**Top 10 features by mean WFO rank:**

| Rank | Feature | Mean Rank | Rank Std | Mean |Importance| |
|------|---------|-----------|----------|----------------|
| 3 | combined_ratio_ttm | 3.9 | 3.0 | 0.0180 |
| 4 | mom_12m | 4.7 | 4.1 | 0.0221 |
| 5 | nfci | 5.7 | 1.6 | 0.0124 |
| 5 | real_yield_change_6m | 5.8 | 3.8 | 0.0164 |
| 6 | real_rate_10y | 6.2 | 2.7 | 0.0127 |
| 6 | credit_spread_hy | 6.2 | 3.5 | 0.0153 |
| 6 | book_value_per_share_growth_yoy | 6.5 | 3.7 | 0.0123 |
| 7 | vol_63d | 7.2 | 4.0 | 0.0130 |
| 7 | investment_income_growth_yoy | 7.2 | 3.2 | 0.0083 |
| 7 | yield_slope | 7.4 | 3.5 | 0.0072 |

---

## Calibration Phase

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Raw BayesianRidge posterior (uncalibrated) | ⬛ Superseded |
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (n=1,188  ECE=2.3% [1.7%–6.8%]) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

**Mean empirical coverage:** 90.1% (target ≥ 80%) ✅  

**Mean trailing 12-point empirical coverage:** 56.2% (gap -23.8% vs nominal) ❌  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | Trailing 12 Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|---------------|----------------------|-------|
| VOO | S&P 500 | +1.40% | -26.51% | +29.30% | 55.81% | 94.4% ✅ | 58.3% | 108 |
| VXUS | Total International Stock | +2.96% | -33.74% | +39.66% | 73.40% | 98.1% ✅ | 50.0% | 108 |
| VWO | Emerging Markets | -0.84% | -30.34% | +28.66% | 59.00% | 92.5% ✅ | 50.0% | 174 |
| VMBS | Mortgage-Backed Securities | +3.17% | -17.96% | +24.29% | 42.25% | 83.3% ✅ | 75.0% | 120 |
| BND | Total Bond Market | +3.37% | -17.98% | +24.73% | 42.70% | 86.7% ✅ | 75.0% | 150 |
| GLD | Gold Shares | -7.26% | -42.19% | +27.66% | 69.85% | 91.7% ✅ | 33.3% | 180 |
| DBC | DB Commodity Index | -9.58% | -37.48% | +18.32% | 55.80% | 88.7% ✅ | 58.3% | 168 |
| VDE | Energy | -10.03% | -40.97% | +20.91% | 61.88% | 85.6% ✅ | 50.0% | 180 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | OOS R² | NW IC | Hit Rate | CW t | CW p |
|-----------|-------------|-------|--------|-------|----------|------|------|
| DBC | DB Commodity Index | 168 | -1.16% | 0.2441 | 72.6% | 2.2363 | 0.0133 |
| VOO | S&P 500 | 108 | -8.79% | -0.0093 | 54.6% | 1.8698 | 0.0321 |
| VDE | Energy | 180 | -4.02% | 0.1084 | 65.6% | 1.5432 | 0.0623 |
| VXUS | Total International Stock | 108 | -5.99% | 0.1140 | 75.0% | 1.4022 | 0.0819 |
| VMBS | Mortgage-Backed Securities | 120 | -11.28% | 0.2344 | 80.8% | 1.2419 | 0.1084 |
| VWO | Emerging Markets | 174 | -8.68% | 0.1486 | 62.6% | 0.9853 | 0.1629 |
| BND | Total Bond Market | 150 | -12.65% | 0.2875 | 74.7% | 0.8384 | 0.2016 |
| GLD | Gold Shares | 180 | -13.00% | 0.1360 | 55.0% | 0.7891 | 0.2155 |

**IC summary:** 7 ✅  0 ⚠️  1 ❌  (of 8 benchmarks)  
**Hit rate ✅:** 7/8 benchmarks above 55% threshold  
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
| P(Actionable Sell) | 36.0% |

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