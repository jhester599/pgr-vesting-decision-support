# PGR Diagnostic Report — April 2026

**As-of Date:** 2026-04-11  
**Horizon:** 6M  
**OOS observations (aggregate):** 1188  
**Newey-West lags:** 5 (accounts for 5-month return-window overlap)  

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² (Campbell-Thompson) | -0.0275 (-2.75%) | ❌ | ≥ 2.00% |
| IC (Newey-West HAC) | 0.2006 | ✅ | ≥ 0.07 |
| IC significance | 0.0000 | ✅ p < 0.05 | p < 0.05 |
| Clark-West t-stat | 3.4268 | ✅ p < 0.05 | p < 0.05 |
| Clark-West p-value | 0.0003 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 68.1% | ✅ | ≥ 55.0% |
| CPCV Positive Paths | 1/7 (12.5%) | ❌ | ≥ 5/7 |

> **Representative CPCV:** benchmark=VOO, model=ridge, paths=7, mean IC=-0.0710, IC std=0.1101.
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
| Mean consecutive-fold Spearman ρ | 0.5487 | ⚠️ MARGINAL | ≥ 0.70 |
| Folds included | 18 | — | — |

> Stability score measures mean pairwise Spearman rank-correlation between consecutive WFO fold importance rankings. A score < 0.40 indicates unstable feature rankings; model predictions may be driven by different features each period.

**Top 10 features by mean WFO rank:**

| Rank | Feature | Mean Rank | Rank Std | Mean |Importance| |
|------|---------|-----------|----------|----------------|
| 4 | combined_ratio_ttm | 4.0 | 3.0 | 0.0204 |
| 4 | mom_12m | 4.7 | 4.2 | 0.0241 |
| 5 | nfci | 5.1 | 2.2 | 0.0163 |
| 5 | real_rate_10y | 5.6 | 2.7 | 0.0143 |
| 6 | real_yield_change_6m | 6.1 | 3.9 | 0.0152 |
| 6 | book_value_per_share_growth_yoy | 6.6 | 3.7 | 0.0113 |
| 6 | credit_spread_hy | 6.6 | 3.6 | 0.0135 |
| 7 | investment_income_growth_yoy | 7.3 | 3.1 | 0.0082 |
| 7 | vix | 7.4 | 2.7 | 0.0105 |
| 7 | vol_63d | 7.7 | 3.7 | 0.0129 |

---

## Calibration Phase

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Raw BayesianRidge posterior (uncalibrated) | ⬛ Superseded |
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (n=1,188  ECE=1.1% [1.4%–6.0%]) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

**Mean empirical coverage:** 88.8% (target ≥ 80%) ✅  

**Mean trailing 12-point empirical coverage:** 57.3% (gap -22.7% vs nominal) ❌  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | Trailing 12 Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|---------------|----------------------|-------|
| VOO | S&P 500 | +0.54% | -26.57% | +27.65% | 54.22% | 94.4% ✅ | 66.7% | 108 |
| VXUS | Total International Stock | +3.63% | -32.70% | +39.95% | 72.65% | 98.1% ✅ | 58.3% | 108 |
| VWO | Emerging Markets | -2.37% | -30.76% | +26.02% | 56.78% | 92.5% ✅ | 50.0% | 174 |
| VMBS | Mortgage-Backed Securities | +0.33% | -20.20% | +20.86% | 41.06% | 82.5% ✅ | 75.0% | 120 |
| BND | Total Bond Market | +0.54% | -20.48% | +21.55% | 42.04% | 86.7% ✅ | 75.0% | 150 |
| GLD | Gold Shares | -0.02% | -26.45% | +26.42% | 52.87% | 86.7% ✅ | 41.7% | 180 |
| DBC | DB Commodity Index | -9.53% | -36.29% | +17.24% | 53.53% | 83.9% ✅ | 58.3% | 168 |
| VDE | Energy | -12.72% | -43.24% | +17.79% | 61.03% | 85.6% ✅ | 33.3% | 180 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | OOS R² | NW IC | Hit Rate | CW t | CW p |
|-----------|-------------|-------|--------|-------|----------|------|------|
| DBC | DB Commodity Index | 168 | -0.67% | 0.1836 | 72.0% | 2.1149 | 0.0180 |
| VOO | S&P 500 | 108 | -6.58% | 0.0764 | 56.5% | 2.0095 | 0.0235 |
| VDE | Energy | 180 | -0.78% | 0.1434 | 62.2% | 1.7917 | 0.0374 |
| VXUS | Total International Stock | 108 | -1.06% | 0.2323 | 75.9% | 1.7378 | 0.0426 |
| VWO | Emerging Markets | 174 | -3.97% | 0.2454 | 67.8% | 1.2809 | 0.1010 |
| VMBS | Mortgage-Backed Securities | 120 | -15.08% | 0.2396 | 80.0% | 1.2216 | 0.1121 |
| BND | Total Bond Market | 150 | -14.59% | 0.2479 | 74.7% | 0.8106 | 0.2095 |
| GLD | Gold Shares | 180 | -11.84% | 0.1602 | 59.4% | 0.7638 | 0.2230 |

**IC summary:** 8 ✅  0 ⚠️  0 ❌  (of 8 benchmarks)  
**Hit rate ✅:** 8/8 benchmarks above 55% threshold  
**Clark-West ✅:** 4/8 benchmarks with p < 0.05  

---

## Shadow Gate Overlay

| Field | Value |
|-------|-------|
| Variant | gemini_veto_0.50 |
| Recommendation Mode | DEFER-TO-TAX-DEFAULT |
| Recommended Sell % | 50% |
| Would Change Live Output | No |
| Reason | no regression sell to veto |
| P(Actionable Sell) | 35.2% |

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