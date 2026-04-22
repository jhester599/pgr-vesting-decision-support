# PGR Diagnostic Report — April 2026

**As-of Date:** 2026-04-22  
**Horizon:** 6M  
**OOS observations (aggregate):** 1188  
**Newey-West lags:** 5 (accounts for 5-month return-window overlap)  

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² (Campbell-Thompson) | -0.0263 (-2.63%) | ❌ | ≥ 2.00% |
| IC (Newey-West HAC) | 0.1530 | ✅ | ≥ 0.07 |
| IC significance | 0.0005 | ✅ p < 0.05 | p < 0.05 |
| Clark-West t-stat | 3.4097 | ✅ p < 0.05 | p < 0.05 |
| Clark-West p-value | 0.0003 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 66.4% | ✅ | ≥ 55.0% |
| CPCV Positive Paths | 1/7 (12.5%) | ❌ | ≥ 5/7 |

> **Representative CPCV:** benchmark=VOO, model=ridge, paths=7, mean IC=-0.0484, IC std=0.1060.
> Stability verdict: FAIL. Scaled monthly threshold: ≥ 5/7 (maps from the full C(8,2) standard of ≥ 19/28 positive paths).

---


## Feature Governance

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Full obs/feature ratio | 20.17 | ✅ | ≥ 4.0 |
| Per-fold obs/feature ratio | 5.00 | ✅ | ≥ 4.0 |
| Features in monthly run | 12 | — | — |
| Fully populated observations | 242 | — | — |

> obs/feature ratio: 20.2 (full matrix), 5.0 (per WFO fold, 60M window).  n_obs=242, n_features=12.  Verdict: OK.

### Feature Importance Stability

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Mean consecutive-fold Spearman ρ | 0.5002 | ⚠️ MARGINAL | ≥ 0.70 |
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
| 6 | book_value_per_share_growth_yoy | 6.6 | 3.7 | 0.0123 |
| 7 | vol_63d | 7.2 | 4.0 | 0.0129 |
| 7 | investment_income_growth_yoy | 7.2 | 3.2 | 0.0083 |
| 7 | yield_slope | 7.4 | 3.5 | 0.0071 |

---

## Calibration Phase

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Raw BayesianRidge posterior (uncalibrated) | ⬛ Superseded |
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (n=1,188  ECE=2.5% [1.8%–6.9%]) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

**Mean empirical coverage:** 91.4% (target ≥ 80%) ✅  

**Mean trailing 12-point empirical coverage:** 54.2% (gap -25.8% vs nominal) ❌  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | Trailing 12 Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|---------------|----------------------|-------|
| VOO | S&P 500 | +0.71% | -27.31% | +28.74% | 56.05% | 94.4% ✅ | 58.3% | 108 |
| VXUS | Total International Stock | +3.34% | -33.02% | +39.70% | 72.72% | 98.1% ✅ | 50.0% | 108 |
| VWO | Emerging Markets | -1.93% | -37.24% | +33.38% | 70.62% | 97.7% ✅ | 50.0% | 174 |
| VMBS | Mortgage-Backed Securities | +2.92% | -18.38% | +24.23% | 42.61% | 83.3% ✅ | 66.7% | 120 |
| BND | Total Bond Market | +3.67% | -17.82% | +25.16% | 42.98% | 86.7% ✅ | 75.0% | 150 |
| GLD | Gold Shares | -9.81% | -42.94% | +23.33% | 66.27% | 91.7% ✅ | 33.3% | 180 |
| DBC | DB Commodity Index | -9.39% | -37.70% | +18.92% | 56.62% | 88.7% ✅ | 58.3% | 168 |
| VDE | Energy | -11.52% | -46.91% | +23.87% | 70.79% | 90.6% ✅ | 41.7% | 180 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | OOS R² | NW IC | Hit Rate | CW t | CW p |
|-----------|-------------|-------|--------|-------|----------|------|------|
| DBC | DB Commodity Index | 168 | 1.34% | 0.2131 | 72.0% | 2.1275 | 0.0174 |
| VOO | S&P 500 | 108 | -8.96% | -0.0023 | 51.9% | 1.9587 | 0.0264 |
| VDE | Energy | 180 | -1.02% | 0.0653 | 65.6% | 1.7335 | 0.0424 |
| VXUS | Total International Stock | 108 | -5.96% | 0.1282 | 71.3% | 1.4367 | 0.0769 |
| VMBS | Mortgage-Backed Securities | 120 | -12.42% | 0.2481 | 80.0% | 1.2220 | 0.1121 |
| GLD | Gold Shares | 180 | -8.99% | 0.1322 | 55.0% | 1.0366 | 0.1507 |
| VWO | Emerging Markets | 174 | -6.81% | 0.1180 | 63.2% | 0.8518 | 0.1977 |
| BND | Total Bond Market | 150 | -12.79% | 0.2814 | 74.7% | 0.7583 | 0.2247 |

**IC summary:** 6 ✅  1 ⚠️  1 ❌  (of 8 benchmarks)  
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
| P(Actionable Sell) | 37.0% |

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