# PGR Diagnostic Report — May 2026

**As-of Date:** 2026-05-21  
**Horizon:** 6M  
**OOS observations (aggregate):** 1200  
**Newey-West lags:** 5 (accounts for 5-month return-window overlap)  

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² (Campbell-Thompson) | -0.0351 (-3.51%) | ❌ | ≥ 2.00% |
| IC (Newey-West HAC) | 0.0979 | ✅ | ≥ 0.07 |
| IC significance | 0.0288 | ✅ p < 0.05 | p < 0.05 |
| Clark-West t-stat | 2.7721 | ✅ p < 0.05 | p < 0.05 |
| Clark-West p-value | 0.0028 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 64.2% | ✅ | ≥ 55.0% |
| CPCV Positive Paths | 1/7 (12.5%) | ❌ | ≥ 5/7 |

> **Representative CPCV:** benchmark=VOO, model=ridge, paths=7, mean IC=-0.0607, IC std=0.1171.
> Stability verdict: FAIL. Scaled monthly threshold: ≥ 5/7 (maps from the full C(8,2) standard of ≥ 19/28 positive paths).

---


## Feature Governance

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Full obs/feature ratio | 20.25 | ✅ | ≥ 4.0 |
| Per-fold obs/feature ratio | 5.00 | ✅ | ≥ 4.0 |
| Features in monthly run | 12 | — | — |
| Fully populated observations | 243 | — | — |

> obs/feature ratio: 20.2 (full matrix), 5.0 (per WFO fold, 60M window).  n_obs=243, n_features=12.  Verdict: OK.

### Feature Importance Stability

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Mean consecutive-fold Spearman ρ | 0.5373 | ⚠️ MARGINAL | ≥ 0.70 |
| Folds included | 19 | — | — |

> Stability score measures mean pairwise Spearman rank-correlation between consecutive WFO fold importance rankings. A score < 0.40 indicates unstable feature rankings; model predictions may be driven by different features each period.

**Top 10 features by mean WFO rank:**

| Rank | Feature | Mean Rank | Rank Std | Mean |Importance| |
|------|---------|-----------|----------|----------------|
| 3 | combined_ratio_ttm | 3.5 | 2.3 | 0.0180 |
| 4 | mom_12m | 4.6 | 3.8 | 0.0211 |
| 5 | real_yield_change_6m | 5.5 | 3.9 | 0.0168 |
| 5 | nfci | 5.8 | 2.0 | 0.0118 |
| 6 | credit_spread_hy | 6.2 | 3.7 | 0.0155 |
| 6 | book_value_per_share_growth_yoy | 6.4 | 3.3 | 0.0121 |
| 6 | vol_63d | 6.8 | 3.6 | 0.0132 |
| 6 | real_rate_10y | 6.8 | 3.0 | 0.0125 |
| 7 | investment_income_growth_yoy | 7.1 | 3.1 | 0.0077 |
| 7 | vix | 7.5 | 2.7 | 0.0080 |

---

## Calibration Phase

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Raw BayesianRidge posterior (uncalibrated) | ⬛ Superseded |
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (n=1,200  ECE=1.0% [1.1%–5.7%]) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

**Mean empirical coverage:** 90.6% (target ≥ 80%) ✅  

**Mean trailing 12-point empirical coverage:** 55.2% (gap -24.8% vs nominal) ❌  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | Trailing 12 Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|---------------|----------------------|-------|
| VOO | S&P 500 | +1.43% | -25.21% | +28.06% | 53.27% | 93.0% ✅ | 58.3% | 114 |
| VXUS | Total International Stock | -1.50% | -37.11% | +34.10% | 71.21% | 97.2% ✅ | 50.0% | 108 |
| VWO | Emerging Markets | +0.75% | -31.64% | +33.15% | 64.79% | 95.6% ✅ | 50.0% | 180 |
| VMBS | Mortgage-Backed Securities | +2.27% | -19.96% | +24.50% | 44.46% | 85.8% ✅ | 66.7% | 120 |
| BND | Total Bond Market | +2.00% | -19.65% | +23.66% | 43.31% | 86.0% ✅ | 83.3% | 150 |
| GLD | Gold Shares | -6.60% | -39.37% | +26.17% | 65.54% | 91.7% ✅ | 33.3% | 180 |
| DBC | DB Commodity Index | -11.93% | -41.02% | +17.16% | 58.18% | 88.7% ✅ | 50.0% | 168 |
| VDE | Energy | -9.59% | -41.07% | +21.89% | 62.96% | 86.7% ✅ | 50.0% | 180 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | OOS R² | NW IC | Hit Rate | CW t | CW p |
|-----------|-------------|-------|--------|-------|----------|------|------|
| DBC | DB Commodity Index | 168 | 0.22% | 0.1150 | 69.0% | 1.9981 | 0.0237 |
| VXUS | Total International Stock | 108 | -5.00% | 0.1301 | 69.4% | 1.2736 | 0.1028 |
| VOO | S&P 500 | 114 | -11.63% | -0.0223 | 54.4% | 1.2494 | 0.1070 |
| VDE | Energy | 180 | -2.69% | 0.0074 | 56.7% | 1.2041 | 0.1151 |
| GLD | Gold Shares | 180 | -6.02% | 0.1001 | 54.4% | 1.1840 | 0.1190 |
| VMBS | Mortgage-Backed Securities | 120 | -21.05% | 0.0678 | 80.0% | 0.9921 | 0.1616 |
| VWO | Emerging Markets | 180 | -5.26% | 0.1507 | 62.2% | 0.8718 | 0.1922 |
| BND | Total Bond Market | 150 | -18.74% | 0.2111 | 73.3% | 0.4938 | 0.3111 |

**IC summary:** 5 ✅  1 ⚠️  2 ❌  (of 8 benchmarks)  
**Hit rate ✅:** 6/8 benchmarks above 55% threshold  
**Clark-West ✅:** 1/8 benchmarks with p < 0.05  

---

## Shadow Gate Overlay

| Field | Value |
|-------|-------|
| Variant | gemini_veto_0.50 |
| Recommendation Mode | DEFER-TO-TAX-DEFAULT |
| Recommended Sell % | 50% |
| Would Change Live Output | No |
| Reason | no regression sell to veto |
| P(Actionable Sell) | 40.9% |

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