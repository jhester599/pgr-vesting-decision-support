# PGR Diagnostic Report — February 2026

**As-of Date:** 2026-02-28  
**Horizon:** 6M  
**OOS observations (aggregate):** 1176  
**Newey-West lags:** 5 (accounts for 5-month return-window overlap)  

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² (Campbell-Thompson) | -0.0604 (-6.04%) | ❌ | ≥ 2.00% |
| IC (Newey-West HAC) | 0.1990 | ✅ | ≥ 0.07 |
| IC significance | 0.0000 | ✅ p < 0.05 | p < 0.05 |
| Clark-West t-stat | 2.8441 | ✅ p < 0.05 | p < 0.05 |
| Clark-West p-value | 0.0023 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 67.9% | ✅ | ≥ 55.0% |
| CPCV Positive Paths | 1/7 (12.5%) | ❌ | ≥ 5/7 |

> **Representative CPCV:** benchmark=VOO, model=ridge, paths=7, mean IC=-0.0590, IC std=0.1053.
> Stability verdict: FAIL. Scaled monthly threshold: ≥ 5/7 (maps from the full C(8,2) standard of ≥ 19/28 positive paths).

---


## Feature Governance

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Full obs/feature ratio | 19.25 | ✅ | ≥ 4.0 |
| Per-fold obs/feature ratio | 5.00 | ✅ | ≥ 4.0 |
| Features in monthly run | 12 | — | — |
| Fully populated observations | 231 | — | — |

> obs/feature ratio: 19.2 (full matrix), 5.0 (per WFO fold, 60M window).  n_obs=231, n_features=12.  Verdict: OK.

### Feature Importance Stability

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Mean consecutive-fold Spearman ρ | 0.5557 | ⚠️ MARGINAL | ≥ 0.70 |
| Folds included | 18 | — | — |

> Stability score measures mean pairwise Spearman rank-correlation between consecutive WFO fold importance rankings. A score < 0.40 indicates unstable feature rankings; model predictions may be driven by different features each period.

**Top 10 features by mean WFO rank:**

| Rank | Feature | Mean Rank | Rank Std | Mean |Importance| |
|------|---------|-----------|----------|----------------|
| 3 | combined_ratio_ttm | 3.9 | 2.7 | 0.0193 |
| 5 | mom_12m | 5.0 | 4.2 | 0.0238 |
| 5 | real_rate_10y | 5.4 | 2.5 | 0.0137 |
| 5 | nfci | 5.6 | 2.1 | 0.0120 |
| 5 | credit_spread_hy | 5.9 | 3.5 | 0.0152 |
| 6 | real_yield_change_6m | 6.2 | 4.1 | 0.0147 |
| 6 | book_value_per_share_growth_yoy | 6.4 | 3.3 | 0.0152 |
| 7 | investment_income_growth_yoy | 7.1 | 3.3 | 0.0089 |
| 7 | yield_slope | 7.6 | 3.3 | 0.0090 |
| 7 | vol_63d | 7.7 | 4.2 | 0.0134 |

---

## Calibration Phase

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Raw BayesianRidge posterior (uncalibrated) | ⬛ Superseded |
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (n=1,176  ECE=1.9% [1.6%–6.9%]) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

**Mean empirical coverage:** 89.7% (target ≥ 80%) ✅  

**Mean trailing 12-point empirical coverage:** 60.4% (gap -19.6% vs nominal) ❌  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | Trailing 12 Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|---------------|----------------------|-------|
| VOO | S&P 500 | -1.36% | -30.18% | +27.46% | 57.64% | 94.4% ✅ | 66.7% | 108 |
| VXUS | Total International Stock | +0.34% | -35.50% | +36.17% | 71.67% | 95.1% ✅ | 58.3% | 102 |
| VWO | Emerging Markets | -3.11% | -32.87% | +26.64% | 59.50% | 92.5% ✅ | 50.0% | 174 |
| VMBS | Mortgage-Backed Securities | +2.28% | -17.97% | +22.54% | 40.51% | 82.5% ✅ | 66.7% | 120 |
| BND | Total Bond Market | +3.03% | -18.03% | +24.09% | 42.12% | 86.7% ✅ | 83.3% | 150 |
| GLD | Gold Shares | -10.04% | -73.27% | +53.19% | 126.46% | 100.0% ✅ | 33.3% | 180 |
| DBC | DB Commodity Index | -4.54% | -31.00% | +21.92% | 52.92% | 84.6% ✅ | 66.7% | 162 |
| VDE | Energy | -6.28% | -35.43% | +22.87% | 58.30% | 81.7% ✅ | 58.3% | 180 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | OOS R² | NW IC | Hit Rate | CW t | CW p |
|-----------|-------------|-------|--------|-------|----------|------|------|
| DBC | DB Commodity Index | 162 | 1.82% | 0.2976 | 74.1% | 2.0555 | 0.0207 |
| VOO | S&P 500 | 108 | -11.69% | 0.0036 | 63.9% | 1.7008 | 0.0459 |
| VDE | Energy | 180 | -5.73% | 0.1705 | 64.4% | 1.6451 | 0.0509 |
| VMBS | Mortgage-Backed Securities | 120 | -13.97% | 0.2345 | 80.0% | 1.1521 | 0.1258 |
| VXUS | Total International Stock | 102 | -9.51% | 0.0472 | 68.6% | 1.0310 | 0.1525 |
| VWO | Emerging Markets | 174 | -12.03% | 0.1015 | 62.6% | 0.6387 | 0.2619 |
| BND | Total Bond Market | 150 | -15.94% | 0.2441 | 72.7% | 0.5719 | 0.2841 |
| GLD | Gold Shares | 180 | -14.88% | 0.2198 | 61.1% | 0.4714 | 0.3190 |

**IC summary:** 6 ✅  1 ⚠️  1 ❌  (of 8 benchmarks)  
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
| P(Actionable Sell) | 34.7% |

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