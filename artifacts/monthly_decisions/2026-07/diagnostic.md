# PGR Diagnostic Report — July 2026

**As-of Date:** 2026-07-22  
**Horizon:** 6M  
**OOS observations (aggregate):** 1218  
**Newey-West lags:** 5 (accounts for 5-month return-window overlap)  

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² (Campbell-Thompson) | -0.0278 (-2.78%) | ❌ | ≥ 2.00% |
| IC (Newey-West HAC) | 0.1195 | ✅ | ≥ 0.07 |
| IC significance | 0.0043 | ✅ p < 0.05 | p < 0.05 |
| Clark-West t-stat | 3.2467 | ✅ p < 0.05 | p < 0.05 |
| Clark-West p-value | 0.0006 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 64.2% | ✅ | ≥ 55.0% |
| CPCV Positive Paths | 1/7 (12.5%) | ❌ | ≥ 5/7 |

> **Representative CPCV:** benchmark=VOO, model=ridge, paths=7, mean IC=-0.0709, IC std=0.1187.
> Stability verdict: FAIL. Scaled monthly threshold: ≥ 5/7 (maps from the full C(8,2) standard of ≥ 19/28 positive paths).

---


## Feature Governance

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Full obs/feature ratio | 20.42 | ✅ | ≥ 4.0 |
| Per-fold obs/feature ratio | 5.00 | ✅ | ≥ 4.0 |
| Features in monthly run | 12 | — | — |
| Fully populated observations | 245 | — | — |

> obs/feature ratio: 20.4 (full matrix), 5.0 (per WFO fold, 60M window).  n_obs=245, n_features=12.  Verdict: OK.

### Feature Importance Stability

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Mean consecutive-fold Spearman ρ | 0.6193 | ⚠️ MARGINAL | ≥ 0.70 |
| Folds included | 19 | — | — |

> Stability score measures mean pairwise Spearman rank-correlation between consecutive WFO fold importance rankings. A score < 0.40 indicates unstable feature rankings; model predictions may be driven by different features each period.

**Top 10 features by mean WFO rank:**

| Rank | Feature | Mean Rank | Rank Std | Mean |Importance| |
|------|---------|-----------|----------|----------------|
| 3 | combined_ratio_ttm | 3.9 | 3.2 | 0.0182 |
| 4 | mom_12m | 4.2 | 3.5 | 0.0228 |
| 5 | credit_spread_hy | 5.3 | 3.1 | 0.0210 |
| 5 | nfci | 5.6 | 2.1 | 0.0163 |
| 6 | real_rate_10y | 6.1 | 3.2 | 0.0206 |
| 6 | book_value_per_share_growth_yoy | 6.4 | 3.6 | 0.0141 |
| 6 | real_yield_change_6m | 6.4 | 3.9 | 0.0216 |
| 6 | vol_63d | 6.9 | 3.7 | 0.0142 |
| 7 | investment_income_growth_yoy | 7.2 | 3.1 | 0.0095 |
| 8 | vix | 8.0 | 2.7 | 0.0070 |

---

## Calibration Phase

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Raw BayesianRidge posterior (uncalibrated) | ⬛ Superseded |
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (n=1,218  ECE=3.0% [2.1%–7.6%]) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

**Mean empirical coverage:** 90.7% (target ≥ 80%) ✅  

**Mean trailing 12-point empirical coverage:** 57.3% (gap -22.7% vs nominal) ❌  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | Trailing 12 Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|---------------|----------------------|-------|
| VOO | S&P 500 | -1.14% | -26.39% | +24.12% | 50.51% | 88.6% ✅ | 66.7% | 114 |
| VXUS | Total International Stock | +3.33% | -30.79% | +37.45% | 68.24% | 94.4% ✅ | 58.3% | 108 |
| VWO | Emerging Markets | -5.11% | -34.27% | +24.05% | 58.32% | 91.7% ✅ | 58.3% | 180 |
| VMBS | Mortgage-Backed Securities | +0.13% | -21.84% | +22.10% | 43.94% | 82.5% ✅ | 66.7% | 120 |
| BND | Total Bond Market | +1.96% | -19.04% | +22.96% | 42.00% | 85.9% ✅ | 75.0% | 156 |
| GLD | Gold Shares | -7.47% | -44.27% | +29.32% | 73.59% | 95.7% ✅ | 41.7% | 186 |
| DBC | DB Commodity Index | -14.19% | -58.22% | +29.84% | 88.06% | 98.8% ✅ | 50.0% | 168 |
| VDE | Energy | -13.47% | -46.24% | +19.29% | 65.53% | 87.6% ✅ | 41.7% | 186 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | OOS R² | NW IC | Hit Rate | CW t | CW p |
|-----------|-------------|-------|--------|-------|----------|------|------|
| VDE | Energy | 186 | -3.38% | 0.1363 | 59.7% | 2.0841 | 0.0193 |
| DBC | DB Commodity Index | 168 | 0.73% | 0.1932 | 69.6% | 1.7244 | 0.0432 |
| GLD | Gold Shares | 186 | 1.45% | 0.2031 | 55.4% | 1.3839 | 0.0840 |
| VMBS | Mortgage-Backed Securities | 120 | -11.26% | 0.2036 | 76.7% | 1.3447 | 0.0906 |
| VOO | S&P 500 | 114 | -12.02% | -0.0364 | 60.5% | 1.3110 | 0.0962 |
| BND | Total Bond Market | 156 | -9.75% | 0.3227 | 70.5% | 0.9293 | 0.1771 |
| VXUS | Total International Stock | 108 | -16.28% | -0.1650 | 58.3% | 0.6300 | 0.2650 |
| VWO | Emerging Markets | 180 | -12.22% | -0.1004 | 65.0% | 0.0474 | 0.4811 |

**IC summary:** 5 ✅  0 ⚠️  3 ❌  (of 8 benchmarks)  
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
| P(Actionable Sell) | 41.2% |

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