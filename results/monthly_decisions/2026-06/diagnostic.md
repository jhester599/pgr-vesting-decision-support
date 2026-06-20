# PGR Diagnostic Report — June 2026

**As-of Date:** 2026-06-22  
**Horizon:** 6M  
**OOS observations (aggregate):** 1212  
**Newey-West lags:** 5 (accounts for 5-month return-window overlap)  

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² (Campbell-Thompson) | -0.0213 (-2.13%) | ❌ | ≥ 2.00% |
| IC (Newey-West HAC) | 0.1282 | ✅ | ≥ 0.07 |
| IC significance | 0.0025 | ✅ p < 0.05 | p < 0.05 |
| Clark-West t-stat | 3.5237 | ✅ p < 0.05 | p < 0.05 |
| Clark-West p-value | 0.0002 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 63.4% | ✅ | ≥ 55.0% |
| CPCV Positive Paths | 1/7 (12.5%) | ❌ | ≥ 5/7 |

> **Representative CPCV:** benchmark=VOO, model=ridge, paths=7, mean IC=-0.0586, IC std=0.1095.
> Stability verdict: FAIL. Scaled monthly threshold: ≥ 5/7 (maps from the full C(8,2) standard of ≥ 19/28 positive paths).

---


## Feature Governance

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Full obs/feature ratio | 20.33 | ✅ | ≥ 4.0 |
| Per-fold obs/feature ratio | 5.00 | ✅ | ≥ 4.0 |
| Features in monthly run | 12 | — | — |
| Fully populated observations | 244 | — | — |

> obs/feature ratio: 20.3 (full matrix), 5.0 (per WFO fold, 60M window).  n_obs=244, n_features=12.  Verdict: OK.

### Feature Importance Stability

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Mean consecutive-fold Spearman ρ | 0.5699 | ⚠️ MARGINAL | ≥ 0.70 |
| Folds included | 19 | — | — |

> Stability score measures mean pairwise Spearman rank-correlation between consecutive WFO fold importance rankings. A score < 0.40 indicates unstable feature rankings; model predictions may be driven by different features each period.

**Top 10 features by mean WFO rank:**

| Rank | Feature | Mean Rank | Rank Std | Mean |Importance| |
|------|---------|-----------|----------|----------------|
| 3 | combined_ratio_ttm | 3.5 | 2.8 | 0.0173 |
| 4 | mom_12m | 4.4 | 3.6 | 0.0216 |
| 5 | credit_spread_hy | 5.5 | 3.2 | 0.0193 |
| 5 | nfci | 5.7 | 2.4 | 0.0150 |
| 6 | real_yield_change_6m | 6.4 | 4.2 | 0.0177 |
| 6 | book_value_per_share_growth_yoy | 6.5 | 3.4 | 0.0122 |
| 6 | real_rate_10y | 6.6 | 3.0 | 0.0187 |
| 7 | vol_63d | 7.2 | 4.0 | 0.0114 |
| 7 | investment_income_growth_yoy | 7.2 | 3.2 | 0.0071 |
| 7 | vix | 7.6 | 2.3 | 0.0065 |

---

## Calibration Phase

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Raw BayesianRidge posterior (uncalibrated) | ⬛ Superseded |
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (n=1,212  ECE=1.3% [1.4%–6.1%]) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

**Mean empirical coverage:** 90.6% (target ≥ 80%) ✅  

**Mean trailing 12-point empirical coverage:** 57.3% (gap -22.7% vs nominal) ❌  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | Trailing 12 Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|---------------|----------------------|-------|
| VOO | S&P 500 | -1.10% | -27.09% | +24.89% | 51.98% | 91.2% ✅ | 58.3% | 114 |
| VXUS | Total International Stock | +4.34% | -30.44% | +39.12% | 69.56% | 94.4% ✅ | 58.3% | 108 |
| VWO | Emerging Markets | -0.71% | -30.21% | +28.79% | 58.99% | 91.7% ✅ | 58.3% | 180 |
| VMBS | Mortgage-Backed Securities | +1.55% | -20.48% | +23.59% | 44.07% | 82.5% ✅ | 75.0% | 120 |
| BND | Total Bond Market | +3.12% | -18.77% | +25.01% | 43.78% | 85.9% ✅ | 75.0% | 156 |
| GLD | Gold Shares | -7.22% | -38.94% | +24.49% | 63.43% | 91.7% ✅ | 41.7% | 180 |
| DBC | DB Commodity Index | -14.35% | -59.29% | +30.58% | 89.87% | 98.8% ✅ | 50.0% | 168 |
| VDE | Energy | -14.07% | -47.34% | +19.21% | 66.55% | 88.7% ✅ | 41.7% | 186 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | OOS R² | NW IC | Hit Rate | CW t | CW p |
|-----------|-------------|-------|--------|-------|----------|------|------|
| VDE | Energy | 186 | -0.29% | 0.1053 | 57.5% | 2.3019 | 0.0112 |
| DBC | DB Commodity Index | 168 | 2.04% | 0.1903 | 69.6% | 1.7741 | 0.0389 |
| VOO | S&P 500 | 114 | -5.81% | 0.1038 | 58.8% | 1.5796 | 0.0585 |
| GLD | Gold Shares | 180 | -2.20% | 0.1767 | 56.7% | 1.5552 | 0.0608 |
| VMBS | Mortgage-Backed Securities | 120 | -17.26% | 0.1239 | 75.0% | 1.1122 | 0.1341 |
| BND | Total Bond Market | 156 | -20.07% | 0.2086 | 67.9% | 1.0528 | 0.1470 |
| VXUS | Total International Stock | 108 | -11.68% | -0.0176 | 63.0% | 0.8862 | 0.1887 |
| VWO | Emerging Markets | 180 | -8.12% | 0.0233 | 61.7% | 0.5484 | 0.2920 |

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
| P(Actionable Sell) | 44.9% |

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