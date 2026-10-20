# PGR Diagnostic Report — October 2026

**As-of Date:** 2026-10-20  
**Horizon:** 6M  
**OOS observations (aggregate):** 1236 over 186 months  
**HAC lags:** 5 months (accounts for the 5-month return-window overlap). Pooled p-values are Driscoll-Kraay: Newey-West on the monthly sums of the scores, i.e. clustered by date. Per-benchmark p-values are Newey-West.  
**Prequential shrinkage alpha (live):** 0.500  

> All metrics below are realised-only (review 2026-09-25, WP7): every OOS month uses
> ensemble weights, shrinkage and calibration fitted on targets realised by that month,
> and OOS R² / Clark-West compare with each benchmark's prevailing mean of the targets
> realised by then (training history included).

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² vs prevailing mean (gate) | 0.0485 (4.85%) | ✅ | ≥ 2.00% |
| Pooled IC (Spearman) | 0.1166 | ✅ | ≥ 0.07 |
| IC significance (clustered by date) | 0.1243 | ❌ not sig. | p < 0.05 |
| Clark-West t-stat | 3.4081 | ✅ p < 0.05 | p < 0.05 |
| Clark-West p-value | 0.0004 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 63.0% | — | not gated |
| Base rate P(PGR outperforms), same rows | 68.1% | — | constant-sign rule hits 68.1% |
| Hit rate − base rate | -5.1% | — | > 0 |
| Directional skill (Pesaran–Timmermann p, gate) | 0.3490 | ❌ | p < 0.05 |
| Walk-forward validation complete (gate) | complete (16 required model/benchmark pairs) | ✅ | every required pair |
| Required inputs ready at as-of (gate) | ready | ✅ | finite and fresh |

> The recommendation gate uses the equal-weight mean of the per-benchmark ICs
> (see recommendation.md), not the pooled IC above.

> **Validation:** walk-forward only (TimeSeriesSplit, 60-month window, 6-month test folds,
> gap = horizon + purge buffer = 8 months). The combinatorial purged K-fold diagnostic
> was retired (pre-v200 remediation R3): it trained on folds after its test folds.

---


## Feature Governance

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Full obs/feature ratio | 21.08 | ✅ | ≥ 4.0 |
| Per-fold obs/feature ratio | 5.00 | ✅ | ≥ 4.0 |
| Features in monthly run | 12 | — | — |
| Fully populated observations | 253 | — | — |

> obs/feature ratio: 21.1 (full matrix), 5.0 (per WFO fold, 60M window).  n_obs=253, n_features=12.  Verdict: OK.

### Feature Importance Stability

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Mean consecutive-fold Spearman ρ | 0.5571 | ⚠️ MARGINAL | ≥ 0.70 |
| Folds included | 19 | — | — |

> Stability score measures mean pairwise Spearman rank-correlation between consecutive WFO fold importance rankings. A score < 0.40 indicates unstable feature rankings; model predictions may be driven by different features each period.

**Top 10 features by mean WFO rank:**

| Rank | Feature | Mean Rank | Rank Std | Mean |Importance| |
|------|---------|-----------|----------|----------------|
| 3 | mom_12m | 3.9 | 2.1 | 0.0257 |
| 4 | combined_ratio_ttm | 4.6 | 4.2 | 0.0355 |
| 4 | credit_spread_hy | 4.8 | 3.6 | 0.0230 |
| 5 | real_yield_change_6m | 5.2 | 3.5 | 0.0173 |
| 5 | yield_slope | 5.9 | 3.3 | 0.0160 |
| 6 | vix | 6.8 | 3.5 | 0.0173 |
| 7 | book_value_per_share_growth_yoy | 7.0 | 3.4 | 0.0114 |
| 7 | real_rate_10y | 7.3 | 2.1 | 0.0120 |
| 7 | vol_63d | 7.4 | 3.2 | 0.0144 |
| 7 | nfci | 7.7 | 3.2 | 0.0119 |

### Multicollinearity (VIF)

**Overall:** ❌ HIGH multicollinearity  
Features flagged high (VIF > 10): **76**  
Features flagged moderate (VIF 5–10): **0**  

> VIF measures how much variance in a feature is explained by the other features. VIF > 10 indicates severe multicollinearity and may cause unstable coefficient estimates.

**All features by VIF (descending):**

| Feature | VIF | Status |
|---------|-----|--------|
| combined_ratio_ttm | 1000799917193443.50 | ❌ HIGH |
| roe_trend | 2939026.51 | ❌ HIGH |
| vol_63d | 1683471.63 | ❌ HIGH |
| pif_growth_acceleration | 1421617.01 | ❌ HIGH |
| duration_rate_shock_3m | 1377737.81 | ❌ HIGH |
| monthly_combined_ratio_delta | 1274703.51 | ❌ HIGH |
| pgr_vs_vfh_6m | 1259123.91 | ❌ HIGH |
| real_yield_change_6m | 1250003.57 | ❌ HIGH |
| npw_vs_npe_spread_pct | 1203184.65 | ❌ HIGH |
| underwriting_income | 1135023.47 | ❌ HIGH |
| mortgage_spread_30y_10y | 1116568.74 | ❌ HIGH |
| pgr_price_to_book_relative | 1063452.69 | ❌ HIGH |
| credit_spread_ratio | 931538.45 | ❌ HIGH |
| mom_12m | 900908.41 | ❌ HIGH |
| gainshare_est | 790288.99 | ❌ HIGH |
| breakeven_momentum_3m | 760988.06 | ❌ HIGH |
| pgr_pe_vs_market_pe | 736686.79 | ❌ HIGH |
| medical_cpi_yoy | 715695.32 | ❌ HIGH |
| pif_growth_yoy | 713037.25 | ❌ HIGH |
| unrealized_gain_pct_equity | 679574.71 | ❌ HIGH |
| vwo_vxus_spread_6m | 660908.89 | ❌ HIGH |
| loss_ratio_ttm | 654455.84 | ❌ HIGH |
| nfci | 623599.93 | ❌ HIGH |
| usd_momentum_6m | 581109.41 | ❌ HIGH |
| term_premium_10y | 552001.38 | ❌ HIGH |
| book_value_per_share_growth_yoy | 534339.82 | ❌ HIGH |
| expense_ratio_ttm | 510992.45 | ❌ HIGH |
| excess_bond_premium_proxy | 463897.12 | ❌ HIGH |
| commodity_equity_momentum | 457338.84 | ❌ HIGH |
| roe_net_income_ttm | 455416.73 | ❌ HIGH |
| usd_broad_return_3m | 455390.81 | ❌ HIGH |
| high_52w | 435330.87 | ❌ HIGH |
| ppi_auto_ins_yoy | 429618.29 | ❌ HIGH |
| underwriting_income_growth_yoy | 428343.13 | ❌ HIGH |
| credit_spread_hy | 387552.56 | ❌ HIGH |
| wti_return_3m | 384042.12 | ❌ HIGH |
| unearned_premium_to_npw_ratio | 375030.76 | ❌ HIGH |
| baa10y_spread | 344685.63 | ❌ HIGH |
| pe_ratio | 317155.80 | ❌ HIGH |
| equity_risk_premium | 300470.04 | ❌ HIGH |
| realized_gain_to_net_income_ratio | 295362.33 | ❌ HIGH |
| npw_growth_yoy | 271418.44 | ❌ HIGH |
| rate_adequacy_gap_yoy | 269772.79 | ❌ HIGH |
| channel_mix_direct_pct_yoy | 256928.55 | ❌ HIGH |
| npw_per_pif_yoy | 235193.91 | ❌ HIGH |
| cr_acceleration | 228043.67 | ❌ HIGH |
| underwriting_margin_ttm | 226360.64 | ❌ HIGH |
| used_car_cpi_yoy | 190490.08 | ❌ HIGH |
| investment_income_growth_yoy | 186475.48 | ❌ HIGH |
| gold_vs_treasury_6m | 173687.73 | ❌ HIGH |
| legal_services_ppi_relative | 173650.52 | ❌ HIGH |
| gasoline_retail_sales_delta | 172177.56 | ❌ HIGH |
| direct_channel_pif_share_ttm | 166762.31 | ❌ HIGH |
| underwriting_income_3m | 166304.64 | ❌ HIGH |
| motor_vehicle_ins_cpi_yoy | 157243.70 | ❌ HIGH |
| breakeven_inflation_10y | 154885.05 | ❌ HIGH |
| roe | 151976.73 | ❌ HIGH |
| yield_slope | 138309.25 | ❌ HIGH |
| pgr_premium_to_surplus | 132421.42 | ❌ HIGH |
| unearned_premium_growth_yoy | 115012.56 | ❌ HIGH |
| vmt_yoy | 97084.09 | ❌ HIGH |
| yield_curvature | 95562.83 | ❌ HIGH |
| reserve_to_npe_ratio | 95206.08 | ❌ HIGH |
| auto_pricing_power_spread | 92575.21 | ❌ HIGH |
| vix | 90206.16 | ❌ HIGH |
| mom_6m | 87338.19 | ❌ HIGH |
| pgr_vs_kie_6m | 84861.69 | ❌ HIGH |
| buyback_acceleration | 77026.75 | ❌ HIGH |
| channel_mix_agency_pct | 73970.44 | ❌ HIGH |
| pgr_vs_peers_6m | 72734.13 | ❌ HIGH |
| buyback_yield | 63320.44 | ❌ HIGH |
| pb_ratio | 59066.92 | ❌ HIGH |
| real_rate_10y | 35146.33 | ❌ HIGH |
| mom_3m | 30276.86 | ❌ HIGH |
| severity_index_yoy | 28482.82 | ❌ HIGH |
| investment_book_yield | 20818.88 | ❌ HIGH |

---

## Calibration Phase

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Uncalibrated (P = 50%; too few realised OOS rows) | ⬛ Superseded |
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (prequential ECE=15.6% [10.5%–24.6%] over n=1,025 OOS benchmark-months) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

> ECE is prequential: each OOS month is scored by the per-benchmark calibrator fitted only
> on months whose 6-month outcome was already known, as the monthly run would have done.
> Confidence tiers read the calibrated P(outperform) in the direction of each signal.

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

> Coverage is trailing and prequential: each of the last 12 OOS months per benchmark is
> scored with an interval calibrated only on residuals realised by that month.

**Mean trailing 12-point empirical coverage:** 41.7% (gap -38.3% vs nominal) ❌  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Trailing 12 Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|----------------------|-------|
| VOO | S&P 500 | -0.72% | -26.28% | +24.84% | 51.12% | 50.0% ❌ | 114 |
| VXUS | Total International Stock | +1.23% | -33.97% | +36.44% | 70.41% | 41.7% ❌ | 114 |
| VWO | Emerging Markets | +2.46% | -31.05% | +35.97% | 67.02% | 41.7% ❌ | 180 |
| VMBS | Mortgage-Backed Securities | +3.50% | -23.12% | +30.12% | 53.24% | 41.7% ❌ | 126 |
| BND | Total Bond Market | +2.88% | -22.86% | +28.63% | 51.50% | 50.0% ❌ | 156 |
| GLD | Gold Shares | +2.43% | -31.06% | +35.93% | 66.99% | 33.3% ❌ | 186 |
| DBC | DB Commodity Index | -2.06% | -32.83% | +28.71% | 61.55% | 41.7% ❌ | 174 |
| VDE | Energy | -8.60% | -45.95% | +28.75% | 74.70% | 33.3% ❌ | 186 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | OOS R² | IC | NW p | Hit Rate | Base Rate | PT p | CW t | CW p |
|-----------|-------------|-------|--------|----|------|----------|-----------|------|------|------|
| GLD | Gold Shares | 186 | 17.32% | 0.1649 | 0.2403 | 61.8% | 67.2% | 0.2459 | 3.5452 | 0.0002 |
| DBC | DB Commodity Index | 174 | 15.15% | 0.2749 | 0.0213 | 70.1% | 67.8% | 0.0168 | 2.9165 | 0.0020 |
| VWO | Emerging Markets | 180 | 13.53% | 0.0862 | 0.5094 | 61.1% | 68.3% | 0.9923 | 2.7982 | 0.0029 |
| BND | Total Bond Market | 156 | -6.87% | 0.1229 | 0.3381 | 71.8% | 75.0% | 0.1783 | 1.4690 | 0.0719 |
| VOO | S&P 500 | 114 | -3.88% | 0.1090 | 0.4465 | 57.0% | 63.2% | 0.3679 | 1.4638 | 0.0730 |
| VDE | Energy | 186 | -0.45% | 0.0164 | 0.8840 | 58.1% | 61.8% | 0.5104 | 1.3629 | 0.0873 |
| VMBS | Mortgage-Backed Securities | 126 | -14.70% | 0.0008 | 0.9950 | 65.9% | 75.4% | 0.9855 | 0.6355 | 0.2631 |
| VXUS | Total International Stock | 114 | -12.75% | -0.0112 | 0.9441 | 56.1% | 67.5% | 0.9342 | 0.0968 | 0.4615 |

**IC summary:** 5 ✅  0 ⚠️  3 ❌  (of 8 benchmarks)  
**Directional skill ✅:** 1/8 benchmarks with Pesaran–Timmermann p < 0.05  
**Clark-West ✅:** 3/8 benchmarks with p < 0.05  

---

## Shadow Gate Overlay

| Field | Value |
|-------|-------|
| Variant | gemini_veto_0.50 |
| Recommendation Mode | DEFER-TO-TAX-DEFAULT |
| Recommended Sell % | 50% |
| Would Change Live Output | No |
| Reason | no regression deviation to veto |
| P(Actionable Sell) | 45.2% |

---

## Classifier Monitoring

| Metric | Value |
|--------|-------|
| Matured observations | 6 |
| Brier score | 0.3546 |
| Log loss | 0.9077 |
| ECE (10-bin) | 0.4700 |

> Matured-horizon diagnostics are computed only once the forecast horizon has elapsed.

---

## Threshold Reference

| Metric | Good | Marginal | Failing | Source |
|--------|------|----------|---------|--------|
| OOS R² vs prevailing mean (gate) | > 2% | 0–2% | < 0% | Campbell & Thompson (2008) |
| Mean IC, equal-weight (gate) | > 0.07 | 0.03–0.07 | < 0.03 | Harvey et al. (2016) |
| Directional skill, PT p (gate) | < 0.05 | 0.05–0.10 | ≥ 0.10 | Pesaran & Timmermann (1992, 2009) |
| Clark-West | p < 0.05 | p < 0.10 | ≥ 0.10 | Clark & West (2007) |
| PBO | < 15% | 15–40% | > 40% | Bailey et al. (2014) |

> Incomplete walk-forward results or required inputs that are missing, non-finite or stale
> at the as-of date withhold ACTIONABLE (fail closed).

---

*Generated by `scripts/monthly_decision.py`*