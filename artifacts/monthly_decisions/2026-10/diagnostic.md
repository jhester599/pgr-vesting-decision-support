# PGR Diagnostic Report — October 2026

**As-of Date:** 2026-10-15  
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
| OOS R² vs prevailing mean (gate) | 0.0494 (4.94%) | ✅ | ≥ 2.00% |
| Pooled IC (Spearman) | 0.1165 | ✅ | ≥ 0.07 |
| IC significance (clustered by date) | 0.1244 | ❌ not sig. | p < 0.05 |
| Clark-West t-stat | 3.4035 | ✅ p < 0.05 | p < 0.05 |
| Clark-West p-value | 0.0004 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 63.3% | — | not gated |
| Base rate P(PGR outperforms), same rows | 68.1% | — | constant-sign rule hits 68.1% |
| Hit rate − base rate | -4.9% | — | > 0 |
| Directional skill (Pesaran–Timmermann p, gate) | 0.3113 | ❌ | p < 0.05 |
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
| Mean consecutive-fold Spearman ρ | 0.5563 | ⚠️ MARGINAL | ≥ 0.70 |
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
| 6 | book_value_per_share_growth_yoy | 6.9 | 3.4 | 0.0114 |
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
| npw_vs_npe_spread_pct | 3572931.43 | ❌ HIGH |
| medical_cpi_yoy | 2042135.11 | ❌ HIGH |
| buyback_yield | 1902561.22 | ❌ HIGH |
| legal_services_ppi_relative | 1663019.06 | ❌ HIGH |
| monthly_combined_ratio_delta | 1658501.15 | ❌ HIGH |
| unearned_premium_to_npw_ratio | 1607189.91 | ❌ HIGH |
| loss_ratio_ttm | 1600361.74 | ❌ HIGH |
| excess_bond_premium_proxy | 1362172.76 | ❌ HIGH |
| pif_growth_acceleration | 1332416.91 | ❌ HIGH |
| npw_per_pif_yoy | 1268408.62 | ❌ HIGH |
| credit_spread_ratio | 1090261.14 | ❌ HIGH |
| npw_growth_yoy | 1052881.49 | ❌ HIGH |
| roe_net_income_ttm | 995415.66 | ❌ HIGH |
| mom_6m | 976070.55 | ❌ HIGH |
| baa10y_spread | 823284.72 | ❌ HIGH |
| investment_book_yield | 802376.29 | ❌ HIGH |
| vwo_vxus_spread_6m | 767428.03 | ❌ HIGH |
| expense_ratio_ttm | 664714.40 | ❌ HIGH |
| mortgage_spread_30y_10y | 662758.96 | ❌ HIGH |
| ppi_auto_ins_yoy | 588399.58 | ❌ HIGH |
| pgr_vs_vfh_6m | 546037.68 | ❌ HIGH |
| pgr_vs_kie_6m | 523587.74 | ❌ HIGH |
| gold_vs_treasury_6m | 496951.65 | ❌ HIGH |
| yield_curvature | 457705.57 | ❌ HIGH |
| pgr_price_to_book_relative | 441980.89 | ❌ HIGH |
| usd_momentum_6m | 440578.25 | ❌ HIGH |
| underwriting_margin_ttm | 424268.49 | ❌ HIGH |
| nfci | 413875.44 | ❌ HIGH |
| vix | 407754.43 | ❌ HIGH |
| pe_ratio | 404869.29 | ❌ HIGH |
| underwriting_income | 391109.50 | ❌ HIGH |
| severity_index_yoy | 388587.46 | ❌ HIGH |
| book_value_per_share_growth_yoy | 379417.18 | ❌ HIGH |
| mom_3m | 372422.03 | ❌ HIGH |
| credit_spread_hy | 347187.51 | ❌ HIGH |
| underwriting_income_growth_yoy | 339196.29 | ❌ HIGH |
| gainshare_est | 334026.87 | ❌ HIGH |
| roe_trend | 330964.06 | ❌ HIGH |
| channel_mix_direct_pct_yoy | 302185.44 | ❌ HIGH |
| real_rate_10y | 276176.65 | ❌ HIGH |
| commodity_equity_momentum | 263036.52 | ❌ HIGH |
| pif_growth_yoy | 246923.83 | ❌ HIGH |
| auto_pricing_power_spread | 246407.85 | ❌ HIGH |
| vol_63d | 245323.09 | ❌ HIGH |
| rate_adequacy_gap_yoy | 225619.00 | ❌ HIGH |
| mom_12m | 221865.11 | ❌ HIGH |
| real_yield_change_6m | 207589.23 | ❌ HIGH |
| channel_mix_agency_pct | 207209.11 | ❌ HIGH |
| wti_return_3m | 203479.23 | ❌ HIGH |
| cr_acceleration | 203207.67 | ❌ HIGH |
| breakeven_inflation_10y | 202247.62 | ❌ HIGH |
| equity_risk_premium | 191737.29 | ❌ HIGH |
| term_premium_10y | 168551.20 | ❌ HIGH |
| motor_vehicle_ins_cpi_yoy | 166868.26 | ❌ HIGH |
| usd_broad_return_3m | 162413.68 | ❌ HIGH |
| pgr_pe_vs_market_pe | 153117.67 | ❌ HIGH |
| reserve_to_npe_ratio | 152382.60 | ❌ HIGH |
| high_52w | 150943.83 | ❌ HIGH |
| pgr_vs_peers_6m | 147626.85 | ❌ HIGH |
| direct_channel_pif_share_ttm | 136081.77 | ❌ HIGH |
| pb_ratio | 131667.92 | ❌ HIGH |
| realized_gain_to_net_income_ratio | 131541.79 | ❌ HIGH |
| unrealized_gain_pct_equity | 121313.83 | ❌ HIGH |
| duration_rate_shock_3m | 113386.20 | ❌ HIGH |
| buyback_acceleration | 109344.84 | ❌ HIGH |
| vmt_yoy | 102184.98 | ❌ HIGH |
| roe | 100905.57 | ❌ HIGH |
| unearned_premium_growth_yoy | 98684.22 | ❌ HIGH |
| breakeven_momentum_3m | 89453.12 | ❌ HIGH |
| gasoline_retail_sales_delta | 73118.85 | ❌ HIGH |
| underwriting_income_3m | 67140.04 | ❌ HIGH |
| used_car_cpi_yoy | 50767.06 | ❌ HIGH |
| pgr_premium_to_surplus | 43548.80 | ❌ HIGH |
| investment_income_growth_yoy | 37092.44 | ❌ HIGH |
| yield_slope | 29469.23 | ❌ HIGH |

---

## Calibration Phase

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Uncalibrated (P = 50%; too few realised OOS rows) | ⬛ Superseded |
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (prequential ECE=15.9% [10.7%–24.9%] over n=1,025 OOS benchmark-months) |
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

**Mean trailing 12-point empirical coverage:** 42.7% (gap -37.3% vs nominal) ❌  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Trailing 12 Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|----------------------|-------|
| VOO | S&P 500 | -0.71% | -29.56% | +28.14% | 57.69% | 58.3% ❌ | 114 |
| VXUS | Total International Stock | +1.23% | -33.97% | +36.43% | 70.40% | 41.7% ❌ | 114 |
| VWO | Emerging Markets | +2.46% | -31.05% | +35.97% | 67.02% | 41.7% ❌ | 180 |
| VMBS | Mortgage-Backed Securities | +3.50% | -23.12% | +30.12% | 53.24% | 41.7% ❌ | 126 |
| BND | Total Bond Market | +2.89% | -22.86% | +28.64% | 51.50% | 50.0% ❌ | 156 |
| GLD | Gold Shares | +2.43% | -31.06% | +35.93% | 66.99% | 33.3% ❌ | 186 |
| DBC | DB Commodity Index | -2.06% | -32.88% | +28.75% | 61.63% | 41.7% ❌ | 174 |
| VDE | Energy | -8.36% | -46.10% | +29.38% | 75.47% | 33.3% ❌ | 186 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | OOS R² | IC | NW p | Hit Rate | Base Rate | PT p | CW t | CW p |
|-----------|-------------|-------|--------|----|------|----------|-----------|------|------|------|
| GLD | Gold Shares | 186 | 17.35% | 0.1649 | 0.2405 | 61.8% | 67.2% | 0.2459 | 3.5441 | 0.0002 |
| DBC | DB Commodity Index | 174 | 15.19% | 0.2759 | 0.0205 | 70.7% | 67.8% | 0.0047 | 2.9164 | 0.0020 |
| VWO | Emerging Markets | 180 | 13.53% | 0.0858 | 0.5113 | 61.1% | 68.3% | 0.9923 | 2.7945 | 0.0029 |
| BND | Total Bond Market | 156 | -6.26% | 0.1217 | 0.3428 | 72.4% | 75.0% | 0.1555 | 1.4780 | 0.0707 |
| VOO | S&P 500 | 114 | -4.09% | 0.1069 | 0.4527 | 57.9% | 63.2% | 0.3400 | 1.4370 | 0.0767 |
| VDE | Energy | 186 | -0.44% | 0.0195 | 0.8625 | 58.1% | 61.8% | 0.5104 | 1.3557 | 0.0884 |
| VMBS | Mortgage-Backed Securities | 126 | -13.93% | 0.0002 | 0.9985 | 65.9% | 75.4% | 0.9855 | 0.6592 | 0.2555 |
| VXUS | Total International Stock | 114 | -12.82% | -0.0110 | 0.9451 | 56.1% | 67.5% | 0.9342 | 0.0866 | 0.4656 |

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