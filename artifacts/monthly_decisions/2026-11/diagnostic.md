# PGR Diagnostic Report — November 2026

**As-of Date:** 2026-11-20  
**Horizon:** 6M  
**OOS observations (aggregate):** 1248 over 186 months  
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
| OOS R² vs prevailing mean (gate) | 0.0352 (3.52%) | ✅ | ≥ 2.00% |
| Pooled IC (Spearman) | 0.1111 | ✅ | ≥ 0.07 |
| IC significance (clustered by date) | 0.1358 | ❌ not sig. | p < 0.05 |
| Clark-West t-stat | 3.0635 | ✅ p < 0.05 | p < 0.05 |
| Clark-West p-value | 0.0013 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 62.3% | — | not gated |
| Base rate P(PGR outperforms), same rows | 68.3% | — | constant-sign rule hits 68.3% |
| Hit rate − base rate | -6.0% | — | > 0 |
| Directional skill (Pesaran–Timmermann p, gate) | 0.4624 | ❌ | p < 0.05 |
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
| Full obs/feature ratio | 21.17 | ✅ | ≥ 4.0 |
| Per-fold obs/feature ratio | 5.00 | ✅ | ≥ 4.0 |
| Features in monthly run | 12 | — | — |
| Fully populated observations | 254 | — | — |

> obs/feature ratio: 21.2 (full matrix), 5.0 (per WFO fold, 60M window).  n_obs=254, n_features=12.  Verdict: OK.

### Feature Importance Stability

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Mean consecutive-fold Spearman ρ | 0.6250 | ⚠️ MARGINAL | ≥ 0.70 |
| Folds included | 20 | — | — |

> Stability score measures mean pairwise Spearman rank-correlation between consecutive WFO fold importance rankings. A score < 0.40 indicates unstable feature rankings; model predictions may be driven by different features each period.

**Top 10 features by mean WFO rank:**

| Rank | Feature | Mean Rank | Rank Std | Mean |Importance| |
|------|---------|-----------|----------|----------------|
| 4 | mom_12m | 4.1 | 2.0 | 0.0236 |
| 4 | credit_spread_hy | 4.2 | 3.1 | 0.0235 |
| 4 | combined_ratio_ttm | 4.7 | 4.2 | 0.0305 |
| 5 | real_yield_change_6m | 5.3 | 3.8 | 0.0165 |
| 5 | yield_slope | 5.5 | 2.9 | 0.0157 |
| 5 | book_value_per_share_growth_yoy | 5.9 | 3.3 | 0.0138 |
| 6 | vix | 6.8 | 3.2 | 0.0155 |
| 7 | vol_63d | 7.7 | 3.2 | 0.0126 |
| 8 | real_rate_10y | 8.0 | 2.5 | 0.0092 |
| 8 | npw_growth_yoy | 8.1 | 2.7 | 0.0088 |

### Multicollinearity (VIF)

**Overall:** ❌ HIGH multicollinearity  
Features flagged high (VIF > 10): **76**  
Features flagged moderate (VIF 5–10): **0**  

> VIF measures how much variance in a feature is explained by the other features. VIF > 10 indicates severe multicollinearity and may cause unstable coefficient estimates.

**All features by VIF (descending):**

| Feature | VIF | Status |
|---------|-----|--------|
| combined_ratio_ttm | 1000799917193443.50 | ❌ HIGH |
| mom_3m | 1704084.75 | ❌ HIGH |
| unearned_premium_to_npw_ratio | 1319789.97 | ❌ HIGH |
| pgr_vs_vfh_6m | 1275236.96 | ❌ HIGH |
| underwriting_income | 1264748.53 | ❌ HIGH |
| mortgage_spread_30y_10y | 1247197.11 | ❌ HIGH |
| expense_ratio_ttm | 1132913.88 | ❌ HIGH |
| pif_growth_yoy | 1068415.67 | ❌ HIGH |
| breakeven_momentum_3m | 1033267.60 | ❌ HIGH |
| legal_services_ppi_relative | 1027832.17 | ❌ HIGH |
| gasoline_retail_sales_delta | 906117.18 | ❌ HIGH |
| commodity_equity_momentum | 862136.95 | ❌ HIGH |
| pgr_pe_vs_market_pe | 857991.99 | ❌ HIGH |
| credit_spread_hy | 845017.84 | ❌ HIGH |
| high_52w | 830373.26 | ❌ HIGH |
| pgr_price_to_book_relative | 760487.41 | ❌ HIGH |
| pgr_vs_kie_6m | 708261.89 | ❌ HIGH |
| loss_ratio_ttm | 613139.42 | ❌ HIGH |
| pif_growth_acceleration | 580026.62 | ❌ HIGH |
| motor_vehicle_ins_cpi_yoy | 539890.86 | ❌ HIGH |
| pgr_vs_peers_6m | 524040.01 | ❌ HIGH |
| nfci | 494511.42 | ❌ HIGH |
| gainshare_est | 475301.54 | ❌ HIGH |
| investment_income_growth_yoy | 457619.08 | ❌ HIGH |
| investment_book_yield | 453833.97 | ❌ HIGH |
| usd_broad_return_3m | 447487.03 | ❌ HIGH |
| monthly_combined_ratio_delta | 437787.50 | ❌ HIGH |
| npw_per_pif_yoy | 429892.01 | ❌ HIGH |
| duration_rate_shock_3m | 416595.19 | ❌ HIGH |
| term_premium_10y | 408067.35 | ❌ HIGH |
| auto_pricing_power_spread | 395614.43 | ❌ HIGH |
| roe_trend | 369608.10 | ❌ HIGH |
| mom_12m | 366963.66 | ❌ HIGH |
| ppi_auto_ins_yoy | 360703.93 | ❌ HIGH |
| pe_ratio | 322189.15 | ❌ HIGH |
| realized_gain_to_net_income_ratio | 315800.64 | ❌ HIGH |
| vwo_vxus_spread_6m | 313389.81 | ❌ HIGH |
| unrealized_gain_pct_equity | 311374.69 | ❌ HIGH |
| medical_cpi_yoy | 302606.43 | ❌ HIGH |
| yield_slope | 297143.34 | ❌ HIGH |
| gold_vs_treasury_6m | 291239.52 | ❌ HIGH |
| roe | 279238.40 | ❌ HIGH |
| usd_momentum_6m | 250440.17 | ❌ HIGH |
| yield_curvature | 242593.37 | ❌ HIGH |
| baa10y_spread | 225514.26 | ❌ HIGH |
| buyback_acceleration | 222533.37 | ❌ HIGH |
| underwriting_income_3m | 210812.51 | ❌ HIGH |
| equity_risk_premium | 209332.91 | ❌ HIGH |
| roe_net_income_ttm | 202763.02 | ❌ HIGH |
| wti_return_3m | 197209.45 | ❌ HIGH |
| underwriting_income_growth_yoy | 192270.67 | ❌ HIGH |
| excess_bond_premium_proxy | 191953.60 | ❌ HIGH |
| underwriting_margin_ttm | 180921.55 | ❌ HIGH |
| pgr_premium_to_surplus | 159288.77 | ❌ HIGH |
| book_value_per_share_growth_yoy | 150796.48 | ❌ HIGH |
| npw_vs_npe_spread_pct | 142757.19 | ❌ HIGH |
| severity_index_yoy | 134335.60 | ❌ HIGH |
| credit_spread_ratio | 124625.57 | ❌ HIGH |
| channel_mix_direct_pct_yoy | 119791.82 | ❌ HIGH |
| mom_6m | 119239.07 | ❌ HIGH |
| channel_mix_agency_pct | 109746.98 | ❌ HIGH |
| real_yield_change_6m | 106618.60 | ❌ HIGH |
| unearned_premium_growth_yoy | 88746.87 | ❌ HIGH |
| rate_adequacy_gap_yoy | 84602.06 | ❌ HIGH |
| cr_acceleration | 80670.59 | ❌ HIGH |
| vmt_yoy | 69899.01 | ❌ HIGH |
| used_car_cpi_yoy | 58723.94 | ❌ HIGH |
| breakeven_inflation_10y | 49781.01 | ❌ HIGH |
| real_rate_10y | 47295.33 | ❌ HIGH |
| vol_63d | 46307.14 | ❌ HIGH |
| reserve_to_npe_ratio | 33429.24 | ❌ HIGH |
| npw_growth_yoy | 31722.92 | ❌ HIGH |
| vix | 31003.75 | ❌ HIGH |
| pb_ratio | 25733.67 | ❌ HIGH |
| buyback_yield | 15149.78 | ❌ HIGH |
| direct_channel_pif_share_ttm | 13398.48 | ❌ HIGH |

---

## Calibration Phase

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Uncalibrated (P = 50%; too few realised OOS rows) | ⬛ Superseded |
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (prequential ECE=19.5% [13.3%–26.5%] over n=1,038 OOS benchmark-months) |
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

**Mean trailing 12-point empirical coverage:** 54.2% (gap -25.8% vs nominal) ❌  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Trailing 12 Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|----------------------|-------|
| VOO | S&P 500 | +0.49% | -28.07% | +29.04% | 57.11% | 58.3% ❌ | 120 |
| VXUS | Total International Stock | +3.84% | -26.66% | +34.33% | 60.99% | 58.3% ❌ | 114 |
| VWO | Emerging Markets | +1.74% | -26.44% | +29.92% | 56.36% | 50.0% ❌ | 186 |
| VMBS | Mortgage-Backed Securities | +3.71% | -23.08% | +30.51% | 53.59% | 50.0% ❌ | 126 |
| BND | Total Bond Market | +4.82% | -21.37% | +31.01% | 52.38% | 50.0% ❌ | 156 |
| GLD | Gold Shares | +0.70% | -34.57% | +35.98% | 70.55% | 58.3% ❌ | 186 |
| DBC | DB Commodity Index | -2.09% | -34.17% | +29.99% | 64.16% | 66.7% ❌ | 174 |
| VDE | Energy | -5.13% | -37.27% | +27.02% | 64.28% | 41.7% ❌ | 186 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | OOS R² | IC | NW p | Hit Rate | Base Rate | PT p | CW t | CW p |
|-----------|-------------|-------|--------|----|------|----------|-----------|------|------|------|
| VWO | Emerging Markets | 186 | 19.52% | 0.1775 | 0.1516 | 58.1% | 68.3% | 0.9974 | 3.2270 | 0.0007 |
| GLD | Gold Shares | 186 | 10.97% | 0.1133 | 0.4020 | 58.1% | 67.7% | 0.7121 | 2.7088 | 0.0037 |
| DBC | DB Commodity Index | 174 | 7.40% | 0.1475 | 0.2546 | 66.1% | 67.8% | 0.2570 | 1.9872 | 0.0242 |
| VOO | S&P 500 | 120 | -0.93% | 0.1377 | 0.3155 | 60.0% | 64.2% | 0.2883 | 1.4718 | 0.0719 |
| BND | Total Bond Market | 156 | -7.82% | 0.1015 | 0.3946 | 69.2% | 75.0% | 0.2808 | 1.0995 | 0.1366 |
| VDE | Energy | 186 | -2.94% | 0.0361 | 0.7727 | 58.1% | 61.8% | 0.3376 | 1.0250 | 0.1534 |
| VXUS | Total International Stock | 114 | -4.63% | 0.0745 | 0.6546 | 61.4% | 67.5% | 0.5266 | 0.6089 | 0.2719 |
| VMBS | Mortgage-Backed Securities | 126 | -14.60% | -0.0001 | 0.9993 | 70.6% | 76.2% | 0.8862 | 0.4101 | 0.3412 |

**IC summary:** 6 ✅  1 ⚠️  1 ❌  (of 8 benchmarks)  
**Directional skill ✅:** 0/8 benchmarks with Pesaran–Timmermann p < 0.05  
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
| P(Actionable Sell) | 49.2% |

---

## Classifier Monitoring

| Metric | Value |
|--------|-------|
| Matured observations | 9 |
| Brier score | 0.2923 |
| Log loss | 0.7807 |
| ECE (10-bin) | 0.4498 |

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