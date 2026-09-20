# PGR Diagnostic Report — September 2026

**As-of Date:** 2026-09-21  
**Horizon:** 6M  
**OOS observations (aggregate):** 1224  
**Newey-West lags:** 5 (accounts for 5-month return-window overlap)  

---

## Aggregate Model Health

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| OOS R² (Campbell-Thompson) | -0.0113 (-1.13%) | ❌ | ≥ 2.00% |
| IC (Newey-West HAC) | 0.1649 | ✅ | ≥ 0.07 |
| IC significance | 0.0000 | ✅ p < 0.05 | p < 0.05 |
| Clark-West t-stat | 3.8847 | ✅ p < 0.05 | p < 0.05 |
| Clark-West p-value | 0.0001 | ✅ p < 0.05 | p < 0.05 |
| Hit Rate | 66.6% | ✅ | ≥ 55.0% |
| CPCV Positive Paths | 1/7 (12.5%) | ❌ | ≥ 5/7 |

> **Representative CPCV:** benchmark=VOO, model=ridge, paths=7, mean IC=-0.0538, IC std=0.1166.
> Stability verdict: FAIL. Scaled monthly threshold: ≥ 5/7 (maps from the full C(8,2) standard of ≥ 19/28 positive paths).

---


## Feature Governance

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Full obs/feature ratio | 20.58 | ✅ | ≥ 4.0 |
| Per-fold obs/feature ratio | 5.00 | ✅ | ≥ 4.0 |
| Features in monthly run | 12 | — | — |
| Fully populated observations | 247 | — | — |

> obs/feature ratio: 20.6 (full matrix), 5.0 (per WFO fold, 60M window).  n_obs=247, n_features=12.  Verdict: OK.

### Feature Importance Stability

| Metric | Value | Status | Threshold (Good) |
|--------|-------|--------|-----------------|
| Mean consecutive-fold Spearman ρ | 0.5707 | ⚠️ MARGINAL | ≥ 0.70 |
| Folds included | 19 | — | — |

> Stability score measures mean pairwise Spearman rank-correlation between consecutive WFO fold importance rankings. A score < 0.40 indicates unstable feature rankings; model predictions may be driven by different features each period.

**Top 10 features by mean WFO rank:**

| Rank | Feature | Mean Rank | Rank Std | Mean |Importance| |
|------|---------|-----------|----------|----------------|
| 3 | combined_ratio_ttm | 3.8 | 2.7 | 0.0191 |
| 5 | mom_12m | 5.1 | 4.1 | 0.0230 |
| 5 | nfci | 5.4 | 2.0 | 0.0120 |
| 5 | real_rate_10y | 5.8 | 2.7 | 0.0130 |
| 5 | credit_spread_hy | 5.8 | 3.4 | 0.0149 |
| 6 | real_yield_change_6m | 6.3 | 4.0 | 0.0141 |
| 6 | book_value_per_share_growth_yoy | 6.5 | 3.3 | 0.0144 |
| 7 | investment_income_growth_yoy | 7.3 | 3.3 | 0.0085 |
| 7 | vol_63d | 7.4 | 4.3 | 0.0137 |
| 7 | vix | 7.6 | 2.8 | 0.0082 |

### Multicollinearity (VIF)

**Overall:** ❌ HIGH multicollinearity  
Features flagged high (VIF > 10): **76**  
Features flagged moderate (VIF 5–10): **0**  

> VIF measures how much variance in a feature is explained by the other features. VIF > 10 indicates severe multicollinearity and may cause unstable coefficient estimates.

**All features by VIF (descending):**

| Feature | VIF | Status |
|---------|-----|--------|
| gainshare_est | 47854863.56 | ❌ HIGH |
| channel_mix_direct_pct_yoy | 22346011.78 | ❌ HIGH |
| pgr_vs_kie_6m | 17825649.69 | ❌ HIGH |
| pif_growth_yoy | 13921351.65 | ❌ HIGH |
| unearned_premium_growth_yoy | 12987849.76 | ❌ HIGH |
| baa10y_spread | 10288229.13 | ❌ HIGH |
| credit_spread_ratio | 8390704.89 | ❌ HIGH |
| commodity_equity_momentum | 7776706.90 | ❌ HIGH |
| nfci | 7359458.40 | ❌ HIGH |
| vwo_vxus_spread_6m | 6536346.66 | ❌ HIGH |
| ppi_auto_ins_yoy | 6389803.04 | ❌ HIGH |
| monthly_combined_ratio_delta | 5474539.66 | ❌ HIGH |
| severity_index_yoy | 5409306.41 | ❌ HIGH |
| rate_adequacy_gap_yoy | 5226587.03 | ❌ HIGH |
| direct_channel_pif_share_ttm | 4945063.57 | ❌ HIGH |
| breakeven_inflation_10y | 4159074.39 | ❌ HIGH |
| pe_ratio | 4047754.66 | ❌ HIGH |
| legal_services_ppi_relative | 3441316.55 | ❌ HIGH |
| vmt_yoy | 3289804.79 | ❌ HIGH |
| cr_acceleration | 3071289.50 | ❌ HIGH |
| term_premium_10y | 2902062.77 | ❌ HIGH |
| mom_12m | 2783678.06 | ❌ HIGH |
| medical_cpi_yoy | 2746275.76 | ❌ HIGH |
| pif_growth_acceleration | 2672344.49 | ❌ HIGH |
| motor_vehicle_ins_cpi_yoy | 2530832.23 | ❌ HIGH |
| vol_63d | 2521027.45 | ❌ HIGH |
| pgr_price_to_book_relative | 2519497.40 | ❌ HIGH |
| credit_spread_hy | 2443641.92 | ❌ HIGH |
| underwriting_income_3m | 2358647.15 | ❌ HIGH |
| yield_curvature | 2260306.50 | ❌ HIGH |
| used_car_cpi_yoy | 2184309.63 | ❌ HIGH |
| realized_gain_to_net_income_ratio | 2182272.41 | ❌ HIGH |
| real_yield_change_6m | 2155592.50 | ❌ HIGH |
| loss_ratio_ttm | 2052730.64 | ❌ HIGH |
| high_52w | 1874316.82 | ❌ HIGH |
| combined_ratio_ttm | 1842518.89 | ❌ HIGH |
| wti_return_3m | 1617974.20 | ❌ HIGH |
| usd_momentum_6m | 1563081.57 | ❌ HIGH |
| roe_trend | 1555032.70 | ❌ HIGH |
| book_value_per_share_growth_yoy | 1483922.40 | ❌ HIGH |
| mom_3m | 1461832.18 | ❌ HIGH |
| real_rate_10y | 1384257.19 | ❌ HIGH |
| roe_net_income_ttm | 1376168.35 | ❌ HIGH |
| mortgage_spread_30y_10y | 1360005.48 | ❌ HIGH |
| auto_pricing_power_spread | 1351322.62 | ❌ HIGH |
| gold_vs_treasury_6m | 1344896.31 | ❌ HIGH |
| npw_vs_npe_spread_pct | 1177706.52 | ❌ HIGH |
| duration_rate_shock_3m | 1061027.38 | ❌ HIGH |
| underwriting_income | 901296.42 | ❌ HIGH |
| yield_slope | 858686.22 | ❌ HIGH |
| pgr_pe_vs_market_pe | 841488.61 | ❌ HIGH |
| pb_ratio | 822400.68 | ❌ HIGH |
| buyback_acceleration | 817745.94 | ❌ HIGH |
| npw_growth_yoy | 792563.56 | ❌ HIGH |
| pgr_vs_peers_6m | 759980.64 | ❌ HIGH |
| investment_book_yield | 665068.80 | ❌ HIGH |
| gasoline_retail_sales_delta | 644635.20 | ❌ HIGH |
| investment_income_growth_yoy | 556600.71 | ❌ HIGH |
| pgr_vs_vfh_6m | 550765.59 | ❌ HIGH |
| reserve_to_npe_ratio | 514220.58 | ❌ HIGH |
| pgr_premium_to_surplus | 502070.02 | ❌ HIGH |
| unearned_premium_to_npw_ratio | 494401.07 | ❌ HIGH |
| expense_ratio_ttm | 493833.48 | ❌ HIGH |
| npw_per_pif_yoy | 471366.50 | ❌ HIGH |
| excess_bond_premium_proxy | 449830.44 | ❌ HIGH |
| channel_mix_agency_pct | 415124.52 | ❌ HIGH |
| underwriting_income_growth_yoy | 385661.96 | ❌ HIGH |
| underwriting_margin_ttm | 371153.58 | ❌ HIGH |
| usd_broad_return_3m | 354010.92 | ❌ HIGH |
| unrealized_gain_pct_equity | 314804.27 | ❌ HIGH |
| breakeven_momentum_3m | 298518.23 | ❌ HIGH |
| roe | 291976.72 | ❌ HIGH |
| mom_6m | 274458.39 | ❌ HIGH |
| equity_risk_premium | 272744.79 | ❌ HIGH |
| buyback_yield | 226515.83 | ❌ HIGH |
| vix | 197985.42 | ❌ HIGH |

---

## Calibration Phase

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Raw BayesianRidge posterior (uncalibrated) | ⬛ Superseded |
| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | ✅ Active (n=1,224  ECE=3.4% [2.1%–7.2%]) |
| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | ⏳ Activates at n ≥ 500 |

---

## Conformal Prediction Intervals

**Method:** ACI (Adaptive Conformal Inference — adjusts α_t for distribution shift)  
**Nominal Coverage:** 80%  

**Mean empirical coverage:** 89.2% (target ≥ 80%) ✅  

**Mean trailing 12-point empirical coverage:** 58.3% (gap -21.7% vs nominal) ❌  

| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Emp. Coverage | Trailing 12 Coverage | N Cal |
|-----------|-------------|----------------|----------|----------|----------|---------------|----------------------|-------|
| VOO | S&P 500 | +3.04% | -21.84% | +27.93% | 49.77% | 88.6% ✅ | 66.7% | 114 |
| VXUS | Total International Stock | -3.51% | -32.57% | +25.55% | 58.11% | 88.9% ✅ | 66.7% | 108 |
| VWO | Emerging Markets | -3.98% | -28.69% | +20.73% | 49.42% | 86.7% ✅ | 66.7% | 180 |
| VMBS | Mortgage-Backed Securities | +2.40% | -18.70% | +23.50% | 42.20% | 85.7% ✅ | 58.3% | 126 |
| BND | Total Bond Market | +3.37% | -17.63% | +24.36% | 41.99% | 85.9% ✅ | 75.0% | 156 |
| GLD | Gold Shares | -2.54% | -38.92% | +33.85% | 72.78% | 93.5% ✅ | 50.0% | 186 |
| DBC | DB Commodity Index | -11.22% | -46.28% | +23.85% | 70.13% | 97.0% ✅ | 33.3% | 168 |
| VDE | Energy | -13.62% | -45.72% | +18.48% | 64.19% | 87.6% ✅ | 50.0% | 186 |

> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate
> larger historical prediction errors.  ACI dynamically adjusts coverage when errors
> cluster (distribution shift), providing stronger guarantees than static split conformal.

---

## Per-Benchmark Health

| Benchmark | Description | N OOS | OOS R² | NW IC | Hit Rate | CW t | CW p |
|-----------|-------------|-------|--------|-------|----------|------|------|
| DBC | DB Commodity Index | 168 | 6.51% | 0.3350 | 71.4% | 2.3116 | 0.0110 |
| VDE | Energy | 186 | 1.37% | 0.1576 | 61.3% | 2.2898 | 0.0116 |
| VOO | S&P 500 | 114 | -12.21% | -0.0459 | 60.5% | 1.6836 | 0.0475 |
| GLD | Gold Shares | 186 | -4.62% | 0.2219 | 60.8% | 1.3718 | 0.0859 |
| VMBS | Mortgage-Backed Securities | 126 | -12.94% | 0.1622 | 79.4% | 1.1794 | 0.1202 |
| VXUS | Total International Stock | 108 | -9.07% | -0.0177 | 64.8% | 1.0380 | 0.1508 |
| BND | Total Bond Market | 156 | -13.65% | 0.1924 | 73.7% | 0.6481 | 0.2589 |
| VWO | Emerging Markets | 180 | -8.25% | 0.0098 | 63.3% | 0.6370 | 0.2625 |

**IC summary:** 5 ✅  0 ⚠️  3 ❌  (of 8 benchmarks)  
**Hit rate ✅:** 8/8 benchmarks above 55% threshold  
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
| P(Actionable Sell) | 41.8% |

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