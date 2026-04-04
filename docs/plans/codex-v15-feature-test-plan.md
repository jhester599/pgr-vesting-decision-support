# codex-v15-feature-test-plan.md

Created: 2026-04-04

## Goal

Run an exhaustive fixed-budget feature-replacement cycle for v15, testing each research-backed candidate one feature at a time against the current v14 winner:

- forecast universe: `VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE`
- leading prediction-layer candidate: `ensemble_ridge_gbt`
- working baselines:
  - `ridge_lean_v1`
  - `gbt_lean_plus_two`

The objective is not to add more features. The objective is to replace weaker features with stronger, more causal, more benchmark-aware ones while keeping the model lean.

## Source Reports Reviewed

The v15 queue below was synthesized from:

- `docs/history/v15-research-reports/v15_chatgptdeepresearch_20260404.md`
- `docs/history/v15-research-reports/v15_geminideepresearch_20260404.md`
- `docs/history/v15-research-reports/v15_geminipro_20260404.md`
- `docs/history/v15-research-reports/v15_claudeopusresearch_20260404.md`

## Report-by-Report Findings

### ChatGPT Deep Research

Strongest contributions:

- underwriting decomposition: `loss_ratio_ttm`, `expense_ratio_ttm`, `underwriting_margin_ttm`
- rate-adequacy / severity family: `rate_adequacy_gap_yoy`, `ppi_auto_ins_yoy`, `severity_index_yoy`
- insurer pipeline signals: `npw_per_pif_yoy`, `npw_vs_npe_spread_pct`, `unearned_premium_growth_yoy`
- investment-book sensitivity: `duration_rate_shock_3m`, `portfolio_yield_spread`
- benchmark-side drivers: `usd_broad_return_3m`, `wti_return_3m`, `mortgage_spread_30y_10y`, `baa10y_spread`

Most useful framing:

- v15 should replace redundant momentum / risk-regime features, not expand the stack
- benchmark-predictive features belong in the model because the target is relative return

### Gemini Deep Research

Strongest contributions:

- use PGR's monthly 8-K advantage more aggressively
- prioritize stationary spread/ratio features over generic macro levels
- add "social inflation" and benchmark-specific drivers
- explicitly tailor inputs to Ridge vs GBT

Most useful concrete ideas:

- `auto_severity_inflation_spread`
- `monthly_combined_ratio_delta`
- `pif_growth_acceleration`
- `legal_services_ppi_relative`
- `term_premium_10y`
- `mbs_treasury_spread`
- `pgr_price_to_book_relative`
- `pgr_premium_to_surplus`

### Claude Opus Research

Strongest contributions:

- very clear triage of which current features are weakest:
  - `vmt_yoy`
  - `yield_curvature`
  - `mom_3m`
  - `nfci`
- strongest proposed replacements were both economically grounded and implementation-light
- best model-specific thinking for relative price features

Most useful concrete ideas:

- `pgr_peer_relative_mom_6m`
- `breakeven_inflation_10y`
- `usd_broad_return`
- `combined_ratio_acceleration`
- `pgr_relative_volatility_ratio`
- `book_value_per_share_growth_yoy`
- `short_term_relative_reversal_1m`
- `relative_drawdown_depth`

### Gemini Pro

Strongest contributions:

- strongest warning against low-value technical indicators and nominal-dollar features
- best articulation of benchmark-side regime features
- most useful "replace noisy absolute feature with relative feature" framing

Most useful concrete ideas:

- `auto_pricing_power_spread`
- `equity_risk_premium`
- `real_yield_change_6m`
- `pgr_cr_vs_peer_cr`
- `credit_spread_ratio`
- `pgr_pe_vs_market_pe`
- `gold_vs_treasury_6m`
- `vwo_vxus_spread_6m`

## Cross-Report Synthesis

The four reports converge on five feature themes:

1. Replace generic exposure proxies with direct insurance economics.
2. Decompose underwriting quality into cleaner, more causal sub-signals.
3. Add a small number of benchmark-side macro drivers because the target is relative return.
4. Prefer relative and stationary features over absolute and nominal features.
5. Keep the feature budget lean and test one swap at a time.

## Canonical Feature Families And Naming Crosswalk

Many report suggestions were the same economic idea under different names. v15 should test canonical families, while still retaining the proposed variants when the construction materially differs.

| Canonical Family | Report Names / Variants | v15 Handling |
|---|---|---|
| Insurance pricing power | `rate_adequacy_gap_yoy`, `auto_pricing_power_spread`, `auto_severity_inflation_spread` | Test all three, but in that order |
| Underwriting trend | `monthly_combined_ratio_delta`, `combined_ratio_acceleration`, `cr_acceleration` | Use `cr_acceleration` as the repo-native first test, then build the target-delta version |
| Unit / policy growth | `pif_growth_acceleration`, `pif_growth_yoy` | Test existing `pif_growth_yoy` first, then acceleration |
| Premium-rate decomposition | `npw_per_pif_yoy`, `npw_growth_minus_pif_growth`, `npw_vs_npe_spread_pct` | Treat as a three-part pricing / pipeline family |
| Benchmark USD driver | `usd_broad_return_3m`, `usd_momentum_6m`, `USD_Momentum_Index`, `trade-weighted USD` | Test 3-month version first, then 6-month variant |
| Benchmark inflation driver | `breakeven_inflation_10y`, `breakeven_momentum_3m` | Test level first in Ridge, momentum first in GBT |
| Benchmark bond driver | `mortgage_spread_30y_10y`, `MBS_Treasury_Spread` | Treat as the same family |
| Benchmark credit driver | `baa10y_spread`, `US_Corporate_Credit_Spread`, `credit_spread_ratio`, `excess_bond_premium_proxy` | Test `baa10y_spread` first, then harder variants |
| Relative valuation | `pgr_price_to_book_relative`, `pgr_pe_vs_market_pe` | Prefer P/B relative before P/E relative |
| Relative momentum | `pgr_peer_relative_mom_6m`, `pgr_vs_peers_6m`, `insurance_sector_relative_momentum_vs_broad_financials` | Test peer-relative first |
| Relative risk / mean reversion | `pgr_relative_volatility_ratio`, `relative_drawdown_depth`, `short_term_relative_reversal_1m` | Secondary price-derived family after peer-relative momentum |

## Canonical v15 Testing Rules

1. Keep the v13.1 recommendation layer fixed during v15.
2. Keep the v14 selected forecast universe fixed during v15.
3. `v15.0`: test all candidate features one at a time on Ridge and GBT first.
4. `v15.1`: retest only the `v15.0` winners across all deployed model types.
5. `v15.2`: compare the best confirmed modified models against their baselines and `historical_mean`.
6. Test one feature at a time before any pairwise or grouped follow-up.
7. Prefer one-for-one swaps.
8. Allow one-for-two swaps only after both individual adds prove useful.
9. Use benchmark-aware acceptance criteria, not only raw IC.
10. Do not promote a candidate just because it improves one benchmark slice.

## Baseline Features To Challenge

Current Ridge baseline:

- `mom_12m`
- `vol_63d`
- `yield_slope`
- `yield_curvature`
- `real_rate_10y`
- `credit_spread_hy`
- `nfci`
- `vix`
- `combined_ratio_ttm`
- `investment_income_growth_yoy`
- `roe_net_income_ttm`
- `npw_growth_yoy`

Current GBT baseline:

- `mom_3m`
- `mom_6m`
- `mom_12m`
- `vol_63d`
- `yield_slope`
- `yield_curvature`
- `real_rate_10y`
- `credit_spread_hy`
- `nfci`
- `vix`
- `vmt_yoy`
- `pif_growth_yoy`
- `investment_book_yield`
- `underwriting_income_growth_yoy`

## Acceptance Gate For An Individual Feature

A feature passes its individual test only if all are true:

- it improves policy utility vs the baseline model on the reduced universe
- it does not worsen mean OOS R² materially
- it does not create a worse coverage burden than the feature it replaces
- it does not make the recommendation output less stable or less explainable

Secondary tie-breakers:

- better mean IC
- better hit rate
- lower implementation complexity
- stronger economic interpretation

## Phase Structure

### Phase 1 — Repo-Ready Immediate Tests

These are either already in the feature matrix or can be built trivially from already-fetched price series.

### Phase 2 — Easy New Derived Features

These require feature-engineering changes, but only from data the repo already fetches or can fetch easily from FRED / EDGAR.

### Phase 3 — Harder Public-Data Features

These are still public-data feasible, but require new loaders, peer normalization, or more careful point-in-time handling.

### Phase 4 — Stretch / Deferred Queue

These are worth recording, but should only be built after the higher-consensus, lower-cost queue is exhausted.

## Phase 1 — Repo-Ready Immediate Tests

| Order | Feature | Why It Is In Scope Immediately | Test Against | Model(s) | Notes |
|---|---|---|---|---|---|
| 1 | `ppi_auto_ins_yoy` | Already in matrix; strongest insurance-pricing cycle proxy | `vmt_yoy`, `mom_3m`, `mom_6m` | both | Individual test before building full pricing-power spread |
| 2 | `unearned_premium_growth_yoy` | Already in matrix; strong cross-report support for revenue pipeline | `npw_growth_yoy`, `mom_12m`, `investment_income_growth_yoy` | ridge first, then gbt | Already validated historically but not in current winning stack |
| 3 | `cr_acceleration` | Existing repo-native proxy for monthly combined-ratio delta / acceleration | `combined_ratio_ttm` | ridge first, then gbt | Canonical stand-in for Gemini/Claude combined-ratio trend ideas |
| 4 | `pgr_vs_peers_6m` | Existing relative-price feature aligned to Claude peer-relative momentum thesis | `mom_3m`, `mom_6m` | gbt first, then ridge | Strong candidate to replace generic short-horizon momentum |
| 5 | `medical_cpi_yoy` | Already in matrix; building block for severity-pressure family | `vmt_yoy`, `mom_3m` | both | Test alone before composite severity features |
| 6 | `used_car_cpi_yoy` | Already in matrix; direct physical-damage severity proxy | `vmt_yoy`, `mom_3m` | both | Test alone before composite severity features |
| 7 | `pif_growth_yoy` | Already in matrix and already in GBT; test as Ridge replacement candidate | `investment_income_growth_yoy`, `npw_growth_yoy` | ridge | Claude/Gemini both argued for more direct unit-growth signals |
| 8 | `gainshare_est` | Existing repo feature; low priority but already available | `investment_income_growth_yoy`, `roe_net_income_ttm` | ridge | Keep only if it clearly beats baseline features |
| 9 | `pgr_vs_kie_6m` | Existing relative insurance-benchmark price feature | `mom_3m`, `mom_6m` | gbt | Only keep if it beats peer-relative momentum cleanly |
| 10 | `pgr_vs_vfh_6m` | Existing financials-relative price feature | `mom_3m`, `mom_6m` | gbt | Lower priority than `pgr_vs_peers_6m` because diversification-first logic deprioritizes VFH |

## Phase 2 — Easy New Derived Features

| Order | Feature | Replace / Compete With | Model(s) | Source | Consensus | Notes |
|---|---|---|---|---|---|---|
| 11 | `rate_adequacy_gap_yoy` | `vmt_yoy`, `mom_3m`, `mom_6m` | both | FRED | ChatGPT, Claude | Primary v15 candidate; likely highest priority new feature |
| 12 | `severity_index_yoy` | `vmt_yoy`, `mom_3m` | both | FRED | ChatGPT | Use used-car + medical CPI average |
| 13 | `auto_pricing_power_spread` | `vmt_yoy`, `mom_3m` | both | FRED | Gemini Pro | Alternative pricing-power formulation; compare against rate-adequacy gap |
| 14 | `monthly_combined_ratio_delta` | `combined_ratio_ttm` | ridge first | EDGAR | Gemini | Distinct from `cr_acceleration`; use 96% target framing |
| 15 | `pif_growth_acceleration` | `pif_growth_yoy`, `investment_income_growth_yoy` | gbt first, then ridge | EDGAR | Gemini | Direct second-derivative growth signal |
| 16 | `npw_per_pif_yoy` | `npw_growth_yoy`, `pif_growth_yoy` | both | EDGAR | ChatGPT | Strong pricing / average premium per policy feature |
| 17 | `npw_vs_npe_spread_pct` | `npw_growth_yoy`, `mom_12m` | both | EDGAR | ChatGPT, Claude | Growth pipeline / written-vs-earned spread |
| 18 | `underwriting_margin_ttm` | `combined_ratio_ttm`, `underwriting_income_growth_yoy` | ridge first, then gbt | EDGAR | ChatGPT | Cleaner profitability ratio than nominal underwriting income |
| 19 | `reserve_to_npe_ratio` | `credit_spread_hy`, `nfci` | both | EDGAR | ChatGPT | Insurer-specific earnings-quality / reserve signal |
| 20 | `book_value_per_share_growth_yoy` | `roe_net_income_ttm`, `buyback_yield` | ridge first | EDGAR | Claude | Better balance-sheet compounding feature |
| 21 | `duration_rate_shock_3m` | `real_rate_10y`, `yield_curvature` | both | EDGAR + FRED | ChatGPT | Direct bond-book sensitivity proxy |
| 22 | `direct_channel_pif_share_ttm` | `investment_income_growth_yoy`, `roe_net_income_ttm` | ridge | EDGAR | Claude | Use direct-channel mix as slow structural moat signal |
| 23 | `channel_mix_direct_pct_yoy` | `investment_income_growth_yoy`, `roe_net_income_ttm` | ridge | EDGAR | ChatGPT | Faster-moving version of direct-channel thesis |
| 24 | `realized_gain_to_net_income_ratio` | `investment_income_growth_yoy`, `roe_net_income_ttm` | ridge | EDGAR | ChatGPT | Earnings-quality check; likely secondary |
| 25 | `unrealized_gain_pct_equity` | `roe_net_income_ttm`, `investment_income_growth_yoy` | ridge | EDGAR | ChatGPT | OCI / capital-flexibility view |

## Phase 3 — Price / Macro / Benchmark-Side Additions

| Order | Feature | Replace / Compete With | Model(s) | Source | Consensus | Notes |
|---|---|---|---|---|---|---|
| 26 | `usd_broad_return_3m` | `nfci`, `vix` | both | FRED | ChatGPT, Claude | Highest-priority benchmark-side add |
| 27 | `usd_momentum_6m` | `nfci`, `vix` | gbt first | FRED | Gemini Pro | Test as horizon variant after 3m USD test |
| 28 | `wti_return_3m` | `yield_curvature`, `mom_3m`, `mom_6m` | both | FRED | ChatGPT | Strong for `VDE` / `DBC` denominator forecasting |
| 29 | `mortgage_spread_30y_10y` | `yield_curvature`, `real_rate_10y` | ridge first, then gbt | FRED | ChatGPT | Most targeted `VMBS` feature in all reports |
| 30 | `baa10y_spread` | `credit_spread_hy`, `nfci` | both | FRED | ChatGPT, Gemini | Smoother credit-stress proxy |
| 31 | `credit_spread_ratio` | `credit_spread_hy` | ridge first | FRED | Gemini Pro | Test only if extra IG spread source is easy and point-in-time safe |
| 32 | `breakeven_inflation_10y` | `yield_curvature` | both | FRED | Claude | Level version |
| 33 | `breakeven_momentum_3m` | `yield_curvature`, `vol_63d` | gbt first | FRED | Gemini Pro | Momentum version; likely more tree-friendly |
| 34 | `real_yield_change_6m` | `real_rate_10y` | both | FRED | Gemini Pro | Use if real-rate level is too blunt |
| 35 | `term_premium_10y` | `yield_curvature`, `yield_slope` | ridge first | FRED | Gemini | Distinct from classic curve slope |
| 36 | `gold_vs_treasury_6m` | `mom_3m`, `mom_12m` | gbt first | prices | Gemini Pro | Cross-asset fear / safety switch |
| 37 | `commodity_equity_momentum` | `mom_3m`, `mom_6m` | gbt | prices | Gemini | Inflation / real-asset regime signal |
| 38 | `legal_services_ppi_relative` | `vix`, `nfci`, `credit_spread_hy` | ridge first | FRED | Gemini | Social-inflation proxy; lower priority than rate adequacy |
| 39 | `gasoline_retail_sales_delta` | `wti_return_3m`, `vmt_yoy` | gbt | FRED | Gemini | Energy-demand variant; test after WTI |
| 40 | `vwo_vxus_spread_6m` | `mom_6m`, `mom_12m` | gbt | prices | Gemini Pro | Lower priority; may overlap with USD |

## Phase 4 — Harder Public-Data / Peer / Valuation Features

| Order | Feature | Replace / Compete With | Model(s) | Difficulty | Why Deferred |
|---|---|---|---|---|---|
| 41 | `pgr_cr_vs_peer_cr` | `combined_ratio_ttm` | gbt first, then ridge | medium-high | Very strong idea, but requires robust peer underwriting normalization |
| 42 | `pgr_price_to_book_relative` | `roe_net_income_ttm`, `pb_ratio` | ridge | medium | Needs reliable point-in-time peer / sector valuation series |
| 43 | `pgr_pe_vs_market_pe` | `pe_ratio`, `roe_net_income_ttm` | ridge | medium | Lower confidence than P/B relative, but worth one test |
| 44 | `equity_risk_premium` | `yield_curvature`, `real_rate_10y`, `credit_spread_hy` | ridge first | medium | Valuable for `VOO` / `BND`, but requires a trustworthy market earnings-yield series |
| 45 | `pgr_fcf_yield` | `underwriting_income_growth_yoy`, `investment_income_growth_yoy` | ridge | medium-high | Potentially strong, but quarterly cash-flow timing needs care |
| 46 | `pgr_premium_to_surplus` | `roe_net_income_ttm`, `npw_growth_yoy` | ridge | medium | Good operating-leverage idea, but build after simpler EDGAR ratios |

## Phase 5 — Stretch / Low-Priority Queue

These should be documented and only tested if the earlier queue stalls.

| Order | Feature | Model(s) | Reason To Defer |
|---|---|---|---|
| 47 | `excess_bond_premium_proxy` | both | Strong literature support, but materially more complex to build correctly |
| 48 | `relative_drawdown_depth` | gbt | Useful mean-reversion idea, but lower consensus and more tactical |
| 49 | `short_term_relative_reversal_1m` | gbt | Highest risk of noise / sign instability |
| 50 | `reserve_development_pct` | ridge | Valuable if available cleanly, but point-in-time construction is tricky |
| 51 | `vmt_momentum` | gbt | Included because Gemini supported it, but earlier repo work has been skeptical |
| 52 | `vmt_gas_proxy_yoy` | gbt | Interesting combined claims-frequency / energy-demand idea, but lower conviction |
| 53 | `insurance_sector_relative_momentum_vs_broad_financials` | both | Potentially useful, but less diversification-friendly than peer-relative momentum |
| 54 | `portfolio_yield_spread` | ridge | Strong idea from ChatGPT, but requires careful definition of benchmark rate |

## Exact Test Protocol For Each Feature

For every feature in Phases 1 through 4:

1. Add or confirm the feature in the feature matrix.
2. Record:
   - first non-null date
   - non-null row count
   - observations-per-feature effect on Ridge and GBT candidates
3. Run one-for-one swap tests separately for Ridge and GBT.
4. For each model, compare against the unchanged baseline on:
   - mean IC
   - mean hit rate
   - mean OOS R²
   - policy utility
   - recommendation stability
5. Mark the feature as:
   - `reject`
   - `keep for Ridge`
   - `keep for GBT`
   - `keep for both`
   - `defer pending pairwise follow-up`

## Recommended Execution Order

### Pass A — Highest-Consensus, Lowest-Cost

- `ppi_auto_ins_yoy`
- `cr_acceleration`
- `pgr_vs_peers_6m`
- `unearned_premium_growth_yoy`
- `rate_adequacy_gap_yoy`
- `severity_index_yoy`
- `npw_per_pif_yoy`
- `underwriting_margin_ttm`
- `usd_broad_return_3m`
- `wti_return_3m`
- `mortgage_spread_30y_10y`
- `baa10y_spread`

### Pass B — Strong Secondary Queue

- `pif_growth_acceleration`
- `npw_vs_npe_spread_pct`
- `reserve_to_npe_ratio`
- `book_value_per_share_growth_yoy`
- `duration_rate_shock_3m`
- `breakeven_inflation_10y`
- `real_yield_change_6m`
- `term_premium_10y`
- `gold_vs_treasury_6m`
- `legal_services_ppi_relative`

### Pass C — Harder But Plausibly Valuable

- `pgr_cr_vs_peer_cr`
- `pgr_price_to_book_relative`
- `equity_risk_premium`
- `pgr_fcf_yield`
- `pgr_premium_to_surplus`

## Expected First Swaps By Model

### Ridge First-Swap Queue

1. `combined_ratio_ttm` -> `cr_acceleration`
2. `investment_income_growth_yoy` -> `unearned_premium_growth_yoy`
3. `npw_growth_yoy` -> `npw_per_pif_yoy`
4. `yield_curvature` -> `mortgage_spread_30y_10y`
5. `credit_spread_hy` -> `baa10y_spread`
6. `nfci` -> `usd_broad_return_3m`

### GBT First-Swap Queue

1. `vmt_yoy` -> `rate_adequacy_gap_yoy`
2. `mom_3m` -> `pgr_vs_peers_6m`
3. `yield_curvature` -> `wti_return_3m`
4. `nfci` -> `usd_broad_return_3m`
5. `mom_12m` -> `gold_vs_treasury_6m`
6. `real_rate_10y` -> `duration_rate_shock_3m`

## Deliverables

The v15 implementation loop should write:

- normalized candidate inventory
- generated swap queue
- one summary CSV per phase
- one detailed CSV per phase
- one markdown memo per phase
- one final closeout memo with:
  - winning replacements
  - rejected features
  - deferred features
  - exact post-v15 candidate stack recommendation

## v15 Success Condition

v15 succeeds if it identifies a post-v14 lean feature set that:

- materially improves policy utility vs the current v14 candidate baseline
- stays within the lean feature budget
- improves or at least stabilizes OOS fit
- preserves the clearer v13.1 recommendation layer
- creates a credible path to a v15.1 or v16 promotion study
