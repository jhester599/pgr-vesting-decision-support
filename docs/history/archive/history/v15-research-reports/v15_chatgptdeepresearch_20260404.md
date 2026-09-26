# Candidate Replacement Features for v15

## Executive Summary

- **Split the underwriting signal you already trust into cleaner, more targetable parts.** The current v14 lean sets rely heavily on `combined_ratio_ttm` plus a macro block (`yield_*`, `credit_spread_hy`, `nfci`, `vix`).fileciteturn15file0L1-L1 For an insurer, the combined ratio is *the* core KPI, but it is an aggregate of **loss ratio** and **expense ratio**, which often move for different economic reasons (claims severity/frequency vs distribution/scale efficiency). The repo’s own EDGAR dictionary already flags `loss_ratio_ttm` and `expense_ratio_ttm` as high-theory, EDGAR-native candidates.fileciteturn19file0L1-L1
- **Add a “rate adequacy vs loss-cost inflation” feature family (lean but high signal).** In personal auto, underwriting cycles are dominated by whether premium rate increases keep up with (or outrun) claim severities. Your pipeline already has the right public building blocks: **auto insurance PPI** and **used car + medical CPI** (all monthly, interpretable).citeturn0search3turn16search0 This is a direct upgrade over generic macro proxies like `vmt_yoy` (driving exposure) because it links **pricing power** to **cost inflation**, which is the economic mechanism that moves insurer margins.
- **Use EDGAR-derived “premium per policy” and “growth mode vs runoff” to replace weaker generic growth proxies.** The EDGAR cache supports `net_premiums_written`, `net_premiums_earned`, and `pif_total` at monthly cadence.fileciteturn19file0L1-L1 From first principles, **NPW per policy** is a proxy for *rate level* (pricing) separate from *unit growth*, and **NPW–NPE spread** is a proxy for whether the book is expanding (future earned premium tailwind) or not.
- **Upgrade investment-book sensitivity with “duration × rate shock” and “book yield spread.”** Insurers’ equity returns are materially influenced by the investment portfolio; the 2025 10‑K highlights investment income magnitude and sensitivity to yield and portfolio composition.citeturn5view2 The EDGAR cache contains `fixed_income_duration` (high coverage) and (from ~2015) `investment_book_yield`.fileciteturn19file0L1-L1 A simple interaction feature `duration × Δ10Y` gives a more *structural* and benchmark-relevant measure than `real_rate_10y` alone (especially versus bond benchmarks).
- **Include a small number of benchmark-predictive features—because the target is relative return.** The reduced benchmark universe spans US equity, international equity, bonds/MBS, gold, broad commodities, and energy. The ETFs’ own fact sheets show these exposures are anchored to very different underlying drivers: S&P 500 equities, global ex‑US equities, aggregate IG USD bonds, agency MBS, gold bullion, diversified commodity futures, and US energy equities.citeturn9search44turn10search14turn9search40turn10search15turn14search37turn14search0turn13search24 If you do not model at least a few **USD / oil / rates / mortgage spread** drivers, your relative-return forecast will often be “right on PGR but wrong on the benchmark.”
- **Where v14 likely remains insufficient (feature-side, not model-side).** v14’s Ridge/GBT lean sets are still dominated by: (a) overlapping momentum windows, (b) generic risk proxies (`nfci`, `vix`, HY OAS), and (c) a small number of insurer fundamentals.fileciteturn15file0L1-L1 The v14 bakeoff detail also shows the ensemble does **not** consistently dominate `baseline_historical_mean` across the chosen universe (e.g., BND/DBC/VDE cases where the baseline is comparable or higher on policy return).fileciteturn15file0L1-L1 This is consistent with the broader literature point that beating a historical-mean baseline out of sample is hard, often requiring very disciplined signals and constraints.citeturn20search1
- **Most “weak/redundant generic feature types” in this project class:** multiple correlated momentum windows (`mom_3m`, `mom_6m`, `mom_12m`), multiple correlated “risk regime” variables (`nfci`, `vix`, `credit_spread_hy`), and curve “shape” factors that may be less directly tied to your benchmark mix than *rate changes* and *mortgage spreads*.fileciteturn15file0L1-L1
- **Highest expected lift per unit of complexity:** (1) underwriting decomposition, (2) rate-adequacy gap, (3) premium-per-policy and unearned-premium pipeline indicators, (4) duration × rates shock, (5) USD and oil (for VXUS/VWO and DBC/VDE) as minimal benchmark-predictive coverage.fileciteturn19file0L1-L1citeturn23search2turn23search1

## Ranked Candidate Feature Table

| feature_name | category | replace_or_compete_with | definition | economic_rationale | expected_direction | likely_frequency | likely_source | implementation_difficulty | likely_signal_quality | why_it_might_outperform_existing_feature | key_risks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loss_ratio_ttm | PGR_specific | combined_ratio_ttm | 12‑month rolling mean of `loss_lae_ratio` from monthly EDGAR supplements.fileciteturn19file0L1-L1 | Separates claims cost pressure from expense changes; closer to the mechanism driven by severity/frequency and reserving.citeturn5view4 | Higher → worse underwriting → lower relative returns | monthly | SEC monthly 8‑K “Monthly Results” (EDGAR cache columns `loss_lae_ratio`).fileciteturn19file0L1-L1 | low | high | More specific than `combined_ratio_ttm`, reduces “aggregation noise” (expense ratio can improve while losses worsen, and vice versa). | Loss ratio can be distorted by reserve actions or cat events; may need outlier handling. |
| expense_ratio_ttm | PGR_specific | combined_ratio_ttm; channel_mix_agency_pct | 12‑month rolling mean of `expense_ratio`.fileciteturn19file0L1-L1 | Captures distribution/scale efficiency and channel mix effects; complements loss ratio. | Higher → worse efficiency → lower relative returns | monthly | SEC monthly 8‑K (EDGAR cache `expense_ratio`).fileciteturn19file0L1-L1 | low | medium-high | Often smoother and structurally persistent vs claims metrics; may be a “slow signal” Ridge can use well. | Structural breaks (strategy shifts, acquisitions) can change baseline; also correlated with growth initiatives. |
| rate_adequacy_gap_yoy | PGR_specific | vmt_yoy; mom_3m/mom_6m; credit_spread_hy | `ppi_auto_ins_yoy − severity_index_yoy`, where `ppi_auto_ins_yoy` is YoY % change of auto insurance PPI and `severity_index_yoy` is a simple average of used-car CPI YoY and medical CPI YoY.citeturn0search3turn16search0 | Encodes “pricing vs loss-cost inflation” directly, a primary driver of auto insurer underwriting cycles. | Higher gap → improving rate adequacy → higher relative returns | monthly | FRED: PCU5241265241261 (auto insurance PPI) + CPI components.citeturn0search3turn16search0 | medium | high | Replaces a generic exposure proxy (VMT) with a margin mechanism; likely more stable and interpretable. | Weighting choices (used-car vs medical) matter; structural shifts (repair tech, litigiousness) may weaken CPI proxies. |
| ppi_auto_ins_yoy | shared_regime | vmt_yoy; mom_3m/mom_6m | YoY % change in “Direct P&C insurers: private passenger auto insurance” PPI.citeturn0search3 | Proxy for industry price level / rate actions; captures the underwriting pricing cycle. | Higher → premium pricing rising → higher relative returns (esp. if costs not rising faster) | monthly | FRED (BLS): PCU5241265241261.citeturn0search3 | low | medium-high | More “insurance-native” than broad macro; should be less redundant with risk proxies like VIX. | Does not capture company-specific pricing mix; could lag actual filed/earned rate changes. |
| severity_index_yoy | shared_regime | vmt_yoy; mom_3m/mom_6m | Average (or weighted average) of used-car CPI YoY and medical CPI YoY.citeturn16search0 | Captures two major components of claims severity inflation (auto physical damage + bodily injury/medical). | Higher → margin pressure → lower relative returns | monthly | FRED (BLS CPI components): CUSR0000SETA02 (used cars & trucks) + medical CPI series (already used in repo features).citeturn16search0 | low | medium-high | More causal than price momentum: it explains *why* underwriting margin expectations may deteriorate. | CPI components may not track insurer loss costs perfectly (labor, parts, legal inflation). |
| npw_per_pif_yoy | PGR_specific | npw_growth_yoy; pif_growth_yoy | YoY % change of `(net_premiums_written / pif_total)` (a proxy for average premium per policy).fileciteturn19file0L1-L1 | Separates rate level changes from unit growth; rising premium per policy often signals pricing power / rate actions. | Higher → improving pricing → higher relative returns (conditional on retention) | monthly | SEC monthly 8‑K: `net_premiums_written`, `pif_total`.fileciteturn19file0L1-L1 | medium | high | Upgrades `npw_growth_yoy` by splitting growth into *rate* vs *count*—more predictive and interpretable. | Mix changes (coverage, geography) can bias “premium per policy”; needs winsorization. |
| npw_vs_npe_spread_pct | PGR_specific | npw_growth_yoy; mom_12m | `(net_premiums_written − net_premiums_earned) / net_premiums_earned` (or NPE TTM) at time t.fileciteturn19file0L1-L1 | Signals whether written premium is outrunning earned premium (pipeline growth tailwind) vs decelerating. | Higher → growth mode → higher relative returns (if profitable) | monthly | SEC monthly 8‑K: NPW & NPE.fileciteturn19file0L1-L1 | low | medium | More direct than generic momentum; maps to an insurer’s revenue recognition pipeline. | Growth can be value-destructive if underpriced; sign may flip in competitive wars. |
| unearned_premium_growth_yoy | PGR_specific | mom_12m; pif_growth_yoy | YoY % change in `unearned_premiums`.fileciteturn19file0L1-L1 | Unearned premium is forward revenue inventory; increases can precede earned premium growth. | Higher → future revenue tailwind → higher relative returns (if margins stable) | monthly | SEC monthly 8‑K balance sheet.fileciteturn19file0L1-L1 | low | medium-high | Often smoother than NPW/PIF; may help Ridge as a stable leading indicator. | Can be affected by seasonality, billing terms, or reporting changes; requires proper lagging to avoid look-ahead. |
| reserve_to_npe_ratio | PGR_specific | credit_spread_hy; nfci | `loss_lae_reserves / net_premiums_earned` (monthly).fileciteturn19file0L1-L1 | Proxy for reserve adequacy / conservatism; rising ratio may signal adverse development risk.citeturn5view4 | Higher → worse reserve signal → lower relative returns | monthly | SEC monthly 8‑K: reserves and NPE.fileciteturn19file0L1-L1 | low | medium | More insurer-specific than generic financial conditions; targets a key risk channel in P&C earnings quality. | Ratio can rise in high growth phases without being “bad”; needs context. |
| channel_mix_direct_pct_yoy | PGR_specific | channel_mix_agency_pct; expense_ratio_ttm | YoY change (or YoY % change) in `pif_direct_auto / pif_total_personal_lines`.fileciteturn19file0L1-L1 | Mix shift toward direct can improve expense structure and margin durability. | Higher direct mix → higher relative returns | monthly | SEC monthly 8‑K: PIF by channel.fileciteturn19file0L1-L1 | medium | medium | More causal and interpretable than generic “growth”; links to distribution economics. | Product/model changes can affect channel economics; data coverage differs early vs later periods. |
| portfolio_yield_spread | PGR_specific | investment_income_growth_yoy; real_rate_10y | `investment_book_yield − GS10` (or another Treasury yield).fileciteturn19file0L1-L1citeturn18search4 | Captures carry advantage vs risk-free and the reinvestment tailwind/headwind for the insurer’s bond book.citeturn5view2 | Higher spread → higher carry → higher relative returns (with credit-risk caveat) | monthly | SEC monthly 8‑K (book yield) + FRED Treasury yields.fileciteturn19file0L1-L1citeturn18search4 | medium | medium-high | More structural than pure investment income YoY (which can be noisy); ties to bond benchmark behavior too. | Book yield series has shorter history (~2015+); can reduce effective sample. |
| duration_rate_shock_3m | shared_regime | real_rate_10y; yield_curvature | `fixed_income_duration × ΔGS10_3m` (3‑month change in 10Y yield).fileciteturn19file0L1-L1citeturn18search4 | Approximates mark-to-market pressure on the bond portfolio and book value when yields move; relevant to insurer equity and bond/MBS benchmarks.citeturn5view2 | More positive Δy × duration → more negative relative returns | monthly | SEC monthly 8‑K (duration) + FRED Treasury yields.fileciteturn19file0L1-L1citeturn18search4 | medium | high | Replaces “curve shape” with the driver that actually moves bond total returns (rate changes) and insurer OCI sensitivity. | Interaction features can be unstable if not scaled; requires careful leakage-safe timing. |
| unrealized_gain_pct_equity | PGR_specific | roe_net_income_ttm; pb_ratio | `net_unrealized_gains_fixed / shareholders_equity`.fileciteturn19file0L1-L1 | Captures OCI sensitivity and embedded AFS gain/loss position; relates to capital flexibility and rate exposure. | Higher → stronger equity buffer → higher relative returns | monthly | SEC monthly 8‑K: unrealized gains, equity.fileciteturn19file0L1-L1 | low | medium | May explain valuation moves not captured in ROE or P/B, especially around rate shocks. | Accounting classification changes; noise during rapid rate regimes; may be mean reverting. |
| realized_gain_to_net_income_ratio | PGR_specific | roe_net_income_ttm; investment_income_growth_yoy | `total_net_realized_gains / net_income` (TTM or smoothed).fileciteturn19file0L1-L1 | Earnings quality: high reliance on realized gains can be less repeatable than underwriting+core investment income.citeturn5view2 | Higher → lower quality → lower relative returns | monthly | SEC monthly 8‑K income statement lines.fileciteturn19file0L1-L1 | medium | medium | Adds a “quality” dimension missing from v14 lean sets. | Net income can be small/negative → ratio instability; needs capping. |
| underwriting_margin_ttm | PGR_specific | combined_ratio_ttm; underwriting_income_growth_yoy | `underwriting_income_ttm / net_premiums_earned_ttm` (or `−(combined_ratio_ttm−100%)` if you prefer).fileciteturn19file0L1-L1 | Direct measure of core insurance profitability (before investment/taxes); aligns with the 10‑K’s emphasis on underwriting profitability goals.citeturn5view0 | Higher → higher relative returns | monthly | SEC monthly 8‑K (underwriting income derivable; NPE).fileciteturn19file0L1-L1 | medium | high | Frequently more stable and interpretable than net income/ROE; strong candidate for Ridge. | Needs consistent derivation across history (early filing format risk). |
| usd_broad_return_3m | benchmark_predictive | nfci; vix | 3‑month % change in broad trade-weighted USD index (DTWEXBGS), aligned to month-end.citeturn23search2 | USD is a first-order driver of USD-based international equity returns (VXUS/VWO) and contributes to commodity returns. | Higher USD → PGR likely outperforms VXUS/VWO (relative return up) | daily→monthly | FRED (Board of Governors): DTWEXBGS.citeturn23search2 | low | medium-high | Gives “benchmark-side” explanatory power absent from insurer-only metrics; reduces unexplained relative moves vs VXUS/VWO. | USD effects can be regime dependent; may overlap with risk-off proxies. |
| wti_return_3m | benchmark_predictive | yield_curvature; mom_3m/mom_6m | 3‑month % change in WTI spot price (DCOILWTICO), aligned to month-end.citeturn23search1 | Oil is a core driver for energy equities (VDE) and a major weight in broad commodity exposure (DBC).citeturn13search24turn14search0 | Higher oil → PGR likely underperforms VDE/DBC (relative return down) | daily→monthly | FRED (EIA): DCOILWTICO.citeturn23search1 | low | high | Directly targets the benchmark universe; replaces “curve curvature” with a commodity/energy driver that should matter for VDE/DBC relative forecasts. | Geopolitical shocks can dominate (fat tails); may hurt stability if not robustified. |
| mortgage_spread_30y_10y | benchmark_predictive | yield_curvature; real_rate_10y | `MORTGAGE30US − GS10` (monthly average of weekly mortgage rate minus 10Y yield).citeturn24search2turn18search4 | Captures mortgage basis/spread pressure relevant to agency MBS total returns (VMBS).citeturn10search15 | Wider spread → VMBS weaker → PGR relative return vs VMBS up | weekly→monthly | FRED: MORTGAGE30US + GS10.citeturn24search2turn18search4 | low | medium | More benchmark-specific for VMBS than generic yield curvature; should reduce benchmark-side error. | Mortgage spread can be influenced by technicals; may be noisy with monthly sampling. |
| baa10y_spread | shared_regime | credit_spread_hy | Use Moody’s Baa–10Y Treasury spread (BAA10Y) as a smoother, more “core credit” spread than HY OAS.citeturn17search1 | Credit conditions affect equities, credit, and financial sector returns; Baa spread may be less jumpy than HY. | Higher spread → risk-off → ambiguous (often negative for equities) | daily→monthly | FRED: BAA10Y.citeturn17search1 | low | medium | Potentially less redundant with VIX/NFCI and less tail-driven than HY OAS. | Might be too slow-moving; may reduce sensitivity to sharp risk episodes. |

## Best Fixed-Budget Replacement Ideas

Below are **10 concrete swaps** designed to keep feature count roughly flat while upgrading economic specificity. The “remove” side references features inside the v14 lean Ridge/GBT sets.fileciteturn15file0L1-L1

1) **Remove:** `combined_ratio_ttm`  
**Add:** `loss_ratio_ttm` + `expense_ratio_ttm`  
**Why better:** same core concept but decomposed—lets the model separate claims inflation pressure from distribution efficiency.fileciteturn19file0L1-L1  
**Best for:** Ridge (primary), also GBT.

2) **Remove:** `vmt_yoy`  
**Add:** `rate_adequacy_gap_yoy`  
**Why better:** VMT is an exposure proxy; rate adequacy gap directly targets underwriting margin cycle (pricing vs costs).citeturn16search3turn0search3turn16search0  
**Best for:** both (likely strongest for GBT due to nonlinearity).

3) **Remove:** `mom_3m` + `mom_6m` (GBT)  
**Add:** `underwriting_margin_ttm` + `npw_per_pif_yoy`  
**Why better:** replaces short-horizon price chasing with fundamentals that explain *why* the stock should outperform diversified benchmarks.fileciteturn19file0L1-L1  
**Best for:** both (Ridge benefits from smooth fundamentals; GBT can pick thresholds).

4) **Remove:** `yield_curvature`  
**Add:** `mortgage_spread_30y_10y`  
**Why better:** for the selected universe including VMBS, mortgage spread targets MBS-specific risk premia more directly than curve curvature.citeturn10search15turn24search2turn18search4  
**Best for:** Ridge (stable linear relation), also useful for GBT.

5) **Remove:** `real_rate_10y`  
**Add:** `duration_rate_shock_3m`  
**Why better:** bond total returns and insurer OCI sensitivity are driven by yield moves times duration; interaction is closer to mechanism and benchmark impact.fileciteturn19file0L1-L1citeturn18search4  
**Best for:** GBT (handles interactions naturally), but also Ridge if scaled.

6) **Remove:** `investment_income_growth_yoy`  
**Add:** `portfolio_yield_spread`  
**Why better:** moves from noisy realized income changes to a structural carry/tailwind measure tied to rates and credit environment.fileciteturn19file0L1-L1  
**Best for:** Ridge.

7) **Remove:** `npw_growth_yoy`  
**Add:** `npw_vs_npe_spread_pct`  
**Why better:** NPW growth is ambiguous; NPW–NPE spread directly indicates growth pipeline vs runoff, closer to near-term revenue tailwind.fileciteturn19file0L1-L1  
**Best for:** both.

8) **Remove:** `credit_spread_hy`  
**Add:** `baa10y_spread`  
**Why better:** Baa spread is typically less jumpy and may be more stable month-to-month; can reduce overreaction to HY tails while still capturing credit regime.citeturn17search1  
**Best for:** Ridge; GBT may still prefer HY in crises.

9) **Remove:** `nfci` **or** `vix` (drop one to control redundancy)  
**Add:** `usd_broad_return_3m`  
**Why better:** adds benchmark-side explanatory power for VXUS/VWO and commodities while keeping a risk-regime proxy via the remaining NFCI/VIX.citeturn23search2turn10search14turn11search37  
**Best for:** both.

10) **Remove:** `yield_curvature` (or one momentum window)  
**Add:** `wti_return_3m`  
**Why better:** directly targets energy/commodity benchmarks (VDE/DBC) that otherwise inject unexplained variance into relative returns.citeturn13search24turn14search0turn23search1  
**Best for:** GBT (nonlinear), also Ridge if winsorized.

## Model-Specific Recommendations

### Ridge / linear models

Prioritize features that are **smooth, monotonic, and economically interpretable**, and avoid heavy tail sensitivity unless clipped.

1. `loss_ratio_ttm`fileciteturn19file0L1-L1  
2. `expense_ratio_ttm`fileciteturn19file0L1-L1  
3. `underwriting_margin_ttm`fileciteturn19file0L1-L1  
4. `npw_per_pif_yoy`fileciteturn19file0L1-L1  
5. `unearned_premium_growth_yoy`fileciteturn19file0L1-L1  
6. `mortgage_spread_30y_10y`citeturn24search2turn18search4  
7. `usd_broad_return_3m`citeturn23search2  

### GBT / tree models

Prioritize features with **threshold effects** and **regime interactions**, where trees can add value without adding model complexity.

1. `rate_adequacy_gap_yoy`citeturn0search3turn16search0  
2. `duration_rate_shock_3m`fileciteturn19file0L1-L1  
3. `wti_return_3m`citeturn23search1  
4. `reserve_to_npe_ratio`fileciteturn19file0L1-L1  
5. `realized_gain_to_net_income_ratio`fileciteturn19file0L1-L1  
6. `channel_mix_direct_pct_yoy`fileciteturn19file0L1-L1  
7. `baa10y_spread` (if you swap out HY OAS)citeturn17search1  

## Benchmark-Predictive Features

Benchmark-predictive features **should be included**, but **under a strict fixed-budget rule**: you add them by *replacing* redundant generic regime features, not by expanding the feature set.

The selected benchmark universe contains structurally different exposures—US equity, ex‑US equity and EM equity, IG bonds, agency MBS, gold bullion, commodity futures, and US energy equities.citeturn9search44turn10search14turn11search37turn9search40turn10search15turn14search37turn14search0turn13search24 A lean relative-return forecaster benefits when it has at least one or two strong “benchmark drivers” for each major block.

**Broad US equity (VOO)**: best candidates are “risk price” and “discount rate” proxies already in the stack (VIX, credit spreads, curve slope). The FRED definitions support VIX as an options-implied volatility gauge and spreads as credit risk pricing measures.citeturn19search0turn0search0turn16search2

**International equity (VXUS) and emerging markets (VWO)**: a broad USD index is one of the most direct benchmark-side drivers for USD-denominated foreign equity returns; it also interacts with global risk-on/off.citeturn23search2turn10search14turn11search37

**Fixed income / rate-sensitive assets (BND, VMBS)**: bond total returns are dominated by yield level and yield changes; VMBS has additional mortgage basis/spread exposure. Treasury yields and the mortgage rate series are public and updatable at low cadence.citeturn18search4turn24search2turn10search15turn9search40

**Gold / commodities / real assets (GLD, DBC, VDE)**: gold is explicitly tied to gold bullion price, and DBC is explicitly tied to a diversified commodity futures index; energy equities are anchored to the energy sector.citeturn14search37turn14search0turn13search24 In a lean framework, oil (WTI) and USD provide disproportionate benchmark-side explanatory power.citeturn23search1turn23search2

## Data Feasibility

### Feasible with SEC filings

Monthly EDGAR 8‑K supplements provide a rare high-cadence insurer dataset: underwriting ratios, NPW/NPE, PIF by channel, balance sheet items, capital actions, duration, and (partly) book yield.fileciteturn19file0L1-L1 The company itself states underwriting profitability (combined ratio) is central to its strategy, making these features economically grounded.citeturn5view0

**High-feasibility EDGAR-derived ideas (no new vendors):** `loss_ratio_ttm`, `expense_ratio_ttm`, `npw_per_pif_yoy`, `npw_vs_npe_spread_pct`, `unearned_premium_growth_yoy`, `reserve_to_npe_ratio`, `channel_mix_direct_pct_yoy`, `underwriting_margin_ttm`, `duration_rate_shock_3m`, `unrealized_gain_pct_equity`, `realized_gain_to_net_income_ratio`.fileciteturn19file0L1-L1

### Feasible with FRED public macro series

Public series already map directly to your strongest proposed “shared-regime” and “benchmark driver” ideas:

- Auto insurance price proxy: PPI for private passenger auto insurance.citeturn0search3  
- Claims severity proxies: CPI used cars & trucks (and medical CPI series that your pipeline already uses).citeturn16search0  
- Credit risk pricing: HY OAS (current feature) and Baa spread alternative.citeturn0search0turn17search1  
- Volatility/risk: VIX series definition on FRED.citeturn19search0  
- USD: broad trade-weighted dollar index.citeturn23search2  
- Oil: WTI spot.citeturn23search1  
- Rates/curve slope and inflation expectations: Treasury spreads and breakeven inflation rate.citeturn16search2turn15search0  
- Mortgage rate: Freddie Mac PMMS mortgage rate, enabling a mortgage spread.citeturn24search2  

### Feasible with existing price history and simple derived series

Your current feature base already relies on momentum/volatility and valuations computed from price history.fileciteturn11file0L1-L1 Those can be extended in a fixed budget by swapping *better* price-derived constructs (e.g., fewer redundant momentum windows, more “state variables” like drawdown to high, or relative strength vs the reduced universe).

If you rely on Alpha Vantage for adjusted monthly series, their documentation explicitly supports split/dividend-adjusted outputs including monthly adjusted time series.citeturn28search0

### Ideas that would require new external data

- **Peer underwriting/combined ratio spreads** (would require ingesting peers’ filings and normalizing across accounting/segments), even though it’s theoretically compelling as a “relative fundamentals” feature.fileciteturn19file0L1-L1  
- **Catastrophe intensity / weather-loss indices** (NOAA or insured loss vendors) to isolate cat-adjusted underwriting; could be valuable but violates the “avoid exotic paid data” preference unless you find a clean free proxy.citeturn5view3  
- **High-quality implied rate volatility** (MOVE) is often paywalled or licensing constrained; not recommended under your constraints.

## Final Shortlist

Exactly **15** candidate replacement features, ranked for v15 testing under a fixed feature budget:

1. **loss_ratio_ttm** — best one-for-one upgrade to `combined_ratio_ttm` because it isolates claims cost pressure.fileciteturn19file0L1-L1  
2. **expense_ratio_ttm** — pairs with loss ratio to decompose underwriting quality and capture durable efficiency.fileciteturn19file0L1-L1  
3. **rate_adequacy_gap_yoy** — directly encodes premium pricing vs loss-cost inflation (high lift, low complexity).citeturn0search3turn16search0  
4. **ppi_auto_ins_yoy** — a clean, monthly “price cycle” proxy that is insurer-specific and benchmark-agnostic.citeturn0search3  
5. **severity_index_yoy** — interpretable claims severity pressure proxy (used cars + medical).citeturn16search0  
6. **npw_per_pif_yoy** — separates rate increases from unit growth; stronger mechanism signal than `npw_growth_yoy` alone.fileciteturn19file0L1-L1  
7. **npw_vs_npe_spread_pct** — pipeline indicator for near-term revenue tailwind (growth mode vs runoff).fileciteturn19file0L1-L1  
8. **unearned_premium_growth_yoy** — forward inventory of revenue; typically smoother and Ridge-friendly.fileciteturn19file0L1-L1  
9. **reserve_to_npe_ratio** — insurer-specific earnings-quality / adverse development risk proxy.citeturn5view4turn19file0L1-L1  
10. **channel_mix_direct_pct_yoy** — captures durable structural margin improvements from distribution mix shift.fileciteturn19file0L1-L1  
11. **underwriting_margin_ttm** — “core profit” signal aligned to management’s underwriting profitability focus.citeturn5view0turn19file0L1-L1  
12. **portfolio_yield_spread** — investment carry tailwind/headwind feature tied to insurer economics and bond benchmarks.citeturn5view2turn18search4turn19file0L1-L1  
13. **duration_rate_shock_3m** — structural rate-risk/OCI feature (duration × yield move) with cross-benchmark relevance.citeturn18search4turn19file0L1-L1  
14. **usd_broad_return_3m** — minimal but high-value benchmark driver for VXUS/VWO-relative performance.citeturn23search2turn10search14turn11search37  
15. **wti_return_3m** — minimal but high-value benchmark driver for VDE/DBC-relative performance.citeturn23search1turn13search24turn14search0