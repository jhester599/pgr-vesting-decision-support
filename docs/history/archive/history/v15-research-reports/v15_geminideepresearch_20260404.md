# **Quantitative Feature Engineering for Relative-Return Forecasting: Progressive Corporation (PGR) versus Reduced Benchmark Universe**

## **Section 1: Executive Summary**

The following analysis outlines the strategic feature-replacement framework for version 15 (v15) of the Progressive Corporation (PGR) relative-return forecasting model. Operating under a strict fixed-budget dimensionality constraint, the objective is to optimize the inputs for the existing ensemble architecture (Ridge regression and Gradient Boosted Trees), predicting PGR returns against a highly specific reduced benchmark universe comprising VOO, VXUS, VWO, VMBS, BND, GLD, DBC, and VDE.

The transition from v14 to v15 requires a philosophical shift from broad, generic macroeconomic indicators toward highly granular, economically interpretable signals that directly map to the idiosyncratic risk premia of property and casualty (P\&C) insurance and the specific covariance structures of the benchmark assets.

* **Failure of Generic Macroeconomic Inputs:** The inability of the v14 ensemble to consistently outperform the historical mean baseline suggests an over-reliance on lagging, highly aggregated features such as Headline CPI, generic bond yields, or quarterly net income. These generic features fail to capture the high-frequency, idiosyncratic realities of P\&C insurance reserving and fail to accurately price the specific risk premia driving the reduced benchmark universe.  
* **Exploiting the Monthly Reporting Anomaly:** Progressive Corporation is uniquely transparent among its peers (e.g., Allstate, Travelers), publishing highly detailed monthly 8-K filings that disclose net premiums written, policies in force, and, crucially, the monthly combined ratio.1 V15 must pivot from quarterly, lagged earnings data to high-frequency, real-time monthly operational metrics that lead quarterly earnings surprises.  
* **Deconstructing PGR's Idiosyncratic Alpha:** PGR’s relative outperformance is primarily governed by its underwriting margin and top-line policy growth. The highest-value feature themes for v15 involve isolating the exact micro-inflationary pressures that drive auto insurance severity—specifically, the divergence between used car prices and auto repair costs relative to generic inflation.2  
* **Capturing the "Social Inflation" Penalty:** A critical blind spot in standard financial modeling of liability insurers is "social inflation"—the systemic impact of outsized jury awards, nuclear verdicts, and aggressive litigation tactics on loss reserves.4 Features that proxy legal service costs and reserve development trends are vital for predicting periods where PGR will face valuation compression due to unfavorable prior-year reserve adjustments.6  
* **Actuarial Frequency via Mobility Data:** Rather than relying on generic alternative data (e.g., smartphone mobility indices), the model must utilize the Federal Highway Administration’s Vehicle Miles Traveled (VMT).7 VMT is the actuarial standard for exposure; accelerating VMT without corresponding premium rate approvals mathematically guarantees margin compression.  
* **Optimizing Shared-Regime Benchmark Predictors:** Empirical research dictates that relative returns are often more predictable out-of-sample than aggregate absolute returns, provided the correct dimensional reduction is applied.8 V15 must swap absolute yield metrics for specialized risk premia (e.g., 10-Year Real Interest Rates, MBS Spreads, and Term Premia) to dictate the allocation flows between broad equities (VOO), fixed income (BND, VMBS), and real assets (GLD, DBC).  
* **Alignment with Ensemble Dynamics:** The proposed feature swaps are explicitly engineered to feed the dual-model architecture. Spreads, relative valuation ratios, and rates of change are tailored to enforce strict stationarity for the Ridge regression component. Conversely, threshold-based macro indicators (e.g., real rate regime shifts or credit spread spikes) are curated to allow the Gradient Boosted Trees (GBT) to map non-linear conditional dependencies.  
* **Strict Feasibility and Lean Implementation:** All 15 candidate features recommended for the v15 testing queue are entirely derivable from highly reliable, low-cost public data sources. By relying exclusively on SEC EDGAR (PGR's 8-K filings) and the Federal Reserve Economic Data (FRED) database, the architecture avoids the fragility, backtest-overfitting, and expense of exotic alternative data.

## **First Principles Reasoning**

To construct a robust feature set that improves signal quality without expanding dimensionality, the feature engineering process must be rooted in fundamental economic and financial principles.

### **1\. What drives PGR stock performance relative to diversified alternatives?**

Progressive Corporation is a property and casualty (P\&C) insurer. The economic engine of a P\&C insurer is fundamentally different from a standard corporate entity. Insurers generate value through two distinct mechanisms: underwriting profit and investment income on float.10 Underwriting profit is measured by the combined ratio, which is the sum of incurred losses and operating expenses divided by earned premiums. PGR explicitly targets a combined ratio of 96% or better.11 When PGR operates below this threshold, it generates essentially cost-free capital (float) which is then deployed into its investment portfolio.

Therefore, PGR's relative outperformance is driven by its ability to accurately price risk ahead of inflation. If claims severity (the cost to fix a crashed car) or claims frequency (how often cars crash) spikes unexpectedly, the combined ratio breaches 100%, float is destroyed, and the stock underperforms.13 Severity is driven by hyper-specific physical inflation—namely, the cost of used cars, auto parts, and mechanic wages 2—as well as "social inflation," which represents the rising cost of legal settlements and nuclear verdicts.4 Frequency is driven by macroscopic mobility, explicitly Vehicle Miles Traveled (VMT).7 PGR's relative stock performance accelerates when it demonstrates pricing power (measured by Policies in Force growth) while simultaneously keeping these severity and frequency vectors contained below its 96% combined ratio target.1

### **2\. What drives the reduced benchmark universe itself?**

The reduced universe (VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE) represents a diversified cross-section of global capital markets. Predicting PGR's *relative* return requires predicting the denominator. The performance of these assets is governed by structural macroeconomic risk premia rather than idiosyncratic corporate earnings:

* **VOO (US Equities):** Driven by the Equity Risk Premium (ERP)—the excess yield offered by earnings over the risk-free rate.15  
* **BND & VMBS (Fixed Income):** Driven by the Term Premium (the compensation for bearing duration risk) 16 and specific credit spreads. VMBS is heavily reliant on the spread between mortgage rates and Treasury yields.17  
* **GLD (Gold):** A zero-yielding asset driven almost entirely by the inverse of the 10-Year Real Interest Rate. When real rates are negative, the opportunity cost of holding gold disappears, driving outperformance.18  
* **DBC & VDE (Commodities & Energy):** Driven by physical demand throughput, which can be proxied by downstream consumption metrics like retail gasoline sales, acting as a barometer for the physical economy.20  
* **VXUS & VWO (International/Emerging):** Inversely correlated to the strength of the US Dollar, which dictates global liquidity and capital flight dynamics.

### **3\. Which current common feature types in these systems are likely too weak, redundant, or generic?**

Standard quantitative models often rely on features that are mathematically convenient but economically detached.

* *Headline CPI:* Too broad. Rent and food prices do not dictate auto insurance claims.  
* *Quarterly EPS:* P\&C earnings are highly distorted by quarterly mark-to-market accounting on massive bond portfolios and one-off reserve developments.21 Furthermore, quarterly data is severely lagged.  
* *Generic VIX:* Mean-reverts too quickly and reflects equity options hedging rather than true systemic financial stress.  
* *Trailing P/E Ratios:* Non-stationary and fundamentally flawed for insurers, whose intrinsic value is tied to Book Value and Return on Equity (ROE).21  
* *Raw Yield Curve Slopes (e.g., 2Y/10Y):* Conflates central bank policy rate expectations with the actual risk premium demanded by the market.

### **4\. Which replacement features could improve signal quality without expanding dimensionality?**

To improve signal quality within a fixed feature budget, the model must deploy *composite* features—metrics that combine two or more economic variables into a single, stationary spread or ratio. For example, combining the earnings yield of the S\&P 500 and the 10-Year Treasury yield into a single "Equity Risk Premium Proxy" replaces two raw features with one highly predictive, stationary signal. Similarly, utilizing PGR's unique monthly 8-K data allows the creation of a "Monthly Combined Ratio Delta" against the 96% target, providing a real-time, high-frequency profitability signal that renders lagging quarterly EPS data obsolete.1

## **Section 2: Ranked Candidate Feature Table**

| feature\_name | category | replace\_or\_compete\_with | definition | economic\_rationale | expected\_direction | likely\_frequency | likely\_source | implementation\_difficulty | likely\_signal\_quality | why\_it\_might\_outperform\_existing\_feature | key\_risks |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Auto\_Severity\_Inflation\_Spread | PGR\_specific | Headline\_CPI\_YoY | Average of YoY changes in FRED CUSR0000SETA02 (Used Cars) and CUSR0000SETD (Auto Repair) minus Headline CPI YoY. | Auto repair and replacement costs dictate claim severity. When these specific components outpace general inflation, underwriting margins compress, leading to unfavorable reserve development and multiple contraction.2 | Negative (Higher spread \= lower PGR relative return) | Monthly | FRED | Low | High | Headline CPI is too broad. Auto-specific inflation isolates the exact cost of goods sold for PGR's core product, predicting margin compression before quarterly earnings. | Used car prices can be volatile; repair costs tend to be sticky. Lag in BLS reporting. |
| Monthly\_Combined\_Ratio\_Delta | PGR\_specific | Quarterly\_Net\_Income | 3-month rolling average of PGR's monthly reported combined ratio minus the company's historical 96% target. | The combined ratio is the definitive metric of underwriting profitability. A ratio below 96 indicates profit; rising ratios indicate margin pressure. PGR uniquely reports this monthly via 8-K.1 | Negative | Monthly | SEC EDGAR (PGR 8-K) | Low | High | Replaces lagged, heavily smoothed quarterly EPS with real-time monthly underwriting efficiency, directly tied to PGR's stated enterprise goals. | Vulnerable to one-off catastrophic weather events causing temporary, mean-reverting spikes. |
| PIF\_Growth\_Acceleration | PGR\_specific | Trailing\_12M\_Revenue\_Growth | 3-month annualized growth rate of PGR Policies in Force (PIF) minus the 12-month trailing growth rate. | Measures the second derivative of growth. Accelerating PIF indicates successful rate-taking and market share capture, signaling strong forward premium generation.1 | Positive | Monthly | SEC EDGAR (PGR 8-K) | Low | High | Revenue is a lagging indicator influenced by past rate hikes. PIF measures active volume. Acceleration provides a stationary, leading signal of future cash flow. | Monthly PIF data can exhibit seasonality which must be properly adjusted during preprocessing. |
| 10Y\_Real\_Interest\_Rate | shared\_regime | Nominal\_10Y\_Yield | 10-Year Treasury Inflation-Indexed Security Yield (FRED: REAINTRATREARAT10Y). | Real rates represent the true cost of capital and the discount rate for equities. They inversely drive gold (GLD) and dictate the attractiveness of fixed income (BND) over equities (VOO).18 | Complex (Non-linear across benchmark) | Daily/Monthly | FRED | Low | High | Nominal yields conflate inflation expectations and real growth. Real rates isolate the specific liquidity regime driving broad asset allocation across the reduced universe. | May be overly correlated with pure inflation expectations during extreme, rapid macroeconomic shocks. |
| Legal\_Services\_PPI\_Relative | PGR\_specific | Generic\_Legal\_Cost\_Index | YoY change in Producer Price Index: Legal Services (FRED: PCU54115411) divided by YoY Headline PPI. | Proxies "social inflation"—the rising cost of litigation and nuclear verdicts. Social inflation forces insurers to increase reserves retroactively, which destroys equity value.4 | Negative | Monthly | FRED | Low | Medium | Directly attacks the most opaque risk in casualty insurance (social inflation) using a measurable federal index, replacing anecdotal or absent litigation metrics. | The BLS index may understate the true extreme tail risks of "nuclear verdicts" in commercial auto. |
| VMT\_Momentum | PGR\_specific | Generic\_Mobility\_Index | YoY percentage change in the 12-Month Moving Total Vehicle Miles Traveled (FRED: TRFVOLUSM227SFWA). | VMT is the primary macro driver of claim frequency. More miles driven equals more accidents. Rising VMT without corresponding premium rate increases compresses margins.20 | Negative | Monthly | FRED | Low | Medium | Unlike generic mobility data, VMT is the actuarial standard for exposure. It serves as a direct leading indicator for claim frequency. | Data is published with a 1-2 month lag by the Federal Highway Administration, requiring careful time-alignment. |
| Term\_Premium\_10Y | benchmark\_predictive | Yield\_Curve\_Slope\_2Y10Y | Term Premium on a 10-Year Zero Coupon Bond (FRED: THREEFYTP10). | The term premium is the excess yield investors require to commit to holding a long-term bond instead of a series of shorter-term bonds. It drives the performance of BND and VOO.16 | Complex | Daily/Monthly | FRED | Low | High | The raw 2Y/10Y spread conflates rate expectations with the risk premium. Term premium isolates the pure risk compensation driving bond benchmark returns. | Model relies on theoretical constructs (e.g., Adrian-Crump-Moench model) which can undergo revisions by the Federal Reserve. |
| MBS\_Treasury\_Spread | benchmark\_predictive | Generic\_Corporate\_Spread | Yield spread between Mortgage-Backed Securities and 10-Year Treasuries (or FRED TMBACBW027NBOG growth). | Captures the specific risk premium in the mortgage market, a primary driver of the VMBS benchmark. Widening spreads indicate housing/credit stress and impact bond allocations.27 | Negative (for VMBS relative) | Weekly/Monthly | FRED / Derived | Medium | High | Directly targets the pricing mechanism of the VMBS benchmark component, providing much better relative predictability than broad corporate spreads. | Highly subject to central bank interventions (e.g., Federal Reserve quantitative easing/tightening in the MBS market). |
| ERP\_Proxy\_Spread | benchmark\_predictive | Trailing\_SP500\_PE | S\&P 500 Earnings Yield (1/PE) minus the 10-Year Treasury Yield. | The Equity Risk Premium (ERP) determines the excess return of broad equities (VOO) over risk-free assets. A shrinking ERP suggests equities are overvalued relative to bonds.15 | Positive (for VOO relative) | Monthly | Derived (Price/FRED) | Low | High | Raw P/E ratios are non-stationary and ignore the cost of capital. ERP adjusts equity valuation for the prevailing interest rate regime, creating a stationary oscillator. | Earnings yields rely on forward estimates which can be systematically biased by analysts during turning points. |
| PGR\_Price\_to\_Book\_Relative | PGR\_specific | PGR\_Trailing\_PE | PGR Price-to-Book ratio divided by the XLF (Financial Sector) Price-to-Book ratio. | In the insurance industry, Book Value (and ROE) is a far more reliable valuation metric than P/E due to the volatility of quarterly underwriting and mark-to-market investment gains.21 | Mean-reverting | Daily/Monthly | Derived | Low | High | P/E is distorted by reserve developments and investment portfolio swings. Relative P/B is the industry standard for intrinsic valuation mean-reversion. | Book value can be temporarily distorted by Accumulated Other Comprehensive Income (AOCI) swings from bond portfolios during rate shocks. |
| US\_Corporate\_Credit\_Spread | shared\_regime | Generic\_VIX | Moody's Seasoned Baa Corporate Bond Yield minus 10-Year Treasury Yield (FRED: BAA10Y). | A widening spread indicates growing systemic financial stress and rising default probabilities, triggering risk-off regimes that favor BND/GLD over VOO/VXUS/PGR.30 | Negative (Risk-off) | Daily/Monthly | FRED | Low | High | VIX is highly mean-reverting and often noisy. The BAA spread provides a more persistent, economically grounded signal of true credit market stress and liquidity drain. | Can remain artificially compressed during periods of heavy corporate stock buybacks or low net debt issuance. |
| Gasoline\_Retail\_Sales\_Delta | benchmark\_predictive | WTI\_Crude\_Price | YoY change in Advance Retail Sales: Gasoline Stations (FRED: RSGASS). | Measures actual consumer end-demand for energy, serving as a highly correlated driver for the energy sector (VDE) and broad commodities (DBC).20 | Positive (for VDE/DBC) | Monthly | FRED | Low | Medium | Raw oil prices are driven by speculative futures and supply shocks. Retail sales proxy actual economic throughput and physical demand strength. | Highly sensitive to price per gallon changes rather than purely volumetric demand, requiring careful normalization. |
| USD\_Momentum\_Index | benchmark\_predictive | Static\_FX\_Rates | 3-month rate of change in the Broad Trade-Weighted US Dollar Index. | A strong USD negatively impacts international equities (VXUS) and emerging markets (VWO) due to capital flight and tightened global financial conditions. | Negative (for VXUS/VWO) | Monthly | FRED | Low | High | Replaces noisy, static bilateral FX rates with a stationary, macro-regime indicator that governs global equity flows and cross-border liquidity. | Currency momentum can reverse violently due to unpredictable central bank interventions or geopolitical shocks. |
| Commodity\_Equity\_Momentum | benchmark\_predictive | Generic\_Commodity\_Index | 6-month price return of DBC minus 6-month price return of VOO. | Captures the relative rotation between financial assets and real assets. Identifies inflationary regimes where input costs rise and corporate margins compress.31 | Momentum-following | Monthly | Derived | Low | Medium | Absolute commodity prices lack context. Relative momentum identifies the specific asset-allocation regime driving institutional flows between the benchmarks. | Commodities can experience sharp, idiosyncratic supply shocks (e.g., geopolitical conflicts) unrelated to broader macro trends. |
| PGR\_Premium\_to\_Surplus | PGR\_specific | PGR\_Dividend\_Yield | PGR Net Premiums Written (Annualized) divided by Total Shareholders' Equity (Surplus). | Operating leverage metric. A high ratio indicates the company is writing maximum business against its capital base; if combined ratios slip, equity is rapidly destroyed.13 | Mean-reverting | Quarterly/Monthly | SEC EDGAR | Medium | High | Dividend yields for insurers are noisy and discretionary. Premium-to-surplus measures the actual mechanical risk to the balance sheet and regulatory capital constraints. | Requires careful interpolation between quarterly 10-Q reporting and monthly 8-K premium data to maintain a smooth time series. |

## **Section 3: Best Fixed-Budget Replacement Ideas**

To maintain the lean architecture of the v15 ensemble\_ridge\_gbt model, expanding the feature space is prohibited. The strategy requires ruthlessly pruning features that offer weak, non-stationary, or logically disjointed signals, replacing them one-for-one (or two-for-one) with the superior, economically grounded metrics outlined above.

**1\. Swap: Headline Inflation for Micro-Severity Inflation**

* **Remove:** Headline\_CPI\_YoY  
* **Add:** Auto\_Severity\_Inflation\_Spread  
* **Why the swap is better:** Headline CPI is a generic macroeconomic indicator heavily weighted by housing and food, which have zero bearing on P\&C insurance liabilities. PGR's cost of goods sold is entirely dictated by auto parts, mechanic wages, and used car replacement costs. By blending FRED series CUSR0000SETA02 (Used Cars) and CUSR0000SETD (Auto Repair) and measuring their spread against broad inflation, we create a highly specific, economically interpretable proxy for PGR's impending claims severity.2 When this spread widens, PGR's margins will inherently compress unless aggressive rate action is taken.  
* **Target Model:** Both (Ridge for linear margin compression mapping; GBT for identifying severity spikes that cross critical profitability thresholds).

**2\. Swap: Lagged Earnings for High-Frequency Efficiency**

* **Remove:** Quarterly\_Net\_Income  
* **Add:** Monthly\_Combined\_Ratio\_Delta  
* **Why the swap is better:** Progressive is unique among its major publicly traded peers in that it releases monthly 8-K filings detailing its combined ratio and net premiums written.1 Quarterly net income is a severely lagged, heavily smoothed metric that the market has already priced in. The monthly combined ratio provides a real-time, leading indicator of underwriting profitability relative to management's strict 96% target.11 This delta mathematically dictates the generation of float.  
* **Target Model:** Both (Highly predictive of near-term relative price action and highly stationary).

**3\. Swap: Static Valuations for Relative Book Value**

* **Remove:** PGR\_Trailing\_PE  
* **Add:** PGR\_Price\_to\_Book\_Relative  
* **Why the swap is better:** In the P\&C insurance industry, P/E ratios are notoriously deceptive. Earnings are heavily distorted by mark-to-market fluctuations in the massive fixed-income investment portfolios and irregular, one-time prior-year reserve development adjustments.21 Price-to-Book (calculated relative to the XLF financial sector to remove broad market beta) captures the true intrinsic valuation mean-reversion of the enterprise, aligning with industry-standard actuarial valuation methods.  
* **Target Model:** Ridge (Highly mean-reverting, stationary, and linear).

**4\. Swap: Generic Mobility for Actuarial Exposure**

* **Remove:** Generic\_Mobility\_Index (e.g., Apple/Google mobility data)  
* **Add:** VMT\_Momentum  
* **Why the swap is better:** Alternative smartphone mobility data is noisy, subject to localized methodological changes, and lacks historical depth for robust backtesting. The Federal Highway Administration’s Vehicle Miles Traveled (VMT) is the gold standard actuarial exposure metric. A year-over-year acceleration in VMT directly correlates with higher accident frequency, which pressures the combined ratio if not offset by lower severity.7  
* **Target Model:** GBT (Captures threshold effects where frequency suddenly overwhelms the premium pricing models, triggering non-linear drawdowns).

**5\. Swap: Nominal Yields for Real Liquidity**

* **Remove:** Nominal\_10Y\_Yield  
* **Add:** 10Y\_Real\_Interest\_Rate  
* **Why the swap is better:** Nominal yields conflate economic growth expectations with inflation expectations. Real interest rates (FRED: REAINTRATREARAT10Y) isolate the true cost of capital and the restrictiveness of monetary policy. This is the primary driver for the reduced benchmark universe: it inversely prices the zero-yield GLD asset, sets the fundamental discount rate for long-duration equities like VOO, and dictates fixed-income flows into BND.18  
* **Target Model:** GBT (Models distinct regime shifts in broad asset allocation).

**6\. Swap: Opaque Legal Risks for Federal Proxies**

* **Remove:** Generic\_Legal\_Cost\_Index (or alternative NLP sentiment features)  
* **Add:** Legal\_Services\_PPI\_Relative  
* **Why the swap is better:** "Social inflation"—driven by nuclear verdicts, aggressive plaintiff litigation, and shifting jury demographics—is the largest systemic, unmodeled risk to commercial and personal auto liability.4 Paid NLP datasets attempting to track this are often expensive, brittle, and overfitted. The BLS Producer Price Index for Legal Services (PCU54115411) offers a robust, free, monthly gauge of rising legal costs, acting as a highly stationary proxy for social inflation risk.  
* **Target Model:** Ridge (Captures the linear erosion of liability margins).

**7\. Swap: Revenue Growth for Unit Economics**

* **Remove:** Trailing\_12M\_Revenue\_Growth  
* **Add:** PIF\_Growth\_Acceleration  
* **Why the swap is better:** Top-line revenue growth can be artificially inflated by desperate rate hikes even as an insurer bleeds customers. Policies in Force (PIF) represents actual unit volume. Accelerating PIF growth proves that PGR's pricing algorithms are competitive enough to steal market share while maintaining margin.1 By taking the second derivative (acceleration), we create a stationary oscillator that leads future cash flow generation.  
* **Target Model:** Both.

**8\. Swap: Yield Curve Slope for Pure Risk Premium**

* **Remove:** Yield\_Curve\_Slope\_2Y10Y  
* **Add:** Term\_Premium\_10Y  
* **Why the swap is better:** The raw 2Y/10Y spread is a blunt instrument that mixes expected central bank policy rate paths with risk compensation. The 10-Year Term Premium (THREEFYTP10) isolates the exact excess yield investors demand for bearing duration risk. This serves as a vastly superior and more precise predictor for the BND and VMBS aggregate bond benchmark components.16  
* **Target Model:** Ridge.

**9\. Swap: Broad Corporate Spreads for Sector Specifics**

* **Remove:** Generic\_Corporate\_Spread (e.g., CDX IG)  
* **Add:** MBS\_Treasury\_Spread  
* **Why the swap is better:** To predict the VMBS (Mortgage-Backed Securities) component of the benchmark universe, generic corporate spreads introduce unnecessary noise related to industrial and tech sector balance sheets. The spread between MBS and Treasuries directly prices the convexity, prepayment, and credit risk of the underlying VMBS asset, offering a pure signal for this specific benchmark constituent.27  
* **Target Model:** Ridge.

**10\. Swap: Absolute Yields for Equity Attractiveness (Two-for-One)**

* **Remove:** Trailing\_SP500\_PE AND Generic\_VIX  
* **Add:** ERP\_Proxy\_Spread (Equity Risk Premium)  
* **Why the swap is better:** This two-for-one swap reduces dimensionality while improving signal quality. The ERP combines both equity valuation (S\&P 500 Earnings Yield) and the macro rate environment (10-Year Treasury Yield) into a single, highly stationary metric. It dictates the relative attractiveness of VOO against risk-free or defensive assets like BND. By removing the non-stationary P/E ratio and the overly noisy VIX, the model relies on an economically rigorous signal that drives institutional asset allocation.15  
* **Target Model:** Ridge.

## **Section 4: Model-Specific Recommendations**

The v15 architecture relies on a proven ensemble\_ridge\_gbt methodology. It is critical to understand that these models learn fundamentally different representations of the underlying data. Ridge Regression requires stationary, normalized inputs to prevent coefficient explosion and isolate linear correlations. Gradient Boosted Trees (GBT) do not require strict stationarity and thrive on discovering hierarchical, regime-based thresholds and non-linear interactions. Features must be curated specifically to feed these varying mathematical appetites.

**Top Candidate Replacement Features for Ridge (Linear Models):**

Linear models excel at identifying mean-reversion and stable, continuous correlations over time. Inputs must be formulated as rates of change, spreads, or relative ratios to ensure strict stationarity.

1. **PGR\_Price\_to\_Book\_Relative:** Highly mean-reverting. A low relative P/B historically predicts strong forward returns as P\&C valuations normalize back to industry medians.21  
2. **Monthly\_Combined\_Ratio\_Delta:** A continuous, stationary spread oscillating around the 96% management target. This linearly maps to operating profit generation and subsequent float deployment.  
3. **ERP\_Proxy\_Spread:** A classic, stationary financial spread that dictates the mechanical flow of capital between equities (VOO) and fixed income (BND).  
4. **Auto\_Severity\_Inflation\_Spread:** A continuous variable tracking the linear pressure on margins caused by physical replacement costs.

**Top Candidate Replacement Features for GBT (Tree Models):**

Tree models excel at finding non-linear regime shifts, structural breaks, and complex conditional interactions (e.g., "If VMT is high AND Real Rates are high, then relative return drops precipitously").

1. **10Y\_Real\_Interest\_Rate:** Real rates often act as step-functions in global markets. A shift from a negative to a positive real rate regime radically alters the fundamental behavior of GLD and VOO. Trees can map these inflection points.19  
2. **US\_Corporate\_Credit\_Spread (BAA10Y):** Credit spreads stay artificially compressed for long periods and spike violently during liquidity crises. GBTs easily isolate these specific "risk-off" panic regimes where PGR acts as a defensive haven.  
3. **VMT\_Momentum:** Driving behavior has elastic limits. Trees can identify the critical threshold where excess VMT overwhelms the actuarial pricing models, triggering drawdowns.  
4. **PIF\_Growth\_Acceleration:** Trees can capture the non-linear compounding effects and network advantages of rapid market share acquisition during periods where competitors are shedding policies.

## **Section 5: Benchmark-Predictive Features**

**Should benchmark-predictive features be included?** Explicitly, yes. Empirical quantitative research demonstrates that relative returns—the components of returns beyond the aggregate index—are highly predictable, typically more so than the absolute returns of the index itself.8 Forecasting relative returns is a two-sided equation: ![][image1]. If the model relies solely on PGR-specific data, it will fail during periods of severe macroeconomic dislocation that disproportionately affect the benchmark. Because the reduced universe is heavily diversified, the model must include features that dictate the covariance and expected returns of these specific assets, allowing the ensemble to predict the denominator of the relative-return equation.

**Broad US Equity (VOO):**

* *Feature:* **ERP\_Proxy\_Spread** (Earnings Yield minus 10-Year Yield).  
* *Rationale:* Broad equities are driven by the excess return they offer over risk-free alternatives. When the ERP compresses, VOO is highly vulnerable to systemic drawdowns, making defensive, low-beta assets (like PGR) or fixed income (BND) more attractive on a relative basis.15

**International and Emerging Equity (VXUS, VWO):**

* *Feature:* **USD\_Momentum\_Index**.  
* *Rationale:* International equities are mechanically tied to currency fluctuations. A strong US Dollar acts as a tightening of global financial conditions, drawing capital away from emerging markets (VWO) and developed international markets (VXUS), while making domestic US assets (VOO, PGR) relatively stronger due to capital flight.

**Fixed Income and Rate-Sensitive Assets (BND, VMBS):**

* *Features:* **Term\_Premium\_10Y** and **MBS\_Treasury\_Spread**.  
* *Rationale:* BND is a duration-heavy aggregate bond fund, making it highly sensitive to the term premium.16 VMBS requires a specific spread metric to account for housing market credit risk and prepayment convexity, which standard corporate spreads fail to capture.28

**Gold, Commodities, and Real Assets (GLD, DBC, VDE):**

* *Features:* **10Y\_Real\_Interest\_Rate** and **Gasoline\_Retail\_Sales\_Delta**.  
* *Rationale:* Gold (GLD) is a zero-yield asset; its primary pricing mechanism is the inverse of the real interest rate.18 The energy sector (VDE) and broad commodities (DBC) are driven by physical demand and inventory drawdowns, which are excellently proxied by downstream retail gasoline sales data.20

By including exactly one highly targeted, economically grounded feature for each sub-asset class in the benchmark, the ensemble model can accurately map the macro regime without bloating the feature space with hundreds of noisy, collinear variables.

## **Section 6: Data Feasibility**

A primary constraint of the v15 update is to avoid exotic, expensive, or low-frequency alternative data that introduces pipeline fragility. All proposed features have been vetted for high-frequency availability, historical depth, and public accessibility.

**Feasible using SEC EDGAR / Company Filings:**

* Monthly\_Combined\_Ratio\_Delta: Sourced directly from PGR's unique monthly 8-K earnings releases. This requires simple NLP/regex parsing of the standardized monthly tables.1  
* PIF\_Growth\_Acceleration: Sourced from the same monthly 8-K filings, tracking the explicitly reported "Policies in Force" line items.  
* PGR\_Premium\_to\_Surplus: Derived by combining quarterly 10-Q/10-K Shareholders' Equity data with annualized monthly Net Premiums Written from the 8-Ks.

**Feasible using FRED / Public Macro Series:**

The Federal Reserve Economic Data (FRED) API provides robust, free, and historically deep access to the required macroeconomic drivers.

* Auto\_Severity\_Inflation\_Spread: Requires pulling FRED series CUSR0000SETA02 (Used Cars) and CUSR0000SETD (Auto Repair).2  
* VMT\_Momentum: FRED series TRFVOLUSM227SFWA.7  
* Legal\_Services\_PPI\_Relative: FRED series PCU54115411.24  
* 10Y\_Real\_Interest\_Rate: FRED series REAINTRATREARAT10Y.19  
* US\_Corporate\_Credit\_Spread: FRED series BAA10Y.  
* MBS\_Treasury\_Spread: Derived using FRED series TMBACBW027NBOG or directly via spread series.28  
* Term\_Premium\_10Y: FRED series THREEFYTP10.16  
* Gasoline\_Retail\_Sales\_Delta: FRED series RSGASS.20

**Feasible using Existing Price History / Simple Derived Series:**

* ERP\_Proxy\_Spread: Derived using S\&P 500 Trailing Earnings Yield and the 10-Year Treasury Yield.  
* PGR\_Price\_to\_Book\_Relative: Derived using standard daily closing prices and book value estimates.  
* Commodity\_Equity\_Momentum: Derived using DBC vs VOO daily price momentum.  
* USD\_Momentum\_Index: Derived from the DXY index price momentum.

**Data requiring new external sourcing:**

* **None.** Every single one of the 15 proposed features can be engineered via robust, free APIs (FRED API, SEC EDGAR scraping, standard Yahoo/YFinance market data feeds). This ensures absolute operational stability for the production-facing recommendation layer and guarantees that backtesting can extend deep into previous economic cycles.

## **Section 7: Final Shortlist**

The following 15 features constitute the exact candidate queue for v15 testing. They adhere strictly to the fixed-budget dimensionality constraint, systematically replacing weaker generic features with specialized, economically interpretable signals optimized for the ensemble\_ridge\_gbt architecture.

1. **Auto\_Severity\_Inflation\_Spread**: Isolates the exact physical inflationary pressures (used cars, auto repair) destroying P\&C underwriting margins, acting as a leading indicator of loss reserving stress.  
2. **Monthly\_Combined\_Ratio\_Delta**: Leverages PGR's unique monthly 8-K reporting to provide a real-time, high-frequency measure of operational profitability relative to management's 96% target.  
3. **PIF\_Growth\_Acceleration**: Measures the second derivative of Policies in Force, capturing true unit-economic momentum and market share capture independent of raw premium price hikes.  
4. **10Y\_Real\_Interest\_Rate**: Replaces nominal yields to provide the true macroeconomic discount rate, serving as the primary directional driver for GLD, VOO, and fixed-income benchmarks.  
5. **Legal\_Services\_PPI\_Relative**: Provides a robust, monthly public-data proxy for "social inflation" and nuclear verdicts, which represent the largest opaque tail-risk to liability insurers.  
6. **VMT\_Momentum**: Uses the Federal Highway Administration’s actual miles-driven data to perfectly proxy actuarial accident frequency, replacing noisy alternative mobility data.  
7. **MBS\_Treasury\_Spread**: Specifically targets the mortgage credit and prepayment risk premia necessary to accurately forecast the VMBS component of the benchmark universe.  
8. **Term\_Premium\_10Y**: Isolates the pure risk compensation demanded by bondholders, stripping out central bank rate expectations to better predict the BND aggregate bond benchmark.  
9. **ERP\_Proxy\_Spread**: Combines equity valuations and interest rates into a single stationary metric to predict the relative attractiveness of VOO against defensive assets.  
10. **PGR\_Price\_to\_Book\_Relative**: Utilizes the insurance industry standard for intrinsic valuation, offering a highly mean-reverting signal for the Ridge model while avoiding the noise of P/E ratios.  
11. **Gasoline\_Retail\_Sales\_Delta**: Proxies end-consumer physical energy demand to predict the relative momentum of the VDE and DBC benchmark components.  
12. **USD\_Momentum\_Index**: Captures the global liquidity and currency regime that mechanically dictates the relative performance of VXUS and VWO international equities.  
13. **US\_Corporate\_Credit\_Spread (BAA10Y)**: Provides a persistent, economically grounded signal of systemic financial stress and risk-off regimes, replacing the highly volatile and noisy VIX.  
14. **Commodity\_Equity\_Momentum**: Tracks the relative flow of capital between real assets (DBC) and financial assets (VOO) to map broad inflationary and stagflationary market regimes.  
15. **PGR\_Premium\_to\_Surplus**: Measures critical operating leverage on the balance sheet, identifying periods where PGR is mathematically overexposed to sudden spikes in claims severity.

#### **Works cited**

1. Progressive Reports February 2026 Results \- PGR \- Stock Titan, accessed April 4, 2026, [https://www.stocktitan.net/news/PGR/progressive-reports-february-2026-2f9vmmp1x9p7.html](https://www.stocktitan.net/news/PGR/progressive-reports-february-2026-2f9vmmp1x9p7.html)  
2. Consumer Price Index for All Urban Consumers: Used Cars and Trucks in U.S. City Average (CUSR0000SETA02) | FRED, accessed April 4, 2026, [https://fred.stlouisfed.org/series/CUSR0000SETA02](https://fred.stlouisfed.org/series/CUSR0000SETA02)  
3. Table 1\. Consumer Price Index for All Urban Consumers (CPI-U): U. S. city average, by expenditure category \- Bureau of Labor Statistics, accessed April 4, 2026, [https://www.bls.gov/news.release/cpi.t01.htm](https://www.bls.gov/news.release/cpi.t01.htm)  
4. Social Inflation \- korean securities association, accessed April 4, 2026, [https://apjfs.org/file/download/6690?view=1](https://apjfs.org/file/download/6690?view=1)  
5. cas research paper \- social inflation and loss development — an update \- Casualty Actuarial Society, accessed April 4, 2026, [https://www.casact.org/sites/default/files/2023-03/RP\_Social\_Inflation\_Update.pdf](https://www.casact.org/sites/default/files/2023-03/RP_Social_Inflation_Update.pdf)  
6. Document \- SEC.gov, accessed April 4, 2026, [https://www.sec.gov/Archives/edgar/data/80661/000008066123000034/pgr2023630ex99earningsrele.htm](https://www.sec.gov/Archives/edgar/data/80661/000008066123000034/pgr2023630ex99earningsrele.htm)  
7. Vehicle Miles Traveled (TRFVOLUSM227SFWA) | FRED | St. Louis Fed, accessed April 4, 2026, [https://fred.stlouisfed.org/series/TRFVOLUSM227SFWA](https://fred.stlouisfed.org/series/TRFVOLUSM227SFWA)  
8. The predictability of relative asset returns | Macrosynergy, accessed April 4, 2026, [https://macrosynergy.com/research/the-predictability-of-relative-returns/](https://macrosynergy.com/research/the-predictability-of-relative-returns/)  
9. Predicting Relative Returns \- NBER, accessed April 4, 2026, [https://www.nber.org/system/files/working\_papers/w23886/w23886.pdf](https://www.nber.org/system/files/working_papers/w23886/w23886.pdf)  
10. PGR (Progressive Corp) vs S\&P 500 Comparison \- Alpha Spread, accessed April 4, 2026, [https://www.alphaspread.com/comparison/nyse/pgr/vs/indx/gspc](https://www.alphaspread.com/comparison/nyse/pgr/vs/indx/gspc)  
11. Is Progressive's Profitability Anchored by Combined Ratio? \- August 26, 2025 \- Zacks.com, accessed April 4, 2026, [https://www.zacks.com/stock/news/2743063/is-progressives-profitability-anchored-by-combined-ratio](https://www.zacks.com/stock/news/2743063/is-progressives-profitability-anchored-by-combined-ratio)  
12. Earnings call transcript: Progressive Q4 2025 shows robust growth, stock dips, accessed April 4, 2026, [https://www.investing.com/news/transcripts/earnings-call-transcript-progressive-q4-2025-shows-robust-growth-stock-dips-93CH-4538474](https://www.investing.com/news/transcripts/earnings-call-transcript-progressive-q4-2025-shows-robust-growth-stock-dips-93CH-4538474)  
13. The Progressive Corporation \- 2025 Annual Report to Shareholders, accessed April 4, 2026, [https://s202.q4cdn.com/605347829/files/doc\_financials/2025/q4/interactive/Progressive%202025%20Interactive%20Annual%20Report/pdfs/Progressive-2025-Financial-Review.pdf](https://s202.q4cdn.com/605347829/files/doc_financials/2025/q4/interactive/Progressive%202025%20Interactive%20Annual%20Report/pdfs/Progressive-2025-Financial-Review.pdf)  
14. Three Reasons Why Your Car Insurance Rate Might Go Up in 2024, accessed April 4, 2026, [https://www.fredcchurch.com/three-reasons-why-your-car-insurance-premium-may-increase/](https://www.fredcchurch.com/three-reasons-why-your-car-insurance-premium-may-increase/)  
15. A Look at Stock Valuations Relative to Bonds \- DSB Rock Island Wealth Management, accessed April 4, 2026, [https://dsb-rockisland.com/wealth-management-and-planning/wealth-management-and-planning/a-look-at-stock-valuations-relative-to-bonds/](https://dsb-rockisland.com/wealth-management-and-planning/wealth-management-and-planning/a-look-at-stock-valuations-relative-to-bonds/)  
16. Term Premium \- Economic Data Series | FRED | St. Louis Fed, accessed April 4, 2026, [https://fred.stlouisfed.org/tags/series?t=term+premium](https://fred.stlouisfed.org/tags/series?t=term+premium)  
17. Monetary Policy and the Mortgage Market \- FDIC, accessed April 4, 2026, [https://www.fdic.gov/center-financial-research/dominik-supera-paper.pdf](https://www.fdic.gov/center-financial-research/dominik-supera-paper.pdf)  
18. The Role of Gold in a Portfolio \- WisdomTree, accessed April 4, 2026, [https://www.wisdomtree.com/investments/-/media/us-media-files/documents/resource-library/market-insights/gannatti-commentary/the-role-of-gold-in-a-portfolio.pdf](https://www.wisdomtree.com/investments/-/media/us-media-files/documents/resource-library/market-insights/gannatti-commentary/the-role-of-gold-in-a-portfolio.pdf)  
19. 10-Year Real Interest Rate | ALFRED | St. Louis Fed, accessed April 4, 2026, [https://alfred.stlouisfed.org/series?seid=REAINTRATREARAT10Y\&utm\_source=series\_page\&utm\_medium=related\_content\&utm\_term=related\_resources\&utm\_campaign=alfred](https://alfred.stlouisfed.org/series?seid=REAINTRATREARAT10Y&utm_source=series_page&utm_medium=related_content&utm_term=related_resources&utm_campaign=alfred)  
20. Moving 12-Month Total Vehicle Miles Traveled | FRED | St. Louis Fed, accessed April 4, 2026, [https://fred.stlouisfed.org/graph/?chart\_type=line\&height=600\&s%5B1%5D%5Bid%5D=M12MTVUSM227NFWA\&s%5B1%5D%5Brange%5D=max\&s%5B1%5D%5Bscale%5D=Left\&s%5B2%5D%5Bid%5D=RSGASS\&s%5B2%5D%5Brange%5D=max\&s%5B2%5D%5Bscale%5D=Right](https://fred.stlouisfed.org/graph/?chart_type=line&height=600&s%5B1%5D%5Bid%5D=M12MTVUSM227NFWA&s%5B1%5D%5Brange%5D=max&s%5B1%5D%5Bscale%5D=Left&s%5B2%5D%5Bid%5D=RSGASS&s%5B2%5D%5Brange%5D=max&s%5B2%5D%5Bscale%5D=Right)  
21. Relative Valuation of U.S. Insurance Companies \- Columbia University, accessed April 4, 2026, [http://www.columbia.edu/\~dn75/Relative%20Valuation%20of%20U.S.%20Insurance%20Companies%20-%20Nissim%20-%2012-20-2011.pdf](http://www.columbia.edu/~dn75/Relative%20Valuation%20of%20U.S.%20Insurance%20Companies%20-%20Nissim%20-%2012-20-2011.pdf)  
22. The Progressive Corporation (PGR): Business Model Canvas \[Dec-2025 Updated\], accessed April 4, 2026, [https://dcf-model.com/products/pgr-business-model-canvas](https://dcf-model.com/products/pgr-business-model-canvas)  
23. THE PROGRESSIVE CORPORATION 2022 ANNUAL REPORT, accessed April 4, 2026, [https://www.progressive.com/content/pdf/art/2022-annual-report.pdf](https://www.progressive.com/content/pdf/art/2022-annual-report.pdf)  
24. Producer Price Index by Industry: Legal Services (PCU54115411) | FRED | St. Louis Fed, accessed April 4, 2026, [https://fred.stlouisfed.org/series/PCU54115411](https://fred.stlouisfed.org/series/PCU54115411)  
25. Vehicle Miles Traveled (TRFVOLUSM227NFWA) | FRED | St. Louis Fed, accessed April 4, 2026, [https://fred.stlouisfed.org/series/TRFVOLUSM227NFWA](https://fred.stlouisfed.org/series/TRFVOLUSM227NFWA)  
26. Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis (DGS10) | FRED, accessed April 4, 2026, [https://fred.stlouisfed.org/series/DGS10](https://fred.stlouisfed.org/series/DGS10)  
27. Mortgage-Backed \- Economic Data Series | FRED | St. Louis Fed, accessed April 4, 2026, [https://fred.stlouisfed.org/tags/series?t=mortgage-backed](https://fred.stlouisfed.org/tags/series?t=mortgage-backed)  
28. Treasury and Agency Securities: Mortgage-Backed Securities (MBS), All Commercial Banks (TMBACBW027NBOG) | FRED | St. Louis Fed, accessed April 4, 2026, [https://fred.stlouisfed.org/series/TMBACBW027NBOG](https://fred.stlouisfed.org/series/TMBACBW027NBOG)  
29. INVESTMENT STRATEGY: FACTORS Index Dashboard: S\&P 500® Factor Indices \- S\&P Global, accessed April 4, 2026, [https://www.spglobal.com/spdji/en/documents/performance-reports/dashboard-sp-500-factor.pdf](https://www.spglobal.com/spdji/en/documents/performance-reports/dashboard-sp-500-factor.pdf)  
30. Fiscal Multipliers and Financial Crises \- European Central Bank, accessed April 4, 2026, [https://www.ecb.europa.eu/press/conferences/shared/pdf/20171120\_fiscal\_conference/3a\_paper\_Faria-e-Castro.pdf](https://www.ecb.europa.eu/press/conferences/shared/pdf/20171120_fiscal_conference/3a_paper_Faria-e-Castro.pdf)  
31. Private Equity & Macro Factors \- PGIM, accessed April 4, 2026, [https://www.pgim.com/content/dam/pgim/us/en/pgim-multi-asset-solutions/active/documents/research/PGIM\_Private-Equity\_Macro-Factors.pdf](https://www.pgim.com/content/dam/pgim/us/en/pgim-multi-asset-solutions/active/documents/research/PGIM_Private-Equity_Macro-Factors.pdf)  
32. The Progressive Corporation 2023 Annual Report, accessed April 4, 2026, [https://www.progressive.com/content/pdf/art/2023-annual-report.pdf](https://www.progressive.com/content/pdf/art/2023-annual-report.pdf)  
33. Consumer Price Index for All Urban Consumers: Motor Vehicle Maintenance and Repair in U.S. City Average (CUSR0000SETD) | FRED, accessed April 4, 2026, [https://fred.stlouisfed.org/series/CUSR0000SETD](https://fred.stlouisfed.org/series/CUSR0000SETD)  
34. ANNUAL REPORT 2024 \- Bureau of Transportation Statistics, accessed April 4, 2026, [https://www.bts.gov/sites/bts.dot.gov/files/2025-03/TSAR-2024\_dot\_79039\_DS1.pdf](https://www.bts.gov/sites/bts.dot.gov/files/2025-03/TSAR-2024_dot_79039_DS1.pdf)

[image1]: <data:image/png;base64,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>