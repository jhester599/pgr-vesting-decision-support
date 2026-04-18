# Executive Summary

Technical indicators are transformations of price/volume data that often encapsulate momentum, mean-reversion, or volatility signals.  Our in-depth review finds **limited evidence** that novel technical indicators will reliably improve 6-month relative return forecasts beyond the strong momentum already in the baseline model.  The few well-supported signals at mid-horizons are primarily directional/relative signals that momentum already captures (e.g. relative strength/mean-reversion measures like RSI) or regime filters (e.g. trend strength or volatility bands).  Academic studies show **some predictive power for technical factors in cross-sectional portfolios and fixed-income markets【28†L46-L54】【26†L77-L85】**, but generic technical rules in individual stocks often fail rigorous out-of-sample tests.  In practice, adding dozens of highly correlated TA features risks overfitting.  A prudent approach is to focus on a small number of Tier-1 indicators (momentum oscillators and volatility regimes) with clear theory or prior evidence, and use strict walk-forward validation with multiple-testing controls.  We recommend first testing a few oscillators (RSI, Stochastic) and volatility bands (ATR, Bollinger) on PGR and key benchmarks, using stationary transforms (e.g. RSI centered at 0, ATR/price), and comparing baseline vs. expanded models with Diebold-Mariano or Clark-West tests.  The strongest potential signals are *mean-reversion* oscillators (high RSI or Stoch → likely future underperformance) and *regime filters* (high ADX/trend strength → momentum signals become stronger).  However, **modest/negative expected magnitude skill** (current OOS R² is strongly negative) suggests any found signal may not lift predictive R² much.  We recommend continuing only if a few Tier-1 features yield robust IC > 0.15 or Clark-West t>1.65 on multiple benchmarks.  If not, resources should shift to other ideas.  The single most important methodological guardrail is **strict walk-forward testing with multiple-testing corrections**: only technical features that survive stringent out-of-sample scrutiny (e.g. new monthly data or bootstrap tests) should be considered further.

# 1. Problem Framing and Target Formulation

**Target horizon:**  Classic momentum studies find strongest price predictability at **short to medium horizons** (1–12 months).  Recent evidence suggests that even **one-month lagged returns have positive serial correlation across many markets**【55†L81-L89】.  Traditional stock-market studies find momentum peaks at around 6–12 months (and sometimes reverses beyond a year).  However, **day-to-day technical oscillators (RSI, Stochastic) are usually used for very short horizons**, whereas long horizons (>1 year) tend to shift to fundamentals.  For our 6-month horizon, the signal‐to‐noise is low: if any technical signal is present, it is most likely at shorter horizons (1–3 months) and gradually decays.  **In summary, we expect the richest TA signal at short (monthly) lags, with diminishing power by 6–12 months**; thus we should particularly evaluate 1–3 and 6-month forecasts for evidence of technical persistence, but remain skeptical beyond 6 months.

**Target type:**  Given the model’s poor R² but decent directional IC (~0.19, hit ~69%), directional targets (binary classification or cross-sectional ranking) may be more appropriate for technical features.  *Momentum and oscillator signals often translate into rank/direction signals rather than calibrated magnitude*【28†L46-L54】.  For example, a high RSI typically signals *underperformance* (overbought → mean reversion) rather than a precise return amount.  Therefore, *binary/outperform vs underperform* or *cross-sectional rank* formulations likely align better with TA signals than raw regression of continuous alpha.  We will test both regression and classification, but emphasize ranking accuracy (IC, hit rate) as primary metrics for TA features.  This also helps mitigate calibration error in magnitude forecasts.

**Relative-return definition:**  The key question is whether to compute relative returns as **PGR minus broad market (VOO)**, **PGR minus sector/peer**, or **PGR minus a composite of insurers**.  Broad-market (e.g. S&P 500/VOO) is the standard benchmark and lets us interpret TA signals relative to overall equity momentum.  Sector-level (e.g. an insurance ETF or peer index) would isolate industry cycles, but no dedicated insurance ETF is in the pipeline (we have 4 peers which could form an index).  Intuition suggests: if PGR’s moves are idiosyncratic within the **insurance/financial sector**, relative signals vs peers could be informative.  If macro/market factors dominate PGR, then benchmarking vs VOO or cross-asset ETFs is clearer.  In practice, we can pursue both: e.g. compute TA on *PGR/VOO* ratio and on *PGR minus average of peers*.  However, initial focus should be on the simplest: PGR vs VOO (broad market), since that is defensible and widely used. 

**Combined formulation:**  The “best” combination likely pairs a **directional target** (classification/rank) at a relatively **short horizon** (1–6 months) using a **broad-market relative return**.  This maximizes the chance that pure price signals (momentum vs mean-reversion) show up.  Under this scheme, momentum-like indicators (price vs MA, MACD) would aim to time *continuation*, whereas oscillators (RSI, CCI, Stochastic) target *mean-reversion*.  In particular, one expects that high recent relative strength (e.g. high PGR/VOO ratio or high PGR RSI) might mean *reversion* over the next 6 months.  The relatively shorter horizons and categorical targets play to the strengths of TA, per the literature【55†L81-L89】【62†L347-L356】.  Longer horizons (12+ months) or non-directional targets are likely to wash out TA signals and capture macro/fundamental effects instead.

# 2. Alpha Vantage Indicator Taxonomy

Alpha Vantage provides dozens of built-in technical indicators.  We organize them into economically motivated groups (with examples) and discuss their mechanisms:

- **Trend/Overlap (Moving Averages, Trendlines):**  *Indicators:* SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3, MAMA, HT_TRENDLINE (plus Parabolic SAR).  *Mechanism:* These smooth prices to identify persistent trends.  For example, an asset price consistently above its moving average suggests an uptrend【60†L344-L352】【60†L359-L368】.  The slope of the MA indicates momentum in the trend.  Many crossover rules (e.g. short MA crossing long MA, “golden cross”) are lagging confirmations of trends【60†L374-L383】【60†L389-L398】.  *Typical parameters:* commonly 10–200 days; shorter MAs (10–50d) capture recent trends, longer (100–200d) capture secular trend.  The smoothing effect means these indicators are highly collinear: e.g. SMA and EMA of the same period track each other closely.  KAMA, TEMA, and MAMA are variations with different smoothing (faster adaptation).  *Signal:* These generally align with momentum (trend-following).  For relative performance, one might compare PGR’s MA vs benchmark’s MA.  However, since raw momentum (past return) is already in the model, these risk redundancy.  Overlap indicators tend to produce a *persistent* signal (trend) rather than quick reversal.

- **Momentum Oscillators:**  *Indicators:* RSI, Stochastic (STOCH, STOCHF, STOCHRSI), William %R (WILLR), MACD (including histograms, PPO, APO), ADX/ADXR (with +DI/−DI), Momentum (MOM), ROC/ROCR, CCI, TRIX, Ultimate Oscillator (ULTOSC), Commodity Channel Index (CCI), Chaikin Money Flow (CMO), Aroon/AroonOscillator, Balance of Power (BOP), MFI, Volume/SVI (not listed but VWAP).  *Mechanism:* These capture price changes relative to recent history. RSI, Stoch, CCI, and WillR indicate overbought/oversold conditions (mean-reversion signals).  For example, RSI compares average gains vs losses (higher RSI near 100 means recent strong gains)【62†L347-L356】.  When RSI >70 or stochastic >80, prices are often considered *overbought*, suggesting a reversion down; below 30/20 signals *oversold*, suggesting bounce【62†L347-L356】【62†L375-L383】.  MACD and ROC measure raw momentum (difference or ratio of fast vs slow EMAs or percent change); positive MACD suggests an uptrend, negative suggests downtrend.  ADX measures trend strength (0–100) without direction, often used to *gate* momentum signals【64†L267-L274】.  CCI, TRIX, ULTOSC, MOM, and Aroon also gauge trend momentum or exhaustion.  Many are highly correlated: e.g. RSI, Stochastic, and CCI all capture similar short-term oscillations.  *Typical parameters:* Defaults are often 14 or similar. Sensitivity varies: RSI14, Stochastic K14/D3, MACD(12,26,9), ADX14 are common.  *Signal orientation:* Most oscillators imply *mean-reversion* when extreme (high RSI→ future weakness), whereas MACD/ADX are *trend-following* (e.g. rising MACD → continued outperformance).  For relative returns, oscillators translate into “reversion to mean” signals for PGR vs benchmark, whereas momentum measures suggest continuation.  

- **Volatility/Range:**  *Indicators:* ATR, NATR, True Range (TRANGE), Bollinger Bands (BBANDS).  *Mechanism:* These quantify price variability.  ATR (Average True Range) and True Range measure raw volatility; NATR = ATR/price.  Bollinger Bands construct an envelope around a moving average (typically ±2 standard deviations on 20-day MA【67†L25-L33】).  When bands are wide, volatility is high; narrow bands mean calm markets.  Price touching the upper or lower band indicates an extreme relative to volatility.  For example, a close near the upper Bollinger band suggests an “overbought” move【67†L1-L4】.  *Parameters:* BBANDS often use 20-day MA, 2σ (alpha defaults). ATR/NATR often 14-day.  *Signal:* Volatility itself can imply mean-reversion or risk-aversion regimes.  A breakout beyond bands can signal continuation (volatility expansion) or reversal (mean reversion to mean).  Since volatility is a well-known factor, adding ATR/NATR may partly proxy a volatility risk factor.  These indicators align with regime detection: e.g. high ATR means volatile regime, possibly changing return dynamics; BBANDS capture when price is at extreme quantiles.

- **Volume/Money-Flow:**  *Indicators:* On-Balance Volume (OBV), Accumulation/Distribution (AD), Money Flow Index (MFI), Accumulation/Distribution Oscillator (ADOSC), Chaikin Oscillator, VWAP (intraday).  *Mechanism:* These attempt to incorporate trading volume.  OBV cumulatively adds volume on up-days and subtracts on down-days【69†L330-L339】; it shows whether “smart money” is flowing in (rising OBV) or out (falling OBV).  ADL/ADOSC weigh volume by price positioning (close vs range)【71†L410-L417】.  MFI is like RSI but uses typical price and volume.  These are all related: they track net volume trends.  *Parameters:* MFI commonly 14-day.  ADL/OBV have no “period” parameter.  *Signal:* Theoretically, rising OBV (divergence from price) can forecast price moves. In practice, their predictive value is weak.  They may capture accumulation vs distribution.  For relative returns, a divergence between PGR’s volume indicator and the benchmark’s could hint at relative strength.  However, volume-flow signals tend to align with momentum (accumulation precedes price rises) and are highly correlated with price moves themselves.  

- **Cycle/Hilbert:**  *Indicators:* HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE.  *Mechanism:* These apply Hilbert transform techniques to find dominant cycle periods and phase information.  These are essentially *spectral* indicators, identifying cyclical behavior.  *Parameters:* None beyond default.  *Signal:* Generally designed for short-cycle trading (intraday swings).  There is little evidence these capture anything beyond noise for monthly horizons.  They likely add no unique information for 6-month forecasting, as cycles in daily data rarely align with multi-month moves.

- **Price Transforms:**  *Indicators:* AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE.  *Mechanism:* These compute simple price averages: (high+low+close)/3 etc, or (high+low)/2.  *Signal:* They are simply linear transformations of OHLC price, not adding new information.  They serve as intermediate values for some indicators (e.g. Bollinger). They do not independently provide forecasting signals beyond raw price.

- **Pattern Recognition (Candlesticks):**  *Indicators:* CDL* (e.g. CDL3INSIDE, CDLHAMMER, plus ~60 other candlestick patterns).  *Mechanism:* Each identifies a specific candlestick formation (e.g. “hammer” or “doji”) which some traders interpret as reversal signals.  *Signal:* These are short-lived, often spurious signals. The large number of patterns introduces massive data-snooping risk. There is no credible evidence that monthly counts of such patterns reliably predict 6-month returns. They essentially encode short-term price patterns rather than enduring signals.

**Redundancies and Collinearity:**  Many indicators overlap heavily.  For example, RSI, Stochastic, WillR, and CCI all measure price relative to recent range and tend to co-move.  MACD, PPO, and momentum (MOM/ROC) are all momentum proxies (linear vs log scale).  ATR, NATR and True Range are closely related (NATR = ATR/price).  DMI/ADX and their components (+DI/−DI) are derived together.  Price-transform indicators are trivial linear scalings of price. In short, most TA features are different *representations* of the same underlying price action (momentum vs reversal vs volatility).  We must be cautious that the regression model not treat each as independent signal. 

**Trend vs Mean-Reversion Mechanisms:**  Roughly, moving averages, MACD, and ADX (with DI lines) are *trend-following* – they tend to predict continuation.  Oscillators like RSI, Stochastic, CCI, WillR, and Bollinger act as *mean-reversion* signals (extreme values mean revert).  ATR/BBAND width are regime measures: e.g. extremely low Bollinger bandwidth sometimes precedes breakouts (volatility regimes).  Volume indicators (OBV, ADL) implicitly suggest momentum (volume divergence can foreshadow trend) but can also signal divergence setups.  Most TA signals were originally conceived for daily swing-trading; their sign at a medium (6-mo) horizon depends on whether short-term “exhaustion” leads to multi-month reversals or if trends persist beyond usual horizons.  

# 3. Literature Synthesis

**(A) Well-supported findings:**  Recent out-of-sample research yields **mixed results, but some encouraging signals in cross-sectional contexts**.  Notably, Zeng et al. (2022) show that a set of common technical indicators *collectively* explain cross-sectional stock returns beyond traditional factors【28†L46-L54】.  Their 14-indicator combination yields smaller estimation errors than Fama-French, and long-short portfolios of their signals “generate substantial profits” consistently【28†L46-L54】.  This suggests that, *at least cross-sectionally*, technical features contain independent information.  Similarly, Chin et al. (2025) find that technical indicators significantly predict **corporate bond returns** out-of-sample, often outperforming standard bond fundamentals【26†L77-L85】.  In that study, machine-learning yielded no better results than a linear model – implying that any nonlinearity in TA signals might be modest.  Moreover, empirical factor studies show strong **momentum effects** in stocks at 6–12 months (Jegadeesh-Titman (1993)), and even one-month momentum appears pervasive across many asset classes【55†L81-L89】.  Short-term reversal is typically found in highly liquid individual equities (contrary to the cross-asset momentum found by Zaremba et al.【55†L81-L89】).  Overall, the **biggest chunk of robust evidence** is for *relative momentum* (stocks that recently outperformed continue to do so) and *volatility/momentum regimes*.  These align with testing oscillators and trend filters.

**(B) Weak or inconsistent findings:**  Conversely, *many published “technical analyses” fail out-of-sample*.  Classic studies (e.g. Brock, Lakonishok, LeBaron 1992) reported in-sample success for moving-average and trading-range rules, but later work with reality-check bootstraps (Sullivan et al. 2001) found that *most simple trading rules do not survive data-snooping adjustments*.  Harvey, Liu, and Zhu (2016) warn that numerous “anomalies” (including technical) may be artifacts of mining.  In equity returns, pure technical signals have **no consensus** for reliably predicting multi-month returns outside of momentum.  For example, most evidence for oscillators (RSI, CCI) is anecdotal or local; academic replication of mean-reversion (beyond the known cross-stock reversal at multi-year lags) is weak.  In fact, Zaremba et al. explicitly note that their result (“short-term momentum everywhere”) is *contrary to traditional stock-level evidence of short-term reversal*【55†L81-L89】.  In sum, *isolated trading signals often underperform once transaction costs and robust testing are applied*.  

**(C) Proxies and redundancy:**  Many apparent “signals” are just proxies for known factors.  For instance, **momentum oscillators** (RSI, Stochastic, CCI, MACD) largely capture variations of the price momentum factor.  If we already include raw 6-month price momentum in the model, an indicator like MACD or ROC is likely collinear.  Studies (e.g. Zeng 2022) suggest that these cannot be fully explained by simple momentum, but the incremental gain beyond momentum is usually small.  Similarly, **volatility indicators** (ATR, Bollinger, VIX) often proxy for a volatility risk factor or regime, which may not be new information if we already have VIX/volatility index in the model.  So far, **no strong evidence** exists that, say, ATR provides independent long-horizon information once volatility is accounted for.  

**Cross-sectional vs. time-series:**  Technical indicators seem to have **stronger backing in cross-sectional stock selection** than in time-series market timing.  Zeng (2022) explicitly focused on explaining *cross-sectional expected returns* with TA, independent of market factors【28†L46-L54】.  This suggests TA features may be better at *ranking stocks relative to each other* (i.e. picking winners/losers) than at predicting aggregate market moves.  Our use-case (PGR vs benchmarks) is partly cross-sectional (stock vs ETFs), but also partly market timing (forecasting PGR’s alpha relative to asset classes).  Thus, we should not expect TA’s best utility to be on overall market timing – rather, if it adds, it may come from relative strength insights. 

**Relative returns:**  Direct evidence that TA predicts *relative* stock–benchmark returns is scarce.  Most literature looks at absolute or cross-sectional returns.  One might infer from cross-asset momentum studies that when an asset class has momentum, equities vs bonds, etc., we could see relative patterns.  For example, if gold has momentum (GLD uptrend), investors might shift from stocks (PGR underperform).  However, this is highly speculative.  One somewhat relevant angle is *pairs trading*: traders often compute technical signals on the **price ratio** of two securities (akin to option A in Section 5).  Some institutional strategies do this (e.g. long one stock/short another when their ratio hits extremes), but academic literature on it is limited.  We found no academic papers explicitly testing PGR vs S&P or vs sector technical signals.  Thus, our approach to relative returns will be largely heuristics guided by theory rather than prior results.

**Nonlinear models:**  The incremental evidence (Zeng 2022; Chin 2025) suggests that a linear or simple model can extract most of the value of TA features.  In Chin et al., machine-learning offered little edge over linear regression【26†L77-L85】.  In practice, boosting/Trees may find complex interactions (e.g. momentum * trend-strength).  But given our small sample, any nonlinear fit risks overfitting.  We should test both ridge (linear) and GBT, but treat GBT gains with skepticism.  If any deep nonlinear pattern exists (e.g. a high-degree interaction), it must be strong to persist out-of-sample.  

**Regime/market dependence:**  TA usefulness likely depends on market regime.  For instance, short-term momentum seems stronger during turbulent periods with high cross-sectional dispersion【55†L83-L89】.  Conversely, in very tranquil or mean-reverting markets, TA signals die.  PGR (an insurance stock) may be influenced by interest-rate and credit cycles.  If bond yields are trending up, perhaps one expects financial stocks like insurers to underperform (since their liabilities change).  If such macro trends are captured by bond market technical signals, that could incidentally predict PGR vs bond ETF.  However, **we lack conclusive evidence** that financial-sector stocks respond predictably to technical signals on macro asset classes.  Therefore, regime dependence is an open question; our tests must explicitly check sub-periods (pre-2020/2020/2021-22 etc.) to guard against spurious fits.

**Decay and data-mining:**  All TA-based predictive relationships are susceptible to decay under data-snooping.  The body of published technical trading rules is enormous, and most likely fail robust out-of-sample tests.  For every positive report, there are many unpublished failures.  Given this, any apparent TA signal we find must be validated out-of-sample.  We should assume a prior that *true TA alpha is low* (e.g. Holden & Peel 1999).  Only if multiple tests (e.g. rolling windows, new data) confirm a signal do we consider it real.  In practice, this means requiring that *several* Tier-1 indicators show consistent significance (not just one lucky result).

**Sector-specific evidence:**  We found almost **no literature on technical analysis of insurance stocks or P&C insurers**.  Insurers have relatively stable cash flows but low beta.  Conventional wisdom: low-beta or dividend stocks often show *momentum anomalies similar to broader market*, but no known twist unique to insurers.  If anything, they might be slightly mean-reverting (as slower-moving stocks) – but this is speculative.  Absent guidance, the best analogy is financial stocks.  Studies of financial-sector momentum vs economy may hint at interaction with interest rates.  But ultimately, we must treat this as a new area: our tests on PGR will effectively be *prospective tests* with no guide beyond general TA theory.

**Summary of findings:**  Technical indicators can sometimes forecast returns, particularly in cross-sectional stock selection and bond markets【28†L46-L54】【26†L77-L85】.  However, most TA rules in isolation have historically failed rigorous out-of-sample tests, and many signals simply proxy momentum or volatility factors.  We must therefore be highly selective: focus on those signals for which there is at least *some* prior out-of-sample support or strong theoretical rationale, and treat any discovered effect very cautiously.  

# 4. Tiered Indicator Ranking

We classify each indicator from Section 2 into **Tier 1 (strong evidence)**, **Tier 2 (weak/plausible)**, or **Tier 3 (unlikely)** for 6-month relative-return prediction.  We list expected signal direction, evidence type, and assets where applicable:

| **Indicator**         | **Tier** | **Expected Signal (PGR)**                | **Stronger For** | **Assets/Sectors**           |
|-----------------------|---------:|------------------------------------------|------------------|------------------------------|
| RSI (14-day)          | **1**    | Overbought (RSI↑) → underperformance (neg. rel. ret)【62†L347-L356】 | Relative (mean-reversion) | Stocks, broader markets (works on equities) |
| Bollinger %B (20,2σ)  | **1**    | PGR near upper band → underperformance; near lower band → outperformance【67†L1-L4】 | Relative (volatility filter) | Stocks, commodities (vol regimes) |
| ATR (14-day NATR)     | **1**    | High ATR → volatile regime → possible mean-reversion or *higher drawdown risk* | Absolute (volatility measure) | All asset classes (volatility regimes) |
| ADX (14-day)          | **1**    | High ADX (trend strong) → momentum signals reinforced; Low ADX → flat → weaker signals【64†L267-L274】 | Both (tuning momentum) | Stocks/bonds when trending strongly |
| (MACD Histogram)      | **2**    | Positive MACD → continued outperformance; negative → underperformance | Absolute (momentum) | Stocks, indices (broad momentum) |
| Stochastic %K (14)    | **2**    | Overbought >80 → underperformance; oversold <20 → outperformance【62†L375-L383】 | Relative (mean-reversion) | Stocks (range-bound regimes) |
| CCI (20)              | **2**    | High CCI (>100) → underperformance; low (<–100) → outperformance | Similar to RSI/Stoch | Stocks, commodities |
| WillR (14)            | **2**    | WillR near 0 (overbought) → underperformance; near –100 → outperformance | Similar to RSI | Stocks |
| MACD Signal line/Hist | **2**    | (If cross above signal) bullish continuation | Absolute (trend) | Stocks, indices |
| PPO (12,26)           | **2**    | Like MACD but normalized by price | Absolute (trend) | Stocks, indices |
| Momentum (MOM)        | **2**    | Positive momentum → continued outperformance | Absolute (raw momentum) | Stocks (redundant with baseline) |
| ROC (14)              | **2**    | Similarly, positive ROC → outperformance | Absolute (raw momentum) | Stocks |
| Aroon / AroonOsc      | **3**    | (Aroon Up above Aroon Down) → uptrend, etc. | Absolute/Relative (trend) | Stocks (limited evidence) |
| MFI (14)              | **3**    | High MFI (overbought) → underperformance; low (oversold) → outperformance | Similar to RSI | Stocks (volume-weighted) |
| OBV / AD / ADOSC      | **3**    | Rising OBV → bullish; falling → bearish | Absolute (volume momentum) | Stocks (weak evidence) |
| VWAP (daily)          | **3**    | (Typically intraday; not applicable monthly) | — | — |
| HT_* (cycle indicators)| **3**   | N/A (intended for intraday cycles; no evidence at 6M) | — | — |
| Price Transforms      | **3**    | (Trivial; no signal beyond raw price) | — | — |
| Candlestick patterns  | **3**    | (No reliable monthly predictive power) | — | — |

**Notes:** Tier-1 indicators are those most justified to test first.  RSI and Stochastic capture mean-reversion (“overbought→sell”)【62†L347-L356】【62†L375-L383】, BBands capture volatility extremes【67†L1-L4】, and ATR/ADX signal regimes of volatility/trend【64†L267-L274】.  These have at least some theoretical/empirical support.  Tier-2 indicators (MACD, PPO, CCI, WillR, ROC) are plausible momentum signals or variations of Tier-1, but with weaker specific evidence at our horizon.  Tier-3 (candlestick, cycle, price-transforms, raw volume) are deprioritized.  

**Direction and assets:**  For Tier-1 and 2, we expect *mean-reverting* signals (RSI↑ ⇒ α↓; Stoch high ⇒ α↓) and *trend-following* signals (MACD up ⇒ α↑), depending on context.  Evidence is generally stronger for *absolute returns* than strictly relative, but since we do relative forecasts, we interpret them as relative signals (e.g. “PGR overbought relative to index → PGR underperforms index”).  Many signals are cross-market (e.g. Bollinger, ADX) and likely work similarly across sectors, though certain assets (energy, commodities) have their own vol patterns.  We highlight that **technical evidence is often strongest in equity indexes and commodity markets** (per [55], [66]), rather than in low-vol stocks.  However, we have no PGR-specific studies, so we assume broad applicability across our benchmarks. 

# 5. Absolute vs. Relative Signal Construction

We consider four approaches to using TA in a relative-return model:

- **Option A – Indicator on the price ratio:**  Compute technical indicators on the ratio (or difference) of PGR price to benchmark price (e.g. RSI of PGR/VOO).  *Justification:* This directly measures PGR’s strength vs. the benchmark’s performance. It is conceptually similar to pairs trading.  If PGR has gained relative to VOO, a high RSI on the ratio would indicate *relative overbought*, potentially forecasting a drop in PGR/VOO.  *Evidence:* This specific approach is rarely studied, but one can argue it isolates purely relative movements.  It essentially treats the ratio like a “synthetic stock”.  However, technical rules on ratios can be unstable because the ratio’s volatility and distribution differ from a single stock.  Empirically, such a construction is untested in academia as far as we know.  We should experiment with it (especially for RSI or MACD) as it may capture relative momentum.  

- **Option B – Difference of indicators:**  Compute the indicator separately on PGR and on the benchmark, then form the difference.  E.g. `RSI_diff = RSI(PGR) – RSI(VOO)`.  *Justification:* This is a simpler analog to Option A.  A large positive `RSI_diff` means PGR’s recent momentum exceeds VOO’s, suggesting potential mean-reversion (PGR overpriced relative to benchmark).  *Evidence:* This approach is logically similar to ratio, but it assumes linearity of the indicator.  It's more heuristic; no direct literature comparison.  However, it has been used in some trading systems (e.g. RSI spread strategies).  We expect it to be noisy but worth testing alongside Option A.  

- **Option C – PGR-only indicators:**  Compute indicators on PGR alone, ignoring benchmark.  *Justification:* If the benchmark is relatively stable, PGR’s absolute technical signals may still correlate with relative performance.  For example, if PGR enters an upswing, it may outperform any slower-moving benchmark.  Many models simply use stock-only indicators even for relative forecasts.  *Evidence:* This is the simplest to implement, but it risks confounding overall market moves (e.g. if all stocks rally, a bullish PGR-only signal might just reflect market bullishness).  We should always include market factors (which we do in baseline) to account for that.  PGR-only TA can be useful especially if the benchmark is bond or commodity (an entirely different asset class).  

- **Option D – Benchmark/ETF technical signals (regime signals):**  Use indicators computed on benchmark ETFs as *market/regime features*.  For example, calculate `ADX(VOO)`, `RSI(GLD)`, etc.  *Justification:* Technical conditions in major asset classes may signal rotation opportunities.  For instance, if bond market (BND) shows a strong downtrend (e.g. rising yields), that might historically coincide with financial stocks (like insurers) outperforming fixed income.  Or if GLD has strong momentum, capital may be flowing to safe havens, implying *PGR likely to underperform equities*.  *Evidence:* There is no specific academic proof for these cross-asset TA signals, but the logic is akin to momentum in risk-on vs risk-off regimes.  Some practitioner strategies use equity/bond momentum for asset allocation.  We should test whether e.g. `ADX(BND)` or `RSI(GLD)` interacts with PGR’s relative return.  This approach effectively treats TA on benchmarks as proxies for macro regimes (like interest-rate trends).  

- **Peer-relative signals:**  Compute TA indicators on PGR’s insurance peers (ALL, TRV, CB, HIG) to gauge sector strength.  *Justification:* If the insurance sector as a whole is in a strong trend (or overheated), PGR may follow.  E.g. high RSI on an average of ALL/TRV means sector overbought → PGR maybe underperforms.  *Evidence:* This is a form of cross-sectional momentum within a sector.  Very little literature, but it is intuitive: industries often move together.  Practically, we can create a composite peer index (e.g. average price of peers) and compute TA on that (or differences).  We’ll test key indicators like RSI and MACD on the peer average.  

In summary, we will experiment with all four approaches: (A) and (B) for direct relative measures, (C) as a baseline check, (D) and peer signals as additional context.  There is **no clear verdict from literature**, so our validation will determine which yields incremental signal. 

# 6. Frequency Adaptation and Monthly Aggregation

Most TA indicators are defined on high-frequency data, but our model uses **monthly features**.  We must aggregate or adapt them:

- **Time-period scaling:**  An indicator’s `time_period` parameter should reflect the horizon of interest.  For a 6-month forward return (~180 trading days), trend indicators should span a similar scale.  For example, a 200-day SMA covers ~10 months.  We should consider using longer windows than daily defaults.  **Recommendations:** 
  - Oscillators (RSI/Stoch): often default to 14 days; to capture monthly conditions, we might use 14–21 days (3–4 weeks) or even 63 days for slower signals.  We will test both short (14) and medium (63) periods. 
  - Bollinger Bands / ATR: default 14–20.  For medium-term, 20–63 might be used.  A 20d BBand captures ~1m volatility, 63d ~3m.
  - ADX: default 14 (1m).  We may test 14 vs 40 (2m) to smooth noise. 
  - MACD: standard 12/26/9 corresponds to weeks; we could also test longer (e.g. 26/52/9 for month vs quarter).
  
  In practice, we can treat `time_period = 20` as roughly monthly, `63` as quarter, `125` as half-year.  We will start with standard values (14,20,50) and refine after initial tests.

- **Aggregation methods (monthly):**  Once we compute daily indicator values, we need a single monthly feature.  Common choices:
  - *End-of-month snapshot:* simply take the indicator’s value on the last trading day of the month.  This is easy and retains the indicator’s meaning.  It is the approach we will default to.  
  - *Trailing-month average or median:* smooth out daily noise by averaging the indicator over the last 20 trading days.  This could help if the indicator is volatile.  We will compare this for more stable signals like MACD or ATR.
  - *Fraction of days in state:* for binary-like features (e.g. RSI>70), we could record what fraction of days in the month were overbought.  This is speculative; likely we just use end-of-month levels.  
  - *Sign of change:* for some indicators (like MACD histogram), maybe the change from previous month is used (e.g. positive → buy).  We will focus on levels first.
  
  **Specific suggestions:** 
  - **Oscillators (RSI, Stoch, CCI, WillR):** Use the end-of-month value (scaled to [–1,1]).  This captures the current overbought/oversold reading.  Alternatively, a *month-average RSI* might reduce noise, but loses the precise threshold interpretation.
  - **Trend/Overlap (SMA crossover):** We may use *binary flags* (e.g. “price > SMA”) or the distance `(Price – SMA)/SMA`.  For monthly, a sensible feature is `(EOM Price – EOM SMA) / EOM SMA`.  We will implement at least one such measure for a Tier-1 moving average (e.g. 63-day SMA).
  - **ATR/Volatility:**  Use the end-of-month ATR or NATR.  Possibly also track change in ATR (volatility momentum).
  - **Bollinger Bands:**  Use %B at month’s end: `(Close – LowerBand)/(UpperBand – LowerBand)`.  This is already bounded [0,1].  
  - **ADX:**  Use EOM ADX (strength 0–100) or normalized to –1..1.  
  - **Volume (OBV, AD, etc):**  Compute cumulative changes over the month as features (OBV_delta = OBV_end – OBV_start).  But likely unhelpful.
  
  **Candlestick patterns:**  If used at all, one could count how many bullish vs bearish patterns occurred during the month.  There is no evidence this helps at 6-month horizons.  We will not prioritize these.

Overall, we prefer **end-of-month values** for most indicators, as they best preserve the intended signal.  Where appropriate, we will test alternatives (e.g. average or count) if end-of-month seems too noisy for a given indicator.

# 7. Stationarity Transformations and Feature Engineering

For ridge regression, features must be stationary and comparable in scale.  We propose the following transforms for each Tier 1/2 indicator:

- **RSI (0–100):**  Already bounded.  We center and scale to ±1: e.g. `RSI_stat = (RSI - 50) / 25`.  This maps RSI=75 → +1, 25 → –1【62†L347-L356】.

- **Stochastic %K (0–100):**  Similar to RSI.  `Stoch_stat = (StochK - 50) / 25`.  (High Stoch → positive, but since Stoch is overbought at 80, we may flip sign in use.)

- **Williams %R (–100 to 0):**  We first convert to 0–100 scale (`100 + WillR`).  Then use `(WillR_adj - 50)/25`.

- **MACD Histogram:**  Value in price units.  Normalize by price: `MACD_h_stat = MACD_hist / Close`.  This removes level-dependence.

- **PPO (Price Oscillator):**  Already normalized by price (percent).  We can center: `PPO_stat = PPO / 100` (since PPO is usually in percent).

- **CCI (Commodity Channel):**  Typically unbounded (±100).  We can scale by a nominal bound, say 100: `CCI_stat = CCI / 100` (so ±1 corresponds to ±100).

- **Momentum (MOM):**  Percentage or price diff.  Use percent (e.g. 6M-return).  Already in model.

- **ADX (0–100):**  Center at 50: `ADX_stat = (ADX - 50) / 50`.  This maps strong trend (ADX>50) to positive.  

- **Bollinger %B:**  Already 0–1.  We use `BB_pcent = %B` (or center at 0.5: `(B-0.5)/0.5` if desired).  Alternatively, distance from 50% band: `(Close - MidMA)/(Upper-Lower)`.  We prefer %B as given.

- **ATR/NATR:**  Use `NATR = ATR / Close`.  Already dimensionless (percent volatility).  This equals nATR endpoint of Alpha.

- **Volume indicators (OBV, AD):**  These accumulate, so not stationary.  We can take monthly change and divide by price or z-score.  E.g. `OBV_stat = ΔOBV / AvgVolume`.  We suspect these add noise; at best we can include normalized OBV change.

- **Price vs MA:**  If used, use `(Close - SMA_n)/SMA_n`.

**Normalization across assets/time:**  Since our model is time-series, we do *per-asset* transforms (as above).  Cross-sectional normalization is less relevant because we predict one asset’s relative return at a time.  However, we might apply a rolling z-score (demeaning) for some indicators to track deviation from recent norm (e.g. RSI_z = (RSI - mean(RSI_past60d))/std(RSI_past60d)).  This can help if indicator levels drift.  We will experiment with a few z-score variants for the most promising features.

**Indicator momentum (slope):**  We will also include **first differences** (`Δindicator`) for key TA features (RSI, MACD, SMA gap).  Often, the *change* in an oscillator may signal emerging momentum shift.  However, given small N, we will first test levels only and then consider adding Δterms if any signal is weak.  The literature does not strongly favor slope over level for multi-month forecasting, but it is trivial to include if warranted.

**Binary thresholds:**  Instead of continuous RSI, one could use binary overbought/oversold flags (RSI>70, <30).  With small data, binarizing can stabilize feature effect.  We will test simple threshold flags for Tier-1 oscillators (RSI, Stoch) as a sensitivity check.  There is no strong evidence this beats continuous values, but it might reduce noise for linear models.

**Regime interactions:**  We will test interactions between key TA indicators and regime variables (VIX, yield slope).  For example, **ADX × momentum** or RSI × volatility.  There is conceptual support (ADX gating momentum), but given our sample size, we will add interactions sparingly (only if they add big lift).  

**Redundancy handling:**  Given high collinearity among indicators, we will group them by latent factor (e.g. “oscillator factor” vs “trend factor”) and use correlation/principal components to prune.  For instance, if RSI, Stochastic, and WillR are 90% correlated, we might include only RSI.  We could apply PCA or hierarchical clustering to identify clusters of correlated TA features.  A pre-screen at correlation >0.9 can drop redundant variables.  This is crucial to avoid multicollinearity in ridge regression.

# 8. Testable Hypotheses

We translate insights into explicit, prioritized hypotheses about PGR’s relative returns.  Each hypothesis states the indicator, expected effect, target formulation, and support strength:

1. **RSI Mean Reversion:** *High RSI on PGR (relative to benchmark) predicts underperformance.*  
   - *Indicator:* RSI (or relative RSI_diff).  
   - *Expected:* RSI_PGR >70 ⇒ PGR will *underperform* the benchmark over the next 6 months (and RSI <30 ⇒ outperformance).  
   - *Target:* Classification (outperform vs underperform PGR vs VOO).  
   - *Support:* Strong theoretical mean-reversion; RSI is a classic oscillator【62†L347-L356】.  Zeng et al. (2022) and practitioners note RSI signals in cross-section.  No PGR-specific test exists.

2. **Trend-Strength Gating:** *ADX on PGR gates momentum usefulness.*  
   - *Indicator:* ADX (PGR).  
   - *Mechanism:* When ADX > 25–30, PGR’s trend is strong, so traditional momentum (past returns) is more predictive; when ADX is low, momentum signals fade.  We hypothesize an interaction: ADX × 6M momentum >0 predicts extra alpha.  
   - *Target:* Regression (alpha) or classification (only consider momentum signal when ADX high).  
   - *Support:* ADX is known measure of trend strength【64†L267-L274】.  This is speculative but plausible: a strong trending regime is needed for technical momentum to matter.

3. **Bollinger Over-Extension:** *PGR near upper Bollinger band vs bond index predicts mean-reversion.*  
   - *Indicator:* %B(PGR vs BND) – Bollinger band position on ratio PGR/BND.  
   - *Expected:* PGR/BND ratio touching upper band → PGR underperforms BND over 6m.  
   - *Target:* Classification (PGR relative to BND).  
   - *Support:* If PGR has rallied far relative to bonds (upper band), a pullback is likely.  Bollinger theory says ~95% prices stay within ±2σ【67†L1-L4】.  We tailor it to PGR vs BND (reflecting bond-vs-equity regimes).

4. **Cross-Asset Momentum:** *Positive momentum in broad market ETFs enhances PGR’s relative strength.*  
   - *Indicator:* MACD or RSI on VOO (or a 3-asset composite of VOO, VXUS, VDE).  
   - *Expected:* Strong uptrend in equity ETF(s) (e.g. MACD(VOO)>0) → PGR more likely to outperform (risk-on).  
   - *Target:* Both regression and classification (PGR vs equity benchmarks).  
   - *Support:* No direct study, but intuitively, if the equity asset class is trending up, individual stocks like PGR are likely to outperform bond/commodity benchmarks.  This tests Option D above.  

5. **Sector Co-movement:** *PGR’s peers’ technical signals predict relative performance vs diversified benchmarks.*  
   - *Indicator:* Average RSI or MACD of ALL, TRV, CB, HIG.  
   - *Expected:* If peer-average RSI is high (sector overbought) → PGR underperforms broad market; if low, PGR outperforms.  
   - *Target:* Classification (PGR vs VOO/VXUS).  
   - *Support:* Sector momentum is a known factor (stocks often move in industry clusters).  This is heuristic (Option D/Peer).  

6. **Momentum Oscillator Decay:** *Mean-reversion for high momentum readings beyond short-term.*  
   - *Indicator:* 3-month momentum (or 1M return) on PGR.  
   - *Expected:* If PGR’s 3M return is very high, its *6M* return is relatively lower (over-reversal).  
   - *Target:* Regression (6M alpha) and classification.  
   - *Support:* The momentum “gap” literature (e.g. Novy-Marx, Moskowitz) shows very strong winners sometimes reverse slightly in later months (momentum decay).  This is more generic and partially covered by momentum feature; included for completeness.

7. **Volatility Breakouts:** *Sharp increase in volatility signals regime change.*  
   - *Indicator:* A jump in ATR or BB width for PGR.  
   - *Expected:* A sudden ATR spike (e.g. >mean+2σ) predicts either continued trend (if momentum remains) or a reversal (if it's a crash).  We test both possibilities via interactions (ATR × momentum).  
   - *Target:* Regression.  
   - *Support:* High ATR often coincides with large news events; direction uncertain.  Not strongly backed but included as possible risk filter.

**Ranking:**  Hypotheses 1–3 are highest priority: they rely on well-known mechanics (RSI reversion, trend strength gating, Bollinger breakpoints). Hypothesis 4–5 (macro and peer signals) are plausible but more speculative. Hypothesis 6 is weak (basically raw momentum effect, already partially in baseline). Hypothesis 7 is the least specific (volatility jump).

Each hypothesis should be tested in the WFO framework, comparing baseline vs. baseline+indicator.  We will note that none of these have been explicitly validated on PGR/insurance stocks, so any positive result will be new evidence.

# 9. Evaluation and Validation Framework

To credibly assess incremental TA value, we propose:

- **Baseline model:**  Ridge regression (and GBT) using the current 12–13 features (6M momentum, 63d vol, yield slope, real rate, credit spread, NFCI, VIX, combined ratio, NPW growth, book value growth, invest income growth).  This is the proper benchmark.  All TA tests must show improvement *over this baseline*.

- **Test design:**  We will adopt a strict walk-forward (rolling window) approach (same as baseline).  For each new TA feature, we compare two models at each fold: (A) Baseline only vs. (B) Baseline + TA feature.  We will record the difference in out-of-sample forecast error (or IC/hit rate) fold-by-fold.  This is essentially an A/B test.  We must avoid any peeking or selection on test folds.  

- **Statistical tests:**  Given overlapping 6-month returns, simple t-tests are invalid.  We will use *Diebold-Mariano* (DM) test with Newey-West adjustments (accounting for overlap) to compare predictive accuracy between models【55†L81-L89】.  Alternatively, the *Clark-West* (CW) test for nested models can gauge if adding TA significantly improves mean squared error.  For directional skill, we can use Wilcoxon or Sign tests on hit rate, and Newey-West corrected *information coefficient (IC) t-statistics*.  We may also compute *certainty-equivalent gains* for a simple long-short strategy.  All p-values will be adjusted for time-series dependence.

- **Multiple-testing control:**  As we will screen many indicators, we guard against false positives.  We will employ a false discovery rate (FDR) approach (Benjamini–Hochberg) on p-values of incremental tests.  Given possible indicator correlations, we may use the more conservative Benjamini–Yekutieli method.  We will also pre-register a shortlist of Tier-1 tests (limiting ~5–10) to reduce multiple comparisons.  If many indicators are screened, we may first apply a theoretical prior filter (as in our tiers) and only formally test those few.  

- **Metrics:**  Key metrics include OOS $R^2$ (pooled over time and benchmarks), Newey–West IC (and its t-stat), directional hit rate, and area under ROC (for classification).  We will compute model *CW t-statistic* for MSE difference, and *DM t-statistic* for loss-difference.  For classification, we’ll report balanced accuracy and the Clark-West analog for logit (if feasible).  The primary gate will be *IC t-stat* and *CW t-stat*: a new TA feature must consistently generate t-stat > ~1.65 (one-sided 5% level) across several folds/benchmarks to be considered genuine.  We may also set a threshold on policy gain (e.g. requiring >5% certainty-equivalent improvement) if trading application is intended.

- **Stop/continue rule:**  We will stop adding new features to the main model if *no TA indicator meets significance criteria*.  Concretely, if after testing all Tier-1 features, **none produce a Clark-West t-statistic >1.65 on at least 3 of the 8 benchmark targets**, and no IC t-stat >1.65, we conclude lack of robust signal and halt.  If a few pass (e.g. RSI passes for VOO/VMBS/GLD), we continue to Phase 3.  We set this high bar to avoid chasing noise.

- **Regime robustness:**  We will explicitly segment performance by sub-periods: pre-COVID (pre-2020), mid-COVID (2020–21), post-2021.  We can include a regime dummy or run separate WFO on each era.  This ensures a purported TA signal isn’t driven by one extreme period (e.g. 2020 crash).

In sum, we require **consistent out-of-sample improvement, assessed with time-series-aware tests, and controlled for multiple trials**.  Only features satisfying these stringent criteria should be considered real.

# 10. Recommended Model Sequence

Given our goals and small sample, we propose the following sequence:

1. **Penalized Linear Model (Ridge/ElasticNet):**  Start with a ridge regression including only Tier-1 indicators (after pre-filtering, plus baseline features).  Ridge is interpretable (coefficients) and handles multicollinearity.  It gives a baseline for how much linear signal exists in TA features.  Because N≈180 and feature count maybe ~10, ridge should suffice to prevent overfit.  We can examine coefficient magnitudes to identify signal vs noise.  **Vulnerability:** if we throw too many features (20+), Ridge may dilute signals (shrinking them) – but still safer than unpenalized.  We'll control complexity by only adding a handful at a time.

2. **Gradient-Boosted Trees (GBT):**  Next, use a simple GBT (depth ~2–3).  GBT can capture nonlinear interactions (e.g. RSI crossing a threshold, or RSI combined with VIX level).  We will train GBT on the same WFO splits, adding one or two TA features at a time.  **Interpretability:** Partial dependence and SHAP analysis can reveal non-linear patterns.  **Caution:** GBT can easily overfit with so few data points.  We'll use shallow trees and limited boosting iterations, and drop features that show no interaction effect.  We will not rely on GBT for final signals unless effects are large and consistent.

3. **Firth-Penalized Logistic Regression:**  For classification (outperform/underperform), we use penalized logistic (Firth regression avoids complete separation with small N).  This will use the same features as the linear model.  It is more tailored to directional hit-rate metric.  **Ridge vs Lasso:** We prefer ridge penalty to allow small contributions from multiple indicators.  Lasso could be tried as feature selector but risk is unstable paths given few obs.

**Feature selection vs regularization:**  Given extreme multiple-testing risk, we lean toward **theory-driven pre-selection** rather than brute-regularization of dozens.  While in principle ridge can “handle” many inputs, in practice with ~180 obs we cannot trust letting 30+ indicators be automatically zeroed out without bias.  We will pre-select Tier-1/Tier-2 only, then rely on regularization to fine-tune weights.  If many features survive Tier-1 screening with significance, we may consider PCA or dimension reduction to avoid overfitting.

**Overall sequence:**  (1) Test Tier-1 features in ridge (one at a time); (2) Re-test best subset in a combined ridge; (3) GBT on the subset to discover interactions; (4) If classification is key, try logistic on that subset.  If multiple features are promising, we may also try small model ensembles.  

Importantly, **no strategy** should rely on simply dumping all 50+ TA features into a model and hoping ridge prunes them.  That is likely to produce spurious fits.  Instead, we emphasize **controlled incremental testing** as outlined.  

# 11. Failure Modes and Caveats

We must candidly acknowledge how this approach can fail:

- **Redundancy with existing features:**  Our baseline already includes 6-month price momentum and 63-day volatility.  Many TA indicators are just different flavors of momentum or volatility.  For example, MACD and ROC essentially encode momentum already captured by returns features; ATR and Bollinger width echo realized volatility (overlap with our 63d vol and VIX).  The realistic probability that an indicator adds *new* signal (beyond what momentum and volatility features give) is low.  At best they might slightly refine timing; at worst they do nothing.  

- **Horizon mismatch:**  Most TA were developed for days/weeks.  Their signal may not persist at 6-month horizons.  For instance, an RSI reading today might predict reversal in weeks, but 6 months is long.  If the underlying price drifts due to fundamentals, an RSI fade signal might dissipate.  We risk *false positives* if we ignore this: an indicator that “works” on daily data may be pure noise monthly.  The horizon gap is significant and not often addressed in TA literature (which rarely covers 6M forecasts).  

- **Multiple testing/data mining:**  Even with Tier filtering, we may test dozens of features.  Under reasonable assumptions, one would expect several false positives.  For example, if we test 20 indicators at 5% significance, ~1 will show spurious t>1.65 by chance.  If we lower thresholds via FDR, still expect some noise hits.  In our sequential filtering, even Tier-1 features could randomly appear significant in one benchmark.  This is why we need replication across splits, periods, and benchmarks.  

- **Regime dependence:**  TA signals might flip sign across regimes.  For example, momentum strategies often crash in panics.  A pattern that held pre-2020 might fail in 2020–21.  If we only get a few decades of data, a regime-specific anomaly can be misleading.  We mitigate by sub-sample tests, but small sub-samples are even noisier.  

- **Parameter sensitivity:**  Many indicators are sensitive to lookback length.  E.g. RSI(14) vs RSI(21) might give different signals.  We may unintentionally pick a “magic” parameter that happened to correlate this sample.  Without a principled way to choose periods (beyond defaults or financial meaning), we risk overfitting by tuning.  We will restrict ourselves to a few candidate settings and avoid extensive optimization.  

- **Alpha Vantage limitations:**  The free Alpha Vantage API has known constraints:  limited call frequency (so computing dozens of daily endpoints for ~14 tickers can be slow or incomplete under free tier), occasional missing data or gaps, and end-of-day updates only.  It can throttle if we exceed rates.  Also, intraday endpoints (like VWAP, some candlestick patterns) are not available or relevant for our monthly context.  We must ensure data integrity: e.g. confirm that each TA series is complete and consistent before using.  In practice, if API limits hinder daily updates, we may have to cache data or use the premium tier (if available) to run tests reliably.

- **Overfitting vs live performance gap:**  Even if we find one or two indicators that improve OOS metrics modestly, live trading/human implementation will likely underperform backtests.  Reasons: lookahead in data (we use EOM data which is future relative to some real-time decisions), transaction costs (more signals lead to more turnover), and the fact that machine learning skill often decays on new data (selection bias).  The “barrier to entry” in TA is low, so any discovered pattern is likely quickly arbitraged away in live markets.  We must treat positive WFO results as preliminary and require forward-testing on true new data (e.g. waiting one year of live trading) before believing them.

In short, the **default expectation** is failure or trivial effects.  We must guard vigorously against jumping to conclusions from a small data set.  The only defense is strict methodology: no p-hacking, only robust out-of-sample success counts.  

# 12. Prioritized Research Roadmap

We propose a phased approach with clear decision points:

- **Phase 1 – Theory & Screening:**  
  - *Objective:* Refine hypotheses and indicator list (Sections 2–4). Build data pipeline for candidate TA signals (daily data ingestion from AlphaVantage, monthly aggregation).  
  - *Data:* OHLCV daily for PGR, 8 ETFs, 4 peers.  
  - *Features:* Tier-1 indicators (RSI, ATR, Bollinger, ADX, etc) and baseline features for sanity.  
  - *Metric:* Preliminary walk-forward correlation (IC) of each new indicator with future relative return (univariate).  
  - *Success:* Identify any Tier-1 indicator with consistent IC >0.1 (t >1.0) across >3 splits.  
  - *Stop/Continue:* If none shows even a hint of directional predictive power (IC~0), then pause adding more.  If one or more show promise, move to Phase 2.

- **Phase 2 – Minimal Empirical Test (Tier 1 only):**  
  - *Objective:* Test Tier-1 indicators properly in WFO.  
  - *Model:* Ridge regression with baseline vs baseline+each Tier-1 feature (one at a time), and possibly logistic classification.  
  - *Comparison:* Use Diebold-Mariano and Clark-West tests on 6m returns (overlap adjusted).  
  - *Metric:* Out-of-sample $R^2$, IC t-stats, hit rate improvement.  
  - *Success:* At least one indicator yields (a) Clark-West t-stat > 1.65 on multiple benchmarks and (b) NW-corrected IC t-stat >1.65, and positive combined OOS $R^2$ improvement.  
  - *Stop:* If none meet threshold, conclude TA not promising.  If one or more pass, keep those and proceed.

- **Phase 3 – Expanded Features (Tier 2 + relative constructs):**  
  - *Objective:* Test weaker but plausible signals, and test Options A–C (relative or difference signals).  
  - *Features:* Tier-2 indicators, plus RSI and MACD on PGR/benchmark ratios or differences.  
  - *Model:* Same WFO setup with ridge/GBT on expanded set.  
  - *Metric:* Same as Phase 2, plus classification AUC.  
  - *Success:* New indicators show incremental improvement over Phase-2 results.  Specifically, more benchmarks where CW t>1.65.  
  - *Stop:* If Phase 3 indicators fail to improve further, revert focus on only Phase-2 successes.

- **Phase 4 – Nonlinear & Interaction (Regime Conditioning):**  
  - *Objective:* Explore GBT or logistic to capture interactions (e.g. ADX×momentum) and regime conditioning (e.g. feature × VIX high).  
  - *Features:* Interaction terms of top indicators with VIX, yield slope, etc.  
  - *Model:* Shallow GBT and Firth-Logit.  
  - *Metric:* Whether any new interactions significantly boost predictive stats.  
  - *Success:* Identify specific regimes where TA signals are stronger (e.g. high volatility periods).  
  - *Stop:* If no clear interactions survive cross-validation, deem model explained by main effects.

- **Phase 5 – Robustness & Go/No-Go:**  
  - *Objective:* Final validation and decision.  
  - *Tasks:* Conduct out-of-time test (e.g. use most recent unseen period as holdout). Test subperiod stability. Check sensitivity to parameter changes (like RSI window).  
  - *Metric:* Consistency of results, significant figures.  
  - *Success:* TA-enhanced model yields materially higher IC/hit in true out-of-sample (e.g. last-year).  
  - *Go/No-Go:* If robust criteria are met on holdout (e.g. CW t>2.0, Sharpe or policy gain decent), prepare for production. Otherwise, declare TA not sufficiently reliable.

This roadmap ensures that we only proceed to complex analyses if earlier, simpler tests indicate promise.  At each stage, failure to achieve benchmarks means halting further effort on TA indicators and exploring other ideas.

# 13. Concrete First Test Queue

Below are **the first 8 Alpha Vantage endpoints** we recommend implementing and testing.  For each, we specify the exact parameters, stationarity transform, tickers, relevant benchmark contexts, signal direction, and rationale:

| **Indicator**        | **Endpoint/Params**                   | **Transform (monthly)**           | **Tickers**               | **Benchmark Targets**      | **Expected Signal**                      | **Model** | **Justification/Citation**                                       |
|----------------------|---------------------------------------|-----------------------------------|---------------------------|---------------------------|------------------------------------------|-----------|------------------------------------------------------------------|
| RSI(14)              | `RSI`: time_period=14 (daily)         | `(RSI-50)/25`                     | PGR **and** VOO (compute both) | VOO, VXUS, VWO, VDE etc. | High PGR_RSI - VOO_RSI → negative rel. return (mean-revert) | Both      | RSI ≈70 means overbought【62†L347-L356】→ expect underperformance vs index. Tier-1 oscillator. |
| Stoch (14,3,3)       | `STOCH`: %K time_period=14, %D=3      | `(StochK-50)/25`                  | PGR, possibly difference PGR-VOO | VOO, emerging markets etc. | StochK>80 (overbought) → PGR underperf. | Both      | Stochastic oscillator signals exhaustion【62†L375-L383】, akin to RSI. |
| MACD (12,26,9)       | `MACD`: fast=12, slow=26, signal=9     | `MACD_hist/Close` (normalized)    | PGR                       | VOO, sector ETFs           | MACD_hist>0 → momentum → PGR outperf. | Ridge/GBT | Popular momentum indicator; tests broad trend.  Gains may be modest.【64†L267-L274】. |
| ADX (14)             | `ADX`: time_period=14 (daily)         | `(ADX-50)/50`                     | PGR                       | All benchmarks            | ADX>50 → strong trend, so momentum signals stronger. | Ridge   | ADX >25 indicates trend【64†L267-L274】; could modulate momentum’s effect. |
| ATR (14)             | `ATR`: time_period=14 (daily)         | `ATR/Close` (i.e. NATR)           | PGR, possibly ratio PGR/VOO | All benchmarks            | ATR high → volatile regime → possibly PGR underperf (risk-off). | Ridge   | ATR gives volatility. No direct predictive evidence; test as regime signal.  |
| BBANDS (20,2)        | `BBANDS`: time_period=20, std=2       | `%B = (Close-LB)/(UB-LB)`         | PGR                       | VOO, BND, commodity ETFS   | High %B (near upper band) → PGR underperf; low %B → outperf. | Both    | Bollinger signals extremes【67†L25-L33】; Tier-1 volatility filter. |
| Price/200-day SMA    | N/A (compute custom using `SMA` )    | `(Close - SMA200)/SMA200`         | PGR                       | All benchmarks            | Price >> SMA → strong trend → PGR likely outperf. | Ridge   | Overlap/trend feature (common in technical trading)【60†L344-L352】.  Serves as a baseline trend filter. |
| OBV (n/a)            | `OBV` (no time period)               | ΔOBV (month)/AvgVol or z-score    | PGR                       | VOO or peers             | Rising OBV suggests accumulation → PGR outperf; falling → underperf. | Ridge   | Volume momentum indicator【69†L330-L339】 (weak evidence); included for completeness. |

**Notes:**  
- Each indicator is computed on daily data and then summarized to month-end (by taking end-of-month value or monthly change as shown).  
- For RSI, we include both PGR and a benchmark (VOO) to form a difference or ratio-based feature.  This directly targets relative strength.  E.g. `RSI_diff = RSI_PGR - RSI_VOO`.  A high positive RSI_diff (PGR more overbought than VOO) is expected to predict negative relative return.  
- For MACD, ADX, ATR, BBANDS we start with PGR-only.  We also experiment with computing them on benchmark ETFs (e.g. `ADX(VOO)`, `MACD(VOO)`) as regime features (Option D above).  
- **Transform formulas:** Provided formulas ensure stationarity and scale-invariance.  For example, ATR/Close is the standard NATR.  BB %B is by design 0–1.  Price – SMA ratio centers the trend gap.  
- **Benchmarks:** We list likely ETFs where each feature might matter.  RSI/Stoch most applicable to equity benchmarks (VOO, VXUS, VWO, VDE), less so to bonds or gold.  Bollinger bands and ATR are tested against any benchmark (volatility could matter vs any).
- **Expected signals:** E.g. *“High RSI on PGR relative to VOO → PGR underperforms VOO.”* We specify direction where theory suggests it.  
- **Model:**  We test in both regression (Ridge) and classification (Logistic/GBT), depending on directionality.  For directional signals (RSI, Stoch), binary classification may work well.  For others (MACD, ATR), regression yields an alpha contribution.  

**Rationale and Citations:**  
We justify each choice with either theoretical reasoning or known usage.  For instance, RSI and Stochastic are well-known mean-reversion tools【62†L347-L356】【62†L375-L383】.  Bollinger bands use standard 20d,2σ settings【67†L25-L33】.  ATR is included as a volatility proxy, despite weak prior evidence; we treat it as exploratory.  We cite relevant descriptions ([62], [64], [67]) to ground the expected behavior.  

This prioritized list covers at least one indicator from each key group and follows the Tier ranking.  After implementing these, we will move on to additional Tier-2 candidates if any prove promising.

# 14. Final Bottom-Line Recommendation

1. **Is it worth pursuing TA indicators?**  *Likely only with low expectations.* The preponderance of evidence suggests that, in a small-sample setting with many benchmarks, technical indicators will at best yield only **marginal incremental signal** beyond existing momentum/volatility features【28†L46-L54】【26†L77-L85】.  The current negative OOS R² indicates calibration issues; improving direction via classification is feasible, but improving magnitude forecasts is hard.  Given the heavy multiple-testing burden and redundancy with momentum, we should approach TA feature addition cautiously.  It is **worth a careful search (as outlined)**, but the default assumption is that any discovered “alpha” will be weak.  

2. **Most promising families:**  The evidence favors **momentum oscillators and volatility bands**.  In particular, mean-reversion oscillators (RSI, Stochastic, CCI, WillR) and volatility envelope signals (Bollinger %B, ATR/NATR) have some out-of-sample backing in related contexts【62†L347-L356】【67†L25-L33】.  Trend filters (ADX, MACD) are secondary – they might refine when momentum holds, but are less likely to add new prediction beyond momentum itself.  **Volume indicators and pattern recognition are least promising**; their signals tend to be noisy and redundant.  

3. **Best target formulation:**  A *directional, cross-sectional* target seems most compatible with TA.  Concretely, a classification of “PGR outperforms (or underperforms) benchmark over next 6 months” is likely more informative than raw regression.  Among relative-return definitions, comparing PGR to a broad market ETF (VOO or another major index) is easiest to interpret and test.  We should also consider stock minus a peer composite as a variant.  As for horizon, *shorter is better*: significant TA signal is most likely seen in 1–3 month returns【55†L81-L89】.  However, since our model horizon is fixed at 6 months, we should *augment* it by also evaluating 1- and 3-month relative returns during development (as additional validation of signal stability).  

4. **Key methodological recommendation:**  The single most important step is **strict walk-forward validation with multiple-testing correction**.  Every indicator must be evaluated out-of-sample (rolling forward) without peek bias.  We must apply FDR or Bonferroni adjustments across our indicator tests to control false discoveries.  In practice, that means requiring *consistency* across splits and benchmarks (not just one isolated hit) before trusting a signal.  Without this rigor, any discovered “alpha” is likely spurious.  

5. **When to quit:**  If after testing the Tier-1 indicators none produce a robust improvement (e.g. **no TA feature yields a Clark-West t-stat > 1.65 on ≥3 benchmarks**, or improves pooled IC significantly above the 0.19 baseline), we should conclude diminishing returns.  A concrete rule: *if no TA indicator achieves IC t-stat >1.65 (p<0.05) in at least two separate WFO folds or across at least three different ETF comparisons, drop the pursuit.*  In that case, TA has shown no generalizable benefit, and we should pivot to other research (e.g. alternative macro signals, fundamental factors, or different modeling approaches).  

**Bottom line:** TA indicators could add *some* directional signal, but the effort is substantial and success is uncertain.  We proceed in a **highly prioritized, step-by-step** manner.  Only if a few Tier-1 signals survive our rigorous out-of-sample testing – boosting directional accuracy without overfitting – will we incorporate them.  Otherwise, we will save the work (and capacity) for more promising avenues.  

