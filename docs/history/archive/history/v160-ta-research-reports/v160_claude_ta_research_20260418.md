# Technical indicators on PGR: a limited, disciplined bet worth one decisive test

## 1. Executive summary

**Bottom line: Technical indicators from Alpha Vantage are unlikely to meaningfully improve 6-month forward PGR-vs-benchmark forecasts, but a tightly scoped, pre-registered test of 5–8 theory-grounded TA features is worth one concentrated research cycle before abandoning TA as a research direction.** The dominant academic finding — Han, Yang & Zhou (2013, JFQA) — is that moving-average and trend-based TA profitability concentrates in *high-volatility, high-information-uncertainty, small-cap* stocks. PGR is the opposite: large-cap, low-vol, heavily covered, institutionally held S&P 500 insurer. This is the worst-case sector profile for TA edge. Add Bajgrowicz–Scaillet (2012) FDR results — essentially zero TA rules survive honest multiple-testing plus transaction costs on liquid US names — and the prior is strongly negative.

**What remains defensible**: Neely–Rapach–Tu–Zhou (2014, *Management Science*) established that a *principal component of technical indicators* (PC-TECH) adds ~0.9% monthly OOS R² to equity-premium forecasts beyond macro, concentrated in recessions. That result is at the index level with monthly TA aggregation — the closest analog to the user's problem. The honest read is: TA may carry a small macro-timing signal; it is far less likely to carry individual-stock relative-return alpha for PGR specifically.

**Top Tier 1 candidates to test first (all stationarized, monthly interval or 6M-horizon-matched daily):**
1. `ROC_126` computed on the **ratio series** `P_PGR / P_ETF_k` (direct relative-momentum — the single most theoretically justified feature).
2. `(C − EMA_252)/EMA_252` on the same ratio series (long-trend deviation, horizon-matched).
3. `NATR_14` on PGR daily data (vol regime conditioner — not standalone alpha, an interaction term).
4. `%B` with `BBANDS(monthly, period=6, nbdev=2)` on PGR and ratio series (mean-reversion at horizon-matched window).
5. `PC-TECH` composite: first principal component of 6–10 TA signals on VOO (equity-premium PC-TECH analog, used as cross-asset regime signal).

**Recommended target formulation**: **binary classification** of `sign(R_{PGR,t+6} − R_{ETF,t+6})` via Firth-penalized logistic regression is the most TA-compatible target. Rationale: (a) the user's current diagnosis (positive IC ≈ 0.19, hit rate ≈ 69%, negative R²) proves directional skill exists despite magnitude miscalibration; (b) TA signals historically show hit-rate gains larger than R² gains; (c) Firth stabilizes small-N logistic inference.

**The single most important methodological caveat**: With N_eff ≈ 19 non-overlapping 6M observations in the walk-forward OOS window and 50 candidate TA features naively, expected false positives at α=0.05 ≈ 2.5 per 50 tests, with P(≥1 false positive) ≈ 92%. **Any TA research must use Benjamini–Hochberg–Yekutieli FDR at q=0.10 (effective threshold t ≈ 2.85–3.05) staged by theoretical prior. The user's ad-hoc "≥3 of 8 benchmarks at CW t>1.65" rule has FWER ≈ 0.15, not 0.05, and should be replaced with Romano–Wolf stepdown or a pre-specified t>2.0 on ≥4 of 8 threshold.**

**Stop rule**: If, after Phase 2 of the roadmap (Tier 1 test of 5 pre-registered features), no TA feature or composite achieves Clark–West t > 2.0 (HAC-adjusted with NW lag=10) on ≥4 of 8 benchmarks with Campbell–Thompson ΔCE > 50 bp/year and no regime catastrophic failure, **stop TA research and redirect to alternative signal sources (peer-relative fundamentals, tax/vesting-schedule features, or re-examining the structural weakness of the predictive stack itself).**

---

## 2. Problem framing and target formulation

The user is forecasting 6-month forward PGR return *minus* benchmark-ETF return. Three target formulations are live: continuous regression, binary outperform/underperform classification, and cross-sectional rank across the 9-symbol universe.

**Horizon selection**: 6M is fixed by the decision-support application (monthly vesting decisions with a 6M RSU horizon). Shorter horizons (1M, 3M) carry stronger TA evidence academically — Neely et al. use monthly; Jegadeesh-Titman (1993) finds 3-12M momentum most robust — but do not match the decision. The 12M horizon would carry more momentum signal (MOP 2012) but further reduces effective sample size (N_eff ≈ 180/12 ≈ 15). **6M is the right horizon; accept the associated power constraint.**

**Target type**: Three considerations favor **binary classification as primary** with regression as secondary diagnostic:
- The user's existing R² ≈ −13% with IC ≈ 0.19 and hit rate ≈ 69% is the signature of **directional skill with miscalibrated magnitudes**. A hit-rate-optimizing target (binary) preserves the skill; a variance-optimizing target (R²) penalizes magnitude error that TA is unlikely to fix.
- Cenesizoglu–Timmermann (2012, JBF) document that statistical R² often fails to translate to economic value; it is *directional accuracy in bad regimes* that drives utility gains. This matches the vesting application: correctly flagging the 10% worst 6M PGR-underperformance months dominates the CE calculus.
- Firth (1993, *Biometrika*) penalized logistic is demonstrably the right tool for N_eff ≈ 19 binary labels — it removes O(1/n) bias, handles separation, and has smaller confidence ellipsoids than MLE or L2 logistic (Kosmidis–Firth 2021).

**Use regression (Ridge + shallow GBT) as a secondary pipeline** to discipline magnitude estimates for position-sizing, but promote features on binary-classification performance first.

**Relative-return definition**: The 8 ETFs span macro regimes (VOO = broad US, VXUS = ex-US, VWO = EM, VMBS/BND = rates, GLD = inflation/safe haven, DBC = commodities, VDE = energy). The most TA-compatible benchmarks are the **broad-equity anchors** (VOO, VXUS) because both PGR and these benchmarks respond to equity-market regimes captured by TA. The least TA-compatible are **VMBS and BND**: PGR-vs-bond spread is largely driven by duration/credit-spread fundamentals the user already has (yield slope, credit spread, real rate). Expect TA features to contribute most against VOO, VXUS, VWO, and VDE; least against VMBS, BND, GLD, DBC.

Adding a **peer-relative composite** (PGR vs equal-weighted ALL/TRV/CB/HIG) is theoretically the cleanest test for sector-specific TA efficacy, because it removes equity-market direction and isolates insurance-sector idiosyncratic signal. The user has these peer tickers ingested — **add this as a ninth benchmark**.

---

## 3. Alpha Vantage indicator taxonomy

Alpha Vantage exposes **53 technical indicator endpoints**. Price transforms (AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE) and all TA-Lib candlestick patterns (CDL*) are **not available** as AV endpoints — compute locally from OHLCV. Four families organize the space.

**Overlap / trend-following** — SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3, MAMA, HT_TRENDLINE, MIDPOINT, MIDPRICE, SAR (VWAP is premium, intraday only). Economic mechanism: delayed mean estimates of price; captures trend via `C_t − MA_n`. Parameter sensitivity: period choice dominates; weighting scheme (simple vs exponential vs adaptive) is secondary. Expected within-group correlation ρ > 0.95 when windows match — **all are mathematical repackagings of the same smoothed-price information**. A single representative (EMA) captures >95% of the family's information. KAMA adds marginal independent signal via its volatility-adaptive weighting, useful only in trending/choppy regime distinctions.

**Momentum oscillators** — RSI, CMO, STOCH (SlowK/D), STOCHF, STOCHRSI, WILLR, MFI, ULTOSC, CCI, TRIX, MOM, ROC, ROCR, APO, PPO, MACD, MACDEXT, BOP, AROON/AROONOSC, ADX/ADXR/DX/PLUS_DI/MINUS_DI/PLUS_DM/MINUS_DM. Economic mechanism: compare recent gain vs loss, or position within range. For relative-return prediction, the mechanistic story is **underreaction → continuation** (momentum) or **overreaction → reversal** (oscillators at extremes). These mechanisms are in tension and must be tested separately. Collinearity: RSI↔CMO ρ≈0.95; MOM↔ROC↔ROCR ρ≈1.0 exactly; MACD=APO with same periods; PPO = 100×MACD/slowMA (scale-invariant variant); DX/ADX/ADXR/DI all share DM+TR machinery with pairwise ρ>0.85. Recommended representatives: **RSI** (oscillator), **ROC** (momentum), **PPO** (MACD-normalized), **(PLUS_DI − MINUS_DI)** for direction + **ADX** for strength, **AROONOSC**, **ULTOSC** (multi-timeframe).

**Volatility / range** — ATR, NATR, TRANGE, BBANDS. These are **risk measures, not alpha**. Realized vol already in baseline. BBANDS contributes two derived features: %B = (C−Lower)/(Upper−Lower) as a mean-reversion positioning signal, and Bandwidth = (Upper−Lower)/Middle as a vol-regime scalar. NATR = 100×ATR/C is the scale-invariant ATR. Recommend NATR and %B + Bandwidth only.

**Volume / money flow** — OBV, AD, ADOSC, MFI. OBV and AD are **cumulative and non-stationary** — must be differenced or oscillator-transformed before use in Ridge. ADOSC (Chaikin oscillator) is the pre-stationarized form of AD. MFI is a volume-weighted RSI. In NRTZ (2014), OBV-based rules performed comparably to MA rules and loaded onto PC-TECH — the strongest volume-TA evidence. For individual stock prediction, volume signal is known to weaken in heavily institutional/benchmarked names (PGR qualifies).

**Cycle / Hilbert Transform** — HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE, HT_TRENDLINE. HT_DCPHASE is angular in [-180°, 180°] and **must be sin/cos encoded** before any linear model. HT_TRENDLINE is effectively collinear with a long EMA. HT_TRENDMODE is already binary {0,1}. No peer-reviewed top-journal OOS evidence supports Hilbert-transform TA for individual stock forecasting — treat as Tier 3 except HT_TRENDMODE as a regime switch.

**Pattern recognition (candlesticks)** — compute locally via TA-Lib (61 patterns). Aggregated to monthly as `count(bullish) − count(bearish)`. Marshall, Young & Rose (and multiple replications) find no OOS predictive value at daily frequency net of data-snooping adjustment. Tier 3 — exclude.

---

## 4. Literature synthesis — what is actually supported

The academic record on TA is mixed and sector-conditional. Six findings are well-supported, four are weak or regime-dependent, and most of the indicator zoo has no credible evidence.

**Well-supported**: **Cross-sectional momentum** (Jegadeesh–Titman 1993, 2001) at 6-12M formation / 6M holding clears Harvey–Liu–Zhu t>3.0 with ~1%/month WML returns (t-stats 3–4 through 1990s, weaker but still positive post-2000). **Time-series momentum** (Moskowitz, Ooi, Pedersen 2012, JFE) shows past 12M excess return predicts next 1M across 58 futures contracts with 52 of 58 significant; diversified TSMOM delivers Sharpe ≈ 1.2. **Principal component of TA signals predicts equity premium in recessions** (Neely, Rapach, Tu, Zhou 2014, *Management Science*): PC-TECH R²_OS ≈ 0.88% monthly (t≈2.3 on MSFE-adj), PC-ALL combining PC-TECH and PC-ECON ≈ 1.98%, with gains 5–10% of R² during NBER recessions and near zero in expansions. **Trees/neural nets extract more TA signal than linear models** (Gu, Kelly, Xiu 2020, *RFS*): NN3 achieves monthly OOS R² ≈ 0.40% vs OLS-3 at 0.16%, with long-short decile Sharpe ≈ 1.35 vs 0.6. **Trend-factor cross-sectional regression spans MA timing** (Han, Zhou, Zhu 2016, JFE): a regression on past returns at multiple horizons captures MA effects — i.e., MA is a repackaging of multi-horizon momentum. **Campbell–Thompson (2008, RFS) sign restrictions** on return forecasts improve OOS economic value even when statistical R² is tiny.

**Regime-dependent or weak**: Moving-average timing profitability **concentrates in high-volatility, high-information-uncertainty stocks** (Han, Yang, Zhou 2013, JFQA) — this is the single most important sector-conditional result. Lo–Mamaysky–Wang (2000, JF) automated patterns show conditional distributions differ from unconditional but no economic OOS R² net of costs. Zhu–Zhou (2009, JFE) provide theoretical support for MA under parameter/model uncertainty. Goh, Jiang, Tu, Zhou (2013) find TA signals have OOS power for US Treasury bond premia, complementary to macro — relevant for cross-asset signals.

**No credible evidence / evidence against**: Bajgrowicz–Scaillet (2012, JFE) apply FDR to 7,846 Sullivan–Timmermann–White rules on DJIA 1897–2011: **(a) best historical rules fail OOS persistence tests, (b) in-sample performance is completely offset by 5 bp one-way transaction costs, (c) net economic value ≈ 0**. Hsu–Kuan (2005, JFE) find profitable TA only in "young" markets (NASDAQ, R2000) pre-ETF; DJIA/S&P 500 show none. Harvey–Liu–Zhu (2016, RFS) argue virtually no individually-tested TA rule clears the adjusted t>3.0 hurdle given the factor zoo. **No peer-reviewed top-journal paper establishes OOS predictive value specifically for financial/insurance stocks at 1-6M individual stock level for MACD, RSI, Bollinger, stochastics, ADX, or Hilbert indicators.**

**The critical sector caveat**: Han–Yang–Zhou's finding that MA profitability scales with volatility and information uncertainty is directly disqualifying for PGR. Baker–Bradley–Wurgler (2011, FAJ) show low-vol stocks are dominated by benchmarked institutional flow, not trend-following retail — the behavioral mechanism for momentum/MA doesn't operate strongly. PGR has ~90% institutional ownership, high analyst coverage, 30+ year history, and bottom-decile volatility for S&P 500 financials. This is the worst profile for TA.

**Fraction of daily TA signal surviving aggregation to 6M**: under 5% net of multiple-testing and costs, based on BS (2012) FDR work. The signal that does survive at 6M is essentially *momentum at various horizons*, which the user already has as `mom_6m`.

---

## 5. Tiered indicator ranking

| Tier | Indicators | Evidence level | Expected signal direction for PGR outperformance | Strongest asset class |
|---|---|---|---|---|
| **1** | ROC_126 / MOM_126 (6M return) | Strong — JT 1993, MOP 2012, NRTZ 2014 | Positive: prior 6M relative outperformance → continuation | All equity benchmarks, weakest vs bond ETFs |
| **1** | (C − EMA_252)/EMA_252 on ratio series | Strong — HZZ 2016 "Trend Factor" spans MA via multi-horizon momentum | Positive | Equity benchmarks |
| **1** | PC-TECH composite (first PC of 6–10 aggregate TA signals on VOO) | Strong — NRTZ 2014, R²_OS ≈ 0.88% | Regime-conditional; stronger in recessions | Macro/cross-asset regime |
| **1** | NATR_14 on PGR (interaction/conditioner only, not standalone) | Strong — GKX 2020 retvol top feature | Interaction — amplifies momentum signal in high-vol regimes | Universal risk scaler |
| **2** | %B with BBANDS(monthly, period=6) on ratio series | Weak — mean-reversion hypothesis at horizon-matched window, no direct OOS evidence | Negative at extremes (overbought → underperformance) | Equity benchmarks |
| **2** | ADX on ratio series + PLUS_DI − MINUS_DI | Weak — trend strength regime conditioner; no direct OOS | Regime gate (trend follow vs mean revert) | Equity benchmarks |
| **2** | PPO or MACD_hist/C (scale-invariant) | Weak — collinear with ROC/EMA family; minor independent info | Positive momentum | Equity benchmarks |
| **2** | ADOSC normalized by dollar volume | Weak — NRTZ 2014 OBV loaded on PC-TECH; ADOSC is stationary variant | Positive (accumulation) | Equity benchmarks |
| **2** | HT_TRENDMODE (binary regime flag) | Weak — no OOS evidence for individual stocks; theoretical regime use only | Regime gate | Universal |
| **2** | RSI_126 on ratio series, transformed (RSI−50)/50 | Weak — tension between momentum (continuation) and oscillator (reversal) mechanisms at same indicator | Ambiguous direction — test both | Equity benchmarks |
| **3 (exclude)** | All other overlap studies (WMA, DEMA, TEMA, TRIMA, T3, MAMA, TRIMA, MIDPOINT, MIDPRICE, SAR) | Collinear with EMA; no independent evidence | — | — |
| **3 (exclude)** | MACDEXT, APO, MOM, ROCR, TRIX, BOP, CMO, AROON/AROONOSC, DX, ADXR, PLUS_DM, MINUS_DM, STOCH, STOCHF, STOCHRSI, WILLR, ULTOSC, MFI, CCI | Redundant or no OOS evidence at 1-6M for individual stocks | — | — |
| **3 (exclude)** | OBV (raw — cumulative, use ADOSC instead) | Non-stationary; ADOSC supersedes | — | — |
| **3 (exclude)** | HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDLINE | No OOS evidence for individual stock prediction | — | — |
| **3 (exclude)** | All 61 candlestick patterns | No OOS evidence net of data-snooping at daily; aggregation to monthly destroys remaining signal | — | — |
| **3 (exclude)** | AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE | Near-collinear with close, add noise | — | — |
| **3 (exclude)** | TRANGE, ATR (use NATR), BBANDS Upper/Middle/Lower raw (use %B, Bandwidth) | Subsumed by scale-invariant siblings | — | — |

---

## 6. Absolute vs relative signal construction

Four construction options carry very different theoretical and empirical weight for this relative-return target.

**Option A — indicator computed on the ratio series `P_PGR,t / P_ETF_k,t`**: *Theoretically the cleanest.* The feature measures momentum, mean reversion, or trend *of the relative-performance curve itself* — precisely what the target measures forward. Valid for all momentum/volatility indicators requiring only close (RSI, EMA/SMA family, MACD, ROC, CMO, BBANDS, ATR on ratio, TRIX, HT family, PPO). Invalid for indicators requiring volume (OBV, AD, ADOSC, MFI) or H/L (STOCH, WILLR, CCI) because the ratio series has no native volume and range concepts don't transfer. **This is the first construction to test.**

**Option B — difference of independent indicators (PGR_RSI − VOO_RSI)**: Valid only for bounded unitless oscillators (RSI, STOCH, %B, ADX, MFI, ULTOSC, AROONOSC). Captures *momentum divergence*. Theoretically weaker than Option A because the indicators are non-linear functions of price and their difference is not mathematically equivalent to the ratio-series indicator. Use as a robustness check on Option A.

**Option C — PGR-only indicators**: Captures PGR absolute signal. Weak for *relative* return prediction because any broad-market component is not partialled out. Weakest expected to help.

**Option D — benchmark-ETF indicators as cross-asset regime signals**: BND's MACD telling you whether you're in a rate-hawkish environment has a **fundamental channel** (NIM, investment-income) to PGR earnings, not a TA-to-TA channel. Goh–Jiang–Tu–Zhou (2013) supports bond-TA as a useful bond-premium predictor; translation to PGR relative returns is indirect. Theoretically plausible but expect small effects. PC-TECH on VOO is the cleanest instantiation (NRTZ 2014 analog).

**Peer-relative signals (PGR vs ALL/TRV/CB/HIG)**: Compute ratio-series indicators on `P_PGR / mean(P_ALL, P_TRV, P_CB, P_HIG)`. This removes equity-market direction and tests for sector-idiosyncratic TA signal. Theoretically the most novel construction — no published evidence, but the mechanism (insurance-peer relative over/underreaction) is at least specific. Worth one concentrated test in Phase 3.

**Recommendation order**: Option A first (ratio series), Option D second (cross-asset VOO PC-TECH), peer-relative third, Option B as robustness, Option C only if A-C all fail.

---

## 7. Frequency adaptation and monthly aggregation

TA signals computed on daily data are overwhelmingly designed for days-to-weeks prediction. Aggregating them to monthly features predicting 6M ahead requires choosing both a **data frequency** and an **aggregation rule**.

**Data-frequency principle**: for monthly features predicting a 6M horizon, **prefer indicators computed on monthly-bar data** with `time_period ∈ {3, 6, 12}`. This matches Rapach–Zhou's equity-premium conventions, minimizes intraday noise, aligns feature and target sampling frequency for cleaner Ridge inference, and uses many fewer degrees of freedom. Supplement with 1-2 long-period daily indicators (e.g., SMA_252, RSI_126) as slow-moving regime signals. **Abandon the 14-day classic defaults for this problem** — they capture noise at 6M.

Alpha Vantage accepts `interval=monthly` on technical-indicator endpoints, so monthly-bar indicators are free.

**Aggregation rules by indicator family**:

- **Momentum / trend (ROC, EMA, MACD, PPO)**: use **end-of-month snapshot** of the monthly-interval indicator. Don't trailing-average across days within the month — that reintroduces noise.
- **Bounded oscillators (RSI, STOCH, %B, ADX, MFI)**: end-of-month snapshot plus a **sign-of-change** feature (`sign(RSI_t − RSI_{t-1})` on monthly bars) to capture direction of regime change.
- **Volatility (NATR, BB Bandwidth)**: end-of-month snapshot. Both level and `Δ` matter; include both.
- **Volume (ADOSC)**: end-of-month snapshot after normalizing by trailing dollar-volume.
- **Cycle (HT_TRENDMODE, sin/cos HT_DCPHASE)**: end-of-month snapshot on monthly-bar indicator.
- **Threshold flags (I(C>SMA_200), I(RSI>70), I(ADX>25))**: aggregate to monthly as **fraction of days in the month with condition true** (yields continuous [0,1] feature rather than binary, reducing variance).
- **Candlestick patterns (if tested)**: `count(bullish) − count(bearish)` per month. But: exclude per Tier 3.

---

## 8. Stationarity transformation formulas

Every feature entering Ridge must be stationary and scale-invariant. Exact transforms for Tier 1 and Tier 2:

| Indicator | Raw | Transform | Notes |
|---|---|---|---|
| SMA_n, EMA_n | price | `(C_t − EMA_n)/EMA_n` | Price-deviation ratio; works on raw or ratio series |
| EMA_fast − EMA_slow | price | `(EMA_fast − EMA_slow)/EMA_slow` | Trend-slope proxy, equivalent to PPO |
| MACD, MACD_hist, APO | price-scale | `MACD_hist / C_t` or use PPO directly | PPO already scale-invariant |
| PPO | % | `PPO / 100` (to decimal) | Already scale-invariant |
| ROC_n | % | `log(C_t / C_{t−n})` | Log return preferred for symmetry |
| RSI_n | [0,100] | `(RSI − 50) / 50` | Centered bounded |
| MFI_n | [0,100] | `(MFI − 50) / 50` | |
| STOCH %K, %D | [0,100] | `(K − 50)/50`; also spread `(K−D)/50` | |
| WILLR | [−100, 0] | `(WILLR + 50)/50` | |
| ULTOSC | [0,100] | `(ULTOSC − 50)/50` | |
| CMO | [−100,100] | `CMO / 100` | |
| CCI | unbounded heavy-tailed | `tanh(CCI / 100)` | Winsorizing alternative; don't use raw z-score |
| ADX | [0,100] | `ADX / 100` (strength, unsigned) | Not centered — ADX is magnitude |
| PLUS_DI, MINUS_DI | [0,100] | `(PLUS_DI − MINUS_DI) / (PLUS_DI + MINUS_DI + ε)` | Signed trend-direction |
| AROONOSC | [−100,100] | `AROONOSC / 100` | Pre-combined |
| BBANDS | 3 price lines | %B = `(C − Lower)/(Upper − Lower)`; Bandwidth = `(Upper − Lower)/Middle` | Two derived features |
| ATR_n, NATR_n | price / % | `NATR/100` | Scale-invariant vol |
| OBV | cumulative | `(OBV_t − SMA_n(OBV))/rolling_std_n(OBV)` or use ADOSC | Differencing essential |
| ADOSC | vol-scale | `ADOSC / SMA_n(Volume × C)` | Dollar-volume normalize |
| HT_TRENDMODE | {0,1} | use as-is | |
| HT_DCPHASE | [−180°, 180°] | `sin(DCPHASE·π/180), cos(DCPHASE·π/180)` | Two features — angular encoding mandatory |
| HT_DCPERIOD | ~[10,50] | `(HT_DCPERIOD − 30)/20` | Center and scale |

**Additional feature constructions** (for every transformed `f_t`): first difference `Δf_t = f_t − f_{t−1}`; acceleration `Δ²f_t`; binary thresholds aggregated monthly; cross-sectional z-score across the 9-symbol universe; rolling 24-month z-score within symbol. **Recommended hybrid**: rolling within-symbol z first, then cross-sectional demean at each t. This is the standard equity cross-sectional factor recipe (Asness–Frazzini–Pedersen 2013).

**Multicollinearity handling**: after transforms, (a) compute pairwise correlation matrix on an initial 60-month window; (b) run complementary-pairs stability selection (Shah–Samworth 2013) with ElasticNet and π_thr = 0.75; (c) drop any feature with VIF > 10 against the rest; (d) do NOT use PCA blindly — PC-TECH per NRTZ 2014 is valuable precisely as a *pre-specified* composite, not as an automatic dimension-reduction step.

---

## 9. Testable hypotheses, ranked by plausibility

| Rank | Hypothesis | Indicator | Direction | Mechanism | Target | Evidence strength |
|---|---|---|---|---|---|---|
| 1 | 6M ratio-series momentum continues | `ROC_126(P_PGR/P_VOO)` and `(P_PGR/P_VOO)` vs EMA_252 | Positive | Cross-sectional + time-series momentum (JT, MOP, HZZ 2016) | Both regression and classification; all equity benchmarks | Strong |
| 2 | PC-TECH on VOO predicts PGR vs equity benchmarks in recessions | First PC of {EMA dev, ROC_126, ADX, NATR, %B, ADOSC} on VOO | Positive in recessions, near 0 expansions | NRTZ 2014 — PC-TECH in NBER recessions | Regression primarily (NRTZ style); VOO, VXUS, VWO, VDE | Strong for macro; unclear for single-stock |
| 3 | NATR interaction with momentum | `NATR_PGR × ROC_126(ratio)` | Positive interaction coefficient | Han-Yang-Zhou 2013 — TA/MA pays in high-vol; GKX 2020 retvol as top feature | GBT first (captures interaction); then Ridge with explicit interaction term | Moderate theoretical |
| 4 | Peer-relative ratio momentum | `ROC_126(P_PGR / mean(ALL,TRV,CB,HIG))` | Positive | Sector-idiosyncratic momentum removes equity-market direction | Binary classification, VOO/VXUS benchmarks | Novel, no published evidence |
| 5 | Bond-TA regime signal conditions PGR-vs-bond spread | `(C − EMA_252)/EMA_252` on BND and VMBS | Negative (BND trend up → PGR underperforms bonds) | Fundamental duration/NIM channel, not TA-to-TA | Regression, VMBS and BND benchmarks specifically | Weak |
| 6 | Extreme %B reverses at 6M | `I(%B_6M_monthly > 1)` on ratio series | Negative (overbought → underperformance) | Mean-reversion at horizon-matched window | Binary classification, equity benchmarks | Weak |
| 7 | ADX regime gate boosts momentum | `I(ADX_ratio > 25)` gates momentum feature | Positive interaction | Trend-strength regime conditioning | GBT, equity benchmarks | Weak |
| 8 | VIX-conditioned cycle state predicts defensive rotation | `HT_TRENDMODE × VIX_level` | PGR outperforms equity benchmarks when `(HT_TRENDMODE=0 AND VIX high)` (low-beta defensive rotation) | Defensive-stock flow in cycle-mode regimes; Baker-Bradley-Wurgler | Classification, VOO/VXUS | Weak, speculative |
| 9 | ADOSC on PGR accumulation predicts outperformance | normalized ADOSC on PGR | Positive | NRTZ 2014 OBV loaded on PC-TECH | Regression, equity benchmarks | Weak |
| 10 | RSI_126 mean reversion on ratio series | `(RSI_126(ratio) − 50)/50` | Negative at extremes | Oscillator reversal hypothesis (tension with #1) | Nonlinear binary classification; GBT | Weak, in tension with H1 |
| 11 | Cross-asset GLD and DBC TA as inflation regime signal | `(C − EMA_252)/EMA_252` on GLD, DBC | Positive (inflation good for commodity, tests whether PGR benefits or loses to commodity ETFs) | Macro regime | Regression vs GLD, DBC benchmarks specifically | Weak |
| 12 | VDE TA as energy-cycle signal | `ROC_126` on VDE | Negative for PGR-vs-VDE (energy beta ≠ insurance beta) | Sector rotation | Regression vs VDE | Weak |

---

## 10. Evaluation framework and stop/continue rules

**Baseline**: the 12-13 feature Ridge + GBT + Firth logistic stack exactly as currently implemented. Every TA test is incremental vs this baseline. **Per user clarification 1**, also test swap variants: replace each of `vmt_yoy`, `yield_curvature`, `mom_3m`, `nfci` with a Tier 1 TA feature and compare both addition and swap models against the original baseline.

**Primary promotion gate — Clark–West (2007) MSPE-adjusted statistic**. For nested models (baseline ⊂ baseline + TA) the Diebold–Mariano test is *biased against* the larger model under H0 because the larger model's estimation noise inflates its MSPE. CW corrects this by adding a noise-refund term `(ŷ₁ − ŷ₂)²`:

`f_{t+h} = e₁,t+h² − e₂,t+h² + (ŷ₁,t+h − ŷ₂,t+h)²`

Regress `f_{t+h}` on a constant with HAC standard errors; the t-stat on the constant is CW. One-sided rejection at z=1.645 gives approximately α=0.05 per Clark–West 48-DGP simulations. **Use z=1.645 as minimum; prefer 2.0 to build in the correlated-benchmark aggregation discount.**

**HAC lag selection**: with 6M overlap, residuals have MA(h−1)=MA(5) structure mechanically. Set NW bandwidth `L = 2(h−1) = 10` as the default; hard floor L ≥ h−1 = 5. Ang–Bekaert (2007) and Boudoukh–Richardson–Whitelaw (2008) document that NW/Hansen–Hodrick are *downward-biased* in small samples. For critical inference, cross-check with **Hodrick 1992 "1B" reverse-regression standard errors** which impose the no-predictability null and typically give 30–50% larger t-stats.

**Effective sample size**: with T_OOS = 180 − 60 − 8 = 112 monthly OOS observations and h=6, N_eff ≈ 112/6 ≈ **19 non-overlapping 6M obs**. This is tight. Single-predictor IC SE ≈ 1/√19 ≈ 0.23 — a raw IC of 0.46 is needed for single-test t=2. Grinold-Kahn IR ≈ IC × √BR with BR capped by cross-signal correlation at maybe 5–15; expect realistic IR uplift of 0.2 at most per TA feature.

**Multiple-testing control**: use **Benjamini–Hochberg–Yekutieli (2001) FDR at q=0.10 with arbitrary-dependence factor c(m) = Σ 1/i ≈ ln(m)+0.577**. For m=50, c ≈ 4.5, threshold for k=5 discoveries ≈ p < 0.00222 (t ≈ 3.06). This converges to HLZ's t>3 rule of thumb. **For pre-specified Tier 1 short list of 5-10 features, use BH (not BHY) at q=0.10** — the theoretical pre-commitment reduces the effective testing space. Romano–Wolf stepdown is the most powerful option for the correlated 8-benchmark cross-section.

**Full metric set**:
- **Pooled OOS R²** (Campbell-Thompson version with sign/variance restrictions)
- **Newey-West HAC-adjusted IC** (Spearman on ranks)
- **Hit rate** (binary accuracy) with block-bootstrap SE at block length ≥ 6
- **Covered balanced accuracy** for imbalanced periods
- **Clark–West t-stat** (primary gate)
- **Campbell-Thompson CE gain** at γ=3 and γ=5 (primary economic gate)
- **Deflated Sharpe Ratio** (Bailey-Lopez de Prado 2014) and **PBO** (Bailey et al. 2017) computed via CPCV

**Primary promotion gate (compound)**: A TA feature or composite is **promoted** if and only if: (a) Clark–West t > 2.0 HAC-adjusted on ≥4 of 8 benchmarks OR on the peer-relative composite, (b) Campbell-Thompson CE gain > 50 bp/year at γ=3, (c) no regime with R²_OOS < −1% in pre-specified cuts (pre-2020, 2020–21, 2022–present), (d) DSR > 0 and PBO < 0.3 via CPCV with N=10, k=2.

**Concrete stop rules**:
- **After Phase 2 (Tier 1 test of 5 features)**: if no individual feature or the PC-TECH composite clears CW t > 2.0 on ≥3 benchmarks, stop — TA is not incrementally informative. Redirect research.
- **After Phase 3 (Tier 2 + ratio-series constructions)**: if promoted features total ≤ 1 net of the baseline swap, stop adding TA and consolidate the one survivor.
- **Hard ceiling**: any phase in which adding TA features *degrades* OOS R² or hit rate on ≥5 of 8 benchmarks → abandon TA immediately.
- **Anti-p-hacking commitment**: pre-register feature list, CW threshold, and BHY q level in a Git commit hash *before* running OOS evaluation.

---

## 11. Recommended model sequence

The user runs three model types: Ridge regression, shallow GBT (depth=2), Firth logistic. Each answers a different question.

**Firth penalized logistic first**. Rationale: the user's diagnosis is that directional skill exists (IC 0.19, hit rate 69%) but magnitude is miscalibrated (R² = −13%). Binary targets preserve the skill; Firth handles the small-N binary setting without the separation and bias pathologies of MLE or L2 logistic at N_eff=19. Use penalized likelihood ratio tests (not Wald) for coefficient inference (Heinze 2006). Report McFadden adjusted R² and Brier score OOS. This is the gate that most likely shows TA value if TA has any value.

**Ridge regression second**. Rationale: confirms direction-consistent continuous prediction; provides input for magnitude-based sizing. Fix λ ex-ante on a 48-month tuning slab; do **not** re-CV each fold (inner CV with <30 obs is dominated by noise; shrinkage prior beats cross-validated λ in this regime). Ridge alone is **insufficient** for 25+ correlated TA features at N_eff=19 — the user is correct that regularization does not save pure addition. **Theory-based pre-selection (Tier 1-2 only) is required, even with regularization.** The direct answer: **including 20-50 stationarized TA features on Ridge/Lasso with N_eff=19 is not defensible. Pre-selection to ≤8-10 TA candidates with strong theoretical prior is mandatory.** Stability selection (Meinshausen-Bühlmann 2010 complementary-pairs version, Shah-Samworth 2013) as a robustness check if the user wants to include 20+ candidates.

**Gradient-boosted trees (depth=2) third**. Rationale: captures nonlinear interactions (Gu-Kelly-Xiu 2020 found trees/NNs extract TA × vol interactions linear models miss). Use heavy shrinkage (learning rate ≤ 0.01, min_samples_leaf ≥ 10, ~100 trees max), early stop on walk-forward OOS. At N_eff=19 a single depth-2 tree is fitting 4 noisy subgroup means — avoid single trees; use the ensemble only. The primary value of GBT here is testing whether Hypothesis 3 (NATR × momentum interaction) has incremental value beyond additive Ridge.

**Critical principle**: in the N_eff=19 regime, Ridge/Firth and GBT are likely to perform comparably; the interpretability win for Ridge and Firth is decisive for *understanding* which TA features contribute vs merely *predicting* that they do. Prefer interpretable linear models for the Tier 1 screen; escalate to GBT only for the specific interaction hypotheses.

---

## 12. Failure modes and caveats

**Information redundancy**: This is the single largest risk. The user's baseline already contains `mom_6m` and `vol_63d` — which are mathematically very close to what MA, MACD, PPO, ROC, NATR, and BBANDS Bandwidth measure. The Gu–Kelly–Xiu (2020) finding is that after momentum at multiple horizons + realized volatility + liquidity are in the model, the marginal OOS R² from additional TA indicators is under 0.1% monthly — likely negative after costs and multiple testing. **Realistic probability that TA provides non-redundant signal beyond user's baseline: 20-30%.**

**Horizon mismatch**: nearly all TA academic evidence is daily/weekly. At 6M aggregation, <5% of the rule-level signal survives net of FDR and costs (BS 2012). The signal that survives is essentially multi-horizon momentum — already captured by `mom_6m`.

**Multiple testing**: with 50 TA features tested naively, expected false positives = 2.5, P(at least one) = 92%. Under BHY FDR at q=0.10 with correlation adjustment c(50) ≈ 4.5, effective threshold p < 0.00044 for k=1 (t ≈ 3.52), p < 0.00222 for k=5 (t ≈ 3.06). The user's power to detect a Sharpe improvement of 0.2 at these thresholds is <1%. **To have any power, reduce m to 5-10 via pre-registration, or treat as a hypothesis-generation exercise requiring independent-sample confirmation (which does not exist).**

**Regime dependence**: NRTZ 2014 shows TA signal concentrates in NBER recessions and disappears in expansions. User's sample spans 2010-2025 with only the 2020 recession — very limited recession data. Expect PC-TECH-style composites to appear noise-dominated outside recession, limiting the apparent evidence.

**Parameter sensitivity**: RSI-14 vs RSI-21 vs RSI-50 will give different answers. This is itself a form of multiple testing. Discipline: fix `time_period` ex-ante using horizon-matching (3, 6, 12 on monthly bars), do not grid-search.

**Alpha Vantage free-tier constraints** (critical operational finding): free tier as of April 2026 is **25 API requests/day** (reduced from historical 500/day and later 100/day). Full history on `TIME_SERIES_DAILY_ADJUSTED` is premium-only; free gets 100 bars. `TIME_SERIES_MONTHLY_ADJUSTED` and `TIME_SERIES_WEEKLY_ADJUSTED` return full 20+ year history on free tier. **Recommended path: compute all indicators locally using pandas-ta / TA-Lib from free monthly + weekly adjusted downloads. Do not use AV's technical-indicator endpoints for this project.** Nine symbols × 2 endpoints (monthly + weekly) = 18 requests/month, well within budget. Benefit: enables price transforms, candlestick patterns, ratio-series indicators, and corporate-action-adjusted indicator computation (AV's indicator endpoints use raw price and can have artificial jumps on split/dividend days — a real data-quality issue for long-horizon indicators).

**Overfitting vs live-performance gap**: Even if WFO improves, structural reasons live may be worse include: (a) PGR's capital-return policy (variable dividends) creates adjusted-price discontinuities that TA indicators handle imperfectly; (b) RSU vesting introduces a secondary supply/demand pressure on PGR specifically (insider selling) that is idiosyncratic to the user's own decision and not in historical data; (c) insurance-sector capital cycles (soft/hard market) are long-wavelength regimes with only 2-3 cycles in the sample, leaving no OOS validation headroom; (d) TA's strongest evidence is in recessions — if the forward period is an expansion, expect degradation.

**The honest meta-caveat**: the user's diagnosis of "variance dominance, not absent signal" (IC 0.19, hit 69%, R² −13%) plus the external peer review concluding "predictive stack may be structurally weak rather than architecturally fixable" is consistent with a model that has a real but small signal being swamped by calibration noise. **TA is unlikely to change this diagnosis** because TA at 6M horizon is well-documented to offer only modest index-level signal and near-zero individual-stock signal for low-vol large-cap financials. If this test returns negative, it is confirmatory of the structural weakness finding and the correct response is to redirect to non-TA signal sources or accept that the system will be limited to low-resolution directional guidance.

---

## 13. Prioritized research roadmap

**Phase 1 — theory-first screening (1 week, no coding)**: lock the Tier 1 feature list (5 features), the CW threshold (t > 2.0), the BH q=0.10 level, the regime cuts (pre-2020, 2020-21, 2022+), the CPCV parameters (N=10, k=2), and the benchmark aggregation rule (≥4 of 8 or Romano-Wolf). Commit to Git with timestamp. **Stop rule**: no Phase 2 until this is written and reviewed.

**Phase 2 — Tier 1 MVP empirical test (2 weeks)**: download monthly + weekly adjusted history for 13 symbols (9 benchmarks/PGR + 4 peers). Compute 5 Tier 1 features: `ROC_126(ratio)` for each of 8 ETFs, `(C − EMA_252)/EMA_252` on ratio, `NATR_14` on PGR, `%B(monthly,6,2)` on ratio, and PC-TECH on VOO. Run Firth logistic + Ridge + GBT (in that order) as both additions to baseline and swaps for `mom_3m` and `nfci`. Evaluate: CW t-stat, CT ΔCE, regime R²_OOS, DSR/PBO via CPCV. **Success criterion**: ≥1 Tier 1 feature or composite achieves CW t > 2.0 on ≥3 benchmarks with CT ΔCE > 50 bp. **Stop if not met — this is the key gate; failure here means TA research terminates.**

**Phase 3 — Tier 2 + ratio-series + peer-relative (2-3 weeks, only if Phase 2 passes)**: add 5 Tier 2 features; explore Options B, D, and peer-relative construction; test interactions (NATR × momentum, ADX × momentum). Apply BHY q=0.10 to the expanded test space. **Success criterion**: total promoted features (Tier 1 + Tier 2) ≥ 2 with consistent signs across regimes.

**Phase 4 — nonlinear interactions and regime conditioning (2 weeks, if Phase 3 passes)**: escalate to GBT with the full Phase-3-survivor set; test formal regime conditioning via interaction with VIX or NFCI; compute Giacomini-Rossi fluctuation test for time-varying predictive ability.

**Phase 5 — robustness, promotion, and deployment (2 weeks, if Phase 4 passes)**: full CPCV with DSR and PBO; Romano-Wolf stepdown across 8 benchmarks; production promotion gate check; write final feature documentation; deploy with monitoring.

**Total research budget**: if Phase 2 passes, 7-9 weeks. If Phase 2 fails, 3-4 weeks and terminate. **Pre-committed stop at Phase 2 is the most important element of this roadmap.**

---

## 14. Concrete first-test queue

| # | Feature | AV endpoint (compute locally preferred) | time_period / params | Ticker(s) | Benchmarks most helped | Expected direction | Model first | Citation |
|---|---|---|---|---|---|---|---|---|
| 1 | `log(P_PGR/P_ETF)` 6M return on ratio | Compute locally from `TIME_SERIES_MONTHLY_ADJUSTED` | period=6 monthly | Ratio PGR/{VOO, VXUS, VWO, VDE} | All equity ETFs | Positive | Firth logistic, then Ridge | Jegadeesh–Titman 1993; Moskowitz–Ooi–Pedersen 2012 |
| 2 | `(C − EMA_252)/EMA_252` on ratio series | Local `EMA` from monthly adjusted | time_period=12 monthly (≈252 daily) | Ratio PGR/{VOO, VXUS, VWO} | VOO, VXUS, VWO | Positive | Firth logistic | Han–Zhou–Zhu 2016; NRTZ 2014 |
| 3 | `NATR_14` interaction with ROC_6M | Local from `TIME_SERIES_DAILY_ADJUSTED` (or compute from monthly as `std(mo returns, 12)/C`) | time_period=14 daily or 12 monthly | PGR | All equity ETFs via interaction | Positive interaction coefficient | GBT first; Ridge with explicit `NATR × ROC` | Gu–Kelly–Xiu 2020; Han–Yang–Zhou 2013 |
| 4 | `%B` with BBANDS(monthly,6,2) on ratio | Local `BBANDS` from `TIME_SERIES_MONTHLY_ADJUSTED` | time_period=6, nbdev=2 monthly | Ratio PGR/{VOO, VXUS} | VOO, VXUS | Negative at extremes (overbought → underperformance) | Firth logistic (threshold features natural) | Bollinger/mean-reversion literature; no direct OOS evidence — genuinely exploratory |
| 5 | PC-TECH VOO composite | Local: PCA on {EMA_252 dev, ROC_126, NATR, %B, ADX, ADOSC} for VOO | See components | VOO only | VOO, VXUS, VWO, VDE | Positive in recessions | Ridge with regime interaction | Neely–Rapach–Tu–Zhou 2014 *Management Science* |
| 6 | Peer-relative 6M momentum | Local: `log(P_PGR / geomean(P_ALL,P_TRV,P_CB,P_HIG))` 6M return | period=6 monthly | PGR vs insurance peer composite | New 9th "insurance-peer" benchmark | Positive | Firth logistic | Novel; no direct citation |
| 7 | `(ADX_14 > 25) × ROC_126` regime gate | Local `ADX` from daily adjusted | ADX time_period=14 daily | PGR | Equity ETFs | Positive interaction | GBT | Wilder 1978; no formal OOS paper |
| 8 | ADOSC normalized | Local `ADOSC` from daily adjusted | fast=3, slow=10 daily; EOM snapshot | PGR | Equity ETFs | Positive (accumulation) | Ridge | Neely–Rapach–Tu–Zhou 2014 (OBV loaded on PC-TECH) |

---

## 15. Bottom-line recommendation

**(1) Is TA worth pursuing given the constraints?** **Qualified yes — as a two-week, pre-registered, Tier-1-only test with a hard stop gate, not as an open-ended research program.** The prior is negative: PGR is the worst sector profile for TA (low-vol, large-cap, heavily covered, institutionally owned), the existing baseline already contains momentum and realized vol (which capture most survivable TA signal per Gu-Kelly-Xiu 2020), and the N_eff ≈ 19 sample size gives brutal statistical power under honest multiple-testing. But the Neely-Rapach-Tu-Zhou 2014 PC-TECH result and the ratio-series momentum hypothesis are specific enough that one disciplined test is worth doing before abandoning TA as a direction.

**(2) Most promising indicator families for incremental 6M relative-return value**: (a) **Multi-horizon momentum applied to the ratio series `P_PGR/P_ETF_k`** — theoretically cleanest and subsumes trend/MA; (b) **PC-TECH composite on VOO** as a macro-regime signal à la NRTZ 2014; (c) **NATR as a vol-regime interaction conditioner** — not standalone alpha but theoretically strong interaction with momentum (Han-Yang-Zhou 2013, Gu-Kelly-Xiu 2020). Everything else is Tier 2 or 3 — test only if Phase 2 succeeds.

**(3) Target formulation most likely to reveal TA signal**: **Binary classification via Firth penalized logistic regression of `sign(R_PGR − R_ETF) at t+6`**. The user's own diagnostic (IC 0.19, hit rate 69%, R² −13%) proves directional skill exists with magnitude miscalibration — classification preserves skill, regression penalizes calibration error TA cannot fix. Keep regression as secondary for magnitude-based sizing.

**(4) Single most important methodological recommendation**: **Pre-register the Tier 1 feature list, Clark-West threshold (t > 2.0), and BH/BHY q-level in a timestamped Git commit *before* running OOS evaluation.** Without pre-commitment, the 92% probability of at least one false positive across 50 tests guarantees you will find something that looks significant and be unable to honestly distinguish it from noise. The user's existing v37-v158 research cycle has already absorbed substantial specification search; one more cycle without pre-registration is not an independent test.

**(5) Empirical results that should trigger abandoning TA**: stop and redirect effort if any of the following occur in Phase 2: **(a)** no Tier 1 feature achieves Clark-West t > 2.0 HAC-adjusted on ≥3 of 8 benchmarks; **(b)** Campbell-Thompson ΔCE is negative or under 25 bp/year at γ=3 for the best-performing feature; **(c)** adding TA features degrades pooled OOS R² or hit rate on ≥5 of 8 benchmarks; **(d)** regime subsample R²_OOS is catastrophically negative (< −1%) in any of the three pre-specified regimes; **(e)** PBO via CPCV exceeds 0.4. Any of these outcomes confirms the external peer-review conclusion that the predictive stack's weakness is structural rather than architecturally fixable by adding features — at which point the right research direction is (i) non-price signal sources (consensus earnings revisions via existing peer data, tax/vesting schedule features specific to the user's decision, macro-regime classification as a standalone model), (ii) accepting the model as a low-resolution directional tool and designing the decision-support UI around that limitation rather than demanding higher statistical precision, or (iii) reformulating the target to something inherently more predictable (12M horizon despite power cost, or simple outperform/underperform binary with no magnitude claim).

**The honest meta-conclusion**: this research will most likely produce a negative result. That is not a failure — it is a clean, defensible answer to a well-posed question, which is more valuable than a false-positive promoted feature that degrades live performance. The explicit stop rules and pre-registration are what convert "TA didn't help" from an inconclusive disappointment into a decisive finding that closes a direction and frees research attention for more promising ones.