# PGR v15 feature replacement candidates for relative-return forecasting

**The most promising swaps target four weak features — `vmt_yoy`, `yield_curvature`, `mom_3m`, and `nfci` — replacing them with rate-adequacy gap, peer-relative momentum, breakeven inflation, and trade-weighted USD signals that carry stronger economic grounding and broader benchmark-predictive power.** The v14 ensemble of Ridge + GBT failed to clearly beat the historical-mean baseline, a problem rooted in low signal-to-noise and features that either overlap (vix/nfci/credit_spread_hy all measure risk appetite) or lack a clear causal mechanism for PGR-specific relative returns (vmt_yoy). The candidates below were selected by cross-referencing insurance-sector fundamentals, cross-asset macro research, price-based feature literature (Gu, Kelly & Xiu 2020; Jegadeesh & Titman 1993; Barroso & Santa-Clara 2015), and Ridge-vs-GBT engineering best practices. Every candidate is freely sourceable, monthly or faster, and economically interpretable.

---

## Which current features are weakest and most replaceable

Before adding anything, the existing feature set needs triage. Based on evidence strength, overlap with other features, and match to model type, these are the replacement priorities:

**Tier 1 — Replace first.** `vmt_yoy` is the single weakest feature. Vehicle miles traveled carries a 2-month reporting lag, has an ambiguous net direction (more driving raises both claims frequency *and* premium volume), and the academic link to insurance *stock returns* is thin. `yield_curvature` ranks next: while yield slope (T10Y2Y) is one of the strongest documented predictors of financial-sector relative returns, the curvature term (2×GS5 − GS2 − GS10) has limited empirical support for predicting either PGR alpha or cross-asset relative returns. `mom_3m` is the third weakest — the 3-month lookback sits in the "no man's land" between short-term reversal (1 month) and medium-term momentum (6–12 months), and for a large-cap liquid stock like PGR, the signal is noisy.

**Tier 2 — Replace if budget allows.** `nfci` overlaps heavily with `credit_spread_hy` and `vix` (all three capture financial stress / risk appetite). Fed Board research (FEDS 2013/39) warns that NFCI's predictive power for equity returns is "weak unless the financial crisis period is included." Among the Ridge-only fundamentals, `investment_income_growth_yoy` could be replaced by a more mechanistic feature (book yield gap or rate-adequacy gap), and `buyback_acceleration` (BayesianRidge) has weak empirical backing.

**Tier 3 — Keep.** `mom_6m`, `mom_12m`, `vol_63d`, `yield_slope`, `real_rate_10y`, `credit_spread_hy`, `vix`, `combined_ratio_ttm`, and `roe_net_income_ttm` all have strong theoretical grounding and empirical support. Some benefit from minor re-engineering (e.g., converting `mom_12m` to skip-month format) rather than outright replacement.

---

## Ranked candidate replacement features

### 1. Rate adequacy gap

| Field | Detail |
|---|---|
| **Definition** | YoY Δ(CPI motor vehicle insurance) minus weighted average of YoY Δ(CPI used cars, CPI medical care, CPI motor vehicle repair). Positive = rate increases outpacing loss-cost inflation. |
| **Source** | FRED: `CUSR0000SETA10`, `CUSR0000SETA02`, `CPIMEDSL`, `CUSR0000SETD`. Monthly, ~2 weeks lag. |
| **Direction** | Positive gap → PGR combined ratio improving → PGR outperforms all 8 benchmarks. |
| **Replaces** | `vmt_yoy` in Group B (all models). |
| **Implementation** | Easy. Four FRED pulls, simple arithmetic. Z-score for Ridge; raw for GBT. |
| **Rationale** | Directly captures the pricing-vs-cost dynamic that is *the* primary driver of P&C profitability. The 78% historical correlation between CPI insurance and rate adjustments is well documented. Unlike VMT, this feature has an unambiguous directional link to insurance stock returns and is available with shorter lag. |
| **Model fit** | **Both Ridge and GBT.** Approximately linear in Ridge; GBT can capture threshold effects (e.g., gap turning negative signals margin compression). |

### 2. PGR peer-relative momentum (6 month)

| Field | Detail |
|---|---|
| **Definition** | PGR cumulative 6-month return minus equal-weighted 6-month return of ALL, TRV, CB, HIG. |
| **Source** | Daily price data for 5 stocks. Free from Yahoo Finance or SEC. |
| **Direction** | Positive → continued PGR outperformance (within-industry momentum). Da, Liu & Schaumburg (2013) show industry-relative returns are "significantly more powerful predictors of future returns than raw returns." |
| **Replaces** | `mom_3m` in Group B. |
| **Implementation** | Easy. Monthly computation from daily closes. |
| **Rationale** | Isolates PGR-specific information (pricing advantage, telematics edge, market-share gains) from shared industry factors (rates, CAT losses, macro). Residual/industry-relative momentum has half the volatility of raw momentum with no decrease in return (Blitz et al. 2011). Directly addresses the relative-return prediction task. |
| **Model fit** | **Both.** Linear momentum effect for Ridge; GBT captures regime-dependent strength. |

### 3. Breakeven inflation (10Y)

| Field | Detail |
|---|---|
| **Definition** | 10-year Treasury breakeven inflation rate (`T10YIE`), used as level or 3-month change. |
| **Source** | FRED: `T10YIE`. Daily. |
| **Direction** | Rising breakevens → PGR underperforms GLD, DBC, VDE (inflation hedges rally); mixed vs. BND (both hurt, but insurance can reprice); underperforms VOO if margins lag inflation. |
| **Replaces** | `yield_curvature` in Group B. |
| **Implementation** | Trivial. Single FRED series. |
| **Rationale** | Inflation expectations are the key missing macro dimension. The current set has yield *shape* (slope, curvature) and yield *level* (real_rate_10y) but no inflation expectations signal. For the benchmark universe, breakevens predict gold/commodity relative returns (rising breakevens favor GLD, DBC over BND/VMBS) far more directly than curvature does. For PGR specifically, inflation drives claims costs — the rate-adequacy gap (feature #1) captures this from the insurance side, while breakevens capture the macro side. |
| **Model fit** | **Both.** Nearly linear relationship with gold and bond returns in Ridge. GBT can capture regime thresholds (e.g., breakevens above 3% triggering different dynamics). |

### 4. Trade-weighted USD (broad)

| Field | Detail |
|---|---|
| **Definition** | 3-month change in the Federal Reserve broad trade-weighted US dollar index. |
| **Source** | FRED: `DTWEXBGS`. Daily. |
| **Direction** | USD strengthening → PGR outperforms VXUS, VWO, GLD, DBC (all inversely correlated with USD). Correlation of **−0.40** between MSCI EM and DXY monthly returns (Marquette Associates). PGR is 100% domestic revenue with zero FX translation risk. |
| **Replaces** | `nfci` in Group B. |
| **Implementation** | Trivial. Single FRED series, 3-month log difference. |
| **Rationale** | NFCI captures financial stress, but `credit_spread_hy` and `vix` already cover that channel with stronger evidence. The USD fills an entirely different and critical gap: **it is the single strongest predictor of PGR's relative performance versus international equity (VXUS, VWO), gold (GLD), and commodity (DBC) benchmarks** — four of the eight benchmark assets. In 2025, the dollar fell 9% and EM stocks rose 25.6%. No other feature in the current set captures cross-currency dynamics. |
| **Model fit** | **Both.** The PGR-vs-international relationship is approximately linear (Ridge). GBT can capture non-linear regime effects (e.g., dollar strength above a threshold triggering EM crisis dynamics). |

### 5. Combined ratio acceleration

| Field | Detail |
|---|---|
| **Definition** | Second derivative of the trailing 12-month combined ratio: Δ(CR_ttm) − Δ(CR_ttm, 3 months ago). Positive = deterioration accelerating; negative = improvement accelerating. |
| **Source** | PGR monthly 8-K filings (combined_ratio field). 256 rows, 2004–2026. |
| **Direction** | Negative acceleration (improvement speeding up) → strongly positive for PGR. Swiss Re data shows combined ratio improved to 89% in Q3 2025 (lowest since 2001), but forecasts deterioration to 97% by 2026 — detecting this inflection early is the feature's purpose. |
| **Replaces** | Can replace `combined_ratio_ttm` in GBT (which captures threshold effects on raw CR level already); or serves as an *upgrade* to `combined_ratio_ttm` in Ridge (where the change signal is more linear than the level). |
| **Implementation** | Easy. Monthly 8-K data already available; two subtractions. |
| **Rationale** | The combined ratio *level* matters, but the *direction and acceleration* drive stock returns more at the margin (Eling & Schmit 2018). Markets price the trend. An accelerating improvement cycle signals hard-market pricing power; decelerating improvement is the early warning of cycle peak. This is the single highest-alpha PGR-specific signal. |
| **Model fit** | **Primarily Ridge.** The linear change-of-change signal is well-suited for Ridge. For GBT, keep the raw CR level instead (tree splits capture thresholds at 96%, 100% naturally). |

### 6. Policies-in-force growth (YoY)

| Field | Detail |
|---|---|
| **Definition** | YoY percentage change in PGR total policies-in-force from monthly 8-K data. |
| **Source** | PGR monthly 8-K (pif_total column). |
| **Direction** | Accelerating PIF growth → positive for PGR. PIF growth leads premium growth by 1–3 months and is the strongest market-share signal. PGR reached 38.9M policies in January 2026 (+10% YoY). |
| **Replaces** | `investment_income_growth_yoy` in Ridge/BayesianRidge. |
| **Implementation** | Easy. Already in the 8-K dataset. |
| **Rationale** | PIF growth is uniquely available monthly only for PGR — no major peer provides it at this frequency. It is a **leading indicator** of premium growth, retention quality, and competitive positioning. Investment income growth is a lagging indicator driven primarily by rate levels (already captured by `real_rate_10y` and `yield_slope`). Replacing a lagging, redundant fundamental with a leading, unique one should improve forecast quality. |
| **Model fit** | **Both Ridge and GBT.** Approximately linear for Ridge. GBT can detect interactions (e.g., PIF growth + stable CR = very bullish; PIF growth + rising CR = bearish). |

### 7. Volatility-adjusted relative momentum (PGR vs VOO, 12-1 skip-month)

| Field | Detail |
|---|---|
| **Definition** | PGR 12-1 month cumulative return minus VOO 12-1 month return, divided by the trailing 63-day standard deviation of the daily PGR−VOO return spread. |
| **Source** | Daily prices for PGR and VOO. |
| **Direction** | Positive → continued relative outperformance. Barroso & Santa-Clara (2015) show risk-managing momentum nearly doubles the Sharpe ratio. Zaremba's VARMOM yields 2–3× higher Sharpe than raw momentum. |
| **Replaces** | `mom_12m` (raw 12-month momentum) in Group B. |
| **Implementation** | Low. Requires daily returns; rolling vol computation is standard. |
| **Rationale** | The current `mom_12m` is absolute PGR momentum — it does not distinguish PGR-specific signal from broad market moves and does not account for risk. Volatility-adjusting and computing relative to VOO directly targets what the model predicts (PGR-vs-benchmark) and produces a more stationary, stable signal. The skip-month design avoids 1-month reversal contamination. |
| **Model fit** | **Primarily Ridge.** The Z-score normalization makes it an ideal linear feature. GBT can use this or the raw relative momentum interchangeably. |

### 8. CPI used cars & trucks (YoY change)

| Field | Detail |
|---|---|
| **Definition** | Year-over-year percentage change in CPI for used cars and trucks. |
| **Source** | FRED: `CUSR0000SETA02`. Monthly. |
| **Direction** | Rising used car CPI → negative for PGR (higher total-loss claim payouts). 78% historical correlation between used car prices and insurance rate adjustments. Average total-loss payout rose from ~$17,500 (2023) to ~$22,600 (2025). |
| **Replaces** | Can replace `vmt_yoy` if rate-adequacy gap (#1) is not used, or serves as an alternative/component. Also a candidate for replacing `buyback_acceleration` in BayesianRidge. |
| **Implementation** | Trivial. Single FRED pull. |
| **Rationale** | Used car prices are the **primary claims severity driver** for physical damage coverage — PGR's largest expense category. This is a more direct, causal link to PGR profitability than VMT. Also has cross-asset relevance: used car CPI captures inflation dynamics that affect the BND/VMBS vs. equity relative return (inflation surprise hurts bonds more than insurers who can reprice). |
| **Model fit** | **Both.** Linear cost-pressure signal for Ridge. GBT captures threshold effects (e.g., used car deflation flipping to inflation). |

### 9. PGR relative volatility ratio

| Field | Detail |
|---|---|
| **Definition** | Ratio of PGR trailing 63-day realized volatility to VOO trailing 63-day realized volatility. |
| **Source** | Daily prices for PGR and VOO. |
| **Direction** | High ratio (PGR vol elevated vs. market) → PGR tends to underperform near-term (IVOL puzzle; Ang et al. 2006). Low ratio → PGR is behaving as a defensive, stable compounder → tends to outperform. |
| **Replaces** | `vol_63d` (absolute PGR volatility) in Group B. |
| **Implementation** | Easy. Standard rolling-window calculation. |
| **Rationale** | Absolute volatility conflates market-wide vol with PGR-specific vol. The ratio isolates PGR's *relative* risk profile. When PGR vol is low relative to the market, it signals the stock is in "defensive compounder" mode — exactly when P&C insurers tend to outperform diversified benchmarks. Gu, Kelly & Xiu (2020) identify volatility as a top-3 predictive category; the ratio form is more informative for relative-return prediction. |
| **Model fit** | **Primarily GBT.** The IVOL-return relationship is non-linear and conditional (Zhu et al. 2023 show it depends on fundamental quality). GBT captures these interactions naturally. For Ridge, use the log ratio for better distributional properties. |

### 10. Book value per share growth (YoY)

| Field | Detail |
|---|---|
| **Definition** | Year-over-year percentage change in PGR book value per share from monthly 8-K. |
| **Source** | PGR monthly 8-K (book_value_per_share column). |
| **Direction** | Positive BVPS growth → positive for PGR. P/B is the primary valuation anchor for insurance stocks. PGR's justified P/B = (ROE − g)/(r − g); with 40% comprehensive ROE in 2025, consistent BVPS growth compounds intrinsic value. |
| **Replaces** | `buyback_yield` in BayesianRidge. More comprehensive than buyback yield because it captures *all* capital creation — retained earnings, unrealized investment gains, and capital actions combined. |
| **Implementation** | Easy. Already in the 8-K dataset. |
| **Rationale** | BVPS growth is a higher-quality, more comprehensive signal than buyback yield. It reflects the net effect of operating profitability, investment returns, and capital allocation. Monthly BVPS is rare — PGR is virtually the only major insurer providing it — creating an information advantage. |
| **Model fit** | **Both Ridge and GBT.** Linear compounding signal for Ridge. GBT can detect regime shifts (e.g., BVPS declining from unrealized losses triggers different dynamics). |

### 11. Insurance sector relative momentum vs. broad financials

| Field | Detail |
|---|---|
| **Definition** | 6-month return of equal-weighted P&C basket (PGR, ALL, TRV, CB, HIG) minus 6-month return of XLF (Financial Select Sector SPDR). |
| **Source** | Daily prices for 5 insurance stocks + XLF. |
| **Direction** | Positive → P&C insurance outperforming banks and diversified financials → favorable underwriting cycle → PGR benefits. Industry momentum persists up to 12 months (Moskowitz & Grinblatt 1999). |
| **Replaces** | `nfci` (if USD in #4 is not used) or adds as a GBT-specific feature replacing `yield_curvature`. |
| **Implementation** | Low. Standard price-return calculations. |
| **Rationale** | Captures the **underwriting cycle position** in price-derived form. When P&C insurers outperform banks, it signals hard-market profitability that is likely to persist. This is a market-consensus signal about insurance cycle positioning, complementing the accounting-based combined ratio signal. |
| **Model fit** | **Both.** Linear industry momentum for Ridge; GBT captures interactions with individual stock momentum. |

### 12. NPW growth rate (YoY, trailing 12 months)

| Field | Detail |
|---|---|
| **Definition** | Year-over-year growth in PGR net premiums written, trailing 12-month sum. |
| **Source** | PGR monthly 8-K (net_premiums_written column). |
| **Direction** | Moderate growth with stable CR → positive. Rapid growth with deteriorating CR → negative. The CAS "Optimal Growth" paper shows maximum sustainable growth of ~5.5% at typical leverage — growth above this requires either capital injection or leverage expansion. |
| **Replaces** | `underwriting_income` in ElasticNet (NPW growth is a leading indicator of underwriting income). Or `buyback_acceleration` in BayesianRidge. |
| **Implementation** | Easy. Already in the 8-K dataset. |
| **Rationale** | Premium growth is the top-line signal — PGR added almost $9B in NWP in 2025 (+18% YoY) while maintaining sub-90 combined ratios, a combination that is "exceptionally bullish and rare" per Swiss Re analysis. The interaction of NPW growth × combined ratio is the strongest two-variable signal in P&C insurance stock prediction. |
| **Model fit** | **Both.** Linear growth signal for Ridge. GBT naturally discovers the growth × profitability interaction. |

### 13. Net premiums written growth minus PIF growth (pricing signal)

| Field | Detail |
|---|---|
| **Definition** | YoY NPW growth minus YoY PIF growth. Positive = rate increases (premium per policy rising); negative = rate moderation or cuts. |
| **Source** | PGR monthly 8-K (derived from NPW and PIF fields). |
| **Direction** | Positive (pricing power) → positive for PGR margins. When NPW growth exceeds PIF growth, PGR is successfully raising prices. When the gap narrows or turns negative, it signals competitive pressure or deliberate rate cuts. |
| **Replaces** | Could replace `investment_income_growth_yoy` as an alternative to PIF growth (#6). |
| **Implementation** | Easy. Two 8-K fields, simple arithmetic. |
| **Rationale** | This decomposition separates *volume* growth from *price* growth — a distinction that is critical for forecasting margin sustainability. Rising prices are immediately margin-positive; rising volume is positive only if loss ratios on new business are adequate (which they typically aren't in the first renewal cycle per Nissim 2010). |
| **Model fit** | **Both.** Clean linear signal for Ridge. GBT can use the raw decomposition. |

### 14. Short-term relative reversal (1 month, PGR vs peers)

| Field | Detail |
|---|---|
| **Definition** | PGR's most recent 1-month return minus insurance peer average 1-month return. Used with a **negative** sign expectation (reversal). |
| **Source** | Daily prices for PGR + 4 peers. |
| **Direction** | Negative (reversal). Within-industry reversal generates **1.20% per month** (t = 5.87) per Da, Liu & Schaumburg (2014). However, for large-cap liquid stocks, Medhat & Schmeling (2022) find short-term *momentum* — so the sign is uncertain and GBT can learn the conditional relationship. |
| **Replaces** | Could replace `mom_3m` as an alternative to peer-relative momentum (#2). |
| **Implementation** | Easy. Monthly returns for 5 stocks. |
| **Rationale** | The current feature set has medium-term momentum (6m, 12m) but no short-term signal designed to capture mean reversion. This fills a gap. The key insight is that short-term *industry-relative* return contains more signal than short-term *absolute* return. |
| **Model fit** | **GBT only.** The conditional direction (reversal for illiquid stocks, momentum for liquid stocks like PGR) makes this unsuitable for Ridge. GBT can learn the correct relationship. |

### 15. Excess bond premium proxy

| Field | Detail |
|---|---|
| **Definition** | Residual from regressing HY credit spread (`BAMLH0A0HYM2`) on macro fundamentals (unemployment, IP growth, VIX). Captures the investor-risk-appetite component of credit spreads, isolating it from default risk. |
| **Source** | FRED: `BAMLH0A0HYM2`, `UNRATE`, `INDPRO`, `VIXCLS`. Monthly. Alternatively, use the Gilchrist-Zakrajšek excess bond premium directly if available from the Federal Reserve Board data releases. |
| **Direction** | Rising EBP → strongly negative for PGR vs. GLD and BND (flight to safety). Gilchrist & Zakrajšek (2012, AER) show EBP "leads to declines in economic activity and asset prices" and has predictive power exceeding the term spread for recessions. |
| **Replaces** | `credit_spread_hy` (upgrades it by extracting the more predictive component). Or `nfci` if both are in the feature set. |
| **Implementation** | Medium. Requires rolling regression to extract the residual, or sourcing the Fed data directly. |
| **Rationale** | Raw HY credit spread confounds two signals: (1) expected default losses (captured by economic fundamentals already in the model via real_rate_10y, yield_slope) and (2) risk-bearing capacity of the financial sector (the true predictive component). The EBP isolates signal #2, which is the portion that actually predicts asset returns. This is one of the strongest findings in the macro-finance prediction literature. |
| **Model fit** | **Both.** The EBP has a relatively linear negative relationship with subsequent equity returns (Ridge-friendly). GBT can capture non-linear crisis dynamics. |

### 16. Relative drawdown depth (PGR vs VOO)

| Field | Detail |
|---|---|
| **Definition** | Current PGR/VOO price ratio divided by its trailing 12-month maximum, minus 1. Ranges from 0 (at peak relative performance) to large negative values (deep relative underperformance). |
| **Source** | Daily prices for PGR and VOO. |
| **Direction** | Deep drawdowns predict recovery (contrarian/mean-reversion). Kim et al. (2021) show drawdown-ranked portfolios outperform momentum portfolios in both direction forecasting and return capture. |
| **Replaces** | Candidate for GBT feature set, potentially replacing `yield_curvature` or `nfci`. |
| **Implementation** | Easy. Rolling max of price ratio. |
| **Rationale** | Fills the mean-reversion gap in the current feature set, which is pure momentum-driven. PGR drawdowns relative to VOO often result from transitory events (catastrophe quarters, one-time reserve charges) that subsequently reverse. This feature gives GBT a contrarian signal to balance the momentum signals. |
| **Model fit** | **GBT only.** The relationship is highly non-linear — shallow drawdowns are noise, only deep drawdowns are informative. Trees handle this threshold effect naturally. Not suited for Ridge. |

### 17. Direct channel PIF share (trailing 12 months)

| Field | Detail |
|---|---|
| **Definition** | Direct auto PIF as a percentage of total auto PIF, or YoY change in this share, from PGR monthly 8-K channel breakdown. |
| **Source** | PGR monthly 8-K (channel_mix or segment PIF fields). |
| **Direction** | Rising direct share → structurally lower expense ratio → positive for PGR. PGR's direct auto grew 14% YoY in January 2026 while agency also grew strongly — dual-channel growth signals broad competitive advantage. |
| **Replaces** | `buyback_acceleration` in BayesianRidge. |
| **Implementation** | Easy. Already in 8-K dataset. |
| **Rationale** | Channel mix is a slow-moving structural signal that captures PGR's long-term competitive moat evolution. Direct distribution has lower customer acquisition costs and higher lifetime value. This feature is orthogonal to the cyclical signals (momentum, macro) and provides a genuine PGR-specific alpha signal not available for any peer. |
| **Model fit** | **Ridge only.** The relationship is approximately linear and slow-moving — suited for Ridge's ability to capture persistent, low-frequency signals. Too slow-moving to help GBT at monthly prediction horizons. |

---

## Specific Ridge vs GBT feature set recommendations

The fundamental difference: **Ridge needs linear, normalized, pre-computed features; GBT needs raw levels with threshold effects and interaction potential.** With feature budgets of ~14 (Ridge) and ~11 (GBT), each slot must earn its place.

### Proposed v15 GBT feature set (~11 features)

| Slot | Current (v14) | Proposed (v15) | Change rationale |
|---|---|---|---|
| 1 | mom_3m | **pgr_peer_relative_mom_6m** | Industry-relative signal isolates PGR alpha; 6m better than 3m lookback |
| 2 | mom_6m | mom_6m | **Keep.** Well-supported horizon |
| 3 | mom_12m | **vol_adjusted_relative_mom_12_1** | Skip-month + vol-adjustment + relative to VOO = strictly better signal |
| 4 | vol_63d | **vol_ratio_pgr_mkt** | Relative vol isolates PGR-specific risk; IVOL puzzle is non-linear → ideal for GBT |
| 5 | yield_slope | yield_slope | **Keep.** Strongest single macro predictor for financial-sector relative returns |
| 6 | yield_curvature | **breakeven_inflation_10y** | Breakevens predict gold/commodity/bond relative returns far better than curvature |
| 7 | real_rate_10y | real_rate_10y | **Keep.** Very strong inverse relationship with GLD; positive for PGR investment income |
| 8 | credit_spread_hy | credit_spread_hy | **Keep.** Strong evidence. Consider upgrading to EBP proxy in v16 |
| 9 | nfci | **usd_broad_3m_chg** | USD captures 4 of 8 benchmark assets' relative return dynamics; NFCI is redundant with vix + credit spread |
| 10 | vix | vix | **Keep.** Broad risk-regime indicator; partially redundant but GBT can use it for interaction splits |
| 11 | vmt_yoy | **rate_adequacy_gap** | Direct causal link to PGR profitability vs. ambiguous VMT signal with 2-month lag |

Net changes: **4 replacements.** The new set maintains 3 momentum signals (now all relative/adjusted), adds inflation and USD dimensions, and introduces the rate-adequacy gap as the insurance-specific fundamental in the GBT set.

### Proposed v15 Ridge feature set (~14 features)

| Slot | Current (v14) | Proposed (v15) | Change rationale |
|---|---|---|---|
| 1–11 | Group B (same as GBT above) | v15 GBT set above | Same base with same 4 swaps |
| 12 | combined_ratio_ttm | **combined_ratio_acceleration** | Ridge benefits from the linear change-of-change signal rather than the non-linear level |
| 13 | investment_income_growth_yoy | **pif_growth_yoy** | Leading indicator replaces lagging indicator; unique monthly PGR signal |
| 14 | roe_net_income_ttm | roe_net_income_ttm | **Keep.** INS5 model identifies ROE as a priced factor for insurance stocks (Eling & Schmit 2018) |

For **BayesianRidge** (currently has buyback_yield and buyback_acceleration on top): replace both with **bvps_growth_yoy** and **npw_growth_yoy**. BVPS growth is a strictly more comprehensive capital-creation signal than buyback yield, and NPW growth provides the top-line signal that buyback acceleration cannot.

---

## Benchmark-predictive features for the reduced universe

Several candidate features specifically help forecast the *benchmark* side of the relative return (predicting where VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE are heading), which is half the prediction problem:

**Trade-weighted USD** (candidate #4) is the single most powerful benchmark-predictive addition. Dollar strength directly and negatively predicts VXUS returns, VWO returns (−0.40 correlation), GLD returns, and DBC returns. It also indirectly helps predict VOO (40% international revenue exposure creates FX headwind) and VDE (oil is dollar-denominated). This one feature carries information about **6 of 8 benchmarks**.

**Breakeven inflation** (candidate #3) predicts the GLD-vs-BND relative return (rising breakevens favor gold, hurt bonds), DBC (commodity demand-inflation link), and VDE (energy as inflation hedge). It also predicts VMBS (mortgage rates rise with inflation expectations, hurting MBS).

**Real rate 10Y** (already in set — keep) is the strongest single predictor of GLD returns (strong inverse relationship), and predicts BND/VMBS returns (rising real rates hurt bonds).

**Credit spread HY** (already in set — keep) predicts the equity-vs-bond relative return and the risk-on/risk-off regime that drives VOO vs. BND/GLD.

**VIX** (already in set — keep) predicts the gold-vs-equity and bond-vs-equity relative return during stress episodes.

The current set's weakness is **no USD signal and no inflation expectations signal** — the two additions that most directly target the cross-asset benchmark prediction problem.

---

## How feature types map to model architecture

The research literature provides clear guidance on which features belong in which model:

**Features with linear, monotonic relationships** belong in Ridge: momentum (all horizons), yield slope changes, rate-adequacy gap, PIF growth, NPW growth, BVPS growth, ROE, combined ratio *acceleration* (change-of-change). These features should be **z-scored using an expanding window** and winsorized at 3σ to reduce outlier impact. Ridge coefficients should be **sign-constrained** to match economic priors (Campbell & Thompson 2008 show this significantly improves out-of-sample performance).

**Features with threshold effects, non-monotonic relationships, or conditional interactions** belong in GBT: combined ratio *level* (100% is a regime break), volatility ratio (IVOL puzzle is conditional on fundamental quality), relative drawdown depth (only informative at extremes), short-term reversal (direction depends on liquidity context), and VIX (relationship with returns is non-linear). These features should be fed in **raw form** — no normalization needed since trees are scale-invariant.

**Features that work in both**: yield slope, real rate, credit spread, USD, breakeven inflation, peer-relative momentum. For Ridge, use changes or z-scores; for GBT, use levels.

A critical engineering point: **Ridge needs pre-computed interaction terms** that GBT discovers automatically. The two highest-priority interaction terms for Ridge are (1) **rate_adequacy_gap × combined_ratio_acceleration** (insurance pricing power × profitability trend) and (2) **yield_slope × pif_growth** (macro tailwind × company-specific momentum). These should be evaluated as potential v15.1 additions if the base swaps prove successful.

---

## Conclusion

The v15 replacement strategy centers on four high-confidence swaps in the shared Group B — `vmt_yoy` → rate adequacy gap, `mom_3m` → peer-relative momentum, `yield_curvature` → breakeven inflation, and `nfci` → trade-weighted USD — plus model-specific fundamental upgrades. The common thread is shifting from **generic, indirect features to relative, causal, benchmark-aware features**. The rate-adequacy gap directly captures insurance profitability dynamics. Peer-relative momentum isolates PGR-specific alpha. Breakeven inflation and USD address the biggest gap in the current set: predicting the benchmark side of the relative return, particularly for gold, commodities, and international equity. These changes preserve the feature budget, maintain economic interpretability, and should reduce the overlap between `credit_spread_hy`, `nfci`, and `vix` that currently wastes capacity on redundant risk-appetite signals. The path to beating the historical-mean baseline likely runs through features that the unconditional average cannot capture — regime transitions in insurance pricing, currency-driven cross-asset dynamics, and PGR-specific competitive signals that no generic macro model can replicate.