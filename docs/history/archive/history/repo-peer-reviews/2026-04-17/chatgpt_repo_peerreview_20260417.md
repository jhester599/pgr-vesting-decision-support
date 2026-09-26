# Executive Summary

The PGR vesting‐support project has made steady progress, but its core weakness remains low prediction confidence.  Past research shows that expanding feature sets broadly hurts performance【81†L6-L15】, and even intensive searches yield only marginal gains【84†L8-L16】【111†L138-L141】.  The next phase must therefore be tightly focused on the few high-impact, low‐complexity changes still untried.  Our review of the repo finds that **(a)** classification models show promise (especially a pooled BA edge), but calibration and stability remain open issues; **(b)** some targeted macro features (e.g. USD index, oil) and small-sample fixes have not yet been fully exploited; and **(c)** regression ensemble improvements are likely smaller and should follow classification improvements.  We recommend a mixed program starting with classification-first experiments.  Key steps will be adding a **Firth-penalized logistic** for very sparse benchmarks, and trying two single-feature enhancements (a *broad USD index* feature and an *oil-momentum* feature) on those sectors most likely to benefit.  Each experiment is modest in scope (re-using existing data pipelines), has clear hypotheses, and acceptance criteria tied to improved balanced accuracy or calibration.  We explicitly avoid any broad feature-sprawl or architectural overhaul.  The goal is to squeeze out measurable gains in BA and reliability, so that the shadow classifier can be considered for eventual promotion, all while keeping complexity minimal and transparent. 

# What the Repo’s Research History Already Proves

The repository’s archives make it clear that **broad feature expansion has failed** and only *targeted* tweaks yield gains.  The v88 feature sweep established that the original 12‐feature “lean_baseline” beat any larger curated bundle【81†L6-L15】.  A full 72‐feature benchmark‐specific search in v128 showed only *tiny* pooled gains (BA 0.5000→0.5016) albeit improved calibration【84†L8-L16】.  In practice, only four benchmarks switched models and only modestly: e.g. BND, DBC, VIG each gained a few BA points【84†L13-L20】.  Any large “all features” model performed worse.  In short, the data strongly favor *keeping models simple*.  

Classification has shown **some promise but also clear limits**.  A lean logistic per-benchmark (Path A) can beat random guessing on covered cases (pooled BA ≈0.63 with a reasonable threshold), outperforming the regression consensus on those points【84†L8-L16】.  A portfolio‐level composite classifier (Path B) further raised BA but at the cost of worse calibration【58†L578-L581】【58†L558-L562】.  The highest gains came on a few benchmarks: e.g. an exotic 2‐feature model on VGT briefly achieved BA>0.94【84†L18-L24】.  But that VGT signal failed a temporal stability audit【67†L28-L36】【67†L57-L63】, and was rightly dropped (VGT reverts to baseline).  The roadmaps and closeouts consistently note that **only a minority of tickers ever showed material improvement**, and the *overall* benefit was minimal【84†L8-L16】【111†L138-L141】.  In short, classification research has proven the *approach* viable but not yet robust; important issues like calibration, coverage gating, and small-sample stability remain.

Meanwhile, the regression ensemble has been heavily tuned (shrinkage, consensus blending, Kelly sizing, etc.), yielding high *hold‐out* accuracy on its small coverage (e.g. ~85% accuracy at ~12% coverage as of early 2026【11†L9-L17】).  However, it offers limited upside: the regression side has largely exhausted straightforward improvements (no simple model changes remain), and adding more regressors or broad new features carries complexity.  The change logs show recent wins (e.g. blended ensemble, correlation pruning) are already integrated【58†L531-L539】.  In other words, **regression is near its engineered limit**. 

Overall, the repo history tells us: **keep it lean**.  Classification can help, but only in narrow ways; we must avoid any broad “throw everything at it” expansions.  Instead, we should pursue the small number of **specific opportunities identified in backlog and roadmaps**: Firth logistics for thin data【111†L91-L99】, a dollar‐index feature【111†L94-L97】, an oil‐momentum feature【111†L95-L97】, and careful gating rules.  The roadmaps explicitly prioritize these (CLS-02, FEAT-01, FEAT-02)【111†L91-L99】, because past artifacts suggest they are the last low-hanging fruit.  Anything else (e.g. giant new feature sets or alternative model families) is either already tried or too complex relative to expected gain.  

# What Remains Unresolved

Despite extensive work, **prediction confidence is still insufficient**.  Our audit identifies these likely root causes:

- **Feature signal is weak.**  PGR vs ETF relative returns are inherently hard to predict, and the lean models have limited information.  Existing features (macroeconomic, valuation, PGR ops, etc.) only pushed BA slightly above random【84†L8-L16】.  It’s possible a few key signals have been overlooked or only partially used: e.g. dynamic FX exposure, energy prices, or any latent insurance‐cycle indicator.  If such signals exist in the current data, they haven’t been effectively found yet.  

- **Calibration and threshold issues.**  Even when balanced accuracy improved, calibration worsened【58†L578-L581】.  The classification layer currently uses fixed probability cut-offs (0.30/0.70), which may be suboptimal given the low event rate.  The Path B model gained raw accuracy but at the cost of probability reliability【58†L578-L581】【58†L558-L562】.  Until we fix calibration, the system cannot safely raise threshold to gain coverage without risking too many false calls.  In other words, decision-layer policy is hampered by mis-calibrated scores.  

- **Small-sample instability.**  For some benchmarks (e.g. newer or less volatile sectors like VDE, VMBS, DBC), there are very few sell signals, so logistic fits are highly variable.  The VGT case shows how a spurious two-feature combo can look great in one period and disappear in another【67†L28-L36】.  This instability lowers our confidence in any winner unless it’s convincingly stable.  The repo’s backlog already flags “short-history” benchmarks as a concern (CLS-02).  

- **Benchmark heterogeneity.**  Each ETF behaves differently (equity vs bond vs commodity sectors).  A single pooled model struggles; indeed, the roadmap notes only “a minority” of benchmarks had large gains from specialization【111†L138-L141】.  This heterogeneity means a one-size-fits-all fix isn’t possible—improvements must respect per-ticker differences without adding undue complexity.  

- **Decision-layer vs prediction-layer.**  Some recent changes (Black-Litterman tau tuning, Kelly size) improved the *mapping* from model score to portfolio decisions, but they don’t raise *predictive* accuracy directly.  Those efforts are mostly orthogonal to our goal: they reduce risk of false confidence in decisions, but we still need better P(yield vs hold).  In short, we’ve likely extracted the low-hanging fruit of policy calibration; now the bottleneck is the prediction layer itself.  

In summary, the **primary problem** is a lack of robust predictive signal in the current models, aggravated by calibration and sample-size issues.  Secondary problems include gating policy nuances, but until scores are both more accurate and better-calibrated, we cannot raise coverage in a confidence-preserving way.  The project must therefore focus on *mild feature/model tweaks that boost true signal*, rather than more policy juggling.

# Best Low-Complexity Accuracy Opportunities Still Open

Given the above, the most promising features to try next are those already in the data plumbing or very easy to add, which economic reasoning suggests could matter but have not been fully leveraged:

1. **Broad USD Index (DTWEXBGS) / USD momentum features.**  The roadmaps explicitly list a post-v128 USD feature search【111†L94-L97】.  The `v128_feature_inventory.csv` indicates that “usd_broad_return_3m” and “usd_momentum_6m” features exist in the matrix【109†L1-L9】 (likely derived from DTWEXBGS), but no model selectors chose them strongly.  However, currency moves can affect PGR’s relative performance (insurance payouts vs foreign exposure).  Complexity: *low* (the series is public and already partly present; we may only need to emphasize or re-scale it).  Potential signal: *medium* (hard to know, but backlog prioritized it for “benchmarks most likely to benefit”, likely BND, VXUS, etc).  If USD momentum helps classification on bond or international ETFs (or regression for BND, VWO/VXUS), it’s worth checking with a targeted search or inclusion.  

2. **WTI Oil Price Momentum.**  Already present in the matrix as “wti_return_3m”【97†L1-L9】.  Roadmap backlog (FEAT-02) calls for a focused test on commodity/energy ETFs (DBC, VDE)【111†L95-L98】.  Complexity: *very low* (feature exists).  Expected signal: *potentially modest but clear for those sectors*.  WTI is highly relevant for energy (VDE) and commodities (DBC), but may have been drowned out in the pooled search.  A small experiment (e.g. add WTI momentum just for VDE/DBC classifiers, or in regression for those) could yield a measurable BA lift or new “sell” triggers around oil cycles, without harming other tickers.  

3. **Term Premium / Rates Spreads.**  The 10-year term premium (“THREEFYTP10”) and mortgage yield spreads are already in the data and used.  But one variant not explicitly mentioned in results is a multi-month change in term premium or an extreme-threshold flag (beyond inclusion as a raw feature).  Complexity: *low* (already have term_premium_10y and mortgage_spread_30y_10y).  Possibly add a 3-month change.  These have conceptual relevance (rising term premium often signals equity declines) and have moderate availability (some nulls, but most data present【95†L1-L10】).  We can propose testing a short-horizon differencing of term_premium or gating logic (“if term premium jump >X, sell”) as a small augmentation.  

4. **Insurance-cycle features.**  Some insurance-specific signals (combined ratio, severity index, rate adequacy) are in use.  But one obvious missing piece is **motor-vehicle insurance CPI (CUSR0000SETE)** – the repo noted it was added in “research-only” macro series.  That series could be loaded to compute a “motor vehicle CPI yoy” feature; it has low cost to add and is insurance-specific.  Inventory shows the fallback “motor_vehicle_ins_cpi_yoy” feature with many missing values【99†L1-L10】, because they used an incomplete source.  Adding the correct series may give better quality and coverage.  Complexity: *low to moderate* (requires fetching one BLS series into the pipeline).  Upside: *unknown* – but insurance cost trends (auto insurance inflation) arguably drive PGR margins, so a cleaner CPI feature might improve predictive power.  This is lower priority than USD/oil but in the same vein of “insurance macro” features that the classification analysis found salient (severity_index etc.).  

5. **PGR Operations Ratios (extended features).**  Many of these already exist, but some may not be used to full effect.  For example, “net premiums written” growth (`npw_growth_yoy`) and PIF ratios are in inventory【88†L33-L42】【107†L82-L93】.  One under-tested candidate is **NPW per PIF** as a measure of new business growth.  The feature set includes `npw_per_pif_yoy`【107†L89-L93】.  Also, “channel mix” (direct vs agency) or acceleration of growth.  These are already in the lean and extended families.  The question is whether any such feature has been underused; broad searches imply they haven’t stood out.  We mention them only to note we likely shouldn’t expand further here (v88 showed adding “extended_ops” as a block hurt【81†L10-L15】).  So we suggest *no new PGR-ops features beyond existing ones*.  

6. **Equity Valuation Ratios.**  Features like P/E, P/B, and relative valuations exist (e.g. `pgr_vs_market_pe`, `equity_risk_premium` in inventory【88†L36-L44】【107†L72-L81】).  However, inventory shows many of these have sparse data or low prevalence, and previous broad tests showed “valuation” family hurt performance【81†L11-L15】.  Any deep dive here is likely low yield relative to complexity.  We would **not** prioritize these as new experiments (they are effectively deprioritized).  

In ranking by **signal-to-complexity**: *WTI momentum* and *Firth logistic* (see below) are very low effort with reasonable upside; *USD index* is next; *term premium tweaks* are easy but marginal; *insurance CPI* moderate lift; *other valuation/op features* we recommend dropping.  

# Paths Already Explored or to Deprioritize

Several ideas are already dead ends or mature enough to drop:

- **Broad feature expansion.**  The v88 sweep conclusively showed that adding whole families (inflation, context, etc.) *reduced* performance【81†L6-L15】.  Similarly, v128’s 72-feature search found almost no pooled BA gain【84†L8-L16】.  We should **drop** any plan to add many new features at once or try “all curated features”.  The inventory and results suggest diminishing returns beyond a small core set.

- **VGT-specific exotic model.**  The two-feature VGT classifier was a one-off that failed stability【67†L28-L36】【67†L57-L63】.  It has been retired.  Any further VGT-only pursuit is unlikely without new features or data.  

- **SCHD and new benchmarks.**  Adding SCHD as a classifier was studied but deferred (needs 185+ obs).  It remains a future item, not for now.  Similarly, adding any new tickers or heavy rebalancing of benchmark universe is out of scope for the next accuracy sprint.

- **Alternate model families for classification.**  Almost all recent efforts stayed with logistic models (Paths A/B).  The ensemble (via elastic-net consensus) was tried in v128 and gave small gains for some ETFs (BND, DBC, VIG).  But these are integrated as slight method tweaks rather than new architectures.  We see no need for a fresh model family (no evidence ensemble classifiers like GBT would help without big data or features).  

- **Decision-layer fixes that don’t boost accuracy.**  Many engineering tasks (Kelly sizing, Black-Litterman tau, correlation pruning) have been done【58†L531-L539】【111†L91-L99】.  These improved the *resource allocation and confidence scoring* but not core prediction quality.  They should be deferred if they distract from predictive experiments.  

- **More calibration sweeps.**  V127’s calibration (Platt/temperature) found no method that preserved BA【58†L558-L562】.  Unless a new insight arises, we should not revisit this until we get the features right.  

In summary, we should **deprioritize any broad or “shiny” ideas** already vetted, and **drop** those with proven negative or negligible impact (e.g. adding large feature bundles, new classifiers for context ETFs, further complex ensembling).  The focus now is narrow follow-up on items hinted by prior results (as the roadmap backlog suggests).

# Next-Phase Experiments (3–5, Ranked)

We propose the following tightly-scoped experiments, in priority order.  Each is a focused test that reuses existing data and code, with a clear hypothesis and pass/fail criteria.

1. **CLS-02: Firth Logistic for Small Samples (Classification).**  
   **Objective:** Improve stability for benchmarks with very limited data (thin sell-rate).  **Why now:** The backlog flags small-sample bias as a remaining issue【111†L91-L99】.  Benchmarks like DBC, SCHD (if SCHD is included), or any with <~100 training positives may benefit from a penalty that prevents infinite coefficients.  **Data/Features:** Use the existing lean 12 features for each benchmark; identify the ones with <X positive cases in training (we can define X=30).  **Hypothesis:** Replacing standard sklearn logistic (`class_weight='balanced'`) with a Firth-penalized logistic will raise covered balanced accuracy (BA_cov) and/or reduce calibration error on these few-benchmark cases.  **Upside:** *Moderate.* Reduces variance in rare-case models, likely improving held-out BA for those few ETFs.  **Complexity:** *Low to moderate.* Implementation via a statsmodels GLM with logit+firth (or use an existing library).  A few lines of code change.  **Acceptance Criteria:** For each targeted benchmark (e.g. DBC, maybe VDE or others with small n), Firth logistic should meet either: (a) ≥+0.02 BA_cov improvement over standard logistic, or (b) noticeably lower ECE (even if BA_cov steady).  Globally, pooled BA_cov should not decline.  **Stop Conditions:** If Firth yields no BA gain or causes overfitting on any (e.g. much worse accuracy), stop.  **Adopt vs Research:** This is research-only initially (shadow test).  If successful, integrate into classification shadow (replace Path A solver for small benchmarks).

2. **FEAT-01: USD Index Momentum (Classification & Regression).**  
   **Objective:** Test adding the DTWEXBGS-based dollar-index features to models, especially where currency moves matter.  **Why now:** Roadmap explicitly calls for exploring USD momentum【111†L94-L97】.  Though the feature exists in the pool, it was never selected.  Perhaps a focused inclusion could help, e.g. for BND (bonds often respond to USD strength) or VWO/VXUS (EM currency effects).  **Data/Features:** The repo already computes `usd_broad_return_3m` and `usd_momentum_6m`【109†L1-L9】.  (If not, add DTWEXBGS through the existing pipeline.)  We will add these features to the candidate set for relevant benchmarks.  **Hypothesis:** Including USD momentum will raise balanced accuracy (or induce sell signals) for those benchmarks (e.g. better distinguishing PGR outperformance when USD is weak).  **Upside:** *Modest.* If one or two benchmarks see noticeable BA gains (e.g. +0.03-0.05), it's worthwhile.  **Complexity:** *Low.* Add feature to feature list or override for a couple benchmarks.  **Acceptance:** Requires a clear improvement on targeted benchmarks (e.g. BA_cov increase ≥0.03 or coverage up without loss of precision) and no degradation elsewhere.  **Stop:** If no improvements emerge (or calibration degrades), drop this feature.  **Mode:** Research-only shadow; could also try adding USD feature to the regression models and check IC, but primarily classification.

3. **FEAT-02: Oil Price Momentum (Classification).**  
   **Objective:** Leverage WTI momentum (3m return) in classifiers for commodity/energy ETFs.  **Why now:** Roadmap calls this out specifically for DBC and VDE【111†L95-L98】.  **Data/Features:** Use the existing `wti_return_3m` feature【97†L1-L9】.  Run classification search focusing on DBC and VDE: e.g. a small forward-stepwise selection or simple logistic that forces `wti_return_3m` into the model.  **Hypothesis:** Incorporating recent oil price trends will improve predictions: e.g. if oil has spiked, likely tough for PGR vs. energy/commodity benchmarks.  **Upside:** *High for these niches.* If WTI is a real driver, we may capture some previously-missed signals.  **Complexity:** *Very low.* Feature exists; just ensure it’s included.  **Acceptance:** Look for an increase in BA_cov (e.g. +0.04+) for DBC/VDE classification folds.  Also check business logic: verify that a sell recommendation is triggered in months of rising oil in backtest.  **Stop:** If WTI feature is never picked by forward search or degrades calibration, cease.  **Mode:** Classification research only (then shadow). Optionally test the same feature in the regression model for these tickers as a bonus.

4. **FEAT-03: Term Premium Change Signal (Regression/Classification).**  
   **Objective:** Capture interest-rate risk more sharply.  **Why now:** Term premium data is available and plausibly predictive, but we may not have tuned it finely.  **Data:** We already have `term_premium_10y`; create a 3-month difference (`term_premium_diff_3m`) or threshold feature (e.g. >0.25%).  **Hypothesis:** Sudden jumps in term premium hurt stock returns (raise rates), so adding this feature should improve timing in both regression and classification.  **Upside:** *Modest.* If term premium trends are meaningful, it may improve BA a bit or improve early warning.  **Complexity:** *Low.* Computation of a simple lagged diff.  **Acceptance:** Any measurable BA or IC gain.  If minimal, drop.  **Mode:** Could test in both streams (logistic search for classification on relevant tickers, and in regression pipelines).  If one side shows gain, proceed with that.  

5. **Calibration Adjustment (Classification).**  
   **Objective:** While not a new feature, this experiment is to tune the coverage thresholds or gating for the current best classifier.  **Why now:** Even after adding features, we may need to adjust decision thresholds.  This is borderline with decision policy, but can be framed as *improving calibration/truthfulness of classifier outputs*.  **Data:** Use existing classifier probabilities (raw or with Platt scaling).  **Hypothesis:** Small shifts in cutoff (e.g. from 0.30/0.70 to 0.25/0.75) might increase utility (more coverage or balanced accuracy).  **Upside:** *Low to moderate.* Possibly gain recall without too much precision loss.  **Complexity:** *Very low.* Just adjust constants in evaluation.  **Acceptance:** If balanced accuracy or realized decision accuracy (over past months) improves.  If not, revert.  **Mode:** Shadow (until we trust model outputs).

*(If needed) [Optional]:* **CLS-04: Combined Path A/B Ensembling.**  Blend the per-benchmark and portfolio classifiers (Path A vs Path B) in some gated way.  The roadmap’s CLS-03 was blocked, but one might test a simple ensemble.  However, given constraints, we’d skip this unless classification-only fixes fail.

Each experiment’s outcomes will be judged by out-of-sample (strict WFO) metrics, not in-sample fit. We keep the acceptance bar high (must beat current classifier BA/ECE significantly) to justify any promotion from “research” to “shadow candidate.”

# Recommended Sequencing

We recommend **classification-first**.  The evidence suggests the classifier has untapped potential that could meaningfully improve accuracy, while regression is mostly locked in place.  Specifically, start with **Experiment 1 (Firth logistic)**, because it requires no new data and fixes an obvious instability. If Firth yields stable gains on thin data, it should be implemented immediately in the shadow stack. 

In parallel or immediately after, run **Experiment 3 (Oil Momentum)** for commodity/energy tickers, since WTI data exists already.  This targets a narrow sector where improvement could be clear, and its simplicity (just forcing one feature into VDE/DBC models) means quick feedback. 

Next, try **Experiment 2 (USD Features)**.  This may require verifying or slightly reconfiguring the USD series, but the ROI is moderate.  Do the term premium experiment (**4**) last or in parallel, given its very low risk but possibly smaller impact. 

Regression enhancements (if any) should follow classification results: if classification BA jumps significantly, one might revisit regression.  But absent new strong signals, skip large regression work in this phase.  The classification path is shadow-only right now, so any upgrades are risk-guarded; but because *false confidence is to be avoided*, it's acceptable to get classifier improvements working in shadow before considering regression changes for production.

In summary, **classification experiments first** (Firth → oil momentum → USD dollar → term premium) because they directly target the largest identified bottleneck (lack of predictive signal in the decision layer) and align with backlog priorities【111†L91-L99】.  Regression can be revisited if, for example, USD or oil features also improve their performance, but given complexity constraints, we keep the emphasis on the classification pipeline for now.

# Do Now / Defer / Drop

| Action / Idea                                 | Status    | Reasoning                                                     |
|----------------------------------------------|-----------|---------------------------------------------------------------|
| **Add Firth logistic for small-sample ETFs** | **Do Now**   | Low complexity; addresses thin-data instability (CLS-02)【111†L91-L99】. Proven in backlog. |
| **Add USD index features (DTWEXBGS)**       | **Do Now**   | Low cost (already in data), targeted per roadmap (FEAT-01)【111†L94-L97】. May boost BND/VXUS. |
| **Add 3M WTI momentum to DBC/VDE models**   | **Do Now**   | Feature exists; roadmap (FEAT-02) calls it out. Likely benefit commodity/energy. |
| **Tune term-premium indicator**             | **Do Soon**  | Very low effort. Already have term_premium; test 3M change. Potential signal on bonds vs stocks. |
| **Ins. CPI (motor vehicle)**                | **Defer** | Data exists but moderate effort to fetch BLS. If others fail, revisit for insurance cycle effect. |
| **Add broad valuation features**            | **Drop**   | Prior analyses showed valuation features hurt or null【81†L11-L15】. Low signal and high risk. |
| **Add broad inflation/ops feature sets**    | **Drop**   | Already shown negative impact【81†L6-L15】. Would add complexity with no ROI. |
| **Benchmark-specific gating (beyond CLS-03)** | **Drop**   | The VGT gate and SCHD plans are blocked by matured-month requirements. Not for next cycle. |
| **Major regression rework (e.g. reopen GBT)**| **Defer**  | No evident simple levers remain; backlog defers REG-02. Focus on classification first. |
| **Further calibration sweeps**              | **Drop**   | V127 found no better calibrator that preserved BA【58†L558-L562】. Skip new calibration trials. |
| **Composite classifier ensemble (CLS-03)** | **Drop**   | Blocked by data maturity, and v128 results showed modest benefit【111†L138-L141】. |

# Implementation Guidance for the First Experiment (CLS-02: Firth Logistic)

**Objective:** Mitigate small-sample bias in classification for benchmarks with very few sell signals. 

**Steps:**

1. **Identify thin benchmarks.**  From historical data (pre-2019 backtest), determine which ETFs have limited actionable-sell events. A simple rule: benchmarks with <30 (or <20) positive cases in training over the full WFO window. Candidates might include DBC, VMBS, VDE, (SCHD if included), etc.

2. **Implement Firth logistic.**  Using a statsmodels GLM (Binomial family) or another package supporting Firth’s correction, fit logistic regression with the same 12 baseline features and `class_weight='balanced'`. Ensure we use the identical train/validation split logic (purged rolling windows). The penalty reduces bias in coefficient estimates for rare classes.

3. **Evaluate carefully.**  For each thin ETF, compute the prequential (rolling) covered metrics: BA_covered, precision, recall, Brier, ECE10, as with the existing classifier. Compare Firth vs ordinary logistic. Also check coverage (the fraction of months hitting the threshold). 

4. **Cross-check model outputs.**  Ensure probabilities remain well-calibrated post-Firth (they usually improve calibration). Confirm that no obvious data leak or overfitting appears (e.g. probabilities stuck at 0 or 1).

5. **Acceptance criteria:**  
   - *Per-Benchmark:* If a thin ETF’s balanced_accuracy_covered increases by ≥0.02 with similar or better coverage, it is a win. Or if BA stays ~same but ECE10 halves, still beneficial.  
   - *Pooled:* The overall pooled BA_covered should not drop. Ideally, it rises slightly. At minimum, no negative impact on the aggregate classifier performance.

6. **Decision:**  
   - If criteria are met, *promote* Firth logistic for those benchmarks into the shadow classifier (update Path A training code to use Firth for identified ETFs).  
   - If it underperforms (worse BA or overfits), revert to standard logistic and declare the experiment “no benefit”.

7. **Stop Condition:** If after testing on 2–3 thin ETFs we see no improvement (or destabilization), cease further Firth attempts. Otherwise, implement it and move on.

**Complexity Considerations:**  
This is a low-complexity change: it uses existing feature sets and data splits, only swapping the logistic solver. Because it is a small tweak, coding and testing should be quick. Tests should be added to ensure the new code path executes only for the intended ETFs.  

**Outcome:** The expectation is a small but meaningful reduction in estimation variance for the most volatile or data-poor benchmarks, improving overall reliability. If successful, it directly addresses an identified weakness (small sample bias) without adding new data or features, thus fully aligning with the mandate to boost confidence with minimal complexity.  

  

