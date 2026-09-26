# **Research Memo: Absolute Direction & Price Forecasting Lane**

## **1\. Executive Recommendation**

**Primary Path:** **Multi-Horizon Logistic Classification** (predicting binary/ordinal outcomes for forward returns).

**Secondary Benchmark Path:** **Structural Decomposition** (modeling TTM BVPS growth × Mean-Reverting P/B multiple).

**What Not to Pursue Initially:** **Direct Future Price/Log-Price Regression.** **Rationale:** Raw price series are non-stationary, and standardizing them (e.g., via differences) results in noisy continuous targets that cause catastrophic overfitting in small-sample monthly datasets. Predicting *whether* the price will be above the current level (or a hurdle rate) transforms a high-variance regression problem into a more stable, lower-variance classification problem. The structural decomposition serves as an economically grounded, explainable benchmark to anchor the classifier's predictions.

## **2\. Why This Fits the Repo**

* **Architecture Fit:** The repository already operates on a monthly feature matrix. Classification and modular decomposition easily plug into the existing src/processing/feature\_engineering.py and wfo\_engine without architectural rewrites.  
* **Data Sufficiency:** Predicting exact dollar returns at a 12-month horizon with N \< 300 monthly rows is practically impossible. Predicting directional probability (up/down) is statistically viable with simple, heavily regularized models on this exact sample size.  
* **Methodological Fit:** By enforcing strictly expanded Walk-Forward Optimization (WFO) and keeping models simple (Ridge/ElasticNet/shallow GBT), this approach avoids the pitfalls of K-Fold leakage and respects the temporal arrow of time.  
* **Governance Fit:** This new lane can easily be appended to monthly\_decision.py as a "Shadow Artifact," allowing the repository to track its out-of-sample (OOS) performance for several months before allowing it to influence live vesting execution.

## **3\. Target Design**

We will evaluate targets across $h \\in \\{1, 3, 6, 12\\}$ months.

**The Classification Formulation (Primary):**

* **Target:** Y\_direction\_h \= 1 if Forward\_Total\_Return(h) \> Risk\_Free\_Rate(h) else 0  
* *Why:* Comparing against the risk-free rate (using FRED data) provides an economic threshold for holding. It is far superior to predicting raw prices or relative percentages because it bounds the error penalty.

**The Decomposition Formulation (Secondary/Anchor):**

* **BVPS Leg:** Target\_BVPS\_Growth\_h \= TTM\_BVPS(t+h) / TTM\_BVPS(t) \- 1  
* **P/B Multiple Leg:** Target\_PB\_h \= P(t+h) / BVPS(t+h)  
* **Synthesis:** Implied\_Price(t+h) \= BVPS(t) \* (1 \+ Forecast\_BVPS\_Growth\_h) \* Forecast\_PB\_h  
* *Why:* BVPS growth is highly autoregressive and fundamentally driven (underwriting profit), while P/B is regime/macro-driven. Separating them reduces cross-contamination in feature weights.

**Why Not Raw Price?** Raw prices drift. Any model learning raw prices effectively just learns the recent moving average. Forecasting forward returns or directional probabilities directly sidesteps unit roots and scaling issues.

## **4\. Feature Plan**

**Features to Reuse Immediately:**

* **Valuation:** Current P/B, P/E, Dividend Yield.  
* **Underwriting Fundamentals (EDGAR):** PGR Combined Ratio, Loss Ratio, YoY Policies in Force (PIF) growth, Net Premiums Written.  
* **Capital:** TTM ROE, Share repurchases as a % of float.  
* **Macro/Rates (FRED):** Yield curve slope (10Y-2Y), High Yield Credit Spreads, CPI inflation YoY.

**Features to Exclude (or Isolate):**

* **Technical Indicators (TA):** *Exclude from the core absolute model.* Monthly TA features (like 14-month RSI or MACD) are notorious for overfitting in small samples and fighting with structural fundamentals.  
* *Recommendation for TA:* Keep TA features strictly in a **separate shadow lane** (as referenced in v160-v164 research) to see if a purely technical model outperforms the fundamental absolute model over time, but do not combine them into a single feature matrix matrix to preserve degrees of freedom.

## **5\. Model Plan**

We face strict small-sample monthly constraints. Parsimony is survival.

* **Baseline Benchmark:** Naive Drift / Random Walk for price; Long-term historical average for P/B multiple.  
* **Recommended Core Models:**  
  * *Classifier:* L2-Regularized Logistic Regression (Ridge Classifier).  
  * *Decomposition:* Ridge Regression for the P/B multiple; Simple Exponential Smoothing (or AR1) for BVPS growth.  
* **Recommended Challenger Model:** Conformal Histogram GBT (HistGradientBoostingClassifier). Must be heavily constrained (max\_depth=2 or 3, min\_samples\_leaf=15) to prevent memorizing the sparse monthly history.  
* **Horizon Approach:** Use **independent models per horizon**. The features driving a 1-month return (mostly momentum/sentiment) are vastly different from those driving a 12-month return (mostly valuation and underwriting cycle).

## **6\. Evaluation Plan**

**Walk-Forward Design:** Strictly expanding window WFO. Retrain annually or semi-annually.

*CRITICAL:* When predicting $h$ months ahead, the temporal split must enforce an **embargo gap of $h$ months** between the training set and the test set to prevent target overlap leakage (e.g., month 12's 3-month forward return overlaps with month 14's 3-month forward return).

**Metrics by Task:**

* **Classification:** Brier Score (for calibration quality), Log Loss, and Precision for the "Downside/Sell" class (preventing capital loss is the primary goal of vesting liquidation).  
* **Decomposition:** RMSE and MAE on the implied final price, tracking the error contribution from the BVPS leg vs. the P/B leg.  
* **Economic Usefulness:** "Strategy Return." Compare the portfolio curve of holding PGR unconditionally vs. liquidating upon a high-confidence "Below Current Price" forecast.

**Minimum Evidence for Promotion:** The absolute model must shadow in monthly\_decision.py for at least 6 live months. Its out-of-sample Brier Score must consistently beat a naive "always predict long-term base rate" model before being used for live execution.

## **7\. Implementation Plan for Codex / Claude Code**

Sequence the build in small, reviewable PRs to respect the research/production boundary:

1. **PR 1: Target Construction**  
   * *File:* src/processing/absolute\_targets.py  
   * *Task:* Create functions to safely shift and compute \+1m, \+3m, \+6m, \+12m forward returns, binary indicators, and forward P/B ratios. Include rigorous tests for leakage.  
2. **PR 2: WFO Embargo Implementation**  
   * *File:* src/models/wfo\_engine.py / src/models/evaluation.py  
   * *Task:* Implement Purged/Embargoed K-Fold or expanding window gaps to handle overlapping multi-horizon targets.  
3. **PR 3: Research Script (Classification)**  
   * *File:* scripts/research/v170\_absolute\_classification.py  
   * *Task:* Generate an artifact results/research/v170\_absolute\_classification\_summary.md comparing Logistic vs. Shallow GBT across all horizons using existing features.  
4. **PR 4: Research Script (Decomposition)**  
   * *File:* scripts/research/v171\_absolute\_decomposition.py  
   * *Task:* Generate artifacts forecasting BVPS and P/B separately, recombining, and scoring against the classifier.  
5. **PR 5: Shadow Integration**  
   * *File:* src/models/absolute\_shadow.py and scripts/monthly\_decision.py  
   * *Task:* Wire the winning formulation (likely Logistic Classification) to output probabilities in the monthly JSON artifact, keeping it explicitly isolated from the live relative-benchmark voting logic.

## **Final Summary**

Build **multi-horizon logistic classification first**. It translates a noisy prediction into a bounded, actionable probability. Build the **BVPS x P/B decomposition second**, utilizing it as a sanity-check benchmark.

**Greatest Expected Failure Mode:** Target overlap leakage. If you do not explicitly purge $h$ months of data between your train and test periods during WFO, your 12-month models will look like magic in backtesting and fail immediately in production.