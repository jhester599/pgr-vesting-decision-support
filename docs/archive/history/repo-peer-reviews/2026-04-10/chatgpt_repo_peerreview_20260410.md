# PGR Vesting Decision Support — Findings from v37–v60 and a Concrete Next-Step Plan

## Project context and what is currently in production

The repository is a monthly batch decision-support system for managing semi-annual RSU vesting decisions in entity["company","Progressive Corporation","pgr stock issuer"] stock, with the core business question framed as: over the next ~6 months, will PGR outperform a diversified benchmark portfolio (hold more if “yes,” sell more if “no”). fileciteturn3file0L1-L1

Operationally, the system is built around scheduled ingestion → SQLite → monthly feature engineering → walk-forward modeling → a tax- and redeploy-aware recommendation/reporting layer. fileciteturn3file0L1-L1 The production surface area is script-driven (notebooks are intentionally not the operational boundary). fileciteturn3file0L1-L1

The modeling core that matters for this research cycle is:

- **Small-sample WFO**: rolling 60-month training windows, 6-month forecast horizon, and an explicit gap logic (horizon + purge buffer) to reduce overlap-leakage. fileciteturn8file0L1-L1 fileciteturn13file0L1-L1  
- **Primary production universe**: the repo ingests a broad ETF universe but the **production forecast universe is a selected set of 8 benchmarks** used by the monthly decision engine. fileciteturn9file0L1-L1  
- **Lean model stack**: production is intentionally **ridge + gradient-boosted trees** with model-specific lean feature sets (v18 feature sets), explicitly acknowledging sample-size constraints. fileciteturn8file0L1-L1 fileciteturn9file0L1-L1  
- **Model-health reporting**: the system already computes Campbell–Thompson OOS R² and a Newey–West HAC-adjusted IC (to account for overlapping 6-month windows), with CPCV used as a stability diagnostic. fileciteturn12file0L1-L1 fileciteturn13file0L1-L1 fileciteturn15file0L1-L1  
- **Probability calibration + uncertainty**: probability calibration (Platt scaling, with isotonic deferred until far larger samples) and conformal prediction intervals are implemented in the repo and used in the monthly reporting path. fileciteturn20file0L1-L1 fileciteturn15file0L1-L1 This aligns with the “decision support” framing, where reliable probabilities and uncertainty bounds can matter more than noisy point forecasts.

These design choices are consistent with the earlier deep-research reports’ thesis: in return forecasting, it is common to see weak (or negative) OOS R² even when ranking/direction metrics show some signal; therefore, calibration and variance control can be higher leverage than adding “bigger” models. fileciteturn0file1 fileciteturn0file2 The finance literature supports this broad claim: out-of-sample return predictability is typically small and unstable, and naive historical-mean benchmarks are hard to beat consistently. citeturn0search4turn0search0

## Results recap from the v37–v60 cycle

The v37–v60 research plan and the consolidated results summary point to a very clean empirical story: nearly every step toward “more sophistication” degraded pooled OOS R², while the one consistent winner was **post-hoc shrinkage**. fileciteturn0file0 fileciteturn0file3

Key quantitative outcomes reported for this cycle:

- **v37 baseline**: pooled OOS R² ≈ **−0.2269**, with **positive IC ≈ 0.1579** and a **high hit rate ≈ 0.7002**. fileciteturn0file3  
- **Best overall**: **v38 shrinkage**, pooled post-hoc shrinkage with **α = 0.50**, improving pooled OOS R² to **≈ −0.1310** while leaving IC and hit rate unchanged (as expected for pure scaling). fileciteturn0file3  
- **Best later-phase regression variant after v38**: **v50 clip+shrink** was still essentially at baseline (≈ −0.2300), not competitive with v38. fileciteturn0file3  
- **Most promising “non-regression” branch**: **v46 classification** achieved modest directional metrics (accuracy ≈ 0.6533; balanced accuracy ≈ 0.5292; Brier ≈ 0.2502). fileciteturn0file3  
- **v60 diagnostics**: pooled Clark–West statistic **t ≈ 3.3567 (p ≈ 0.0004)**, pooled certainty-equivalent gain **+0.0330**, and MSE decomposition showing **variance dominating error (≈ 38.4% variance share vs ≈ 1.4% bias share)**. fileciteturn0file3

The earlier research reports predicted this pattern qualitatively: “negative OOS R² + positive IC + high hit rate often implies ranking/direction skill but poor magnitude calibration,” and recommended calibration/shrinkage as the first lever. fileciteturn0file1 fileciteturn0file2

From an academic-methods perspective, the v60 Clark–West finding is particularly important. Clark & West show that when comparing a larger/nested forecasting model to a parsimonious benchmark, the larger model’s MSFE can look worse mechanically because parameter estimation adds noise; the Clark–West adjustment provides an “approximately normal” test for whether the larger model actually improves predictive accuracy after accounting for that effect. citeturn1search0turn1search6 This is an evidence-based explanation for why OOS R² can remain negative while diagnostics still indicate real incremental signal.

## What these results imply about the model’s current failure mode

### The system is variance-limited, not “signal-free”

The v60 MSE decomposition’s low bias share means “the model isn’t mainly wrong because it’s systematically pushing the mean the wrong way”; rather, it is wrong because the predictions are too noisy (high variance) relative to the weak predictability of the target. fileciteturn0file3

This is exactly the regime where:

- adding features,
- increasing flexibility (PCA, more aggressive model classes),
- pooling across heterogeneous assets,

tends to harm OOS R² because each change consumes degrees of freedom and raises estimation noise. fileciteturn0file3 fileciteturn0file0

This is not just a “your repo” pattern; it’s consistent with broader evidence that return predictability is small OOS and that unstable predictors can look strong in-sample but fail out-of-sample. citeturn0search4turn0search0

### Shrinkage wins because it reduces “overconfident amplitude,” preserving whatever directional signal exists

A pure shrinkage map \(\hat{y}_{\text{cal}} = \alpha \hat{y}\) leaves rank ordering unchanged and typically leaves directional sign unchanged (for \(\alpha>0\)), so IC and hit rate naturally remain the same while MSE can improve through reduced variance. This is exactly what v38 showed. fileciteturn0file3

This aligns with the earlier deep-research reports’ central recommendation: if IC is positive but OOS R² is negative, the highest-leverage intervention is often a post-processing calibration layer that rescales forecasts without adding model degrees-of-freedom. fileciteturn0file1 fileciteturn0file2

### The “complexity vs. sample size” tension is binding, and it will remain binding for a long time

Your production WFO already acknowledges the constraint: training windows are 60 months, targets have a 6-month horizon, and the code adds purge/embargo separation explicitly because overlapping windows create serial dependence. fileciteturn8file0L1-L1 fileciteturn13file0L1-L1

That means:

- effective independent sample size is smaller than “number of months,” and  
- many sophisticated structures (high-dimensional feature expansions, latent factor models, richer Bayesian structures, regime splits) will overfit unless you can materially increase usable OOS history.

The empirical record from v44–v59 is basically a controlled demonstration of this. fileciteturn0file3

### Classification is promising, but only as a decision-layer gating tool

v46 classification’s balanced accuracy barely clears 0.5 despite decent raw accuracy, which likely indicates **class imbalance and/or weak separability**. fileciteturn0file3 That is not good enough to replace the regression engine outright, but it can still be useful as a conservative “action permission” layer (only act when probability is high enough), especially when paired with shrinkage-calibrated magnitudes for sizing. This hybrid framing was also recommended in the results summary and earlier reports. fileciteturn0file3 fileciteturn0file2

## Prioritized enhancement plan focused on the v37–v60 findings

This plan is intentionally conservative, and it explicitly avoids “more complex models” as the default next move because the completed experiment cycle indicates those moves are net-negative under the current sample regime. fileciteturn0file3

### Highest priority production enhancements

These are the changes most likely to produce measurable improvements (or at minimum, reduce decision risk) with minimal overfitting risk.

#### Implement v38-style shrinkage in the production prediction path

The v37–v60 cycle’s only clear regression winner was post-hoc shrinkage (v38), improving pooled OOS R² materially with α=0.50. fileciteturn0file3 Today, the repo implements probability calibration (Platt scaling) and conformal intervals, but **there is no parallel “magnitude shrinkage/calibration” layer** applied to the point forecasts in the monthly decision script, and “shrinkage” does not appear as a first-class module/config. fileciteturn15file0L1-L1

Concrete implementation (scikit-learn / statsmodels friendly):

1. Add `config.MODEL_PREDICTION_SHRINKAGE_ALPHA = 0.50` and `MODEL_PREDICTION_SHRINKAGE_MODE = {"static","rolling"}` in `config/model.py`. fileciteturn8file0L1-L1  
2. Create a small module `src/models/magnitude_calibration.py` with:
   - `apply_shrinkage(y_hat: np.ndarray | float, alpha: float) -> same`
   - (optional) `fit_alpha_no_intercept(y_hat_hist, y_true_hist) -> float` using the closed-form MSE-optimal scaling:
     \[
       \alpha^*=\frac{\sum y_{\text{true}} y_{\text{hat}}}{\sum y_{\text{hat}}^2}
     \]
     and then clip to a safe range like `[0, 1]` to prevent sign-flips from tiny samples.
3. In `scripts/monthly_decision.py`, after `get_ensemble_signals(...)` and before consensus and reporting, apply shrinkage to:
   - `signals["predicted_relative_return"]`
   - all reconstructed OOS sequences used for evaluation (next item), **so reported OOS R² matches what the decision layer is actually using**. fileciteturn15file0L1-L1

Why this is the top priority:

- it is already empirically validated in your own v38 results; fileciteturn0file3  
- it adds effectively **one** degree-of-freedom (or zero if alpha is fixed); and  
- it directly targets the diagnosed failure mode (variance/calibration). fileciteturn0file3

Safety note: treat α=0.50 as a **frozen promotion candidate** until it passes one clean holdout evaluation; do not continuously re-optimize α every month unless you institute a strict “as-of” rolling calibration design (see below) to avoid reintroducing drift-based multiple-testing. The general danger of repeated configuration search in finance is well documented. citeturn1search1turn1search2

#### Align model-health metrics with the actual ensemble forecast you deploy

Right now, the monthly decision script computes aggregate OOS R² and Newey–West IC using `model_result = elasticnet if present else first available model`, rather than using the reconstructed inverse-variance ensemble OOS predictions that are already available via `_reconstruct_ensemble_oos(...)`. fileciteturn15file0L1-L1 This creates a high-risk misalignment: you may be gating decisions, tracking drift, and triggering retrains using metrics that do not reflect the actual forecast used in the decision/reporting layer.

Concrete changes:

- In `_compute_aggregate_health(...)` and `_write_diagnostic_report(...)`, replace the per-benchmark `(y_true, y_hat)` extraction from a single model with:
  - `y_hat_ens, y_true = _reconstruct_ensemble_oos(ens_result)`
  - then compute OOS R² and Newey–West IC on the *ensemble* series, not a component model. fileciteturn15file0L1-L1  
- Maintain a second “component-model” view if you want diagnostics, but the **promotion gate metrics must be computed on the deployed forecast**. The Campbell–Thompson OOS R² you already use is specifically defined as model vs naive benchmark using realized data sequences, and it should evaluate the forecast actually used. fileciteturn12file0L1-L1

This is an unusually high-leverage fix because it improves the reliability of every downstream decision (monitoring, retrain triggers, and eventually vesting actions) without changing the underlying model class.

#### Add Clark–West reporting as a first-class diagnostic gate alongside OOS R²

Your v60 result summary reports a statistically significant pooled Clark–West test even while OOS R² remains negative. fileciteturn0file3 The repo currently reports Campbell–Thompson OOS R² and Newey–West IC, but not Clark–West. fileciteturn12file0L1-L1

Why this matters:

- When models are nested or near-nested relative to a benchmark (e.g., historical mean baseline), MSFE comparisons are biased against the larger model due to parameter estimation noise; Clark–West proposes an MSFE adjustment and a near-normal test statistic for equal predictive accuracy. citeturn1search0turn1search6  
- If Clark–West consistently rejects the null in favor of the model, that is strong evidence that the model contains real predictive information even if squared-error metrics (like OOS R²) are depressed by variance.

Concrete implementation (statsmodels-friendly):

- Add `src/models/forecast_tests.py` implementing Clark–West:
  1. Define benchmark forecast \(f_{0,t}\) (historical mean expanding) and model forecast \(f_{1,t}\) (your calibrated/shrunk ensemble).
  2. Compute \(e_{0,t} = y_t - f_{0,t}\), \(e_{1,t}=y_t-f_{1,t}\).
  3. Compute the Clark–West adjustment term and the adjusted loss differential series (per Clark & West). citeturn1search0turn1search6
  4. Regress the adjusted differential on a constant and use HAC SE (Newey–West lags = horizon−1) because your monthly targets overlap. fileciteturn12file0L1-L1  
- Surface the pooled statistic and per-benchmark statistics in `diagnostic.md` next to OOS R² and Newey–West IC.

This directly operationalizes the v60 diagnostic insight rather than leaving it as a one-off research result. fileciteturn0file3

### Highest priority research enhancements

These are “next research cycle” items that should be run as a small, curated set of experiments (not another wide search), because v37–v60 already demonstrated that breadth-first exploration amplifies overfitting risk.

#### Add a magnitude calibration layer that goes beyond scalar shrinkage

v38 shows that “multiply by α” improved OOS R². fileciteturn0file3 A natural next extension (not equivalent to adding a richer predictive model) is **affine recalibration**:

\[
\hat{y}_{\text{cal}} = a + b \hat{y}
\]

Why it is worth testing despite v60’s low bias share:

- the pooled bias share is small, but bias can still be **benchmark-specific** (e.g., some benchmarks systematically over/understate relative returns), which can damage aggregate metrics and decision thresholds. fileciteturn0file3  
- affine calibration can be regularized heavily (and can be fitted “as-of” on prior OOS points only), keeping variance low.

Implementation detail:

- Use `sklearn.linear_model.Ridge` as a calibrator on the single predictor \( \hat{y} \) (or two predictors: \(\hat{y}\) and |\(\hat{y}\)| for piecewise behavior), fit on historical OOS points.
- Evaluate “prequentially”: at time t, calibrator trained on {1…t−1}, applied at t. This avoids leakage and is compatible with your WFO discipline.

This approach is informationally aligned to the earlier research reports, which stressed “calibration-only fixes first.” fileciteturn0file2

#### Benchmark-quality weighting for consensus aggregation

Current consensus in the monthly decision script treats benchmarks symmetrically in aggregation (mean predicted return; majority vote). fileciteturn15file0L1-L1 The v60 results summary explicitly suggests **uneven benchmark quality** and recommends reviewing benchmark-level diagnostics before changing weights. fileciteturn0file3

A concrete, low-variance way to act on this without “optimizing weights” into overfitting:

- Define **quality scores** \(q_b\) per benchmark based on stable OOS diagnostics (e.g., Newey–West IC or Clark–West t-stat), computed as-of. fileciteturn12file0L1-L1 fileciteturn0file3  
- Convert to weights via a shrinkage-to-equal-weight rule:
  \[
  w_b = (1-\lambda)\cdot \frac{1}{B} + \lambda \cdot \frac{\max(q_b,0)}{\sum \max(q_b,0)}
  \]
  with \(\lambda\) small (e.g., 0.25) until you have much more OOS history.
- Use weights in:
  - consensus expected relative return (weighted mean),
  - consensus P(outperform) (weighted mean of calibrated per-benchmark probabilities),
  - and possibly in the decision policy backtest section.

This is directly aligned with the v37–v60 conclusion that “variance control beats complexity”: you’re not increasing model capacity, you’re downweighting known-noisy benchmark channels.

#### Decision-layer hybridization: use classification as a gate, not as a replacement

The v46 classification branch is the only “non-regression” direction that looks promising, but its balanced accuracy indicates it is not robust enough to be a standalone action engine. fileciteturn0file3

The actionable way to use it is:

- Let the shrunk/calibrated regression forecast control *sizing* (continuous expected benefit).
- Let the classification probability control *whether you act at all* via conservative thresholds.

Concrete policy proposal to evaluate:

- Compute \(p = P(\text{outperform})\) (your existing Platt-calibrated probabilities already produce this). fileciteturn20file0L1-L1  
- Define a high-confidence band (example):
  - act “risk-on PGR” only if \(p \ge 0.65\),
  - act “risk-off / diversify” only if \(p \le 0.35\),
  - otherwise default to tax/diversification baseline (e.g., 50% sale).
- Inside the “act” regions, use the shrunk magnitude forecast (and tax breakeven logic) to size deviations (e.g., 25%/50%/75%/100% sell). fileciteturn15file0L1-L1

This provides a concrete way to convert modest directional skill into reduced decision regret, and it is consistent with both the earlier deep-research reports and the results summary’s recommendation. fileciteturn0file2 fileciteturn0file3

### Medium priority experiments worth a small, disciplined trial

These are “candidate upgrades” that may reduce variance, but are not supported directly by the v37–v60 winner set. They should be tested in a very limited and pre-registered way to avoid another broad search cycle.

#### Replace the current GBT with monotonic-constrained entity["organization","scikit-learn","python ml library"] Histogram Gradient Boosting for variance control

Your v37–v60 experience suggests flexible models hurt, but monotonic constraints can *reduce* effective flexibility by ruling out economically nonsensical fit distortions, which is often beneficial in small samples.

`HistGradientBoostingRegressor` supports monotonic constraints (`monotonic_cst`) and interaction constraints, which can reduce variance and improve stability. citeturn4search1turn4search2

Concrete approach:

- Keep the ridge model unchanged.
- Create a second tree model candidate:
  - `HistGradientBoostingRegressor(max_depth=2 or 3, learning_rate small, max_iter modest)`
  - supply `monotonic_cst` only for **2–4 features with very strong monotonic priors** (example candidates might include credit spreads and volatility measures, depending on the empirical sign stability in your WFO fold importance diagnostics). fileciteturn15file0L1-L1
- Evaluate only as: “GBT → HGBR with monotonic constraints,” keeping everything else fixed, and compare to the v38-shrunk baseline.

This is not “more complexity”; it is a specific variance-control mechanism supported by primary library documentation. citeturn4search1turn4search2

#### Add a block-bootstrap bagging wrapper around ridge for stability

Given you are in a variance-dominated regime, another classical approach is to stabilize the linear model’s predictions by averaging over resampled training windows.

Implementation concept:

- For each fold (or for the live refit), sample K bootstrap replicates using contiguous month-blocks (block length ≈ 6 months), fit ridge on each replicate, average predictions.
- This can be implemented without new dependencies; it is CPU-friendly.
- Evaluate it only as incremental improvement on top of v38 shrinkage (otherwise you won’t know which variance-control lever did anything).

This is not in your completed v37–v60 set, and it should be treated as experimental only because it introduces additional moving parts.

## Implementation details and governance guardrails

### How to do “rolling calibration” without leakage

If you decide to move from fixed α=0.50 to a rolling calibration, the design must be **as-of**:

- Generate historical OOS predictions with the base model (already available via WFO folds).
- For each OOS timestamp \(t\), fit the calibrator using only OOS points \(<t\).
- Apply calibrator at \(t\) and store the calibrated prediction.
- Evaluate OOS R² / Clark–West / CE gain on this prequential calibrated series.

This preserves the repo’s strong time-series discipline (no K-fold, no leakage), which is also critical in finance because leakage can masquerade as “alpha.” fileciteturn13file0L1-L1 citeturn4search0turn1search1

### Use multiple metrics, but gate on the forecast actually used in decisions

Given you have explicit evidence that OOS R² can be negative while Clark–West and CE gain are positive, your promotion gate should be multi-metric, not single-metric. fileciteturn0file3 citeturn1search0

A reasonable production gate set for this project (with small sample) is:

- **Campbell–Thompson OOS R²** on the final calibrated forecast, not raw. fileciteturn12file0L1-L1  
- **Clark–West** vs historical mean baseline on the same forecast. citeturn1search0turn1search6  
- **Newey–West IC** with lags = horizon−1 to account for overlap. fileciteturn12file0L1-L1  
- **Calibration metrics** (ECE / reliability) for probability forecasts. fileciteturn20file0L1-L1  
- **Policy-level utility metrics** (certainty equivalent, capture ratio, etc.), because the business objective is a vesting decision rule, not minimizing MSE. fileciteturn0file3

### Constrain your search process

The v37–v60 cycle already shows how quickly broad exploration can fail to beat a simple shrinkage baseline. fileciteturn0file3 In finance, repeated configuration search is a known failure mode and can lead to negative expected OOS results. citeturn1search1turn1search2

Concretely:

- Keep the next cycle to **5–8 experiments total**, each a single-factor change from baseline.
- Pre-register which metric determines promotion.
- Reserve the final “holdout window” for exactly one or two candidates.

## Recommendations that would require new data sources

The v37–v60 evidence suggests you should not chase new signals by adding broad features by default; nonetheless, if you *choose* to expand data sources, these are the kinds of sources that could plausibly add orthogonal information, but they are outside your current stack:

- **Options-implied information** (implied volatility/skew, insurer-cat risk repricing indicators): requires options data vendor access (not currently in repo). This can materially change infrastructure and is not recommended unless you have a low-friction data feed.  
- **Analyst expectations / earnings revisions** (I/B/E/S or similar): paid data in most cases; not currently in your ingestion inventory.  
- **Industry catastrophe-loss estimates** to build “combined ratio ex-cats” or CAT-weighted underwriting features: would require a CAT-model proxy or a third-party dataset and may introduce licensing complications.  

If you do add external data, the sample-size tension becomes even more acute: more predictors without significantly more truly point-in-time observations usually increases variance and gets punished OOS (exactly what you saw in many v44–v59 attempts). fileciteturn0file3 citeturn0search0

## Summary of what should be concluded from v37–v60 and what to do next

The completed v37–v60 cycle supports three firm conclusions:

1. **There is real signal, but the system is variance/calibration constrained.** The diagnostic evidence (especially v60) is inconsistent with a pure-noise model. fileciteturn0file3 citeturn1search0  
2. **Model complexity generally harms out-of-sample performance in the current sample regime.** The experiment sweep demonstrated that PCA, broader feature sets, alternate regression classes, and structural pooling mostly degrade metrics. fileciteturn0file3  
3. **The best next steps are calibration- and decision-layer improvements, not “new models.”** v38 shrinkage is the only clear winner, and classification is promising only as a gating layer. fileciteturn0file3 fileciteturn0file2

Accordingly, the concrete next-step sequence that best fits your own evidence and the broader forecasting literature is:

- **Promote v38-style shrinkage into the production prediction path**, and ensure **all reported metrics evaluate the same calibrated forecast actually used in decisions**. fileciteturn0file3  
- **Add Clark–West diagnostics** alongside Campbell–Thompson OOS R² and Newey–West IC so you can distinguish “no signal” from “signal overwhelmed by estimation noise.” citeturn1search0turn1search6 fileciteturn12file0L1-L1  
- **Run a tight, calibration-only research mini-cycle**: (a) per-benchmark shrinkage, (b) affine recalibration, and (c) benchmark-quality-weighted consensus—all designed to reduce variance without adding degrees-of-freedom.  
- **Use classification only as an action gate**, not as a replacement forecaster, and evaluate success primarily in policy/utility terms, not only in squared-error terms. fileciteturn0file3