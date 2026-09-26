# Improving PGR Relative Return Prediction Accuracy

## System baseline and why the problem is hard

This repository implements a monthly batch decision-support pipeline that produces a vesting-time sell-versus-hold recommendation and diagnostics for a 6-month horizon. The modeling core is a walk-forward optimization engine with rolling time-series validation and an explicit gap to reduce leakage risk.

Two facts frame the problem.

First, out-of-sample return forecasting is hard. In the academic literature, many predictors that look good in-sample fail out-of-sample, and even when predictability exists, out-of-sample R² is usually small.

Second, your current architecture is already doing several things right: rolling 60-month training windows, a 6-month target horizon, an embargo gap, lean feature sets, and a preference for interpretable models. That means the most plausible route to improvement is not more model complexity. It is better target construction, stronger shrinkage, and better magnitude calibration.

## What the current evidence says

Your current pattern is the key clue:
- OOS R² is materially negative.
- IC is positive and statistically significant.
- Hit rate is high.

That combination usually means the model has ranking or directional skill, but poor magnitude calibration. In other words, it is often on the right side, but the forecast amplitudes are too noisy or too aggressive for squared-error loss.

## Best target changes to test

### 1. Make classification the primary prediction task

Your business question is fundamentally directional: will PGR outperform a diversified benchmark over the next 6 months?

So the cleanest primary target is:
- binary outperform / underperform for decision support
- continuous excess return only as a secondary sizing layer

This keeps the core model aligned with the real decision and reduces sensitivity to extreme return magnitudes.

### 2. Replace per-ETF primary targets with one composite benchmark target

Instead of training the main decision model on eight separate relative-return labels, create one primary target:
- PGR 6-month return minus a fixed composite of the benchmark ETFs

This should reduce label noise, align the target with the actual diversification decision, and cut estimation variance.

Keep the per-benchmark models as diagnostics, not as the main production objective.

### 3. Treat 12-month horizon as a secondary branch, not the next default

A 12-month target may improve signal-to-noise, but it sharply reduces the effective sample and makes embargo costs larger. It is worth testing later, but not before fixing magnitude calibration on the 6-month horizon.

## Best feature-engineering changes to test

### 1. Add a continuous shrinkage and calibration layer to predictions

This is the single highest-leverage intervention.

Take the raw OOS regression prediction and transform it with either:
- y_cal = c * y_hat
or
- y_cal = a + b * y_hat

Fit c, or a and b, only on prior OOS predictions in a rolling as-of manner. This preserves directional skill while reducing MSE blowups from overconfident amplitudes.

### 2. Use conservative rolling normalization

For numeric features, apply transformations within each training window only:
- rolling z-score standardization
- optional rank normalization for highly skewed variables
- winsorization of extreme tails

These are simple, interpretable, and directly helpful when squared-error loss is dominated by a few extreme months.

### 3. Handle missingness in a feature-specific way

Do not treat all missingness as generic.

Recommended policy:
- slow fundamentals: carry forward last known value only after publication lag
- dense macro series: standard scaling or z-scoring only
- sparse or recently introduced features: either exclude, or use missingness indicators plus simple within-fold imputation

Avoid complex multivariate imputation across the full 72-feature set at this stage. With your sample size, that is more likely to add variance than signal.

### 4. Avoid broad interaction expansion

With n around 30 to 60 in early folds, adding many interactions is likely to overfit. If interactions are tested at all, they should be limited to one or two economically motivated terms, such as a pricing-cycle variable interacting with a rates variable.

### 5. Consider structured PCA only as a second-wave experiment

If you want to unlock more of the 72 features, the safest version is blockwise compression:
- one small PCA on macro features
- one small PCA on insurance-operation or pricing features
- feed 1 to 2 components from each block into a simple Ridge model

Do not run unconstrained PCA across everything as the first move.

## Best model architecture changes to test

### 1. Hierarchical or panel pooling across the 8 benchmarks

Right now, training separate models per benchmark wastes statistical strength.

A better small-sample design is to stack benchmark-month observations into one panel and estimate:
- common coefficients across benchmarks
- benchmark fixed effects
- optionally a very small number of benchmark-specific adjustments with strong regularization

This is one of the few ways to increase effective sample size without cheating.

### 2. Bayesian models can help, but only with real priors

A Bayesian linear model is worth considering only if you impose economically defensible priors such as:
- shrink intercept toward zero excess return
- shrink coefficients strongly toward zero
- possibly sign-informed priors for a few core macro relationships

Otherwise it is just another version of regularized regression.

### 3. Prefer regime features over regime splits

Do not split the data into separate hard-market and soft-market models yet. That will likely make the sample too small.

Instead, add one or two regime indicators and let a simple model adjust continuously.

## Best walk-forward design changes to test

### 1. Increase the number of OOS prediction points for calibration

If feasible, reduce test window size from 6 months to 1 month while keeping the gap discipline. That creates more OOS predictions and improves the sample available for post-model calibration.

### 2. Test expanding-window estimation against rolling-window estimation

Because your problem is variance-limited, an expanding window may outperform a rolling 60-month window even if it is less adaptive.

This is especially plausible for Ridge-style models with strong shrinkage.

### 3. Keep the embargo conservative

Your current 8-month gap for a 6-month horizon is reasonable and should not be relaxed unless you can prove no leakage. Leakage prevention matters more than squeezing out a few extra observations.

## Why IC and hit rate can be strong while OOS R² is negative

This is a known and common phenomenon in financial ML.

IC rewards ranking.
Hit rate rewards correct direction.
OOS R² punishes squared magnitude errors.

So a model can:
- get the sign right often
- rank outcomes meaningfully
- still lose badly to the historical mean on squared error because a few amplitude mistakes dominate MSE

That means your negative OOS R² does not imply zero signal. It implies that the current signal extraction and scaling are not yet aligned with a squared-error objective.

## Should OOS R² remain the primary gate?

It should remain an important safety gate, but not the only one.

Recommended production gate set:
- calibrated OOS R² on the final shrunk prediction
- IC
- hit rate or balanced accuracy for direction
- Brier score or ECE for the probability layer
- CPCV stability

The important shift is that OOS R² should be computed on the final calibrated forecast actually used in the decision layer, not on raw unshrunk model outputs.

## Ranked action plan

### 1. Add continuous magnitude shrinkage and calibration
Complexity: Small
Expected impact on OOS R²: High
Main risk: limited if implemented with strict as-of OOS-only calibration

### 2. Make a composite benchmark the primary target
Complexity: Medium
Expected impact: Medium to high
Main risk: benchmark weights could introduce noise if chosen poorly

### 3. Replace separate benchmark models with a pooled panel model
Complexity: Medium
Expected impact: Medium
Main risk: over-pooling could blur benchmark-specific effects

### 4. Increase calibration sample by shortening WFO test windows
Complexity: Small to medium
Expected impact: Medium
Main risk: noisier per-fold estimates unless evaluated carefully

### 5. Add structured factor compression for the long tail of features
Complexity: Medium
Expected impact: Low to medium
Main risk: factor instability and reduced interpretability

## Suggested experimental sequence

1. First test only continuous shrinkage on top of the current production model.
2. If that helps, lock it before changing any features or targets.
3. Next test the composite benchmark target.
4. Then test pooled panel estimation across benchmarks.
5. Only after those steps, test structured PCA and possibly a 12-month horizon branch.

Use a preregistered experiment log and limit grid sizes so that you do not create multiple-comparison bias.

## Honest assessment

A +2 percent OOS R² target is ambitious but not impossible.

With the current data volume, I would not expect large positive OOS R² from a monthly single-stock relative-return model. A realistic ceiling is probably low single digits if the signal is real and stable. If the signal is weak or regime-dependent, the ceiling may be near zero.

The best chance of reaching +2 percent is not adding more complex learners. It is reducing label noise and reducing forecast variance through:
- composite target construction
- post-model shrinkage
- pooled estimation across benchmarks

If those three do not materially improve results, the likely conclusion is that the current monthly data volume is not sufficient to support a stable positive OOS R² gate at the desired level.
