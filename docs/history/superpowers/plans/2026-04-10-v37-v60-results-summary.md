# v37-v60 Results Summary and Recommended Next Steps

## Purpose

This document consolidates the completed `v37`-`v60` enhancement cycle into one place:

- what was tested
- which variants improved or degraded the baseline
- what the results imply about the model's current failure mode
- what should happen next in research and in production

The active research baseline remains `v38` unless and until a later holdout-tested candidate beats it.

---

## Executive Summary

- Historical unmodified baseline: `v37`
  - Pooled OOS R²: `-0.2269`
  - IC: `0.1579`
  - Hit rate: `0.7002`
  - Sigma ratio: `0.6851`

- Best overall result: `v38_shrinkage_best_results.csv`
  - Variant: pooled post-hoc shrinkage with `alpha = 0.50`
  - Pooled OOS R²: `-0.1310`
  - IC: `0.1579`
  - Hit rate: `0.7002`
  - Sigma ratio: `0.3426`

- Best non-`v38` result from later phases:
  - `v50` clip+shrink: `-0.2300`
  - This was still worse than both `v38` and slightly worse than `v37`

- Most promising non-regression branch:
  - `v46` classification
  - Accuracy: `0.6533`
  - Balanced accuracy: `0.5292`
  - Brier score: `0.2502`

- Diagnostic conclusion from `v60`:
  - Pooled Clark-West `t = 3.3567`, `p = 0.0004`
  - Pooled CE gain: `+0.0330`
  - Pooled MSE variance share: `38.4%`
  - Pooled MSE bias share: `1.4%`
  - Interpretation: the ensemble appears to contain real predictive signal, but performance is constrained mainly by variance/calibration rather than mean-direction failure

---

## Phase-by-Phase Summary

### Phase 1: Calibration and Baseline Control (`v37`-`v43`)

- `v37` established the current-repo baseline and showed the repo/data state no longer matched the older over-dispersion diagnosis exactly.
- `v38` was the only clear phase-1 winner and the only variant that materially improved pooled OOS R² without harming IC or hit rate.
- `v39`, `v40`, `v41`, `v42`, and `v43` all underperformed `v38`.

Phase 1 conclusion:
- keep `v38` as the active research baseline
- treat the rest of phase 1 as investigated and not promoted

### Phase 2-3: Architecture and Structural Changes (`v44`-`v49`)

- `v44` PCA: worse than `v38`
- `v45` BayesianRidge variants: materially worse than `v38`
- `v46` classification: promising for future decision-layer work, but not a regression replacement
- `v47` composite benchmarks: worse than `v38`
- `v48` panel pooling: worse than `v38`
- `v49` regime features: worse than `v38`

Phase 2-3 conclusion:
- no regression architecture beat the simple `v38` shrinkage control
- `v46` should remain on the roadmap as a parallel classification branch

### Phase 4-5: Advanced Models, Data, and Diagnostics (`v50`-`v60`)

- `v50` prediction winsorization:
  - clip-only variants were poor
  - clip+shrink was the best later-phase regression result at `-0.2300`, but still worse than `v38`
- `v51` peer pooling: strongly negative
- `v52` shorter WFO windows: strongly negative on OOS R² despite one variant having stronger IC
- `v53` ARDRegression: strongly negative
- `v54` GPR:
  - best variant `B_gpr_rbf` reached `-0.2828`
  - still worse than `v38`
- `v55` rank targets: strongly negative
- `v56` 12-month horizon: strongly negative
- `v57` transformations:
  - rank-normalized GBT was the best of the phase at `-0.2440`
  - still worse than `v38`
- `v58` domain-specific FRED features: strongly negative
- `v59` imputation expansion to 18 features: strongly negative
- `v60` diagnostics:
  - produced the most useful late-cycle insight
  - supported the interpretation that the model has real signal but remains calibration-limited

Phase 4-5 conclusion:
- later-phase architecture and feature expansion mostly added variance
- diagnostics reinforced the case for conservative promotion of `v38`, not a more complex model

---

## Ranked Outcome Snapshot

Top pooled regression results:

1. `v38` shrinkage best: `-0.1310`
2. `v37` baseline: `-0.2269`
3. `v50` clip+shrink: `-0.2300`
4. `v40` constrained GBT: `-0.2355`
5. `v57` rank-normalized GBT: `-0.2440`
6. `v54` GPR-RBF: `-0.2828`

Bottom line:

- no post-`v38` regression variant beat the shrinkage baseline
- complexity generally hurt OOS R²
- additional features usually raised dispersion without improving directional quality enough to offset it

---

## What the Results Mean

### 1. The model likely has signal

`v60` is the strongest evidence here. The pooled Clark-West result is statistically significant and the pooled certainty-equivalent gain is positive. That is not consistent with a pure-noise model.

### 2. The main issue is still calibration and variance control

The winning change across the entire cycle was still the simplest one: shrink the predictions. The best diagnostic evidence also points to error being dominated by prediction variance rather than systematic bias.

### 3. More model complexity is not the right next default

PCA, BayesianRidge replacement, GPR replacement, peer pooling, regime features, longer horizon, broader feature sets, and expanded imputation all failed to improve the baseline. The current dataset remains firmly in a small-sample regime where low-variance methods dominate.

### 4. Classification may be the right place to expand next

`v46` did not solve regression, but it produced respectable directional metrics for the actual decision problem. That suggests future gains may come more from better decision-layer framing than from trying to extract more raw return magnitude accuracy.

---

## Recommended Production Changes

### Recommended now

1. Promote `v38` shrinkage into production.
   - Apply the fold-external prediction shrinkage used in `v38` with `alpha = 0.50`.
   - This is the only change that consistently improved pooled OOS R² while preserving IC and hit rate.
   - It is low-complexity, low-risk, and directly aligned with the dominant calibration diagnosis.

2. Add calibration-oriented monitoring to production evaluation.
   - Track sigma ratio, Clark-West, and MSE variance-vs-bias decomposition in the evaluation/reporting layer.
   - These are not live decision inputs, but they should become standard promotion diagnostics.

3. Keep the broader production model architecture unchanged for now.
   - Do not replace Ridge with GPR, BayesianRidge, ARD, or expanded feature sets.
   - Do not adopt peer pooling, 12M horizon, or imputation-expanded feature panels in production.

### Recommended after `v38` production promotion

4. Run one holdout evaluation for the promoted `v38` production candidate.
   - Use the reserved holdout window only once for the final promotion decision.
   - If holdout confirms the gain, lock `v38` in as the new production baseline.

5. Add benchmark-level diagnostics before changing benchmark weights.
   - `v60` suggests uneven quality across benchmarks.
   - Do not change the benchmark set immediately, but add a review of per-benchmark CW, CE gain, and calibration stability before any weighting or pruning decision.

---

## Recommended Research Next Steps

### Highest-priority research

1. Calibration-only extensions of `v38`
   - benchmark-specific shrinkage estimated only from training folds
   - fold-local volatility targeting or sigma-ratio targeting
   - conservative piecewise shrinkage based on prediction magnitude

These are the most logical next experiments because the full cycle showed calibration fixes outperforming architecture changes.

2. Decision-layer branch built on `v46`
   - use classification for sign or action gating
   - keep `v38` as the magnitude model
   - test hybrid rules such as: act only when `v46` probability exceeds a threshold and use `v38` for sizing

This is the most promising non-regression path revealed by the cycle.

3. Benchmark-specific action thresholds
   - use `v60` diagnostics to test whether some benchmarks should require stronger evidence before contributing to recommendations
   - especially review weaker cases such as `GLD`

### Medium-priority research

4. Simpler post-processing ensembles around `v38`
   - blend only if the companion model clearly improves calibration without adding variance
   - candidates worth limited follow-up: `v50` clip+shrink and `v57` rank-normalized GBT
   - avoid broad multi-model exploration unless the blend is explicitly calibration-focused

5. Production-style recommendation evaluation
   - test whether `v38` plus a directional gate improves recommendation quality even if raw OOS R² does not move much
   - emphasize decision metrics alongside return metrics

### Lower-priority or deprioritized research

6. Deprioritize these families unless the data regime changes materially
   - PCA or other dimensionality compression as a default path
   - BayesianRidge or ARD as full replacements
   - aggressive feature expansion and imputation
   - peer pooling / sector two-stage structures
   - 12-month horizon as the next main branch

These all added complexity without improving the main objective.

---

## Recommended Order of Action

1. Implement `v38` shrinkage in production.
2. Run the single reserved holdout evaluation for that production candidate.
3. Add `v60`-style diagnostics to the evaluation toolkit.
4. Start a new research branch focused on calibration-only extensions of `v38`.
5. Start a parallel research branch that combines `v46` classification with `v38` regression for recommendation gating.

---

## Final Recommendation

The enhancement cycle does not support a broader production refactor. It supports a narrow, evidence-based production improvement:

- promote `v38` shrinkage
- keep the current production architecture otherwise intact
- shift future research away from more complex regressors and toward:
  - better calibration
  - better decision framing
  - benchmark-aware gating and diagnostics

That is the highest-signal path supported by the full `v37`-`v60` result set.
