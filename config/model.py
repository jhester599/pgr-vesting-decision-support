"""
Walk-Forward Optimization parameters, Kelly sizing, ensemble models, CPCV,
Black-Litterman, fractional differentiation, diagnostic thresholds (DIAG_*,
VIF_*), probability calibration, conformal prediction, recommendation-layer
constants, V13 shadow settings, and BLP parameters.
"""

import os

# ---------------------------------------------------------------------------
# Walk-Forward Optimization parameters
# ---------------------------------------------------------------------------
WFO_TRAIN_WINDOW_MONTHS: int = 60    # 5-year rolling training window
WFO_TEST_WINDOW_MONTHS: int = 6      # 6-month out-of-sample test period
WFO_MIN_GAINSHARE_OBS: int = 60      # Min non-NaN rows to include Gainshare features

# v1 bug: WFO_EMBARGO_MONTHS = 1 was too short for a 6M overlapping target.
# v2 fix: embargo must equal the target horizon to prevent autocorrelation
# leakage between consecutive monthly observations.
WFO_EMBARGO_MONTHS: int = 1          # Retained for v1 backward-compat; DO NOT USE in v2
WFO_EMBARGO_MONTHS_6M: int = 6       # Correct embargo for 6-month target horizon
WFO_EMBARGO_MONTHS_12M: int = 12     # Correct embargo for 12-month target horizon
WFO_TARGET_HORIZONS: list[int] = [6, 12]

# v3.0: additional purge buffer beyond the target horizon to account for
# serial autocorrelation in monthly data (research report recommendation).
# Total gap = target_horizon + purge_buffer:
#   6M horizon  → gap = 6 + 2 = 8 months
#   12M horizon → gap = 12 + 3 = 15 months
WFO_PURGE_BUFFER_6M: int = 2
WFO_PURGE_BUFFER_12M: int = 3

# ---------------------------------------------------------------------------
# v3.1 ensemble and Kelly sizing parameters
# ---------------------------------------------------------------------------
KELLY_FRACTION: float = 0.25          # quarter-Kelly to control risk
KELLY_MAX_POSITION: float = 0.20      # v4.1: reduced from 0.30 (Meulbroek 2005: 25% employer stock = 42% CE loss)
# v5.0: added shallow GBT as 4th ensemble member (max_depth=2, n_estimators=50)
ENSEMBLE_MODELS: list[str] = ["elasticnet", "ridge", "bayesian_ridge", "gbt"]

# ---------------------------------------------------------------------------
# v4.0 CPCV parameters — v5.0: upgraded from C(6,2)=15 paths to C(8,2)=28 paths
# ---------------------------------------------------------------------------
CPCV_N_FOLDS: int = 8         # Number of folds for CombinatorialPurgedCV (v5.0: was 6)
CPCV_N_TEST_FOLDS: int = 2    # Test folds per split; yields C(8,2)=28 paths (v5.0)

# ---------------------------------------------------------------------------
# v4.0 Black-Litterman parameters
# ---------------------------------------------------------------------------
BL_RISK_AVERSION: float = 2.5           # Moderate risk aversion (1=aggressive, 5=conservative)
BL_TAU: float = 0.05                    # Uncertainty in equilibrium returns (small = trust prior)
BL_VIEW_CONFIDENCE_SCALAR: float = 1.0  # Scales Ω = RMSE² × scalar

# Use BayesianRidge posterior variance (σ²_pred) as the Ω diagonal in the
# Black-Litterman model instead of MAE².
BL_USE_BAYESIAN_VARIANCE: bool = True

# ---------------------------------------------------------------------------
# v4.0 Fractional differentiation parameters
# ---------------------------------------------------------------------------
FRACDIFF_MAX_D: float = 0.5             # Maximum differentiation order (preserves memory)
FRACDIFF_CORR_THRESHOLD: float = 0.90   # Minimum correlation with original series
FRACDIFF_ADF_ALPHA: float = 0.05        # Stationarity significance level

# ---------------------------------------------------------------------------
# v4.3.1 — Diagnostic OOS Evaluation Report thresholds
# Used in _write_diagnostic_report() to flag model health vs. peer-review
# benchmarks (Harvey et al. 2016; Campbell & Thompson 2008; Gu et al. 2020).
# ---------------------------------------------------------------------------
# Campbell-Thompson OOS R²: >2% = good, 0.5–2% = marginal, <0% = failing.
DIAG_MIN_OOS_R2: float = 0.02
# Newey-West HAC-adjusted Spearman IC: >0.07 = good, 0.03–0.07 = marginal.
DIAG_MIN_IC: float = 0.07
# Hit rate (directional accuracy): >55% = good, 52–55% = marginal.
DIAG_MIN_HIT_RATE: float = 0.55
# CPCV positive paths (out of C(8,2)=28): ≥19 = good (~67%), 14–18 = marginal.
# v5.0: updated from C(6,2)=15 thresholds (was: ≥13/15 good, 10–12 marginal).
DIAG_CPCV_MIN_POSITIVE_PATHS: int = 19
# Variance Inflation Factor thresholds for multicollinearity checks (v32.1).
# VIF > HIGH_THRESHOLD is flagged as high multicollinearity (❌).
# VIF > WARN_THRESHOLD is flagged as moderate multicollinearity (⚠️).
VIF_HIGH_THRESHOLD: float = 10.0
VIF_WARN_THRESHOLD: float = 5.0

# ---------------------------------------------------------------------------
# v5.1 — Probability Calibration
# ---------------------------------------------------------------------------
# Minimum OOS observations required before activating each calibration tier.
# At n < CALIBRATION_MIN_OBS_PLATT the raw BayesianRidge posterior is returned.
# At n >= CALIBRATION_MIN_OBS_ISOTONIC the two-stage Platt → Isotonic model
# is used, which is non-parametric and benefits from larger samples.
CALIBRATION_MIN_OBS_PLATT: int = 20     # Activate Platt scaling above this n
# Isotonic requires far more per-benchmark data to avoid plateau collapse on
# out-of-sample inputs.  With n=78–260 per benchmark (2026), the step function
# returns a single constant for most live predictions.  Re-evaluate at ~500+.
CALIBRATION_MIN_OBS_ISOTONIC: int = 500  # Per-benchmark isotonic threshold
# ECE computation parameters
CALIBRATION_N_BINS: int = 10            # Equal-width probability bins for ECE
CALIBRATION_BOOTSTRAP_REPS: int = 500   # Block bootstrap replications for ECE CI

# ---------------------------------------------------------------------------
# v5.2 — Conformal Prediction Intervals
# ---------------------------------------------------------------------------
# Nominal coverage for the prediction interval shown in recommendation.md.
# 0.80 = 80% CI; interpretation: "over any 5 predictions, at least 4 will
# contain the true 6M relative return."
CONFORMAL_COVERAGE: float = 0.80
# Method: "aci" (Adaptive Conformal Inference; handles distribution shift)
# or "split" (standard split conformal; simpler, slightly narrower intervals).
CONFORMAL_METHOD: str = "aci"
# ACI step size γ — controls how fast α_t adapts to coverage misses.
# 0.05 is the default from Gibbs & Candès (2021); smaller = slower adaptation.
CONFORMAL_ACI_GAMMA: float = 0.05

# ---------------------------------------------------------------------------
# v13 recommendation-layer promotion study
# ---------------------------------------------------------------------------
# Keep the live model stack unchanged, but allow the monthly report/email layer
# to include or eventually promote the simpler diversification-first baseline
# that performed best in the v11/v12 recommendation studies.
RECOMMENDATION_LAYER_MODE: str = os.getenv("RECOMMENDATION_LAYER_MODE", "shadow_promoted")
RECOMMENDATION_LAYER_VALID_MODES: tuple[str, ...] = (
    "live_only",
    "live_with_shadow",
    "shadow_promoted",
)
# v22: keep the active recommendation layer unchanged, but replace the
# previously displayed live-stack cross-check with the historically stronger
# reduced-universe Ridge+GBT candidate selected in v21.
V22_PROMOTED_CROSS_CHECK_CANDIDATE: str = os.getenv(
    "V22_PROMOTED_CROSS_CHECK_CANDIDATE",
    "ensemble_ridge_gbt_v18",
)
V13_SHADOW_BASELINE_STRATEGY: str = "historical_mean"
V13_SHADOW_BASELINE_POLICY: str = "neutral_band_3pct"
V13_SHADOW_FORECAST_UNIVERSE: list[str] = [
    "VOO",
    "VXUS",
    "VWO",
    "VMBS",
    "BND",
    "GLD",
    "DBC",
    "VDE",
    "VFH",
]
V13_REDEPLOY_UNIVERSE: list[str] = [
    "VOO",
    "VXUS",
    "VWO",
    "VMBS",
    "BND",
    "GLD",
    "DBC",
    "VDE",
]

# ---------------------------------------------------------------------------
# v6.0 — Beta-Transformed Linear Pool (BLP) aggregation
# Replaces naive equal-weight ensemble averaging with a calibrated pool.
# Ranjan & Gneiting (2010): any linear pool of calibrated forecasts is
# necessarily uncalibrated; BLP corrects this via a Beta CDF transformation.
#
# The BLP has 5 free parameters:
#   a, b   — Beta distribution shape parameters (the transformation)
#   w_1..3 — Linear pool weights for models 1-3; w_4 = 1 − Σw_1..3
#             (4 models → 3 independent weights)
#
# Parameter fitting uses maximum likelihood on OOS probability sequences.
# Requires BLP_MIN_OOS_MONTHS months of live OOS predictions to accumulate
# before fitting — guard enforced by BLPModel.fit().
# Target activation: Week 8 = 2026-05-20 (counting from first live run).
# ---------------------------------------------------------------------------
BLP_MIN_OOS_MONTHS: int = 12     # Minimum live OOS months required to fit BLP
BLP_N_PARAMS: int = 5            # 2 Beta shape + 3 independent weights
BLP_BETA_A_INIT: float = 1.0     # Initial Beta(a) shape (a=b=1 → uniform)
BLP_BETA_B_INIT: float = 1.0     # Initial Beta(b) shape
BLP_WEIGHT_INIT: float = 0.25    # Initial equal weight per model (4 models)
