"""Renders ``diagnostic.md`` (OOS evaluation) and ``plots/calibration_curve.png``.
Functions in other ``pgr_vds.decision`` modules are called through the module
(``health.compute_aggregate_health(...)``), so a test patches a function
once, in the module that defines it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from pgr_vds.decision import health, rendering
from src.models.calibration import CalibrationResult
from src.models.conformal import ConformalCoverageBacktest
from src.models.evaluation import FeatureImportanceStability
from src.reporting.decision_rendering import evaluate_quality_gates


def plot_calibration_curve(
    out_dir: Path,
    cal_probs: np.ndarray,
    cal_outcomes: np.ndarray,
    cal_result: CalibrationResult,
    n_bins: int | None = None,
) -> Path | None:
    """Generate a reliability diagram (calibration curve) and save to disk.

    The reliability diagram plots predicted probability (x-axis) against the
    fraction of positive outcomes in each bin (y-axis).  A perfectly calibrated
    model lies on the diagonal.  Points above the diagonal are under-confident;
    points below are over-confident.

    The plot includes:
      - Binned calibration curve (blue circles, connected)
      - Diagonal perfect-calibration reference (dashed grey)
      - Histogram of predicted probabilities (bottom subpanel)
      - ECE annotation with 95% bootstrap CI

    Args:
        out_dir:      Output directory (YYYY-MM folder).
        cal_probs:    Array of pooled calibrated P(outperform) values.
        cal_outcomes: Array of corresponding binary outcomes (1 = outperform).
        cal_result:   CalibrationResult for ECE annotation.
        n_bins:       Number of probability bins (default: config.CALIBRATION_N_BINS).

    Returns:
        Path to the saved PNG, or ``None`` if insufficient data.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [calibration plot] matplotlib not available — skipping plot.")
        return None

    if len(cal_probs) < 4 or cal_result.method == "uncalibrated":
        return None

    if n_bins is None:
        n_bins = getattr(config, "CALIBRATION_N_BINS", 10)

    # Compute reliability bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers: list[float] = []
    fraction_pos: list[float] = []
    bin_counts: list[int] = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (cal_probs >= lo) & (cal_probs < hi)
        if mask.sum() == 0:
            continue
        bin_centers.append(float(cal_probs[mask].mean()))
        fraction_pos.append(float(cal_outcomes[mask].mean()))
        bin_counts.append(int(mask.sum()))

    if len(bin_centers) < 2:
        return None

    # ---- Plot ----
    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 7),
        gridspec_kw={"height_ratios": [4, 1]},
    )

    # Reliability curve
    ax_main.plot([0, 1], [0, 1], "--", color="grey", linewidth=1.2, label="Perfect calibration")
    ax_main.plot(bin_centers, fraction_pos, "o-", color="#1f77b4", linewidth=2,
                 markersize=7, label="Model calibration")

    # Annotate ECE
    ece_txt = (
        f"ECE = {cal_result.ece:.1%}  "
        f"[95% CI: {cal_result.ece_ci_lower:.1%}–{cal_result.ece_ci_upper:.1%}]"
    )
    ax_main.text(
        0.04, 0.95, ece_txt,
        transform=ax_main.transAxes,
        fontsize=9, verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.8},
    )

    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlabel("Mean predicted probability")
    ax_main.set_ylabel("Fraction of positives (actual)")
    ax_main.set_title(
        f"Calibration Reliability Diagram  "
        f"(n={cal_result.n_obs:,} obs, {cal_result.method})"
    )
    ax_main.legend(fontsize=9)
    ax_main.grid(True, alpha=0.3)

    # Histogram of predicted probabilities
    ax_hist.hist(cal_probs, bins=n_bins, range=(0, 1), color="#1f77b4", alpha=0.6)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / "calibration_curve.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  Wrote {save_path}")
    return save_path


def write_diagnostic_report(
    out_dir: Path,
    as_of: date,
    ensemble_results: dict,
    target_horizon_months: int = 6,
    cal_result: CalibrationResult | None = None,
    signals: pd.DataFrame | None = None,
    obs_feature_report: dict | None = None,
    conformal_coverage_summary: ConformalCoverageBacktest | None = None,
    importance_stability: FeatureImportanceStability | None = None,
    vif_series: pd.Series | None = None,
    benchmark_quality_df: pd.DataFrame | None = None,
    shadow_gate_overlay: dict[str, Any] | None = None,
    classifier_monitoring_summary: dict[str, Any] | None = None,
    aggregate_health: dict | None = None,
    shrinkage_alpha: float | None = None,
    panel: pd.DataFrame | None = None,
) -> None:
    """
    Write diagnostic.md alongside recommendation.md.

    Aggregates the realised-only (prequential) ensemble OOS record across all
    benchmarks (review 2026-09-25, WP7), then reports:

    - Campbell-Thompson OOS R² against each benchmark's prevailing mean of
      the targets realised by each forecast date
    - pooled Spearman IC with a Driscoll-Kraay p-value clustered by date
    - hit rate against the base rate, with the Pesaran-Timmermann test
    - Clark-West against the same naive benchmark
    - the readiness gates: walk-forward completion for every required
      model/benchmark pair and required inputs at the as-of date (R3)
    - prequential ECE and trailing conformal coverage
    - Per-benchmark health table from ensemble OOS predictions

    Args:
        out_dir:                Output directory (YYYY-MM folder).
        as_of:                  As-of date for the report header.
        ensemble_results:       Dict of ETF ticker → EnsembleWFOResult from
                                ``run_ensemble_benchmarks()``.
        target_horizon_months:  Forward return horizon used during training.
        obs_feature_report:     Output of ``compute_obs_feature_ratio()`` for the
                                feature matrix used in this monthly run.
        aggregate_health:       ``health.compute_aggregate_health`` output; computed
                                here when omitted.
        shrinkage_alpha:        Live prequential shrinkage alpha.
        panel:                  Prequential OOS panel; built when omitted.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "diagnostic.md"

    nw_lags = target_horizon_months - 1  # HAC lags over months (5 for 6M)
    if aggregate_health is None:
        aggregate_health = health.compute_aggregate_health(
            ensemble_results,
            target_horizon_months=target_horizon_months,
            panel=panel,
        )
    if aggregate_health is not None:
        agg_realized = aggregate_health["agg_realized"]
        computed_quality_df = aggregate_health["benchmark_quality_df"]
    else:
        agg_realized = pd.Series(dtype=float)
        computed_quality_df = pd.DataFrame()
    if benchmark_quality_df is None:
        benchmark_quality_df = computed_quality_df

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    if aggregate_health is None or len(agg_realized) < 4:
        # Not enough data — write a minimal report
        lines = [
            f"# PGR Diagnostic Report — {as_of.strftime('%B %Y')}",
            "",
            "> ⚠️ Insufficient OOS observations for aggregate diagnostics "
            f"(need ≥ 4, got {len(agg_realized)}).",
            "",
            f"*Generated by `{rendering.REPORT_GENERATED_BY}`*",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Wrote {path} (insufficient data)")
        return

    oos_r2 = float(aggregate_health["oos_r2"])
    nw_ic = float(aggregate_health["nw_ic"])
    nw_pval = float(aggregate_health["nw_pval"])
    agg_hit = float(aggregate_health["agg_hit"])
    base_rate = float(aggregate_health["base_rate"])
    constant_rule = float(aggregate_health["constant_rule_hit_rate"])
    hit_excess = float(aggregate_health["hit_rate_excess"])
    pt_p_value = float(aggregate_health["pt_p_value"])
    cw_t_stat = float(aggregate_health["cw_t_stat"])
    cw_p_value = float(aggregate_health["cw_p_value"])
    n_agg = int(aggregate_health["n_agg"])
    n_calls = int(aggregate_health.get("n_calls", n_agg))
    n_dates = int(aggregate_health.get("n_dates", 0))

    r2_flag = health.flag(oos_r2, config.DIAG_MIN_OOS_R2, 0.005)
    ic_flag_agg = health.flag(nw_ic, config.DIAG_MIN_IC, 0.03)
    pt_flag = health.directional_flag(pt_p_value)
    pt_text = f"{pt_p_value:.4f}" if np.isfinite(pt_p_value) else "undefined"

    sig_marker = "\u2705 p < 0.05" if (not np.isnan(nw_pval) and nw_pval < 0.05) else (
        "\u26a0\ufe0f p < 0.10" if (not np.isnan(nw_pval) and nw_pval < 0.10) else "\u274c not sig."
    )
    cw_marker = "\u2705 p < 0.05" if (not np.isnan(cw_p_value) and cw_p_value < 0.05) else (
        "\u26a0\ufe0f p < 0.10" if (not np.isnan(cw_p_value) and cw_p_value < 0.10) else "\u274c not sig."
    )

    # ------------------------------------------------------------------
    # Build markdown
    # ------------------------------------------------------------------
    readiness_gates = {
        gate.name: gate
        for gate in evaluate_quality_gates(float("nan"), aggregate_health)
        if gate.name in {"wfo_completed", "data_ready"}
    }
    status_marks = {"PASS": "✅", "MARGINAL": "⚠️", "FAIL": "❌"}
    readiness_rows = [
        f"| Walk-forward validation complete (gate) | {readiness_gates['wfo_completed'].current} | "
        f"{status_marks[readiness_gates['wfo_completed'].status]} | every required pair |",
        f"| Required inputs ready at as-of (gate) | {readiness_gates['data_ready'].current} | "
        f"{status_marks[readiness_gates['data_ready'].status]} | finite and fresh |",
    ]
    readiness_note_lines = [
        "> **Validation:** walk-forward only (TimeSeriesSplit, 60-month window, 6-month test folds,",
        f"> gap = horizon + purge buffer = {target_horizon_months + config.WFO_PURGE_BUFFER_6M} months). "
        "The combinatorial purged K-fold diagnostic",
        "> was retired (pre-v200 remediation R3): it trained on folds after its test folds.",
    ]

    obs_feature_lines: list[str] = []
    if obs_feature_report is not None:
        obs_feature_lines = [
            "",
            "## Feature Governance",
            "",
            "| Metric | Value | Status | Threshold (Good) |",
            "|--------|-------|--------|-----------------|",
            f"| Full obs/feature ratio | {obs_feature_report['ratio']:.2f} | "
            f"{ {'OK': '✅', 'WARNING': '⚠️', 'FAIL': '❌'}.get(obs_feature_report['verdict'], '⚠️') } | ≥ 4.0 |",
            f"| Per-fold obs/feature ratio | {obs_feature_report['per_fold_ratio']:.2f} | "
            f"{ {'OK': '✅', 'WARNING': '⚠️', 'FAIL': '❌'}.get(obs_feature_report['verdict'], '⚠️') } | ≥ 4.0 |",
            f"| Features in monthly run | {obs_feature_report['n_features']} | — | — |",
            f"| Fully populated observations | {obs_feature_report['n_obs']} | — | — |",
            "",
            f"> {obs_feature_report['message']}",
            "",
        ]

        # v32.0 — Feature importance stability subsection
        if importance_stability is not None:
            _stab_icon = {
                "STABLE": "✅",
                "MARGINAL": "⚠️",
                "UNSTABLE": "❌",
            }.get(importance_stability.verdict, "⚠️")
            obs_feature_lines += [
                "### Feature Importance Stability",
                "",
                "| Metric | Value | Status | Threshold (Good) |",
                "|--------|-------|--------|-----------------|",
                f"| Mean consecutive-fold Spearman ρ | "
                f"{importance_stability.stability_score:.4f} | "
                f"{_stab_icon} {importance_stability.verdict} | ≥ 0.70 |",
                f"| Folds included | {importance_stability.n_folds} | — | — |",
                "",
                "> Stability score measures mean pairwise Spearman rank-correlation "
                "between consecutive WFO fold importance rankings. "
                "A score < 0.40 indicates unstable feature rankings; "
                "model predictions may be driven by different features each period.",
                "",
                "**Top 10 features by mean WFO rank:**",
                "",
                "| Rank | Feature | Mean Rank | Rank Std | Mean |Importance| |",
                "|------|---------|-----------|----------|----------------|",
            ]
            top10 = importance_stability.per_feature.head(10)
            for feat, row in top10.iterrows():
                obs_feature_lines.append(
                    f"| {int(row['mean_rank']):.0f} | {feat} | "
                    f"{row['mean_rank']:.1f} | {row['rank_std']:.1f} | "
                    f"{row['mean_importance']:.4f} |"
                )
            obs_feature_lines += [""]

        # v32.1 — VIF multicollinearity subsection
        if vif_series is not None and not vif_series.empty:
            high_vif = vif_series[vif_series > config.VIF_HIGH_THRESHOLD]
            warn_vif = vif_series[
                (vif_series > config.VIF_WARN_THRESHOLD) &
                (vif_series <= config.VIF_HIGH_THRESHOLD)
            ]
            if not high_vif.empty:
                vif_overall = "❌ HIGH multicollinearity"
            elif not warn_vif.empty:
                vif_overall = "⚠️ MODERATE multicollinearity"
            else:
                vif_overall = "✅ LOW multicollinearity"
            obs_feature_lines += [
                "### Multicollinearity (VIF)",
                "",
                f"**Overall:** {vif_overall}  ",
                f"Features flagged high (VIF > {config.VIF_HIGH_THRESHOLD:.0f}): "
                f"**{len(high_vif)}**  ",
                f"Features flagged moderate (VIF {config.VIF_WARN_THRESHOLD:.0f}–"
                f"{config.VIF_HIGH_THRESHOLD:.0f}): **{len(warn_vif)}**  ",
                "",
                "> VIF measures how much variance in a feature is explained by the "
                "other features. VIF > 10 indicates severe multicollinearity and "
                "may cause unstable coefficient estimates.",
                "",
                "**All features by VIF (descending):**",
                "",
                "| Feature | VIF | Status |",
                "|---------|-----|--------|",
            ]
            for feat, vif_val in vif_series.items():
                if vif_val > config.VIF_HIGH_THRESHOLD:
                    flag = "❌ HIGH"
                elif vif_val > config.VIF_WARN_THRESHOLD:
                    flag = "⚠️ MODERATE"
                else:
                    flag = "✅ OK"
                obs_feature_lines.append(
                    f"| {feat} | {vif_val:.2f} | {flag} |"
                )
            obs_feature_lines += [""]

        obs_feature_lines += [
            "---",
            "",
        ]

    lines = [
        f"# PGR Diagnostic Report — {as_of.strftime('%B %Y')}",
        "",
        f"**As-of Date:** {as_of}  ",
        f"**Horizon:** {target_horizon_months}M  ",
        f"**OOS observations (aggregate):** {n_agg} over {n_dates} months  ",
        f"**HAC lags:** {nw_lags} months (accounts for the {target_horizon_months - 1}-month "
        "return-window overlap). Pooled p-values are Driscoll-Kraay: Newey-West on the monthly "
        "sums of the scores, i.e. clustered by date. Per-benchmark p-values are Newey-West.  ",
        (
            f"**Prequential shrinkage alpha (live):** {float(shrinkage_alpha):.3f}  "
            if shrinkage_alpha is not None and np.isfinite(float(shrinkage_alpha))
            else "**Prequential shrinkage alpha (live):** n/a  "
        ),
        "",
        "> All metrics below are realised-only (review 2026-09-25, WP7): every OOS month uses",
        "> ensemble weights, shrinkage and calibration fitted on targets realised by that month,",
        "> and OOS R² / Clark-West compare with each benchmark's prevailing mean of the targets",
        "> realised by then (training history included).",
        "",
        "---",
        "",
        "## Aggregate Model Health",
        "",
        "| Metric | Value | Status | Threshold (Good) |",
        "|--------|-------|--------|-----------------|",
        f"| OOS R² vs prevailing mean (gate) | {oos_r2:.4f} ({oos_r2:.2%}) | {r2_flag} | ≥ {config.DIAG_MIN_OOS_R2:.2%} |",
        f"| Pooled IC (Spearman) | {nw_ic:.4f} | {ic_flag_agg} | ≥ {config.DIAG_MIN_IC:.2f} |",
        f"| IC significance (clustered by date) | {nw_pval:.4f} | {sig_marker} | p < 0.05 |",
        f"| Clark-West t-stat | {cw_t_stat:.4f} | {cw_marker} | p < 0.05 |",
        f"| Clark-West p-value | {cw_p_value:.4f} | {cw_marker} | p < 0.05 |",
        (
            f"| Hit Rate ({n_calls:,} calls; {n_agg - n_calls:,} zero forecasts make no call) | {agg_hit:.1%} | — | not gated |"
            if n_calls < n_agg
            else f"| Hit Rate | {agg_hit:.1%} | — | not gated |"
        ),
        f"| Base rate P(PGR outperforms), same rows | {base_rate:.1%} | — | constant-sign rule hits {constant_rule:.1%} |",
        f"| Hit rate − base rate | {hit_excess:+.1%} | — | > 0 |",
        f"| Directional skill (Pesaran–Timmermann p, gate) | {pt_text} | {pt_flag} | p < {config.DIAG_MAX_DIRECTIONAL_PVALUE:.2f} |",
        *readiness_rows,
        "",
        "> The recommendation gate uses the equal-weight mean of the per-benchmark ICs",
        "> (see recommendation.md), not the pooled IC above.",
        "",
        *readiness_note_lines,
        "",
        "---",
        "",
        *obs_feature_lines,
        "## Calibration Phase",
        "",
        "| Phase | Description | Status |",
        "|-------|-------------|--------|",
    ]

    # Determine which phase is active based on cal_result
    if cal_result is None or cal_result.method == "uncalibrated":
        phase1_status = "✅ Active"
        phase2_status = f"⏳ Activates at n ≥ {config.CALIBRATION_MIN_OBS_PLATT}"
        phase3_status = f"⏳ Activates at n ≥ {config.CALIBRATION_MIN_OBS_ISOTONIC}"
    elif cal_result.method == "platt":
        phase1_status = "⬛ Superseded"
        phase2_status = (
            f"✅ Active (prequential ECE={cal_result.ece:.1%} "
            f"[{cal_result.ece_ci_lower:.1%}–{cal_result.ece_ci_upper:.1%}] "
            f"over n={cal_result.n_obs:,} OOS benchmark-months)"
        )
        phase3_status = f"⏳ Activates at n ≥ {config.CALIBRATION_MIN_OBS_ISOTONIC}"
    else:  # isotonic
        phase1_status = "⬛ Superseded"
        phase2_status = "⬛ Superseded by Phase 3"
        phase3_status = (
            f"✅ Active (n={cal_result.n_obs:,}  ECE={cal_result.ece:.1%} "
            f"[{cal_result.ece_ci_lower:.1%}–{cal_result.ece_ci_upper:.1%}])"
        )

    lines += [
        f"| Phase 1 | Uncalibrated (P = 50%; too few realised OOS rows) | {phase1_status} |",
        f"| Phase 2 | Platt scaling (logistic regression on OOS scores → binary) | {phase2_status} |",
        f"| Phase 3 | Platt → Isotonic (non-parametric; monotone reliability) | {phase3_status} |",
        "",
        "> ECE is prequential: each OOS month is scored by the per-benchmark calibrator fitted only",
        "> on months whose 6-month outcome was already known, as the monthly run would have done.",
        "> Confidence tiers read the calibrated P(outperform) in the direction of each signal.",
        "",
        "---",
        "",
        "## Conformal Prediction Intervals",
        "",
        f"**Method:** {config.CONFORMAL_METHOD.upper()} "
        f"({'Adaptive Conformal Inference — adjusts α_t for distribution shift' if config.CONFORMAL_METHOD == 'aci' else 'Split Conformal — finite-sample corrected quantile of absolute residuals'})  ",
        f"**Nominal Coverage:** {config.CONFORMAL_COVERAGE:.0%}  ",
        "",
        "> Coverage is trailing and prequential: each of the last 12 OOS months per benchmark is",
        "> scored with an interval calibrated only on residuals realised by that month.",
        "",
    ]

    # Build per-benchmark coverage table from signals CI columns
    has_ci_data = (
        signals is not None
        and not signals.empty
        and "ci_trailing_empirical_coverage" in signals.columns
        and "ci_n_calibration" in signals.columns
    )
    if has_ci_data and signals is not None:
        valid_rows = signals[signals["ci_n_calibration"] > 0]
        if not valid_rows.empty:
            if conformal_coverage_summary is not None:
                trailing_cov_flag = health.flag(
                    abs(conformal_coverage_summary.trailing_coverage_gap),
                    good=0.05,
                    marginal=0.10,
                    higher_is_better=False,
                )
                lines += [
                    f"**Mean trailing 12-point empirical coverage:** "
                    f"{conformal_coverage_summary.trailing_empirical_coverage:.1%} "
                    f"(gap {conformal_coverage_summary.trailing_coverage_gap:+.1%} vs nominal) "
                    f"{trailing_cov_flag}  ",
                    "",
                ]
            lines += [
                "| Benchmark | Description | Predicted Return | CI Lower | CI Upper | CI Width | Trailing 12 Coverage | N Cal |",
                "|-----------|-------------|----------------|----------|----------|----------|----------------------|-------|",
            ]
            for ticker, row in signals.iterrows():
                if pd.isna(row.get("ci_lower")):
                    continue
                desc = rendering.ETF_DESCRIPTIONS.get(str(ticker), str(ticker))
                pred_str = f"{row['predicted_relative_return']:+.2%}" if not pd.isna(row.get("predicted_relative_return")) else "n/a"
                ci_lo = f"{row['ci_lower']:+.2%}"
                ci_hi = f"{row['ci_upper']:+.2%}"
                ci_w = f"{row['ci_width']:.2%}"
                trailing_value = row.get("ci_trailing_empirical_coverage")
                if pd.isna(trailing_value):
                    trailing_cov = "n/a"
                else:
                    trailing_flag = health.flag(
                        abs(float(trailing_value) - config.CONFORMAL_COVERAGE),
                        good=0.05,
                        marginal=0.10,
                        higher_is_better=False,
                    )
                    trailing_cov = f"{float(trailing_value):.1%} {trailing_flag}"
                n_cal = int(row["ci_n_calibration"])
                lines.append(
                    f"| {ticker} | {desc} | {pred_str} | {ci_lo} | {ci_hi} | {ci_w} | "
                    f"{trailing_cov} | {n_cal} |"
                )
        else:
            lines.append("> ⚠️ No benchmarks had sufficient calibration data for conformal intervals.")
    else:
        lines.append("> ⚠️ Conformal interval data not available (signals not passed to diagnostic).")

    lines += [
        "",
        "> **Interpretation:** The CI width reflects model uncertainty — wider intervals indicate",
        "> larger historical prediction errors.  ACI dynamically adjusts coverage when errors",
        "> cluster (distribution shift), providing stronger guarantees than static split conformal.",
        "",
        "---",
        "",
        "## Per-Benchmark Health",
        "",
        "| Benchmark | Description | N OOS | OOS R² | IC | NW p | Hit Rate | Base Rate | PT p | CW t | CW p |",
        "|-----------|-------------|-------|--------|----|------|----------|-----------|------|------|------|",
    ]

    benchmark_rows = (
        benchmark_quality_df.to_dict("records")
        if benchmark_quality_df is not None and not benchmark_quality_df.empty
        else []
    )
    for row in benchmark_rows:
        desc = rendering.ETF_DESCRIPTIONS.get(str(row["benchmark"]), str(row["benchmark"]))
        base = row.get("base_rate", float("nan"))
        pt_p = row.get("pt_p_value", float("nan"))
        lines.append(
            f"| {row['benchmark']} | {desc} | {row['n_obs']} "
            f"| {row['oos_r2']:.2%} | {row['nw_ic']:.4f} | {row.get('nw_p_value', float('nan')):.4f} "
            f"| {row['hit_rate']:.1%} | {float(base):.1%} | {float(pt_p):.4f} "
            f"| {row['cw_t_stat']:.4f} | {row['cw_p_value']:.4f} |"
        )

    # Summary counts
    n_ok_ic = sum(1 for r in benchmark_rows if r["ic_flag"] == "✅")
    n_warn_ic = sum(1 for r in benchmark_rows if r["ic_flag"] == "⚠️")
    n_fail_ic = sum(1 for r in benchmark_rows if r["ic_flag"] == "❌")
    n_ok_hr = sum(1 for r in benchmark_rows if r["hr_flag"] == "✅")
    n_ok_cw = sum(1 for r in benchmark_rows if float(r["cw_p_value"]) < 0.05)

    lines += [
        "",
        f"**IC summary:** {n_ok_ic} ✅  {n_warn_ic} ⚠️  {n_fail_ic} ❌  "
        f"(of {len(benchmark_rows)} benchmarks)  ",
        f"**Directional skill ✅:** {n_ok_hr}/{len(benchmark_rows)} benchmarks with "
        f"Pesaran–Timmermann p < {config.DIAG_MAX_DIRECTIONAL_PVALUE:.2f}  ",
        f"**Clark-West ✅:** {n_ok_cw}/{len(benchmark_rows)} benchmarks with p < 0.05  ",
        "",
        "---",
        "",
        "## Shadow Gate Overlay",
        "",
        "| Field | Value |",
        "|-------|-------|",
        (
            f"| Variant | {shadow_gate_overlay.get('variant', 'n/a')} |"
            if isinstance(shadow_gate_overlay, dict)
            else "| Variant | n/a |"
        ),
        (
            f"| Recommendation Mode | {shadow_gate_overlay.get('recommendation_mode', 'n/a')} |"
            if isinstance(shadow_gate_overlay, dict)
            else "| Recommendation Mode | n/a |"
        ),
        (
            f"| Recommended Sell % | {float(shadow_gate_overlay.get('recommended_sell_pct', 0.0)):.0%} |"
            if isinstance(shadow_gate_overlay, dict)
            else "| Recommended Sell % | n/a |"
        ),
        (
            f"| Would Change Live Output | {'Yes' if shadow_gate_overlay.get('would_change') else 'No'} |"
            if isinstance(shadow_gate_overlay, dict)
            else "| Would Change Live Output | n/a |"
        ),
        (
            f"| Reason | {shadow_gate_overlay.get('reason', 'n/a')} |"
            if isinstance(shadow_gate_overlay, dict)
            else "| Reason | n/a |"
        ),
        (
            f"| P(Actionable Sell) | {float(shadow_gate_overlay['classifier_prob_actionable_sell']):.1%} |"
            if isinstance(shadow_gate_overlay, dict)
            and shadow_gate_overlay.get("classifier_prob_actionable_sell") is not None
            else "| P(Actionable Sell) | n/a |"
        ),
        "",
        "---",
        "",
        "## Classifier Monitoring",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        (
            f"| Matured observations | {classifier_monitoring_summary.get('matured_n', 0)} |"
            if isinstance(classifier_monitoring_summary, dict)
            else "| Matured observations | 0 |"
        ),
        (
            f"| Brier score | {float(classifier_monitoring_summary['brier_score']):.4f} |"
            if isinstance(classifier_monitoring_summary, dict)
            and classifier_monitoring_summary.get("brier_score") is not None
            else "| Brier score | n/a |"
        ),
        (
            f"| Log loss | {float(classifier_monitoring_summary['log_loss']):.4f} |"
            if isinstance(classifier_monitoring_summary, dict)
            and classifier_monitoring_summary.get("log_loss") is not None
            else "| Log loss | n/a |"
        ),
        (
            f"| ECE (10-bin) | {float(classifier_monitoring_summary['ece_10']):.4f} |"
            if isinstance(classifier_monitoring_summary, dict)
            and classifier_monitoring_summary.get("ece_10") is not None
            else "| ECE (10-bin) | n/a |"
        ),
        "",
        "> Matured-horizon diagnostics are computed only once the forecast horizon has elapsed.",
        "",
        "---",
        "",
        "## Threshold Reference",
        "",
        "| Metric | Good | Marginal | Failing | Source |",
        "|--------|------|----------|---------|--------|",
        "| OOS R² vs prevailing mean (gate) | > 2% | 0–2% | < 0% | Campbell & Thompson (2008) |",
        "| Mean IC, equal-weight (gate) | > 0.07 | 0.03–0.07 | < 0.03 | Harvey et al. (2016) |",
        "| Directional skill, PT p (gate) | < 0.05 | 0.05–0.10 | ≥ 0.10 | Pesaran & Timmermann (1992, 2009) |",
        "| Clark-West | p < 0.05 | p < 0.10 | ≥ 0.10 | Clark & West (2007) |",
        "| PBO | < 15% | 15–40% | > 40% | Bailey et al. (2014) |",
        "",
        "> Incomplete walk-forward results or required inputs that are missing, non-finite or stale",
        "> at the as-of date withhold ACTIONABLE (fail closed).",
        "",
        "---",
        "",
        f"*Generated by `{rendering.REPORT_GENERATED_BY}`*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {path}")
