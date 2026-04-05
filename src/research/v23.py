"""Helpers for the v23 extended-history benchmark-proxy study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.processing.multi_total_return import build_etf_monthly_returns
from src.research.v20 import V20_FORECAST_UNIVERSE


V23_REVIEW_PATHS: tuple[str, ...] = (
    "live_production_ensemble_reduced",
    "ensemble_ridge_gbt_v18",
    "ensemble_ridge_gbt_v20_best",
)


@dataclass(frozen=True)
class V23Decision:
    """Decision record for the extended-history proxy study."""

    status: str
    recommended_path: str
    rationale: str


def fit_proxy_blend_weights(
    target_returns: pd.Series,
    component_returns: dict[str, pd.Series],
    fallback_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Fit non-negative, sum-to-one proxy blend weights on the overlapping history."""
    frame = pd.DataFrame({"target": target_returns})
    for name, series in component_returns.items():
        frame[name] = series
    aligned = frame.dropna()
    if aligned.empty:
        if fallback_weights is not None:
            return dict(fallback_weights)
        n = len(component_returns)
        return {name: 1.0 / n for name in component_returns}

    X = aligned[list(component_returns)].to_numpy(copy=True)
    y = aligned["target"].to_numpy(copy=True)
    weights, *_ = np.linalg.lstsq(X, y, rcond=None)
    weights = np.clip(weights, 0.0, None)
    if float(weights.sum()) <= 0.0:
        if fallback_weights is not None:
            return dict(fallback_weights)
        weights = np.repeat(1.0 / len(component_returns), len(component_returns))
    else:
        weights = weights / float(weights.sum())
    return {
        name: float(weight)
        for name, weight in zip(component_returns.keys(), weights)
    }


def stitch_proxy_series(
    actual: pd.Series,
    proxy: pd.Series,
) -> pd.Series:
    """Use the proxy only before the first valid actual observation."""
    actual_valid = actual.dropna()
    if actual_valid.empty:
        return proxy.sort_index()
    first_actual = actual_valid.index.min()
    combined = pd.concat([actual.rename("actual"), proxy.rename("proxy")], axis=1).sort_index()
    stitched = combined["actual"].copy()
    pre_inception_mask = stitched.isna() & (combined.index < first_actual)
    stitched.loc[pre_inception_mask] = combined.loc[pre_inception_mask, "proxy"]
    return stitched


def build_stitched_benchmark_returns(
    conn: Any,
    forward_months: int,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Build research-only stitched benchmark forward-return histories."""
    benchmark_returns: dict[str, pd.Series] = {}
    manifest_rows: list[dict[str, object]] = []

    voo_actual = build_etf_monthly_returns(conn, "VOO", forward_months)
    voo_proxy = build_etf_monthly_returns(conn, "VTI", forward_months)
    benchmark_returns["VOO"] = stitch_proxy_series(voo_actual, voo_proxy)
    voo_overlap = pd.concat([voo_actual.rename("actual"), voo_proxy.rename("proxy")], axis=1).dropna()
    manifest_rows.append(
        {
            "benchmark": "VOO",
            "proxy_type": "single",
            "proxy_components": "VTI",
            "weights": "VTI=1.0000",
            "actual_start": voo_actual.dropna().index.min().date().isoformat() if not voo_actual.dropna().empty else "",
            "proxy_start": voo_proxy.dropna().index.min().date().isoformat() if not voo_proxy.dropna().empty else "",
            "stitched_start": benchmark_returns["VOO"].dropna().index.min().date().isoformat()
            if not benchmark_returns["VOO"].dropna().empty
            else "",
            "overlap_corr": float(voo_overlap["actual"].corr(voo_overlap["proxy"])) if not voo_overlap.empty else float("nan"),
        }
    )

    vxus_actual = build_etf_monthly_returns(conn, "VXUS", forward_months)
    vea_proxy = build_etf_monthly_returns(conn, "VEA", forward_months)
    vwo_proxy = build_etf_monthly_returns(conn, "VWO", forward_months)
    vxus_weights = fit_proxy_blend_weights(
        vxus_actual,
        {"VEA": vea_proxy, "VWO": vwo_proxy},
        fallback_weights={"VEA": 0.75, "VWO": 0.25},
    )
    vxus_proxy = sum(series * vxus_weights[name] for name, series in {"VEA": vea_proxy, "VWO": vwo_proxy}.items())
    benchmark_returns["VXUS"] = stitch_proxy_series(vxus_actual, vxus_proxy)
    vxus_overlap = pd.concat([vxus_actual.rename("actual"), vxus_proxy.rename("proxy")], axis=1).dropna()
    manifest_rows.append(
        {
            "benchmark": "VXUS",
            "proxy_type": "blend",
            "proxy_components": "VEA,VWO",
            "weights": ",".join(f"{name}={weight:.4f}" for name, weight in vxus_weights.items()),
            "actual_start": vxus_actual.dropna().index.min().date().isoformat() if not vxus_actual.dropna().empty else "",
            "proxy_start": vxus_proxy.dropna().index.min().date().isoformat() if not vxus_proxy.dropna().empty else "",
            "stitched_start": benchmark_returns["VXUS"].dropna().index.min().date().isoformat()
            if not benchmark_returns["VXUS"].dropna().empty
            else "",
            "overlap_corr": float(vxus_overlap["actual"].corr(vxus_overlap["proxy"])) if not vxus_overlap.empty else float("nan"),
        }
    )

    vmbs_actual = build_etf_monthly_returns(conn, "VMBS", forward_months)
    bnd_proxy = build_etf_monthly_returns(conn, "BND", forward_months)
    benchmark_returns["VMBS"] = stitch_proxy_series(vmbs_actual, bnd_proxy)
    vmbs_overlap = pd.concat([vmbs_actual.rename("actual"), bnd_proxy.rename("proxy")], axis=1).dropna()
    manifest_rows.append(
        {
            "benchmark": "VMBS",
            "proxy_type": "single",
            "proxy_components": "BND",
            "weights": "BND=1.0000",
            "actual_start": vmbs_actual.dropna().index.min().date().isoformat() if not vmbs_actual.dropna().empty else "",
            "proxy_start": bnd_proxy.dropna().index.min().date().isoformat() if not bnd_proxy.dropna().empty else "",
            "stitched_start": benchmark_returns["VMBS"].dropna().index.min().date().isoformat()
            if not benchmark_returns["VMBS"].dropna().empty
            else "",
            "overlap_corr": float(vmbs_overlap["actual"].corr(vmbs_overlap["proxy"])) if not vmbs_overlap.empty else float("nan"),
        }
    )

    for benchmark in V20_FORECAST_UNIVERSE:
        if benchmark in benchmark_returns:
            continue
        series = build_etf_monthly_returns(conn, benchmark, forward_months)
        benchmark_returns[benchmark] = series
        manifest_rows.append(
            {
                "benchmark": benchmark,
                "proxy_type": "actual_only",
                "proxy_components": "",
                "weights": "",
                "actual_start": series.dropna().index.min().date().isoformat() if not series.dropna().empty else "",
                "proxy_start": "",
                "stitched_start": series.dropna().index.min().date().isoformat() if not series.dropna().empty else "",
                "overlap_corr": float("nan"),
            }
        )

    return benchmark_returns, pd.DataFrame(manifest_rows).sort_values("benchmark").reset_index(drop=True)


def build_extended_relative_return_series(
    conn: Any,
    forward_months: int,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Build PGR-minus-benchmark relative-return series using stitched benchmark histories."""
    pgr_returns = build_etf_monthly_returns(conn, "PGR", forward_months)
    benchmark_returns, manifest = build_stitched_benchmark_returns(conn, forward_months)

    relative_map: dict[str, pd.Series] = {}
    coverage_rows: list[dict[str, object]] = []
    for benchmark, benchmark_series in benchmark_returns.items():
        aligned = pd.DataFrame({"pgr": pgr_returns, "benchmark": benchmark_series}).dropna()
        if aligned.empty:
            continue
        relative = (aligned["pgr"] - aligned["benchmark"]).rename(f"{benchmark}_{forward_months}m")
        relative_map[benchmark] = relative
        coverage_rows.append(
            {
                "benchmark": benchmark,
                "relative_start": relative.index.min().date().isoformat(),
                "relative_end": relative.index.max().date().isoformat(),
                "n_obs": int(relative.shape[0]),
            }
        )
    coverage_df = pd.DataFrame(coverage_rows).sort_values("benchmark").reset_index(drop=True)
    return relative_map, manifest.merge(coverage_df, on="benchmark", how="left")


def choose_v23_decision(
    metric_summary_df: pd.DataFrame,
    review_summary_df: pd.DataFrame,
) -> V23Decision:
    """Choose whether the v21 promotion result survives the extended-history check."""
    if metric_summary_df.empty or review_summary_df.empty:
        return V23Decision(
            status="insufficient_data",
            recommended_path="live_production_ensemble_reduced",
            rationale="Required v23 inputs were missing.",
        )

    def _metric(name: str) -> pd.Series | None:
        match = metric_summary_df[metric_summary_df["candidate_name"] == name]
        return None if match.empty else match.iloc[0]

    def _review(name: str) -> pd.Series | None:
        match = review_summary_df[review_summary_df["path_name"] == name]
        return None if match.empty else match.iloc[0]

    live_metric = _metric("live_production_ensemble_reduced")
    baseline_metric = _metric("baseline_historical_mean")
    live_review = _review("live_production_ensemble_reduced")
    if live_metric is None or baseline_metric is None or live_review is None:
        return V23Decision(
            status="insufficient_data",
            recommended_path="live_production_ensemble_reduced",
            rationale="A required live or baseline row was missing from the v23 inputs.",
        )

    candidate_rows = metric_summary_df[metric_summary_df["candidate_name"].isin(V23_REVIEW_PATHS)].copy()
    candidate_rows = candidate_rows[candidate_rows["candidate_name"] != "live_production_ensemble_reduced"]
    candidate_rows = candidate_rows.sort_values(
        by=["mean_policy_return_sign", "mean_oos_r2", "mean_ic"],
        ascending=[False, False, False],
    )
    if candidate_rows.empty:
        return V23Decision(
            status="insufficient_data",
            recommended_path="live_production_ensemble_reduced",
            rationale="No v23 candidate rows were available.",
        )
    best_candidate = candidate_rows.iloc[0]
    candidate_name = str(best_candidate["candidate_name"])
    candidate_review = _review(candidate_name)
    if candidate_review is None:
        return V23Decision(
            status="insufficient_data",
            recommended_path="live_production_ensemble_reduced",
            rationale="The selected v23 candidate was missing from the extended-history review summary.",
        )

    if (
        float(best_candidate["mean_policy_return_sign"]) >= float(baseline_metric["mean_policy_return_sign"]) - 0.002
        and float(best_candidate["mean_oos_r2"]) > float(live_metric["mean_oos_r2"])
        and float(candidate_review["signal_agreement_with_shadow_rate"])
        >= float(live_review["signal_agreement_with_shadow_rate"])
    ):
        return V23Decision(
            status="extended_history_confirms_candidate",
            recommended_path=candidate_name,
            rationale=(
                "The leading candidate retained or improved its historical behavior versus the simpler baseline "
                "even after extending the benchmark histories with research-only proxies."
            ),
        )

    return V23Decision(
        status="extended_history_reverts_to_live",
        recommended_path="live_production_ensemble_reduced",
        rationale=(
            "The extended-history proxy study did not preserve the v21 promotion edge cleanly enough to keep "
            "the candidate ahead of the current visible cross-check."
        ),
    )


__all__ = [
    "V23Decision",
    "V23_REVIEW_PATHS",
    "build_extended_relative_return_series",
    "build_stitched_benchmark_returns",
    "choose_v23_decision",
    "fit_proxy_blend_weights",
    "stitch_proxy_series",
]
