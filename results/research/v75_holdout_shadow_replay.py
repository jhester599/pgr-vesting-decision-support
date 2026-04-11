"""v75 - Holdout-era monthly replay for the v74 quality-weighted shadow path."""

from __future__ import annotations

import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

import config
from scripts import monthly_decision
from src.database import db_client
from src.research.v37_utils import RESULTS_DIR, print_footer, print_header
from src.research.v75 import V75Decision, choose_v75_decision, holdout_monthly_review_dates, summarize_v75_review


def _build_path_rows(as_of: date, conn) -> list[dict[str, object]]:
    signals, ensemble_results, diagnostics = monthly_decision._generate_signals(  # noqa: SLF001
        conn,
        as_of,
        target_horizon_months=6,
    )
    if signals.empty or not ensemble_results:
        return []

    aggregate_health = monthly_decision._compute_aggregate_health(  # noqa: SLF001
        ensemble_results,
        target_horizon_months=6,
    )
    if aggregate_health is None:
        return []

    consensus, mean_pred, mean_ic, mean_hr, mean_prob, confidence_tier = monthly_decision._consensus_signal(  # noqa: SLF001
        signals
    )
    live_mode = monthly_decision._determine_recommendation_mode(  # noqa: SLF001
        consensus,
        mean_pred,
        mean_ic,
        mean_hr,
        aggregate_health,
        diagnostics.get("representative_cpcv"),
    )
    rows = [
        {
            "as_of": as_of.isoformat(),
            "path_name": "live_equal_weight",
            "consensus": consensus,
            "recommendation_mode": str(live_mode["label"]),
            "sell_pct": float(live_mode["sell_pct"]),
            "mean_predicted": float(mean_pred),
            "mean_ic": float(mean_ic),
            "mean_hit_rate": float(mean_hr),
            "mean_prob_outperform": float(mean_prob),
            "confidence_tier": confidence_tier,
            "top_benchmark": "",
            "top_benchmark_weight": 1.0 / max(len(signals), 1),
            "signal_agrees_with_live": True,
            "mode_agrees_with_live": True,
            "sell_agrees_with_live": True,
            "abs_sell_pct_diff_vs_live": 0.0,
        }
    ]

    shadow_df = monthly_decision._build_v74_shadow_consensus(  # noqa: SLF001
        signals,
        aggregate_health,
        diagnostics.get("representative_cpcv"),
    )
    if shadow_df is None or shadow_df.empty:
        return rows

    quality_row = shadow_df[shadow_df["variant"] == "quality_weighted"]
    if quality_row.empty:
        return rows

    quality = quality_row.iloc[0]
    rows.append(
        {
            "as_of": as_of.isoformat(),
            "path_name": "v74_quality_weighted",
            "consensus": str(quality["consensus"]),
            "recommendation_mode": str(quality["recommendation_mode"]),
            "sell_pct": float(quality["recommended_sell_pct"]),
            "mean_predicted": float(quality["mean_predicted_return"]),
            "mean_ic": float(quality["mean_ic"]),
            "mean_hit_rate": float(quality["mean_hit_rate"]),
            "mean_prob_outperform": float(quality["mean_prob_outperform"]),
            "confidence_tier": str(quality["confidence_tier"]),
            "top_benchmark": str(quality["top_benchmark"]),
            "top_benchmark_weight": float(quality["top_benchmark_weight"]),
            "signal_agrees_with_live": str(quality["consensus"]) == consensus,
            "mode_agrees_with_live": str(quality["recommendation_mode"]) == str(live_mode["label"]),
            "sell_agrees_with_live": abs(float(quality["recommended_sell_pct"]) - float(live_mode["sell_pct"])) < 1e-9,
            "abs_sell_pct_diff_vs_live": abs(float(quality["recommended_sell_pct"]) - float(live_mode["sell_pct"])),
        }
    )
    return rows


def run_v75_holdout_shadow_replay(end_as_of: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame, V75Decision]:
    """Run the v75 holdout-era replay and return detail, summary, and decision."""
    if end_as_of is None:
        end_as_of = date.today()

    review_dates = holdout_monthly_review_dates(end_as_of, holdout_start="2024-04-01")
    conn = db_client.get_connection(config.DB_PATH)
    try:
        rows: list[dict[str, object]] = []
        for as_of in review_dates:
            rows.extend(_build_path_rows(as_of, conn))
    finally:
        conn.close()

    detail_df = pd.DataFrame(rows)
    summary_df = summarize_v75_review(detail_df)
    decision = choose_v75_decision(summary_df)
    return detail_df, summary_df, decision


def _write_decision_memo(path: Path, review_dates: list[date], decision: V75Decision) -> None:
    lines = [
        "# v75 Holdout Shadow Replay Decision",
        "",
        f"- Review window start: `{review_dates[0].isoformat()}`" if review_dates else "- Review window start: `n/a`",
        f"- Review window end: `{review_dates[-1].isoformat()}`" if review_dates else "- Review window end: `n/a`",
        f"- Decision status: **{decision.status}**",
        f"- Recommended path: **{decision.recommended_path}**",
        "",
        f"{decision.rationale}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    end_as_of = date.today()
    review_dates = holdout_monthly_review_dates(end_as_of, holdout_start="2024-04-01")
    detail_df, summary_df, decision = run_v75_holdout_shadow_replay(end_as_of=end_as_of)

    print_header("v75", "Holdout-Era Shadow Replay")
    if not summary_df.empty:
        print(summary_df.to_string(index=False, float_format="{:.4f}".format))
    print("")
    print(f"Decision: {decision.status} -> {decision.recommended_path}")
    print(decision.rationale)
    print_footer()

    detail_path = RESULTS_DIR / "v75_holdout_shadow_replay_detail.csv"
    summary_path = RESULTS_DIR / "v75_holdout_shadow_replay_summary.csv"
    memo_path = RESULTS_DIR / "v75_holdout_shadow_replay_decision.md"
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    _write_decision_memo(memo_path, review_dates, decision)
    print(f"\nSaved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {memo_path}")


if __name__ == "__main__":
    main()
