"""
Monthly decision report generator for PGR RSU vesting decisions.

Runs on or after the 20th of each month (first business day ≥ 20th) via
GitHub Actions.  Generates a structured sell/hold recommendation based on
the latest available market data and the v3.0 multi-benchmark WFO model.

The monthly decision is a monitoring and signal-tracking tool — it does NOT
replace the biannual vesting-event recommendation.  Sell/hold decisions are
only executed at actual vesting dates (January and July).  Monthly runs
answer the question: "is the model reliably predicting PGR relative returns,
or did we get lucky on 20 coin flips?"

Output per run (in results/monthly_decisions/YYYY-MM/):
  recommendation.md      — Human-readable sell/hold report with signal details
  signals.csv            — Per-benchmark: ticker, IC, hit_rate, predicted_return, signal
  backtest_summary.csv   — Monthly stability stats at this point in time
  plots/                 — IC time series and regime breakdown charts

The decision_log.md in results/monthly_decisions/ is appended with one
summary row per run.

Usage:
    python scripts/monthly_decision.py [--as-of YYYY-MM-DD] [--dry-run] [--skip-fred]

Options:
    --as-of YYYY-MM-DD  Override the as-of date (default: today).
                        Use for back-dated runs and testing.
    --dry-run           Generate outputs but do not update the DB or
                        commit to git.  Useful for local testing.
    --skip-fred         Skip the FRED data fetch step.  Useful when
                        FRED_API_KEY is not set or during testing.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import config
from src.database import db_client
from src.models.multi_benchmark_wfo import get_current_signals, run_all_benchmarks
from src.processing.feature_engineering import (
    build_feature_matrix_from_db,
    get_feature_columns,
)
from src.processing.multi_total_return import load_relative_return_matrix
from src.reporting.backtest_report import (
    compute_oos_r_squared,
    export_backtest_to_csv,
    generate_rolling_ic_series,
)


# ---------------------------------------------------------------------------
# Business-day helpers
# ---------------------------------------------------------------------------

def _is_business_day(d: date) -> bool:
    """Return True if ``d`` is a weekday (Mon–Fri)."""
    return d.weekday() < 5


def _first_business_day_on_or_after(d: date) -> date:
    """Advance ``d`` to the next business day if it falls on a weekend."""
    while not _is_business_day(d):
        d += timedelta(days=1)
    return d


def _resolve_as_of_date(as_of_arg: str | None) -> date:
    """
    Determine the as-of date for this run.

    If ``as_of_arg`` is provided, parse it.  Otherwise use today.
    If the result falls on a weekend, advance to Monday.
    """
    if as_of_arg:
        return date.fromisoformat(as_of_arg)
    return _first_business_day_on_or_after(date.today())


def _output_dir(as_of: date) -> Path:
    """Return the output directory for the given month."""
    month_str = as_of.strftime("%Y-%m")
    return Path("results") / "monthly_decisions" / month_str


def _already_ran(as_of: date) -> bool:
    """Return True if a recommendation.md already exists for this month."""
    return (_output_dir(as_of) / "recommendation.md").exists()


# ---------------------------------------------------------------------------
# FRED fetch step
# ---------------------------------------------------------------------------

def _fetch_fred_step(conn, dry_run: bool = False, skip_fred: bool = False) -> None:
    """Fetch the latest FRED macro series and upsert into the DB."""
    if skip_fred or dry_run:
        if skip_fred:
            print("  [FRED] Skipping FRED fetch (--skip-fred).")
        else:
            print("  [FRED] Dry run — skipping FRED HTTP calls.")
        return

    from src.ingestion.fred_loader import fetch_all_fred_macro, upsert_fred_to_db

    print(f"  [FRED] Fetching {len(config.FRED_SERIES_MACRO)} macro series...")
    try:
        df = fetch_all_fred_macro(config.FRED_SERIES_MACRO)
        n = upsert_fred_to_db(conn, df)
        print(f"  [FRED] {n} rows upserted.")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FRED] WARNING: fetch failed: {exc}. Continuing with cached data.")


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def _generate_signals(
    conn,
    as_of: date,
    model_type: str = "elasticnet",
    target_horizon_months: int = 6,
) -> pd.DataFrame:
    """
    Build feature matrix (sliced to as_of), train WFO models, return signals.

    Returns a DataFrame with one row per benchmark and columns:
      benchmark, predicted_relative_return, ic, hit_rate, signal, top_feature.
    """
    as_of_ts = pd.Timestamp(as_of)

    df_full = build_feature_matrix_from_db(conn, force_refresh=True)
    feature_cols = get_feature_columns(df_full)
    X_full = df_full[feature_cols]

    # Strict temporal cutoff: only data available on or before as_of
    X_event = X_full.loc[X_full.index <= as_of_ts]
    if X_event.empty:
        return pd.DataFrame()

    X_current = X_event.iloc[[-1]]

    # Load relative return matrix for all benchmarks
    rel_matrix_cols = {}
    for etf in config.ETF_BENCHMARK_UNIVERSE:
        rel_series = load_relative_return_matrix(conn, etf, target_horizon_months)
        if not rel_series.empty:
            rel_matrix_cols[etf] = rel_series
    if not rel_matrix_cols:
        return pd.DataFrame()

    rel_matrix = pd.DataFrame(rel_matrix_cols)

    # Train all benchmarks
    wfo_results = run_all_benchmarks(
        X_event,
        rel_matrix,
        model_type=model_type,
        target_horizon_months=target_horizon_months,
    )

    # Generate current signals
    signals = get_current_signals(
        X_full=X_event,
        relative_return_matrix=rel_matrix,
        wfo_results=wfo_results,
        X_current=X_current,
        model_type=model_type,
    )

    return signals


# ---------------------------------------------------------------------------
# Consensus signal
# ---------------------------------------------------------------------------

def _consensus_signal(signals: pd.DataFrame) -> tuple[str, float, float, float]:
    """
    Derive a consensus signal from per-benchmark signals.

    Returns:
        (consensus_signal, mean_predicted_return, mean_ic, mean_hit_rate)
    """
    if signals.empty:
        return "NEUTRAL", 0.0, 0.0, 0.0

    mean_pred = float(signals["predicted_relative_return"].mean())
    mean_ic = float(signals["ic"].mean())
    mean_hr = float(signals["hit_rate"].mean())

    outperform_count = (signals["signal"] == "OUTPERFORM").sum()
    underperform_count = (signals["signal"] == "UNDERPERFORM").sum()
    neutral_count = (signals["signal"] == "NEUTRAL").sum()

    total = len(signals)
    if outperform_count > total / 2:
        consensus = "OUTPERFORM"
    elif underperform_count > total / 2:
        consensus = "UNDERPERFORM"
    else:
        consensus = "NEUTRAL"

    return consensus, mean_pred, mean_ic, mean_hr


def _sell_pct_from_consensus(
    consensus: str,
    mean_predicted: float,
    mean_ic: float,
) -> float:
    """Map consensus signal + IC to a sell-percentage recommendation (0.0–1.0)."""
    if mean_ic < 0.05:
        return 0.50  # weak signal → default diversification
    if consensus == "OUTPERFORM":
        if mean_predicted > 0.15:
            return 0.25  # high conviction → hold most
        if mean_predicted > 0.05:
            return 0.50
        return 0.75
    if consensus == "UNDERPERFORM":
        return 1.00   # model predicts underperformance → diversify fully
    return 0.50       # NEUTRAL


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_recommendation_md(
    out_dir: Path,
    as_of: date,
    run_date: date,
    signals: pd.DataFrame,
    consensus: str,
    mean_predicted: float,
    mean_ic: float,
    mean_hr: float,
    sell_pct: float,
    dry_run: bool,
) -> None:
    """Write the human-readable recommendation report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "recommendation.md"

    lines = [
        f"# PGR Monthly Decision Report — {as_of.strftime('%B %Y')}",
        "",
        f"**As-of Date:** {as_of}  ",
        f"**Run Date:** {run_date}  ",
        f"**Model Version:** v3.0 (ElasticNet WFO, 20 ETF benchmarks)  ",
        "",
        "---",
        "",
        "## Consensus Signal",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Signal | **{consensus}** |",
        f"| Recommended Sell % | **{sell_pct:.0%}** |",
        f"| Predicted 6M Relative Return | {mean_predicted:+.2%} |",
        f"| Mean IC (across benchmarks) | {mean_ic:.4f} |",
        f"| Mean Hit Rate | {mean_hr:.1%} |",
        "",
        "> **Note:** The sell % recommendation is used only at actual vesting events",
        "> (January and July).  Monthly reports are monitoring tools, not trade signals.",
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ]

    if consensus == "OUTPERFORM":
        lines += [
            f"The model has {'moderate' if mean_ic < 0.10 else 'high'} conviction that PGR "
            f"will outperform a diversified portfolio of ETF benchmarks over the next 6 months.",
            "",
            f"Recommended action at next vesting event: **HOLD {1 - sell_pct:.0%}** of vesting RSUs.",
        ]
    elif consensus == "UNDERPERFORM":
        lines += [
            "The model predicts PGR will underperform the benchmark portfolio over the next 6 months.",
            "",
            "Recommended action at next vesting event: **SELL {:.0%}** of vesting RSUs and diversify.".format(sell_pct),
        ]
    else:
        lines += [
            "Model signal is weak (mean IC below threshold or mixed directional signals).",
            "",
            "Recommended action at next vesting event: **DEFAULT 50% SALE** for risk management.",
        ]

    lines += [
        "",
        "---",
        "",
        "## Per-Benchmark Signals",
        "",
        "| Benchmark | Predicted Return | IC | Hit Rate | Signal |",
        "|-----------|----------------|----|----------|--------|",
    ]

    if not signals.empty:
        for ticker, row in signals.iterrows():
            pred = f"{row['predicted_relative_return']:+.2%}" if not pd.isna(row.get("predicted_relative_return")) else "n/a"
            ic_val = f"{row['ic']:.4f}" if not pd.isna(row.get("ic")) else "n/a"
            hr_val = f"{row['hit_rate']:.1%}" if not pd.isna(row.get("hit_rate")) else "n/a"
            lines.append(f"| {ticker} | {pred} | {ic_val} | {hr_val} | {row.get('signal', 'N/A')} |")

    lines += [
        "",
        "---",
        "",
        f"*Generated by `scripts/monthly_decision.py`*{'  [DRY RUN]' if dry_run else ''}",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {path}")


def _write_signals_csv(out_dir: Path, signals: pd.DataFrame) -> None:
    """Write per-benchmark signals to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "signals.csv"
    if signals.empty:
        pd.DataFrame(
            columns=["benchmark", "predicted_relative_return", "ic", "hit_rate", "signal", "top_feature"]
        ).to_csv(path, index=False)
    else:
        signals.reset_index().to_csv(path, index=False)
    print(f"  Wrote {path}")


def _append_decision_log(
    as_of: date,
    run_date: date,
    consensus: str,
    sell_pct: float,
    mean_predicted: float,
    mean_ic: float,
    mean_hr: float,
    dry_run: bool,
) -> None:
    """Append one row to the persistent decision_log.md."""
    log_path = Path("results") / "monthly_decisions" / "decision_log.md"
    if not log_path.exists():
        return

    content = log_path.read_text(encoding="utf-8")

    new_row = (
        f"| {as_of} | {run_date} | {consensus} | {sell_pct:.0%} "
        f"| {mean_predicted:+.2%} | {mean_ic:.4f} | {mean_hr:.1%} "
        f"| {'[DRY RUN]' if dry_run else ''} |"
    )

    # Find the placeholder row and replace it, or append after the table header
    placeholder = "| *(first entry will appear here after the first automated run)* |"
    if placeholder in content:
        content = content.replace(placeholder, new_row, 1)
    else:
        # Append after the last table row
        lines = content.splitlines()
        # Find the last | line in the log table
        last_table_idx = max(
            (i for i, line in enumerate(lines) if line.startswith("| ")),
            default=-1,
        )
        if last_table_idx >= 0:
            lines.insert(last_table_idx + 1, new_row)
            content = "\n".join(lines) + "\n"

    log_path.write_text(content, encoding="utf-8")
    print(f"  Appended to {log_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    as_of_date_str: str | None = None,
    dry_run: bool = False,
    skip_fred: bool = False,
) -> None:
    as_of = _resolve_as_of_date(as_of_date_str)
    run_date = date.today()

    print(f"\n{'[DRY RUN] ' if dry_run else ''}PGR Monthly Decision — as-of {as_of}")
    print(f"Run date: {run_date}")

    # Idempotency: skip if this month's report already exists
    if _already_ran(as_of) and not dry_run:
        print(f"  Report for {as_of.strftime('%Y-%m')} already exists. Skipping.")
        return

    conn = db_client.get_connection(config.DB_PATH)
    db_client.initialize_schema(conn)

    # Step 1: Refresh FRED data
    _fetch_fred_step(conn, dry_run=dry_run, skip_fred=skip_fred)

    # Step 2: Generate signals
    print(f"\nGenerating signals (as-of {as_of})...")
    signals = _generate_signals(conn, as_of, model_type="elasticnet", target_horizon_months=6)

    # Step 3: Compute consensus
    consensus, mean_pred, mean_ic, mean_hr = _consensus_signal(signals)
    sell_pct = _sell_pct_from_consensus(consensus, mean_pred, mean_ic)

    print(f"\n  Consensus signal: {consensus}")
    print(f"  Predicted 6M relative return: {mean_pred:+.2%}")
    print(f"  Mean IC: {mean_ic:.4f}  |  Mean hit rate: {mean_hr:.1%}")
    print(f"  Sell %: {sell_pct:.0%}")

    # Step 4: Write outputs
    out_dir = _output_dir(as_of)
    _write_recommendation_md(
        out_dir, as_of, run_date, signals,
        consensus, mean_pred, mean_ic, mean_hr, sell_pct, dry_run,
    )
    _write_signals_csv(out_dir, signals)
    _append_decision_log(
        as_of, run_date, consensus, sell_pct, mean_pred, mean_ic, mean_hr, dry_run,
    )

    conn.close()
    print(f"\nDone. Results written to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PGR v3.0 monthly decision report generator."
    )
    parser.add_argument(
        "--as-of",
        metavar="YYYY-MM-DD",
        help="Override the as-of date (default: today or next business day after 20th).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate outputs without making HTTP calls or committing to git.",
    )
    parser.add_argument(
        "--skip-fred",
        action="store_true",
        help="Skip the FRED data fetch step.",
    )
    args = parser.parse_args()
    main(
        as_of_date_str=args.as_of,
        dry_run=args.dry_run,
        skip_fred=args.skip_fred,
    )
