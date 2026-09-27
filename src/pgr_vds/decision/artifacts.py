"""Output folders, CSV artifacts, the decision log and the shadow ledgers.

Production runs write ``artifacts/monthly_decisions/YYYY-MM/`` and append to
``decision_log.md`` and the classifier/TA shadow ledgers; dry runs write the
gitignored ``results/dry_run/monthly_decisions/YYYY-MM/`` and append nothing.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

import config
from src.models.classification_monitoring import (
    attach_matured_classifier_outcomes,
    attach_matured_ta_outcomes,
)
from src.reporting.classification_artifacts import (
    append_classifier_history,
    append_ta_shadow_variant_history,
    build_classifier_history_entry,
    build_ta_shadow_variant_history_entries,
    classification_history_path,
    ta_shadow_variant_history_path,
)
from src.reporting.confidence import benchmark_role_for_ticker
from src.reporting.run_manifest import build_run_manifest, write_run_manifest

logger = logging.getLogger(__name__)

# src/pgr_vds/decision/artifacts.py -> repository root (git SHA lookup).
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

# The entry point recorded as ``script_name`` in ``run_manifest.json``.
MANIFEST_SCRIPT_NAME: str = "cli/monthly_decision.py"


def output_dir(as_of: date) -> Path:
    """Return the output directory for the given month."""
    month_str = as_of.strftime("%Y-%m")
    return Path(config.MONTHLY_DECISIONS_DIR) / month_str


def dry_run_output_dir(as_of: date) -> Path:
    """Return the gitignored output directory used by ``--dry-run``.

    Dry runs must never overwrite the committed production artifacts under
    ``artifacts/monthly_decisions/``.
    """
    month_str = as_of.strftime("%Y-%m")
    return Path(config.DRY_RUN_MONTHLY_DECISIONS_DIR) / month_str


def already_ran(as_of: date) -> bool:
    """Return True if a report already exists for this exact as-of date.

    Checks the run_manifest.json as_of_date field so that re-running with a
    different date in the same month (e.g. after a code update) overwrites the
    stale report rather than skipping.
    """
    out_dir = output_dir(as_of)
    manifest_path = out_dir / "run_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        import json as _json
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("as_of_date") == as_of.isoformat()
    except Exception:
        # Unreadable manifest — fall back to presence check
        return (out_dir / "recommendation.md").exists()


def write_step_output(name: str, value: str) -> None:
    """Append ``name=value`` to ``$GITHUB_OUTPUT`` when running in Actions."""
    output_path = os.getenv("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def load_previous_decision_summary(as_of: date) -> dict | None:
    """Load the most recent prior row from decision_log.md, if available."""
    path = Path(config.DECISION_LOG_PATH)
    if not path.exists():
        return None

    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        if "As-Of Date" in line or "---" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 8:
            continue
        rows.append(
            {
                "as_of": parts[0],
                "run_date": parts[1],
                "consensus": parts[2],
                "sell_pct": parts[3],
                "predicted": parts[4],
                "mean_ic": parts[5],
                "mean_hr": parts[6],
                "notes": parts[7],
            }
        )

    prior_rows: list[dict[str, str]] = []
    for row in rows:
        try:
            if date.fromisoformat(row["as_of"]) < as_of:
                prior_rows.append(row)
        except ValueError:
            continue
    if not prior_rows:
        return None
    prior_rows.sort(key=lambda row: row["as_of"])
    return prior_rows[-1]


def write_signals_csv(out_dir: Path, signals: pd.DataFrame) -> None:
    """Write per-benchmark signals to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "signals.csv"
    if signals.empty:
        pd.DataFrame(columns=[
            "benchmark", "predicted_relative_return", "ic", "hit_rate",
            "signal", "prob_outperform", "confidence_tier", "benchmark_role",
        ]).to_csv(path, index=False)
    else:
        export = signals.reset_index()
        export["benchmark_role"] = export["benchmark"].map(
            lambda ticker: benchmark_role_for_ticker(str(ticker))["role"]
        )
        export.to_csv(path, index=False)
    print(f"  Wrote {path}")


def write_benchmark_quality_csv(
    out_dir: Path,
    benchmark_quality_df: pd.DataFrame | None,
) -> None:
    """Write ensemble OOS benchmark-quality diagnostics to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "benchmark_quality.csv"
    columns = [
        "benchmark",
        "n_obs",
        "oos_r2",
        "nw_ic",
        "nw_p_value",
        "hit_rate",
        "cw_t_stat",
        "cw_p_value",
        "cw_mean_adjusted_differential",
        "r2_flag",
        "ic_flag",
        "hr_flag",
    ]
    if benchmark_quality_df is None or benchmark_quality_df.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    else:
        benchmark_quality_df.loc[:, columns].to_csv(path, index=False)
    print(f"  Wrote {path}")


def write_consensus_shadow_csv(
    out_dir: Path,
    consensus_shadow_df: pd.DataFrame | None,
) -> None:
    """Write the v74 shadow consensus comparison table to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "consensus_shadow.csv"
    columns = [
        "variant",
        "n_benchmarks",
        "consensus",
        "mean_predicted_return",
        "mean_ic",
        "mean_hit_rate",
        "mean_prob_outperform",
        "confidence_tier",
        "weight_mode",
        "score_col",
        "lambda_mix",
        "top_benchmark",
        "top_benchmark_weight",
        "recommendation_mode",
        "recommended_sell_pct",
        "is_live_path",
    ]
    if consensus_shadow_df is None or consensus_shadow_df.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    else:
        consensus_shadow_df.loc[:, columns].to_csv(path, index=False)
    print(f"  Wrote {path}")


def append_decision_log(
    as_of: date,
    run_date: date,
    consensus: str,
    sell_pct: float,
    mean_predicted: float,
    mean_ic: float,
    mean_hr: float,
    dry_run: bool,
    _log_path_override: Path | None = None,
) -> None:
    """Append one row to the persistent decision_log.md.

    Inserts after the last data row of the ## Log table, identified by the
    separator line immediately below the log table's header row.  This prevents
    orphaned rows appearing in the Column Definitions or other sections.

    v7.3 fix: replaced the previous "find last | line in entire file" approach
    which incorrectly anchored to rows in the Column Definitions table.

    Args:
        _log_path_override: If provided, write to this path instead of the
                            default artifacts/monthly_decisions/decision_log.md.
                            Used in tests only.
    """
    log_path = _log_path_override or Path(config.DECISION_LOG_PATH)
    if not log_path.exists():
        return

    content = log_path.read_text(encoding="utf-8")

    new_row = (
        f"| {as_of} | {run_date} | {consensus} | {sell_pct:.0%} "
        f"| {mean_predicted:+.2%} | {mean_ic:.4f} | {mean_hr:.1%} "
        f"| {'[DRY RUN]' if dry_run else ''} |"
    )
    row_prefix = (
        f"| {as_of} | {run_date} | {consensus} | {sell_pct:.0%} "
        f"| {mean_predicted:+.2%} | {mean_ic:.4f} | {mean_hr:.1%} |"
    )
    if new_row in content or any(line.startswith(row_prefix) for line in content.splitlines()):
        print(f"  Decision log already contains this row; skipping append.")
        return

    # Replace the placeholder on first use.
    placeholder = "| *(first entry will appear here after the first automated run)* |"
    if placeholder in content:
        content = content.replace(placeholder, new_row, 1)
        log_path.write_text(content, encoding="utf-8")
        print(f"  Appended to {log_path}")
        return

    # Locate the log table by its fixed separator line.  The log table header
    # is the only table whose separator contains "Consensus Signal".  Find the
    # last data row within that table (before the next "---" section divider or
    # end of file) and insert after it.
    LOG_SEPARATOR = "|------------|----------|-----------------|"
    lines = content.splitlines()

    sep_idx = next(
        (i for i, line in enumerate(lines) if line.startswith(LOG_SEPARATOR)),
        -1,
    )
    if sep_idx < 0:
        # Fallback: can't find the log table — append at end of file.
        content = content.rstrip("\n") + "\n" + new_row + "\n"
        log_path.write_text(content, encoding="utf-8")
        print(f"  Appended to {log_path} (fallback: no log table separator found)")
        return

    # Find the last log-table row: scan forward from sep_idx while lines start
    # with "| " and stop at the first "---" divider or non-table line.
    last_data_idx = sep_idx
    for i in range(sep_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("---") or (stripped and not stripped.startswith("|")):
            break
        if stripped.startswith("|"):
            last_data_idx = i

    lines.insert(last_data_idx + 1, new_row)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Appended to {log_path}")


def update_shadow_ledgers(
    conn: sqlite3.Connection,
    *,
    as_of: date,
    run_date: date,
    history_base_dir: Path,
    dry_run: bool,
    classification_shadow_summary: dict | None,
    classification_shadow_variants: list[dict[str, object]],
    live_recommendation_mode: str,
    live_sell_pct: float,
    shadow_gate_overlay: dict | None,
) -> pd.DataFrame:
    """Update the classifier and TA shadow ledgers; return the classifier history.

    The ledgers live next to the production month folders
    (``history_base_dir``). Dry runs read them for monitoring but never
    append to or rewrite them. Matured outcomes are attached to the returned
    history either way.
    """
    history_path = classification_history_path(history_base_dir)
    if history_path.exists():
        history_df = pd.read_csv(history_path)
    else:
        history_df = pd.DataFrame()
    history_df = attach_matured_classifier_outcomes(
        conn, history_df, horizon_months=6, as_of=as_of
    )
    if not history_df.empty and not dry_run:
        history_df.to_csv(history_path, index=False)

    if dry_run:
        logger.info("[DRY RUN] Not appending to the classifier or TA shadow ledgers.")
    elif isinstance(classification_shadow_summary, dict) and classification_shadow_summary.get("enabled"):
        history_entry = build_classifier_history_entry(
            as_of_date=as_of,
            run_date=run_date,
            feature_anchor_date=(
                str(classification_shadow_summary.get("feature_anchor_date"))
                if classification_shadow_summary.get("feature_anchor_date") is not None
                else None
            ),
            forecast_horizon_months=6,
            classification_shadow_summary=classification_shadow_summary,
            live_recommendation_mode=live_recommendation_mode,
            live_sell_pct=live_sell_pct,
            shadow_gate_overlay=shadow_gate_overlay,
        )
        history_path = append_classifier_history(
            base_dir=history_base_dir,
            entry=history_entry,
        )
        history_df = pd.read_csv(history_path)
        history_df = attach_matured_classifier_outcomes(
            conn, history_df, horizon_months=6, as_of=as_of
        )
        history_df.to_csv(history_path, index=False)

    ta_history_entries = build_ta_shadow_variant_history_entries(
        as_of_date=as_of,
        run_date=run_date,
        forecast_horizon_months=6,
        classification_shadow_variants=classification_shadow_variants,
    )
    if ta_history_entries and not dry_run:
        ta_history_path = append_ta_shadow_variant_history(
            base_dir=history_base_dir,
            entries=ta_history_entries,
        )
        print(f"  Appended TA shadow history to {ta_history_path}")
    ta_history_path = ta_shadow_variant_history_path(history_base_dir)
    if ta_history_path.exists():
        ta_history_df = attach_matured_ta_outcomes(
            conn, pd.read_csv(ta_history_path), as_of=as_of,
        )
        if not dry_run:
            ta_history_df.to_csv(ta_history_path, index=False)
    return history_df


# The files a run writes into its month folder (``run_manifest.json`` lists them).
MONTHLY_OUTPUT_FILES: tuple[str, ...] = (
    "recommendation.md",
    "diagnostic.md",
    "signals.csv",
    "benchmark_quality.csv",
    "consensus_shadow.csv",
    "classification_shadow.csv",
    "decision_overlays.csv",
    "dashboard.html",
    "monthly_summary.json",
)


def write_monthly_run_manifest(
    out_dir: Path,
    *,
    snapshot: dict[str, Any],
    as_of: date,
    manifest_warnings: list[str],
    dry_run: bool,
    nan_live_features: list[str],
) -> Path:
    """Write ``run_manifest.json`` for this run and return its path.

    ``snapshot`` is ``db_client.get_operational_snapshot`` taken before the
    run records its retrain-trigger evaluation.
    """
    manifest = build_run_manifest(
        workflow_name="monthly_decision",
        script_name=MANIFEST_SCRIPT_NAME,
        as_of_date=as_of,
        schema_version=snapshot["schema_version"],
        latest_dates=snapshot["latest_dates"],
        row_counts=snapshot["row_counts"],
        warnings=manifest_warnings,
        outputs=[str(out_dir / name) for name in MONTHLY_OUTPUT_FILES],
        artifact_classification="dry_run" if dry_run else "production",
        cwd=str(REPO_ROOT),
    )
    manifest["dry_run"] = bool(dry_run)
    manifest["nan_live_features"] = nan_live_features
    return write_run_manifest(out_dir, manifest)
