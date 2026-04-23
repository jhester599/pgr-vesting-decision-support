"""Run x18 dividend policy regime audit and target construction."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.database import db_client
from src.research.x18_dividend_policy_regime import (
    POLICY_CHANGE_DATE,
    build_regime_aware_dividend_targets,
    classify_dividend_regime,
    summarize_x18_policy_targets,
)

OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "x18_dividend_policy_targets.csv"
DIVIDEND_AUDIT_PATH = OUTPUT_DIR / "x18_dividend_policy_audit.csv"
SUMMARY_PATH = OUTPUT_DIR / "x18_dividend_policy_summary.json"
MEMO_PATH = OUTPUT_DIR / "x18_research_memo.md"


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = db_client.get_connection(config.DB_PATH)
    try:
        dividends = db_client.get_dividends(conn, "PGR")
        pgr_monthly = db_client.get_pgr_edgar_monthly(conn)
    finally:
        conn.close()
    return dividends, pgr_monthly


def _monthly_snapshot_frame(pgr_monthly: pd.DataFrame) -> pd.DataFrame:
    frame = pgr_monthly.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    frame = frame.sort_index()
    if "book_value_per_share" not in frame.columns:
        for candidate in ("book_value_per_share", "bvps", "current_bvps"):
            if candidate in frame.columns:
                frame["book_value_per_share"] = pd.to_numeric(frame[candidate], errors="coerce")
                break
    return frame


def run_x18_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run the x18 policy audit and regime-aware target build."""
    dividends, pgr_monthly = _load_inputs()
    classified = classify_dividend_regime(dividends)
    targets = build_regime_aware_dividend_targets(
        _monthly_snapshot_frame(pgr_monthly),
        dividends,
    )
    summary = summarize_x18_policy_targets(targets)
    summary["policy_change_date"] = str(POLICY_CHANGE_DATE.date())
    summary["post_policy_special_count"] = int(
        targets["special_dividend_occurred_regime"].fillna(0).sum()
    )
    return classified, targets, summary


def write_artifacts(
    classified: pd.DataFrame,
    targets: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    """Write x18 artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    classified.to_csv(DIVIDEND_AUDIT_PATH)
    targets.to_csv(DETAIL_PATH)
    payload = {
        "version": "x18",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "summary": summary,
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_memo(summary, targets)


def write_memo(summary: dict[str, object], targets: pd.DataFrame) -> None:
    """Write a compact x18 memo."""
    lines = [
        "# x18 Research Memo",
        "",
        "## Scope",
        "",
        "x18 audits the dividend-policy regime break and rebuilds the annual",
        "target window around the actual year-end payment behavior rather than",
        "a generic Q1-only frame.",
        "",
        "## Findings",
        "",
        f"- Policy change anchor: `{summary['policy_change_date']}`.",
        f"- Total November snapshots: {summary['snapshot_count']}.",
        f"- Post-policy eligible snapshots: {summary['post_policy_snapshot_count']}.",
        f"- Post-policy special occurrences in rebuilt labels: {summary['post_policy_special_count']}.",
        "",
        "## Interpretation",
        "",
        "- Pre-policy observations should be treated as annual quantitative",
        "  capital-return history, not pooled naively into the post-2018",
        "  regular-plus-special label.",
        "- The next dividend model should use the post-policy label plus",
        "  persistent-BVPS and capital-generation features.",
    ]
    if not targets.empty:
        lines.extend(
            [
                "",
                "## Window Convention",
                "",
                "- Target window: December through February following the",
                "  November snapshot.",
            ]
        )
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    classified, targets, summary = run_x18_audit()
    write_artifacts(classified, targets, summary)
    print(f"Wrote {DIVIDEND_AUDIT_PATH}")
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
