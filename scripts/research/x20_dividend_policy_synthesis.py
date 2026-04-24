"""Run x20 synthesis for the dividend-policy rebuild lane."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.x20_dividend_policy_synthesis import build_x20_summary

OUTPUT_DIR = Path("results") / "research"
SUMMARY_PATH = OUTPUT_DIR / "x20_dividend_policy_synthesis.json"
MEMO_PATH = OUTPUT_DIR / "x20_research_memo.md"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    return pd.read_csv(path).to_dict(orient="records")


def write_artifacts(summary: dict[str, Any]) -> None:
    """Write x20 JSON and memo artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_memo(summary)


def write_memo(summary: dict[str, Any]) -> None:
    """Write a compact x20 memo."""
    overlap = summary.get("overlap_comparison", {})
    sample_scope = summary.get("sample_scope", {})
    recommendation = summary.get("recommendation", {})
    lines = [
        "# x20 Research Memo",
        "",
        "## Scope",
        "",
        "x20 compares the x19 post-policy rebuild against x10 using the",
        "overlapping post-policy test years, then records what that means for",
        "the next dividend-lane step.",
        "",
        "## Findings",
        "",
        (
            f"- Overlap OOS observations: {sample_scope.get('x19_oos_n_obs', 0)} "
            f"within {sample_scope.get('post_policy_snapshot_count', 0)} "
            "post-policy November snapshots."
        ),
        (
            f"- Best x10 overlap EV MAE: {overlap.get('x10_overlap_ev_mae', float('nan')):.3f}."
            if overlap
            else "- Best x10 overlap EV MAE: n/a."
        ),
        (
            f"- Best x19 overlap EV MAE: {overlap.get('x19_overlap_ev_mae', float('nan')):.3f}."
            if overlap
            else "- Best x19 overlap EV MAE: n/a."
        ),
        (
            f"- Recommendation: `{recommendation.get('status', 'continue_research')}`."
        ),
        "",
        "## Interpretation",
        "",
        f"- {recommendation.get('rationale', 'Continue research.')}",
        "- If occurrence is still one-class on the current-policy overlap,",
        "  the next useful experiment is target-scale work on dividend size.",
    ]
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    x18 = _read_json(OUTPUT_DIR / "x18_dividend_policy_summary.json")
    x10 = _read_json(OUTPUT_DIR / "x10_dividend_capital_summary.json")
    x19 = _read_json(OUTPUT_DIR / "x19_post_policy_dividend_summary.json")
    x10["detail_rows"] = _read_csv_rows(OUTPUT_DIR / "x10_dividend_capital_detail.csv")
    x19["detail_rows"] = _read_csv_rows(OUTPUT_DIR / "x19_post_policy_dividend_detail.csv")
    payloads = {
        "x18": x18.get("summary", {}),
        "x10": x10,
        "x19": x19,
    }
    summary = build_x20_summary(payloads)
    write_artifacts(summary)
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
