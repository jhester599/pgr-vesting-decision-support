"""Write x11 synthesis artifacts for x9/x10 capital research."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.x11_capital_synthesis import build_x11_summary

OUTPUT_DIR = Path("results") / "research"
SUMMARY_PATH = OUTPUT_DIR / "x11_capital_synthesis_summary.json"
MEMO_PATH = OUTPUT_DIR / "x11_capital_synthesis_memo.md"
INPUT_SUMMARIES = {
    "x4": OUTPUT_DIR / "x4_bvps_forecasting_summary.json",
    "x6": OUTPUT_DIR / "x6_special_dividend_summary.json",
    "x8": OUTPUT_DIR / "x8_synthesis_summary.json",
    "x9": OUTPUT_DIR / "x9_bvps_bridge_summary.json",
    "x10": OUTPUT_DIR / "x10_dividend_capital_summary.json",
}


def load_payloads() -> dict[str, dict[str, object]]:
    """Load required x11 input payloads."""
    payloads: dict[str, dict[str, object]] = {}
    for version, path in INPUT_SUMMARIES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required x11 input: {path}")
        payloads[version] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def write_memo(summary: dict[str, object]) -> None:
    """Write human-readable x11 memo."""
    recommendation = summary["recommendation"]
    dividend = summary["dividend_comparison"]
    lines = [
        "# x11 Capital Synthesis Memo",
        "",
        "## Scope",
        "",
        "x11 synthesizes x9 BVPS bridge evidence and x10 special-dividend",
        "capital-feature evidence against earlier x4/x6 baselines. It does",
        "not train models or alter production/monthly/shadow artifacts.",
        "",
        "## Recommendation",
        "",
        f"- Status: `{recommendation['status']}`.",
        f"- Rationale: {recommendation['rationale']}.",
        "",
        "## BVPS Comparison",
        "",
    ]
    for row in summary["bvps_comparison"]:
        direction = "beat" if row["x9_beats_x4"] else "did not beat"
        lines.append(
            f"- {row['horizon_months']}m: x9 `{row['x9_model_name']}` "
            f"({row['x9_feature_block']}) {direction} x4 "
            f"(delta MAE {row['future_bvps_mae_delta']:.3f})."
        )
    lines.extend(
        [
            "",
            "## Dividend Comparison",
            "",
            (
                f"- x10 `{dividend['x10_feature_set']}` / "
                f"`{dividend['x10_model_name']}` vs x6 "
                f"`{dividend['x6_model_name']}`: EV MAE delta "
                f"{dividend['expected_value_mae_delta']:.3f}; "
                f"confidence `{dividend['confidence']}` with "
                f"{dividend['n_obs']} OOS annual observations."
            ),
            "",
            "## Decision Questions",
            "",
        ]
    )
    for item in summary["decision_questions"]:
        lines.append(
            f"- {item['question']} Criterion: {item['criterion']} "
            f"Answer: {item['answer']}"
        )
    lines.extend(
        [
            "",
            "## Next Research Step",
            "",
            "Run a narrower x12 BVPS target audit focused on capital-return",
            "adjusted BVPS, December special-dividend discontinuities, and",
            "whether the 12m x4 tree result is robust or target-regime driven.",
        ]
    )
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = build_x11_summary(load_payloads())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_memo(summary)
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
