"""Write x8 synthesis artifacts from checked-in x-series research outputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.x8_synthesis import build_x8_summary

OUTPUT_DIR = Path("results") / "research"
SUMMARY_PATH = OUTPUT_DIR / "x8_synthesis_summary.json"
MEMO_PATH = OUTPUT_DIR / "x8_synthesis_memo.md"

INPUT_SUMMARIES = {
    "x2": OUTPUT_DIR / "x2_absolute_classification_summary.json",
    "x3": OUTPUT_DIR / "x3_direct_return_summary.json",
    "x4": OUTPUT_DIR / "x4_bvps_forecasting_summary.json",
    "x5": OUTPUT_DIR / "x5_decomposition_summary.json",
    "x6": OUTPUT_DIR / "x6_special_dividend_summary.json",
    "x7": OUTPUT_DIR / "x7_targeted_ta_summary.json",
}


def _format_gate_count(label: str, count: dict[str, object]) -> str:
    horizons = int(count["true_horizon_count"])
    horizon_word = "horizon" if horizons == 1 else "horizons"
    return (
        f"- {label}: {count['true_count']} "
        f"({horizons} {horizon_word})."
    )


def load_payloads() -> dict[str, dict[str, object]]:
    """Load required x-series summary payloads."""
    payloads: dict[str, dict[str, object]] = {}
    for version, path in INPUT_SUMMARIES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required x8 input: {path}")
        payloads[version] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def write_memo(summary: dict[str, object]) -> None:
    """Write the human-readable x8 synthesis memo."""
    readiness = summary["shadow_readiness"]
    ta_leader = summary["ta_leader"]
    dividend = summary["special_dividend_leader"]
    gate_counts = summary["gate_counts"]

    lines = [
        "# x8 Synthesis Memo",
        "",
        "## Scope",
        "",
        "x8 synthesizes checked-in x2-x7 research artifacts. It does not",
        "train models, refresh data, or alter production/monthly/shadow paths.",
        "",
        "## Shadow Readiness",
        "",
        f"- Status: `{readiness['status']}`.",
        f"- Rationale: {readiness['rationale']}.",
        "",
        "## Cross-Lane Findings",
        "",
        "- Absolute classification remains research-only. x2 did not clear",
        "  the base-rate gate, while x7 targeted TA replacements improved",
        "  selected 3m/6m classification evidence.",
        "- Direct forward-return modeling remains a benchmark lane. The",
        "  checked-in x3 summary shows only limited no-change gate clearance.",
        "- BVPS forecasting is the strongest structural leg. The x5",
        "  recombination still relies on no-change P/B as the stable anchor.",
        "- Special-dividend forecasting should remain an annual sidecar",
        "  because the November snapshot sample is very small.",
        "",
        "## Gate Counts",
        "",
        _format_gate_count("x2 base-rate gate true rows", gate_counts["x2"]),
        _format_gate_count("x3 no-change gate true rows", gate_counts["x3"]),
        _format_gate_count("x4 BVPS no-change gate true rows", gate_counts["x4"]),
        (
            f"- x7 best cleared horizons: "
            f"{gate_counts['x7_best_cleared_horizon_count']}."
        ),
        "",
        "## Leading Sidecars",
        "",
        (
            f"- TA leader: `{ta_leader['variant']}` cleared "
            f"{ta_leader['cleared_horizon_count']}/"
            f"{ta_leader['tested_horizon_count']} horizons."
        ),
        (
            f"- Special-dividend leader: `{dividend['model_name']}` with "
            f"{dividend['n_obs']} annual observations and expected-value MAE "
            f"{dividend['expected_value_mae']:.3f}."
        ),
        "",
        "## Recommendation",
        "",
        "Do not wire x-series into shadow artifacts yet. The next research",
        "step should be a narrow x9 robustness pass: confirm x7 3m/6m TA",
        "evidence, pressure-test BVPS/PB recombination, and keep the",
        "special-dividend model framed as low-confidence annual research.",
    ]
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payloads = load_payloads()
    summary = build_x8_summary(payloads)
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
