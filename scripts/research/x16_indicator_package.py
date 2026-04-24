"""Write x16 research-only indicator package artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.x16_indicator_package import (
    build_indicator_package,
    build_peer_review_prompt,
)

OUTPUT_DIR = Path("results") / "research"
PACKAGE_PATH = OUTPUT_DIR / "x16_indicator_package.json"
PROMPT_PATH = OUTPUT_DIR / "x16_peer_review_prompt.md"
MEMO_PATH = OUTPUT_DIR / "x16_research_memo.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing x16 input artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_memo(indicator_package: dict[str, Any]) -> None:
    """Write a short x16 memo."""
    lines = [
        "# x16 Research Memo",
        "",
        "## Scope",
        "",
        "x16 packages the current x-series structural indicator candidate",
        "for later monthly-report/dashboard discussion without changing any",
        "production or shadow behavior.",
        "",
        "## Indicator",
        "",
        f"- Name: `{indicator_package['indicator_name']}`.",
        f"- Horizon: `{indicator_package['horizon_months']}m`.",
        f"- P/B anchor policy: `{indicator_package['pb_anchor_policy']}`.",
        "",
        "## Interpretation",
        "",
        "- x16 is documentation and packaging only.",
        "- It preserves the x14 candidate and the x15 negative result.",
    ]
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    x14 = _load_json(OUTPUT_DIR / "x14_indicator_synthesis_summary.json")
    x15 = _load_json(OUTPUT_DIR / "x15_pb_regime_overlay_summary.json")
    package = build_indicator_package(x14_summary=x14, x15_summary=x15)
    prompt = build_peer_review_prompt(
        indicator_package=package,
        x14_summary=x14,
        x15_summary=x15,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_PATH.write_text(
        json.dumps(package, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    PROMPT_PATH.write_text(prompt + "\n", encoding="utf-8")
    write_memo(package)
    print(f"Wrote {PACKAGE_PATH}")
    print(f"Wrote {PROMPT_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
