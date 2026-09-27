"""Current-use governance contracts; archived claims are not adoption proof."""

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _registry() -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(
        (REPO_ROOT / "research/registry.yaml").read_text(encoding="utf-8")
    )
    return {row["id"]: row for row in data["studies"]}


def test_current_use_docs_label_firth_research_only() -> None:
    backlog = (REPO_ROOT / "docs/research/backlog.md").read_text(
        encoding="utf-8"
    )
    assert "research-only pending v204 evidence" in backlog
    assert "Historical status claimed" in backlog
    assert "2026-09-27 R4 evidence note (V07)" in backlog
    registry = _registry()
    assert registry["v154"]["status"] == "closed"
    assert registry["v154"]["promoted_to"] is None
    assert "research-only pending v204" in registry["v154"]["question"].lower()
    readme = (REPO_ROOT / "research/README.md").read_text(encoding="utf-8")
    assert "research-only pending v204" in readme.lower()


def test_current_use_docs_distinguish_baseline_metadata_from_fit() -> None:
    registry = _registry()
    for study in ("v141", "v143", "v144", "v149", "v150"):
        assert (
            "baseline-derived candidate metadata only"
            in registry[study]["question"]
        )
        assert "2026-09-27" in registry[study]["notes"]
