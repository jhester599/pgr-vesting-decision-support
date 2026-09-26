from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_docs_root_has_navigation_map() -> None:
    text = _read("docs/README.md")

    assert "## Active Operator Docs" in text
    assert "## Decisions, Reviews And History" in text
    assert "decisions/" in text
    assert "history/" in text
    assert "legacy" in text.lower()


def test_legacy_plan_and_result_dirs_are_labeled() -> None:
    plan_text = _read("docs/history/plans/README.md")
    result_text = _read("docs/history/results/README.md")

    assert "Legacy" in plan_text
    assert "../superpowers/plans/" in plan_text
    assert "Legacy" in result_text
    assert "research/studies/" in result_text
    assert "research/legacy/" in result_text


def test_artifact_policy_mentions_current_shadow_ledgers() -> None:
    text = _read("docs/artifact-policy.md")

    assert "classification_shadow_history.csv" in text
    assert "ta_shadow_variant_history.csv" in text
    assert "scripts/verify_monthly_outputs.py" in text


def test_retired_code_index_replaces_root_archive() -> None:
    """Root ``archive/`` was retired in phase 4; its index says how to get a file back."""
    assert not Path("archive").exists()
    text = _read("docs/history/retired-code/README.md")

    assert "git show 282a6b3:archive/scripts/" in text
    assert "research/legacy/" in text


def test_gitignore_has_single_hypothesis_entry() -> None:
    lines = _read(".gitignore").splitlines()

    assert lines.count(".hypothesis/") == 1
