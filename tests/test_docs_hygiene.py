from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_docs_root_has_navigation_map() -> None:
    text = _read("docs/README.md")

    assert "## Active Operator Docs" in text
    assert "## Historical And Research Docs" in text
    assert "docs/superpowers/plans/" in text
    assert "docs/plans/" in text
    assert "legacy" in text.lower()


def test_legacy_plan_and_result_dirs_are_labeled() -> None:
    plan_text = _read("docs/plans/README.md")
    result_text = _read("docs/results/README.md")

    assert "Legacy" in plan_text
    assert "docs/superpowers/plans/" in plan_text
    assert "Legacy" in result_text
    assert "results/research/" in result_text


def test_artifact_policy_mentions_current_shadow_ledgers() -> None:
    text = _read("docs/artifact-policy.md")

    assert "classification_shadow_history.csv" in text
    assert "ta_shadow_variant_history.csv" in text
    assert "scripts/verify_monthly_outputs.py" in text


def test_root_archive_is_labeled_as_code_archive() -> None:
    text = _read("archive/README.md")

    assert "retired code and tests" in text
    assert "docs/archive/" in text


def test_gitignore_has_single_hypothesis_entry() -> None:
    lines = _read(".gitignore").splitlines()

    assert lines.count(".hypothesis/") == 1
