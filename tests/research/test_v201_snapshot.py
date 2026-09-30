"""Independent registry snapshot and unchanged-input contracts."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from pgr_vds.research_lib import provenance


@pytest.fixture
def setup_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, Path, Path, dict]:
    module = importlib.import_module("pgr_vds.research_lib.snapshot")
    root = Path(__file__).resolve().parents[2]
    lock = json.loads(
        (
            root
            / (
                "research/studies/v200_clean_baseline/outputs/baseline_lock.json"
            )
        ).read_text(encoding="utf-8")
    )
    original = b"studies:\n  - id: v200\n    status: closed\n"
    registry = tmp_path / "research/registry.yaml"
    registry.parent.mkdir()
    registry.write_bytes(original + b"  - id: v201\n    status: closed\n")
    (tmp_path / "input.bin").write_bytes(b"abc")
    lock["input_files"] = [
        {
            "path": "research/registry.yaml",
            "sha256": hashlib.sha256(original).hexdigest(),
            "hash_basis": "git_source_lf",
        },
        {
            "path": "input.bin",
            "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9c"
            "b410ff61f20015ad",
        },
    ]
    commit = "1" * 40
    lock["research_execution_code_commit"] = commit
    location = tmp_path / "lock.json"
    location.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    def git_blob(args: list[str], **kwargs: object) -> bytes:
        assert args == ["git", "show", f"{commit}:research/registry.yaml"]
        return original

    monkeypatch.setattr(module, "git_blob", git_blob)
    return module, location, registry, lock


def test_registry_growth_preserves_exact_lock_and_all_pins(
    setup_snapshot: tuple,
) -> None:
    module, location, registry, _ = setup_snapshot
    before = location.read_bytes()
    checked = module.verify_registry_growth(location, location.parent / "tmp")
    assert location.read_bytes() == before
    assert checked["registry_check"]["added_entries"] == ["v201"]
    assert checked["registry_check"]["pinned_entries_unchanged"] == 1
    assert checked["registry_check"]["pinned_at_commit"] == "1" * 40
    assert checked["registry_check"]["current_sha256"] == (
        hashlib.sha256(registry.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "content",
    [
        "studies:\n  - id: v200\n    status: promoted\n",
        "studies:\n  - id: v201\n    status: closed\n",
        "studies:\n  - id: v200\n    status: closed\n"
        "  - id: v200\n    status: promoted\n",
    ],
)
def test_rewritten_deleted_and_duplicate_entries_fail(
    setup_snapshot: tuple,
    content: str,
) -> None:
    module, location, registry, _ = setup_snapshot
    registry.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        module.verify_registry_growth(location, location.parent / "tmp")


def test_historical_registry_wrong_hash_fails(setup_snapshot: tuple) -> None:
    module, location, _, lock = setup_snapshot
    lock["input_files"][0]["sha256"] = "0" * 64
    location.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="historical registry SHA256"):
        module.verify_registry_growth(location, location.parent / "tmp")


def test_other_consumed_file_mutation_still_fails(
    setup_snapshot: tuple,
) -> None:
    module, location, _, _ = setup_snapshot
    (location.parent / "input.bin").write_bytes(b"abcd")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module.verify_registry_growth(location, location.parent / "tmp")


def test_abbreviated_execution_commit_is_rejected(
    setup_snapshot: tuple,
) -> None:
    module, location, _, lock = setup_snapshot
    lock["research_execution_code_commit"] = "master"
    location.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="full execution commit"):
        module.verify_registry_growth(location, location.parent / "tmp")
