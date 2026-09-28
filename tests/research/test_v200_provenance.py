"""Independent expected-output fixtures for the v200 provenance contract."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any

import pytest

from pgr_vds.research_lib import provenance


ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
BASELINE_COMMIT = "c3b4798b90a43f9dbd616981861e9ff01fa2de72"
REFRESH_COMMIT = "ed7997f6f540f664e59dd44a4079616d74a8e8cb"
DB_SHA256 = "38991c7653f6c8dc6eb5f3740f3a499a7121f093019aeacbfeb1bc85001a94e6"
SEED_COMMIT = "aae0be883309bdc068ee9afbb78b0aa9a4b52b8d"
SEED_SHA256 = (
    "7c68efbd90ef5c10a12e05dc2a9e03402e353a06219fd2db129d711754c1c35d"
)


def clean_lock(input_path: str) -> dict[str, Any]:
    """Construct an independently specified lock schema."""
    return {
        "schema_version": 1,
        "status": "clean",
        "baseline_code_commit": BASELINE_COMMIT,
        "input_db": {
            "git_commit": REFRESH_COMMIT,
            "relative_path": "data/pgr_financials.db",
            "sha256": DB_SHA256,
        },
        "parent_seed": {
            "git_commit": SEED_COMMIT,
            "db_sha256": SEED_SHA256,
        },
        "repairs": [
            {"id": "R2-lite", "commit": REFRESH_COMMIT},
            {"id": "R3b", "commit": BASELINE_COMMIT},
        ],
        "accepted_exceptions": [
            {
                "id": "VWO_MARCH_2026_GAP",
                "description": "Accepted March gap; twelve targets.",
            },
        ],
        "input_files": [{"path": input_path, "sha256": ABC_SHA256}],
    }


def write_lock(path: Path, lock: dict[str, Any]) -> Path:
    """Save fixture JSON outside the repository data tree."""
    path.write_text(json.dumps(lock), encoding="utf-8")
    return path


def test_sha256_has_independent_known_answer(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"abc")
    assert provenance.sha256_file(path) == ABC_SHA256


def test_source_digest_accepts_only_git_line_ending_equivalence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture.py"
    path.write_bytes(b"abc\r\n")
    expected = (
        "edeaaff3f1774ad2888673770c6d64097e391bc362d7d6fb34982ddf0efd18cb"
    )
    assert provenance.source_sha256(path) == expected
    path.write_bytes(b"abc\n")
    assert provenance.source_sha256(path) == expected
    path.write_bytes(b"abcd\n")
    assert provenance.source_sha256(path) != expected


def test_exact_artifact_digest_never_normalizes_csv_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture.csv"
    path.write_bytes(b"abc\r\n")
    with pytest.raises(ValueError, match="source text"):
        provenance.source_sha256(path)
    first = provenance.sha256_file(path)
    path.write_bytes(b"abc\n")
    assert provenance.sha256_file(path) != first


def test_lock_checks_declared_source_basis_and_rejects_content_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    source = tmp_path / "fixture.py"
    source.write_bytes(b"abc\r\n")
    lock = clean_lock("fixture.py")
    lock["input_files"][0].update(
        sha256=(
            "edeaaff3f1774ad2888673770c6d64097e391bc362d7d6fb34982ddf0efd18cb"
        ),
        hash_basis="git_source_lf",
    )
    location = write_lock(tmp_path / "lock.json", lock)
    provenance.verify_baseline_lock(location)
    source.write_bytes(b"abc\n")
    provenance.verify_baseline_lock(location)
    source.write_bytes(b"abcd\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        provenance.verify_baseline_lock(location)


def test_immutable_copy_rejects_wrong_hash_and_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture.db"
    with sqlite3.connect(path) as writer:
        writer.execute("CREATE TABLE known (value INTEGER)")
        writer.execute("INSERT INTO known VALUES (17)")
    before = provenance.sha256_file(path)
    with pytest.raises(ValueError, match="SHA256"):
        provenance.read_immutable(path, ABC_SHA256)
    connection = provenance.read_immutable(path, before)
    try:
        assert connection.execute("SELECT value FROM known").fetchone() == (
            17,
        )
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("INSERT INTO known VALUES (99)")
    finally:
        connection.close()
    assert provenance.sha256_file(path) == before
    assert not Path(str(path) + "-wal").exists()
    assert not Path(str(path) + "-shm").exists()


def test_git_export_rejects_short_ref_before_execution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full"):
        provenance.export_git_blob(
            "ed7997f",
            "data/pgr_financials.db",
            tmp_path / "copy.db",
            ABC_SHA256,
        )
    assert not (tmp_path / "copy.db").exists()


def test_git_export_checks_bytes_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_git(
        args: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=b"abc", stderr=b"")

    monkeypatch.setattr(provenance.subprocess, "run", fake_git)
    destination = tmp_path / "copy.db"
    result = provenance.export_git_blob(
        REFRESH_COMMIT, "data/pgr_financials.db", destination, ABC_SHA256
    )
    assert result == destination.resolve()
    assert destination.read_bytes() == b"abc"
    assert calls[0][-1] == REFRESH_COMMIT + ":data/pgr_financials.db"
    with pytest.raises(ValueError, match="SHA256"):
        provenance.export_git_blob(
            REFRESH_COMMIT,
            "data/pgr_financials.db",
            tmp_path / "bad.db",
            "0" * 64,
        )
    assert not (tmp_path / "bad.db").exists()


def test_git_export_cannot_write_repository_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="data"):
        provenance.export_git_blob(
            REFRESH_COMMIT,
            "data/pgr_financials.db",
            tmp_path / "data" / "nested" / "copy.db",
            ABC_SHA256,
        )
    assert not (tmp_path / "data").exists()


def test_clean_lock_verifies_exact_pins_and_consumed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    (tmp_path / "input.csv").write_bytes(b"abc")
    lock = clean_lock("input.csv")
    path = write_lock(tmp_path / "lock.json", lock)
    assert provenance.verify_baseline_lock(path) == lock
    (tmp_path / "input.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        provenance.verify_baseline_lock(path)


@pytest.mark.parametrize("status", ["provisional", "blocked", None])
def test_provisional_or_missing_clean_status_rejected(
    tmp_path: Path, status: str | None
) -> None:
    lock = clean_lock("input.csv")
    if status is None:
        del lock["status"]
    else:
        lock["status"] = status
    path = write_lock(tmp_path / "lock.json", lock)
    with pytest.raises(ValueError, match="clean"):
        provenance.verify_baseline_lock(path)


def test_absent_lock_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        provenance.verify_baseline_lock(tmp_path / "absent.json")


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("baseline_code_commit", "c3b4798"),
        ("baseline_code_commit", "0" * 40),
        ("parent_seed", {"git_commit": "0" * 40, "db_sha256": SEED_SHA256}),
        ("repairs", []),
        ("accepted_exceptions", []),
        ("input_files", []),
    ],
)
def test_lock_rejects_unpinned_or_missing_evidence(
    tmp_path: Path, key: str, replacement: Any
) -> None:
    lock = clean_lock("input.csv")
    lock[key] = replacement
    path = write_lock(tmp_path / "lock.json", lock)
    with pytest.raises(ValueError):
        provenance.verify_baseline_lock(path)


def test_lock_rejects_absolute_input_convenience_as_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    input_path = tmp_path / "input.csv"
    input_path.write_bytes(b"abc")
    lock = clean_lock(str(input_path))
    with pytest.raises(ValueError, match="relative"):
        provenance.verify_baseline_lock(
            write_lock(tmp_path / "lock.json", lock)
        )


def test_runtime_lock_records_installed_namespace_and_versions() -> None:
    lock = provenance.runtime_lock()
    assert lock["python"]
    assert lock["platform"]
    assert Path(lock["pgr_vds_path"]).is_file()
    assert lock["packages"]["numpy"]
    assert lock["packages"]["scikit-learn"]
    provenance.verify_runtime(lock)


def test_runtime_rejects_silent_dependency_upgrade() -> None:
    lock = provenance.runtime_lock()
    lock["packages"]["scikit-learn"] = "999.0.0"
    with pytest.raises(ValueError, match="scikit-learn"):
        provenance.verify_runtime(lock)
