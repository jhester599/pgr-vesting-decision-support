"""Probe tests for the repo guard (review 2026-09-25, F28, step 9).

Not collected by default (the name does not match ``test_*.py``).
``tests/test_wp12_test_hygiene.py`` runs this file in a subprocess and
checks which probes the guard fails. Every probe that touches the
repository is refused before the file is created, so nothing is written.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import config
from src.processing import feature_engineering
from tests import repo_guard

_PROBE_FILE = repo_guard.REPO_ROOT / "results" / "guard_probe_wp12.txt"


def test_probe_write_inside_repo() -> None:
    _PROBE_FILE.write_text("x", encoding="utf-8")


def test_probe_swallowed_write_inside_repo() -> None:
    try:
        _PROBE_FILE.write_text("x", encoding="utf-8")
    except OSError:
        pass


def test_probe_mkdir_inside_repo() -> None:
    (repo_guard.REPO_ROOT / "results" / "guard_probe_wp12_dir").mkdir()


def test_probe_unmarked_read_only_db_open() -> None:
    sqlite3.connect(f"{repo_guard.COMMITTED_DB.as_uri()}?mode=ro", uri=True).close()


def test_probe_unmarked_plain_file_read_of_db() -> None:
    with open(repo_guard.COMMITTED_DB, "rb") as fh:
        fh.read(16)


@pytest.mark.artifact
def test_probe_artifact_read_write_db_open() -> None:
    sqlite3.connect(str(repo_guard.COMMITTED_DB)).close()


@pytest.mark.artifact
def test_probe_artifact_read_only_db_open() -> None:
    conn = sqlite3.connect(f"{repo_guard.COMMITTED_DB.as_uri()}?mode=ro", uri=True)
    conn.execute("SELECT 1").fetchone()
    conn.close()


@pytest.mark.artifact
def test_probe_artifact_committed_db_copy(committed_db_copy: Path) -> None:
    assert Path(config.DB_PATH) == committed_db_copy
    sqlite3.connect(str(committed_db_copy)).close()


def test_probe_default_paths_are_in_tmp(tmp_path: Path) -> None:
    assert Path(config.DB_PATH).parent == tmp_path
    assert Path(feature_engineering._PROCESSED_PATH).parent == tmp_path


def test_probe_writes_outside_repo_are_allowed(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("ok", encoding="utf-8")
    sqlite3.connect(str(tmp_path / "ok.db")).close()
