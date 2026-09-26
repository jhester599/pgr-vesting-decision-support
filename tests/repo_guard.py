"""Keep tests out of the repository tree and the committed database.

Review 2026-09-25, F28 (step 9): 50 tests wrote
``data/processed/feature_matrix.parquet`` into the working tree, and research
tests opened the committed ``data/pgr_financials.db`` (a read-write open
switches it to WAL mode and leaves ``-wal``/``-shm`` files beside it).

``tests/conftest.py`` installs ``audit_hook`` once per process. While a test
runs (setup, call and teardown), every Python-level file open, SQLite
connect and directory change is checked by ``classify_access``:

- any write inside the repository tree is refused, except ``__pycache__``
  and ``.pytest_cache`` (the interpreter and pytest, not the test);
- opening the committed database is refused, unless the test is marked
  ``artifact`` and the open is read-only (a ``mode=ro`` SQLite URI or a
  plain read), because artifact tests check committed data by design.

A refused access raises ``PermissionError`` at the call site and is also
recorded, so a test that swallows the error still fails at teardown.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_DB = REPO_ROOT / "data" / "pgr_financials.db"

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_EXEMPT_PARTS = frozenset({"__pycache__", ".pytest_cache"})
_WATCHED_EVENTS = frozenset(
    {"open", "sqlite3.connect", "os.mkdir", "os.remove", "os.rename", "os.rmdir", "shutil.rmtree"}
)


@dataclass
class GuardState:
    """What the guard is checking now, and what it has refused."""

    test_id: str | None = None
    allow_committed_reads: bool = False
    violations: list[str] = field(default_factory=list)


STATE = GuardState()
_INSTALLED = False


def _as_path(raw: object) -> tuple[Path | None, bool]:
    """Absolute path for an audited path argument, and whether a SQLite URI
    asked for read-only access. Returns (None, False) for file descriptors,
    ``:memory:`` and anything that is not a path."""
    if isinstance(raw, int) or raw is None:
        return None, False
    try:
        text = os.fsdecode(raw)
    except TypeError:
        return None, False
    read_only = False
    if text.startswith("file:"):
        parsed = urlparse(text)
        read_only = parse_qs(parsed.query).get("mode", [""])[0] == "ro"
        text = unquote(parsed.path)
    if text in ("", ":memory:"):
        return None, False
    return Path(os.path.abspath(text)), read_only


def _in_repo(path: Path) -> bool:
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError:
        return False
    return not _EXEMPT_PARTS.intersection(relative.parts)


def _is_committed_db(path: Path) -> bool:
    return path == COMMITTED_DB


def classify_access(event: str, args: tuple, allow_committed_reads: bool) -> str | None:
    """Return why an audited access is refused, or None when it is allowed."""
    if event not in _WATCHED_EVENTS or not args:
        return None
    if event == "os.rename":
        targets = [args[0], args[1]]
    else:
        targets = [args[0]]
    for raw in targets:
        path, read_only_uri = _as_path(raw)
        if path is None:
            continue
        if event == "sqlite3.connect":
            if _is_committed_db(path):
                if allow_committed_reads and read_only_uri:
                    continue
                if allow_committed_reads:
                    return f"read-write SQLite connection to the committed database {path}"
                return f"SQLite connection to the committed database {path}"
            if _in_repo(path) and not read_only_uri:
                return f"SQLite connection that can write inside the repository: {path}"
            continue
        if event == "open":
            mode, flags = args[1], args[2]
            if isinstance(mode, str):
                writes = any(ch in mode for ch in "wax+")
            else:
                writes = bool(flags & _WRITE_FLAGS)
            if _is_committed_db(path) and not (allow_committed_reads and not writes):
                return f"opened the committed database {path} (mode={mode!r})"
            if writes and _in_repo(path):
                return f"write inside the repository: {path} (mode={mode!r})"
            continue
        if event == "os.mkdir" and path.is_dir():
            continue  # makedirs(exist_ok=True) on an existing directory
        if _in_repo(path):
            return f"{event} inside the repository: {path}"
    return None


def audit_hook(event: str, args: tuple) -> None:
    """``sys.addaudithook`` callback; active only while a test runs."""
    if STATE.test_id is None or event not in _WATCHED_EVENTS:
        return
    reason = classify_access(event, args, STATE.allow_committed_reads)
    if reason is None:
        return
    STATE.violations.append(reason)
    raise PermissionError(f"[repo guard] {STATE.test_id}: {reason}")


def install() -> None:
    """Register the audit hook once per process (hooks cannot be removed)."""
    global _INSTALLED
    if not _INSTALLED:
        sys.addaudithook(audit_hook)
        _INSTALLED = True


def start(test_id: str, allow_committed_reads: bool) -> None:
    STATE.test_id = test_id
    STATE.allow_committed_reads = allow_committed_reads
    STATE.violations = []


def stop() -> list[str]:
    """Stop checking and return the refused accesses of the finished test."""
    violations = STATE.violations
    STATE.test_id = None
    STATE.allow_committed_reads = False
    STATE.violations = []
    return violations
