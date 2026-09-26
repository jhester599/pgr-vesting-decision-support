"""Fail on new ``sys.path`` edits outside ``tests/conftest.py``.

Review 2026-09-25, section 5, phase 0. The code is installable
(``pip install -e .``), so new modules must import ``src`` and ``config``
without editing ``sys.path``. The files that already edit it are listed in
``sys_path_allowlist.txt`` next to this script. The list only shrinks:

- a tracked file that edits ``sys.path`` and is not listed fails;
- a listed file that no longer edits ``sys.path`` (or no longer exists) also
  fails, so remove it from the list in the same change.

Python files and GitHub workflow files (inline scripts) are scanned.

Usage:
    python scripts/checks/check_sys_path_edits.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_PATH = Path(__file__).with_name("sys_path_allowlist.txt")
ALWAYS_ALLOWED: frozenset[str] = frozenset({"tests/conftest.py"})
SCANNED_SUFFIXES: tuple[str, ...] = (".py", ".yml", ".yaml")

SYS_PATH_EDIT = re.compile(
    r"sys\.path\s*\.\s*(?:insert|append|extend|remove|pop|clear)\s*\("
    r"|sys\.path\s*(?:\+=|=(?!=))"
    r"|sys\.path\s*\[[^\]]*\]\s*=(?!=)"
    r"|site\.addsitedir\s*\("
)


def tracked_files(root: Path = REPO_ROOT) -> list[str]:
    """Tracked (and staged) files, as repo-relative POSIX paths."""
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(line for line in output.splitlines() if line.endswith(SCANNED_SUFFIXES))


def edits_sys_path(text: str) -> bool:
    """True when the text edits ``sys.path`` outside a comment line."""
    for line in text.splitlines():
        code = line.split("#", 1)[0] if not line.lstrip().startswith("#") else ""
        if SYS_PATH_EDIT.search(code):
            return True
    return False


def load_allowlist(path: Path = ALLOWLIST_PATH) -> set[str]:
    """Allowlisted repo-relative paths (blank lines and # comments ignored)."""
    entries: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            entries.add(entry)
    return entries


def find_violations(
    files: list[str],
    allowlist: set[str],
    root: Path = REPO_ROOT,
) -> tuple[list[str], list[str]]:
    """Return (new edits not allowlisted, stale allowlist entries)."""
    editing: set[str] = set()
    for rel in files:
        path = root / rel
        if not path.is_file():
            continue
        if edits_sys_path(path.read_text(encoding="utf-8", errors="replace")):
            editing.add(rel)
    new = sorted(editing - allowlist - ALWAYS_ALLOWED)
    stale = sorted(allowlist - editing)
    return new, stale


def main() -> int:
    new, stale = find_violations(tracked_files(), load_allowlist())
    for rel in new:
        print(f"{rel}: new sys.path edit (import src/config via `pip install -e .` instead)")
    for rel in stale:
        print(f"{rel}: listed in {ALLOWLIST_PATH.name} but no longer edits sys.path; remove it")
    print(f"[sys-path] {len(new)} new edits, {len(stale)} stale allowlist entries")
    return 1 if new or stale else 0


if __name__ == "__main__":
    sys.exit(main())
