"""Verify historical registry bytes and preserve all other baseline pins."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from pgr_vds.research_lib.provenance import (
    REPO_ROOT,
    source_sha256,
    verify_baseline_lock,
)


REGISTRY = "research/registry.yaml"
# PyYAML is runtime-pinned but ships no typing stubs. Validate its decoded
# values below rather than adding a dev dependency to the locked runtime.
yaml = importlib.import_module("yaml")


def git_blob(args: list[str], **kwargs: Any) -> bytes:
    """Read local Git bytes without a network call."""
    return subprocess.check_output(args, **kwargs)


def _entries(content: bytes) -> dict[str, dict]:
    """Require unique study IDs before checking semantic preservation."""
    document = yaml.safe_load(content)
    rows = document.get("studies") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Registry must contain a study list")
    entries: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("Registry entries need string IDs")
        if row["id"] in entries:
            raise ValueError("Duplicate registry IDs")
        entries[row["id"]] = row
    return entries


def verify_registry_growth(lock_path: Path, scratch: Path) -> dict[str, Any]:
    """Check the unchanged v200 lock using its exact historical registry.

    The mandatory downstream entry cannot match v200's old file bytes.
    Only that source file is checked at v200's full execution commit. Its
    original entries must still exist unchanged; every other input retains
    the original current-checkout SHA256 check. The accepted lock is never
    edited and the returned lock is the original full document.
    """
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    commit = lock.get("research_execution_code_commit")
    if not isinstance(commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", commit
    ):
        raise ValueError("Require a full execution commit for the registry")
    files = lock.get("input_files", [])
    pins = [item for item in files if item.get("path") == REGISTRY]
    if len(pins) != 1 or pins[0].get("hash_basis") != "git_source_lf":
        raise ValueError("Require one Git-LF registry input pin")
    pinned = git_blob(
        ["git", "show", f"{commit}:{REGISTRY}"],
        cwd=REPO_ROOT,
    ).replace(b"\r\n", b"\n")
    if hashlib.sha256(pinned).hexdigest() != pins[0].get("sha256"):
        raise ValueError("Pinned historical registry SHA256 differs")
    before = _entries(pinned)
    after = _entries((REPO_ROOT / REGISTRY).read_bytes())
    changed = [key for key, row in before.items() if after.get(key) != row]
    if changed:
        raise ValueError(f"Pinned registry entries changed: {changed}")
    filtered = {
        **lock,
        "input_files": [
            item for item in files if item.get("path") != REGISTRY
        ],
    }
    scratch.mkdir(parents=True, exist_ok=True)
    temporary = scratch / "baseline_other_inputs.json"
    temporary.write_text(json.dumps(filtered), encoding="utf-8")
    verify_baseline_lock(temporary)
    return {
        "lock": lock,
        "registry_check": {
            "path": REGISTRY,
            "pinned_sha256": pins[0]["sha256"],
            "pinned_at_commit": commit,
            "current_sha256": source_sha256(REPO_ROOT / REGISTRY),
            "pinned_entries_unchanged": len(before),
            "added_entries": [key for key in after if key not in before],
        },
    }
