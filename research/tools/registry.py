"""Study registry: validate ``research/registry.yaml`` and render the README.

Review 2026-09-25, section 5, phase 3. Every folder under
``research/studies/`` is one study and must have exactly one registry entry
(and every entry a folder). ``research/README.md`` is generated from the
registry; do not edit it by hand.

Usage (from the repository root):
    python research/tools/registry.py            # check (CI); exit 1 on errors
    python research/tools/registry.py --write    # regenerate research/README.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / "research"
REGISTRY_PATH = RESEARCH_DIR / "registry.yaml"
STUDIES_DIR = RESEARCH_DIR / "studies"
README_PATH = RESEARCH_DIR / "README.md"

REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "slug",
    "date",
    "question",
    "status",
    "promoted_to",
    "closeout",
)
OPTIONAL_FIELDS: tuple[str, ...] = ("notes",)
STATUSES: dict[str, str] = {
    "promoted": "its result sets a live production setting",
    "retained": "it tested a live setting and kept the incumbent value",
    "shadow": "its result feeds a reporting-only (shadow) lane of the monthly run",
    "closed": "research only; nothing it found is used by the monthly run",
}
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def load_registry(path: Path = REGISTRY_PATH) -> list[dict[str, Any]]:
    """Return the registry entries, in file order."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("studies"), list):
        raise ValueError(f"{path}: expected a mapping with a 'studies' list")
    return list(data["studies"])


def folder_name(entry: dict[str, Any]) -> str:
    """Return the study folder name, ``<id>_<slug>``."""
    return f"{entry['id']}_{entry['slug']}"


def study_folders(studies_dir: Path = STUDIES_DIR) -> list[str]:
    """Return the names of the study folders on disk."""
    if not studies_dir.is_dir():
        return []
    return sorted(
        p.name for p in studies_dir.iterdir()
        if p.is_dir() and not p.name.startswith((".", "__"))
    )


def validate(
    entries: list[dict[str, Any]],
    studies_dir: Path = STUDIES_DIR,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Return a list of problems; empty when the registry is consistent."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    registered: set[str] = set()
    for n, entry in enumerate(entries, 1):
        where = f"entry {n} ({entry.get('id', '?')})"
        if not isinstance(entry, dict):
            errors.append(f"{where}: not a mapping")
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        unknown = sorted(set(entry) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
        if missing:
            errors.append(f"{where}: missing {', '.join(missing)}")
            continue
        if unknown:
            errors.append(f"{where}: unknown field(s) {', '.join(unknown)}")
        sid, slug = str(entry["id"]), str(entry["slug"])
        if not _ID.match(sid) or not _SLUG.match(slug):
            errors.append(f"{where}: id and slug must be lower-case [a-z0-9_] names")
        if sid in seen_ids:
            errors.append(f"{where}: duplicate id")
        seen_ids.add(sid)
        if not _DATE.match(str(entry["date"])):
            errors.append(f"{where}: date must be a quoted YYYY-MM-DD string")
        if not str(entry["question"]).strip():
            errors.append(f"{where}: empty question")
        status = entry["status"]
        if status not in STATUSES:
            errors.append(f"{where}: status {status!r} not in {sorted(STATUSES)}")
        promoted_to = entry["promoted_to"]
        if status in ("promoted", "retained", "shadow") and not promoted_to:
            errors.append(f"{where}: status {status} needs promoted_to")
        if status == "closed" and promoted_to:
            errors.append(f"{where}: a closed study has no promoted_to")
        closeout = entry["closeout"]
        if closeout and not (repo_root / str(closeout)).is_file():
            errors.append(f"{where}: closeout {closeout} does not exist")
        registered.add(folder_name(entry))

    on_disk = set(study_folders(studies_dir))
    for name in sorted(on_disk - registered):
        errors.append(f"research/studies/{name}: folder is not in the registry")
    for name in sorted(registered - on_disk):
        errors.append(f"research/studies/{name}: registered but the folder does not exist")
    return errors


def _cell(text: Any) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def render_readme(entries: list[dict[str, Any]]) -> str:
    """Render ``research/README.md`` from the registry entries."""
    lines = [
        "# Research",
        "",
        "<!-- Generated by research/tools/registry.py from research/registry.yaml."
        " Do not edit by hand. -->",
        "",
        "One folder per study under [`studies/`](studies/), named `<id>_<slug>`.",
        "Each holds the study's script(s), a README, and `outputs/` (summary",
        "CSV/MD/JSON files). Large per-fold detail files go to `outputs/detail/`,",
        "which is not committed; the study README says how to regenerate them.",
        "[`legacy/`](legacy/) holds the v9–v28 result folders unchanged.",
        "",
        "To add a study: create `studies/<id>_<slug>/`, add an entry to",
        "[`registry.yaml`](registry.yaml), then run",
        "`python research/tools/registry.py --write`. CI runs",
        "`python research/tools/registry.py`, which fails when a study folder is",
        "not registered or this README is out of date.",
        "",
        "Run study scripts from the repository root, for example",
        "`python research/studies/v38_shrinkage/v38_shrinkage.py`. Their tests",
        "are in [`tests/research/`](../tests/research/).",
        "",
        "## Status",
        "",
    ]
    for status, meaning in STATUSES.items():
        count = sum(1 for e in entries if e["status"] == status)
        lines.append(f"- **{status}** ({count}): {meaning}.")
    lines += [
        "",
        "## Studies",
        "",
        "| Study | Date | Question | Status | Promoted to | Closeout |",
        "|---|---|---|---|---|---|",
    ]
    for entry in sorted(entries, key=_sort_key):
        folder = folder_name(entry)
        closeout = entry["closeout"]
        closeout_cell = f"[{Path(str(closeout)).name}](../{closeout})" if closeout else ""
        promoted = entry["promoted_to"]
        promoted_cell = f"`{_cell(promoted)}`" if promoted else ""
        lines.append(
            f"| [{folder}](studies/{folder}/) | {entry['date']} | "
            f"{_cell(entry['question'])} | {entry['status']} | {promoted_cell} | "
            f"{closeout_cell} |"
        )
    lines.append("")
    return "\n".join(lines)


def _sort_key(entry: dict[str, Any]) -> tuple[str, int, str]:
    m = re.match(r"^([a-z]+)(\d+)$", str(entry["id"]))
    if m:
        return (m.group(1), int(m.group(2)), "")
    return ("~", 0, str(entry["id"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="Regenerate research/README.md.")
    args = parser.parse_args(argv)

    entries = load_registry()
    errors = validate(entries)
    expected = render_readme(entries)
    if args.write:
        README_PATH.write_text(expected, encoding="utf-8")
        print(f"[registry] wrote {README_PATH.relative_to(REPO_ROOT)}")
    elif not README_PATH.exists() or README_PATH.read_text(encoding="utf-8") != expected:
        errors.append(
            "research/README.md is out of date; run python research/tools/registry.py --write"
        )
    for error in errors:
        print(f"[registry] {error}")
    print(f"[registry] {len(entries)} studies, {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
