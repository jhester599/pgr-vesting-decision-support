"""Check the relative links in the active Markdown docs.

Review 2026-09-25, section 5, phase 0: a link checker for the active docs, so
that file moves cannot silently break them. External links (``http:``,
``https:``, ``mailto:``) are not fetched. Every relative link must point to a
tracked-or-present file or directory, and a ``#fragment`` on a Markdown target
(or on the same file) must match one of its GitHub heading anchors.

The active docs are listed in ``ACTIVE_DOC_GLOBS``; the history trees
(``docs/plans``, ``docs/superpowers``, ``docs/closeouts``, ``docs/results``,
``docs/archive``, ``archive/`` and ``results/``) are frozen records and are
not checked, and neither are the generated reports under ``artifacts/``
(only its READMEs) or the study outputs under ``research/studies/`` (only
the index, the legacy README and each study's README).

Usage:
    python scripts/checks/check_doc_links.py            # exit 1 on a broken link
    python scripts/checks/check_doc_links.py FILE.md ... # check only these files
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[2]

ACTIVE_DOC_GLOBS: tuple[str, ...] = (
    "*.md",
    "docs/*.md",
    "docs/data/*.md",
    "docs/research/*.md",
    "docs/reviews/*.md",
    "artifacts/README.md",
    "artifacts/*/README.md",
    "research/README.md",
    "research/legacy/README.md",
    "research/studies/*/README.md",
)

_FENCE = re.compile(r"^\s*(```|~~~)")
_INLINE_CODE = re.compile(r"(`+)(?:(?!\1).)*?\1")
# [text](target "title") and ![alt](target); nested brackets one level deep.
_INLINE_LINK = re.compile(r"!?\[(?:[^\[\]]|\[[^\[\]]*\])*\]\(\s*<?([^()\s>]+(?:\([^()\s]*\))?)>?(?:\s+[\"'(][^)]*)?\s*\)")
_REF_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+.*)?$")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_HTML_ANCHOR = re.compile(r"<a\s+[^>]*(?:name|id)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
_EXTERNAL = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


@dataclass(frozen=True)
class BrokenLink:
    """One link that does not resolve."""

    source: str
    line: int
    target: str
    reason: str

    def __str__(self) -> str:
        return f"{self.source}:{self.line}: {self.target} ({self.reason})"


def active_docs(root: Path = REPO_ROOT) -> list[Path]:
    """Return the active Markdown docs, sorted."""
    found: set[Path] = set()
    for pattern in ACTIVE_DOC_GLOBS:
        for path in root.glob(pattern):
            if path.is_file():
                found.add(path)
    return sorted(found)


def github_slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower-case, punctuation dropped, spaces to '-'."""
    text = re.sub(r"<[^>]+>", "", heading)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace("`", "").strip().lower()
    kept = []
    for char in text:
        category = unicodedata.category(char)
        if char in "-_ " or category[0] in ("L", "N") or category == "Mn":
            kept.append(char)
    return "".join(kept).replace(" ", "-")


def _lines_outside_fences(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append((number, line))
    return lines


def anchors(path: Path) -> set[str]:
    """Every anchor a Markdown file defines (headings, with -1/-2 duplicates)."""
    counts: dict[str, int] = {}
    result: set[str] = set()
    for _, line in _lines_outside_fences(path.read_text(encoding="utf-8")):
        match = _HEADING.match(line)
        if match:
            slug = github_slug(match.group(2))
            seen = counts.get(slug, 0)
            result.add(slug if seen == 0 else f"{slug}-{seen}")
            counts[slug] = seen + 1
        for html in _HTML_ANCHOR.finditer(line):
            result.add(html.group(1))
    return result


def links(path: Path) -> list[tuple[int, str]]:
    """(line, target) for every inline link, image and reference definition."""
    found: list[tuple[int, str]] = []
    for number, line in _lines_outside_fences(path.read_text(encoding="utf-8")):
        ref = _REF_DEF.match(line)
        if ref:
            found.append((number, ref.group(1)))
            continue
        stripped = _INLINE_CODE.sub("", line)
        for match in _INLINE_LINK.finditer(stripped):
            found.append((number, match.group(1)))
    return found


def check_file(path: Path, root: Path = REPO_ROOT) -> list[BrokenLink]:
    """Return the broken relative links in one Markdown file."""
    broken: list[BrokenLink] = []
    rel_source = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
    anchor_cache: dict[Path, set[str]] = {}
    for line, raw in links(path):
        if _EXTERNAL.match(raw):
            continue
        target, _, fragment = raw.partition("#")
        target = unquote(target.split("?", 1)[0])
        if target:
            resolved = (root / target.lstrip("/")) if target.startswith("/") else (path.parent / target)
            if not resolved.exists():
                broken.append(BrokenLink(rel_source, line, raw, "missing file"))
                continue
        else:
            resolved = path
        if fragment and resolved.is_file() and resolved.suffix.lower() == ".md":
            if resolved not in anchor_cache:
                anchor_cache[resolved] = anchors(resolved)
            if unquote(fragment).lower() not in {a.lower() for a in anchor_cache[resolved]}:
                broken.append(BrokenLink(rel_source, line, raw, "missing anchor"))
    return broken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", type=Path, help="Markdown files (default: active docs)")
    args = parser.parse_args(argv)
    files = [p.resolve() for p in args.files] if args.files else active_docs()
    broken = [item for path in files for item in check_file(path)]
    for item in broken:
        print(item)
    print(f"[doc-links] {len(files)} files, {len(broken)} broken links")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
