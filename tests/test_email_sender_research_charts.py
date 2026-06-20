"""Tests for inline research chart embedding in the monthly decision email.

Covers:
- _build_research_charts_html: CID references, labels, empty input, HTML escaping
- build_email_message with charts: MIME structure, image attachment, CID headers
- build_email_message graceful degradation: no paths, missing files, partial files
- build_email_html: chart section presence/absence
- _CHART_LABEL_MAP / _RESEARCH_CHART_NAMES: constant integrity
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from src.reporting.email_sender import (
    _CHART_LABEL_MAP,
    _RESEARCH_CHART_NAMES,
    _build_research_charts_html,
    build_email_html,
    build_email_message,
)

# ---------------------------------------------------------------------------
# Minimal valid 1×1 white PNG bytes (built without PIL dependency)
# ---------------------------------------------------------------------------

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


_MINIMAL_PNG: bytes = (
    b"\x89PNG\r\n\x1a\n"
    + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    + _png_chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    + _png_chunk(b"IEND", b"")
)

_MINIMAL_BODY = """\
# PGR Monthly Decision - June 2026

**As-of Date:** 2026-06-20

## Consensus Signal

| Field | Value |
|-------|-------|
| Signal | **NEUTRAL** |
| Recommendation Mode | **MONITORING-ONLY** |
| Recommended Sell % | **50%** |
| Predicted 6M Relative Return | +1.00% |
"""

# ---------------------------------------------------------------------------
# MIME navigation helpers
# ---------------------------------------------------------------------------

def _is_with_charts(msg) -> bool:
    return msg.get_content_subtype() == "mixed"


def _plain_text(msg) -> str:
    if _is_with_charts(msg):
        alt = msg.get_payload()[0]
        return alt.get_payload()[0].get_payload(decode=True).decode("utf-8")
    return msg.get_payload()[0].get_payload(decode=True).decode("utf-8")


def _html_text(msg) -> str:
    if _is_with_charts(msg):
        alt = msg.get_payload()[0]
        related = alt.get_payload()[1]
        return related.get_payload()[0].get_payload(decode=True).decode("utf-8")
    return msg.get_payload()[1].get_payload(decode=True).decode("utf-8")


def _image_parts(msg) -> list:
    if not _is_with_charts(msg):
        return []
    alt = msg.get_payload()[0]
    related = alt.get_payload()[1]
    return related.get_payload()[1:]


# ---------------------------------------------------------------------------
# _build_research_charts_html
# ---------------------------------------------------------------------------

def test_build_research_charts_html_empty_returns_empty_string() -> None:
    assert _build_research_charts_html([]) == ""


def test_build_research_charts_html_single_chart_has_cid_img_tag() -> None:
    html = _build_research_charts_html([("pgrchart0@pgr", "PGR Share Price")])
    assert 'src="cid:pgrchart0@pgr"' in html


def test_build_research_charts_html_single_chart_has_label_text() -> None:
    html = _build_research_charts_html([("pgrchart0@pgr", "PGR Share Price")])
    assert "PGR Share Price" in html


def test_build_research_charts_html_section_heading_present() -> None:
    html = _build_research_charts_html([("pgrchart0@pgr", "Some Chart")])
    assert "PGR Research Charts" in html


def test_build_research_charts_html_multiple_charts_all_have_cid_tags() -> None:
    pairs = [
        ("pgrchart0@pgr", "Share Price"),
        ("pgrchart1@pgr", "BVPS"),
        ("pgrchart2@pgr", "P/B Ratio"),
    ]
    html = _build_research_charts_html(pairs)
    for cid, _ in pairs:
        assert f'src="cid:{cid}"' in html


def test_build_research_charts_html_escapes_label_for_html() -> None:
    html = _build_research_charts_html([("cid0@pgr", "A <b>bold</b> & label")])
    assert "<b>" not in html
    assert "&lt;b&gt;" in html or "bold" in html


# ---------------------------------------------------------------------------
# Constant integrity
# ---------------------------------------------------------------------------

def test_all_research_chart_names_are_in_label_map() -> None:
    for name in _RESEARCH_CHART_NAMES:
        assert name in _CHART_LABEL_MAP, f"{name!r} missing from _CHART_LABEL_MAP"


def test_research_chart_names_order() -> None:
    assert _RESEARCH_CHART_NAMES == [
        "pgr_share_price.png",
        "pgr_book_value_per_share.png",
        "pgr_price_to_book.png",
        "pgr_repurchase_dollar_amount_capped.png",
    ]


# ---------------------------------------------------------------------------
# build_email_message — graceful degradation (no files)
# ---------------------------------------------------------------------------

def test_no_chart_paths_produces_flat_alternative() -> None:
    msg = build_email_message(
        _MINIMAL_BODY, "from@example.com", "to@example.com", chart_paths=None
    )
    assert msg.get_content_subtype() == "alternative"


def test_empty_chart_paths_list_produces_flat_alternative() -> None:
    msg = build_email_message(
        _MINIMAL_BODY, "from@example.com", "to@example.com", chart_paths=[]
    )
    assert msg.get_content_subtype() == "alternative"


def test_all_chart_paths_missing_produces_flat_alternative(tmp_path: Path) -> None:
    paths = [tmp_path / "nonexistent1.png", tmp_path / "nonexistent2.png"]
    msg = build_email_message(
        _MINIMAL_BODY, "from@example.com", "to@example.com", chart_paths=paths
    )
    assert msg.get_content_subtype() == "alternative"


# ---------------------------------------------------------------------------
# build_email_message — with charts present
# ---------------------------------------------------------------------------

@pytest.fixture()
def one_chart(tmp_path: Path) -> list[Path]:
    p = tmp_path / "pgr_share_price.png"
    p.write_bytes(_MINIMAL_PNG)
    return [p]


@pytest.fixture()
def four_charts(tmp_path: Path) -> list[Path]:
    paths = []
    for name in _RESEARCH_CHART_NAMES:
        p = tmp_path / name
        p.write_bytes(_MINIMAL_PNG)
        paths.append(p)
    return paths


def test_with_charts_root_is_multipart_mixed(one_chart: list[Path]) -> None:
    msg = build_email_message(
        _MINIMAL_BODY, "from@example.com", "to@example.com", chart_paths=one_chart
    )
    assert msg.get_content_subtype() == "mixed"


def test_with_charts_plain_text_is_accessible(one_chart: list[Path]) -> None:
    msg = build_email_message(
        _MINIMAL_BODY, "from@example.com", "to@example.com", chart_paths=one_chart
    )
    plain = _plain_text(msg)
    assert "PGR Monthly Decision" in plain


def test_with_charts_html_is_accessible(one_chart: list[Path]) -> None:
    msg = build_email_message(
        _MINIMAL_BODY, "from@example.com", "to@example.com", chart_paths=one_chart
    )
    html = _html_text(msg)
    assert "<html" in html


def test_with_charts_html_contains_cid_reference(one_chart: list[Path]) -> None:
    msg = build_email_message(
        _MINIMAL_BODY, "from@example.com", "to@example.com", chart_paths=one_chart
    )
    html = _html_text(msg)
    assert "cid:pgrchart0@pgr" in html


def test_with_charts_one_file_produces_one_image_part(one_chart: list[Path]) -> None:
    msg = build_email_message(
        _MINIMAL_BODY, "from@example.com", "to@example.com", chart_paths=one_chart
    )
    assert len(_image_parts(msg)) == 1


def test_with_four_charts_produces_four_image_parts(four_charts: list[Path]) -> None:
    msg = build_email_message(
        _MINIMAL_BODY, "from@example.com", "to@example.com", chart_paths=four_charts
    )
    assert len(_image_parts(msg)) == 4


def test_image_parts_have_correct_content_id_headers(four_charts: list[Path]) -> None:
    msg = build_email_message(
        _MINIMAL_BODY, "from@example.com", "to@example.com", chart_paths=four_charts
    )
    for i, part in enumerate(_image_parts(msg)):
        assert part["Content-ID"] == f"<pgrchart{i}@pgr>"


def test_image_parts_have_inline_content_disposition(one_chart: list[Path]) -> None:
    msg = build_email_message(
        _MINIMAL_BODY, "from@example.com", "to@example.com", chart_paths=one_chart
    )
    part = _image_parts(msg)[0]
    assert "inline" in part.get("Content-Disposition", "")


def test_image_part_data_matches_file_content(one_chart: list[Path]) -> None:
    msg = build_email_message(
        _MINIMAL_BODY, "from@example.com", "to@example.com", chart_paths=one_chart
    )
    part = _image_parts(msg)[0]
    assert part.get_payload(decode=True) == _MINIMAL_PNG


def test_partial_charts_only_existing_files_are_attached(tmp_path: Path) -> None:
    existing = tmp_path / "pgr_share_price.png"
    existing.write_bytes(_MINIMAL_PNG)
    missing = tmp_path / "pgr_price_to_book.png"  # not written
    msg = build_email_message(
        _MINIMAL_BODY, "from@example.com", "to@example.com",
        chart_paths=[existing, missing],
    )
    assert _is_with_charts(msg)
    assert len(_image_parts(msg)) == 1


def test_known_filename_uses_chart_label_map(tmp_path: Path) -> None:
    p = tmp_path / "pgr_price_to_book.png"
    p.write_bytes(_MINIMAL_PNG)
    msg = build_email_message(
        _MINIMAL_BODY, "from@example.com", "to@example.com", chart_paths=[p]
    )
    html = _html_text(msg)
    assert "PGR Price / Book Ratio" in html


def test_unknown_filename_uses_stem_as_label(tmp_path: Path) -> None:
    p = tmp_path / "my_custom_chart.png"
    p.write_bytes(_MINIMAL_PNG)
    msg = build_email_message(
        _MINIMAL_BODY, "from@example.com", "to@example.com", chart_paths=[p]
    )
    html = _html_text(msg)
    assert "My Custom Chart" in html


def test_subject_and_headers_preserved_with_charts(one_chart: list[Path]) -> None:
    msg = build_email_message(
        _MINIMAL_BODY,
        "from@example.com",
        "to@example.com",
        "June 2026",
        chart_paths=one_chart,
    )
    assert "June 2026" in msg["Subject"]
    assert msg["From"] == "from@example.com"
    assert msg["To"] == "to@example.com"


# ---------------------------------------------------------------------------
# build_email_html — chart section presence/absence
# ---------------------------------------------------------------------------

def test_build_email_html_without_cids_has_no_chart_section() -> None:
    html = build_email_html(_MINIMAL_BODY, chart_cids=None)
    assert "PGR Research Charts" not in html


def test_build_email_html_with_cids_has_chart_section() -> None:
    html = build_email_html(
        _MINIMAL_BODY,
        chart_cids=[("pgrchart0@pgr", "PGR Share Price")],
    )
    assert "PGR Research Charts" in html
    assert 'src="cid:pgrchart0@pgr"' in html
