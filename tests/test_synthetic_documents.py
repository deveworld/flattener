"""Reproducibility, actual font coverage, and generated document provenance."""

import hashlib
import json
import struct
import unicodedata
from importlib.resources import files

import numpy as np
import pytest

from flattener.synthetic.documents import (
    ENGLISH_SENTENCES,
    FONT_SHA256,
    KOREAN_SENTENCES,
    TITLES,
    _font,
    _wrap,
    generate_document,
)


def test_document_is_reproducible_and_hash_describes_the_pixels():
    first = generate_document("doc-1", 17)
    second = generate_document("doc-1", 17)
    np.testing.assert_array_equal(first.image, second.image)
    assert first.text == second.text
    assert first.metadata == second.metadata
    assert first.image.shape == (256, 192, 3)
    assert first.image.dtype == np.uint8
    assert first.image.flags.writeable
    assert first.metadata["source_sha256"] == hashlib.sha256(first.image.tobytes()).hexdigest()
    json.dumps(first.metadata, allow_nan=False)


@pytest.mark.parametrize(("document_id", "seed"), [("doc-2", 17), ("doc-1", 18)])
def test_seed_and_document_id_each_change_the_document(document_id, seed):
    base = generate_document("doc-1", 17)
    changed = generate_document(document_id, seed)
    assert changed.metadata["source_sha256"] != base.metadata["source_sha256"]
    assert changed.text != base.text


@pytest.mark.parametrize("size", [(128, 128), (192, 256)])
def test_document_layout_keeps_drawn_text_on_the_page(size):
    page = generate_document("layout", 41, size)
    assert page.image.shape == (*size, 3)
    height, width = size
    assert 0.01 < np.mean(np.any(page.image < 190, axis=-1)) < 0.6
    for record in page.metadata["text"]["boxes"]:
        x0, y0, x1, y1 = record["bounds_xyxy"]
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height
    assert page.metadata["original_aspect_ratio"] == width / height


def test_generated_page_contains_authored_bilingual_text_table_and_digits():
    page = generate_document("bilingual", 7)
    text = " ".join(page.text)
    assert any("가" <= char <= "힣" for char in text)
    assert any("a" <= char.lower() <= "z" for char in text)
    assert any(char.isdigit() for char in text)
    assert "표본 A" in text and "표본 B" in text
    assert unicodedata.normalize("NFC", text) == text
    assert page.metadata["text"]["license"] == "Apache-2.0"
    assert page.metadata["font"]["license"] == "OFL-1.1"
    assert page.metadata["font"]["modified"] is False
    assert page.metadata["external_photographs"] is False
    assert page.metadata["document_group"] == "bilingual"


def test_bundled_font_cmap_has_real_glyphs_for_all_authored_korean_and_latin():
    # Inspect the font's format-12 Unicode character map rather than comparing
    # pictures of .notdef boxes, which can give false positives for coverage.
    resource = files("flattener").joinpath("assets", "fonts", "NotoSansKR[wght].ttf")
    data = resource.read_bytes()
    assert hashlib.sha256(data).hexdigest() == FONT_SHA256
    count = struct.unpack_from(">H", data, 4)[0]
    table = None
    for index in range(count):
        tag, _, offset, _ = struct.unpack_from(">4sIII", data, 12 + 16 * index)
        if tag == b"cmap":
            table = offset
            break
    assert table is not None
    _, subtables = struct.unpack_from(">HH", data, table)
    groups = []
    for index in range(subtables):
        platform, encoding, relative = struct.unpack_from(">HHI", data, table + 4 + 8 * index)
        subtable = table + relative
        if platform not in (0, 3) or (platform == 3 and encoding != 10):
            continue
        if struct.unpack_from(">H", data, subtable)[0] != 12:
            continue
        n_groups = struct.unpack_from(">I", data, subtable + 12)[0]
        groups.extend(
            struct.unpack_from(">III", data, subtable + 16 + 12 * group)
            for group in range(n_groups)
        )
    assert groups
    chars = set(
        "".join((*KOREAN_SENTENCES, *ENGLISH_SENTENCES, *TITLES, "항목수량값표본0123456789 /"))
    )
    for char in chars:
        codepoint = ord(char)
        assert any(
            start <= codepoint <= end and glyph + codepoint - start > 0
            for start, end, glyph in groups
        ), repr(char)
    assert _font(24).getname()[0] == "Noto Sans KR"
    license_text = resource.parent.joinpath("OFL.txt").read_text()
    assert "SIL OPEN FONT LICENSE Version 1.1" in license_text
    assert "Copyright 2014-2021 Adobe" in license_text


def test_wrap_preserves_letters_and_words_without_inserting_internal_spaces():
    font = _font(20)
    english = "Three careful measurements form a useful pattern."
    assert " ".join(_wrap(english, font, 200)) == english
    korean = "나무의그림자는바람을따라천천히움직입니다."
    lines = _wrap(korean, font, 90)
    assert "".join(lines) == korean
    assert all(font.getlength(line) <= 90 for line in lines)


@pytest.mark.parametrize("document_id", ["", None])
def test_bad_document_ids_are_rejected(document_id):
    with pytest.raises(ValueError, match="document_id"):
        generate_document(document_id, 0)


@pytest.mark.parametrize("seed", [-1, True, "1"])
def test_bad_seeds_are_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        generate_document("doc", seed)


@pytest.mark.parametrize("size", [(0, 128), (128, True), None])
def test_bad_sizes_are_rejected(size):
    with pytest.raises(ValueError, match="output_size"):
        generate_document("doc", 0, size)
