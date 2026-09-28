"""Image adapter checks using hand-written orientations and pixel expectations."""

import json
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from flattener import (
    FlattenerError,
    RenderConfig,
    Status,
    load_image,
    render_file,
    save_result,
)

# The stored test image is 2x3 with labels 1..6; each EXIF orientation shows it upright as:
UPRIGHT = {
    1: [[1, 2, 3], [4, 5, 6]],
    2: [[3, 2, 1], [6, 5, 4]],
    3: [[6, 5, 4], [3, 2, 1]],
    4: [[4, 5, 6], [1, 2, 3]],
    5: [[1, 4], [2, 5], [3, 6]],
    6: [[4, 1], [5, 2], [6, 3]],
    7: [[6, 3], [5, 2], [4, 1]],
    8: [[3, 6], [2, 5], [1, 4]],
}


def _rgb(labels):
    labels = np.asarray(labels, dtype=np.uint8)
    return np.stack((labels * 30, labels * 11, labels * 7), axis=-1)


def _save(path, orientation=None):
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    Image.fromarray(_rgb(UPRIGHT[1])).save(path, exif=exif)
    return path


def _centers(height, width):
    rows, columns = np.indices((height, width))
    return np.stack(
        (2 * (columns + 0.5) / width - 1, 2 * (rows + 0.5) / height - 1), axis=-1
    ).astype(np.float32)


def test_every_exif_orientation_is_applied(tmp_path):
    for orientation, labels in UPRIGHT.items():
        actual = load_image(_save(tmp_path / f"oriented-{orientation}.png", orientation))
        assert actual.dtype == np.uint8 and actual.flags.writeable
        np.testing.assert_array_equal(actual, _rgb(labels), err_msg=f"orientation {orientation}")


def test_grayscale_is_replicated_into_rgb(tmp_path):
    for extension in ("png", "jpg"):
        path = tmp_path / f"gray.{extension}"
        Image.new("L", (3, 2), color=91).save(path)
        np.testing.assert_array_equal(load_image(path), np.full((2, 3, 3), 91, np.uint8))


def test_broken_and_non_eight_bit_files_are_refused(tmp_path):
    def chunk(name, payload):
        checksum = zlib.crc32(name + payload)
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", checksum)

    # A 1x1 16-bit RGB PNG: Pillow narrows it to RGB, so a mode check alone would pass it.
    ihdr = struct.pack(">IIBBBBB", 1, 1, 16, 2, 0, 0, 0)
    row = b"\x00" + struct.pack(">HHH", 65535, 32768, 1)
    deep = tmp_path / "sixteen_bit.png"
    deep.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(row))
        + chunk(b"IEND", b"")
    )
    with Image.open(deep) as decoded:
        assert decoded.mode == "RGB"
    with pytest.raises(FlattenerError) as caught:
        load_image(deep)
    assert caught.value.reason_code == "unsupported_mode"

    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    for path in (broken, tmp_path / "missing.png"):
        with pytest.raises(FlattenerError) as caught:
            load_image(path)
        assert caught.value.status == Status.INPUT_ERROR
        assert caught.value.reason_code == "decode_failed"


def test_input_size_is_checked_before_decoding(tmp_path, monkeypatch):
    source = _save(tmp_path / "input.png")

    def no_decode(*args, **kwargs):
        raise AssertionError("Image pixels should not be decoded")

    monkeypatch.setattr(Image.Image, "load", no_decode)
    with pytest.raises(FlattenerError) as caught:
        load_image(source, config=RenderConfig(max_input_pixels=5))
    assert caught.value.reason_code == "size_limit_exceeded"


def test_render_file_saves_the_upright_result_and_never_overwrites_silently(tmp_path):
    # Orientation 6 stores the page on its side: the upright image is 3 high, 2 wide.
    source = _save(tmp_path / "oriented.png", 6)
    upright = _rgb(UPRIGHT[6])
    output_dir = tmp_path / "result"
    result = render_file(source, _centers(3, 2), output_dir=output_dir)
    assert result.status == Status.SUCCESS
    np.testing.assert_array_equal(result.image, upright)  # the EXIF turn is applied once
    record = json.loads((output_dir / "metadata.json").read_text())
    assert record["status"] == "success" and record["used_fallback"] is False
    assert record["artifacts"] == result.metadata["artifacts"]
    assert set(record["artifacts"]) == {
        "image",
        "source_mask",
        "target_mask",
        "valid_mask",
        "metadata",
    }
    np.testing.assert_array_equal(load_image(output_dir / "image.png"), upright)
    for name in ("source_mask", "target_mask", "valid_mask"):
        with Image.open(output_dir / f"{name}.png") as mask:
            assert mask.mode == "L"
            np.testing.assert_array_equal(np.asarray(mask), np.full((3, 2), 255, np.uint8))
    assert record["metadata"]["timing_scope"] == "file"
    assert record["metadata"]["source_size"] == [3, 2]

    original = (output_dir / "metadata.json").read_bytes()
    with pytest.raises(FlattenerError) as caught:
        render_file(source, _centers(3, 2), output_dir=output_dir)
    assert caught.value.reason_code == "output_exists"
    with pytest.raises(FlattenerError) as caught:
        save_result(result, output_dir)
    assert caught.value.reason_code == "output_exists"
    assert caught.value.status == Status.IO_ERROR
    assert (output_dir / "metadata.json").read_bytes() == original
    result.metadata["test_marker"] = "replacement"
    save_result(result, output_dir, overwrite=True)
    record = json.loads((output_dir / "metadata.json").read_text())
    assert record["metadata"]["test_marker"] == "replacement"


def test_failure_status_is_saved_and_fallback_is_marked(tmp_path):
    source = _save(tmp_path / "input.png")
    grid = _centers(2, 3)
    grid[0, 0, 0] = np.nan
    for policy in ("return", "original"):
        output_dir = tmp_path / policy
        config = RenderConfig(on_error=policy)
        result = render_file(source, grid, config=config, output_dir=output_dir)
        assert result.status == Status.INVALID_TRANSFORM
        assert result.reason_code == "non_finite_grid"
        assert result.source_mask is result.target_mask is result.valid_mask is None
        assert result.used_fallback == (policy == "original")
        if policy == "original":
            np.testing.assert_array_equal(result.image, _rgb(UPRIGHT[1]))
        else:
            assert result.image is None
        record = json.loads((output_dir / "metadata.json").read_text())
        assert record["status"] == "invalid_transform"
        assert record["used_fallback"] == (policy == "original")
        assert set(record["artifacts"]) == (
            {"metadata", "image"} if policy == "original" else {"metadata"}
        )
