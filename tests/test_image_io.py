"""Image adapter checks using hand-written orientations and pixel expectations."""

import json
import math
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


@pytest.fixture
def rgb():
    labels = np.arange(1, 7, dtype=np.uint8).reshape(2, 3)
    return np.stack((labels * 30, labels * 11, labels * 7), axis=-1)


@pytest.fixture
def grid():
    return np.array(
        [[[-2 / 3, -0.5], [0, -0.5], [2 / 3, -0.5]], [[-2 / 3, 0.5], [0, 0.5], [2 / 3, 0.5]]],
        dtype=np.float32,
    )


@pytest.fixture
def source(tmp_path, rgb):
    path = tmp_path / "input.png"
    Image.fromarray(rgb).save(path)
    return path


@pytest.mark.parametrize(
    ("orientation", "labels"),
    [
        (1, [[1, 2, 3], [4, 5, 6]]),
        (2, [[3, 2, 1], [6, 5, 4]]),
        (3, [[6, 5, 4], [3, 2, 1]]),
        (4, [[4, 5, 6], [1, 2, 3]]),
        (5, [[1, 4], [2, 5], [3, 6]]),
        (6, [[4, 1], [5, 2], [6, 3]]),
        (7, [[6, 3], [5, 2], [4, 1]]),
        (8, [[3, 6], [2, 5], [1, 4]]),
    ],
)
def test_exif_orientations_have_explicit_pixel_expectations(tmp_path, rgb, orientation, labels):
    path = tmp_path / "oriented.png"
    exif = Image.Exif()
    exif[274] = orientation
    Image.fromarray(rgb).save(path, exif=exif)
    actual = load_image(path)
    labels = np.asarray(labels, dtype=np.uint8)
    expected = np.stack((labels * 30, labels * 11, labels * 7), axis=-1)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8
    assert actual.flags.writeable


@pytest.mark.parametrize("extension", ["png", "jpg"])
def test_grayscale_is_replicated_into_rgb(tmp_path, extension):
    path = tmp_path / f"gray.{extension}"
    Image.new("L", (3, 2), color=91).save(path)
    np.testing.assert_array_equal(load_image(path), np.full((2, 3, 3), 91, np.uint8))


@pytest.mark.parametrize(
    ("bit_depth", "color_type", "row", "pillow_mode"),
    [
        (16, 2, b"\x00" + struct.pack(">HHH", 65535, 32768, 1), "RGB"),
        (2, 0, b"\x00\x80", "L"),
        (4, 0, b"\x00\x80", "L"),
    ],
)
def test_non_eight_bit_png_cannot_be_silently_converted(
    tmp_path, bit_depth, color_type, row, pillow_mode
):
    def chunk(name, payload):
        checksum = zlib.crc32(name + payload)
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", checksum)

    ihdr = struct.pack(">IIBBBBB", 1, 1, bit_depth, color_type, 0, 0, 0)
    encoded = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(row))
        + chunk(b"IEND", b"")
    )
    path = tmp_path / "unsupported_bit_depth.png"
    path.write_bytes(encoded)
    # A plain mode check would accept this file despite its original bit depth.
    with Image.open(path) as decoded:
        assert decoded.mode == pillow_mode
    with pytest.raises(FlattenerError) as caught:
        load_image(path)
    assert caught.value.reason_code == "unsupported_mode"


@pytest.mark.parametrize("data", [None, b"not an image"])
def test_missing_or_broken_input_is_input_error(tmp_path, data):
    path = tmp_path / "broken.png"
    if data is not None:
        path.write_bytes(data)
    with pytest.raises(FlattenerError) as caught:
        load_image(path)
    assert caught.value.status == Status.INPUT_ERROR
    assert caught.value.reason_code == "decode_failed"


def test_input_size_is_checked_before_decoding(source, monkeypatch):
    def no_decode(*args, **kwargs):
        raise AssertionError("Image pixels should not be decoded")

    monkeypatch.setattr(Image.Image, "load", no_decode)
    with pytest.raises(FlattenerError) as caught:
        load_image(source, config=RenderConfig(max_input_pixels=5))
    assert caught.value.reason_code == "size_limit_exceeded"


def test_success_artifacts_and_finite_timings(tmp_path, source, rgb, grid):
    output_dir = tmp_path / "result"
    result = render_file(source, grid, output_dir=output_dir)
    assert result.status == Status.SUCCESS
    np.testing.assert_array_equal(result.image, rgb)
    record = json.loads((output_dir / "metadata.json").read_text())
    assert record["status"] == "success"
    assert record["used_fallback"] is False
    assert record["artifacts"] == result.metadata["artifacts"]
    assert set(record["artifacts"]) == {
        "image",
        "source_mask",
        "target_mask",
        "valid_mask",
        "metadata",
    }
    np.testing.assert_array_equal(load_image(output_dir / "image.png"), rgb)
    for name in ("source_mask", "target_mask", "valid_mask"):
        with Image.open(output_dir / f"{name}.png") as mask:
            assert mask.mode == "L"
            np.testing.assert_array_equal(np.asarray(mask), np.full((2, 3), 255, np.uint8))
    metadata = record["metadata"]
    assert metadata["source_size"] == [2, 3]
    assert metadata["requested_output_size"] == [2, 3]
    assert metadata["actual_output_size"] == [2, 3]
    assert metadata["timing_scope"] == "file"
    assert metadata["model_id"] is None
    assert metadata["model_sha256"] is None
    for name in ("load_model", "predict"):
        assert metadata["timings"][name] is None
    for name in ("prepare_input", "process_coordinates", "render", "save", "total"):
        assert math.isfinite(metadata["timings"][name])
        assert metadata["timings"][name] >= 0


def test_orientation_is_not_applied_twice_in_render_file(tmp_path, rgb):
    path = tmp_path / "oriented.png"
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(rgb).save(path, exif=exif)
    grid = np.array(
        [[[-0.5, -2 / 3], [0.5, -2 / 3]], [[-0.5, 0], [0.5, 0]], [[-0.5, 2 / 3], [0.5, 2 / 3]]],
        dtype=np.float32,
    )
    result = render_file(path, grid, output_dir=tmp_path / "output")
    expected = np.array([[4, 1], [5, 2], [6, 3]], np.uint8)
    np.testing.assert_array_equal(result.image[..., 0], expected * 30)
    np.testing.assert_array_equal(load_image(tmp_path / "output/image.png"), result.image)


def test_no_implicit_overwrite_and_explicit_overwrite_works(tmp_path, source, grid):
    output_dir = tmp_path / "output"
    result = render_file(source, grid, output_dir=output_dir)
    original = (output_dir / "metadata.json").read_bytes()
    with pytest.raises(FlattenerError) as caught:
        save_result(result, output_dir)
    assert caught.value.reason_code == "output_exists"
    assert caught.value.status == Status.IO_ERROR
    assert (output_dir / "metadata.json").read_bytes() == original
    result.metadata["test_marker"] = "replacement"
    save_result(result, output_dir, overwrite=True)
    assert (
        json.loads((output_dir / "metadata.json").read_text())["metadata"]["test_marker"]
        == "replacement"
    )


@pytest.mark.parametrize("policy", ["return", "original"])
def test_failure_status_is_saved_and_fallback_is_marked(tmp_path, source, rgb, grid, policy):
    grid[0, 0, 0] = np.nan
    output_dir = tmp_path / "failed"
    result = render_file(source, grid, config=RenderConfig(on_error=policy), output_dir=output_dir)
    assert result.status == Status.INVALID_TRANSFORM
    assert result.reason_code == "non_finite_grid"
    assert result.source_mask is result.target_mask is result.valid_mask is None
    assert result.used_fallback == (policy == "original")
    if policy == "original":
        np.testing.assert_array_equal(result.image, rgb)
    else:
        assert result.image is None
    record = json.loads((output_dir / "metadata.json").read_text())
    assert record["status"] == "invalid_transform"
    assert record["used_fallback"] == (policy == "original")
    assert set(record["artifacts"]) == (
        {"metadata", "image"} if policy == "original" else {"metadata"}
    )
