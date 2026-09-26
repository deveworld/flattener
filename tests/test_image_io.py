"""Image adapter checks using hand-written orientations and pixel expectations."""

import json
import math
import os
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flattener import (
    FlattenerError,
    RenderConfig,
    Status,
    image_io,
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


def test_rgb_jpeg_is_supported_without_claiming_lossless_roundtrip(tmp_path):
    path = tmp_path / "color.jpg"
    Image.new("RGB", (3, 2), color=(240, 90, 40)).save(path)
    actual = load_image(path)
    assert actual.shape == (2, 3, 3)
    assert actual.dtype == np.uint8
    assert np.max(np.abs(actual.astype(int) - np.array([240, 90, 40]))) <= 2


@pytest.mark.parametrize(
    ("mode", "extension"),
    [("RGBA", "png"), ("P", "png"), ("I;16", "png"), ("CMYK", "jpg")],
)
def test_unsupported_modes_are_rejected(tmp_path, mode, extension):
    path = tmp_path / f"unsupported.{extension}"
    Image.new(mode, (3, 2)).save(path)
    with pytest.raises(FlattenerError) as caught:
        load_image(path)
    assert caught.value.status == Status.INPUT_ERROR
    assert caught.value.reason_code == "unsupported_mode"


def test_png_color_key_transparency_is_rejected(tmp_path, rgb):
    path = tmp_path / "transparent.png"
    Image.fromarray(rgb).save(path, transparency=(30, 11, 7))
    with pytest.raises(FlattenerError, match="without transparency"):
        load_image(path)


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


def test_animated_png_is_rejected(tmp_path):
    path = tmp_path / "animated.png"
    first = Image.new("RGB", (3, 2), color="red")
    second = Image.new("RGB", (3, 2), color="blue")
    first.save(path, save_all=True, append_images=[second], duration=30, loop=0)
    with pytest.raises(FlattenerError) as caught:
        load_image(path)
    assert caught.value.reason_code == "unsupported_frames"


def test_other_file_formats_are_rejected(tmp_path):
    path = tmp_path / "image.gif"
    Image.new("RGB", (3, 2)).save(path)
    with pytest.raises(FlattenerError) as caught:
        load_image(path)
    assert caught.value.reason_code == "unsupported_format"


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


def test_unexpected_decode_errors_are_not_hidden(source, monkeypatch):
    def out_of_memory(*args, **kwargs):
        raise MemoryError("deliberate allocation failure")

    monkeypatch.setattr(image_io.ImageOps, "exif_transpose", out_of_memory)
    with pytest.raises(MemoryError):
        load_image(source)


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


def test_decode_failure_cannot_fallback_but_can_save_metadata(tmp_path, grid):
    result = render_file(
        tmp_path / "missing.png",
        grid,
        config=RenderConfig(on_error="original"),
        output_dir=tmp_path / "failed",
    )
    assert result.status == Status.INPUT_ERROR
    assert result.image is None
    assert not result.used_fallback
    assert result.metadata["timing_scope"] == "file"
    assert result.metadata["timings"]["render"] is None
    assert result.metadata["actual_output_size"] is None
    assert set(result.metadata["artifacts"]) == {"metadata"}


def test_render_failure_raises_with_file_scope_and_correct_stage(source, grid):
    grid[0, 0, 0] = np.nan
    with pytest.raises(FlattenerError) as caught:
        render_file(source, grid)
    assert caught.value.metadata["timing_scope"] == "file"
    assert caught.value.metadata["failure_stage"] == "process_coordinates"
    assert caught.value.metadata["timings"]["prepare_input"] >= 0


@pytest.mark.parametrize("policy", ["return", "original"])
def test_save_failure_keeps_processing_status_and_obeys_policy(tmp_path, source, rgb, grid, policy):
    output_dir = tmp_path / "output"
    render_file(source, grid, output_dir=output_dir)
    result = render_file(source, grid, output_dir=output_dir, config=RenderConfig(on_error=policy))
    assert result.status == Status.IO_ERROR
    assert result.reason_code == "output_exists"
    assert result.metadata["processing_status"] == "success"
    assert result.metadata["failure_stage"] == "save"
    assert result.used_fallback == (policy == "original")
    if policy == "original":
        np.testing.assert_array_equal(result.image, rgb)
    else:
        assert result.image is None


def test_save_failure_retains_prior_transform_failure(tmp_path, source, grid):
    output_dir = tmp_path / "output"
    render_file(source, grid, output_dir=output_dir)
    grid[0, 0, 0] = np.nan
    result = render_file(
        source, grid, output_dir=output_dir, config=RenderConfig(on_error="return")
    )
    assert result.status == Status.IO_ERROR
    assert result.metadata["processing_status"] == "invalid_transform"
    assert result.metadata["processing_reason_code"] == "non_finite_grid"


def test_invalid_failure_policy_always_raises(source, grid):
    with pytest.raises(FlattenerError) as caught:
        render_file(source, grid, config=RenderConfig(on_error="invalid"))
    assert caught.value.status == Status.INPUT_ERROR
    assert caught.value.metadata["timing_scope"] == "file"
    assert caught.value.metadata["failure_stage"] == "prepare_input"
    assert caught.value.metadata["timings"]["prepare_input"] >= 0


def test_failed_new_output_commit_rolls_back_created_files(tmp_path, source, grid, monkeypatch):
    result = render_file(source, grid)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    marker = output_dir / "unrelated.txt"
    marker.write_text("preserve")
    actual_link = os.link

    def fail_metadata(source_path, target_path, *args, **kwargs):
        if Path(target_path).name == "metadata.json":
            raise OSError("injected commit failure")
        return actual_link(source_path, target_path, *args, **kwargs)

    monkeypatch.setattr(image_io.os, "link", fail_metadata)
    with pytest.raises(FlattenerError) as caught:
        save_result(result, output_dir)
    assert caught.value.status == Status.IO_ERROR
    assert caught.value.metadata["save_transaction"]["rolled_back"] is True
    assert list(output_dir.iterdir()) == [marker]
    assert marker.read_text() == "preserve"


def test_failed_overwrite_restores_every_original_artifact(tmp_path, source, grid, monkeypatch):
    result = render_file(source, grid)
    output_dir = tmp_path / "output"
    save_result(result, output_dir)
    before = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    result.image[:] = 0
    actual_replace = os.replace

    def fail_one_commit(source_path, target_path, *args, **kwargs):
        if Path(source_path).name == "valid_mask.png":
            raise OSError("injected overwrite failure")
        return actual_replace(source_path, target_path, *args, **kwargs)

    monkeypatch.setattr(image_io.os, "replace", fail_one_commit)
    with pytest.raises(FlattenerError) as caught:
        save_result(result, output_dir, overwrite=True)
    assert caught.value.metadata["save_transaction"]["rolled_back"] is True
    after = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    assert after == before


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, MemoryError])
def test_interrupted_overwrite_restores_files_and_reraises(
    tmp_path, source, grid, monkeypatch, exception_type
):
    result = render_file(source, grid)
    output_dir = tmp_path / "output"
    save_result(result, output_dir)
    before = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    result.image[:] = 0
    actual_replace = os.replace
    interrupt = exception_type("injected interruption")

    def interrupt_commit(source_path, target_path, *args, **kwargs):
        if Path(source_path).name == "valid_mask.png":
            raise interrupt
        return actual_replace(source_path, target_path, *args, **kwargs)

    monkeypatch.setattr(image_io.os, "replace", interrupt_commit)
    with pytest.raises(exception_type) as caught:
        save_result(result, output_dir, overwrite=True)
    assert caught.value is interrupt
    after = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    assert after == before


def test_rollback_failure_preserves_backups_for_manual_recovery(
    tmp_path, source, grid, monkeypatch
):
    result = render_file(source, grid)
    output_dir = tmp_path / "output"
    save_result(result, output_dir)
    original_image = (output_dir / "image.png").read_bytes()
    result.image[:] = 0
    actual_replace = os.replace

    def fail_commit_and_image_recovery(source_path, target_path, *args, **kwargs):
        if Path(source_path).name in {"valid_mask.png", "previous-image.png"}:
            raise OSError("injected commit/recovery failure")
        return actual_replace(source_path, target_path, *args, **kwargs)

    monkeypatch.setattr(image_io.os, "replace", fail_commit_and_image_recovery)
    with pytest.raises(FlattenerError) as caught:
        save_result(result, output_dir, overwrite=True)
    recovery = caught.value.metadata["save_transaction"]
    assert recovery["rolled_back"] is False
    assert recovery["rollback_errors"]
    backup = Path(recovery["recovery_directory"]) / "previous-image.png"
    assert backup.read_bytes() == original_image


def test_non_json_metadata_does_not_publish_partial_artifacts(tmp_path, source, grid):
    result = render_file(source, grid)
    result.metadata["invalid"] = float("nan")
    output_dir = tmp_path / "output"
    with pytest.raises(FlattenerError) as caught:
        save_result(result, output_dir)
    assert caught.value.reason_code == "metadata_serialization_failed"
    assert list(output_dir.iterdir()) == []


def test_non_utf8_metadata_is_an_expected_io_error(tmp_path, source, grid):
    result = render_file(source, grid)
    result.metadata["invalid_text"] = "\ud800"
    output_dir = tmp_path / "output"
    with pytest.raises(FlattenerError) as caught:
        save_result(result, output_dir)
    assert caught.value.reason_code == "metadata_serialization_failed"
    assert list(output_dir.iterdir()) == []


def test_nul_in_output_path_is_an_expected_io_error(source, grid):
    result = render_file(source, grid)
    with pytest.raises(FlattenerError) as caught:
        save_result(result, "invalid\x00path")
    assert caught.value.status == Status.IO_ERROR
    assert caught.value.reason_code == "invalid_path"


def test_inconsistent_result_shapes_are_rejected_before_writing(tmp_path, source, grid):
    result = render_file(source, grid)
    result.valid_mask = np.ones((1, 3), dtype=bool)
    with pytest.raises(FlattenerError) as caught:
        save_result(result, tmp_path / "output")
    assert caught.value.reason_code == "invalid_result"
    assert not (tmp_path / "output").exists()
