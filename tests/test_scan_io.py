"""Scan source decoding (EXIF, draft decode, ICC) and atomic image/record writes."""

import hashlib
import io
import os

import numpy as np
import pytest
from PIL import Image, JpegImagePlugin

from flattener import FlattenerError, RenderConfig, Status, load_image
from flattener.scan import io as scan_io
from flattener.scan import records as rec

ICC = b"\x00\x00\x02\x0cfake-icc-profile-bytes" * 8


def smooth_rgb(height, width):
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    red = 128 + 100 * np.sin(x / 37.0) * np.cos(y / 53.0)
    green = 255 * x / width
    blue = 255 * y / height
    return np.clip(np.stack((red, green, blue), axis=-1), 0, 255).astype(np.uint8)


def write_jpeg(path, pixels, *, orientation=None, focal=None, icc=None, quality=95):
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
    if focal is not None:
        exif.get_ifd(0x8769)[0xA405] = focal
    options = {"quality": quality, "exif": exif.tobytes()}
    if icc is not None:
        options["icc_profile"] = icc
    Image.fromarray(pixels).save(path, format="JPEG", **options)
    return path


def test_small_jpeg_goes_through_load_image(tmp_path):
    path = write_jpeg(tmp_path / "photo.jpg", smooth_rgb(60, 80), orientation=6, focal=26, icc=ICC)
    source = scan_io.read_source(path)
    assert np.array_equal(source.image, load_image(path))
    assert source.image.shape == (80, 60, 3)
    info = source.info
    assert (info.size, info.full_size, info.scale, info.format) == ((80, 60), (80, 60), 1, "JPEG")
    assert (info.exif_orientation, info.focal_length_35mm) == (6, 26.0)
    assert info.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert source.icc_profile == ICC and info.icc_sha256 == hashlib.sha256(ICC).hexdigest()
    assert source.reasons == () and source.path == str(path)


def test_large_jpeg_is_draft_decoded(tmp_path):
    pixels = smooth_rgb(480, 640)
    path = write_jpeg(tmp_path / "large.jpg", pixels, orientation=6, icc=ICC)
    config = RenderConfig(max_input_pixels=100_000)
    with pytest.raises(FlattenerError) as caught:
        load_image(path, config=config)
    assert caught.value.reason_code == "size_limit_exceeded"
    source = scan_io.read_source(path, config=config)
    info = source.info
    assert source.image.shape == (320, 240, 3)  # half size, turned by EXIF 6
    assert (info.size, info.full_size, info.scale) == ((320, 240), (640, 480), 2)
    assert info.downscaled and source.icc_profile == ICC
    (reason,) = source.reasons
    assert (reason.code, reason.severity, reason.status) == (
        "source_downscaled",
        rec.Severity.WARNING,
        Status.SUCCESS,
    )
    assert reason.detail == {"scale": 2, "full_size": [640, 480]}
    full = load_image(path)  # default limits: the full decode
    box = full.reshape(320, 2, 240, 2, 3).mean(axis=(1, 3))
    assert np.abs(source.image.astype(np.float64) - box).mean() < 2.0


def test_draft_scale_is_the_smallest_that_fits(tmp_path):
    path = write_jpeg(tmp_path / "large.jpg", smooth_rgb(480, 640))
    source = scan_io.read_source(path, config=RenderConfig(max_input_pixels=20_000))
    assert source.info.scale == 4 and source.image.shape == (120, 160, 3)
    wide = write_jpeg(tmp_path / "wide.jpg", smooth_rgb(101, 1001))
    source = scan_io.read_source(wide, config=RenderConfig(max_side=600))
    assert source.info.scale == 2 and source.image.shape == (51, 501, 3)
    assert source.info.full_size == (101, 1001)
    assert scan_io.draft_scale(640, 480, RenderConfig(max_input_pixels=4_800)) == 8
    assert scan_io.draft_scale(640, 480, RenderConfig(max_input_pixels=4_799)) is None
    assert scan_io.draft_scale(640, 480) == 1


def test_files_above_the_limits_otherwise_fail_as_before(tmp_path):
    config = RenderConfig(max_input_pixels=1_000)
    jpeg = write_jpeg(tmp_path / "large.jpg", smooth_rgb(480, 640))
    png = tmp_path / "large.png"
    Image.fromarray(smooth_rgb(480, 640)).save(png)
    for path, draft in ((jpeg, False), (png, True), (jpeg, True)):
        with pytest.raises(FlattenerError) as caught:
            scan_io.read_source(path, config=config, draft=draft)
        assert (caught.value.status, caught.value.reason_code) == (
            Status.INPUT_ERROR,
            "size_limit_exceeded",
        )
    garbage = tmp_path / "garbage.jpg"
    garbage.write_bytes(b"not an image")
    for path in (garbage, tmp_path / "missing.jpg"):
        with pytest.raises(FlattenerError) as caught:
            scan_io.read_source(path)
        assert caught.value.reason_code == "decode_failed"


def test_multi_picture_jpeg_decodes_its_primary_image(tmp_path):
    primary = smooth_rgb(480, 640)
    gain_map = np.full((120, 160, 3), 200, np.uint8)
    exif = Image.Exif()
    exif[0x0112] = 6
    path = tmp_path / "hdr.jpg"
    Image.fromarray(primary).save(
        path,
        format="MPO",
        save_all=True,
        append_images=[Image.fromarray(gain_map)],
        exif=exif.tobytes(),
        quality=95,
    )
    with pytest.raises(FlattenerError) as caught:
        load_image(path)
    assert caught.value.reason_code == "unsupported_format"
    source = scan_io.read_source(path)
    assert source.info.format == "MPO" and source.info.scale == 1
    assert source.image.shape == (640, 480, 3)
    turned = np.rot90(primary, k=-1)  # EXIF 6: turn clockwise
    assert np.abs(source.image.astype(np.float64) - turned).mean() < 2.0
    small = scan_io.read_source(path, config=RenderConfig(max_input_pixels=100_000))
    assert small.info.scale == 2 and small.image.shape == (320, 240, 3)
    with pytest.raises(FlattenerError) as caught:
        scan_io.read_source(path, config=RenderConfig(max_input_pixels=100_000), draft=False)
    assert caught.value.reason_code == "size_limit_exceeded"


def test_png_source_keeps_icc(tmp_path):
    path = tmp_path / "scan.png"
    pixels = smooth_rgb(30, 40)
    Image.fromarray(pixels).save(path, icc_profile=ICC)
    source = scan_io.read_source(path)
    assert np.array_equal(source.image, pixels) and source.icc_profile == ICC
    assert source.info.format == "PNG" and source.info.exif_orientation is None


def test_pyramid_levels_cover_the_full_extent(tmp_path):
    path = tmp_path / "level.png"
    Image.fromarray(np.full((101, 203, 3), 77, np.uint8)).save(path)
    source = scan_io.read_source(path)
    assert source.level(1) is source.image
    assert source.level(2).shape == (51, 102, 3) and (source.level(2) == 77).all()
    assert source.level(8).shape == (13, 25, 3)
    assert source.level(2) is source.level(2)
    with pytest.raises(ValueError):
        source.level(3)


def test_jpeg_output_is_q95_444_with_icc_and_dpi(tmp_path):
    pixels = smooth_rgb(64, 96)
    path = scan_io.save_image(tmp_path / "out" / "page.jpg", pixels, icc_profile=ICC, dpi=300)
    with Image.open(path) as saved:
        assert saved.format == "JPEG" and JpegImagePlugin.get_sampling(saved) == 0
        assert saved.info["icc_profile"] == ICC
        assert saved.info["dpi"] == pytest.approx((300, 300))
        decoded = np.asarray(saved.convert("RGB"), dtype=np.float64)
    assert np.abs(decoded - pixels).mean() < 1.5
    reference = io.BytesIO()
    Image.fromarray(pixels).save(reference, format="JPEG", quality=95, subsampling=0)
    with Image.open(path) as saved, Image.open(reference) as expected:
        assert saved.quantization == expected.quantization


def test_png_output_is_lossless(tmp_path):
    pixels = smooth_rgb(33, 47)
    path = scan_io.save_image(tmp_path / "page.png", pixels, icc_profile=ICC)
    with Image.open(path) as saved:
        assert saved.format == "PNG" and saved.info["icc_profile"] == ICC
        assert np.array_equal(np.asarray(saved), pixels)
    with pytest.raises(FlattenerError) as caught:
        scan_io.save_image(tmp_path / "page.tif", pixels)
    assert caught.value.reason_code == "unsupported_format"
    with pytest.raises(FlattenerError) as caught:
        scan_io.save_image(tmp_path / "bad.png", pixels.astype(np.float32))
    assert caught.value.reason_code == "invalid_result"


def temporary_files(directory):
    return sorted(path.name for path in directory.iterdir() if path.name.startswith("."))


def test_save_image_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "page.png"
    first = smooth_rgb(20, 30)
    scan_io.save_image(path, first)
    with pytest.raises(FlattenerError) as caught:
        scan_io.save_image(path, first[::-1].copy())
    assert (caught.value.status, caught.value.reason_code) == (Status.IO_ERROR, "output_exists")

    original_save = Image.Image.save

    def failing_save(self, fp, *args, **kwargs):
        fp.write(b"partial bytes")
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", failing_save)
    with pytest.raises(FlattenerError) as caught:
        scan_io.save_image(path, first[::-1].copy(), overwrite=True)
    assert caught.value.reason_code == "save_failed"
    monkeypatch.setattr(Image.Image, "save", original_save)
    with Image.open(path) as kept:
        assert np.array_equal(np.asarray(kept), first)
    assert temporary_files(tmp_path) == []
    scan_io.save_image(path, first[::-1].copy(), overwrite=True)
    with Image.open(path) as replaced:
        assert np.array_equal(np.asarray(replaced), first[::-1])


def make_result(image):
    reasons = [rec.reason("page_fills_frame")]
    return rec.ScanResult(
        status=Status.SUCCESS,
        outcome=rec.decide_outcome("dewarp", reasons),
        reasons=reasons,
        image=image,
        source=rec.SourceInfo(size=image.shape[:2], format="JPEG"),
        timings={"read": 0.01},
        qa={"coverage": np.float32(0.99)},
        input="photo.jpg",
    )


def test_save_scan_writes_image_and_record(tmp_path):
    result = make_result(smooth_rgb(40, 30))
    artifacts = scan_io.save_scan(result, tmp_path, "photo", icc_profile=ICC)
    assert artifacts == {
        "image": str(tmp_path / "photo.jpg"),
        "record": str(tmp_path / "photo.scan.json"),
    }
    assert result.artifacts == artifacts and result.timings["save"] >= 0
    record = scan_io.read_scan_record(tmp_path / "photo.scan.json")
    assert record.to_dict() == result.to_dict()
    with Image.open(tmp_path / "photo.jpg") as saved:
        assert JpegImagePlugin.get_sampling(saved) == 0 and saved.info["icc_profile"] == ICC
    png = scan_io.save_scan(make_result(smooth_rgb(8, 8)), tmp_path, "bench", fmt="png")
    assert png["image"].endswith("bench.png") and temporary_files(tmp_path) == []
    with pytest.raises(FlattenerError) as caught:
        scan_io.save_scan(make_result(smooth_rgb(8, 8)), tmp_path, "photo")
    assert caught.value.reason_code == "output_exists"
    for stem in ("", "..", "a/b"):
        with pytest.raises(FlattenerError) as caught:
            scan_io.save_scan(result, tmp_path, stem)
        assert caught.value.reason_code == "invalid_path"


def test_failed_record_rolls_back_the_image(tmp_path, monkeypatch):
    first = make_result(smooth_rgb(40, 30))
    scan_io.save_scan(first, tmp_path, "photo", fmt="png")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    real_replace = os.replace

    def failing_replace(source, target):
        if str(target).endswith(".scan.json"):
            raise OSError("no space left")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(FlattenerError) as caught:
        scan_io.save_scan(
            make_result(smooth_rgb(40, 30)[::-1].copy()),
            tmp_path,
            "photo",
            fmt="png",
            overwrite=True,
        )
    assert (caught.value.status, caught.value.reason_code) == (Status.IO_ERROR, "save_failed")
    assert caught.value.metadata["save_transaction"]["rolled_back"] is True
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before

    real_link = os.link

    def failing_link(source, target):
        if str(target).endswith(".scan.json"):
            raise OSError("link failed")
        return real_link(source, target)

    monkeypatch.setattr(os, "link", failing_link)
    fresh = tmp_path / "fresh"
    with pytest.raises(FlattenerError):
        scan_io.save_scan(make_result(smooth_rgb(8, 8)), fresh, "photo")
    assert list(fresh.iterdir()) == []


def test_unrestored_previous_image_is_kept(tmp_path, monkeypatch):
    scan_io.save_scan(make_result(smooth_rgb(12, 12)), tmp_path, "photo", fmt="png")
    old_bytes = (tmp_path / "photo.png").read_bytes()
    real_replace = os.replace

    def failing_replace(source, target):
        if str(target).endswith(".scan.json") or str(source).endswith(".old"):
            raise OSError("device error")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(FlattenerError) as caught:
        scan_io.save_scan(
            make_result(smooth_rgb(12, 12)[::-1].copy()),
            tmp_path,
            "photo",
            fmt="png",
            overwrite=True,
        )
    transaction = caught.value.metadata["save_transaction"]
    assert transaction["rolled_back"] is False
    (kept,) = [path for path in tmp_path.iterdir() if path.name.endswith(".old")]
    assert kept.read_bytes() == old_bytes
    assert any(str(kept) in error for error in transaction["errors"])


def test_record_without_image_removes_a_stale_image(tmp_path):
    scan_io.save_scan(make_result(smooth_rgb(8, 8)), tmp_path, "photo")
    failure = rec.ScanResult(
        status=Status.INPUT_ERROR,
        outcome=rec.Outcome.FAILED,
        reason_code="decode_failed",
        message="Could not decode image",
    )
    artifacts = scan_io.save_scan(failure, tmp_path, "photo", overwrite=True)
    assert artifacts == {"record": str(tmp_path / "photo.scan.json")}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["photo.scan.json"]
    record = scan_io.read_scan_record(tmp_path / "photo.scan.json")
    assert (record.status, record.outcome, record.image) == (Status.INPUT_ERROR, "failed", None)
    (tmp_path / "broken.scan.json").write_text("{", encoding="utf-8")
    with pytest.raises(FlattenerError) as caught:
        scan_io.read_scan_record(tmp_path / "broken.scan.json")
    assert caught.value.reason_code == "invalid_scan_record"
