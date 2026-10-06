"""Scan source decoding (EXIF, draft decode, ICC) and atomic image/record writes."""

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


def test_jpegs_decode_whole_or_draft_with_exif_and_icc(tmp_path):
    path = write_jpeg(tmp_path / "photo.jpg", smooth_rgb(60, 80), orientation=6, focal=26, icc=ICC)
    source = scan_io.read_source(path)
    assert np.array_equal(source.image, load_image(path))
    assert source.image.shape == (80, 60, 3)
    info = source.info
    assert (info.size, info.full_size, info.scale, info.format) == ((80, 60), (80, 60), 1, "JPEG")
    assert (info.exif_orientation, info.focal_length_35mm) == (6, 26.0)
    assert source.icc_profile == ICC
    assert source.reasons == () and source.path == str(path)
    # A photo over the input limit is draft-decoded at a reduced scale.
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


def temporary_files(directory):
    return sorted(path.name for path in directory.iterdir() if path.name.startswith("."))


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
    for stem in ("..", "a/b"):
        with pytest.raises(FlattenerError) as caught:
            scan_io.save_scan(result, tmp_path, stem)
        assert caught.value.reason_code == "invalid_path"
    # A record without an image removes the image of an earlier scan.
    failure = rec.ScanResult(
        status=Status.INPUT_ERROR,
        outcome=rec.Outcome.FAILED,
        reason_code="decode_failed",
        message="Could not decode image",
    )
    artifacts = scan_io.save_scan(failure, tmp_path, "photo", overwrite=True)
    assert artifacts == {"record": str(tmp_path / "photo.scan.json")}
    assert not (tmp_path / "photo.jpg").exists()
    record = scan_io.read_scan_record(tmp_path / "photo.scan.json")
    assert (record.status, record.outcome, record.image) == (Status.INPUT_ERROR, "failed", None)
    (tmp_path / "broken.scan.json").write_text("{", encoding="utf-8")
    with pytest.raises(FlattenerError) as caught:
        scan_io.read_scan_record(tmp_path / "broken.scan.json")
    assert caught.value.reason_code == "invalid_scan_record"


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

    real_place = scan_io._place_new

    def failing_place(staged, target):
        if str(target).endswith(".scan.json"):
            raise OSError("place failed")
        return real_place(staged, target)

    monkeypatch.setattr(scan_io, "_place_new", failing_place)
    fresh = tmp_path / "fresh"
    with pytest.raises(FlattenerError):
        scan_io.save_scan(make_result(smooth_rgb(8, 8)), fresh, "photo")
    assert list(fresh.iterdir()) == []


def test_saving_works_without_hard_links_and_follows_the_umask(tmp_path, monkeypatch):
    previous = os.umask(0o022)
    try:
        artifacts = scan_io.save_scan(make_result(smooth_rgb(8, 8)), tmp_path / "linked", "photo")
        page = scan_io.save_image(tmp_path / "linked" / "page.jpg", smooth_rgb(8, 8))
    finally:
        os.umask(previous)
    for path in [*artifacts.values(), page]:
        assert os.stat(path).st_mode & 0o777 == 0o644

    def no_links(source, target):
        raise PermissionError(1, "Operation not permitted", str(source), None, str(target))

    monkeypatch.setattr(os, "link", no_links)  # as on FAT and exFAT drives
    first = make_result(smooth_rgb(40, 30))
    assert scan_io.save_scan(first, tmp_path, "photo")["image"] == str(tmp_path / "photo.jpg")
    with pytest.raises(FlattenerError) as caught:
        scan_io.save_scan(make_result(smooth_rgb(8, 8)), tmp_path, "photo")
    assert caught.value.reason_code == "output_exists"
    scan_io.save_scan(make_result(smooth_rgb(8, 8)), tmp_path, "photo", overwrite=True)
    with Image.open(tmp_path / "photo.jpg") as saved:
        assert saved.size == (8, 8)
    page = scan_io.save_image(tmp_path / "page.png", smooth_rgb(8, 8))
    assert np.array_equal(np.asarray(Image.open(page)), smooth_rgb(8, 8))
    assert temporary_files(tmp_path) == []
