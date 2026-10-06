"""Scan input and output files.

``read_source`` decodes one JPEG or PNG for a scan:

* A file within the ``RenderConfig`` input limits (16 MP, 8192 px a side) goes through
  ``image_io.load_image``, which applies the EXIF orientation once.
* A larger JPEG is decoded in Pillow's draft mode at the smallest reduction of 1/2, 1/4
  or 1/8 that fits the limits (libjpeg scales during the DCT, so this is fast) and then
  turned by its EXIF orientation once. Decoded pixel i covers full-resolution pixels
  ``[scale i, scale (i + 1))``. The source records ``scale`` and the
  ``source_downscaled`` reason. Other files above the limits fail as in ``load_image``,
  and Pillow's decompression-bomb limit (about 179 MP) still applies.
* A JPEG with Multi-Picture Format data (Pillow's ``MPO``, as written by phones that
  add a depth map or an HDR gain map) decodes its primary image the same way;
  ``load_image`` refuses such files as multi-frame.
* The ICC profile bytes of the file are kept (``icc_profile``) for the output file;
  pixel values are used as stored.
* ``SourceImage.info`` records the sizes, the EXIF orientation and the
  35 mm equivalent focal length.
* ``SourceImage.level(f)`` is the image reduced by f (2, 4 or 8) with a box filter over
  the full extent, so flattener coordinates are the same on every level.

``save_scan`` writes ``<stem>.jpg`` (or ``.png``) and ``<stem>.scan.json``. JPEG is
quality 95 with 4:4:4 chroma (``subsampling=0``); PNG is lossless. The source's ICC
profile and a known DPI can be embedded. Every file is written to a temporary file in
the target directory, flushed to disk and moved into place, so a reader sees either the
old file or the new one. Files get the permissions of the umask. Without ``overwrite``
an existing target is an ``output_exists`` error, and a new file is moved into place
only while its name is still free: by a hard link, or on file systems without hard
links (FAT, exFAT, some network shares) by an exclusive create that the file then
replaces. The image is committed first and rolled back if the record cannot be
written. A result without an image removes an earlier image of the same name (with
``overwrite``), so the record and the files agree. Use one writer per output directory.
"""

from __future__ import annotations

import io
import json
import os
import secrets
import shutil
import tempfile
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from ..contracts import FlattenerError, RenderConfig, Status
from ..image_io import _path, load_image
from ..validation import validate_config, validate_image
from .records import Reason, ScanResult, SourceInfo, reason

FORMATS = {"jpeg": ".jpg", "png": ".png"}
FORMAT_NAMES = {"jpeg": "jpeg", "jpg": "jpeg", "png": "png"}
JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0  # Pillow: 0 is 4:4:4 (no chroma subsampling)
PNG_COMPRESS_LEVEL = 6
RECORD_SUFFIX = ".scan.json"
DRAFT_SCALES = (2, 4, 8)
JPEG_FORMATS = ("JPEG", "MPO")  # MPO: a JPEG with Multi-Picture data (e.g. an HDR gain map)
LEVELS = (1, 2, 4, 8)
_TRANSPOSED_ORIENTATIONS = frozenset({5, 6, 7, 8})
_EXIF_ORIENTATION = 0x0112
_EXIF_IFD = 0x8769
_FOCAL_35MM = 0xA405


def _input_error(reason_code: str, message: str) -> FlattenerError:
    return FlattenerError(Status.INPUT_ERROR, reason_code, message)


def _io_error(reason_code: str, message: str) -> FlattenerError:
    return FlattenerError(Status.IO_ERROR, reason_code, message)


@dataclass(eq=False)
class SourceImage:
    """A decoded source: EXIF-oriented RGB pixels, their record and the ICC bytes."""

    image: NDArray[np.uint8]
    info: SourceInfo
    icc_profile: bytes | None = None
    path: str | None = None
    _levels: dict[int, NDArray[np.uint8]] = field(default_factory=dict, repr=False)

    @property
    def reasons(self) -> tuple[Reason, ...]:
        if not self.info.downscaled:
            return ()
        return (
            reason("source_downscaled", scale=self.info.scale, full_size=list(self.info.full_size)),
        )

    def level(self, factor: int) -> NDArray[np.uint8]:
        """The image reduced by ``factor`` (``LEVELS``) with a box filter; cached.

        The level covers the full image extent at ``round(size / factor)`` pixels (at
        least 1), so a flattener coordinate points at the same place on every level.
        """
        if factor not in LEVELS:
            raise ValueError(f"factor must be one of {LEVELS}")
        if factor == 1:
            return self.image
        if factor not in self._levels:
            height, width = self.image.shape[:2]
            size = (max(1, int(width / factor + 0.5)), max(1, int(height / factor + 0.5)))
            with Image.fromarray(self.image) as full:
                with full.resize(size, Image.Resampling.BOX) as reduced:
                    self._levels[factor] = np.asarray(reduced, dtype=np.uint8).copy()
        return self._levels[factor]


@dataclass(frozen=True, slots=True)
class _Header:
    format: str | None
    width: int
    height: int
    icc_profile: bytes | None
    orientation: int | None
    focal_35mm: float | None


def _exif_fields(decoded: Image.Image) -> tuple[int | None, float | None]:
    try:
        exif = decoded.getexif()
        orientation = exif.get(_EXIF_ORIENTATION)
        focal = exif.get_ifd(_EXIF_IFD).get(_FOCAL_35MM)
    except (AttributeError, KeyError, OSError, SyntaxError, TypeError, ValueError):
        return None, None
    orientation = orientation if isinstance(orientation, int) and 1 <= orientation <= 8 else None
    if isinstance(focal, (int, float)) and not isinstance(focal, bool) and focal > 0:
        return orientation, float(focal)
    return orientation, None


def _probe(data: bytes) -> _Header | None:
    """Format, stored size, ICC bytes and EXIF fields without decoding the pixels;
    ``None`` when Pillow cannot identify the file."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as decoded:
                orientation, focal = _exif_fields(decoded)
                icc = decoded.info.get("icc_profile")
                return _Header(
                    decoded.format,
                    decoded.width,
                    decoded.height,
                    icc if isinstance(icc, bytes) and icc else None,
                    orientation,
                    focal,
                )
    except Image.DecompressionBombError as exc:
        raise _input_error(
            "size_limit_exceeded", "Pillow rejected the image dimensions before decoding."
        ) from exc
    except (OSError, SyntaxError, ValueError):
        return None


def _fits(width: int, height: int, config: RenderConfig) -> bool:
    return width * height <= config.max_input_pixels and max(width, height) <= config.max_side


def draft_scale(width: int, height: int, config: RenderConfig | None = None) -> int | None:
    """The smallest draft reduction whose decoded size fits the input limits, or
    ``None`` (1 when the full size fits)."""
    cfg = validate_config(config)
    for scale in (1, *DRAFT_SCALES):
        if _fits(-(-width // scale), -(-height // scale), cfg):
            return scale
    return None


def _decode_jpeg(
    data: bytes, header: _Header, config: RenderConfig, *, draft: bool
) -> tuple[NDArray, int]:
    """Decode the primary image of a JPEG (or MPO) file, in draft mode when it is above
    the input limits; returns the EXIF-oriented RGB pixels and the draft scale."""
    scale = draft_scale(header.width, header.height, config) if draft else None
    if not draft and _fits(header.width, header.height, config):
        scale = 1
    if scale is None:
        raise _input_error(
            "size_limit_exceeded",
            f"input ({header.height}, {header.width}) exceeds the configured size limits"
            + (" even at 1/8 draft scale" if draft else ""),
        )
    expected = (-(-header.width // scale), -(-header.height // scale))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as decoded:
                if decoded.format not in JPEG_FORMATS:
                    raise _input_error("unsupported_format", "Expected a JPEG file.")
                if decoded.format == "JPEG" and getattr(decoded, "n_frames", 1) != 1:
                    raise _input_error(
                        "unsupported_frames", "Multi-frame images are not supported."
                    )
                if decoded.mode not in {"RGB", "L"}:
                    raise _input_error(
                        "unsupported_mode", "Only 8-bit RGB or grayscale images are supported."
                    )
                if scale > 1:
                    decoded.draft(decoded.mode, (header.width // scale, header.height // scale))
                if decoded.size != expected:
                    raise _input_error(
                        "decode_failed",
                        f"draft decode gave {decoded.size}, expected {expected}",
                    )
                oriented = ImageOps.exif_transpose(decoded)
                try:
                    rgb = oriented.convert("RGB")
                    try:
                        image = np.array(rgb, dtype=np.uint8, copy=True)
                    finally:
                        rgb.close()
                finally:
                    oriented.close()
    except Image.DecompressionBombError as exc:
        raise _input_error(
            "size_limit_exceeded", "Pillow rejected the image dimensions before decoding."
        ) from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise _input_error("decode_failed", f"Could not decode image: {exc}") from exc
    validate_image(image, config)
    return image, scale


def read_source(
    path: str | os.PathLike[str], *, config: RenderConfig | None = None, draft: bool = True
) -> SourceImage:
    """Decode one JPEG or PNG for a scan (see the module docstring).

    ``draft=False`` keeps ``load_image``'s behaviour for large files (a
    ``size_limit_exceeded`` input error).
    """
    cfg = validate_config(config)
    source = _path(path)
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise _input_error("decode_failed", f"Could not decode image: {exc}") from exc
    header = _probe(data)
    scale = 1
    if header is not None and (
        header.format == "MPO"
        or (draft and header.format == "JPEG" and not _fits(header.width, header.height, cfg))
    ):
        image, scale = _decode_jpeg(data, header, cfg, draft=draft)
    else:
        image = load_image(source, config=cfg)
    height, width = image.shape[:2]
    full_size = (height, width)
    if header is not None and scale > 1:
        turned = header.orientation in _TRANSPOSED_ORIENTATIONS
        full_size = (header.width, header.height) if turned else (header.height, header.width)
    icc = None if header is None else header.icc_profile
    info = SourceInfo(
        size=(height, width),
        full_size=full_size,
        scale=scale,
        format=None if header is None else header.format,
        exif_orientation=None if header is None else header.orientation,
        focal_length_35mm=None if header is None else header.focal_35mm,
    )
    return SourceImage(image=image, info=info, icc_profile=icc, path=str(source))


def image_format(name: str) -> str:
    """Canonical format name (``jpeg`` or ``png``) of ``jpg``, ``jpeg`` or ``png``."""
    key = name.lower().lstrip(".") if isinstance(name, str) else None
    if key not in FORMAT_NAMES:
        raise _input_error("unsupported_format", "Output format must be jpeg (jpg) or png.")
    return FORMAT_NAMES[key]


def _check_image(image: Any) -> NDArray[np.uint8]:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 1
    ):
        raise _input_error("invalid_result", "Output image must be uint8 (height, width, 3).")
    return np.ascontiguousarray(image)


def _save_options(
    fmt: str, quality: int, icc_profile: bytes | None, dpi: float | tuple[float, float] | None
) -> dict[str, Any]:
    if fmt == "jpeg":
        if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
            raise _input_error("invalid_output_option", "JPEG quality must be in 1..100.")
        options: dict[str, Any] = {
            "format": "JPEG",
            "quality": quality,
            "subsampling": JPEG_SUBSAMPLING,
        }
    else:
        options = {"format": "PNG", "compress_level": PNG_COMPRESS_LEVEL}
    if icc_profile is not None:
        if not isinstance(icc_profile, bytes):
            raise _input_error("invalid_output_option", "icc_profile must be bytes.")
        options["icc_profile"] = icc_profile
    if dpi is not None:
        pair = (dpi, dpi) if isinstance(dpi, (int, float)) else tuple(dpi)
        if len(pair) != 2 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < float("inf")
            for v in pair
        ):
            raise _input_error("invalid_output_option", "dpi must be positive.")
        options["dpi"] = (float(pair[0]), float(pair[1]))
    return options


def _encoder(image: NDArray[np.uint8], options: dict[str, Any]) -> Callable[[BinaryIO], None]:
    def write(stream: BinaryIO) -> None:
        with Image.fromarray(image) as encoded:
            encoded.save(stream, **options)

    return write


def _check_target(target: Path, overwrite: bool) -> bool:
    """Whether ``target`` exists; refuse it without ``overwrite`` or when it is not a
    regular file."""
    if not (target.exists() or target.is_symlink()):
        return False
    if not overwrite:
        raise _io_error("output_exists", f"Output already exists: {target}")
    if target.is_symlink() or not target.is_file():
        raise _io_error("invalid_output_target", f"Cannot replace a non-regular file: {target}")
    return True


def _temporary(directory: Path, name: str) -> tuple[int, str]:
    """A new file ``.<name>.<random>.tmp`` in ``directory``, opened for writing. Its
    permissions follow the umask (``tempfile.mkstemp`` files are owner-only)."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    while True:
        path = directory / f".{name}.{secrets.token_hex(4)}.tmp"
        try:
            return os.open(path, flags, 0o666), str(path)
        except FileExistsError:
            continue


def _stage(directory: Path, name: str, write: Callable[[BinaryIO], None]) -> Path:
    """Write a temporary file in ``directory`` and flush it to disk."""
    handle, temporary = _temporary(directory, name)
    try:
        with os.fdopen(handle, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return Path(temporary)


def _place_new(staged: Path, target: Path) -> None:
    """Move ``staged`` to ``target`` only while that name is free (``FileExistsError``
    otherwise): by a hard link, or by an exclusive create of ``target`` that ``staged``
    then replaces where the file system has no hard links."""
    try:
        os.link(staged, target)
    except FileExistsError:
        raise
    except OSError:
        os.close(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
        try:
            os.replace(staged, target)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
    else:
        staged.unlink()


class _Transaction:
    """Files moved into place one at a time, undone in reverse on failure."""

    def __init__(self, directory: Path, overwrite: bool) -> None:
        self.directory = directory
        self.overwrite = overwrite
        self.done: list[tuple[Path, Path | None]] = []  # (target, backup of the old file)
        self.temporary: list[Path] = []

    def _backup(self, target: Path) -> Path | None:
        if not target.exists():
            return None
        handle, name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".old", dir=self.directory
        )
        os.close(handle)
        backup = Path(name)
        backup.unlink()
        try:
            os.link(target, backup)
        except OSError:  # no hard links on this file system
            shutil.copy2(target, backup)
        self.temporary.append(backup)
        return backup

    def place(self, staged: Path, target: Path) -> None:
        if self.overwrite:
            backup = self._backup(target)
            os.replace(staged, target)
        else:
            backup = None
            _place_new(staged, target)  # fails if the name was taken meanwhile
        self.done.append((target, backup))

    def remove(self, target: Path) -> None:
        backup = self._backup(target)
        if backup is not None:
            target.unlink()
            self.done.append((target, backup))

    def rollback(self) -> list[str]:
        """Undo the moves; a backup that could not be restored stays on disk."""
        errors = []
        for target, backup in reversed(self.done):
            try:
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
            except OSError as exc:
                errors.append(f"{target}: {exc}")
                if backup is not None:
                    self.temporary.remove(backup)
                    errors.append(f"previous file kept at {backup}")
        self.done.clear()
        return errors

    def cleanup(self) -> None:
        for path in self.temporary:
            path.unlink(missing_ok=True)


def save_image(
    path: str | os.PathLike[str],
    image: NDArray[np.uint8],
    *,
    fmt: str | None = None,
    quality: int = JPEG_QUALITY,
    icc_profile: bytes | None = None,
    dpi: float | tuple[float, float] | None = None,
    overwrite: bool = False,
) -> str:
    """Write one RGB image atomically; the format follows ``fmt`` or the suffix."""
    target = _path(path, output=True)
    fmt = image_format(fmt if fmt is not None else target.suffix)
    options = _save_options(fmt, quality, icc_profile, dpi)
    pixels = _check_image(image)
    staged = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        _check_target(target, overwrite)
        staged = _stage(target.parent, target.name, _encoder(pixels, options))
        if overwrite:
            os.replace(staged, target)
        else:
            _place_new(staged, target)  # fails if the name was taken meanwhile
    except FileExistsError as exc:
        raise _io_error("output_exists", f"Output already exists: {target}") from exc
    except OSError as exc:
        raise _io_error("save_failed", f"Could not save image: {exc}") from exc
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    return str(target)


def output_paths(
    output_dir: str | os.PathLike[str], stem: str, fmt: str = "jpeg"
) -> dict[str, Path]:
    """``{"image": <dir>/<stem>.jpg|.png, "record": <dir>/<stem>.scan.json}``."""
    directory = _path(output_dir, output=True)
    if (
        not isinstance(stem, str)
        or stem in {"", ".", ".."}
        or any(mark in stem for mark in ("/", "\\", "\x00"))
    ):
        raise _io_error("invalid_path", "stem must be a plain file name.")
    return {
        "image": directory / f"{stem}{FORMATS[image_format(fmt)]}",
        "record": directory / f"{stem}{RECORD_SUFFIX}",
    }


def save_scan(
    result: ScanResult,
    output_dir: str | os.PathLike[str],
    stem: str,
    *,
    fmt: str = "jpeg",
    quality: int = JPEG_QUALITY,
    icc_profile: bytes | None = None,
    dpi: float | tuple[float, float] | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
    """Write the result's image (if any) and ``<stem>.scan.json`` (see the module docstring).

    On success ``result.artifacts`` holds the written paths and ``result.timings["save"]``
    the seconds spent encoding and staging the image (the record is excluded).
    """
    if not isinstance(result, ScanResult):
        raise _input_error("invalid_result", "Expected a ScanResult.")
    if not isinstance(overwrite, bool):
        raise _input_error("invalid_overwrite", "overwrite must be a bool.")
    paths = output_paths(output_dir, stem, fmt)
    options = _save_options(image_format(fmt), quality, icc_profile, dpi)
    pixels = None if result.image is None else _check_image(result.image)
    directory = paths["image"].parent
    started = time.perf_counter()
    transaction = _Transaction(directory, overwrite)
    staged: list[Path] = []
    try:
        directory.mkdir(parents=True, exist_ok=True)
        _check_target(paths["image"], overwrite)
        _check_target(paths["record"], overwrite)
        artifacts = {"record": str(paths["record"])}
        if pixels is not None:
            staged.append(_stage(directory, paths["image"].name, _encoder(pixels, options)))
            artifacts = {"image": str(paths["image"]), **artifacts}
        timings = {**result.timings, "save": time.perf_counter() - started}
        record = {**result.to_dict(), "timings": timings, "artifacts": artifacts}
        try:
            text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        except (TypeError, ValueError) as exc:
            raise _io_error(
                "record_serialization_failed", f"Scan record is not JSON: {exc}"
            ) from exc
        staged.append(
            _stage(directory, paths["record"].name, lambda s: s.write(text.encode("utf-8")))
        )
        if pixels is not None:
            transaction.place(staged[0], paths["image"])
        else:
            transaction.remove(paths["image"])
        transaction.place(staged[-1], paths["record"])
    except BaseException as exc:
        errors = transaction.rollback()
        if errors:
            exc.add_note("Output rollback incomplete: " + "; ".join(errors))
        if isinstance(exc, FileExistsError):
            target = exc.filename2 or exc.filename
            raise _io_error("output_exists", f"Output already exists: {target}") from exc
        if isinstance(exc, OSError):
            failure = _io_error("save_failed", f"Could not save scan: {exc}")
            failure.metadata = {"save_transaction": {"rolled_back": not errors, "errors": errors}}
            raise failure from exc
        raise
    finally:
        transaction.cleanup()
        for path in staged:
            path.unlink(missing_ok=True)
    result.timings = timings
    result.artifacts = artifacts
    return dict(artifacts)


def read_scan_record(path: str | os.PathLike[str]) -> ScanResult:
    """Read ``<stem>.scan.json`` back into a ``ScanResult`` (no arrays)."""
    source = _path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise _input_error("invalid_scan_record", f"Could not read scan record: {exc}") from exc
    return ScanResult.from_dict(data)
