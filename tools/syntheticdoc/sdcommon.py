"""Shared helpers for the SyntheticDoc tooling: HTTP, zip parsing, state, logging.

Standard library only (numpy and Pillow are used by ``fetch.py`` alone).

Transport. The dataset (DOI 10.3929/ethz-c-000801994) and its asset share are
Nextcloud public shares on libdrive.ethz.ch. Two endpoints are used:

* WebDAV ``/public.php/webdav/<path>`` (HTTP Basic, user = share token): PROPFIND
  listings, HEAD and single-range GET. Responses carry ``ETag``,
  ``Last-Modified`` and ``OC-Checksum``, which are recorded and re-checked.
* Share download ``/index.php/s/<token>/download?path=<dir>&files=<name>``: the
  only endpoint that answers multi-range requests with ``multipart/byteranges``
  (WebDAV ignores them and would stream the whole 6 GB shard). One request can
  carry about 350 ranges before Apache's 8 KB request-header limit (HTTP 400),
  so ranges are packed up to ``MAX_RANGE_HEADER`` bytes. This endpoint sends no
  ETag, so consistency is checked with WebDAV HEAD before and after each shard,
  the total size in every ``Content-Range`` and the CRC-32 of every member.

Every thread owns one persistent HTTPS connection, so the number of workers is
the number of connections (default 6). Transient failures (network errors,
429, 5xx) are retried with exponential backoff and jitter, honouring
``Retry-After``.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import http.client
import json
import os
import random
import re
import sqlite3
import ssl
import struct
import sys
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOOL_VERSION = "syntheticdoc-tools-1.0"
HOST = "libdrive.ethz.ch"
DATASET_TOKEN = "AmgqqJb6owHMZBn"  # SyntheticDoc dataset share (DOI 10.3929/ethz-c-000801994)
ASSET_TOKEN = "Gp5rdkRS5dXHExP"  # SyntheticDoc asset share (DOI 10.3929/ethz-c-000804058)
DATASET_DOI = "10.3929/ethz-c-000801994"
ASSET_DOI = "10.3929/ethz-c-000804058"
USER_AGENT = (
    "flattener-syntheticdoc/1.0 (document-unwarping research data mirror; "
    "metadata-first range reads; <=6 connections; python-http.client)"
)
DEFAULT_CONNECTIONS = 6
MAX_RANGE_HEADER = 7000  # bytes of "bytes=a-b,..." per request; Apache rejects > ~8190
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


# --------------------------------------------------------------------------- logging


_log_lock = threading.Lock()


def utcnow() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(*parts: Any) -> None:
    line = " ".join(str(p) for p in parts)
    with _log_lock:
        print(f"{utcnow()} {line}", flush=True)


def _replace(tmp: Path, path: Path) -> None:
    """os.replace with retries: on Windows a reader holding ``path`` open blocks it briefly."""
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 39:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.25)


def write_json_atomic(path: Path, value: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}-{threading.get_ident()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=indent, ensure_ascii=False, sort_keys=False)
        handle.write("\n")
    _replace(tmp, path)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}-{threading.get_ident()}")
    with open(tmp, "wb") as handle:
        handle.write(data)
    _replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_unit(key: str, salt: str) -> float:
    """Deterministic value in [0, 1) from ``key`` (used for splits and selection)."""
    digest = hashlib.sha256(f"{salt}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


# --------------------------------------------------------------------------- HTTP


class HttpError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = True):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class ConsistencyError(RuntimeError):
    """The remote file changed (ETag/size) or returned bytes that fail verification."""


class TruncatedArchive(ConsistencyError):
    """No end-of-central-directory record where the zip format requires one (the last
    65,557 bytes): the file on the server is truncated, so retrying cannot help."""


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests = 0
        self.bytes = 0
        self.retries = 0
        self.started = time.monotonic()

    def add(self, nbytes: int) -> None:
        with self.lock:
            self.requests += 1
            self.bytes += nbytes

    def retry(self) -> None:
        with self.lock:
            self.retries += 1

    def snapshot(self) -> dict[str, float]:
        with self.lock:
            elapsed = max(time.monotonic() - self.started, 1e-9)
            return {
                "requests": self.requests,
                "bytes": self.bytes,
                "retries": self.retries,
                "elapsed_s": round(elapsed, 1),
                "mb_per_s": round(self.bytes / elapsed / 1e6, 3),
            }


class Http:
    """Thread-safe client: one keep-alive connection per thread, retries with backoff."""

    def __init__(
        self,
        *,
        host: str = HOST,
        timeout: float = 180.0,
        max_attempts: int = 8,
        backoff_base: float = 2.0,
        backoff_cap: float = 300.0,
    ) -> None:
        self.host = host
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.local = threading.local()
        self.stats = Stats()
        self.context = ssl.create_default_context()

    def thread_bytes(self) -> int:
        """Response bytes received by the calling thread (exact under concurrency)."""
        return getattr(self.local, "bytes", 0)

    def _conn(self) -> http.client.HTTPSConnection:
        conn = getattr(self.local, "conn", None)
        if conn is None:
            conn = http.client.HTTPSConnection(
                self.host, timeout=self.timeout, context=self.context
            )
            self.local.conn = conn
        return conn

    def _drop(self) -> None:
        conn = getattr(self.local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        self.local.conn = None

    def request(
        self,
        method: str,
        target: str,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        *,
        ok: Iterable[int] = (200, 206, 207),
        max_body: int | None = None,
        what: str = "",
    ) -> Response:
        """Send with retries. ``max_body`` guards against an ignored Range header: a
        larger Content-Length closes the connection without reading the body."""
        ok = tuple(ok)
        hdrs = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        hdrs.update(headers or {})
        attempt = 0
        while True:
            attempt += 1
            retry_after = None
            try:
                conn = self._conn()
                conn.request(method, target, body=body, headers=hdrs)
                resp = conn.getresponse()
                status = resp.status
                rh = {k.lower(): v for k, v in resp.getheaders()}
                length = rh.get("content-length")
                size = int(length) if length is not None and length.isdigit() else None
                if method == "HEAD":
                    resp.read()
                    data = b""
                elif status in ok:
                    if max_body is not None and size is not None and size > max_body:
                        self._drop()
                        raise HttpError(
                            f"{what or target}: {status} with Content-Length {size} > "
                            f"{max_body} (range ignored?)",
                            status=status,
                            retryable=False,
                        )
                    if max_body is None:
                        data = resp.read()
                    else:
                        data = resp.read(max_body + 1)
                        if len(data) > max_body:
                            self._drop()
                            raise HttpError(
                                f"{what or target}: body exceeds {max_body} bytes",
                                status=status,
                                retryable=False,
                            )
                        if not resp.isclosed() and resp.read(1):
                            self._drop()
                            raise HttpError(f"{what or target}: trailing body data")
                elif size is not None and size <= 65536:
                    data = resp.read()
                else:  # unexpected (possibly huge, e.g. a 200 for an ignored Range): do not drain
                    data = resp.read(2048)
                    self._drop()
                if rh.get("connection", "").lower() == "close":
                    self._drop()
                if status in ok:
                    self.stats.add(len(data))
                    self.local.bytes = getattr(self.local, "bytes", 0) + len(data)
                    return Response(status, rh, data)
                retryable = status in RETRYABLE_STATUS
                err = HttpError(
                    f"{what or target}: HTTP {status} {data[:200]!r}",
                    status=status,
                    retryable=retryable,
                )
                retry_after = rh.get("retry-after")
                if not retryable:
                    raise err
            except HttpError as exc:
                if not exc.retryable:
                    raise
                err = exc
            except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
                self._drop()
                err = HttpError(f"{what or target}: {type(exc).__name__}: {exc}")
            if attempt >= self.max_attempts:
                raise err
            delay = min(self.backoff_cap, self.backoff_base * 2 ** (attempt - 1))
            delay = delay * (0.5 + random.random())
            if retry_after and retry_after.strip().isdigit():
                delay = max(delay, min(float(retry_after), 900.0))
            self.stats.retry()
            log(f"retry {attempt}/{self.max_attempts - 1} in {delay:.1f}s: {err}")
            time.sleep(delay)


def _basic(token: str) -> str:
    return "Basic " + base64.b64encode(f"{token}:".encode()).decode()


def _basic_hdr(token: str) -> dict[str, str]:
    return {"Authorization": _basic(token)}


@dataclass(frozen=True)
class RemoteFile:
    """One file on a share: ``path`` like ``train/train_000.zip``."""

    token: str
    path: str

    @property
    def webdav(self) -> str:
        return "/public.php/webdav/" + urllib.parse.quote(self.path)

    @property
    def download(self) -> str:
        directory, _, name = self.path.rpartition("/")
        query = urllib.parse.urlencode({"path": "/" + directory, "files": name})
        return f"/index.php/s/{self.token}/download?{query}"

    @property
    def auth(self) -> dict[str, str]:
        return {"Authorization": _basic(self.token)}


@dataclass(frozen=True)
class Listing:
    path: str
    size: int | None
    etag: str | None
    last_modified: str | None
    is_dir: bool


def propfind(client: Http, token: str, directory: str) -> list[Listing]:
    """Depth-1 PROPFIND of ``directory`` (``""`` for the share root)."""
    body = (
        b'<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop><d:getcontentlength/>'
        b"<d:getetag/><d:getlastmodified/><d:resourcetype/></d:prop></d:propfind>"
    )
    target = "/public.php/webdav/" + urllib.parse.quote(directory.strip("/"))
    if directory.strip("/"):
        target += "/"
    headers = {"Depth": "1", "Content-Type": "application/xml", **_basic_hdr(token)}
    resp = client.request("PROPFIND", target, headers, body, ok=(207,), what=f"PROPFIND {target}")
    ns = {"d": "DAV:"}
    out = []
    for node in ET.fromstring(resp.body).findall("d:response", ns):
        href = urllib.parse.unquote(node.findtext("d:href", "", ns))
        prop = node.find("d:propstat/d:prop", ns)
        if prop is None:
            continue
        length = prop.findtext("d:getcontentlength", None, ns)
        rtype = prop.find("d:resourcetype", ns)
        is_dir = rtype is not None and rtype.find("d:collection", ns) is not None
        rel = href.split("/public.php/webdav/", 1)[-1].strip("/")
        out.append(
            Listing(
                path=rel,
                size=int(length) if length else None,
                etag=prop.findtext("d:getetag", None, ns),
                last_modified=prop.findtext("d:getlastmodified", None, ns),
                is_dir=is_dir,
            )
        )
    return out


@dataclass(frozen=True)
class Head:
    size: int
    etag: str | None
    last_modified: str | None
    checksum: str | None


def head(client: Http, remote: RemoteFile) -> Head:
    resp = client.request("HEAD", remote.webdav, remote.auth, ok=(200,), what=f"HEAD {remote.path}")
    return Head(
        size=int(resp.headers["content-length"]),
        etag=resp.headers.get("etag"),
        last_modified=resp.headers.get("last-modified"),
        checksum=resp.headers.get("oc-checksum"),
    )


def get_file(client: Http, remote: RemoteFile, *, expect_size: int | None = None) -> Response:
    resp = client.request("GET", remote.webdav, remote.auth, ok=(200,), what=f"GET {remote.path}")
    if expect_size is not None and len(resp.body) != expect_size:
        raise ConsistencyError(f"{remote.path}: got {len(resp.body)} bytes, expected {expect_size}")
    return resp


_CR = re.compile(rb"content-range:\s*bytes\s+(\d+)-(\d+)/(\d+|\*)", re.I)


def get_range(
    client: Http,
    remote: RemoteFile,
    start: int,
    end: int,
    *,
    total: int | None = None,
    etag: str | None = None,
) -> bytes:
    """Single inclusive range over WebDAV; checks Content-Range, total size and ETag."""
    headers = {"Range": f"bytes={start}-{end}", **remote.auth}
    resp = client.request(
        "GET",
        remote.webdav,
        headers,
        ok=(206,),
        max_body=end - start + 1,
        what=f"GET {remote.path} [{start}-{end}]",
    )
    match = _CR.match(b"content-range: " + resp.headers.get("content-range", "").encode())
    if not match or int(match[1]) != start or int(match[2]) != end:
        raise ConsistencyError(
            f"{remote.path}: bad Content-Range {resp.headers.get('content-range')}"
        )
    if total is not None and match[3] != b"*" and int(match[3]) != total:
        raise ConsistencyError(f"{remote.path}: size changed to {match[3]!r} (expected {total})")
    if etag is not None and resp.headers.get("etag") not in (None, etag):
        raise ConsistencyError(f"{remote.path}: ETag {resp.headers.get('etag')} != {etag}")
    if len(resp.body) != end - start + 1:
        raise ConsistencyError(f"{remote.path}: short range body {len(resp.body)}")
    return resp.body


def pack_ranges(ranges: Sequence[tuple[int, int]], limit: int = MAX_RANGE_HEADER) -> list[list]:
    """Split sorted inclusive ranges into groups whose Range header stays under ``limit``."""
    groups: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    length = len("bytes=")
    for a, b in ranges:
        piece = len(f"{a}-{b},")
        if current and length + piece > limit:
            groups.append(current)
            current, length = [], len("bytes=")
        current.append((a, b))
        length += piece
    if current:
        groups.append(current)
    return groups


def parse_multipart(body: bytes, content_type: str) -> list[tuple[int, int, int | None, bytes]]:
    """Parse ``multipart/byteranges``; returns (start, end, total, data) per part.

    Part lengths come from each part's Content-Range, so the boundary string may
    appear inside binary data without harm.
    """
    match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not match:
        raise ConsistencyError(f"multipart without boundary: {content_type}")
    delimiter = b"--" + match[1].encode()
    parts = []
    pos = body.find(delimiter)
    if pos < 0:
        raise ConsistencyError("multipart: first boundary missing")
    while True:
        pos += len(delimiter)
        if body[pos : pos + 2] == b"--":
            break
        header_end = body.find(b"\r\n\r\n", pos)
        if header_end < 0:
            raise ConsistencyError("multipart: truncated part header")
        header = body[pos:header_end]
        cr = _CR.search(header)
        if not cr:
            raise ConsistencyError(f"multipart: part without Content-Range: {header[:200]!r}")
        start, end = int(cr[1]), int(cr[2])
        total = None if cr[3] == b"*" else int(cr[3])
        data_start = header_end + 4
        data = body[data_start : data_start + end - start + 1]
        if len(data) != end - start + 1:
            raise ConsistencyError("multipart: truncated part body")
        parts.append((start, end, total, data))
        pos = body.find(delimiter, data_start + len(data))
        if pos < 0:
            raise ConsistencyError("multipart: closing boundary missing")
    return parts


def get_ranges(
    client: Http,
    remote: RemoteFile,
    ranges: Sequence[tuple[int, int]],
    *,
    total: int,
) -> dict[tuple[int, int], bytes]:
    """Fetch many inclusive ranges via the share download endpoint (multipart).

    Returns data for every requested range; raises ConsistencyError when the
    reported total size differs from ``total`` or a range is missing.
    """
    out: dict[tuple[int, int], bytes] = {}
    for group in pack_ranges(sorted(ranges)):
        spec = ",".join(f"{a}-{b}" for a, b in group)
        expected = sum(b - a + 1 for a, b in group)
        resp = client.request(
            "GET",
            remote.download,
            {"Range": "bytes=" + spec},
            ok=(206,),
            max_body=expected + 400 * len(group) + 4096,
            what=f"GET {remote.path} x{len(group)} ranges",
        )
        ctype = resp.headers.get("content-type", "")
        if ctype.startswith("multipart/byteranges"):
            parts = parse_multipart(resp.body, ctype)
        else:
            cr = _CR.match(b"content-range: " + resp.headers.get("content-range", "").encode())
            if not cr:
                raise ConsistencyError(f"{remote.path}: 206 without Content-Range")
            total_s = None if cr[3] == b"*" else int(cr[3])
            parts = [(int(cr[1]), int(cr[2]), total_s, resp.body)]
        for start, end, part_total, _ in parts:
            if part_total is not None and part_total != total:
                raise ConsistencyError(
                    f"{remote.path}: size changed to {part_total} (expected {total})"
                )
        for a, b in group:
            for start, end, _, data in parts:
                if start <= a and b <= end:
                    out[(a, b)] = data[a - start : b - start + 1]
                    break
            else:
                raise ConsistencyError(f"{remote.path}: range {a}-{b} missing from response")
    return out


# --------------------------------------------------------------------------- zip


@dataclass(frozen=True)
class ZipEntry:
    name: str
    method: int
    flags: int
    crc: int
    csize: int
    usize: int
    offset: int  # local header offset


def locate_central_directory(tail: bytes, tail_offset: int) -> tuple[int, int, int, int | None]:
    """From the file tail return (cd_offset, cd_size, entries, zip64_eocd_offset).

    If the zip64 EOCD record lies before ``tail_offset`` the caller must read it
    (the returned cd values are then the 32-bit placeholders and zip64 offset is set).
    """
    i = tail.rfind(b"PK\x05\x06")
    if i < 0 or len(tail) - i < 22:
        raise ConsistencyError("zip: end of central directory not found")
    _, _, _, _, n, cd_size, cd_offset, _ = struct.unpack("<IHHHHIIH", tail[i : i + 22])
    j = tail.rfind(b"PK\x06\x07", 0, i)
    if j >= 0 and i - j == 20:
        (z64_offset,) = struct.unpack("<Q", tail[j + 8 : j + 16])
        rel = z64_offset - tail_offset
        if 0 <= rel <= len(tail) - 56:
            return parse_zip64_eocd(tail[rel : rel + 56]) + (None,)
        return cd_offset, cd_size, n, z64_offset
    return cd_offset, cd_size, n, None


def parse_zip64_eocd(record: bytes) -> tuple[int, int, int]:
    if record[:4] != b"PK\x06\x06":
        raise ConsistencyError("zip: bad zip64 end of central directory record")
    n, cd_size, cd_offset = struct.unpack("<QQQ", record[32:56])
    return cd_offset, cd_size, n


def parse_central_directory(cd: bytes) -> list[ZipEntry]:
    entries = []
    p = 0
    while p + 46 <= len(cd) and cd[p : p + 4] == b"PK\x01\x02":
        (_, _, _, flags, method, _, _, crc, csize, usize, fnl, efl, fcl, _, _, _, lho) = (
            struct.unpack("<IHHHHHHIIIHHHHHII", cd[p : p + 46])
        )
        raw = cd[p + 46 : p + 46 + fnl]
        name = raw.decode("utf-8" if flags & 0x800 else "cp437", "replace")
        if not flags & 0x800:
            try:
                name = raw.decode("utf-8")
            except UnicodeDecodeError:
                pass
        extra = cd[p + 46 + fnl : p + 46 + fnl + efl]
        q = 0
        while q + 4 <= len(extra):
            hid, hl = struct.unpack("<HH", extra[q : q + 4])
            data = extra[q + 4 : q + 4 + hl]
            if hid == 1:
                k = 0
                if usize == 0xFFFFFFFF:
                    (usize,) = struct.unpack("<Q", data[k : k + 8])
                    k += 8
                if csize == 0xFFFFFFFF:
                    (csize,) = struct.unpack("<Q", data[k : k + 8])
                    k += 8
                if lho == 0xFFFFFFFF:
                    (lho,) = struct.unpack("<Q", data[k : k + 8])
                    k += 8
            q += 4 + hl
        entries.append(ZipEntry(name, method, flags, crc, csize, usize, lho))
        p += 46 + fnl + efl + fcl
    if p != len(cd):
        raise ConsistencyError(f"zip: central directory parse stopped at {p}/{len(cd)}")
    return entries


def extract_member(buf: bytes, rel: int, entry: ZipEntry) -> bytes:
    """Payload of ``entry`` whose local header starts at ``buf[rel]``; CRC-32 verified."""
    if buf[rel : rel + 4] != b"PK\x03\x04":
        raise ConsistencyError(f"zip: no local header for {entry.name}")
    fnl, efl = struct.unpack("<HH", buf[rel + 26 : rel + 30])
    name = buf[rel + 30 : rel + 30 + fnl]
    if name.decode("utf-8", "replace") != entry.name:
        raise ConsistencyError(f"zip: local name {name!r} != {entry.name!r}")
    start = rel + 30 + fnl + efl
    data = buf[start : start + entry.csize]
    if len(data) != entry.csize:
        raise ConsistencyError(f"zip: truncated member {entry.name}")
    if entry.method == 8:
        data = zlib.decompress(data, -15)
    elif entry.method != 0:
        raise ConsistencyError(f"zip: unsupported method {entry.method} for {entry.name}")
    if len(data) != entry.usize or (zlib.crc32(data) & 0xFFFFFFFF) != entry.crc:
        raise ConsistencyError(f"zip: CRC/size mismatch for {entry.name}")
    return data


def read_central_directory(
    client: Http,
    remote: RemoteFile,
    size: int,
    etag: str | None,
    *,
    tail_bytes: int = 1 << 20,
) -> tuple[list[ZipEntry], int, int]:
    """Read and parse the central directory with at most three range requests."""
    start = max(0, size - tail_bytes)
    tail = get_range(client, remote, start, size - 1, total=size, etag=etag)
    try:
        cd_offset, cd_size, n, z64 = locate_central_directory(tail, start)
    except ConsistencyError as exc:
        if len(tail) >= min(size, 65557):
            raise TruncatedArchive(
                f"{remote.path}: no end-of-central-directory record in the last {len(tail)} "
                f"bytes of {size} (truncated upstream)"
            ) from exc
        raise
    if z64 is not None:
        record = get_range(client, remote, z64, z64 + 55, total=size, etag=etag)
        cd_offset, cd_size, n = parse_zip64_eocd(record)
    if cd_offset >= start and cd_offset + cd_size <= size:
        cd = tail[cd_offset - start : cd_offset - start + cd_size]
    else:
        cd = get_range(client, remote, cd_offset, cd_offset + cd_size - 1, total=size, etag=etag)
    entries = parse_central_directory(cd)
    if len(entries) != n:
        raise ConsistencyError(f"{remote.path}: {len(entries)} entries, EOCD says {n}")
    return entries, cd_offset, cd_size


def member_spans(entries: Sequence[ZipEntry], cd_offset: int) -> dict[str, tuple[int, int]]:
    """Inclusive byte span of every member (local header .. byte before the next one)."""
    ordered = sorted(entries, key=lambda e: e.offset)
    spans = {}
    for k, entry in enumerate(ordered):
        nxt = ordered[k + 1].offset if k + 1 < len(ordered) else cd_offset
        if nxt <= entry.offset:
            raise ConsistencyError(f"zip: overlapping members at {entry.name}")
        spans[entry.name] = (entry.offset, nxt - 1)
    return spans


# --------------------------------------------------------------------------- state


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS shards (
  shard TEXT PRIMARY KEY, split TEXT NOT NULL, idx INTEGER NOT NULL, path TEXT NOT NULL,
  size INTEGER, etag TEXT, last_modified TEXT, oc_checksum TEXT, listed_at TEXT,
  cd_offset INTEGER, cd_size INTEGER, n_entries INTEGER, n_samples INTEGER,
  status TEXT NOT NULL DEFAULT 'pending', harvested_at TEXT, attempts INTEGER DEFAULT 0,
  error TEXT, seconds REAL, bytes INTEGER
);
CREATE TABLE IF NOT EXISTS samples (
  uid TEXT PRIMARY KEY, split TEXT NOT NULL, shard TEXT NOT NULL, sample_id TEXT NOT NULL,
  r_off INTEGER, r_size INTEGER, r_crc INTEGER, r_method INTEGER,
  b_off INTEGER, b_size INTEGER, b_crc INTEGER, b_method INTEGER,
  m_off INTEGER, m_size INTEGER, m_crc INTEGER, m_method INTEGER,
  blk_start INTEGER, blk_end INTEGER, members TEXT,
  status TEXT, mesh_family TEXT, mesh_id TEXT, doc_source TEXT, doc_name TEXT, bg_name TEXT,
  orientation INTEGER, meta_z BLOB,
  doc_class TEXT, bg_class TEXT, lic_class TEXT, doc_ref TEXT, doc_license TEXT,
  bg_license TEXT, doc_group TEXT, mesh_group TEXT,
  our_split TEXT, test_overlap INTEGER,
  sel_role TEXT, sel_rank REAL, dl_status TEXT, dl_at TEXT, dl_error TEXT,
  dl_attempts INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS samples_shard ON samples(shard);
CREATE TABLE IF NOT EXISTS docs (
  doc_ref TEXT PRIMARY KEY, source TEXT, license_id TEXT, license_url TEXT, license_class TEXT,
  title TEXT, authors TEXT, url TEXT, citation TEXT, extra TEXT
);
CREATE TABLE IF NOT EXISTS backgrounds (
  name TEXT PRIMARY KEY, license TEXT, license_class TEXT, authors TEXT, source TEXT,
  meta TEXT
);
"""

POST_INDEXES = """
CREATE INDEX IF NOT EXISTS samples_sel ON samples(sel_role, dl_status);
CREATE INDEX IF NOT EXISTS samples_split ON samples(our_split, lic_class);
"""


class State:
    """SQLite index + resumable state. All access is serialized by one lock."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=120)
        self.db.execute("PRAGMA busy_timeout=120000")
        self.db.executescript(SCHEMA)
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.commit()

    def get(self, key: str, default: Any = None) -> Any:
        with self.lock:
            row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key: str, value: Any) -> None:
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO kv(key, value) VALUES (?, ?)", (key, json.dumps(value))
            )
            self.db.commit()

    def query(self, sql: str, args: Sequence[Any] = ()) -> list[tuple]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def execute(self, sql: str, args: Sequence[Any] = ()) -> None:
        with self.lock:
            self.db.execute(sql, args)
            self.db.commit()

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self.lock:
            self.db.executemany(sql, rows)
            self.db.commit()

    def close(self) -> None:
        with self.lock:
            self.db.close()


# --------------------------------------------------------------------------- layout


@dataclass(frozen=True)
class Layout:
    data_root: Path  # e.g. D:\flattener-sota\data

    @property
    def research(self) -> Path:
        return self.data_root / "research" / "syntheticdoc"

    @property
    def release(self) -> Path:
        return self.data_root / "release" / "syntheticdoc"

    @property
    def index_db(self) -> Path:
        return self.research / "index" / "syntheticdoc-index.sqlite"

    @property
    def tables(self) -> Path:
        return self.research / "index" / "license-tables"

    @property
    def reports(self) -> Path:
        return self.research / "reports"


def default_data_root() -> Path | None:
    candidate = Path(r"D:\flattener-sota\data")
    return candidate if sys.platform == "win32" and candidate.is_dir() else None
