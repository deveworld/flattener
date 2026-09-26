"""HTTP Range access to remote files and zip archives (standard library only).

Building blocks for ``fetch_research.py``:

* ``RangeClient``: one persistent HTTP(S) connection (= one "connection slot") that follows
  redirects, keeps credentials on the original host only, validates ``206`` / ``Content-Range``
  and uses ``If-Range`` so a changed remote file raises ``RemoteChanged``.
* ``read_central_directory``: parses a remote zip's central directory (zip64 aware) with two or
  three small Range requests.
* ``extract_remote_members``: streams contiguous runs of wanted members with one Range request per
  run, verifies every member's CRC-32 and size against the central directory, and writes files
  atomically. A journal makes it resumable at member granularity.
* ``download_file``: resumable full download (``.part`` + Range) followed by a SHA-256 check.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import socket
import ssl
import struct
import threading
import time
import urllib.parse
import zlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

USER_AGENT = "flattener-research-data/1 (resumable range fetch)"
NETWORK_ERRORS = (
    OSError,
    http.client.HTTPException,
    socket.timeout,
    ssl.SSLError,
    ConnectionError,
)


RETRYABLE_STATUS = (408, 425, 429, 500, 502, 503, 504)


class HttpStatusError(Exception):
    def __init__(self, status: int, url: str, detail: str = ""):
        super().__init__(f"HTTP {status} for {url} {detail}".strip())
        self.status = status
        self.url = url


class RemoteChanged(Exception):
    """The remote file differs from the recorded identity (ETag/size)."""


class IntegrityError(Exception):
    """CRC-32, size or SHA-256 mismatch."""


class RateLimiter:
    """Token bucket shared by all connections; ``rate`` in bytes/s, ``None`` = unlimited."""

    def __init__(self, rate: float | None):
        self.rate = rate
        self._lock = threading.Lock()
        self._tokens = 0.0
        self._last = time.monotonic()

    def consume(self, n: int) -> None:
        if not self.rate:
            return
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.rate, self._tokens + (now - self._last) * self.rate)
            self._last = now
            self._tokens -= n
            deficit = -self._tokens
        if deficit > 0:
            time.sleep(deficit / self.rate)


class Meter:
    """Thread-safe byte counter with a sliding-window rate (for progress/ETA)."""

    def __init__(self, window: float = 60.0):
        self.window = window
        self.total = 0
        self._lock = threading.Lock()
        self._samples: list[tuple[float, int]] = [(time.monotonic(), 0)]

    def add(self, n: int) -> None:
        with self._lock:
            self.total += n
            now = time.monotonic()
            if now - self._samples[-1][0] >= 1.0:
                self._samples.append((now, self.total))
                cutoff = now - self.window
                while len(self._samples) > 2 and self._samples[1][0] < cutoff:
                    self._samples.pop(0)

    def rate(self) -> float:
        with self._lock:
            now = time.monotonic()
            t0, b0 = self._samples[0]
            dt = now - t0
            return (self.total - b0) / dt if dt > 0 else 0.0


@dataclass
class RemoteInfo:
    url: str
    final_url: str
    size: int
    etag: str | None
    last_modified: str | None
    linked_etag: str | None  # Hugging Face X-Linked-ETag (SHA-256 of LFS objects)
    accept_ranges: bool

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "size": self.size,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "linked_etag": self.linked_etag,
            "accept_ranges": self.accept_ranges,
        }


class RangeClient:
    """A single persistent connection. Not thread-safe: use one client per worker."""

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        limiter: RateLimiter | None = None,
        meter: Meter | None = None,
    ):
        self.url = url
        self.auth_host = urllib.parse.urlsplit(url).netloc
        self.auth_headers = dict(headers or {})
        self.timeout = timeout
        self.limiter = limiter or RateLimiter(None)
        self.meter = meter
        self._conn: http.client.HTTPConnection | None = None
        self._conn_key: tuple[str, str] | None = None
        self._final_url: str | None = None
        self.info: RemoteInfo | None = None

    # -- connection handling -------------------------------------------------------------
    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._conn_key = None

    def reset(self, *, forget_redirect: bool = False) -> None:
        self.close()
        if forget_redirect:
            self._final_url = None

    def _connection(self, scheme: str, netloc: str) -> http.client.HTTPConnection:
        key = (scheme, netloc)
        if self._conn is None or self._conn_key != key:
            self.close()
            if scheme == "https":
                ctx = ssl.create_default_context()
                self._conn = http.client.HTTPSConnection(netloc, timeout=self.timeout, context=ctx)
            else:
                self._conn = http.client.HTTPConnection(netloc, timeout=self.timeout)
            self._conn_key = key
        return self._conn

    def _request(
        self, url: str, headers: dict[str, str], method: str = "GET"
    ) -> http.client.HTTPResponse:
        parts = urllib.parse.urlsplit(url)
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        hdrs = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity", **headers}
        if parts.netloc == self.auth_host:
            hdrs.update(self.auth_headers)  # credentials never leave the original host
        conn = self._connection(parts.scheme, parts.netloc)
        try:
            conn.request(method, path, headers=hdrs)
            return conn.getresponse()
        except NETWORK_ERRORS:
            self.close()
            raise

    def _open(
        self, headers: dict[str, str], method: str = "GET"
    ) -> tuple[http.client.HTTPResponse, dict[str, str]]:
        """Follow redirects; return the final response and headers seen on redirect hops."""
        url = self._final_url or self.url
        hop_headers: dict[str, str] = {}
        for _ in range(10):
            resp = self._request(url, headers, method)
            if resp.status in (301, 302, 303, 307, 308):
                for k in ("x-linked-etag", "x-linked-size", "etag"):
                    v = resp.getheader(k)
                    if v and k not in hop_headers:
                        hop_headers[k] = v
                location = resp.getheader("location")
                resp.read()
                if not location:
                    raise HttpStatusError(resp.status, url, "redirect without Location")
                url = urllib.parse.urljoin(url, location)
                continue
            if url != self.url:
                self._final_url = url
            return resp, hop_headers
        raise HttpStatusError(0, self.url, "too many redirects")

    # -- public API ------------------------------------------------------------------------
    def probe(self) -> RemoteInfo:
        """Identify the remote file with a 1-byte Range GET (works where HEAD differs)."""
        self._final_url = None
        resp, hops = self._open({"Range": "bytes=0-0"})
        body = resp.read()
        if resp.status not in (200, 206):
            raise HttpStatusError(resp.status, self.url, body[:200].decode("latin-1", "replace"))
        size = None
        cr = resp.getheader("content-range")
        if resp.status == 206 and cr and "/" in cr:
            total = cr.rsplit("/", 1)[1]
            size = int(total) if total.isdigit() else None
        if size is None:
            cl = resp.getheader("content-length")
            size = int(cl) if cl and resp.status == 200 else None
        if size is None:
            raise HttpStatusError(resp.status, self.url, "cannot determine size")
        linked = hops.get("x-linked-etag") or resp.getheader("x-linked-etag")
        self.info = RemoteInfo(
            url=self.url,
            final_url=self._final_url or self.url,
            size=size,
            etag=resp.getheader("etag") or hops.get("etag"),
            last_modified=resp.getheader("last-modified"),
            linked_etag=linked.strip('"') if linked else None,
            accept_ranges=resp.status == 206,
        )
        return self.info

    def stream(
        self, start: int, end: int, *, if_range: str | None = None, chunk: int = 1 << 18
    ) -> Iterator[bytes]:
        """Yield bytes [start, end] (inclusive). Raises RemoteChanged if the server ignores
        Range/If-Range (i.e. the file changed). The caller must exhaust or close the iterator."""
        headers = {"Range": f"bytes={start}-{end}"}
        if if_range:
            headers["If-Range"] = if_range
        resp, _ = self._open(headers)
        if resp.status in (401, 403, 404, 410) and self._final_url:
            # Signed CDN URLs expire: resolve the redirect again once.
            resp.read()
            self._final_url = None
            resp, _ = self._open(headers)
        if resp.status == 200:
            self.close()
            raise RemoteChanged(f"server answered 200 to a Range request for {self.url}")
        if resp.status != 206:
            body = resp.read()[:200].decode("latin-1", "replace")
            raise HttpStatusError(resp.status, self.url, body)
        cr = resp.getheader("content-range") or ""
        expect = f"bytes {start}-{end}/"
        if not cr.startswith(expect):
            self.close()
            raise RemoteChanged(f"unexpected Content-Range {cr!r}, wanted {expect}...")
        remaining = end - start + 1
        ok = False
        try:
            while remaining > 0:
                data = resp.read(min(chunk, remaining))
                if not data:
                    raise http.client.IncompleteRead(b"", remaining)
                remaining -= len(data)
                self.limiter.consume(len(data))
                if self.meter is not None:
                    self.meter.add(len(data))
                yield data
            ok = True
        finally:
            if not ok:
                self.close()  # connection state unknown after a partial read

    def fetch(self, start: int, end: int, *, if_range: str | None = None) -> bytes:
        return b"".join(self.stream(start, end, if_range=if_range))


def with_retries(
    fn: Callable[[], object],
    *,
    client: RangeClient | None,
    what: str,
    log: Callable[[str], None],
    attempts: int = 30,
    max_sleep: float = 300.0,
):
    """Retry ``fn`` on network errors and 5xx/429 with exponential backoff."""
    delay = 5.0
    for i in range(1, attempts + 1):
        try:
            return fn()
        except HttpStatusError as exc:
            if exc.status not in RETRYABLE_STATUS or i == attempts:
                raise
            log(f"RETRY {what}: {exc} (attempt {i}/{attempts}, sleep {delay:.0f}s)")
        except NETWORK_ERRORS as exc:
            if i == attempts:
                raise
            log(
                f"RETRY {what}: {type(exc).__name__}: {exc} (attempt {i}/{attempts}, "
                f"sleep {delay:.0f}s)"
            )
        if client is not None:
            client.reset()
        time.sleep(delay)
        delay = min(max_sleep, delay * 2)
    raise AssertionError("unreachable")


# -- zip central directory -------------------------------------------------------------------
EOCD_SIG = b"PK\x05\x06"
EOCD64_LOC_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CDH_SIG = b"PK\x01\x02"
LFH_SIG = b"PK\x03\x04"


@dataclass
class ZipEntry:
    name: str
    flag: int
    method: int
    crc: int
    csize: int
    usize: int
    lho: int
    end: int = 0  # start of the next record: bytes [lho, end) belong to this entry

    @property
    def is_dir(self) -> bool:
        return self.name.endswith("/")


@dataclass
class CentralDirectory:
    entries: list[ZipEntry]
    cd_offset: int
    cd_size: int
    cd_sha256: str
    archive_size: int
    zip64: bool = False
    by_name: dict[str, ZipEntry] = field(default_factory=dict)


def parse_central_directory(fetch: Callable[[int, int], bytes], size: int) -> CentralDirectory:
    """``fetch(start, end_inclusive)`` returns archive bytes; ``size`` is the archive size."""
    tail_len = min(size, 65535 + 22 + 20)
    tail = fetch(size - tail_len, size - 1)
    pos = tail.rfind(EOCD_SIG)
    if pos < 0:
        raise IntegrityError("end of central directory record not found")
    (_, _, _, _, n_total, cd_size, cd_offset, _) = struct.unpack("<IHHHHIIH", tail[pos : pos + 22])
    zip64 = False
    loc = tail.rfind(EOCD64_LOC_SIG, 0, pos)
    if loc >= 0 and pos - loc == 20:
        (_, _, eocd64_off, _) = struct.unpack("<IIQI", tail[loc : loc + 20])
        rec = fetch(eocd64_off, eocd64_off + 55)
        if rec[:4] != EOCD64_SIG:
            raise IntegrityError("zip64 end of central directory record not found")
        (n_total, cd_size, cd_offset) = struct.unpack("<QQQ", rec[32:56])
        zip64 = True
    cd = fetch(cd_offset, cd_offset + cd_size - 1)
    entries = parse_central_directory_bytes(cd)
    if len(entries) != n_total:
        raise IntegrityError(f"central directory has {len(entries)} entries, EOCD says {n_total}")
    order = sorted(entries, key=lambda e: e.lho)
    for a, b in zip(order, order[1:], strict=False):
        a.end = b.lho
    if order:
        order[-1].end = cd_offset
    cdir = CentralDirectory(
        entries=entries,
        cd_offset=cd_offset,
        cd_size=cd_size,
        cd_sha256=hashlib.sha256(cd).hexdigest(),
        archive_size=size,
        zip64=zip64,
    )
    cdir.by_name = {e.name: e for e in entries}
    return cdir


def parse_central_directory_bytes(cd: bytes) -> list[ZipEntry]:
    entries: list[ZipEntry] = []
    p = 0
    while p < len(cd):
        if cd[p : p + 4] != CDH_SIG:
            raise IntegrityError(f"bad central directory header at +{p}")
        (_, _, _, flag, method, _, _, crc, csize, usize, fnl, efl, fcl, disk, _, _, lho) = (
            struct.unpack("<IHHHHHHIIIHHHHHII", cd[p : p + 46])
        )
        raw_name = cd[p + 46 : p + 46 + fnl]
        name = raw_name.decode("utf-8" if flag & 0x800 else "cp437")
        extra = cd[p + 46 + fnl : p + 46 + fnl + efl]
        q = 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack("<HH", extra[q : q + 4])
            data = extra[q + 4 : q + 4 + hlen]
            if hid == 0x0001:
                k = 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", data[k : k + 8])[0]
                    k += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", data[k : k + 8])[0]
                    k += 8
                if lho == 0xFFFFFFFF:
                    lho = struct.unpack("<Q", data[k : k + 8])[0]
                    k += 8
            q += 4 + hlen
        entries.append(ZipEntry(name, flag, method, crc, csize, usize, lho))
        p += 46 + fnl + efl + fcl
    return entries


def safe_relpath(name: str, strip_prefix: str = "") -> Path:
    """Map an archive member name to a safe relative path (no absolute paths, no '..')."""
    rel = name.replace("\\", "/")
    if strip_prefix:
        if not rel.startswith(strip_prefix):
            raise ValueError(f"{name!r} does not start with {strip_prefix!r}")
        rel = rel[len(strip_prefix) :]
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or ":" in p for p in parts):
        raise ValueError(f"unsafe member name {name!r}")
    return Path(*parts)


# -- journal ---------------------------------------------------------------------------------
class Journal:
    """Append-only TSV of completed members: name, size, crc32 (hex)."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.done: dict[str, tuple[int, int]] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) == 3 and parts[1].isdigit():
                        self.done[parts[0]] = (int(parts[1]), int(parts[2], 16))
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("a", encoding="utf-8", newline="\n")
        self._pending = 0

    def add(self, name: str, size: int, crc: int) -> None:
        with self._lock:
            self.done[name] = (size, crc)
            self._f.write(f"{name}\t{size}\t{crc:08x}\n")
            self._pending += 1
            if self._pending >= 50:
                self._f.flush()
                self._pending = 0

    def flush(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.flush()
                self._pending = 0

    def close(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.flush()
                self._f.close()


def file_crc32(path: Path, bufsize: int = 1 << 20) -> int:
    crc = 0
    with path.open("rb") as f:
        while chunk := f.read(bufsize):
            crc = zlib.crc32(chunk, crc)
    return crc


def is_member_done(entry: ZipEntry, dest: Path, journal: Journal, *, recheck: bool) -> bool:
    """Done = file present with the central-directory size and (journal or local CRC) match."""
    try:
        st = dest.stat()
    except FileNotFoundError:
        return False
    if st.st_size != entry.usize:
        return False
    rec = journal.done.get(entry.name)
    if rec == (entry.usize, entry.crc) and not recheck:
        return True
    if file_crc32(dest) == entry.crc:
        if rec != (entry.usize, entry.crc):
            journal.add(entry.name, entry.usize, entry.crc)
        return True
    return False


# -- streaming extraction ----------------------------------------------------------------------
class _StreamReader:
    def __init__(self, chunks: Iterator[bytes], start: int):
        self._it = chunks
        self._buf = bytearray()
        self.pos = start  # absolute archive offset of self._buf[0]

    def _fill(self, n: int) -> None:
        while len(self._buf) < n:
            try:
                self._buf += next(self._it)
            except StopIteration:
                raise http.client.IncompleteRead(bytes(self._buf), n - len(self._buf)) from None

    def read(self, n: int) -> bytes:
        self._fill(n)
        out = bytes(self._buf[:n])
        del self._buf[:n]
        self.pos += n
        return out

    def iter_bytes(self, n: int, chunk: int = 1 << 18) -> Iterator[bytes]:
        while n > 0:
            if not self._buf:
                self._fill(1)
            take = min(n, len(self._buf), chunk)
            out = bytes(self._buf[:take])
            del self._buf[:take]
            self.pos += take
            n -= take
            yield out

    def skip_to(self, offset: int) -> None:
        if offset < self.pos:
            raise ValueError(f"cannot seek backwards ({offset} < {self.pos})")
        for _ in self.iter_bytes(offset - self.pos):
            pass

    def drain(self) -> None:
        """Consume the rest of the stream so the HTTP connection stays reusable."""
        self.pos += len(self._buf)
        self._buf.clear()
        for chunk in self._it:
            self.pos += len(chunk)

    def close(self) -> None:
        close = getattr(self._it, "close", None)
        if close is not None:
            close()


def _write_member(reader: _StreamReader, entry: ZipEntry, dest: Path) -> None:
    hdr = reader.read(30)
    if hdr[:4] != LFH_SIG:
        raise IntegrityError(f"bad local header for {entry.name} at {entry.lho}")
    fnl, efl = struct.unpack("<HH", hdr[26:30])
    lname = reader.read(fnl)
    if lname.decode("utf-8" if entry.flag & 0x800 else "cp437") != entry.name:
        raise IntegrityError(f"local name {lname!r} != central name {entry.name!r}")
    reader.read(efl)
    if entry.method not in (0, 8):
        raise IntegrityError(f"unsupported compression method {entry.method} for {entry.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    crc = 0
    size = 0
    inflater = zlib.decompressobj(-15) if entry.method == 8 else None
    with tmp.open("wb") as f:
        for data in reader.iter_bytes(entry.csize):
            if inflater is not None:
                data = inflater.decompress(data)
            crc = zlib.crc32(data, crc)
            size += len(data)
            f.write(data)
        if inflater is not None:
            tail = inflater.flush()
            crc = zlib.crc32(tail, crc)
            size += len(tail)
            f.write(tail)
    if size != entry.usize or crc != entry.crc:
        tmp.unlink(missing_ok=True)
        raise IntegrityError(
            f"{entry.name}: size {size} crc {crc:08x}, expected {entry.usize} {entry.crc:08x}"
        )
    os.replace(tmp, dest)


def plan_runs(
    entries: Iterable[ZipEntry], *, max_gap: int = 1 << 20, max_run: int = 256 << 20
) -> list[list[ZipEntry]]:
    """Group entries (by archive offset) into runs fetched with one Range request each."""
    runs: list[list[ZipEntry]] = []
    for e in sorted(entries, key=lambda x: x.lho):
        if runs:
            run = runs[-1]
            if e.lho - run[-1].end <= max_gap and e.end - run[0].lho <= max_run:
                run.append(e)
                continue
        runs.append([e])
    return runs


def extract_remote_members(
    client: RangeClient,
    entries: list[ZipEntry],
    dest_for: Callable[[ZipEntry], Path],
    journal: Journal,
    *,
    if_range: str | None,
    log: Callable[[str], None],
    on_member: Callable[[ZipEntry], None] | None = None,
    stop: threading.Event | None = None,
    max_gap: int = 1 << 20,
    max_run: int = 256 << 20,
) -> int:
    """Extract ``entries`` (files only). Returns the number of members written."""
    written = 0
    runs = plan_runs(entries, max_gap=max_gap, max_run=max_run)
    for run in runs:
        pending = list(run)
        crc_failures = 0
        attempt = 0
        delay = 5.0
        while pending:
            if stop is not None and stop.is_set():
                return written
            start, end = pending[0].lho, pending[-1].end - 1
            reader = _StreamReader(client.stream(start, end, if_range=if_range), start)
            try:
                while pending:
                    e = pending[0]
                    reader.skip_to(e.lho)
                    _write_member(reader, e, dest_for(e))
                    journal.add(e.name, e.usize, e.crc)
                    pending.pop(0)
                    written += 1
                    if on_member is not None:
                        on_member(e)
                reader.drain()
            except IntegrityError as exc:
                reader.close()
                client.reset()
                crc_failures += 1
                log(f"INTEGRITY {exc} (failure {crc_failures}/3)")
                if crc_failures >= 3:
                    raise
            except NETWORK_ERRORS as exc:
                reader.close()
                client.reset()
                attempt += 1
                if attempt > 40:
                    raise
                log(
                    f"RETRY range {start}-{end}: {type(exc).__name__}: {exc} "
                    f"({len(pending)} members left in run, attempt {attempt}, sleep {delay:.0f}s)"
                )
                time.sleep(delay)
                delay = min(300.0, delay * 2)
            except HttpStatusError as exc:
                reader.close()
                client.reset(forget_redirect=True)
                attempt += 1
                if exc.status not in RETRYABLE_STATUS or attempt > 40:
                    raise
                log(f"RETRY range {start}-{end}: {exc} (attempt {attempt}, sleep {delay:.0f}s)")
                time.sleep(delay)
                delay = min(300.0, delay * 2)
    return written


# -- full download -----------------------------------------------------------------------------
def sha256_file(path: Path, bufsize: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    client: RangeClient,
    dest: Path,
    info: RemoteInfo,
    *,
    if_range: str | None,
    log: Callable[[str], None],
    on_bytes: Callable[[int], None] | None = None,
    stop: threading.Event | None = None,
    segment: int = 64 << 20,
) -> Path:
    """Resumable download to ``dest.part``; renamed to ``dest`` when complete (size-checked).
    The identity of the partial file (URL, size, ETag) is kept in ``dest.part.id``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    ident = dest.with_name(dest.name + ".part.id")
    want_id = f"{info.url}\n{info.size}\n{info.etag or ''}\n{info.linked_etag or ''}\n"
    if part.exists() and (not ident.exists() or ident.read_text("utf-8") != want_id):
        log(f"RESTART {dest.name}: partial file belongs to a different remote version")
        part.unlink()
    ident.write_text(want_id, "utf-8")
    offset = part.stat().st_size if part.exists() else 0
    if offset > info.size:
        part.unlink()
        offset = 0
    if offset:
        log(f"RESUME {dest.name} at {offset}/{info.size}")
    attempt = 0
    delay = 5.0
    with part.open("r+b" if part.exists() else "wb") as f:
        f.seek(offset)
        f.truncate()
        while offset < info.size:
            if stop is not None and stop.is_set():
                return part
            end = min(info.size, offset + segment) - 1
            try:
                for data in client.stream(offset, end, if_range=if_range):
                    f.write(data)
                    offset += len(data)
                    if on_bytes is not None:
                        on_bytes(len(data))
                attempt = 0
                delay = 5.0
            except (HttpStatusError, *NETWORK_ERRORS) as exc:
                retryable = not isinstance(exc, HttpStatusError) or exc.status in RETRYABLE_STATUS
                attempt += 1
                if not retryable or attempt > 40:
                    raise
                f.flush()
                offset = f.tell()
                client.reset()
                log(
                    f"RETRY {dest.name} at {offset}: {type(exc).__name__}: {exc} "
                    f"(attempt {attempt}, sleep {delay:.0f}s)"
                )
                time.sleep(delay)
                delay = min(300.0, delay * 2)
        f.flush()
        os.fsync(f.fileno())
    if part.stat().st_size != info.size:
        raise IntegrityError(f"{part} has {part.stat().st_size} bytes, expected {info.size}")
    os.replace(part, dest)
    ident.unlink(missing_ok=True)
    return dest
