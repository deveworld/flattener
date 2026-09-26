"""SyntheticDoc data tooling (tools/syntheticdoc): zip/multipart parsing, licenses, split."""

import io
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "syntheticdoc"
sys.path.insert(0, str(TOOLS))

import fetch  # noqa: E402
import sdcommon  # noqa: E402
import sdlicense  # noqa: E402
import sdsplit  # noqa: E402
from harvest import parse_metadata  # noqa: E402

FILES = ("uv_inverse.exr", "render.png", "backward_map.npy", "metadata.json", "shadow.png")


def _shard(n: int = 3) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for k in range(n):
            for name in FILES:
                info = zipfile.ZipInfo(f"{k:07d}/{name}")
                data = f"{k}-{name}-".encode() * (7 + k)
                z.writestr(info, data, compress_type=zipfile.ZIP_STORED)
        z.writestr("0000000/../deflated.txt", b"hello" * 50, compress_type=zipfile.ZIP_DEFLATED)
    return buf.getvalue()


def _zip64(data: bytes) -> bytes:
    """Rewrite the EOCD into zip64 form (0xFFFFFFFF placeholders) like the real shards."""
    i = data.rfind(b"PK\x05\x06")
    _, _, _, n1, n, cd_size, cd_offset, clen = struct.unpack("<IHHHHIIH", data[i : i + 22])
    body = data[:i]
    z64_offset = len(body)
    z64 = struct.pack("<IQHHIIQQQQ", 0x06064B50, 44, 45, 45, 0, 0, n, n, cd_size, cd_offset)
    loc = struct.pack("<IIQI", 0x07064B50, 0, z64_offset, 1)
    eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, n1, n, cd_size, 0xFFFFFFFF, 0)
    return body + z64 + loc + eocd


@pytest.mark.parametrize("zip64", [False, True])
def test_central_directory_and_members(zip64):
    data = _shard()
    if zip64:
        data = _zip64(data)
    tail_offset = max(0, len(data) - 600)
    cd_offset, cd_size, n, z64 = sdcommon.locate_central_directory(data[tail_offset:], tail_offset)
    if z64 is not None:
        cd_offset, cd_size, n = sdcommon.parse_zip64_eocd(data[z64 : z64 + 56])
    entries = sdcommon.parse_central_directory(data[cd_offset : cd_offset + cd_size])
    assert len(entries) == n == 3 * len(FILES) + 1
    reference = {i.filename: i for i in zipfile.ZipFile(io.BytesIO(_shard())).infolist()}
    spans = sdcommon.member_spans(entries, cd_offset)
    for entry in entries:
        assert entry.crc == reference[entry.name].CRC
        a, b = spans[entry.name]
        payload = sdcommon.extract_member(data[a : b + 1], 0, entry)
        assert zlib.crc32(payload) == entry.crc
    # render, map and metadata are contiguous: one block per sample
    e = {x.name: x for x in entries}
    r, m = spans["0000001/render.png"], spans["0000001/metadata.json"]
    block = data[r[0] : m[1] + 1]
    for name in ("render.png", "backward_map.npy", "metadata.json"):
        entry = e[f"0000001/{name}"]
        assert sdcommon.extract_member(block, entry.offset - r[0], entry).startswith(b"1-")
    corrupt = bytearray(block)
    corrupt[-3] ^= 0xFF
    with pytest.raises(sdcommon.ConsistencyError, match="CRC"):
        entry = e["0000001/metadata.json"]
        sdcommon.extract_member(bytes(corrupt), entry.offset - r[0], entry)


def test_multipart_parse_and_range_packing():
    boundary = "b0undary"
    parts = [(10, 19, b"--b0undary"), (100, 103, b"\r\n\r\n")]
    body = b""
    for a, b, data in parts:
        body += (
            f"\r\n--{boundary}\r\nContent-type: application/zip\r\n"
            f"Content-range: bytes {a}-{b}/5000\r\n\r\n"
        ).encode() + data
    body += f"\r\n--{boundary}--\r\n".encode()
    got = sdcommon.parse_multipart(body, f"multipart/byteranges; boundary={boundary}")
    assert got == [(10, 19, 5000, b"--b0undary"), (100, 103, 5000, b"\r\n\r\n")]
    ranges = [(k * 10_000_000_000, k * 10_000_000_000 + 1300) for k in range(1000)]
    groups = sdcommon.pack_ranges(ranges, limit=7000)
    assert sum(len(g) for g in groups) == 1000
    assert all(len("bytes=" + ",".join(f"{a}-{b}" for a, b in g)) <= 7000 for g in groups)


class _FakeHttp:
    def __init__(self, blob: bytes, total: int):
        self.blob, self.total, self.calls = blob, total, 0

    def request(self, method, target, headers=None, body=None, **_):
        self.calls += 1
        spec = headers["Range"].split("=", 1)[1].split(",")
        out = b""
        for piece in spec:
            a, b = map(int, piece.split("-"))
            out += (
                f"\r\n--XyZ\r\nContent-range: bytes {a}-{b}/{self.total}\r\n\r\n"
            ).encode() + self.blob[a : b + 1]
        out += b"\r\n--XyZ--\r\n"
        return sdcommon.Response(206, {"content-type": "multipart/byteranges; boundary=XyZ"}, out)


def test_get_ranges_maps_parts_and_checks_total():
    blob = bytes(range(256)) * 40
    fake = _FakeHttp(blob, len(blob))
    remote = sdcommon.RemoteFile("tok", "train/train_000.zip")
    ranges = [(5, 9), (300, 330), (9000, 9100)]
    got = sdcommon.get_ranges(fake, remote, ranges, total=len(blob))
    assert all(got[(a, b)] == blob[a : b + 1] for a, b in ranges)
    with pytest.raises(sdcommon.ConsistencyError, match="size changed"):
        sdcommon.get_ranges(fake, remote, ranges, total=len(blob) + 1)
    assert remote.download.startswith("/index.php/s/tok/download?path=%2Ftrain&files=")


def test_license_normalization():
    cc = sdlicense.cc_license
    assert cc("https://creativecommons.org/licenses/by/4.0/legalcode", "CC BY") == (
        "CC-BY-4.0",
        "clean",
    )
    assert cc("http://creativecommons.org/licenses/by-4.0/", "CC BY 4.0") == ("CC-BY-4.0", "clean")
    assert cc("https://creativecommons.org/licenses/by-sa/4.0/deed.de", "CC BY SA")[1] == "sa"
    assert cc("https://creativecommons.org/licenses/by-nc/3.0/", "CC BY NC") == (
        "CC-BY-NC-3.0",
        "nc",
    )
    assert cc("https://creativecommons.org/licenses/by-nc-sa/4.0/", "")[1] == "nc"
    assert cc("", "CC BY NC SA")[1] == "nc"
    assert cc("", "CC BY-ND")[1] == "unknown"
    assert cc("", "all rights reserved")[1] == "unknown"
    assert sdlicense.background_class("CC0") == ("CC0-1.0", "clean")
    assert sdlicense.background_class("CC-BY") == ("CC-BY", "clean")
    assert sdlicense.background_class("CC BY-NC-ND 2.0")[1] == "nc"
    assert sdlicense.background_class(None)[1] == "unknown"
    assert sdlicense.combine("clean", "sa") == "sa"
    assert sdlicense.combine("sa", "unknown") == "unknown"
    assert sdlicense.combine("unknown", "nc", "clean") == "nc"


def test_parse_metadata_paths():
    raw = (
        b'{"files": {"mesh": "/x/src/mesh_generator/results/fold_by_pull/01234.obj",'
        b' "document": "/x/src/document_gathering/results/imslp/'
        b'page_011043_T\xc3\xa4n WWV_p56.png",'
        b' "surface_texture": "/x/surfaces/flat-clean/train_Stone_0386_elf_floor/"},'
        b' "status": "success", "orientation": 90}'
    )
    got = parse_metadata(raw)
    assert got["mesh_family"] == "fold_by_pull" and got["mesh_id"] == "01234"
    assert got["doc_source"] == "imslp" and got["doc_name"].startswith("page_011043_Tän")
    assert got["bg_name"] == "train_Stone_0386_elf_floor" and got["orientation"] == 90


def test_split_is_group_disjoint_and_deterministic():
    rng = np.random.default_rng(0)
    rows = []
    for k in range(6000):
        split = "test" if k % 50 == 0 else ("val" if k % 10 == 0 else "train")
        source = ["arxiv", "doab", "res_txt2img"][k % 3]
        doc = f"{source}:{int(rng.integers(0, 400 if source != 'res_txt2img' else 5000))}"
        mesh = f"fam{k % 6}/{int(rng.integers(0, 1500))}"
        status = "success" if k % 97 else "incomplete"
        rows.append((f"{split}/{k:07d}", split, status, source, "clean", doc, mesh))
    out, info = sdsplit.assign(rows, val_fraction=0.02)
    assert out == sdsplit.assign(rows, val_fraction=0.02)[0]
    by = {r[0]: r for r in rows}
    train = [by[u] for u, (s, _) in out.items() if s == "train"]
    val = [by[u] for u, (s, _) in out.items() if s == "val"]
    assert {r[5] for r in train}.isdisjoint({r[5] for r in val})
    assert {r[6] for r in train}.isdisjoint({r[6] for r in val})
    assert all(out[r[0]][0] == "official_test" for r in rows if r[1] == "test")
    assert all(
        out[r[0]][0] == "excluded_invalid" for r in rows if r[2] != "success" and r[1] != "test"
    )
    pool = sum(1 for r in rows if r[1] != "test" and r[2] == "success")
    assert 0.01 < len(val) / pool < 0.035
    assert info["leak_check"] == {"doc_groups_train_and_val": 0, "mesh_groups_train_and_val": 0}


def test_convert_sample_full_resolution_and_fp16():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (1440, 1024, 3), dtype=np.uint8)
    png = io.BytesIO()
    Image.fromarray(image).save(png, "PNG", compress_level=1)
    bm = rng.uniform(-1, 1, (177, 121, 2)).astype(np.float32)
    npy = io.BytesIO()
    np.save(npy, bm)
    jpeg, bm16_bytes, info = fetch.convert_sample(png.getvalue(), npy.getvalue())
    with Image.open(io.BytesIO(jpeg)) as im:
        assert im.format == "JPEG" and im.size == (1024, 1440) and im.mode == "RGB"
    bm16 = np.load(io.BytesIO(bm16_bytes))
    assert bm16.dtype == np.float16 and bm16.shape == (177, 121, 2)
    assert np.abs(bm16.astype(np.float32) - bm).max() <= 2.0**-11 + 1e-9
    assert info["backward_map"]["fp16_max_error_render_px_xy"][0] <= 0.25 + 1e-6
    wrong = io.BytesIO()
    np.save(wrong, bm[:100])
    with pytest.raises(sdcommon.ConsistencyError):
        fetch.convert_sample(png.getvalue(), wrong.getvalue())


def test_state_roundtrip(tmp_path):
    state = sdcommon.State(tmp_path / "index.sqlite")
    state.put("k", {"a": 1})
    assert state.get("k") == {"a": 1} and state.get("missing", 3) == 3
    state.close()


class _RangeHttp:
    """Serves single Range requests (WebDAV style) from an in-memory file."""

    def __init__(self, blob: bytes):
        self.blob = blob

    def request(self, method, target, headers=None, body=None, **_):
        a, b = map(int, headers["Range"].split("=", 1)[1].split("-"))
        cr = f"bytes {a}-{b}/{len(self.blob)}"
        return sdcommon.Response(206, {"content-range": cr}, self.blob[a : b + 1])


def test_truncated_shard_is_terminal_and_intact_shard_parses():
    blob = _zip64(_shard(40))
    remote = sdcommon.RemoteFile("tok", "train/train_106.zip")
    entries, cd_offset, _ = sdcommon.read_central_directory(
        _RangeHttp(blob), remote, len(blob), None, tail_bytes=70_000
    )
    assert len(entries) == 40 * len(FILES) + 1 and cd_offset < len(blob)
    cut = blob[: (len(blob) // 2 // 4096) * 4096]  # like the upstream partial uploads
    with pytest.raises(sdcommon.TruncatedArchive):
        sdcommon.read_central_directory(_RangeHttp(cut), remote, len(cut), None, tail_bytes=70_000)
