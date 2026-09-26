"""Tests for tools/research_data (local HTTP server with Range support; no internet).

Run: python -m pytest tools/research_data/tests -q
"""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import random
import sys
import threading
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_research as fr  # noqa: E402
import remote_zip as rz  # noqa: E402


class _State:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.etag = '"v1"'
        self.redirect: dict[str, str] = {}
        self.drop_after: int | None = None  # close the socket after N body bytes (once)
        self.corrupt_at: int | None = None  # flip one byte at this absolute offset
        self.seen_auth: list[tuple[str, str | None]] = []
        self.requests = 0


def _make_handler(state: _State):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence
            pass

        def do_GET(self):  # noqa: N802
            state.requests += 1
            state.seen_auth.append((self.path, self.headers.get("Authorization")))
            if self.path in state.redirect:
                self.send_response(302)
                self.send_header("Location", state.redirect[self.path])
                self.send_header("X-Linked-ETag", '"' + "ab" * 32 + '"')
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            data = state.files.get(self.path)
            if data is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            rng = self.headers.get("Range")
            if_range = self.headers.get("If-Range")
            if rng and (if_range is None or if_range == state.etag):
                a, b = rng.split("=", 1)[1].split("-")
                start, end = int(a), min(int(b), len(data) - 1)
                body = bytearray(data[start : end + 1])
                if state.corrupt_at is not None and start <= state.corrupt_at <= end:
                    body[state.corrupt_at - start] ^= 0xFF
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            else:
                body = bytearray(data)
                self.send_response(200)
            self.send_header("ETag", state.etag)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state.drop_after is not None and len(body) > state.drop_after:
                n = state.drop_after
                state.drop_after = None
                self.wfile.write(bytes(body[:n]))
                self.wfile.flush()
                self.close_connection = True
                self.connection.shutdown(2)
                return
            self.wfile.write(bytes(body))

    return Handler


@pytest.fixture
def server():
    state = _State()
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield state, base
    httpd.shutdown()
    httpd.server_close()


def _zip_bytes(n: int = 40, force_zip64: bool = False) -> tuple[bytes, dict[str, bytes]]:
    rnd = random.Random(1)
    members: dict[str, bytes] = {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("DS/", b"")
        for i in range(n):
            for sub, method in (("img", zipfile.ZIP_STORED), ("grid", zipfile.ZIP_DEFLATED)):
                name = f"DS/{sub}/{i:04d}.bin"
                data = bytes(rnd.getrandbits(8) for _ in range(rnd.randint(0, 3000)))
                if sub == "grid":
                    data = data + b"\0" * 2000  # compressible
                zi = zipfile.ZipInfo(name)
                zi.compress_type = method
                with zf.open(zi, "w", force_zip64=force_zip64) as w:
                    w.write(data)
                members[name] = data
                zf.writestr(f"__MACOSX/DS/{sub}/._{i:04d}.bin", b"junk" * 20)
        zf.writestr("DS/split.json", b'{"train": [1, 2]}')
        members["DS/split.json"] = b'{"train": [1, 2]}'
    return buf.getvalue(), members


@pytest.mark.parametrize("zip64", [False, True])
def test_central_directory_and_range_extraction(server, tmp_path, zip64):
    state, base = server
    blob, members = _zip_bytes(force_zip64=zip64)
    state.files["/a.zip"] = blob
    client = rz.RangeClient(base + "/a.zip")
    info = client.probe()
    assert info.size == len(blob) and info.etag == '"v1"' and info.accept_ranges
    cd = rz.parse_central_directory(lambda a, b: client.fetch(a, b, if_range=info.etag), info.size)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        ref = {zi.filename: (zi.CRC, zi.file_size, zi.header_offset) for zi in zf.infolist()}
    assert {e.name: (e.crc, e.usize, e.lho) for e in cd.entries} == ref

    wanted = [e for e in cd.entries if e.name.startswith("DS/") and not e.is_dir]
    wanted = [e for e in wanted if not fr.is_junk(e.name)]
    journal = rz.Journal(tmp_path / "j.tsv")

    def dest(e):
        return tmp_path / "out" / rz.safe_relpath(e.name, "DS/")

    before = state.requests
    n = rz.extract_remote_members(
        client, wanted, dest, journal, if_range=info.etag, log=print, max_gap=4096
    )
    journal.close()
    assert n == len(members)
    for name, data in members.items():
        assert dest(cd.by_name[name]).read_bytes() == data
    # junk entries interleave with wanted ones but gaps are small: very few range requests
    assert state.requests - before <= 3

    # resume: remove one file; only it is fetched again
    victim = cd.by_name["DS/img/0007.bin"]
    dest(victim).unlink()
    journal = rz.Journal(tmp_path / "j.tsv")
    todo = [e for e in wanted if not rz.is_member_done(e, dest(e), journal, recheck=False)]
    assert [e.name for e in todo] == [victim.name]
    assert rz.extract_remote_members(client, todo, dest, journal, if_range=info.etag, log=print)
    assert dest(victim).read_bytes() == members[victim.name]
    journal.close()


def test_extraction_survives_dropped_connection_and_detects_corruption(
    server, tmp_path, monkeypatch
):
    state, base = server
    blob, members = _zip_bytes()
    state.files["/a.zip"] = blob
    client = rz.RangeClient(base + "/a.zip")
    info = client.probe()
    cd = rz.parse_central_directory(lambda a, b: client.fetch(a, b), info.size)
    wanted = [e for e in cd.entries if e.name.startswith("DS/img/")]

    def dest(e):
        return tmp_path / "o" / rz.safe_relpath(e.name, "DS/")

    monkeypatch.setattr(rz.time, "sleep", lambda s: None)  # no backoff in tests
    state.drop_after = 20000
    journal = rz.Journal(tmp_path / "j.tsv")
    rz.extract_remote_members(client, wanted, dest, journal, if_range=info.etag, log=print)
    for e in wanted:
        assert dest(e).read_bytes() == members[e.name]
    journal.close()

    # a persistent corruption must raise after 3 attempts, leaving no bad file behind
    target = cd.by_name["DS/grid/0003.bin"]
    state.corrupt_at = target.end - 30  # inside the deflated data of that member
    journal = rz.Journal(tmp_path / "j2.tsv")
    with pytest.raises(rz.IntegrityError):
        rz.extract_remote_members(
            client,
            [target],
            lambda e: tmp_path / "bad" / "x.bin",
            journal,
            if_range=info.etag,
            log=print,
        )
    assert not (tmp_path / "bad" / "x.bin").exists()
    journal.close()


def test_download_resume_and_remote_change(server, tmp_path):
    state, base = server
    blob = bytes(random.Random(2).getrandbits(8) for _ in range(300_000))
    state.files["/f.zip"] = blob
    client = rz.RangeClient(base + "/f.zip")
    info = client.probe()
    dest = tmp_path / "f.zip"
    part = tmp_path / "f.zip.part"
    # simulate an earlier interrupted run of the same remote version
    (tmp_path / "f.zip.part.id").write_text(f"{info.url}\n{info.size}\n{info.etag}\n\n", "utf-8")
    part.write_bytes(blob[:123_457])
    got = rz.download_file(client, dest, info, if_range=info.etag, log=print, segment=50_000)
    assert got == dest and dest.read_bytes() == blob and not part.exists()
    assert rz.sha256_file(dest) == hashlib.sha256(blob).hexdigest()

    # the remote changes while a partial download exists: If-Range mismatch -> 200 -> error
    part.write_bytes(blob[:1000])
    (tmp_path / "f.zip.part.id").write_text(f"{info.url}\n{info.size}\n{info.etag}\n\n", "utf-8")
    state.etag = '"v2"'
    with pytest.raises(rz.RemoteChanged):
        rz.download_file(client, dest, info, if_range=info.etag, log=print)


def test_redirect_keeps_credentials_on_origin_host(server, tmp_path):
    state, base = server
    other = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    threading.Thread(target=other.serve_forever, daemon=True).start()
    try:
        cdn = f"http://127.0.0.1:{other.server_address[1]}"
        state.files["/cdn/obj"] = b"x" * 5000
        state.redirect["/resolve/obj"] = cdn + "/cdn/obj"
        client = rz.RangeClient(base + "/resolve/obj", headers={"Authorization": "Bearer T"})
        info = client.probe()
        assert info.size == 5000 and info.linked_etag == "ab" * 32
        assert client.fetch(10, 19) == b"x" * 10
        auth = dict(state.seen_auth)
        assert auth["/resolve/obj"] == "Bearer T"
        assert auth["/cdn/obj"] is None
    finally:
        other.shutdown()
        other.server_close()


def test_local_extraction_mappers_and_pairing(tmp_path):
    ddir = tmp_path / "doc3d"
    # grid archive in the Doc3D_grid.zip layout (+ macOS junk)
    grid_zip = tmp_path / "Doc3D_grid.zip"
    with zipfile.ZipFile(grid_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for g in ("grid2D", "grid3D"):
            zf.writestr(f"Doc3D_grid/{g}/.DS_Store", b"junk")
            for k in range(3):
                zf.writestr(f"Doc3D_grid/{g}/1/{k}_x.mat", b"\0" * 512 + b"\x89HDF\r\n\x1a\n")
        zf.writestr("Doc3D_grid/traindoc.txt", "1/0_x\n1/1_x\n")
        zf.writestr("Doc3D_grid/valdoc3D.txt", "1/2_x\n")
    j = rz.Journal(tmp_path / "g.tsv")
    res = fr.extract_local_zip(
        grid_zip,
        ddir,
        fr.strip_mapper("Doc3D_grid/"),
        j,
        recheck=False,
        stop=threading.Event(),
    )
    j.close()
    assert res["by_dir"]["grid2D"]["files"] == 3 and res["members_skipped_junk"] == 2
    # image archive with an unknown leading directory
    img_zip = tmp_path / "img_1.zip"
    from PIL import Image

    with zipfile.ZipFile(img_zip, "w") as zf:
        for k in range(2):  # 2_x has no image -> reported as grid_without_img
            b = io.BytesIO()
            Image.new("RGB", (8, 8)).save(b, "PNG")
            zf.writestr(f"doc3d/img/1/{k}_x.png", b.getvalue())
    j = rz.Journal(tmp_path / "i.tsv")
    fr.extract_local_zip(
        img_zip, ddir, fr.doc3d_img_mapper, j, recheck=False, stop=threading.Event()
    )
    j.close()
    pairing = fr.pairing_doc3d(ddir)
    assert pairing["img"] == 2 and pairing["grid2D"] == 3 and pairing["img_with_both_grids"] == 2
    assert pairing["grid_without_img"] == 1 and pairing["ok"]
    assert pairing["traindoc.txt"]["missing_grids"] == 0
    assert pairing["content_check"]["mat_headers"]["kinds"] == {"matlab-v7.3-hdf5": 3}
    with pytest.raises(ValueError):
        rz.safe_relpath("DS/../evil", "DS/")


def test_uvdoc_pairing(tmp_path):
    from PIL import Image

    ddir = tmp_path / "uvdoc_final"
    for sub in ("img", "metadata_sample", "grid2d", "grid3d", "metadata_geom"):
        (ddir / sub).mkdir(parents=True)
    for sid, geom in (("00000", "g0"), ("00001", "g0"), ("00002", "g1")):
        Image.new("RGB", (4, 4)).save(ddir / "img" / f"{sid}.png")
        (ddir / "metadata_sample" / f"{sid}.json").write_text(json.dumps({"geom_name": geom}))
    for geom in ("g0",):
        for sub in ("grid2d", "grid3d"):
            (ddir / sub / f"{geom}.mat").write_bytes(b"MATLAB 5.0")
    out = fr.pairing_uvdoc(ddir)
    assert out["img"] == 3 and out["img_whose_geometry_grid_is_missing"] == 1 and not out["ok"]
    (ddir / "grid2d" / "g1.mat").write_bytes(b"MATLAB")
    (ddir / "grid3d" / "g1.mat").write_bytes(b"MATLAB")
    assert fr.pairing_uvdoc(ddir)["ok"]


def _fake_archive(prefix: str, layout: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}/", b"")
        for name, data in layout.items():
            zf.writestr(f"{prefix}/{name}", data)
            head, _, base = f"{prefix}/{name}".rpartition("/")
            zf.writestr(f"__MACOSX/{head}/._{base}", b"junk")
    return buf.getvalue()


def _png(w: int = 8, h: int = 12) -> bytes:
    from PIL import Image

    b = io.BytesIO()
    Image.new("RGB", (w, h), (200, 10, 10)).save(b, "PNG")
    return b.getvalue()


def test_cli_end_to_end(server, tmp_path, monkeypatch):
    state, base = server
    mat = b"\0" * 512 + b"\x89HDF\r\n\x1a\n" + b"\0" * 100
    bench = {
        f"{d}/{i:02d}.{ext}": (_png() if ext == "png" else mat)
        for d, ext in (("img", "png"), ("grid2d", "mat"))
        for i in range(50)
    }
    grid = {f"{g}/1/{k}_a.mat": mat for g in ("grid2D", "grid3D") for k in range(4)}
    grid["traindoc.txt"] = b"1/0_a\n1/1_a\n1/2_a\n"
    grid["valdoc3D.txt"] = b"1/3_a\n"
    final = {"split.json": b'{"geoms": [1]}'}
    for k in range(6):
        final[f"img/{k:05d}.png"] = _png()
        final[f"metadata_sample/{k:05d}.json"] = json.dumps({"geom_name": f"g{k % 2}"}).encode()
        final[f"uvmap/{k:05d}.mat"] = mat  # not selected
    for g in ("g0", "g1"):
        final[f"grid2d/{g}.mat"] = mat
        final[f"grid3d/{g}.mat"] = mat
        final[f"metadata_geom/{g}.json"] = b"{}"
    blobs = {
        "UVDoc_benchmark.zip": _fake_archive("UVDoc_benchmark", bench),
        "Doc3D_grid.zip": _fake_archive("Doc3D_grid", grid),
        "UVDoc_final.zip": _fake_archive("UVDoc_final", final),
    }
    for name, blob in blobs.items():
        state.files[f"/igl/{name}"] = blob
    monkeypatch.setattr(fr, "IGL", base + "/igl")
    monkeypatch.setattr(
        fr, "IGL_SIZES", {k: len(v) for k, v in blobs.items() if k != "UVDoc_final.zip"}
    )
    monkeypatch.setattr(fr, "_cd_cache", {})
    monkeypatch.setattr(fr, "_journals", {})
    root = tmp_path / "data"
    args = [
        "run",
        "--root",
        str(root),
        "--connections",
        "2",
        "--uvdoc-parts",
        "3",
        "--progress-file",
        str(tmp_path / "p.json"),
    ]
    assert fr.main(args) == 0
    m = json.loads((root / "research/uvdoc_final/MANIFEST.json").read_text())
    assert m["track"] == "research" and m["complete"] and m["pairing"]["ok"]
    assert m["pairing"]["img"] == 6 and m["contents"]["grid2d"]["files"] == 2
    assert not (root / "research/uvdoc_final/uvmap").exists()
    assert m["remote_archive_identity"]["central_directory_entries"] > 0
    d = json.loads((root / "research/doc3d/MANIFEST.json").read_text())
    assert d["track"] == "research" and not d["complete"]  # img still pending
    assert d["tasks"]["doc3d-grid"]["state"] == "done"
    assert d["pairing"]["grid2D"] == 4 and d["pairing"]["traindoc.txt"]["missing_grids"] == 0
    arch = d["sources"][0]["archive"]
    assert arch["sha256"] == hashlib.sha256(blobs["Doc3D_grid.zip"]).hexdigest()
    b = json.loads((root / "eval/uvdoc_benchmark/MANIFEST.json").read_text())
    assert b["track"] == "eval-only" and b["complete"] and b["pairing"]["all_dirs_have_50"]
    assert json.loads((root / "research/doc3d/TRACK.json").read_text())["track"] == "research"
    # idempotent rerun: nothing is fetched again
    before = state.requests
    monkeypatch.setattr(fr, "_cd_cache", {})
    monkeypatch.setattr(fr, "_journals", {})
    assert fr.main(args) == 0
    assert state.requests - before <= 4  # probe + central directory of UVDoc_final only
    # gated Doc3D images: 403 -> blocked (exit 3), other data untouched
    monkeypatch.setattr(fr, "HF_URL", base + "/gated/img_{i}.zip")
    state.files.update({f"/gated/img_{i}.zip": b"" for i in range(1, 22)})

    orig_probe = rz.RangeClient.probe

    def probe_403(self):
        if "/gated/" in self.url:
            raise rz.HttpStatusError(403, self.url)
        return orig_probe(self)

    monkeypatch.setattr(rz.RangeClient, "probe", probe_403)
    assert (
        fr.main(
            [
                "run",
                "--root",
                str(root),
                "--tasks",
                "doc3d-img",
                "--progress-file",
                str(tmp_path / "p2.json"),
            ]
        )
        == 3
    )
    p2 = json.loads((tmp_path / "p2.json").read_text())
    assert {it["state"] for it in p2["items"]} == {"blocked"}
