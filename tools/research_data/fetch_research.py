"""Fetch the research-track dewarping datasets (and the small UVDoc benchmark) resumably.

RESEARCH TRACK ONLY. Doc3D and UVDoc_final carry non-commercial or unclear upstream terms
(see ``DATASETS`` below). Weights trained on any of this data are research-track and stay
private. Every manifest written here carries ``"track"``.

Tasks (each needs one HTTP connection; ``--connections`` caps how many run at once):

  uvdoc-benchmark  UVDoc_benchmark.zip (775 MB, 50 samples) -> <root>/eval/uvdoc_benchmark
                   full download, CRC-checked extraction, archive kept. Evaluation only.
  doc3d-grid       UVDoc's Doc3D_grid.zip (2.44 GB; grid2D/grid3D .mat + traindoc/valdoc3D
                   split lists) -> <root>/research/doc3d, archive kept.
  uvdoc-final      UVDoc_final.zip (29.5 GB): only img, grid2d, grid3d, metadata_sample,
                   metadata_geom and split.json (12.2 GB) extracted by HTTP Range, CRC-checked.
  doc3d-img        Doc3D img_1..21.zip (25.0 GB) from the gated Hugging Face dataset, pinned to
                   revision REV; SHA-256 checked against the LFS object id, extracted, archive
                   deleted unless --keep-archives. Needs an HF token whose account was granted
                   access (env HF_TOKEN, --hf-token-file, or
                   %USERPROFILE%/.cache/huggingface/token).

Everything is idempotent: finished tasks leave ``.state/<task>.done.json`` and are skipped on the
next run; partial downloads resume from ``.part`` files, Range extraction resumes per member
from a journal. After the network work, ``MANIFEST.json`` is (re)built per dataset with source
identity (URL, size, ETag, SHA-256), license summary, track, file counts and pairing checks.

Usage (``--root`` is the data root that holds ``research/`` and ``eval/``):
  python fetch_research.py run --root <data-root> \\
      --tasks uvdoc-benchmark,doc3d-grid,uvdoc-final --connections 2
  python fetch_research.py status --root <data-root>
  python fetch_research.py finalize --root <data-root>   # manifests + pairing only
Exit codes: 0 all requested work done, 1 a task failed, 3 some task blocked (e.g. gated access).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import queue
import random
import sys
import threading
import time
import traceback
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from remote_zip import (  # noqa: E402
    HttpStatusError,
    IntegrityError,
    Journal,
    Meter,
    RangeClient,
    RateLimiter,
    RemoteInfo,
    ZipEntry,
    download_file,
    extract_remote_members,
    file_crc32,
    is_member_done,
    parse_central_directory,
    safe_relpath,
    sha256_file,
    with_retries,
)

SCHEMA = "flattener-research-data/1"
IGL = "https://igl.ethz.ch/projects/uvdoc"
HF_REPO = "StonyBrook-CVLab/doc3D-dataset"
REV = "fd3d37e7c9125c45c587a2eb9a6973f9183436e9"  # HF dataset revision (2025-12-18)
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{REV}/doc3d/img_{{i}}.zip"
# Sizes from the HF tree API at REV; a mismatch means the pinned revision changed.
DOC3D_IMG_SIZES = {
    1: 1252670277,
    2: 1251189816,
    3: 1191818532,
    4: 1239232546,
    5: 1131604388,
    6: 1165141545,
    7: 1234117135,
    8: 520551898,
    9: 1273215553,
    10: 1277179239,
    11: 1286570490,
    12: 1284211592,
    13: 1229783036,
    14: 1238875061,
    15: 1238792987,
    16: 1258747871,
    17: 1265834412,
    18: 1184198224,
    19: 1108514482,
    20: 1126374665,
    21: 1268295729,
}
# igl.ethz.ch archive sizes (HTTP Content-Length, 2026-09-26).
IGL_SIZES = {"UVDoc_benchmark.zip": 775129594, "Doc3D_grid.zip": 2435935670}
UVDOC_FINAL_PARTS = ("img", "grid2d", "grid3d", "metadata_sample", "metadata_geom")

RESEARCH_NOTE = (
    "RESEARCH TRACK ONLY. Weights trained with any of this data are research-track and are "
    "never released (owner decision D1(a)). Do not mix into release-track runs."
)

DATASETS: dict[str, dict] = {
    "doc3d": {
        "subdir": "research/doc3d",
        "track": "research",
        "release_weights_allowed": False,
        "title": "Doc3D (DewarpNet, ICCV 2019): img modality + UVDoc's Doc3D_grid.zip",
        "license": {
            "declared": (
                "MIT label on the Hugging Face dataset card and on "
                "github.com/cvlab-stonybrook/doc3D-dataset; Doc3D_grid.zip is distributed with "
                "the MIT-licensed UVDoc code (github.com/tanguymagne/UVDoc) and is derived from "
                "Doc3D, so it inherits Doc3D's upstream terms."
            ),
            "summary": (
                "The MIT label cannot grant more than the upstream sources allow. Document "
                "textures include YES! Magazine pages (CC BY-NC-ND 3.0) and CVF Open Access "
                "papers (author/IEEE copyright); about 70% of renders are lit with Laval Indoor "
                "HDR maps, whose license is free only for non-profit/academic research and "
                "forbids commercial use without a separate license. Treat as non-commercial "
                "research data: usable for research-track training and comparison with prior "
                "SOTA (which trained on Doc3D), never for released weights."
            ),
            "access": (
                "Hugging Face gated='auto': the downloading account must click 'Agree and access "
                "repository', which shares its username/e-mail with the dataset authors."
            ),
            "references": [
                "https://huggingface.co/datasets/StonyBrook-CVLab/doc3D-dataset",
                "https://github.com/cvlab-stonybrook/doc3D-dataset",
                "https://github.com/tanguymagne/UVDoc",
                "https://wiki.creativecommons.org/wiki/Yes!_Magazine",
                "http://hdrdb.com/indoor/",
            ],
            "citation": "Das et al., DewarpNet, ICCV 2019; Verhoeven et al., UVDoc, SIGGRAPH "
            "Asia 2023 (grids)",
        },
        "tasks": ["doc3d-grid", "doc3d-img"],
    },
    "uvdoc_final": {
        "subdir": "research/uvdoc_final",
        "track": "research",
        "release_weights_allowed": False,
        "title": "UVDoc dataset (UVDoc_final.zip), img + grid2d + grid3d + metadata only",
        "license": {
            "declared": (
                "MIT on github.com/tanguymagne/UVDoc-Dataset and github.com/tanguymagne/UVDoc; "
                "UVDoc_final.zip itself contains no license file."
            ),
            "summary": (
                "Composited from real captures of blank paper. Document textures include images "
                "generated with DeepFloyd IF (its license bars any data produced by the model "
                "from commercial or production use), ACM TOG and conference paper pages "
                "(publisher/author copyright), PLOS ONE (CC BY) and Gutenberg pages; "
                "backgrounds come from DTD, which is made available for research purposes only. "
                "Research-track training only; never for released weights."
            ),
            "access": "Public direct HTTP from igl.ethz.ch (Range supported).",
            "references": [
                "https://github.com/tanguymagne/UVDoc-Dataset",
                "https://igl.ethz.ch/projects/uvdoc/",
                "https://github.com/deep-floyd/IF/blob/develop/LICENSE-MODEL",
                "https://www.robots.ox.ac.uk/~vgg/data/dtd/",
            ],
            "citation": "Verhoeven, Magne, Sorkine-Hornung, UVDoc, SIGGRAPH Asia 2023",
        },
        "tasks": ["uvdoc-final"],
    },
    "uvdoc_benchmark": {
        "subdir": "eval/uvdoc_benchmark",
        "track": "eval-only",
        "training_use": "forbidden (benchmark; training on it is test-set leakage)",
        "release_weights_allowed": None,
        "title": "UVDoc benchmark (50 real samples, all modalities)",
        "license": {
            "declared": "MIT on github.com/tanguymagne/UVDoc; the zip has no license file.",
            "summary": (
                "Evaluation only. Report metrics; do not redistribute the files and never train "
                "on them (neither track)."
            ),
            "access": "Public direct HTTP from igl.ethz.ch.",
            "references": [
                "https://github.com/tanguymagne/UVDoc",
                "https://igl.ethz.ch/projects/uvdoc/",
            ],
            "citation": "Verhoeven, Magne, Sorkine-Hornung, UVDoc, SIGGRAPH Asia 2023",
        },
        "tasks": ["uvdoc-benchmark"],
    },
}


def utcnow() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"{utcnow()} {msg}", flush=True)


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}-{threading.get_ident()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def tool_identity() -> dict:
    here = Path(__file__).resolve().parent
    out = {}
    for name in ("fetch_research.py", "remote_zip.py"):
        p = here / name
        out[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    return out


def dataset_dir(root: Path, key: str) -> Path:
    return root / DATASETS[key]["subdir"]


def is_junk(name: str) -> bool:
    base = name.rstrip("/").rsplit("/", 1)[-1]
    return name.startswith("__MACOSX/") or base == ".DS_Store" or base.startswith("._")


# -- work items ----------------------------------------------------------------------------------
class Blocked(Exception):
    """The task cannot proceed without owner action (e.g. gated access not granted)."""


@dataclass
class Progress:
    total: int = 0
    done: int = 0
    state: str = "pending"
    note: str = ""


@dataclass
class Item:
    name: str  # unique work-item name, e.g. "uvdoc-final:part03"
    task: str  # task family, e.g. "uvdoc-final"
    dataset: str
    run: Callable[[Item, Context], dict | None]
    progress: Progress = field(default_factory=Progress)
    done_marker: Path | None = None
    params: dict = field(default_factory=dict)


@dataclass
class Context:
    root: Path
    limiter: RateLimiter
    meter: Meter
    stop: threading.Event
    keep_archives: bool
    hf_token: str | None
    recheck: bool
    net: threading.Semaphore  # one permit per allowed concurrent HTTP connection


def new_client(ctx: Context, url: str, headers: dict | None = None) -> RangeClient:
    return RangeClient(url, headers=headers, limiter=ctx.limiter, meter=ctx.meter, timeout=90)


def probe(ctx: Context, client: RangeClient, what: str) -> RemoteInfo:
    try:
        return with_retries(client.probe, client=client, what=f"probe {what}", log=log, attempts=8)
    except HttpStatusError as exc:
        if exc.status in (401, 403):
            raise Blocked(f"{what}: HTTP {exc.status} (access not granted / no token)") from exc
        raise


def done_marker(ctx: Context, dataset: str, item: str) -> Path:
    return dataset_dir(ctx.root, dataset) / ".state" / f"{item.replace(':', '_')}.done.json"


# -- full archive download + local extraction ---------------------------------------------------
def extract_local_zip(
    archive: Path,
    dest_root: Path,
    mapper: Callable[[str], Path | None],
    journal: Journal,
    *,
    recheck: bool,
    stop: threading.Event,
) -> dict:
    """Extract every mapped member with CRC-32 + size verification; resumable via journal."""
    counts: dict[str, list[int]] = {}
    skipped = 0
    written = 0
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        for zi in infos:
            if stop.is_set():
                raise KeyboardInterrupt("stop requested")
            if zi.is_dir():
                continue
            rel = mapper(zi.filename)
            if rel is None:
                skipped += 1
                continue
            top = rel.parts[0] if len(rel.parts) > 1 else "."
            c = counts.setdefault(top, [0, 0])
            c[0] += 1
            c[1] += zi.file_size
            dest = dest_root / rel
            rec = journal.done.get(zi.filename)
            try:
                size_ok = dest.stat().st_size == zi.file_size
            except FileNotFoundError:
                size_ok = False
            if size_ok and rec == (zi.file_size, zi.CRC) and not recheck:
                continue
            if size_ok and file_crc32(dest) == zi.CRC:
                journal.add(zi.filename, zi.file_size, zi.CRC)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            crc = 0
            size = 0
            with zf.open(zi) as src, tmp.open("wb") as out:  # ZipExtFile also checks CRC
                while chunk := src.read(1 << 20):
                    crc = zlib.crc32(chunk, crc)
                    size += len(chunk)
                    out.write(chunk)
            if crc != zi.CRC or size != zi.file_size:
                tmp.unlink(missing_ok=True)
                raise IntegrityError(f"{archive.name}:{zi.filename} CRC/size mismatch")
            os.replace(tmp, dest)
            journal.add(zi.filename, zi.file_size, zi.CRC)
            written += 1
    return {
        "members_total": len(infos),
        "members_extracted": sum(c[0] for c in counts.values()),
        "members_skipped_junk": skipped,
        "written_this_run": written,
        "by_dir": {k: {"files": v[0], "bytes": v[1]} for k, v in sorted(counts.items())},
        "crc32_verified": True,
    }


def run_archive_item(item: Item, ctx: Context) -> dict:
    p = item.params
    ddir = dataset_dir(ctx.root, item.dataset)
    archive: Path = ddir / "archives" / p["archive_name"]
    headers = p.get("headers") or {}
    item.progress.state = "waiting-for-connection"
    with ctx.net:  # network phase holds a connection permit; extraction does not
        client = new_client(ctx, p["url"], headers)
        try:
            info = probe(ctx, client, item.name)
            if p.get("expected_size") and info.size != p["expected_size"]:
                raise IntegrityError(
                    f"{item.name}: remote size {info.size} != expected {p['expected_size']}"
                )
            expected_sha = p.get("expected_sha256") or (
                info.linked_etag if info.linked_etag and len(info.linked_etag) == 64 else None
            )
            item.progress.total = info.size
            sidecar = archive.with_name(archive.name + ".sha256.json")
            side = read_json(sidecar)
            if archive.exists() and side and side.get("size") == info.size:
                sha = side["sha256"]
                item.progress.done = info.size
                log(f"HAVE {archive.name} sha256={sha}")
            else:
                part = archive.with_name(archive.name + ".part")
                item.progress.done = part.stat().st_size if part.exists() else 0
                item.progress.state = "downloading"
                t0 = time.monotonic()
                b0 = item.progress.done

                def on_bytes(n: int) -> None:
                    item.progress.done += n

                if_range = info.etag if p.get("use_if_range", True) else None
                path = download_file(
                    client,
                    archive,
                    info,
                    if_range=if_range,
                    log=log,
                    on_bytes=on_bytes,
                    stop=ctx.stop,
                )
                if path != archive:
                    raise KeyboardInterrupt("stop requested")
                dt_s = time.monotonic() - t0
                log(
                    f"DOWNLOADED {archive.name} {info.size} B in {dt_s:.0f}s "
                    f"({(info.size - b0) / max(dt_s, 1e-6) / 1e6:.2f} MB/s this run)"
                )
                item.progress.state = "hashing"
                sha = sha256_file(archive)
                if expected_sha and sha != expected_sha:
                    bad = archive.with_name(archive.name + ".bad")
                    os.replace(archive, bad)
                    raise IntegrityError(f"{archive.name}: sha256 {sha} != expected {expected_sha}")
                write_json_atomic(
                    sidecar,
                    {
                        "file": archive.name,
                        "size": info.size,
                        "sha256": sha,
                        "sha256_expected": expected_sha,
                        "remote": info.as_dict(),
                        "verified_utc": utcnow(),
                    },
                )
                log(
                    f"SHA256 {archive.name} {sha}"
                    + (" (matches HF LFS oid)" if expected_sha else "")
                )
        finally:
            client.close()

    item.progress.state = "extracting"
    journal = Journal(ddir / ".state" / f"{item.name.replace(':', '_')}.extract.tsv")
    try:
        ext = extract_local_zip(
            archive, ddir, p["mapper"], journal, recheck=ctx.recheck, stop=ctx.stop
        )
    finally:
        journal.close()
    log(
        f"EXTRACTED {archive.name}: {ext['members_extracted']} files "
        f"({ext['written_this_run']} written this run), CRC-32 verified"
    )
    kept = True
    if not p.get("keep_archive", True) and not ctx.keep_archives:
        archive.unlink()
        kept = False
        log(f"DELETED archive {archive.name} after verified extraction")
    return {
        "method": "full download + local CRC-checked extraction",
        "source": info.as_dict(),
        "archive": {
            "file": str(archive.relative_to(ddir)).replace("\\", "/"),
            "size": info.size,
            "sha256": sha,
            "sha256_expected": expected_sha,
            "sha256_verified_against_publisher": bool(expected_sha),
            "kept_on_disk": kept,
        },
        "extraction": ext,
    }


# -- HTTP Range extraction (UVDoc_final) ---------------------------------------------------------
_cd_cache: dict[str, tuple[RemoteInfo, object]] = {}
_cd_lock = threading.Lock()


def remote_cd(ctx: Context, url: str):
    with _cd_lock:
        if url not in _cd_cache:
            client = new_client(ctx, url)
            try:
                info = probe(ctx, client, url)

                def fetch(a: int, b: int) -> bytes:
                    return with_retries(
                        lambda: client.fetch(a, b, if_range=info.etag),
                        client=client,
                        what=f"cd {a}-{b}",
                        log=log,
                    )

                cd = parse_central_directory(fetch, info.size)
            finally:
                client.close()
            _cd_cache[url] = (info, cd)
            log(
                f"CENTRAL-DIRECTORY {url}: {len(cd.entries)} entries, zip64={cd.zip64}, "
                f"sha256={cd.cd_sha256}"
            )
        return _cd_cache[url]


def uvdoc_final_wanted(cd) -> list[ZipEntry]:
    out = []
    for e in cd.entries:
        if e.is_dir or is_junk(e.name):
            continue
        if e.name == "UVDoc_final/split.json" or any(
            e.name.startswith(f"UVDoc_final/{p}/") for p in UVDOC_FINAL_PARTS
        ):
            out.append(e)
    return sorted(out, key=lambda e: e.lho)


def split_parts(entries: list[ZipEntry], n_parts: int) -> list[list[ZipEntry]]:
    total = sum(e.csize for e in entries)
    per = max(1, total // n_parts)
    parts: list[list[ZipEntry]] = [[]]
    acc = 0
    for e in entries:
        if acc >= per * len(parts) and len(parts) < n_parts:
            parts.append([])
        parts[-1].append(e)
        acc += e.csize
    return [p for p in parts if p]


_journals: dict[Path, Journal] = {}
_journals_lock = threading.Lock()


def shared_journal(path: Path) -> Journal:
    with _journals_lock:
        if path not in _journals:
            _journals[path] = Journal(path)
        return _journals[path]


def run_range_item(item: Item, ctx: Context) -> dict:
    p = item.params
    ddir = dataset_dir(ctx.root, item.dataset)
    info, cd = remote_cd(ctx, p["url"])
    entries: list[ZipEntry] = [cd.by_name[n] for n in p["members"]]
    journal = shared_journal(ddir / ".state" / "uvdoc-final.extract.tsv")

    def dest_for(e: ZipEntry) -> Path:
        return ddir / safe_relpath(e.name, "UVDoc_final/")

    item.progress.total = sum(e.csize for e in entries)
    item.progress.state = "checking"
    todo = []
    for e in entries:
        if is_member_done(e, dest_for(e), journal, recheck=ctx.recheck):
            item.progress.done += e.csize
        else:
            todo.append(e)
    log(
        f"{item.name}: {len(entries) - len(todo)}/{len(entries)} members already present, "
        f"{sum(e.csize for e in todo) / 1e9:.2f} GB to fetch"
    )

    def on_member(e: ZipEntry) -> None:
        item.progress.done += e.csize

    item.progress.state = "waiting-for-connection"
    with ctx.net:
        item.progress.state = "range-extracting"
        client = new_client(ctx, p["url"])
        try:
            extract_remote_members(
                client,
                todo,
                dest_for,
                journal,
                if_range=info.etag,
                log=log,
                on_member=on_member,
                stop=ctx.stop,
            )
        finally:
            client.close()
            journal.flush()
    if ctx.stop.is_set():
        raise KeyboardInterrupt("stop requested")
    missing = [
        e.name for e in entries if not is_member_done(e, dest_for(e), journal, recheck=False)
    ]
    if missing:
        raise IntegrityError(f"{item.name}: {len(missing)} members missing after extraction")
    return {
        "method": "HTTP Range extraction of selected members, CRC-32 verified per member",
        "members_sha256": p["members_sha256"],
        "part_count": p["part_count"],
        "members": len(entries),
        "compressed_bytes": item.progress.total,
        "uncompressed_bytes": sum(e.usize for e in entries),
    }


# -- item construction ---------------------------------------------------------------------------
def strip_mapper(prefix: str) -> Callable[[str], Path | None]:
    def mapper(name: str) -> Path | None:
        if is_junk(name) or not name.startswith(prefix):
            return None
        return safe_relpath(name, prefix)

    return mapper


def doc3d_img_mapper(name: str) -> Path | None:
    """Map any 'img/<folder>/<id>.png' member (with or without a leading directory)."""
    if is_junk(name) or not name.lower().endswith(".png"):
        return None
    parts = name.replace("\\", "/").split("/")
    if "img" not in parts:
        raise IntegrityError(f"unexpected Doc3D img member layout: {name!r}")
    i = len(parts) - 1 - parts[::-1].index("img")
    return safe_relpath("/".join(parts[i:]))


def build_items(ctx: Context, tasks: list[str], uvdoc_parts: int) -> list[Item]:
    items: list[Item] = []
    for task in tasks:
        if task == "uvdoc-benchmark":
            items.append(
                Item(
                    "uvdoc-benchmark",
                    task,
                    "uvdoc_benchmark",
                    run_archive_item,
                    params={
                        "url": f"{IGL}/UVDoc_benchmark.zip",
                        "archive_name": "UVDoc_benchmark.zip",
                        "expected_size": IGL_SIZES["UVDoc_benchmark.zip"],
                        "keep_archive": True,
                        "mapper": strip_mapper("UVDoc_benchmark/"),
                    },
                )
            )
        elif task == "doc3d-grid":
            items.append(
                Item(
                    "doc3d-grid",
                    task,
                    "doc3d",
                    run_archive_item,
                    params={
                        "url": f"{IGL}/Doc3D_grid.zip",
                        "archive_name": "Doc3D_grid.zip",
                        "expected_size": IGL_SIZES["Doc3D_grid.zip"],
                        "keep_archive": True,
                        "mapper": strip_mapper("Doc3D_grid/"),
                    },
                )
            )
        elif task == "uvdoc-final":
            url = f"{IGL}/UVDoc_final.zip"
            _, cd = remote_cd(ctx, url)
            wanted = uvdoc_final_wanted(cd)
            parts = split_parts(wanted, uvdoc_parts)
            for k, part in enumerate(parts):
                names = [e.name for e in part]
                items.append(
                    Item(
                        f"uvdoc-final:part{k:02d}",
                        task,
                        "uvdoc_final",
                        run_range_item,
                        params={
                            "url": url,
                            "members": names,
                            "part_count": len(parts),
                            "compressed_bytes": sum(e.csize for e in part),
                            "members_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
                        },
                    )
                )
        elif task == "doc3d-img":
            headers = {"Authorization": f"Bearer {ctx.hf_token}"} if ctx.hf_token else {}
            for i in range(1, 22):
                items.append(
                    Item(
                        f"doc3d-img:{i:02d}",
                        task,
                        "doc3d",
                        run_archive_item,
                        params={
                            "url": HF_URL.format(i=i),
                            "archive_name": f"img_{i}.zip",
                            "expected_size": DOC3D_IMG_SIZES[i],
                            "keep_archive": False,
                            "headers": headers,
                            "use_if_range": False,  # immutable at pinned REV
                            "mapper": doc3d_img_mapper,
                        },
                    )
                )
        else:
            raise SystemExit(f"unknown task {task!r}")
    for it in items:
        it.done_marker = done_marker(ctx, it.dataset, it.name)
        if it.run is run_archive_item:
            it.progress.total = it.params["expected_size"]
        else:
            it.progress.total = it.params["compressed_bytes"]
    return items


# -- runner --------------------------------------------------------------------------------------
def run_items(ctx: Context, items: list[Item], connections: int, progress_file: Path) -> int:
    q: queue.Queue[Item] = queue.Queue()
    for it in items:
        rec = read_json(it.done_marker) if it.done_marker else None
        if rec and rec.get("members_sha256") != it.params.get("members_sha256"):
            rec = None  # a different part split: re-check members (the journal skips done ones)
        if rec and not ctx.recheck:
            it.progress.state = "done"
            it.progress.note = "done earlier"
            tot = rec.get("archive", {}).get("size") or rec.get("compressed_bytes") or 0
            it.progress.total = it.progress.done = tot
            continue
        q.put(it)
    results: dict[str, str] = {}

    def worker() -> None:
        while not ctx.stop.is_set():
            try:
                it = q.get_nowait()
            except queue.Empty:
                return
            it.progress.state = "starting"
            log(f"START {it.name}")
            try:
                rec = it.run(it, ctx)
                rec = {
                    "item": it.name,
                    "task": it.task,
                    "dataset": it.dataset,
                    "track": DATASETS[it.dataset]["track"],
                    "finished_utc": utcnow(),
                    **rec,
                }
                write_json_atomic(it.done_marker, rec)
                it.progress.state = "done"
                results[it.name] = "done"
                log(f"DONE {it.name}")
            except Blocked as exc:
                it.progress.state = "blocked"
                it.progress.note = str(exc)
                results[it.name] = "blocked"
                log(f"BLOCKED {it.name}: {exc}")
            except KeyboardInterrupt:
                it.progress.state = "stopped"
                results[it.name] = "stopped"
                return
            except Exception as exc:  # noqa: BLE001 - report and continue with other items
                it.progress.state = "failed"
                it.progress.note = f"{type(exc).__name__}: {exc}"
                results[it.name] = "failed"
                log(f"FAILED {it.name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()

    threads = [
        threading.Thread(target=worker, name=f"worker{k}", daemon=True)
        for k in range(max(1, connections) + 1)  # +1: local extraction overlaps downloads
    ]
    for t in threads:
        t.start()
    last_log = 0.0
    while any(t.is_alive() for t in threads):
        time.sleep(2.0)
        now = time.monotonic()
        snap = snapshot(items, ctx.meter)
        write_json_atomic(progress_file, snap)
        if now - last_log >= 60.0:
            last_log = now
            log("PROGRESS " + snap["line"])
    snap = snapshot(items, ctx.meter)
    write_json_atomic(progress_file, snap)
    log("PROGRESS " + snap["line"])
    states = [it.progress.state for it in items]
    if "failed" in states:
        return 1
    if "blocked" in states or "stopped" in states or "pending" in states:
        return 3
    return 0


_t_start = time.monotonic()


def snapshot(items: list[Item], meter: Meter) -> dict:
    total = sum(it.progress.total for it in items)
    done = sum(min(it.progress.done, it.progress.total) for it in items)
    rate = meter.rate()
    known_all = all(
        it.progress.total > 0 or it.progress.state in ("blocked", "failed") for it in items
    )
    blocked_bytes = sum(it.progress.total for it in items if it.progress.state == "blocked")
    remaining = max(0, total - done - blocked_bytes)
    eta = remaining / rate if rate > 0 else None
    by_state: dict[str, int] = {}
    for it in items:
        by_state[it.progress.state] = by_state.get(it.progress.state, 0) + 1
    elapsed = time.monotonic() - _t_start
    line = (
        f"items={by_state} bytes={done / 1e9:.2f}/{total / 1e9:.2f}GB"
        f"{'' if known_all else '+'} rate60s={rate / 1e6:.2f}MB/s "
        f"session={meter.total / 1e9:.2f}GB avg={meter.total / max(elapsed, 1) / 1e6:.2f}MB/s "
        f"eta={'?' if eta is None else f'{eta / 60:.0f}min'}"
    )
    return {
        "updated_utc": utcnow(),
        "pid": os.getpid(),
        "line": line,
        "rate_60s_bytes_per_s": rate,
        "session_bytes": meter.total,
        "session_elapsed_s": round(elapsed, 1),
        "total_bytes": total,
        "done_bytes": done,
        "eta_s": eta,
        "items": [
            {
                "name": it.name,
                "state": it.progress.state,
                "done": it.progress.done,
                "total": it.progress.total,
                "note": it.progress.note,
            }
            for it in items
        ],
    }


# -- manifests and pairing -----------------------------------------------------------------------
def _walk_files(base: Path):
    """Yield (relative_dir, name, size) for every file below ``base`` (scandir-based)."""
    stack = [(base, "")]
    while stack:
        d, rel = stack.pop()
        with os.scandir(d) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append((Path(e.path), f"{rel}/{e.name}" if rel else e.name))
                elif e.is_file(follow_symlinks=False) and not e.name.endswith(".part"):
                    yield rel, e.name, e.stat().st_size


SKIP_TOP = (".state", "archives", "MANIFEST.json", "TRACK.json")


def scan_tree(base: Path) -> dict:
    out: dict[str, dict] = {}
    if not base.exists():
        return out
    for child in sorted(base.iterdir()):
        if child.name in SKIP_TOP or child.name.endswith(".part"):
            continue
        if child.is_file():
            out[child.name] = {"files": 1, "bytes": child.stat().st_size}
            continue
        n = b = 0
        exts: dict[str, int] = {}
        for _rel, name, size in _walk_files(child):
            n += 1
            b += size
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            exts[ext] = exts.get(ext, 0) + 1
        out[child.name] = {"files": n, "bytes": b, "by_ext": exts}
    return out


def stems(base: Path, ext: str) -> set[str]:
    """Relative '<sub>/<stem>' (or '<stem>') keys of files with ``ext`` under ``base``."""
    if not base.exists():
        return set()
    n = len(ext)
    return {
        (f"{rel}/{name[:-n]}" if rel else name[:-n])
        for rel, name, _size in _walk_files(base)
        if name.lower().endswith(ext)
    }


def mat_kind(path: Path) -> str:
    with path.open("rb") as f:
        head = f.read(1024)
    if head[512:520] == b"\x89HDF\r\n\x1a\n":
        return "matlab-v7.3-hdf5"
    if head[:6] == b"MATLAB":
        return "matlab-v5"
    return "unknown"


def sample_check(img_paths: list[Path], mat_paths: list[Path], n: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    res: dict = {}
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is in the remote env
        Image = None
    if img_paths and Image is not None:
        sizes: dict[str, int] = {}
        bad = []
        for p in rng.sample(img_paths, min(n, len(img_paths))):
            try:
                with Image.open(p) as im:
                    im.load()
                    key = f"{im.size[0]}x{im.size[1]} {im.mode}"
                sizes[key] = sizes.get(key, 0) + 1
            except Exception as exc:  # noqa: BLE001
                bad.append(f"{p.name}: {exc}")
        res["images_decoded"] = {"sampled": min(n, len(img_paths)), "sizes": sizes, "bad": bad}
    if mat_paths:
        kinds: dict[str, int] = {}
        for p in rng.sample(mat_paths, min(n, len(mat_paths))):
            k = mat_kind(p)
            kinds[k] = kinds.get(k, 0) + 1
        res["mat_headers"] = {"sampled": min(n, len(mat_paths)), "kinds": kinds}
    return res


def examples(s: set[str], k: int = 10) -> list[str]:
    return sorted(s)[:k]


def pairing_doc3d(ddir: Path) -> dict:
    img = stems(ddir / "img", ".png")
    g2 = stems(ddir / "grid2D", ".mat")
    g3 = stems(ddir / "grid3D", ".mat")
    grids = g2 & g3
    out: dict = {
        "img": len(img),
        "grid2D": len(g2),
        "grid3D": len(g3),
        "grid2D_without_grid3D": len(g2 - g3),
        "grid3D_without_grid2D": len(g3 - g2),
        "img_with_both_grids": len(img & grids),
        "img_without_grid": len(img - grids),
        "img_without_grid_examples": examples(img - grids),
        "grid_without_img": len(grids - img) if img else None,
        "grid_without_img_examples": examples(grids - img) if img else [],
        "img_present": bool(img),
    }
    for split in ("traindoc.txt", "valdoc3D.txt"):
        p = ddir / split
        if p.exists():
            ids = [x.strip() for x in p.read_text("utf-8").splitlines() if x.strip()]
            out[split] = {
                "entries": len(ids),
                "unique": len(set(ids)),
                "missing_grids": len(set(ids) - grids),
                "missing_img": len(set(ids) - img) if img else None,
            }
    if "traindoc.txt" in out and "valdoc3D.txt" in out:
        tr = {x.strip() for x in (ddir / "traindoc.txt").read_text("utf-8").splitlines()}
        va = {x.strip() for x in (ddir / "valdoc3D.txt").read_text("utf-8").splitlines()}
        out["train_val_overlap"] = len((tr & va) - {""})
    imgs = [ddir / "img" / f"{k}.png" for k in sorted(img)]
    mats = [ddir / "grid2D" / f"{k}.mat" for k in sorted(g2)]
    out["content_check"] = sample_check(imgs, mats, 200)
    out["ok"] = bool(img) and not (img - grids) and not (g2 ^ g3)
    return out


def pairing_uvdoc(ddir: Path) -> dict:
    img = stems(ddir / "img", ".png")
    meta = stems(ddir / "metadata_sample", ".json")
    g2 = stems(ddir / "grid2d", ".mat")
    g3 = stems(ddir / "grid3d", ".mat")
    mg = stems(ddir / "metadata_geom", ".json")
    missing_geom: set[str] = set()
    bad_meta: list[str] = []
    geoms_used: set[str] = set()
    for sid in sorted(img & meta):
        try:
            geom = json.loads((ddir / "metadata_sample" / f"{sid}.json").read_text("utf-8"))[
                "geom_name"
            ]
        except Exception as exc:  # noqa: BLE001
            bad_meta.append(f"{sid}: {exc}")
            continue
        geoms_used.add(geom)
        if geom not in g2 or geom not in g3:
            missing_geom.add(sid)
    split_info = None
    sp = ddir / "split.json"
    if sp.exists():
        s = json.loads(sp.read_text("utf-8"))
        split_info = {k: (len(v) if hasattr(v, "__len__") else v) for k, v in s.items()}
    imgs = [ddir / "img" / f"{k}.png" for k in sorted(img)]
    mats = [ddir / "grid2d" / f"{k}.mat" for k in sorted(g2)]
    out = {
        "img": len(img),
        "metadata_sample": len(meta),
        "grid2d": len(g2),
        "grid3d": len(g3),
        "metadata_geom": len(mg),
        "img_without_metadata": len(img - meta),
        "img_whose_geometry_grid_is_missing": len(missing_geom),
        "missing_examples": examples(missing_geom),
        "bad_metadata": bad_meta[:10],
        "distinct_geometries_used": len(geoms_used),
        "grid2d_without_grid3d": len(g2 ^ g3),
        "split_json_keys": split_info,
        "content_check": sample_check(imgs, mats, 200),
    }
    out["ok"] = bool(img) and not (img - meta) and not missing_geom and not bad_meta
    return out


def pairing_benchmark(ddir: Path) -> dict:
    tree = scan_tree(ddir)
    counts = {k: v["files"] for k, v in tree.items() if isinstance(v, dict)}
    dirs = [k for k in counts if (ddir / k).is_dir()]
    imgs = sorted((ddir / "img").glob("*.png")) if (ddir / "img").exists() else []
    return {
        "files_per_dir": counts,
        "all_dirs_have_50": all(counts[k] == 50 for k in dirs),
        "content_check": sample_check(imgs, sorted((ddir / "grid2d").glob("*.mat")), 50),
        "ok": bool(dirs) and all(counts[k] == 50 for k in dirs),
    }


def finalize(root: Path, only: list[str] | None = None) -> dict:
    summary = {}
    for key, meta in DATASETS.items():
        if only and key not in only:
            continue
        ddir = dataset_dir(root, key)
        if not ddir.exists():
            continue
        state = ddir / ".state"
        done = {}
        if state.exists():
            for p in sorted(state.glob("*.done.json")):
                rec = read_json(p)
                if rec:
                    done[rec["item"]] = rec
        tasks_status = {}
        for t in meta["tasks"]:
            recs = [r for r in done.values() if r["task"] == t]
            if t == "uvdoc-final":
                expected = max((r.get("part_count", 0) for r in recs), default=0) or None
            elif t == "doc3d-img":
                expected = len(DOC3D_IMG_SIZES)
            else:
                expected = 1
            ok = expected is not None and len(recs) >= expected
            tasks_status[t] = {
                "state": "done" if ok else ("partial" if recs else "pending"),
                "done_items": len(recs),
                "expected_items": expected,
            }
        log(f"FINALIZE {key}: scanning {ddir}")
        contents = scan_tree(ddir)
        if key == "doc3d":
            pairing = pairing_doc3d(ddir)
        elif key == "uvdoc_final":
            pairing = pairing_uvdoc(ddir)
        else:
            pairing = pairing_benchmark(ddir)
        sources = []
        for name, rec in sorted(done.items()):
            src = {
                "item": name,
                "method": rec.get("method"),
                "finished_utc": rec.get("finished_utc"),
            }
            if "source" in rec:
                src["remote"] = rec["source"]
            if "archive" in rec:
                src["archive"] = rec["archive"]
            if "extraction" in rec:
                src["members_extracted"] = rec["extraction"]["members_extracted"]
                src["crc32_verified"] = rec["extraction"]["crc32_verified"]
            if "members" in rec:
                src["members"] = rec["members"]
                src["crc32_verified"] = True
            sources.append(src)
        if key == "uvdoc_final" and _cd_cache.get(f"{IGL}/UVDoc_final.zip"):
            info, cd = _cd_cache[f"{IGL}/UVDoc_final.zip"]
            remote_id = {
                **info.as_dict(),
                "central_directory_sha256": cd.cd_sha256,
                "central_directory_entries": len(cd.entries),
            }
        else:
            prev = read_json(ddir / "MANIFEST.json") or {}
            remote_id = prev.get("remote_archive_identity")
        complete = all(v["state"] == "done" for v in tasks_status.values()) and bool(
            pairing.get("ok")
        )
        manifest = {
            "schema": SCHEMA,
            "dataset": key,
            "title": meta["title"],
            "track": meta["track"],
            "track_note": (
                RESEARCH_NOTE if meta["track"] == "research" else meta.get("training_use")
            ),
            "release_weights_allowed": meta["release_weights_allowed"],
            "license": meta["license"],
            "location": str(ddir),
            "complete": complete,
            "tasks": tasks_status,
            "sources": sources,
            "contents": contents,
            "pairing": pairing,
            "tool": {"path": "tools/research_data/fetch_research.py", "sha256": tool_identity()},
            "updated_utc": utcnow(),
        }
        if key == "uvdoc_final":
            manifest["remote_archive_identity"] = remote_id
            manifest["selection"] = {
                "archive": f"{IGL}/UVDoc_final.zip",
                "members": [f"UVDoc_final/{p}/" for p in UVDOC_FINAL_PARTS]
                + ["UVDoc_final/split.json"],
                "not_downloaded": ["uvmap", "wc", "warped_textures", "textures", "img_geom", "seg"],
            }
        if key == "doc3d":
            manifest["hf_revision"] = REV
            manifest["split_note"] = (
                "traindoc.txt / valdoc3D.txt come from Doc3D_grid.zip (UVDoc's split). Doc3D's "
                "own guidance is to split by mesh ID (cvlab-stonybrook/doc3D-dataset issue #8)."
            )
        write_json_atomic(ddir / "MANIFEST.json", manifest)
        summary[key] = {
            "complete": complete,
            "tasks": tasks_status,
            "pairing_ok": pairing.get("ok"),
            "counts": {k: v.get("files") for k, v in contents.items()},
        }
        log(f"MANIFEST {ddir / 'MANIFEST.json'} complete={complete} pairing_ok={pairing.get('ok')}")
    return summary


# -- CLI -----------------------------------------------------------------------------------------
def load_hf_token(path: str | None) -> str | None:
    for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(env):
            return os.environ[env].strip()
    cands = [Path(path)] if path else []
    cands.append(Path.home() / ".cache" / "huggingface" / "token")
    for p in cands:
        if p.is_file():
            tok = p.read_text("utf-8").strip()
            if tok:
                return tok
    return None


def default_progress_file(root: Path) -> Path:
    job = os.environ.get("FLATTENER_SOTA_JOB") or "research-data"
    logs = root.parent / "logs"
    return (logs if logs.is_dir() else root / "research") / f"{job}.progress.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="download/extract tasks, then rebuild manifests")
    r.add_argument("--root", type=Path, required=True, help="data root (holds research/, eval/)")
    r.add_argument("--tasks", default="uvdoc-benchmark,doc3d-grid,uvdoc-final")
    r.add_argument("--connections", type=int, default=2, help="max concurrent HTTP connections")
    r.add_argument("--max-mbps", type=float, default=None, help="global cap in MB/s (optional)")
    r.add_argument("--uvdoc-parts", type=int, default=8)
    r.add_argument("--keep-archives", action="store_true")
    r.add_argument("--recheck", action="store_true", help="re-verify CRCs of existing files")
    r.add_argument("--hf-token-file", default=None)
    r.add_argument("--progress-file", type=Path, default=None)
    f = sub.add_parser("finalize", help="rebuild MANIFEST.json + pairing checks (no network)")
    f.add_argument("--root", type=Path, required=True)
    f.add_argument("--datasets", default=None)
    s = sub.add_parser("status", help="print progress file and manifest summaries")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--progress-file", type=Path, default=None)
    a = ap.parse_args(argv)

    if a.cmd == "status":
        pf = a.progress_file or default_progress_file(a.root)
        out = {"progress_file": str(pf), "progress": read_json(pf)}
        for key in DATASETS:
            m = read_json(dataset_dir(a.root, key) / "MANIFEST.json")
            if m:
                out[key] = {
                    "complete": m["complete"],
                    "tasks": m["tasks"],
                    "track": m["track"],
                    "pairing_ok": m["pairing"].get("ok"),
                    "updated_utc": m["updated_utc"],
                }
        print(json.dumps(out, indent=1))
        return 0
    if a.cmd == "finalize":
        only = a.datasets.split(",") if a.datasets else None
        print(json.dumps(finalize(a.root, only), indent=1))
        return 0

    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    ctx = Context(
        root=a.root,
        limiter=RateLimiter(a.max_mbps * 1e6 if a.max_mbps else None),
        meter=Meter(),
        stop=threading.Event(),
        keep_archives=a.keep_archives,
        hf_token=load_hf_token(a.hf_token_file) if "doc3d-img" in tasks else None,
        recheck=a.recheck,
        net=threading.BoundedSemaphore(max(1, a.connections)),
    )
    log(
        f"RUN tasks={tasks} connections={a.connections} max_mbps={a.max_mbps} root={a.root} "
        f"pid={os.getpid()} hf_token={'yes' if ctx.hf_token else 'no'} tool={tool_identity()}"
    )
    for key in DATASETS:
        if any(t in DATASETS[key]["tasks"] for t in tasks):
            d = dataset_dir(a.root, key)
            d.mkdir(parents=True, exist_ok=True)
            track_file = d / "TRACK.json"
            if not track_file.exists():
                write_json_atomic(
                    track_file,
                    {
                        "track": DATASETS[key]["track"],
                        "dataset": key,
                        "note": RESEARCH_NOTE
                        if DATASETS[key]["track"] == "research"
                        else DATASETS[key].get("training_use"),
                    },
                )
    items = build_items(ctx, tasks, a.uvdoc_parts)
    pf = a.progress_file or default_progress_file(a.root)
    try:
        rc = run_items(ctx, items, a.connections, pf)
    except KeyboardInterrupt:
        ctx.stop.set()
        rc = 3
    finally:
        for j in _journals.values():
            j.close()
    datasets = sorted({it.dataset for it in items})
    try:
        summary = finalize(a.root, datasets)
        log("SUMMARY " + json.dumps(summary))
    except Exception as exc:  # noqa: BLE001
        log(f"FINALIZE-FAILED {type(exc).__name__}: {exc}")
        traceback.print_exc()
        rc = rc or 1
    log(f"EXIT {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
