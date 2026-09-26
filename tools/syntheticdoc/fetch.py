"""Step 5: selection, download + conversion, manifests and verification.

Selection (release track): per official train shard the ``per_shard`` clean
samples of our ``train`` split with the lowest salted-hash rank (so all 1,000
shards, i.e. all six mesh-family blocks, are covered evenly), plus every clean
sample of our ``val`` split. Raising ``per_shard`` later extends the same
ranking, so existing downloads stay selected.

Download: only render.png, backward_map.npy and metadata.json. They are
contiguous in each shard (render, map, metadata), so one range per sample;
``samples_per_request`` ranges go into one multipart request. Every member is
CRC-32 checked against the central directory recorded during the harvest.
Work is interleaved across shards so a partial download is still stratified.

Output (``data/<track>/syntheticdoc``)::

    samples/<shard>/<sid>.render.jpg   RGB JPEG q95, 4:2:0, full 1024x1440 (W x H)
    samples/<shard>/<sid>.bm16.npy     float16 (177, 121, 2), last axis (x, y),
                                       RAW upstream values (see dataset-manifest)
    samples/<shard>/<sid>.json         flattener-dewarp-sample-1 record + provenance,
                                       license class, track, attribution, upstream metadata
    manifest.jsonl                     one record per downloaded sample (paths relative
                                       to this directory), read by
                                       flattener.dewarp.data.ConvertedSource
    dataset-manifest.json, license-manifest.json, progress.json, verify-report.json

Files are written atomically; the per-sample JSON is written last and the index
row is marked ``done`` only after it, so a killed job resumes cleanly.
"""

from __future__ import annotations

import io
import json
import random
import shutil
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sdcommon import (
    DATASET_DOI,
    DATASET_TOKEN,
    TOOL_VERSION,
    ConsistencyError,
    Http,
    Layout,
    RemoteFile,
    State,
    ZipEntry,
    extract_member,
    get_ranges,
    head,
    log,
    stable_unit,
    utcnow,
    write_bytes_atomic,
    write_json_atomic,
)
from sdlicense import MATSYNTH_CITATION, SYNTHETICDOC_CITATION, attribution_for

SELECT_SALT = "flattener-syntheticdoc-select-v1"
RENDER_SIZE_HW = (1440, 1024)
BM_SHAPE = (177, 121, 2)
JPEG_QUALITY = 95
SAMPLE_SCHEMA = "flattener-dewarp-sample-1"
BM_CONVENTION_TEXT = (
    "Raw SyntheticDoc backward_map.npy values, only cast to float16. Shape (177, 121, 2) "
    "= (rows, cols, xy); entry (i, j) sits at output (flat page) pixel center "
    "((2j+1)/121-1, (2i+1)/177-1) and holds the normalized render position of that page "
    "point. Upstream uv_to_backward_map.py stores b = 2x/n - 1 for render pixel-center index "
    "x and render size n (1024 wide, 1440 high); the task brief calls this the "
    "align_corners=True convention. Convert to the flattener convention (align_corners=False, "
    "+-1 = outer pixel edges) with flattener.dewarp.coords.syntheticdoc_to_flattener(values, "
    "(1440, 1024)), i.e. q = b + (1/1024, 1/1440). ConvertedSource does this automatically "
    "for backward_map_convention 'syntheticdoc' with original_render_size [1440, 1024]."
)


# --------------------------------------------------------------------------- selection


def select(
    state: State, layout: Layout | None = None, *, per_shard: int = 200, track: str = "release"
) -> dict[str, Any]:
    if track != "release":
        raise ValueError("only the release selection (clean samples) is implemented")
    rows = state.query(
        "SELECT uid, shard, our_split FROM samples WHERE lic_class='clean' AND status='success' "
        "AND split IN ('train','val') AND our_split IN ('train','val') AND r_off IS NOT NULL "
        "AND b_off IS NOT NULL AND m_off IS NOT NULL"
    )
    by_shard: dict[str, list[tuple[float, str]]] = defaultdict(list)
    val = []
    for uid, shard, our in rows:
        rank = stable_unit(uid, SELECT_SALT)
        if our == "val":
            val.append((rank, uid))
        elif shard.startswith("train_"):
            by_shard[shard].append((rank, uid))
    updates = []
    short = {}
    for shard, cands in sorted(by_shard.items()):
        cands.sort()
        if len(cands) < per_shard:
            short[shard] = len(cands)
        updates += [(f"{track}-train", rank, uid) for rank, uid in cands[:per_shard]]
    updates += [(f"{track}-val", rank, uid) for rank, uid in val]
    # Reconcile samples downloaded under an earlier split/selection: a downloaded clean sample
    # keeps a role matching its CURRENT split (extra samples are harmless); one that is no
    # longer clean train/val is orphaned (dropped from the manifest, files left on disk).
    wanted = {uid: role for role, _, uid in updates}
    fixes, orphans = [], []
    for uid, role, our, cls, shard, sid in state.query(
        "SELECT uid, sel_role, our_split, lic_class, shard, sample_id FROM samples "
        "WHERE sel_role LIKE ? AND dl_status='done'",
        (f"{track}-%",),
    ):
        want = wanted.get(uid)
        if want is None and cls == "clean" and our in ("train", "val"):
            want = f"{track}-{our}"
            updates.append((want, stable_unit(uid, SELECT_SALT), uid))
        if want is None:
            orphans.append(uid)
        elif want != role:
            fixes.append((shard, sid, want.split("-", 1)[1]))
    train_shards = state.query("SELECT COUNT(*) FROM shards WHERE split='train'")[0][0]
    with state.lock:
        state.db.execute(
            "UPDATE samples SET sel_role=NULL, sel_rank=NULL WHERE sel_role LIKE ?",
            (f"{track}-%",),
        )
        state.db.executemany("UPDATE samples SET sel_role=?, sel_rank=? WHERE uid=?", updates)
        state.db.executemany(
            "UPDATE samples SET dl_status='orphaned' WHERE uid=?", [(u,) for u in orphans]
        )
        state.db.execute(  # re-selected after being orphaned: fetch again
            "UPDATE samples SET dl_status=NULL WHERE dl_status='orphaned' AND sel_role IS NOT NULL"
        )
        state.db.commit()
    if fixes and layout is not None:
        root = layout.data_root / track / "syntheticdoc" / "samples"
        for shard, sid, split in fixes:
            path = root / shard / f"{sid}.json"
            if path.is_file():
                record = json.loads(path.read_text(encoding="utf-8"))
                record["split"] = split
                write_json_atomic(path, record)
    if fixes or orphans:
        log(f"selection reconciled {len(fixes)} downloaded samples, orphaned {len(orphans)}")
    counts = dict(
        state.query(
            "SELECT sel_role, COUNT(*) FROM samples WHERE sel_role LIKE ? GROUP BY sel_role",
            (f"{track}-%",),
        )
    )
    info = {
        "at": utcnow(),
        "track": track,
        "per_shard": per_shard,
        "salt": SELECT_SALT,
        "policy": "per official train shard: the per_shard clean samples of our train split "
        "with the lowest salted SHA-256 rank; plus every clean sample of our val split",
        "train_shards_with_candidates": len(by_shard),
        "train_shards_total": train_shards,
        "shards_short": short,
        "reconciled": len(fixes),
        "orphaned": len(orphans),
        "counts": counts,
    }
    state.put(f"selection.{track}", info)
    log("selection", counts, f"short shards: {len(short)}")
    return info


# --------------------------------------------------------------------------- conversion


def convert_sample(render_png: bytes, bm_npy: bytes) -> tuple[bytes, bytes, dict[str, Any]]:
    with Image.open(io.BytesIO(render_png)) as im:
        im.load()
        mode, size = im.mode, im.size
        rgb = im if im.mode == "RGB" else im.convert("RGB")
        out = io.BytesIO()
        rgb.save(out, "JPEG", quality=JPEG_QUALITY, subsampling=2, optimize=False)
    jpeg = out.getvalue()
    bm = np.load(io.BytesIO(bm_npy), allow_pickle=False)
    if bm.shape != BM_SHAPE or bm.dtype != np.float32:
        raise ConsistencyError(f"backward map shape/dtype {bm.shape}/{bm.dtype}")
    finite = np.isfinite(bm)
    bm16 = bm.astype(np.float16)
    if not np.array_equal(np.isfinite(bm16), finite):
        raise ConsistencyError("float16 cast changed finiteness")
    err = np.abs(bm16.astype(np.float32) - bm)
    err = np.where(finite, err, 0.0)
    buf = io.BytesIO()
    np.save(buf, bm16, allow_pickle=False)
    w, h = size
    info = {
        "render_png": {"mode": mode, "size_wh": [w, h], "bytes": len(render_png)},
        "image": {
            "format": "JPEG",
            "quality": JPEG_QUALITY,
            "subsampling": "4:2:0",
            "size_wh": [w, h],
            "bytes": len(jpeg),
        },
        "backward_map": {
            "dtype": "float16",
            "shape": list(BM_SHAPE),
            "nonfinite_values": int((~finite).sum()),
            "range": [float(np.nanmin(bm)), float(np.nanmax(bm))] if finite.any() else None,
            "fp16_max_abs_error": float(err.max()),
            "fp16_max_error_render_px_xy": [
                float(err[..., 0].max() * w / 2),
                float(err[..., 1].max() * h / 2),
            ],
        },
    }
    return jpeg, buf.getvalue(), info


# --------------------------------------------------------------------------- download

_SAMPLE_COLS = (
    "uid, split, shard, sample_id, r_off, r_size, r_crc, r_method, b_off, b_size, b_crc, "
    "b_method, m_off, m_size, m_crc, m_method, blk_start, blk_end, lic_class, doc_ref, "
    "doc_license, bg_license, bg_name, doc_group, mesh_group, mesh_family, mesh_id, doc_source, "
    "doc_name, test_overlap, our_split, sel_role"
)


def _entries(row: dict[str, Any]) -> dict[str, ZipEntry]:
    sid = row["sample_id"]
    out = {}
    for key, name in (("r", "render.png"), ("b", "backward_map.npy"), ("m", "metadata.json")):
        size = row[f"{key}_size"]
        out[name] = ZipEntry(
            f"{sid}/{name}",
            row[f"{key}_method"],
            0,
            row[f"{key}_crc"],
            size,
            size,
            row[f"{key}_off"],
        )
    return out


def _ranges(row: dict[str, Any], entries: dict[str, ZipEntry], cd_offset: int):
    if row["blk_start"] is not None:
        return [(row["blk_start"], row["blk_end"])]
    spans = []
    for entry in entries.values():  # rare fallback: members not contiguous
        end = min(
            entry.offset + 30 + len(entry.name.encode()) + 1024 + entry.csize - 1, cd_offset - 1
        )
        spans.append((entry.offset, end))
    return spans


class Downloader:
    def __init__(
        self,
        client: Http,
        state: State,
        layout: Layout,
        *,
        track: str,
        connections: int,
        samples_per_request: int,
        min_free_gb: float,
        manifest_every_s: float = 1800.0,
        max_attempts: int = 5,
    ) -> None:
        self.client, self.state, self.layout, self.track = client, state, layout, track
        self.root = layout.data_root / track / "syntheticdoc"
        self.connections = connections
        self.samples_per_request = samples_per_request
        self.min_free_gb = min_free_gb
        self.manifest_every_s = manifest_every_s
        self.max_attempts = max_attempts
        self.lock = threading.Lock()
        self.shard_checked: dict[str, float] = {}
        self.shard_info = {
            s: (path, size, etag, cd)
            for s, path, size, etag, cd in state.query(
                "SELECT shard, path, size, etag, cd_offset FROM shards"
            )
        }
        self.done = 0
        self.failed = 0
        self.src_bytes = 0
        self.out_bytes = 0
        self.t0 = time.monotonic()
        self.stop = False

    def _check_shard(self, shard: str) -> None:
        path, size, etag, _ = self.shard_info[shard]
        with self.lock:
            last = self.shard_checked.get(shard)
        if last is not None and time.monotonic() - last < 3600:
            return
        info = head(self.client, RemoteFile(DATASET_TOKEN, path))
        if (info.size, info.etag) != (size, etag):
            raise ConsistencyError(f"{shard}: upstream changed ({info.etag}); re-harvest needed")
        with self.lock:
            self.shard_checked[shard] = time.monotonic()

    def _write_sample(
        self, row: dict[str, Any], render: bytes, bm: bytes, meta_raw: bytes, shard_etag: str
    ) -> int:
        jpeg, bm16, conv = convert_sample(render, bm)
        if conv["render_png"]["size_wh"] != [RENDER_SIZE_HW[1], RENDER_SIZE_HW[0]]:
            raise ConsistencyError(f"render size {conv['render_png']['size_wh']}")
        sid, shard = row["sample_id"], row["shard"]
        folder = self.root / "samples" / shard
        stem = sid
        role_split = row["sel_role"].split("-", 1)[1]
        record = {
            "schema": SAMPLE_SCHEMA,
            "sample_id": f"syntheticdoc-{row['split']}-{sid}",
            "split": role_split,
            "image": f"{stem}.render.jpg",
            "backward_map": f"{stem}.bm16.npy",
            "backward_map_convention": "syntheticdoc",
            "original_render_size": list(RENDER_SIZE_HW),
            "track": self.track,
            "license_class": row["lic_class"],
            "dataset": "SyntheticDoc",
            "official": {
                "split": row["split"],
                "shard": shard,
                "sample_id": sid,
                "shard_etag": shard_etag,
                "members": {
                    "render.png": {"crc32": f"{row['r_crc']:08x}", "size": row["r_size"]},
                    "backward_map.npy": {"crc32": f"{row['b_crc']:08x}", "size": row["b_size"]},
                    "metadata.json": {"crc32": f"{row['m_crc']:08x}", "size": row["m_size"]},
                },
            },
            "groups": {
                "doc_group": row["doc_group"],
                "mesh_group": row["mesh_group"],
                "mesh_family": row["mesh_family"],
                "doc_source": row["doc_source"],
                "official_test_overlap": {
                    "mesh": bool(row["test_overlap"] & 1),
                    "document": bool(row["test_overlap"] & 2),
                },
            },
            "license": {
                "class": row["lic_class"],
                "dataset": "CC-BY-4.0",
                "document": row["doc_license"],
                "background": row["bg_license"],
                "mesh": "CC-BY-4.0",
            },
            "attribution": attribution_for(self.state, row["doc_ref"], row["bg_name"]),
            "conversion": conv,
            "syntheticdoc_metadata": json.loads(meta_raw),
            "tool": TOOL_VERSION,
            "downloaded_at": utcnow(),
        }
        write_bytes_atomic(folder / f"{stem}.render.jpg", jpeg)
        write_bytes_atomic(folder / f"{stem}.bm16.npy", bm16)
        write_json_atomic(folder / f"{stem}.json", record)
        return len(jpeg) + len(bm16)

    def fetch_batch(self, shard: str, rows: list[dict[str, Any]]) -> tuple[int, int]:
        path, size, etag, cd_offset = self.shard_info[shard]
        self._check_shard(shard)
        remote = RemoteFile(DATASET_TOKEN, path)
        plan = []
        ranges = []
        for row in rows:
            entries = _entries(row)
            spans = _ranges(row, entries, cd_offset)
            plan.append((row, entries, spans))
            ranges += spans
        got = get_ranges(self.client, remote, ranges, total=size)
        ok = bad = 0
        for row, entries, spans in plan:
            try:
                parts = {}
                for name, entry in entries.items():
                    for a, b in spans:
                        if a <= entry.offset <= b:
                            parts[name] = extract_member(got[(a, b)], entry.offset - a, entry)
                            break
                    else:
                        raise ConsistencyError(f"{name} not inside fetched ranges")
                nbytes = self._write_sample(
                    row,
                    parts["render.png"],
                    parts["backward_map.npy"],
                    parts["metadata.json"],
                    etag,
                )
                src = sum(b - a + 1 for a, b in spans)
                self.state.execute(
                    "UPDATE samples SET dl_status='done', dl_at=?, dl_error=NULL WHERE uid=?",
                    (utcnow(), row["uid"]),
                )
                with self.lock:
                    self.done += 1
                    self.src_bytes += src
                    self.out_bytes += nbytes
                ok += 1
            except Exception as exc:  # noqa: BLE001 - record per sample, retried next round
                bad += 1
                self.state.execute(
                    "UPDATE samples SET dl_status='error', dl_error=?, "
                    "dl_attempts=COALESCE(dl_attempts,0)+1 WHERE uid=?",
                    (f"{type(exc).__name__}: {exc}"[:500], row["uid"]),
                )
                with self.lock:
                    self.failed += 1
                log(f"sample {row['uid']} failed: {type(exc).__name__}: {exc}")
        return ok, bad

    def plan(self) -> list[tuple[str, list[dict[str, Any]]]]:
        rows = self.state.query(
            f"SELECT {_SAMPLE_COLS} FROM samples WHERE sel_role LIKE ? AND "
            "(dl_status IS NULL OR dl_status = 'error') AND COALESCE(dl_attempts, 0) < ?",
            (f"{self.track}-%", self.max_attempts),
        )
        names = _SAMPLE_COLS.replace(" ", "").split(",")
        by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for values in rows:
            row = dict(zip(names, values))
            by_shard[row["shard"]].append(row)
        batches: list[tuple[int, int, str, list[dict[str, Any]]]] = []
        for shard, items in by_shard.items():
            # val first within a shard, then by selection rank
            items.sort(key=lambda r: (r["sel_role"] != f"{self.track}-val", r["uid"]))
            order = stable_unit(shard, SELECT_SALT)
            for k in range(0, len(items), self.samples_per_request):
                batches.append((k, order, shard, items[k : k + self.samples_per_request]))
        batches.sort(key=lambda b: (b[0], b[1]))
        return [(shard, items) for _, _, shard, items in batches]

    def progress(self, total: int, final: bool = False) -> dict[str, Any]:
        elapsed = max(time.monotonic() - self.t0, 1e-9)
        with self.lock:
            done, failed, src, out = self.done, self.failed, self.src_bytes, self.out_bytes
        rate = done / elapsed
        counts = dict(
            self.state.query(
                "SELECT sel_role || ':' || COALESCE(dl_status, 'pending'), COUNT(*) FROM samples "
                "WHERE sel_role LIKE ? GROUP BY sel_role, dl_status",
                (f"{self.track}-%",),
            )
        )
        free = shutil.disk_usage(self.root).free
        snap = self.client.stats.snapshot()
        info = {
            "track": self.track,
            "step": "download",
            "at": utcnow(),
            "final": final,
            "this_run": {
                "samples_done": done,
                "samples_failed": failed,
                "samples_planned": total,
                "elapsed_s": round(elapsed, 1),
                "samples_per_s": round(rate, 3),
                "source_mb_per_s": round(src / elapsed / 1e6, 3),
                "stored_mb_per_s": round(out / elapsed / 1e6, 3),
                "stored_bytes_per_sample": round(out / done) if done else None,
                "eta_hours": round((total - done - failed) / rate / 3600, 2) if rate else None,
            },
            "http": snap,
            "index_counts": counts,
            "disk_free_gb": round(free / 1e9, 1),
        }
        try:
            write_json_atomic(self.root / "progress.json", info)
        except OSError as exc:  # informational file; never fail the job over it
            log(f"progress.json not written: {exc}")
        return info

    def run(self) -> bool:
        self.root.mkdir(parents=True, exist_ok=True)
        batches = self.plan()
        total = sum(len(items) for _, items in batches)
        log(
            f"download[{self.track}]: {total} samples in {len(batches)} requests, "
            f"{self.connections} connections"
        )
        if not batches:
            return True
        last_log = last_manifest = time.monotonic()
        with ThreadPoolExecutor(max_workers=self.connections) as pool:
            pending: set = set()
            queue = list(reversed(batches))
            while (queue or pending) and not self.stop:
                while queue and len(pending) < self.connections and not self.stop:
                    shard, items = queue.pop()
                    fut = pool.submit(self.fetch_batch, shard, items)
                    fut.meta = (shard, items)  # type: ignore[attr-defined]
                    pending.add(fut)
                done, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001 - whole batch retried next round
                        shard, items = fut.meta  # type: ignore[attr-defined]
                        log(f"batch {shard} x{len(items)} failed: {exc}")
                        uids = [r["uid"] for r in items]
                        self.state.executemany(
                            "UPDATE samples SET dl_status='error', dl_error=?, "
                            "dl_attempts=COALESCE(dl_attempts,0)+1 WHERE uid=?",
                            [(f"batch: {exc}"[:500], u) for u in uids],
                        )
                        with self.lock:
                            self.failed += len(items)
                now = time.monotonic()
                if now - last_log > 60:
                    last_log = now
                    info = self.progress(total)
                    run = info["this_run"]
                    log(
                        f"download {run['samples_done']}/{total} "
                        f"({run['samples_failed']} failed), {run['samples_per_s']} samples/s, "
                        f"{run['source_mb_per_s']} MB/s source, ETA {run['eta_hours']} h, "
                        f"free {info['disk_free_gb']} GB"
                    )
                    if info["disk_free_gb"] < self.min_free_gb:
                        log(f"STOP: free space below {self.min_free_gb} GB")
                        self.stop = True
                if now - last_manifest > self.manifest_every_s:
                    last_manifest = now
                    build_manifests(self.state, self.layout, self.track)
        self.progress(total, final=True)
        # errored samples get more attempts on the next run; give up after max_attempts
        self.state.execute(
            "UPDATE samples SET dl_status='failed' WHERE dl_status='error' AND sel_role LIKE ? "
            "AND COALESCE(dl_attempts, 0) >= ?",
            (f"{self.track}-%", self.max_attempts),
        )
        left = self.state.query(
            "SELECT COUNT(*) FROM samples WHERE sel_role LIKE ? AND "
            "(dl_status IS NULL OR dl_status='error')",
            (f"{self.track}-%",),
        )[0][0]
        return left == 0 and not self.stop


def download(
    client: Http,
    state: State,
    layout: Layout,
    *,
    track: str,
    connections: int,
    samples_per_request: int,
    min_free_gb: float,
    rounds: int = 4,
    round_pause: float = 300.0,
) -> bool:
    ok = False
    for round_no in range(1, rounds + 1):
        dl = Downloader(
            client,
            state,
            layout,
            track=track,
            connections=connections,
            samples_per_request=samples_per_request,
            min_free_gb=min_free_gb,
        )
        ok = dl.run()
        build_manifests(state, layout, track)
        if ok or dl.stop:
            break
        if round_no < rounds:
            log(f"download round {round_no} left errors; retrying after {round_pause:.0f}s")
            time.sleep(round_pause)
    return ok


# --------------------------------------------------------------------------- manifests


def build_manifests(state: State, layout: Layout, track: str = "release") -> dict[str, Any]:
    root = layout.data_root / track / "syntheticdoc"
    rows = state.query(
        "SELECT uid, split, shard, sample_id, sel_role, lic_class, doc_group, mesh_group, "
        "mesh_family, doc_source, doc_license, bg_license, doc_ref, bg_name, test_overlap, "
        "orientation FROM samples WHERE sel_role LIKE ? AND dl_status='done' ORDER BY shard, uid",
        (f"{track}-%",),
    )
    lines = []
    counts: Counter = Counter()
    fam: Counter = Counter()
    src: Counter = Counter()
    doc_lic: Counter = Counter()
    bg_lic: Counter = Counter()
    doc_refs: Counter = Counter()
    bgs: Counter = Counter()
    for (
        uid,
        split,
        shard,
        sid,
        role,
        cls,
        dgroup,
        mgroup,
        family,
        source,
        dlic,
        blic,
        ref,
        bg,
        overlap,
        orientation,
    ) in rows:
        our = role.split("-", 1)[1]
        lines.append(
            json.dumps(
                {
                    "schema": SAMPLE_SCHEMA,
                    "sample_id": f"syntheticdoc-{split}-{sid}",
                    "split": our,
                    "image": f"samples/{shard}/{sid}.render.jpg",
                    "backward_map": f"samples/{shard}/{sid}.bm16.npy",
                    "meta": f"samples/{shard}/{sid}.json",
                    "backward_map_convention": "syntheticdoc",
                    "original_render_size": list(RENDER_SIZE_HW),
                    "track": track,
                    "license_class": cls,
                    "official_split": split,
                    "shard": shard,
                    "doc_group": dgroup,
                    "mesh_group": mgroup,
                    "mesh_family": family,
                    "doc_source": source,
                    "doc_license": dlic,
                    "background_license": blic,
                    "official_test_overlap": overlap,
                    "orientation": orientation,
                },
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        counts[our] += 1
        fam[f"{our}:{family}"] += 1
        src[f"{our}:{source}"] += 1
        doc_lic[dlic] += 1
        bg_lic[blic] += 1
        if ref:
            doc_refs[ref] += 1
        if bg:
            bgs[bg] += 1
    root.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(
        root / "manifest.jsonl", ("\n".join(lines) + "\n").encode() if lines else b""
    )
    selection = state.get(f"selection.{track}", {})
    split_info = state.get("split", {})
    tables = state.get("license_tables", {})
    selected = dict(
        state.query(
            "SELECT sel_role, COUNT(*) FROM samples WHERE sel_role LIKE ? GROUP BY sel_role",
            (f"{track}-%",),
        )
    )
    dataset_manifest = {
        "schema": "flattener-syntheticdoc-dataset-manifest-1",
        "track": track,
        "name": f"syntheticdoc-{track}",
        "generated_at": utcnow(),
        "tool": TOOL_VERSION,
        "source": {
            "dataset": "SyntheticDoc",
            "doi": DATASET_DOI,
            "url": "https://libdrive.ethz.ch/index.php/s/AmgqqJb6owHMZBn",
            "license": "CC-BY-4.0",
            "citation": SYNTHETICDOC_CITATION,
            "members_used": ["render.png", "backward_map.npy", "metadata.json"],
            "members_skipped": ["uv_inverse.exr", "albedo.png", "shadow.png"],
        },
        "complete": sum(counts.values()) == sum(selected.values()),
        "counts": {"downloaded": dict(counts), "selected": selected},
        "by_split_and_family": dict(sorted(fam.items())),
        "by_split_and_document_source": dict(sorted(src.items())),
        "files": {
            "manifest": "manifest.jsonl (flattener-dewarp-sample-1 records; paths relative to "
            "this directory; loader: flattener.dewarp.data.ConvertedSource)",
            "image": {
                "pattern": "samples/<shard>/<sid>.render.jpg",
                "format": "JPEG",
                "quality": JPEG_QUALITY,
                "chroma_subsampling": "4:2:0",
                "mode": "RGB",
                "size_wh": [RENDER_SIZE_HW[1], RENDER_SIZE_HW[0]],
                "note": "full upstream resolution, decoded from render.png (no resize)",
            },
            "backward_map": {
                "pattern": "samples/<shard>/<sid>.bm16.npy",
                "dtype": "float16",
                "shape": list(BM_SHAPE),
                "axes": "(rows, cols, xy)",
                "convention": "syntheticdoc",
                "original_render_size_hw": list(RENDER_SIZE_HW),
                "description": BM_CONVENTION_TEXT,
                "float16_error": "max |fp16 - fp32| <= 2**-11 in normalized units, i.e. up to "
                "~0.25 px in x and ~0.35 px in y at 1024x1440 (measured on 200 local samples: "
                "0.125 / 0.176 px); per-sample values in conversion.backward_map",
            },
            "sample_json": "samples/<shard>/<sid>.json: record + upstream metadata, CRC-32 of "
            "upstream members, groups, license, attribution",
        },
        "split": {
            "policy": "own split grouped by document group AND mesh group (see research "
            "reports/split-report.json); official test split never used",
            **split_info,
        },
        "selection": selection,
        "license_tables": {
            k: {kk: v.get(kk) for kk in ("etag", "sha256", "size")}
            for k, v in (tables.get("files") or {}).items()
        },
    }
    write_json_atomic(root / "dataset-manifest.json", dataset_manifest)
    # license manifest with attribution lists
    docs = []
    for ref, n in sorted(doc_refs.items()):
        row = state.query(
            "SELECT source, license_id, license_url, title, authors, url, citation FROM docs "
            "WHERE doc_ref=?",
            (ref,),
        )
        if row:
            s, spdx, url, title, authors, link, citation = row[0]
            docs.append(
                {
                    "ref": ref,
                    "samples": n,
                    "source": s,
                    "license": spdx,
                    "license_url": url,
                    "title": title,
                    "authors": json.loads(authors) if authors else None,
                    "url": link,
                    "citation": citation,
                }
            )
    by_bg_author = []
    for name, n in sorted(bgs.items()):
        row = state.query("SELECT license, authors, source FROM backgrounds WHERE name=?", (name,))
        if row and row[0][0] not in ("CC0-1.0",):
            by_bg_author.append(
                {
                    "material": name,
                    "samples": n,
                    "license": row[0][0],
                    "authors": json.loads(row[0][1]),
                    "source": row[0][2],
                }
            )
    license_manifest = {
        "schema": "flattener-syntheticdoc-license-manifest-1",
        "track": track,
        "generated_at": utcnow(),
        "policy": "release track: every sample is class 'clean' (document CC BY / CC0 / "
        "ODC-BY / CC BY 4.0 FLUX outputs, background CC0 / CC BY, meshes CC BY 4.0); no NC, "
        "ND, SA or unknown components"
        if track == "release"
        else "research track",
        "class_counts": dict(Counter(r[5] for r in rows)),
        "document_licenses": dict(doc_lic),
        "background_licenses": dict(bg_lic),
        "notices": {
            "dataset": SYNTHETICDOC_CITATION,
            "backgrounds": MATSYNTH_CITATION + " Deschaintre et al., Single-Image SVBRDF Capture "
            "with a Rendering-Aware Deep Network, ACM TOG 37(4), 2018.",
            "meshes": "SyntheticDoc meshes, CC BY 4.0 (simulated with ARCSim).",
        },
        "attribution_documents": docs,
        "attribution_backgrounds_non_cc0": by_bg_author,
    }
    write_json_atomic(root / "license-manifest.json", license_manifest)
    return {"downloaded": dict(counts), "selected": selected}


# --------------------------------------------------------------------------- verification


def verify(
    client: Http,
    state: State,
    layout: Layout,
    *,
    track: str = "release",
    local: int = 64,
    remote: int = 6,
    seed: int = 0,
) -> dict[str, Any]:
    """Decode a random sample of outputs; for a few, re-fetch upstream members and compare."""
    root = layout.data_root / track / "syntheticdoc"
    rows = state.query(
        f"SELECT {_SAMPLE_COLS} FROM samples WHERE sel_role LIKE ? AND dl_status='done'",
        (f"{track}-%",),
    )
    names = _SAMPLE_COLS.replace(" ", "").split(",")
    rows = [dict(zip(names, r)) for r in rows]
    rng = random.Random(seed)
    picks = rng.sample(rows, min(local, len(rows)))
    results = []
    manifest_ids = set()
    mpath = root / "manifest.jsonl"
    if mpath.is_file():
        for line in mpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                manifest_ids.add(json.loads(line)["sample_id"])
    for k, row in enumerate(picks):
        folder = root / "samples" / row["shard"]
        sid = row["sample_id"]
        res: dict[str, Any] = {"uid": row["uid"]}
        try:
            with Image.open(folder / f"{sid}.render.jpg") as im:
                im.load()
                res["jpeg"] = {"mode": im.mode, "size_wh": list(im.size)}
                jpeg_arr = np.asarray(im.convert("RGB"), dtype=np.float32)
            bm16 = np.load(folder / f"{sid}.bm16.npy", allow_pickle=False)
            meta = json.loads((folder / f"{sid}.json").read_text(encoding="utf-8"))
            res["bm16"] = {
                "shape": list(bm16.shape),
                "dtype": str(bm16.dtype),
                "finite": bool(np.isfinite(bm16).all()),
            }
            res["json_ok"] = meta["schema"] == SAMPLE_SCHEMA and meta["track"] == track
            if track == "release":
                res["json_ok"] = res["json_ok"] and meta["license_class"] == "clean"
            res["in_manifest"] = f"syntheticdoc-{row['split']}-{sid}" in manifest_ids
            if k < remote:
                path, size, etag, cd = state.query(
                    "SELECT path, size, etag, cd_offset FROM shards WHERE shard=?", (row["shard"],)
                )[0]
                entries = _entries(row)
                spans = _ranges(row, entries, cd)
                got = get_ranges(client, RemoteFile(DATASET_TOKEN, path), spans, total=size)
                parts = {}
                for name, entry in entries.items():
                    for a, b in spans:
                        if a <= entry.offset <= b:
                            parts[name] = extract_member(got[(a, b)], entry.offset - a, entry)
                res["upstream_crc32_ok"] = True  # extract_member raised otherwise
                with Image.open(io.BytesIO(parts["render.png"])) as im:
                    ref = np.asarray(im.convert("RGB"), dtype=np.float32)
                mse = float(np.mean((ref - jpeg_arr) ** 2))
                res["jpeg_psnr_db"] = round(10 * np.log10(255.0**2 / max(mse, 1e-12)), 2)
                bm = np.load(io.BytesIO(parts["backward_map.npy"]), allow_pickle=False)
                diff = np.abs(bm16.astype(np.float32) - bm)
                res["bm_fp16_max_abs_error"] = float(np.nanmax(diff))
                res["metadata_equal"] = (
                    json.loads(parts["metadata.json"]) == meta["syntheticdoc_metadata"]
                )
                res["upstream_crc32"] = {n: f"{e.crc:08x}" for n, e in entries.items()}
            res["ok"] = (
                res["jpeg"]["size_wh"] == [RENDER_SIZE_HW[1], RENDER_SIZE_HW[0]]
                and res["bm16"]["shape"] == list(BM_SHAPE)
                and res["bm16"]["dtype"] == "float16"
                and res["json_ok"]
                and res.get("metadata_equal", True)
                and res.get("jpeg_psnr_db", 99) > 35
                and res.get("bm_fp16_max_abs_error", 0) < 1e-3
            )
        except Exception as exc:  # noqa: BLE001 - report every failure
            res["ok"] = False
            res["error"] = f"{type(exc).__name__}: {exc}"
        results.append(res)
    report = {
        "schema": "flattener-syntheticdoc-verify-report-1",
        "track": track,
        "generated_at": utcnow(),
        "checked": len(results),
        "remote_rechecked": sum(1 for r in results if "upstream_crc32_ok" in r),
        "all_ok": all(r["ok"] for r in results) and bool(results),
        "results": results,
    }
    write_json_atomic(root / "verify-report.json", report)
    log(
        f"verify[{track}]: {report['checked']} checked, {report['remote_rechecked']} re-fetched, "
        f"all_ok={report['all_ok']}"
    )
    return report


def local_dir(layout: Layout, track: str) -> Path:
    return layout.data_root / track / "syntheticdoc"
