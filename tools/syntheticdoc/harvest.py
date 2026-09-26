"""Step 2: harvest the central directory and every sample's metadata.json.

For each train/val/test shard (``<split>/<split>_NNN.zip``, about 6.4 GB and
1,000 samples each) this reads only the zip central directory and the
metadata.json members (about 1.3 KB each, fetched ~300 per multipart request).
Renders, maps and ``uv_inverse.exr`` are never read here.

Per sample the index keeps: member offsets, sizes, CRC-32 and methods of
render.png / backward_map.npy / metadata.json, the contiguous byte block
render..metadata used later for a one-range download, the parsed
license-relevant fields (mesh family/id, document source/name, background
material, orientation, status) and the original metadata bytes (zlib).

Consistency: the shard's ETag/size from the PROPFIND listing must match a
WebDAV HEAD before and after the harvest; every multipart part must report the
same total size; every metadata member must pass its CRC-32. A shard whose
ETag changed later is re-listed as ``changed`` and harvested again.
Resumable: shards already ``harvested`` are skipped.
"""

from __future__ import annotations

import json
import re
import threading
import time
import zlib
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from sdcommon import (
    DATASET_TOKEN,
    ConsistencyError,
    Http,
    Layout,
    RemoteFile,
    State,
    extract_member,
    get_ranges,
    head,
    log,
    member_spans,
    propfind,
    read_central_directory,
    utcnow,
    write_json_atomic,
)

SPLITS = ("train", "val", "test")
EXPECTED_FILES = frozenset(
    {
        "uv_inverse.exr",
        "render.png",
        "backward_map.npy",
        "metadata.json",
        "shadow.png",
        "albedo.png",
    }
)
_SHARD = re.compile(r"^(train|val|test)/(train|val|test)_(\d+)\.zip$")
_RESULTS = re.compile(r"/results/([^/]+)/(.+)$")


def list_shards(client: Http, state: State) -> dict[str, int]:
    """PROPFIND every split; insert new shards and flag changed ones for re-harvest."""
    counts = {}
    now = utcnow()
    for split in SPLITS:
        rows = [r for r in propfind(client, DATASET_TOKEN, split) if not r.is_dir]
        shards = []
        for row in rows:
            match = _SHARD.match(row.path)
            if match and match[1] == split and match[2] == split:
                shards.append((row, int(match[3])))
        counts[split] = len(shards)
        for row, idx in shards:
            name = row.path.rsplit("/", 1)[-1][: -len(".zip")]
            old = state.query("SELECT etag, size, status FROM shards WHERE shard=?", (name,))
            if not old:
                state.execute(
                    "INSERT INTO shards(shard, split, idx, path, size, etag, last_modified, "
                    "listed_at, status) VALUES (?,?,?,?,?,?,?,?, 'pending')",
                    (name, split, idx, row.path, row.size, row.etag, row.last_modified, now),
                )
            elif (old[0][0], old[0][1]) != (row.etag, row.size):
                log(f"shard {name} changed upstream: {old[0][0]} -> {row.etag}; re-harvest")
                state.execute(
                    "UPDATE shards SET size=?, etag=?, last_modified=?, listed_at=?, "
                    "status='changed', attempts=0, error=NULL WHERE shard=?",
                    (row.size, row.etag, row.last_modified, now, name),
                )
    state.put("listing", {"at": now, "counts": counts})
    log("listing", counts)
    return counts


def parse_metadata(raw: bytes) -> dict[str, Any]:
    meta = json.loads(raw)
    files = meta.get("files") or {}
    mesh = str(files.get("mesh") or "")
    match = _RESULTS.search(mesh)
    parts = match.groups() if match else tuple(mesh.rsplit("/", 2)[-2:])
    family = parts[0] if len(parts) == 2 else ""
    mesh_id = parts[-1] if parts else ""
    mesh_id = mesh_id[:-4] if mesh_id.endswith(".obj") else mesh_id
    document = str(files.get("document") or "")
    match = _RESULTS.search(document)
    doc = match.groups() if match else tuple(document.rsplit("/", 2)[-2:])
    texture = str(files.get("surface_texture") or "").rstrip("/")
    orientation = meta.get("orientation")
    return {
        "status": str(meta.get("status", "")),
        "mesh_family": family,
        "mesh_id": mesh_id,
        "doc_source": doc[0] if len(doc) == 2 else "",
        "doc_name": doc[-1] if doc else "",
        "bg_name": texture.rsplit("/", 1)[-1],
        "orientation": orientation if isinstance(orientation, int) else None,
    }


def harvest_shard(client: Http, state: State, shard: str) -> dict[str, Any]:
    started = time.monotonic()
    bytes0 = client.stats.bytes
    ((split, path, size, etag),) = state.query(
        "SELECT split, path, size, etag FROM shards WHERE shard=?", (shard,)
    )
    remote = RemoteFile(DATASET_TOKEN, path)
    before = head(client, remote)
    if (before.size, before.etag) != (size, etag):
        log(
            f"{shard}: HEAD {before.etag}/{before.size} differs from listing {etag}/{size}; "
            "harvesting the current version"
        )
        size, etag = before.size, before.etag
    entries, cd_offset, cd_size = read_central_directory(client, remote, size, etag)
    spans = member_spans(entries, cd_offset)
    by_sample: dict[str, dict[str, Any]] = defaultdict(dict)
    for entry in entries:
        if entry.name.endswith("/"):
            continue
        sid, _, fname = entry.name.partition("/")
        by_sample[sid][fname] = entry
    wanted = {
        sid: files["metadata.json"] for sid, files in by_sample.items() if "metadata.json" in files
    }
    got = get_ranges(client, remote, [spans[e.name] for e in wanted.values()], total=size)
    rows = []
    for sid in sorted(by_sample):
        files = by_sample[sid]
        r, b, m = (files.get(k) for k in ("render.png", "backward_map.npy", "metadata.json"))
        parsed: dict[str, Any] = {
            "status": "incomplete",
            "mesh_family": None,
            "mesh_id": None,
            "doc_source": None,
            "doc_name": None,
            "bg_name": None,
            "orientation": None,
        }
        meta_z = None
        if m is not None:
            raw = extract_member(got[spans[m.name]], 0, m)
            meta_z = zlib.compress(raw, 6)
            parsed = parse_metadata(raw)
        blk = (None, None)
        if r and b and m:
            (r0, r1), (b0, b1), (m0, m1) = spans[r.name], spans[b.name], spans[m.name]
            if r1 + 1 == b0 and b1 + 1 == m0:
                blk = (r0, m1)
        else:
            parsed["status"] = "incomplete"
        extra = sorted(set(files) - EXPECTED_FILES)
        missing = sorted(EXPECTED_FILES - set(files))
        members = json.dumps({"missing": missing, "extra": extra}) if (missing or extra) else None

        def cols(e):
            return (e.offset, e.csize, e.crc, e.method) if e else (None, None, None, None)

        rows.append(
            (
                f"{split}/{sid}",
                split,
                shard,
                sid,
                *cols(r),
                *cols(b),
                *cols(m),
                *blk,
                members,
                parsed["status"],
                parsed["mesh_family"],
                parsed["mesh_id"],
                parsed["doc_source"],
                parsed["doc_name"],
                parsed["bg_name"],
                parsed["orientation"],
                meta_z,
            )
        )
    after = head(client, remote)
    if (after.size, after.etag) != (size, etag):
        raise ConsistencyError(f"{shard}: ETag/size changed during harvest ({after.etag})")
    elapsed = time.monotonic() - started
    nbytes = client.stats.bytes - bytes0  # approximate under concurrency; informational only
    with state.lock:
        dup = state.db.execute(
            f"SELECT uid, shard FROM samples WHERE uid IN ({','.join('?' * len(rows))}) "
            "AND shard != ?",
            [row[0] for row in rows] + [shard],
        ).fetchall()
        if dup:
            taken = {uid for uid, _ in dup}
            log(f"WARNING {shard}: {len(dup)} sample ids also in other shards, e.g. {dup[:3]}")
            rows = [(f"{row[0]}@{shard}" if row[0] in taken else row[0], *row[1:]) for row in rows]
        state.db.execute("DELETE FROM samples WHERE shard=?", (shard,))
        state.db.executemany(
            "INSERT INTO samples(uid, split, shard, sample_id, r_off, r_size, r_crc, r_method, "
            "b_off, b_size, b_crc, b_method, m_off, m_size, m_crc, m_method, blk_start, blk_end, "
            "members, status, mesh_family, mesh_id, doc_source, doc_name, bg_name, orientation, "
            "meta_z) VALUES (" + ",".join("?" * 27) + ")",
            rows,
        )
        state.db.execute(
            "UPDATE shards SET status='harvested', harvested_at=?, cd_offset=?, cd_size=?, "
            "n_entries=?, n_samples=?, oc_checksum=?, last_modified=?, error=NULL, seconds=?, "
            "bytes=?, size=?, etag=? WHERE shard=?",
            (
                utcnow(),
                cd_offset,
                cd_size,
                len(entries),
                len(rows),
                before.checksum,
                before.last_modified,
                round(elapsed, 2),
                nbytes,
                size,
                etag,
                shard,
            ),
        )
        state.db.commit()
    return {"shard": shard, "samples": len(rows), "seconds": elapsed}


def _order(shards: list[tuple[str, str, int]]) -> list[str]:
    """Interleave so partial progress covers all splits and mesh-family blocks."""
    return [s for s, _, _ in sorted(shards, key=lambda t: (t[2] % 10, t[1] != "train", t[2]))]


def harvest_all(
    client: Http,
    state: State,
    layout: Layout,
    *,
    connections: int,
    max_attempts: int = 6,
    rounds: int = 6,
    round_pause: float = 300.0,
    limit: int | None = None,
) -> bool:
    """Harvest every pending shard. Returns True when all shards are harvested."""
    lock = threading.Lock()
    progress_path = layout.reports / "harvest-progress.json"
    for round_no in range(1, rounds + 1):
        todo = state.query(
            "SELECT shard, split, idx FROM shards WHERE status != 'harvested' AND attempts < ?",
            (max_attempts,),
        )
        total = state.query("SELECT COUNT(*) FROM shards")[0][0]
        if not todo:
            break
        order = _order(todo)[: limit or None]
        done_before = state.query("SELECT COUNT(*) FROM shards WHERE status='harvested'")[0][0]
        log(f"harvest round {round_no}: {len(order)} shards to do ({done_before}/{total} done)")
        t0 = time.monotonic()
        finished = 0
        samples = 0

        def run(shard: str) -> dict[str, Any]:
            try:
                return harvest_shard(client, state, shard)
            except Exception as exc:  # noqa: BLE001 - record, retry in a later round
                state.execute(
                    "UPDATE shards SET status='error', attempts=attempts+1, error=? WHERE shard=?",
                    (f"{type(exc).__name__}: {exc}"[:1000], shard),
                )
                log(f"ERROR {shard}: {type(exc).__name__}: {exc}")
                return {"shard": shard, "error": str(exc)}

        with ThreadPoolExecutor(max_workers=connections) as pool:
            pending = set()
            queue = list(order)
            while queue or pending:
                while queue and len(pending) < connections:
                    pending.add(pool.submit(run, queue.pop(0)))
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    res = fut.result()
                    with lock:
                        finished += 1
                        samples += res.get("samples", 0)
                        rate = finished / max(time.monotonic() - t0, 1e-9)
                        left = len(order) - finished
                        eta = left / rate if rate else float("nan")
                        if "error" not in res and (finished % 10 == 0 or left < 6):
                            snap = client.stats.snapshot()
                            log(
                                f"harvest {finished}/{len(order)} shards, {samples} samples, "
                                f"{rate * 3600:.0f} shards/h, ETA {eta / 60:.1f} min, "
                                f"{snap['requests']} req, {snap['retries']} retries"
                            )
                            try:
                                write_json_atomic(
                                    progress_path,
                                    {
                                        "track": "research",
                                        "step": "harvest",
                                        "at": utcnow(),
                                        "round": round_no,
                                        "finished_this_round": finished,
                                        "todo_this_round": len(order),
                                        "harvested_total": done_before + finished,
                                        "shards_total": total,
                                        "samples_this_round": samples,
                                        "shards_per_hour": round(rate * 3600, 1),
                                        "eta_minutes": round(eta / 60, 1),
                                        "http": snap,
                                    },
                                )
                            except OSError as exc:
                                log(f"progress file not written: {exc}")
        if limit:
            break
        left = state.query("SELECT COUNT(*) FROM shards WHERE status != 'harvested'")[0][0]
        if left == 0:
            break
        if round_no < rounds:
            log(f"{left} shards not harvested; pausing {round_pause:.0f}s before retry round")
            time.sleep(round_pause)
    left = state.query(
        "SELECT shard, status, attempts, error FROM shards WHERE status!='harvested'"
    )
    if left:
        log(f"harvest incomplete: {len(left)} shards, e.g. {left[:3]}")
    return not left


def harvest_report(state: State, layout: Layout) -> dict[str, Any]:
    rows = state.query(
        "SELECT split, COUNT(*), SUM(n_samples), SUM(size), SUM(seconds), SUM(bytes) "
        "FROM shards WHERE status='harvested' GROUP BY split"
    )
    per_split = {
        r[0]: {
            "shards": r[1],
            "samples": r[2],
            "shard_bytes": r[3],
            "harvest_seconds": r[4],
            "harvest_bytes": r[5],
        }
        for r in rows
    }
    status = dict(state.query("SELECT status, COUNT(*) FROM samples GROUP BY status"))
    incomplete = state.query("SELECT COUNT(*) FROM samples WHERE members IS NOT NULL")[0][0]
    no_block = state.query(
        "SELECT COUNT(*) FROM samples WHERE blk_start IS NULL AND status='success'"
    )[0][0]
    fam = state.query(
        "SELECT split, shard, mesh_family, COUNT(*) FROM samples GROUP BY split, shard, mesh_family"
    )
    fam_by_shard: dict[str, dict[str, int]] = defaultdict(dict)
    for _, shard, family, n in fam:
        fam_by_shard[shard][family or "?"] = n
    families = dict(state.query("SELECT mesh_family, COUNT(*) FROM samples GROUP BY mesh_family"))
    sources = dict(state.query("SELECT doc_source, COUNT(*) FROM samples GROUP BY doc_source"))
    mixed = {s: f for s, f in fam_by_shard.items() if len(f) > 1}
    report = {
        "schema": "flattener-syntheticdoc-harvest-report-1",
        "track": "research",
        "generated_at": utcnow(),
        "dataset_doi": "10.3929/ethz-c-000801994",
        "per_split": per_split,
        "sample_status": status,
        "samples_with_missing_or_extra_members": incomplete,
        "success_samples_without_contiguous_block": no_block,
        "mesh_families": families,
        "document_sources": sources,
        "shards_with_several_families": mixed,
        "shard_family": {s: max(f, key=f.get) for s, f in fam_by_shard.items()},
    }
    write_json_atomic(layout.reports / "harvest-report.json", report)
    return report
