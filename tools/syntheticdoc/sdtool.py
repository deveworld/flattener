#!/usr/bin/env python3
"""SyntheticDoc release/research data tooling (flattener).

Steps (each resumable and idempotent; ``pipeline`` runs them in order):

  licenses   copy license tables from the asset share (sources_*.json, READMEs,
             background material licenses) into <research>/index/license-tables
  harvest    list all shards, read every central directory and metadata.json
  classify   per-sample license class (clean/sa/nc/unknown) + license report
  split      own train/val split grouped by document AND mesh group
  select     ~N clean samples per train shard + all clean val samples
  download   render/backward-map/metadata of the selection -> JPEG q95 + fp16 npy
  manifest   manifest.jsonl, dataset-manifest.json, license-manifest.json
  verify     decode a random subset; re-fetch a few upstream members and compare
  status     print progress (safe to run while a job is running)

Layout under --data-root (default D:\\flattener-sota\\data on the remote PC):
  research/syntheticdoc/index/syntheticdoc-index.sqlite   index + state (all classes)
  research/syntheticdoc/index/license-tables/             verbatim license tables
  research/syntheticdoc/reports/                          harvest/license/split reports
  release/syntheticdoc/                                   clean samples + manifests

Stdlib + numpy + Pillow, Python 3.12. Bounded concurrency (--connections, default 6).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from sdcommon import (
    DEFAULT_CONNECTIONS,
    TOOL_VERSION,
    Http,
    Layout,
    State,
    default_data_root,
    log,
    utcnow,
)


def _layout(args: argparse.Namespace) -> Layout:
    root = Path(args.data_root) if args.data_root else default_data_root()
    if root is None:
        sys.exit("--data-root is required (no D:\\flattener-sota\\data here)")
    return Layout(root)


def _signature(state: State, *extra: object) -> str:
    rows = state.query(
        "SELECT COUNT(*), COALESCE(SUM(n_samples), 0), GROUP_CONCAT(etag) FROM "
        "(SELECT n_samples, etag FROM shards WHERE status='harvested' ORDER BY shard)"
    )
    tables = state.get("license_tables", {})
    tsig = sorted((k, v.get("sha256")) for k, v in (tables.get("files") or {}).items())
    blob = json.dumps([rows[0][0], rows[0][1], rows[0][2], tsig, list(map(str, extra))])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _script_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(here.glob("*.py"))
    }


def cmd_licenses(args, layout, state, client):
    from sdlicense import fetch_tables

    fetch_tables(client, state, layout)
    return 0


def cmd_harvest(args, layout, state, client):
    from harvest import harvest_all, harvest_report, list_shards

    list_shards(client, state)
    ok = harvest_all(
        client,
        state,
        layout,
        connections=args.connections,
        limit=args.limit,
        rounds=args.rounds,
        round_pause=args.round_pause,
    )
    harvest_report(state, layout)
    return 0 if ok or args.limit else 3


def cmd_classify(args, layout, state, client):
    from sdlicense import classify, license_report, load_tables

    load_tables(state, layout)
    classify(state, layout)
    report = license_report(state, layout)
    log(
        "license classes",
        report["class_counts"],
        "clean fraction",
        report["clean_fraction_of_success"],
    )
    state.put("stage.classify", _signature(state))
    return 0


def cmd_split(args, layout, state, client):
    from sdsplit import resplit

    resplit(state, layout, val_fraction=args.val_fraction)
    state.put("stage.split", _signature(state, args.val_fraction))
    return 0


def cmd_select(args, layout, state, client):
    from fetch import select

    select(state, layout, per_shard=args.per_shard)
    state.put("stage.select", _signature(state, args.val_fraction, args.per_shard))
    with state.lock:
        from sdcommon import POST_INDEXES

        state.db.executescript(POST_INDEXES)
    return 0


def cmd_download(args, layout, state, client):
    from fetch import download

    ok = download(
        client,
        state,
        layout,
        track="release",
        connections=args.connections,
        samples_per_request=args.samples_per_request,
        min_free_gb=args.min_free_gb,
        rounds=args.rounds,
        round_pause=args.round_pause,
    )
    return 0 if ok else 4


def cmd_manifest(args, layout, state, client):
    from fetch import build_manifests

    log("manifest", build_manifests(state, layout, "release"))
    return 0


def cmd_verify(args, layout, state, client):
    from fetch import verify

    report = verify(client, state, layout, track="release", local=args.local, remote=args.remote)
    return 0 if report["all_ok"] else 5


def cmd_status(args, layout, state, client):
    out = {"at": utcnow(), "tool": TOOL_VERSION}
    out["shards"] = dict(state.query("SELECT status, COUNT(*) FROM shards GROUP BY status"))
    out["samples"] = state.query("SELECT COUNT(*) FROM samples")[0][0]
    out["classes"] = dict(state.query("SELECT lic_class, COUNT(*) FROM samples GROUP BY lic_class"))
    out["our_split"] = dict(
        state.query("SELECT our_split, COUNT(*) FROM samples GROUP BY our_split")
    )
    out["selection"] = dict(
        state.query(
            "SELECT sel_role || ':' || COALESCE(dl_status, 'pending'), COUNT(*) FROM samples "
            "WHERE sel_role IS NOT NULL GROUP BY sel_role, dl_status"
        )
    )
    out["stages"] = {
        k: state.get(k) for k in ("stage.classify", "stage.split", "stage.select", "stage.pipeline")
    }
    for name, path in (
        ("harvest_progress", layout.reports / "harvest-progress.json"),
        ("download_progress", layout.release / "progress.json"),
    ):
        if path.is_file():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    out["disk_free_gb"] = round(shutil.disk_usage(layout.data_root).free / 1e9, 1)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_pipeline(args, layout, state, client):
    from fetch import build_manifests, download, select, verify
    from harvest import harvest_all, harvest_report, list_shards
    from sdlicense import classify, fetch_tables, license_report, load_tables
    from sdsplit import resplit

    state.put(
        "stage.pipeline", {"started_at": utcnow(), "args": vars(args), "scripts": _script_hashes()}
    )
    log(f"pipeline {TOOL_VERSION} scripts {_script_hashes()}")
    fetch_tables(client, state, layout)
    list_shards(client, state)
    if not harvest_all(
        client,
        state,
        layout,
        connections=args.connections,
        rounds=args.rounds,
        round_pause=args.round_pause,
    ):
        harvest_report(state, layout)
        log("pipeline: harvest incomplete; relaunch to resume")
        return 3
    harvest_report(state, layout)
    sig = _signature(state)
    if state.get("stage.classify") != sig:
        load_tables(state, layout)
        classify(state, layout)
        state.put("stage.classify", sig)
    report = license_report(state, layout)
    log(
        "license classes",
        report["class_counts"],
        "clean fraction",
        report["clean_fraction_of_success"],
    )
    sig_split = _signature(state, args.val_fraction)
    if state.get("stage.split") != sig_split:
        if state.query("SELECT COUNT(*) FROM samples WHERE dl_status='done'")[0][0]:
            log("WARNING: split inputs changed after downloads started; recomputing split")
        resplit(state, layout, val_fraction=args.val_fraction)
        state.put("stage.split", sig_split)
    sig_sel = _signature(state, args.val_fraction, args.per_shard)
    if state.get("stage.select") != sig_sel:
        select(state, layout, per_shard=args.per_shard)
        state.put("stage.select", sig_sel)
        with state.lock:
            from sdcommon import POST_INDEXES

            state.db.executescript(POST_INDEXES)
    if args.skip_download:
        build_manifests(state, layout, "release")
        return 0
    ok = download(
        client,
        state,
        layout,
        track="release",
        connections=args.connections,
        samples_per_request=args.samples_per_request,
        min_free_gb=args.min_free_gb,
        rounds=args.rounds,
        round_pause=args.round_pause,
    )
    build_manifests(state, layout, "release")
    rep = verify(
        client, state, layout, track="release", local=args.verify_local, remote=args.verify_remote
    )
    state.put(
        "stage.pipeline",
        {"finished_at": utcnow(), "download_complete": ok, "verify_ok": rep["all_ok"]},
    )
    log(f"pipeline finished: download_complete={ok} verify_ok={rep['all_ok']}")
    return 0 if ok and rep["all_ok"] else 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", help="data root holding release/ and research/")
    parser.add_argument(
        "--connections",
        type=int,
        default=DEFAULT_CONNECTIONS,
        help="concurrent HTTPS connections (default 6)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in (
        "licenses",
        "harvest",
        "classify",
        "split",
        "select",
        "download",
        "manifest",
        "verify",
        "status",
        "pipeline",
    ):
        p = sub.add_parser(name)
        p.add_argument("--limit", type=int, default=None, help="harvest at most N shards (test)")
        p.add_argument("--rounds", type=int, default=6, help="retry rounds for failed items")
        p.add_argument("--round-pause", type=float, default=300.0)
        p.add_argument("--val-fraction", type=float, default=0.02)
        p.add_argument("--per-shard", type=int, default=200)
        p.add_argument("--samples-per-request", type=int, default=8)
        p.add_argument("--min-free-gb", type=float, default=40.0)
        p.add_argument("--local", type=int, default=64)
        p.add_argument("--remote", type=int, default=6)
        p.add_argument("--verify-local", type=int, default=64)
        p.add_argument("--verify-remote", type=int, default=6)
        p.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.connections <= 32:
        parser.error("--connections must be in 1..32")
    layout = _layout(args)
    for path in (layout.research, layout.release, layout.reports, layout.tables):
        path.mkdir(parents=True, exist_ok=True)
    state = State(layout.index_db)
    client = Http()
    handler = globals()[f"cmd_{args.cmd}"]
    t0 = time.monotonic()
    try:
        code = handler(args, layout, state, client)
    finally:
        if args.cmd != "status":
            log(f"{args.cmd} done in {time.monotonic() - t0:.0f}s; http {client.stats.snapshot()}")
        state.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
