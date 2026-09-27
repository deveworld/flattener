"""DocUNet benchmark (65 scans, 130 photos) scored with the calibrated MATLAB protocol.

Predictions are rectified outputs of the ``crop/{k}_{m} copy.png`` inputs. Every
report records the aggregation conventions, the AD solver, the SIFT-flow source
hash, software versions and the SHA-256 of every input file.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import string
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

from . import metrics
from .reporting import (
    environment,
    failure_counts,
    mean,
    prediction_record,
    prediction_status,
    report_evidence,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from .siftflow import SiftFlow, SiftFlowBuild

REPORT_SCHEMA = "flattener-benchmark-docunet-1.0"
DOCUMENTS = tuple(range(1, 66))
VIEWS = (1, 2)
ROTATED_DOCUMENT = 64
CONVENTIONS = ("130_raw", "130_rot64", "128_excl64")
# Report every convention unless told otherwise: published tables mix all three.
DEFAULT_CONVENTIONS = CONVENTIONS
CONVENTION_DESCRIPTIONS = {
    "130_raw": "all 130 photos as published (64_1/64_2 upside down); what PaperEdge "
    "adres.txt and the released DocTr/DocScanner/DocGeoNet MATLAB scripts compute "
    "(their rot90 block for document 64 is commented out); validated by the PaperEdge "
    "MATLAB check",
    "130_rot64": "64_1/64_2 rotated 180 degrees before scoring (those scripts with the "
    "rot90 block enabled); not covered by the calibration checks",
    "128_excl64": "64_1/64_2 excluded (UVDoc: DocUnet_results has 128 images and "
    "evalScript.m skips missing files); validated by the UVDoc table checks",
}
REF_RESIZE_MODES = ("once", "uvdoc_view2_double")
DEFAULT_WORKERS = 2
DEFAULT_REFERENCE_PATTERN = "docunet-reference-{k}.png"
DEFAULT_REFERENCES_DIR = Path("outputs/dataset/files")

# SHA-256 of the 65 DocUNet flatbed scans (scan.zip, scan/{k}.png) as distributed
# by the DocUNet authors; identical to outputs/dataset/manifest.json.
REFERENCE_SHA256 = {
    1: "6e7b954887cbda909f864ef1034039fc1f88672a188a7682eae86989a78d1820",
    2: "a0c79a3c801cb55ccf5ecab8d355f4064960830d0188ba4b8889d902acacbb13",
    3: "5815a08284d8f56059fffa7d459c41ac9e60da1799846b43e5eeaa29361f3d04",
    4: "b846be55877c705e651ef322f8fba5ad799070866ca641dd221ff1b7dc26f618",
    5: "5875880bb33016de0a9e933c449d341c2d2607c8158afd5973029a40ce0f5f68",
    6: "0b8655156be5e4f9b3404e638c44cf82654c3da347bb1725d9455cf0bf99a03c",
    7: "85134744afad490b8216421d3ca3d0b80d3ef81d791938a52cd9a798d85f8e9d",
    8: "c74e25eda4d44beb207a0e89ed9d8b970cec3295130820bd873ea09b1e4352cc",
    9: "252d64ec12c56fd34ab17a976d06d21f0c14c7ce1f210a193831538b5cc5b2ea",
    10: "c03fea29cb50a1a01da54a82d7409bfb0bc832fe37e561a261bef0b52f471a86",
    11: "8bb96354f75bf34c3a4b7b24d289c23578829cfef98f8e6755f0c42eb6901e08",
    12: "22e65fc05941e2a4b2600b4cf60f81eaf9a73ce86fa433c96e4b6e7d8eddae07",
    13: "61d027aa6c74be7599095b41ebd4abb766bdf249c16388175d46c2519fb3539c",
    14: "d4a421af766ad0639dec1706176b9fd2bfeeda3956819abed6d8851b09410718",
    15: "1b1cf8d8748532fb5af6bf5ef5d793083e5907a91dfaadfc566e1b5ab7ee92cd",
    16: "cd2123e4847162b4f40a50e9b4c7314813ca6f7c3b0820928ec94d4ae971a477",
    17: "fcab7528dffe7fb88132e08d782b7d2794c8e2db59417f0135700a74574e42e7",
    18: "6b09000b561a11154c3bd74212ced3fc9b262278e958b72fa2ca189577e4b5d5",
    19: "a2d44a9a8ac855bbe4925e9d47fd9456dd67f9cb893b1ef0cd7f8c57bc7288e1",
    20: "1b2dd7170ad43b1d33aa8308fcca20eed285a6dca7fae2e44c5c354a0abfe102",
    21: "1ce22f1950a8959300e573990e4cdb95a44fbc4ea4c22bf2216b60991c6a5c31",
    22: "6d5213c37a38381229ee8ada7ccaaf65f6ae108b599ea08c67542082e0e45b2c",
    23: "d8d64fe10f6b59db5c452f92eda4dcf4db08fa8c16bd27c8a36376f1a7a1f7db",
    24: "1d64a9351c8c4df46d770d13bf1a0cad0548d018907395e3935642d0faa47416",
    25: "fd7a14aa998ff857be82075fb92aa3751f56d28346dfd5074b3e62bc5c6fe013",
    26: "5fd34160d7702e7c3bc27cdd22fe8178b0e9c4969cc09900cf2d4b60d0f58891",
    27: "a7f607cf9a16ee230497c4d3763fd43a1ad476000bfaeaddec17a26e7e057fa7",
    28: "98458b8aec610e9102ecb981abf0d12780f21730e846a8dd7359eae826fe7f05",
    29: "d87fb0c2804a51f6f3a9ddea9b51508d50a760cb7f3488c57fa1f46eebcefa66",
    30: "82e1ab4fc6e7f3f0e82d393c1048a9fa58d016ae340f739c90417e3e8c09455e",
    31: "18bd52793e35e12b9f113badc441d46ee5c00ed3120340de79de2780a7d34907",
    32: "ea9309935b0b5896e70d94657870288c9c20416bcad304a1ba1142a0d3bae215",
    33: "69575a0053c183ab17d141bb01e6e2b1faece92102af1d2d2f24a995fd63f896",
    34: "fcf0288a66a60fa2478a58ff0e279133606b15704ad8f760e7afba125b023cf5",
    35: "3cb3bd198ef4a5ab11f98b33e92dc62885d96f7d11c05a325875bbc44248de59",
    36: "0880cec3721acc8426bba534fae2b90706650fe952dadc1ed5fb77bcf34b5c8d",
    37: "5373fc1f0014bfd68135b07711781aef069eb16fa01f70f6a1b26155652457b3",
    38: "a74cc330ef7083d5371a4d13a417c749c3495209df080522e485e4dd7f843932",
    39: "717f245d3e6540c448c993f495be2c8039cb99818c1474a5a4b360a5282eca2e",
    40: "b9a7d00ad1f5de7f3851572787dd547855a20733548da61ac4ae199e9d666343",
    41: "fb1520ac4c535e5c5db35515cb22cc0b8517551bb45068364293b221833fb308",
    42: "e04698d67a701746dcae119bebb336ad30e201d2b18a799ae27ba6a0236710da",
    43: "00f136864821c5f1517ccb0bcad42c3e06bd9933c2914fa1f14a3eb97804b00f",
    44: "8e72f9291a13e5f2c0118ddf5de299e1bbbef62f6311ef5d6b99d0119ffe97aa",
    45: "beff7c4aa8e2d485f5a502b8979c5f2990df00ee07a7fb4f597209cc2466bc89",
    46: "31862ae24388a18490366d254f01751c676d4204c064d2fa9300cf5536c3a684",
    47: "afbce9bd5de95482809e0c054932ee018a3c8363f212b40cc90ef52953896361",
    48: "3df8e6060eea5516d8c93655dce2b8da19c209b5d4ba34b0c6dbad1e6c6cb02f",
    49: "6082e44194d8d1515eb536c6d26a26047c7fa15f7fdb8ce75a98086712bb2d33",
    50: "bf362309f8bb55ff9bd3bb971f774742670d3e6c404d67128f8e51d5e404c413",
    51: "8b92f710e1734719b063982701cafe2c008e1b09b4d3cc4f2ea3a5e789b775b1",
    52: "5b79349aa8d04695fd98df9f0d29f5bf9bad5e048bce488d0ae5dc29402fcab3",
    53: "7cd122c00ee7c70dbbb1311b3a13eb57b81f2e0b5e283ee3a253e25ac94288dc",
    54: "6b7d19270e6832a76dd2bdd33b1ab229085ec93c6fe7779e349d91ee3563ffb3",
    55: "b653b20c8958ab0ea65d0ad62629009bd48450ac9b9f404c4967955778603c36",
    56: "7d7109ae15ec9701d98ab40dc34381b320c4c0fddca081071b81b1504989c8ca",
    57: "3b620c48e69d7b27fbbc40a832b4c8d9a41626482462aaae07e9b4b6a8023395",
    58: "2128fa3e66792fa99644f9e3f9a50ba5bd88a758f7d18e414231944567f06749",
    59: "999409f9e486ecb96a1a286a1b90f89a1bf5aed692f0ef98fd21e2839951d31f",
    60: "8ef0fd25a43be0ee02ac634c0e84ba701a272baaa582167290ccbb18979c40fa",
    61: "50ea5e8487029470090fdffc694cc55d84ebf467ff58a99d6881dcb779a153db",
    62: "9910afdc0f6286018c89885dfe77226fc51cb5a02f1afade6d67262f23ce8d0f",
    63: "145d8b15c8811dde382c9dc79966683214910fc9e902f41f7a852648d40f86f0",
    64: "722348a72f14e5dfce2130802d752d5eab222fe425a2c81a06d4346cb4ae2924",
    65: "5ca8e7d064a198af58665ef84566609799b589203731e49a0b619abfcc60ac07",
}

PROTOCOL = {
    "alpha_composite_for_metrics": "black (MATLAB one-output imread; PNG bKGD if present)",
    "flow_direction": "scan -> prediction (siftFlow(scan, prediction))",
    "gray_before_resize": True,
    "metric_area": metrics.METRIC_AREA,
    "ms_ssim": "weighted sum of 5 MATLAB ssim levels with impyramid reduce",
    "ms_ssim_weights": [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
    "resize": "MATLAB imresize bicubic (Keys a=-0.5), antialias when shrinking, ceil size",
    "siftflow": {
        "alpha": 510.0,
        "cell_size": 3,
        "d": 10200.0,
        "gamma": 1.275,
        "grid_spacing": 1,
        "nIterations": 30,
        "nTopIterations": 60,
        "nlevels": 4,
        "prefilter": "7x7 sigma=1 gaussian (replicate), then 0.5x bicubic",
        "topwsize": 10,
        "wsize": 2,
    },
    "ad": "PaperEdge evalAlignedUnwarp.m: Sobel mask > 0.5, per-axis scale+shift removed",
    "ad_solvers": {
        "exact_lstsq": "exact least squares (numpy lstsq)",
        "lsqr_4iter": "LSQR with MATLAB lsqr(A,B) defaults; its means match MATLAB in the "
        "DocUNet calibration checks, per-image AD can differ by up to ~0.016",
    },
    "conventions": CONVENTION_DESCRIPTIONS,
}


def convention_views(convention: str) -> tuple[tuple[int, int, bool], ...]:
    """Return ``(document, view, rotated_180)`` triples scored under a convention."""
    if convention == "130_raw":
        return tuple((k, m, False) for k in DOCUMENTS for m in VIEWS)
    if convention == "130_rot64":
        return tuple((k, m, k == ROTATED_DOCUMENT) for k in DOCUMENTS for m in VIEWS)
    if convention == "128_excl64":
        return tuple((k, m, False) for k in DOCUMENTS if k != ROTATED_DOCUMENT for m in VIEWS)
    raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")


def validate_conventions(conventions: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(conventions))
    if not selected:
        raise ValueError("at least one convention is required")
    for convention in selected:
        convention_views(convention)
    return selected


def required_views(conventions: Iterable[str]) -> tuple[tuple[int, int, bool], ...]:
    views = {view for convention in conventions for view in convention_views(convention)}
    return tuple(sorted(views))


def reference_resizes(ref_resize: str, view: int) -> int:
    """How many times the scan is resized before scoring ``view``."""
    if ref_resize == "once":
        return 1
    if ref_resize == "uvdoc_view2_double":
        # UVDoc evalScript.m resizes rimg inside the view loop, so view 2 sees it twice.
        return view
    raise ValueError(f"ref_resize must be one of {REF_RESIZE_MODES}, got {ref_resize!r}")


def format_name(pattern: str, *, fields: Sequence[str], **values: int) -> str:
    """Fill ``{k}``/``{m}`` in a relative file pattern after strict validation."""
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("file pattern must be a non-empty string")
    used = set()
    for _, field, spec, conversion in string.Formatter().parse(pattern):
        if field is None:
            continue
        if field not in fields or spec or conversion:
            raise ValueError(f"pattern {pattern!r} may only use plain {sorted(fields)} fields")
        used.add(field)
    if used != set(fields):
        raise ValueError(f"pattern {pattern!r} must contain {sorted(fields)}")
    name = pattern.format(**values)
    # Check both path flavours so a rooted ("/x", "\\x") or drive path cannot escape
    # the directory on either platform.
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    if posix.is_absolute() or windows.anchor or ".." in windows.parts or not posix.parts:
        raise ValueError(f"pattern {pattern!r} must produce a relative path inside the directory")
    return name


@dataclass(frozen=True)
class ViewTask:
    view: int
    rotated: bool
    prediction_path: str


@dataclass(frozen=True)
class DocumentTask:
    document: int
    reference_path: str
    views: tuple[ViewTask, ...]
    ref_resize: str
    ad_solver: str


def score_document(task: DocumentTask, siftflow: SiftFlow | None) -> list[dict[str, Any]]:
    """Score all requested views of one document; the scan is decoded once."""
    reference_bytes = Path(task.reference_path).read_bytes()
    reference = metrics.read_image(reference_bytes)
    prepared: dict[int, metrics.PreparedReference] = {}
    rows = []
    for view in task.views:
        started = time.perf_counter()
        resizes = reference_resizes(task.ref_resize, view.view)
        if resizes not in prepared:
            prepared[resizes] = metrics.prepare_reference(reference, resizes=resizes)
        prediction_bytes = Path(view.prediction_path).read_bytes()
        prediction = metrics.read_image(prediction_bytes)
        if view.rotated:
            prediction = np.ascontiguousarray(np.rot90(prediction, 2))
        score = metrics.score_prepared(
            prediction, prepared[resizes], siftflow=siftflow, ad_solver=task.ad_solver
        )
        row = score.to_dict()
        row.update(
            {
                "id": f"{task.document}_{view.view}",
                "k": task.document,
                "m": view.view,
                "prediction_sha256": sha256_bytes(prediction_bytes),
                "reference_sha256": sha256_bytes(reference_bytes),
                "rotated_180": view.rotated,
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
        rows.append(row)
    return rows


ScoreTask = Callable[[Any, SiftFlow | None], list[dict[str, Any]]]
_WORKER_SIFTFLOW: SiftFlow | None = None


def _initialize_worker(build: SiftFlowBuild | None) -> None:
    global _WORKER_SIFTFLOW
    import torch

    torch.set_num_threads(1)
    _WORKER_SIFTFLOW = None if build is None else SiftFlow(build)


def _score_in_worker(score: ScoreTask, task: Any) -> list[dict[str, Any]]:
    return score(task, _WORKER_SIFTFLOW)


@contextlib.contextmanager
def _single_thread_environment():
    """Children inherit one-thread BLAS/OpenMP settings; the parent is restored."""
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_tasks(
    tasks: Sequence[Any],
    build: SiftFlowBuild | None,
    *,
    workers: int = DEFAULT_WORKERS,
    progress: Callable[[dict[str, Any], int, int], None] | None = None,
    score: ScoreTask = score_document,
    total: int | None = None,
) -> list[dict[str, Any]]:
    """Score tasks in-process (``workers == 1``) or in a spawned process pool.

    ``score`` turns one task into rows; it must be a module-level function so the pool
    can pickle it. ``total`` is the expected row count for ``progress`` (default: the
    views of the ``DocumentTask`` list).
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if total is None:
        total = sum(len(task.views) for task in tasks)
    rows: list[dict[str, Any]] = []

    def collect(new_rows: list[dict[str, Any]]) -> None:
        for row in new_rows:
            rows.append(row)
            if progress is not None:
                progress(row, len(rows), total)

    if workers == 1:
        siftflow = None if build is None else SiftFlow(build)
        for task in tasks:
            collect(score(task, siftflow))
        return rows
    context = multiprocessing.get_context("spawn")
    with _single_thread_environment():
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)) or 1,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(build,),
        ) as executor:
            futures = [executor.submit(_score_in_worker, score, task) for task in tasks]
            try:
                for future in as_completed(futures):
                    collect(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    return rows


def _row_key(row: dict[str, Any]) -> tuple[int, int, bool]:
    return (row["k"], row["m"], row["rotated_180"])


def summarize(rows: Sequence[dict[str, Any]], conventions: Sequence[str]) -> dict[str, Any]:
    """Means per convention (and per view) over the rows each convention selects."""
    by_key = {_row_key(row): row for row in rows}
    summary = {}
    for convention in conventions:
        selected = [by_key[view] for view in convention_views(convention)]
        summary[convention] = _means(selected)
        summary[convention]["views"] = {
            str(view): _means([row for row in selected if row["m"] == view]) for view in VIEWS
        }
    return summary


def _means(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Means, plus ``prediction_failures`` (scored images not recorded as passing their
    checks: force-rendered failures and images ``predictions.json`` does not list) and
    ``prediction_unrecorded`` (the latter); both ``None`` when the predictions carry no
    record."""
    result: dict[str, Any] = {"n": len(rows)}
    for name in ("ms_ssim", "ld", "ad", "ad_exact_lstsq", "ad_lsqr_4iter"):
        values = [row[name] for row in rows]
        result[name] = None if any(value is None for value in values) else mean(values)
    result.update(failure_counts([row.get("prediction_status") for row in rows]))
    return result


def hash_references(
    references_dir: Path,
    reference_pattern: str,
    documents: Iterable[int],
    *,
    verify: bool = True,
) -> dict[int, str]:
    """Hash scans and (optionally) require the published DocUNet scan bytes."""
    hashes = {}
    mismatched = []
    for document in sorted(set(documents)):
        path = references_dir / format_name(reference_pattern, fields=("k",), k=document)
        if not path.is_file():
            raise FileNotFoundError(f"missing DocUNet scan {path}")
        hashes[document] = sha256_file(path)
        if verify and hashes[document] != REFERENCE_SHA256[document]:
            mismatched.append(str(path))
    if mismatched:
        raise ValueError(f"scan bytes differ from the published DocUNet scans: {mismatched}")
    return hashes


def evaluate_docunet(
    predictions_dir: str | os.PathLike[str],
    *,
    pattern: str,
    conventions: Sequence[str],
    build: SiftFlowBuild | None,
    ref_resize: str = "once",
    ad_solver: str = metrics.DEFAULT_AD_SOLVER,
    references_dir: str | os.PathLike[str] = DEFAULT_REFERENCES_DIR,
    reference_pattern: str = DEFAULT_REFERENCE_PATTERN,
    verify_references: bool = True,
    workers: int = DEFAULT_WORKERS,
    progress: Callable[[dict[str, Any], int, int], None] | None = None,
) -> dict[str, Any]:
    """Score a directory of DocUNet predictions and return the report dictionary.

    ``build`` comes from :func:`flattener.benchmark.siftflow.build_siftflow`; ``None``
    computes MS-SSIM only, and the report has ``comparable: false``.
    """
    conventions = validate_conventions(conventions)
    reference_resizes(ref_resize, 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if ad_solver not in metrics.AD_SOLVERS:
        raise ValueError(f"ad_solver must be one of {metrics.AD_SOLVERS}, got {ad_solver!r}")
    predictions_root = Path(predictions_dir).expanduser().resolve()
    references_root = Path(references_dir).expanduser().resolve()
    if not predictions_root.is_dir():
        raise NotADirectoryError(f"predictions directory not found: {predictions_root}")
    views = required_views(conventions)
    documents = sorted({k for k, _, _ in views})
    reference_hashes = hash_references(
        references_root, reference_pattern, documents, verify=verify_references
    )
    record = prediction_record(predictions_root)
    missing, names = [], {}
    grouped: dict[int, list[ViewTask]] = {}
    for k, m, rotated in views:
        names[f"{k}_{m}"] = format_name(pattern, fields=("k", "m"), k=k, m=m)
        path = predictions_root / names[f"{k}_{m}"]
        if not path.is_file():
            missing.append(names[f"{k}_{m}"])
        grouped.setdefault(k, []).append(ViewTask(m, rotated, str(path)))
    if missing:
        recorded = [prediction_status(record, name) for name in missing]
        failed = [status for status in recorded if status not in (None, "success", "unrecorded")]
        note = (
            f"; predictions.json records {len(failed)} of them as failed checks "
            "(predict with --force-render to score the forced images)"
            if failed
            else ""
        )
        raise FileNotFoundError(
            f"{len(missing)} predictions missing in {predictions_root}, first: {missing[:5]}{note}"
        )
    tasks = [
        DocumentTask(
            document=k,
            reference_path=str(
                references_root / format_name(reference_pattern, fields=("k",), k=k)
            ),
            views=tuple(sorted(grouped[k], key=lambda item: (item.view, item.rotated))),
            ref_resize=ref_resize,
            ad_solver=ad_solver,
        )
        for k in documents
    ]
    started = time.perf_counter()
    rows = run_tasks(tasks, build, workers=workers, progress=progress)
    wall_seconds = time.perf_counter() - started
    rows.sort(key=_row_key)
    for row in rows:
        if row["reference_sha256"] != reference_hashes[row["k"]]:
            raise RuntimeError(f"scan {row['k']} changed during evaluation")
        row["prediction_status"] = prediction_status(record, names[row["id"]])
    return {
        "benchmark": "docunet",
        "comparable": build is not None and verify_references,
        "config": {
            "ad_solver": ad_solver,
            "conventions": list(conventions),
            "prediction_pattern": pattern,
            "ref_resize": ref_resize,
            "reference_pattern": reference_pattern,
            "references_verified": verify_references,
            "workers": workers,
        },
        "created_utc": utc_now(),
        "environment": environment(),
        "evidence": report_evidence(record),
        "inputs": {
            "predictions": {row["id"]: row["prediction_sha256"] for row in rows},
            "predictions_dir": str(predictions_root),
            "references": {str(k): digest for k, digest in reference_hashes.items()},
            "references_dir": str(references_root),
        },
        "predictions_record": record,
        "protocol": PROTOCOL,
        "rows": rows,
        "schema": REPORT_SCHEMA,
        "siftflow": None if build is None else build.metadata(),
        "summary": summarize(rows, conventions),
        "timing": {
            "image_seconds_total": round(sum(row["seconds"] for row in rows), 3),
            "wall_seconds": round(wall_seconds, 3),
        },
    }


def print_progress(row: dict[str, Any], done: int, total: int) -> None:
    rotated = " rot180" if row["rotated_180"] else ""
    parts = [f"ms_ssim={row['ms_ssim']:.5f}"]
    if row["ld"] is not None:
        parts += [f"ld={row['ld']:.3f}", f"ad={row['ad']:.4f}"]
    print(
        f"[{done}/{total}] {row['id']}{rotated} {' '.join(parts)} ({row['seconds']:.1f}s)",
        file=sys.stderr,
        flush=True,
    )
