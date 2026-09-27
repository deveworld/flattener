"""Install and reproduce the v0 page gate from the locator study archive.

The archive is ``outputs/models/locator.zip``: the locator study's ``best.pt``
(region-context-unet-2.0, 140,545 parameters), its training script, records and code.

Usage (run at low CPU priority, for example ``nice -n 19``):
  .venv/bin/python tools/locator_v0.py install [--zip Z] [--out DIR]
  .venv/bin/python tools/locator_v0.py reproduce [--zip Z] [--smartdoc DIR] [--weights W]
      [--report R]

Steps:
  install    Copies ``best.pt`` to DIR/locator-v0-pilot.pt (schema
             independent-locator-pilot-1.0) and exports DIR/page-gate-v0.pt (schema
             region-context-1: the weights plus the provenance record
             ``tools/locator_v0_provenance.json``), each with a ``.sha256`` sidecar. The
             export's sha256 should be ``RECORDED_EXPORT_SHA256``. DIR defaults to
             $FLATTENER_CACHE_DIR/scan-weights (~/.cache/flattener/scan-weights).
  reproduce  Runs the weights on the 48 SmartDoc capture validation samples of the study
             (``gate_v0`` in ``outputs/benchmarks/smartdoc/manifest.json``, the files under
             DIR/gate-v0/) with the study's metric: sigmoid, bilinear upsampling to 384x288,
             threshold 0.5. It passes when the capture IoU is within 1e-5 of the recorded
             0.9508340768516064, every per-image loss is within 1e-5 of the recorded loss,
             and the archived model code gives probabilities within 1e-5 of the port. The
             report goes to --report.

The provenance record was frozen from the study's dataset manifest before that dataset was
deleted; the paths inside it (and in ``locator_v0_smartdoc.json``) are where the files were
at study time. Every archive member used is checked against the archive's
``manifest.json`` hashes, the provenance record and the samples against the hashes the
study recorded, and every sample file against the SmartDoc manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import platform
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flattener.benchmark.siftflow import cache_root  # noqa: E402
from flattener.evidence import evidence  # noqa: E402
from flattener.regions import context  # noqa: E402

OUTPUTS = Path(os.environ.get("FLATTENER_OUTPUTS", ROOT / "outputs"))
DEFAULT_ZIP = OUTPUTS / "models" / "locator.zip"
DEFAULT_SMARTDOC = OUTPUTS / "benchmarks" / "smartdoc"
DEFAULT_WEIGHTS_DIR = cache_root() / "scan-weights"
PROVENANCE = ROOT / "tools" / "locator_v0_provenance.json"
PILOT_NAME = "locator-v0-pilot.pt"
EXPORT_NAME = "page-gate-v0.pt"
# The archive path at study time; the export's provenance record (and so its sha256) holds it.
ARCHIVE_LABEL = "outputs/coordinate-fold/locator.zip"
RECORDED_EXPORT_SHA256 = "49b80249f2e5e140de9b96ce95d1614be800b8fa311ab72dc0490f9b5940c155"
PHOTO = (384, 288)
RECORDED_CAPTURE_IOU = 0.9508340768516064
TOLERANCE = 1e-5
MEMBERS = ("best.pt", "experiment.json", "best-validation.json", "run.py", "source.tar")
ARCHIVED_MODULE = "src/flattener/regions/context_segmentation.py"
SMARTDOC_CITATION = (
    "Jean-Christophe Burie, Joseph Chazalon, Mickael Coustaty, Sebastien Eskenazi, "
    "Muhammad Muzzamil Luqman, Maroua Mehri, Nibal Nayef, Jean-Marc Ogier, Sophea Prum and "
    "Marcal Rusinol: ICDAR2015 Competition on Smartphone Document Capture and OCR "
    "(SmartDoc). ICDAR 2015."
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fail(message: str):
    raise SystemExit(f"locator_v0: {message}")


def read_archive(path: Path) -> dict:
    """The archive members this tool uses, each checked against the archive manifest."""
    path = Path(path)
    if not path.is_file():
        _fail(f"{path} does not exist")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        members = {}
        for name in MEMBERS:
            data = archive.read(name)
            if _sha(data) != manifest["entries"][name]["sha256"]:
                _fail(f"{name} differs from the archive manifest")
            members[name] = data
    experiment = json.loads(members["experiment.json"])
    validation = json.loads(members["best-validation.json"])
    best = _sha(members["best.pt"])
    run_hashes = [v for k, v in experiment["source_sha256"].items() if k.endswith("/run.py")]
    if (
        experiment.get("state") != "complete"
        or experiment.get("best_sha256") != best
        or experiment.get("best_validation_sha256") != _sha(members["best-validation.json"])
        or validation.get("checkpoint_sha256") != best
        or run_hashes != [_sha(members["run.py"])]
    ):
        _fail("archive records disagree with best.pt, best-validation.json or run.py")
    return {
        "path": path,
        "sha256": _file_sha(path),
        "members": members,
        "experiment": experiment,
        "validation": validation,
        "best_sha256": best,
    }


def _recorded_manifest_sha256(experiment: dict) -> str:
    recorded = [v for k, v in experiment["source_sha256"].items() if k.endswith("/manifest.json")]
    if len(recorded) != 1:
        _fail("the study records no single dataset manifest")
    return recorded[0]


def read_provenance(archive: dict, path: Path = PROVENANCE) -> dict:
    """The frozen provenance record of the export; its hashes must match the archive."""
    path = Path(path)
    if not path.is_file():
        _fail(f"{path} does not exist")
    record = json.loads(path.read_bytes())
    source = record["source"]
    if (
        source["archive_sha256"] != archive["sha256"]
        or source["member_sha256"] != archive["best_sha256"]
        or source["experiment_sha256"] != _sha(archive["members"]["experiment.json"])
        or record["dataset"]["manifest_sha256"] != _recorded_manifest_sha256(archive["experiment"])
    ):
        _fail(f"{path} does not match the archive")
    return record


def read_gate_samples(directory: Path, archive: dict) -> dict:
    """The study's SmartDoc capture validation records, by id, from the SmartDoc manifest."""
    manifest_path = Path(directory) / "manifest.json"
    if not manifest_path.is_file():
        _fail(f"{manifest_path} does not exist")
    gate = json.loads(manifest_path.read_bytes())["gate_v0"]
    digest = _recorded_manifest_sha256(archive["experiment"])
    if gate["dataset_manifest_sha256"] != digest:
        _fail("gate_v0 records come from another dataset manifest than the study's")
    return {
        "directory": Path(directory),
        "sha256": digest,
        "records": {r["id"]: r for r in gate["records"]},
    }


def _kind(record: dict) -> str:
    if record.get("source", "authored") != "paper":
        return "authored"
    return record.get("kind", "simulated")


def install(archive: dict, provenance: dict, out: Path) -> dict:
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    pilot = out / PILOT_NAME
    temporary = pilot.with_name(pilot.name + ".tmp")
    temporary.write_bytes(archive["members"]["best.pt"])
    os.replace(temporary, pilot)
    if context.write_sidecar(pilot) != archive["best_sha256"]:
        _fail("installed pilot checkpoint differs from best.pt")
    loaded = context.load_context_segmenter(pilot)
    if loaded.metadata["parameter_count"] != archive["experiment"]["parameter_count"]:
        _fail("parameter count differs from the study's")
    export = out / EXPORT_NAME
    digest = context.save_context_checkpoint(
        export, loaded.model.state_dict(), loaded.config, provenance
    )
    exported = context.load_context_segmenter(export)
    for key, value in loaded.model.state_dict().items():
        if not torch.equal(value, exported.model.state_dict()[key]):
            _fail(f"exported weight {key} differs")
    return {
        "pilot": {"path": str(pilot), "sha256": archive["best_sha256"]},
        "export": {
            "path": str(export),
            "sha256": digest,
            "schema": context.CHECKPOINT_SCHEMA,
            "matches_recorded": digest == RECORDED_EXPORT_SHA256,
        },
        "parameter_count": loaded.metadata["parameter_count"],
    }


def _load_png(samples: dict, record: dict, name: str) -> np.ndarray:
    path = samples["directory"] / record["paths"][name]
    data = path.read_bytes()
    if _sha(data) != record["sha256"][name]:
        _fail(f"{path} differs from the SmartDoc manifest")
    with Image.open(io.BytesIO(data)) as encoded:
        return np.array(encoded, copy=True)


def load_sample(samples: dict, record: dict) -> tuple[np.ndarray, np.ndarray]:
    """RGB image and boolean page mask of one validation record at the photo size."""
    image = _load_png(samples, record, "image.png")
    values = _load_png(samples, record, "mask.png")
    if image.shape != (*PHOTO, 3) or values.shape != PHOTO or not np.isin(values, (0, 255)).all():
        _fail(f"{record['id']} is not a {PHOTO} RGB image with a 0/255 mask")
    return image, values == 255


def archived_model_class(archive: dict):
    """``ContextPageSegmenter`` from the archived source, for parity checks."""
    with tarfile.open(fileobj=io.BytesIO(archive["members"]["source.tar"])) as tar:
        source = tar.extractfile(ARCHIVED_MODULE).read()
    name = "flattener.regions._archived_context_segmentation"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "flattener.regions"
    sys.modules[name] = module
    try:
        exec(compile(source, f"locator.zip:source.tar:{ARCHIVED_MODULE}", "exec"), module.__dict__)
    finally:
        del sys.modules[name]
    return module.ContextPageSegmenter, module.ContextSegmenterConfig


def image_metrics(logits: torch.Tensor, mask: np.ndarray) -> dict:
    """The study's per-image metrics (run.py ``evaluate``) from native-size logits."""
    logits = logits[None, None]
    target = torch.from_numpy(mask)[None, None]
    loss_target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest-exact")
    probability = logits.sigmoid()
    dims = (1, 2, 3)
    bce = F.binary_cross_entropy_with_logits(logits, loss_target, reduction="none").mean(dims)
    dice = 1 - (2 * (probability * loss_target).sum(dims) + 1) / (
        probability.sum(dims) + loss_target.sum(dims) + 1
    )
    prediction = F.interpolate(probability, size=PHOTO, mode="bilinear", align_corners=False) >= 0.5
    tp = (prediction & target).sum(dims).float()
    truth, predicted = target.sum(dims), prediction.sum(dims)
    union = truth + predicted - tp
    values = torch.stack(
        (tp / union, tp / predicted.clamp_min(1), tp / truth, bce + dice, predicted), dim=1
    )[0]
    keys = ("iou", "precision", "recall", "loss", "predicted_pixels")
    return dict(zip(keys, values.double().tolist(), strict=True))


def reproduce(archive: dict, samples: dict, weights: Path) -> dict:
    """The study's capture metrics of ``weights`` on its 48 SmartDoc validation samples."""
    loaded = context.load_context_segmenter(weights)
    archived_class, archived_config = archived_model_class(archive)
    raw_config = archive["experiment"]["model_config"]
    archived = archived_class(
        archived_config(
            input_size=tuple(raw_config["input_size"]), channels=tuple(raw_config["channels"])
        )
    ).eval()
    archived.load_state_dict(loaded.model.state_dict(), strict=True)
    recorded = archive["validation"]
    rows, parity = [], 0.0
    for row in recorded["rows"]:
        if row["domain"] != "capture":
            continue
        record = samples["records"].get(row["id"])
        if record is None or record["split"] != "validation" or _kind(record) != "capture":
            _fail(f"{row['id']} is not a capture validation record")
        image, mask = load_sample(samples, record)
        logits = torch.from_numpy(loaded.predict_logits(image))
        with torch.inference_mode():
            reference = archived(context.preprocess_image(image, loaded.config)[None])[0, 0]
        parity = max(parity, float((logits.sigmoid() - reference.sigmoid()).abs().max()))
        metrics = image_metrics(logits, mask)
        rows.append(
            {
                "id": row["id"],
                "domain": row["domain"],
                "document": row["document"],
                **metrics,
                "delta": {key: metrics[key] - row[key] for key in metrics},
            }
        )
    if not rows:
        _fail("the archive records no capture validation rows")
    iou = float(np.mean([r["iou"] for r in rows]))
    expected = recorded["summary"]["domains"]["capture"]["iou"]
    capture = {
        "count": len(rows),
        "iou": iou,
        "recorded_iou": expected,
        "iou_difference": iou - expected,
        "max_abs_image_iou_difference": max(abs(r["delta"]["iou"]) for r in rows),
        "max_abs_image_loss_difference": max(abs(r["delta"]["loss"]) for r in rows),
        "predicted_pixel_mismatches": sum(r["delta"]["predicted_pixels"] != 0 for r in rows),
    }
    checks = {
        "archived_code_max_abs_probability_difference": parity,
        "archived_code_parity": parity <= TOLERANCE,
        "recorded_loss_agreement": capture["max_abs_image_loss_difference"] <= TOLERANCE,
        "capture_iou_reproduced": abs(iou - RECORDED_CAPTURE_IOU) <= TOLERANCE
        and expected == RECORDED_CAPTURE_IOU,
    }
    model = {"provenance": "own_previous_model", "sha256": loaded.model_sha256}
    return {
        "schema": "page-gate-reproduction-1",
        "evidence": evidence("gt_full_frame_validation", model),
        "passed": all(value for key, value in checks.items() if isinstance(value, bool)),
        "checks": checks,
        "tolerance": TOLERANCE,
        "summary": {"capture": capture},
        "model": {
            "path": str(weights),
            "sha256": loaded.model_sha256,
            "schema": loaded.metadata["checkpoint_schema"],
            "model_id": loaded.model_id,
        },
        "archive": {
            "path": ARCHIVE_LABEL,
            "sha256": archive["sha256"],
            "recorded_device": archive["experiment"]["device"],
            "recorded_torch": archive["experiment"]["torch_version"],
        },
        "dataset_manifest_sha256": samples["sha256"],
        "samples": str(samples["directory"]),
        "attribution": SMARTDOC_CITATION,
        "metric": archive["experiment"]["mask_metric_policy"],
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "rows": rows,
    }


def _write_json(path: Path, value: dict) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("install", "reproduce"):
        command = commands.add_parser(name)
        command.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
        command.add_argument("--threads", type=int, default=2)
    commands.choices["install"].add_argument("--out", type=Path, default=DEFAULT_WEIGHTS_DIR)
    reproduce_command = commands.choices["reproduce"]
    reproduce_command.add_argument("--smartdoc", type=Path, default=DEFAULT_SMARTDOC)
    reproduce_command.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS_DIR / EXPORT_NAME
    )
    reproduce_command.add_argument(
        "--report", type=Path, default=DEFAULT_WEIGHTS_DIR / "page-gate-v0-reproduction.json"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(args.threads)
    archive = read_archive(args.zip)
    if args.command == "install":
        print(json.dumps(install(archive, read_provenance(archive), args.out), indent=2))
        return 0
    samples = read_gate_samples(args.smartdoc, archive)
    report = reproduce(archive, samples, args.weights.expanduser())
    _write_json(args.report, report)
    print(
        json.dumps(
            {"passed": report["passed"], "checks": report["checks"], "summary": report["summary"]},
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
