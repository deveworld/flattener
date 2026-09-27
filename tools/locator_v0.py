"""Install, reproduce and audit the v0 page gate from the locator study archive.

The archive is ``outputs/coordinate-fold/locator.zip``: the locator study's ``best.pt``
(region-context-unet-2.0, 140,545 parameters), its training script, records and code.

Usage (run at low CPU priority, for example ``nice -n 19``):
  .venv/bin/python tools/locator_v0.py install [--zip Z] [--dataset D] [--out DIR]
  .venv/bin/python tools/locator_v0.py reproduce [--zip Z] [--dataset D] [--weights W]
      [--domains capture|all] [--report R]
  .venv/bin/python tools/locator_v0.py exposure [--zip Z] [--dataset D] [--output JSON]

Steps:
  install    Copies ``best.pt`` to DIR/locator-v0-pilot.pt (schema
             independent-locator-pilot-1.0) and exports DIR/page-gate-v0.pt (schema
             region-context-1, weights plus a provenance record), each with a ``.sha256``
             sidecar. DIR defaults to ~/.cache/flattener-sota-20260926/scan-weights.
  reproduce  Runs the weights on the recorded validation samples (``best-validation.json``
             rows, read from ``outputs/dataset``) with the study's metric: sigmoid,
             bilinear upsampling to 384x288, threshold 0.5. It passes when the capture IoU
             is within 1e-5 of the recorded 0.9508340768516064, every per-image loss is
             within 1e-5 of the recorded loss, and the archived model code gives
             probabilities within 1e-5 of the port. The report goes to --report.
  exposure   Writes the SmartDoc templates that v0 trained on or was selected with
             (default: src/flattener/regions/locator_v0_smartdoc.json).

Every archive member used is checked against the archive's ``manifest.json`` hashes, and
every dataset file against ``outputs/dataset/manifest.json``.
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
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flattener.evidence import evidence  # noqa: E402
from flattener.regions import context  # noqa: E402

OUTPUTS = Path(os.environ.get("FLATTENER_OUTPUTS", ROOT / "outputs"))
DEFAULT_ZIP = OUTPUTS / "coordinate-fold" / "locator.zip"
DEFAULT_DATASET = OUTPUTS / "dataset"
DEFAULT_WEIGHTS_DIR = Path("~/.cache/flattener-sota-20260926/scan-weights").expanduser()
DEFAULT_EXPOSURE = ROOT / "src" / "flattener" / "regions" / context.SMARTDOC_EXPOSURE_FILE
PILOT_NAME = "locator-v0-pilot.pt"
EXPORT_NAME = "page-gate-v0.pt"
ARCHIVE_LABEL = "outputs/coordinate-fold/locator.zip"
PHOTO = (384, 288)
RECORDED_CAPTURE_IOU = 0.9508340768516064
TOLERANCE = 1e-5
DOMAINS = ("authored", "simulated", "capture")
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


def read_dataset(directory: Path, experiment: dict) -> dict:
    """The dataset manifest; its sha256 must be the one the study recorded."""
    manifest_path = Path(directory) / "manifest.json"
    if not manifest_path.is_file():
        _fail(f"{manifest_path} does not exist")
    digest = _file_sha(manifest_path)
    recorded = [v for k, v in experiment["source_sha256"].items() if k.endswith("/manifest.json")]
    if recorded != [digest]:
        _fail("dataset manifest differs from the one the locator study used")
    manifest = json.loads(manifest_path.read_bytes())
    paper = manifest["collection"]["paper_augmentation"]
    records = {r["id"]: r for r in manifest["samples"]}
    records.update({r["id"]: r for r in paper["records"]})
    return {
        "directory": Path(directory),
        "manifest": manifest,
        "sha256": digest,
        "paper": paper,
        "records": records,
    }


def _kind(record: dict) -> str:
    if record.get("source", "authored") != "paper":
        return "authored"
    return record.get("kind", "simulated")


def train_records(dataset: dict) -> list[dict]:
    """Records the study trained on: printed pages of the train split, paper records included."""
    return [
        r for r in dataset["records"].values() if r["split"] == "train" and r["has_page"] is True
    ]


def _checked_counts(archive: dict, dataset: dict) -> Counter:
    counts = Counter(_kind(r) for r in train_records(dataset))
    if dict(counts) != archive["experiment"]["train_counts"]:
        _fail(f"train counts {dict(counts)} differ from the study's")
    return counts


def _scratch_basis(archive: dict) -> str:
    text = archive["members"]["run.py"].decode()
    if (
        "model = ContextPageSegmenter(config).to(device)" not in text
        or text.count("load_state_dict(") != 1
        or 'saved = torch.load(WORK / "best.pt"' not in text
    ):
        _fail("run.py does not show a scratch start")
    return (
        f"run.py (sha256 {_sha(archive['members']['run.py'])}) seeds the generator, builds "
        "ContextPageSegmenter(config) and trains it; its only load_state_dict reloads its "
        "own best.pt after training."
    )


def provenance(archive: dict, dataset: dict) -> dict:
    """The provenance record of the region-context-1 export."""
    experiment, recorded = archive["experiment"], archive["validation"]["summary"]
    counts = _checked_counts(archive, dataset)
    manifest, paper = dataset["manifest"], dataset["paper"]
    sources = {t["source_dataset"] for t in paper["templates"]} | {
        c["source_dataset"] for c in paper["captures"]
    }
    licenses = {t["license_spdx"] for t in paper["templates"]} | {
        c["license_spdx"] for c in paper["captures"]
    }
    if sources != {"smartdoc2015-ch1"} or licenses != {"CC-BY-4.0"}:
        _fail(f"paper sources {sources} or licenses {licenses} are not SmartDoc CC BY 4.0")
    if manifest["external_images"] is not False or manifest["is_real_photo"] is not False:
        _fail("authored scenes record external or real-photo images")
    training_keys = (
        "seed",
        "epochs",
        "best_epoch",
        "batch_size",
        "device",
        "torch_version",
        "loss",
        "optimizer",
        "learning_rate",
        "gradient_clip_norm",
        "rotation_probability",
        "background_probability",
        "augmentation",
        "selection",
        "train_counts",
        "validation_counts",
    )
    return {
        "name": "page-gate-v0",
        "model_version": context.MODEL_VERSION,
        "parameter_count": experiment["parameter_count"],
        "source": {
            "archive": ARCHIVE_LABEL,
            "archive_sha256": archive["sha256"],
            "member": "best.pt",
            "member_sha256": archive["best_sha256"],
            "member_schema": context.PILOT_SCHEMA,
            "experiment_sha256": _sha(archive["members"]["experiment.json"]),
        },
        "track": "release",
        "initialization": "scratch",
        "lineage": [{"kind": "scratch"}],
        "initialization_basis": _scratch_basis(archive),
        "training": {key: experiment[key] for key in training_keys},
        "dataset": {
            "path": "outputs/dataset",
            "manifest_sha256": dataset["sha256"],
            "schema_version": manifest["schema_version"],
            "include_paper_augmentation": True,
        },
        "data": [
            {
                "kind": "authored",
                "count": counts["authored"],
                "source": manifest["source"],
                "generator": f"{manifest['schema_version']} paired scenes",
                "license": "Apache-2.0 AND OFL-1.1",
                "terms": manifest["terms"],
            },
            {
                "kind": "simulated",
                "count": counts["simulated"],
                "source": "synthetic/paper.py cylinder and perspective augmentation of "
                "SmartDoc 2015 Challenge 1 templates",
                "generator_sha256": paper["generator_code_sha256"],
                "license": "CC-BY-4.0",
            },
            {
                "kind": "capture",
                "count": counts["capture"],
                "source": "SmartDoc 2015 Challenge 1 frames with four-corner planar masks",
                "license": "CC-BY-4.0",
            },
        ],
        "attribution": SMARTDOC_CITATION,
        "license_notes": [
            "SmartDoc 2015 Challenge 1 is CC BY 4.0; its readme says original images remain "
            "the property of their respective authors.",
        ],
        "recorded_validation": {
            "measurement": "mask IoU, 384x288, threshold 0.5, mean over images",
            "all": recorded["all"]["iou"],
            "authored": recorded["domains"]["authored"]["iou"],
            "simulated": recorded["domains"]["simulated"]["iou"],
            "capture": recorded["domains"]["capture"]["iou"],
            "small_pages": recorded["by_size"]["small"]["iou"],
            "count": recorded["all"]["count"],
        },
        "limitations": list(experiment["limitations"])
        + ["trained on printed pages only; no-page negatives were never evaluated"],
        "smartdoc_exposure": f"flattener/regions/{context.SMARTDOC_EXPOSURE_FILE}",
    }


def install(archive: dict, dataset: dict, out: Path) -> dict:
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
        export, loaded.model.state_dict(), loaded.config, provenance(archive, dataset)
    )
    exported = context.load_context_segmenter(export)
    for key, value in loaded.model.state_dict().items():
        if not torch.equal(value, exported.model.state_dict()[key]):
            _fail(f"exported weight {key} differs")
    return {
        "pilot": {"path": str(pilot), "sha256": archive["best_sha256"]},
        "export": {"path": str(export), "sha256": digest, "schema": context.CHECKPOINT_SCHEMA},
        "parameter_count": loaded.metadata["parameter_count"],
    }


def _load_png(dataset: dict, record: dict, name: str) -> np.ndarray:
    path = dataset["directory"] / record["paths"][name]
    data = path.read_bytes()
    if _sha(data) != record["sha256"][name]:
        _fail(f"{path} differs from the dataset manifest")
    with Image.open(io.BytesIO(data)) as encoded:
        return np.array(encoded, copy=True)


def load_sample(dataset: dict, record: dict) -> tuple[np.ndarray, np.ndarray]:
    """RGB image and boolean page mask of one validation record at the photo size."""
    image = _load_png(dataset, record, "image.png")
    values = _load_png(dataset, record, "mask.png")
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


def reproduce(archive: dict, dataset: dict, weights: Path, domains=("capture",)) -> dict:
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
        if row["domain"] not in domains:
            continue
        record = dataset["records"].get(row["id"])
        if record is None or record["split"] != "validation" or _kind(record) != row["domain"]:
            _fail(f"{row['id']} is not a {row['domain']} validation record")
        image, mask = load_sample(dataset, record)
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
    summary = {}
    for domain in domains:
        selected = [r for r in rows if r["domain"] == domain]
        iou = float(np.mean([r["iou"] for r in selected]))
        expected = recorded["summary"]["domains"][domain]["iou"]
        summary[domain] = {
            "count": len(selected),
            "iou": iou,
            "recorded_iou": expected,
            "iou_difference": iou - expected,
            "max_abs_image_iou_difference": max(abs(r["delta"]["iou"]) for r in selected),
            "max_abs_image_loss_difference": max(abs(r["delta"]["loss"]) for r in selected),
            "predicted_pixel_mismatches": sum(
                r["delta"]["predicted_pixels"] != 0 for r in selected
            ),
        }
    capture = summary.get("capture")
    checks = {
        "archived_code_max_abs_probability_difference": parity,
        "archived_code_parity": parity <= TOLERANCE,
        "recorded_loss_agreement": all(
            s["max_abs_image_loss_difference"] <= TOLERANCE for s in summary.values()
        ),
    }
    if capture is not None:
        checks["capture_iou_reproduced"] = (
            abs(capture["iou"] - RECORDED_CAPTURE_IOU) <= TOLERANCE
            and capture["recorded_iou"] == RECORDED_CAPTURE_IOU
        )
    model = {"provenance": "own_previous_model", "sha256": loaded.model_sha256}
    return {
        "schema": "page-gate-reproduction-1",
        "evidence": evidence("gt_full_frame_validation", model),
        "passed": all(value for key, value in checks.items() if isinstance(value, bool)),
        "checks": checks,
        "tolerance": TOLERANCE,
        "summary": summary,
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
        "dataset_manifest_sha256": dataset["sha256"],
        "metric": archive["experiment"]["mask_metric_policy"],
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "rows": rows,
    }


def exposure(archive: dict, dataset: dict) -> dict:
    """SmartDoc templates the v0 gate trained on, was selected with, or never saw."""
    paper = dataset["paper"]
    _checked_counts(archive, dataset)
    used = Counter(
        (r["template_id"], _kind(r), "train")
        for r in train_records(dataset)
        if _kind(r) != "authored"
    )
    for row in archive["validation"]["rows"]:
        if row["domain"] == "authored":
            continue
        record = dataset["records"][row["id"]]
        if record["split"] != "validation":
            _fail(f"{row['id']} is a {record['split']} record")
        used[(record["template_id"], row["domain"], "validation")] += 1
    frames: dict[str, set] = {}
    for record in dataset["records"].values():
        if _kind(record) == "capture" and record["split"] in ("train", "validation"):
            frames.setdefault(record["template_id"], set()).add(record["capture_id"])
    captures = {c["id"]: c for c in paper["captures"]}
    templates, groups = [], {"train": [], "selection": [], "none": []}
    for template in sorted(paper["templates"], key=lambda t: t["id"]):
        name = template["archive_member"].rsplit("/", 1)[-1].removesuffix(".png")
        train = {k: used[(template["id"], k, "train")] for k in ("simulated", "capture")}
        selection = {k: used[(template["id"], k, "validation")] for k in ("simulated", "capture")}
        level = (
            "train" if any(train.values()) else "selection" if any(selection.values()) else ("none")
        )
        expected = {"train": "train", "selection": "validation", "none": "test"}[level]
        if template["split"] != expected:
            _fail(f"{name}: dataset split {template['split']} with exposure {level}")
        members = sorted(captures[c]["archive_member"] for c in frames.get(template["id"], ()))
        groups[level].append(name)
        templates.append(
            {
                "model_name": name,
                "template_id": template["id"],
                "document_group": template["document_group"],
                "dataset_split": template["split"],
                "exposure": level,
                "training_records": train,
                "selection_records": selection,
                "frames_used": members,
                "backgrounds_used": sorted({m.split("/", 1)[0] for m in members}),
                "template_archive_member": template["archive_member"],
                "original_document_source": template["original_document_source"],
            }
        )
    backgrounds = sorted({c["metadata_row"]["bg_name"] for c in paper["captures"]})
    return {
        "schema": "smartdoc-exposure-1",
        "model": {
            "name": "page-gate-v0",
            "model_version": context.MODEL_VERSION,
            "checkpoint_sha256": archive["best_sha256"],
            "archive": ARCHIVE_LABEL,
            "archive_sha256": archive["sha256"],
        },
        "dataset": {
            "path": "outputs/dataset",
            "manifest_sha256": dataset["sha256"],
            "source_dataset": "smartdoc2015-ch1",
            "split_rule": paper["split_protocol"]["rule"],
        },
        "rule": (
            "Evaluations of page-gate-v0 on SmartDoc drop every frame of an "
            "exclude_from_evaluation document, on any background. selection_exposed documents "
            "chose the checkpoint epoch; report them apart from unexposed ones."
        ),
        "exclude_from_evaluation": groups["train"],
        "selection_exposed": groups["selection"],
        "unexposed": groups["none"],
        "backgrounds_seen_in_training": backgrounds,
        "notes": [
            "Grouping follows document_group: all magazine and tax documents are train.",
            "Frames of every background in the dataset (background01-04) were used; "
            "background05 frames were never used.",
            "Capture masks come from four-corner planar annotations on 1920x1080 frames "
            "stored at 384x288 without keeping the aspect ratio.",
        ],
        "templates": templates,
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
    for name in ("install", "reproduce", "exposure"):
        command = commands.add_parser(name)
        command.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
        command.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
        command.add_argument("--threads", type=int, default=2)
    commands.choices["install"].add_argument("--out", type=Path, default=DEFAULT_WEIGHTS_DIR)
    reproduce_command = commands.choices["reproduce"]
    reproduce_command.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS_DIR / EXPORT_NAME
    )
    reproduce_command.add_argument("--domains", choices=("capture", "all"), default="capture")
    reproduce_command.add_argument(
        "--report", type=Path, default=DEFAULT_WEIGHTS_DIR / "page-gate-v0-reproduction.json"
    )
    commands.choices["exposure"].add_argument("--output", type=Path, default=DEFAULT_EXPOSURE)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(args.threads)
    archive = read_archive(args.zip)
    dataset = read_dataset(args.dataset, archive["experiment"])
    if args.command == "install":
        print(json.dumps(install(archive, dataset, args.out), indent=2))
        return 0
    if args.command == "exposure":
        value = exposure(archive, dataset)
        _write_json(args.output, value)
        print(
            json.dumps(
                {
                    key: value[key]
                    for key in ("exclude_from_evaluation", "selection_exposed", "unexposed")
                },
                indent=2,
            )
        )
        return 0
    domains = DOMAINS if args.domains == "all" else ("capture",)
    report = reproduce(archive, dataset, args.weights.expanduser(), domains)
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
