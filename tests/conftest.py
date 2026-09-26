"""Shared test options and dataset fixtures.

Tests marked ``training`` run only with ``--run-training``. The scene and paper datasets are
built once per run and shared read-only; tests that change files work on a copy.
"""

import hashlib
import json
import shutil

import numpy as np
import pytest
from PIL import Image

SCENE_PHOTO_SIZE = (128, 160)


def pytest_addoption(parser):
    parser.addoption(
        "--run-training",
        action="store_true",
        default=False,
        help="Explicitly allow tests marked training (may perform CPU training).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-training"):
        return
    skip_training = pytest.mark.skip(reason="training requires explicit --run-training")
    for item in items:
        if item.get_closest_marker("training") is not None:
            item.add_marker(skip_training)


def _read_manifest(directory):
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(directory, value):
    (directory / "manifest.json").write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(scope="session")
def scene_source(tmp_path_factory):
    """Paired-scene dataset with one document per split, built once per run."""
    from flattener.synthetic.paired_scenes import PairedSceneConfig, generate_paired_scene_dataset

    root = tmp_path_factory.mktemp("coordinate-data-only")
    directory = root / "scenes"
    generate_paired_scene_dataset(
        directory,
        PairedSceneConfig(
            train_documents=1,
            validation_documents=1,
            test_documents=1,
            image_size=SCENE_PHOTO_SIZE,
            seed=397,
        ),
    )
    try:
        yield directory
    finally:
        shutil.rmtree(root)


@pytest.fixture(scope="session")
def paper_source(scene_source, tmp_path_factory):
    """Scene dataset plus a three-split paper-augmentation block, built once per run."""
    from flattener.learning import scene_data
    from flattener.synthetic import paper

    root = tmp_path_factory.mktemp("paper-adapter-contract")
    directory = root / "dataset"
    shutil.copytree(scene_source, directory)
    (directory / "files").mkdir()
    manifest = _read_manifest(directory)
    block = {
        "schema_version": "paper-augmentation-1.0",
        "complete": True,
        "photo_size": list(SCENE_PHOTO_SIZE),
        "generator_code_sha256": {
            "synthetic/paper.py": scene_data._data_hashes(True)["synthetic/paper.py"]
        },
        "templates": [],
        "captures": [],
        "records": [],
    }
    for index, split in enumerate(("train", "validation", "test")):
        flat = np.full((128, 160, 3), 245 - index, np.uint8)
        flat[20:105:8, 15:145] = 30 + index
        filename = f"files/template-{split}.png"
        Image.fromarray(flat).save(directory / filename)
        template = {
            "id": f"template-{split}",
            "document_group": f"smartdoc:fixture-{split}",
            "split": split,
            "capture_id": f"flat-capture-{split}",
            "source_dataset": "smartdoc2015-ch1",
            "license_spdx": "CC-BY-4.0",
            "path": filename,
            "sha256": scene_data._sha256(directory / filename),
            "bytes": (directory / filename).stat().st_size,
            "content_sha256": hashlib.sha256(flat.tobytes()).hexdigest(),
        }
        block["templates"].append(template)
        seed = 720 + index
        params, _, sampling = paper.sample_paper_geometry(
            "gentle", seed, size_bin="medium", photo_size=SCENE_PHOTO_SIZE, output_size=(137, 149)
        )
        sample = paper.generate_paper_sample(flat, seed + 10, params, photo_size=SCENE_PHOTO_SIZE)
        capture_rgb = np.full((*SCENE_PHOTO_SIZE, 3), 100 + index, np.uint8)
        capture_name = f"files/capture-{split}.png"
        Image.fromarray(capture_rgb).save(directory / capture_name)
        capture = {
            "id": f"capture-{split}",
            "document_group": template["document_group"],
            "split": split,
            "source_dataset": "smartdoc2015-ch1",
            "license_spdx": "CC-BY-4.0",
            "path": capture_name,
            "sha256": scene_data._sha256(directory / capture_name),
            "bytes": (directory / capture_name).stat().st_size,
            "original_size": list(SCENE_PHOTO_SIZE),
            "corners_xy": [[20, 18], [138, 21], [140, 109], [18, 111]],
        }
        block["captures"].append(capture)
        capture_geometry = paper.make_capture_geometry(
            capture["corners_xy"], SCENE_PHOTO_SIZE, (192, 144), SCENE_PHOTO_SIZE
        )
        for kind, image, geometry in (
            ("simulated", sample.image, sample.geometry),
            ("capture", capture_rgb, capture_geometry),
        ):
            identifier = f"paper-{split}-{kind}"
            fraction = float(geometry.photo_page_mask.mean())
            size_bin = "small" if fraction < 0.25 else "medium" if fraction < 0.5 else "large"
            record = {
                "id": identifier,
                "source": "paper",
                "kind": kind,
                "document_group": template["document_group"],
                "split": split,
                "capture_session": f"session-{split}-{kind}",
                "has_page": True,
                "template_id": template["id"],
                "template_sha256": template["sha256"],
                "family": "gentle" if kind == "simulated" else "capture_perspective",
                "size_bin": size_bin,
            }
            metadata = {key: record[key] for key in scene_data._PAPER_FIELDS}
            metadata.update(
                schema_version="paper-augmentation-1.0",
                sample_id=identifier,
                image_size=list(SCENE_PHOTO_SIZE),
                is_real_photo=kind == "capture",
                source_template_is_real_capture=True,
                template_content_sha256=template["content_sha256"],
                geometry=geometry.metadata,
            )
            if kind == "simulated":
                metadata.update(geometry_seed=seed, sampling=sampling)
            else:
                record["capture_id"] = capture["id"]
                metadata.update(
                    capture_id=capture["id"],
                    capture_sha256=capture["sha256"],
                    original_size=capture["original_size"],
                    corners_xy=capture["corners_xy"],
                    photometric_augmented=False,
                )
            record["paths"] = {
                name: f"files/{identifier}-{name}"
                for name in ("image.png", "mask.png", "metadata.json")
            }
            Image.fromarray(image).save(directory / record["paths"]["image.png"])
            Image.fromarray(geometry.photo_page_mask.astype(np.uint8) * 255).save(
                directory / record["paths"]["mask.png"]
            )
            (directory / record["paths"]["metadata.json"]).write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            record["sha256"] = {
                name: scene_data._sha256(directory / relative)
                for name, relative in record["paths"].items()
            }
            block["records"].append(record)
    manifest["collection"] = {"paper_augmentation": block}
    _write_manifest(directory, manifest)
    try:
        yield directory
    finally:
        shutil.rmtree(root)


@pytest.fixture
def mutable_paper(paper_source, tmp_path):
    directory = tmp_path / "paper-copy"
    shutil.copytree(paper_source, directory)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


@pytest.fixture
def tiny_docunet(tmp_path, monkeypatch):
    """65 tiny scans and 130 predictions; the metric area is shrunk to keep it fast."""
    from flattener.benchmark import metrics

    monkeypatch.setattr(metrics, "METRIC_AREA", 300)
    rng = np.random.default_rng(7)
    scans = tmp_path / "scans"
    predictions = tmp_path / "pred"
    scans.mkdir()
    predictions.mkdir()
    for k in range(1, 66):
        scan = rng.integers(0, 256, size=(24, 18, 4), dtype=np.uint8)
        Image.fromarray(scan, "RGBA").save(scans / f"docunet-reference-{k}.png")
        for m in (1, 2):
            prediction = rng.integers(0, 256, size=(20, 16, 3), dtype=np.uint8)
            Image.fromarray(prediction).save(predictions / f"{k}_{m}.png")
    return scans, predictions
