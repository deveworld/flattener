"""tools/locator_v0.py: v0 install, the recorded capture IoU and the SmartDoc exposure file.

The archive checks need ``outputs/coordinate-fold/locator.zip`` and ``outputs/dataset``
(or ``FLATTENER_OUTPUTS`` pointing at an ``outputs`` directory that has both).
"""

import importlib.util
from pathlib import Path

import pytest

from flattener.regions import context

TOOL = Path(__file__).resolve().parents[1] / "tools" / "locator_v0.py"
_spec = importlib.util.spec_from_file_location("locator_v0", TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

needs_archive = pytest.mark.skipif(
    not (tool.DEFAULT_ZIP.is_file() and (tool.DEFAULT_DATASET / "manifest.json").is_file()),
    reason="outputs/coordinate-fold/locator.zip or outputs/dataset is not present",
)


@pytest.fixture(scope="module")
def archive():
    return tool.read_archive(tool.DEFAULT_ZIP)


@pytest.fixture(scope="module")
def dataset(archive):
    return tool.read_dataset(tool.DEFAULT_DATASET, archive["experiment"])


@needs_archive
def test_install_and_reproduce_the_capture_iou(archive, dataset, tmp_path):
    installed = tool.install(archive, dataset, tmp_path)
    assert installed["pilot"]["sha256"] == archive["best_sha256"]
    assert installed["parameter_count"] == 140_545
    export = context.load_context_segmenter(tmp_path / tool.EXPORT_NAME)
    provenance = export.metadata["provenance"]
    assert (provenance["track"], provenance["initialization"]) == ("release", "scratch")
    assert [item["kind"] for item in provenance["data"]] == ["authored", "simulated", "capture"]
    assert [item["count"] for item in provenance["data"]] == [3840, 1584, 528]
    report = tool.reproduce(archive, dataset, tmp_path / tool.EXPORT_NAME)
    assert report["passed"], report["checks"]
    capture = report["summary"]["capture"]
    assert capture["count"] == 48
    assert abs(capture["iou"] - tool.RECORDED_CAPTURE_IOU) <= 1e-5
    assert capture["max_abs_image_loss_difference"] <= 1e-5
    assert report["checks"]["archived_code_max_abs_probability_difference"] <= 1e-5
    assert report["evidence"]["model_provenance"] == "own_previous_model"


@needs_archive
def test_committed_exposure_file_matches_the_archive(archive, dataset):
    assert tool.exposure(archive, dataset) == context.smartdoc_exposure()
