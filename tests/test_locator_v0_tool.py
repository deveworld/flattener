"""tools/locator_v0.py: v0 install rebuilds the recorded export, and reproduce gives the
recorded capture IoU.

Both need ``outputs/models/locator.zip``; reproduce also needs the SmartDoc gate samples
(``outputs/benchmarks/smartdoc/manifest.json`` and ``gate-v0/``). ``FLATTENER_OUTPUTS`` may
point at another ``outputs`` directory that has them.
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
    not tool.DEFAULT_ZIP.is_file(), reason="outputs/models/locator.zip is not present"
)
needs_samples = pytest.mark.skipif(
    not (tool.DEFAULT_SMARTDOC / "manifest.json").is_file(),
    reason="outputs/benchmarks/smartdoc is not present",
)


@pytest.fixture(scope="module")
def archive():
    return tool.read_archive(tool.DEFAULT_ZIP)


@pytest.fixture(scope="module")
def installed(archive, tmp_path_factory):
    out = tmp_path_factory.mktemp("page-gate")
    return out, tool.install(archive, tool.read_provenance(archive), out)


@needs_archive
def test_install_rebuilds_the_recorded_export(archive, installed):
    out, result = installed
    assert result["export"]["sha256"] == tool.RECORDED_EXPORT_SHA256
    assert result["pilot"]["sha256"] == archive["best_sha256"]
    assert result["parameter_count"] == 140_545
    provenance = context.load_context_segmenter(out / tool.EXPORT_NAME).metadata["provenance"]
    assert (provenance["track"], provenance["initialization"]) == ("release", "scratch")
    assert [item["kind"] for item in provenance["data"]] == ["authored", "simulated", "capture"]
    assert [item["count"] for item in provenance["data"]] == [3840, 1584, 528]


@needs_archive
@needs_samples
def test_reproduce_the_capture_iou(archive, installed):
    out, _ = installed
    samples = tool.read_gate_samples(tool.DEFAULT_SMARTDOC, archive)
    report = tool.reproduce(archive, samples, out / tool.EXPORT_NAME)
    assert report["passed"], report["checks"]
    capture = report["summary"]["capture"]
    assert capture["count"] == 48 and len(report["rows"]) == 48
    assert abs(capture["iou"] - tool.RECORDED_CAPTURE_IOU) <= 1e-5
    assert capture["max_abs_image_loss_difference"] <= 1e-5
    assert report["checks"]["archived_code_max_abs_probability_difference"] <= 1e-5
    assert report["evidence"]["model_provenance"] == "own_previous_model"
