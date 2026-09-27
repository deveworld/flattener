"""Scan reason codes, guidance, outcomes and JSON round-trips of ScanPlan and ScanResult."""

import json
import math

import numpy as np
import pytest

from flattener import FlattenerError, Status
from flattener.scan import records as rec
from flattener.scan.output_policy import plan_output


def json_round_trip(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def make_grid(gh=31, gw=45):
    x, y = np.meshgrid(np.linspace(-0.6, 0.55, gw), np.linspace(-0.8, 0.85, gh))
    bend = 0.03 * np.sin(np.pi * (x + 0.6) / 1.15)
    return np.stack((x + 0.01 * y, y + bend), axis=-1).astype(np.float32)


def make_plan(**changes):
    grid = changes.pop("grid", make_grid())
    source = rec.SourceInfo(
        size=(3000, 4000),
        full_size=(6000, 8000),
        scale=2,
        format="JPEG",
        file_sha256="ab" * 32,
        exif_orientation=6,
        focal_length_35mm=26.0,
        icc_sha256="cd" * 32,
    )
    fields = dict(
        source=source,
        output=plan_output(make_grid(), source.size),
        grid=grid,
        region={"box": [100, 50, 3900, 2950], "source": "gate", "dominance": np.float32(0.97)},
        angle_deg=-3.25,
        quarter=1,
        orientation_probs=(0.01, 0.97, 0.015, 0.005),
        views=({"pass": "A", "box": (100, 50, 3900, 2950), "angle_deg": 86.75},),
        inset=(0.0, 0.0),
        models=({"id": "r4", "sha256": "ef" * 32, "track": "release", "shippable": True},),
        thresholds={"version": "m1-0", "coverage": 0.95, "drift": 0.04},
        provenance={"region": "auto", "orientation": "manual"},
    )
    fields.update(changes)
    return rec.ScanPlan(**fields)


def test_outcomes_and_exit_codes():
    info = rec.reason("page_fills_frame")
    warning = rec.reason("possible_occlusion", solidity=0.8)
    assert rec.decide_outcome("dewarp", [info]) == rec.Outcome.RECTIFIED
    assert rec.decide_outcome("dewarp", [info, warning]) == rec.Outcome.RECTIFIED_WITH_WARNINGS
    assert rec.decide_outcome("perspective", []) == rec.Outcome.PERSPECTIVE
    assert rec.decide_outcome("original", [warning]) == rec.Outcome.ORIGINAL_RETURNED
    assert rec.decide_outcome(None) == rec.Outcome.FAILED
    with pytest.raises(ValueError):
        rec.decide_outcome("crop")
    assert rec.exit_code(["rectified", rec.Outcome.RECTIFIED_WITH_WARNINGS]) == 0
    assert rec.exit_code(["rectified", "perspective"]) == rec.EXIT_FALLBACK == 1
    assert rec.exit_code(["original_returned"]) == 1
    assert rec.exit_code(["failed"]) == 1
    assert rec.EXIT_USAGE == 2


def test_plan_json_round_trip_is_exact():
    plan = make_plan()
    data = json_round_trip(plan.to_dict())
    again = rec.ScanPlan.from_dict(data)
    assert again == plan and again.plan_sha256 == plan.plan_sha256
    assert again.to_dict() == plan.to_dict()
    assert np.array_equal(again.grid.astype(np.float32), make_grid())
    assert again.grid.dtype == np.float64 and not again.grid.flags.writeable
    assert again.region["dominance"] == float(np.float32(0.97))
    assert again.views[0]["box"] == [100, 50, 3900, 2950]
    assert data["plan_sha256"] == plan.plan_sha256
    body = rec.canonical_json(plan.body())
    assert plan.plan_sha256 == __import__("hashlib").sha256(body).hexdigest()
    assert (plan.nodes, plan.fill_rgb) == ("flattener", (255, 255, 255))
    uvdoc = make_plan(nodes="uvdoc", fill_rgb=[10, 20, 30])
    again = rec.ScanPlan.from_dict(json_round_trip(uvdoc.to_dict()))
    assert again == uvdoc and (again.nodes, again.fill_rgb) == ("uvdoc", (10, 20, 30))


def test_plan_sha256_tracks_every_value():
    plan = make_plan()
    grid = make_grid().astype(np.float64)
    grid[7, 9, 0] = np.nextafter(grid[7, 9, 0], np.inf)
    assert make_plan(grid=grid).plan_sha256 != plan.plan_sha256
    assert make_plan(quarter=2).plan_sha256 != plan.plan_sha256
    assert make_plan(quantize="truncate").plan_sha256 != plan.plan_sha256
    assert make_plan(nodes="uvdoc").plan_sha256 != plan.plan_sha256
    assert make_plan(fill_rgb=(0, 0, 0)).plan_sha256 != plan.plan_sha256
    reordered = {"drift": 0.04, "coverage": 0.95, "version": "m1-0"}
    assert make_plan(thresholds=reordered).plan_sha256 == plan.plan_sha256
    assert hash(make_plan()) == hash(plan)


def test_tampered_plan_is_refused():
    data = json_round_trip(make_plan().to_dict())
    data["grid"][0][0][0] += 1e-9
    with pytest.raises(FlattenerError) as caught:
        rec.ScanPlan.from_dict(data)
    assert (caught.value.status, caught.value.reason_code) == (
        Status.INPUT_ERROR,
        "plan_sha256_mismatch",
    )
    assert rec.ScanPlan.from_dict(data, verify=False).plan_sha256 != data["plan_sha256"]
    extra = {**make_plan().to_dict(), "note": "unhashed"}
    with pytest.raises(FlattenerError) as caught:
        rec.ScanPlan.from_dict(extra)
    assert caught.value.reason_code == "invalid_plan"
    assert set(make_plan().to_dict()) == rec.PLAN_KEYS


@pytest.mark.parametrize(
    "changes",
    [
        {"nodes": "center"},
        {"fill_rgb": (0, 0)},
        {"fill_rgb": (0, 0, 256)},
    ],
)
def test_invalid_plans_are_input_errors(changes):
    with pytest.raises(FlattenerError) as caught:
        make_plan(**changes)
    assert (caught.value.status, caught.value.reason_code) == (Status.INPUT_ERROR, "invalid_plan")


def test_perspective_plan_round_trip():
    homography = np.array([[0.5, 0.02, 0.01], [-0.01, 0.7, 0.03], [0.05, -0.04, 1.0]])
    plan = make_plan(kind="perspective", grid=None, homography=homography)
    again = rec.ScanPlan.from_dict(json_round_trip(plan.to_dict()))
    assert again == plan and np.array_equal(again.homography, homography)
    with pytest.raises(FlattenerError):
        rec.ScanPlan.from_dict({**plan.to_dict(), "schema": "other"})
    broken = {key: value for key, value in plan.to_dict().items() if key != "output"}
    with pytest.raises(FlattenerError) as caught:
        rec.ScanPlan.from_dict(broken)
    assert caught.value.reason_code == "invalid_plan"


def make_result(plan):
    reasons = [
        rec.reason("source_downscaled", scale=2, full_size=[6000, 8000]),
        rec.reason("page_cut_off", sides=["top"]),
    ]
    return rec.ScanResult(
        status=Status.SUCCESS,
        outcome=rec.decide_outcome("dewarp", reasons),
        reasons=reasons,
        guidance=rec.collect_guidance(reasons),
        candidates={"orientations": [(1, np.float32(0.97)), (3, 0.02)], "regions": []},
        image=np.zeros((4, 3, 3), np.uint8),
        valid_mask=np.ones((4, 3), bool),
        plan=plan,
        source=plan.source,
        qa={"coverage": np.float64(0.991), "drift": 0.012, "orientation_margin": math.nan},
        timings={"read": 0.05, "infer": np.float32(0.19)},
        metadata={"validation": {"checked_cells": np.int64(1320), "fold_check": "passed"}},
        models=[{"id": "r4", "sha256": "ef" * 32, "shippable": True}],
        shippable=True,
        evidence={"evidence_kind": "predictions"},
        input="photo.jpg",
    )


def test_result_json_round_trip():
    result = make_result(make_plan())
    data = result.to_dict()
    text = json.dumps(data, ensure_ascii=False, allow_nan=False)
    again = rec.ScanResult.from_dict(json.loads(text))
    assert again.to_dict() == data
    assert again.outcome == rec.Outcome.RECTIFIED_WITH_WARNINGS
    assert again.plan == result.plan and data["plan_sha256"] == result.plan.plan_sha256
    assert again.image is None and again.valid_mask is None
    assert data["qa"]["orientation_margin"] is None
    assert data["metadata"]["validation"]["checked_cells"] == 1320
    assert data["guidance"][0]["message_ko"] == "위쪽이 잘렸어요. 조금 물러나 찍어 주세요"
    assert data["reasons"][0] == {
        "code": "source_downscaled",
        "severity": "warning",
        "status": "success",
        "detail": {"scale": 2, "full_size": [6000, 8000]},
    }
    assert data["quality_status"] == "not_evaluated" and data["research"] is False


def test_failure_record_round_trip():
    failure = rec.ScanResult(
        status=Status.MODEL_ERROR,
        outcome=rec.Outcome.ORIGINAL_RETURNED,
        reason_code="page_not_found",
        message="no page component",
        used_fallback=True,
        reasons=[rec.reason("page_not_found")],
        guidance=rec.collect_guidance([rec.reason("page_not_found")]),
        research=True,
    )
    data = json_round_trip(failure.to_dict())
    assert rec.ScanResult.from_dict(data).to_dict() == failure.to_dict()
    assert data["plan"] is None and data["guidance"][0]["actions"] == ["reshoot", "set_region"]
