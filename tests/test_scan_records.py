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
        exif_orientation=6,
        focal_length_35mm=26.0,
    )
    fields = dict(
        source=source,
        output=plan_output(make_grid(), source.size),
        grid=grid,
        region={"box": [100, 50, 3900, 2950], "source": "gate", "dominance": np.float32(0.97)},
        angle_deg=-3.25,
        quarter=1,
        orientation_probs=(0.01, 0.97, 0.015, 0.005),
        views=({"pass": "initial", "box": (100, 50, 3900, 2950), "angle_deg": 86.75},),
        inset=(0.0, 0.0),
        models=({"id": "dewarp", "path": "dewarp.pt"},),
        thresholds={"version": "test", "coverage": 0.95, "drift": 0.04},
        provenance={"region": "auto", "orientation": "manual"},
    )
    fields.update(changes)
    return rec.ScanPlan(**fields)


def test_plans_round_trip_exactly_and_refuse_changes():
    plan = make_plan()
    data = json_round_trip(plan.to_dict())
    again = rec.ScanPlan.from_dict(data)
    assert again == plan
    assert again.to_dict() == plan.to_dict()
    assert np.array_equal(again.grid.astype(np.float32), make_grid())
    assert again.grid.dtype == np.float64 and not again.grid.flags.writeable
    assert again.region["dominance"] == float(np.float32(0.97))
    assert again.views[0]["box"] == [100, 50, 3900, 2950]
    assert (plan.nodes, plan.fill_rgb) == ("flattener", (255, 255, 255))
    uvdoc = make_plan(nodes="uvdoc", fill_rgb=[10, 20, 30])
    again = rec.ScanPlan.from_dict(json_round_trip(uvdoc.to_dict()))
    assert again == uvdoc and (again.nodes, again.fill_rgb) == ("uvdoc", (10, 20, 30))
    # A perspective plan keeps its homography.
    homography = np.array([[0.5, 0.02, 0.01], [-0.01, 0.7, 0.03], [0.05, -0.04, 1.0]])
    perspective = make_plan(kind="perspective", grid=None, homography=homography)
    again = rec.ScanPlan.from_dict(json_round_trip(perspective.to_dict()))
    assert again == perspective and np.array_equal(again.homography, homography)
    with pytest.raises(FlattenerError):
        rec.ScanPlan.from_dict({**perspective.to_dict(), "schema": "other"})
    assert make_plan(quarter=2) != plan
    assert make_plan(quantize="truncate") != plan
    data = json_round_trip(plan.to_dict())
    data["grid"][0][0][0] += 0.01
    assert rec.ScanPlan.from_dict(data) != plan
    extra = {**make_plan().to_dict(), "note": "extra"}
    with pytest.raises(FlattenerError) as caught:
        rec.ScanPlan.from_dict(extra)
    assert caught.value.reason_code == "invalid_plan"
    # Optional sharpening round-trips.
    assert set(make_plan().to_dict()) == rec.PLAN_KEYS - {"sharpen"}
    sharpened = make_plan(sharpen=1.5)
    assert set(sharpened.to_dict()) == rec.PLAN_KEYS
    assert rec.ScanPlan.from_dict(json_round_trip(sharpened.to_dict())).sharpen == 1.5
    broken = {key: value for key, value in perspective.to_dict().items() if key != "output"}
    with pytest.raises(FlattenerError) as caught:
        rec.ScanPlan.from_dict(broken)
    assert caught.value.reason_code == "invalid_plan"
    for changes in ({"nodes": "center"}, {"fill_rgb": (0, 0)}, {"sharpen": -1.0}):
        with pytest.raises(FlattenerError) as caught:
            make_plan(**changes)
        assert (caught.value.status, caught.value.reason_code) == (
            Status.INPUT_ERROR,
            "invalid_plan",
        )


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
        models=[{"id": "dewarp", "path": "dewarp.pt"}],
        input="photo.jpg",
    )


def test_results_round_trip_through_json_with_their_outcome():
    result = make_result(make_plan())
    data = result.to_dict()
    text = json.dumps(data, ensure_ascii=False, allow_nan=False)
    again = rec.ScanResult.from_dict(json.loads(text))
    assert again.to_dict() == data
    assert again.outcome == rec.Outcome.RECTIFIED_WITH_WARNINGS
    assert again.plan == result.plan
    assert again.image is None and again.valid_mask is None
    assert data["qa"]["orientation_margin"] is None
    assert data["metadata"]["validation"]["checked_cells"] == 1320
    assert data["guidance"][0]["message_en"] == (
        "The top edge is cut off. Step back a little and take the photo again"
    )
    # Records written while guidance also carried message_ko still load.
    older = json.loads(text)
    older["guidance"][0]["message_ko"] = "localized text"
    assert rec.ScanResult.from_dict(older).to_dict() == data
    assert data["reasons"][0] == {
        "code": "source_downscaled",
        "severity": "warning",
        "status": "success",
        "detail": {"scale": 2, "full_size": [6000, 8000]},
    }
    assert data["quality_status"] == "not_evaluated"
    # The outcome follows the path taken and the worst reason.
    info = rec.reason("page_fills_frame")
    warning = rec.reason("possible_occlusion", solidity=0.8)
    assert rec.decide_outcome("dewarp", [info]) == rec.Outcome.RECTIFIED
    assert rec.decide_outcome("dewarp", [info, warning]) == rec.Outcome.RECTIFIED_WITH_WARNINGS
    assert rec.decide_outcome("perspective", []) == rec.Outcome.PERSPECTIVE
    assert rec.decide_outcome("original", [warning]) == rec.Outcome.ORIGINAL_RETURNED
    assert rec.decide_outcome(None) == rec.Outcome.FAILED
    with pytest.raises(ValueError):
        rec.decide_outcome("crop")
    assert (rec.EXIT_SCANNED, rec.EXIT_FALLBACK, rec.EXIT_USAGE) == (0, 1, 2)
    # A failure record round-trips as well.
    failure = rec.ScanResult(
        status=Status.MODEL_ERROR,
        outcome=rec.Outcome.ORIGINAL_RETURNED,
        reason_code="page_not_found",
        message="no page component",
        used_fallback=True,
        reasons=[rec.reason("page_not_found")],
        guidance=rec.collect_guidance([rec.reason("page_not_found")]),
    )
    data = json_round_trip(failure.to_dict())
    assert rec.ScanResult.from_dict(data).to_dict() == failure.to_dict()
    assert data["plan"] is None and data["guidance"][0]["actions"] == ["reshoot", "set_region"]
