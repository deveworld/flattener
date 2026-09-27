"""Scan reason codes, guidance, outcomes and JSON round-trips of ScanPlan and ScanResult."""

import json
import math

import numpy as np
import pytest

from flattener import FlattenerError, Status
from flattener.scan import records as rec
from flattener.scan.output_policy import OutputPlan, plan_output

SECTION_1_5 = {
    "page_not_found",
    "page_too_small",
    "page_fills_frame",
    "ambiguous_page_region",
    "two_page_candidate",
    "neighbor_page_partial",
    "page_cut_off",
    "possible_occlusion",
    "orientation_uncertain",
    "unstable_geometry",
    "invalid_transform",
    "curvature_not_corrected",
    "unverified_region",
    "source_downscaled",
    "low_resolution",
    "rerun_discarded",  # L1: a candidate keeps pass A when only its crop passes fail
    "candidate_fallback",  # a later candidate passed after C1 failed
}
KOREAN = {  # design section 1.5, verbatim
    "page_not_found": "페이지 전체가 화면에 들어오게 찍어 주세요",
    "page_too_small": "더 가까이에서 찍어 주세요",
    "ambiguous_page_region": "페이지를 골라 주세요",
    "two_page_candidate": "페이지를 골라 주세요",
    "possible_occlusion": "손가락이나 물건이 페이지를 가려요",
    "orientation_uncertain": "방향을 확인해 주세요",
    "unstable_geometry": "다시 찍거나 영역을 지정해 주세요",
    "invalid_transform": "다시 찍거나 영역을 지정해 주세요",
}


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


def test_registry_covers_section_1_5():
    assert set(rec.REASONS) == SECTION_1_5
    for code, spec in rec.REASONS.items():
        assert spec.code == code
        assert (spec.message_ko is None) == (spec.message_en is None)
        assert bool(spec.actions) == (spec.message_ko is not None)
        if spec.severity == rec.Severity.FAILURE:
            assert spec.status != Status.SUCCESS
        elif code != "curvature_not_corrected":
            assert spec.status == Status.SUCCESS
    for code, message in KOREAN.items():
        assert rec.REASONS[code].message_ko == message
    silent = {
        "page_fills_frame",
        "neighbor_page_partial",
        "source_downscaled",
        "low_resolution",
        "rerun_discarded",
    }
    assert {code for code, spec in rec.REASONS.items() if spec.message_ko is None} >= silent


@pytest.mark.parametrize(
    ("detail", "ko", "en"),
    [
        (
            {"side": "top"},
            "위쪽이 잘렸어요. 조금 물러나 찍어 주세요",
            "The top edge is cut off. Step back a little and take the photo again",
        ),
        (
            {"sides": ["right", "left"]},
            "왼쪽과 오른쪽이 잘렸어요. 조금 물러나 찍어 주세요",
            "The left and right edges are cut off. Step back a little and take the photo again",
        ),
        (
            {"sides": ["bottom", "top", "right"]},
            "위쪽, 아래쪽, 오른쪽이 잘렸어요. 조금 물러나 찍어 주세요",
            "The top, bottom and right edges are cut off. Step back a little and take the photo"
            " again",
        ),
        (
            {"side": "middle"},
            "페이지 가장자리가 잘렸어요. 조금 물러나 찍어 주세요",
            "An edge of the page is cut off. Step back a little and take the photo again",
        ),
    ],
)
def test_cut_off_guidance_names_the_sides(detail, ko, en):
    guidance = rec.guidance_for(rec.reason("page_cut_off", **detail))
    assert (guidance.message_ko, guidance.message_en) == (ko, en)
    assert guidance.actions == (rec.Action.RESHOOT,)


def test_unregistered_codes_keep_their_status():
    fold = rec.reason("fold_detected", status=Status.INVALID_TRANSFORM, cells=3)
    assert fold.severity == rec.Severity.FAILURE and fold.detail == {"cells": 3}
    assert rec.guidance_for(fold).message_ko == KOREAN["invalid_transform"]
    assert rec.guidance_for(fold).code == "fold_detected"
    note = rec.reason("model_note", status=Status.SUCCESS)
    assert note.severity == rec.Severity.WARNING and rec.guidance_for(note) is None
    with pytest.raises(ValueError):
        rec.reason("unknown_code")


def test_guidance_is_collected_once_per_message():
    reasons = [
        rec.reason("unstable_geometry", drift=0.07),
        rec.reason("invalid_transform"),
        rec.reason("source_downscaled", scale=2),
        rec.reason("orientation_uncertain", margin=0.1),
    ]
    guidance = rec.collect_guidance(reasons)
    assert [item.code for item in guidance] == ["unstable_geometry", "orientation_uncertain"]
    assert guidance[0].actions == (rec.Action.RESHOOT, rec.Action.SET_REGION)


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
        {"quarter": 4},
        {"quarter": True},
        {"grid": np.full((3, 3, 2), np.nan)},
        {"grid": np.zeros((1, 3, 2))},
        {"grid": None},
        {"kind": "perspective"},
        {"kind": "warp"},
        {"inset": (1.0, 0.0)},
        {"angle_deg": math.inf},
        {"orientation_probs": (0.5, 0.5)},
        {"provenance": {"region": "guessed"}},
        {"region": {"box": object()}},
        {"quantize": "floor"},
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


def test_source_info_focal_length():
    info = rec.SourceInfo(size=(3000, 4000), focal_length_35mm=26.0)
    assert info.full_size == (3000, 4000) and not info.downscaled
    assert info.focal_px() == pytest.approx(26 * 5000 / math.hypot(36, 24))
    assert rec.SourceInfo(size=(10, 10)).focal_px() is None
    with pytest.raises(FlattenerError):
        rec.SourceInfo(size=(0, 10))


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


def test_record_with_mismatched_plan_hash_is_refused():
    data = make_result(make_plan()).to_dict()
    data["plan_sha256"] = "0" * 64
    with pytest.raises(FlattenerError) as caught:
        rec.ScanResult.from_dict(data)
    assert caught.value.reason_code == "plan_sha256_mismatch"
    with pytest.raises(FlattenerError) as caught:
        rec.ScanResult.from_dict({**data, "schema": "flattener-scan-record-0"})
    assert caught.value.reason_code == "invalid_scan_record"
    bad_outcome = {**make_result(make_plan()).to_dict(), "outcome": "scanned"}
    with pytest.raises(FlattenerError) as caught:
        rec.ScanResult.from_dict(bad_outcome)
    assert caught.value.reason_code == "invalid_scan_record"


def test_output_plan_is_part_of_the_plan_hash():
    plan = make_plan()
    output = OutputPlan.from_dict({**plan.output.to_dict(), "height": plan.output.height + 1})
    assert make_plan(output=output).plan_sha256 != plan.plan_sha256
