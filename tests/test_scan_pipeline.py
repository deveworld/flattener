"""The scanner pipeline with fake models and injected failures."""

import json

import numpy as np
import pytest
import torch
from scan_fakes import (
    FakeGate,
    FakeOrienter,
    Page,
    oracle_dewarper,
    ramp_scene,
    shrink,
    twist,
)

from flattener.contracts import FlattenerError, RenderConfig, Status
from flattener.dewarp.checkpoint import CHECKPOINT_SCHEMA, save_checkpoint
from flattener.dewarp.model import DewarpModelConfig, UVDocNet
from flattener.dewarp.predict import BatchInput, rectify, rectify_batch, rerun_grid
from flattener.scan import locate
from flattener.scan.io import read_scan_record, save_scan
from flattener.scan.pipeline import (
    BUNDLE_SCHEMA,
    PlannedScan,
    Scanner,
    ScanOptions,
    ScanOverrides,
    homography_map,
    median_drift,
    quad_homography,
    view_geometry,
)
from flattener.scan.placement import Placement, compose
from flattener.scan.records import Outcome, ScanPlan
from flattener.scan.views import crop_placement

SIZE = (240, 320)
TRUE = Page((170, 125), (150, 190), 7.0)


@pytest.fixture(autouse=True)
def small_pages(monkeypatch):
    monkeypatch.setitem(locate.THRESHOLDS, "min_page_long_side_px", 50)


def _scanner(page=TRUE, *, gate=True, orienter=None, always=None, hooks=None):
    dewarper = oracle_dewarper(page, SIZE)
    dewarper.model.always = always
    dewarper.model.hooks = hooks or {}
    fake_gate = FakeGate([page], SIZE) if gate else None
    return Scanner(dewarper, gate=fake_gate, orienter=orienter), dewarper.model


def _codes(result):
    return [item.code for item in result.reasons]


def _chain(result):
    return [
        (entry["name"], entry.get("accepted"), [f["failure"]["code"] for f in entry["failures"]])
        for entry in result.metadata["chain"]
        if "failures" in entry
    ]


def _error_px(grid, page=TRUE):
    return float(np.abs(grid - page.grid(source_size=SIZE)).max() * SIZE[1] / 2)


def test_the_default_path_rectifies_a_skewed_page():
    scanner, model = _scanner()
    image = ramp_scene(SIZE, TRUE)
    result = scanner.scan(image)
    assert result.status == Status.SUCCESS and result.reason_code is None
    assert result.outcome == Outcome.RECTIFIED_WITH_WARNINGS
    assert _codes(result) == ["orientation_uncertain"]  # no orientation model loaded
    assert result.reasons[0].detail["why"] == "no_orientation_model"
    assert [g.code for g in result.guidance] == ["orientation_uncertain"]
    assert model.calls == 2  # pass A and one crop pass; the confident gate needs no pass 0
    plan = result.plan
    assert _error_px(plan.grid) < 0.5
    assert plan.region["localization"] == "gate" and plan.region["candidate"] == "C1"
    assert [view["pass"] for view in plan.views] == ["A", 1]
    assert plan.views[1]["rotated"] and plan.views[1]["deskew_skipped"] is None
    assert plan.output.size == (190, 150) and result.image.shape == (190, 150, 3)
    assert result.valid_mask.all()
    assert set(result.timings) >= {"read", "detect", "orient", "infer", "verify", "plan"}
    assert {"render", "total"} <= set(result.timings)
    assert result.qa["coverage"] >= 0.95 and result.qa["drift"] < 0.01
    assert result.qa["orientation"]["method"] == "prior"
    assert result.candidates["orientations"][0] == {"quarter": 0, "probability": None}
    # The fake weights carry no lineage record: not shippable, results stamped research.
    assert result.shippable is False and result.research is True
    assert [model["role"] for model in result.models] == ["dewarp", "gate"]
    assert result.evidence["measurement"] == "predictions"
    assert result.metadata["renderer"]["validation"]["fold_check"] == "passed"
    assert plan.thresholds["locate"]["consensus_coverage"] == 0.95


@pytest.mark.parametrize(
    "angle, pre, post, action",
    [
        (90.0 + 7.0, True, True, None),  # OrientNet before dewarping turns the view
        (180.0 + 7.0, False, True, "flip_180"),  # the flat preview flips the nodes
        (-90.0 + 7.0, False, True, "reinfer_quarter_3"),  # the flat preview re-runs
    ],
)
def test_orientation_before_and_after_dewarping(angle, pre, post, action):
    page = Page((170, 125), (150, 190), angle)
    orienter = FakeOrienter(page, SIZE, pre=pre, post=post)
    scanner, _ = _scanner(page, orienter=orienter)
    result = scanner.scan(ramp_scene(SIZE, page))
    assert result.outcome == Outcome.RECTIFIED and _codes(result) == []
    assert result.plan.quarter == page.quarter()
    assert _error_px(result.plan.grid, page) < 0.5
    post_record = result.qa["orientation"]["post"]
    assert (post_record or {}).get("action") == action
    assert [model["role"] for model in result.models] == ["dewarp", "gate", "orient"]


def test_an_unsure_orientation_keeps_the_photo_orientation():
    page = Page((170, 125), (150, 190), 90.0 + 7.0)
    scanner, _ = _scanner(page, orienter=FakeOrienter(page, SIZE, pre=False, post=False))
    result = scanner.scan(ramp_scene(SIZE, page))
    assert result.outcome == Outcome.RECTIFIED_WITH_WARNINGS
    assert _codes(result) == ["orientation_uncertain"]
    assert result.plan.quarter == 0
    assert result.reasons[0].detail["probabilities"] == [0.4, 0.2, 0.2, 0.2]
    assert [item["quarter"] for item in result.candidates["orientations"]] == [0, 1]


def test_a_quarter_override_and_the_oracle_mode_skip_orientnet():
    page = Page((170, 125), (150, 190), 180.0 + 7.0)
    orienter = FakeOrienter(page, SIZE)
    scanner, _ = _scanner(page, orienter=orienter)
    image = ramp_scene(SIZE, page)
    result = scanner.scan(image, overrides=ScanOverrides(quarter=2))
    assert orienter.calls == [] and result.plan.quarter == 2
    assert result.plan.provenance["orientation"] == "manual"
    assert _error_px(result.plan.grid, page) < 0.5
    oracle = scanner.scan(image, ScanOptions(orient="oracle"), ScanOverrides(quarter=2))
    assert oracle.qa["orientation"]["method"] == "oracle"
    with pytest.raises(FlattenerError, match="oracle"):
        scanner.scan(image, ScanOptions(orient="oracle"))
    none = scanner.scan(image, ScanOptions(orient="none"))
    assert "orientation_uncertain" not in _codes(none) and none.plan.quarter == 0


def test_only_the_crop_pass_fails_keeps_pass_a():
    scanner, model = _scanner(hooks={1: twist})
    result = scanner.scan(ramp_scene(SIZE, TRUE))
    assert result.status == Status.SUCCESS and result.qa["rerun_discarded"]
    assert _codes(result) == ["orientation_uncertain", "rerun_discarded"]
    assert result.reasons[1].detail["failures"] == ["fold_or_degenerate"]
    assert [view["pass"] for view in result.plan.views] == ["A"]
    assert _chain(result) == [("C1", "A", ["fold_or_degenerate"])]


def test_drift_between_passes_discards_the_crop_pass():
    # A crop pass 20 % larger than the page keeps the page covered but moves the nodes.
    scanner, _ = _scanner(hooks={1: shrink(1.2)})
    result = scanner.scan(ramp_scene(SIZE, TRUE))
    assert result.qa["rerun_discarded"]
    failures = result.metadata["chain"][0]["failures"]
    assert failures[0]["failure"]["code"] == "unstable_geometry"
    assert failures[0]["failure"]["detail"]["check"] == "drift"


def test_every_candidate_failing_returns_the_original_with_the_reason():
    scanner, model = _scanner(always=twist)
    image = ramp_scene(SIZE, TRUE)
    result = scanner.scan(image)
    assert result.outcome == Outcome.ORIGINAL_RETURNED and result.used_fallback
    assert result.status == Status.INVALID_TRANSFORM
    assert result.reason_code == "fold_or_degenerate"  # the renderer's own reason
    assert result.reasons[-1].detail["candidate"] == "C1"
    assert result.image.tobytes() == image.tobytes() and result.plan is None
    guidance = [item.to_dict() for item in result.guidance]
    assert guidance[-1]["message_ko"] == "다시 찍거나 영역을 지정해 주세요"
    assert [name for name, _, _ in _chain(result)] == ["C1", "C2", "C3", "C4"]
    assert model.calls == 7  # C1 2, C2 2, pass 0, C3 2; C4 reuses pass 0
    returned = scanner.scan(image, ScanOptions(on_error="return"))
    assert returned.outcome == Outcome.FAILED and returned.image is None
    with pytest.raises(FlattenerError) as error:
        scanner.scan(image, ScanOptions(on_error="raise"))
    assert error.value.reason_code == "fold_or_degenerate"
    assert error.value.metadata["scan"]["outcome"] == "original_returned"
    single = scanner.scan(image, ScanOptions(chain=False))
    assert [name for name, _, _ in _chain(single)] == ["C1"]


def test_the_perspective_fallback_needs_a_sure_quad():
    scanner, _ = _scanner(always=twist)
    image = ramp_scene(SIZE, TRUE)
    result = scanner.scan(image, ScanOptions(fallback="perspective"))
    assert result.outcome == Outcome.PERSPECTIVE and result.used_fallback
    assert result.status == Status.INVALID_TRANSFORM and result.reason_code == "fold_or_degenerate"
    assert _codes(result)[-2:] == ["fold_or_degenerate", "curvature_not_corrected"]
    assert result.plan.kind == "perspective" and result.image.shape[:2] == result.plan.output.size
    # Pass A's corners are the page's corners (the twist keeps them) and the gate agrees.
    corners = np.array(result.plan.region["quad"])
    np.testing.assert_allclose(corners, TRUE.corners(), atol=0.5)
    # A folded outline gives no sure quad: the original comes back.
    scanner, _ = _scanner(always=shrink(0.7))
    result = scanner.scan(image, ScanOptions(fallback="perspective"))
    assert result.outcome == Outcome.ORIGINAL_RETURNED
    assert result.reason_code == "unstable_geometry"
    assert result.reasons[-1].detail["check"] == "coverage"


def test_gate_rejections_return_the_original_with_guidance():
    scanner, model = _scanner()
    blank = ramp_scene(SIZE)
    scanner.gate.pages = []
    result = scanner.scan(blank)
    assert result.outcome == Outcome.ORIGINAL_RETURNED and result.status == Status.MODEL_ERROR
    assert result.reason_code == "page_not_found" and model.calls == 0
    assert result.guidance[0].message_en == "Take the photo with the whole page in the frame"
    left, right = Page((85, 120), (120, 170)), Page((235, 120), (120, 170))
    scanner.gate.pages = [left, right]
    result = scanner.scan(ramp_scene(SIZE, left))
    assert result.reason_code == "two_page_candidate"
    assert len(result.candidates["regions"]) == 2
    assert result.guidance[0].message_ko == "페이지를 골라 주세요"


def test_without_a_gate_the_outline_is_used_unverified():
    scanner, model = _scanner(gate=False)
    result = scanner.scan(ramp_scene(SIZE, TRUE))
    assert result.status == Status.SUCCESS and "unverified_region" in _codes(result)
    assert result.plan.region["localization"] == "outline_unverified"
    assert model.calls == 3 and _error_px(result.plan.grid) < 0.5


def test_manual_overrides_take_the_same_path():
    scanner, model = _scanner()
    image = ramp_scene(SIZE, TRUE)
    region = scanner.scan(image, overrides=ScanOverrides(region=(80, 20, 262, 232)))
    assert region.plan.provenance["region"] == "manual"
    assert region.plan.region["localization"] == "manual_region"
    assert region.metadata["localization"]["box"] == [72, 11, 270, 240]  # region + 4 %
    quad = scanner.scan(image, overrides=ScanOverrides(quad=TRUE.corners().tolist()))
    assert quad.plan.region["localization"] == "manual_quad" and quad.plan.quarter == 0
    assert quad.qa["orientation"]["method"] == "quad" and _error_px(quad.plan.grid) < 0.5
    sized = scanner.scan(image, overrides=ScanOverrides(output_size=(300, 200), aspect="a4"))
    assert sized.plan.output.size == (300, 200) and sized.image.shape == (300, 200, 3)
    assert sized.plan.provenance["output_size"] == "manual"
    long_side = scanner.scan(image, ScanOptions(output=400, aspect="a4"))
    assert max(long_side.plan.output.size) == 400 and long_side.plan.output.paper == "iso"
    flat = scanner.scan(
        image, overrides=ScanOverrides(quad=TRUE.corners().tolist(), skip_dewarp=True)
    )
    assert flat.outcome == Outcome.PERSPECTIVE and flat.status == Status.SUCCESS
    assert not flat.used_fallback and flat.plan.kind == "perspective"
    assert flat.plan.provenance["dewarp"] == "manual"
    turned = scanner.scan(
        image, overrides=ScanOverrides(region=(80, 20, 262, 232), skip_dewarp=True, quarter=1)
    )
    assert turned.plan.kind == "perspective" and turned.plan.quarter == 1
    assert turned.plan.output.width > turned.plan.output.height  # read from the right edge
    with pytest.raises(FlattenerError):
        ScanOverrides(region=(0, 0, 1, 1), quad=TRUE.corners().tolist())
    with pytest.raises(FlattenerError):
        ScanOverrides(quarter=4)


def test_the_stage_cache_reruns_only_what_changed():
    scanner, model = _scanner()
    image = ramp_scene(SIZE, TRUE)
    first = scanner.scan(image)
    assert first.metadata["cache"]["misses"] == {"detect": 1, "infer": 2}
    again = scanner.scan(image, overrides=ScanOverrides(output_size=(200, 160)))
    assert model.calls == 2 and scanner.gate.calls == 1
    assert again.metadata["cache"] == {"hits": {"detect": 1, "infer": 2}, "misses": {}}
    assert again.plan.grid.tobytes() == first.plan.grid.tobytes()
    turned = scanner.scan(image, overrides=ScanOverrides(quarter=1))
    assert scanner.gate.calls == 1 and model.calls == 4
    assert turned.metadata["cache"]["hits"] == {"detect": 1}
    scanner.cache.clear()
    scanner.scan(image)
    assert scanner.gate.calls == 2 and model.calls == 6


def test_whole_image_passes_render_what_rectify_renders():
    page = Page((160, 120), (230, 180), 6.0)
    dewarper = oracle_dewarper(page, SIZE)
    scanner = Scanner(dewarper)
    image = ramp_scene(SIZE, page)
    whole = ScanOverrides(region=(0, 0, SIZE[1], SIZE[0]))
    config = RenderConfig(on_error="return")
    for passes, deskew in ((0, False), (1, False), (1, True)):
        options = ScanOptions(passes=passes, deskew=deskew, orient="none", output=SIZE)
        result = scanner.scan(image, options, whole)
        assert result.status == Status.SUCCESS and not result.qa["rerun_discarded"]
        expected = rectify(dewarper, image, render_config=config, rerun=passes, deskew=deskew)
        assert result.image.tobytes() == expected.image.tobytes()
        grid, _ = rerun_grid(dewarper, image, passes, deskew)
        assert result.plan.grid.tobytes() == grid.tobytes()


def test_plan_then_render_matches_scan_and_plans_round_trip(tmp_path):
    scanner, _ = _scanner()
    image = ramp_scene(SIZE, TRUE)
    planned = scanner.plan(image, ScanOptions(preview=64))
    assert isinstance(planned, PlannedScan)
    plan, preview = planned
    assert max(preview.shape[:2]) == 64 and plan is planned.plan
    result = scanner.render(image, planned)
    assert result.image.tobytes() == scanner.scan(image).image.tobytes()
    assert result.preview is preview and result.plan.plan_sha256 == plan.plan_sha256
    stored = ScanPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    again = scanner.render(image, stored)
    assert again.image.tobytes() == result.image.tobytes()
    other = ramp_scene(SIZE, Page((150, 110), (140, 180)))
    with pytest.raises(FlattenerError, match="another source"):
        scanner.render(other, stored)
    paths = save_scan(result, tmp_path, "page", fmt="png")
    record = read_scan_record(paths["record"])
    assert record.plan.plan_sha256 == plan.plan_sha256
    assert record.outcome == result.outcome and "save" in record.timings


def test_the_batch_rectifier_records_its_pipeline(tmp_path):
    scanner, _ = _scanner()
    image = ramp_scene(SIZE, TRUE)
    options = ScanOptions(passes=1)
    inputs = [BatchInput("a.png", "a copy.png", lambda: image)]
    report = rectify_batch(
        scanner.dewarper,
        inputs,
        tmp_path,
        rectifier=scanner.rectifier(options),
        pipeline=scanner.pipeline_record(options),
    )
    item = report["items"][0]
    assert item["status"] == "success" and item["output_size"] == list(SIZE)
    assert item["outcome"] == "rectified_with_warnings" and len(item["crops"]) == 2
    assert report["pipeline"]["name"] == "flattener.scan"
    with pytest.raises(FlattenerError, match="another pipeline"):
        rectify_batch(scanner.dewarper, inputs, tmp_path)
    failing, _ = _scanner(always=twist)
    forced = rectify_batch(
        failing.dewarper,
        inputs,
        tmp_path / "forced",
        force_render=True,
        rectifier=failing.rectifier(options, force_render=True),
        pipeline=failing.pipeline_record(options),
    )
    item = forced["items"][0]
    assert item["status"] == "invalid_transform" and item["forced"] and item["action"] == "written"


def _tiny_checkpoint(path):
    config = DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4)
    torch.manual_seed(0)
    save_checkpoint(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "step": 3,
            "track": "release",
            "model_config": config.to_dict(),
            "lineage": [{"kind": "scratch"}],
            "data_spec_sha256": "0" * 64,
            "code": {"sha256": "1" * 64},
            "model_state": UVDocNet(config).eval().state_dict(),
        },
    )


def test_bundles_load_through_the_ship_gate(tmp_path):
    _tiny_checkpoint(tmp_path / "model.pt")
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"schema": BUNDLE_SCHEMA, "dewarp": "model.pt"}))
    with pytest.raises(FlattenerError) as error:
        Scanner.load(bundle)
    assert error.value.reason_code == "not_shippable"
    scanner = Scanner.load(bundle, research=True)
    assert scanner.research and not scanner.shippable
    assert "unknown_lineage_entry" in scanner.models[0]["gate_reasons"]  # no training step
    image = np.random.default_rng(0).integers(0, 256, (120, 90, 3), dtype=np.uint8)
    result = scanner.scan(image, ScanOptions(orient="none"))
    assert result.research and result.models[0]["role"] == "dewarp"
    with pytest.raises(FlattenerError) as error:
        Scanner.load(bundle, backend="onnx")
    assert error.value.reason_code == "unsupported_backend"
    entry = {"path": "model.pt", "sha256": "0" * 64}
    bundle.write_text(json.dumps({"schema": BUNDLE_SCHEMA, "dewarp": entry}))
    with pytest.raises(FlattenerError) as error:
        Scanner.load(bundle, research=True)
    assert error.value.reason_code == "checksum_mismatch"
    bundle.write_text(json.dumps({"schema": "other"}))
    with pytest.raises(FlattenerError) as error:
        Scanner.load(bundle, research=True)
    assert error.value.reason_code == "invalid_bundle"


def test_geometry_helpers():
    placement = crop_placement((20, 30, 180, 150), 90.0 + 8.0, SIZE)
    geometry = view_geometry(placement)
    height, width = placement.view_size
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], float)
    pixels = corners @ geometry.matrix[:, :2].T + geometry.matrix[:, 2]
    values = compose(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float), placement)
    np.testing.assert_allclose(pixels, (values + 1) / 2 * [SIZE[1], SIZE[0]], atol=1e-9)
    homography = quad_homography(TRUE.corners(), SIZE)
    dense = homography_map(homography, (4, 4))
    assert dense.dtype == np.float32 and dense.shape == (4, 4, 2)
    corner = np.array([-1, -1, 1.0]) @ homography.T
    np.testing.assert_allclose(
        (corner[:2] / corner[2] + 1) / 2 * [SIZE[1], SIZE[0]], TRUE.corners()[0], atol=1e-9
    )
    grid = TRUE.grid(source_size=SIZE)
    assert median_drift(grid, grid, SIZE) == 0.0
    assert Placement.whole(SIZE).kind == "whole"


def test_invalid_options_are_refused():
    for kwargs in (
        {"passes": 3},
        {"passes": True},
        {"orient": "maybe"},
        {"output": 1},
        {"output": (10,)},
        {"border": 1.0},
        {"inset": 2.0},
        {"preview": 1},
        {"deskew": "yes"},
        {"view_extent": "loose"},
    ):
        with pytest.raises(FlattenerError) as error:
            ScanOptions(**kwargs)
        assert error.value.reason_code == "invalid_scan_option"


@pytest.mark.parametrize(
    "options",
    [ScanOptions(view_source="pyramid"), ScanOptions(view_extent="tight"), ScanOptions(passes=2)],
)
def test_view_flags_and_two_passes_keep_the_page(options):
    scanner, model = _scanner()
    result = scanner.scan(ramp_scene(SIZE, TRUE), options)
    assert result.status == Status.SUCCESS and not result.qa["rerun_discarded"]
    assert _error_px(result.plan.grid) < 1.0
    inputs = [view["input"] for view in result.plan.views]
    if options.view_source == "pyramid":
        assert all(item == "view_map:pyramid" for item in inputs)
    if options.view_extent == "tight":
        assert result.plan.views[1]["placement"]["kind"] == "outline"
    assert len(result.plan.views) == 1 + options.rerun


def test_a_non_finite_pass_is_refused_and_the_chain_goes_on():
    def corner_nan(grid):
        broken = grid.copy()
        broken[0, 0] = np.nan
        return broken

    scanner, model = _scanner(hooks={0: corner_nan})
    result = scanner.scan(ramp_scene(SIZE, TRUE))
    first = result.metadata["chain"][0]
    assert first["passes"][1]["skipped"] == "non_finite_outline"
    assert [item["failure"]["code"] for item in first["failures"]] == ["non_finite_grid"] * 2
    assert result.status == Status.SUCCESS and result.plan.region["candidate"] == "C2"
