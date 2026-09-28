"""Composite training samples, the tilt view and the tilt probe keep exact targets."""

import json
import pickle

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from flattener import FlattenerError
from flattener.dewarp import composite
from flattener.dewarp.composite import (
    BACKGROUND_SCHEMA,
    BackgroundBank,
    CompositeConfig,
    apply_homography,
    compose,
    corner_homography,
    plan_placement,
    warp_homography,
)
from flattener.dewarp.coords import pixels_to_flattener
from flattener.dewarp.data import (
    SAMPLE_SCHEMA,
    SPEC_SCHEMA,
    AugmentConfig,
    DataSpec,
    RawSample,
    ViewConfig,
    apply_geometry,
    build_dataset,
    evaluation_view,
    release_license_ok,
)
from flattener.dewarp.probe import corner_patches, grid_metrics, summarize

OUT = (96, 64)  # (height, width) of the composites here
GRID = (9, 7)
# Lighting off, so page colours can be compared with the source.
PLAIN = dict(
    illumination=0.0,
    white_balance=0.0,
    page_gain=(1.0, 1.0),
    drop_shadow_prob=0.0,
    cast_shadow_prob=0.0,
    edge_trim=0.0,
)


def _page(uv, size=(120, 90)):
    """Edge-pixel position of page point (u, v) in [0, 1]^2: a slanted, gently curved
    page well inside a (H, W) image."""
    height, width = size
    u, v = uv[..., 0], uv[..., 1]
    x = width * (0.2 + 0.55 * u + 0.05 * v) + 2.0 * np.sin(np.pi * v)
    y = height * (0.15 + 0.65 * v - 0.04 * u)
    return np.stack([x, y], axis=-1)


def _nodes(size=(120, 90), rows=GRID[0], cols=GRID[1]):
    u, v = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    return _page(np.stack([u, v], axis=-1), size)


def _ramp(size=(120, 90)):
    """RGB ramps along x (red) and y (green); bilinear sampling reproduces them."""
    height, width = size
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    image = np.stack([xx * 255 / (width - 1), yy * 255 / (height - 1), 0 * xx + 128], -1)
    return image.round().astype(np.uint8)


def _values(nodes_px, size):
    height, width = size
    return (2.0 * nodes_px / [width, height] - 1.0).astype(np.float32)


def _sample_at(image, target):
    """Colours of a (3, H, W) image at flattener values (bilinear)."""
    grid = torch.from_numpy(np.asarray(target, np.float64))[None]
    sampled = F.grid_sample(image[None].double(), grid, mode="bilinear", align_corners=False)
    return sampled[0].permute(1, 2, 0).numpy()


def test_corner_homographies_round_trip():
    rng = np.random.default_rng(0)
    source = rng.uniform(0, 100, (4, 2))
    target = source + rng.uniform(-10, 10, (4, 2))
    transform = corner_homography(source, target)
    np.testing.assert_allclose(apply_homography(transform, source), target, atol=1e-9)
    nodes = _nodes()
    back = apply_homography(np.linalg.inv(transform), apply_homography(transform, nodes))
    np.testing.assert_allclose(back, nodes, atol=1e-9)


@pytest.mark.parametrize("step", [1.0, 2.2])
def test_the_warp_samples_source_coordinates_exactly(step):
    """Warped coordinate images, read at the target nodes, give back the source nodes."""
    size = (120, 90)
    height, width = size
    nodes = _nodes(size)
    config = CompositeConfig(prob=1.0, perspective=0.12, area=(0.3, 0.3), wide_angle_prob=0.0)
    transform, _ = plan_placement(nodes, OUT, config, np.random.default_rng(3))
    coordinates = np.mgrid[0:height, 0:width].astype(np.float32)[::-1] + 0.5  # x, y edges
    warped = [
        warp_homography(Image.fromarray(channel, mode="F"), transform, OUT, step=step)
        for channel in coordinates
    ]
    target = 2.0 * apply_homography(transform, nodes) / [OUT[1], OUT[0]] - 1.0
    found = _sample_at(torch.from_numpy(np.stack(warped)), target)
    assert np.abs(found - nodes).max() < 1e-3  # pixels


def test_composites_place_the_page_where_the_target_says():
    size = (120, 90)
    nodes = _nodes(size)
    config = CompositeConfig(prob=1.0, area=(0.2, 0.5), **PLAIN)
    for seed in range(4):
        image, target, turn = compose(
            _ramp(size),
            _values(nodes, size),
            np.ones(GRID, bool),
            OUT,
            config,
            np.random.default_rng(seed),
        )
        assert image.shape == (3, *OUT) and image.dtype == torch.uint8 and turn == 0
        inner = (slice(1, -1), slice(1, -1))  # border nodes mix in the background
        expected = _sample_at(torch.from_numpy(_ramp(size)).permute(2, 0, 1), _values(nodes, size))
        np.testing.assert_allclose(_sample_at(image, target)[inner], expected[inner], atol=3)
        outline = (target + 1) / 2 * [OUT[1], OUT[0]]
        area = abs(composite.polygon_area(composite.outline(outline))) / (OUT[0] * OUT[1])
        assert 0.15 < area <= 0.5 + 1e-6  # shrunk only when the page does not fit
    again = compose(
        _ramp(size),
        _values(nodes, size),
        np.ones(GRID, bool),
        OUT,
        config,
        np.random.default_rng(0),
    )
    first = compose(
        _ramp(size),
        _values(nodes, size),
        np.ones(GRID, bool),
        OUT,
        config,
        np.random.default_rng(0),
    )
    assert torch.equal(again[0], first[0]) and np.array_equal(again[1], first[1])


def test_quarter_turns_keep_the_upright_page_as_target():
    size = (120, 90)
    nodes = _nodes(size)
    config = CompositeConfig(
        prob=1.0,
        turns=(0, 1, 0, 0),
        angle_std=0.0,
        wide_angle_prob=0.0,
        perspective=0.0,
        area=(0.3, 0.3),
        **PLAIN,
    )
    image, target, turn = compose(
        _ramp(size),
        _values(nodes, size),
        np.ones(GRID, bool),
        OUT,
        config,
        np.random.default_rng(1),
    )
    assert turn == 1
    top_left, top_right, bottom_left = target[0, 0], target[0, -1], target[-1, 0]
    assert top_left[1] > top_right[1] + 0.2  # counter-clockwise: the top edge runs upward
    assert bottom_left[0] > top_left[0] + 0.2  # and the left edge runs to the right
    colours = _sample_at(image, target)
    middle = GRID[0] // 2
    assert target[middle, -2, 1] < target[middle, 1, 1] - 0.2  # page right is up on screen
    assert colours[middle, -2, 0] > colours[middle, 1, 0] + 50  # where the red ramp is high


def test_unplaceable_samples_stay_plain_crops():
    size = (120, 90)
    nodes = _nodes(size)
    config = CompositeConfig(prob=1.0)
    valid = np.ones(GRID, bool)
    valid[0, 0] = False
    rng = np.random.default_rng(0)
    assert compose(_ramp(size), _values(nodes, size), valid, OUT, config, rng) is None
    outside = nodes + [60.0, 0.0]  # the right edge leaves the image
    assert (
        compose(_ramp(size), _values(outside, size), np.ones(GRID, bool), OUT, config, rng) is None
    )


def _write_bank(folder, licenses):
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for index, license_text in enumerate(licenses):
        Image.fromarray(_ramp((40, 40))).save(folder / f"t{index}.jpg")
        files.append({"file": f"t{index}.jpg", "license": license_text})
    manifest = {"schema": BACKGROUND_SCHEMA, "files": files}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def test_background_bank_checks_licenses_and_pickles_without_its_cache(tmp_path):
    with pytest.raises(PermissionError):
        BackgroundBank(
            _write_bank(tmp_path / "mixed", ["CC0-1.0", "CC BY-NC 4.0"]),
            license_ok=release_license_ok,
        )
    bank = BackgroundBank(
        _write_bank(tmp_path / "cc0", ["CC0-1.0"] * 2), license_ok=release_license_ok
    )
    view = bank.draw(np.random.default_rng(0), OUT)
    assert view.shape == (*OUT, 3) and view.dtype == np.float32 and len(bank._cache) == 1
    copy = pickle.loads(pickle.dumps(bank))
    assert len(copy._cache) == 0 and copy.describe() == bank.describe()
    assert bank.describe()["licenses"] == {"CC0-1.0": 2}


def test_composite_config_checks_values_and_stays_out_of_disabled_configs():
    assert "composite" not in AugmentConfig().to_dict()
    enabled = AugmentConfig(composite={"prob": 0.3, "area": [0.1, 0.9]})
    data = enabled.to_dict()
    assert data["composite"]["prob"] == 0.3 and data["composite"]["area"] == [0.1, 0.9]
    assert AugmentConfig(**json.loads(json.dumps(data))) == enabled
    for bad in ({"prob": 1.5}, {"area": [0.0, 0.5]}, {"turns": [0, 0, 0, 0]}, {"nope": 1}):
        with pytest.raises(ValueError):
            AugmentConfig(composite=bad)


def _write_source(root, size=(120, 90), count=3, split="train"):
    """Converted-layout samples: the ramp image and a centre-aligned 18x12 map."""
    root.mkdir(parents=True, exist_ok=True)
    u, v = np.meshgrid((np.arange(12) + 0.5) / 12, (np.arange(18) + 0.5) / 18)
    centres = _page(np.stack([u, v], axis=-1), size)
    entries = []
    for index in range(count):
        Image.fromarray(_ramp(size)).save(root / f"s{index}.png")
        np.save(root / f"s{index}.npy", pixels_to_flattener(centres - 0.5, size))
        entries.append(
            {
                "schema": SAMPLE_SCHEMA,
                "sample_id": f"{split}{index}",
                "split": split,
                "image": f"s{index}.png",
                "backward_map": f"s{index}.npy",
                "track": "release",
                "license_class": "clean",
            }
        )
    (root / "manifest.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))


def _dataset(tmp_path, augment, track="release"):
    _write_source(tmp_path / "src")
    source = {
        "name": "s",
        "kind": "converted",
        "root": str(tmp_path / "src"),
        "track": track,
        "license": "CC BY 4.0 test fixture",
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps({"schema": SPEC_SCHEMA, "name": "s", "track": track, "sources": [source]})
    )
    return build_dataset(
        DataSpec.load(spec_path),
        track=track,
        split="train",
        input_size=OUT,
        grid_size=GRID,
        augment=augment,
        seed=5,
        train=True,
    )


def test_training_items_mix_composites_with_unchanged_crops(tmp_path):
    bank = _write_bank(tmp_path / "bank", ["CC0-1.0"])
    place = {"prob": 0.5, "background_dir": str(bank), **PLAIN, "page_gain": [1.0, 1.0]}
    mixed = _dataset(tmp_path, AugmentConfig(composite=place))
    plain = _dataset(tmp_path / "plain", AugmentConfig())
    turns, composites = [], 0
    for position in range(12):
        item, reference = mixed[(0, position)], plain[(0, position)]
        assert "turn" in item and "turn" not in reference
        turns.append(int(item["turn"]))
        if torch.equal(item["image"], reference["image"]):  # stayed a plain crop
            assert torch.equal(item["grid2d"], reference["grid2d"])
            torch.testing.assert_close(item["color"], reference["color"])
        else:
            composites += 1
            image, target = item["image"], item["grid2d"].permute(1, 2, 0).numpy()
            source = mixed.sources[0].load(0)
            expected = _sample_at(
                torch.from_numpy(np.array(source.image)).permute(2, 0, 1), source.grid
            )
            inner = (slice(1, -1), slice(1, -1))
            np.testing.assert_allclose(_sample_at(image, target)[inner], expected[inner], atol=3)
    assert 2 <= composites <= 10 and set(turns) == {0}
    described = mixed.describe()[0]
    assert described["backgrounds"]["licenses"] == {"CC0-1.0": 1}
    assert "backgrounds" not in plain.describe()[0]
    again = pickle.loads(pickle.dumps(mixed))[(0, 3)]
    assert torch.equal(again["image"], mixed[(0, 3)]["image"])


def test_release_sets_refuse_backgrounds_without_a_release_license(tmp_path):
    bank = _write_bank(tmp_path / "bank", ["CC BY-NC-ND 4.0"])
    augment = AugmentConfig(composite={"prob": 0.5, "background_dir": str(bank)})
    with pytest.raises(FlattenerError) as error:
        _dataset(tmp_path, augment)
    assert error.value.reason_code == "track_violation"
    assert _dataset(tmp_path / "research", augment, track="research") is not None


def test_plain_crops_match_apply_geometry(tmp_path):
    dataset = _dataset(tmp_path, AugmentConfig())
    sample = dataset.sources[0].load(1)
    image, grid = apply_geometry(sample, OUT, dataset.augment, np.random.default_rng([5, 4]))
    item = dataset[(1, 4)]
    assert torch.equal(item["image"], image)


def _view_sample(size=(120, 90)):
    nodes = _nodes(size)
    return RawSample(_ramp(size), _values(nodes, size), np.ones(GRID, bool), None)


def test_the_tilt_view_keeps_exact_targets_inside_the_frame():
    sample = _view_sample()
    page_image, page = evaluation_view(sample, OUT, ViewConfig("p", margin=(0.05, 0.05)), "k")
    view = ViewConfig("t", "tilt", (0.05, 0.05), tilt=0.15)
    image, target = evaluation_view(sample, OUT, view, "k")
    assert image.shape == (3, *OUT) and np.abs(target).max() <= 1.0
    inner = (slice(1, -1), slice(1, -1))
    np.testing.assert_allclose(
        _sample_at(image, target)[inner], _sample_at(page_image, page)[inner], atol=4
    )
    assert not np.allclose(target, page, atol=0.02)  # the camera really moved
    again, _ = evaluation_view(sample, OUT, view, "k")
    assert torch.equal(again, image)
    assert ViewConfig.from_dict(view.to_dict()) == view
    for bad in ({"crop": "tilt", "tilt": 0.0}, {"crop": "tilt", "rotation": 90}, {"tilt": 0.2}):
        with pytest.raises(ValueError):
            ViewConfig("v", **bad)


def test_probe_metrics_read_foreshortening():
    quad = np.array([[10.0, 12.0], [70.0, 5.0], [80.0, 95.0], [3.0, 88.0]])
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    u, v = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 9))
    target = apply_homography(corner_homography(unit, quad), np.stack([u, v], -1))
    bilinear, homography = corner_patches(target)
    np.testing.assert_allclose(homography, target, atol=1e-9)
    exact = grid_metrics(target, target)
    assert exact["grid_error_pct"] == 0 and exact["undercorrection"] == 0
    flat = grid_metrics(bilinear, target)
    assert flat["undercorrection"] == pytest.approx(1.0) and flat["grid_error_pct"] > 0
    half = grid_metrics(target + 0.5 * (bilinear - target), target)
    assert half["undercorrection"] == pytest.approx(0.5)
    summary = summarize([exact, flat])
    assert summary["undercorrection"] == pytest.approx(0.5) and summary["samples"] == 2


@pytest.mark.training
def test_two_training_steps_with_composites_and_a_tilt_view(tmp_path):
    from flattener.dewarp.checkpoint import load_checkpoint
    from flattener.dewarp.model import DewarpModelConfig
    from flattener.dewarp.train import TrainConfig, train

    _write_source(tmp_path / "train")
    _write_source(tmp_path / "val", count=2, split="val")
    sources = [
        {
            "name": split,
            "kind": "converted",
            "root": str(tmp_path / split),
            "track": "release",
            "license": "CC BY 4.0 test fixture",
            "split": split,
        }
        for split in ("train", "val")
    ]
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"schema": SPEC_SCHEMA, "name": "t", "track": "release", "sources": sources})
    )
    config = TrainConfig(
        data_spec=str(spec),
        output_dir=str(tmp_path / "run"),
        track="release",
        model=DewarpModelConfig(input_size=(64, 48), grid_size=(5, 4), num_filter=4),
        augment=AugmentConfig(composite={"prob": 1.0}),
        batch_size=2,
        total_steps=2,
        warmup_steps=1,
        num_workers=0,
        device="cpu",
        log_every=1,
        val_every=2,
        checkpoint_every=2,
        val_views=(ViewConfig("page"), ViewConfig("tilt", "tilt", (0.02, 0.1))),
    )
    summary = train(config)
    assert summary["state"] == "completed" and summary["step"] == 2
    checkpoint = load_checkpoint(summary["checkpoint"])
    assert checkpoint["metrics"]["views"]["tilt"]["epe_px"] > 0
    assert checkpoint["train_config"]["augment"]["composite"]["prob"] == 1.0
    assert TrainConfig.from_dict(checkpoint["train_config"]) == config


def test_a_facing_page_shows_at_the_fold_and_the_target_stays_the_page():
    size = (120, 90)
    nodes, valid = _values(_nodes(size), size), np.ones(GRID, bool)
    red = np.zeros((*size, 3), np.uint8)
    red[..., 0] = 250  # the facing page is plain red
    config = CompositeConfig(
        prob=1.0,
        area=(0.1, 0.2),
        neighbor_prob=1.0,
        fold_shadow=(0.0, 0.0),
        panel_prob=0.0,
        **PLAIN,
    )
    expected = _sample_at(torch.from_numpy(_ramp(size)).permute(2, 0, 1), nodes)
    inner = (slice(1, -1), slice(1, -1))
    beside = 0
    for seed in range(6):
        image, target, _ = compose(
            _ramp(size),
            nodes,
            valid,
            OUT,
            config,
            np.random.default_rng(seed),
            neighbor=(red, nodes, valid),
        )
        np.testing.assert_allclose(_sample_at(image, target)[inner], expected[inner], atol=3)
        edge = (target[GRID[0] // 2] + 1) / 2 * [OUT[1], OUT[0]]  # the middle row, output px
        step = (edge[-1] - edge[0]) * 0.05
        for x, y in (edge[-1] + step, edge[0] - step):  # just outside the right and left edges
            if 0 <= x < OUT[1] and 0 <= y < OUT[0]:
                r, g, b = image[:, int(y), int(x)].tolist()
                beside += r >= 245 and g <= 5 and b <= 5
    assert beside >= 3
