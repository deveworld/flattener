"""Forward-only augmentation tests: no models, optimizers or backward calls."""

import math

import pytest
import torch

from flattener.contracts import FlattenerError
from flattener.learning import augmentation as aug


def _centers(height, width):
    y, x = torch.meshgrid((torch.arange(height) + 0.5) * 2 / height - 1,
                          (torch.arange(width) + 0.5) * 2 / width - 1, indexing="ij")
    return torch.stack((x, y), -1)


def _batch(count=4, photo=(80, 60), preview=(40, 30)):
    coordinates = _centers(*preview)
    image = torch.stack(((coordinates[..., 0] + 1) / 2,
                         (coordinates[..., 1] + 1) / 2,
                         torch.full(preview, 0.6))).repeat(count, 1, 1, 1)
    target = (_centers(9, 7) * 0.4).repeat(count, 1, 1, 1)
    valid = torch.ones((count, 9, 7), dtype=torch.bool)
    valid[:, 0, 0] = False
    target[:, 0, 0] = torch.nan
    source = torch.zeros((count, *photo), dtype=torch.bool)
    source[:, photo[0] // 3:2 * photo[0] // 3,
           photo[1] // 3:2 * photo[1] // 3] = True
    return {"image": image, "target": target, "valid": valid, "source_mask": source,
            "extra": "untouched"}


def _bytes(batch):
    return {name: value.numpy().tobytes() for name, value in batch.items()
            if isinstance(value, torch.Tensor)}


def test_disabled_is_exact_identity_without_rng_or_admission(monkeypatch):
    batch = object()
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: pytest.fail("RNG draw"))
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: pytest.fail("RNG draw"))
    assert aug.augment_training_batch(batch, photo_size=None) is batch


@pytest.mark.parametrize(
    "name,value",
    [
        ("rotation_probability", 1.1),
        ("background_probability", float("nan")),
        ("rotation_probability", True),
    ],
)
def test_invalid_probability_rejected_before_rng(name, value, monkeypatch):
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: pytest.fail("RNG draw"))
    with pytest.raises(FlattenerError, match="finite in"):
        aug.augment_training_batch({}, photo_size=(80, 60), **{name: value})


def test_background_preserves_protected_page_labels_and_input_exactly():
    batch = _batch()
    before = _bytes(batch)
    result = aug.augment_training_batch(batch, photo_size=(80, 60), background_probability=1)
    protected = aug._protected_preview(batch["source_mask"], (40, 30)).expand(-1, 3, -1, -1)
    assert torch.equal(result["image"][protected], batch["image"][protected])
    assert not torch.equal(result["image"][~protected], batch["image"][~protected])
    for key in ("target", "valid", "source_mask"):
        assert result[key] is batch[key]
    assert _bytes(batch) == before
    assert result["extra"] == "untouched"
    assert result["image"].data_ptr() != batch["image"].data_ptr()
    assert bool(((result["image"] >= 0) & (result["image"] <= 1)).all())


def test_background_gate_is_per_image_and_unselected_image_exact(monkeypatch):
    batch = _batch()
    original = torch.rand

    def draws(shape, **kwargs):
        if shape == (4,):
            return torch.tensor([0.1, 0.9, 0.2, 0.8])
        return original(shape, **kwargs)

    monkeypatch.setattr(torch, "rand", draws)
    result = aug.augment_training_batch(batch, photo_size=(80, 60), background_probability=0.5)
    assert torch.equal(result["image"][[1, 3]], batch["image"][[1, 3]])
    assert not torch.equal(result["image"][[0, 2]], batch["image"][[0, 2]])


@pytest.mark.parametrize("preview", [(40, 30), (48, 48)])
def test_thin_page_mask_remains_protected_after_any_preview_resize(preview):
    source = torch.zeros((1, 80, 60), dtype=torch.bool)
    source[:, 20:61, 31] = True
    protected = aug._protected_preview(source, preview)[0, 0]
    # Independently map each source foreground pixel center into preview cells.
    rows, columns = source[0].nonzero(as_tuple=True)
    rows = ((rows + 0.5) * preview[0] / 80).long().clamp_max(preview[0] - 1)
    columns = ((columns + 0.5) * preview[1] / 60).long().clamp_max(preview[1] - 1)
    assert bool(protected[rows, columns].all())


@pytest.mark.parametrize("broken", ["missing", "empty"])
def test_background_requires_safe_source_mask_even_on_validated_path(broken):
    batch = _batch()
    if broken == "missing":
        batch.pop("source_mask")
    else:
        batch["source_mask"].zero_()
    with pytest.raises(FlattenerError, match="mask"):
        aug.augment_training_batch(batch, photo_size=(80, 60), background_probability=1,
                                   _validated=True)


def test_empty_source_mask_can_defer_failure_without_device_scalar_read():
    batch = _batch()
    batch["source_mask"].zero_()
    checks = []
    aug.augment_training_batch(batch, photo_size=(80, 60), background_probability=1,
                               _validated=True, _checks=checks)
    assert checks and not bool(torch.stack(checks).all())


def test_per_image_turns_transform_values_not_flat_domain_and_preserve_valid(monkeypatch):
    batch = _batch()
    before = _bytes(batch)
    monkeypatch.setattr(aug, "_sample_turns", lambda n, p: torch.arange(4))
    result = aug.augment_training_batch(batch, photo_size=(80, 60), rotation_probability=1)
    target = batch["target"]
    x, y = target.unbind(-1)
    expected = torch.stack((target[0], torch.stack((y[1], -0.5625 * x[1]), -1),
                            -target[2], torch.stack((-y[3], 0.5625 * x[3]), -1)))
    torch.testing.assert_close(result["target"], expected, rtol=0, atol=0, equal_nan=True)
    assert result["valid"] is batch["valid"]
    assert _bytes(batch) == before
    assert torch.equal(result["image"][0], batch["image"][0])
    assert torch.equal(result["image"][2], batch["image"][2].flip((-2, -1)))
    assert torch.equal(result["source_mask"][0], batch["source_mask"][0])
    assert torch.equal(result["source_mask"][2], batch["source_mask"][2].flip((-2, -1)))
    assert torch.isnan(result["target"][~result["valid"]]).all()


@pytest.mark.parametrize("photo,preview,filtered", [
    ((384, 288), (192, 144), (144, 108)),
    ((288, 384), (144, 192), (108, 144)),
])
def test_antialias_prefilter_uses_physical_photo_aspect(photo, preview, filtered, monkeypatch):
    batch = _batch(count=1, photo=photo, preview=preview)
    original, calls = aug.F.interpolate, []

    def spy(value, **kwargs):
        if kwargs.get("antialias"):
            calls.append(kwargs)
        return original(value, **kwargs)

    monkeypatch.setattr(aug.F, "interpolate", spy)
    aug._rotate_fit(batch, torch.tensor([1]), photo, torch.zeros_like(batch["image"]), [])
    assert len(calls) == 1
    assert calls[0]["size"] == filtered
    assert calls[0]["align_corners"] is False and calls[0]["antialias"] is True


@pytest.mark.parametrize("photo,preview", [((80, 60), (48, 48)), ((60, 80), (30, 40))])
def test_inverse_sampling_agrees_with_independent_coordinate_color_oracle(photo, preview):
    batch = _batch(count=1, photo=photo, preview=preview)
    h, w = photo
    scale = min(h / w, w / h)
    b, c = scale * h / w, scale * w / h
    q = _centers(*preview)
    for turn in (1, 3):
        result = aug._rotate_fit(batch, torch.tensor([turn]), photo,
                                 torch.zeros_like(batch["image"]), [])
        x = (-q[..., 1] / c) if turn == 1 else q[..., 1] / c
        y = (q[..., 0] / b) if turn == 1 else -q[..., 0] / b
        inside = (x.abs() < 0.7) & (y.abs() < 0.7)
        expected = torch.stack(((x + 1) / 2, (y + 1) / 2, torch.full_like(x, 0.6)))
        # Antialiased linear-ramp filtering can shift samples by <0.2 source
        # pixels due to discrete weights; this does not move physical geometry.
        torch.testing.assert_close(result["image"][0, :, inside], expected[:, inside],
                                   rtol=0, atol=0.004)


@pytest.mark.parametrize("turn", [1, 3])
def test_square_photo_rotation_matches_rot90_and_page_mask(turn):
    batch = _batch(count=1, photo=(64, 64), preview=(32, 32))
    result = aug._rotate_fit(batch, torch.tensor([turn]), (64, 64),
                             torch.zeros_like(batch["image"]), [])
    torch.testing.assert_close(result["image"], torch.rot90(batch["image"], turn, (-2, -1)),
                               rtol=0, atol=1e-6)
    assert torch.equal(result["source_mask"], torch.rot90(batch["source_mask"], turn, (-2, -1)))


def test_fit_preserves_boundary_gt_validity_without_coordinate_clipping():
    batch = _batch(count=4)
    target = torch.tensor([[[-1., -1.], [1., -1.]], [[-1., 1.], [1., 1.]]])
    batch["target"] = target.repeat(4, 1, 1, 1)
    batch["valid"] = torch.ones((4, 2, 2), dtype=torch.bool)
    checks = []
    result = aug._rotate_fit(batch, torch.arange(4), (80, 60),
                             torch.zeros_like(batch["image"]), checks)
    assert result["valid"] is batch["valid"]
    assert torch.stack(checks).all()
    assert (result["target"].abs() <= 1).all()
    assert result["target"][1, ..., 1].abs().max() == 0.5625


def test_rng_resume_reproduces_all_outputs_and_next_draw_exactly():
    batch = _batch()
    state = torch.get_rng_state()
    first = aug.augment_training_batch(batch, photo_size=(80, 60), rotation_probability=0.6,
                                      background_probability=0.8)
    next_draw = torch.rand(3)
    torch.set_rng_state(state)
    second = aug.augment_training_batch(batch, photo_size=(80, 60), rotation_probability=0.6,
                                       background_probability=0.8)
    assert _bytes(first) == _bytes(second)
    assert torch.equal(next_draw, torch.rand(3))


def test_validated_deferred_path_never_reads_a_tensor_scalar(monkeypatch):
    batch = _batch()
    monkeypatch.setattr(torch.Tensor, "__bool__", lambda self: pytest.fail("tensor bool sync"))
    monkeypatch.setattr(torch.Tensor, "item", lambda self: pytest.fail("tensor scalar sync"))
    checks = []
    result = aug.augment_training_batch(batch, photo_size=(80, 60), rotation_probability=1,
                                       background_probability=1, _validated=True, _checks=checks)
    assert len(checks) == 2 and result["valid"] is batch["valid"]


def test_rotation_can_omit_source_mask_but_background_cannot():
    batch = _batch()
    batch.pop("source_mask")
    result = aug.augment_training_batch(batch, photo_size=(80, 60), rotation_probability=1)
    assert "source_mask" not in result and result["valid"] is batch["valid"]


def test_procedural_texture_is_bounded_varied_and_asset_free():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)  # the bounds are statistical; earlier tests move the global RNG
        texture = aug._texture(torch.zeros((8, 3, 48, 36)))
    assert torch.isfinite(texture).all() and texture.min() >= 0 and texture.max() <= 1
    assert texture.flatten(1).std(1).min() > 0.015
    assert texture.mean((2, 3)).std(0).min() > 0.025
    assert math.isfinite(float(texture.mean()))


def test_xpu_augmentation_admission_without_accelerator_initialization(monkeypatch):
    from torch._subclasses import fake_tensor

    monkeypatch.setattr(fake_tensor, "init_gpu_context", lambda device: None)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "_lazy_init", lambda: pytest.fail("real XPU initialization"))
    mode = fake_tensor.FakeTensorMode()

    def tensor(shape, dtype=torch.float32, device="xpu:0"):
        return fake_tensor.FakeTensor(mode, torch.empty(shape, dtype=dtype, device="meta"),
                                      torch.device(device))

    batch = {"image": tensor((2, 3, 40, 30)), "target": tensor((2, 9, 7, 2)),
             "valid": tensor((2, 9, 7), dtype=torch.bool)}
    aug._admit(batch, (80, 60), False, True, [])
    batch["target"] = tensor((2, 9, 7, 2), device="xpu:1")
    with pytest.raises(FlattenerError, match="matching"):
        aug._admit(batch, (80, 60), False, True, [])


def _semantic_corners(degrees, photo=(80, 60)):
    """Independent physical-pixel rectangle, then normalize into the source."""
    corners = torch.tensor([[-8., -5.], [8., -5.], [8., 5.], [-8., 5.]])
    angle = math.radians(degrees)
    matrix = torch.tensor([[math.cos(angle), -math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])
    return (corners @ matrix.T) / torch.tensor([photo[1] / 2, photo[0] / 2])


def _with_semantic(batch, degrees=0, photo=(80, 60)):
    corners = _semantic_corners(degrees, photo).repeat(len(batch["image"]), 1, 1)
    direction, eligible = aug._direction_labels(corners, photo)
    return {**batch, "corner_target": corners,
            "corner_valid": torch.ones(corners.shape[:-1], dtype=torch.bool),
            "direction_target": direction, "direction_valid": eligible}


@pytest.mark.parametrize("degrees,expected", [(0, 0), (50, 1), (180, 2), (-50, 3)])
def test_semantic_direction_uses_clockwise_physical_horizontal_axis(degrees, expected):
    corners = _semantic_corners(degrees)
    direction, eligible = aug._direction_labels(corners[None], (80, 60))
    assert direction.tolist() == [expected] and eligible.tolist() == [True]


def test_semantic_direction_boundary_is_masked_not_arbitrarily_assigned():
    direction, eligible = aug._direction_labels(_semantic_corners(45)[None], (80, 60))
    assert not eligible[0] and direction[0] == 0


def test_semantic_direction_degenerate_and_nonfinite_are_masked():
    corners = torch.zeros((2, 4, 2))
    corners[1, 0, 0] = torch.nan
    _, eligible = aug._direction_labels(corners, (80, 60))
    assert not eligible.any()


def test_fit_rotation_preserves_semantic_order_and_clockwise_direction(monkeypatch):
    batch = _with_semantic(_batch())
    before = _bytes(batch)
    turns = torch.arange(4)
    monkeypatch.setattr(aug, "_sample_turns", lambda n, p: turns)
    result = aug.augment_training_batch(batch, photo_size=(80, 60), rotation_probability=1)
    corner = batch["corner_target"]
    x, y = corner.unbind(-1)
    expected = torch.stack((corner[0], torch.stack((y[1], -0.5625 * x[1]), -1),
                            -corner[2], torch.stack((-y[3], 0.5625 * x[3]), -1)))
    torch.testing.assert_close(result["corner_target"], expected, rtol=0, atol=0)
    assert result["corner_valid"] is batch["corner_valid"]
    assert result["direction_target"].tolist() == [0, 3, 2, 1]
    assert result["direction_valid"].all()
    # Semantic TL is still element 0 even when it lies at screen bottom-right.
    assert result["corner_target"][2, 0, 0] > 0 and result["corner_target"][2, 0, 1] > 0
    assert _bytes(batch) == before


def test_cropped_corners_keep_nan_mask_and_conformal_direction_fallback():
    batch = _with_semantic(_batch())
    batch["corner_target"][:, 0] = torch.nan
    batch["corner_valid"][:, 0] = False
    turns = torch.arange(4)
    forward = aug._fit_matrices(turns, (80, 60), torch.device("cpu"))[0]
    result = aug.transform_semantic_page_labels(batch, forward, (80, 60), turns=turns)
    assert torch.isnan(result["corner_target"][:, 0]).all()
    assert result["corner_valid"] is batch["corner_valid"]
    assert result["direction_target"].tolist() == [0, 3, 2, 1]
    assert result["direction_valid"].all()
    no_fallback = aug.transform_semantic_page_labels(batch, forward, (80, 60))
    assert not no_fallback["direction_valid"].any()


def test_legacy_anisotropic_rotation_rederives_direction_instead_of_q_minus_k():
    batch = _with_semantic(_batch(count=1), degrees=50)
    assert batch["direction_target"].tolist() == [1]
    legacy = torch.tensor([[[0., 1.], [-1., 0.]]])
    result = aug.transform_semantic_page_labels(batch, legacy, (80, 60))
    assert result["direction_target"].tolist() == [3]  # q-k would incorrectly yield 0.
    assert result["direction_valid"].tolist() == [True]
    with pytest.raises(FlattenerError, match="exact physical fit"):
        aug.transform_semantic_page_labels(batch, legacy, (80, 60), turns=torch.tensor([1]))


def test_background_only_does_not_change_semantic_labels_or_existing_rng_flow():
    plain = _batch()
    semantic = _with_semantic(plain)
    state = torch.get_rng_state()
    original = aug.augment_training_batch(plain, photo_size=(80, 60), background_probability=1)
    after = torch.get_rng_state()
    torch.set_rng_state(state)
    result = aug.augment_training_batch(semantic, photo_size=(80, 60), background_probability=1)
    assert torch.equal(original["image"], result["image"])
    assert torch.equal(torch.get_rng_state(), after)
    for key in aug._SEMANTIC_FIELDS:
        assert result[key] is semantic[key]


@pytest.mark.parametrize("broken", ["partial", "nan", "direction"])
def test_semantic_transform_rejects_inconsistent_labels(broken):
    batch = _with_semantic(_batch(count=1))
    if broken == "partial":
        batch.pop("direction_valid")
    elif broken == "nan":
        batch["corner_target"][:, 0] = torch.nan
    else:
        batch["direction_target"][:] = 4
    with pytest.raises(FlattenerError):
        aug.transform_semantic_page_labels(batch, torch.eye(2)[None], (80, 60))


def test_semantic_transform_rejects_nonfinite_matrix_even_when_all_corners_are_masked():
    batch = _with_semantic(_batch(count=1))
    batch["corner_target"].fill_(torch.nan)
    batch["corner_valid"].zero_()
    batch["direction_valid"].zero_()
    forward = torch.full((1, 2, 2), torch.nan)
    with pytest.raises(FlattenerError, match="matrix must be finite"):
        aug.transform_semantic_page_labels(batch, forward, (80, 60))
    checks = []
    result = aug.transform_semantic_page_labels(batch, forward, (80, 60), _checks=checks)
    assert not bool(torch.stack(checks).all())
    assert torch.isnan(result["corner_target"]).all()
    assert not result["corner_valid"].any() and not result["direction_valid"].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA augmentation parity requires a GPU")
def test_cuda_parity_protected_page_and_disabled_rng_identity():
    """Half-size network preview with the original photo aspect."""
    preview = (192, 144)
    device = torch.device("cuda", torch.cuda.current_device())
    with torch.random.fork_rng(devices=[device.index]):
        cpu = _with_semantic(_batch(count=8, photo=(384, 288), preview=preview),
                             photo=(384, 288))
        gpu = {name: value.to(device) if isinstance(value, torch.Tensor) else value
               for name, value in cpu.items()}
        original = _bytes(cpu)
        # The disabled call must not consume either host or device RNG and must
        # return the exact same dictionary/tensors, not a numerical copy.
        cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        assert aug.augment_training_batch(gpu, photo_size=(384, 288)) is gpu
        assert torch.equal(torch.get_rng_state(), cpu_state)
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_state)

        torch.manual_seed(20260926)
        draw_state = torch.get_rng_state()
        turns = aug._sample_turns(len(cpu["image"]), 0.75)
        assert set(turns.tolist()) == {0, 1, 2, 3}
        torch.set_rng_state(draw_state)
        expected = aug.augment_training_batch(cpu, photo_size=(384, 288),
                                              rotation_probability=0.75,
                                              background_probability=0.8)
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(draw_state)
        checks = []
        actual = aug.augment_training_batch(gpu, photo_size=(384, 288),
                                            rotation_probability=0.75,
                                            background_probability=0.8,
                                            _validated=True, _checks=checks)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert checks and bool(torch.stack(checks).all())
        # GPU interpolation/sine kernels may round differently; coordinate
        # values, masks and unchanged samples have substantially tighter gates.
        torch.testing.assert_close(actual["image"].cpu(), expected["image"],
                                   rtol=0, atol=3e-5)
        torch.testing.assert_close(actual["target"].cpu(), expected["target"],
                                   rtol=0, atol=1e-7, equal_nan=True)
        assert torch.equal(actual["valid"].cpu(), expected["valid"])
        assert torch.equal(actual["source_mask"].cpu(), expected["source_mask"])
        assert actual["valid"] is gpu["valid"]
        torch.testing.assert_close(actual["corner_target"].cpu(), expected["corner_target"],
                                   rtol=0, atol=1e-7, equal_nan=True)
        for name in ("corner_valid", "direction_target", "direction_valid"):
            assert torch.equal(actual[name].cpu(), expected[name])

        protected = aug._protected_preview(gpu["source_mask"], preview).expand(-1, 3, -1, -1)
        background = aug.augment_training_batch(gpu, photo_size=(384, 288),
                                                background_probability=1)
        assert torch.equal(background["image"][protected], gpu["image"][protected])
        assert not torch.equal(background["image"][~protected], gpu["image"][~protected])
        for name in ("target", "valid", "source_mask"):
            assert background[name] is gpu[name]
        assert _bytes(cpu) == original
        assert _bytes({name: value.cpu() if isinstance(value, torch.Tensor) else value
                       for name, value in gpu.items()}) == original
