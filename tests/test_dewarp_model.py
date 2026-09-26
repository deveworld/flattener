"""UVDocnet re-implementation: configuration, shapes and upstream forward parity."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from flattener.dewarp.model import (
    UVDOC_CONFIG,
    UVDOC_WEIGHTS_SHA256,
    DewarpModelConfig,
    UVDocNet,
    count_parameters,
    load_uvdoc_weights,
    natural_grid_size,
    uvdoc_model,
)

TINY = DewarpModelConfig(input_size=(64, 48), grid_size=(4, 3), num_filter=4)


def _uvdoc_repo() -> Path:
    """The upstream UVDoc checkout named by ``FLATTENER_UVDOC_REPO``."""
    if not os.environ.get("FLATTENER_UVDOC_REPO"):
        pytest.skip("set FLATTENER_UVDOC_REPO to an upstream UVDoc checkout")
    repo = Path(os.environ["FLATTENER_UVDOC_REPO"])
    if not (repo / "model.py").is_file() or not (repo / "model/best_model.pkl").is_file():
        pytest.skip(f"UVDoc reference repository not found at {repo}")
    return repo


def test_natural_grid_size_follows_four_stride_two_stages():
    assert natural_grid_size((712, 488)) == (45, 31)
    assert natural_grid_size((1440, 1024)) == (90, 64)
    assert DewarpModelConfig().natural_grid_size == DewarpModelConfig().grid_size


def test_upstream_configuration_has_upstream_parameter_count():
    assert count_parameters(UVDocNet(UVDOC_CONFIG)) == {"total": 8_003_463, "trainable": 8_003_463}


def test_config_validation_and_round_trip():
    config = DewarpModelConfig(input_size=(1440, 1024), grid_size=(90, 64), with_3d=False)
    assert DewarpModelConfig.from_dict(config.to_dict()) == config
    for bad in (
        {"kernel_size": 4},
        {"grid_size": (2, 31)},
        {"input_size": (712,)},
        {"grid_convention": "other"},
        {"num_filter": True},
    ):
        with pytest.raises(ValueError):
            DewarpModelConfig(**bad)
    with pytest.raises(ValueError, match="unknown"):
        DewarpModelConfig.from_dict({"channels_last": True})


@pytest.mark.parametrize("grid_size", [(4, 3), (9, 7)])
def test_output_shapes_for_natural_and_resized_grids(grid_size):
    config = DewarpModelConfig(input_size=(64, 48), grid_size=grid_size, num_filter=4)
    model = UVDocNet(config).eval()
    with torch.no_grad():
        grid2d, grid3d = model(torch.rand(2, 3, 64, 48))
    assert grid2d.shape == (2, 2, *grid_size) and grid3d.shape == (2, 3, *grid_size)
    no3d = UVDocNet(
        DewarpModelConfig(input_size=(64, 48), grid_size=grid_size, num_filter=4, with_3d=False)
    ).eval()
    with torch.no_grad():
        assert no3d(torch.rand(1, 3, 64, 48))[1] is None


def test_upstream_format_weights_load_into_matching_and_2d_only_models(tmp_path):
    source = UVDocNet(TINY)
    path = tmp_path / "best_model.pkl"
    torch.save({"model_state": source.state_dict()}, path)
    target = UVDocNet(TINY)
    lineage = load_uvdoc_weights(target, path)
    assert lineage["kind"] == "uvdoc" and lineage["license"] == "unverified"  # not upstream
    assert lineage["sha256_matches_upstream"] is False
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])
    load_uvdoc_weights(UVDocNet(DewarpModelConfig(**{**TINY.to_dict(), "with_3d": False})), path)
    torch.save({"model": source.state_dict()}, tmp_path / "other.pkl")
    # A dewarp checkpoint also holds 'model_state'; only the bare upstream format is accepted.
    torch.save({"model_state": source.state_dict(), "track": "research"}, tmp_path / "ckpt.pt")
    for name in ("other.pkl", "ckpt.pt"):
        with pytest.raises(ValueError, match="model_state"):
            load_uvdoc_weights(target, tmp_path / name)


def test_forward_parity_with_upstream_uvdoc():
    repo = _uvdoc_repo()
    spec = importlib.util.spec_from_file_location("uvdoc_upstream_model", repo / "model.py")
    upstream_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream_module)
    upstream = upstream_module.UVDocnet(num_filter=32, kernel_size=5)
    state = torch.load(repo / "model/best_model.pkl", map_location="cpu", weights_only=True)
    upstream.load_state_dict(state["model_state"])
    upstream.eval()
    ours, lineage = uvdoc_model(repo / "model/best_model.pkl")
    assert lineage["sha256"] == UVDOC_WEIGHTS_SHA256 and lineage["license"] == "MIT"
    assert list(ours.state_dict()) == list(upstream.state_dict())
    generator = torch.Generator().manual_seed(20260926)
    with torch.no_grad():
        for _ in range(3):
            image = torch.rand(1, 3, 712, 488, generator=generator)
            reference, candidate = upstream(image), ours(image)
            for expected, actual in zip(reference, candidate, strict=True):
                assert actual.shape == expected.shape
                assert (actual - expected).abs().max().item() <= 1e-5
