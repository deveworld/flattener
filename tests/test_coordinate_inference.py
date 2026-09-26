"""Untrained checkpoint fixtures: inference and validation, never fitting."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flattener import RenderConfig, identity_map, render
from flattener.contracts import FlattenerError
from flattener.learning import coordinate_study as study
from flattener.learning.coordinate_inference import load_coordinate_predictor
from flattener.learning.inference import LoadedPredictor, unwarp
from flattener.learning.model import ModelConfig, coarse_identity
from flattener.learning.spatial import make_coordinate_model


@pytest.fixture(autouse=True, scope="module")
def never_train():
    def forbidden(*args, **kwargs):
        pytest.fail("inference fixtures must never train or construct an optimizer")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.Tensor, "backward", forbidden)
        patch.setattr(torch.autograd, "backward", forbidden)
        patch.setattr(torch.optim.Optimizer, "__init__", forbidden)
        patch.setattr(study, "SceneCoordinateDataset", forbidden)
        yield


@pytest.fixture(scope="module")
def study_models(tmp_path_factory, never_train):
    root = tmp_path_factory.mktemp("untrained-coordinate-fixtures")
    for architecture in ("global", "spatial", "page", "projective", "surface", "oriented_surface"):
        config = study.StudyConfig(
            architecture=architecture,
            map_mode=("analytic" if architecture in (
                "projective", "surface", "oriented_surface") else "legacy"),
            page_weight=1.0 if architecture == "oriented_surface" else 0.0,
            direction_weight=1.0 if architecture == "oriented_surface" else 0.0,
            corner_weight=1.0 if architecture == "oriented_surface" else 0.0,
            model_config=ModelConfig(
                input_size=(32, 24), grid_size=(9, 7), channels=(4, 8, 12), hidden=16
            ),
            output_size=(128, 128),
        )
        with torch.random.fork_rng(devices=[]):
            model = make_coordinate_model(architecture, config.model_config)
        payload = {
            "schema_version": study.CHECKPOINT_SCHEMA,
            "checkpoint_kind": "best",
            "config": asdict(config),
            "model_spec": study._model_spec(config),
            "model_state": model.state_dict(),
            "optimizer_state": {
                "state": {},
                "param_groups": [study._adam_group(config, len(list(model.parameters())))],
            },
            "epoch": 0,
            "best_epoch": 0,
            "history": [],
            "initial_validation_epe_pixels": 1.0,
            "best_validation_epe_pixels": 1.0,
            "manifest_sha256": "0" * 64,
            "provenance": study._provenance(),
            "dataset_output_size": config.output_size,
            "dataset_photo_size": (128, 128),
            "dataset_counts": {"train": 6, "validation": 6, "test": 6},
            "torch_rng_state": torch.get_rng_state(),
            "shuffle_rng_state": torch.Generator().manual_seed(config.seed).get_state(),
            # Serialized shape fixture only, never passed to any CUDA RNG API.
            "cuda_rng_state": torch.zeros(16, dtype=torch.uint8),
            "xpu_rng_state": None,
            "device_info": {"device": "cuda:0", "name": "fixture", "capability": (8, 0)},
            "best_checkpoint_sha256": None,
        }
        directory = root / architecture
        directory.mkdir()
        study._save_checkpoint(directory / "best.pt", payload)
    return root


@pytest.mark.parametrize("architecture", ["global", "spatial", "page", "oriented_surface"])
def test_model_load_and_existing_renderer_exact(study_models, architecture):
    checkpoint = study_models / architecture / "best.pt"
    before_rng = torch.get_rng_state().clone()
    before_threads = torch.get_num_threads()
    before_deterministic = torch.are_deterministic_algorithms_enabled()
    predictor = load_coordinate_predictor(checkpoint)
    assert isinstance(predictor, LoadedPredictor)
    assert predictor.checksum_verified
    assert architecture in predictor.model_id
    assert predictor.model_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert predictor.checkpoint_path == str(checkpoint.resolve())
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert torch.get_num_threads() == before_threads
    assert torch.are_deterministic_algorithms_enabled() == before_deterministic
    assert all(
        p.device.type == "cpu" and not p.requires_grad for p in predictor._model.parameters()
    )
    image = np.random.default_rng(111).integers(0, 256, (129, 151, 3), dtype=np.uint8)
    original = image.copy()
    grid = predictor.predict_map(image, (128, 136))
    np.testing.assert_array_equal(grid, identity_map((128, 136)))
    direct = render(image, grid, config=RenderConfig(on_error="return"))
    public = unwarp(image, predictor, (128, 136), config=RenderConfig(on_error="return"))
    assert public.status == direct.status and public.reason_code == direct.reason_code
    for key in ("image", "source_mask", "target_mask", "valid_mask"):
        np.testing.assert_array_equal(getattr(public, key), getattr(direct, key))
    exact = unwarp(image, predictor, image.shape[:2])
    np.testing.assert_array_equal(exact.image, image)
    np.testing.assert_array_equal(image, original)
    assert public.metadata["model_id"] == predictor.model_id
    assert torch.equal(torch.get_rng_state(), before_rng)


@pytest.mark.parametrize("outcome", ["success", "return", "original"])
def test_photo_cli_exit_status_and_saved_results(study_models, tmp_path, outcome):
    """Use actual subprocess inference with hand-set weights, never training."""
    saved = torch.load(study_models / "global" / "best.pt", weights_only=True)
    if outcome != "success":
        residual = torch.zeros(2, *saved["config"]["model_config"]["grid_size"])
        residual[0] = -2 * coarse_identity(tuple(residual.shape[1:]))[..., 0]
        saved["model_state"]["head.3.bias"] = residual.flatten()
    checkpoint = tmp_path / "model.pt"
    study._save_checkpoint(checkpoint, saved)
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    input_path = tmp_path / "input.png"
    Image.fromarray(image).save(input_path)
    output = tmp_path / "output"
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "examples" / "unwarp_photo.py"),
            "--checkpoint", str(checkpoint),
            "--input", str(input_path),
            "--output-dir", str(output),
            "--height", "32", "--width", "32",
            "--on-error", "return" if outcome == "success" else outcome,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert process.returncode == (0 if outcome == "success" else 1), process.stderr
    record = json.loads((output / "metadata.json").read_text())
    expected_status = "success" if outcome == "success" else "invalid_transform"
    assert record["status"] == record["metadata"]["processing_status"] == expected_status
    assert record["metadata"]["quality_status"] == "not_evaluated"
    assert record["used_fallback"] == (outcome == "original")
    assert record["reason_code"] == (None if outcome == "success" else "fold_or_degenerate")
    assert expected_status in process.stdout
    if outcome == "return":
        assert {path.name for path in output.iterdir()} == {"metadata.json"}
    else:
        with Image.open(output / "image.png") as decoded:
            np.testing.assert_array_equal(np.asarray(decoded), image)


@pytest.mark.parametrize("architecture", ["global", "spatial", "page", "oriented_surface"])
def test_explicit_cpu_checkpoint_loads_without_cuda(
    study_models, architecture, monkeypatch, tmp_path
):
    saved = torch.load(study_models / architecture / "best.pt", weights_only=True)
    saved["config"]["device"] = "cpu"
    saved["device_info"] = {"device": "cpu", "name": "fixture", "capability": None}
    saved["cuda_rng_state"] = None
    checkpoint = tmp_path / "cpu.pt"
    study._save_checkpoint(checkpoint, saved)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU checkpoint loading attempted CUDA access")

    for name in ("is_available", "get_device_name", "get_rng_state", "set_rng_state"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    predictor = load_coordinate_predictor(checkpoint)
    grid = predictor.predict_map(np.zeros((128, 128, 3), dtype=np.uint8), (128, 128))
    np.testing.assert_array_equal(grid, identity_map((128, 128)))
    assert all(p.device.type == "cpu" for p in predictor._model.parameters())


def _legacy_payload(study_models, architecture, schema="coordinate-study-0.1"):
    """Recreate exact historical fields, not the current dataclass shape."""
    saved = torch.load(study_models / architecture / "best.pt", weights_only=True)
    assert schema in (
        "coordinate-study-0.1", "coordinate-study-0.2", "coordinate-study-0.3",
        "coordinate-study-0.4", "coordinate-study-0.5", "coordinate-study-0.6",
        "coordinate-study-0.7", "coordinate-study-0.8", "coordinate-study-0.9",
        "coordinate-study-0.10", "coordinate-study-0.11",
    )
    version = int(schema.rsplit(".", 1)[1])
    saved["schema_version"] = schema
    if version < 11:
        del saved["xpu_rng_state"]
    fields = (
        "architecture", "device", "seed", "epochs", "batch_size", "learning_rate",
        "model_config", "output_size",
    )
    if version >= 2:
        fields += ("topology_weight", "curvature_weight", "gradient_clip_norm")
    if version >= 3:
        fields += ("page_weight", "boundary_weight", "differential_weight")
    if version >= 4:
        fields += ("include_paper_augmentation", "initial_checkpoint_sha256")
    if version >= 5:
        fields += ("map_mode", "topology_mode", "initialization_mode")
    if version >= 6:
        fields += ("page_relative_weight",)
    if version >= 7:
        fields += ("orientation_weight", "rotation_probability")
    if version >= 9:
        fields += ("background_probability", "rotation_mode", "initial_model_config")
    if version >= 10:
        fields += ("direction_weight", "corner_weight")
    saved["config"] = {key: saved["config"][key] for key in fields}
    saved["provenance"] = study._provenance(schema)
    if schema == "coordinate-study-0.1":
        assert "loss_protocol" not in saved["provenance"]
    elif schema == "coordinate-study-0.2":
        protocol = saved["provenance"]["loss_protocol"]
        assert protocol["version"] == "coordinate-objective-0.2"
        assert not {"page", "boundary", "differential"} & protocol.keys()
    elif schema == "coordinate-study-0.5":
        assert saved["provenance"]["loss_protocol"]["version"] == "coordinate-objective-0.5"
        assert "page_relative" not in saved["provenance"]["loss_protocol"]
    elif schema == "coordinate-study-0.6":
        assert saved["provenance"]["loss_protocol"]["version"] == "coordinate-objective-0.6"
        assert "orientation" not in saved["provenance"]["loss_protocol"]
    elif schema == "coordinate-study-0.7":
        assert saved["provenance"]["loss_protocol"]["version"] == "coordinate-objective-0.7"
        assert "reduction" not in saved["provenance"]["loss_protocol"]
    elif schema == "coordinate-study-0.8":
        assert saved["provenance"]["loss_protocol"]["version"] == "coordinate-objective-0.8"
        assert "rotation_mode" not in saved["provenance"]["augmentation_protocol"]
    elif schema == "coordinate-study-0.9":
        assert saved["provenance"]["loss_protocol"]["version"] == "coordinate-objective-0.8"
        assert saved["provenance"]["augmentation_protocol"]["version"] == (
            "coordinate-augmentation-0.9")
    elif version >= 10:
        assert saved["provenance"]["loss_protocol"]["version"] == "coordinate-objective-0.10"
        assert "oriented_surface" in saved["provenance"]["map_protocol"]
    else:
        assert saved["provenance"]["loss_protocol"]["version"] == "coordinate-objective-0.3"
    if version < 4:
        assert "include_paper_augmentation" not in saved["config"]
        assert "initial_checkpoint_sha256" not in saved["config"]
    if version < 5:
        assert not {"map_mode", "topology_mode", "initialization_mode"} & saved["config"].keys()
    if version < 6:
        assert "page_relative_weight" not in saved["config"]
    if version < 7:
        assert not {"orientation_weight", "rotation_probability"} & saved["config"].keys()
        assert "augmentation_protocol" not in saved["provenance"]
    if version < 10:
        assert not {"direction_weight", "corner_weight"} & saved["config"].keys()
    if version < 11:
        assert "torch_xpu" not in saved["provenance"]["versions"]
        assert "execution_protocol" not in saved["provenance"]
    assert "direction_training" not in saved["config"]
    assert "direction_training_protocol" not in saved["provenance"]
    return saved


@pytest.mark.parametrize("architecture", ["global", "spatial"])
@pytest.mark.parametrize("schema", [
    "coordinate-study-0.1", "coordinate-study-0.2", "coordinate-study-0.3",
    "coordinate-study-0.4", "coordinate-study-0.5", "coordinate-study-0.6",
    "coordinate-study-0.7", "coordinate-study-0.8", "coordinate-study-0.9",
    "coordinate-study-0.10", "coordinate-study-0.11",
])
def test_legacy_schema_loads_without_rewriting_or_adding_new_objectives(
    study_models, architecture, schema, tmp_path
):
    saved = _legacy_payload(study_models, architecture, schema)
    regularizers = (0, 0, 0)
    if schema != "coordinate-study-0.1":
        saved["config"].update(topology_weight=0.5, curvature_weight=0.25, gradient_clip_norm=2.0)
        regularizers = (0.5, 0.25, 2.0)
    config = study._validate_saved(saved)
    assert (
        config.topology_weight, config.curvature_weight, config.gradient_clip_norm
    ) == regularizers
    assert (config.page_weight, config.boundary_weight, config.differential_weight) == (0, 0, 0)
    assert config.include_paper_augmentation is False
    assert config.initial_checkpoint_sha256 is None
    assert config.page_relative_weight == 0
    assert config.orientation_weight == config.rotation_probability == 0
    assert config.background_probability == 0 and config.rotation_mode == "batch"
    assert config.initial_model_config is None
    assert config.direction_weight == config.corner_weight == 0
    assert config.direction_training == "predicted"
    assert (config.map_mode, config.topology_mode, config.initialization_mode) == (
        "legacy", "absolute", "full"
    )
    checkpoint = tmp_path / "old.pt"
    study._save_checkpoint(checkpoint, saved)
    before = checkpoint.read_bytes()
    loaded = study.coordinate_checkpoint(checkpoint)
    assert loaded["schema_version"] == schema
    assert len(loaded["config"]) == {
        "coordinate-study-0.1": 8, "coordinate-study-0.2": 11, "coordinate-study-0.3": 14,
        "coordinate-study-0.4": 16, "coordinate-study-0.5": 19, "coordinate-study-0.6": 20,
        "coordinate-study-0.7": 22, "coordinate-study-0.8": 22, "coordinate-study-0.9": 25,
        "coordinate-study-0.10": 27, "coordinate-study-0.11": 27,
    }[schema]
    predictor = load_coordinate_predictor(checkpoint)
    grid = predictor.predict_map(np.zeros((128, 128, 3), dtype=np.uint8), (128, 128))
    np.testing.assert_array_equal(grid, identity_map((128, 128)))
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("architecture", ["global", "spatial", "page"])
def test_new_schema_records_explicit_regularization_contract(study_models, architecture):
    saved = torch.load(study_models / architecture / "best.pt", weights_only=True)
    assert saved["schema_version"] == study.CHECKPOINT_SCHEMA == "coordinate-study-0.12"
    assert len(saved["config"]) == 28
    assert saved["config"]["direction_training"] == "predicted"
    assert saved["config"]["include_paper_augmentation"] is False
    assert saved["config"]["initial_checkpoint_sha256"] is None
    protocol = saved["provenance"]["loss_protocol"]
    assert protocol["version"] == "coordinate-objective-0.10"
    assert "final FP32 bits/gradients may differ" in protocol["reduction"]
    assert {
        "version", "topology", "minimum_normalized_jacobian", "curvature", "gradient_clip",
        "page", "boundary", "differential", "page_relative", "orientation",
    } <= protocol.keys()
    assert {key: saved["config"][key] for key in (
        "map_mode", "topology_mode", "initialization_mode"
    )} == {"map_mode": "legacy", "topology_mode": "absolute", "initialization_mode": "full"}
    assert "original source-mask" in protocol["page"]
    assert "outer 8 output pixels" in protocol["boundary"]
    assert "align_corners=False" in protocol["differential"]
    assert "max-min+1" in protocol["page_relative"]
    assert "180-degree" in protocol["orientation"]
    assert "CPU-global" in saved["provenance"]["augmentation_protocol"]["quarter_turn"]
    assert "nearest-exact" in saved["provenance"]["augmentation_protocol"]["source_mask"]
    page_weight = 0.75 if architecture == "page" else 0.0
    saved["config"].update(
        topology_weight=0.5, curvature_weight=0.25, gradient_clip_norm=2.0,
        page_weight=page_weight, boundary_weight=0.125, differential_weight=0.0625,
        page_relative_weight=0.25,
        orientation_weight=50.0, rotation_probability=0.5,
    )
    config = study._validate_saved(saved)
    assert (config.topology_weight, config.curvature_weight, config.gradient_clip_norm) == (
        0.5, 0.25, 2.0
    )
    assert (config.page_weight, config.boundary_weight, config.differential_weight) == (
        page_weight, 0.125, 0.0625
    )
    assert config.page_relative_weight == 0.25
    assert config.orientation_weight == 50.0 and config.rotation_probability == 0.5


@pytest.mark.parametrize("architecture", ["global", "surface"])
def test_07_objective_load_and_warm_start_keep_saved_batch_and_weights(
    study_models, architecture, tmp_path
):
    saved = _legacy_payload(study_models, architecture, study.ORIENTATION_CHECKPOINT_SCHEMA)
    saved["config"].update(batch_size=16, orientation_weight=50.0, rotation_probability=0.5)
    validated = study._validate_saved(saved)
    assert validated.batch_size == 16 and study.StudyConfig().batch_size == 32
    assert validated.orientation_weight == 50 and validated.rotation_probability == 0.5
    checkpoint = tmp_path / "old07.pt"
    digest = study._save_checkpoint(checkpoint, saved)
    fresh = replace(validated, epochs=2)
    initialized, state = study._initialization(fresh, checkpoint, None)
    assert initialized.initial_checkpoint_sha256 == digest
    assert study._same_state(state, saved["model_state"])
    assert saved["provenance"]["loss_protocol"]["version"] == "coordinate-objective-0.7"
    saved["schema_version"] = study.CHECKPOINT_SCHEMA
    saved["xpu_rng_state"] = None
    saved["config"].update(background_probability=0.0, rotation_mode="batch",
                           initial_model_config=None, direction_weight=0.0, corner_weight=0.0,
                           direction_training="predicted")
    old_objective = saved["provenance"]["loss_protocol"]
    saved["provenance"] = study._provenance()
    saved["provenance"]["loss_protocol"] = old_objective
    with pytest.raises(FlattenerError, match="regularization protocol"):
        study._validate_saved(saved)


def test_checkpoint_validation_does_not_rehash_current_source(study_models, monkeypatch):
    saved = torch.load(study_models / "surface" / "best.pt", weights_only=True)

    def forbidden(*args, **kwargs):
        pytest.fail("checkpoint contract validation must not hash the current source tree")

    monkeypatch.setattr(study, "_sha256", forbidden)
    study._validate_saved(saved)
    # Structural/source-digest validation remains strict without re-reading
    # the current implementation (which may legitimately differ from an old checkpoint).
    saved["provenance"]["source_code_sha256"]["learning/spatial.py"] = "bad"
    with pytest.raises(FlattenerError, match="fingerprints"):
        study._validate_saved(saved)


@pytest.mark.parametrize("architecture", ["global", "spatial", "page", "projective", "surface"])
def test_resolution_lineage_from_08_checkpoint_loads_at_native_input_resolution(
    study_models, architecture, tmp_path
):
    old = _legacy_payload(study_models, architecture, study.OPTIMIZED_CHECKPOINT_SCHEMA)
    source_config = study._validate_saved(old)
    source = tmp_path / "source.pt"
    digest = study._save_checkpoint(source, old)
    requested = replace(source_config, initialization_mode="resolution",
                        model_config=replace(source_config.model_config, input_size=(384, 288)))
    before_rng = torch.get_rng_state().clone()
    config, state = study._initialization(requested, source, None)
    assert config.initial_model_config == source_config.model_config
    assert config.initial_checkpoint_sha256 == digest
    assert study._same_state(state, old["model_state"])
    saved = dict(old, schema_version=study.CHECKPOINT_SCHEMA, config=asdict(config),
                 model_state=state, model_spec=study._model_spec(config),
                 provenance=study._provenance(), xpu_rng_state=None)
    checkpoint = tmp_path / "native.pt"
    study._save_checkpoint(checkpoint, saved)
    loaded = load_coordinate_predictor(checkpoint)
    assert loaded._model.config.input_size == (384, 288)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    np.testing.assert_array_equal(loaded.predict_map(image, (128, 128)), identity_map((128, 128)))
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert study._validate_saved(saved).initial_model_config.input_size == (32, 24)


@pytest.mark.parametrize("field", ["input_size", "grid_size", "channels", "hidden", "extra"])
def test_resolution_checkpoint_requires_complete_exact_source_model_config(study_models, field):
    saved = torch.load(study_models / "surface" / "best.pt", weights_only=True)
    source = dict(saved["config"]["model_config"])
    saved["config"].update(initialization_mode="resolution",
                           initial_model_config=source, initial_checkpoint_sha256="e" * 64)
    if field == "extra":
        source["unknown"] = 1
    else:
        del source[field]
    with pytest.raises(FlattenerError, match="source ModelConfig fields"):
        study._validate_saved(saved)


@pytest.mark.parametrize("field", ["initial_model_config", "initial_checkpoint_sha256"])
def test_resolution_checkpoint_cannot_omit_bound_source(study_models, field):
    saved = torch.load(study_models / "surface" / "best.pt", weights_only=True)
    saved["config"].update(initialization_mode="resolution",
                           initial_model_config=dict(saved["config"]["model_config"]),
                           initial_checkpoint_sha256="e" * 64)
    saved["config"][field] = None
    with pytest.raises(FlattenerError, match="requires source config and checkpoint SHA"):
        study._validate_saved(saved)


@pytest.mark.parametrize("field,value", [("background_probability", 0.0),
                                         ("rotation_mode", "batch"),
                                         ("initial_model_config", None)])
def test_08_schema_rejects_new_augmentation_or_lineage_fields(study_models, field, value):
    saved = _legacy_payload(study_models, "surface", study.OPTIMIZED_CHECKPOINT_SCHEMA)
    saved["config"][field] = value
    with pytest.raises(FlattenerError, match="configuration fields"):
        study._validate_saved(saved)
    current = torch.load(study_models / "surface" / "best.pt", weights_only=True)
    del current["config"][field]
    with pytest.raises(FlattenerError, match="configuration fields"):
        study._validate_saved(current)


@pytest.mark.parametrize("schema", [
    "coordinate-study-0.9", "coordinate-study-0.10", "coordinate-study-0.11",
    "coordinate-study-0.12",
])
def test_09_augmentation_protocol_is_explicit_and_strict(study_models, schema):
    saved = (_legacy_payload(study_models, "surface", schema)
             if schema != "coordinate-study-0.12" else
             torch.load(study_models / "surface" / "best.pt", weights_only=True))
    saved["config"].update(rotation_mode="per_image_fit", rotation_probability=0.25,
                           background_probability=0.5)
    study._validate_saved(saved)
    protocol = saved["provenance"]["augmentation_protocol"]
    assert protocol["version"] == "coordinate-augmentation-0.9"
    assert "original-photo physical pixel aspect ratio" in protocol["rotation_mode"]
    assert "before rotation" in protocol["background"]
    assert protocol["fit_module_version"] == study.AUGMENTATION_VERSION
    assert protocol["fit_module_protocol"] == study.AUGMENTATION_PROTOCOL
    assert "initial_model_config" in saved["provenance"]["initialization_protocol"]["resolution"]
    saved["provenance"]["augmentation_protocol"] = study._provenance(
        study.OPTIMIZED_CHECKPOINT_SCHEMA)["augmentation_protocol"]
    with pytest.raises(FlattenerError, match="augmentation protocol"):
        study._validate_saved(saved)


@pytest.mark.parametrize("schema", [
    "coordinate-study-0.9", "coordinate-study-0.10", "coordinate-study-0.11",
    "coordinate-study-0.12",
])
def test_09_checkpoint_requires_augmentation_source_fingerprint(study_models, schema):
    saved = (_legacy_payload(study_models, "surface", schema)
             if schema != "coordinate-study-0.12" else
             torch.load(study_models / "surface" / "best.pt", weights_only=True))
    del saved["provenance"]["source_code_sha256"]["learning/augmentation.py"]
    with pytest.raises(FlattenerError, match="source fingerprints"):
        study._validate_saved(saved)
    old = _legacy_payload(study_models, "surface", study.OPTIMIZED_CHECKPOINT_SCHEMA)
    old["provenance"]["source_code_sha256"].pop("learning/augmentation.py", None)
    study._validate_saved(old)


@pytest.mark.parametrize("schema", [f"coordinate-study-0.{version}" for version in range(1, 10)])
@pytest.mark.parametrize("field", ["direction_weight", "corner_weight"])
def test_pre_oriented_schemas_reject_even_zero_semantic_objective_fields(
    study_models, schema, field
):
    saved = _legacy_payload(study_models, "global", schema)
    saved["config"][field] = 0.0
    with pytest.raises(FlattenerError, match="configuration fields"):
        study._validate_saved(saved)


@pytest.mark.parametrize("schema", [f"coordinate-study-0.{version}" for version in range(1, 10)])
def test_pre_oriented_schema_cannot_claim_oriented_architecture(study_models, schema):
    saved = _legacy_payload(study_models, "oriented_surface", schema)
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


@pytest.mark.parametrize("field,value", [
    ("page_weight", 0.0), ("direction_weight", 0.0), ("corner_weight", 0.0),
    ("direction_weight", -1), ("corner_weight", float("nan")),
    ("direction_weight", True), ("corner_weight", None), ("map_mode", "legacy"),
])
def test_oriented_checkpoint_requires_explicit_analytic_semantic_supervision(
    study_models, field, value
):
    saved = torch.load(study_models / "oriented_surface" / "best.pt", weights_only=True)
    saved["config"][field] = value
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


@pytest.mark.parametrize("head", ["orientation_head.bias", "page_head.2.bias"])
@pytest.mark.parametrize("fault", ["missing", "shape", "nonfinite"])
def test_oriented_checkpoint_requires_complete_finite_semantic_head_weights(
    study_models, head, fault
):
    saved = torch.load(study_models / "oriented_surface" / "best.pt", weights_only=True)
    if fault == "missing":
        del saved["model_state"][head]
    elif fault == "shape":
        saved["model_state"][head] = torch.zeros(9)
    else:
        saved["model_state"][head].fill_(float("nan"))
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


def test_surface_weights_cannot_be_relabelled_as_oriented_surface(study_models):
    saved = torch.load(study_models / "oriented_surface" / "best.pt", weights_only=True)
    saved["model_state"] = torch.load(
        study_models / "surface" / "best.pt", weights_only=True)["model_state"]
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


@pytest.mark.parametrize("protocol,key", [
    ("loss_protocol", "direction"), ("loss_protocol", "corners"),
    ("loss_protocol", "semantic_page"), ("map_protocol", "oriented_surface"),
    ("initialization_protocol", "oriented_backbone"),
])
def test_oriented_checkpoint_protocol_cannot_be_dropped_or_changed(study_models, protocol, key):
    saved = torch.load(study_models / "oriented_surface" / "best.pt", weights_only=True)
    assert study._validate_saved(saved).architecture == "oriented_surface"
    del saved["provenance"][protocol][key]
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)
    saved["provenance"][protocol][key] = "different coordinate convention"
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


@pytest.mark.parametrize("photo_size", [(47, 83), (83, 47), (64, 64)])
def test_oriented_inference_passes_original_photo_aspect_and_preserves_metadata(
    study_models, photo_size, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("portable oriented inference attempted CUDA access")

    for name in ("is_available", "get_device_name", "get_rng_state", "set_rng_state"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    checkpoint = study_models / "oriented_surface" / "best.pt"
    rng = torch.get_rng_state().clone()
    predictor = load_coordinate_predictor(checkpoint)
    assert all(not p.requires_grad and p.device.type == "cpu"
               for p in predictor._model.parameters())
    calls = []
    original_forward = predictor._model.forward_dense

    def forward(images, size, *, photo_size):
        calls.append((tuple(images.shape), size, photo_size))
        return original_forward(images, size, photo_size=photo_size)

    monkeypatch.setattr(predictor._model, "forward_dense", forward)
    image = np.random.default_rng(22).integers(0, 256, (*photo_size, 3), dtype=np.uint8)
    before = image.copy()
    size = (31, 41)
    grid = predictor.predict_map(image, size)
    assert grid.shape == (*size, 2) and grid.dtype == np.float32 and np.isfinite(grid).all()
    # Fresh zero geometry/direction heads choose the upright identity, even
    # though page-attention parameters are nonzero random initialization.
    np.testing.assert_array_equal(grid, identity_map(size))
    result = unwarp(image, predictor, size, config=RenderConfig(on_error="return"))
    assert calls == [((1, 3, 32, 24), size, photo_size)] * 2
    assert result.metadata["model_input_size"] == [32, 24]
    assert result.metadata["model_map_mode"] == "analytic"
    assert "oriented_surface" in result.metadata["model_id"]
    assert result.metadata["model_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert result.metadata["processing_status"] == str(result.status)
    assert result.metadata["quality_status"] == "not_evaluated"
    np.testing.assert_array_equal(image, before)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(p.grad is None for p in predictor._model.parameters())


@pytest.mark.parametrize("input_size", [(32, 24), (40, 56)])
def test_oriented_network_input_size_does_not_replace_runtime_photo_size(
    study_models, tmp_path, input_size, monkeypatch
):
    saved = torch.load(study_models / "oriented_surface" / "best.pt", weights_only=True)
    saved["config"]["model_config"]["input_size"] = input_size
    checkpoint = tmp_path / "resized-input.pt"
    study._save_checkpoint(checkpoint, saved)
    predictor = load_coordinate_predictor(checkpoint)
    calls = []

    def forward(images, size, *, photo_size):
        calls.append((tuple(images.shape[-2:]), photo_size))
        # An independent aspect-sensitive decoder probe catches accidental use
        # of checkpoint photo_size, network input_size, or requested output_size.
        dense = coarse_identity(size).unsqueeze(0)
        dense[..., 0] *= photo_size[0] / photo_size[1]
        return torch.zeros(1, 2, 9, 7), dense

    monkeypatch.setattr(predictor._model, "forward_dense", forward)
    size = (17, 19)
    actual = predictor.predict_map(np.zeros((45, 90, 3), dtype=np.uint8), size)
    expected = identity_map(size).copy()
    expected[..., 0] *= 0.5
    np.testing.assert_array_equal(actual, expected)
    assert calls == [(input_size, (45, 90))]


@pytest.mark.parametrize("photo_size", [(47, 83), (83, 47)])
@pytest.mark.parametrize("quarter_turn", [1, 2, 3])
def test_oriented_checkpoint_quarter_turn_map_is_physical_unclipped_and_original_frame(
    study_models, tmp_path, photo_size, quarter_turn
):
    saved = torch.load(study_models / "oriented_surface" / "best.pt", weights_only=True)
    saved["model_state"]["orientation_head.bias"][quarter_turn] = 2.0
    checkpoint = tmp_path / "turned.pt"
    study._save_checkpoint(checkpoint, saved)
    predictor = load_coordinate_predictor(checkpoint)
    size = (29, 37)
    image = np.zeros((*photo_size, 3), dtype=np.uint8)
    actual = predictor.predict_map(image, size)
    flat = identity_map(size)
    x, y = flat[..., 0], flat[..., 1]
    height, width = photo_size
    expected = np.stack({
        1: (-y * (height / width), x * (width / height)),
        2: (-x, -y),
        3: (y * (height / width), -x * (width / height)),
    }[quarter_turn], axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert actual.shape == (*size, 2)  # The upright flat document lattice is unchanged.
    if quarter_turn % 2:
        assert np.max(np.abs(actual)) > 1.0  # Never fit/crop/clamp predicted source coordinates.
    assert all(p.grad is None for p in predictor._model.parameters())


def test_diversity_09_checkpoint_preserves_nondefault_controls_and_resolution_lineage(
    study_models, tmp_path
):
    assert study.DIVERSITY_CHECKPOINT_SCHEMA == "coordinate-study-0.9"
    saved = _legacy_payload(study_models, "surface", study.DIVERSITY_CHECKPOINT_SCHEMA)
    source_config = dict(saved["config"]["model_config"])
    saved["config"]["model_config"]["input_size"] = (40, 56)
    saved["config"].update(
        initial_model_config=source_config, initial_checkpoint_sha256="a" * 64,
        initialization_mode="resolution", background_probability=0.5,
        rotation_probability=0.25, rotation_mode="per_image_fit")
    saved["model_state"]["head.3.bias"][0] = 0.1
    checkpoint = tmp_path / "old-09.pt"
    study._save_checkpoint(checkpoint, saved)
    before = checkpoint.read_bytes()
    rng = torch.get_rng_state().clone()
    config = study._validate_saved(study.coordinate_checkpoint(checkpoint))
    assert config.direction_weight == config.corner_weight == 0
    assert config.model_config.input_size == (40, 56)
    assert asdict(config.initial_model_config) == source_config
    assert config.initial_checkpoint_sha256 == "a" * 64
    assert config.initialization_mode == "resolution"
    assert config.rotation_mode == "per_image_fit"
    assert config.rotation_probability == 0.25 and config.background_probability == 0.5
    loaded = load_coordinate_predictor(checkpoint)
    assert study._same_state(loaded._model.state_dict(), saved["model_state"])
    grid = loaded.predict_map(np.zeros((47, 83, 3), dtype=np.uint8), (31, 41))
    assert np.isfinite(grid).all() and not np.array_equal(grid, identity_map((31, 41)))
    assert checkpoint.read_bytes() == before and torch.equal(torch.get_rng_state(), rng)


def test_semantic_010_checkpoint_loads_without_xpu_state_or_execution_protocol(
    study_models, tmp_path
):
    assert study.SEMANTIC_CHECKPOINT_SCHEMA == "coordinate-study-0.10"
    saved = _legacy_payload(study_models, "oriented_surface", study.SEMANTIC_CHECKPOINT_SCHEMA)
    assert "xpu_rng_state" not in saved
    assert len(saved["config"]) == 27
    config = study._validate_saved(saved)
    assert config.direction_weight == config.corner_weight == config.page_weight == 1
    checkpoint = tmp_path / "semantic-010.pt"
    study._save_checkpoint(checkpoint, saved)
    before = checkpoint.read_bytes()
    loaded = load_coordinate_predictor(checkpoint)
    np.testing.assert_array_equal(
        loaded.predict_map(np.zeros((47, 83, 3), dtype=np.uint8), (31, 41)),
        identity_map((31, 41)),
    )
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("schema", [f"coordinate-study-0.{version}" for version in range(1, 11)])
@pytest.mark.parametrize("claim", ["rng", "version", "execution", "device"])
def test_pre_xpu_checkpoints_reject_new_backend_claims(study_models, schema, claim):
    saved = _legacy_payload(study_models, "global", schema)
    if claim == "rng":
        saved["xpu_rng_state"] = None
    elif claim == "version":
        saved["provenance"]["versions"]["torch_xpu"] = None
    elif claim == "execution":
        saved["provenance"]["execution_protocol"] = study._provenance()["execution_protocol"]
    else:
        saved["config"]["device"] = "xpu:0"
        saved["device_info"] = {"device": "xpu:0", "name": "portable XPU fixture",
                                 "capability": None}
        saved["cuda_rng_state"] = None
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


@pytest.mark.parametrize("field", ["missing_rng", "missing_version", "bad_version",
                                    "missing_execution", "changed_execution"])
def test_current_checkpoint_requires_explicit_xpu_execution_provenance(study_models, field):
    saved = torch.load(study_models / "global" / "best.pt", weights_only=True)
    assert saved["xpu_rng_state"] is None
    assert "torch_xpu" in saved["provenance"]["versions"]
    assert saved["provenance"]["execution_protocol"]["version"] == "explicit-device-0.11"
    if field == "missing_rng":
        del saved["xpu_rng_state"]
    elif field == "missing_version":
        del saved["provenance"]["versions"]["torch_xpu"]
    elif field == "bad_version":
        saved["provenance"]["versions"]["torch_xpu"] = True
    elif field == "missing_execution":
        del saved["provenance"]["execution_protocol"]
    else:
        saved["provenance"]["execution_protocol"]["precision"] = "implicit autocast"
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


def _xpu_payload(study_models, architecture="oriented_surface"):
    """Serialized portable state only: never initialize or query the XPU backend."""
    saved = torch.load(study_models / architecture / "best.pt", weights_only=True)
    saved["config"]["device"] = "xpu:0"
    saved["device_info"] = {"device": "xpu:0", "name": "portable XPU fixture",
                             "capability": None}
    saved["cuda_rng_state"] = None
    saved["xpu_rng_state"] = torch.arange(16, dtype=torch.uint8)
    saved["provenance"]["versions"]["torch_xpu"] = "portable-fixture-runtime"
    return saved


@pytest.mark.parametrize("architecture", ["global", "surface", "oriented_surface"])
@pytest.mark.parametrize("device", ["xpu", "xpu:0"])
@pytest.mark.parametrize("schema", ["coordinate-study-0.11", "coordinate-study-0.12"])
def test_xpu_checkpoint_loads_and_predicts_on_cpu_without_any_accelerator_api(
    study_models, tmp_path, monkeypatch, architecture, device, schema
):
    saved = _xpu_payload(study_models, architecture)
    saved["config"]["device"] = device
    if schema == "coordinate-study-0.11":
        assert study.XPU_CHECKPOINT_SCHEMA == schema
        saved["schema_version"] = schema
        del saved["config"]["direction_training"]
        saved["provenance"] = study._provenance(schema)
        saved["provenance"]["versions"]["torch_xpu"] = "portable-fixture-runtime"
    checkpoint = tmp_path / "portable-xpu.pt"
    study._save_checkpoint(checkpoint, saved)
    before = checkpoint.read_bytes()
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        pytest.fail("portable inference attempted to initialize/query an accelerator")

    for backend in (torch.cuda, torch.xpu):
        for name in ("is_available", "is_initialized", "device_count", "current_device",
                     "get_device_name", "get_device_properties", "get_device_capability",
                     "get_rng_state", "get_rng_state_all", "set_rng_state", "set_rng_state_all",
                     "manual_seed", "manual_seed_all", "set_device", "init", "_lazy_init"):
            monkeypatch.setattr(backend, name, forbidden, raising=False)
    loaded = study.coordinate_checkpoint(checkpoint)
    assert loaded["config"]["device"] == device
    assert study._validate_saved(loaded).direction_training == "predicted"
    assert torch.equal(loaded["xpu_rng_state"], saved["xpu_rng_state"])
    predictor = load_coordinate_predictor(checkpoint)
    assert all(p.device.type == "cpu" and not p.requires_grad
               for p in predictor._model.parameters())
    np.testing.assert_array_equal(
        predictor.predict_map(np.zeros((47, 83, 3), dtype=np.uint8), (31, 41)),
        identity_map((31, 41)),
    )
    assert checkpoint.read_bytes() == before and torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("schema", ["coordinate-study-0.1", "coordinate-study-0.9",
                                    "coordinate-study-0.10", "coordinate-study-0.11"])
@pytest.mark.parametrize("claim", ["config", "protocol"])
def test_pre_teacher_checkpoints_reject_new_training_direction_claims(
        study_models, schema, claim):
    saved = _legacy_payload(study_models, "global", schema)
    if claim == "config":
        saved["config"]["direction_training"] = "predicted"
    else:
        saved["provenance"]["direction_training_protocol"] = (
            study._provenance()["direction_training_protocol"])
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


@pytest.mark.parametrize("fault", ["missing_config", "invalid_config", "missing_protocol",
                                    "changed_protocol"])
def test_teacher_checkpoint_requires_explicit_strict_training_protocol(study_models, fault):
    saved = torch.load(study_models / "oriented_surface" / "best.pt", weights_only=True)
    saved["config"]["direction_training"] = "teacher"
    assert study._validate_saved(saved).direction_training == "teacher"
    assert saved["provenance"]["direction_training_protocol"]["version"] == (
        "training-direction-0.12")
    if fault == "missing_config":
        del saved["config"]["direction_training"]
    elif fault == "invalid_config":
        saved["config"]["direction_training"] = True
    elif fault == "missing_protocol":
        del saved["provenance"]["direction_training_protocol"]
    else:
        saved["provenance"]["direction_training_protocol"]["evaluation"] = "use GT geometry"
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


def test_teacher_checkpoint_public_inference_uses_only_predicted_direction(
        study_models, tmp_path, monkeypatch):
    saved = torch.load(study_models / "oriented_surface" / "best.pt", weights_only=True)
    saved["config"]["direction_training"] = "teacher"
    saved["model_state"]["orientation_head.bias"][2] = 4.0
    checkpoint = tmp_path / "teacher.pt"
    study._save_checkpoint(checkpoint, saved)
    predictor = load_coordinate_predictor(checkpoint)
    original = predictor._model._forward_with_geometry_validated
    calls = []

    def forward(images, size, **kwargs):
        assert not predictor._model.training
        assert "teacher_direction" not in kwargs and "teacher_valid" not in kwargs
        calls.append(True)
        return original(images, size, **kwargs)

    monkeypatch.setattr(predictor._model, "_forward_with_geometry_validated", forward)
    grid = predictor.predict_map(np.zeros((47, 83, 3), dtype=np.uint8), (31, 41))
    np.testing.assert_array_equal(grid, -identity_map((31, 41)))
    assert calls == [True]
    assert all(parameter.grad is None for parameter in predictor._model.parameters())


@pytest.mark.parametrize("fault", ["missing", "none", "float", "bool", "scalar", "matrix",
                                    "short", "huge", "sparse", "cuda_state", "device", "index",
                                    "capability"])
def test_xpu_checkpoint_rejects_invalid_serialized_rng_and_device_metadata(study_models, fault):
    saved = _xpu_payload(study_models)
    if fault == "missing":
        del saved["xpu_rng_state"]
    elif fault == "none":
        saved["xpu_rng_state"] = None
    elif fault == "float":
        saved["xpu_rng_state"] = torch.zeros(16, dtype=torch.float32)
    elif fault == "bool":
        saved["xpu_rng_state"] = torch.zeros(16, dtype=torch.bool)
    elif fault == "scalar":
        saved["xpu_rng_state"] = torch.tensor(1, dtype=torch.uint8)
    elif fault == "matrix":
        saved["xpu_rng_state"] = torch.zeros((4, 4), dtype=torch.uint8)
    elif fault == "short":
        saved["xpu_rng_state"] = torch.zeros(15, dtype=torch.uint8)
    elif fault == "huge":
        saved["xpu_rng_state"] = torch.zeros(65_537, dtype=torch.uint8)
    elif fault == "sparse":
        saved["xpu_rng_state"] = saved["xpu_rng_state"].to_sparse()
    elif fault == "cuda_state":
        saved["cuda_rng_state"] = torch.zeros(16, dtype=torch.uint8)
    elif fault == "device":
        saved["device_info"]["device"] = "cuda:0"
    elif fault == "index":
        saved["device_info"]["device"] = "xpu:1"
    else:
        saved["device_info"]["capability"] = (1, 0)
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_non_xpu_checkpoint_cannot_contain_xpu_rng_state(study_models, device):
    saved = torch.load(study_models / "global" / "best.pt", weights_only=True)
    saved["config"]["device"] = device
    if device == "cpu":
        saved["device_info"] = {"device": "cpu", "name": "fixture", "capability": None}
        saved["cuda_rng_state"] = None
    saved["xpu_rng_state"] = torch.zeros(16, dtype=torch.uint8)
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


@pytest.mark.parametrize("field", [
    "topology_weight", "curvature_weight", "gradient_clip_norm",
    "page_weight", "boundary_weight", "differential_weight", "page_relative_weight",
    "orientation_weight", "rotation_probability", "direction_weight", "corner_weight",
])
def test_checkpoint_schema_does_not_silently_add_or_drop_objective_fields(study_models, field):
    current = torch.load(study_models / "global" / "best.pt", weights_only=True)
    del current["config"][field]
    with pytest.raises(FlattenerError):
        study._validate_saved(current)
    legacy = _legacy_payload(study_models, "global")
    legacy["config"][field] = 0.0
    with pytest.raises(FlattenerError):
        study._validate_saved(legacy)


@pytest.mark.parametrize("field", ["page_weight", "boundary_weight", "differential_weight"])
def test_regularized_legacy_schema_rejects_new_objective_fields(study_models, field):
    legacy = _legacy_payload(study_models, "global", "coordinate-study-0.2")
    legacy["config"][field] = 0.0
    with pytest.raises(FlattenerError):
        study._validate_saved(legacy)


@pytest.mark.parametrize("schema", [
    "coordinate-study-0.1", "coordinate-study-0.2", "coordinate-study-0.3",
])
@pytest.mark.parametrize("field,value", [
    ("include_paper_augmentation", False), ("initial_checkpoint_sha256", None),
])
def test_legacy_schema_cannot_silently_enable_new_data_or_initialization_contract(
    study_models, schema, field, value
):
    legacy = _legacy_payload(study_models, "global", schema)
    legacy["config"][field] = value
    with pytest.raises(FlattenerError):
        study._validate_saved(legacy)


@pytest.mark.parametrize("field", ["include_paper_augmentation", "initial_checkpoint_sha256"])
def test_current_schema_requires_explicit_data_and_initialization_contract(study_models, field):
    current = torch.load(study_models / "global" / "best.pt", weights_only=True)
    del current["config"][field]
    with pytest.raises(FlattenerError):
        study._validate_saved(current)


@pytest.mark.parametrize("architecture", ["global", "spatial", "page"])
def test_current_schema_loads_explicit_augmentation_and_initial_model_provenance(
    study_models, architecture, tmp_path
):
    saved = torch.load(study_models / architecture / "best.pt", weights_only=True)
    saved["config"].update(include_paper_augmentation=True, initial_checkpoint_sha256="b" * 64)
    config = study._validate_saved(saved)
    assert config.include_paper_augmentation is True
    assert config.initial_checkpoint_sha256 == "b" * 64
    checkpoint = tmp_path / "initialized.pt"
    study._save_checkpoint(checkpoint, saved)
    # Portable inference needs neither the source dataset nor the initialization file.
    predictor = load_coordinate_predictor(checkpoint)
    assert predictor.model_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    np.testing.assert_array_equal(
        predictor.predict_map(np.zeros((128, 128, 3), dtype=np.uint8), (128, 128)),
        identity_map((128, 128)),
    )


@pytest.mark.parametrize("field,value", [
    ("include_paper_augmentation", 0), ("include_paper_augmentation", 1),
    ("include_paper_augmentation", None), ("include_paper_augmentation", "true"),
    ("initial_checkpoint_sha256", True), ("initial_checkpoint_sha256", 0),
    ("initial_checkpoint_sha256", ""), ("initial_checkpoint_sha256", "a" * 63),
    ("initial_checkpoint_sha256", "a" * 65), ("initial_checkpoint_sha256", "A" * 64),
    ("initial_checkpoint_sha256", "g" * 64),
])
def test_current_schema_validates_augmentation_and_initial_model_provenance(
    study_models, field, value
):
    saved = torch.load(study_models / "global" / "best.pt", weights_only=True)
    saved["config"][field] = value
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


def test_page_schema_03_loads_with_existing_loss_weights_and_no_new_data_contract(
    study_models, tmp_path
):
    saved = _legacy_payload(study_models, "page", "coordinate-study-0.3")
    saved["config"].update(page_weight=0.5, boundary_weight=0.25, differential_weight=0.125)
    config = study._validate_saved(saved)
    assert (config.page_weight, config.boundary_weight, config.differential_weight) == (
        0.5, 0.25, 0.125
    )
    assert config.include_paper_augmentation is False and config.initial_checkpoint_sha256 is None
    checkpoint = tmp_path / "page-03.pt"
    study._save_checkpoint(checkpoint, saved)
    before = checkpoint.read_bytes()
    predictor = load_coordinate_predictor(checkpoint)
    grid = predictor.predict_map(np.zeros((128, 128, 3), dtype=np.uint8), (128, 128))
    np.testing.assert_array_equal(grid, identity_map((128, 128)))
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("field", ["topology_weight", "curvature_weight", "gradient_clip_norm"])
def test_regularized_legacy_schema_requires_its_original_objective_fields(study_models, field):
    legacy = _legacy_payload(study_models, "global", "coordinate-study-0.2")
    del legacy["config"][field]
    with pytest.raises(FlattenerError):
        study._validate_saved(legacy)


@pytest.mark.parametrize("schema", ["coordinate-study-0.1", "coordinate-study-0.2"])
def test_legacy_schema_cannot_claim_new_page_architecture(study_models, schema):
    saved = _legacy_payload(study_models, "page", schema)
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


@pytest.mark.parametrize("schema", [
    "coordinate-study-0.2", "coordinate-study-0.3", "coordinate-study-0.4",
    "coordinate-study-0.5", "coordinate-study-0.6", "coordinate-study-0.7",
    "coordinate-study-0.8", "coordinate-study-0.9",
])
def test_loss_protocol_cannot_be_swapped_between_regularized_schemas(study_models, schema):
    if schema == "coordinate-study-0.2":
        saved = _legacy_payload(study_models, "global", schema)
        other = "coordinate-study-0.3"
    elif schema in ("coordinate-study-0.3", "coordinate-study-0.4"):
        saved = _legacy_payload(study_models, "global", schema)
        other = "coordinate-study-0.2"
    elif schema in ("coordinate-study-0.5", "coordinate-study-0.6"):
        saved = _legacy_payload(study_models, "global", schema)
        other = "coordinate-study-0.7"
    else:
        saved = _legacy_payload(study_models, "global", schema)
        other = study.CHECKPOINT_SCHEMA
    saved["provenance"]["loss_protocol"] = study._provenance(other)["loss_protocol"]
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


@pytest.mark.parametrize("schema", [
    "coordinate-study-0.1", "coordinate-study-0.2", "coordinate-study-0.3",
    "coordinate-study-0.4", "coordinate-study-0.5",
])
def test_older_schemas_reject_even_zero_page_relative_weight(study_models, schema):
    saved = _legacy_payload(study_models, "global", schema)
    saved["config"]["page_relative_weight"] = 0.0
    with pytest.raises(FlattenerError, match="configuration fields"):
        study._validate_saved(saved)


@pytest.mark.parametrize("architecture", ["global", "spatial", "page", "projective", "surface"])
def test_analytic_schema_05_loads_and_warmstarts_06_without_changing_weights(
    study_models, architecture, tmp_path
):
    assert study.ANALYTIC_CHECKPOINT_SCHEMA == "coordinate-study-0.5"
    saved = _legacy_payload(study_models, architecture, study.ANALYTIC_CHECKPOINT_SCHEMA)
    config = study._validate_saved(saved)
    assert config.page_relative_weight == 0
    path = tmp_path / "old-05.pt"
    digest = study._save_checkpoint(path, saved)
    before = path.read_bytes()
    rng = torch.get_rng_state().clone()
    model = study.model_from_checkpoint(study.coordinate_checkpoint(path))
    assert study._same_state(model.state_dict(), saved["model_state"])
    upgraded, state = study._initialization(
        replace(config, page_relative_weight=0.25), path, None)
    assert upgraded.page_relative_weight == 0.25
    assert upgraded.initial_checkpoint_sha256 == digest
    assert study._same_state(state, saved["model_state"])
    assert torch.equal(rng, torch.get_rng_state())
    assert path.read_bytes() == before


@pytest.mark.parametrize("protocol", ["map_protocol", "initialization_protocol"])
def test_analytic_schema_05_still_strictly_validates_its_protocol(study_models, protocol):
    saved = _legacy_payload(study_models, "surface", study.ANALYTIC_CHECKPOINT_SCHEMA)
    saved["provenance"][protocol][next(iter(saved["provenance"][protocol]))] = "altered"
    with pytest.raises(FlattenerError, match="incompatible map or initialization protocol"):
        study._validate_saved(saved)


@pytest.mark.parametrize("schema", [
    "coordinate-study-0.1", "coordinate-study-0.2", "coordinate-study-0.3",
    "coordinate-study-0.4", "coordinate-study-0.5", "coordinate-study-0.6",
])
@pytest.mark.parametrize("field", ["orientation_weight", "rotation_probability"])
def test_pre_orientation_schemas_reject_even_zero_new_controls(study_models, schema, field):
    saved = _legacy_payload(study_models, "global", schema)
    saved["config"][field] = 0
    with pytest.raises(FlattenerError, match="configuration fields"):
        study._validate_saved(saved)


@pytest.mark.parametrize("architecture", ["global", "spatial", "page", "projective", "surface"])
def test_page_relative_schema_06_preserves_objective_and_warmstarts_07(
    study_models, architecture, tmp_path
):
    assert study.PAGE_RELATIVE_CHECKPOINT_SCHEMA == "coordinate-study-0.6"
    saved = _legacy_payload(study_models, architecture, study.PAGE_RELATIVE_CHECKPOINT_SCHEMA)
    saved["config"]["page_relative_weight"] = 0.25
    config = study._validate_saved(saved)
    assert config.page_relative_weight == 0.25
    assert config.orientation_weight == config.rotation_probability == 0
    path = tmp_path / "old-06.pt"
    digest = study._save_checkpoint(path, saved)
    before = path.read_bytes()
    upgraded, state = study._initialization(
        replace(config, orientation_weight=50, rotation_probability=0.5), path, None)
    assert upgraded.initial_checkpoint_sha256 == digest
    assert upgraded.orientation_weight == 50 and upgraded.rotation_probability == 0.5
    assert study._same_state(state, saved["model_state"])
    assert study._same_state(study.model_from_checkpoint(saved).state_dict(), state)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["missing", "changed", "legacy_claim"])
def test_rotation_protocol_is_strict_and_cannot_be_claimed_by_old_schema(study_models, change):
    saved = torch.load(study_models / "global" / "best.pt", weights_only=True)
    if change == "missing":
        del saved["provenance"]["augmentation_protocol"]
    elif change == "changed":
        saved["provenance"]["augmentation_protocol"]["target"] = "rotate output domain"
    else:
        old = _legacy_payload(study_models, "global", study.PAGE_RELATIVE_CHECKPOINT_SCHEMA)
        old["provenance"]["augmentation_protocol"] = saved["provenance"]["augmentation_protocol"]
        saved = old
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


def test_new_checkpoint_requires_loss_protocol_and_legacy_must_not_claim_it(study_models):
    current = torch.load(study_models / "global" / "best.pt", weights_only=True)
    protocol = current["provenance"].pop("loss_protocol")
    with pytest.raises(FlattenerError):
        study._validate_saved(current)
    legacy = _legacy_payload(study_models, "global")
    legacy["provenance"]["loss_protocol"] = protocol
    with pytest.raises(FlattenerError):
        study._validate_saved(legacy)


@pytest.mark.parametrize("architecture", ["global", "spatial", "page", "oriented_surface"])
def test_study_predictor_preserves_input_error_policy(study_models, architecture):
    predictor = load_coordinate_predictor(study_models / architecture / "best.pt")
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    returned = unwarp(image, predictor, (2, 2), config=RenderConfig(on_error="original"))
    assert returned.used_fallback and returned.reason_code == "output_smaller_than_grid"
    np.testing.assert_array_equal(returned.image, image)
    assert returned.image is not image
    assert returned.source_mask is returned.target_mask is returned.valid_mask is None


def test_missing_checkpoint_is_structured(tmp_path):
    with pytest.raises(FlattenerError) as caught:
        load_coordinate_predictor(tmp_path / "missing.pt")
    assert caught.value.reason_code == "checkpoint_read_failed"


def test_changed_checkpoint_is_rejected(study_models, tmp_path):
    source = study_models / "global" / "best.pt"
    destination = tmp_path / "bad.pt"
    destination.write_bytes(source.read_bytes() + b"tampered")
    Path(str(destination) + ".sha256").write_text(
        hashlib.sha256(source.read_bytes()).hexdigest() + "\n"
    )
    with pytest.raises(FlattenerError):
        load_coordinate_predictor(destination)


@pytest.mark.parametrize("architecture", ["global", "spatial", "page", "oriented_surface"])
@pytest.mark.parametrize(
    "mutation",
    ["schema", "extra", "shape", "dtype", "sparse", "nan", "epoch", "optimizer", "rng", "huge"],
)
def test_valid_hash_does_not_bypass_checkpoint_validation(
    study_models, architecture, mutation, tmp_path
):
    saved = torch.load(study_models / architecture / "best.pt", weights_only=True)
    key = next(iter(saved["model_state"]))
    if mutation == "schema":
        saved["schema_version"] = "learning-0.1"
    elif mutation == "extra":
        saved["config"]["unknown"] = True
    elif mutation == "shape":
        saved["model_state"][key] = torch.zeros(1)
    elif mutation == "dtype":
        saved["model_state"][key] = saved["model_state"][key].double()
    elif mutation == "sparse":
        saved["model_state"][key] = saved["model_state"][key].to_sparse()
    elif mutation == "nan":
        saved["model_state"][key].fill_(float("nan"))
    elif mutation == "epoch":
        saved["epoch"] = 1
    elif mutation == "optimizer":
        saved["optimizer_state"]["param_groups"][0]["lr"] = 3.0
    elif mutation == "rng":
        saved["torch_rng_state"] = torch.zeros(2, dtype=torch.float32)
    else:
        saved["config"]["model_config"]["hidden"] = 10**9
    destination = tmp_path / "bad.pt"
    torch.save(saved, destination)
    destination.with_suffix(".pt.sha256").write_text(
        hashlib.sha256(destination.read_bytes()).hexdigest() + "\n"
    )
    with pytest.raises(FlattenerError):
        load_coordinate_predictor(destination)


def test_checksum_sidecar_is_mandatory(study_models, tmp_path):
    path = tmp_path / "no-sidecar.pt"
    path.write_bytes((study_models / "global" / "best.pt").read_bytes())
    with pytest.raises(FlattenerError):
        study.coordinate_checkpoint(path)


def test_inspect_cli_does_not_require_cuda(study_models, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("inspecting a checkpoint must not access CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert study.main(["inspect", str(study_models / "spatial" / "best.pt")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "spatial" in json.dumps(result)


def test_loader_digest_refers_to_the_exact_bytes_it_deserialized(study_models, monkeypatch):
    source = study_models / "global" / "best.pt"
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    original_hash = study._sha256

    def no_checkpoint_reread(path):
        assert Path(path).suffix != ".pt", "loaded weights must not be attributed to a later file"
        return original_hash(path)

    monkeypatch.setattr(study, "_sha256", no_checkpoint_reread)
    assert load_coordinate_predictor(source).model_sha256 == expected


@pytest.mark.parametrize("kind", ["best", "last"])
def test_synthetic_epoch_one_state_is_validated_without_optimizer(study_models, kind):
    saved = torch.load(study_models / "global" / "best.pt", weights_only=True)
    saved.update(
        epoch=1,
        best_epoch=1,
        best_validation_epe_pixels=0.5,
        checkpoint_kind=kind,
        best_checkpoint_sha256=None if kind == "best" else "a" * 64,
        history=[{"epoch": 1, "train_loss": 1.0, "validation_epe_pixels": 0.5}],
    )
    # Synthetic moments validate the schema, not actual GPU optimizer behavior.
    saved["optimizer_state"]["state"] = {
        index: {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros_like(value),
            "exp_avg_sq": torch.zeros_like(value),
        }
        for index, value in enumerate(saved["model_state"].values())
    }
    assert study._validate_saved(saved).architecture == "global"
    saved["optimizer_state"]["state"][0]["exp_avg_sq"].fill_(-1)
    with pytest.raises(FlattenerError):
        study._validate_saved(saved)


def _synthetic_epoch(study_models, epoch):
    """Fabricate portable state to test disk transactions without training."""
    saved = torch.load(study_models / "global" / "best.pt", weights_only=True)
    saved.update(
        epoch=epoch,
        best_epoch=min(epoch, 1),
        best_validation_epe_pixels=0.5 if epoch else 1.0,
        history=[
            {"epoch": i, "train_loss": 1.0, "validation_epe_pixels": 0.5 if i == 1 else 0.75}
            for i in range(1, epoch + 1)
        ],
    )
    if epoch:
        saved["optimizer_state"]["state"] = {
            index: {
                "step": torch.tensor(float(epoch)),
                "exp_avg": torch.zeros_like(value),
                "exp_avg_sq": torch.zeros_like(value),
            }
            for index, value in enumerate(saved["model_state"].values())
        }
    return saved


@pytest.mark.parametrize("portable", [False, True] if os.name != "nt" else [True])
def test_checkpoint_generation_selects_best_and_preserves_previous_pair(
    study_models, tmp_path, monkeypatch, portable
):
    monkeypatch.setattr(study, "_PORTABLE_CHECKPOINTS", portable)
    initial_hash = study._commit_checkpoints(tmp_path, _synthetic_epoch(study_models, 0))
    initial_best = study._checkpoint_location(tmp_path / "best.pt", strict=True)
    initial_last = study._checkpoint_location(tmp_path / "last.pt", strict=True)
    assert initial_best.parent == initial_last.parent
    assert initial_best.parent.parent == tmp_path.resolve()
    assert initial_best.parent.name.startswith(".checkpoint-")
    if portable:
        pointer = tmp_path / "current.json"
        assert pointer.is_file() and not pointer.is_symlink()
        assert json.loads(pointer.read_text(encoding="ascii")) == {
            "generation": initial_best.parent.name}
        assert not (tmp_path / "best.pt").exists()
    else:
        assert (tmp_path / "current").is_symlink()
        assert os.readlink(tmp_path / "current") == initial_best.parent.name
    assert study.coordinate_checkpoint(initial_last)["best_checkpoint_sha256"] == initial_hash
    chosen_hash = study._commit_checkpoints(tmp_path, _synthetic_epoch(study_models, 1))
    assert chosen_hash != initial_hash
    chosen_best = study._checkpoint_location(tmp_path / "best.pt", strict=True)
    assert chosen_best.parent != initial_best.parent
    assert study.coordinate_checkpoint(tmp_path / "last.pt")["epoch"] == 1
    assert study._commit_checkpoints(tmp_path, _synthetic_epoch(study_models, 2)) == chosen_hash
    assert study.coordinate_checkpoint(tmp_path / "best.pt")["epoch"] == 1
    latest = study.coordinate_checkpoint(tmp_path / "last.pt")
    assert latest["epoch"] == 2 and latest["best_epoch"] == 1
    assert latest["best_checkpoint_sha256"] == chosen_hash
    assert hashlib.sha256(chosen_best.read_bytes()).hexdigest() == chosen_hash
    assert study.coordinate_checkpoint(initial_best)["epoch"] == 0
    assert study.coordinate_checkpoint(initial_last)["epoch"] == 0


@pytest.mark.parametrize("failed_write", [1, 2, 3, 4, "publish"])
@pytest.mark.parametrize("portable", [False, True] if os.name != "nt" else [True])
def test_interrupted_checkpoint_publication_keeps_old_pair_readable(
    study_models, tmp_path, monkeypatch, failed_write, portable
):
    monkeypatch.setattr(study, "_PORTABLE_CHECKPOINTS", portable)
    expected = study._commit_checkpoints(tmp_path, _synthetic_epoch(study_models, 0))
    old_current = study._checkpoint_location(tmp_path / "last.pt", strict=True).parent
    pointer = tmp_path / ("current.json" if portable else "current")
    old_pointer = pointer.read_bytes() if portable else os.readlink(pointer)
    atomic = study._atomic_bytes
    replace = study.os.replace
    calls = 0

    def fail_after_write(path, payload):
        nonlocal calls
        atomic(path, payload)
        calls += 1
        if calls == failed_write:
            raise OSError("injected interruption after partial checkpoint write")

    def fail_before_publish(source, destination):
        if failed_write == "publish" and Path(destination).name == pointer.name:
            raise OSError("injected interruption before generation publication")
        return replace(source, destination)

    monkeypatch.setattr(study, "_atomic_bytes", fail_after_write)
    monkeypatch.setattr(study.os, "replace", fail_before_publish)
    with pytest.raises(OSError, match="injected interruption"):
        study._commit_checkpoints(tmp_path, _synthetic_epoch(study_models, 1))
    assert study._checkpoint_location(tmp_path / "last.pt", strict=True).parent == old_current
    assert (pointer.read_bytes() if portable else os.readlink(pointer)) == old_pointer
    if portable:
        assert json.loads(pointer.read_text(encoding="ascii")) == {
            "generation": old_current.name}
    best = study.coordinate_checkpoint(tmp_path / "best.pt")
    last = study.coordinate_checkpoint(tmp_path / "last.pt")
    assert best["epoch"] == last["epoch"] == 0
    assert last["best_checkpoint_sha256"] == expected
    best_path = study._checkpoint_location(tmp_path / "best.pt", strict=True)
    assert best_path.parent == old_current
    assert hashlib.sha256(best_path.read_bytes()).hexdigest() == expected


def test_loading_alias_uses_one_generation_during_publication(study_models, tmp_path, monkeypatch):
    original_hash = study._commit_checkpoints(tmp_path, _synthetic_epoch(study_models, 0))
    next_epoch = _synthetic_epoch(study_models, 1)
    torch_load = study.torch.load
    swapped = False

    def publish_while_loading(*args, **kwargs):
        nonlocal swapped
        saved = torch_load(*args, **kwargs)
        if not swapped:
            swapped = True
            study._commit_checkpoints(tmp_path, next_epoch)
        return saved

    monkeypatch.setattr(study.torch, "load", publish_while_loading)
    predictor = load_coordinate_predictor(tmp_path / "best.pt")
    assert swapped
    assert predictor.model_sha256 == original_hash
    assert study.coordinate_checkpoint(tmp_path / "best.pt")["epoch"] == 1


@pytest.mark.parametrize("failure", [False, True])
def test_portable_checkpoint_pointer_preserves_complete_generations(
    study_models, tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(study, "_PORTABLE_CHECKPOINTS", True)
    digest = study._commit_checkpoints(tmp_path, _synthetic_epoch(study_models, 0))
    old = study._checkpoint_location(tmp_path / "last.pt", strict=True)
    assert not (tmp_path / "best.pt").exists()
    assert study._read_checkpoint(tmp_path / "best.pt")[1] == digest
    atomic = study._atomic_bytes

    def interrupted(path, payload):
        if path.name == "current.json":
            raise OSError("publication interrupted")
        return atomic(path, payload)

    if failure:
        monkeypatch.setattr(study, "_atomic_bytes", interrupted)
        with pytest.raises(OSError, match="publication interrupted"):
            study._commit_checkpoints(tmp_path, _synthetic_epoch(study_models, 1))
    else:
        study._commit_checkpoints(tmp_path, _synthetic_epoch(study_models, 1))
    assert study.coordinate_checkpoint(old)["epoch"] == 0
    expected_epoch = 0 if failure else 1
    assert study.coordinate_checkpoint(tmp_path / "last.pt")["epoch"] == expected_epoch
    assert study.coordinate_checkpoint(tmp_path / "best.pt")["epoch"] == expected_epoch
    loaded = load_coordinate_predictor(tmp_path / "best.pt")
    assert Path(loaded.checkpoint_path).parent.name.startswith(".checkpoint-")


@pytest.mark.parametrize("pointer", [
    "../escaped", "/abs/path", ".checkpoint-../../escape", "plain",
])
def test_portable_checkpoint_pointer_rejects_unowned_paths(tmp_path, pointer):
    (tmp_path / "current.json").write_text(json.dumps({"generation": pointer}))
    with pytest.raises(FlattenerError):
        study.coordinate_checkpoint(tmp_path / "last.pt")
