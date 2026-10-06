"""Default model downloads, offline reuse and local bundle overrides."""

import json

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from flattener import FlattenerError, runtime
from flattener.scan import download, pipeline

BACKENDS = ("onnx", "torch")


def _bundle(directory, backend="torch"):
    directory.mkdir(exist_ok=True)
    files = download.MODEL_FILES[backend]
    for name in files[1:]:
        (directory / name).write_bytes(b"weights")
    path = directory / files[0]
    if backend == "torch":
        data = {
            "schema": "flattener-scan-bundle-1",
            "dewarp": {"path": "dewarp.pt", "tta": "flip"},
            "gate": {"path": "gate.pt"},
            "orient": {"path": "orient.pt"},
        }
    else:
        data = {
            "schema": "flattener-web-bundle-1",
            "models": {"dewarp": {"file": "dewarp.onnx"}},
        }
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("backend", BACKENDS)
def test_cached_models_load_offline_without_a_remote_request(tmp_path, monkeypatch, backend):
    bundle = _bundle(tmp_path, backend)

    def snapshot(**kwargs):
        assert kwargs["local_files_only"] is True
        return str(tmp_path)

    monkeypatch.setattr(download, "snapshot_download", snapshot)
    assert download.download_bundle(backend) == bundle


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("partial", [False, True])
def test_downloads_missing_models_and_reuses_the_completed_cache(
    tmp_path, monkeypatch, partial, backend
):
    files = download.MODEL_FILES[backend]
    cached = partial
    if partial:
        (tmp_path / files[0]).write_text("{}")
    remote_requests = []

    def snapshot(**kwargs):
        nonlocal cached
        if kwargs.get("local_files_only"):
            if not cached:
                raise LocalEntryNotFoundError("no cached models")
            return str(tmp_path)
        remote_requests.append(kwargs)
        _bundle(tmp_path, backend)
        cached = True
        return str(tmp_path)

    monkeypatch.setattr(download, "snapshot_download", snapshot)
    assert download.download_bundle(backend) == tmp_path / files[0]
    assert download.download_bundle(backend) == tmp_path / files[0]
    assert len(remote_requests) == 1
    request = remote_requests[0]
    assert request["repo_id"] == "DevWorld/flattener"
    assert request["revision"] == "1.1.0"
    assert request["token"] is False
    assert set(request["allow_patterns"]) == set(files)


def test_download_failure_explains_how_to_use_local_models(tmp_path, monkeypatch):
    def offline(**kwargs):
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("no cached models")
        raise OSError("connection unavailable")

    monkeypatch.setattr(download, "snapshot_download", offline)
    with pytest.raises(FlattenerError, match="--bundle PATH") as failure:
        download.download_bundle()
    assert failure.value.reason_code == "model_download_failed"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_incomplete_snapshot_is_not_used_as_a_bundle(tmp_path, monkeypatch, backend):
    files = download.MODEL_FILES[backend]
    (tmp_path / files[0]).write_text("{}")
    monkeypatch.setattr(download, "snapshot_download", lambda **kwargs: str(tmp_path))
    with pytest.raises(FlattenerError, match=f"incomplete.*{files[1]}"):
        download.download_bundle(backend)


def test_scanner_loads_default_or_explicit_bundle(tmp_path, monkeypatch):
    torch_bundle = _bundle(tmp_path / "torch", "torch")
    onnx_bundle = _bundle(tmp_path / "onnx", "onnx")
    requested = []
    loaded = []
    scanner = object()

    def default(backend):
        requested.append(backend)
        return onnx_bundle

    def from_files(weights, **kwargs):
        loaded.append(("torch", weights, kwargs))
        return scanner

    def from_onnx(manifest, base, **kwargs):
        loaded.append(("onnx", base, manifest))
        return scanner

    monkeypatch.setattr(runtime, "cuda_available", lambda: False)
    monkeypatch.setattr(pipeline, "download_bundle", default)
    monkeypatch.setattr(pipeline.Scanner, "from_files", from_files)
    monkeypatch.setattr(pipeline.Scanner, "from_onnx", from_onnx)
    assert pipeline.Scanner.load() is scanner  # no GPU: the ONNX models
    assert pipeline.Scanner.load(onnx_bundle) is scanner
    assert pipeline.Scanner.load(torch_bundle) is scanner
    assert requested == ["onnx"]
    assert [entry[0] for entry in loaded] == ["onnx", "onnx", "torch"]
    assert loaded[0][1] == tmp_path / "onnx"
    _, weights, options = loaded[2]
    assert weights == tmp_path / "torch" / "dewarp.pt"
    assert options["gate"] == tmp_path / "torch" / "gate.pt"
    assert options["orient"] == tmp_path / "torch" / "orient.pt"
    assert options["tta"] == "flip"
    assert options["device"] == "cpu"


def test_bundles_refuse_the_other_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "cuda_available", lambda: False)
    with pytest.raises(FlattenerError, match="torch backend"):
        pipeline.Scanner.load(_bundle(tmp_path / "torch", "torch"), backend="onnx")
    with pytest.raises(FlattenerError, match="onnx backend"):
        pipeline.Scanner.load(_bundle(tmp_path / "onnx", "onnx"), backend="torch")
    with pytest.raises(FlattenerError, match="cpu"):
        pipeline.Scanner.load(_bundle(tmp_path / "onnx", "onnx"), device="cuda")
