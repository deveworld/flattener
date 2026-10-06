"""Default model downloads, offline reuse and local bundle overrides."""

import json

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from flattener import FlattenerError
from flattener.scan import download, pipeline


def _bundle(directory):
    directory.mkdir(exist_ok=True)
    for name in download.MODEL_FILES:
        (directory / name).write_bytes(b"weights")
    path = directory / "bundle.json"
    path.write_text(
        json.dumps(
            {
                "schema": "flattener-scan-bundle-1",
                "dewarp": {"path": "dewarp.pt", "tta": "flip"},
                "gate": {"path": "gate.pt"},
                "orient": {"path": "orient.pt"},
            }
        )
    )
    return path


def test_cached_models_load_offline_without_a_remote_request(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path)

    def snapshot(**kwargs):
        assert kwargs["local_files_only"] is True
        return str(tmp_path)

    monkeypatch.setattr(download, "snapshot_download", snapshot)
    assert download.download_bundle() == bundle


@pytest.mark.parametrize("partial", [False, True])
def test_downloads_missing_models_and_reuses_the_completed_cache(tmp_path, monkeypatch, partial):
    cached = partial
    if partial:
        (tmp_path / "bundle.json").write_text("{}")
    remote_requests = []

    def snapshot(**kwargs):
        nonlocal cached
        if kwargs.get("local_files_only"):
            if not cached:
                raise LocalEntryNotFoundError("no cached models")
            return str(tmp_path)
        remote_requests.append(kwargs)
        _bundle(tmp_path)
        cached = True
        return str(tmp_path)

    monkeypatch.setattr(download, "snapshot_download", snapshot)
    assert download.download_bundle() == tmp_path / "bundle.json"
    assert download.download_bundle() == tmp_path / "bundle.json"
    assert len(remote_requests) == 1
    request = remote_requests[0]
    assert request["repo_id"] == "DevWorld/flattener"
    assert request["revision"] == "1.0.0"
    assert request["token"] is False
    assert set(request["allow_patterns"]) == set(download.MODEL_FILES)


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


def test_incomplete_snapshot_is_not_used_as_a_bundle(tmp_path, monkeypatch):
    (tmp_path / "bundle.json").write_text("{}")
    monkeypatch.setattr(download, "snapshot_download", lambda **kwargs: str(tmp_path))
    with pytest.raises(FlattenerError, match="incomplete.*dewarp.pt"):
        download.download_bundle()


def test_scanner_loads_default_or_explicit_bundle(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path)
    requested = []
    loaded = []
    scanner = object()

    def default():
        requested.append(True)
        return bundle

    def from_files(weights, **kwargs):
        loaded.append((weights, kwargs))
        return scanner

    monkeypatch.setattr(pipeline, "download_bundle", default)
    monkeypatch.setattr(pipeline.Scanner, "from_files", from_files)
    assert pipeline.Scanner.load() is scanner
    assert pipeline.Scanner.load(bundle) is scanner
    assert requested == [True]
    for weights, options in loaded:
        assert weights == tmp_path / "dewarp.pt"
        assert options["gate"] == tmp_path / "gate.pt"
        assert options["orient"] == tmp_path / "orient.pt"
        assert options["tta"] == "flip"
