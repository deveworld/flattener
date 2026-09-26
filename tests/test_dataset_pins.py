"""Generator sources still match the hashes recorded in outputs/dataset."""

import hashlib
import json
from pathlib import Path

import pytest

import flattener
from flattener.learning.scene_data import _GENERATOR_SOURCES, _PAPER_SOURCE

MANIFEST = Path(__file__).resolve().parents[1] / "outputs" / "dataset" / "manifest.json"
PACKAGE = Path(flattener.__file__).resolve().parent

pytestmark = pytest.mark.skipif(not MANIFEST.is_file(), reason="outputs/dataset is not present")


def _sha256(name):
    return hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()


def test_generator_sources_match_dataset_manifest():
    manifest = json.loads(MANIFEST.read_bytes())
    assert len(_GENERATOR_SOURCES) == 11
    assert manifest["generator_code_sha256"] == {name: _sha256(name) for name in _GENERATOR_SOURCES}
    paper = manifest["collection"]["paper_augmentation"]
    assert paper["generator_code_sha256"] == {_PAPER_SOURCE: _sha256(_PAPER_SOURCE)}
