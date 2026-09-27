"""The 14 dataset-pinned sources keep their bytes.

Built datasets, caches and trained weights depend on these files, so new behaviour goes
into other modules. The hashes were recorded before the test cleanup of 2026-09-26
(pinned-hashes-before-cleanup.txt). The first two tests need no built dataset; the last
one also checks the hashes recorded in outputs/dataset when it is present.
"""

import hashlib
import json
from pathlib import Path

import pytest

import flattener
from flattener.learning.scene_data import _DATA_SOURCES, _GENERATOR_SOURCES, _PAPER_SOURCE

PACKAGE = Path(flattener.__file__).resolve().parent
MANIFEST = Path(__file__).resolve().parents[1] / "outputs" / "dataset" / "manifest.json"
# sha256sum lines, as recorded.
PINNED = """\
1d52e3be1a99560c7e6d111431e221219e5b92e67b35a3a5d2cd8d6505d5e7e7  synthetic/paired_scenes.py
5b7d1b6834656a24d4cfc822189ce32c4be576ec9252b1248db10bd9c83ccd01  synthetic/scene_v2.py
0aefacf846ec3c1a4c9cf62351dc74d2d3f19f7276b801c93cf5c159f45a28b2  synthetic/scenes.py
2cf8808da603ce530eb6e6c92391255662239ab088f4ece119257c97d9357e54  synthetic/documents.py
0be1f353a6131a44ab2f33e0afb08df1cdece3d3e705913fe78ef5cb7815f942  synthetic/geometry.py
665a10b569743e61a490727d57620923a095ac7de7580e6e61e77419d0982a05  synthetic/pipeline.py
c8c80bbe79f31a49b585e5ea6ef970b5aef3504b8782628e6e2670495da79f99  renderer.py
5f0f9742c69cd3bdbb2f2dd9b796c81de4e25f43b460ae9e6642e26ec1267dde  coordinates.py
36049346f8d0bb8d9900c9c32e81112ebab73eaba5dc1d6759d0fc022e8af05f  validation.py
77ce42b9bab81db55768d26736d6578eaf3e1be550e53fe7a87cdfc7ec901f94  contracts.py
4db411553b2c4c29ae68c7823e2c33e00d6bc3c2bb6522bf6a20dff98d40c259  regions/segmentation.py
b2bbd6742717f5bc9840f9c1a6a95e497e31df3b3be34847aeb149edf624a75b  synthetic/paper.py
b1cc2da2c34f65f3a4c60832e000d45a319f508126a1c60f910ae285889b17d9  learning/model.py
0861fdabd8b560d31de061f631a17f43f4f0fb184c4079750206d25daee42309  learning/scene_data.py
"""
PINNED_SHA256 = {path: digest for digest, path in map(str.split, PINNED.splitlines())}


def _sha256(name):
    return hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()


def test_pinned_list_is_the_dataset_source_list():
    assert len(PINNED_SHA256) == 14
    assert set(PINNED_SHA256) == {*_DATA_SOURCES, _PAPER_SOURCE}


def test_pinned_sources_are_unchanged():
    changed = sorted(name for name, digest in PINNED_SHA256.items() if _sha256(name) != digest)
    assert not changed, f"dataset-pinned files changed: {changed}"


@pytest.mark.skipif(not MANIFEST.is_file(), reason="outputs/dataset is not present")
def test_generator_sources_match_dataset_manifest():
    manifest = json.loads(MANIFEST.read_bytes())
    assert len(_GENERATOR_SOURCES) == 11
    assert manifest["generator_code_sha256"] == {name: _sha256(name) for name in _GENERATOR_SOURCES}
    paper = manifest["collection"]["paper_augmentation"]
    assert paper["generator_code_sha256"] == {_PAPER_SOURCE: _sha256(_PAPER_SOURCE)}
