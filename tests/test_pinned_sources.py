"""Six pinned sources keep their bytes.

Stored scores depend on the renderer and its coordinate and validation rules, the page
gate check depends on the segmentation preprocessing, and the orientation data records the
hash of ``synthetic/documents.py``. Change these files only together with a re-score.
The hashes were recorded before the test cleanup of 2026-09-26.
"""

import hashlib
from pathlib import Path

import flattener

PACKAGE = Path(flattener.__file__).resolve().parent
# sha256sum lines, as recorded.
PINNED = """\
c8c80bbe79f31a49b585e5ea6ef970b5aef3504b8782628e6e2670495da79f99  renderer.py
5f0f9742c69cd3bdbb2f2dd9b796c81de4e25f43b460ae9e6642e26ec1267dde  coordinates.py
36049346f8d0bb8d9900c9c32e81112ebab73eaba5dc1d6759d0fc022e8af05f  validation.py
77ce42b9bab81db55768d26736d6578eaf3e1be550e53fe7a87cdfc7ec901f94  contracts.py
4db411553b2c4c29ae68c7823e2c33e00d6bc3c2bb6522bf6a20dff98d40c259  regions/segmentation.py
2cf8808da603ce530eb6e6c92391255662239ab088f4ece119257c97d9357e54  synthetic/documents.py
"""


def test_pinned_sources_are_unchanged():
    pinned = {path: digest for digest, path in map(str.split, PINNED.splitlines())}
    assert len(pinned) == 6
    changed = sorted(
        name
        for name, digest in pinned.items()
        if hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() != digest
    )
    assert not changed, f"pinned sources changed: {changed}"
