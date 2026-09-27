"""Scanner label provider: ``krreal.py draft --provider tools/krreal/scan_provider.py:propose``.

The draft quad and curved edges come from the final grid of a scan of the photo, through
``drafts.pages_from_grid``. A saved scan record (``<stem>.scan.json`` from
``python -m flattener.scan``) in ``scan-quarter/`` or ``scan-default/`` next to
``photos/`` is reused when it was made from the same bytes; ``scan-quarter/`` comes first
because there the page's up direction was given, so the grid's first row is the page top.
Without a saved record the scanner runs on the photo (release dewarp weights and page
gate v0, CPU, 2 threads, no orientation model). A photo the scanner gave back as the
original gets no geometry.
"""

from __future__ import annotations

import json
import os
import sys
from functools import cache
from pathlib import Path
from typing import Any

from drafts import pages_from_grid

REPO = Path(__file__).resolve().parents[2]
RECORD_FOLDERS = ("scan-quarter", "scan-default")
SCANNED = ("rectified", "rectified_with_warnings")
WEIGHTS = os.environ.get(
    "KRREAL_SCAN_WEIGHTS",
    str(Path.home() / ".cache/flattener-sota-20260926/vast-runs/R4/step_00140625.pt"),
)
GATE = os.environ.get(
    "KRREAL_SCAN_GATE",
    str(
        Path(os.environ.get("FLATTENER_CACHE_DIR", Path.home() / ".cache/flattener"))
        / "scan-weights/page-gate-v0.pt"
    ),
)


def saved_record(photo: Path, sha256: str) -> dict[str, Any] | None:
    """The first saved scan record of these exact bytes, or None."""
    root = photo.resolve().parent.parent
    for folder in RECORD_FOLDERS:
        path = root / folder / f"{photo.stem}.scan.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            if (record.get("source") or {}).get("file_sha256") == sha256:
                return record
    return None


@cache
def scanner() -> Any:
    """The scanner, loaded once per run."""
    source = REPO / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import torch

    from flattener.scan.pipeline import Scanner

    torch.set_num_threads(2)
    return Scanner.from_files(WEIGHTS, gate=GATE, device="cpu")


def propose(photo: Path, info: dict[str, Any]) -> dict[str, Any]:
    record = saved_record(photo, info["sha256"])
    if record is None:
        record = scanner().scan(photo).to_dict()
    plan = record.get("plan") or {}
    # Only a dewarped result has a grid; the original (and a perspective-only result,
    # which the default settings never give) is left for the labeler.
    if record.get("outcome") not in SCANNED or plan.get("kind") != "dewarp":
        return {}
    # The grid holds flattener values of the EXIF-applied photo; info["size"] is its
    # (width, height), so the corners come out in the label's pixel coordinates.
    return {"pages": [pages_from_grid(plan["grid"], image_size=tuple(info["size"]))]}
