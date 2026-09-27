"""Stage timing of scans (design section 3, T4).

``StageClock`` times stages with ``time.perf_counter_ns``. With ``cuda_sync`` it calls
``torch.cuda.synchronize()`` at the start and the end of each stage (only when torch is
already imported and CUDA is initialized), so queued GPU work is counted in the stage
that queued it; ``stage(name, sync=...)`` sets this per stage. A stage that raises is
recorded with ``ok=False`` before the exception propagates. ``add`` records a duration
measured elsewhere, such as the model load (cold start).

Summaries: ``median_of_repeats`` takes the per-stage median over repeated runs of one
image, ``summarize_stages`` the count, median and p95 of each stage over images, and
``summarize_by_bucket`` groups them by source megapixels (``megapixel_bucket``:
``<=2``, ``2-6``, ``6-12``, ``12-16``, ``>16``; above 16 MP the JPEG is draft decoded).
Percentiles interpolate linearly between order statistics (NumPy's default).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MEGAPIXEL_BUCKETS = ((2.0, "<=2"), (6.0, "2-6"), (12.0, "6-12"), (16.0, "12-16"))
TOP_BUCKET = ">16"


def cuda_synchronize() -> None:
    """Wait for queued CUDA work when torch is imported and CUDA is initialized."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


@dataclass(frozen=True, slots=True)
class StageRecord:
    name: str
    start_ns: int  # from the clock's start
    duration_ns: int
    synced: bool = False
    ok: bool = True

    @property
    def seconds(self) -> float:
        return self.duration_ns / 1e9

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_ns": self.start_ns,
            "duration_ns": self.duration_ns,
            "seconds": self.seconds,
            "synced": self.synced,
            "ok": self.ok,
        }


class StageClock:
    """Per-stage wall time of one scan (see the module docstring)."""

    def __init__(
        self,
        *,
        cuda_sync: bool = False,
        synchronize: Callable[[], None] = cuda_synchronize,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.cuda_sync = cuda_sync
        self._synchronize = synchronize
        self._clock = clock
        self._origin = clock()
        self._records: list[StageRecord] = []

    @property
    def records(self) -> tuple[StageRecord, ...]:
        return tuple(self._records)

    @contextmanager
    def stage(self, name: str, *, sync: bool | None = None) -> Iterator[None]:
        synced = self.cuda_sync if sync is None else sync
        if synced:
            self._synchronize()
        start = self._clock()
        ok = False
        try:
            yield
            ok = True
        finally:
            if synced:
                self._synchronize()
            end = self._clock()
            self._records.append(StageRecord(name, start - self._origin, end - start, synced, ok))

    def add(self, name: str, seconds: float) -> None:
        """Record a duration measured elsewhere (for example the model load)."""
        if not seconds >= 0:
            raise ValueError("seconds must be non-negative")
        now = self._clock() - self._origin
        self._records.append(StageRecord(name, now, round(seconds * 1e9)))

    def seconds(self) -> dict[str, float]:
        """Total seconds per stage name, in first-recorded order."""
        totals: dict[str, int] = {}
        for record in self._records:
            totals[record.name] = totals.get(record.name, 0) + record.duration_ns
        return {name: value / 1e9 for name, value in totals.items()}

    def elapsed(self) -> float:
        """Seconds since the clock was made."""
        return (self._clock() - self._origin) / 1e9

    def to_dict(self) -> dict[str, Any]:
        return {
            "seconds": self.seconds(),
            "total": self.elapsed(),
            "stages": [record.to_dict() for record in self._records],
            "cuda_sync": self.cuda_sync,
        }


def percentile(values: Iterable[float], q: float) -> float:
    """The ``q``-th percentile (0-100) with linear interpolation."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("percentile of no values")
    return float(np.percentile(array, q))


def summarize(values: Iterable[float]) -> dict[str, Any]:
    """Count, median, p95, min and max of some durations."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def median_of_repeats(runs: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Per-stage median over repeated runs of one image (stages in first-seen order)."""
    values: dict[str, list[float]] = {}
    for run in runs:
        for name, seconds in run.items():
            values.setdefault(name, []).append(float(seconds))
    return {name: float(np.median(items)) for name, items in values.items()}


def summarize_stages(rows: Iterable[Mapping[str, float]]) -> dict[str, dict[str, Any]]:
    """``summarize`` of each stage over rows of per-stage seconds (one row per image)."""
    values: dict[str, list[float]] = {}
    for row in rows:
        for name, seconds in row.items():
            values.setdefault(name, []).append(float(seconds))
    return {name: summarize(items) for name, items in values.items()}


def megapixel_bucket(height: int, width: int) -> str:
    """Bucket label of a source of (height, width) full-resolution pixels."""
    megapixels = height * width / 1e6
    for limit, label in MEGAPIXEL_BUCKETS:
        if megapixels <= limit:
            return label
    return TOP_BUCKET


def summarize_by_bucket(
    rows: Iterable[Mapping[str, Any]], *, timings: str = "timings", bucket: str = "bucket"
) -> dict[str, dict[str, dict[str, Any]]]:
    """``summarize_stages`` per megapixel bucket; each row holds its bucket label and a
    mapping of per-stage seconds. Buckets come in ``MEGAPIXEL_BUCKETS`` order."""
    groups: dict[str, list[Mapping[str, float]]] = {}
    for row in rows:
        groups.setdefault(row[bucket], []).append(row[timings])
    order = [label for _, label in MEGAPIXEL_BUCKETS] + [TOP_BUCKET]
    labels = sorted(groups, key=lambda label: (order.index(label) if label in order else 99, label))
    return {label: summarize_stages(groups[label]) for label in labels}


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append one JSON line (one timing row per image)."""
    line = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
