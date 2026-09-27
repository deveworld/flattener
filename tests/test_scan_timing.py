"""StageClock records and the median/p95 summaries of scan timings."""

import json

import numpy as np
import pytest

from flattener.scan import timing


class FakeClock:
    def __init__(self):
        self.now = 1_000

    def __call__(self):
        return self.now

    def advance(self, ns):
        self.now += ns


def test_stage_clock_records_each_stage():
    clock = FakeClock()
    stages = timing.StageClock(clock=clock)
    clock.advance(5)
    with stages.stage("read"):
        clock.advance(2_000_000)
    with stages.stage("infer"):
        clock.advance(190_000_000)
        with stages.stage("infer_pass1"):
            clock.advance(90_000_000)
    with stages.stage("infer"):
        clock.advance(10_000_000)
    stages.add("model_load", 0.5)
    names = [record.name for record in stages.records]
    assert names == ["read", "infer_pass1", "infer", "infer", "model_load"]
    assert stages.records[0].start_ns == 5 and stages.records[0].duration_ns == 2_000_000
    assert stages.seconds() == pytest.approx(
        {"read": 0.002, "infer_pass1": 0.09, "infer": 0.29, "model_load": 0.5}
    )
    assert list(stages.seconds()) == ["read", "infer_pass1", "infer", "model_load"]
    assert stages.elapsed() == pytest.approx(0.292000005)
    json.dumps(stages.to_dict(), allow_nan=False)
    with pytest.raises(ValueError):
        stages.add("model_load", -1.0)


def test_sync_runs_at_stage_start_and_end():
    calls = []
    clock = FakeClock()
    stages = timing.StageClock(
        cuda_sync=True, synchronize=lambda: calls.append(clock.now), clock=clock
    )
    with stages.stage("infer"):
        clock.advance(3)
    with stages.stage("save", sync=False):
        clock.advance(4)
    assert calls == [1_000, 1_003]
    assert [record.synced for record in stages.records] == [True, False]
    plain = timing.StageClock(synchronize=lambda: calls.append(-1), clock=clock)
    with plain.stage("detect", sync=True):
        pass
    assert calls[-2:] == [-1, -1]


def test_percentiles_and_summaries():
    values = list(range(1, 21))
    assert timing.percentile(values, 50) == 10.5
    assert timing.percentile(values, 95) == pytest.approx(19.05)
    summary = timing.summarize(values)
    assert summary == {
        "count": 20,
        "median": 10.5,
        "p95": pytest.approx(19.05),
        "min": 1,
        "max": 20,
    }
    assert timing.summarize([])["median"] is None
    with pytest.raises(ValueError):
        timing.percentile([], 50)
    repeats = [
        {"read": 0.3, "infer": 1.0},
        {"read": 0.1, "infer": 3.0, "save": 0.2},
        {"read": 0.2, "infer": 2.0},
    ]
    assert timing.median_of_repeats(repeats) == {"read": 0.2, "infer": 2.0, "save": 0.2}
    stages = timing.summarize_stages([{"read": 0.1, "total": 1.0}, {"read": 0.3, "total": 1.4}])
    assert stages["read"]["median"] == pytest.approx(0.2)
    assert stages["total"]["p95"] == pytest.approx(1.38)


def test_summary_by_bucket_and_jsonl(tmp_path):
    rows = [
        {"image": "a", "bucket": ">16", "timings": {"total": 1.2}},
        {"image": "b", "bucket": "6-12", "timings": {"total": 0.8}},
        {"image": "c", "bucket": "6-12", "timings": {"total": 1.0}},
        {"image": "d", "bucket": "<=2", "timings": {"total": 0.2}},
    ]
    summary = timing.summarize_by_bucket(rows)
    assert list(summary) == ["<=2", "6-12", ">16"]
    assert summary["6-12"]["total"]["median"] == pytest.approx(0.9)
    cached = [*rows, {"image": "c", "bucket": "6-12", "timings": {"total": 0.1}, "cached": True}]
    assert timing.summarize_by_bucket(cached) == summary
    assert timing.summarize_by_bucket(cached, include_cached=True)["6-12"]["total"]["count"] == 3
    path = tmp_path / "timing.jsonl"
    for row in rows:
        timing.append_jsonl(path, row)
    assert timing.read_jsonl(path) == rows
    assert path.read_text(encoding="utf-8").count("\n") == 4
    with pytest.raises(ValueError):
        timing.append_jsonl(path, {"total": np.nan})
