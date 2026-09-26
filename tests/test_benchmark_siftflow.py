"""SIFT-flow plumbing. Building needs the user-supplied C++ sources, so the compiled
backend is only exercised (on two tiny images) when FLATTENER_SIFTFLOW_SOURCE is set."""

import hashlib
import os

import numpy as np
import pytest

from flattener.benchmark import metrics, siftflow


def test_missing_source_explains_how_to_supply_it(monkeypatch, tmp_path):
    monkeypatch.delenv(siftflow.SIFTFLOW_SOURCE_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="FLATTENER_SIFTFLOW_SOURCE"):
        siftflow.resolve_source_dir(None)
    with pytest.raises(FileNotFoundError, match="not a SIFT-flow source tree"):
        siftflow.resolve_source_dir(tmp_path)


def test_source_tree_hash_is_the_sorted_sha256sum_listing(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "x.h").write_bytes(b"header")
    (tmp_path / "a.cpp").write_bytes(b"code")
    (tmp_path / "B.txt").write_bytes(b"")
    listing = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in sorted(
            [("a.cpp", b"code"), ("b/x.h", b"header"), ("B.txt", b"")],
            key=lambda item: item[0].encode(),
        )
    )
    expected = hashlib.sha256(listing.encode()).hexdigest()
    assert siftflow.source_tree_sha256(tmp_path) == expected


def test_exclusive_lock_is_reentrant_across_calls(tmp_path):
    lock = tmp_path / "locks" / "build.lock"
    with siftflow._exclusive_lock(lock):
        assert lock.exists()
    with siftflow._exclusive_lock(lock):
        pass


def test_level_offsets_follow_siftflowc2f():
    offset_x, offset_y = siftflow._level_offset((3, 4), (5, 7))
    xx = np.arange(1, 5)
    expected_x = np.floor((xx - 1) * (7 - 1) / (4 - 1) + 1 - xx + 0.5)
    assert np.array_equal(offset_x[0], expected_x)
    assert offset_y[:, 0].tolist() == [0.0, 1.0, 2.0]


def test_compile_flags_define_linux_mac_on_every_platform():
    assert any(flag.endswith("D_LINUX_MAC") for flag in siftflow.COMPILE_FLAGS)
    assert "_MSC_VER" in siftflow._MEX_SHIM and "#undef __min" in siftflow._MEX_SHIM


def test_reference_pyramid_is_reused_only_for_an_identical_scan():
    backend = object.__new__(siftflow.SiftFlow)
    backend._reference_cache = None
    built = []

    def pyramid(image):
        built.append(image.copy())
        return [image.astype(np.float64) + len(built)]

    backend._pyramid = pyramid
    backend._coarse_to_fine = lambda first, second: (first[0], second[0])
    rng = np.random.default_rng(1)
    scan = rng.integers(0, 256, size=(6, 5), dtype=np.uint8)
    one, two = (rng.integers(0, 256, size=(6, 5), dtype=np.uint8) for _ in range(2))
    first = backend.flow(scan, one)
    assert len(built) == 2  # scan + prediction
    again = backend.flow(scan.copy(), two)
    assert len(built) == 3 and np.array_equal(first[0], again[0])
    scan[0, 0] ^= 1  # in-place change of the caller's array must not hit the cache
    changed = backend.flow(scan, one)
    assert len(built) == 5 and not np.array_equal(changed[0], first[0])


@pytest.mark.skipif(
    not os.environ.get(siftflow.SIFTFLOW_SOURCE_ENV),
    reason="set FLATTENER_SIFTFLOW_SOURCE to the SIFT-flow source tree to build it",
)
def test_compiled_flow_is_zero_for_identical_images_and_tracks_a_shift():
    backend = siftflow.load_siftflow()
    rng = np.random.default_rng(0)
    texture = rng.integers(0, 256, size=(48, 40), dtype=np.uint8)
    texture = np.kron(texture[::4, ::4], np.ones((4, 4), dtype=np.uint8))[:48, :40]
    vx, vy = backend.flow(texture, texture)
    assert vx.shape == vy.shape == (24, 20)
    assert metrics.local_distortion(vx, vy) == 0.0
    shifted = np.roll(texture, 4, axis=1)
    vx, vy = backend.flow(texture, shifted)
    assert np.median(vx[4:-4, 4:-4]) == pytest.approx(2.0, abs=0.5)
    assert np.median(np.abs(vy[4:-4, 4:-4])) <= 0.5
    backend._reference_cache = None  # the cached scan pyramid gives identical flows
    fresh_x, fresh_y = backend.flow(texture, shifted)
    assert np.array_equal(fresh_x, vx) and np.array_equal(fresh_y, vy)
