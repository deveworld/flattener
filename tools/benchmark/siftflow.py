# Derived from OARDoc (https://github.com/GreatV/oardoc, commit
# 9c45c0e783748554a83abea482924a2443294bef, file oardoc/metrics/siftflow.py),
# Copyright the OARDoc authors, licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0).
#
# Modified by the flattener authors: sources are built into a
# local cache with a process lock, the MEX shim accepts both the
# original DocUNet and the UVDoc-patched SIFT-flow trees, the pyramid uses the
# NumPy MATLAB emulation, built libraries can be reloaded in worker processes, the
# scan pyramid is reused across views, and the build also works with MSVC on
# Windows. See LICENSE.
#
# The coarse-to-fine driver (``SiftFlow._coarse_to_fine`` and ``_level_offset``), as
# in OARDoc, reimplements the control flow and parameter schedule of
# ``SIFTflowc2f.m`` from Ce Liu's SIFT-flow package. No original MATLAB or C++ file
# is included; its C++ is compiled from the user's own copy. The original package
# has no package-wide redistribution license; ``mexDiscreteFlow.m`` includes an
# "All rights reserved" notice. See LICENSE for the scope of this provenance issue.
"""SIFT flow (Ce Liu) compiled at run time from a user-supplied source tree.

The original distribution has no package-wide license. Each user supplies a copy:
the directory that contains ``mexDenseSIFT`` and ``mexDiscreteFlow`` (the
``SIFTflow`` folder of the DocUNet ``eval.zip`` or of UVDoc's ``eval_code``).
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import matlab

SIFTFLOW_SOURCE_ENV = "FLATTENER_SIFTFLOW_SOURCE"
CACHE_DIR_ENV = "FLATTENER_CACHE_DIR"
BINDING_VERSION = "flattener-siftflow-binding-2"
_WINDOWS = os.name == "nt"
# ``_LINUX_MAC`` selects the template ``__min``/``__max`` and ``strcasecmp`` code paths
# on every platform so Linux and Windows builds compile the same arithmetic.
COMPILE_FLAGS = ("/O2", "/D_LINUX_MAC") if _WINDOWS else ("-O3", "-D_LINUX_MAC")
REQUIRED_SOURCES = (
    "mexDenseSIFT/Image.h",
    "mexDenseSIFT/ImageFeature.h",
    "mexDenseSIFT/ImageProcessing.h",
    "mexDenseSIFT/project.h",
    "mexDiscreteFlow/BPFlow.cpp",
    "mexDiscreteFlow/BPFlow.h",
    "mexDiscreteFlow/Image.h",
    "mexDiscreteFlow/ImageProcessing.h",
    "mexDiscreteFlow/Stochastic.cpp",
    "mexDiscreteFlow/Stochastic.h",
    "mexDiscreteFlow/project.h",
)

# SIFTflowc2f parameters used by evalUnwarp.m / siftFlow.m in every DocUNet script.
SIFT_CELL_SIZE = 3
SIFT_GRID_SPACING = 1
FLOW_ALPHA = 2.0 * 255.0
FLOW_D = 40.0 * 255.0
FLOW_GAMMA = 0.005 * 255.0
FLOW_LEVELS = 4
FLOW_WSIZE = 2
FLOW_TOP_WSIZE = 10
FLOW_TOP_ITERATIONS = 60
FLOW_ITERATIONS = 30

_MEX_SHIM = r"""#pragma once
// Minimal stand-in for MATLAB's mex.h so the header-only image classes compile
// outside MATLAB. No MATLAB array function is ever called by the bindings.
#include <cstddef>
#include <cstdlib>
#ifdef _MSC_VER
// MSVC's <stdlib.h> defines __min/__max macros; project.h defines the templates
// used on Linux instead (enabled by _LINUX_MAC), so drop the macros first.
#undef __min
#undef __max
#include <string.h>
inline int strcasecmp(const char* first, const char* second) {
    return _stricmp(first, second);
}
#else
#include <strings.h>
inline int _strcmpi(const char* first, const char* second) {
    return strcasecmp(first, second);
}
#endif
using mwSize = std::size_t;
struct mxArray {};
enum mxClassID { mxDOUBLE_CLASS, mxSINGLE_CLASS, mxINT8_CLASS, mxUINT8_CLASS,
                 mxINT16_CLASS, mxUINT16_CLASS, mxINT32_CLASS, mxUINT32_CLASS };
constexpr int mxREAL = 0;
struct FlattenerMexDimensions {
    operator const int*() const { return nullptr; }
    operator const mwSize*() const { return nullptr; }
};
inline int mxGetNumberOfDimensions(const mxArray*) { return 0; }
inline FlattenerMexDimensions mxGetDimensions(const mxArray*) { return {}; }
inline void* mxGetData(const mxArray*) { return nullptr; }
inline bool mxIsClass(const mxArray*, const char*) { return false; }
template <class T>
inline mxArray* mxCreateNumericArray(int, const T*, mxClassID, int) { return nullptr; }
inline void mexErrMsgTxt(const char*) {}
"""

_DENSE_BINDING = r"""
#include <torch/extension.h>
#include <cstring>
#include "Image.h"
#include "ImageFeature.h"

torch::Tensor dense_sift(torch::Tensor input, int64_t cell_size, int64_t step_size) {
    TORCH_CHECK(input.device().is_cpu(), "dense_sift requires a CPU tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat64, "dense_sift requires float64");
    TORCH_CHECK(input.dim() == 2, "dense_sift requires an HW tensor");
    input = input.contiguous();
    const int height = input.size(0);
    const int width = input.size(1);
    DImage image(width, height, 1);
    std::memcpy(image.data(), input.data_ptr<double>(), sizeof(double) * width * height);
    UCImage descriptor;
    ImageFeature::imSIFT(image, descriptor, cell_size, step_size, true);
    auto output = torch::empty(
        {descriptor.height(), descriptor.width(), descriptor.nchannels()},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    std::memcpy(output.data_ptr<unsigned char>(), descriptor.data(), descriptor.nelements());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("dense_sift", &dense_sift);
}
"""

_FLOW_BINDING = r"""
#include <torch/extension.h>
#include <cstring>
#include "BPFlow.h"

torch::Tensor discrete_flow(
    torch::Tensor first, torch::Tensor second, double alpha, double distance,
    double gamma, int64_t iterations, int64_t hierarchy, int64_t window,
    torch::Tensor offset_x, torch::Tensor offset_y,
    torch::Tensor window_x, torch::Tensor window_y) {
    TORCH_CHECK(first.device().is_cpu() && second.device().is_cpu(), "CPU tensors required");
    TORCH_CHECK(first.scalar_type() == torch::kUInt8 && second.scalar_type() == torch::kUInt8,
                "features must be uint8");
    TORCH_CHECK(first.dim() == 3 && second.dim() == 3, "features must use HWC layout");
    TORCH_CHECK(first.size(2) == second.size(2), "feature channel counts differ");
    first = first.contiguous();
    second = second.contiguous();
    offset_x = offset_x.to(torch::kInt16).contiguous();
    offset_y = offset_y.to(torch::kInt16).contiguous();
    window_x = window_x.to(torch::kInt16).contiguous();
    window_y = window_y.to(torch::kInt16).contiguous();
    const int height = first.size(0);
    const int width = first.size(1);
    for (const auto& field : {offset_x, offset_y, window_x, window_y}) {
        TORCH_CHECK(field.dim() == 2 && field.size(0) == height && field.size(1) == width,
                    "offset and window fields must match the first feature image");
    }
    BPFlow solver;
    solver.setDisplay(false);
    solver.LoadImages(width, height, first.size(2), first.data_ptr<unsigned char>(),
                      second.size(1), second.size(0), second.data_ptr<unsigned char>());
    solver.setPara(alpha, distance);
    solver.setHomogeneousMRF(window);
    solver.LoadOffset(offset_x.data_ptr<short>(), offset_y.data_ptr<short>());
    solver.LoadWinSize(window_x.data_ptr<short>(), window_y.data_ptr<short>());
    solver.ComputeDataTerm();
    solver.ComputeRangeTerm(gamma);
    solver.MessagePassing(iterations, hierarchy);
    solver.ComputeVelocity();
    const DImage& flow = solver.flow();
    auto output = torch::empty({flow.height(), flow.width(), 2},
                               torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    std::memcpy(output.data_ptr<double>(), flow.data(),
                sizeof(double) * flow.height() * flow.width() * 2);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("discrete_flow", &discrete_flow);
}
"""


def resolve_source_dir(source_dir: str | os.PathLike[str] | None) -> Path:
    """Return the validated SIFT-flow source directory (argument, then environment)."""
    candidate = source_dir if source_dir is not None else os.environ.get(SIFTFLOW_SOURCE_ENV)
    if candidate is None or str(candidate) == "":
        raise FileNotFoundError(
            f"SIFT-flow source is required: pass --siftflow-source DIR or set "
            f"{SIFTFLOW_SOURCE_ENV} to the folder containing mexDenseSIFT and mexDiscreteFlow"
        )
    path = Path(candidate).expanduser().resolve()
    missing = [name for name in REQUIRED_SOURCES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path} is not a SIFT-flow source tree; missing {missing}")
    return path


def cache_root(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    configured = os.environ.get(CACHE_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "flattener"


@dataclass(frozen=True)
class SiftFlowBuild:
    """Picklable description of a compiled SIFT-flow backend."""

    source_dir: str
    build_key: str
    build_dir: str
    dense_module: str
    dense_library: str
    flow_module: str
    flow_library: str
    torch_version: str
    compile_flags: tuple[str, ...]
    compiler: str = ""

    def metadata(self) -> dict[str, Any]:
        record = asdict(self)
        record["compile_flags"] = list(self.compile_flags)
        record["binding_version"] = BINDING_VERSION
        return record


@contextlib.contextmanager
def _exclusive_lock(path: Path):
    """Inter-process lock on ``path`` (``flock`` on POSIX, ``msvcrt.locking`` on Windows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if _WINDOWS:
            import msvcrt
            import time

            stream.seek(0)
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.2)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _stage_sources(source_dir: Path, staging: Path) -> None:
    for folder in ("mexDenseSIFT", "mexDiscreteFlow"):
        destination = staging / folder
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted((source_dir / folder).iterdir()):
            if path.is_file() and path.suffix.lower() in {".h", ".cpp"}:
                shutil.copy2(path, destination / path.name)
        (destination / "mex.h").write_text(_MEX_SHIM, encoding="utf-8")
    (staging / "mexDenseSIFT" / "flattener_dense_binding.cpp").write_text(
        _DENSE_BINDING, encoding="utf-8"
    )
    (staging / "mexDiscreteFlow" / "flattener_flow_binding.cpp").write_text(
        _FLOW_BINDING, encoding="utf-8"
    )


def build_siftflow(
    source_dir: str | os.PathLike[str] | None = None,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    verbose: bool = False,
) -> SiftFlowBuild:
    """Compile (or reuse) the SIFT-flow torch extensions using PyTorch’s build cache."""
    import torch
    from torch.utils.cpp_extension import load

    source = resolve_source_dir(source_dir)
    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
    torch_version = torch.__version__.split("+")[0].replace(".", "_")
    build_key = f"v{BINDING_VERSION}_py{python_version}_torch{torch_version}"
    root = cache_root(cache_dir) / "torch_extensions"
    build_dir = root / f"siftflow-{build_key}"
    record_path = build_dir / "build.json"
    with _exclusive_lock(root / f"siftflow-{build_key}.lock"):
        build_dir.mkdir(parents=True, exist_ok=True)
        staging = build_dir / "src"
        _stage_sources(source, staging)
        modules = {}
        for kind, folder, sources in (
            ("dense", "mexDenseSIFT", ("flattener_dense_binding.cpp",)),
            (
                "flow",
                "mexDiscreteFlow",
                ("flattener_flow_binding.cpp", "BPFlow.cpp", "Stochastic.cpp"),
            ),
        ):
            name = f"flattener_siftflow_{kind}_{build_key}"
            output = build_dir / kind
            output.mkdir(exist_ok=True)
            module = load(
                name=name,
                sources=[str(staging / folder / item) for item in sources],
                extra_include_paths=[str(staging / folder)],
                extra_cflags=list(COMPILE_FLAGS),
                build_directory=str(output),
                with_cuda=False,
                verbose=verbose,
            )
            library = Path(getattr(module, "__file__", "") or output / f"{name}.so")
            if not library.is_file():
                raise RuntimeError(f"SIFT-flow build did not produce {library}")
            sys.modules.setdefault(name, module)
            modules[kind] = (name, library)
        build = SiftFlowBuild(
            source_dir=str(source),
            build_key=build_key,
            build_dir=str(build_dir),
            dense_module=modules["dense"][0],
            dense_library=str(modules["dense"][1]),
            flow_module=modules["flow"][0],
            flow_library=str(modules["flow"][1]),
            torch_version=torch.__version__,
            compile_flags=COMPILE_FLAGS,
            compiler=_compiler_version(),
        )
        _write_record(record_path, build.metadata())
        return build


def _compiler_version() -> str:
    """First banner line of the C++ compiler torch uses (``cl`` or ``$CXX``/``c++``)."""
    import subprocess

    command = ["cl"] if _WINDOWS else [os.environ.get("CXX", "c++"), "--version"]
    try:
        completed = subprocess.run(
            command, capture_output=True, stdin=subprocess.DEVNULL, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    # ``cl`` prints its version banner on stderr and a usage line on stdout.
    streams = (completed.stderr, completed.stdout) if _WINDOWS else (completed.stdout,)
    for line in b"".join(streams).decode("utf-8", "replace").splitlines():
        if line.strip():
            return line.strip()
    return "unknown"


def _write_record(path: Path, record: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=".build-", suffix=".json", delete=False, encoding="utf-8"
    ) as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _import_library(name: str, library: str) -> Any:
    import torch  # noqa: F401  (loads libtorch before the extension)

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, library)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load SIFT-flow extension {library}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


class SiftFlow:
    """Dense SIFT + coarse-to-fine SIFT flow exactly as ``siftFlow.m`` calls it."""

    def __init__(self, build: SiftFlowBuild) -> None:
        if not isinstance(build, SiftFlowBuild):
            raise TypeError("build must be a SiftFlowBuild")
        self.build = build
        self._dense = _import_library(build.dense_module, build.dense_library)
        self._flow = _import_library(build.flow_module, build.flow_library)
        self._reference_cache: tuple[np.ndarray, list[np.ndarray]] | None = None

    def dense_sift(self, image: np.ndarray) -> np.ndarray:
        """``mexDenseSIFT(image, 3, 1)`` for a float64 image in ``[0, 1]``."""
        import torch

        if image.dtype != np.float64 or image.ndim != 2:
            raise ValueError("dense SIFT input must be a float64 HW image")
        descriptor = self._dense.dense_sift(
            torch.from_numpy(np.ascontiguousarray(image)), SIFT_CELL_SIZE, SIFT_GRID_SPACING
        )
        return descriptor.numpy()

    def flow(self, reference: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(vx, vy)`` from ``siftFlow(reference, prediction)`` at half resolution."""
        for name, image in (("reference", reference), ("prediction", prediction)):
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 2:
                raise ValueError(f"{name} must be a grayscale uint8 image")
        if reference.shape != prediction.shape:
            raise ValueError(f"SIFT-flow shapes differ: {reference.shape} and {prediction.shape}")
        first = self._reference_pyramid(reference)
        second = self._pyramid(prediction)
        return self._coarse_to_fine(first, second)

    def _reference_pyramid(self, reference: np.ndarray) -> list[np.ndarray]:
        """The scan pyramid, reused while consecutive calls pass an identical scan.

        Both views of a document are usually scored against the same resized scan, so
        its blur, dense SIFT and 128-channel pyramid are built once. The cache holds
        one scan and is keyed by content, so results are identical to recomputing.
        """
        cached = self._reference_cache
        if (
            cached is not None
            and cached[0].shape == reference.shape
            and np.array_equal(cached[0], reference)
        ):
            return cached[1]
        pyramid = self._pyramid(reference)
        self._reference_cache = (reference.copy(), pyramid)
        return pyramid

    def _pyramid(self, image: np.ndarray) -> list[np.ndarray]:
        blurred = matlab.imresize(matlab.gaussian_filter(image, 7, 1.0), scale=0.5)
        levels = [self.dense_sift(blurred.astype(np.float64) / 255.0)]
        for _ in range(1, FLOW_LEVELS):
            smoothed = matlab.gaussian_filter(levels[-1], 5, 0.67)
            levels.append(matlab.imresize(smoothed, scale=0.5))
        return levels

    def _coarse_to_fine(
        self, first: list[np.ndarray], second: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        offsets = [
            _level_offset(level_one.shape[:2], level_two.shape[:2])
            for level_one, level_two in zip(first, second, strict=True)
        ]
        vx = vy = None
        for level in range(FLOW_LEVELS - 1, -1, -1):
            height, width = first[level].shape[:2]
            base_x, base_y = offsets[level]
            if vx is None:
                start_x, start_y = base_x, base_y
                window, iterations, hierarchy, mrf_window = (
                    FLOW_TOP_WSIZE,
                    FLOW_TOP_ITERATIONS,
                    2,
                    FLOW_TOP_WSIZE,
                )
            else:
                previous_x, previous_y = offsets[level + 1]
                upsampled_x = matlab.imresize(vx - previous_x, output_shape=(height, width))
                upsampled_y = matlab.imresize(vy - previous_y, output_shape=(height, width))
                start_x = matlab.matlab_round(base_x + upsampled_x * 2.0)
                start_y = matlab.matlab_round(base_y + upsampled_y * 2.0)
                window = FLOW_WSIZE + level
                iterations, hierarchy, mrf_window = (
                    FLOW_ITERATIONS,
                    FLOW_LEVELS - 1 - level,
                    FLOW_WSIZE,
                )
            windows = torch.full((height, width), window, dtype=torch.int16)
            flow = self._flow.discrete_flow(
                torch.from_numpy(np.ascontiguousarray(first[level])),
                torch.from_numpy(np.ascontiguousarray(second[level])),
                FLOW_ALPHA,
                FLOW_D,
                FLOW_GAMMA * 2.0**level,
                iterations,
                hierarchy,
                mrf_window,
                torch.from_numpy(np.asarray(start_x, dtype=np.int16)),
                torch.from_numpy(np.asarray(start_y, dtype=np.int16)),
                windows,
                windows,
            ).numpy()
            vx = np.ascontiguousarray(flow[:, :, 0])
            vy = np.ascontiguousarray(flow[:, :, 1])
        assert vx is not None and vy is not None
        return vx, vy


def _level_offset(first: tuple[int, int], second: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """``round((xx-1)*(width2-1)/(width-1)+1-xx)`` offsets from ``SIFTflowc2f.m``."""
    (height, width), (height2, width2) = first, second
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij"
    )
    offset_x = matlab.matlab_round(xx * (width2 - 1) / max(1, width - 1) - xx)
    offset_y = matlab.matlab_round(yy * (height2 - 1) / max(1, height - 1) - yy)
    return offset_x, offset_y


def load_siftflow(
    source_dir: str | os.PathLike[str] | None = None,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    verbose: bool = False,
) -> SiftFlow:
    """Build (if needed) and load the SIFT-flow backend from a user-supplied source tree."""
    return SiftFlow(build_siftflow(source_dir, cache_dir=cache_dir, verbose=verbose))
