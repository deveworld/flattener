"""SOTA-track document dewarping: UVDoc-family grid regression with license tracks.

Independent of ``flattener.learning``. Modules: ``model`` (UVDocnet, MIT-derived),
``coords`` (grid conventions), ``data`` (sources, tracks, augmentation),
``composite`` (pages placed on new backgrounds with perspective), ``checkpoint``,
``train`` (``python -m flattener.dewarp.train``), ``predict``
(``python -m flattener.dewarp.predict``) and ``probe`` (the tilt probe,
``python -m flattener.dewarp.probe``).
"""

import importlib

# Submodules load on first use, so inference without PyTorch never imports training code.
_EXPORTS = {
    "align_corners_to_flattener": "coords",
    "grid_to_map": "coords",
    "map_to_grid": "coords",
    "syntheticdoc_to_flattener": "coords",
    "uvdoc_dense_map": "coords",
    "TRACKS": "data",
    "AugmentConfig": "data",
    "DataSpec": "data",
    "SourceSpec": "data",
    "build_dataset": "data",
    "require_track": "data",
    "UVDOC_CONFIG": "config",
    "DewarpModelConfig": "config",
    "natural_grid_size": "config",
    "UVDocNet": "model",
    "count_parameters": "model",
    "load_uvdoc_weights": "model",
    "uvdoc_model": "model",
}


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


__all__ = [
    "TRACKS",
    "UVDOC_CONFIG",
    "AugmentConfig",
    "DataSpec",
    "DewarpModelConfig",
    "SourceSpec",
    "UVDocNet",
    "align_corners_to_flattener",
    "build_dataset",
    "count_parameters",
    "grid_to_map",
    "load_uvdoc_weights",
    "map_to_grid",
    "natural_grid_size",
    "require_track",
    "syntheticdoc_to_flattener",
    "uvdoc_dense_map",
    "uvdoc_model",
]
