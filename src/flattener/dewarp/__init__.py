"""SOTA-track document dewarping: UVDoc-family grid regression with license tracks.

Independent of ``flattener.learning``. Modules: ``model`` (UVDocnet, MIT-derived),
``coords`` (grid conventions), ``data`` (sources, tracks, augmentation),
``composite`` (pages placed on new backgrounds with perspective), ``checkpoint``,
``train`` (``python -m flattener.dewarp.train``), ``predict``
(``python -m flattener.dewarp.predict``) and ``probe`` (the tilt probe,
``python -m flattener.dewarp.probe``).
"""

from .coords import (
    align_corners_to_flattener,
    grid_to_map,
    map_to_grid,
    syntheticdoc_to_flattener,
    uvdoc_dense_map,
)
from .data import TRACKS, AugmentConfig, DataSpec, SourceSpec, build_dataset, require_track
from .model import (
    UVDOC_CONFIG,
    DewarpModelConfig,
    UVDocNet,
    count_parameters,
    load_uvdoc_weights,
    natural_grid_size,
    uvdoc_model,
)

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
