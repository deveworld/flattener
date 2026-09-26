"""Experimental CPU coordinate prediction, training, and evaluation.

The analytical renderer remains usable without model weights. Trained models
require an explicit output size and do not promise real-photo correction.
"""

from .inference import LoadedPredictor, load_predictor, unwarp
from .model import CoordinatePredictor, ModelConfig, dense_maps, preprocess_image
from .training import TrainConfig, train

__all__ = [
    "CoordinatePredictor",
    "LoadedPredictor",
    "ModelConfig",
    "TrainConfig",
    "dense_maps",
    "load_predictor",
    "preprocess_image",
    "train",
    "unwarp",
]
