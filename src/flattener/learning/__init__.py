"""CPU coordinate prediction for trained models.

Trained models need an explicit output size.
"""

from .inference import LoadedPredictor, load_predictor, unwarp
from .model import CoordinatePredictor, ModelConfig, dense_maps, preprocess_image

__all__ = [
    "CoordinatePredictor",
    "LoadedPredictor",
    "ModelConfig",
    "dense_maps",
    "load_predictor",
    "preprocess_image",
    "unwarp",
]
