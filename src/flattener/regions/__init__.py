"""Experimental, opt-in page-region localization and guarded correction."""

from .checkpoints import load_region_model
from .experiment_training import StudyTrainConfig, train_region_study
from .geometry import PageRegion, RegionConfig, RegionEstimate, detect_region
from .inference import crop_page, estimate_page_region, prepare_page_view, unwarp_page
from .segmentation import LoadedSegmenter, SegmenterConfig, load_segmenter
from .training import RegionTrainConfig, train_regions

__all__ = [
    "LoadedSegmenter",
    "PageRegion",
    "RegionConfig",
    "RegionEstimate",
    "RegionTrainConfig",
    "SegmenterConfig",
    "StudyTrainConfig",
    "crop_page",
    "detect_region",
    "estimate_page_region",
    "load_segmenter",
    "load_region_model",
    "prepare_page_view",
    "train_regions",
    "train_region_study",
    "unwarp_page",
]
