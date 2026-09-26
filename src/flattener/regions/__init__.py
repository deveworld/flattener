"""Opt-in page-region localization and guarded correction."""

from .geometry import PageRegion, RegionConfig, RegionEstimate, detect_region
from .inference import crop_page, estimate_page_region, prepare_page_view, unwarp_page
from .segmentation import LoadedSegmenter, SegmenterConfig, load_segmenter

__all__ = [
    "LoadedSegmenter",
    "PageRegion",
    "RegionConfig",
    "RegionEstimate",
    "SegmenterConfig",
    "crop_page",
    "detect_region",
    "estimate_page_region",
    "load_segmenter",
    "prepare_page_view",
    "unwarp_page",
]
