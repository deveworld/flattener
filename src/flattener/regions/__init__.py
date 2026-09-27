"""Opt-in page-region localization and guarded correction."""

from .geometry import PageRegion, RegionConfig, RegionEstimate, detect_region
from .inference import crop_page, estimate_page_region, unwarp_page
from .segmentation import SegmenterConfig

__all__ = [
    "PageRegion",
    "RegionConfig",
    "RegionEstimate",
    "SegmenterConfig",
    "crop_page",
    "detect_region",
    "estimate_page_region",
    "unwarp_page",
]
