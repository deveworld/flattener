"""Reproducible, independently authored documents and analytic distortions."""

from .documents import Document, generate_document
from .geometry import GeometryMaps, WarpParameters, make_geometry, sample_warp
from .pipeline import (
    DatasetConfig,
    PhotometricConfig,
    SyntheticSample,
    generate_dataset,
    generate_sample,
    load_sample,
    validate_dataset,
)

__all__ = [
    "DatasetConfig",
    "Document",
    "GeometryMaps",
    "PhotometricConfig",
    "SyntheticSample",
    "WarpParameters",
    "generate_dataset",
    "generate_document",
    "generate_sample",
    "load_sample",
    "make_geometry",
    "sample_warp",
    "validate_dataset",
]
