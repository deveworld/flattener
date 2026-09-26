"""Compatibility import for the consolidated coordinate-study loader.

Loading a study checkpoint does not promote it or certify photo quality.
"""

from .coordinate_study import load_coordinate_predictor

__all__ = ["load_coordinate_predictor"]
