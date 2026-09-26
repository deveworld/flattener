"""Explicit scene-schema routing, without changing the historical generator."""

import json
from pathlib import Path

from ..contracts import FlattenerError, Status


def scene_backend(directory):
    """Select a known schema from a bounded manifest; never decode test images."""
    path = Path(directory) / "manifest.json"
    try:
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("scene manifest exceeds 64 MiB")
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        schema = manifest.get("schema_version") if isinstance(manifest, dict) else None
        if schema == "scene-0.1":
            from . import scenes as backend
        elif schema == "scene-0.2":
            from . import scene_v2 as backend
        elif schema == "scene-0.3":
            from . import paired_scenes as backend
        else:
            raise ValueError("unsupported scene manifest schema")
        return backend
    except (OSError, ValueError, TypeError) as error:
        raise FlattenerError(Status.INPUT_ERROR, "invalid_scene_dataset", str(error)) from error


def validate_scene_dataset(directory):
    backend = scene_backend(directory)
    if getattr(backend, "SCENE_SCHEMA_VERSION", None) == "scene-0.1":
        return backend.validate_scene_dataset(directory)
    if getattr(backend, "SCENE_SCHEMA_VERSION", None) == "scene-0.3":
        return backend.validate_paired_scene_dataset(directory)
    return backend.validate_scene_v2_dataset(directory)


def load_scene(directory, record):
    return scene_loader(directory)(directory, record)


def scene_loader(directory):
    """Resolve once before a batch; the selected loader still verifies samples."""
    backend = scene_backend(directory)
    if getattr(backend, "SCENE_SCHEMA_VERSION", None) == "scene-0.3":
        return backend.load_paired_scene
    return getattr(backend, "load_scene_v2", None) or backend.load_scene
