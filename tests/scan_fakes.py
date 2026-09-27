"""Fake models for scanner tests (imported by the scan test modules).

The scene is a ramp photo: red holds x / W and green y / H of each pixel center, so a
bilinear sample tells which source point a view pixel shows. ``Oracle`` fits that affine
map on the middle of its input and returns the page grid as a dewarp model sees it: node
(0, 0) is the page corner at the top left of its view, whatever the page's text
direction. ``FakeGate`` rasterizes the page on the gate grid; ``FakeOrienter`` knows the
page's text-up direction and reports its quarter turn in any view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from flattener.dewarp.coords import center_positions, edge_positions
from flattener.dewarp.model import DewarpModelConfig
from flattener.dewarp.predict import Dewarper
from flattener.scan.locate import rasterize_polygon
from flattener.scan.orient import OrientResult

GRID = (9, 7)
CONFIG = DewarpModelConfig(input_size=(96, 72), grid_size=GRID, num_filter=4)
RELEASE_INFO = {"sha256": "fake-dewarp", "kind": "checkpoint", "step": 20000, "track": "release"}


def rotation(angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


class Page:
    """A page whose position u in [-1, 1]^2 (page frame, text up at -v) lies at
    ``center + R(angle) (u * size / 2)`` source pixels. ``angle`` includes quarter
    turns: 90 puts the page's top edge on the right."""

    def __init__(self, center, size, angle_deg=0.0):
        self.center = np.asarray(center, float)
        self.size = np.asarray(size, float)
        self.angle = float(angle_deg)
        self.matrix = rotation(angle_deg) @ np.diag(self.size / 2)

    def pixels(self, uv):
        return np.asarray(uv, float) @ self.matrix.T + self.center

    def grid(self, grid_size=GRID, source_size=None):
        """The true grid in page order: flattener values at edge-aligned nodes."""
        u = np.stack(np.meshgrid(edge_positions(grid_size[1]), edge_positions(grid_size[0])), -1)
        pixels = self.pixels(u)
        if source_size is None:
            return pixels
        height, width = source_size
        return pixels / np.array([width, height]) * 2 - 1

    def corners(self):
        return self.pixels([[-1, -1], [1, -1], [1, 1], [-1, 1]])

    def quarter(self):
        """Clockwise quarter turns of the text-up direction in the photo."""
        return round(self.angle / 90.0) % 4


def ramp_scene(source_size, page=None, *, background=40):
    """The ramp photo; the blue channel is 200 on the page and ``background`` elsewhere."""
    height, width = source_size
    rows, cols = np.mgrid[0:height, 0:width]
    red = (cols + 0.5) / width * 255
    green = (rows + 0.5) / height * 255
    blue = np.full(source_size, background, float)
    if page is not None:
        values = page.corners() / np.array([width, height]) * 2 - 1
        blue[rasterize_polygon(values, source_size)] = 200
    return np.stack([red, green, blue], -1).round().astype(np.uint8)


def fit_view(tensor_or_image, source_size):
    """Affine map (matrix, offset) from view values (flattener) to source pixels, fitted
    on the middle of a ramp view (tensor (1, 3, h, w) in [0, 1] or uint8 image)."""
    if isinstance(tensor_or_image, torch.Tensor):
        red = tensor_or_image[0, 0].double().cpu().numpy() * 255
        green = tensor_or_image[0, 1].double().cpu().numpy() * 255
    else:
        red = tensor_or_image[..., 0].astype(float)
        green = tensor_or_image[..., 1].astype(float)
    height, width = source_size
    rows, cols = red.shape
    qx, qy = np.meshgrid(center_positions(cols), center_positions(rows))
    middle = (np.abs(qx) < 0.25) & (np.abs(qy) < 0.25)
    design = np.stack([qx[middle], qy[middle], np.ones(middle.sum())], 1)
    fit_x = np.linalg.lstsq(design, red[middle] / 255 * width, rcond=None)[0]
    fit_y = np.linalg.lstsq(design, green[middle] / 255 * height, rcond=None)[0]
    return np.array([fit_x[:2], fit_y[:2]]), np.array([fit_x[2], fit_y[2]])


class Oracle(torch.nn.Module):
    """A perfect dewarp model of ``page`` on a ramp photo of ``source_size`` that orders
    its nodes by the view (see the module docstring). ``hooks`` maps a call number (from
    0) to a function that changes the grid of that call, for failure injection; ``always``
    changes every call."""

    def __init__(self, page, source_size, grid_size=GRID):
        super().__init__()
        self.page, self.source_size, self.grid_size = page, source_size, grid_size
        self.calls = 0
        self.hooks = {}
        self.always = None
        self.seen = []

    def forward(self, x):
        matrix, offset = fit_view(x, self.source_size)
        inverse = np.linalg.inv(matrix)

        def to_view(points):
            return (np.asarray(points) - offset) @ inverse.T

        axis = to_view(self.page.pixels([1, 0])) - to_view(self.page.pixels([-1, 0]))
        turn = round(math.degrees(math.atan2(axis[1], axis[0])) / 90.0) % 4
        gh, gw = self.grid_size
        nodes = np.stack(np.meshgrid(edge_positions(gw), edge_positions(gh)), -1)
        page_uv = nodes @ rotation(-90.0 * turn).T  # view-frame nodes -> page positions
        grid = to_view(self.page.pixels(page_uv))
        hook = self.hooks.get(self.calls, self.always)
        if hook is not None:
            grid = hook(grid)
        self.calls += 1
        self.seen.append(turn)
        tensor = torch.from_numpy(grid.transpose(2, 0, 1).astype(np.float32))[None]
        return tensor, None


def oracle_dewarper(page, source_size, info=None):
    model = Oracle(page, source_size)
    return Dewarper(model, CONFIG, torch.device("cpu"), dict(info or RELEASE_INFO))


def twist(grid):
    """Swap two interior node columns: cells fold, the outline stays."""
    twisted = grid.copy()
    twisted[1:-1, 2], twisted[1:-1, 4] = grid[1:-1, 4], grid[1:-1, 2]
    return twisted


def shrink(factor):
    def change(grid):
        center = grid.mean(axis=(0, 1))
        return center + (grid - center) * factor

    return change


@dataclass
class _GateConfig:
    input_size: tuple = (48, 36)


@dataclass
class FakeGate:
    """A page gate that rasterizes pages (``probability`` inside, ``low`` outside)."""

    pages: list
    source_size: tuple
    probability: float = 0.97
    low: float = 0.02
    config: _GateConfig = field(default_factory=_GateConfig)
    model_sha256: str = "fake-gate"
    model_id: str = "fake-gate"
    load_seconds: float = 0.0
    calls: int = 0
    metadata: dict = field(
        default_factory=lambda: {
            "model_version": "fake",
            "parameter_count": 1,
            "provenance": {
                "track": "release",
                "lineage": [{"kind": "scratch"}],
                "data": [{"kind": "authored", "license": "Apache-2.0 AND OFL-1.1"}],
            },
        }
    )

    def predict_probability(self, image):
        self.calls += 1
        height, width = self.source_size
        grid = np.full(self.config.input_size, self.low, np.float32)
        for page in self.pages:
            values = page.corners() / np.array([width, height]) * 2 - 1
            grid[rasterize_polygon(values, self.config.input_size)] = self.probability
        return grid


class FakeOrienter:
    """Reports the quarter turn of ``page``'s text-up direction in a view: from the view
    geometry when one is given (the pre-dewarp view), else from the ramp colors of the
    image (the flat preview). ``pre`` and ``post`` set False make that side uncertain."""

    def __init__(self, page, source_size, *, pre=True, post=True, confidence=0.97):
        self.page, self.source_size = page, source_size
        self.pre, self.post, self.confidence = pre, post, confidence
        self.temperature, self.threshold = 1.0, 0.9
        self.info = {"sha256": "fake-orient", "model_id": "fake", "shippable": True}
        self.calls = []

    def predict(self, image, geometry=None):
        up = self.page.pixels([0, -1]) - self.page.pixels([0, 1])
        if geometry is not None:
            matrix = geometry.matrix[:, :2]
            accepted = self.pre
        else:
            matrix, _ = fit_view(image, self.source_size)
            accepted = self.post
        view_up = np.linalg.solve(matrix, up)
        angle = math.degrees(math.atan2(view_up[0], -view_up[1]))  # clockwise from up
        k = round(angle / 90.0) % 4
        self.calls.append(("pre" if geometry is not None else "post", k))
        probabilities = [(1 - self.confidence) / 3] * 4
        probabilities[k] = self.confidence
        if not accepted:
            probabilities = [0.4, 0.2, 0.2, 0.2]
            k = 0
        order = sorted(probabilities, reverse=True)
        return OrientResult(
            k=k,
            probabilities=probabilities,
            confidence=order[0],
            margin=order[0] - order[1],
            accepted=accepted,
            reason_code=None if accepted else "orientation_uncertain",
            temperature=1.0,
            threshold=0.9,
        )
