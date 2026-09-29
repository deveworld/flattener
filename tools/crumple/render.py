"""Crumpled and creased pages: a height field over a flat page, seen by a pinhole camera.

``render(texture, rng, out_hw)`` returns the photo (uint8 (H, W, 3)) and its exact
forward map: for each point of a center-aligned (``MAP_HW``) grid over the flat page, the
flattener value of where it lands in the photo. That map is the dewarp target
(``backward_map_convention: "flattener"`` in the converted data layout).

Surface (page width 1, height ``aspect``): the sum of
- creases: line segments with a V profile ``a * (sqrt(s^2 + w^2) - w)`` across them
  (``s`` the signed distance, ``w`` the rounding), faded out past their ends;
- folds: a few creases across the whole page with larger slopes (letters, maps);
- bends: a quadratic curl and low-frequency waves.
The camera turns the page by a tilt of up to ``MAX_TILT_DEG`` about a random axis in its
plane and a small roll, and looks at it from a distance that frames it at a random
scale. Shading is Lambertian under a light from the camera side, plus ambient, with a
darker band in folds that face away. The page is sampled ``OVERSAMPLE`` times denser
than it covers the photo and splatted bilinearly; the few holes left take the
background. Nothing occludes (slopes stay moderate), so the forward map is exact.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

MAP_HW = (177, 121)
MAX_TILT_DEG = 32.0
OVERSAMPLE = 2.2


def _creases(rng: np.random.Generator, aspect: float) -> list[tuple]:
    creases = []
    for _ in range(int(rng.integers(3, 26))):
        center = rng.uniform([0.0, 0.0], [1.0, aspect])
        angle = rng.uniform(0, math.pi)
        half = rng.uniform(0.08, 0.7) * math.hypot(1.0, aspect)
        slope = float(np.clip(rng.normal(0, 0.07), -0.2, 0.2))
        width = rng.uniform(0.002, 0.02)
        creases.append((center, angle, half, slope, width))
    for _ in range(int(rng.choice(3, p=[0.45, 0.35, 0.2]))):  # folds across the page
        center = rng.uniform([0.2, 0.2 * aspect], [0.8, 0.8 * aspect])
        angle = rng.choice([0.0, math.pi / 2]) + rng.normal(0, 0.08)
        slope = float(rng.choice([-1, 1]) * rng.uniform(0.05, 0.3))
        creases.append((center, angle, 10.0, slope, rng.uniform(0.002, 0.01)))
    return creases


def height_field(
    rng: np.random.Generator, x: torch.Tensor, y: torch.Tensor, aspect: float
) -> torch.Tensor:
    """Heights (page-width units) at page points ``x`` in [0, 1], ``y`` in [0, aspect]."""
    z = torch.zeros_like(x)
    for center, angle, half, slope, width in _creases(rng, aspect):
        direction = torch.tensor([math.cos(angle), math.sin(angle)], dtype=x.dtype)
        dx, dy = x - float(center[0]), y - float(center[1])
        along = dx * direction[0] + dy * direction[1]
        across = -dx * direction[1] + dy * direction[0]
        fade = torch.sigmoid((half - along.abs()) / (0.08 * half + 1e-3))
        z = z + slope * (torch.sqrt(across**2 + width**2) - width) * fade
    curl = rng.normal(0, 0.12, size=2)
    z = z + float(curl[0]) * (x - 0.5) ** 2 + float(curl[1]) * (y - aspect / 2) ** 2
    for _ in range(int(rng.integers(0, 3))):
        k = rng.uniform(2, 8) * 2 * math.pi
        phase, theta = rng.uniform(0, 2 * math.pi), rng.uniform(0, math.pi)
        amplitude = rng.uniform(0.001, 0.005)
        z = z + amplitude * torch.sin(k * (x * math.cos(theta) + y * math.sin(theta)) + phase)
    return z - z.mean()


def _rotation(tilt: float, axis_angle: float, roll: float) -> torch.Tensor:
    axis = np.array([math.cos(axis_angle), math.sin(axis_angle), 0.0])
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    tilt_matrix = np.eye(3) + math.sin(tilt) * k + (1 - math.cos(tilt)) * k @ k
    c, s = math.cos(roll), math.sin(roll)
    roll_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return torch.tensor(roll_matrix @ tilt_matrix, dtype=torch.float32)


def render(
    texture: NDArray[np.uint8],
    rng: np.random.Generator,
    out_hw: tuple[int, int],
    background: NDArray[np.uint8] | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """(photo, forward map) of ``texture`` (the flat page) crumpled and photographed."""
    out_h, out_w = out_hw
    tex_h, tex_w = texture.shape[:2]
    aspect = tex_h / tex_w
    fill = rng.uniform(0.55, 0.95)  # of the frame's limiting side
    # Page sampling density: enough samples per output pixel the page covers.
    page_px = fill * min(out_w, out_h / aspect)
    cols = int(page_px * OVERSAMPLE)
    rows = int(cols * aspect)
    xs = torch.linspace(0, 1, cols)
    ys = torch.linspace(0, aspect, rows)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    z = height_field(rng, x, y, aspect)
    rotation = _rotation(
        math.radians(rng.uniform(0, MAX_TILT_DEG)), rng.uniform(0, math.pi),
        math.radians(rng.normal(0, 8)),
    )  # fmt: skip
    points = torch.stack([x - 0.5, y - aspect / 2, z], dim=-1) @ rotation.T
    distance = rng.uniform(1.4, 3.0)
    focal = fill * min(out_w, out_h / aspect) * distance  # page width -> fill of the frame
    depth = points[..., 2] + distance
    px = focal * points[..., 0] / depth
    py = focal * points[..., 1] / depth
    # Fit the page into the frame (shrink if needed) at a random place with a margin.
    margin = rng.uniform(0.02, 0.1) * min(out_w, out_h)
    low = torch.stack([px.min(), py.min()])
    span = torch.stack([px.max(), py.max()]) - low
    room = torch.tensor([out_w - 2 * margin, out_h - 2 * margin])
    scale = float(torch.clamp((room / span).min(), max=1.0))
    px, py = (px - low[0]) * scale, (py - low[1]) * scale
    slack = (room - span * scale).clamp(min=0).numpy()
    px = px + margin + rng.uniform(0, slack[0])
    py = py + margin + rng.uniform(0, slack[1])
    # Shading from the surface normal, camera-side light.
    gz_y, gz_x = torch.gradient(z, spacing=(ys, xs))
    normal = torch.stack([gz_x, gz_y, -torch.ones_like(z)], dim=-1) @ rotation.T
    normal = normal / normal.norm(dim=-1, keepdim=True)
    light = np.array([rng.normal(0, 0.5), rng.normal(0, 0.5), -1.0])
    light = torch.tensor(light / np.linalg.norm(light), dtype=torch.float32)
    lambert = (normal * light).sum(-1).clamp(min=0)
    ambient = rng.uniform(0.35, 0.7)
    shade = ambient + (1 - ambient) * lambert / max(float(lambert.max()), 1e-6)
    tex = torch.from_numpy(np.ascontiguousarray(texture)).permute(2, 0, 1)[None].float() / 255
    sample_grid = torch.stack([x * 2 - 1, y / aspect * 2 - 1], dim=-1)[None]
    colour = F.grid_sample(tex, sample_grid, align_corners=True, mode="bilinear")[0]
    colour = (colour * shade[None]).permute(1, 2, 0).reshape(-1, 3)
    # Bilinear splat into the photo.
    fx, fy = px.reshape(-1) - 0.5, py.reshape(-1) - 0.5
    x0, y0 = torch.floor(fx), torch.floor(fy)
    acc = torch.zeros(out_h * out_w, 3)
    weight = torch.zeros(out_h * out_w)
    for ox in (0, 1):
        for oy in (0, 1):
            xi, yi = x0 + ox, y0 + oy
            w = (1 - (fx - xi).abs()) * (1 - (fy - yi).abs())
            inside = (xi >= 0) & (xi < out_w) & (yi >= 0) & (yi < out_h) & (w > 0)
            index = (yi[inside] * out_w + xi[inside]).long()
            acc.index_add_(0, index, colour[inside] * w[inside, None])
            weight.index_add_(0, index, w[inside])
    photo = acc / weight.clamp(min=1e-6)[:, None]
    covered = (weight > 0.3).float()[:, None]
    if background is None:
        base = rng.uniform(0.1, 0.9, 3)
        background = np.full((out_h, out_w, 3), base * 255, np.uint8)
    back = torch.from_numpy(np.ascontiguousarray(background)).reshape(-1, 3).float() / 255
    photo = (photo * covered + back * (1 - covered)).reshape(out_h, out_w, 3)
    image = (photo.clamp(0, 1) * 255 + 0.5).to(torch.uint8).numpy()
    # Forward map at the center-aligned map grid: exact projection of those page points.
    map_h, map_w = MAP_HW
    grid_x = (torch.arange(map_w) + 0.5) / map_w
    grid_y = (torch.arange(map_h) + 0.5) / map_h * aspect
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    at = torch.stack([gx * 2 - 1, gy / aspect * 2 - 1], dim=-1)[None]
    fields = torch.stack([px, py], dim=0)[None]
    mapped = F.grid_sample(fields, at, align_corners=True, mode="bilinear")[0].permute(1, 2, 0)
    forward = torch.stack([mapped[..., 0] / out_w * 2 - 1, mapped[..., 1] / out_h * 2 - 1], -1)
    return image, forward.numpy().astype(np.float32)
