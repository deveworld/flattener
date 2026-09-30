"""Crumpled and creased pages: a height field over a flat page, seen by a pinhole camera.

``render(texture, rng, out_hw)`` returns the photo (uint8 (H, W, 3)), its exact forward
map (for each point of a center-aligned (``MAP_HW``) grid over the flat page, the
flattener value of where it lands in the photo; the dewarp target,
``backward_map_convention: "flattener"`` in the converted data layout) and the surface's
shape at those points in the page's frame (the 3-D target).

Surface (page width 1, height ``aspect``), by ``mode`` (``surface``):
- ``crumple``: creases, line segments with a V profile ``a * (sqrt(s^2 + w^2) - w)``
  across them (``s`` the signed distance, ``w`` the rounding) faded out past their ends,
  a few folds across the whole page, a quadratic curl and low-frequency waves;
- ``crease``: the same with ridges that level off across (``tanh`` over a reach) and
  fade over that reach, so crumpled paper stays within real slopes;
- ``fold``: a letter folded in halves or thirds each way and opened again;
- ``curve``: a smooth curl under a steep, close camera; ``flat``: a slight curl;
- ``book``: two pages open at a gutter that dips away, arching or sagging outward, the
  texture on one of them; ``spread``: the same with both pages as the texture (the
  scanner's target for an open book, which it splits at the gutter).
With ``isometric`` the texture lies on the surface as the paper laid flat (``lay_flat``,
``to_ground``), so it keeps its length and slopes foreshorten it; without, each texture
point sits straight above its place on the ground plane, which stretches the paper on
slopes (the v1-v3 builds).
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
CURVE_TILT_DEG = 45.0
OVERSAMPLE = 2.2
FLAT_COLS = 385  # ground-plane grid across the page for laying it flat
FLAT_PAD = 0.08  # its margin around the page, share of each side


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


def _ridges(rng: np.random.Generator, aspect: float) -> list[tuple]:
    """Ridges of crumpled paper, each with a ``reach`` across past which it stays level,
    and a few page-wide folds."""
    ridges = []
    for _ in range(int(rng.integers(4, 30))):
        center = rng.uniform([0.0, 0.0], [1.0, aspect])
        angle = rng.uniform(0, math.pi)
        half = rng.uniform(0.05, 0.6) * math.hypot(1.0, aspect)
        slope = float(np.clip(rng.normal(0, 0.18), -0.5, 0.5))
        width = rng.uniform(0.002, 0.015)
        reach = rng.uniform(0.03, 0.25)
        ridges.append((center, angle, half, slope, width, reach))
    for _ in range(int(rng.choice(3, p=[0.5, 0.3, 0.2]))):
        center = rng.uniform([0.2, 0.2 * aspect], [0.8, 0.8 * aspect])
        angle = rng.choice([0.0, math.pi / 2]) + rng.normal(0, 0.08)
        slope = float(rng.choice([-1, 1]) * rng.uniform(0.05, 0.4))
        ridges.append((center, angle, 10.0, slope, rng.uniform(0.002, 0.01), 10.0))
    return ridges


def surface(rng: np.random.Generator, aspect: float, mode: str = "crumple"):
    """A height function ``z(x, y)`` (page-width units) over page points ``x`` in [0, 1],
    ``y`` in [0, aspect], drawn now so it can be evaluated anywhere: ``crumple`` sums
    creases, folds, curl and waves; ``curve`` is a smooth curl (a page lying on a book or
    curling off a table) with at most one soft fold; ``flat`` is a page lying almost flat;
    ``fold`` is a letter folded in halves or thirds and opened again, with a slight curl."""
    terms = []

    def across(x, y, center, direction):
        return -(x - float(center[0])) * math.sin(direction) + (y - float(center[1])) * math.cos(
            direction
        )

    if mode in ("flat", "curve"):
        axis = rng.uniform(0, math.pi)

        def along(x, y, axis=axis):
            return (x - 0.5) * math.cos(axis) + (y - aspect / 2) * math.sin(axis)

        if mode == "flat":  # paper lying almost flat: a slight curl, no creases
            a = float(rng.uniform(-0.06, 0.06))
            terms.append(lambda x, y: a * along(x, y) ** 2)
        else:
            a, b = float(rng.uniform(-0.3, 0.3)), float(rng.uniform(-0.08, 0.08))
            terms.append(lambda x, y: a * along(x, y) ** 2 + b * along(x, y) ** 3)
            if rng.random() < 0.4:
                center = rng.uniform([0.3, 0.3 * aspect], [0.7, 0.7 * aspect])
                direction = rng.choice([0.0, math.pi / 2]) + rng.normal(0, 0.1)
                width = rng.uniform(0.01, 0.05)
                c = float(rng.uniform(-0.12, 0.12))
                terms.append(
                    lambda x, y: (
                        c * (torch.sqrt(across(x, y, center, direction) ** 2 + width**2) - width)
                    )
                )
    elif mode in ("book", "spread"):
        # book: two pages over x in [0, 2] page widths, the gutter at x = 1; spread: the
        # same as one sheet (the texture holds both pages), x in [0, 1], the gutter at 0.5
        unit = 1.0 if mode == "book" else 0.5  # a page width in the sheet's units
        page_aspect = aspect / unit
        reach = rng.uniform(0.05, 0.3)
        depth = reach * rng.uniform(0.3, 1.5)
        spine = rng.normal(0, 0.15)  # the gutter deeper at one end
        arch = rng.uniform(-0.12, 0.04, size=2)  # outer parts rising (-) or sagging (+)

        def book(x, y):
            s = (x - unit) / unit  # page widths from the gutter (at x = unit)
            along = y / unit
            gutter = (
                depth
                * (1 + spine * (along - page_aspect / 2) / page_aspect)
                * torch.exp(-s.abs() / reach)
            )
            side = torch.where(s < 0, float(arch[0]), float(arch[1]))
            return unit * (gutter + side * s**2)

        terms.append(book)
    elif mode == "fold":  # folded in halves or thirds along each side, then opened
        for side, length in ((0, 1.0), (1, aspect)):
            parts = int(rng.choice([1, 2, 3], p=[0.35, 0.35, 0.3]))
            for k in range(1, parts):
                at = length * (k / parts + rng.normal(0, 0.015))
                direction = (math.pi / 2 if side == 0 else 0.0) + rng.normal(0, 0.03)
                center = (at, aspect / 2) if side == 0 else (0.5, at)
                slope = float(rng.choice([-1, 1]) * rng.uniform(0.1, 0.6))
                width = rng.uniform(0.002, 0.012)
                terms.append(
                    lambda x, y, center=center, direction=direction, slope=slope, width=width: (
                        slope
                        * (torch.sqrt(across(x, y, center, direction) ** 2 + width**2) - width)
                    )
                )
        curl = rng.normal(0, 0.08, size=2)
        terms.append(
            lambda x, y: float(curl[0]) * (x - 0.5) ** 2 + float(curl[1]) * (y - aspect / 2) ** 2
        )
    elif mode == "crease":  # crumpled paper: ridges and valleys that die out all round
        for center, angle, half, slope, width, reach in _ridges(rng, aspect):
            direction = (math.cos(angle), math.sin(angle))

            def ridge(x, y, center=center, direction=direction, half=half, slope=slope,
                      width=width, reach=reach):  # fmt: skip
                dx, dy = x - float(center[0]), y - float(center[1])
                along = dx * direction[0] + dy * direction[1]
                across = -dx * direction[1] + dy * direction[0]
                fade = torch.sigmoid((half - along.abs()) / (0.5 * reach + 1e-3))
                rise = torch.tanh((torch.sqrt(across**2 + width**2) - width) / reach)
                return slope * reach * rise * fade

            terms.append(ridge)
        curl = rng.normal(0, 0.12, size=2)
        terms.append(lambda x, y: float(curl[0]) * (x - 0.5) ** 2)
        terms.append(lambda x, y: float(curl[1]) * (y - aspect / 2) ** 2)
    else:
        for center, angle, half, slope, width in _creases(rng, aspect):
            direction = (math.cos(angle), math.sin(angle))

            def crease(x, y, center=center, direction=direction, half=half, slope=slope,
                       width=width):  # fmt: skip
                dx, dy = x - float(center[0]), y - float(center[1])
                along = dx * direction[0] + dy * direction[1]
                across = -dx * direction[1] + dy * direction[0]
                fade = torch.sigmoid((half - along.abs()) / (0.08 * half + 1e-3))
                return slope * (torch.sqrt(across**2 + width**2) - width) * fade

            terms.append(crease)
        curl = rng.normal(0, 0.12, size=2)
        terms.append(lambda x, y: float(curl[0]) * (x - 0.5) ** 2)
        terms.append(lambda x, y: float(curl[1]) * (y - aspect / 2) ** 2)
        for _ in range(int(rng.integers(0, 3))):
            k = rng.uniform(2, 8) * 2 * math.pi
            phase, theta = rng.uniform(0, 2 * math.pi), rng.uniform(0, math.pi)
            amplitude = rng.uniform(0.001, 0.005)
            terms.append(
                lambda x, y, k=k, phase=phase, theta=theta, amplitude=amplitude: (
                    amplitude * torch.sin(k * (x * math.cos(theta) + y * math.sin(theta)) + phase)
                )
            )

    def z(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(x)
        for term in terms:
            total = total + term(x, y)
        return total

    return z


def height_field(
    rng: np.random.Generator,
    x: torch.Tensor,
    y: torch.Tensor,
    aspect: float,
    mode: str = "crumple",
) -> torch.Tensor:
    """Heights of a new ``surface`` at page points ``x``, ``y``, less their mean."""
    z = surface(rng, aspect, mode)(x, y)
    return z - z.mean()


def _dct_matrix(n: int, device: str) -> torch.Tensor:
    k = torch.arange(n, device=device, dtype=torch.float64)[:, None]
    j = torch.arange(n, device=device, dtype=torch.float64)[None]
    matrix = torch.cos(math.pi * k * (j + 0.5) / n) * math.sqrt(2.0 / n)
    matrix[0] /= math.sqrt(2.0)
    return matrix


def lay_flat(z, aspect: float, device: str = "cpu", cols: int = FLAT_COLS, width: float = 1.0):
    """Where the surface ``(x, y, z(x, y))`` lies when the paper is laid flat: fields
    (u, v) on a grid over the page's ground plane (with a margin), in page-width units,
    and the grid's bounds. The Jacobian of (u, v) is fitted in least squares to the
    metric's symmetric square root ``I + c g g^T`` (``g`` the height gradient,
    ``c = (sqrt(1 + |g|^2) - 1) / |g|^2``), exact arc length for bends along one
    direction; the Neumann Poisson solve is diagonal in the DCT. The page centre stays
    put. ``width``: the sheet's width in page widths (2 for a book spread)."""
    pad = FLAT_PAD
    cols = round(cols * width)
    rows = max(8, round(cols / width * aspect))
    x0, x1, y0, y1 = -pad * width, width * (1 + pad), -pad * aspect, aspect * (1 + pad)
    xs = torch.linspace(x0, x1, cols, device=device, dtype=torch.float64)
    ys = torch.linspace(y0, y1, rows, device=device, dtype=torch.float64)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    heights = z(x, y)
    gy, gx = torch.gradient(heights, spacing=(ys, xs))
    g2 = gx**2 + gy**2
    c = torch.where(g2 > 1e-12, (torch.sqrt(1 + g2) - 1) / g2.clamp(min=1e-12), 0.5)
    hx, hy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    cx, cy = _dct_matrix(cols, device), _dct_matrix(rows, device)
    kx = 2 - 2 * torch.cos(math.pi * torch.arange(cols, device=device, dtype=torch.float64) / cols)
    ky = 2 - 2 * torch.cos(math.pi * torch.arange(rows, device=device, dtype=torch.float64) / rows)
    eigen = ky[:, None] + kx[None]
    eigen[0, 0] = 1.0
    fields = []
    for jx, jy in ((1 + c * gx * gx, c * gx * gy), (c * gx * gy, 1 + c * gy * gy)):
        ex = hx * (jx[:, 1:] + jx[:, :-1]) / 2  # wanted differences along x edges
        ey = hy * (jy[1:] + jy[:-1]) / 2
        rhs = torch.zeros_like(heights)
        rhs[:, :-1] -= ex
        rhs[:, 1:] += ex
        rhs[:-1] -= ey
        rhs[1:] += ey
        spectrum = cy @ rhs @ cx.T / eigen
        spectrum[0, 0] = 0.0
        fields.append(cy.T @ spectrum @ cx)
    u, v = fields
    row, col = (rows - 1) // 2, (cols - 1) // 2
    u = u - u[row, col] + x[row, col]
    v = v - v[row, col] + y[row, col]
    return torch.stack([u, v]).float(), (x0, x1, y0, y1)


def to_ground(
    tx: torch.Tensor, ty: torch.Tensor, fields: torch.Tensor, bounds: tuple, steps: int = 12
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ground-plane points whose laid-flat place is (``tx``, ``ty``), by Newton steps on
    the fields of ``lay_flat``. Raises ValueError when they do not settle."""
    x0, x1, y0, y1 = bounds
    rows, cols = fields.shape[1:]
    du_y, du_x = torch.gradient(fields[0], spacing=((y1 - y0) / (rows - 1), (x1 - x0) / (cols - 1)))
    dv_y, dv_x = torch.gradient(fields[1], spacing=((y1 - y0) / (rows - 1), (x1 - x0) / (cols - 1)))
    stack = torch.stack([fields[0], fields[1], du_x, du_y, dv_x, dv_y])[None]
    target = torch.stack([tx, ty], dim=-1).reshape(1, -1, 1, 2)
    point = target.clone()
    scale = torch.tensor([2 / (x1 - x0), 2 / (y1 - y0)], device=tx.device)
    offset = torch.tensor([x0, y0], device=tx.device)
    for _ in range(steps):
        at = (point - offset) * scale - 1
        u, v, ux, uy, vx, vy = F.grid_sample(stack, at, align_corners=True, mode="bilinear")[0]
        miss_u, miss_v = target[..., 0] - u[None], target[..., 1] - v[None]
        det = (ux * vy - uy * vx)[None].clamp(min=0.05)
        step_x = (vy[None] * miss_u - uy[None] * miss_v) / det
        step_y = (ux[None] * miss_v - vx[None] * miss_u) / det
        point = point + torch.stack([step_x, step_y], dim=-1)
    if float(torch.maximum(miss_u.abs(), miss_v.abs()).max()) > 1e-4:
        raise ValueError("the laid-flat page does not invert")
    point = point.reshape(*tx.shape, 2)
    return point[..., 0], point[..., 1]


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
    mode: str = "crumple",
    device: str = "cpu",
    isometric: bool = False,
    close: bool | None = None,
    facing: NDArray[np.uint8] | None = None,
    gloss: bool = False,
    shadow: bool = False,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32]]:
    """(photo, forward map, shape) of ``texture`` (the flat page) crumpled and photographed.
    ``shape`` holds the surface point of each map point in the page's own frame (x across,
    y down, z away from the camera, page-width units, centred), the same at any camera.
    ``mode`` ``curve`` is a gently curved page under a steep, close camera (tilt up to
    ``CURVE_TILT_DEG``). A page that folds over itself in the photo raises ValueError.
    ``device`` runs the tensor work there (the draws stay on the CPU generator).
    ``isometric`` keeps the paper's length: the texture lies on the surface as the page
    laid flat (``lay_flat``), so slopes foreshorten it as they do real paper; otherwise
    each texture point sits straight above its place on the ground plane. ``close`` picks
    the steep, close camera (``None``: only for ``curve``). ``mode`` ``book`` needs
    ``facing``, a page of the texture's size: the two lie open at a gutter, the texture on
    a random side, and only the texture is framed and mapped (the facing page may leave
    the photo). ``gloss`` adds a Blinn-Phong highlight of the light (glossy paper under a
    flash); ``shadow`` darkens the photo past one or two soft lines (a hand or a phone
    between the light and the page)."""
    out_h, out_w = out_hw
    tex_h, tex_w = texture.shape[:2]
    aspect = tex_h / tex_w
    width, left = 1, 0  # the sheet in page widths, and where the page starts on it
    if mode == "book":
        if facing is None or facing.shape != texture.shape:
            raise ValueError("a book needs a facing page of the texture's size")
        width, left = 2, int(rng.random() < 0.5)
        texture = np.concatenate([facing, texture] if left else [texture, facing], axis=1)
    fill = rng.uniform(0.55, 0.95)  # of the frame's limiting side
    # Page sampling density: enough samples per output pixel the page covers.
    page_px = fill * min(out_w, out_h / aspect)
    cols = int(page_px * OVERSAMPLE)
    rows = int(cols * aspect)
    xs = torch.linspace(0, width, cols * width, device=device)
    ys = torch.linspace(0, aspect, rows, device=device)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    shape = surface(rng, aspect, mode)
    ground_x, ground_y = x, y
    if isometric:
        ground_x, ground_y = to_ground(x, y, *lay_flat(shape, aspect, device, width=width))
    z = shape(ground_x, ground_y)
    z = z - z.mean()
    centre = left + 0.5  # the page's centre on the sheet
    page = (x >= left) & (x <= left + 1)
    close = mode == "curve" if close is None else close
    max_tilt = CURVE_TILT_DEG if close else MAX_TILT_DEG
    rotation = _rotation(
        math.radians(rng.uniform(0, max_tilt)), rng.uniform(0, math.pi),
        math.radians(rng.normal(0, 8)),
    )  # fmt: skip
    rotation = rotation.to(device)
    points = torch.stack([ground_x - centre, ground_y - aspect / 2, z], dim=-1) @ rotation.T
    distance = rng.uniform(0.9, 1.8) if close else rng.uniform(1.4, 3.0)
    focal = fill * min(out_w, out_h / aspect) * distance  # page width -> fill of the frame
    depth = points[..., 2] + distance
    if float(depth.min()) <= 0.2:
        raise ValueError("page reaches the camera")
    px = focal * points[..., 0] / depth
    py = focal * points[..., 1] / depth
    jx_y, jx_x = torch.gradient(px)
    jy_y, jy_x = torch.gradient(py)
    if float((jx_x * jy_y - jx_y * jy_x).min()) <= 0:
        raise ValueError("page folds over itself in the photo")
    # Fit the page into the frame (shrink if needed) at a random place with a margin.
    margin = rng.uniform(0.02, 0.1) * min(out_w, out_h)
    low = torch.stack([px[page].min(), py[page].min()])
    span = torch.stack([px[page].max(), py[page].max()]) - low
    room = torch.tensor([out_w - 2 * margin, out_h - 2 * margin], device=device)
    scale = float(torch.clamp((room / span).min(), max=1.0))
    px, py = (px - low[0]) * scale, (py - low[1]) * scale
    slack = (room - span * scale).clamp(min=0).cpu().numpy()
    px = px + margin + rng.uniform(0, slack[0])
    py = py + margin + rng.uniform(0, slack[1])
    # Shading from the surface normal, camera-side light.
    if isometric:
        surface_points = torch.stack([ground_x, ground_y, z], dim=-1)
        along_y, along_x = torch.gradient(surface_points, dim=(0, 1))
        normal = -torch.linalg.cross(along_x, along_y) @ rotation.T
    else:
        gz_y, gz_x = torch.gradient(z, spacing=(ys, xs))
        normal = torch.stack([gz_x, gz_y, -torch.ones_like(z)], dim=-1) @ rotation.T
    normal = normal / normal.norm(dim=-1, keepdim=True)
    light = np.array([rng.normal(0, 0.5), rng.normal(0, 0.5), -1.0])
    light = torch.tensor(light / np.linalg.norm(light), dtype=torch.float32, device=device)
    lambert = (normal * light).sum(-1).clamp(min=0)
    ambient = rng.uniform(0.35, 0.7)
    shade = ambient + (1 - ambient) * lambert / max(float(lambert.max()), 1e-6)
    tex = torch.from_numpy(np.ascontiguousarray(texture)).to(device).permute(2, 0, 1)[None]
    tex = tex.float() / 255
    sample_grid = torch.stack([x / width * 2 - 1, y / aspect * 2 - 1], dim=-1)[None]
    colour = F.grid_sample(tex, sample_grid, align_corners=True, mode="bilinear")[0]
    colour = (colour * shade[None]).permute(1, 2, 0).reshape(-1, 3)
    if gloss:
        eye = torch.tensor([0.0, 0.0, distance], device=device)
        view = -(points + eye)  # towards the camera
        half = light + view / view.norm(dim=-1, keepdim=True)
        half = half / half.norm(dim=-1, keepdim=True)
        strength, shininess = rng.uniform(0.15, 0.5), rng.uniform(15, 80)
        highlight = strength * (normal * half).sum(-1).clamp(min=0) ** shininess
        colour = colour + highlight.reshape(-1, 1)
    # Bilinear splat into the photo.
    fx, fy = px.reshape(-1) - 0.5, py.reshape(-1) - 0.5
    x0, y0 = torch.floor(fx), torch.floor(fy)
    acc = torch.zeros(out_h * out_w, 3, device=device)
    weight = torch.zeros(out_h * out_w, device=device)
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
    back = torch.from_numpy(np.ascontiguousarray(background)).to(device).reshape(-1, 3)
    back = back.float() / 255
    photo = (photo * covered + back * (1 - covered)).reshape(out_h, out_w, 3)
    if shadow:
        side = min(out_h, out_w)
        rows_at = (torch.arange(out_h, device=device) - out_h / 2) / side
        cols_at = (torch.arange(out_w, device=device) - out_w / 2) / side
        at_y, at_x = torch.meshgrid(rows_at, cols_at, indexing="ij")
        soft, dark = rng.uniform(0.01, 0.08), rng.uniform(0.2, 0.55)
        mask = torch.ones_like(at_x)
        for _ in range(1 + int(rng.random() < 0.5)):
            angle, offset = rng.uniform(0, 2 * math.pi), rng.uniform(-0.1, 0.45)
            edge = at_x * math.cos(angle) + at_y * math.sin(angle) - offset
            mask = mask * torch.sigmoid(edge / soft)
        photo = photo * (1 - dark * mask)[..., None]
    image = (photo.clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()
    # Forward map at the center-aligned map grid: exact projection of those page points.
    map_h, map_w = MAP_HW
    grid_x = left + (torch.arange(map_w, device=device) + 0.5) / map_w
    grid_y = (torch.arange(map_h, device=device) + 0.5) / map_h * aspect
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    at = torch.stack([gx / width * 2 - 1, gy / aspect * 2 - 1], dim=-1)[None]
    fields = torch.stack([px, py], dim=0)[None]
    mapped = F.grid_sample(fields, at, align_corners=True, mode="bilinear")[0].permute(1, 2, 0)
    forward = torch.stack([mapped[..., 0] / out_w * 2 - 1, mapped[..., 1] / out_h * 2 - 1], -1)
    frame = torch.stack([ground_x - centre, ground_y - aspect / 2, z], dim=0)[None]
    shape = F.grid_sample(frame, at, align_corners=True, mode="bilinear")[0].permute(1, 2, 0)
    return image, forward.cpu().numpy().astype(np.float32), shape.cpu().numpy().astype(np.float32)
