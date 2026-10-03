"""Re-texture UVDoc's raw captures with license-clean pages: ``python tools/uvdoc/retexture.py
--spec S --output D``.

UVDoc's raw set (github.com/tanguymagne/UVDoc-Dataset, MIT, (c) 2023 Tanguy Magne) holds the
authors' photos of deformed blank paper: ``samples/rgb/<id>.png``, a per-pixel texture
coordinate map ``samples/uvmap/<id>.mat`` (``uv``, 0-1, NaN off the page), a page mask
``samples/seg/<id>.mat`` and the page grid ``samples/grid2d/<id>.mat`` (89x61 pixel
coordinates). Its own ``create_final.py`` lays a document on each photo through the UV
map, multiplies it into the photographed paper (shading strength 0.75) and swaps the
background; the released composites use DTD backgrounds (research only) and partly
DeepFloyd-generated pages (non-commercial), so they stay off the release track. This tool
does the same with our pages: clean SyntheticDoc renders flattened through their ground
truth (``tools/crumple/build.py``'s ``flat_texture``) or own generated pages, on CC0 or
procedural backgrounds, and it never reads the raw ``textures/`` or ``backgrounds/``.

The build spec (``flattener-uvdoc-retexture-1``, see ``examples/dewarp``) names the raw
root, the benchmark whose captures stay out (``exclude_benchmark``: a folder with
``metadata_geom/*.json``; their ``sample_id`` captures are skipped), the dewarp data
spec the pages come from, the photo size, counts per split (captures are split by a hash
of their id and ``split_name``, the spec's name when absent, ``val_share`` to val; a
further build of the same captures names the first one here so both hold out the same
captures), ``own_texture_prob``, ``background_dir``,
``capture_background_prob`` (keep the photo's own surroundings), ``shade`` (strength
range) and ``margin`` (crop margin around the page, share of its size per side). Each
item flips the capture as UVDoc does (texture coordinates follow, so text never
mirrors), crops a window around the page, lays the page at the photo size (texture
coordinates averaged over each output pixel) and stores the exact page grid as a
center-aligned flattener map in the converted layout, with
``manifest.jsonl`` rows like ``tools/crumple/build.py``. Captures are processed one at a
time (all items of a capture share its files), ``chunk`` of them per task; parts resume.

``"geometry": "doc3d_grid"`` takes the paper's shape from Doc3D instead (MIT, (c) 2020
CVLab@StonyBrook): ``raw_root`` holds ``grid2D/<folder>/<name>.mat`` (45x31 node positions
in the 448x448 render) and ``grid3D`` (the nodes in space), as in UVDoc's
``Doc3D_grid.zip``. Doc3D's renders, albedo and lighting are never read. Each grid is
refined bicubically to the map's 177x121 nodes, the page is drawn through it (points
splatted at most half a pixel apart, at twice the photo size), lit by a Lambertian
light from its 3-D shape with a paper colour, a soft gradient and grain, and the refined
grid is the stored map. A page that turns over in view is skipped. Renders of one mesh
share a split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flattener.dewarp.checkpoint import write_json  # noqa: E402
from flattener.dewarp.composite import procedural_background  # noqa: E402
from flattener.dewarp.coords import grid_to_map, pixels_to_flattener  # noqa: E402
from flattener.dewarp.data import canonical_json, expand_path  # noqa: E402

BUILD_SCHEMA = "flattener-uvdoc-retexture-1"
MAP_HW = (177, 121)
SOURCE = "UVDoc raw captures, MIT, (c) 2023 Tanguy Magne (github.com/tanguymagne/UVDoc-Dataset)"


def _crumple_build():
    path = HERE.parent / "crumple" / "build.py"
    spec = importlib.util.spec_from_file_location("crumple_build", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _seed(*parts: Any) -> int:
    text = "\0".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def read_matrix(path: Path, key: str) -> np.ndarray:
    """A MATLAB v7.3 (HDF5) or older .mat array in numpy axis order."""
    try:
        import h5py

        with h5py.File(path, "r") as stream:
            return np.array(stream[key][:]).T
    except OSError:
        from scipy.io import loadmat

        return np.asarray(loadmat(path)[key])


def load_capture(raw: Path, capture: str) -> dict[str, np.ndarray]:
    """One raw capture: the photo (H, W, 3) uint8, ``uv`` (H, W, 2) with NaN off the page,
    the page mask (H, W) bool and the page grid (89, 61, 2) in pixel coordinates (rows
    down the page)."""
    samples = raw / "samples"
    with Image.open(samples / "rgb" / f"{capture}.png") as image:
        photo = np.asarray(image.convert("RGB"))
    uv = read_matrix(samples / "uvmap" / f"{capture}.mat", "uv").astype(np.float32)
    seg = read_matrix(samples / "seg" / f"{capture}.mat", "seg") > 0.5
    grid = read_matrix(samples / "grid2d" / f"{capture}.mat", "grid2d").astype(np.float64)
    height, width = photo.shape[:2]
    if uv.shape[:2] != (height, width) and uv.shape[:2] == (width, height):
        uv = uv.transpose(1, 0, 2)
    if seg.shape != (height, width) and seg.shape == (width, height):
        seg = seg.T
    if grid.shape[0] < grid.shape[1]:  # rows run down the page (89 of them)
        grid = grid.transpose(1, 0, 2)
    return {"photo": photo, "uv": uv, "seg": seg, "grid": grid}


def flip(capture: dict[str, np.ndarray], horizontal: bool, vertical: bool) -> dict:
    """The capture mirrored as UVDoc's augment_with_flip does: the photo, mask and texture
    coordinates flip together and the grid's node order follows, so the page reads the
    same way round. Pixel centers sit at integers here (x -> W - 1 - x)."""
    photo, uv, seg, grid = capture["photo"], capture["uv"], capture["seg"], capture["grid"]
    height, width = photo.shape[:2]
    uv, grid = uv.copy(), grid.copy()
    if horizontal:
        photo, seg, uv = photo[:, ::-1], seg[:, ::-1], uv[:, ::-1]
        uv[..., 0] = 1 - uv[..., 0]
        grid[..., 0] = width - 1 - grid[..., 0]
        grid = grid[:, ::-1]
    if vertical:
        photo, seg, uv = photo[::-1], seg[::-1], uv[::-1]
        uv[..., 1] = 1 - uv[..., 1]
        grid[..., 1] = height - 1 - grid[..., 1]
        grid = grid[::-1]
    return {"photo": np.ascontiguousarray(photo), "uv": np.ascontiguousarray(uv),
            "seg": np.ascontiguousarray(seg), "grid": np.ascontiguousarray(grid)}  # fmt: skip


def shrink(
    capture: dict[str, np.ndarray],
    window: tuple[int, int, int, int],
    out_hw: tuple[int, int],
    device: str = "cpu",
) -> dict[str, Any]:
    """The capture inside ``window`` at the photo size, on ``device``: the photo (H, W, 3,
    0-1), texture coordinates averaged over each output pixel's share of the page and the
    page mask (more than half of the pixel on the page)."""
    import torch
    import torch.nn.functional as F

    x0, y0, x1, y1 = window
    photo = torch.from_numpy(np.ascontiguousarray(capture["photo"][y0:y1, x0:x1])).to(device)
    photo = photo.permute(2, 0, 1)[None].float() / 255
    photo = F.interpolate(photo, size=out_hw, mode="bilinear", antialias=True, align_corners=False)
    uv = torch.from_numpy(np.ascontiguousarray(capture["uv"][y0:y1, x0:x1])).to(device)
    seg = torch.from_numpy(np.ascontiguousarray(capture["seg"][y0:y1, x0:x1])).to(device)
    valid = (torch.isfinite(uv).all(dim=-1) & seg).float()
    uv = torch.nan_to_num(uv, nan=0.0) * valid[..., None]
    stack = torch.cat([uv.permute(2, 0, 1), valid[None]])[None]
    stack = F.interpolate(stack, size=out_hw, mode="area")[0]
    share = stack[2]
    uv = (stack[:2] / share.clamp_min(1e-6)).permute(1, 2, 0)
    return {"photo": photo[0].permute(1, 2, 0), "uv": uv, "inside": share > 0.5}


def compose(
    view: dict[str, Any], texture: np.ndarray, background: np.ndarray | None, shade: float
) -> np.ndarray:
    """The page laid on the photographed paper through the UV map, ``shade`` of the
    paper's light multiplied in (create_final.py uses 0.75), on ``background`` (None keeps
    the photo's own surroundings). ``view`` comes from ``shrink``."""
    import torch
    import torch.nn.functional as F

    photo, uv, inside = view["photo"], view["uv"], view["inside"]
    device = photo.device
    coords = torch.where(inside[..., None], uv * 2 - 1, torch.full_like(uv, -2.0))
    source = torch.from_numpy(np.ascontiguousarray(texture)).to(device)
    source = source.permute(2, 0, 1)[None].float() / 255
    warped = F.grid_sample(source, coords[None], align_corners=False)
    warped = _blur(warped, 0.5)[0].permute(1, 2, 0)
    page = photo * (shade * warped + (1 - shade))
    mask = inside.float()
    # A soft edge one pixel past the page, so the paper's border blends into the background.
    grown = F.max_pool2d(mask[None, None], 3, stride=1, padding=1)
    edge = torch.maximum(_blur(grown, 0.75)[0, 0], mask)[..., None]
    if background is None:
        surround = photo
    else:
        surround = torch.from_numpy(np.ascontiguousarray(background)).to(device).float() / 255
    out = page * mask[..., None] + surround * (1 - mask[..., None])
    if background is not None:
        out = out * edge + surround * (1 - edge)
    return (out.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()


def _spread(values, valid, steps: int):
    """``values`` (C, H, W) carried ``steps`` pixels past ``valid`` (H, W) by repeated
    3x3 means of the valid neighbours (the nearest-value fill of create_final.py)."""
    import torch.nn.functional as F

    valid = valid.float()
    for _ in range(steps):
        total = F.avg_pool2d((values * valid)[None], 3, stride=1, padding=1)[0]
        count = F.avg_pool2d(valid[None, None], 3, stride=1, padding=1)[0, 0]
        grown = (count > 0) & (valid == 0)
        values = values.where(~grown[None], total / count.clamp_min(1e-6))
        valid = valid.maximum(grown.float())
    return values


def compose_uvdoc(
    view: dict[str, Any], texture: np.ndarray, background: np.ndarray, shade: float
) -> np.ndarray:
    """``compose`` as UVDoc's create_final.py does it (MIT): the page and its light are
    carried past the page's edge, the blend mask is the page blurred (sigma 0.75) and
    grown twice by 3x3, so the paper shows about two pixels beyond its labelled edge,
    and the background is shifted toward the paper's colour cast (per channel, within
    +-64/255) and, when darker, a fifth of the way to its brightness."""
    import torch
    import torch.nn.functional as F

    photo, uv, inside = view["photo"], view["uv"], view["inside"]
    device = photo.device
    small = -F.max_pool2d(-inside.float()[None, None], 3, stride=1, padding=1)[0, 0] > 0.5
    coords = torch.where(inside[..., None], uv * 2 - 1, torch.full_like(uv, -2.0))
    source = torch.from_numpy(np.ascontiguousarray(texture)).to(device)
    source = source.permute(2, 0, 1)[None].float() / 255
    warped = F.grid_sample(source, coords[None], align_corners=False)[0]
    warped = _spread(warped, small, 4)
    warped = _blur(warped[None], 0.5)[0]
    light = _spread(photo.permute(2, 0, 1), small, 4)
    page = light * (shade * warped + (1 - shade))
    mask = _blur(inside.float()[None, None], 0.75)
    mask = F.max_pool2d(F.max_pool2d(mask, 3, stride=1, padding=1), 3, stride=1, padding=1)[0]
    surround = torch.from_numpy(np.ascontiguousarray(background)).to(device).float() / 255
    surround = surround.permute(2, 0, 1)
    paper = photo[small]
    if len(paper):
        cast = paper.mean(dim=0)
        cast = (cast - cast.median()).clamp(-0.25, 0.25)
        surround = (surround + cast[:, None, None]).clamp(0, 1)
        diff = float(paper.mean()) * (shade * 0.556 + 1 - shade) - float(surround.mean())
        surround = (surround + (0.2 * diff if diff > 0 else diff)).clamp(0, 1)
    out = page * mask + surround * (1 - mask)
    return (out.permute(1, 2, 0).clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()


def _blur(image, sigma: float):
    """Gaussian blur of an (N, C, H, W) tensor, per channel, edges replicated."""
    import torch
    import torch.nn.functional as F

    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=image.device)
    taps = torch.exp(-(offsets**2) / (2 * sigma**2))
    taps = taps / taps.sum()
    channels = image.shape[1]
    out = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    out = F.conv2d(out, taps.view(1, 1, 1, -1).repeat(channels, 1, 1, 1), groups=channels)
    return F.conv2d(out, taps.view(1, 1, -1, 1).repeat(channels, 1, 1, 1), groups=channels)


def clean_texture(rgb: np.ndarray, device: str = "cpu") -> np.ndarray:
    """A flattened render with its own light divided out, so its creases and shadows do not
    reach the new photo as false shading: the paper's brightness is estimated per block
    (1/64 of the long side) as the 90th percentile of luminance, median-filtered over 3x3
    blocks (so a picture on the page does not count as paper) and enlarged bicubically;
    each pixel is brightened by the paper's 95th-percentile level over that estimate (gain
    1-1.8), the paper is balanced to white from its brightest fifth, and a light unsharp
    mask (0.5, sigma 1) restores edges the flattening softened."""
    import torch
    import torch.nn.functional as F

    image = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).float().permute(2, 0, 1)[None]
    image = image / 255
    height, width = image.shape[-2:]
    block = max(8, round(max(height, width) / 64))
    lum = image.mean(dim=1, keepdim=True)
    rows, cols = -(-height // block), -(-width // block)
    padded = F.pad(lum, (0, cols * block - width, 0, rows * block - height), mode="replicate")
    tiles = F.unfold(padded, block, stride=block)[0]  # (block * block, rows * cols)
    paper = torch.quantile(tiles, 0.9, dim=0).view(1, 1, rows, cols)
    near = F.unfold(F.pad(paper, (1, 1, 1, 1), mode="replicate"), 3)[0]
    paper = near.median(dim=0).values.view(1, 1, rows, cols)
    level = torch.quantile(paper.flatten(), 0.95)
    light = F.interpolate(paper, size=(height, width), mode="bicubic", align_corners=False)
    image = image * (level / light.clamp_min(0.05)).clamp(1.0, 1.8)
    bright = lum.flatten() >= torch.quantile(lum.flatten(), 0.8)
    white = image.flatten(2)[0][:, bright].median(dim=1).values.clamp_min(0.05)
    image = image * (0.97 / white).view(1, 3, 1, 1)
    image = image + 0.5 * (image - _blur(image, 1.0))
    return (image[0].permute(1, 2, 0).clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()


def pattern_background(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """A busy patterned surface like a textile or printed paper: stripes, checks, a thin
    mesh, dots, zigzags or blotches in two or three colours, at a random angle and scale,
    softened and lit by a gentle gradient."""
    height, width = size
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    angle = rng.uniform(0, np.pi)
    u = x * np.cos(angle) + y * np.sin(angle)
    v = -x * np.sin(angle) + y * np.cos(angle)
    period = rng.uniform(6, 48)
    kind = rng.integers(6)
    if kind == 0:  # stripes
        mask = (u / period) % 1 < rng.uniform(0.2, 0.6)
    elif kind == 1:  # checks
        mask = ((u // period) + (v // (period * rng.uniform(0.6, 1.6)))) % 2 == 0
    elif kind == 2:  # thin mesh
        line = rng.uniform(0.06, 0.2)
        mask = ((u / period) % 1 < line) | ((v / period) % 1 < line)
    elif kind == 3:  # dots
        du, dv = (u / period) % 1 - 0.5, (v / period) % 1 - 0.5
        mask = du**2 + dv**2 < rng.uniform(0.04, 0.16)
    elif kind == 4:  # zigzag
        wave = np.abs((v / (period * rng.uniform(0.5, 2))) % 2 - 1) * period
        mask = ((u + wave) / period) % 1 < 0.5
    else:  # blotches
        small = rng.normal(size=(max(2, height // 24), max(2, width // 24))).astype(np.float32)
        field = np.asarray(Image.fromarray(small).resize((width, height), Image.BICUBIC))
        mask = field > rng.uniform(-0.5, 0.5)
    colours = rng.uniform(0, 255, size=(3, 3)).astype(np.float32)
    image = np.where(mask[..., None], colours[0], colours[1])
    if rng.random() < 0.5:  # a third colour in a second, coarser pattern
        coarse = (v / (period * rng.uniform(2, 6))) % 1 < 0.5
        image = np.where(coarse[..., None], image * 0.6 + colours[2] * 0.4, image)
    image = np.asarray(Image.fromarray(image.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(rng.uniform(0, 1.5))), np.float32)  # fmt: skip
    gx, gy = rng.uniform(-0.3, 0.3, size=2)
    light = 1 + gx * (x / width - 0.5) + gy * (y / height - 0.5)
    return np.clip(image * light[..., None], 0, 255).astype(np.uint8)


def crop_window(
    grid: np.ndarray,
    size: tuple[int, int] | None,
    out_hw: tuple[int, int],
    margin: np.ndarray,
) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) around the page with ``margin`` (left, top, right, bottom, share of
    the page's extent) widened to the photo's aspect, clipped to whole pixels of an image
    of ``size``; ``size`` None keeps the exact window (nothing but the page is drawn)."""
    low, high = grid.reshape(-1, 2).min(0), grid.reshape(-1, 2).max(0)
    extent = high - low
    x0, y0 = low - margin[:2] * extent
    x1, y1 = high + margin[2:] * extent
    aspect = out_hw[0] / out_hw[1]  # height / width
    if (y1 - y0) / (x1 - x0) < aspect:  # too wide: grow the height
        grow = (x1 - x0) * aspect - (y1 - y0)
        y0, y1 = y0 - grow / 2, y1 + grow / 2
    else:
        grow = (y1 - y0) / aspect - (x1 - x0)
        x0, x1 = x0 - grow / 2, x1 + grow / 2
    if size is None:
        return float(x0), float(y0), float(x1), float(y1)
    height, width = size
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
    return x0, y0, x1, y1


def render_item(
    capture: dict[str, np.ndarray],
    texture: np.ndarray,
    background: Any,
    rng: np.random.Generator,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """(photo, center-aligned flattener map MAP_HW) of one item. ``background`` is None
    (keep the photo's surroundings) or a function of the photo size returning one."""
    out_hw = tuple(settings["photo_hw"])
    flipped = flip(capture, bool(rng.random() < 0.5), bool(rng.random() < 0.5))
    height, width = flipped["photo"].shape[:2]
    shade = float(rng.uniform(*settings.get("shade", (0.65, 0.85))))
    margin = rng.uniform(*settings.get("margin", (0.02, 0.15)), size=4)
    x0, y0, x1, y1 = crop_window(flipped["grid"], (height, width), out_hw, margin)
    view = shrink(flipped, (x0, y0, x1, y1), out_hw, settings.get("device", "cpu"))
    if settings.get("recipe") == "uvdoc" and background is not None:
        photo = compose_uvdoc(view, texture, background(out_hw), shade)
    else:
        photo = compose(view, texture, None if background is None else background(out_hw), shade)
    scale = np.array([out_hw[1] / (x1 - x0), out_hw[0] / (y1 - y0)])
    pixels = (flipped["grid"] - [x0, y0] + 0.5) * scale - 0.5  # pixel centers at integers
    values = pixels_to_flattener(pixels, out_hw)
    return photo, grid_to_map(values, MAP_HW)


DOC3D_SOURCE = (
    "Doc3D geometry: grid2D/grid3D of UVDoc's Doc3D_grid.zip (igl.ethz.ch/projects/uvdoc), "
    "derived from Doc3D (MIT, (c) 2020 CVLab@StonyBrook); renders, albedo and lighting unused"
)
DOC3D_SIZE = 448  # side of the renders grid2D's pixel coordinates refer to


def doc3d_ids(root: Path) -> list[str]:
    """``<folder>/<name>`` of the grid2D files under ``root`` (.mat or .npy)."""
    ids = {
        f"{path.parent.name}/{path.stem}"
        for path in (root / "grid2D").glob("*/*")
        if path.suffix in (".mat", ".npy") and not path.name.startswith(".")
    }
    return sorted(ids)


def doc3d_group(sample_id: str) -> str:
    """The mesh a Doc3D render was made from (``1/1000_4-pp_Page_847-Pwi0001`` -> ``1000``),
    so that renders of one mesh share a split."""
    return sample_id.split("/")[-1].split("-")[0].rsplit("_", 1)[0]


def load_doc3d(root: Path, sample_id: str) -> dict[str, np.ndarray]:
    """One Doc3D geometry: ``grid`` (45, 31, 2), pixel coordinates in the 448x448 render,
    rows down the page, and ``grid3d`` (45, 31, 3), the same nodes in space."""
    folder, name = sample_id.split("/")
    geometry = {}
    for key, field in (("grid2D", "grid"), ("grid3D", "grid3d")):
        npy = root / key / folder / f"{name}.npy"
        if npy.is_file():
            array = np.load(npy, allow_pickle=False)
        else:
            array = read_matrix(root / key / folder / f"{name}.mat", key)
        if array.shape[0] < array.shape[1]:
            array = array.transpose(1, 0, 2)
        geometry[field] = array.astype(np.float64)
    return geometry


def flip_doc3d(geometry: dict[str, np.ndarray], horizontal: bool, vertical: bool) -> dict:
    """``flip`` for a geometry without a photo: the nodes mirror in the render and their
    order follows, so the page reads the same way round."""
    grid, grid3d = geometry["grid"].copy(), geometry["grid3d"]
    if horizontal:
        grid[..., 0] = DOC3D_SIZE - 1 - grid[..., 0]
        grid, grid3d = grid[:, ::-1], grid3d[:, ::-1]
    if vertical:
        grid[..., 1] = DOC3D_SIZE - 1 - grid[..., 1]
        grid, grid3d = grid[::-1], grid3d[::-1]
    return {"grid": np.ascontiguousarray(grid), "grid3d": np.ascontiguousarray(grid3d)}


def _refine(grid: np.ndarray, hw: tuple[int, int], device: str):
    """An edge-aligned grid resampled bicubically to ``hw`` nodes, (H, W, C) on ``device``."""
    import torch
    import torch.nn.functional as F

    array = torch.from_numpy(np.ascontiguousarray(grid, dtype=np.float32)).to(device)
    array = F.interpolate(array.permute(2, 0, 1)[None], size=hw, mode="bicubic", align_corners=True)
    return array[0].permute(1, 2, 0)


def folded(points) -> bool:
    """Whether the page turns over in view: some cell of the (H, W, 2) image positions is
    wound against the rest or nearly edge-on (under 5% of the median cell area). The
    page then hides part of itself, which a grid cannot draw."""
    down = points[1:, :-1] - points[:-1, :-1]
    right = points[:-1, 1:] - points[:-1, :-1]
    area = right[..., 0] * down[..., 1] - right[..., 1] * down[..., 0]
    sign = 1.0 if float(area.sum()) >= 0 else -1.0
    return bool(((area * sign) < 0.05 * area.abs().median()).any())


def paper_light(grid3d, rng: np.random.Generator):
    """Lambertian light on the page from its (H, W, 3) shape: a light within 55 degrees
    of the page's mean normal (the camera's side), ambient 0.35-0.6, brightest 1."""
    import torch

    du = torch.gradient(grid3d, dim=1)[0]
    dv = torch.gradient(grid3d, dim=0)[0]
    normal = torch.linalg.cross(du, dv, dim=-1)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    mean = normal.mean(dim=(0, 1))
    mean = mean / mean.norm().clamp_min(1e-9)
    side = torch.from_numpy(rng.normal(size=3).astype(np.float32)).to(grid3d.device)
    side = side - (side @ mean) * mean
    side = side / side.norm().clamp_min(1e-9)
    angle = np.deg2rad(rng.uniform(0.0, 55.0))
    light = mean * float(np.cos(angle)) + side * float(np.sin(angle))
    ambient = float(rng.uniform(0.35, 0.6))
    shade = ambient + (1 - ambient) * (normal @ light).clamp_min(0)
    return shade / shade.max().clamp_min(1e-6)


def doc3d_view(points, light, out_hw: tuple[int, int], rng: np.random.Generator,
               supersample: int = 2) -> dict[str, Any]:  # fmt: skip
    """A ``shrink``-like view of a page drawn from its refined grid: ``points`` (H, W, 2)
    pixel positions in the photo (centers at integers), ``light`` (H, W). Points spread
    over each cell at most half a pixel apart (at ``supersample`` times the photo size)
    are splatted with their texture coordinates and light; the photo of the blank paper is
    that light times a paper colour, a soft gradient across the photo and fine grain."""
    import torch
    import torch.nn.functional as F

    device = points.device
    height, width = out_hw
    big = (height * supersample, width * supersample)
    scaled = (points + 0.5) * supersample - 0.5
    step = max(float((scaled[1:] - scaled[:-1]).norm(dim=-1).max()),
               float((scaled[:, 1:] - scaled[:, :-1]).norm(dim=-1).max()))  # fmt: skip
    k = int(min(max(np.ceil(2 * step), 1), 24))
    rows, cols = (points.shape[0] - 1) * k + 1, (points.shape[1] - 1) * k + 1
    stack = torch.cat([scaled.permute(2, 0, 1), light[None]])[None]
    dense = F.interpolate(stack, size=(rows, cols), mode="bilinear", align_corners=True)[0]
    u = torch.linspace(0, 1, cols, device=device).expand(rows, cols)
    v = torch.linspace(0, 1, rows, device=device)[:, None].expand(rows, cols)
    x, y = dense[0].round().long(), dense[1].round().long()
    keep = (x >= 0) & (x < big[1]) & (y >= 0) & (y < big[0])
    index = (y * big[1] + x)[keep]
    ones = torch.ones_like(index, dtype=torch.float32)
    values = torch.stack([u[keep], v[keep], dense[2][keep], ones])
    sums = torch.zeros(4, big[0] * big[1], device=device).index_add_(1, index, values)
    sums = sums.view(4, *big)
    # A pixel the points missed, among at least five covered neighbours, takes their mean.
    near = F.avg_pool2d(sums[None], 3, stride=1, padding=1)[0] * 9
    count = F.avg_pool2d((sums[3:] > 0).float()[None], 3, stride=1, padding=1)[0, 0] * 9
    hole = (sums[3] == 0) & (count >= 5)
    sums = torch.where(hole[None], near, sums)
    covered = (sums[3] > 0).float()
    mean = sums[:3] / sums[3].clamp_min(1e-6)
    pooled = F.avg_pool2d(torch.cat([mean * covered, covered[None]])[None], supersample)[0]
    share = pooled[3]
    uv = (pooled[:2] / share.clamp_min(1e-6)).permute(1, 2, 0)
    shade = pooled[2] / share.clamp_min(1e-6)
    tint = torch.from_numpy((rng.uniform(0.84, 0.97) + rng.uniform(-0.03, 0.03, 3)).astype(
        np.float32)).to(device)  # fmt: skip
    yy = torch.linspace(-1, 1, height, device=device)[:, None]
    xx = torch.linspace(-1, 1, width, device=device)[None]
    angle, strength = rng.uniform(0, 2 * np.pi), rng.uniform(0.0, 0.15)
    ramp = 1 + strength * (np.cos(angle) * xx + np.sin(angle) * yy)
    grain = torch.from_numpy(rng.normal(0, 0.012, (1, 1, height, width)).astype(np.float32))
    grain = _blur(grain.to(device), 0.7)[0, 0]
    photo = (shade * ramp)[..., None] * tint + grain[..., None]
    return {"photo": photo.clamp(0, 1), "uv": uv, "inside": share > 0.5}


def render_doc3d_item(
    geometry: dict[str, np.ndarray],
    texture: np.ndarray,
    background: Any,
    rng: np.random.Generator,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """``render_item`` for a Doc3D geometry: the page is drawn through its grid refined to
    MAP_HW nodes, lit from its 3-D shape, and that refined grid is the stored map.
    ``background`` is a function of the photo size. Raises ValueError when the page
    turns over in view."""
    out_hw = tuple(settings["photo_hw"])
    device = settings.get("device", "cpu")
    flipped = flip_doc3d(geometry, bool(rng.random() < 0.5), bool(rng.random() < 0.5))
    shade = float(rng.uniform(*settings.get("shade", (0.65, 0.85))))
    margin = rng.uniform(*settings.get("margin", (0.02, 0.15)), size=4)
    fine = _refine(flipped["grid"], MAP_HW, device)
    if folded(fine):
        raise ValueError("the page turns over in view")
    x0, y0, x1, y1 = crop_window(fine.cpu().numpy(), None, out_hw, margin)
    scale = fine.new_tensor([out_hw[1] / (x1 - x0), out_hw[0] / (y1 - y0)])
    points = (fine - fine.new_tensor([x0, y0]) + 0.5) * scale - 0.5
    light = paper_light(_refine(flipped["grid3d"], MAP_HW, device), rng)
    view = doc3d_view(points, light, out_hw, rng)
    photo = compose(view, texture, background(out_hw), shade)
    values = pixels_to_flattener(points.cpu().numpy().astype(np.float64), out_hw)
    return photo, grid_to_map(values, MAP_HW)


def run_capture(task: dict[str, Any]) -> dict[str, Any]:
    """Items of the captures (or Doc3D geometries) of one task."""
    import torch

    torch.set_num_threads(1)
    settings, output = task["settings"], Path(task["output"])
    split = settings["split"]
    doc3d = settings.get("geometry") == "doc3d_grid"
    prefix, source = ("doc3d", DOC3D_SOURCE) if doc3d else ("uvdoc", SOURCE)
    folder = output / "files" / split
    folder.mkdir(parents=True, exist_ok=True)
    build = _crumple_build()
    device = settings.get("device") or "cpu"
    rows, skipped = [], 0
    for group in task["captures"]:
        capture_id = group["capture"]
        try:
            if doc3d:
                capture = load_doc3d(Path(settings["raw_root"]), capture_id)
            else:
                capture = load_capture(Path(settings["raw_root"]), capture_id)
        except (OSError, KeyError, ValueError):
            skipped += len(group["indices"])
            continue
        for index, record in zip(group["indices"], group["items"], strict=True):
            rng = np.random.default_rng(_seed(settings["name"], split, index))
            try:
                own = rng.random() < settings.get("own_texture_prob", 0.0)
                if own:
                    texture = build.own_texture(rng, f"{settings['name']}-{split}-{index}")
                else:
                    texture = build.flat_texture(record, device=device)
                    if settings.get("clean_texture"):
                        texture = clean_texture(texture, device)
                if not doc3d and rng.random() < settings.get("capture_background_prob", 0.0):
                    background = None
                else:

                    def background(size, rng=rng):
                        if rng.random() < settings.get("pattern_background_prob", 0.0):
                            return pattern_background(rng, size)
                        bank = build._bank(settings["background_dir"])
                        if bank is not None and len(bank) and rng.random() < 0.75:
                            drawn = bank.draw(rng, size)
                        else:
                            drawn = procedural_background(rng, size)
                        return np.clip(drawn, 0, 255).astype(np.uint8)

                render = render_doc3d_item if doc3d else render_item
                photo, forward = render(capture, texture, background, rng, settings)
            except (OSError, ValueError, RuntimeError):
                skipped += 1
                continue
            finally:
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            if not np.isfinite(forward).all() or np.abs(forward).max() > 1.0:
                skipped += 1
                continue
            ident = f"{prefix}-{split}-{index:07d}"
            buffer = io.BytesIO()
            Image.fromarray(photo).save(buffer, "JPEG", quality=92)
            image_bytes = buffer.getvalue()
            (folder / f"{ident}.jpg").write_bytes(image_bytes)
            map_buffer = io.BytesIO()
            np.save(map_buffer, forward.astype(np.float16))
            map_bytes = map_buffer.getvalue()
            (folder / f"{ident}.bm.npy").write_bytes(map_bytes)
            rows.append(
                {
                    "schema": "flattener-dewarp-sample-1",
                    "sample_id": ident,
                    "split": split,
                    "image": f"files/{split}/{ident}.jpg",
                    "backward_map": f"files/{split}/{ident}.bm.npy",
                    "backward_map_convention": "flattener",
                    "image_size": list(settings["photo_hw"]),
                    "track": record["track"],
                    "license_class": "own" if own else record["license_class"],
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "image_bytes": len(image_bytes),
                    "backward_map_sha256": hashlib.sha256(map_bytes).hexdigest(),
                    "backward_map_bytes": len(map_bytes),
                    "texture_sample_id": None if own else record["sample_id"],
                    "capture": capture_id,
                    "geometry_source": source,
                }
            )
    return {"key": task["key"], "rows": rows, "skipped": skipped}


def captures(raw: Path, exclude: Path | None) -> list[str]:
    """Raw capture ids, less the benchmark's."""
    ids = sorted(p.stem for p in (raw / "samples" / "rgb").glob("*.png"))
    if exclude is not None:
        held = {
            json.loads(p.read_text(encoding="utf-8")).get("sample_id")
            for p in (exclude / "metadata_geom").glob("*.json")
        }
        ids = [capture for capture in ids if capture not in held]
    return ids


def split_captures(ids: list[str], spec: dict[str, Any], group=None) -> dict[str, list[str]]:
    """Captures per split: ``val_share`` of them to val by a hash of the id (or of
    ``group(id)``, so a group stays together) and ``split_name`` (the spec's name when
    absent), the rest to train."""
    share = spec.get("val_share", 0.05)
    name = spec.get("split_name", spec["name"])
    key = group or (lambda capture: capture)
    val = [c for c in ids if _seed(name, "split", key(c)) % 1000 < share * 1000]
    held = set(val)
    return {"val": val, "train": [c for c in ids if c not in held]}


def build(
    spec_path: Path, output: Path, *, workers: int, limit: int | None, device: str = "cpu"
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema") != BUILD_SCHEMA:
        raise SystemExit(f"error: {spec_path}: schema must be {BUILD_SCHEMA}")
    base = spec_path.resolve().parent
    raw = Path(expand_path(spec["raw_root"]))
    exclude = (
        Path(expand_path(spec["exclude_benchmark"])) if spec.get("exclude_benchmark") else None
    )
    doc3d = spec.get("geometry", "uvdoc_raw") == "doc3d_grid"
    ids = doc3d_ids(raw) if doc3d else captures(raw, exclude)
    if not ids:
        raise SystemExit(f"error: no captures or geometries under {raw}")
    by_split = split_captures(ids, spec, doc3d_group if doc3d else None)
    chunk = int(spec.get("chunk", 1))
    build_module = _crumple_build()
    orient = build_module._orient_builder()
    dewarp_spec = Path(base, expand_path(spec["dewarp_spec"])).resolve()
    _, pages, described = orient.index_dewarp_spec(str(dewarp_spec), (45, 31))
    background_dir = expand_path(spec["background_dir"]) if spec.get("background_dir") else None
    output.mkdir(parents=True, exist_ok=True)
    parts = output / "parts"
    parts.mkdir(exist_ok=True)
    tasks = []
    for split, count in spec["count"].items():
        count = min(count, limit) if limit else count
        geometries = by_split[split]
        records = pages[split]
        rng = np.random.default_rng(_seed(spec["name"], split, "pick"))
        picks = rng.choice(len(records), size=count, replace=count > len(records))
        order = rng.permutation(len(geometries))
        assigned: dict[str, list[int]] = {}
        for index in range(count):  # every capture gets items in turn
            assigned.setdefault(geometries[order[index % len(geometries)]], []).append(index)
        settings = {key: spec.get(key) for key in ("name", "photo_hw", "own_texture_prob",
                                                   "capture_background_prob", "shade",
                                                   "margin", "geometry", "clean_texture",
                                                   "pattern_background_prob",
                                                   "recipe")}  # fmt: skip
        settings.update(split=split, raw_root=str(raw), background_dir=background_dir,
                        device=device)  # fmt: skip
        groups = [{"capture": capture, "indices": indices,
                   "items": [records[int(picks[i])] for i in indices]}
                  for capture, indices in assigned.items()]  # fmt: skip
        for start in range(0, len(groups), chunk):
            part = groups[start : start + chunk]
            name = part[0]["capture"] if chunk == 1 else f"{start // chunk:05d}"
            tasks.append({"key": f"{split}-{name.replace('/', '_')}", "settings": settings,
                          "output": str(output), "captures": part})  # fmt: skip
    pending = [t for t in tasks if not (parts / f"{t['key']}.jsonl").is_file()]
    started, done = time.perf_counter(), 0
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers) if workers > 1 else _Inline() as pool:
        for result in pool.imap_unordered(run_capture, pending):
            lines = "".join(json.dumps(row) + "\n" for row in result["rows"])
            (parts / f"{result['key']}.jsonl").write_text(lines, encoding="utf-8")
            done += 1
            if done % 20 == 0 or done == len(pending):
                rate = done / max(time.perf_counter() - started, 1e-9)
                print(json.dumps({"tasks": done, "of": len(pending), "per_sec": round(rate, 2)}))
    rows = []
    for task in tasks:
        rows.extend(
            json.loads(line)
            for line in (parts / f"{task['key']}.jsonl").read_text(encoding="utf-8").splitlines()
        )
    rows.sort(key=lambda row: (row["split"], row["sample_id"]))
    (output / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    record = {
        "schema": BUILD_SCHEMA,
        "spec": spec,
        "spec_sha256": hashlib.sha256(canonical_json(spec)).hexdigest(),
        "captures": {split: len(by_split[split]) for split in by_split},
        "excluded_benchmark": str(exclude) if exclude else None,
        "geometry_source": DOC3D_SOURCE if doc3d else SOURCE,
        "pages": described,
        "records": len(rows),
    }
    write_json(output / "build.json", record)
    return {"records": len(rows), "captures": record["captures"]}


class _Inline:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def imap_unordered(self, function, items):
        return map(function, items)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="items per split, for a trial")
    parser.add_argument("--device", default="cpu", help="where pages are laid (cuda on home)")
    args = parser.parse_args(argv)
    result = build(args.spec, args.output, workers=args.workers, limit=args.limit,
                   device=args.device)  # fmt: skip
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
