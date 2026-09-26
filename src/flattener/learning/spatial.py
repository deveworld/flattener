"""Explicit CPU/CUDA/XPU coordinate models without implicit device transfers.

The spatial decoder keeps image features at each output-grid position. Legacy
architectures emit free normalized residuals; projective/surface architectures
evaluate bounded geometric parameters directly at requested pixel centers.
Every zero-initialized output head starts at identity. Construction consumes
the caller's torch RNG; callers needing isolation must use ``fork_rng``.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .model import CoordinatePredictor, ModelConfig, coarse_identity

MODEL_VERSION = "spatial-coordinate-1.0"
PAGE_MODEL_VERSION = "page-coordinate-1.0"
PROJECTIVE_MODEL_VERSION = "projective-coordinate-1.0"
SURFACE_MODEL_VERSION = "surface-coordinate-1.0"
ORIENTED_SURFACE_MODEL_VERSION = "oriented-surface-coordinate-1.0"


def _config(config: ModelConfig | None, architecture: str) -> ModelConfig:
    if config is None:
        config = ModelConfig()
    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig or None")
    # Revalidate before allocation even if a deserializer bypassed the frozen
    # dataclass constructor. These limits also bound checkpoint-driven models.
    config.__post_init__()
    if max(config.input_size) > 2048 or math.prod(config.input_size) > 1_048_576:
        raise ValueError("input_size exceeds the supported resource limit")
    if max(config.grid_size) > 256 or math.prod(config.grid_size) > 16_384:
        raise ValueError("grid_size exceeds the supported resource limit")
    if max(config.channels) > 256 or config.hidden > 2048:
        raise ValueError("channels or hidden exceeds the supported resource limit")
    channels = (3, *config.channels)
    count = sum(
        output * (9 * incoming + 1)
        for incoming, output in zip(channels, channels[1:])
    )
    if architecture in ("global", "page", "projective", "surface", "oriented_surface"):
        count += config.hidden * ((config.channels[-1] + (architecture == "page")) * 30 + 1)
        outputs = {"projective": 8, "surface": 48, "oriented_surface": 48}.get(
            architecture, 2 * math.prod(config.grid_size)
        )
        count += outputs * (config.hidden + 1)
        if architecture in ("page", "oriented_surface"):
            count += config.channels[0] * (9 * sum(config.channels) + 1)
            count += config.channels[0] + 1
        if architecture == "oriented_surface":
            count += 4 * (config.hidden + 1)
    else:
        incoming = sum(config.channels) + config.channels[-1] + 2
        count += config.hidden * (9 * incoming + 1)
        count += config.channels[0] * (9 * config.hidden + 1)
        count += 2 * (config.channels[0] + 1)
    if count > 20_000_000:
        raise ValueError("model exceeds the supported parameter limit")
    return config


def _validate_images(
    model: nn.Module, config: ModelConfig, images: torch.Tensor, *, check_values: bool = True
) -> None:
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if (
        images.device.type not in {"cpu", "cuda", "xpu"}
        or images.dtype != torch.float32
        or images.layout != torch.strided
    ):
        raise TypeError("images must be a strided CPU, CUDA or XPU float32 tensor")
    expected = (3, *config.input_size)
    if images.ndim != 4 or images.shape[0] < 1 or tuple(images.shape[1:]) != expected:
        raise ValueError(f"images must have shape (N, {expected}), with N >= 1")
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.dtype != torch.float32 or tensor.layout != torch.strided:
            raise TypeError("model parameters and buffers must be strided float32 tensors")
        if tensor.device != images.device:
            raise ValueError("model and images must be on the same CPU, CUDA or XPU device")
    if check_values:
        if not torch.isfinite(images).all().item():
            raise ValueError("images must contain only finite values")
        if (images < 0).any().item() or (images > 1).any().item():
            raise ValueError("images must have values in [0, 1]")


def _dense_size(size: tuple[int, int], coarse_size: tuple[int, int], batch: int):
    if (not isinstance(size, tuple) or len(size) != 2
            or any(type(value) is not int or not 2 <= value <= 8192 for value in size)):
        raise ValueError("output_size must be a (height,width) tuple with sides in [2,8192]")
    if any(out < coarse for out, coarse in zip(size, coarse_size, strict=True)):
        raise ValueError("output_size may not shrink either coarse grid axis")
    if batch * math.prod(size) > 16_000_000:
        raise ValueError("batched dense output exceeds 16000000 pixels")
    return size


def expand_coordinate_residual(
    residual: torch.Tensor, output_size: tuple[int, int], mode: str = "legacy"
) -> torch.Tensor:
    """Expand half-pixel residuals without clipping or changing their device.

    ``legacy`` is exactly the previous bilinear border-extension protocol.
    ``linear`` uses the nearest two coarse centers even outside their convex
    hull, so affine absolute maps remain affine across the full image extent.
    Linear extrapolation does not guarantee orientation for arbitrary fields.
    """
    if (not isinstance(residual, torch.Tensor) or residual.dtype != torch.float32
            or residual.device.type not in {"cpu", "cuda", "xpu"}
            or residual.layout != torch.strided):
        raise TypeError("residual must be a strided CPU, CUDA or XPU float32 tensor")
    if (residual.ndim != 4 or residual.shape[0] < 1 or residual.shape[1] != 2
            or min(residual.shape[-2:]) < 2 or max(residual.shape[-2:]) > 256
            or math.prod(residual.shape[-2:]) > 16_384):
        raise ValueError("residual must have bounded shape (N,2,Gh>=2,Gw>=2)")
    if not isinstance(mode, str) or mode not in {"legacy", "linear"}:
        raise ValueError("coordinate expansion mode must be legacy or linear")
    coarse_size = tuple(residual.shape[-2:])
    output_size = _dense_size(output_size, coarse_size, residual.shape[0])
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("residual must contain only finite values")
    if mode == "legacy":
        expanded = F.interpolate(residual, size=output_size, mode="bilinear", align_corners=False)
    else:
        expanded = residual
        # Index selection is bounded; interpolation weights deliberately are
        # not. Negative edge weights implement linear extrapolation, not clamp.
        for axis, (incoming, outgoing) in enumerate(zip(coarse_size, output_size), start=2):
            position = ((torch.arange(outgoing, device=residual.device, dtype=torch.float64)
                         + 0.5) * incoming / outgoing - 0.5)
            lower = position.floor().long().clamp(0, incoming - 2)
            weight = (position - lower).float()
            shape = [1, 1, 1, 1]
            shape[axis] = outgoing
            weight = weight.reshape(shape)
            low = expanded.index_select(axis, lower)
            high = expanded.index_select(axis, lower + 1)
            expanded = low + (high - low) * weight
    result = expanded.permute(0, 2, 3, 1) + coarse_identity(output_size).to(residual.device)[None]
    if not bool(torch.isfinite(result).all()):
        raise ValueError("coordinate expansion overflowed")
    return result


class DeviceCoordinatePredictor(CoordinatePredictor):
    """Legacy global architecture and state keys with explicit accelerator support.

    Only validation changes: the inherited layers and initialization match
    ``CoordinatePredictor`` exactly for the same configuration and RNG state.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(_config(config, "global"))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _validate_images(self, self.config, images)
        return self.head(self.encoder(images)).reshape(images.shape[0], 2, *self.config.grid_size)


class SpatialCoordinatePredictor(nn.Module):
    """Multiscale spatial decoder for N,2,Gh,Gw normalized residuals.

    RGB input must already be float32, full-extent resized and in [0, 1].
    ``model.to(device)`` and input placement are the caller's responsibility.
    This module does not resize source images, clip coordinates, or move data
    between devices. It is not a trained model or a quality certification.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = _config(config, "spatial")
        blocks = []
        incoming = 3
        for outgoing in self.config.channels:
            blocks.append(
                nn.Sequential(nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1), nn.ReLU())
            )
            incoming = outgoing
        self.encoder = nn.ModuleList(blocks)
        incoming = sum(self.config.channels) + self.config.channels[-1] + 2
        self.decoder = nn.Sequential(
            nn.Conv2d(incoming, self.config.hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.config.hidden, self.config.channels[0], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.config.channels[0], 2, 1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.register_buffer(
            "xy",
            coarse_identity(self.config.grid_size).permute(2, 0, 1).unsqueeze(0),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _validate_images(self, self.config, images)
        features = []
        encoded = images
        for block in self.encoder:
            encoded = block(encoded)
            features.append(
                F.interpolate(
                    encoded, size=self.config.grid_size, mode="bilinear", align_corners=False
                )
            )
        batch = images.shape[0]
        context = encoded.mean(dim=(-2, -1), keepdim=True).expand(
            -1, -1, *self.config.grid_size
        )
        return self.decoder(torch.cat((*features, self.xy.expand(batch, -1, -1, -1), context), 1))


class PageCoordinatePredictor(nn.Module):
    """Page-supervised multiscale attention feeding a global coordinate head.

    Page logits are at the second encoder level (ceil(input / 4)). Attention
    scales deep features by 0.5 + sigmoid(logits), retaining background context
    instead of hard-cropping. A predicted page is not a quality certification.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = _config(config, "page")
        blocks, incoming = [], 3
        for outgoing in self.config.channels:
            blocks.append(
                nn.Sequential(nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1), nn.ReLU())
            )
            incoming = outgoing
        self.encoder = nn.ModuleList(blocks)
        self.page_head = nn.Sequential(
            nn.Conv2d(sum(self.config.channels), self.config.channels[0], 3, padding=1),
            nn.ReLU(), nn.Conv2d(self.config.channels[0], 1, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 5))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear((self.config.channels[-1] + 1) * 30, self.config.hidden),
            nn.ReLU(),
            nn.Linear(self.config.hidden, 2 * self.config.grid_size[0] * self.config.grid_size[1]),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward_with_page(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return coordinate residuals and page logits from one encoder pass."""
        _validate_images(self, self.config, images)
        features, encoded = [], images
        for block in self.encoder:
            encoded = block(encoded)
            features.append(encoded)
        page_size = features[1].shape[-2:]
        page_features = [F.interpolate(f, size=page_size, mode="bilinear", align_corners=False)
                         for f in features]
        logits = self.page_head(torch.cat(page_features, dim=1))
        attention = F.interpolate(logits.sigmoid(), size=encoded.shape[-2:], mode="bilinear",
                                  align_corners=False)
        gated = torch.cat((encoded * (0.5 + attention), attention), dim=1)
        residual = self.head(self.pool(gated)).reshape(images.shape[0], 2, *self.config.grid_size)
        return residual, logits

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_with_page(images)[0]


class ProjectiveCoordinatePredictor(nn.Module):
    """Eight-DOF positive-orientation homography evaluated at dense centers.

    The encoder and hidden-head state names match the legacy global model;
    the final layer is a new identity-initialized parameter head. Parameters
    are ordered tx, ty, theta, log_sx, log_sy, shear, perspective_x, perspective_y.
    ``forward`` remains a coarse-residual compatibility view, NOT the correct
    dense decoder. Use ``forward_dense`` to retain the analytic geometry.
    """

    parameter_outputs = 8
    _constant_cache_limit = 4
    _constant_cache_max_bytes = 8 * 1024 * 1024

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        architecture = "projective" if self.parameter_outputs == 8 else "surface"
        self.config = _config(config, architecture)
        layers, incoming = [], 3
        for outgoing in self.config.channels:
            layers.extend((nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1), nn.ReLU()))
            incoming = outgoing
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((6, 5)))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(self.config.channels[-1] * 30, self.config.hidden), nn.ReLU(),
            nn.Linear(self.config.hidden, self.parameter_outputs),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        # Geometry constants have no learned/checkpoint state. Entries contain
        # one unbatched grid and (for surface models) its two Bernstein bases.
        self._constant_cache = OrderedDict()
        self._constant_cache_bytes = 0

    def _apply(self, fn, recurse=True):
        # A .to/.float/.double transition must release constants on the previous
        # device, including transitions that later fail the float32 contract.
        self._constant_cache.clear()
        self._constant_cache_bytes = 0
        return super()._apply(fn, recurse=recurse)

    def _parameters_for(self, images, *, check_values=True):
        # forward_dense validates once before allocating features.
        parameters = self.head(self.encoder(images))
        if check_values and not bool(torch.isfinite(parameters).all()):
            raise ValueError("coordinate parameter prediction must be finite")
        return parameters

    @staticmethod
    def _bernstein(coordinates, degree):
        value = (coordinates + 1) / 2
        return torch.stack([
            math.comb(degree, index) * value**index * (1 - value)**(degree - index)
            for index in range(degree + 1)
        ], dim=-1)

    def _decoder_constants(self, raw, size):
        # Keep the original CPU float64 -> float32 center formula and compute
        # bases on the destination device, exactly as the uncached decoder did.
        key = (size, raw.device, torch.float32)
        cached = self._constant_cache.get(key)
        if cached is not None:
            self._constant_cache.move_to_end(key)
            return cached
        # Inference-warmed constants must also be safe to save for a later
        # backward pass; inference tensors cannot be used for that purpose.
        with torch.inference_mode(False), torch.no_grad():
            with torch.device("cpu"):
                centers = coarse_identity(size)
            centers = centers.to(raw.device)
            constants = (centers,)
            if self.parameter_outputs == 48:
                constants += (self._bernstein(centers[0, :, 0], 4),
                              self._bernstein(centers[:, 0, 1], 3))
        size_bytes = sum(value.numel() * value.element_size() for value in constants)
        if size_bytes <= self._constant_cache_max_bytes:
            while self._constant_cache and (
                len(self._constant_cache) >= self._constant_cache_limit
                or self._constant_cache_bytes + size_bytes > self._constant_cache_max_bytes
            ):
                _, evicted = self._constant_cache.popitem(last=False)
                self._constant_cache_bytes -= sum(
                    value.numel() * value.element_size() for value in evicted
                )
            self._constant_cache[key] = constants
            self._constant_cache_bytes += size_bytes
        return constants

    def _decode(self, raw, size, *, check_values=True):
        constants = self._decoder_constants(raw, size)
        centers = constants[0]
        points = centers[None].expand(raw.shape[0], -1, -1, -1)
        if self.parameter_outputs == 48:
            coefficients = 0.2 * raw[:, 8:].reshape(-1, 2, 4, 5).tanh()
            dx = torch.linalg.vector_norm(torch.diff(coefficients, dim=3), dim=1).flatten(1)
            dy = torch.linalg.vector_norm(torch.diff(coefficients, dim=2), dim=1).flatten(1)
            # Bernstein derivative convexity gives column bounds Lx and Ly.
            # Norm, rather than sqrt(sum squares), has a finite zero gradient.
            bound = torch.linalg.vector_norm(torch.stack((2 * dx.amax(1),
                                                          1.5 * dy.amax(1)), dim=1), dim=1)
            coefficients = coefficients * (0.7 / bound.clamp_min(0.7))[:, None, None, None]
            bx, by = constants[1:]
            rows = torch.einsum("hi,ncij->nchj", by, coefficients)
            residual = torch.einsum("nchj,wj->nhwc", rows, bx)
            points = points + residual
        bounded = raw[:, :8].tanh()
        translation = bounded[:, :2]
        theta = math.pi * bounded[:, 2]
        scales = (3 * bounded[:, 3:5]).exp()
        shear = 2 * bounded[:, 5]
        perspective = 0.6 * bounded[:, 6:8] / (1 + bounded[:, 6:8].abs().sum(1, keepdim=True))
        # H(z)=t+A*z/(1+p.z), A=R(theta)[[exp(sx),shear],[0,exp(sy)]].
        # Translation is outside the fraction, preserving det(H)=det(A)>0.
        x = scales[:, 0, None, None] * points[..., 0] + shear[:, None, None] * points[..., 1]
        y = scales[:, 1, None, None] * points[..., 1]
        c, s = theta.cos()[:, None, None], theta.sin()[:, None, None]
        numerator = torch.stack((c * x - s * y, s * x + c * y), dim=-1)
        denominator = 1 + (perspective[:, None, None] * points).sum(-1, keepdim=True)
        result = translation[:, None, None] + numerator / denominator
        if check_values and not bool(torch.isfinite(result).all()):
            raise ValueError("direct coordinate decoder produced nonfinite coordinates")
        return result

    def _forward_dense_validated(
        self, images: torch.Tensor, output_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Internal dense-only path for prevalidated training/evaluation batches.

        The caller guarantees finite [0,1] inputs and checks the returned scalar
        device boolean together with the loss before backward. It includes raw
        parameter finiteness because tanh can conceal an infinite raw value.
        Metadata/resource guards remain; host synchronization and unused coarse
        geometry are omitted here only.
        """
        _validate_images(self, self.config, images, check_values=False)
        size = _dense_size(output_size, self.config.grid_size, images.shape[0])
        raw = self._parameters_for(images, check_values=False)
        dense = self._decode(raw, size, check_values=False)
        return dense, torch.isfinite(raw).all() & torch.isfinite(dense).all()

    def forward_dense(
        self, images: torch.Tensor, output_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return coarse residual and independently evaluated dense absolute map."""
        _validate_images(self, self.config, images)
        size = _dense_size(output_size, self.config.grid_size, images.shape[0])
        raw = self._parameters_for(images)
        coarse = self._decode(raw, self.config.grid_size)
        reference = self._decoder_constants(raw, self.config.grid_size)[0]
        residual = (coarse - reference).permute(0, 3, 1, 2).contiguous()
        dense = coarse if size == self.config.grid_size else self._decode(raw, size)
        return residual, dense

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_dense(images, self.config.grid_size)[0]


class SurfaceCoordinatePredictor(ProjectiveCoordinatePredictor):
    """Positive homography composed with a bounded smooth 4x5 Bernstein warp.

    Coefficients are 0.2*tanh(raw), normalized so sqrt(Lx²+Ly²)<=0.7.
    Thus ||Jr||₂<=0.7 and G=x+r is continuously injective and orientation
    preserving. Each discrete cell's two residual edge-difference columns
    are also bounded by Lx and Ly, so its four corner Jacobians are positive
    in exact arithmetic. The homography preserves those convex cells because
    its denominator stays positive over |G|<=1.2. This is not an absolute
    float32/renderer-pass guarantee; out-of-bounds and finite-precision gates
    remain mandatory. No coordinate or RGB clipping is applied.
    """

    parameter_outputs = 48


class OrientedSurfaceCoordinatePredictor(SurfaceCoordinatePredictor):
    """Four physical directions, page attention, and an analytic Surface map.

    This is a new architecture, not a reinterpretation of old Surface weights.
    Only encoder.* and head.1.* have the old backbone's parameter meaning.
    The new geometry head starts at identity. Direction is a hard argmax with
    first-index tie breaking: right/down/left/up in source pixel coordinates.
    A direction classification objective is required because argmax does not
    transmit coordinate-loss gradients to the direction head. The residual
    angle is bounded to +/- pi/4; maps from different classes are never averaged.

    ``photo_size`` is mandatory at each call: preview aspect ratio need not
    equal physical source aspect ratio. All returned maps and semantic corners
    are in the original normalized source frame. There is no canonical crop,
    coordinate clipping, or inferred output aspect ratio in this first stage.
    D^-1 R D preserves the positive determinant of the scale/shear transform,
    so the existing Bernstein/perspective orientation argument still applies
    in exact arithmetic. Finite, out-of-bounds, and renderer gates remain
    mandatory; neither valid geometry nor four corners certify page coverage.
    """

    requires_photo_size = True

    def __init__(self, config: ModelConfig | None = None):
        config = _config(config, "oriented_surface")
        super().__init__(config)
        self.page_head = nn.Sequential(
            nn.Conv2d(sum(config.channels), config.channels[0], 3, padding=1), nn.ReLU(),
            nn.Conv2d(config.channels[0], 1, 1),
        )
        self.orientation_head = nn.Linear(config.hidden, 4)
        nn.init.zeros_(self.orientation_head.weight)
        nn.init.zeros_(self.orientation_head.bias)

    @staticmethod
    def _photo_size(photo_size):
        if (not isinstance(photo_size, tuple) or len(photo_size) != 2
                or any(type(v) is not int or not 1 <= v <= 8192 for v in photo_size)
                or math.prod(photo_size) > 16_000_000):
            raise ValueError("photo_size must be a bounded (height,width) integer tuple")
        return photo_size

    def _geometry_features(self, images):
        features, encoded = [], images
        # Keep the old encoder.* state keys, but retain its pre-pooling scales.
        for index in (0, 2, 4):
            encoded = self.encoder[index + 1](self.encoder[index](encoded))
            features.append(encoded)
        page_size = features[1].shape[-2:]
        logits = self.page_head(torch.cat([
            F.interpolate(f, size=page_size, mode="bilinear", align_corners=False)
            for f in features
        ], dim=1))
        attention = F.interpolate(logits.sigmoid(), size=encoded.shape[-2:],
                                  mode="bilinear", align_corners=False)
        pooled = self.encoder[-1](encoded * (0.5 + attention))
        hidden = self.head[2](self.head[1](self.head[0](pooled)))
        return self.head[3](hidden), self.orientation_head(hidden), logits

    @staticmethod
    def _geometry_parameters(raw, orientation_logits, *, turns=None):
        coefficients = 0.2 * raw[:, 8:].reshape(-1, 2, 4, 5).tanh()
        dx = torch.linalg.vector_norm(torch.diff(coefficients, dim=3), dim=1).flatten(1)
        dy = torch.linalg.vector_norm(torch.diff(coefficients, dim=2), dim=1).flatten(1)
        bound = torch.linalg.vector_norm(torch.stack((2 * dx.amax(1),
                                                      1.5 * dy.amax(1)), dim=1), dim=1)
        coefficients = coefficients * (0.7 / bound.clamp_min(0.7))[:, None, None, None]
        bounded = raw[:, :8].tanh()
        delta = (math.pi / 4) * bounded[:, 2]
        turns = orientation_logits.argmax(1) if turns is None else turns
        # Exact quarter-turn signs, followed by the small continuous rotation.
        # Evaluating cos(k*pi/2) directly would introduce avoidable roundoff.
        quarter_cos = raw.new_tensor((1, 0, -1, 0))[turns]
        quarter_sin = raw.new_tensor((0, 1, 0, -1))[turns]
        cosine = quarter_cos * delta.cos() - quarter_sin * delta.sin()
        sine = quarter_sin * delta.cos() + quarter_cos * delta.sin()
        perspective = 0.6 * bounded[:, 6:8] / (
            1 + bounded[:, 6:8].abs().sum(1, keepdim=True))
        return (coefficients, bounded[:, :2], cosine, sine, (3 * bounded[:, 3:5]).exp(),
                2 * bounded[:, 5], perspective)

    @staticmethod
    def _project_oriented(points, parameters, photo_size):
        _, translation, cosine, sine, scales, shear, perspective = parameters
        shape = (-1, *([1] * (points.ndim - 2)))
        x = scales[:, 0].reshape(shape) * points[..., 0] + shear.reshape(shape) * points[..., 1]
        y = scales[:, 1].reshape(shape) * points[..., 1]
        cosine, sine = cosine.reshape(shape), sine.reshape(shape)
        height, width = photo_size
        # D^-1 R D U: rotation is physical, scales/shear are source-normalized.
        numerator = torch.stack((cosine * x - sine * (height / width) * y,
                                 sine * (width / height) * x + cosine * y), dim=-1)
        vector_shape = (*shape, 2)
        denominator = 1 + (perspective.reshape(vector_shape) * points).sum(-1, keepdim=True)
        return translation.reshape(vector_shape) + numerator / denominator

    def _decode_oriented(self, raw, parameters, size, photo_size):
        centers, bx, by = self._decoder_constants(raw, size)
        rows = torch.einsum("hi,ncij->nchj", by, parameters[0])
        residual = torch.einsum("nchj,wj->nhwc", rows, bx)
        return self._project_oriented(centers[None] + residual, parameters, photo_size)

    def _corner_coordinates(self, raw, parameters, photo_size):
        # At each flat physical boundary corner Bernstein reduces exactly to
        # one coefficient, not a sampled/extrapolated dense-grid pixel center.
        corners = raw.new_tensor(((-1, -1), (1, -1), (1, 1), (-1, 1)))
        residual = parameters[0][:, :, (0, 0, 3, 3), (0, 4, 4, 0)].permute(0, 2, 1)
        return self._project_oriented(corners[None] + residual, parameters, photo_size)

    def _forward_with_geometry_validated(
        self, images, output_size, *, photo_size, need_coarse=False,
        teacher_direction=None, teacher_valid=None,
    ):
        """Metadata-checked hot path; caller checks finite flag before backward.

        Inputs must already be finite float32 RGB in [0,1]. The returned device
        flag includes raw geometry/direction/page logits, dense and corner
        coordinates, so saturation cannot conceal nonfinite parameters.
        A paired teacher override is training-only. Fixed GT eligible classes
        control geometry, never the predicted logits. Missing GT falls back to
        argmax. Invalid eligible classes use safe indices and return false in
        the admission flag; they must never reach backward. Public/eval paths
        cannot request this override.
        """
        _validate_images(self, self.config, images, check_values=False)
        size = _dense_size(output_size, self.config.grid_size, images.shape[0])
        photo_size = self._photo_size(photo_size)
        if type(need_coarse) is not bool:
            raise TypeError("need_coarse must be bool")
        teacher = teacher_direction is not None or teacher_valid is not None
        if teacher:
            if not self.training:
                raise ValueError("teacher direction is allowed only in training mode")
            for name, value, dtype in (("teacher_direction", teacher_direction, torch.int64),
                                       ("teacher_valid", teacher_valid, torch.bool)):
                if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                        or value.layout != torch.strided or value.device != images.device
                        or value.shape != (images.shape[0],)):
                    raise ValueError(f"{name} must be a same-device strided {dtype} N-vector")
        # Bound the multiscale branch independently of the requested output.
        page_pixels = math.prod(tuple((side + 3) // 4 for side in self.config.input_size))
        if images.shape[0] * page_pixels * sum(self.config.channels) > 64_000_000:
            raise ValueError("batched multiscale features exceed 64000000 elements")
        raw, orientation, page = self._geometry_features(images)
        if teacher:
            in_range = (teacher_direction >= 0) & (teacher_direction < 4)
            teacher_admitted = (~teacher_valid | in_range).all()
            # Select safe values before any quarter-turn lookup. Out-of-range
            # eligible GT fails admission; no labels or supervision masks are
            # clamped to manufacture a valid training example.
            turns = torch.where(teacher_valid & in_range, teacher_direction, orientation.argmax(1))
            parameters = self._geometry_parameters(raw, orientation, turns=turns)
        else:
            parameters = self._geometry_parameters(raw, orientation)
        dense = self._decode_oriented(raw, parameters, size, photo_size)
        corners = self._corner_coordinates(raw, parameters, photo_size)
        residual = None
        if need_coarse:
            coarse = (dense if size == self.config.grid_size else self._decode_oriented(
                raw, parameters, self.config.grid_size, photo_size))
            reference = self._decoder_constants(raw, self.config.grid_size)[0]
            residual = (coarse - reference).permute(0, 3, 1, 2).contiguous()
        finite = (torch.isfinite(raw).all() & torch.isfinite(orientation).all()
                  & torch.isfinite(page).all() & torch.isfinite(dense).all()
                  & torch.isfinite(corners).all())
        if residual is not None:
            finite = finite & torch.isfinite(residual).all()
        if teacher:
            finite = finite & teacher_admitted
        return residual, dense, {
            "orientation_logits": orientation, "page_logits": page, "corners": corners,
        }, finite

    def forward_with_geometry(self, images, output_size, *, photo_size):
        """Return coarse residual, dense map, and original-frame auxiliary data."""
        _validate_images(self, self.config, images)
        residual, dense, auxiliary, finite = self._forward_with_geometry_validated(
            images, output_size, photo_size=photo_size, need_coarse=True)
        if not bool(finite):
            raise ValueError("oriented geometry prediction must contain only finite values")
        return residual, dense, auxiliary

    def forward_dense(self, images, output_size, *, photo_size):
        return self.forward_with_geometry(images, output_size, photo_size=photo_size)[:2]

    def _forward_dense_validated(self, images, output_size, *, photo_size):
        _, dense, _, finite = self._forward_with_geometry_validated(
            images, output_size, photo_size=photo_size)
        return dense, finite

    def forward(self, images, *, photo_size):
        return self.forward_dense(images, self.config.grid_size, photo_size=photo_size)[0]


def make_coordinate_model(
    architecture: str, config: ModelConfig | None = None
) -> (DeviceCoordinatePredictor | SpatialCoordinatePredictor | PageCoordinatePredictor
      | ProjectiveCoordinatePredictor | SurfaceCoordinatePredictor
      | OrientedSurfaceCoordinatePredictor):
    """Choose an explicit architecture; never infer or fall back to another."""
    if not isinstance(architecture, str) or architecture not in {
        "global", "spatial", "page", "projective", "surface", "oriented_surface"
    }:
        raise ValueError("unsupported coordinate architecture")
    if architecture == "global":
        return DeviceCoordinatePredictor(config)
    if architecture == "page":
        return PageCoordinatePredictor(config)
    if architecture == "projective":
        return ProjectiveCoordinatePredictor(config)
    if architecture == "surface":
        return SurfaceCoordinatePredictor(config)
    if architecture == "oriented_surface":
        return OrientedSurfaceCoordinatePredictor(config)
    return SpatialCoordinatePredictor(config)


def parameter_count(model: nn.Module) -> int:
    """Count all parameters (including frozen ones), excluding buffers."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    return sum(parameter.numel() for parameter in model.parameters())
