"""Random-init OSNet-lite appearance embedder.

The block follows the omni-scale residual in Zhou et al., arXiv 1905.00953.
A 1x1 convolution reduces the width to a quarter of the stage width.
Four lite streams follow the exponent:
stream t stacks t lite 3x3 layers, and each later stream reuses the previous
stream's output. A lite 3x3 is a pointwise 1x1 and then a depthwise 3x3
with padding 1, so every stream keeps the same spatial size. One aggregation
gate is shared by the streams in a block. The gate is global average
pooling, a ReLU hidden layer at the paper's reduction ratio of 16 (at least
one unit), and a sigmoid. The sigmoid outputs channel weights. The weighted
streams are summed, a 1x1 convolution restores the width, and the shortcut
is added.

The network is a 7x7 stride-2 stem, two of those stages, global average
pooling, and a linear map to ``dim``. Each output row is L2-normalised.
Widths are 32, then 32, then 64, which is a narrow cut of the paper's
table. The constructor takes ``seed`` and ``dim`` only. It does not read a
parameter file and it does not open a connection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from trace_search.reid.nn.layers import (
    BatchNorm2d,
    Conv2d,
    DepthwiseConv2d,
    GlobalAvgPool,
    Linear,
    PointwiseConv2d,
    ReLU,
    Sigmoid,
)
from trace_search.reid.types import AppearanceEmbedder

_STREAMS = 4
_BOTTLENECK = 4
_GATE_REDUCTION = 16
_HEIGHT = 64
_WIDTH = 32
_CHANNELS = 3
_STEM_CHANNELS = 32
_STAGE_CHANNELS = (32, 64)
_STEM_KERNEL = 7
_STEM_STRIDE = 2
_STEM_PADDING = 3


class _Lite3x3:
    """Pointwise 1x1 then depthwise 3x3, each followed by norm and ReLU.

    The depthwise layer uses padding 1 so the spatial size stays the same.
    """

    def __init__(self, channels: int, rng: np.random.Generator) -> None:
        self.pointwise = PointwiseConv2d(channels, channels, rng)
        self.pointwise_norm = BatchNorm2d(channels, rng).eval()
        self.depthwise = DepthwiseConv2d(channels, rng, padding=1)
        self.depthwise_norm = BatchNorm2d(channels, rng).eval()
        self.relu = ReLU()

    def forward(self, batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """One lite 3x3. The batch is not modified."""
        mixed = self.pointwise_norm.forward(self.pointwise.forward(batch))
        spatial = self.depthwise_norm.forward(self.depthwise.forward(self.relu.forward(mixed)))
        return self.relu.forward(spatial)


class _AggregationGate:
    """Shared channel gate: pool, ReLU hidden layer, sigmoid."""

    def __init__(self, channels: int, rng: np.random.Generator) -> None:
        hidden = _gate_hidden(channels)
        self.pool = GlobalAvgPool()
        self.reduce = Linear(channels, hidden, rng)
        self.expand = Linear(hidden, channels, rng)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return one weight per channel, shape ``(batch, channels)``."""
        pooled = self.pool.forward(batch)
        hidden = self.relu.forward(self.reduce.forward(pooled))
        return self.sigmoid.forward(self.expand.forward(hidden))


class _Passthrough:
    """Shortcut used when the width does not change."""

    def forward(self, batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the batch unchanged."""
        return batch


class _Projection:
    """1x1 shortcut used when a stage changes the width."""

    def __init__(self, in_channels: int, out_channels: int, rng: np.random.Generator) -> None:
        self.pointwise = PointwiseConv2d(in_channels, out_channels, rng)
        self.norm = BatchNorm2d(out_channels, rng).eval()

    def forward(self, batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project the shortcut to the stage width."""
        return self.norm.forward(self.pointwise.forward(batch))


class _OmniScaleBlock:
    """One omni-scale residual stage with four lite streams and one gate."""

    def __init__(self, in_channels: int, out_channels: int, rng: np.random.Generator) -> None:
        mid = out_channels // _BOTTLENECK
        self.reduce = PointwiseConv2d(in_channels, mid, rng)
        self.reduce_norm = BatchNorm2d(mid, rng).eval()
        self.streams = tuple(_Lite3x3(mid, rng) for _ in range(_STREAMS))
        self.gate = _AggregationGate(mid, rng)
        self.expand = PointwiseConv2d(mid, out_channels, rng)
        self.expand_norm = BatchNorm2d(out_channels, rng).eval()
        self.shortcut: _Passthrough | _Projection
        if in_channels == out_channels:
            self.shortcut = _Passthrough()
        else:
            self.shortcut = _Projection(in_channels, out_channels, rng)
        self.relu = ReLU()

    def forward(self, batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Residual omni-scale map. The batch is not modified."""
        skip = self.shortcut.forward(batch)
        reduced = self.relu.forward(self.reduce_norm.forward(self.reduce.forward(batch)))
        restored = self.expand_norm.forward(self.expand.forward(self._mix(reduced)))
        return self.relu.forward(_add(restored, skip))

    def _mix(self, hidden: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        carried = hidden
        parts: list[npt.NDArray[np.float64]] = []
        for stream in self.streams:
            carried = stream.forward(carried)
            parts.append(_scale_channels(carried, self.gate.forward(carried)))
        return _sum_parts(parts)


class _Network:
    """Stem, two omni-scale stages, pooling, and the embedding map."""

    def __init__(self, rng: np.random.Generator, dim: int) -> None:
        self.stem = Conv2d(
            _CHANNELS,
            _STEM_CHANNELS,
            _STEM_KERNEL,
            rng,
            stride=_STEM_STRIDE,
            padding=_STEM_PADDING,
        )
        self.stem_norm = BatchNorm2d(_STEM_CHANNELS, rng).eval()
        self.stem_relu = ReLU()
        first = _OmniScaleBlock(_STEM_CHANNELS, _STAGE_CHANNELS[0], rng)
        second = _OmniScaleBlock(_STAGE_CHANNELS[0], _STAGE_CHANNELS[1], rng)
        self.stages: tuple[_OmniScaleBlock, _OmniScaleBlock] = (first, second)
        self.pool = GlobalAvgPool()
        self.embed = Linear(_STAGE_CHANNELS[1], dim, rng)

    def forward(self, batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map an NCHW batch to ``(batch, dim)`` before L2 normalisation."""
        hidden = self.stem_relu.forward(self.stem_norm.forward(self.stem.forward(batch)))
        for stage in self.stages:
            hidden = stage.forward(hidden)
        return self.embed.forward(self.pool.forward(hidden))


class OsnetLiteEmbedder:
    """Random-init omni-scale embedder.

    Similarity scores from this embedder have no accuracy meaning and may
    not be reported as one (PRD §0).
    """

    placeholder: bool = True

    def __init__(self, seed: int = 0, dim: int = 256) -> None:
        self.dim = _require_dim(dim)
        self._seed = seed
        self._network = _Network(np.random.default_rng(seed), self.dim)

    @property
    def name(self) -> str:
        """Short embedder label. It is read-only and is not a person name."""
        return "osnet-lite-random-init"

    @property
    def stages(self) -> tuple[_OmniScaleBlock, _OmniScaleBlock]:
        """The two omni-scale stages, stem stage first."""
        return self._network.stages

    def embed(self, crops: Sequence[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        """Map HxWx3 uint8 crops to ``dim`` L2-normalised float32 rows."""
        if len(crops) == 0:
            return _empty(self.dim)
        batch = _batch_from_crops(crops)
        return _unit_rows(self._network.forward(batch))


def _as_appearance_embedder(embedder: OsnetLiteEmbedder) -> AppearanceEmbedder:
    """Return ``embedder`` as an ``AppearanceEmbedder``.

    ``mypy src/ --strict`` checks this assignment. CI does not type-check tests.
    """
    checked: AppearanceEmbedder = embedder
    return checked


def _require_dim(dim: object) -> int:
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise ValueError(f"dim must be a positive int, got {dim!r}")
    return dim


def _gate_hidden(channels: int) -> int:
    """Hidden width of the gate. The paper's ratio is 16, and at least 1."""
    hidden = channels // _GATE_REDUCTION
    if hidden < 1:
        return 1
    return hidden


def _scale_channels(
    features: npt.NDArray[np.float64], gates: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Hadamard product of each channel with its gate weight."""
    batch = int(gates.shape[0])
    channels = int(gates.shape[1])
    scale = gates.reshape(batch, channels, 1, 1)
    product = np.asarray(features * scale, dtype=np.float64)
    return cast(npt.NDArray[np.float64], np.ascontiguousarray(product))


def _add(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    total = np.asarray(left + right, dtype=np.float64)
    return cast(npt.NDArray[np.float64], np.ascontiguousarray(total))


def _sum_parts(parts: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    total = parts[0]
    for part in parts[1:]:
        total = _add(total, part)
    return total


def _empty(dim: int) -> npt.NDArray[np.float32]:
    empty = np.zeros((0, dim), dtype=np.float32)
    return cast(npt.NDArray[np.float32], empty)


def _batch_from_crops(crops: Sequence[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float64]:
    """Stack crops into one new NCHW float64 batch. Crops are not modified."""
    frames = [_resize_chw(crop) for crop in crops]
    stacked = np.stack(frames, axis=0)
    copied = np.array(stacked, dtype=np.float64, order="C", copy=True)
    return cast(npt.NDArray[np.float64], copied)


def _resize_chw(crop: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    _require_crop(crop)
    contiguous = np.ascontiguousarray(crop)
    resized = cv2.resize(contiguous, (_WIDTH, _HEIGHT))
    image = np.asarray(resized, dtype=np.float64) / 255.0
    chw = np.transpose(image, (2, 0, 1))
    return cast(npt.NDArray[np.float64], np.ascontiguousarray(chw))


def _require_crop(crop: npt.NDArray[np.uint8]) -> None:
    if not isinstance(crop, np.ndarray):
        raise ValueError(f"crop must be a numpy array, got {type(crop).__name__}")
    if crop.dtype != np.uint8:
        raise ValueError(f"crop must be uint8, got {crop.dtype}")
    if crop.size == 0:
        raise ValueError("crop must have a non-zero height and width")
    if crop.ndim != 3:
        raise ValueError(f"crop must be HxWx3, got ndim {crop.ndim}")
    channels = int(crop.shape[2])
    if channels != _CHANNELS:
        raise ValueError(f"crop must have 3 channels, got {channels}")


def _unit_rows(projected: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
    norms = cast(npt.NDArray[np.float64], np.linalg.norm(projected, axis=1))
    if _any_unusable(norms):
        raise ValueError("embedding row is zero or non-finite and cannot be normalised")
    scale = norms.reshape(-1, 1)
    unit = np.asarray(projected / scale, dtype=np.float32)
    return cast(npt.NDArray[np.float32], np.ascontiguousarray(unit))


def _any_unusable(norms: npt.NDArray[np.float64]) -> bool:
    flat = np.asarray(norms, dtype=np.float64).reshape(-1)
    for value in flat:
        number = float(value)
        if not math.isfinite(number) or number == 0.0:
            return True
    return False
