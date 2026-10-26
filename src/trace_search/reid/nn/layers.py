"""Forward-only convolution, normalisation, and pooling in numpy.

Weights are drawn from the caller-supplied ``numpy.random.Generator`` with
He normal initialisation: zero mean and standard deviation ``sqrt(2 / fan_in)``.
``fan_in`` is the number of inputs that one output value uses. Shifts start
at zero. Batch-norm scale starts at one, its shift starts at zero, its
running mean starts at zero, and its running variance starts at one.

No constructor takes an array, a filesystem path, or a saved parameter file.
This module does not fetch parameters and does not open a connection.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

_DEFAULT_EPS = 1e-5
_DEFAULT_MOMENTUM = 0.1

__all__ = [
    "BatchNorm2d",
    "Conv2d",
    "DepthwiseConv2d",
    "GlobalAvgPool",
    "Linear",
    "PointwiseConv2d",
    "ReLU",
    "Sigmoid",
]


def _require_generator(rng: object) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    raise TypeError(
        f"expected numpy.random.Generator, got {type(rng).__name__}; an array or a path is refused"
    )


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive int, got {value!r}")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative int, got {value!r}")
    return value


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _require_positive_float(value: object, label: str) -> float:
    number = _finite_float(value)
    if number is None or number <= 0.0:
        raise ValueError(f"{label} must be a finite float greater than 0, got {value!r}")
    return number


def _require_momentum(value: object) -> float:
    number = _finite_float(value)
    if number is None or number < 0.0 or number > 1.0:
        raise ValueError(f"momentum must be a finite float from 0 to 1, got {value!r}")
    return number


def _pair(value: int | tuple[int, ...], label: str, *, positive: bool) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{label} must be an int or a pair of ints, got {value!r}")
        first: object = value[0]
        second: object = value[1]
    else:
        first = value
        second = value
    check = _require_positive_int if positive else _require_nonnegative_int
    return check(first, label), check(second, label)


def _f64(array: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a new contiguous float64 array. The caller's object is not reused."""
    values = np.asarray(array, dtype=np.float64)
    copied = np.array(values, dtype=np.float64, order="C", copy=True)
    return cast(npt.NDArray[np.float64], copied)


def _zeros(shape: tuple[int, ...]) -> npt.NDArray[np.float64]:
    return _f64(np.zeros(shape, dtype=np.float64))


def _he_weight(
    rng: np.random.Generator, shape: tuple[int, ...], fan_in: int
) -> npt.NDArray[np.float64]:
    std = math.sqrt(2.0 / fan_in)
    return _f64(rng.standard_normal(shape) * std)


def _as_float(batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return _f64(batch)


def _as_nchw(batch: npt.NDArray[np.float64], channels: int) -> npt.NDArray[np.float64]:
    data = _as_float(batch)
    if data.ndim != 4:
        raise ValueError(f"expected ndim 4, got {data.ndim}")
    found = int(data.shape[1])
    if found != channels:
        raise ValueError(f"expected {channels} channels, got {found}")
    return data


def _pad(data: npt.NDArray[np.float64], pad_h: int, pad_w: int) -> npt.NDArray[np.float64]:
    padded = np.pad(
        data,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0.0,
    )
    return _f64(padded)


def _kernel_windows(
    padded: npt.NDArray[np.float64], kernel_h: int, kernel_w: int
) -> npt.NDArray[np.float64]:
    """im2col windows of shape ``(N, C, rows, cols, kernel_h, kernel_w)``."""
    by_row = sliding_window_view(padded, kernel_h, axis=2)
    by_col = sliding_window_view(by_row, kernel_w, axis=3)
    return cast(npt.NDArray[np.float64], by_col)


def _conv2d(
    data: npt.NDArray[np.float64],
    weight: npt.NDArray[np.float64],
    bias: npt.NDArray[np.float64],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> npt.NDArray[np.float64]:
    """im2col convolution. Windows come from ``sliding_window_view``.

    Callers pass an NCHW batch whose channel count matches ``weight``.
    """
    batch = int(data.shape[0])
    channels = int(data.shape[1])
    height = int(data.shape[2])
    width = int(data.shape[3])
    out_channels = int(weight.shape[0])
    kernel_h = int(weight.shape[2])
    kernel_w = int(weight.shape[3])
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    out_h = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (width + 2 * pad_w - kernel_w) // stride_w + 1
    if out_h < 1 or out_w < 1:
        raise ValueError(f"output spatial size must be positive, got {(out_h, out_w)}")
    windows = _kernel_windows(_pad(data, pad_h, pad_w), kernel_h, kernel_w)
    strided = windows[:, :, ::stride_h, ::stride_w, :, :]
    columns = np.ascontiguousarray(strided.transpose(0, 2, 3, 1, 4, 5)).reshape(
        batch, out_h, out_w, channels * kernel_h * kernel_w
    )
    matrix = weight.reshape(out_channels, channels * kernel_h * kernel_w)
    product = columns @ matrix.T
    shifted = product + bias.reshape(1, 1, 1, out_channels)
    return _f64(np.transpose(shifted, (0, 3, 1, 2)))


def _depthwise_conv2d(
    data: npt.NDArray[np.float64],
    weight: npt.NDArray[np.float64],
    bias: npt.NDArray[np.float64],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> npt.NDArray[np.float64]:
    """Depthwise 3x3 as one im2col and one einsum over the kernel taps.

    ``weight`` has shape ``(channels, 3, 3)``. Channels are not mixed and
    there is no Python loop over them.
    """
    height = int(data.shape[2])
    width = int(data.shape[3])
    kernel_h = int(weight.shape[1])
    kernel_w = int(weight.shape[2])
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    out_h = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (width + 2 * pad_w - kernel_w) // stride_w + 1
    if out_h < 1 or out_w < 1:
        raise ValueError(f"output spatial size must be positive, got {(out_h, out_w)}")
    windows = _kernel_windows(_pad(data, pad_h, pad_w), kernel_h, kernel_w)
    strided = windows[:, :, ::stride_h, ::stride_w, :, :]
    product = np.einsum("nchwjk,cjk->nchw", strided, weight)
    shifted = product + bias.reshape(1, -1, 1, 1)
    return _f64(shifted)


def _moments(
    data: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Population mean and variance over the batch and both spatial axes."""
    count = int(data.shape[0]) * int(data.shape[2]) * int(data.shape[3])
    total = data.sum(axis=(0, 2, 3), dtype=np.float64)
    mean = _f64(total / count)
    centred = data - mean.reshape(1, -1, 1, 1)
    square = np.square(centred).sum(axis=(0, 2, 3), dtype=np.float64)
    return mean, _f64(square / count), count


def _normalize(
    data: npt.NDArray[np.float64],
    mean: npt.NDArray[np.float64],
    var: npt.NDArray[np.float64],
    weight: npt.NDArray[np.float64],
    bias: npt.NDArray[np.float64],
    eps: float,
) -> npt.NDArray[np.float64]:
    mu = mean.reshape(1, -1, 1, 1)
    sigma = np.sqrt(var.reshape(1, -1, 1, 1) + eps)
    scale = weight.reshape(1, -1, 1, 1)
    shift = bias.reshape(1, -1, 1, 1)
    return _f64(scale * (data - mu) / sigma + shift)


def _unbiased_scale(count: int) -> float:
    if count > 1:
        return count / (count - 1)
    return 1.0


def _update_running(
    running_mean: npt.NDArray[np.float64],
    running_var: npt.NDArray[np.float64],
    batch_mean: npt.NDArray[np.float64],
    batch_var: npt.NDArray[np.float64],
    momentum: float,
    count: int,
) -> None:
    keep = 1.0 - momentum
    unbiased = batch_var * _unbiased_scale(count)
    running_mean *= keep
    running_mean += momentum * batch_mean
    running_var *= keep
    running_var += momentum * unbiased


def _sigmoid(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Stable logistic. Each branch only exponentiates a non-positive value."""
    out = np.empty(data.shape, dtype=np.float64)
    positive = data >= 0.0
    pos = np.exp(-data[positive])
    out[positive] = 1.0 / (1.0 + pos)
    neg = np.exp(data[~positive])
    out[~positive] = neg / (1.0 + neg)
    return _f64(out)


class Conv2d:
    """Dense 2D convolution.

    The input is zero-padded, then each kernel window is read with
    ``sliding_window_view`` and multiplied by the kernel (im2col).
    ``fan_in`` is ``in_channels * kernel_height * kernel_width``.
    ``kernel_size``, ``stride``, and ``padding`` are ``(height, width)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        rng: np.random.Generator,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
    ) -> None:
        generator = _require_generator(rng)
        self.in_channels = _require_positive_int(in_channels, "in_channels")
        self.out_channels = _require_positive_int(out_channels, "out_channels")
        self.kernel_size = _pair(kernel_size, "kernel_size", positive=True)
        self.stride = _pair(stride, "stride", positive=True)
        self.padding = _pair(padding, "padding", positive=False)
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        shape = (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        self.weight = _he_weight(generator, shape, fan_in)
        self.bias = _zeros((self.out_channels,))

    def forward(self, batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convolve an NCHW batch. The batch is not modified."""
        data = _as_nchw(batch, self.in_channels)
        return _conv2d(data, self.weight, self.bias, self.stride, self.padding)


class DepthwiseConv2d:
    """3x3 depthwise convolution. Each channel has its own kernel.

    ``fan_in`` is 9. Channels are not mixed.
    """

    def __init__(
        self,
        channels: int,
        rng: np.random.Generator,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
    ) -> None:
        generator = _require_generator(rng)
        self.channels = _require_positive_int(channels, "channels")
        self.kernel_size = (3, 3)
        self.stride = _pair(stride, "stride", positive=True)
        self.padding = _pair(padding, "padding", positive=False)
        self.weight = _he_weight(generator, (self.channels, 3, 3), 3 * 3)
        self.bias = _zeros((self.channels,))

    def forward(self, batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convolve each channel with its own 3x3 kernel."""
        data = _as_nchw(batch, self.channels)
        return _depthwise_conv2d(data, self.weight, self.bias, self.stride, self.padding)


class PointwiseConv2d:
    """1x1 convolution. ``fan_in`` is ``in_channels``."""

    def __init__(self, in_channels: int, out_channels: int, rng: np.random.Generator) -> None:
        generator = _require_generator(rng)
        self.in_channels = _require_positive_int(in_channels, "in_channels")
        self.out_channels = _require_positive_int(out_channels, "out_channels")
        self.kernel_size = (1, 1)
        self.weight = _he_weight(generator, (self.out_channels, self.in_channels), self.in_channels)
        self.bias = _zeros((self.out_channels,))

    def forward(self, batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Mix channels at each spatial site."""
        data = _as_nchw(batch, self.in_channels)
        kernel = _f64(self.weight.reshape(self.out_channels, self.in_channels, 1, 1))
        return _conv2d(data, kernel, self.bias, (1, 1), (0, 0))


class BatchNorm2d:
    """Per-channel batch normalisation with an affine scale and shift.

    Training uses the population mean and variance of the current batch
    (batch axis and both spatial axes). It then blends those moments into
    the running estimates. The variance blend uses the unbiased batch
    variance when more than one element is present. Eval uses the running
    estimates and leaves them unchanged.

    The scale starts at one and the shift starts at zero.
    """

    def __init__(
        self,
        num_features: int,
        rng: np.random.Generator,
        eps: float = _DEFAULT_EPS,
        momentum: float = _DEFAULT_MOMENTUM,
    ) -> None:
        _require_generator(rng)
        self.num_features = _require_positive_int(num_features, "num_features")
        self.eps = _require_positive_float(eps, "eps")
        self.momentum = _require_momentum(momentum)
        self.training = True
        self.weight = _f64(np.ones(self.num_features, dtype=np.float64))
        self.bias = _zeros((self.num_features,))
        self.running_mean = _zeros((self.num_features,))
        self.running_var = _f64(np.ones(self.num_features, dtype=np.float64))

    def train(self) -> BatchNorm2d:
        """Use batch moments on the next forward."""
        self.training = True
        return self

    def eval(self) -> BatchNorm2d:
        """Use running moments on the next forward."""
        self.training = False
        return self

    def forward(self, batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Normalise an NCHW batch. Training also updates the running moments."""
        data = _as_nchw(batch, self.num_features)
        if self.training:
            mean, var, count = _moments(data)
            out = _normalize(data, mean, var, self.weight, self.bias, self.eps)
            _update_running(self.running_mean, self.running_var, mean, var, self.momentum, count)
            return out
        return _normalize(
            data, self.running_mean, self.running_var, self.weight, self.bias, self.eps
        )


class ReLU:
    """Rectified linear unit. Negative values become zero."""

    def __init__(self) -> None:
        """No parameters."""

    def forward(self, batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return ``max(batch, 0)``."""
        return _f64(np.maximum(_as_float(batch), 0.0))


class Sigmoid:
    """Elementwise logistic function."""

    def __init__(self) -> None:
        """No parameters."""

    def forward(self, batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return ``1 / (1 + exp(-batch))`` without overflowing."""
        return _sigmoid(_as_float(batch))


class GlobalAvgPool:
    """Average over height and width. The result has shape ``(batch, channels)``."""

    def __init__(self) -> None:
        """No parameters."""

    def forward(self, batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Average each channel of an NCHW batch."""
        data = _as_float(batch)
        if data.ndim != 4:
            raise ValueError(f"expected ndim 4, got {data.ndim}")
        height = int(data.shape[2])
        width = int(data.shape[3])
        if height < 1 or width < 1:
            raise ValueError(f"spatial size must be positive, got {(height, width)}")
        total = data.sum(axis=(2, 3), dtype=np.float64)
        return _f64(total / float(height * width))


class Linear:
    """Affine map ``y = x W^T + b``.

    A batch with more than two dimensions is flattened after the batch axis,
    so an NCHW tensor is accepted when ``in_features`` equals ``C * H * W``.
    ``fan_in`` is ``in_features``.
    """

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator) -> None:
        generator = _require_generator(rng)
        self.in_features = _require_positive_int(in_features, "in_features")
        self.out_features = _require_positive_int(out_features, "out_features")
        self.weight = _he_weight(generator, (self.out_features, self.in_features), self.in_features)
        self.bias = _zeros((self.out_features,))

    def forward(self, batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project a batch. Leading dimension is the batch."""
        matrix = _as_features(batch, self.in_features)
        return _f64(matrix @ self.weight.T + self.bias)


def _as_features(batch: npt.NDArray[np.float64], features: int) -> npt.NDArray[np.float64]:
    data = _as_float(batch)
    if data.ndim < 2:
        raise ValueError(f"expected at least 2 dimensions, got {data.ndim}")
    matrix = data.reshape(int(data.shape[0]), -1)
    found = int(matrix.shape[1])
    if found != features:
        raise ValueError(f"expected {features} features, got {found}")
    return _f64(matrix)
