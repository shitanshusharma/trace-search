"""Forward-only numpy layers, checked against plain loops.

The 2x3x5x5 inputs are built in the test from a seeded generator.
Nothing is read from disk and nothing is written into the repo tree.
"""

from __future__ import annotations

import inspect
import math
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import trace_search.reid.nn as nn
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

_INPUT_SHAPE = (2, 3, 5, 5)
_LAYER_NAMES = (
    "Conv2d",
    "DepthwiseConv2d",
    "PointwiseConv2d",
    "BatchNorm2d",
    "Linear",
    "ReLU",
    "Sigmoid",
    "GlobalAvgPool",
)
_WEIGHTED = _LAYER_NAMES[:5]
_CLOSE = {"rtol": 1e-5, "atol": 1e-5}


def _nchw(seed: int) -> np.ndarray:
    """Seeded NCHW sample with a zero, a negative, and a positive entry."""
    data = np.random.default_rng(seed).standard_normal(_INPUT_SHAPE)
    flat = data.reshape(-1)
    flat[0] = 0.0
    flat[1] = -1.25
    flat[2] = 2.5
    return np.asarray(data, dtype=np.float64)


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def _assert_close(found: np.ndarray, expected: np.ndarray) -> None:
    assert found.dtype == np.float64
    assert found.shape == expected.shape
    delta = float(np.max(np.abs(found - expected)))
    assert np.allclose(found, expected, **_CLOSE), delta


def _make(kind: str, rng: np.random.Generator, **extra: object) -> Any:
    if kind == "Conv2d":
        return Conv2d(3, 4, 3, rng, **extra)
    if kind == "DepthwiseConv2d":
        return DepthwiseConv2d(3, rng, **extra)
    if kind == "PointwiseConv2d":
        return PointwiseConv2d(3, 4, rng, **extra)
    if kind == "BatchNorm2d":
        return BatchNorm2d(3, rng, **extra)
    if kind == "Linear":
        return Linear(75, 4, rng, **extra)
    if kind == "ReLU":
        return ReLU(**extra)
    if kind == "Sigmoid":
        return Sigmoid(**extra)
    if kind == "GlobalAvgPool":
        return GlobalAvgPool(**extra)
    raise AssertionError(kind)


def _he_weight(kind: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Independent He-normal draw. ``fan_in`` is the input count of one output."""
    if kind == "Conv2d":
        shape: tuple[int, ...] = (4, 3, 3, 3)
        fan_in = 3 * 3 * 3
        bias_shape: tuple[int, ...] = (4,)
    elif kind == "DepthwiseConv2d":
        shape = (3, 3, 3)
        fan_in = 3 * 3
        bias_shape = (3,)
    elif kind == "PointwiseConv2d":
        shape = (4, 3)
        fan_in = 3
        bias_shape = (4,)
    elif kind == "BatchNorm2d":
        shape = (3,)
        fan_in = 1
        bias_shape = (3,)
    elif kind == "Linear":
        shape = (4, 75)
        fan_in = 75
        bias_shape = (4,)
    else:
        raise AssertionError(kind)
    weight = rng.standard_normal(shape) * math.sqrt(2.0 / fan_in)
    bias = np.zeros(bias_shape, dtype=np.float64)
    return np.asarray(weight, dtype=np.float64), bias


def _conv_reference(
    values: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> np.ndarray:
    """Dense convolution by a direct sum over kernel taps. Out-of-frame taps are zero."""
    batch, channels, height, width = values.shape
    out_channels, in_channels, kernel_h, kernel_w = weight.shape
    assert channels == in_channels
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    out_h = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (width + 2 * pad_w - kernel_w) // stride_w + 1
    out = np.zeros((batch, out_channels, out_h, out_w), dtype=np.float64)
    for item in range(batch):
        for out_c in range(out_channels):
            for row in range(out_h):
                for col in range(out_w):
                    total = float(bias[out_c])
                    for in_c in range(channels):
                        for ky in range(kernel_h):
                            for kx in range(kernel_w):
                                in_row = row * stride_h + ky - pad_h
                                in_col = col * stride_w + kx - pad_w
                                if 0 <= in_row < height and 0 <= in_col < width:
                                    total += float(values[item, in_c, in_row, in_col]) * float(
                                        weight[out_c, in_c, ky, kx]
                                    )
                    out[item, out_c, row, col] = total
    return out


def _depthwise_reference(
    values: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> np.ndarray:
    """One 3x3 kernel per channel. Channels are not mixed."""
    batch, channels, height, width = values.shape
    assert weight.shape == (channels, 3, 3)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    out_h = (height + 2 * pad_h - 3) // stride_h + 1
    out_w = (width + 2 * pad_w - 3) // stride_w + 1
    out = np.zeros((batch, channels, out_h, out_w), dtype=np.float64)
    for item in range(batch):
        for channel in range(channels):
            for row in range(out_h):
                for col in range(out_w):
                    total = float(bias[channel])
                    for ky in range(3):
                        for kx in range(3):
                            in_row = row * stride_h + ky - pad_h
                            in_col = col * stride_w + kx - pad_w
                            if 0 <= in_row < height and 0 <= in_col < width:
                                total += float(values[item, channel, in_row, in_col]) * float(
                                    weight[channel, ky, kx]
                                )
                    out[item, channel, row, col] = total
    return out


def _pointwise_reference(values: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """1x1 mix across channels at each spatial site."""
    batch, channels, height, width = values.shape
    out_channels = int(weight.shape[0])
    assert weight.shape == (out_channels, channels)
    out = np.zeros((batch, out_channels, height, width), dtype=np.float64)
    for item in range(batch):
        for out_c in range(out_channels):
            for row in range(height):
                for col in range(width):
                    total = float(bias[out_c])
                    for in_c in range(channels):
                        total += float(values[item, in_c, row, col]) * float(weight[out_c, in_c])
                    out[item, out_c, row, col] = total
    return out


def _linear_reference(flat: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    batch, features = flat.shape
    out_features = int(weight.shape[0])
    assert weight.shape == (out_features, features)
    out = np.zeros((batch, out_features), dtype=np.float64)
    for item in range(batch):
        for out_f in range(out_features):
            total = float(bias[out_f])
            for feature in range(features):
                total += float(flat[item, feature]) * float(weight[out_f, feature])
            out[item, out_f] = total
    return out


def _relu_reference(values: np.ndarray) -> np.ndarray:
    out = np.empty(values.shape, dtype=np.float64)
    for index in np.ndindex(values.shape):
        value = float(values[index])
        out[index] = value if value > 0.0 else 0.0
    return out


def _sigmoid_scalar(value: float) -> float:
    if value >= 0.0:
        positive = math.exp(-value)
        return 1.0 / (1.0 + positive)
    negative = math.exp(value)
    return negative / (1.0 + negative)


def _sigmoid_reference(values: np.ndarray) -> np.ndarray:
    out = np.empty(values.shape, dtype=np.float64)
    for index in np.ndindex(values.shape):
        out[index] = _sigmoid_scalar(float(values[index]))
    return out


def _gap_reference(values: np.ndarray) -> np.ndarray:
    batch, channels, height, width = values.shape
    out = np.zeros((batch, channels), dtype=np.float64)
    area = height * width
    for item in range(batch):
        for channel in range(channels):
            total = 0.0
            for row in range(height):
                for col in range(width):
                    total += float(values[item, channel, row, col])
            out[item, channel] = total / area
    return out


def _moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Population mean and variance over batch and spatial axes, per channel."""
    batch, channels, height, width = values.shape
    count = batch * height * width
    mean = np.zeros(channels, dtype=np.float64)
    var = np.zeros(channels, dtype=np.float64)
    for channel in range(channels):
        total = 0.0
        for item in range(batch):
            for row in range(height):
                for col in range(width):
                    total += float(values[item, channel, row, col])
        mu = total / count
        mean[channel] = mu
        square = 0.0
        for item in range(batch):
            for row in range(height):
                for col in range(width):
                    delta = float(values[item, channel, row, col]) - mu
                    square += delta * delta
        var[channel] = square / count
    return mean, var, count


def _bn_apply(
    values: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float,
) -> np.ndarray:
    out = np.empty(values.shape, dtype=np.float64)
    for item, channel, row, col in np.ndindex(values.shape):
        centred = float(values[item, channel, row, col]) - float(mean[channel])
        scale = math.sqrt(float(var[channel]) + eps)
        out[item, channel, row, col] = float(weight[channel]) * centred / scale + float(
            bias[channel]
        )
    return out


def _ema(
    running_mean: np.ndarray,
    running_var: np.ndarray,
    batch_mean: np.ndarray,
    batch_var: np.ndarray,
    momentum: float,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend batch moments into the running estimates.

    The variance blend uses the unbiased batch variance when the count is
    greater than one, and the population variance for a single element.
    """
    scale = count / (count - 1) if count > 1 else 1.0
    keep = 1.0 - momentum
    new_mean = np.empty_like(running_mean)
    new_var = np.empty_like(running_var)
    for channel in range(int(running_mean.shape[0])):
        new_mean[channel] = keep * float(running_mean[channel]) + momentum * float(
            batch_mean[channel]
        )
        new_var[channel] = keep * float(running_var[channel]) + momentum * (
            float(batch_var[channel]) * scale
        )
    return new_mean, new_var


def test_nn_package_exports_the_layers() -> None:
    exported = {
        "BatchNorm2d": BatchNorm2d,
        "Conv2d": Conv2d,
        "DepthwiseConv2d": DepthwiseConv2d,
        "GlobalAvgPool": GlobalAvgPool,
        "Linear": Linear,
        "PointwiseConv2d": PointwiseConv2d,
        "ReLU": ReLU,
        "Sigmoid": Sigmoid,
    }
    for name, cls in exported.items():
        assert getattr(nn, name) is cls
        assert name in nn.__all__


def test_defaults() -> None:
    conv = Conv2d(3, 2, 1, np.random.default_rng(0))
    assert conv.in_channels == 3
    assert conv.out_channels == 2
    assert conv.kernel_size == (1, 1)
    assert conv.stride == (1, 1)
    assert conv.padding == (0, 0)
    depth = DepthwiseConv2d(3, np.random.default_rng(0))
    assert depth.channels == 3
    assert depth.kernel_size == (3, 3)
    assert depth.stride == (1, 1)
    assert depth.padding == (0, 0)
    point = PointwiseConv2d(3, 5, np.random.default_rng(0))
    assert point.in_channels == 3
    assert point.out_channels == 5
    assert point.kernel_size == (1, 1)
    norm = BatchNorm2d(3, np.random.default_rng(0))
    assert norm.num_features == 3
    assert norm.eps == 1e-5
    assert norm.momentum == 0.1
    assert norm.training is True
    linear = Linear(75, 4, np.random.default_rng(0))
    assert linear.in_features == 75
    assert linear.out_features == 4


@pytest.mark.parametrize("kind", _WEIGHTED)
def test_weight_draw_matches_he_normal(kind: str) -> None:
    probe = np.random.default_rng(7)
    expected_weight, expected_bias = _he_weight(kind, probe)
    layer = _make(kind, np.random.default_rng(7))
    assert np.array_equal(layer.weight, expected_weight)
    assert np.array_equal(layer.bias, expected_bias)
    assert layer.weight.dtype == np.float64
    if kind == "BatchNorm2d":
        assert np.array_equal(layer.running_mean, np.zeros(3, dtype=np.float64))
        assert np.array_equal(layer.running_var, np.ones(3, dtype=np.float64))


@pytest.mark.parametrize("kind", _WEIGHTED)
def test_same_seed_draws_identical_weights(kind: str) -> None:
    left = _make(kind, np.random.default_rng(0))
    right = _make(kind, np.random.default_rng(0))
    assert np.array_equal(left.weight, right.weight)
    assert np.array_equal(left.bias, right.bias)


@pytest.mark.parametrize("kind", _WEIGHTED)
def test_different_seed_draws_different_weights(kind: str) -> None:
    left = _make(kind, np.random.default_rng(0))
    right = _make(kind, np.random.default_rng(1))
    assert not np.array_equal(left.weight, right.weight)


@pytest.mark.parametrize("kind", _LAYER_NAMES)
def test_weights_keyword_raises_type_error(kind: str) -> None:
    kwargs = {"weights": np.zeros(1, dtype=np.float64)}
    with pytest.raises(TypeError, match="'weights'"):
        _make(kind, np.random.default_rng(0), **kwargs)


@pytest.mark.parametrize("kind", _WEIGHTED)
@pytest.mark.parametrize("keyword", ("weights_path", "checkpoint"))
def test_path_and_checkpoint_keywords_raise_type_error(kind: str, keyword: str) -> None:
    kwargs = {keyword: "parameters.bin"}
    with pytest.raises(TypeError, match=f"'{keyword}'"):
        _make(kind, np.random.default_rng(0), **kwargs)


@pytest.mark.parametrize("kind", _WEIGHTED)
def test_array_is_refused_where_a_generator_is_required(kind: str) -> None:
    with pytest.raises(TypeError, match="Generator"):
        _make(kind, np.zeros(4))  # type: ignore[arg-type]


def test_string_is_refused_where_a_generator_is_required() -> None:
    with pytest.raises(TypeError, match="Generator"):
        Conv2d(3, 4, 1, "parameters.bin")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kernel", "stride", "padding"),
    [
        pytest.param(1, 1, 0, id="k1-s1-p0"),
        pytest.param(3, 1, 0, id="k3-s1-p0"),
        pytest.param(3, 2, 0, id="k3-s2-p0"),
        pytest.param(3, 1, 1, id="k3-s1-p1"),
        pytest.param(3, 2, 1, id="k3-s2-p1"),
        pytest.param((3, 1), (2, 1), (0, 1), id="k3x1-s2x1-p0x1"),
    ],
)
def test_conv2d_matches_loop_reference(
    kernel: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> None:
    values = _nchw(11)
    original = values.copy()
    layer = Conv2d(3, 2, kernel, np.random.default_rng(3), stride=stride, padding=padding)
    assert layer.kernel_size == _pair(kernel)
    assert layer.stride == _pair(stride)
    assert layer.padding == _pair(padding)
    found = layer.forward(values)
    expected = _conv_reference(values, layer.weight, layer.bias, stride, padding)
    _assert_close(found, expected)
    assert np.array_equal(values, original)


@pytest.mark.parametrize(
    ("stride", "padding"),
    [
        pytest.param(1, 0, id="s1-p0"),
        pytest.param(2, 1, id="s2-p1"),
        pytest.param((1, 2), (1, 0), id="s1x2-p1x0"),
    ],
)
def test_depthwise_conv2d_matches_loop_reference(
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> None:
    values = _nchw(12)
    original = values.copy()
    layer = DepthwiseConv2d(3, np.random.default_rng(4), stride=stride, padding=padding)
    found = layer.forward(values)
    expected = _depthwise_reference(values, layer.weight, layer.bias, stride, padding)
    _assert_close(found, expected)
    assert np.array_equal(values, original)


def test_pointwise_conv2d_matches_loop_reference() -> None:
    values = _nchw(13)
    original = values.copy()
    layer = PointwiseConv2d(3, 4, np.random.default_rng(5))
    found = layer.forward(values)
    expected = _pointwise_reference(values, layer.weight, layer.bias)
    _assert_close(found, expected)
    assert found.shape == (2, 4, 5, 5)
    assert np.array_equal(values, original)


def test_linear_matches_loop_reference() -> None:
    values = _nchw(14)
    original = values.copy()
    layer = Linear(75, 4, np.random.default_rng(6))
    flat = values.reshape(2, 75)
    found = layer.forward(values)
    found_flat = layer.forward(flat.copy())
    expected = _linear_reference(flat, layer.weight, layer.bias)
    _assert_close(found, expected)
    _assert_close(found_flat, expected)
    assert np.array_equal(values, original)


def test_relu_matches_loop_reference() -> None:
    values = _nchw(15)
    original = values.copy()
    found = ReLU().forward(values)
    _assert_close(found, _relu_reference(values))
    assert np.array_equal(values, original)


def test_sigmoid_matches_loop_reference() -> None:
    values = _nchw(16)
    flat = values.reshape(-1)
    flat[3] = 800.0
    flat[4] = -800.0
    original = values.copy()
    found = Sigmoid().forward(values)
    _assert_close(found, _sigmoid_reference(values))
    assert np.array_equal(values, original)


def test_global_avg_pool_matches_loop_reference() -> None:
    values = _nchw(17)
    original = values.copy()
    found = GlobalAvgPool().forward(values)
    _assert_close(found, _gap_reference(values))
    assert found.shape == (2, 3)
    assert np.array_equal(values, original)


def test_batchnorm2d_train_matches_loop_reference() -> None:
    values = _nchw(18)
    original = values.copy()
    layer = BatchNorm2d(3, np.random.default_rng(8), eps=0.5, momentum=0.25)
    layer.train()
    assert layer.training is True
    mean_0 = layer.running_mean.copy()
    var_0 = layer.running_var.copy()
    found = layer.forward(values)
    batch_mean, batch_var, count = _moments(values)
    _assert_close(found, _bn_apply(values, batch_mean, batch_var, layer.weight, layer.bias, 0.5))
    mean_1, var_1 = _ema(mean_0, var_0, batch_mean, batch_var, 0.25, count)
    _assert_close(layer.running_mean, mean_1)
    _assert_close(layer.running_var, var_1)
    frozen = _bn_apply(values, mean_0, var_0, layer.weight, layer.bias, 0.5)
    assert not np.allclose(found, frozen, **_CLOSE)
    assert np.array_equal(values, original)

    second = _nchw(19)
    found_2 = layer.forward(second)
    batch_mean_2, batch_var_2, count_2 = _moments(second)
    _assert_close(
        found_2, _bn_apply(second, batch_mean_2, batch_var_2, layer.weight, layer.bias, 0.5)
    )
    mean_2, var_2 = _ema(mean_1, var_1, batch_mean_2, batch_var_2, 0.25, count_2)
    _assert_close(layer.running_mean, mean_2)
    _assert_close(layer.running_var, var_2)


def test_batchnorm2d_eval_matches_loop_reference() -> None:
    train_values = _nchw(20)
    eval_values = _nchw(21) + 4.0
    original = eval_values.copy()
    layer = BatchNorm2d(3, np.random.default_rng(9), eps=0.5, momentum=0.25)
    layer.forward(train_values)
    layer.eval()
    assert layer.training is False
    mean = layer.running_mean.copy()
    var = layer.running_var.copy()
    found = layer.forward(eval_values)
    _assert_close(found, _bn_apply(eval_values, mean, var, layer.weight, layer.bias, 0.5))
    batch_mean, batch_var, _count = _moments(eval_values)
    batch_out = _bn_apply(eval_values, batch_mean, batch_var, layer.weight, layer.bias, 0.5)
    assert not np.allclose(found, batch_out, **_CLOSE)
    assert np.array_equal(layer.running_mean, mean)
    assert np.array_equal(layer.running_var, var)
    assert np.array_equal(eval_values, original)

    layer.train()
    assert layer.training is True
    found_train = layer.forward(eval_values)
    _assert_close(
        found_train, _bn_apply(eval_values, batch_mean, batch_var, layer.weight, layer.bias, 0.5)
    )


def test_batchnorm2d_one_sample_matches_loop_reference() -> None:
    values = np.zeros((1, 2, 1, 1), dtype=np.float64)
    values[0, 0, 0, 0] = 2.0
    values[0, 1, 0, 0] = -3.0
    layer = BatchNorm2d(2, np.random.default_rng(1), eps=0.5, momentum=0.4)
    mean_0 = layer.running_mean.copy()
    var_0 = layer.running_var.copy()
    found = layer.forward(values)
    batch_mean, batch_var, count = _moments(values)
    assert count == 1
    _assert_close(found, _bn_apply(values, batch_mean, batch_var, layer.weight, layer.bias, 0.5))
    mean_1, var_1 = _ema(mean_0, var_0, batch_mean, batch_var, 0.4, count)
    _assert_close(layer.running_mean, mean_1)
    _assert_close(layer.running_var, var_1)


def test_batchnorm2d_accepts_momentum_endpoints() -> None:
    assert BatchNorm2d(1, np.random.default_rng(0), momentum=0.0).momentum == 0.0
    assert BatchNorm2d(1, np.random.default_rng(0), momentum=1).momentum == 1.0


def test_constructors_reject_bad_sizes() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="positive int"):
        Conv2d(0, 1, 1, rng)
    with pytest.raises(ValueError, match="positive int"):
        Conv2d(1, 1, 0, rng)
    with pytest.raises(ValueError, match="positive int"):
        Conv2d(1, 1, 1, rng, stride=0)
    with pytest.raises(ValueError, match="non-negative int"):
        Conv2d(1, 1, 1, rng, padding=-1)
    with pytest.raises(ValueError, match="pair"):
        Conv2d(1, 1, (3, 3, 3), rng)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive int"):
        DepthwiseConv2d(True, rng)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="greater than 0"):
        BatchNorm2d(1, rng, eps=0.0)
    with pytest.raises(ValueError, match="greater than 0"):
        BatchNorm2d(1, rng, eps="wide")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="greater than 0"):
        BatchNorm2d(1, rng, eps=float("nan"))
    with pytest.raises(ValueError, match="from 0 to 1"):
        BatchNorm2d(1, rng, momentum=1.1)
    with pytest.raises(ValueError, match="from 0 to 1"):
        BatchNorm2d(1, rng, momentum=-0.1)
    with pytest.raises(ValueError, match="from 0 to 1"):
        BatchNorm2d(1, rng, momentum="fast")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="from 0 to 1"):
        BatchNorm2d(1, rng, momentum=float("inf"))


def test_forward_rejects_bad_shapes() -> None:
    rng = np.random.default_rng(0)
    values = _nchw(2)
    with pytest.raises(ValueError, match="output spatial size"):
        Conv2d(3, 1, 6, rng).forward(values)
    with pytest.raises(ValueError, match="output spatial size"):
        Conv2d(3, 1, (1, 6), rng).forward(values)
    with pytest.raises(ValueError, match="ndim 4"):
        Conv2d(3, 1, 1, rng).forward(np.zeros((2, 3, 5), dtype=np.float64))
    with pytest.raises(ValueError, match="expected 4 channels"):
        Conv2d(4, 1, 1, rng).forward(values)
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        Linear(75, 2, rng).forward(np.zeros(75, dtype=np.float64))
    with pytest.raises(ValueError, match="expected 7 features"):
        Linear(7, 2, rng).forward(values)
    with pytest.raises(ValueError, match="expected 4 channels"):
        BatchNorm2d(4, rng).forward(values)
    with pytest.raises(ValueError, match="ndim 4"):
        BatchNorm2d(3, rng).forward(np.zeros((2, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="spatial"):
        GlobalAvgPool().forward(np.zeros((2, 3, 0, 5), dtype=np.float64))
    with pytest.raises(ValueError, match="ndim 4"):
        GlobalAvgPool().forward(np.zeros((2, 3), dtype=np.float64))


def _layers_source() -> str:
    return Path(inspect.getfile(Conv2d)).read_text(encoding="utf-8")


def test_layers_module_has_no_backward_step_or_zero_grad() -> None:
    source = _layers_source()
    for needle in (".backward(", ".step(", "zero_grad"):
        assert needle not in source


def test_layer_module_does_not_import_a_deep_learning_package() -> None:
    source = _layers_source()
    for name in ("torch", "onnx", "jax", "tensorflow"):
        assert f"import {name}" not in source
        assert f"from {name}" not in source


def test_no_network_guard_records_zero_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def refuse_connect(*_args: object, **_kwargs: object) -> None:
        calls.append("connect")
        raise OSError("refused")

    def refuse_create_connection(*_args: object, **_kwargs: object) -> None:
        calls.append("create_connection")
        raise OSError("refused")

    monkeypatch.setattr(socket.socket, "connect", refuse_connect)
    monkeypatch.setattr(socket, "create_connection", refuse_create_connection)
    values = _nchw(22)
    rng = np.random.default_rng(0)
    found_conv = Conv2d(3, 2, 1, rng).forward(values)
    found_depth = DepthwiseConv2d(3, rng, padding=1).forward(values)
    found_point = PointwiseConv2d(3, 2, rng).forward(values)
    found_norm = BatchNorm2d(3, rng).forward(values)
    found_linear = Linear(75, 2, rng).forward(values)
    found_relu = ReLU().forward(values)
    found_sigmoid = Sigmoid().forward(values)
    found_pool = GlobalAvgPool().forward(values)
    assert found_conv.shape[0] == 2
    assert found_depth.shape[1] == 3
    assert found_point.shape[1] == 2
    assert found_norm.shape == values.shape
    assert found_linear.shape == (2, 2)
    assert found_relu.shape == values.shape
    assert found_sigmoid.shape == values.shape
    assert found_pool.shape == (2, 3)
    assert calls == []
