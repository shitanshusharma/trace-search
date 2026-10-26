"""Random-init OSNet-lite embedder.

Crops are built in the test from a seeded generator. Nothing is read from
disk and nothing is written into the repo tree.
"""

from __future__ import annotations

import inspect
import socket
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from trace_search.reid import AppearanceEmbedder
from trace_search.reid.nn.layers import BatchNorm2d, DepthwiseConv2d
from trace_search.reid.osnet_lite import (
    OsnetLiteEmbedder,
    _as_appearance_embedder,
    _Passthrough,
    _Projection,
)

_NORM_TOLERANCE = 1e-5


def _crops(
    seed: int,
    count: int = 2,
    shape: tuple[int, ...] = (64, 32, 3),
) -> list[npt.NDArray[np.uint8]]:
    """Build crops in the test from a seeded numpy generator."""
    rng = np.random.default_rng(seed)
    crops: list[npt.NDArray[np.uint8]] = []
    for _ in range(count):
        crop = rng.integers(0, 256, size=shape, dtype=np.uint8)
        crops.append(np.asarray(crop, dtype=np.uint8))
    return crops


def _row_norms(rows: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    return np.linalg.norm(rows.astype(np.float64), axis=1)


def _construct_rejected(keyword: str, value: object) -> None:
    """Call the constructor with one keyword it does not accept."""
    rejected: dict[str, object] = {}
    rejected[keyword] = value
    OsnetLiteEmbedder(**rejected)


def _batch_norms(embedder: OsnetLiteEmbedder) -> list[BatchNorm2d]:
    found: list[BatchNorm2d] = []
    seen: set[int] = set()

    def visit(obj: object) -> None:
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, BatchNorm2d):
            found.append(obj)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
            return
        if isinstance(obj, (np.ndarray, np.random.Generator, str, bytes, int, float)):
            return
        data = getattr(obj, "__dict__", None)
        if isinstance(data, dict):
            for value in data.values():
                visit(value)

    visit(embedder)
    return found


def test_defaults_placeholder_and_name() -> None:
    embedder = OsnetLiteEmbedder()
    assert embedder.dim == 256
    assert embedder.placeholder is True
    assert embedder.name == "osnet-lite-random-init"


def test_name_is_a_read_only_property() -> None:
    embedder = OsnetLiteEmbedder()
    name_attr = OsnetLiteEmbedder.__dict__["name"]
    assert isinstance(name_attr, property)
    assert name_attr.fset is None
    with pytest.raises(AttributeError):
        setattr(embedder, "name", "other")
    assert embedder.name == "osnet-lite-random-init"


def test_signature_is_only_seed_and_dim() -> None:
    params = inspect.signature(OsnetLiteEmbedder).parameters
    assert list(params) == ["seed", "dim"]
    assert params["seed"].default == 0
    assert params["dim"].default == 256
    assert all(item.kind is not inspect.Parameter.VAR_KEYWORD for item in params.values())


def test_assignable_to_appearance_embedder() -> None:
    """Same assignment ``mypy src/ --strict`` checks in ``_as_appearance_embedder``."""
    concrete = OsnetLiteEmbedder(seed=0, dim=8)
    embedder: AppearanceEmbedder = concrete
    assert _as_appearance_embedder(concrete) is embedder
    found = embedder.embed(_crops(seed=1, count=1, shape=(16, 12, 3)))
    assert embedder.placeholder is True
    assert embedder.name == "osnet-lite-random-init"
    assert embedder.dim == 8
    assert found.shape == (1, 8)
    assert found.dtype == np.float32


def test_output_shape_dtype_and_unit_norm() -> None:
    crops = _crops(seed=3, count=8)
    found = OsnetLiteEmbedder(seed=0).embed(crops)
    assert found.shape == (8, 256)
    assert found.dtype == np.float32
    assert bool(np.isfinite(found).all())
    norms = _row_norms(found)
    assert np.all(np.abs(norms - 1.0) <= _NORM_TOLERANCE)


def test_custom_dim_output_shape() -> None:
    crops = _crops(seed=4, count=2, shape=(20, 18, 3))
    found = OsnetLiteEmbedder(seed=2, dim=16).embed(crops)
    assert found.shape == (2, 16)
    assert found.dtype == np.float32
    assert np.all(np.abs(_row_norms(found) - 1.0) <= _NORM_TOLERANCE)


def test_same_seed_is_deterministic() -> None:
    crops = _crops(seed=7, count=3, shape=(24, 16, 3))
    first = OsnetLiteEmbedder(seed=0, dim=32)
    second = OsnetLiteEmbedder(seed=0, dim=32)
    once = first.embed(crops)
    twice = first.embed(crops)
    other = second.embed(crops)
    assert np.array_equal(once, twice)
    assert np.array_equal(once, other)


def test_different_seed_differs() -> None:
    crops = _crops(seed=7, count=2)
    seed_zero = OsnetLiteEmbedder(seed=0, dim=32).embed(crops)
    seed_one = OsnetLiteEmbedder(seed=1, dim=32).embed(crops)
    assert not np.array_equal(seed_zero, seed_one)


def test_different_crops_differ_for_one_seed() -> None:
    embedder = OsnetLiteEmbedder(seed=0, dim=32)
    left = embedder.embed(_crops(seed=1, count=1))
    right = embedder.embed(_crops(seed=2, count=1))
    assert not np.array_equal(left, right)


def test_embed_does_not_modify_crops() -> None:
    crops = _crops(seed=9, count=2, shape=(18, 14, 3))
    originals = [crop.copy() for crop in crops]
    found = OsnetLiteEmbedder(seed=3, dim=8).embed(crops)
    assert found.shape == (2, 8)
    for crop, original in zip(crops, originals, strict=True):
        assert np.array_equal(crop, original)
        assert crop.dtype == np.uint8


def test_same_size_streams_use_padding_one() -> None:
    embedder = OsnetLiteEmbedder(seed=0, dim=8)
    depthwise = [stream.depthwise for stage in embedder.stages for stream in stage.streams]
    assert len(depthwise) == 8
    for layer in depthwise:
        assert isinstance(layer, DepthwiseConv2d)
        assert layer.kernel_size == (3, 3)
        assert layer.stride == (1, 1)
        assert layer.padding == (1, 1)


def test_two_stages_have_four_streams_and_one_shared_gate() -> None:
    embedder = OsnetLiteEmbedder(seed=0, dim=8)
    assert len(embedder.stages) == 2
    first, second = embedder.stages
    assert len(first.streams) == 4
    assert len(second.streams) == 4
    assert first.gate is not second.gate
    assert isinstance(first.shortcut, _Passthrough)
    assert isinstance(second.shortcut, _Projection)
    assert first.gate.reduce.in_features == 8
    assert first.gate.reduce.out_features == 1
    assert second.gate.reduce.in_features == 16
    assert second.gate.reduce.out_features == 1


def test_batch_norm_stays_at_init_and_does_not_update() -> None:
    embedder = OsnetLiteEmbedder(seed=0, dim=8)
    norms = _batch_norms(embedder)
    assert norms
    before = [(layer.running_mean.copy(), layer.running_var.copy()) for layer in norms]
    for layer in norms:
        assert layer.training is False
        assert np.array_equal(layer.weight, np.ones_like(layer.weight))
        assert np.array_equal(layer.bias, np.zeros_like(layer.bias))
    embedder.embed(_crops(seed=5, count=2))
    for layer, (mean, var) in zip(norms, before, strict=True):
        assert layer.training is False
        assert np.array_equal(layer.running_mean, mean)
        assert np.array_equal(layer.running_var, var)


def test_empty_crop_list_has_shape_zero_by_dim() -> None:
    found = OsnetLiteEmbedder(seed=0, dim=8).embed([])
    assert found.shape == (0, 8)
    assert found.dtype == np.float32


@pytest.mark.parametrize("dim", [0, -3, True, 1.5])
def test_dim_below_one_raises(dim: object) -> None:
    with pytest.raises(ValueError, match="dim"):
        OsnetLiteEmbedder(seed=0, dim=dim)  # type: ignore[arg-type]


def test_weights_keyword_raises_type_error() -> None:
    with pytest.raises(TypeError, match="'weights'"):
        _construct_rejected("weights", "x")


def test_weights_path_keyword_raises_type_error() -> None:
    with pytest.raises(TypeError, match="'weights_path'"):
        _construct_rejected("weights_path", "x")


def test_checkpoint_keyword_raises_type_error() -> None:
    with pytest.raises(TypeError, match="'checkpoint'"):
        _construct_rejected("checkpoint", "x")


def test_pretrained_keyword_raises_type_error() -> None:
    with pytest.raises(TypeError, match="'pretrained'"):
        _construct_rejected("pretrained", True)


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
    crops = _crops(seed=5, count=2, shape=(15, 12, 3))
    found = OsnetLiteEmbedder(seed=1, dim=8).embed(crops)
    assert found.shape == (len(crops), 8)
    assert calls == []


def test_noncontiguous_crop_embeds() -> None:
    rng = np.random.default_rng(12)
    full = np.asarray(rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8))
    crop = full[::2, ::2, :]
    assert crop.flags.c_contiguous is False
    found = OsnetLiteEmbedder(seed=0, dim=8).embed([crop])
    assert found.shape == (1, 8)
    assert abs(float(np.linalg.norm(found[0].astype(np.float64))) - 1.0) <= _NORM_TOLERANCE


def test_zero_crop_cannot_be_normalised() -> None:
    crop = np.zeros((12, 9, 3), dtype=np.uint8)
    assert crop.size > 0
    embedder = OsnetLiteEmbedder(seed=0, dim=8)
    with pytest.raises(ValueError, match="normalised"):
        embedder.embed([crop])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_non_uint8_crop_raises(dtype: type) -> None:
    crop: npt.NDArray[np.generic] = np.zeros((8, 8, 3), dtype=dtype)
    embedder = OsnetLiteEmbedder(seed=0, dim=4)
    with pytest.raises(ValueError, match="uint8"):
        embedder.embed([crop])


def test_two_dimensional_crop_raises() -> None:
    crop = _crops(seed=6, count=1, shape=(18, 9))[0]
    assert crop.ndim == 2
    embedder = OsnetLiteEmbedder(seed=0, dim=4)
    with pytest.raises(ValueError, match="HxWx3"):
        embedder.embed([crop])


def test_one_channel_crop_raises() -> None:
    crop = _crops(seed=6, count=1, shape=(12, 8, 1))[0]
    embedder = OsnetLiteEmbedder(seed=0, dim=4)
    with pytest.raises(ValueError, match="3 channels"):
        embedder.embed([crop])


def test_zero_sized_crop_raises() -> None:
    crop = np.zeros((0, 8, 3), dtype=np.uint8)
    embedder = OsnetLiteEmbedder(seed=0, dim=4)
    with pytest.raises(ValueError, match="non-zero"):
        embedder.embed([crop])


def test_non_array_crop_raises() -> None:
    embedder = OsnetLiteEmbedder(seed=0, dim=4)
    raw = cast(npt.NDArray[np.uint8], [[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="numpy"):
        embedder.embed([raw])


def test_module_has_no_backward_step_or_zero_grad() -> None:
    source = Path(inspect.getfile(OsnetLiteEmbedder)).read_text(encoding="utf-8")
    for needle in (".backward(", ".step(", "zero_grad"):
        assert needle not in source
    for name in ("torch", "onnx", "jax", "tensorflow"):
        assert f"import {name}" not in source
        assert f"from {name}" not in source


def test_different_spatial_shapes_share_one_output_shape() -> None:
    crops: Sequence[npt.NDArray[np.uint8]] = [
        _crops(seed=1, count=1, shape=(50, 10, 3))[0],
        _crops(seed=2, count=1, shape=(9, 40, 3))[0],
    ]
    found = OsnetLiteEmbedder(seed=0, dim=8).embed(crops)
    assert found.shape == (2, 8)
    assert np.all(np.abs(_row_norms(found) - 1.0) <= _NORM_TOLERANCE)
