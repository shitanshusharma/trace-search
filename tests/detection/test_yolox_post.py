"""Acceptance tests for YOLOX decode and class-aware NMS.

Every array is built in this file. No image, weight, or model runtime is loaded.
"""

import ast
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from trace_search.detection.types import DetClass, Detection
from trace_search.detection.yolox_post import decode, letterbox, nms, to_detections

_SIZE = 640
_STRIDES = (8, 16, 32)
_IMAGE_H = 200
_IMAGE_W = 400
_COLUMNS = 85
_PERSON = 0
_CAR = 2
_BACKPACK = 24
_HANDBAG = 26
_CELL_Y = 30
_CELL_X = 30
_OFFSET = 0.5
_LOG_SIDE = math.log(2.0)

# Image 200x400 letterboxed to 640. The long side sets the scale.
# ratio = 640 / 400 = 1.6
# resized height = int(200 * 1.6) = 320, so the top inset is (640 - 320) / 2 = 160
# Stride-8 cell (30, 30) with a half-cell offset and log-size log(2):
# center = 30.5 * 8 = 244, side = 2 * 8 = 16, letterbox box = (236, 236, 252, 252)
# original x = 236 / 1.6 = 147.5, 252 / 1.6 = 157.5
# original y = (236 - 160) / 1.6 = 47.5, (252 - 160) / 1.6 = 57.5
_EXPECTED_BOX = (147.5, 47.5, 157.5, 57.5)


def _cell_count(size: int = _SIZE, strides: tuple[int, ...] = _STRIDES) -> int:
    return sum((size // stride) ** 2 for stride in strides)


def _cell_index(
    stride: int,
    gy: int,
    gx: int,
    size: int = _SIZE,
    strides: tuple[int, ...] = _STRIDES,
) -> int:
    offset = 0
    for item in strides:
        grid = size // item
        if item == stride:
            return offset + gy * grid + gx
        offset += grid * grid
    raise AssertionError(stride)


def _raw() -> np.ndarray:
    count = _cell_count()
    assert count == 8400
    return np.zeros((count, _COLUMNS), dtype=np.float64)


def _paint(
    raw: np.ndarray,
    stride: int,
    gy: int,
    gx: int,
    class_id: int,
    *,
    obj: float = 0.9,
    cls: float = 0.8,
    xy: tuple[float, float] = (_OFFSET, _OFFSET),
    wh: tuple[float, float] = (_LOG_SIDE, _LOG_SIDE),
) -> int:
    index = _cell_index(stride, gy, gx)
    raw[index, 0] = xy[0]
    raw[index, 1] = xy[1]
    raw[index, 2] = wh[0]
    raw[index, 3] = wh[1]
    raw[index, 4] = obj
    raw[index, 5 + class_id] = cls
    return index


def _continuous_iou(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (float(left[2]) - float(left[0])) * (float(left[3]) - float(left[1]))
    right_area = (float(right[2]) - float(right[0])) * (float(right[3]) - float(right[1]))
    return intersection / (left_area + right_area - intersection)


def _detections(raw: np.ndarray) -> list[Detection]:
    image = np.zeros((_IMAGE_H, _IMAGE_W, 3), dtype=np.uint8)
    _tensor, ratio, pad = letterbox(image, size=_SIZE)
    return to_detections(
        raw,
        ratio=ratio,
        pad=pad,
        camera_id="cam-a",
        frame_number=4,
        timestamp_seconds=1.25,
    )


def test_letterbox_non_square_image_exposes_ratio_and_pad() -> None:
    """A wide frame needs a vertical inset, so both scale and pad are real."""
    image = np.full((_IMAGE_H, _IMAGE_W, 3), 7, dtype=np.uint8)
    tensor, ratio, pad = letterbox(image, size=_SIZE)
    assert tensor.shape == (3, _SIZE, _SIZE)
    assert tensor.dtype == np.float32
    assert ratio == pytest.approx(1.6)
    assert pad == (0.0, 160.0)
    assert tensor[:, 0, 0].tolist() == [114.0, 114.0, 114.0]
    assert tensor[:, 320, 320].tolist() == [7.0, 7.0, 7.0]


def test_stride8_person_lands_on_original_pixels() -> None:
    """One strong person cell survives the letterbox inverse within one pixel."""
    image = np.zeros((_IMAGE_H, _IMAGE_W, 3), dtype=np.uint8)
    _tensor, ratio, pad = letterbox(image, size=_SIZE)
    assert ratio == pytest.approx(1.6)
    assert pad == (0.0, 160.0)
    raw = _raw()
    _paint(raw, 8, _CELL_Y, _CELL_X, _PERSON, obj=0.9, cls=0.8)
    found = to_detections(
        raw,
        ratio=ratio,
        pad=pad,
        camera_id="cam-a",
        frame_number=4,
        timestamp_seconds=1.25,
    )
    assert len(found) == 1
    detection = found[0]
    assert detection.det_class is DetClass.PERSON
    assert detection.camera_id == "cam-a"
    assert detection.frame_number == 4
    assert detection.timestamp_seconds == pytest.approx(1.25)
    assert detection.score == pytest.approx(0.72)
    assert detection.box[0] == pytest.approx(_EXPECTED_BOX[0], abs=1.0)
    assert detection.box[1] == pytest.approx(_EXPECTED_BOX[1], abs=1.0)
    assert detection.box[2] == pytest.approx(_EXPECTED_BOX[2], abs=1.0)
    assert detection.box[3] == pytest.approx(_EXPECTED_BOX[3], abs=1.0)


def test_decode_adds_grid_index_and_stride() -> None:
    raw = _raw()
    index = _paint(raw, 8, _CELL_Y, _CELL_X, _PERSON)
    decoded = decode(raw, strides=_STRIDES, size=_SIZE)
    assert decoded[index, 0] == pytest.approx(244.0)
    assert decoded[index, 1] == pytest.approx(244.0)
    assert decoded[index, 2] == pytest.approx(16.0)
    assert decoded[index, 3] == pytest.approx(16.0)
    assert decoded[index, 4] == pytest.approx(0.9)
    assert decoded[index, 5] == pytest.approx(0.8)
    assert np.array_equal(raw[index, 0:4], np.array([0.5, 0.5, _LOG_SIDE, _LOG_SIDE]))


def test_decode_stride16_uses_that_level_grid() -> None:
    raw = _raw()
    index = _paint(raw, 16, 4, 6, _PERSON, xy=(0.0, 0.0), wh=(0.0, 0.0))
    decoded = decode(raw)
    assert index == 6400 + 4 * 40 + 6
    assert decoded[index, 0] == pytest.approx(96.0)
    assert decoded[index, 1] == pytest.approx(64.0)
    assert decoded[index, 2] == pytest.approx(16.0)
    assert decoded[index, 3] == pytest.approx(16.0)


def test_same_class_iou_point_eight_keeps_one_box() -> None:
    first = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float64)
    second = np.array([0.0, 0.0, 100.0, 80.0], dtype=np.float64)
    assert _continuous_iou(first, second) == pytest.approx(0.8)
    boxes = np.stack((first, second))
    scores = np.zeros((2, 80), dtype=np.float64)
    scores[0, _PERSON] = 0.95
    scores[1, _PERSON] = 0.70
    kept = nms(boxes, scores, iou=0.45)
    assert kept.shape == (1, 6)
    assert kept[0, 4] == pytest.approx(0.95)
    assert int(kept[0, 5]) == _PERSON


def test_person_and_handbag_at_the_same_place_are_both_kept() -> None:
    box = np.array([20.0, 30.0, 80.0, 140.0], dtype=np.float64)
    boxes = np.stack((box, box))
    scores = np.zeros((2, 80), dtype=np.float64)
    scores[0, _PERSON] = 0.91
    scores[1, _HANDBAG] = 0.77
    kept = nms(boxes, scores, iou=0.45)
    assert kept.shape[0] == 2
    assert {int(row[5]) for row in kept} == {_PERSON, _HANDBAG}

    raw = _raw()
    index = _paint(raw, 8, _CELL_Y, _CELL_X, _PERSON, obj=0.9, cls=0.9)
    raw[index, 5 + _HANDBAG] = 0.8
    found = _detections(raw)
    assert len(found) == 2
    assert {item.det_class for item in found} == {DetClass.PERSON, DetClass.HANDBAG}
    assert found[0].box == pytest.approx(found[1].box)


def test_car_class_is_dropped() -> None:
    raw = _raw()
    _paint(raw, 8, _CELL_Y, _CELL_X, _CAR, obj=0.99, cls=0.99)
    _paint(raw, 8, 10, 10, _PERSON, obj=0.9, cls=0.9)
    found = _detections(raw)
    assert [item.det_class for item in found] == [DetClass.PERSON]


def test_backpack_and_handbag_map_to_det_class() -> None:
    raw = _raw()
    _paint(raw, 8, 10, 12, _BACKPACK, obj=0.9, cls=0.9)
    _paint(raw, 8, 40, 50, _HANDBAG, obj=0.85, cls=0.8)
    found = _detections(raw)
    assert {item.det_class for item in found} == {DetClass.BACKPACK, DetClass.HANDBAG}
    assert all(0.0 <= item.score <= 1.0 for item in found)
    assert all(item.box[2] > item.box[0] and item.box[3] > item.box[1] for item in found)


def test_score_not_above_threshold_is_dropped() -> None:
    raw = _raw()
    _paint(raw, 8, _CELL_Y, _CELL_X, _PERSON, obj=0.5, cls=0.5)
    assert _detections(raw) == []
    raw = _raw()
    _paint(raw, 8, _CELL_Y, _CELL_X, _PERSON, obj=0.5, cls=0.52)
    found = _detections(raw)
    assert len(found) == 1
    assert found[0].score == pytest.approx(0.26)


def test_disjoint_same_class_boxes_are_both_kept() -> None:
    first = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    second = np.array([100.0, 100.0, 120.0, 130.0], dtype=np.float64)
    assert _continuous_iou(first, second) == pytest.approx(0.0)
    boxes = np.stack((first, second))
    scores = np.zeros((2, 80), dtype=np.float64)
    scores[:, _PERSON] = (0.6, 0.4)
    kept = nms(boxes, scores, iou=0.45)
    assert kept.shape[0] == 2


def test_decode_rejects_the_wrong_cell_count() -> None:
    with pytest.raises(ValueError):
        decode(np.zeros((3, _COLUMNS), dtype=np.float64))


def test_letterbox_rejects_an_image_without_three_channels() -> None:
    with pytest.raises(ValueError):
        letterbox(np.zeros((_IMAGE_H, _IMAGE_W), dtype=np.uint8), size=_SIZE)


def test_to_detections_rejects_a_non_positive_ratio() -> None:
    with pytest.raises(ValueError):
        to_detections(
            _raw(),
            ratio=0.0,
            pad=(0.0, 0.0),
            camera_id="cam-a",
            frame_number=0,
            timestamp_seconds=0.0,
        )


def test_module_docstring_cites_the_reference_functions() -> None:
    source = Path("src/trace_search/detection/yolox_post.py").read_text(encoding="utf-8")
    assert "yolox/utils/demo_utils.py" in source
    assert "demo_postprocess" in source
    assert "multiclass_nms" in source


def test_postprocess_does_not_import_a_model_runtime() -> None:
    source = Path("src/trace_search/detection/yolox_post.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])
    banned = {"onnxruntime", "torch", "yolox"}
    assert imported.isdisjoint(banned)
    probe = (
        "import sys\n"
        "import trace_search.detection.yolox_post\n"
        "roots = {name.split('.')[0] for name in sys.modules}\n"
        "banned = {'onnxruntime', 'torch', 'yolox'}\n"
        "raise SystemExit(0 if roots.isdisjoint(banned) else 1)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
