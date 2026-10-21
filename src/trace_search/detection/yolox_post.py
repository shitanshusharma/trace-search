"""YOLOX head decode and class-aware NMS, in NumPy.

Reference behaviour is ``demo_postprocess`` and ``multiclass_nms`` in
``yolox/utils/demo_utils.py`` (Megvii-BaseDetection/YOLOX). This repository
does not pin a YOLOX commit: ``docs/``, ``licenses/manifest.json``, and the
task notes name the project, not a revision. Decode follows
the published grid rule. Each cell center is ``(offset + grid index) *
stride``, and each side is ``exp(log size) * stride``. NMS follows the
class-aware branch: classes are suppressed separately, and a lower-scoring
box is dropped when its overlap with a kept box of the same class is greater
than the threshold.

Overlap is the continuous intersection area divided by the union area. The
keep test is the same comparison the reference uses (keep when overlap is
less than or equal to the threshold). Nothing is copied from YOLOX. This
module does not load a runtime or weights.
"""

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from trace_search.detection.types import DetClass, Detection

_SIZE = 640
_STRIDES = (8, 16, 32)
_COLUMNS = 85
_FILL = np.float32(114.0)
_IOU = 0.45
_SCORE_THRESH = 0.25
_KEPT: dict[int, DetClass] = {
    0: DetClass.PERSON,
    24: DetClass.BACKPACK,
    26: DetClass.HANDBAG,
}

__all__ = ["decode", "letterbox", "nms", "to_detections"]


def letterbox(
    image: npt.NDArray[np.uint8],
    size: int = _SIZE,
) -> tuple[npt.NDArray[np.float32], float, tuple[float, float]]:
    """Fit ``image`` in a square and center it.

    The tensor is float32, channel-first, with a 114 border. ``ratio`` is the
    scale applied to the original pixels. ``pad`` is ``(left, top)`` of that
    border, in input pixels. The inverse of a decoded point is
    ``(point - pad) / ratio``.
    """
    height, width = _image_hw(image)
    ratio, resized_w, resized_h, left, top = _square_fit(height, width, size)
    canvas = np.full((size, size, 3), _FILL, dtype=np.float32)
    resized = _nearest(image, resized_h, resized_w).astype(np.float32)
    canvas[top : top + resized_h, left : left + resized_w] = resized
    tensor = np.ascontiguousarray(np.transpose(canvas, (2, 0, 1)))
    return tensor, ratio, (float(left), float(top))


def decode(
    raw: npt.NDArray[np.float64],
    strides: tuple[int, ...] = _STRIDES,
    size: int = _SIZE,
) -> npt.NDArray[np.float64]:
    """Decode one YOLOX head tensor of shape ``(N, 85)``.

    ``N`` is the number of cells on ``strides`` for a square of ``size``.
    Columns 0 and 1 become the box center in letterbox pixels. Columns 2 and
    3 become width and height. Objectness and the 80 class scores are copied
    through. The caller's array is not modified.
    """
    outputs = _copied_head(raw, size, strides)
    grid, stride = _grids(size, strides)
    outputs[:, 0:2] = (outputs[:, 0:2] + grid) * stride
    outputs[:, 2:4] = np.exp(outputs[:, 2:4]) * stride
    return outputs


def nms(
    boxes: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    iou: float = _IOU,
) -> npt.NDArray[np.float64]:
    """Class-aware greedy NMS.

    ``boxes`` is ``(N, 4)`` in ``x1, y1, x2, y2``. ``scores`` is ``(N, C)``.
    A non-positive score is ignored. Each class is suppressed on its own, so
    two classes at the same place both survive. Rows are
    ``x1, y1, x2, y2, score, class_id``, class 0 first.
    """
    selected = np.asarray(boxes, dtype=np.float64)
    class_scores = np.asarray(scores, dtype=np.float64)
    parts: list[npt.NDArray[np.float64]] = []
    for class_id in range(int(class_scores.shape[1])):
        column = class_scores[:, class_id]
        chosen = _suppress(selected, _high_first(column), iou)
        if chosen:
            parts.append(_pack(selected, column, chosen, class_id))
    if not parts:
        return np.zeros((0, 6), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def to_detections(
    raw: npt.NDArray[np.float64],
    *,
    ratio: float,
    pad: tuple[float, float],
    camera_id: str,
    frame_number: int,
    timestamp_seconds: float,
    strides: tuple[int, ...] = _STRIDES,
    size: int = _SIZE,
    score_thresh: float = _SCORE_THRESH,
    iou: float = _IOU,
) -> list[Detection]:
    """Decode ``raw``, suppress overlaps, and return original-pixel detections.

    Only COCO ids 0 (person), 24 (backpack), and 26 (handbag) are kept. A
    class score is ``objectness * class``. Scores not strictly above
    ``score_thresh`` are dropped. Boxes are moved back through the letterbox
    inverse before the ``Detection`` guards run.
    """
    _require_ratio(ratio)
    decoded = decode(raw, strides=strides, size=size)
    kept = nms(_corners(decoded), _confidences(decoded, score_thresh), iou=iou)
    found: list[Detection] = []
    for index in range(int(kept.shape[0])):
        record = _record(
            kept[index],
            ratio,
            pad,
            camera_id,
            frame_number,
            timestamp_seconds,
        )
        if record is not None:
            found.append(record)
    return found


def _image_hw(image: npt.NDArray[np.uint8]) -> tuple[int, int]:
    if image.ndim != 3 or int(image.shape[2]) != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    return int(image.shape[0]), int(image.shape[1])


def _square_fit(height: int, width: int, size: int) -> tuple[float, int, int, int, int]:
    ratio = min(size / height, size / width)
    resized_w = int(width * ratio)
    resized_h = int(height * ratio)
    left = (size - resized_w) // 2
    top = (size - resized_h) // 2
    return ratio, resized_w, resized_h, left, top


def _nearest(
    image: npt.NDArray[np.uint8],
    height: int,
    width: int,
) -> npt.NDArray[np.uint8]:
    source_h = int(image.shape[0])
    source_w = int(image.shape[1])
    y = (np.arange(height) * source_h) // height
    x = (np.arange(width) * source_w) // width
    y = np.minimum(y, source_h - 1)
    x = np.minimum(x, source_w - 1)
    sampled = image[np.ix_(y, x)]
    return np.asarray(sampled, dtype=np.uint8)


def _cell_count(size: int, strides: Sequence[int]) -> int:
    return sum((size // stride) ** 2 for stride in strides)


def _copied_head(
    raw: npt.NDArray[np.float64],
    size: int,
    strides: tuple[int, ...],
) -> npt.NDArray[np.float64]:
    expected = _cell_count(size, strides)
    outputs = np.array(raw, dtype=np.float64, copy=True)
    if outputs.ndim != 2 or outputs.shape[1] != _COLUMNS or outputs.shape[0] != expected:
        raise ValueError(
            f"raw must have shape ({expected}, {_COLUMNS}) for size {size}, got {outputs.shape}"
        )
    return outputs


def _level(grid: int, stride: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    count = grid * grid
    index = np.arange(count, dtype=np.float64)
    cells = np.column_stack((np.remainder(index, grid), np.floor(index / grid)))
    scales = np.full((count, 1), float(stride), dtype=np.float64)
    return cells, scales


def _grids(
    size: int,
    strides: Sequence[int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    cells: list[npt.NDArray[np.float64]] = []
    scales: list[npt.NDArray[np.float64]] = []
    for stride in strides:
        grid = size // stride
        level_cells, level_scales = _level(grid, stride)
        cells.append(level_cells)
        scales.append(level_scales)
    return np.concatenate(cells, axis=0), np.concatenate(scales, axis=0)


def _corners(decoded: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    corners = np.empty((decoded.shape[0], 4), dtype=np.float64)
    half_w = decoded[:, 2] * 0.5
    half_h = decoded[:, 3] * 0.5
    corners[:, 0] = decoded[:, 0] - half_w
    corners[:, 1] = decoded[:, 1] - half_h
    corners[:, 2] = decoded[:, 0] + half_w
    corners[:, 3] = decoded[:, 1] + half_h
    return corners


def _confidences(
    decoded: npt.NDArray[np.float64],
    score_thresh: float,
) -> npt.NDArray[np.float64]:
    scores = decoded[:, 4:5] * decoded[:, 5:]
    kept = np.zeros(scores.shape, dtype=np.float64)
    np.copyto(kept, scores, where=scores > score_thresh)
    return kept


def _high_first(scores: npt.NDArray[np.float64]) -> list[int]:
    chosen = np.flatnonzero(scores > 0.0)
    if chosen.size == 0:
        return []
    rank = np.argsort(-scores[chosen], kind="mergesort")
    ordered = chosen[rank]
    return [int(index) for index in ordered.tolist()]


def _overlap(
    box: npt.NDArray[np.float64],
    others: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    width = np.maximum(0.0, x2 - x1)
    height = np.maximum(0.0, y2 - y1)
    intersection = width * height
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    other_area = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = box_area + other_area - intersection
    overlap = np.zeros(intersection.shape, dtype=np.float64)
    positive = union > 0.0
    overlap[positive] = intersection[positive] / union[positive]
    return overlap


def _survivors(
    box: npt.NDArray[np.float64],
    boxes: npt.NDArray[np.float64],
    pending: list[int],
    iou: float,
) -> list[int]:
    if not pending:
        return []
    rest = boxes[np.asarray(pending, dtype=np.int64)]
    overlaps = _overlap(box, rest)
    return [
        index
        for index, value in zip(pending, overlaps.tolist(), strict=True)
        if float(value) <= iou
    ]


def _suppress(
    boxes: npt.NDArray[np.float64],
    order: list[int],
    iou: float,
) -> list[int]:
    kept: list[int] = []
    pending = order
    while pending:
        current = pending[0]
        kept.append(current)
        pending = _survivors(boxes[current], boxes, pending[1:], iou)
    return kept


def _pack(
    boxes: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    chosen: list[int],
    class_id: int,
) -> npt.NDArray[np.float64]:
    index = np.asarray(chosen, dtype=np.int64)
    rows = np.empty((index.size, 6), dtype=np.float64)
    rows[:, :4] = boxes[index]
    rows[:, 4] = scores[index]
    rows[:, 5] = float(class_id)
    return rows


def _require_ratio(ratio: float) -> None:
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(f"ratio must be finite and > 0, got {ratio!r}")


def _shift(
    row: npt.NDArray[np.float64],
    ratio: float,
    pad: tuple[float, float],
) -> tuple[float, float, float, float]:
    left, top = pad
    return (
        (float(row[0]) - left) / ratio,
        (float(row[1]) - top) / ratio,
        (float(row[2]) - left) / ratio,
        (float(row[3]) - top) / ratio,
    )


def _record(
    row: npt.NDArray[np.float64],
    ratio: float,
    pad: tuple[float, float],
    camera_id: str,
    frame_number: int,
    timestamp_seconds: float,
) -> Detection | None:
    det_class = _KEPT.get(int(row[5]))
    if det_class is None:
        return None
    x1, y1, x2, y2 = _shift(row, ratio, pad)
    return Detection(
        camera_id=camera_id,
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
        det_class=det_class,
        box=(x1, y1, x2, y2),
        score=float(row[4]),
    )
