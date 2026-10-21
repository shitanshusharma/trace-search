"""SMOKE-wiring stub detector.

This is not a real detector. It finds the flat-colour figures drawn by the
synthetic smoke generator with an OpenCV threshold and a contour pass. It
does not load weights. Its boxes are not evidence for any acceptance criterion.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from trace_search.detection.types import DetClass, Detection, Detector
from trace_search.ingest.video_source import Frame

# Longest side of the frame the threshold sees. Boxes are mapped back.
_MAX_SIDE = 480
# Distance from the frame median, in 8-bit channel units. Flat figures sit
# well above this. The cutoff is not a measured accuracy threshold.
_DISTANCE = 30.0
_MIN_AREA = 64.0
# Fixed wiring value so a figure can start a track. Not a model confidence.
_STUB_SCORE = 0.95

_Box = tuple[float, float, float, float]


class ScriptedFiguresDetector:
    """SMOKE-wiring stub, not a real detector.

    OpenCV threshold plus contours on the flat-colour figures from the smoke
    generator. The score is a fixed wiring value, not a model confidence.
    """

    def detect(self, frame: Frame) -> list[Detection]:
        """Return one person box per flat-colour figure in ``frame``."""
        image = _uint8_image(frame.image)
        working, ratio = _shrink(image)
        boxes = [_scale_box(box, ratio) for box in _boxes(_foreground_mask(working))]
        return [_person(frame, box) for box in boxes]


def scripted_figures_detector() -> Detector:
    """Build the SMOKE-wiring stub. This is not a real detector."""
    return ScriptedFiguresDetector()


def _uint8_image(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    return np.ascontiguousarray(image, dtype=np.uint8)


def _shrink(image: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.uint8], float]:
    height = int(image.shape[0])
    width = int(image.shape[1])
    longest = max(height, width)
    if longest <= _MAX_SIDE:
        return image, 1.0
    ratio = _MAX_SIDE / float(longest)
    resized_w = max(1, int(round(width * ratio)))
    resized_h = max(1, int(round(height * ratio)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    return cast(npt.NDArray[np.uint8], np.ascontiguousarray(resized, dtype=np.uint8)), ratio


def _foreground_mask(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    pixels = image.astype(np.int16)
    median = np.median(pixels.reshape(-1, 3), axis=0)
    delta = pixels - median.reshape(1, 1, 3)
    squared = np.sum(delta.astype(np.float32) ** 2, axis=2)
    distance = np.sqrt(squared)
    selected = distance >= _DISTANCE
    mask = selected.astype(np.uint8) * np.uint8(255)
    return cast(npt.NDArray[np.uint8], np.ascontiguousarray(mask))


def _boxes(mask: npt.NDArray[np.uint8]) -> list[_Box]:
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[_Box] = []
    for contour in found[0]:
        if float(cv2.contourArea(contour)) < _MIN_AREA:
            continue
        rect = cast(tuple[int, int, int, int], cv2.boundingRect(contour))
        x, y, width, height = rect
        if width <= 0 or height <= 0:
            continue
        boxes.append((float(x), float(y), float(x + width), float(y + height)))
    return boxes


def _scale_box(box: _Box, ratio: float) -> _Box:
    scale = 1.0 / ratio
    x1, y1, x2, y2 = box
    return (x1 * scale, y1 * scale, x2 * scale, y2 * scale)


def _person(frame: Frame, box: _Box) -> Detection:
    return Detection(
        camera_id=frame.camera_id,
        frame_number=frame.frame_number,
        timestamp_seconds=frame.timestamp_seconds,
        det_class=DetClass.PERSON,
        box=box,
        score=_STUB_SCORE,
    )
