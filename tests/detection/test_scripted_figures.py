"""SMOKE-wiring stub: threshold and contours, not a model."""

from __future__ import annotations

import inspect

import cv2
import numpy as np

from trace_search.detection.scripted_figures import (
    ScriptedFiguresDetector,
    scripted_figures_detector,
)
from trace_search.detection.types import DetClass, Detector
from trace_search.ingest.video_source import Frame


def _frame(image: np.ndarray) -> Frame:
    return Frame(
        image=image,
        timestamp_seconds=1.5,
        frame_number=4,
        camera_id="cam-smoke",
    )


def _gray(width: int = 200, height: int = 160) -> np.ndarray:
    return np.full((height, width, 3), 40, dtype=np.uint8)


def test_scripted_figures_is_a_smoke_wiring_stub() -> None:
    """The class text says this is a SMOKE-wiring stub, not a real detector."""
    text = inspect.getdoc(ScriptedFiguresDetector) or ""
    assert "SMOKE-wiring stub" in text
    assert "not a real detector" in text


def test_scripted_figures_satisfies_the_detector_protocol() -> None:
    detector: Detector = scripted_figures_detector()
    found = detector.detect(_frame(_gray()))
    assert found == []


def test_uniform_frame_has_no_detections() -> None:
    detector = ScriptedFiguresDetector()
    assert detector.detect(_frame(_gray())) == []


def test_large_frame_maps_the_box_back_to_the_original_pixels() -> None:
    image = np.full((400, 800, 3), 40, dtype=np.uint8)
    cv2.rectangle(image, (100, 80), (300, 300), (20, 180, 40), -1)
    found = ScriptedFiguresDetector().detect(_frame(image))
    assert len(found) == 1
    x1, y1, x2, y2 = found[0].box
    assert x1 <= 200.0 <= x2
    assert y1 <= 190.0 <= y2


def test_flat_colour_figure_is_found_and_a_speck_is_not() -> None:
    image = _gray()
    cv2.rectangle(image, (30, 40), (80, 140), (180, 40, 90), -1)
    cv2.rectangle(image, (5, 5), (7, 7), (200, 20, 20), -1)
    detector = ScriptedFiguresDetector()

    found = detector.detect(_frame(image))

    assert len(found) == 1
    detection = found[0]
    assert detection.det_class is DetClass.PERSON
    assert detection.camera_id == "cam-smoke"
    assert detection.frame_number == 4
    assert detection.score >= 0.6
    x1, y1, x2, y2 = detection.box
    assert x1 <= 55.0 <= x2
    assert y1 <= 90.0 <= y2
    assert x2 - x1 < 80
    assert y2 - y1 < 140
