"""Contract tests for Detection, DetClass, and Detector."""

import math
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from trace_search.attributes.attributes import Box
from trace_search.detection import DetClass, Detection, Detector
from trace_search.ingest.video_source import Frame

_EXACT_FIELDS = {
    "camera_id",
    "frame_number",
    "timestamp_seconds",
    "det_class",
    "box",
    "score",
}


def _detection(
    *,
    camera_id: str = "cam-1",
    frame_number: int = 7,
    timestamp_seconds: float = 12.5,
    det_class: DetClass = DetClass.PERSON,
    box: tuple[float, float, float, float] = (1.0, 2.0, 11.0, 22.0),
    score: float = 0.75,
) -> Detection:
    return Detection(
        camera_id=camera_id,
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
        det_class=det_class,
        box=box,
        score=score,
    )


def test_detection_fields_exact() -> None:
    """A planted extra field changes this set and fails the test."""
    assert {field.name for field in fields(Detection)} == _EXACT_FIELDS


def test_det_class_members_exact() -> None:
    assert {member.name for member in DetClass} == {"PERSON", "BACKPACK", "HANDBAG"}
    assert DetClass.PERSON.value == "person"
    assert DetClass.BACKPACK.value == "backpack"
    assert DetClass.HANDBAG.value == "handbag"


def test_box_reuses_attributes_alias() -> None:
    """The field stores the attributes module alias, not a fresh tuple alias."""
    assert Detection.__annotations__["box"] is Box


def test_valid_detection_keeps_each_field() -> None:
    found = _detection(det_class=DetClass.BACKPACK, score=0.0)
    assert found.camera_id == "cam-1"
    assert found.frame_number == 7
    assert found.timestamp_seconds == 12.5
    assert found.det_class is DetClass.BACKPACK
    assert found.box == (1.0, 2.0, 11.0, 22.0)
    assert found.score == 0.0

    edges = _detection(score=1.0, box=(0, 0, 1, 1), frame_number=0)
    assert edges.score == 1.0
    assert edges.box == (0, 0, 1, 1)
    assert edges.frame_number == 0


def test_detection_is_frozen() -> None:
    found = _detection()
    with pytest.raises(FrozenInstanceError):
        found.score = 0.1


@pytest.mark.parametrize(
    ("kwargs",),
    [
        ({"box": (5.0, 0.0, 5.0, 4.0)},),
        ({"box": (5.0, 0.0, 4.0, 4.0)},),
        ({"box": (0.0, 3.0, 4.0, 3.0)},),
        ({"box": (0.0, 3.0, 4.0, 2.0)},),
        ({"score": -0.01},),
        ({"score": 1.01},),
        ({"box": (math.nan, 0.0, 2.0, 2.0)},),
        ({"box": (0.0, math.inf, 2.0, 2.0)},),
        ({"timestamp_seconds": math.nan},),
        ({"timestamp_seconds": math.inf},),
        ({"timestamp_seconds": -math.inf},),
        ({"score": math.nan},),
        ({"score": math.inf},),
        ({"score": -math.inf},),
        ({"frame_number": -1},),
        ({"camera_id": ""},),
    ],
    ids=[
        "x2-equal-x1",
        "x2-less-than-x1",
        "y2-equal-y1",
        "y2-less-than-y1",
        "score-below-zero",
        "score-above-one",
        "box-nan",
        "box-inf",
        "timestamp-nan",
        "timestamp-inf",
        "timestamp-neg-inf",
        "score-nan",
        "score-inf",
        "score-neg-inf",
        "frame-number-negative",
        "camera-id-empty",
    ],
)
def test_invalid_detection_raises_value_error(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _detection(**kwargs)


def test_detector_protocol_returns_detections() -> None:
    class _StubDetector:
        def detect(self, frame: Frame) -> list[Detection]:
            return [
                _detection(
                    camera_id=frame.camera_id,
                    frame_number=frame.frame_number,
                    timestamp_seconds=frame.timestamp_seconds,
                    det_class=DetClass.HANDBAG,
                )
            ]

    detector: Detector = _StubDetector()
    frame = Frame(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        timestamp_seconds=3.5,
        frame_number=9,
        camera_id="cam-9",
    )
    found = detector.detect(frame)
    assert len(found) == 1
    assert found[0].camera_id == "cam-9"
    assert found[0].frame_number == 9
    assert found[0].timestamp_seconds == 3.5
    assert found[0].det_class is DetClass.HANDBAG
