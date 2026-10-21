"""Contract tests for TrackObservation and Tracker."""

import math
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, fields

import pytest

from trace_search.attributes.attributes import Box
from trace_search.detection import DetClass, Detection
from trace_search.tracking import Tracker, TrackObservation

_EXACT_FIELDS = {
    "camera_id",
    "track_id",
    "det_class",
    "frame_number",
    "timestamp_seconds",
    "box",
    "score",
}


def _observation(
    *,
    camera_id: str = "cam-1",
    track_id: int = 3,
    det_class: DetClass = DetClass.PERSON,
    frame_number: int = 7,
    timestamp_seconds: float = 12.5,
    box: tuple[float, float, float, float] = (1.0, 2.0, 11.0, 22.0),
    score: float = 0.75,
) -> TrackObservation:
    return TrackObservation(
        camera_id=camera_id,
        track_id=track_id,
        det_class=det_class,
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
        box=box,
        score=score,
    )


def _detection(
    *,
    camera_id: str = "cam-9",
    frame_number: int = 4,
    timestamp_seconds: float = 3.5,
    det_class: DetClass = DetClass.HANDBAG,
    box: tuple[float, float, float, float] = (1.0, 2.0, 8.0, 9.0),
    score: float = 0.6,
) -> Detection:
    return Detection(
        camera_id=camera_id,
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
        det_class=det_class,
        box=box,
        score=score,
    )


def test_track_fields_exact() -> None:
    """A planted extra field changes this set and fails the test."""
    assert {field.name for field in fields(TrackObservation)} == _EXACT_FIELDS


def test_track_reuses_detection_types() -> None:
    """Shared fields use the same types as Detection."""
    notes = TrackObservation.__annotations__
    detection_notes = Detection.__annotations__
    assert notes["camera_id"] is detection_notes["camera_id"] is str
    assert notes["track_id"] is int
    assert notes["det_class"] is detection_notes["det_class"] is DetClass
    assert notes["frame_number"] is detection_notes["frame_number"] is int
    assert notes["timestamp_seconds"] is detection_notes["timestamp_seconds"] is float
    assert notes["box"] is detection_notes["box"] is Box
    assert notes["score"] is detection_notes["score"] is float


def test_valid_observation_keeps_each_field() -> None:
    found = _observation(det_class=DetClass.BACKPACK, score=0.0, track_id=0)
    assert found.camera_id == "cam-1"
    assert found.track_id == 0
    assert found.det_class is DetClass.BACKPACK
    assert found.frame_number == 7
    assert found.timestamp_seconds == 12.5
    assert found.box == (1.0, 2.0, 11.0, 22.0)
    assert found.score == 0.0

    edges = _observation(score=1.0, box=(0, 0, 1, 1), frame_number=0, track_id=1)
    assert edges.score == 1.0
    assert edges.box == (0, 0, 1, 1)
    assert edges.frame_number == 0
    assert edges.track_id == 1


def test_same_track_id_is_allowed_on_two_cameras() -> None:
    """A track id is unique per camera only."""
    left = _observation(camera_id="cam-a", track_id=4)
    right = _observation(camera_id="cam-b", track_id=4)
    assert left.track_id == right.track_id == 4
    assert left.camera_id != right.camera_id


def test_observation_is_frozen() -> None:
    found = _observation()
    with pytest.raises(FrozenInstanceError):
        found.track_id = 9


@pytest.mark.parametrize(
    ("kwargs",),
    [
        ({"track_id": -1},),
        ({"track_id": True},),
        ({"track_id": False},),
        ({"track_id": 1.5},),
        ({"box": (5.0, 0.0, 5.0, 4.0)},),
        ({"box": (5.0, 0.0, 4.0, 4.0)},),
        ({"box": (0.0, 3.0, 4.0, 3.0)},),
        ({"box": (0.0, 3.0, 4.0, 2.0)},),
        ({"score": -0.01},),
        ({"score": 1.01},),
        ({"box": (math.nan, 0.0, 2.0, 2.0)},),
        ({"box": (0.0, math.inf, 2.0, 2.0)},),
        ({"box": (0.0, 0.0, math.nan, 2.0)},),
        ({"box": (0.0, 0.0, 2.0, -math.inf)},),
        ({"timestamp_seconds": math.nan},),
        ({"timestamp_seconds": math.inf},),
        ({"timestamp_seconds": -math.inf},),
        ({"score": math.nan},),
        ({"score": math.inf},),
        ({"score": -math.inf},),
        ({"frame_number": -1},),
        ({"frame_number": True},),
        ({"frame_number": 1.5},),
        ({"camera_id": ""},),
        ({"camera_id": 12},),
    ],
    ids=[
        "track-id-negative",
        "track-id-true",
        "track-id-false",
        "track-id-not-int",
        "x2-equal-x1",
        "x2-less-than-x1",
        "y2-equal-y1",
        "y2-less-than-y1",
        "score-below-zero",
        "score-above-one",
        "box-nan",
        "box-inf",
        "box-x2-nan",
        "box-y2-neg-inf",
        "timestamp-nan",
        "timestamp-inf",
        "timestamp-neg-inf",
        "score-nan",
        "score-inf",
        "score-neg-inf",
        "frame-number-negative",
        "frame-number-bool",
        "frame-number-not-int",
        "camera-id-empty",
        "camera-id-not-str",
    ],
)
def test_invalid_observation_raises_value_error(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _observation(**kwargs)


def test_tracker_protocol_updates_and_resets() -> None:
    class _StubTracker:
        def update(self, detections: Sequence[Detection]) -> list[TrackObservation]:
            return [
                _observation(
                    camera_id=detection.camera_id,
                    track_id=index,
                    det_class=detection.det_class,
                    frame_number=detection.frame_number,
                    timestamp_seconds=detection.timestamp_seconds,
                    box=detection.box,
                    score=detection.score,
                )
                for index, detection in enumerate(detections)
            ]

        def reset(self) -> None:
            return None

    tracker: Tracker = _StubTracker()
    detection = _detection()
    found = tracker.update([detection])
    assert len(found) == 1
    assert found[0].camera_id == detection.camera_id
    assert found[0].track_id == 0
    assert found[0].det_class is detection.det_class
    assert found[0].frame_number == detection.frame_number
    assert found[0].timestamp_seconds == detection.timestamp_seconds
    assert found[0].box == detection.box
    assert found[0].score == detection.score
    assert tracker.reset() is None
    assert tracker.update([]) == []
