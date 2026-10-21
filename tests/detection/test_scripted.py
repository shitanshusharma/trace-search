"""Tests for the scripted detector used by FX-CODE wiring."""

import numpy as np

from trace_search.detection import DetClass, Detection, Detector, ScriptedDetector
from trace_search.ingest.video_source import Frame


def _frame(camera_id: str, frame_number: int, *, timestamp_seconds: float = 1.25) -> Frame:
    return Frame(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        timestamp_seconds=timestamp_seconds,
        frame_number=frame_number,
        camera_id=camera_id,
    )


def _detection(
    *,
    camera_id: str = "cam-a",
    frame_number: int = 3,
    det_class: DetClass = DetClass.PERSON,
    box: tuple[float, float, float, float] = (1.0, 2.0, 11.0, 22.0),
    score: float = 0.8,
) -> Detection:
    return Detection(
        camera_id=camera_id,
        frame_number=frame_number,
        timestamp_seconds=1.25,
        det_class=det_class,
        box=box,
        score=score,
    )


def test_scripted_detector_satisfies_detector_protocol() -> None:
    """mypy accepts ScriptedDetector where a Detector is expected."""
    person = _detection()
    bag = _detection(
        det_class=DetClass.BACKPACK,
        box=(0.0, 0.0, 4.0, 6.0),
        score=0.4,
    )
    script = {("cam-a", 3): [person, bag]}
    detector: Detector = ScriptedDetector(script)
    frame = _frame("cam-a", 3)

    first = detector.detect(frame)
    second = detector.detect(frame)

    assert first == [person, bag]
    assert second == [person, bag]
    assert first == second
    assert first is not second
    first.append(_detection(det_class=DetClass.HANDBAG, score=0.2))
    assert detector.detect(frame) == [person, bag]


def test_same_script_returns_identical_lists_on_two_runs() -> None:
    handbag = _detection(det_class=DetClass.HANDBAG, score=0.55)
    script = {("cam-b", 0): (handbag,)}
    detector = ScriptedDetector(script)
    frame = _frame("cam-b", 0, timestamp_seconds=9.0)

    first = detector.detect(frame)
    second = detector.detect(frame)

    assert first == second == [handbag]
    assert first is not second


def test_unknown_frame_returns_empty_list() -> None:
    script = {("cam-a", 3): [_detection()]}
    detector = ScriptedDetector(script)

    missing_frame = detector.detect(_frame("cam-a", 4))
    missing_camera = detector.detect(_frame("cam-z", 3))
    missing_both = detector.detect(_frame("cam-z", 9))
    empty_script = ScriptedDetector({}).detect(_frame("cam-a", 3))

    assert missing_frame == []
    assert missing_camera == []
    assert missing_both == []
    assert empty_script == []
    assert missing_frame is not missing_camera


def test_image_is_not_part_of_the_script_key() -> None:
    person = _detection()
    detector = ScriptedDetector({("cam-a", 3): [person]})
    other = _frame("cam-a", 3)
    other.image = np.ones((2, 2, 3), dtype=np.uint8)

    assert detector.detect(other) == [person]


def test_docstring_limits_output_to_wiring() -> None:
    text = ScriptedDetector.__doc__
    assert text is not None
    assert "CODE_TEST" in text
    assert "SMOKE" in text
    assert "never evidence" in text


def test_caller_dict_mutation_does_not_change_detector_output() -> None:
    """A later add, replace, or delete on the caller's mapping is ignored."""
    person = _detection()
    script = {("cam-a", 3): [person]}
    detector = ScriptedDetector(script)
    frame = _frame("cam-a", 3)
    before = detector.detect(frame)

    script[("cam-a", 9)] = [_detection(frame_number=9)]
    script[("cam-a", 3)] = [_detection(det_class=DetClass.HANDBAG, score=0.1)]
    del script[("cam-a", 3)]

    assert detector.detect(frame) == before == [person]


def test_caller_list_append_does_not_change_detector_output() -> None:
    """Appending to the caller's list after construction does not change detect."""
    person = _detection()
    recorded = [person]
    script = {("cam-a", 3): recorded}
    detector = ScriptedDetector(script)
    frame = _frame("cam-a", 3)
    before = detector.detect(frame)

    recorded.append(_detection(det_class=DetClass.BACKPACK, score=0.3))

    assert detector.detect(frame) == before == [person]
