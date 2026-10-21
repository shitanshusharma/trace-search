"""Acceptance tests for the in-repo ByteTrack-style tracker.

Every sequence is scripted here (FX-CODE style). No image, clip, or weight
file is read.
"""

import math
from collections.abc import Sequence

import numpy as np
import pytest

from trace_search.detection import DetClass, Detection
from trace_search.tracking import ByteTracker, Tracker, TrackObservation
from trace_search.tracking.bytetrack import (
    DEFAULT_HIGH_THRESH,
    DEFAULT_LOW_THRESH,
    DEFAULT_MATCH_IOU,
    DEFAULT_MAX_AGE_FRAMES,
    DEFAULT_MIN_HITS,
    _linear_sum_assignment,
)

_WIDTH = 40.0
_HEIGHT = 80.0


def _box(center_x: float, center_y: float) -> tuple[float, float, float, float]:
    return (
        center_x - _WIDTH / 2.0,
        center_y - _HEIGHT / 2.0,
        center_x + _WIDTH / 2.0,
        center_y + _HEIGHT / 2.0,
    )


def _detection(
    frame: int,
    box: tuple[float, float, float, float],
    *,
    det_class: DetClass = DetClass.PERSON,
    score: float = 0.9,
    camera_id: str = "cam-fx",
) -> Detection:
    return Detection(
        camera_id=camera_id,
        frame_number=frame,
        timestamp_seconds=frame / 10.0,
        det_class=det_class,
        box=box,
        score=score,
    )


def _straight(frame: int) -> list[Detection]:
    """Two lanes, same direction, no overlap, 30 frames."""
    step = 8.0 * frame
    return [
        _detection(frame, _box(40.0 + step, 50.0)),
        _detection(frame, _box(40.0 + step, 220.0)),
    ]


def _crossing(frame: int) -> list[Detection]:
    """Opposite straight lines whose centers meet at frame 15."""
    right = 100.0 + 12.0 * frame
    left = 100.0 + 12.0 * (30 - frame)
    return [
        _detection(frame, _box(right, 100.0)),
        _detection(frame, _box(left, 100.0)),
    ]


def _ids_for(
    observations: Sequence[TrackObservation],
    box: tuple[float, float, float, float],
) -> list[int]:
    return [obs.track_id for obs in observations if obs.box == box]


def _run(
    frames: Sequence[Sequence[Detection]],
    tracker: ByteTracker | None = None,
) -> list[list[TrackObservation]]:
    active = ByteTracker() if tracker is None else tracker
    return [active.update(frame) for frame in frames]


def test_defaults_match_the_documented_paper_settings() -> None:
    assert DEFAULT_HIGH_THRESH == 0.6
    assert DEFAULT_LOW_THRESH == 0.1
    assert DEFAULT_MATCH_IOU == 0.2
    assert DEFAULT_MAX_AGE_FRAMES == 30
    assert DEFAULT_MIN_HITS == 1


def test_byte_tracker_satisfies_tracker_protocol() -> None:
    """mypy accepts ByteTracker where a Tracker is expected."""
    tracker: Tracker = ByteTracker()
    box = _box(40.0, 50.0)
    first = tracker.update([_detection(0, box)])
    assert len(first) == 1
    assert first[0].track_id == 1
    empty = tracker.update([])
    assert empty == []
    tracker.reset()
    assert tracker.update([]) == []
    restarted = tracker.update([_detection(2, box)])
    assert restarted[0].track_id == 1
    assert restarted[0].camera_id == "cam-fx"


def test_two_straight_lines_keep_distinct_stable_ids_for_30_frames() -> None:
    tracker = ByteTracker()
    seen: list[tuple[int, int]] = []
    for frame in range(30):
        detections = _straight(frame)
        observations = tracker.update(detections)
        upper = _ids_for(observations, detections[0].box)
        lower = _ids_for(observations, detections[1].box)
        assert upper == [1]
        assert lower == [2]
        seen.append((upper[0], lower[0]))
    assert len(seen) == 30
    assert len({pair[0] for pair in seen}) == 1
    assert len({pair[1] for pair in seen}) == 1


def test_crossing_at_frame_15_keeps_ids_after_the_cross() -> None:
    tracker = ByteTracker()
    before = tracker.update(_crossing(0))
    right_id = _ids_for(before, _crossing(0)[0].box)[0]
    left_id = _ids_for(before, _crossing(0)[1].box)[0]
    assert right_id != left_id
    for frame in range(1, 30):
        detections = _crossing(frame)
        observations = tracker.update(detections)
        # Frame 15 is the meeting point: the two boxes are the same, so the
        # id check is the frames on either side of the cross.
        if detections[0].box == detections[1].box:
            assert frame == 15
            assert {obs.track_id for obs in observations} == {right_id, left_id}
            continue
        assert _ids_for(observations, detections[0].box) == [right_id]
        assert _ids_for(observations, detections[1].box) == [left_id]


def test_gap_inside_max_age_keeps_id_and_past_max_age_mints_a_new_one() -> None:
    max_age = 4
    box = _box(80.0, 90.0)

    short = ByteTracker(max_age_frames=max_age)
    original = short.update([_detection(0, box)])
    assert [obs.track_id for obs in original] == [1]
    for frame in range(1, max_age):
        assert short.update([]) == []
    restored = short.update([_detection(max_age, box)])
    assert [obs.track_id for obs in restored] == [1]

    exact = ByteTracker(max_age_frames=max_age)
    exact.update([_detection(0, box)])
    for frame in range(1, max_age + 1):
        assert exact.update([]) == []
    still = exact.update([_detection(max_age + 1, box)])
    assert [obs.track_id for obs in still] == [1]

    long = ByteTracker(max_age_frames=max_age)
    long.update([_detection(0, box)])
    for frame in range(1, max_age + 2):
        assert long.update([]) == []
    minted = long.update([_detection(max_age + 2, box)])
    assert [obs.track_id for obs in minted] == [2]


def test_score_between_thresholds_extends_but_does_not_start() -> None:
    box = _box(70.0, 60.0)
    mid_score = (DEFAULT_LOW_THRESH + DEFAULT_HIGH_THRESH) / 2.0

    alone = ByteTracker()
    for frame in range(4):
        assert alone.update([_detection(frame, box, score=mid_score)]) == []

    tracker = ByteTracker()
    started = tracker.update([_detection(0, box, score=0.95)])
    assert [obs.track_id for obs in started] == [1]
    extended = tracker.update([_detection(1, box, score=mid_score)])
    assert [obs.track_id for obs in extended] == [1]
    assert extended[0].score == mid_score

    assert tracker.update([_detection(2, box, score=DEFAULT_LOW_THRESH - 0.05)]) == []
    back = tracker.update([_detection(3, box, score=0.95)])
    assert [obs.track_id for obs in back] == [1]


def test_person_and_backpack_at_the_same_box_get_different_ids() -> None:
    box = _box(55.0, 65.0)
    tracker = ByteTracker()
    observations = tracker.update(
        [
            _detection(0, box, det_class=DetClass.BACKPACK, score=0.91),
            _detection(0, box, det_class=DetClass.PERSON, score=0.93),
        ]
    )
    by_class = {obs.det_class: obs.track_id for obs in observations}
    assert set(by_class) == {DetClass.PERSON, DetClass.BACKPACK}
    assert by_class[DetClass.PERSON] != by_class[DetClass.BACKPACK]
    assert observations[0].box == observations[1].box == box


def test_each_camera_tracker_numbers_ids_on_its_own() -> None:
    box = _box(20.0, 30.0)
    left = ByteTracker().update([_detection(0, box, camera_id="cam-a")])
    right = ByteTracker().update([_detection(0, box, camera_id="cam-b")])
    assert left[0].track_id == right[0].track_id == 1
    assert left[0].camera_id == "cam-a"
    assert right[0].camera_id == "cam-b"


def test_same_input_is_identical_on_two_runs() -> None:
    straight = [_straight(frame) for frame in range(30)]
    crossing = [_crossing(frame) for frame in range(30)]
    assert _run(straight) == _run(straight)
    assert _run(crossing) == _run(crossing)


def test_min_hits_waits_for_that_many_matches_before_output() -> None:
    box = _box(25.0, 35.0)
    tracker = ByteTracker(min_hits=3)
    assert tracker.update([_detection(0, box)]) == []
    assert tracker.update([_detection(1, box)]) == []
    confirmed = tracker.update([_detection(2, box)])
    assert [obs.track_id for obs in confirmed] == [1]


def test_unconfirmed_track_is_dropped_when_it_misses() -> None:
    box = _box(25.0, 35.0)
    tracker = ByteTracker(min_hits=2)
    assert tracker.update([_detection(0, box)]) == []
    assert tracker.update([]) == []
    assert tracker.update([_detection(2, box)]) == []
    confirmed = tracker.update([_detection(3, box)])
    assert [obs.track_id for obs in confirmed] == [2]


def test_match_rejects_pairs_below_match_iou() -> None:
    origin = _box(40.0, 40.0)
    far = _box(400.0, 400.0)
    tracker = ByteTracker(match_iou=0.5)
    first = tracker.update([_detection(0, origin)])
    second = tracker.update([_detection(1, far)])
    assert [obs.track_id for obs in second] == [first[0].track_id + 1]
    back = tracker.update([_detection(2, origin), _detection(2, far)])
    assert _ids_for(back, origin) == [first[0].track_id]
    assert _ids_for(back, far) == [first[0].track_id + 1]


def test_linear_sum_assignment_covers_known_rectangular_and_empty_costs() -> None:
    known = np.array(
        [[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]],
        dtype=np.float64,
    )
    rows, cols = _linear_sum_assignment(known)
    assert rows == [0, 1, 2]
    assert cols == [1, 0, 2]
    assert all(type(index) is int for index in rows + cols)

    rectangular = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0]], dtype=np.float64)
    rows, cols = _linear_sum_assignment(rectangular)
    assert rows == [0, 1]
    assert cols == [1, 0]
    assert all(type(index) is int for index in rows + cols)

    for shape in ((0, 0), (0, 3), (2, 0)):
        rows, cols = _linear_sum_assignment(np.zeros(shape, dtype=np.float64))
        assert rows == []
        assert cols == []


def test_crossing_ids_follow_motion_when_detection_order_is_reversed() -> None:
    """After the boxes meet, the input list is reversed. Ids follow the motion."""
    tracker = ByteTracker()
    opening = _crossing(0)
    before = tracker.update(opening)
    right_id = _ids_for(before, opening[0].box)[0]
    left_id = _ids_for(before, opening[1].box)[0]
    assert right_id != left_id
    for frame in range(1, 30):
        ordered = _crossing(frame)
        fed = list(reversed(ordered)) if frame > 15 else ordered
        observations = tracker.update(fed)
        if ordered[0].box == ordered[1].box:
            assert frame == 15
            assert {obs.track_id for obs in observations} == {right_id, left_id}
            continue
        assert _ids_for(observations, ordered[0].box) == [right_id]
        assert _ids_for(observations, ordered[1].box) == [left_id]


def test_coasting_keeps_id_when_a_moving_box_returns_on_its_predicted_path() -> None:
    """A constant-velocity box that vanishes for a few frames keeps id 1.

    The gap is shorter than ``max_age_frames``. The box returns where that
    velocity puts it.
    """
    speed = 4.0
    seen = 8
    gap = 4
    assert gap < DEFAULT_MAX_AGE_FRAMES
    tracker = ByteTracker()

    def center(frame: int) -> tuple[float, float, float, float]:
        return _box(40.0 + speed * frame, 80.0)

    for frame in range(seen):
        observations = tracker.update([_detection(frame, center(frame))])
        assert [obs.track_id for obs in observations] == [1]
    for frame in range(seen, seen + gap):
        assert tracker.update([]) == []
    returned = seen + gap
    observations = tracker.update([_detection(returned, center(returned))])
    assert [obs.track_id for obs in observations] == [1]
    assert observations[0].box == center(returned)


def test_score_equal_to_high_thresh_can_start_a_track() -> None:
    box = _box(70.0, 60.0)
    started = ByteTracker().update([_detection(0, box, score=DEFAULT_HIGH_THRESH)])
    assert [obs.track_id for obs in started] == [1]
    assert started[0].score == DEFAULT_HIGH_THRESH


def test_score_at_low_thresh_is_dropped_and_just_above_is_low() -> None:
    box = _box(70.0, 60.0)
    just_above = math.nextafter(DEFAULT_LOW_THRESH, 1.0)

    assert ByteTracker().update([_detection(0, box, score=DEFAULT_LOW_THRESH)]) == []
    assert ByteTracker().update([_detection(0, box, score=just_above)]) == []

    held = ByteTracker()
    started = held.update([_detection(0, box, score=0.95)])
    assert [obs.track_id for obs in started] == [1]
    assert held.update([_detection(1, box, score=DEFAULT_LOW_THRESH)]) == []
    extended = held.update([_detection(2, box, score=just_above)])
    assert [obs.track_id for obs in extended] == [1]
    assert extended[0].score == just_above


def test_update_output_is_sorted_by_track_id() -> None:
    first_box = _box(40.0, 50.0)
    second_box = _box(40.0, 220.0)
    tracker = ByteTracker()
    opened = tracker.update(
        [
            _detection(0, first_box, score=0.95),
            _detection(0, second_box, score=0.95),
        ]
    )
    assert [obs.track_id for obs in opened] == [1, 2]
    # The high box matches track 2 in pass 1. The low box matches track 1
    # in pass 2. Pass order would emit 2 then 1; the result is sorted.
    observations = tracker.update(
        [
            _detection(1, second_box, score=0.95),
            _detection(1, first_box, score=(DEFAULT_LOW_THRESH + DEFAULT_HIGH_THRESH) / 2.0),
        ]
    )
    assert [obs.track_id for obs in observations] == [1, 2]
    assert observations[0].box == first_box
    assert observations[1].box == second_box


def test_low_score_pass_does_not_take_a_track_matched_in_the_first_pass() -> None:
    origin = _box(100.0, 100.0)
    shifted = _box(120.0, 100.0)
    tracker = ByteTracker()
    started = tracker.update([_detection(0, origin, score=0.95)])
    assert [obs.track_id for obs in started] == [1]
    high = _detection(1, shifted, score=0.95)
    low = _detection(1, origin, score=(DEFAULT_LOW_THRESH + DEFAULT_HIGH_THRESH) / 2.0)
    observations = tracker.update([high, low])
    assert [obs.track_id for obs in observations] == [1]
    assert observations[0].box == shifted
    assert observations[0].score == high.score


def test_bad_parameters_raise_value_error() -> None:
    with pytest.raises(ValueError):
        ByteTracker(high_thresh=1.2)
    with pytest.raises(ValueError):
        ByteTracker(high_thresh=float("nan"))
    with pytest.raises(ValueError):
        ByteTracker(low_thresh=-0.1)
    with pytest.raises(ValueError):
        ByteTracker(low_thresh=0.8, high_thresh=0.2)
    with pytest.raises(ValueError):
        ByteTracker(match_iou=1.1)
    with pytest.raises(ValueError):
        ByteTracker(max_age_frames=-1)
    with pytest.raises(ValueError):
        ByteTracker(min_hits=0)
    with pytest.raises(ValueError):
        ByteTracker(max_age_frames=True)
    with pytest.raises(ValueError):
        ByteTracker(high_thresh=True)
