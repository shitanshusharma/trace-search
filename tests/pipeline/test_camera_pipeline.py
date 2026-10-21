"""Per-camera pipeline: ingest, detect, track, persist.

Frames are in-memory numpy arrays. The track database is pytest's tmp_path,
outside the repository. No media file is read or written.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trace_search.detection import DetClass, Detection, ScriptedDetector
from trace_search.ingest.video_source import Frame, VideoSource
from trace_search.pipeline import RunStats, run_camera, run_cameras
from trace_search.tracking import ByteTracker
from trace_search.tracking.store import TrackStore

_RUN = "run-pipeline"
_FIXTURE = "FX-CODE-PIPE"
_ORIGIN = "CODE_TEST"
_CLOCK = 1_700_000_000.0
_OFFSET = 2.5
_FRAME_COUNT = 5


class MemoryVideoSource(VideoSource):
    """VideoSource whose frames already live in memory."""

    def __init__(self, camera_id: str, frames: Sequence[Frame]) -> None:
        super().__init__(camera_id)
        self._frames = tuple(frames)
        self._index = 0

    def open(self) -> None:
        self._index = 0
        self._is_open = True

    def close(self) -> None:
        self._is_open = False

    def read_frame(self) -> Frame | None:
        if self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def get_fps(self) -> float:
        return 10.0

    def get_frame_count(self) -> int | None:
        return len(self._frames)


class _RaisingDetector:
    """Scripted detector that raises on one frame number."""

    def __init__(self, inner: ScriptedDetector, fail_frame: int) -> None:
        self._inner = inner
        self._fail_frame = fail_frame

    def detect(self, frame: Frame) -> list[Detection]:
        if frame.frame_number == self._fail_frame:
            raise RuntimeError(f"detector failed on frame {frame.frame_number}")
        return self._inner.detect(frame)


def _box(frame_number: int) -> tuple[float, float, float, float]:
    """A box that overlaps the next frame and misses the first frame later."""
    left = 10.0 + (frame_number * 10.0)
    return (left, 10.0, left + 40.0, 90.0)


def _image() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _frames(camera_id: str, count: int, *, start: float = 1_000.0) -> tuple[Frame, ...]:
    return tuple(
        Frame(
            image=_image(),
            timestamp_seconds=start + float(index),
            frame_number=index,
            camera_id=camera_id,
        )
        for index in range(count)
    )


def _detection(
    camera_id: str,
    frame_number: int,
    *,
    timestamp_seconds: float = 0.0,
) -> Detection:
    return Detection(
        camera_id=camera_id,
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
        det_class=DetClass.PERSON,
        box=_box(frame_number),
        score=0.9,
    )


def _script(camera_ids: Sequence[str], count: int) -> dict[tuple[str, int], tuple[Detection, ...]]:
    script: dict[tuple[str, int], tuple[Detection, ...]] = {}
    for camera_id in camera_ids:
        for index in range(count):
            script[(camera_id, index)] = (_detection(camera_id, index),)
    return script


def _source(camera_id: str, count: int, *, start: float = 1_000.0) -> MemoryVideoSource:
    return MemoryVideoSource(camera_id, _frames(camera_id, count, start=start))


def _store(path: Path) -> TrackStore:
    return TrackStore(path, clock=lambda: _CLOCK)


def _factory(created: list[ByteTracker]) -> Callable[[], ByteTracker]:
    def build() -> ByteTracker:
        tracker = ByteTracker()
        created.append(tracker)
        return tracker

    return build


def _miskey_message(det_camera: str, det_frame: int) -> str:
    return (
        "scripted detection does not match script key "
        "(camera_id='cam-a', frame_number=0): "
        f"detection camera_id={det_camera!r} frame_number={det_frame!r}"
    )


def test_two_cameras_with_the_same_boxes_get_track_id_1(tmp_path: Path) -> None:
    """Fresh trackers: both cameras mint track id 1, and rows differ by camera only."""
    created: list[ByteTracker] = []
    cameras = ("cam-a", "cam-b")
    sources = [_source(camera_id, _FRAME_COUNT) for camera_id in cameras]
    store = _store(tmp_path / "tracks.sqlite")
    stats = run_cameras(
        sources,
        ScriptedDetector(_script(cameras, _FRAME_COUNT)),
        _factory(created),
        store,
        _RUN,
        _FIXTURE,
        _ORIGIN,
    )

    assert len(created) == 2
    assert created[0] is not created[1]
    assert stats["cam-a"] == RunStats(
        frames=_FRAME_COUNT,
        detections=_FRAME_COUNT,
        tracks=_FRAME_COUNT,
        errors=0,
    )
    assert stats["cam-b"] == stats["cam-a"]

    rows_a = store.read("cam-a", run_id=_RUN)
    rows_b = store.read("cam-b", run_id=_RUN)
    assert len(rows_a) == stats["cam-a"].tracks
    assert len(rows_b) == stats["cam-b"].tracks
    assert store.count(run_id=_RUN) == stats["cam-a"].tracks + stats["cam-b"].tracks
    assert [row.track_id for row in rows_a] == [1] * _FRAME_COUNT
    assert [row.track_id for row in rows_b] == [1] * _FRAME_COUNT
    for left, right in zip(rows_a, rows_b, strict=True):
        assert left.camera_id == "cam-a"
        assert right.camera_id == "cam-b"
        assert replace(right, camera_id=left.camera_id) == left


def test_persisted_rows_equal_run_stats_tracks(tmp_path: Path) -> None:
    """The store holds exactly RunStats.tracks rows for the run."""
    store = _store(tmp_path / "tracks.sqlite")
    stats = run_camera(
        _source("cam-a", _FRAME_COUNT),
        ScriptedDetector(_script(("cam-a",), _FRAME_COUNT)),
        ByteTracker,
        store,
        _RUN,
        _FIXTURE,
        _ORIGIN,
    )

    assert stats.tracks == _FRAME_COUNT
    assert stats.errors == 0
    assert store.count(run_id=_RUN) == stats.tracks
    assert len(store.read("cam-a", run_id=_RUN)) == stats.tracks


def test_detector_raise_on_frame_5_flushes_then_reraises(tmp_path: Path) -> None:
    """Frame 5 raises, frames 0-4 stay in the store, and the error is not swallowed."""
    count = 8
    source = _source("cam-a", count)
    detector = _RaisingDetector(ScriptedDetector(_script(("cam-a",), count)), fail_frame=5)
    store = _store(tmp_path / "tracks.sqlite")

    with pytest.raises(RuntimeError, match="detector failed on frame 5") as caught:
        run_camera(source, detector, ByteTracker, store, _RUN, _FIXTURE, _ORIGIN)

    notes = getattr(caught.value, "__notes__", [])
    assert any("recorded pipeline error" in note and "errors=1" in note for note in notes)
    rows = store.read("cam-a", run_id=_RUN)
    assert [row.frame_number for row in rows] == [0, 1, 2, 3, 4]
    assert [row.track_id for row in rows] == [1, 1, 1, 1, 1]
    assert store.count(run_id=_RUN) == 5


def test_clock_offset_is_stored_on_each_row(tmp_path: Path) -> None:
    """Stored timestamps are the source timestamps shifted by +2.5 seconds."""
    frames = _frames("cam-a", _FRAME_COUNT)
    source = MemoryVideoSource("cam-a", frames)
    source.open()
    store = _store(tmp_path / "tracks.sqlite")
    stats = run_camera(
        source,
        ScriptedDetector(_script(("cam-a",), _FRAME_COUNT)),
        ByteTracker,
        store,
        _RUN,
        _FIXTURE,
        _ORIGIN,
        _OFFSET,
    )

    rows = store.read("cam-a", run_id=_RUN)
    assert stats.tracks == len(rows) == len(frames)
    assert stats.errors == 0
    for row, frame in zip(rows, frames, strict=True):
        assert row.timestamp_seconds == frame.corrected_timestamp(_OFFSET)
        assert row.timestamp_seconds == frame.timestamp_seconds + _OFFSET


def test_mis_keyed_scripted_detection_raises(tmp_path: Path) -> None:
    """A scripted box whose camera or frame disagrees with its key is an error."""
    cases = (
        ("other-cam", 0),
        ("cam-a", 4),
        ("other-cam", 4),
    )
    for index, (det_camera, det_frame) in enumerate(cases):
        bad = _detection(det_camera, det_frame)
        script = {("cam-a", 0): (bad,)}
        store = _store(tmp_path / f"mis-{index}.sqlite")
        with pytest.raises(ValueError) as caught:
            run_camera(
                _source("cam-a", 1),
                ScriptedDetector(script),
                ByteTracker,
                store,
                _RUN,
                _FIXTURE,
                _ORIGIN,
            )
        assert str(caught.value) == _miskey_message(det_camera, det_frame)
        assert store.count(run_id=_RUN) == 0
