"""Run one camera from frames to stored tracks.

Each camera gets its own tracker. The function reads frames, detects, tracks,
and writes track rows. It does not keep frames or crops.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from trace_search.detection.types import Detector
from trace_search.ingest.video_source import Frame, VideoSource
from trace_search.tracking.store import TrackStore
from trace_search.tracking.types import Tracker, TrackObservation


@dataclass(frozen=True)
class RunStats:
    """Counts for one finished camera run.

    ``frames`` is how many frames were detected and tracked. ``detections``
    is how many boxes the detector returned. ``tracks`` is how many rows were
    written. ``errors`` is 0 on a returned result. A failed camera does not
    return stats: the error is recorded, rows from earlier frames are
    written, and the original exception is re-raised.
    """

    frames: int
    detections: int
    tracks: int
    errors: int


@dataclass
class _Progress:
    frames: int = 0
    detections: int = 0
    errors: int = 0
    rows: list[TrackObservation] = field(default_factory=list)


def _stats(progress: _Progress) -> RunStats:
    return RunStats(
        frames=progress.frames,
        detections=progress.detections,
        tracks=len(progress.rows),
        errors=progress.errors,
    )


def _open_source(source: VideoSource) -> bool:
    """Open a closed source. Return whether this call opened it."""
    if source.is_open():
        return False
    source.open()
    return True


def _close_source(source: VideoSource, opened_here: bool) -> None:
    if opened_here:
        source.close()


def _with_timestamp(observation: TrackObservation, timestamp_seconds: float) -> TrackObservation:
    return replace(observation, timestamp_seconds=timestamp_seconds)


def _track_frame(
    frame: Frame,
    detector: Detector,
    tracker: Tracker,
    clock_offset_seconds: float,
    progress: _Progress,
) -> None:
    found = detector.detect(frame)
    tracked = tracker.update(found)
    corrected = frame.corrected_timestamp(clock_offset_seconds)
    stamped = [_with_timestamp(item, corrected) for item in tracked]
    progress.frames += 1
    progress.detections += len(found)
    progress.rows.extend(stamped)


def _consume(
    source: VideoSource,
    detector: Detector,
    tracker: Tracker,
    clock_offset_seconds: float,
    progress: _Progress,
) -> None:
    for frame in source.frames():
        _track_frame(frame, detector, tracker, clock_offset_seconds, progress)


def _flush(
    store: TrackStore,
    run_id: str,
    fixture_id: str,
    derived_from: str,
    rows: Sequence[TrackObservation],
) -> None:
    store.write(run_id, fixture_id, derived_from, rows)


def _record_failure(camera_id: str, progress: _Progress, exc: BaseException) -> None:
    """Count the failure and attach that count to the exception."""
    progress.errors += 1
    exc.add_note(
        "recorded pipeline error "
        f"camera_id={camera_id} "
        f"frames={progress.frames} "
        f"detections={progress.detections} "
        f"tracks={len(progress.rows)} "
        f"errors={progress.errors}"
    )


def run_camera(
    source: VideoSource,
    detector: Detector,
    tracker_factory: Callable[[], Tracker],
    store: TrackStore,
    run_id: str,
    fixture_id: str,
    derived_from: str,
    clock_offset_seconds: float = 0.0,
) -> RunStats:
    """Detect, track, and store one camera.

    ``tracker_factory`` is called once for this camera. Timestamps written
    to the store come from ``Frame.corrected_timestamp``. On any exception
    the error is recorded, rows from earlier frames are written, and that
    exception is re-raised.
    """
    progress = _Progress()
    tracker = tracker_factory()
    opened_here = _open_source(source)
    try:
        _consume(source, detector, tracker, clock_offset_seconds, progress)
    except Exception as exc:
        _record_failure(source.camera_id, progress, exc)
        _flush(store, run_id, fixture_id, derived_from, progress.rows)
        raise
    finally:
        _close_source(source, opened_here)
    _flush(store, run_id, fixture_id, derived_from, progress.rows)
    return _stats(progress)


def run_cameras(
    sources: Sequence[VideoSource],
    detector: Detector,
    tracker_factory: Callable[[], Tracker],
    store: TrackStore,
    run_id: str,
    fixture_id: str,
    derived_from: str,
    clock_offset_seconds: float = 0.0,
) -> dict[str, RunStats]:
    """Run each source with a new tracker.

    ``tracker_factory`` is called once per camera, inside ``run_camera``.
    Trackers are not shared. Returned stats are keyed by ``camera_id``.
    """
    stats: dict[str, RunStats] = {}
    for source in sources:
        stats[source.camera_id] = run_camera(
            source,
            detector,
            tracker_factory,
            store,
            run_id,
            fixture_id,
            derived_from,
            clock_offset_seconds,
        )
    return stats
