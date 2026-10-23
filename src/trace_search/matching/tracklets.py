"""Per-camera track summaries loaded from the track store.

A tracklet is one ``(camera_id, track_id)`` for one detection class in one
run. The record has no cross-camera id.

Timestamps are the values already stored on the rows. This module does not
add a clock offset.

Rows inside one tracklet are sorted once, by ``(timestamp_seconds, frame_number)``.
``TrackStore.read`` returns ``rowid`` order, and ``sorted`` is stable, so two
rows with the same timestamp and frame keep that row order. ``t_start`` and
``first_box`` come from the first row. ``t_end`` and ``last_box`` come from
the last row. ``frames`` lists every ``frame_number`` in that order, including
repeats. ``n_obs`` is the number of rows.

``camera_ids=None`` asks ``TrackStore.camera_ids`` for every camera that
still has a row for ``run_id``. Expired rows are purged once before that
lookup and before any read. This module opens no database connection.
An empty collection is refused. A bare ``str`` is refused, because a ``str``
is not a collection of camera ids. An unknown camera id adds no tracklet.
Repeated camera ids are read once.

Ids are refused, not rewritten. A whitespace-only id is empty. An id also
raises when it differs from ``str.strip`` of itself, when NFKC changes it,
or when it contains a format character (Unicode category ``Cf``, including
U+200B, U+200C, U+200D, U+2060 and U+FEFF). The id is not stripped, not
case-folded, and not NFKC-folded.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from trace_search.attributes.attributes import Box
from trace_search.detection.types import DetClass
from trace_search.tracking.store import TrackStore
from trace_search.tracking.types import TrackObservation

__all__ = ["Tracklet", "load_tracklets"]


@dataclass(frozen=True)
class Tracklet:
    """One per-camera track, summarized from stored rows.

    ``frames`` is ``frame_number`` in ``(timestamp_seconds, frame_number)``
    order, with store row order kept when those two tie. ``first_box`` and
    ``t_start`` belong to the first row in that order. ``last_box`` and
    ``t_end`` belong to the last row. ``n_obs`` is the number of rows.
    ``track_id`` is the per-camera track id from the store.
    """

    camera_id: str
    track_id: int
    det_class: DetClass
    run_id: str
    t_start: float
    t_end: float
    n_obs: int
    first_box: Box
    last_box: Box
    frames: tuple[int, ...]


def _reject_empty(label: str, value: object) -> str:
    """Reject a missing id. A whitespace-only string counts as empty."""
    if isinstance(value, str) and value.strip() != "":
        return value
    raise ValueError(f"{label} must be a non-empty id, got {value!r}")


def _reject_strip(label: str, value: str) -> None:
    """Refuse surrounding whitespace. Do not strip the id."""
    if value != value.strip():
        raise ValueError(f"{label} must not have leading or trailing whitespace, got {value!r}")


def _reject_nfkc(label: str, value: str) -> None:
    """Refuse an id that NFKC would change. Do not fold the id."""
    if unicodedata.normalize("NFKC", value) != value:
        raise ValueError(f"{label} must already be in NFKC, got {value!r}")


def _reject_format(label: str, value: str) -> None:
    """Refuse Unicode format characters, including the zero-width marks."""
    for character in value:
        if unicodedata.category(character) == "Cf":
            raise ValueError(f"{label} must not contain a format character, got {value!r}")


def _require_id(label: str, value: object) -> str:
    """Return the id unchanged, or raise. Never strip, fold, or change case."""
    text = _reject_empty(label, value)
    _reject_strip(label, text)
    _reject_nfkc(label, text)
    _reject_format(label, text)
    return text


def _reject_bare_str(camera_ids: object) -> None:
    """A str would iterate as characters. That is not a list of camera ids."""
    if isinstance(camera_ids, str):
        raise ValueError(
            f"camera_ids must be a collection of camera ids, not a str, got {camera_ids!r}"
        )


def _as_camera_collection(camera_ids: object) -> Collection[object]:
    if isinstance(camera_ids, Collection):
        return camera_ids
    raise ValueError(f"camera_ids must be a collection of camera ids, got {camera_ids!r}")


def _reject_empty_collection(camera_ids: Collection[object]) -> None:
    """Refuse ``[]``. ``None`` is the signal that means every camera."""
    if len(camera_ids) == 0:
        raise ValueError("camera_ids must contain at least one camera id")


def _require_camera_ids(camera_ids: object) -> tuple[str, ...] | None:
    """Validate camera ids. ``None`` means every camera for the run."""
    if camera_ids is None:
        return None
    _reject_bare_str(camera_ids)
    checked = _as_camera_collection(camera_ids)
    _reject_empty_collection(checked)
    unique: dict[str, None] = {}
    for camera_id in checked:
        unique.setdefault(_require_id("camera_id", camera_id), None)
    return tuple(unique)


def _require_det_class(value: object) -> DetClass:
    if not isinstance(value, DetClass):
        raise ValueError(f"det_class must be a DetClass, got {value!r}")
    return value


def _cameras_to_read(
    store: TrackStore,
    run_id: str,
    camera_ids: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Honour ``camera_ids``. ``None`` asks the store for this run only."""
    if camera_ids is None:
        return store.camera_ids(run_id)
    return camera_ids


def _read_camera(store: TrackStore, camera_id: str, run_id: str) -> list[TrackObservation]:
    """Read one camera. ``run_id`` selects stored rows. It is not an offset."""
    return store.read(camera_id, run_id)


def _with_class(
    observations: Sequence[TrackObservation],
    det_class: DetClass,
) -> tuple[TrackObservation, ...]:
    return tuple(item for item in observations if item.det_class == det_class)


def _group(
    observations: Sequence[TrackObservation],
) -> tuple[tuple[TrackObservation, ...], ...]:
    buckets: dict[tuple[str, int], list[TrackObservation]] = {}
    for observation in observations:
        key = (observation.camera_id, observation.track_id)
        buckets.setdefault(key, []).append(observation)
    return tuple(tuple(bucket) for bucket in buckets.values())


def _observation_order(observation: TrackObservation) -> tuple[float, int]:
    """Timestamp, then frame number. Stable sort keeps store row order."""
    return (observation.timestamp_seconds, observation.frame_number)


def _ordered_observations(
    observations: Sequence[TrackObservation],
) -> tuple[TrackObservation, ...]:
    return tuple(sorted(observations, key=_observation_order))


def _end_boxes(ordered: Sequence[TrackObservation]) -> tuple[Box, Box]:
    """First and last box. ``ordered`` is already in time, frame, row order."""
    return ordered[0].box, ordered[-1].box


def _frame_numbers(ordered: Sequence[TrackObservation]) -> tuple[int, ...]:
    """Frame numbers. ``ordered`` is already in time, frame, row order."""
    return tuple(item.frame_number for item in ordered)


def _observation_count(observations: Sequence[TrackObservation]) -> int:
    """Count stored rows, including repeated frame numbers."""
    return len(observations)


def _build_tracklet(run_id: str, observations: Sequence[TrackObservation]) -> Tracklet:
    ordered = _ordered_observations(observations)
    first_box, last_box = _end_boxes(ordered)
    return Tracklet(
        camera_id=ordered[0].camera_id,
        track_id=ordered[0].track_id,
        det_class=ordered[0].det_class,
        run_id=run_id,
        t_start=ordered[0].timestamp_seconds,
        t_end=ordered[-1].timestamp_seconds,
        n_obs=_observation_count(ordered),
        first_box=first_box,
        last_box=last_box,
        frames=_frame_numbers(ordered),
    )


def _tracklets_from_rows(
    run_id: str,
    observations: Sequence[TrackObservation],
    det_class: DetClass,
) -> list[Tracklet]:
    matched = _with_class(observations, det_class)
    return [_build_tracklet(run_id, group) for group in _group(matched)]


def _sort_key(tracklet: Tracklet) -> tuple[str, float, int]:
    """Tie order is camera id, then ``t_start``, then per-camera track id."""
    camera_id = tracklet.camera_id
    t_start = tracklet.t_start
    track_id = tracklet.track_id
    return (camera_id, t_start, track_id)


def _sort_tracklets(tracklets: list[Tracklet]) -> list[Tracklet]:
    return sorted(tracklets, key=_sort_key)


def load_tracklets(
    store: TrackStore,
    run_id: str,
    camera_ids: Collection[str] | None = None,
    det_class: DetClass = DetClass.PERSON,
) -> list[Tracklet]:
    """Load one tracklet per per-camera track, sorted by camera, start, track id.

    ``camera_ids=None`` reads every camera with a row for ``run_id``. An empty
    collection raises ``ValueError``. A bare ``str`` raises ``ValueError``.
    Unknown camera ids contribute no tracklets. Timestamps are the stored
    ones. ``det_class`` defaults to ``DetClass.PERSON``. Expired rows are
    purged once before any camera lookup or read.
    """
    checked_run = _require_id("run_id", run_id)
    checked_cameras = _require_camera_ids(camera_ids)
    checked_class = _require_det_class(det_class)
    store.purge_expired()
    selected = _cameras_to_read(store, checked_run, checked_cameras)
    built: list[Tracklet] = []
    for camera_id in selected:
        rows = _read_camera(store, camera_id, checked_run)
        built.extend(_tracklets_from_rows(checked_run, rows, checked_class))
    return _sort_tracklets(built)
