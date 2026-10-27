"""Attribute and example-crop search over anonymous per-camera tracks.

``SearchService`` is the in-process search API a later UI can call. It does
not open a network connection and it does not add a free-text embedder.

Inside the window means inclusive overlap, not containment. A track is
inside the window when t_end >= window start and t_start <= window end.
Touching either edge counts as inside. A track that covers the window
and extends past either end is inside. A track that ends before the
window start, or starts after the window end, is outside.

Results are sorted by score descending, then by t_start ascending, then
by camera_id ascending, then by track_id ascending. That last pair makes
the order deterministic when scores and start times tie. Attribute hits
use ``score is None``, so they sort by time and then by those ids.

Scores are not evidence: placeholder similarity has no accuracy meaning.
Crops are cut and embedded in memory and are never written. An audit
entry describes a crop by shape and dtype only.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from trace_search.attributes.attributes import Box
from trace_search.attributes.store import (
    TrackAttributeRecord,
    TrackAttributeStore,
    _require_attribute_filter,
)
from trace_search.audit.log import AuditAction, AuditLog
from trace_search.detection.types import DetClass
from trace_search.matching.tracklets import Tracklet, load_tracklets
from trace_search.reid.types import AppearanceEmbedder, AppearanceEmbedding
from trace_search.reid.vector_store import VectorStore
from trace_search.tracking.store import TrackStore

# ``TrackStore.read`` does not return ``run_id``, and the store has no
# list-runs method. ``load_tracklets`` needs the id. This read is the
# smallest way to discover runs without editing the track store.
_RUN_ID_SQL = "SELECT DISTINCT run_id FROM track_observations ORDER BY run_id"

_TrackKey = tuple[str, int]


@dataclass(frozen=True)
class TrackRef:
    """One anonymous per-camera track returned by a search.

    ``track_id`` is unique inside ``camera_id`` only. ``t_start`` and
    ``t_end`` come from the stored observations.     ``score`` is cosine
    similarity for a crop search, or ``None`` for an attribute search.
    Scores are not evidence: placeholder similarity has no accuracy meaning.

    Sort order, most significant key first: score descending, t_start
    ascending, camera_id ascending, track_id ascending.
    """

    camera_id: str
    track_id: int
    t_start: float
    t_end: float
    score: float | None


@dataclass(frozen=True)
class _Scope:
    """Inputs that every search method checks before it reads a store."""

    operator: str
    case_id: str
    purpose: str
    window: tuple[float, float]
    camera_ids: tuple[str, ...]
    limit: int


class SearchService:
    """Search tracks by attributes or by an example crop.

    Constructor arguments are the stores already on main: ``TrackStore``,
    ``TrackAttributeStore``, a ``VectorStore``, an ``AppearanceEmbedder``,
        and an ``AuditLog``. A successful call appends one ``search`` audit
        entry after the result list is known. If that append raises, the call
        raises and the caller receives no results.

        Inside the window means inclusive overlap, not containment. See the
        module docstring for the edge rule. Scores are not evidence:
        placeholder similarity has no accuracy meaning.
    """

    def __init__(
        self,
        track_store: TrackStore,
        attr_store: TrackAttributeStore,
        vectors: VectorStore,
        embedder: AppearanceEmbedder,
        audit: AuditLog,
    ) -> None:
        self._tracks = track_store
        self._attributes = attr_store
        self._vectors = vectors
        self._embedder = embedder
        self._audit = audit

    def by_attributes(
        self,
        operator: str,
        case_id: str,
        purpose: str,
        query: Mapping[str, object],
        t_window: tuple[float, float],
        camera_ids: Collection[str],
        limit: int = 50,
    ) -> list[TrackRef]:
        """Return matching tracks inside the window and camera set.

        ``query`` uses the attribute-store filter rules. An unknown colour
        or a non-bool bag raises before any store read and before any audit
        write. ``limit`` is a positive int. ``True`` and ``False`` are
        rejected because ``bool`` is a subclass of ``int``.

        Results have ``score is None``. They sort by t_start ascending,
        then by camera_id and track_id. Inside the window means inclusive
        overlap, not containment.
        """
        scope = _scope(operator, case_id, purpose, t_window, camera_ids, limit, "limit")
        checked = _require_attribute_filter(query)
        refs = _attribute_hits(
            self._tracks,
            self._attributes,
            checked,
            scope.window,
            scope.camera_ids,
        )
        return self._finish(scope, checked, _limited(_ordered(refs), scope.limit))

    def by_example_crop(
        self,
        operator: str,
        case_id: str,
        purpose: str,
        crop: npt.NDArray[np.uint8],
        t_window: tuple[float, float],
        camera_ids: Collection[str],
        k: int = 50,
    ) -> list[TrackRef]:
        """Return the nearest tracks to an in-memory crop.

        The crop must be a 3-dimensional ``uint8`` array with a non-zero
        size. It is not written, and the audit entry stores its shape and
        dtype only. Ranking uses cosine similarity against vectors whose
        track is in ``track_store``, overlaps the window, and sits in
        ``camera_ids``. Scores are not evidence: placeholder similarity has
        no accuracy meaning.

        Results are sorted by score descending, then by t_start ascending,
        then by camera_id ascending, then by track_id ascending.
        """
        scope = _scope(operator, case_id, purpose, t_window, camera_ids, k, "k")
        image = _require_uint8_image("crop", crop)
        refs = _vector_hits(
            self._tracks,
            self._vectors,
            self._embedder,
            image,
            scope.window,
            scope.camera_ids,
        )
        return self._finish(scope, _crop_query(image), _limited(_ordered(refs), scope.limit))

    def by_track(
        self,
        operator: str,
        case_id: str,
        purpose: str,
        camera_id: str,
        track_id: int,
        frame_number: int,
        frame: npt.NDArray[np.uint8],
        t_window: tuple[float, float],
        camera_ids: Collection[str],
        k: int = 50,
    ) -> list[TrackRef]:
        """Click-to-search: cut a crop from ``frame`` and search by that crop.

        The box comes from the stored observation for ``camera_id``,
        ``track_id``, and ``frame_number``. The crop is sliced in memory.
        Neither the frame nor the crop is written. The audit entry records
        the source ids and the crop shape and dtype, not the pixels.
        Scores are not evidence: placeholder similarity has no accuracy
        meaning.
        """
        scope = _scope(operator, case_id, purpose, t_window, camera_ids, k, "k")
        source = _require_nonempty_str("camera_id", camera_id)
        track = _require_index("track_id", track_id)
        frame_index = _require_index("frame_number", frame_number)
        image = _require_uint8_image("frame", frame)
        box = _box_for(self._tracks, source, track, frame_index)
        crop = _require_uint8_image("crop", _slice_crop(image, box))
        refs = _vector_hits(
            self._tracks,
            self._vectors,
            self._embedder,
            crop,
            scope.window,
            scope.camera_ids,
        )
        payload = _track_query(source, track, frame_index, crop)
        return self._finish(scope, payload, _limited(_ordered(refs), scope.limit))

    def _finish(
        self,
        scope: _Scope,
        query_payload: Mapping[str, object],
        refs: Sequence[TrackRef],
    ) -> list[TrackRef]:
        """Append one audit entry, then return ``refs``.

        ``append`` runs before the return. If it raises, this raises and
        the caller gets no list.
        """
        returned = list(refs)
        self._audit.append(
            operator_username=scope.operator,
            action=AuditAction.SEARCH,
            query=_canonical_json(query_payload),
            case_id=scope.case_id,
            purpose=scope.purpose,
            result_ids=[_result_id(item) for item in returned],
        )
        return returned


def _scope(
    operator: object,
    case_id: object,
    purpose: object,
    t_window: object,
    camera_ids: object,
    limit: object,
    limit_label: str,
) -> _Scope:
    """Check operator, case, purpose, window, cameras, and limit."""
    return _Scope(
        operator=_require_nonempty_str("operator", operator),
        case_id=_require_nonempty_str("case_id", case_id),
        purpose=_require_nonempty_str("purpose", purpose),
        window=_require_window(t_window),
        camera_ids=_require_camera_ids(camera_ids),
        limit=_require_positive_int(limit_label, limit),
    )


def _require_nonempty_str(label: str, value: object) -> str:
    """Reject a missing, blank, or non-str value. Do not strip a real id."""
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    return value


def _require_positive_int(label: str, value: object) -> int:
    """A positive int. ``bool`` is excluded because it is an ``int``."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive int, got {value!r}")
    return value


def _require_index(label: str, value: object) -> int:
    """An int >= 0. ``bool`` is excluded. Zero is a valid frame and track."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be an int >= 0, got {value!r}")
    return value


def _require_epoch(value: object) -> float:
    """A finite int or float. ``bool`` is excluded. Do not clamp."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"t_window endpoints must be real numbers, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"t_window endpoints must be finite, got {value!r}")
    return number


def _require_pair(value: object) -> tuple[object, object]:
    """A length-2 sequence. A bare str is not a pair of endpoints."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"t_window must be a pair of epochs, got {value!r}")
    if len(value) != 2:
        raise ValueError(f"t_window must be a pair of epochs, got {value!r}")
    return (value[0], value[1])


def _require_window(value: object) -> tuple[float, float]:
    """Inclusive ``(start, end)``. Reversed or non-finite endpoints raise."""
    pair = _require_pair(value)
    start = _require_epoch(pair[0])
    end = _require_epoch(pair[1])
    if start > end:
        raise ValueError(f"t_window start must be <= end, got {value!r}")
    return (start, end)


def _require_camera_id(value: object) -> str:
    """One camera id. Non-str and blank values raise."""
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"camera_id must be a non-empty string, got {value!r}")
    return value


def _require_camera_ids(value: object) -> tuple[str, ...]:
    """A non-empty collection of camera ids. A bare str raises."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
        raise ValueError(f"camera_ids must be a collection of camera ids, got {value!r}")
    if len(value) == 0:
        raise ValueError("camera_ids must contain at least one camera id")
    unique: dict[str, None] = {}
    for item in value:
        unique.setdefault(_require_camera_id(item), None)
    return tuple(unique)


def _require_uint8_image(label: str, value: object) -> npt.NDArray[np.uint8]:
    """A 3-dimensional uint8 array with a non-zero size."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a numpy array, got {type(value).__name__}")
    if value.dtype != np.uint8:
        raise TypeError(f"{label} must be uint8, got {value.dtype}")
    if value.ndim != 3:
        raise ValueError(f"{label} must be 3-dimensional, got ndim {value.ndim}")
    if int(value.size) == 0:
        raise ValueError(f"{label} must have a non-zero size")
    return cast(npt.NDArray[np.uint8], value)


def _overlaps(t_start: float, t_end: float, window: tuple[float, float]) -> bool:
    """Inclusive overlap, not containment.

    Equal to either edge is inside. ``t_end >= start`` keeps a track that
    ends on the window start. ``t_start <= end`` keeps a track that starts
    on the window end.
    """
    start, end = window
    return t_end >= start and t_start <= end


def _camera_allowed(camera_id: str, camera_ids: Collection[str]) -> bool:
    """True when ``camera_id`` is one of the requested cameras."""
    return camera_id in camera_ids


def _stored_run_ids(store: TrackStore) -> tuple[str, ...]:
    """Run ids present in the track database, in ascending order."""
    with sqlite3.connect(store.db_path) as connection:
        cursor = connection.execute(_RUN_ID_SQL)
        found = [str(row[0]) for row in cursor]
    return tuple(found)


def _tracklets_for_run(store: TrackStore, run_id: str) -> list[Tracklet]:
    """Every det class for one run. ``camera_ids=None`` reads every camera."""
    loaded: list[Tracklet] = []
    for det_class in DetClass:
        loaded.extend(load_tracklets(store, run_id, None, det_class))
    return loaded


def _load_tracklets(store: TrackStore) -> tuple[Tracklet, ...]:
    """All tracklets known to ``store``. The search filters them afterwards."""
    loaded: list[Tracklet] = []
    for run_id in _stored_run_ids(store):
        loaded.extend(_tracklets_for_run(store, run_id))
    return tuple(loaded)


def _index_tracklets(tracklets: Sequence[Tracklet]) -> dict[tuple[str, int, str], Tracklet]:
    """Map ``(camera_id, track_id, run_id)`` to the tracklet that carries time."""
    return {(item.camera_id, item.track_id, item.run_id): item for item in tracklets}


def _attribute_ref(
    record: TrackAttributeRecord,
    indexed: Mapping[tuple[str, int, str], Tracklet],
    camera_ids: Collection[str],
    window: tuple[float, float],
) -> TrackRef | None:
    """One attribute row as a ``TrackRef``, or ``None`` when it is out of scope."""
    if not _camera_allowed(record.camera_id, camera_ids):
        return None
    tracklet = indexed[(record.camera_id, record.track_id, record.run_id)]
    if not _overlaps(tracklet.t_start, tracklet.t_end, window):
        return None
    return TrackRef(
        camera_id=tracklet.camera_id,
        track_id=tracklet.track_id,
        t_start=tracklet.t_start,
        t_end=tracklet.t_end,
        score=None,
    )


def _attribute_hits(
    tracks: TrackStore,
    attributes: TrackAttributeStore,
    query: Mapping[str, object],
    window: tuple[float, float],
    camera_ids: Collection[str],
) -> list[TrackRef]:
    """Attribute matches, filtered here for camera and inclusive overlap.

    The store is asked with ``camera_ids=None`` and ``t_window=None`` so
    this module's filters are the ones that decide membership. Duplicate
    rows for one ``(camera_id, track_id)`` become one ref.
    """
    tracklets = _load_tracklets(tracks)
    records = attributes.query(query, None, None, tracklets)
    indexed = _index_tracklets(tracklets)
    chosen: dict[_TrackKey, TrackRef] = {}
    for record in records:
        ref = _attribute_ref(record, indexed, camera_ids, window)
        if ref is None:
            continue
        key = (ref.camera_id, ref.track_id)
        if key in chosen:
            continue
        chosen[key] = ref
    return list(chosen.values())


def _spans(
    tracklets: Sequence[Tracklet],
) -> dict[_TrackKey, tuple[tuple[float, float], ...]]:
    """Time spans for every known ``(camera_id, track_id)``, including out of window."""
    grouped: dict[_TrackKey, list[tuple[float, float]]] = {}
    for tracklet in tracklets:
        grouped.setdefault((tracklet.camera_id, tracklet.track_id), []).append(
            (tracklet.t_start, tracklet.t_end)
        )
    return {key: tuple(ranges) for key, ranges in grouped.items()}


def _matching_span(
    ranges: Sequence[tuple[float, float]],
    window: tuple[float, float],
) -> tuple[float, float] | None:
    """The span of ranges that overlap ``window``, or ``None``."""
    matched = [item for item in ranges if _overlaps(item[0], item[1], window)]
    if not matched:
        return None
    start = min(item[0] for item in matched)
    end = max(item[1] for item in matched)
    return (start, end)


def _allowed_span(
    embedding: AppearanceEmbedding,
    spans: Mapping[_TrackKey, tuple[tuple[float, float], ...]],
    camera_ids: Collection[str],
    window: tuple[float, float],
) -> tuple[float, float] | None:
    """Time span when the embedding's track is known, in the cameras, and inside."""
    if not _camera_allowed(embedding.camera_id, camera_ids):
        return None
    ranges = spans.get((embedding.camera_id, embedding.track_id))
    if ranges is None:
        return None
    return _matching_span(ranges, window)


def _keep_best(
    best: dict[_TrackKey, float],
    times: dict[_TrackKey, tuple[float, float]],
    camera_id: str,
    track_id: int,
    span: tuple[float, float],
    score: float,
) -> None:
    """Keep the higher score for one track. A lower or equal score stays put."""
    key = (camera_id, track_id)
    previous = best.get(key)
    if previous is not None and score <= previous:
        return
    best[key] = score
    times[key] = span


def _embed_crop(
    embedder: AppearanceEmbedder,
    crop: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float32]:
    """One float32 row from ``embedder``. The crop is not retained."""
    embedded = np.asarray(embedder.embed((crop,)), dtype=np.float32)
    if embedded.ndim != 2 or int(embedded.shape[0]) != 1:
        raise ValueError("embed must return one row for one crop")
    row = np.array(embedded[0], dtype=np.float32, copy=True)
    return row


def _all_hits(
    vectors: VectorStore,
    row: npt.NDArray[np.float32],
) -> list[tuple[AppearanceEmbedding, float]]:
    """Every stored vector, before camera, track, and window scoping.

    ``k`` is the full count so a stray top hit cannot consume the caller's
    ``k`` and hide a real track. An empty store does not call ``search``.
    """
    total = vectors.count()
    if total <= 0:
        return []
    return vectors.search(row, total)


def _vector_hits(
    tracks: TrackStore,
    vectors: VectorStore,
    embedder: AppearanceEmbedder,
    crop: npt.NDArray[np.uint8],
    window: tuple[float, float],
    camera_ids: Collection[str],
) -> list[TrackRef]:
    """Rank known in-window tracks. Stray vectors never become refs."""
    row = _embed_crop(embedder, crop)
    spans = _spans(_load_tracklets(tracks))
    best: dict[_TrackKey, float] = {}
    times: dict[_TrackKey, tuple[float, float]] = {}
    for embedding, score in _all_hits(vectors, row):
        span = _allowed_span(embedding, spans, camera_ids, window)
        if span is None:
            continue
        _keep_best(best, times, embedding.camera_id, embedding.track_id, span, float(score))
    return _refs_from_best(best, times)


def _refs_from_best(
    best: Mapping[_TrackKey, float],
    times: Mapping[_TrackKey, tuple[float, float]],
) -> list[TrackRef]:
    """Build refs from the best score kept for each track."""
    refs: list[TrackRef] = []
    for key, score in best.items():
        t_start, t_end = times[key]
        refs.append(TrackRef(key[0], key[1], t_start, t_end, score))
    return refs


def _score_key(ref: TrackRef) -> float:
    """Sort key for score. ``None`` ties with other attribute hits at 0."""
    if ref.score is None:
        return 0.0
    return float(ref.score)


def _time_key(ref: TrackRef) -> float:
    """Sort key for time. Time is ``t_start``."""
    return ref.t_start


def _camera_key(ref: TrackRef) -> str:
    """Sort key for ``camera_id``."""
    return ref.camera_id


def _track_key(ref: TrackRef) -> int:
    """Sort key for ``track_id``."""
    return ref.track_id


def _ordered(refs: Sequence[TrackRef]) -> list[TrackRef]:
    """Score descending, then t_start ascending, then camera_id, then track_id.

    Each call is a stable sort. The last sort is the most significant key,
    so score decides first and the earlier sorts break ties.
    """
    by_track = sorted(refs, key=_track_key)
    by_camera = sorted(by_track, key=_camera_key)
    by_time = sorted(by_camera, key=_time_key)
    return sorted(by_time, key=_score_key, reverse=True)


def _limited(refs: Sequence[TrackRef], limit: int) -> list[TrackRef]:
    """The first ``limit`` refs. ``limit`` is already a positive int."""
    return list(refs[:limit])


def _canonical_json(payload: Mapping[str, object]) -> str:
    """Canonical JSON: sorted keys, compact separators, no extra whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _crop_query(crop: npt.NDArray[np.uint8]) -> dict[str, object]:
    """Non-content description of a crop. Pixels are not included."""
    payload: dict[str, object] = {
        "kind": "example_crop",
        "crop_dtype": "uint8",
        "crop_shape": cast(object, [int(size) for size in crop.shape]),
    }
    return payload


def _track_query(
    camera_id: str,
    track_id: int,
    frame_number: int,
    crop: npt.NDArray[np.uint8],
) -> dict[str, object]:
    """Non-content description of a click-to-search crop."""
    payload: dict[str, object] = {
        "kind": "by_track",
        "camera_id": camera_id,
        "track_id": track_id,
        "frame_number": frame_number,
        "crop_dtype": "uint8",
        "crop_shape": cast(object, [int(size) for size in crop.shape]),
    }
    return payload


def _result_id(ref: TrackRef) -> str:
    """Audit id for one ref, in the same order as the returned list."""
    return f"{ref.camera_id}:{ref.track_id}"


def _box_for(store: TrackStore, camera_id: str, track_id: int, frame_number: int) -> Box:
    """The stored box for one observation, or raise when it is absent."""
    for observation in store.read(camera_id):
        if observation.track_id != track_id:
            continue
        if observation.frame_number != frame_number:
            continue
        return observation.box
    raise ValueError(
        f"no stored observation for camera {camera_id!r} track {track_id} frame {frame_number}"
    )


def _slice_crop(frame: npt.NDArray[np.uint8], box: Box) -> npt.NDArray[np.uint8]:
    """Copy the pixels inside ``box``. The frame and the copy stay in memory."""
    height = int(frame.shape[0])
    width = int(frame.shape[1])
    x1, y1, x2, y2 = box
    left = max(0, math.floor(float(x1)))
    top = max(0, math.floor(float(y1)))
    right = min(width, math.ceil(float(x2)))
    bottom = min(height, math.ceil(float(y2)))
    if left >= right or top >= bottom:
        raise ValueError("track box does not overlap the frame")
    crop = np.array(frame[top:bottom, left:right], dtype=np.uint8, copy=True)
    return crop
