"""Attribute and example-crop search over anonymous per-camera tracks.

``SearchService`` is the in-process search API a later UI can call. It does
not open a network connection and it does not add a free-text embedder.

Inside the window means inclusive overlap, not containment. A track is
inside the window when t_end >= window start and t_start <= window end.
Touching either edge counts as inside. A track that covers the window
and extends past either end is inside. A track that ends before the
window start, or starts after the window end, is outside. The reported
``t_start`` and ``t_end`` are the track's own span, not the window.

Every call takes a ``run_id``. Track ids restart per run, so spans and
dedupe use ``(run_id, camera_id, track_id)``. Vector search and attribute
reads also take that run and the one ``derived_from`` stored for it.
``TrackStore.count`` is the public lookup for that origin. An unknown
run raises before a search and before an audit write.

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
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from trace_search.attributes.attributes import AttributeQuery, Box, Color, PersonAttributes
from trace_search.attributes.store import (
    TrackAttributeRecord,
    TrackAttributeStore,
    _require_attribute_filter,
)
from trace_search.audit.log import AuditAction, AuditLog
from trace_search.detection.types import DetClass
from trace_search.matching.tracklets import Tracklet, load_tracklets
from trace_search.reid.types import AppearanceEmbedder, AppearanceEmbedding
from trace_search.reid.vector_store import ALLOWED_DERIVED_FROM, VectorStore
from trace_search.tracking.store import TrackStore

# Stable probe order. ``ALLOWED_DERIVED_FROM`` is a set.
_ORIGINS = tuple(sorted(ALLOWED_DERIVED_FROM))

# Track ids restart every run, so a camera and track id are not enough.
_TrackKey = tuple[str, str, int]


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
    run_id: str


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
        *,
        run_id: str,
    ) -> list[TrackRef]:
        """Return matching tracks inside the window and camera set.

        ``run_id`` is required. The read uses that run and its stored
        ``derived_from``. ``query`` uses the attribute-store filter rules.
        An unknown colour or a non-bool bag raises before any store read
        and before any audit write. ``limit`` is a positive int. ``True``
        and ``False`` are rejected because ``bool`` is a subclass of ``int``.

        Results have ``score is None``. They sort by t_start ascending,
        then by camera_id and track_id. Inside the window means inclusive
        overlap, not containment. Reported times are the track span.
        """
        scope = _scope(operator, case_id, purpose, run_id, t_window, camera_ids, limit, "limit")
        checked = _require_attribute_filter(query)
        origin = _origin_for_run(self._tracks, scope.run_id)
        refs = _attribute_hits(
            self._tracks,
            self._attributes,
            checked,
            scope.run_id,
            origin,
            scope.window,
            scope.camera_ids,
        )
        return self._finish(
            scope,
            "limit",
            _attribute_query(checked),
            _limited(_ordered(refs), scope.limit),
        )

    def by_example_crop(
        self,
        operator: str,
        case_id: str,
        purpose: str,
        crop: npt.NDArray[np.uint8],
        t_window: tuple[float, float],
        camera_ids: Collection[str],
        k: int = 50,
        *,
        run_id: str,
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
        scope = _scope(operator, case_id, purpose, run_id, t_window, camera_ids, k, "k")
        image = _require_uint8_image("crop", crop)
        origin = _origin_for_run(self._tracks, scope.run_id)
        refs = _vector_hits(
            self._tracks,
            self._vectors,
            self._embedder,
            image,
            scope.run_id,
            origin,
            scope.window,
            scope.camera_ids,
        )
        return self._finish(scope, "k", _crop_query(image), _limited(_ordered(refs), scope.limit))

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
        *,
        run_id: str,
    ) -> list[TrackRef]:
        """Click-to-search: cut a crop from ``frame`` and search by that crop.

        The box comes from the stored observation for ``camera_id``,
        ``track_id``, and ``frame_number``. The crop is sliced in memory.
        Neither the frame nor the crop is written. The audit entry records
        the source ids and the crop shape and dtype, not the pixels.
        Scores are not evidence: placeholder similarity has no accuracy
        meaning.
        """
        scope = _scope(operator, case_id, purpose, run_id, t_window, camera_ids, k, "k")
        source = _require_nonempty_str("camera_id", camera_id)
        track = _require_index("track_id", track_id)
        frame_index = _require_index("frame_number", frame_number)
        image = _require_uint8_image("frame", frame)
        origin = _origin_for_run(self._tracks, scope.run_id)
        box = _box_for(self._tracks, source, track, frame_index, scope.run_id)
        crop = _require_uint8_image("crop", _slice_crop(image, box))
        refs = _vector_hits(
            self._tracks,
            self._vectors,
            self._embedder,
            crop,
            scope.run_id,
            origin,
            scope.window,
            scope.camera_ids,
        )
        payload = _track_query(source, track, frame_index, crop)
        return self._finish(scope, "k", payload, _limited(_ordered(refs), scope.limit))

    def _finish(
        self,
        scope: _Scope,
        limit_key: str,
        query_payload: Mapping[str, object],
        refs: Sequence[TrackRef],
    ) -> list[TrackRef]:
        """Append one audit entry, then return ``refs``.

        ``append`` runs before the return. If it raises, this raises and
        the caller gets no list. The query records the validated scope.
        """
        returned = list(refs)
        self._audit.append(
            operator_username=scope.operator,
            action=AuditAction.SEARCH,
            query=_canonical_json(_with_scope(scope, limit_key, query_payload)),
            case_id=scope.case_id,
            purpose=scope.purpose,
            result_ids=[_result_id(item) for item in returned],
        )
        return returned


def _scope(
    operator: object,
    case_id: object,
    purpose: object,
    run_id: object,
    t_window: object,
    camera_ids: object,
    limit: object,
    limit_label: str,
) -> _Scope:
    """Check operator, case, purpose, run, window, cameras, and limit."""
    return _Scope(
        operator=_require_nonempty_str("operator", operator),
        case_id=_require_nonempty_str("case_id", case_id),
        purpose=_require_nonempty_str("purpose", purpose),
        run_id=_require_nonempty_str("run_id", run_id),
        window=_require_window(t_window),
        camera_ids=_require_camera_ids(camera_ids),
        limit=_require_positive_int(limit_label, limit),
    )


def _require_nonempty_str(label: str, value: object) -> str:
    """Reject a missing, blank, or non-str value. Return it unchanged."""
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
    """A finite ``int`` or ``float``. ``bool`` is excluded. Do not clamp.

    NumPy numbers such as ``np.float32`` are not accepted. Callers pass
    Python ``int`` and ``float`` only.
    """
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
    return tuple(sorted(unique))


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


def _origin_for_run(store: TrackStore, run_id: str) -> str:
    """The one ``derived_from`` stored on ``run_id``.

    ``TrackStore`` has no method that returns the origin. ``count`` is the
    public read: exactly one of ``SMOKE``, ``CODE_TEST``, and ``SITE`` must
    have rows. Zero rows is an unknown run. Two origins is mixed and also
    refused. This runs after the text checks and before a search.
    """
    found = [origin for origin in _ORIGINS if store.count(run_id=run_id, derived_from=origin) > 0]
    if len(found) != 1:
        raise ValueError(f"run_id {run_id!r} is unknown or has more than one derived_from")
    return found[0]


def _tracklets_for_run(store: TrackStore, run_id: str) -> list[Tracklet]:
    """Every det class for one run. ``camera_ids=None`` reads every camera."""
    loaded: list[Tracklet] = []
    for det_class in DetClass:
        loaded.extend(load_tracklets(store, run_id, None, det_class))
    return loaded


def _load_tracklets(store: TrackStore, run_id: str) -> tuple[Tracklet, ...]:
    """Tracklets for ``run_id`` only. The search filters cameras and time."""
    return tuple(_tracklets_for_run(store, run_id))


def _attribute_ref(
    record: TrackAttributeRecord,
    spans: Mapping[_TrackKey, tuple[tuple[float, float], ...]],
    run_id: str,
    camera_ids: Collection[str],
    window: tuple[float, float],
) -> TrackRef | None:
    """One attribute row as a ``TrackRef``, or ``None`` when it is out of scope.

    The span key is the run passed to ``read``. Checking ``record.run_id``
    again would hide a read that dropped that argument.
    """
    if not _camera_allowed(record.camera_id, camera_ids):
        return None
    span = _span_for(spans, (run_id, record.camera_id, record.track_id), window)
    if span is None:
        return None
    return TrackRef(
        camera_id=record.camera_id,
        track_id=record.track_id,
        t_start=span[0],
        t_end=span[1],
        score=None,
    )


def _attributes_match(record: TrackAttributeRecord, query: Mapping[str, object]) -> bool:
    """Apply an already-validated filter. Do not re-check its types."""
    attrs = PersonAttributes(
        upper_clothing_color=Color(record.upper_color),
        lower_clothing_color=Color(record.lower_color),
        bag_on_track=record.bag_on_track,
        upper_confidence=record.upper_confidence,
        lower_confidence=record.lower_confidence,
    )
    return AttributeQuery.matches(attrs, query)


def _attribute_rows(
    attributes: TrackAttributeStore,
    tracklets: Sequence[Tracklet],
    run_id: str,
    origin: str,
) -> list[TrackAttributeRecord]:
    """Rows for ``run_id`` and ``origin``. Other runs and origins stay out.

    ``read`` takes the run. It has no origin argument, so ``derived_from``
    is applied to the rows it returns. Cameras come from this run's
    tracklets, including cameras the caller did not ask for, so the camera
    check below is the one that drops them.
    """
    cameras: dict[str, None] = {}
    for tracklet in tracklets:
        cameras.setdefault(tracklet.camera_id, None)
    rows: list[TrackAttributeRecord] = []
    for camera_id in cameras:
        for record in attributes.read(camera_id, None, run_id):
            if record.derived_from != origin:
                continue
            rows.append(record)
    return rows


def _attribute_hits(
    tracks: TrackStore,
    attributes: TrackAttributeStore,
    query: Mapping[str, object],
    run_id: str,
    origin: str,
    window: tuple[float, float],
    camera_ids: Collection[str],
) -> list[TrackRef]:
    """Attribute matches for one run, filtered here for camera and overlap.

    Duplicate rows for one ``(run_id, camera_id, track_id)`` become one ref.
    The reported times are the union of overlapping spans, not the window.
    """
    tracklets = _load_tracklets(tracks, run_id)
    spans = _spans(tracklets)
    chosen: dict[_TrackKey, TrackRef] = {}
    for record in _attribute_rows(attributes, tracklets, run_id, origin):
        if not _attributes_match(record, query):
            continue
        ref = _attribute_ref(record, spans, run_id, camera_ids, window)
        if ref is None:
            continue
        key = (run_id, ref.camera_id, ref.track_id)
        if key in chosen:
            continue
        chosen[key] = ref
    return list(chosen.values())


def _spans(
    tracklets: Sequence[Tracklet],
) -> dict[_TrackKey, tuple[tuple[float, float], ...]]:
    """Time spans keyed by ``(run_id, camera_id, track_id)``, including out of window."""
    grouped: dict[_TrackKey, list[tuple[float, float]]] = {}
    for tracklet in tracklets:
        grouped.setdefault((tracklet.run_id, tracklet.camera_id, tracklet.track_id), []).append(
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


def _span_for(
    spans: Mapping[_TrackKey, tuple[tuple[float, float], ...]],
    key: _TrackKey,
    window: tuple[float, float],
) -> tuple[float, float] | None:
    """Union of ranges for ``key`` that overlap ``window``, or ``None``."""
    ranges = spans.get(key)
    if ranges is None:
        return None
    return _matching_span(ranges, window)


def _allowed_span(
    embedding: AppearanceEmbedding,
    spans: Mapping[_TrackKey, tuple[tuple[float, float], ...]],
    run_id: str,
    camera_ids: Collection[str],
    window: tuple[float, float],
) -> tuple[float, float] | None:
    """Time span when the embedding's track is known, in the cameras, and inside.

    The embedding has no run id. ``run_id`` is the caller's run, which is
    also the run ``search`` was given. A span is not clipped to the window.
    """
    if not _camera_allowed(embedding.camera_id, camera_ids):
        return None
    return _span_for(spans, (run_id, embedding.camera_id, embedding.track_id), window)


def _keep_best(
    best: dict[_TrackKey, float],
    times: dict[_TrackKey, tuple[float, float]],
    run_id: str,
    camera_id: str,
    track_id: int,
    span: tuple[float, float],
    score: float,
) -> None:
    """Keep the higher score for one track. A lower or equal score stays put."""
    key = (run_id, camera_id, track_id)
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
    run_id: str,
    origin: str,
) -> list[tuple[AppearanceEmbedding, float]]:
    """Every vector in this run and origin, before camera and window scoping.

    ``k`` is the scoped count so a stray top hit cannot consume the caller's
    ``k`` and hide a real track. An empty scope does not call ``search``.
    """
    total = vectors.count(derived_from=origin, run_id=run_id)
    if total <= 0:
        return []
    return vectors.search(row, total, run_id=run_id, derived_from=origin)


def _vector_hits(
    tracks: TrackStore,
    vectors: VectorStore,
    embedder: AppearanceEmbedder,
    crop: npt.NDArray[np.uint8],
    run_id: str,
    origin: str,
    window: tuple[float, float],
    camera_ids: Collection[str],
) -> list[TrackRef]:
    """Rank known in-window tracks in one run. Stray vectors never become refs."""
    row = _embed_crop(embedder, crop)
    spans = _spans(_load_tracklets(tracks, run_id))
    best: dict[_TrackKey, float] = {}
    times: dict[_TrackKey, tuple[float, float]] = {}
    for embedding, score in _all_hits(vectors, row, run_id, origin):
        span = _allowed_span(embedding, spans, run_id, camera_ids, window)
        if span is None:
            continue
        _keep_best(
            best,
            times,
            run_id,
            embedding.camera_id,
            embedding.track_id,
            span,
            float(score),
        )
    return _refs_from_best(best, times)


def _refs_from_best(
    best: Mapping[_TrackKey, float],
    times: Mapping[_TrackKey, tuple[float, float]],
) -> list[TrackRef]:
    """Build refs from the best score kept for each track."""
    refs: list[TrackRef] = []
    for key, score in best.items():
        t_start, t_end = times[key]
        refs.append(TrackRef(key[1], key[2], t_start, t_end, score))
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
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _with_scope(
    scope: _Scope,
    limit_key: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validated scope plus the method payload. Later keys do not replace scope."""
    merged: dict[str, object] = {
        "run_id": scope.run_id,
        "t_window": [scope.window[0], scope.window[1]],
        "camera_ids": list(scope.camera_ids),
        limit_key: scope.limit,
    }
    for key, value in payload.items():
        if key in merged:
            continue
        merged[key] = value
    return merged


def _attribute_query(checked: Mapping[str, object]) -> dict[str, object]:
    """Validated attribute filter plus ``kind``. Numbers are already floats."""
    payload: dict[str, object] = {"kind": "by_attributes"}
    payload.update(checked)
    return payload


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


def _box_for(
    store: TrackStore,
    camera_id: str,
    track_id: int,
    frame_number: int,
    run_id: str,
) -> Box:
    """The stored box for one observation in ``run_id``, or raise when it is absent."""
    for observation in store.read(camera_id, run_id):
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
