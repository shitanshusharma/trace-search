"""Attributes and one placeholder appearance vector per person track.

The source is read once, in order. For each person track, up to
``samples_per_track`` frames are taken at even gaps along that track's
frame numbers, including both ends when more than one frame is taken.
A single frame uses the middle index (``count // 2``). Duplicate frame
numbers on a tracklet are collapsed, keeping the first occurrence.

Each chosen frame is passed to ``AttributeExtractor.extract_frame`` with
every person box and every bag box on that frame. Only the sampled
tracks are kept. The person crop is passed to ``AppearanceEmbedder.embed``
and then dropped. Crops and frames stay inside that read and are never
written.

Per person track, the stored upper and lower colours are the majority
vote of the sampled frames. A tie stores ``Color.UNKNOWN``.
``bag_on_track`` is true when any sampled frame has a bag box assigned
to that person. One L2-normalised mean of the sampled crop vectors is
stored as an ``AppearanceEmbedding`` with ``placeholder`` true. The
vector's ``frame_number`` is the lowest sampled frame on that track.
Bag tracks get no vector. A person track whose boxes are all out of
frame still gets an attribute row and does not get a vector. This pass
does not return a similarity score.

A source that yields no frames raises before any row is deleted.
A source that yields frames but never yields a planned sample frame
raises before any row is deleted. An already-open source positioned
past every planned sample leaves the camera's existing rows in place.
Re-run policy: replace this camera before re-adding, not an in-store
dedupe and not a purge of the whole run. ``VectorStore`` allows
duplicate ``(camera_id, track_id, frame_number)`` keys. Before the new
rows are written, this function deletes attribute rows and appearance
vectors whose ``run_id`` and ``camera_id`` are this call's, then adds
the new batch. Rows for another camera in the same run are not read
and not deleted. A second call for the same camera leaves one vector
per key for that camera. The delete and the writes that follow it are
not atomic: a failure between them can leave this camera partly
replaced.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import numpy.typing as npt

from trace_search.attributes.attributes import AttributeExtractor, Box, Color, PersonAttributes
from trace_search.attributes.store import TrackAttributeStore
from trace_search.detection.types import DetClass
from trace_search.ingest.video_source import Frame, VideoSource
from trace_search.matching.tracklets import Tracklet
from trace_search.reid.types import AppearanceEmbedder, AppearanceEmbedding
from trace_search.reid.vector_store import ALLOWED_DERIVED_FROM, VectorStore
from trace_search.tracking.store import TrackStore
from trace_search.tracking.types import TrackObservation

__all__ = ["AppearanceStats", "describe_tracks"]

_PERSON: frozenset[DetClass] = frozenset((DetClass.PERSON,))
_BAGS: frozenset[DetClass] = frozenset((DetClass.BACKPACK, DetClass.HANDBAG))


@dataclass(frozen=True)
class AppearanceStats:
    """Counts for one appearance pass.

    ``frames_read`` is how many frames the source yielded during the single
    read. ``tracks_described`` is how many person tracks were written to the
    attribute store. ``embeddings_stored`` is how many appearance vectors
    were added. There is no similarity score.
    """

    frames_read: int
    tracks_described: int
    embeddings_stored: int


@dataclass(frozen=True)
class _Sample:
    track_id: int
    frame_number: int
    box: Box


@dataclass
class _TrackParts:
    frame_number: int
    upper: list[tuple[Color, float]] = field(default_factory=list)
    lower: list[tuple[Color, float]] = field(default_factory=list)
    bag: bool = False
    vectors: list[npt.NDArray[np.float64]] = field(default_factory=list)


@dataclass(frozen=True)
class _StoredTrack:
    track_id: int
    upper: str
    lower: str
    bag: bool
    upper_confidence: float
    lower_confidence: float
    n_frames: int


@dataclass(frozen=True)
class _Pending:
    attributes: tuple[_StoredTrack, ...]
    embeddings: tuple[AppearanceEmbedding, ...]


def describe_tracks(
    source: VideoSource,
    tracklets: Sequence[Tracklet],
    bag_tracklets: Sequence[Tracklet],
    store: TrackStore,
    run_id: str,
    extractor: AttributeExtractor,
    embedder: AppearanceEmbedder,
    attr_store: TrackAttributeStore,
    vectors: VectorStore,
    derived_from: str,
    samples_per_track: int = 5,
) -> AppearanceStats:
    """Store colours, a bag flag, and one placeholder vector per person track.

    A closed source is opened for one pass and closed afterwards. A source
    that is already open is read from its current position and left open.
    Boxes come from ``store`` for ``run_id`` and ``source.camera_id``.
    Tracklets for another camera are ignored. ``bag_tracklets`` supply the
    bag boxes. Person tracks are the only rows written.

    Re-runs replace this camera before re-adding. Attribute rows and
    appearance vectors for this ``run_id`` and ``source.camera_id`` are
    deleted, then the new batch is written. Another camera's rows in the
    same run stay where they are. ``VectorStore`` still allows duplicate
    keys; this function does not dedupe inside the store. The delete and
    the writes are not atomic.

    A source that yields no frames raises ``ValueError`` before that
    delete. A source that yields frames but never yields a planned
    sample frame raises ``ValueError`` before that delete. A person
    track whose boxes are all out of frame is stored as attributes
    and gets no vector.
    """
    limit = _require_limit(samples_per_track)
    origin = _require_origin(derived_from)
    camera_id = source.camera_id
    people = _for_camera(tracklets, camera_id, _PERSON)
    bags = _for_camera(bag_tracklets, camera_id, _BAGS)
    observations = store.read(camera_id, run_id)
    person_boxes = _boxes_for(observations, _track_ids(people))
    bag_lookup = _boxes_for(observations, _track_ids(bags))
    planned = _plan(people, person_boxes, limit)
    present = _present_on_frame(people, person_boxes)
    bags_on_frame = _bags_on_frame(bags, bag_lookup)
    parts, frames_read, planned_read = _read_once(
        source, planned, present, bags_on_frame, extractor, embedder
    )
    _require_planned_frame(frames_read, planned_read)
    pending = _pending(parts, camera_id, embedder)
    _replace_camera(attr_store, vectors, run_id, camera_id)
    _store_pending(pending, attr_store, vectors, camera_id, run_id, origin)
    return AppearanceStats(
        frames_read=frames_read,
        tracks_described=len(pending.attributes),
        embeddings_stored=len(pending.embeddings),
    )


def _require_planned_frame(frames_read: int, planned_read: int) -> None:
    """Fail closed before a delete when the read missed every planned sample."""
    if frames_read == 0:
        raise ValueError("source yielded no frames")
    if planned_read == 0:
        raise ValueError("no planned sample frame was read")


def _require_limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"samples_per_track must be an int >= 1, got {value!r}")
    return value


def _require_origin(value: str) -> str:
    if value not in ALLOWED_DERIVED_FROM:
        allowed = ", ".join(sorted(ALLOWED_DERIVED_FROM))
        raise ValueError(f"derived_from must be one of {allowed}, got {value!r}")
    return value


def _for_camera(
    tracklets: Sequence[Tracklet],
    camera_id: str,
    classes: frozenset[DetClass],
) -> tuple[Tracklet, ...]:
    return tuple(
        item for item in tracklets if item.camera_id == camera_id and item.det_class in classes
    )


def _track_ids(tracklets: Sequence[Tracklet]) -> set[int]:
    return {item.track_id for item in tracklets}


def _boxes_for(
    observations: Sequence[TrackObservation],
    track_ids: set[int],
) -> dict[tuple[int, int], Box]:
    found: dict[tuple[int, int], Box] = {}
    for row in observations:
        if row.track_id not in track_ids:
            continue
        found.setdefault((row.track_id, row.frame_number), row.box)
    return found


def _plan(
    tracklets: Sequence[Tracklet],
    boxes: Mapping[tuple[int, int], Box],
    limit: int,
) -> dict[int, tuple[_Sample, ...]]:
    grouped: dict[int, list[_Sample]] = {}
    for tracklet in tracklets:
        for sample in _samples(tracklet, boxes, limit):
            grouped.setdefault(sample.frame_number, []).append(sample)
    return {
        frame_number: tuple(sorted(items, key=_by_track)) for frame_number, items in grouped.items()
    }


def _by_track(sample: _Sample) -> int:
    return sample.track_id


def _samples(
    tracklet: Tracklet,
    boxes: Mapping[tuple[int, int], Box],
    limit: int,
) -> tuple[_Sample, ...]:
    frames = tuple(dict.fromkeys(tracklet.frames))
    chosen: list[_Sample] = []
    for index in _indexes(len(frames), limit):
        frame_number = frames[index]
        box = boxes.get((tracklet.track_id, frame_number))
        if box is None:
            continue
        chosen.append(_Sample(tracklet.track_id, frame_number, box))
    return tuple(chosen)


def _indexes(count: int, limit: int) -> tuple[int, ...]:
    """Even positions along ``count`` frames, at most ``limit`` of them."""
    if count <= 0:
        return ()
    take = count if count < limit else limit
    if take == 1:
        return (count // 2,)
    last = count - 1
    steps = take - 1
    return tuple((offset * last) // steps for offset in range(take))


def _present_on_frame(
    tracklets: Sequence[Tracklet],
    boxes: Mapping[tuple[int, int], Box],
) -> dict[int, tuple[_Sample, ...]]:
    """Every person box on each frame, including tracks that are not sampled."""
    grouped: dict[int, list[_Sample]] = {}
    for tracklet in tracklets:
        for frame_number in dict.fromkeys(tracklet.frames):
            box = boxes.get((tracklet.track_id, frame_number))
            if box is None:
                continue
            grouped.setdefault(frame_number, []).append(
                _Sample(tracklet.track_id, frame_number, box)
            )
    return {
        frame_number: tuple(sorted(items, key=_by_track)) for frame_number, items in grouped.items()
    }


def _bags_on_frame(
    tracklets: Sequence[Tracklet],
    boxes: Mapping[tuple[int, int], Box],
) -> dict[int, tuple[Box, ...]]:
    grouped: dict[int, list[Box]] = {}
    for tracklet in tracklets:
        for frame_number in dict.fromkeys(tracklet.frames):
            box = boxes.get((tracklet.track_id, frame_number))
            if box is None:
                continue
            grouped.setdefault(frame_number, []).append(box)
    return {frame_number: tuple(items) for frame_number, items in grouped.items()}


def _read_once(
    source: VideoSource,
    planned: Mapping[int, tuple[_Sample, ...]],
    present: Mapping[int, tuple[_Sample, ...]],
    bags_on_frame: Mapping[int, tuple[Box, ...]],
    extractor: AttributeExtractor,
    embedder: AppearanceEmbedder,
) -> tuple[dict[int, _TrackParts], int, int]:
    parts: dict[int, _TrackParts] = {}
    frames_read = 0
    planned_read = 0
    opened_here = _open_source(source)
    try:
        for frame in source.frames():
            frames_read += 1
            if frame.frame_number in planned:
                planned_read += 1
            _consume_frame(frame, planned, present, bags_on_frame, extractor, embedder, parts)
    finally:
        _close_source(source, opened_here)
    return parts, frames_read, planned_read


def _open_source(source: VideoSource) -> bool:
    if source.is_open():
        return False
    source.open()
    return True


def _close_source(source: VideoSource, opened_here: bool) -> None:
    if opened_here:
        source.close()


def _consume_frame(
    frame: Frame,
    planned: Mapping[int, tuple[_Sample, ...]],
    present: Mapping[int, tuple[_Sample, ...]],
    bags_on_frame: Mapping[int, tuple[Box, ...]],
    extractor: AttributeExtractor,
    embedder: AppearanceEmbedder,
    parts: dict[int, _TrackParts],
) -> None:
    """Assign bags across every person on the frame, then keep sampled tracks."""
    chosen = planned.get(frame.frame_number)
    if not chosen:
        return
    image = cast(npt.NDArray[np.uint8], frame.image)
    everyone = present.get(frame.frame_number, ())
    boxes = [item.box for item in everyone]
    bag_boxes = list(bags_on_frame.get(frame.frame_number, ()))
    extracted = extractor.extract_frame(image, boxes, bag_boxes)
    by_track = {
        item.track_id: attributes for item, attributes in zip(everyone, extracted, strict=True)
    }
    for item in chosen:
        _record(parts, item, by_track[item.track_id], image, embedder)


def _record(
    parts: dict[int, _TrackParts],
    sample: _Sample,
    attributes: PersonAttributes,
    image: npt.NDArray[np.uint8],
    embedder: AppearanceEmbedder,
) -> None:
    bucket = _bucket(parts, sample.track_id, sample.frame_number)
    bucket.upper.append((attributes.upper_clothing_color, float(attributes.upper_confidence)))
    bucket.lower.append((attributes.lower_clothing_color, float(attributes.lower_confidence)))
    if attributes.bag_on_track:
        bucket.bag = True
    _embed_crop(bucket, image, sample.box, embedder)


def _bucket(parts: dict[int, _TrackParts], track_id: int, frame_number: int) -> _TrackParts:
    found = parts.get(track_id)
    if found is None:
        created = _TrackParts(frame_number=frame_number)
        parts[track_id] = created
        return created
    if frame_number < found.frame_number:
        found.frame_number = frame_number
    return found


def _embed_crop(
    bucket: _TrackParts,
    image: npt.NDArray[np.uint8],
    box: Box,
    embedder: AppearanceEmbedder,
) -> None:
    """Embed one crop and drop it. The crop is not stored or written."""
    crop = _copy_crop(image, box)
    if crop is None:
        return
    bucket.vectors.append(_one_row(embedder, crop))


def _copy_crop(image: npt.NDArray[np.uint8], box: Box) -> npt.NDArray[np.uint8] | None:
    bounds = _bounds(box, int(image.shape[1]), int(image.shape[0]))
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    crop = image[top:bottom, left:right].copy()
    return cast(npt.NDArray[np.uint8], crop)


def _bounds(box: Box, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Clip with floor on the origin and ceil on the far corner."""
    x1, y1, x2, y2 = box
    left = math.floor(max(0.0, min(float(x1), float(width))))
    top = math.floor(max(0.0, min(float(y1), float(height))))
    right = math.ceil(max(0.0, min(float(x2), float(width))))
    bottom = math.ceil(max(0.0, min(float(y2), float(height))))
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def _one_row(
    embedder: AppearanceEmbedder,
    crop: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float64]:
    embedded = np.asarray(embedder.embed((crop,)), dtype=np.float64)
    if embedded.ndim != 2 or int(embedded.shape[0]) != 1:
        raise ValueError("embed must return one row for one crop")
    row = np.array(embedded[0], dtype=np.float64, copy=True)
    return cast(npt.NDArray[np.float64], row)


def _pending(
    parts: Mapping[int, _TrackParts],
    camera_id: str,
    embedder: AppearanceEmbedder,
) -> _Pending:
    attributes: list[_StoredTrack] = []
    embeddings: list[AppearanceEmbedding] = []
    label = embedder.name
    for track_id in sorted(parts):
        bucket = parts[track_id]
        upper, upper_confidence = _majority(bucket.upper)
        lower, lower_confidence = _majority(bucket.lower)
        attributes.append(
            _StoredTrack(
                track_id=track_id,
                upper=upper.value,
                lower=lower.value,
                bag=bucket.bag,
                upper_confidence=upper_confidence,
                lower_confidence=lower_confidence,
                n_frames=len(bucket.upper),
            )
        )
        if not bucket.vectors:
            continue
        embeddings.append(_embedding(camera_id, track_id, bucket, label))
    return _Pending(tuple(attributes), tuple(embeddings))


def _embedding(
    camera_id: str,
    track_id: int,
    bucket: _TrackParts,
    embedder_label: str,
) -> AppearanceEmbedding:
    return AppearanceEmbedding(
        camera_id=camera_id,
        track_id=track_id,
        frame_number=bucket.frame_number,
        vector=_unit_vector(bucket.vectors),
        embedder_name=embedder_label,
        placeholder=True,
    )


def _majority(votes: Sequence[tuple[Color, float]]) -> tuple[Color, float]:
    counts: dict[Color, int] = {}
    totals: dict[Color, float] = {}
    for color, confidence in votes:
        counts[color] = counts.get(color, 0) + 1
        totals[color] = totals.get(color, 0.0) + float(confidence)
    winner = _sole_leader(counts)
    if winner is None:
        return Color.UNKNOWN, 0.0
    return winner, float(totals[winner] / counts[winner])


def _sole_leader(counts: Mapping[Color, int]) -> Color | None:
    best = max(counts.values())
    leaders = [color for color, count in counts.items() if count == best]
    if len(leaders) != 1:
        return None
    return leaders[0]


def _unit_vector(rows: Sequence[npt.NDArray[np.float64]]) -> tuple[float, ...]:
    matrix = np.stack([np.asarray(row, dtype=np.float64) for row in rows], axis=0)
    mean = np.mean(matrix, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = float(np.linalg.norm(np.asarray(mean, dtype=np.float64)))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("mean appearance vector must be finite and non-zero")
    flat = np.asarray(mean, dtype=np.float64).reshape(-1) / norm
    return tuple(float(item) for item in flat)


def _replace_camera(
    attr_store: TrackAttributeStore,
    vectors: VectorStore,
    run_id: str,
    camera_id: str,
) -> None:
    """Delete this camera's previous rows for ``run_id``. Other cameras stay.

    The following writes are separate calls. This delete and those writes
    are not atomic.
    """
    attr_store.delete_where(run_id=run_id, camera_id=camera_id)
    vectors.delete_where(run_id=run_id, camera_id=camera_id)


def _store_pending(
    pending: _Pending,
    attr_store: TrackAttributeStore,
    vectors: VectorStore,
    camera_id: str,
    run_id: str,
    derived_from: str,
) -> None:
    for row in pending.attributes:
        attr_store.write(
            camera_id,
            row.track_id,
            run_id,
            derived_from,
            row.upper,
            row.lower,
            row.bag,
            row.upper_confidence,
            row.lower_confidence,
            row.n_frames,
        )
    vectors.add(run_id, derived_from, pending.embeddings)
