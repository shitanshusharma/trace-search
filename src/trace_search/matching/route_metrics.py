"""Segment recall, segment precision, and false-merge rate.

T-E-6 has not landed, so this module does not define ``Route`` or
``RouteSegment`` and does not import ``matching.route``. ``RouteSegmentLike``
and ``RouteLike`` are read-only protocols with T-E-6's field names. A later
frozen dataclass with those fields satisfies the protocols unchanged.

A predicted segment matches a truth segment when all of the following hold:

- the ownership map gives the predicted track the truth segment's figure
- the camera ids are equal
- the gap between the two intervals is at most 10.0 seconds

The gap is the time between the later start and the earlier end. Overlap and
touching endpoints have a gap of 0. A gap of 10.0 matches. A gap of 10.1 does
not. The tolerance is closed on both sides (predicted entirely before the
truth segment, or entirely after it).

Matching is one-to-one. Candidate pairs are sorted by gap, then predicted
index, then truth index. A pair is kept only when that predicted segment and
that truth segment are both still free. One truth segment cannot match many
predictions, and one prediction cannot match many truth segments.

``segment_recall`` is matched truth segments / truth segments.
``segment_precision`` is matched predicted segments / predicted segments.
An empty denominator raises ``ValueError``. It is not reported as 0.0 or 1.0.
A real zero, when the denominator is positive and nothing matched, is 0.0.

``false_merge_rate`` is the number of cross-camera links whose tracks belong
to different figures, divided by all cross-camera links in the routes.
Cross-camera links for false-merge are consecutive segments in a route with a different camera_id.
Same-camera consecutive pairs aren't links in the denominator.
The seed is not an extra link. No cross-camera link raises ``ValueError``
rather than returning 0.0.

Camera ids and figures are refused, not rewritten. A whitespace-only value is
empty. A value also raises when it differs from ``str.strip`` of itself, when
NFKC would change it, or when it contains a format character (Unicode
category ``Cf``). Track ids reject bools. Times must be finite, and
``t_exit`` must be >= ``t_entry``.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

__all__ = [
    "Ownership",
    "RouteLike",
    "RouteSegmentLike",
    "TruthSegment",
    "false_merge_rate",
    "matched_pairs",
    "segment_precision",
    "segment_recall",
]

# Closed tolerance. 10.0 matches. 10.1 does not. Both sides use this one value.
_TOLERANCE_S = 10.0

Ownership = Mapping[tuple[str, int], str]


class RouteSegmentLike(Protocol):
    """Read-only T-E-6 segment fields. This is not a ``RouteSegment`` class."""

    @property
    def camera_id(self) -> str:
        raise NotImplementedError

    @property
    def track_id(self) -> int:
        raise NotImplementedError

    @property
    def t_entry(self) -> float:
        raise NotImplementedError

    @property
    def t_exit(self) -> float:
        raise NotImplementedError


class RouteLike(Protocol):
    """Read-only T-E-6 route fields. This is not a ``Route`` class."""

    @property
    def seed_camera(self) -> str:
        raise NotImplementedError

    @property
    def seed_track(self) -> int:
        raise NotImplementedError

    @property
    def segments(self) -> Sequence[RouteSegmentLike]:
        raise NotImplementedError


@dataclass(frozen=True)
class TruthSegment:
    """One truth visit: a figure on one camera from ``t_entry`` to ``t_exit``.

    ``figure`` is an opaque truth label. Ids are refused, not rewritten.
    """

    figure: str
    camera_id: str
    t_entry: float
    t_exit: float

    def __post_init__(self) -> None:
        """Reject a bad id, a non-finite time, or an exit before the entry."""
        figure = _checked_label("figure", self.figure)
        camera_id = _checked_label("camera_id", self.camera_id)
        t_entry = _require_finite("t_entry", self.t_entry)
        t_exit = _require_finite("t_exit", self.t_exit)
        _require_interval(t_entry, t_exit)
        object.__setattr__(self, "figure", figure)
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "t_entry", t_entry)
        object.__setattr__(self, "t_exit", t_exit)


@dataclass(frozen=True)
class _TimedTrack:
    """One validated predicted segment plus the figure of its track."""

    camera_id: str
    track_id: int
    t_entry: float
    t_exit: float
    figure: str


def matched_pairs(
    predicted: Sequence[RouteSegmentLike],
    truth: Sequence[TruthSegment],
    ownership: Ownership,
) -> tuple[tuple[int, int], ...]:
    """Return ``(predicted_index, truth_index)`` pairs in assignment order.

    Order is increasing gap, then predicted index, then truth index. Each
    index appears at most once. See the module docstring for the match rule.
    """
    checked = _checked_ownership(ownership)
    predicted_tracks = _load_predicted(predicted, checked)
    truth_segments = _load_truth(truth)
    return _assign(_candidates(predicted_tracks, truth_segments))


def segment_recall(
    predicted: Sequence[RouteSegmentLike],
    truth: Sequence[TruthSegment],
    ownership: Ownership,
) -> float:
    """Matched truth segments divided by the truth-segment count.

    An empty truth sequence raises ``ValueError``. The result is not 1.0.
    """
    pairs = matched_pairs(predicted, truth, ownership)
    return _ratio(_unique_second(pairs), len(truth), "segment_recall")


def segment_precision(
    predicted: Sequence[RouteSegmentLike],
    truth: Sequence[TruthSegment],
    ownership: Ownership,
) -> float:
    """Matched predicted segments divided by the predicted-segment count.

    An empty predicted sequence raises ``ValueError``. The result is not 1.0.
    """
    pairs = matched_pairs(predicted, truth, ownership)
    return _ratio(_unique_first(pairs), len(predicted), "segment_precision")


def false_merge_rate(routes: Sequence[RouteLike], ownership: Ownership) -> float:
    """Bad cross-camera links divided by all cross-camera links.

    Cross-camera links for false-merge are consecutive segments in a route
    with a different camera_id. Same-camera consecutive pairs aren't links
    in the denominator. The seed is not an extra link. No cross-camera link
    raises ``ValueError`` instead of returning 0.0. Every seed track and
    every segment track must be in ``ownership``.
    """
    checked = _checked_ownership(ownership)
    loaded = _load_routes(routes, checked)
    bad, denominator = _link_counts(loaded)
    return _ratio(bad, denominator, "false_merge_rate")


def _ratio(matched: int, total: int, label: str) -> float:
    """Return ``matched / total``. An empty total is an error, not 0 or 1."""
    if total == 0:
        raise ValueError(f"{label} denominator is empty")
    return matched / total


def _unique_first(pairs: Sequence[tuple[int, int]]) -> int:
    return len({pred_index for pred_index, _truth_index in pairs})


def _unique_second(pairs: Sequence[tuple[int, int]]) -> int:
    return len({truth_index for _pred_index, truth_index in pairs})


def _assign(candidates: Sequence[tuple[float, int, int]]) -> tuple[tuple[int, int], ...]:
    """Greedy one-to-one assignment. ``candidates`` are sorted by the caller."""
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    chosen: list[tuple[int, int]] = []
    for _gap, pred_index, truth_index in sorted(candidates):
        if not _pair_is_free(pred_index, truth_index, used_pred, used_truth):
            continue
        used_pred.add(pred_index)
        used_truth.add(truth_index)
        chosen.append((pred_index, truth_index))
    return tuple(chosen)


def _pair_is_free(
    pred_index: int,
    truth_index: int,
    used_pred: set[int],
    used_truth: set[int],
) -> bool:
    """False when this prediction or this truth segment is already matched."""
    if pred_index in used_pred:
        return False
    if truth_index in used_truth:
        return False
    return True


def _candidates(
    predicted: Sequence[_TimedTrack],
    truth: Sequence[TruthSegment],
) -> tuple[tuple[float, int, int], ...]:
    found: list[tuple[float, int, int]] = []
    for pred_index, segment in enumerate(predicted):
        _collect_candidates(found, pred_index, segment, truth)
    return tuple(found)


def _collect_candidates(
    found: list[tuple[float, int, int]],
    pred_index: int,
    segment: _TimedTrack,
    truth: Sequence[TruthSegment],
) -> None:
    for truth_index, truth_segment in enumerate(truth):
        gap = _candidate_gap(segment, truth_segment)
        if gap is None:
            continue
        found.append((gap, pred_index, truth_index))


def _candidate_gap(segment: _TimedTrack, truth_segment: TruthSegment) -> float | None:
    """Return the gap, or None when the pair is not allowed to match."""
    if segment.figure != truth_segment.figure:
        return None
    if segment.camera_id != truth_segment.camera_id:
        return None
    gap = _gap(segment.t_entry, segment.t_exit, truth_segment.t_entry, truth_segment.t_exit)
    if not _within_tolerance(gap):
        return None
    return gap


def _within_tolerance(gap: float) -> bool:
    """Closed check against the single 10 second tolerance."""
    return gap <= _TOLERANCE_S


def _gap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Seconds between two intervals. Overlap and touching both return 0."""
    if end_a < start_b:
        return start_b - end_a
    if end_b < start_a:
        return start_a - end_b
    return 0.0


def _link_counts(routes: Sequence[Sequence[_TimedTrack]]) -> tuple[int, int]:
    """Return ``(bad cross-camera links, all cross-camera links)``.

    A link is consecutive segments whose ``camera_id`` differs.
    """
    bad = 0
    denominator = 0
    for left, right in _adjacent_pairs(routes):
        if not _is_cross_camera(left, right):
            continue
        denominator += 1
        if left.figure != right.figure:
            bad += 1
    return bad, denominator


def _is_cross_camera(left: _TimedTrack, right: _TimedTrack) -> bool:
    """True when these consecutive segments have a different ``camera_id``."""
    return left.camera_id != right.camera_id


def _adjacent_pairs(
    routes: Sequence[Sequence[_TimedTrack]],
) -> tuple[tuple[_TimedTrack, _TimedTrack], ...]:
    pairs: list[tuple[_TimedTrack, _TimedTrack]] = []
    for segments in routes:
        pairs.extend(_adjacent_in(segments))
    return tuple(pairs)


def _adjacent_in(
    segments: Sequence[_TimedTrack],
) -> tuple[tuple[_TimedTrack, _TimedTrack], ...]:
    """Consecutive segments in route order. Non-neighbours are not links."""
    items = tuple(segments)
    return tuple(zip(items, items[1:]))


def _load_routes(
    routes: object,
    ownership: Ownership,
) -> tuple[tuple[_TimedTrack, ...], ...]:
    raw = _require_sequence("routes", routes)
    return tuple(_load_route(route, ownership) for route in raw)


def _load_route(route: object, ownership: Ownership) -> tuple[_TimedTrack, ...]:
    seed_camera = _checked_label("seed_camera", _attr(route, "seed_camera"))
    seed_track = _require_track_id("seed_track", _attr(route, "seed_track"))
    _figure_for(seed_camera, seed_track, ownership)
    raw_segments = _require_sequence("segments", _attr(route, "segments"))
    return tuple(_load_segment(item, ownership) for item in raw_segments)


def _load_predicted(predicted: object, ownership: Ownership) -> tuple[_TimedTrack, ...]:
    raw = _require_sequence("predicted", predicted)
    return tuple(_load_segment(item, ownership) for item in raw)


def _load_segment(segment: object, ownership: Ownership) -> _TimedTrack:
    camera_id = _checked_label("camera_id", _attr(segment, "camera_id"))
    track_id = _require_track_id("track_id", _attr(segment, "track_id"))
    t_entry = _require_finite("t_entry", _attr(segment, "t_entry"))
    t_exit = _require_finite("t_exit", _attr(segment, "t_exit"))
    _require_interval(t_entry, t_exit)
    figure = _figure_for(camera_id, track_id, ownership)
    return _TimedTrack(camera_id, track_id, t_entry, t_exit, figure)


def _load_truth(truth: object) -> tuple[TruthSegment, ...]:
    raw = _require_sequence("truth", truth)
    return tuple(_require_truth(item) for item in raw)


def _require_truth(item: object) -> TruthSegment:
    if isinstance(item, TruthSegment):
        return item
    raise ValueError(f"truth must be TruthSegment records, got {type(item).__name__}")


def _figure_for(camera_id: str, track_id: int, ownership: Ownership) -> str:
    figure = ownership.get((camera_id, track_id))
    if figure is None:
        raise ValueError(f"no figure for camera {camera_id!r} track_id {track_id}")
    return figure


def _checked_ownership(ownership: object) -> dict[tuple[str, int], str]:
    mapping = _require_mapping(ownership)
    checked: dict[tuple[str, int], str] = {}
    for key, figure in mapping:
        camera_id, track_id = _ownership_key(key)
        checked[(camera_id, track_id)] = _checked_label("figure", figure)
    return checked


def _require_mapping(ownership: object) -> tuple[tuple[object, object], ...]:
    if not isinstance(ownership, Mapping):
        raise ValueError(f"ownership must be a mapping, got {type(ownership).__name__}")
    pairs: list[tuple[object, object]] = []
    for key, figure in ownership.items():
        pairs.append((cast(object, key), cast(object, figure)))
    return tuple(pairs)


def _ownership_key(key: object) -> tuple[str, int]:
    if isinstance(key, tuple) and len(key) == 2:
        camera_raw = cast(object, key[0])
        track_raw = cast(object, key[1])
        camera_id = _checked_label("camera_id", camera_raw)
        track_id = _require_track_id("track_id", track_raw)
        return camera_id, track_id
    raise ValueError(f"ownership key must be (camera_id, track_id), got {key!r}")


def _require_sequence(label: str, value: object) -> Sequence[object]:
    """Reject a bare ``str`` or ``bytes``. Those iterate as characters."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence of records, got {type(value).__name__}")
    return value


def _attr(record: object, field_name: str) -> object:
    if not hasattr(record, field_name):
        raise ValueError(f"record is missing {field_name}")
    return cast(object, getattr(record, field_name))


def _checked_label(label: str, value: object) -> str:
    """Return the id unchanged. Never strip, fold, or rewrite it."""
    text = _require_text(label, value)
    _refuse_strip(label, text)
    _refuse_nfkc(label, text)
    _refuse_format(label, text)
    return text


def _require_text(label: str, value: object) -> str:
    """Reject a missing id. A whitespace-only string counts as empty."""
    if isinstance(value, str) and value.strip() != "":
        return value
    raise ValueError(f"{label} must be a non-empty id, got {value!r}")


def _refuse_strip(label: str, value: str) -> None:
    """Refuse surrounding whitespace. Do not strip the id."""
    if value != value.strip():
        raise ValueError(f"{label} must not have leading or trailing whitespace, got {value!r}")


def _refuse_nfkc(label: str, value: str) -> None:
    """Refuse an id that NFKC would change. Do not fold the id."""
    if unicodedata.normalize("NFKC", value) != value:
        raise ValueError(f"{label} must already be in NFKC, got {value!r}")


def _refuse_format(label: str, value: str) -> None:
    """Refuse Unicode format characters, including the zero-width marks."""
    for character in value:
        if unicodedata.category(character) == "Cf":
            raise ValueError(f"{label} must not contain a format character, got {value!r}")


def _require_track_id(label: str, value: object) -> int:
    """Return an int track id. ``True`` and ``False`` are not track ids."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an int track id, not a bool, got {value!r}")
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an int track id, got {value!r}")
    return value


def _require_finite(label: str, value: object) -> float:
    """Return a finite float. Booleans and other types are rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _require_interval(t_entry: float, t_exit: float) -> None:
    if t_exit < t_entry:
        raise ValueError(f"t_exit ({t_exit}) must be >= t_entry ({t_entry})")
