"""Build one ordered route from a seed track.

A route is computed for one call. It is not stored, and it has no id.

``Link`` belongs to the linker (T-E-5) and is not defined here. ``LinkLike``
is the read-only shape this walk follows: ``from_camera``, ``from_track``,
``to_camera``, ``to_track``, ``gap_seconds`` and ``score``. A later ``Link``
with those fields satisfies ``LinkLike``. ``gap_seconds`` and ``score`` do
not choose the path. When several links leave one track, the next track is
the smallest ``(camera_id, track_id)``. The same order breaks ties backward.

``build_route`` walks incoming links backward from the seed, then outgoing
links forward. The walk is a loop bounded by the number of tracklets. It
does not call itself. It stops when the next track was already used, when
the next camera is outside ``camera_ids``, or when the next tracklet sits
outside ``t_window``. It does not jump over a track that stopped it.
``camera_ids=None`` allows every camera. ``t_window=None`` allows every
time. ``t_window`` is an inclusive pair ``(start, end)``. A tracklet overlaps
that pair when ``t_end >= start`` and ``t_start <= end``.

Segments come back in time order: ``t_entry``, ``t_exit``, ``camera_id``,
``track_id``. Times come from the tracklet catalogue (``t_start``, ``t_end``).
The seed picks ``(camera_id, track_id)`` and must be in that catalogue, with
the same ``t_start`` and ``t_end``.

Camera ids and track ids are refused, not rewritten. A whitespace-only id is
empty. An id also raises when it differs from ``str.strip`` of itself, when
NFKC changes it, or when it contains a format character (Unicode category
``Cf``, including U+200B, U+200C, U+200D, U+2060 and U+FEFF). Track ids are
ints. A string track id is subject to the same id rules and is not parsed
into an int. Bools are not track ids.

The returned route satisfies the read-only ``RouteLike`` and
``RouteSegmentLike`` protocols in ``route_metrics``.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from trace_search.matching.route_metrics import RouteLike, RouteSegmentLike
from trace_search.matching.tracklets import Tracklet

__all__ = (
    "LinkLike",
    "Route",
    "RouteSegment",
    "build_route",
)

TrackKey = tuple[str, int]
Window = tuple[float, float]


class LinkLike(Protocol):
    """Read-only T-E-5 link fields. This is not a ``Link`` class."""

    @property
    def from_camera(self) -> str:
        raise NotImplementedError

    @property
    def from_track(self) -> int:
        raise NotImplementedError

    @property
    def to_camera(self) -> str:
        raise NotImplementedError

    @property
    def to_track(self) -> int:
        raise NotImplementedError

    @property
    def gap_seconds(self) -> float:
        raise NotImplementedError

    @property
    def score(self) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class RouteSegment:
    """One camera visit on a route, from ``t_entry`` to ``t_exit``."""

    camera_id: str
    track_id: int
    t_entry: float
    t_exit: float

    def __post_init__(self) -> None:
        """Refuse a bad id or a non-finite interval. Do not rewrite the id."""
        camera_id = _require_camera("camera_id", self.camera_id)
        track_id = _require_track_id("track_id", self.track_id)
        t_entry = _require_finite("t_entry", self.t_entry)
        t_exit = _require_finite("t_exit", self.t_exit)
        _require_interval("t_entry", "t_exit", t_entry, t_exit)
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "t_entry", t_entry)
        object.__setattr__(self, "t_exit", t_exit)


@dataclass(frozen=True)
class Route:
    """One computed route. It has no id and it is not stored."""

    seed_camera: str
    seed_track: int
    segments: tuple[RouteSegment, ...]

    def __post_init__(self) -> None:
        """Refuse a bad seed id or a segments value that is not a tuple."""
        seed_camera = _require_camera("seed_camera", self.seed_camera)
        seed_track = _require_track_id("seed_track", self.seed_track)
        segments = _require_segments(self.segments)
        object.__setattr__(self, "seed_camera", seed_camera)
        object.__setattr__(self, "seed_track", seed_track)
        object.__setattr__(self, "segments", segments)


def build_route(
    seed: Tracklet,
    tracklets: Iterable[Tracklet],
    links: Iterable[LinkLike],
    *,
    t_window: tuple[float, float] | None = None,
    camera_ids: Collection[str] | None = None,
) -> Route:
    """Follow links backward, then forward, and return the segments in time order.

    ``t_window`` is an inclusive ``(start, end)`` pair, or ``None``.
    ``camera_ids=None`` allows every camera. A camera outside ``camera_ids``
    stops the walk. The route is not stored.
    """
    window = _require_window(t_window)
    allowed = _require_cameras(camera_ids)
    seed_tracklet = _require_tracklet(seed)
    index = _index_tracklets(_records("tracklets", tracklets))
    listed = _require_listed_seed(seed_tracklet, index)
    _require_seed_allowed(listed, window, allowed)
    pairs = tuple(_link_ends(link) for link in _records("links", links))
    seed_key = (listed.camera_id, listed.track_id)
    seen: set[TrackKey] = {seed_key}
    backward = _walk(seed_key, _edges(pairs, backward=True), seen, index, window, allowed)
    forward = _walk(seed_key, _edges(pairs, backward=False), seen, index, window, allowed)
    ordered = _chain_tracklets(backward, listed, forward)
    return _checked(Route(listed.camera_id, listed.track_id, _segments_in_time(ordered)))


def _checked(route: Route) -> Route:
    """The metrics protocols accept this route. The same object comes back."""
    view: RouteLike = _as_route_like(route)
    segment: RouteSegmentLike = _as_segment_like(route.segments[0])
    # Read the protocol fields. mypy --strict rejects this if Route no longer matches.
    _ = (view.seed_camera, view.seed_track, len(view.segments), segment.t_entry, segment.t_exit)
    return route


def _as_route_like(route: Route) -> RouteLike:
    """Return ``route`` as the T-E-7 read-only route protocol."""
    return route


def _as_segment_like(segment: RouteSegment) -> RouteSegmentLike:
    """Return ``segment`` as the T-E-7 read-only segment protocol."""
    return segment


def _chain_tracklets(
    backward: tuple[Tracklet, ...],
    seed: Tracklet,
    forward: tuple[Tracklet, ...],
) -> tuple[Tracklet, ...]:
    """Backward steps were discovered near-to-far. Reverse them into time order."""
    return (*reversed(backward), seed, *forward)


def _segments_in_time(tracklets: Sequence[Tracklet]) -> tuple[RouteSegment, ...]:
    segments = tuple(_segment_from(item) for item in tracklets)
    return tuple(sorted(segments, key=_time_key))


def _segment_from(tracklet: Tracklet) -> RouteSegment:
    return RouteSegment(
        camera_id=tracklet.camera_id,
        track_id=tracklet.track_id,
        t_entry=tracklet.t_start,
        t_exit=tracklet.t_end,
    )


def _time_key(segment: RouteSegment) -> tuple[float, float, str, int]:
    """Entry, then exit, then camera id, then track id."""
    return (segment.t_entry, segment.t_exit, segment.camera_id, segment.track_id)


def _walk(
    start: TrackKey,
    edges: Mapping[TrackKey, tuple[TrackKey, ...]],
    seen: set[TrackKey],
    index: Mapping[TrackKey, Tracklet],
    window: Window | None,
    cameras: frozenset[str] | None,
) -> tuple[Tracklet, ...]:
    """Extend ``start`` along ``edges``. One step per tracklet, then stop.

    ``seen`` is shared by the backward walk and the forward walk. A track
    already in it ends this direction. The loop does not call itself.
    """
    found: list[Tracklet] = []
    current = start
    # At most one added segment per catalogue row, plus one step that stops.
    for _step in range(len(index)):
        nxt = _first_dest(edges, current)
        if nxt is None:
            break
        if _stop_reason(nxt, seen, index, window, cameras):
            break
        seen.add(nxt)
        found.append(index[nxt])
        current = nxt
    return tuple(found)


def _first_dest(
    edges: Mapping[TrackKey, tuple[TrackKey, ...]],
    key: TrackKey,
) -> TrackKey | None:
    dests = edges.get(key)
    if dests is None:
        return None
    return dests[0]


def _stop_reason(
    key: TrackKey,
    seen: set[TrackKey],
    index: Mapping[TrackKey, Tracklet],
    window: Window | None,
    cameras: frozenset[str] | None,
) -> bool:
    """True when ``key`` must not be entered. A missing tracklet raises."""
    if _already_visited(key, seen):
        return True
    if _camera_blocked(key[0], cameras):
        return True
    tracklet = _tracklet_for(key, index)
    if _outside_window(tracklet, window):
        return True
    return False


def _already_visited(key: TrackKey, seen: set[TrackKey]) -> bool:
    """True when this track is already on the route. That is a cycle."""
    return key in seen


def _camera_blocked(camera_id: str, cameras: frozenset[str] | None) -> bool:
    """True when ``camera_ids`` was set and this camera is not in it."""
    if cameras is None:
        return False
    return camera_id not in cameras


def _tracklet_for(key: TrackKey, index: Mapping[TrackKey, Tracklet]) -> Tracklet:
    tracklet = index.get(key)
    if tracklet is None:
        raise ValueError(f"link names camera {key[0]!r} track_id {key[1]} which has no tracklet")
    return tracklet


def _outside_window(tracklet: Tracklet, window: Window | None) -> bool:
    """True when the tracklet does not overlap the inclusive window."""
    if window is None:
        return False
    if tracklet.t_end < window[0]:
        return True
    if tracklet.t_start > window[1]:
        return True
    return False


def _edges(
    pairs: Sequence[tuple[TrackKey, TrackKey]],
    *,
    backward: bool,
) -> dict[TrackKey, tuple[TrackKey, ...]]:
    grouped: dict[TrackKey, list[TrackKey]] = {}
    for origin, dest in pairs:
        key, other = _edge_ends(origin, dest, backward=backward)
        grouped.setdefault(key, []).append(other)
    return {key: _by_track_key(group) for key, group in grouped.items()}


def _edge_ends(
    origin: TrackKey,
    dest: TrackKey,
    *,
    backward: bool,
) -> tuple[TrackKey, TrackKey]:
    """Backward walks toward ``from_*``. Forward walks toward ``to_*``."""
    if backward:
        return dest, origin
    return origin, dest


def _by_track_key(group: list[TrackKey]) -> tuple[TrackKey, ...]:
    """Smallest ``(camera_id, track_id)`` first. Input order does not matter."""
    return tuple(sorted(set(group)))


def _link_ends(link: object) -> tuple[TrackKey, TrackKey]:
    origin = (
        _require_camera("from_camera", _read(link, "from_camera")),
        _require_track_id("from_track", _read(link, "from_track")),
    )
    dest = (
        _require_camera("to_camera", _read(link, "to_camera")),
        _require_track_id("to_track", _read(link, "to_track")),
    )
    return origin, dest


def _read(record: object, field_name: str) -> object:
    return cast(object, getattr(record, field_name))


def _index_tracklets(records: Sequence[object]) -> dict[TrackKey, Tracklet]:
    index: dict[TrackKey, Tracklet] = {}
    for record in records:
        tracklet = _require_tracklet(record)
        key = (tracklet.camera_id, tracklet.track_id)
        if key in index:
            raise ValueError(f"duplicate tracklet for camera {key[0]!r} track_id {key[1]}")
        index[key] = tracklet
    return index


def _require_listed_seed(
    seed: Tracklet,
    index: Mapping[TrackKey, Tracklet],
) -> Tracklet:
    """Return the catalogue row for the seed. Times must agree."""
    key = (seed.camera_id, seed.track_id)
    listed = index.get(key)
    if listed is None:
        raise ValueError(f"seed camera {key[0]!r} track_id {key[1]} is not in tracklets")
    _require_same_span(seed, listed)
    return listed


def _require_same_span(seed: Tracklet, listed: Tracklet) -> None:
    if seed.t_start != listed.t_start:
        raise ValueError("seed times do not match the tracklet catalogue")
    if seed.t_end != listed.t_end:
        raise ValueError("seed times do not match the tracklet catalogue")


def _require_seed_allowed(
    seed: Tracklet,
    window: Window | None,
    cameras: frozenset[str] | None,
) -> None:
    if _camera_blocked(seed.camera_id, cameras):
        raise ValueError(f"seed camera {seed.camera_id!r} is outside camera_ids")
    if _outside_window(seed, window):
        raise ValueError("seed is outside t_window")


def _records(label: str, value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{label} must be an iterable of records, got {type(value).__name__}")
    return tuple(value)


def _require_segments(value: object) -> tuple[RouteSegment, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("segments must be a tuple of RouteSegment")
    if not isinstance(value, tuple):
        raise ValueError("segments must be a tuple of RouteSegment")
    checked: list[RouteSegment] = []
    for item in value:
        if not isinstance(item, RouteSegment):
            raise ValueError(f"segments must be a tuple of RouteSegment, got {type(item).__name__}")
        checked.append(item)
    return tuple(checked)


def _require_tracklet(value: object) -> Tracklet:
    if not isinstance(value, Tracklet):
        raise ValueError(f"tracklet must be a Tracklet, got {type(value).__name__}")
    _require_camera("camera_id", value.camera_id)
    _require_track_id("track_id", value.track_id)
    t_start = _require_finite("t_start", value.t_start)
    t_end = _require_finite("t_end", value.t_end)
    _require_interval("t_start", "t_end", t_start, t_end)
    return value


def _require_window(value: object) -> Window | None:
    if value is None:
        return None
    pair = _require_pair(value)
    start = _require_finite("t_window start", pair[0])
    end = _require_finite("t_window end", pair[1])
    _require_interval("t_window start", "t_window end", start, end)
    return (start, end)


def _require_pair(value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("t_window must be a pair of finite times, or None")
    if len(value) != 2:
        raise ValueError("t_window must be a pair of finite times")
    return value


def _require_cameras(value: object) -> frozenset[str] | None:
    """``None`` means every camera. An empty collection is refused."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError(f"camera_ids must be a collection of camera ids, not a str, got {value!r}")
    if not isinstance(value, Collection):
        raise ValueError(f"camera_ids must be a collection of camera ids, got {value!r}")
    found: list[str] = []
    for item in value:
        found.append(_require_camera("camera_id", item))
    if not found:
        raise ValueError("camera_ids must contain at least one camera id")
    return frozenset(found)


def _require_camera(label: str, value: object) -> str:
    """Return the id unchanged. Never strip, fold, or change case."""
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
    """Return an int track id. Strings are not parsed. Bools are not ids."""
    if isinstance(value, str):
        _require_camera(label, value)
        raise ValueError(f"{label} must be an int track id, got {value!r}")
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


def _require_interval(start_label: str, end_label: str, start: float, end: float) -> None:
    if end < start:
        raise ValueError(f"{end_label} ({end}) must be >= {start_label} ({start})")
