"""Cross-camera links from topology, a transit window, and placeholder vectors.

A link joins an exit on camera A to an entry on camera B. The topology must
have an edge A to B, and ``topology.allows`` must accept the gap. There is
no link inside one camera. A track with no placeholder vector is not linked.
``Link`` has no global id, person id, or route id.

The raw similarity is the cosine of the two placeholder vectors plus
``attr_weight`` times the share of agreeing known attributes (upper, lower,
and bag). The placeholder similarity has no accuracy meaning. Cosine plus
``attr_weight`` times share can fall outside ``[0, 1]``, and ``build_route``
refuses those scores. The stored score is

``((cosine + 1) / 2 + attr_weight * share) / (1 + attr_weight)``.

Cosine is clamped to ``[-1, 1]`` before that map, so the score stays in
``[0, 1]``. ``ATTR_WEIGHT`` and ``MIN_SCORE`` are module-level placeholders.
They were not tuned against any stock clip.

An attribute counts only when both tracks know it. ``UNKNOWN`` and ``None``
are not known. Share is the number agreeing divided by the number known.
With 0 known attributes, share is 0. Each camera pair is assigned one-to-one
with the Hungarian method on ``-score``. Pairs under ``min_score`` are then
dropped. Equal scores break on ``(camera_id, track_id)``.

``topology_mask`` is the gallery mask for the AC-06 harness. A true result
keeps the gallery item. These numbers are not evidence for AC-06, AC-07,
or AC-08.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from trace_search.attributes.attributes import Color
from trace_search.attributes.store import TrackAttributeStore
from trace_search.matching.route import LinkLike
from trace_search.matching.topology import CameraTopology
from trace_search.matching.tracklets import Tracklet, _require_id
from trace_search.reid.eval import EvalItem
from trace_search.reid.types import AppearanceEmbedding

# Placeholder floor. Pairs strictly under this score are dropped after
# assignment. The value was not tuned against any stock clip.
MIN_SCORE = 0.0

# Placeholder weight of the known-attribute share. The value was not tuned
# against any stock clip.
ATTR_WEIGHT = 1.0

# Column term of the tie-break. A score gap of this size or more beats it.
# Not a similarity threshold and not fitted to any clip.
_TIE_BREAK_UNIT = 1e-9

# Extra cost per row index when a column is left over. Smaller than any
# score gap the assignment treats as real. Not fitted to any clip.
_ROW_TIE_WEIGHT = 1e-8

# Above every feasible cost. Feasible costs sit in about [-1, 0].
_BLOCKED = 2.0

__all__ = (
    "ATTR_WEIGHT",
    "MIN_SCORE",
    "Link",
    "link_tracklets",
    "topology_mask",
)

TrackKey = tuple[str, int]
_AttrTriple = tuple[str | None, str | None, bool | None]
_UPPER_KEYS = ("upper", "upper_color", "upper_clothing_color")
_LOWER_KEYS = ("lower", "lower_color", "lower_clothing_color")
_BAG_KEYS = ("bag", "bag_on_track")


class _VectorReader(Protocol):
    """The store methods the linker reads. ``search`` lists the rows."""

    @property
    def dim(self) -> int:
        """Length of every stored vector."""

    def count(self, derived_from: str | None = ..., run_id: str | None = ...) -> int:
        """Count rows. Both arguments omitted counts every row."""

    def search(
        self,
        vector: Sequence[float],
        k: int,
        camera_ids: Sequence[str] | None = ...,
    ) -> Sequence[tuple[AppearanceEmbedding, float]]:
        """Return up to ``k`` rows, cosine highest first."""


@dataclass(frozen=True)
class Link:
    """One anonymous cross-camera candidate. The score is in ``[0, 1]``.

    Fields are the six link columns only. There is no global, person, or
    route id.
    """

    from_camera: str
    from_track: int
    to_camera: str
    to_track: int
    gap_seconds: float
    score: float

    def __post_init__(self) -> None:
        """Refuse a bad id, one camera, a non-finite gap, or a score outside ``[0, 1]``."""
        from_camera = _require_id("from_camera", self.from_camera)
        to_camera = _require_id("to_camera", self.to_camera)
        if from_camera == to_camera:
            raise ValueError(f"a link must cross cameras, got {from_camera!r}")
        from_track = _require_track_id("from_track", self.from_track)
        to_track = _require_track_id("to_track", self.to_track)
        gap_seconds = _require_finite("gap_seconds", self.gap_seconds)
        score = _require_unit_interval("score", self.score)
        object.__setattr__(self, "from_camera", from_camera)
        object.__setattr__(self, "to_camera", to_camera)
        object.__setattr__(self, "from_track", from_track)
        object.__setattr__(self, "to_track", to_track)
        object.__setattr__(self, "gap_seconds", gap_seconds)
        object.__setattr__(self, "score", score)


def link_tracklets(
    tracklets: Sequence[Tracklet],
    topology: CameraTopology,
    vectors: object,
    attributes: object,
    *,
    min_score: float = MIN_SCORE,
    attr_weight: float = ATTR_WEIGHT,
) -> list[Link]:
    """Link exits to entries across each directed edge.

    ``vectors`` is a mapping of ``(camera_id, track_id)`` to a placeholder
    vector, or a vector store with ``dim``, ``count``, and ``search``.
    ``attributes`` is a mapping of the same key to a record, or a
    ``TrackAttributeStore``. A track with no vector is left out.
    ``min_score`` and ``attr_weight`` are keyword-only.
    """
    floor = _require_non_negative("min_score", min_score)
    weight = _require_non_negative("attr_weight", attr_weight)
    indexed = _index_tracklets(tracklets)
    vector_table = _vector_table(vectors)
    attribute_table = _attribute_table(attributes, indexed)
    ready = _with_vectors(indexed, vector_table)
    grouped = _group_by_camera(ready)
    links = _links_across(grouped, topology, vector_table, attribute_table, floor, weight)
    links.sort(key=_link_order)
    return links


def topology_mask(
    topology: CameraTopology,
    time_of: Callable[[EvalItem], float],
) -> Callable[[EvalItem, EvalItem], bool]:
    """Return a gallery mask. True keeps the pair when the topology allows it.

    ``time_of`` of the query item is the exit time. ``time_of`` of the
    gallery item is the entry time. A false result excludes that gallery
    item for that query. The mask is not evidence for AC-06.
    """

    def _allow(query: EvalItem, gallery: EvalItem) -> bool:
        return topology.allows(
            query.camera_id,
            time_of(query),
            gallery.camera_id,
            time_of(gallery),
        )

    return _allow


def _checked(link: Link) -> Link:
    """``Link`` satisfies ``LinkLike``. The same object comes back."""
    view: LinkLike = _as_link_like(link)
    _ = (
        view.from_camera,
        view.from_track,
        view.to_camera,
        view.to_track,
        view.gap_seconds,
        view.score,
    )
    return link


def _as_link_like(link: Link) -> LinkLike:
    """Return ``link`` as the route walk's read-only link protocol."""
    return link


def _index_tracklets(tracklets: Sequence[Tracklet]) -> dict[TrackKey, Tracklet]:
    indexed: dict[TrackKey, Tracklet] = {}
    for item in tracklets:
        key = (item.camera_id, item.track_id)
        if key in indexed:
            raise ValueError(f"duplicate tracklet {key!r}")
        indexed[key] = item
    return indexed


def _vector_table(vectors: object) -> dict[TrackKey, tuple[float, ...]]:
    if isinstance(vectors, Mapping):
        return _vectors_from_mapping(vectors)
    return _vectors_from_store(_as_reader(vectors))


def _as_reader(vectors: object) -> _VectorReader:
    dim = getattr(vectors, "dim", None)
    search = getattr(vectors, "search", None)
    count = getattr(vectors, "count", None)
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError("vectors must be a mapping or a vector store")
    if not callable(search) or not callable(count):
        raise TypeError("vectors must be a mapping or a vector store")
    return cast(_VectorReader, vectors)


def _vectors_from_mapping(vectors: Mapping[object, object]) -> dict[TrackKey, tuple[float, ...]]:
    table: dict[TrackKey, tuple[float, ...]] = {}
    for key, value in vectors.items():
        table[_track_key(key)] = _as_vector(value)
    return table


def _vectors_from_store(reader: _VectorReader) -> dict[TrackKey, tuple[float, ...]]:
    """One vector per track: lowest frame, then the smaller vector tuple.

    ``k`` must be positive, so an empty store is returned without a search.
    """
    total = reader.count()
    if total == 0:
        return {}
    chosen: dict[TrackKey, tuple[int, tuple[float, ...]]] = {}
    for embedding, _cosine in reader.search(_axis_probe(reader.dim), total):
        _keep_earliest(chosen, embedding)
    return {key: vector for key, (_frame, vector) in chosen.items()}


def _axis_probe(dim: int) -> tuple[float, ...]:
    return tuple(1.0 if index == 0 else 0.0 for index in range(dim))


def _keep_earliest(
    chosen: dict[TrackKey, tuple[int, tuple[float, ...]]],
    embedding: AppearanceEmbedding,
) -> None:
    key = (embedding.camera_id, embedding.track_id)
    vector = tuple(float(component) for component in embedding.vector)
    candidate = (embedding.frame_number, vector)
    current = chosen.get(key)
    if current is None or candidate < current:
        chosen[key] = candidate


def _attribute_table(
    attributes: object,
    tracklets: Mapping[TrackKey, Tracklet],
) -> dict[TrackKey, _AttrTriple]:
    if isinstance(attributes, TrackAttributeStore):
        return _attributes_from_store(attributes, tracklets)
    if isinstance(attributes, Mapping):
        return _attributes_from_mapping(attributes)
    raise TypeError("attributes must be a mapping or a TrackAttributeStore")


def _attributes_from_store(
    store: TrackAttributeStore,
    tracklets: Mapping[TrackKey, Tracklet],
) -> dict[TrackKey, _AttrTriple]:
    """Last ``rowid`` wins. A track with no row stays unknown."""
    found: dict[TrackKey, _AttrTriple] = {}
    for item in tracklets.values():
        rows = store.read(item.camera_id, item.track_id, item.run_id)
        if not rows:
            continue
        found[(item.camera_id, item.track_id)] = _triple(rows[-1])
    return found


def _attributes_from_mapping(
    attributes: Mapping[object, object],
) -> dict[TrackKey, _AttrTriple]:
    found: dict[TrackKey, _AttrTriple] = {}
    for key, record in attributes.items():
        found[_track_key(key)] = _triple(record)
    return found


def _with_vectors(
    indexed: Mapping[TrackKey, Tracklet],
    vector_table: Mapping[TrackKey, tuple[float, ...]],
) -> dict[TrackKey, Tracklet]:
    return {key: item for key, item in indexed.items() if key in vector_table}


def _group_by_camera(ready: Mapping[TrackKey, Tracklet]) -> dict[str, tuple[Tracklet, ...]]:
    buckets: dict[str, list[Tracklet]] = {}
    for item in ready.values():
        buckets.setdefault(item.camera_id, []).append(item)
    return {camera_id: _ordered(items) for camera_id, items in buckets.items()}


def _ordered(items: Sequence[Tracklet]) -> tuple[Tracklet, ...]:
    return tuple(sorted(items, key=_track_order))


def _track_order(item: Tracklet) -> tuple[str, int]:
    return (item.camera_id, item.track_id)


def _link_order(link: Link) -> tuple[str, int, str, int]:
    return (link.from_camera, link.from_track, link.to_camera, link.to_track)


def _links_across(
    grouped: Mapping[str, tuple[Tracklet, ...]],
    topology: CameraTopology,
    vectors: Mapping[TrackKey, tuple[float, ...]],
    attributes: Mapping[TrackKey, _AttrTriple],
    min_score: float,
    attr_weight: float,
) -> list[Link]:
    found: list[Link] = []
    for origin in sorted(grouped):
        found.extend(
            _links_from_origin(
                origin,
                grouped,
                topology,
                vectors,
                attributes,
                min_score,
                attr_weight,
            )
        )
    return found


def _links_from_origin(
    origin: str,
    grouped: Mapping[str, tuple[Tracklet, ...]],
    topology: CameraTopology,
    vectors: Mapping[TrackKey, tuple[float, ...]],
    attributes: Mapping[TrackKey, _AttrTriple],
    min_score: float,
    attr_weight: float,
) -> list[Link]:
    found: list[Link] = []
    exits = grouped[origin]
    for dest in topology.successors(origin):
        entries = grouped.get(dest)
        if entries is None:
            continue
        found.extend(
            _links_for_edge(
                exits,
                entries,
                vectors,
                attributes,
                topology,
                min_score,
                attr_weight,
            )
        )
    return found


def _links_for_edge(
    exits: Sequence[Tracklet],
    entries: Sequence[Tracklet],
    vectors: Mapping[TrackKey, tuple[float, ...]],
    attributes: Mapping[TrackKey, _AttrTriple],
    topology: CameraTopology,
    min_score: float,
    attr_weight: float,
) -> list[Link]:
    scores = _score_grid(exits, entries, vectors, attributes, topology, attr_weight)
    chosen = _assign(_cost_matrix(scores))
    return _keep_assigned(exits, entries, scores, chosen, min_score)


def _score_grid(
    exits: Sequence[Tracklet],
    entries: Sequence[Tracklet],
    vectors: Mapping[TrackKey, tuple[float, ...]],
    attributes: Mapping[TrackKey, _AttrTriple],
    topology: CameraTopology,
    attr_weight: float,
) -> list[list[float | None]]:
    grid: list[list[float | None]] = []
    for exit_item in exits:
        row = [
            _pair_score(exit_item, entry_item, vectors, attributes, topology, attr_weight)
            for entry_item in entries
        ]
        grid.append(row)
    return grid


def _pair_score(
    exit_item: Tracklet,
    entry_item: Tracklet,
    vectors: Mapping[TrackKey, tuple[float, ...]],
    attributes: Mapping[TrackKey, _AttrTriple],
    topology: CameraTopology,
    attr_weight: float,
) -> float | None:
    if not topology.allows(
        exit_item.camera_id, exit_item.t_end, entry_item.camera_id, entry_item.t_start
    ):
        return None
    exit_key = (exit_item.camera_id, exit_item.track_id)
    entry_key = (entry_item.camera_id, entry_item.track_id)
    cosine = _cosine(vectors[exit_key], vectors[entry_key])
    share = _share(attributes.get(exit_key), attributes.get(entry_key))
    return _map_score(cosine, share, attr_weight)


def _map_score(cosine: float, share: float, attr_weight: float) -> float:
    """Map cosine and share into ``[0, 1]`` for ``build_route``."""
    bounded = min(1.0, max(-1.0, cosine))
    return ((bounded + 1.0) / 2.0 + attr_weight * share) / (1.0 + attr_weight)


def _cost_matrix(scores: Sequence[Sequence[float | None]]) -> npt.NDArray[np.float64]:
    """Hungarian cost is ``-score`` plus a tie-break, including blocked cells.

    The product term prefers the identity permutation when the score sums
    match. The row term prefers the smaller track id when a row is left
    over. Blocked cells carry the same penalty so an open column stays with
    the smaller id.
    """
    row_count = len(scores)
    column_count = len(scores[0]) if row_count else 0
    cost = np.zeros((row_count, column_count), dtype=np.float64)
    for row, row_scores in enumerate(scores):
        for column, score in enumerate(row_scores):
            penalty = _tie_penalty(row, column, column_count)
            cost[row, column] = _cell_cost(score, penalty)
    return cast(npt.NDArray[np.float64], cost)


def _tie_penalty(row: int, column: int, column_count: int) -> float:
    column_term = (column + 1) / (column_count + 1)
    return column_term * (_TIE_BREAK_UNIT ** (row + 1)) + _ROW_TIE_WEIGHT * row


def _cell_cost(score: float | None, penalty: float) -> float:
    if score is None:
        return _BLOCKED + penalty
    return -score + penalty


def _assign(cost: npt.NDArray[np.float64]) -> list[tuple[int, int]]:
    """Hungarian assignment. A matrix with no rows or columns assigns nothing.

    The import is at module scope. ``importlib`` would hide it from the
    project-rules and semgrep fences. scipy 1.17.1 ships no ``py.typed``
    marker, so the import line ignores ``import-untyped``.
    """
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        return []
    row_index, column_index = linear_sum_assignment(cost)
    rows = [int(index) for index in row_index.tolist()]
    columns = [int(index) for index in column_index.tolist()]
    return list(zip(rows, columns, strict=True))


def _keep_assigned(
    exits: Sequence[Tracklet],
    entries: Sequence[Tracklet],
    scores: Sequence[Sequence[float | None]],
    chosen: Sequence[tuple[int, int]],
    min_score: float,
) -> list[Link]:
    links: list[Link] = []
    for row, column in chosen:
        score = scores[row][column]
        if score is None or score < min_score:
            continue
        links.append(_link_between(exits[row], entries[column], score))
    return links


def _link_between(exit_item: Tracklet, entry_item: Tracklet, score: float) -> Link:
    return _checked(
        Link(
            exit_item.camera_id,
            exit_item.track_id,
            entry_item.camera_id,
            entry_item.track_id,
            entry_item.t_start - exit_item.t_end,
            score,
        )
    )


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError(f"vectors must have the same length, got {len(left)} and {len(right)}")
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for first, second in zip(left, right, strict=True):
        dot += first * second
        left_sq += first * first
        right_sq += second * second
    return dot / math.sqrt(left_sq * right_sq)


def _share(left: _AttrTriple | None, right: _AttrTriple | None) -> float:
    if left is None or right is None:
        return 0.0
    agreed, known = _agreement(left, right)
    if known == 0:
        return 0.0
    return agreed / known


def _agreement(left: _AttrTriple, right: _AttrTriple) -> tuple[int, int]:
    agreed = 0
    known = 0
    for first, second in zip(left, right, strict=True):
        if first is None or second is None:
            continue
        known += 1
        if first == second:
            agreed += 1
    return agreed, known


def _triple(record: object) -> _AttrTriple:
    upper = _optional_colour(_field(record, _UPPER_KEYS))
    lower = _optional_colour(_field(record, _LOWER_KEYS))
    bag = _optional_bag(_field(record, _BAG_KEYS))
    return upper, lower, bag


def _field(record: object, keys: tuple[str, ...]) -> object:
    if isinstance(record, Mapping):
        for key in keys:
            if key in record:
                return record[key]
        return None
    for key in keys:
        if hasattr(record, key):
            return getattr(record, key)
    return None


def _optional_colour(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Color):
        if value is Color.UNKNOWN:
            return None
        return str(value.value)
    if isinstance(value, str):
        if value == Color.UNKNOWN.value:
            return None
        return value
    raise TypeError(f"colour must be a Color, a string, or None, got {value!r}")


def _optional_bag(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise TypeError(f"bag must be a bool or None, got {value!r}")


def _as_vector(value: object) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"vector must be a sequence of finite floats, got {value!r}")
    numbers = tuple(_finite_component(item) for item in value)
    if len(numbers) == 0:
        raise ValueError("vector must be non-empty")
    if _squared_length(numbers) == 0.0:
        raise ValueError("vector must be non-zero")
    return numbers


def _finite_component(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"vector component must be a finite float, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"vector component must be finite, got {number}")
    return number


def _squared_length(values: tuple[float, ...]) -> float:
    total = 0.0
    for value in values:
        total += value * value
    return total


def _track_key(value: object) -> TrackKey:
    if isinstance(value, tuple) and len(value) == 2:
        return _require_id("camera_id", value[0]), _require_track_id("track_id", value[1])
    raise ValueError(f"track key must be (camera_id, track_id), got {value!r}")


def _require_track_id(label: str, value: object) -> int:
    """Return an int track id. Strings are not parsed. Bools are not ids."""
    if isinstance(value, bool) or not isinstance(value, int):
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


def _require_non_negative(label: str, value: object) -> float:
    number = _require_finite(label, value)
    if number < 0.0:
        raise ValueError(f"{label} must be >= 0, got {number}")
    return number


def _require_unit_interval(label: str, value: object) -> float:
    """A link score is a finite float in ``[0, 1]``. Bools are not scores."""
    number = _require_finite(label, value)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must be in [0, 1], got {number}")
    return number
