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
dropped. Equal totals break on ``(camera_id, track_id)``. The Hungarian
step uses ``-score`` with no tie-break penalty. Two totals are equal when
``math.isclose`` says so with ``rel_tol`` 0 and ``abs_tol`` ``1e-12`` times
the longer side of the grid. Exits are walked in ``(camera_id, track_id)``
order, and the first entry in that order that still lets the remaining
rows reach the optimal total is fixed. When every cell has the same score,
that pairing is the diagonal of the sorted rows and columns, so a large
tied grid does not re-solve.

``MIN_SCORE`` of 0 keeps every mapped score, including 0. Pairs are dropped
only when the score is strictly under the floor, and the map cannot be
negative.

A pair whose tracklets differ in ``run_id`` or ``det_class`` is skipped.
The other pairs in the call are still linked. A person track is not linked
to a handbag track.

When ``vectors`` or ``attributes`` is a store, ``run_id`` and
``derived_from`` are required. The read uses only that run and that
``derived_from``. A mapping does not require them.

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

# Placeholder floor. Pairs are dropped only when score < MIN_SCORE.
# The mapped score is in [0, 1], so 0 keeps every mapped score, including 0.
# The value was not tuned against any stock clip.
MIN_SCORE = 0.0

# Placeholder weight of the known-attribute share. The value was not tuned
# against any stock clip.
ATTR_WEIGHT = 1.0

# Absolute tolerance per row of the longer grid side. Float64 sums of
# values near 1 drift by a few ulps per add. 1e-12 per row sits above that
# drift and below a 1e-5 score gap. Not a similarity threshold.
_TOTAL_ABS_TOL_PER_ROW = 1e-12

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
        *,
        run_id: str | None = ...,
        derived_from: str | None = ...,
    ) -> Sequence[tuple[AppearanceEmbedding, float]]:
        """Return up to ``k`` rows, cosine highest first.

        ``run_id`` and ``derived_from`` limit the rows. ``None`` keeps
        every value in that column.
        """


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
    run_id: str | None = None,
    derived_from: str | None = None,
) -> list[Link]:
    """Link exits to entries across each directed edge.

    ``vectors`` is a mapping of ``(camera_id, track_id)`` to a placeholder
    vector, or a vector store with ``dim``, ``count``, and ``search``.
    ``attributes`` is a mapping of the same key to a record, or a
    ``TrackAttributeStore``. A track with no vector is left out.
    ``min_score`` and ``attr_weight`` are keyword-only. A store read also
    requires ``run_id`` and ``derived_from``, and uses only those rows.
    """
    floor = _require_non_negative("min_score", min_score)
    weight = _require_non_negative("attr_weight", attr_weight)
    scope_run, scope_origin = _call_scope(vectors, attributes, run_id, derived_from)
    indexed = _index_tracklets(tracklets)
    if scope_run is not None:
        indexed = _tracklets_in_run(indexed, scope_run)
    vector_table = _vector_table(vectors, scope_run, scope_origin)
    attribute_table = _attribute_table(attributes, indexed, scope_run, scope_origin)
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

    The caller's time_of maps a query item to its EXIT time and a gallery
    item to its ENTRY time. A false result excludes that gallery item for
    that query. The mask is not evidence for AC-06.
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


def _call_scope(
    vectors: object,
    attributes: object,
    run_id: str | None,
    derived_from: str | None,
) -> tuple[str | None, str | None]:
    """Require both scope fields when either argument is a store."""
    run = _optional_scope_text("run_id", run_id)
    origin = _optional_scope_text("derived_from", derived_from)
    if _reads_store(vectors, attributes) and (run is None or origin is None):
        raise ValueError("run_id and derived_from are required when reading a store")
    return run, origin


def _reads_store(vectors: object, attributes: object) -> bool:
    if isinstance(attributes, TrackAttributeStore):
        return True
    if isinstance(vectors, Mapping):
        return False
    return _is_reader(vectors)


def _optional_scope_text(label: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    return value


def _tracklets_in_run(
    indexed: Mapping[TrackKey, Tracklet],
    run_id: str,
) -> dict[TrackKey, Tracklet]:
    return {key: item for key, item in indexed.items() if item.run_id == run_id}


def _vector_table(
    vectors: object,
    run_id: str | None,
    derived_from: str | None,
) -> dict[TrackKey, tuple[float, ...]]:
    if isinstance(vectors, Mapping):
        return _vectors_from_mapping(vectors)
    reader = _as_reader(vectors)
    if run_id is None or derived_from is None:
        raise ValueError("run_id and derived_from are required when reading a store")
    return _vectors_from_store(reader, run_id, derived_from)


def _as_reader(vectors: object) -> _VectorReader:
    if not _is_reader(vectors):
        raise TypeError("vectors must be a mapping or a vector store")
    return cast(_VectorReader, vectors)


def _is_reader(vectors: object) -> bool:
    dim = getattr(vectors, "dim", None)
    search = getattr(vectors, "search", None)
    count = getattr(vectors, "count", None)
    if isinstance(dim, bool) or not isinstance(dim, int):
        return False
    return callable(search) and callable(count)


def _vectors_from_mapping(vectors: Mapping[object, object]) -> dict[TrackKey, tuple[float, ...]]:
    table: dict[TrackKey, tuple[float, ...]] = {}
    for key, value in vectors.items():
        table[_track_key(key)] = _as_vector(value)
    return table


def _vectors_from_store(
    reader: _VectorReader,
    run_id: str,
    derived_from: str,
) -> dict[TrackKey, tuple[float, ...]]:
    """One vector per track inside ``run_id`` and ``derived_from``.

    Lowest frame wins, then the smaller vector tuple. ``k`` must be
    positive, so an empty scoped store is returned without a search.
    """
    total = reader.count(derived_from=derived_from, run_id=run_id)
    if total == 0:
        return {}
    chosen: dict[TrackKey, tuple[int, tuple[float, ...]]] = {}
    hits = reader.search(
        _axis_probe(reader.dim),
        total,
        run_id=run_id,
        derived_from=derived_from,
    )
    for embedding, _cosine in hits:
        _keep_earliest(chosen, embedding)
    return {key: vector for key, (_frame, vector) in chosen.items()}


def _axis_probe(dim: int) -> tuple[float, ...]:
    return tuple(1.0 if index == 0 else 0.0 for index in range(dim))


def _keep_earliest(
    chosen: dict[TrackKey, tuple[int, tuple[float, ...]]],
    embedding: AppearanceEmbedding,
) -> None:
    key = (embedding.camera_id, embedding.track_id)
    vector = _as_vector(tuple(embedding.vector))
    candidate = (embedding.frame_number, vector)
    current = chosen.get(key)
    if current is None or candidate < current:
        chosen[key] = candidate


def _attribute_table(
    attributes: object,
    tracklets: Mapping[TrackKey, Tracklet],
    run_id: str | None,
    derived_from: str | None,
) -> dict[TrackKey, _AttrTriple]:
    if isinstance(attributes, TrackAttributeStore):
        if run_id is None or derived_from is None:
            raise ValueError("run_id and derived_from are required when reading a store")
        return _attributes_from_store(attributes, tracklets, run_id, derived_from)
    if isinstance(attributes, Mapping):
        return _attributes_from_mapping(attributes)
    raise TypeError("attributes must be a mapping or a TrackAttributeStore")


def _attributes_from_store(
    store: TrackAttributeStore,
    tracklets: Mapping[TrackKey, Tracklet],
    run_id: str,
    derived_from: str,
) -> dict[TrackKey, _AttrTriple]:
    """Last matching ``rowid`` wins. Other runs and origins are ignored."""
    found: dict[TrackKey, _AttrTriple] = {}
    for item in tracklets.values():
        rows = store.read(item.camera_id, item.track_id, run_id)
        scoped = [row for row in rows if row.derived_from == derived_from]
        if not scoped:
            continue
        found[(item.camera_id, item.track_id)] = _triple(scoped[-1])
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
    if not _same_run_and_class(exit_item, entry_item):
        return None
    if not topology.allows(
        exit_item.camera_id, exit_item.t_end, entry_item.camera_id, entry_item.t_start
    ):
        return None
    exit_key = (exit_item.camera_id, exit_item.track_id)
    entry_key = (entry_item.camera_id, entry_item.track_id)
    cosine = _cosine(vectors[exit_key], vectors[entry_key])
    share = _share(attributes.get(exit_key), attributes.get(entry_key))
    return _map_score(cosine, share, attr_weight)


def _same_run_and_class(exit_item: Tracklet, entry_item: Tracklet) -> bool:
    """Skip a cross-run pair and a pair of different ``det_class`` values."""
    if exit_item.run_id != entry_item.run_id:
        return False
    return exit_item.det_class == entry_item.det_class


def _map_score(cosine: float, share: float, attr_weight: float) -> float:
    """Map cosine and share into ``[0, 1]`` for ``build_route``."""
    bounded = min(1.0, max(-1.0, cosine))
    return ((bounded + 1.0) / 2.0 + attr_weight * share) / (1.0 + attr_weight)


def _cost_matrix(scores: Sequence[Sequence[float | None]]) -> npt.NDArray[np.float64]:
    """Hungarian cost is ``-score``. A missing pair is blocked. No tie penalty."""
    row_count = len(scores)
    column_count = len(scores[0]) if row_count else 0
    cost = np.zeros((row_count, column_count), dtype=np.float64)
    for row, row_scores in enumerate(scores):
        for column, score in enumerate(row_scores):
            cost[row, column] = _cell_cost(score)
    return cast(npt.NDArray[np.float64], cost)


def _cell_cost(score: float | None) -> float:
    if score is None:
        return _BLOCKED
    return -score


def _assign(cost: npt.NDArray[np.float64]) -> list[tuple[int, int]]:
    """Optimal assignment, then the lexicographic tie on ``(camera_id, track_id)``.

    Rows and columns are already in that order. A grid whose every cell has
    the same score is paired on the diagonal, so a large exact tie does not
    re-solve. Any other tie walks the rows and keeps a column only when the
    remaining subproblem still reaches the optimal total.
    """
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        return []
    if _uniform_scores(cost):
        return _diagonal_pairs(cost)
    return _lex_among_optimal(cost)


def _uniform_scores(cost: npt.NDArray[np.float64]) -> bool:
    """True when every cell is feasible and the costs are identical."""
    if cost.size == 0:
        return True
    first = float(cost[0, 0])
    if not math.isfinite(first) or first >= _BLOCKED:
        return False
    return bool(np.all(np.equal(cost, first)))


def _diagonal_pairs(cost: npt.NDArray[np.float64]) -> list[tuple[int, int]]:
    """Smallest rows with the smallest columns. The sorted order is the diagonal."""
    width = min(int(cost.shape[0]), int(cost.shape[1]))
    return [(index, index) for index in range(width)]


def _lex_among_optimal(cost: npt.NDArray[np.float64]) -> list[tuple[int, int]]:
    seed = _hungarian(cost)
    optimal = _pair_sum(cost, seed)
    return _walk_rows(cost, optimal)


def _walk_rows(cost: npt.NDArray[np.float64], optimal: float) -> list[tuple[int, int]]:
    open_rows = list(range(int(cost.shape[0])))
    open_cols = list(range(int(cost.shape[1])))
    fixed: list[tuple[int, int]] = []
    side = max(int(cost.shape[0]), int(cost.shape[1]))
    for row in range(int(cost.shape[0])):
        open_rows = [item for item in open_rows if item != row]
        column = _first_column(cost, row, open_rows, open_cols, fixed, optimal, side)
        if column is None:
            _require_still_optimal(cost, fixed, open_rows, open_cols, optimal, side)
            continue
        fixed.append((row, column))
        open_cols = [item for item in open_cols if item != column]
    return fixed


def _first_column(
    cost: npt.NDArray[np.float64],
    row: int,
    open_rows: Sequence[int],
    open_cols: Sequence[int],
    fixed: Sequence[tuple[int, int]],
    optimal: float,
    side: int,
) -> int | None:
    for column in open_cols:
        rest = [item for item in open_cols if item != column]
        trial = [*fixed, (row, column)]
        if _reaches(cost, trial, open_rows, rest, optimal, side):
            return column
    return None


def _require_still_optimal(
    cost: npt.NDArray[np.float64],
    fixed: Sequence[tuple[int, int]],
    open_rows: Sequence[int],
    open_cols: Sequence[int],
    optimal: float,
    side: int,
) -> None:
    if _reaches(cost, fixed, open_rows, open_cols, optimal, side):
        return
    raise ValueError("assignment lost the optimal score total")


def _reaches(
    cost: npt.NDArray[np.float64],
    fixed: Sequence[tuple[int, int]],
    open_rows: Sequence[int],
    open_cols: Sequence[int],
    optimal: float,
    side: int,
) -> bool:
    if _too_costly(cost, fixed, open_rows, open_cols, optimal, side):
        return False
    total = _pair_sum(cost, fixed) + _sub_sum(cost, open_rows, open_cols)
    return _totals_equal(total, optimal, side)


def _too_costly(
    cost: npt.NDArray[np.float64],
    fixed: Sequence[tuple[int, int]],
    open_rows: Sequence[int],
    open_cols: Sequence[int],
    optimal: float,
    side: int,
) -> bool:
    """True when even the cheapest completion is above the optimal total."""
    optimistic = _pair_sum(cost, fixed) + _row_min_bound(cost, open_rows, open_cols)
    limit = _TOTAL_ABS_TOL_PER_ROW * max(side, 1)
    return optimistic > optimal + limit


def _row_min_bound(
    cost: npt.NDArray[np.float64],
    rows: Sequence[int],
    cols: Sequence[int],
) -> float:
    """Sum of the best row costs, ignoring column clashes. A lower bound."""
    if not rows or not cols:
        return 0.0
    best = [min(float(cost[row, column]) for column in cols) for row in rows]
    best.sort()
    take = min(len(rows), len(cols))
    return sum(best[:take])


def _sub_sum(
    cost: npt.NDArray[np.float64],
    rows: Sequence[int],
    cols: Sequence[int],
) -> float:
    if not rows or not cols:
        return 0.0
    sub = _submatrix(cost, rows, cols)
    if _uniform_scores(sub):
        width = min(len(rows), len(cols))
        return width * float(sub[0, 0])
    return _pair_sum(sub, _hungarian(sub))


def _submatrix(
    cost: npt.NDArray[np.float64],
    rows: Sequence[int],
    cols: Sequence[int],
) -> npt.NDArray[np.float64]:
    picked = cost[np.ix_(list(rows), list(cols))]
    return cast(npt.NDArray[np.float64], np.asarray(picked, dtype=np.float64))


def _totals_equal(left: float, right: float, side: int) -> bool:
    """Equal within ``1e-12`` times the longer grid side. Not a 1e-4 gap."""
    limit = _TOTAL_ABS_TOL_PER_ROW * max(side, 1)
    return math.isclose(left, right, rel_tol=0.0, abs_tol=limit)


def _pair_sum(cost: npt.NDArray[np.float64], pairs: Sequence[tuple[int, int]]) -> float:
    total = 0.0
    for row, column in pairs:
        total += float(cost[row, column])
    return total


def _hungarian(cost: npt.NDArray[np.float64]) -> list[tuple[int, int]]:
    """One Hungarian solution. Ties are not broken here.

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
