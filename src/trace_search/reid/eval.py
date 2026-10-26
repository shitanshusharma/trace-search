"""Rank-1 and mAP for cross-camera appearance vectors.

Gallery items with the same label and the same ``camera_id`` are junk
and are excluded. A query with no remaining match is invalid: it counts
in ``num_invalid_query`` and is left out of CMC and mAP. Ties keep
gallery order. ``--embedder`` is ``placeholder`` (the default) or
``osnet-lite``. ``osnet-lite`` is random init only. The JSON and the
human-readable report carry that embedder's label. The JSON evidence
flag is always false. These numbers are not an AC-06 result.

The synthetic query and gallery set is one function, ``synthetic_evaluation_split``.
It builds flat-colour blocks in memory. The command loads them through
the ReID loader before any embedding. An empty allowlist, an empty
split, or a split whose every query is invalid fails closed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

import trace_search.reid.data as data_module
from trace_search.fixtures.guards import FixtureSeparationError, LoaderContext
from trace_search.reid.osnet_lite import OsnetLiteEmbedder
from trace_search.reid.placeholder import UntrainedPlaceholderEmbedder
from trace_search.reid.synthetic import make_synthetic_reid_set
from trace_search.reid.types import AppearanceEmbedder

ReidSample = data_module.ReidSample

_NOTE = "Placeholder or synthetic embeddings. Not an AC-06 result."
_PLACEHOLDER = "placeholder"
_OSNET_LITE = "osnet-lite"
_OSNET_LITE_LABEL = "OSNet-style, chained streams, random init"
_EMBEDDER_CHOICES = (_PLACEHOLDER, _OSNET_LITE)
_DISTANCE = "cosine"
_CMC_RANKS = 5
_JSON_REQUIRED = "ReID evaluation refused: --json is required. No metric was computed."
_CLOSED_ALLOWLIST = (
    "ReID evaluation refused: the id allowlist is empty and no synthetic "
    "evaluation samples are available. No metric was computed."
)
_CLOSED_EMPTY = (
    "ReID evaluation refused: the query set and the gallery set must both "
    "be non-empty. No metric was computed."
)
_CLOSED_INVALID = (
    "ReID evaluation refused: every query is invalid because it has no "
    "gallery match on another camera. No metric was computed."
)
_DESCRIPTION = (
    "Print Rank-1 and mAP as JSON. "
    "Samples are loaded through the ReID loader before any embedding. "
    "The manifest is the repo licenses/manifest.json. "
    "There is no data directory and no file input. "
    "The JSON is not an AC-06 result."
)

_Prepared = tuple[tuple["EvalItem", ...], tuple["EvalItem", ...], AppearanceEmbedder]


@dataclass(frozen=True)
class EvalItem:
    """One query or gallery vector.

    ``label`` is an anonymous same-appearance group index. It is not a
    person id. ``track_id`` is unique per camera only.
    """

    camera_id: str
    track_id: int
    label: int
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.camera_id == "":
            raise ValueError("camera_id must be a non-empty string")
        object.__setattr__(self, "vector", _copy_vector(self.vector))


@dataclass(frozen=True)
class ReidEvalResult:
    """Rank-1, mAP, and the counts behind them.

    ``mean_ap`` is written as ``mAP`` in :meth:`as_json`. ``num_invalid_query``
    is not part of that JSON object.
    """

    rank1: float
    mean_ap: float
    cmc_top5: tuple[float, float, float, float, float]
    num_query: int
    num_valid_query: int
    num_invalid_query: int
    num_gallery: int

    def as_json(self) -> dict[str, object]:
        """Return the spec JSON object. ``evidence`` is always false."""
        return {
            "rank1": self.rank1,
            "mAP": self.mean_ap,
            "cmc_top5": [float(value) for value in self.cmc_top5],
            "num_query": self.num_query,
            "num_valid_query": self.num_valid_query,
            "num_gallery": self.num_gallery,
            "evidence": _evidence_flag(),
            "note": _result_note(),
        }


@dataclass(frozen=True)
class _QueryScore:
    ap: float
    hits: tuple[bool, ...]


def rank1_map(
    query: Sequence[EvalItem],
    gallery: Sequence[EvalItem],
    distance: str = _DISTANCE,
) -> ReidEvalResult:
    """Score ``query`` against ``gallery`` with the cross-camera protocol.

    ``distance`` must be ``cosine``. Same-label same-camera gallery items
    are junk. A query with no match on another camera is invalid and is
    left out of CMC and mAP. Equal distances keep gallery order.
    """
    _require_distance(distance)
    scores = [_rank_query(item, gallery) for item in query]
    valid = [score for score in scores if score is not None]
    invalid = len(query) - len(valid)
    cmc = _mean_cmc(valid)
    return ReidEvalResult(
        rank1=cmc[0],
        mean_ap=_mean_ap(valid),
        cmc_top5=cmc,
        num_query=len(query),
        num_valid_query=len(valid),
        num_invalid_query=invalid,
        num_gallery=len(gallery),
    )


def synthetic_evaluation_split(
    seed: int,
) -> tuple[tuple[ReidSample, ...], tuple[ReidSample, ...]]:
    """Return seeded query samples and gallery samples.

    Query samples are the ``cam-0`` rows of the in-memory set. Gallery
    samples are the other cameras. The set is not written to disk.
    """
    _require_seed(seed)
    samples = make_synthetic_reid_set(seed=seed)
    query = tuple(sample for sample in samples if sample.camera_id == "cam-0")
    gallery = tuple(sample for sample in samples if sample.camera_id != "cam-0")
    return query, gallery


def pinned_manifest_path() -> Path:
    """The repo ``licenses/manifest.json``. Callers cannot pass another path."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "licenses" / "manifest.json"


def build_embedder(embedder: str, seed: int) -> AppearanceEmbedder:
    """Return one random-init embedder. Any other name is refused.

    ``osnet-lite`` is ``OsnetLiteEmbedder(seed=...)`` only.
    """
    _require_seed(seed)
    if embedder == _PLACEHOLDER:
        return UntrainedPlaceholderEmbedder(seed=seed)
    if embedder == _OSNET_LITE:
        return OsnetLiteEmbedder(seed=seed)
    raise ValueError(f"embedder {embedder!r} is not available")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation command. Return a process status code."""
    args = _parse_args(argv)
    if args.emit_json is not True:
        print(_JSON_REQUIRED, file=sys.stderr)
        return 2
    outcome = _prepare(str(args.embedder), int(args.synthetic_seed))
    if isinstance(outcome, str):
        print(outcome, file=sys.stderr)
        return 1
    query, gallery, embedder = outcome
    print(json.dumps(_labelled_json(rank1_map(query, gallery).as_json(), embedder)))
    return 0


def _label_for(embedder: AppearanceEmbedder) -> str:
    """Label for the JSON and the report. osnet-lite is not paper OSNet."""
    if isinstance(embedder, OsnetLiteEmbedder):
        return _OSNET_LITE_LABEL
    return embedder.name


def _format_report(payload: dict[str, object]) -> str:
    """Human-readable lines. The first line is the embedder label."""
    lines = (
        f"embedder: {payload['embedder']}",
        f"rank1: {payload['rank1']}",
        f"mAP: {payload['mAP']}",
        f"cmc_top5: {payload['cmc_top5']}",
        f"num_query: {payload['num_query']}",
        f"num_valid_query: {payload['num_valid_query']}",
        f"num_gallery: {payload['num_gallery']}",
        f"evidence: {json.dumps(payload['evidence'])}",
        f"note: {payload['note']}",
    )
    return "\n".join(lines)


def _labelled_json(
    payload: dict[str, object],
    embedder: AppearanceEmbedder,
) -> dict[str, object]:
    """Copy ``payload`` and attach the embedder label and its report."""
    labelled = dict(payload)
    labelled["embedder"] = _label_for(embedder)
    labelled["report"] = _format_report(labelled)
    return labelled


def _evidence_flag() -> bool:
    """Evaluation JSON is never evidence for AC-06."""
    return False


def _result_note() -> str:
    return _NOTE


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _parser()
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(list(argv))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("--embedder", choices=_EMBEDDER_CHOICES, default=_PLACEHOLDER)
    parser.add_argument("--synthetic-seed", type=int, required=True)
    parser.add_argument("--json", action="store_true", dest="emit_json")
    return parser


def _prepare(embedder_name: str, seed: int) -> _Prepared | str:
    try:
        _require_seed(seed)
        manifest = _require_manifest(pinned_manifest_path())
        loaded_query, loaded_gallery = _load_split(seed, manifest)
    except (FixtureSeparationError, ValueError) as exc:
        return str(exc)
    closed = _closed_reason(loaded_query, loaded_gallery)
    if closed is not None:
        return closed
    try:
        embedder = build_embedder(embedder_name, seed)
        query_items = _items_from_loaded(loaded_query, embedder)
        gallery_items = _items_from_loaded(loaded_gallery, embedder)
    except (FixtureSeparationError, ValueError) as exc:
        return str(exc)
    return query_items, gallery_items, embedder


def _load_split(
    seed: int,
    manifest: Path,
) -> tuple[list[ReidSample], list[ReidSample]]:
    """Load the seeded split. Embedding has not started."""
    query_samples, gallery_samples = synthetic_evaluation_split(seed)
    loaded_query = data_module.load_reid_samples(
        query_samples,
        LoaderContext.EVALUATION,
        manifest,
    )
    loaded_gallery = data_module.load_reid_samples(
        gallery_samples,
        LoaderContext.EVALUATION,
        manifest,
    )
    return loaded_query, loaded_gallery


def _require_manifest(path: Path) -> Path:
    if path.is_file():
        return path
    raise FixtureSeparationError(f"pinned manifest is missing: {path}")


def _require_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"synthetic seed must be a non-negative int, got {seed!r}")


def _closed_reason(
    query: Sequence[ReidSample],
    gallery: Sequence[ReidSample],
) -> str | None:
    if len(data_module._EVALUATION_ID_PATTERNS) == 0:
        return _CLOSED_ALLOWLIST
    if len(query) == 0 or len(gallery) == 0:
        return _CLOSED_EMPTY
    if not _any_valid_query(query, gallery):
        return _CLOSED_INVALID
    return None


def _any_valid_query(
    query: Sequence[ReidSample],
    gallery: Sequence[ReidSample],
) -> bool:
    """True when some query has the same label on another camera."""
    for sample in query:
        if _has_cross_camera_match(sample, gallery):
            return True
    return False


def _has_cross_camera_match(query: ReidSample, gallery: Sequence[ReidSample]) -> bool:
    for item in gallery:
        if item.label == query.label and item.camera_id != query.camera_id:
            return True
    return False


def _items_from_loaded(
    samples: Sequence[ReidSample],
    embedder: AppearanceEmbedder,
) -> tuple[EvalItem, ...]:
    """Embed crops from ``samples`` only. ``samples`` must come from the loader."""
    crops = _crops_from_loaded(samples)
    vectors = embedder.embed(crops)
    return _items_from_vectors(samples, vectors)


def _crops_from_loaded(samples: Sequence[ReidSample]) -> list[npt.NDArray[np.uint8]]:
    crops: list[npt.NDArray[np.uint8]] = []
    for sample in samples:
        crop = sample.crop
        if crop is None:
            raise FixtureSeparationError(
                f"{sample.fixture_id} is refused: the evaluation embed step "
                "only accepts a crop returned by the loader"
            )
        crops.append(crop)
    return crops


def _items_from_vectors(
    samples: Sequence[ReidSample],
    vectors: npt.NDArray[np.float32],
) -> tuple[EvalItem, ...]:
    _require_rows(vectors, len(samples))
    items: list[EvalItem] = []
    for index, sample in enumerate(samples):
        row = np.asarray(vectors[index], dtype=np.float64)
        values = tuple(float(value) for value in row)
        items.append(
            EvalItem(
                camera_id=sample.camera_id,
                track_id=sample.track_id,
                label=sample.label,
                vector=values,
            )
        )
    return tuple(items)


def _require_rows(vectors: npt.NDArray[np.float32], count: int) -> None:
    if vectors.ndim == 2 and int(vectors.shape[0]) == count:
        return
    raise ValueError(f"embedder returned shape {vectors.shape} for {count} samples")


def _require_distance(distance: str) -> None:
    if distance == _DISTANCE:
        return
    raise ValueError(f"distance must be {_DISTANCE!r}, got {distance!r}")


def _rank_query(query: EvalItem, gallery: Sequence[EvalItem]) -> _QueryScore | None:
    kept = tuple(item for item in gallery if not _is_junk(query, item))
    if not _has_match(query, kept):
        return None
    ordered = _order_by_distance(query, kept)
    return _score_ordered(query, ordered)


def _is_junk(query: EvalItem, item: EvalItem) -> bool:
    """Same label and same camera. Those gallery items are excluded."""
    return item.label == query.label and item.camera_id == query.camera_id


def _has_match(query: EvalItem, gallery: Sequence[EvalItem]) -> bool:
    for item in gallery:
        if item.label == query.label:
            return True
    return False


def _order_by_distance(query: EvalItem, gallery: Sequence[EvalItem]) -> tuple[EvalItem, ...]:
    distances = _cosine_distances(query.vector, gallery)
    order = _stable_nearest_first(distances)
    return tuple(gallery[index] for index in order)


def _stable_nearest_first(distances: npt.NDArray[np.float64]) -> tuple[int, ...]:
    """Lowest distance first. Equal distances keep the incoming order."""
    order = np.argsort(distances, kind="stable")
    return tuple(int(index) for index in order)


def _cosine_distances(
    vector: tuple[float, ...],
    gallery: Sequence[EvalItem],
) -> npt.NDArray[np.float64]:
    query = np.asarray(vector, dtype=np.float64)
    matrix = _gallery_matrix(gallery)
    width = int(matrix.shape[1])
    if width != int(query.shape[0]):
        raise ValueError(
            "query and gallery vectors must have the same length, "
            f"got {int(query.shape[0])} and {width}"
        )
    query_norm = float(np.linalg.norm(query))
    gallery_norms = np.linalg.norm(matrix, axis=1)
    dots = matrix @ query
    distances = 1.0 - (dots / (gallery_norms * query_norm))
    return cast(npt.NDArray[np.float64], np.asarray(distances, dtype=np.float64))


def _gallery_matrix(gallery: Sequence[EvalItem]) -> npt.NDArray[np.float64]:
    rows = [np.asarray(item.vector, dtype=np.float64) for item in gallery]
    width = int(rows[0].shape[0])
    for row in rows:
        if int(row.shape[0]) != width:
            raise ValueError("gallery vectors must have the same length")
    stacked = np.stack(rows)
    return cast(npt.NDArray[np.float64], np.asarray(stacked, dtype=np.float64))


def _score_ordered(query: EvalItem, ordered: Sequence[EvalItem]) -> _QueryScore:
    matches = tuple(item.label == query.label for item in ordered)
    return _QueryScore(ap=_average_precision(matches), hits=_cumulative_hits(matches))


def _average_precision(matches: Sequence[bool]) -> float:
    """Mean of precision at each relevant rank. Ranks are 1-based."""
    relevant = 0
    total = 0.0
    for index, is_match in enumerate(matches, start=1):
        if not is_match:
            continue
        relevant += 1
        total += relevant / index
    return total / relevant


def _cumulative_hits(matches: Sequence[bool]) -> tuple[bool, ...]:
    seen = False
    hits: list[bool] = []
    for is_match in matches:
        if is_match:
            seen = True
        hits.append(seen)
    return tuple(hits)


def _mean_ap(scores: Sequence[_QueryScore]) -> float:
    if len(scores) == 0:
        return 0.0
    return sum(score.ap for score in scores) / len(scores)


def _mean_cmc(scores: Sequence[_QueryScore]) -> tuple[float, float, float, float, float]:
    if len(scores) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    rates = [_hit_rate(scores, rank) for rank in range(1, _CMC_RANKS + 1)]
    return (rates[0], rates[1], rates[2], rates[3], rates[4])


def _hit_rate(scores: Sequence[_QueryScore], rank: int) -> float:
    hits = sum(1 for score in scores if _hit_at(score.hits, rank))
    return hits / len(scores)


def _hit_at(hits: Sequence[bool], rank: int) -> bool:
    if rank <= len(hits):
        return hits[rank - 1]
    return hits[-1]


def _copy_vector(vector: Sequence[float]) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes)):
        raise ValueError("vector must be a sequence of finite floats")
    values = tuple(_finite_float(item) for item in vector)
    if len(values) == 0:
        raise ValueError("vector must be non-empty")
    if _squared_norm(values) == 0.0:
        raise ValueError("vector must be non-zero")
    return values


def _finite_float(item: float) -> float:
    value = float(item)
    if math.isfinite(value):
        return value
    raise ValueError(f"vector values must be finite, got {value!r}")


def _squared_norm(values: tuple[float, ...]) -> float:
    total = 0.0
    for value in values:
        total += value * value
    return total
