"""AC-05 harness for person-box recall and single-camera IDF1.

The metric functions are real. ``run`` does not produce an AC-05 result in
Cut 01. ``status`` is ``DEFERRED``, ``evidence`` is false, and
``ac05_score`` is null. A fixture id that starts with ``FX-SMOKE`` or
``FX-CODE`` raises ``FixtureSeparationError`` unless ``mode`` is
``unit_test``. Unit-test mode still returns no numeric AC-05 score.

IDF1 uses one global min-cost track-id assignment from
``scipy.optimize.linear_sum_assignment``. Per-frame pairs are greedy,
highest IoU first, and a pair counts only at IoU greater than or equal
to the threshold. This module does not load a model or weights.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from trace_search.attributes.attributes import Box
from trace_search.fixtures.guards import FixtureSeparationError

_BLOCKED_PREFIXES: tuple[str, ...] = ("FX-SMOKE", "FX-CODE")
_UNIT_TEST_MODE = "unit_test"

__all__ = [
    "Ac05RunResult",
    "DetBox",
    "Idf1Result",
    "SubsetRecall",
    "SubsetReport",
    "TrackBox",
    "detection_recall",
    "idf1",
    "run",
]


class SubsetRecall(TypedDict):
    """Recall for one ground-truth flag. Null when ``run`` did not measure."""

    recall: float | None
    matched: int | None
    ground_truth: int | None


class SubsetReport(TypedDict):
    """Low-light and occluded recall, each on its own ground-truth set."""

    low_light: SubsetRecall
    occluded: SubsetRecall


class Idf1Result(TypedDict):
    """IDTP, IDFP, IDFN, and IDF1 for one comparison."""

    idtp: int
    idfp: int
    idfn: int
    idf1: float


class Ac05RunResult(TypedDict):
    """Cut 01 AC-05 payload. ``ac05_score`` is always null."""

    criterion: Literal["AC-05"]
    fixture_id: str
    status: Literal["DEFERRED"]
    evidence: Literal[False]
    ac05_score: None
    recall: float | None
    idf1: float | None
    idtp: int | None
    idfp: int | None
    idfn: int | None
    subset: SubsetReport


@dataclass(frozen=True)
class DetBox:
    """One person box in one frame.

    ``box`` is ``(x1, y1, x2, y2)`` in pixels. ``low_light`` and
    ``occluded`` are ground-truth flags used only for the subset report.
    A box may carry both flags.
    """

    frame: int
    box: Box
    low_light: bool = False
    occluded: bool = False

    def __post_init__(self) -> None:
        _require_frame(self.frame)
        _require_box(self.box)
        _require_flag("low_light", self.low_light)
        _require_flag("occluded", self.occluded)


@dataclass(frozen=True)
class TrackBox:
    """One box of one track in one frame.

    ``track_id`` is a label for the metric. It is not a person id.
    """

    frame: int
    track_id: int
    box: Box

    def __post_init__(self) -> None:
        _require_frame(self.frame)
        _require_track_id(self.track_id)
        _require_box(self.box)


def detection_recall(
    gt: Sequence[DetBox],
    pred: Sequence[DetBox],
    iou: float = 0.5,
) -> float:
    """Return person-box recall at ``iou``.

    Matching is one-to-one inside each frame. Candidate pairs are taken
    in descending IoU, and a pair matches when its IoU is greater than
    or equal to ``iou``. A ground-truth box and a prediction are each
    used at most once. Boxes from different frames are never paired.

    Recall is the number of matched ground-truth boxes divided by the
    ground-truth count. Extra predictions do not change it. When ``gt``
    is empty, recall is 1: there is no miss, whether or not ``pred``
    is empty.
    """
    _check_iou(iou)
    matched = _matched_gt_indices(gt, pred, iou)
    return _recall_ratio(len(matched), len(gt))


def idf1(
    gt_tracks: Sequence[TrackBox],
    pred_tracks: Sequence[TrackBox],
    iou: float = 0.5,
) -> Idf1Result:
    """Return IDTP, IDFP, IDFN, and IDF1.

    Each frame is matched with the same greedy IoU rule as
    ``detection_recall``. Those spatial pairs are then assigned globally:
    the cost of pairing a ground-truth track id with a predicted track id
    is the negative of how many spatial pairs they share.
    ``linear_sum_assignment`` minimizes that cost, which maximizes IDTP.
    Each track id is used at most once.

    IDF1 is ``2 * IDTP / (2 * IDTP + IDFP + IDFN)``.
    IDFN is the ground-truth detection count minus IDTP.
    IDFP is the predicted detection count minus IDTP.

    Empty inputs:

    - both sides empty: IDTP = IDFP = IDFN = 0 and IDF1 = 1
    - ground truth empty and predictions not: IDF1 = 0 and IDFP is the
      prediction count
    - predictions empty and ground truth not: IDF1 = 0 and IDFN is the
      ground-truth count
    """
    _check_iou(iou)
    counts = _count_pairs(_spatial_id_pairs(gt_tracks, pred_tracks, iou))
    idtp = _assigned_idtp(
        _unique_ids(gt_tracks),
        _unique_ids(pred_tracks),
        counts,
    )
    idfn = len(gt_tracks) - idtp
    idfp = len(pred_tracks) - idtp
    return _idf1_result(idtp, idfp, idfn)


def run(
    fixture_id: str,
    gt: Sequence[DetBox] | None = None,
    pred: Sequence[DetBox] | None = None,
    gt_tracks: Sequence[TrackBox] | None = None,
    pred_tracks: Sequence[TrackBox] | None = None,
    iou: float = 0.5,
    mode: str = "eval",
) -> Ac05RunResult:
    """Return a Cut 01 AC-05 payload. The payload is never a pass.

    ``status`` is ``DEFERRED``, ``evidence`` is false, and ``ac05_score``
    is null. Metrics are filled in only for a fixture id that does not
    start with ``FX-SMOKE`` or ``FX-CODE``. Those prefixes raise
    ``FixtureSeparationError`` unless ``mode`` is ``unit_test``.
    Unit-test mode still leaves every score null, including when boxes
    were supplied. Omitted sequences are not measured. An explicit empty
    sequence uses the empty-input rule.
    """
    _require_fixture_id(fixture_id)
    if _blocked_fixture(fixture_id) and mode != _UNIT_TEST_MODE:
        _reject_blocked(fixture_id)
    _check_iou(iou)
    if _blocked_fixture(fixture_id):
        return _unscored(fixture_id)
    return _measured(fixture_id, gt, pred, gt_tracks, pred_tracks, iou)


def _measured(
    fixture_id: str,
    gt: Sequence[DetBox] | None,
    pred: Sequence[DetBox] | None,
    gt_tracks: Sequence[TrackBox] | None,
    pred_tracks: Sequence[TrackBox] | None,
    iou: float,
) -> Ac05RunResult:
    recall, subset = _recall_and_subset(gt, pred, iou)
    scores = _track_scores(gt_tracks, pred_tracks, iou)
    return _pack(fixture_id, recall, scores, subset)


def _recall_and_subset(
    gt: Sequence[DetBox] | None,
    pred: Sequence[DetBox] | None,
    iou: float,
) -> tuple[float | None, SubsetReport]:
    if gt is None and pred is None:
        return None, _blank_subset()
    gt_boxes = _present_boxes(gt)
    pred_boxes = _present_boxes(pred)
    recall = detection_recall(gt_boxes, pred_boxes, iou)
    return recall, _subset_report(gt_boxes, pred_boxes, iou)


def _track_scores(
    gt_tracks: Sequence[TrackBox] | None,
    pred_tracks: Sequence[TrackBox] | None,
    iou: float,
) -> Idf1Result | None:
    if gt_tracks is None and pred_tracks is None:
        return None
    return idf1(_present_tracks(gt_tracks), _present_tracks(pred_tracks), iou)


def _unscored(fixture_id: str) -> Ac05RunResult:
    return _pack(fixture_id, None, None, _blank_subset())


def _pack(
    fixture_id: str,
    recall: float | None,
    scores: Idf1Result | None,
    subset: SubsetReport,
) -> Ac05RunResult:
    idtp: int | None = None
    idfp: int | None = None
    idfn: int | None = None
    idf1_score: float | None = None
    if scores is not None:
        idtp = scores["idtp"]
        idfp = scores["idfp"]
        idfn = scores["idfn"]
        idf1_score = scores["idf1"]
    return {
        "criterion": "AC-05",
        "fixture_id": fixture_id,
        "status": "DEFERRED",
        "evidence": False,
        "ac05_score": None,
        "recall": recall,
        "idf1": idf1_score,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "subset": subset,
    }


def _subset_report(
    gt: Sequence[DetBox],
    pred: Sequence[DetBox],
    iou: float,
) -> SubsetReport:
    matched = _matched_gt_indices(gt, pred, iou)
    return {
        "low_light": _recall_where(gt, matched, _is_low_light),
        "occluded": _recall_where(gt, matched, _is_occluded),
    }


def _blank_subset() -> SubsetReport:
    return {
        "low_light": _blank_subset_recall(),
        "occluded": _blank_subset_recall(),
    }


def _blank_subset_recall() -> SubsetRecall:
    return {"recall": None, "matched": None, "ground_truth": None}


def _recall_where(
    gt: Sequence[DetBox],
    matched: set[int],
    selected: Callable[[DetBox], bool],
) -> SubsetRecall:
    """Subset recall from the full-frame matching, not a second match."""
    hits = 0
    total = 0
    for index, box in enumerate(gt):
        if not selected(box):
            continue
        total += 1
        if index in matched:
            hits += 1
    return {
        "recall": _recall_ratio(hits, total),
        "matched": hits,
        "ground_truth": total,
    }


def _is_low_light(box: DetBox) -> bool:
    return box.low_light


def _is_occluded(box: DetBox) -> bool:
    return box.occluded


def _matched_gt_indices(
    gt: Sequence[DetBox],
    pred: Sequence[DetBox],
    iou: float,
) -> set[int]:
    gt_frames = _indices_by_frame(gt)
    pred_frames = _indices_by_frame(pred)
    matched: set[int] = set()
    for frame, gt_indexes in gt_frames.items():
        pred_indexes = pred_frames.get(frame, [])
        gt_boxes = [gt[index].box for index in gt_indexes]
        pred_boxes = [pred[index].box for index in pred_indexes]
        for gt_local, _pred_local in _greedy_pairs(gt_boxes, pred_boxes, iou):
            matched.add(gt_indexes[gt_local])
    return matched


def _spatial_id_pairs(
    gt_tracks: Sequence[TrackBox],
    pred_tracks: Sequence[TrackBox],
    iou: float,
) -> list[tuple[int, int]]:
    gt_by_frame = _tracks_by_frame(gt_tracks)
    pred_by_frame = _tracks_by_frame(pred_tracks)
    frames = list(dict.fromkeys([*gt_by_frame, *pred_by_frame]))
    pairs: list[tuple[int, int]] = []
    for frame in frames:
        gt_items = gt_by_frame.get(frame, [])
        pred_items = pred_by_frame.get(frame, [])
        for gt_index, pred_index in _greedy_pairs(
            _boxes_of(gt_items),
            _boxes_of(pred_items),
            iou,
        ):
            pairs.append((gt_items[gt_index].track_id, pred_items[pred_index].track_id))
    return pairs


def _greedy_pairs(
    gt_boxes: Sequence[Box],
    pred_boxes: Sequence[Box],
    iou: float,
) -> list[tuple[int, int]]:
    """One-to-one pairs, highest IoU first, kept only at ``iou`` or above."""
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    chosen: list[tuple[int, int]] = []
    for _overlap, gt_index, pred_index in _candidate_pairs(gt_boxes, pred_boxes, iou):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        chosen.append((gt_index, pred_index))
    return chosen


def _candidate_pairs(
    gt_boxes: Sequence[Box],
    pred_boxes: Sequence[Box],
    iou: float,
) -> list[tuple[float, int, int]]:
    pairs: list[tuple[float, int, int]] = []
    for gt_index, gt_box in enumerate(gt_boxes):
        for pred_index, pred_box in enumerate(pred_boxes):
            overlap = _box_iou(gt_box, pred_box)
            if overlap >= iou:
                pairs.append((overlap, gt_index, pred_index))
    pairs.sort(key=_pair_order)
    return pairs


def _pair_order(pair: tuple[float, int, int]) -> tuple[float, int, int]:
    """Highest IoU, then the earlier ground-truth box, then the earlier prediction."""
    overlap, gt_index, pred_index = pair
    return (-overlap, gt_index, pred_index)


def _box_iou(left: Box, right: Box) -> float:
    """Intersection over union. Boxes that do not overlap score 0."""
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    overlap_w = min(left_x2, right_x2) - max(left_x1, right_x1)
    overlap_h = min(left_y2, right_y2) - max(left_y1, right_y1)
    if overlap_w <= 0.0 or overlap_h <= 0.0:
        return 0.0
    intersection = overlap_w * overlap_h
    left_area = (left_x2 - left_x1) * (left_y2 - left_y1)
    right_area = (right_x2 - right_x1) * (right_y2 - right_y1)
    return intersection / (left_area + right_area - intersection)


def _assigned_idtp(
    gt_ids: Sequence[int],
    pred_ids: Sequence[int],
    counts: Mapping[tuple[int, int], int],
) -> int:
    """IDTP from the min-cost assignment. An empty side contributes 0."""
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0
    row_index, col_index = linear_sum_assignment(_id_cost(gt_ids, pred_ids, counts))
    rows = [int(index) for index in row_index.tolist()]
    cols = [int(index) for index in col_index.tolist()]
    total = 0
    for row, col in zip(rows, cols, strict=True):
        total += counts.get((gt_ids[row], pred_ids[col]), 0)
    return total


def _id_cost(
    gt_ids: Sequence[int],
    pred_ids: Sequence[int],
    counts: Mapping[tuple[int, int], int],
) -> npt.NDArray[np.float64]:
    """Negative spatial-match counts, so min-cost assignment maximizes IDTP."""
    cost = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for row, gt_id in enumerate(gt_ids):
        for col, pred_id in enumerate(pred_ids):
            cost[row, col] = -float(counts.get((gt_id, pred_id), 0))
    return cost


def _idf1_result(idtp: int, idfp: int, idfn: int) -> Idf1Result:
    return {
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "idf1": _idf1_score(idtp, idfp, idfn),
    }


def _idf1_score(idtp: int, idfp: int, idfn: int) -> float:
    """IDF1 is 1 when every count is 0: the two empty sets agree."""
    denominator = (2 * idtp) + idfp + idfn
    if denominator == 0:
        return 1.0
    return (2 * idtp) / denominator


def _count_pairs(pairs: Sequence[tuple[int, int]]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for pair in pairs:
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _unique_ids(tracks: Sequence[TrackBox]) -> list[int]:
    return list(dict.fromkeys(item.track_id for item in tracks))


def _indices_by_frame(boxes: Sequence[DetBox]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for index, box in enumerate(boxes):
        grouped.setdefault(box.frame, []).append(index)
    return grouped


def _tracks_by_frame(tracks: Sequence[TrackBox]) -> dict[int, list[TrackBox]]:
    grouped: dict[int, list[TrackBox]] = {}
    for item in tracks:
        grouped.setdefault(item.frame, []).append(item)
    return grouped


def _boxes_of(items: Sequence[TrackBox]) -> list[Box]:
    return [item.box for item in items]


def _present_boxes(boxes: Sequence[DetBox] | None) -> list[DetBox]:
    if boxes is None:
        return []
    return list(boxes)


def _present_tracks(tracks: Sequence[TrackBox] | None) -> list[TrackBox]:
    if tracks is None:
        return []
    return list(tracks)


def _recall_ratio(matched: int, ground_truth: int) -> float:
    """1 when ``ground_truth`` is 0: an empty ground-truth set has no miss."""
    if ground_truth == 0:
        return 1.0
    return matched / ground_truth


def _blocked_fixture(fixture_id: str) -> bool:
    return fixture_id.startswith(_BLOCKED_PREFIXES)


def _reject_blocked(fixture_id: str) -> None:
    raise FixtureSeparationError(
        f"{fixture_id} cannot produce an AC-05 result. FX-SMOKE and FX-CODE are not AC-05 fixtures."
    )


def _require_fixture_id(fixture_id: object) -> None:
    if not isinstance(fixture_id, str) or fixture_id == "":
        raise ValueError(f"fixture_id must be a non-empty string, got {fixture_id!r}")


def _require_frame(frame: object) -> None:
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError(f"frame must be an int >= 0, got {frame!r}")


def _require_track_id(track_id: object) -> None:
    if isinstance(track_id, bool) or not isinstance(track_id, int):
        raise ValueError(f"track_id must be an int, got {track_id!r}")


def _require_flag(label: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a bool, got {value!r}")


def _require_box(box: object) -> None:
    if not isinstance(box, tuple) or len(box) != 4:
        raise ValueError(f"box must be an (x1, y1, x2, y2) tuple, got {box!r}")
    x1 = _as_float("x1", box[0])
    y1 = _as_float("y1", box[1])
    x2 = _as_float("x2", box[2])
    y2 = _as_float("y2", box[3])
    if x2 <= x1:
        raise ValueError(f"box x2 must be greater than x1, got x1={x1} x2={x2}")
    if y2 <= y1:
        raise ValueError(f"box y2 must be greater than y1, got y1={y1} y2={y2}")


def _as_float(label: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _check_iou(iou: object) -> None:
    if isinstance(iou, bool) or not isinstance(iou, (int, float)):
        raise ValueError(f"iou must be a real number in [0, 1], got {iou!r}")
    if not math.isfinite(float(iou)) or not 0.0 <= float(iou) <= 1.0:
        raise ValueError(f"iou must be a real number in [0, 1], got {iou!r}")
