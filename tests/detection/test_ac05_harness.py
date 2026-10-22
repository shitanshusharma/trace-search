"""Behaviour tests for the AC-05 harness.

Metrics are checked on hand-built boxes. ``run`` stays DEFERRED and does
not produce an AC-05 score from FX-SMOKE or FX-CODE fixtures.
"""

from __future__ import annotations

import re

import pytest

from trace_search.detection.ac05_harness import (
    DetBox,
    TrackBox,
    detection_recall,
    idf1,
    run,
)
from trace_search.fixtures.guards import FixtureSeparationError

_LEFT = (0.0, 0.0, 10.0, 10.0)
_RIGHT = (100.0, 0.0, 110.0, 10.0)
_BLOCKED_IDS = ("FX-SMOKE-SYN-001", "FX-SMOKE", "FX-CODE-001", "FX-CODE")
_DISGUISED_BLOCKED_IDS = (
    "fx-smoke-syn-001",
    "Fx-Code-001",
    " FX-SMOKE-SYN-001",
    "FX-CODE-001 ",
    "  Fx-Smoke-001  ",
    "FX_SMOKE_SYN_001",
    "fx_code_001",
)
_TOP_LEVEL_KEYS = {
    "criterion",
    "fixture_id",
    "status",
    "evidence",
    "ac05_score",
    "diagnostics",
}
_DIAGNOSTICS_NOTE = (
    "Not an AC-05 result. AC-05 is DEFERRED in Cut 01; these numbers are not evidence."
)


def _numeric_leaves(value: object) -> list[object]:
    """Numbers in a run payload. Bools are flags, not scores."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return []
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, dict):
        found: list[object] = []
        for nested in value.values():
            found.extend(_numeric_leaves(nested))
        return found
    raise AssertionError(f"unexpected result value {value!r}")


def _numbers_outside_diagnostics(result: dict[str, object]) -> list[object]:
    """Walk every top-level value except ``diagnostics``. ``False`` is allowed."""
    found: list[object] = []
    for key, value in result.items():
        if key == "diagnostics":
            continue
        found.extend(_numeric_leaves(value))
    return found


def test_detection_recall_is_three_quarters() -> None:
    """Three matched boxes out of four ground-truth boxes, plus a false positive."""
    gt = [
        DetBox(0, (0.0, 0.0, 10.0, 10.0)),
        DetBox(0, (20.0, 0.0, 30.0, 10.0)),
        DetBox(0, (40.0, 0.0, 50.0, 10.0)),
        DetBox(0, (60.0, 0.0, 70.0, 10.0)),
    ]
    pred = [
        DetBox(0, (0.0, 0.0, 10.0, 10.0)),
        DetBox(0, (20.0, 0.0, 30.0, 10.0)),
        DetBox(0, (40.0, 0.0, 50.0, 10.0)),
        DetBox(0, (200.0, 0.0, 210.0, 10.0)),
    ]
    assert detection_recall(gt, pred) == 0.75
    assert detection_recall(gt, pred) == 3 / 4


def test_detection_recall_keeps_the_higher_iou_pair() -> None:
    """The weaker pair is offered first. Highest IoU still matches both boxes.

    Ground truth A overlaps prediction 1 at IoU 1 and prediction 0 at 4/7.
    Ground truth B overlaps prediction 0 at 5/6 and prediction 1 at 3/7,
    which is below 0.5. Taking prediction 0 for A would leave B unmatched.
    """
    gt = [
        DetBox(0, (0.0, 0.0, 10.0, 10.0)),
        DetBox(0, (4.0, 0.0, 14.0, 10.0)),
    ]
    pred = [
        DetBox(0, (2.0, 0.0, 14.0, 10.0)),
        DetBox(0, (0.0, 0.0, 10.0, 10.0)),
    ]
    assert detection_recall(gt, pred) == 1.0


def test_equal_iou_matches_the_earlier_ground_truth_once() -> None:
    """One prediction cannot match two ground-truth boxes. The tie goes to index 0."""
    gt = [
        DetBox(0, (0.0, 0.0, 10.0, 10.0)),
        DetBox(0, (6.0, 0.0, 16.0, 10.0), low_light=True),
    ]
    pred = [DetBox(0, (3.0, 0.0, 13.0, 10.0))]
    assert detection_recall(gt, pred) == 0.5
    result = run("FX-HOLD-001", gt=gt, pred=pred)
    assert result["diagnostics"]["subset"]["low_light"] == {
        "recall": 0.0,
        "matched": 0,
        "ground_truth": 1,
    }


def test_detection_recall_counts_iou_at_the_threshold() -> None:
    """IoU 0.5 matches. A lower overlap does not. A stricter threshold rejects 0.5."""
    gt = [DetBox(0, (0.0, 0.0, 3.0, 2.0))]
    at_threshold = [DetBox(0, (1.0, 0.0, 4.0, 2.0))]
    below = [DetBox(0, (2.0, 0.0, 5.0, 2.0))]
    assert detection_recall(gt, at_threshold) == 1.0
    assert detection_recall(gt, below) == 0.0
    assert detection_recall(gt, at_threshold, iou=0.6) == 0.0
    assert detection_recall(gt, gt, iou=1.0) == 1.0


def test_detection_recall_does_not_match_across_frames() -> None:
    gt = [DetBox(0, _LEFT)]
    pred = [DetBox(1, _LEFT)]
    assert detection_recall(gt, pred) == 0.0


def test_detection_recall_ignores_non_overlapping_boxes() -> None:
    """A horizontal gap and a vertical gap are both misses."""
    gt = [
        DetBox(0, (0.0, 0.0, 10.0, 10.0)),
        DetBox(1, (0.0, 0.0, 10.0, 10.0)),
    ]
    pred = [
        DetBox(0, (10.0, 0.0, 20.0, 10.0)),
        DetBox(1, (0.0, 10.0, 10.0, 20.0)),
    ]
    assert detection_recall(gt, pred) == 0.0


def test_detection_recall_empty_ground_truth_is_one() -> None:
    """No ground truth means no miss. Predictions do not change that."""
    pred = [DetBox(0, _LEFT)]
    assert detection_recall([], []) == 1.0
    assert detection_recall([], pred) == 1.0
    assert detection_recall(pred, []) == 0.0


def test_idf1_id_swap_halfway_is_one_half() -> None:
    """Two tracks over 10 frames, ids swapped on frames 5-9.

    Each global assignment matches 5 frames per track, so IDTP is 10.
    IDFP and IDFN are the other 10. IDF1 is 20/40 = 0.5.
    A per-frame id change would score 1. A single matched track would
    not survive the perfect-tracking test.
    """
    gt: list[TrackBox] = []
    pred: list[TrackBox] = []
    for frame in range(10):
        gt.append(TrackBox(frame, 1, _LEFT))
        gt.append(TrackBox(frame, 2, _RIGHT))
        if frame < 5:
            pred.append(TrackBox(frame, 10, _LEFT))
            pred.append(TrackBox(frame, 20, _RIGHT))
        else:
            pred.append(TrackBox(frame, 10, _RIGHT))
            pred.append(TrackBox(frame, 20, _LEFT))

    result = idf1(gt, pred)
    assert result["idtp"] == 10
    assert result["idfp"] == 10
    assert result["idfn"] == 10
    assert result["idf1"] == 0.5


def test_idf1_perfect_tracking_is_one() -> None:
    """The same two tracks over 10 frames, in reverse input order."""
    gt: list[TrackBox] = []
    pred: list[TrackBox] = []
    for frame in range(9, -1, -1):
        gt.append(TrackBox(frame, 1, _LEFT))
        gt.append(TrackBox(frame, 2, _RIGHT))
        pred.append(TrackBox(frame, 20, _RIGHT))
        pred.append(TrackBox(frame, 10, _LEFT))

    result = idf1(gt, pred)
    assert result["idtp"] == 20
    assert result["idfp"] == 0
    assert result["idfn"] == 0
    assert result["idf1"] == 1.0


def test_idf1_empty_inputs() -> None:
    """Both empty agree. One empty side is a total miss, not a perfect score."""
    pred = [TrackBox(0, 1, _LEFT), TrackBox(1, 1, _LEFT)]
    gt = [TrackBox(0, 7, _RIGHT)]

    both = idf1([], [])
    assert both == {"idtp": 0, "idfp": 0, "idfn": 0, "idf1": 1.0}

    only_pred = idf1([], pred)
    assert only_pred == {"idtp": 0, "idfp": 2, "idfn": 0, "idf1": 0.0}

    only_gt = idf1(gt, [])
    assert only_gt == {"idtp": 0, "idfp": 0, "idfn": 1, "idf1": 0.0}


def test_idf1_requires_spatial_overlap() -> None:
    """The same track id with disjoint boxes is not a true positive."""
    gt = [TrackBox(0, 1, _LEFT)]
    pred = [TrackBox(0, 1, _RIGHT)]
    assert idf1(gt, pred) == {"idtp": 0, "idfp": 1, "idfn": 1, "idf1": 0.0}


def test_idf1_threshold_matches_recall() -> None:
    gt = [TrackBox(0, 1, (0.0, 0.0, 3.0, 2.0))]
    pred = [TrackBox(0, 4, (1.0, 0.0, 4.0, 2.0))]
    assert idf1(gt, pred)["idf1"] == 1.0
    assert idf1(gt, pred, iou=0.6)["idf1"] == 0.0


def test_idf1_counts_unmatched_sides() -> None:
    matched = TrackBox(0, 1, _LEFT)
    extra_gt = TrackBox(0, 2, _RIGHT)
    extra_pred = TrackBox(0, 9, _RIGHT)
    false_positive = idf1([matched], [matched, extra_pred])
    false_negative = idf1([matched, extra_gt], [TrackBox(0, 3, _LEFT)])
    assert false_positive == {"idtp": 1, "idfp": 1, "idfn": 0, "idf1": 2 / 3}
    assert false_negative == {"idtp": 1, "idfp": 0, "idfn": 1, "idf1": 2 / 3}


def test_idf1_min_cost_assignment_avoids_the_greedy_trap() -> None:
    """The largest single pair is not the best assignment.

    Overlap counts are GT1-P1 = 5, GT1-P2 = 4, GT2-P1 = 4, GT2-P2 = 0.
    Pairing the 5 leaves the other track unmatched (IDTP 5).
    Crossing them scores 4 + 4 (IDTP 8).
    """
    gt: list[TrackBox] = []
    pred: list[TrackBox] = []
    for frame in range(5):
        gt.append(TrackBox(frame, 1, _LEFT))
        pred.append(TrackBox(frame, 1, _LEFT))
    for frame in range(5, 9):
        gt.append(TrackBox(frame, 1, _LEFT))
        gt.append(TrackBox(frame, 2, _RIGHT))
        pred.append(TrackBox(frame, 2, _LEFT))
        pred.append(TrackBox(frame, 1, _RIGHT))

    result = idf1(gt, pred)
    assert result["idtp"] == 8
    assert result["idfp"] == 5
    assert result["idfn"] == 5
    assert result["idf1"] == 8 / 13


def test_subset_reports_low_light_and_occluded_separately() -> None:
    """Overall recall is 3/4. The two flags do not share one denominator.

    The matched box at x=0 carries both flags, so it counts in each subset.
    """
    gt = [
        DetBox(0, (0.0, 0.0, 10.0, 10.0), low_light=True, occluded=True),
        DetBox(0, (20.0, 0.0, 30.0, 10.0), low_light=True),
        DetBox(0, (40.0, 0.0, 50.0, 10.0), occluded=True),
        DetBox(0, (60.0, 0.0, 70.0, 10.0)),
    ]
    pred = [
        DetBox(0, (0.0, 0.0, 10.0, 10.0)),
        DetBox(0, (40.0, 0.0, 50.0, 10.0)),
        DetBox(0, (60.0, 0.0, 70.0, 10.0)),
    ]
    result = run("FX-HOLD-001", gt=gt, pred=pred)
    assert result["status"] == "DEFERRED"
    assert result["evidence"] is False
    assert result["ac05_score"] is None
    assert result["diagnostics"]["recall"] == 0.75
    assert result["diagnostics"]["subset"]["low_light"] == {
        "recall": 0.5,
        "matched": 1,
        "ground_truth": 2,
    }
    assert result["diagnostics"]["subset"]["occluded"] == {
        "recall": 1.0,
        "matched": 2,
        "ground_truth": 2,
    }


def test_subset_recall_is_one_when_the_flag_is_absent() -> None:
    gt = [DetBox(0, _LEFT)]
    pred = [DetBox(0, _LEFT)]
    result = run("FX-HOLD-001", gt=gt, pred=pred)
    assert result["diagnostics"]["subset"]["low_light"] == {
        "recall": 1.0,
        "matched": 0,
        "ground_truth": 0,
    }
    assert result["diagnostics"]["subset"]["occluded"] == {
        "recall": 1.0,
        "matched": 0,
        "ground_truth": 0,
    }


def test_run_empty_sequences_use_the_empty_rules() -> None:
    result = run("FX-HOLD-001", gt=[], pred=[], gt_tracks=[], pred_tracks=[])
    assert result["status"] == "DEFERRED"
    assert result["ac05_score"] is None
    assert result["evidence"] is False
    diagnostics = result["diagnostics"]
    assert diagnostics["recall"] == 1.0
    assert diagnostics["idf1"] == 1.0
    assert diagnostics["idtp"] == 0
    assert diagnostics["idfp"] == 0
    assert diagnostics["idfn"] == 0


def test_run_measures_one_side_when_the_other_is_omitted() -> None:
    gt = [DetBox(0, _LEFT)]
    tracks = [TrackBox(0, 1, _LEFT)]
    recall_only = run("FX-HOLD-001", gt=gt)
    track_only = run("FX-HOLD-001", pred_tracks=tracks)
    assert recall_only["diagnostics"]["recall"] == 0.0
    assert recall_only["diagnostics"]["idf1"] is None
    assert track_only["diagnostics"]["recall"] is None
    assert track_only["diagnostics"]["idf1"] == 0.0
    assert track_only["diagnostics"]["idfp"] == 1
    assert track_only["diagnostics"]["idfn"] == 0


def test_run_hold_without_boxes_is_deferred() -> None:
    result = run("FX-HOLD-001")
    assert result["fixture_id"] == "FX-HOLD-001"
    assert result["criterion"] == "AC-05"
    assert result["status"] == "DEFERRED"
    assert result["evidence"] is False
    assert result["ac05_score"] is None
    assert set(result) == _TOP_LEVEL_KEYS
    diagnostics = result["diagnostics"]
    assert diagnostics["note"] == _DIAGNOSTICS_NOTE
    assert diagnostics["recall"] is None
    assert diagnostics["idf1"] is None
    assert diagnostics["idtp"] is None
    assert diagnostics["idfp"] is None
    assert diagnostics["idfn"] is None
    assert diagnostics["subset"]["low_light"]["recall"] is None
    assert diagnostics["subset"]["occluded"]["recall"] is None
    assert _numbers_outside_diagnostics(result) == []
    assert _numeric_leaves(result) == []


@pytest.mark.parametrize("fixture_id", _BLOCKED_IDS)
def test_run_smoke_and_code_raise(fixture_id: str) -> None:
    with pytest.raises(FixtureSeparationError, match=fixture_id):
        run(fixture_id)
    with pytest.raises(FixtureSeparationError, match=fixture_id):
        run(fixture_id, mode="eval")
    with pytest.raises(FixtureSeparationError, match=fixture_id):
        run(fixture_id, mode="UNIT_TEST")


@pytest.mark.parametrize("fixture_id", _DISGUISED_BLOCKED_IDS)
def test_disguised_smoke_and_code_ids_raise(fixture_id: str) -> None:
    """Lower case, mixed case, surrounding space, and underscores are the same gate."""
    with pytest.raises(FixtureSeparationError, match=re.escape(fixture_id)):
        run(fixture_id)
    gt = [DetBox(0, _LEFT)]
    pred = [DetBox(0, _LEFT)]
    tracks = [TrackBox(0, 1, _LEFT)]
    result = run(
        fixture_id,
        gt=gt,
        pred=pred,
        gt_tracks=tracks,
        pred_tracks=tracks,
        mode="unit_test",
    )
    assert result["fixture_id"] == fixture_id
    assert result["evidence"] is False
    assert result["status"] == "DEFERRED"
    assert set(result) == _TOP_LEVEL_KEYS
    assert result["diagnostics"]["note"] == _DIAGNOSTICS_NOTE
    assert result["diagnostics"]["recall"] is None
    assert result["diagnostics"]["idf1"] is None
    assert _numbers_outside_diagnostics(result) == []
    assert _numeric_leaves(result) == []


def test_embedded_smoke_prefix_is_not_blocked() -> None:
    result = run("X-FX-SMOKE-001")
    assert result["status"] == "DEFERRED"
    assert result["fixture_id"] == "X-FX-SMOKE-001"


@pytest.mark.parametrize("fixture_id", ("FX-SMOKE-SYN-001", "FX-CODE-001"))
def test_run_unit_test_evidence_is_false(fixture_id: str) -> None:
    """Supplied boxes would score perfectly. Unit-test mode still has no score."""
    gt = [DetBox(0, _LEFT)]
    pred = [DetBox(0, _LEFT)]
    tracks = [TrackBox(0, 1, _LEFT)]
    result = run(
        fixture_id,
        gt=gt,
        pred=pred,
        gt_tracks=tracks,
        pred_tracks=tracks,
        mode="unit_test",
    )
    assert result["status"] == "DEFERRED"
    assert result["evidence"] is False
    assert result["ac05_score"] is None
    assert result["fixture_id"] == fixture_id
    assert _numeric_leaves(result) == []


def test_run_hold_metrics_are_not_an_ac05_score() -> None:
    gt = [TrackBox(0, 1, _LEFT)]
    result = run(
        "FX-HOLD-001",
        gt=[DetBox(0, _LEFT)],
        pred=[DetBox(0, _LEFT)],
        gt_tracks=gt,
        pred_tracks=gt,
    )
    assert result["status"] == "DEFERRED"
    assert result["evidence"] is False
    assert result["ac05_score"] is None
    assert set(result) == _TOP_LEVEL_KEYS
    assert set(result["diagnostics"]) == {
        "note",
        "recall",
        "idf1",
        "idtp",
        "idfp",
        "idfn",
        "subset",
    }
    assert result["diagnostics"]["note"] == _DIAGNOSTICS_NOTE
    assert result["diagnostics"]["recall"] == 1.0
    assert result["diagnostics"]["idf1"] == 1.0
    assert _numbers_outside_diagnostics(result) == []
    assert _numeric_leaves(result["diagnostics"]) != []


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        DetBox(-1, _LEFT)
    with pytest.raises(ValueError):
        DetBox(True, _LEFT)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DetBox(1.5, _LEFT)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DetBox(0, (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        DetBox(0, (0.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        DetBox(0, (0.0, 0.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DetBox(0, [0.0, 0.0, 1.0, 1.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DetBox(0, (0.0, 0.0, float("inf"), 1.0))
    with pytest.raises(ValueError):
        DetBox(0, (True, 0.0, 1.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DetBox(0, ("0", 0.0, 1.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DetBox(0, _LEFT, low_light=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TrackBox(0, True, _LEFT)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TrackBox(0, 1.5, _LEFT)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        run("")
    with pytest.raises(ValueError):
        run(None)  # type: ignore[arg-type]
    good = [DetBox(0, _LEFT)]
    with pytest.raises(ValueError):
        detection_recall(good, good, iou=1.1)
    with pytest.raises(ValueError):
        detection_recall(good, good, iou=-0.1)
    with pytest.raises(ValueError):
        detection_recall(good, good, iou=float("nan"))
    with pytest.raises(ValueError):
        detection_recall(good, good, iou=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        detection_recall(good, good, iou="0.5")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        run("FX-HOLD-001", iou=2)
