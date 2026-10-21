"""Per-camera multi-object tracker.

Clean-room implementation of the two-stage association in Zhang, Sun, Jiang,
et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
(arXiv:2110.06864). No code is taken from a ByteTrack, OC-SORT, or boxmot
repository. There is no appearance model, no weight file, and no
cross-camera id.

For each frame, and separately for each detection class:

1. Predict every track with a constant-velocity Kalman filter.
2. Match all tracks to high-score boxes with IoU cost and the Hungarian
   algorithm. Reject a pair whose IoU is smaller than ``match_iou``.
3. Match the tracks that are still free to the low-score boxes the same way.
4. Drop unmatched low-score boxes. They never start a track.
5. Keep an unmatched track until it has been missing for more than
   ``max_age_frames`` frames, then delete it.
6. Start a track from each unmatched high-score box.

Lost tracks are not returned. Track ids are per camera, start at 1, and are
assigned in order. One ``ByteTracker`` instance is one camera.

Argument defaults:

- ``high_thresh`` 0.6. The paper's detection score threshold tau. A box
  scores high when its score is greater than this value, matching the
  paper's ``score > tau`` split. A box at the threshold is low-score.
- ``low_thresh`` 0.1. Boxes below this are dropped. A box in
  ``[low_thresh, high_thresh]`` is matched in the second stage only. The
  paper's occlusion example still associates a box scored 0.1.
- ``match_iou`` 0.2. The paper rejects a pair whose IoU is smaller than 0.2.
- ``max_age_frames`` 30. The paper keeps a lost tracklet for 30 frames.
  The track is deleted once it has been unmatched for more frames than this.
- ``min_hits`` 1. Output gate on top of the paper's algorithm. The frame
  that starts a track counts as one hit. The default emits that track
  immediately. An unconfirmed track is deleted on its first miss.
"""

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from trace_search.attributes.attributes import Box
from trace_search.detection.types import DetClass, Detection
from trace_search.tracking.types import TrackObservation

DEFAULT_HIGH_THRESH = 0.6
DEFAULT_LOW_THRESH = 0.1
DEFAULT_MATCH_IOU = 0.2
DEFAULT_MAX_AGE_FRAMES = 30
DEFAULT_MIN_HITS = 1

# Constant-velocity state: center x, center y, width, height, then the
# per-frame velocity of each. Process noise is discrete white acceleration.
# These are filter constants, not the paper's association hyperparameters.
_STATE = 8
_MEAS = 4
_ACCEL_VAR = 2.0
_MEAS_VAR = 1.0
_INIT_VEL_VAR = 100.0
_MIN_SIDE = 1e-3
_FIRST_TRACK_ID = 1


@dataclass(frozen=True)
class _Params:
    high_thresh: float
    low_thresh: float
    match_iou: float
    max_age_frames: int
    min_hits: int


@dataclass
class _Track:
    track_id: int
    det_class: DetClass
    hits: int
    missed: int
    mean: npt.NDArray[np.float64]
    cov: npt.NDArray[np.float64]
    predicted: Box


@dataclass(frozen=True)
class _Assignment:
    pairs: list[tuple[int, int]]
    unmatched_tracks: list[int]
    unmatched_dets: list[int]


def _unit(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a real number in [0, 1], got {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be a real number in [0, 1], got {value!r}")
    return number


def _count(value: int, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an int >= {minimum}, got {value!r}")
    return value


def _parameters(
    high_thresh: float,
    low_thresh: float,
    match_iou: float,
    max_age_frames: int,
    min_hits: int,
) -> _Params:
    high = _unit(high_thresh, "high_thresh")
    low = _unit(low_thresh, "low_thresh")
    match = _unit(match_iou, "match_iou")
    if low > high:
        raise ValueError(f"low_thresh ({low}) must be <= high_thresh ({high})")
    return _Params(
        high,
        low,
        match,
        _count(max_age_frames, "max_age_frames", 0),
        _count(min_hits, "min_hits", 1),
    )


def _motion_matrix() -> npt.NDArray[np.float64]:
    matrix = np.eye(_STATE, dtype=np.float64)
    for index in range(_MEAS):
        matrix[index, index + _MEAS] = 1.0
    return matrix


def _process_noise(accel_var: float) -> npt.NDArray[np.float64]:
    noise = np.zeros((_STATE, _STATE), dtype=np.float64)
    block = accel_var * np.array([[0.25, 0.5], [0.5, 1.0]], dtype=np.float64)
    for index in range(_MEAS):
        noise[index, index] = block[0, 0]
        noise[index, index + _MEAS] = block[0, 1]
        noise[index + _MEAS, index] = block[1, 0]
        noise[index + _MEAS, index + _MEAS] = block[1, 1]
    return noise


_MOTION_F = _motion_matrix()
_PROCESS_NOISE = _process_noise(_ACCEL_VAR)
_MEAS_NOISE = np.eye(_MEAS, dtype=np.float64) * _MEAS_VAR


def _box_to_meas(box: Box) -> npt.NDArray[np.float64]:
    x1, y1, x2, y2 = box
    return np.array(
        [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1],
        dtype=np.float64,
    )


def _state_to_box(state: npt.NDArray[np.float64]) -> Box:
    center_x = float(state[0])
    center_y = float(state[1])
    width = max(float(state[2]), _MIN_SIDE)
    height = max(float(state[3]), _MIN_SIDE)
    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def _initial_covariance() -> npt.NDArray[np.float64]:
    cov = np.eye(_STATE, dtype=np.float64)
    for index in range(_MEAS, _STATE):
        cov[index, index] = _INIT_VEL_VAR
    return cov


def _predict_motion(
    mean: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    return _MOTION_F @ mean, _MOTION_F @ cov @ _MOTION_F.T + _PROCESS_NOISE


def _correct_motion(
    mean: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64],
    box: Box,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    innovation = _box_to_meas(box) - mean[:_MEAS]
    innovation_cov = cov[:_MEAS, :_MEAS] + _MEAS_NOISE
    gain = np.linalg.solve(innovation_cov.T, cov[:, :_MEAS].T).T
    return mean + gain @ innovation, cov - gain @ cov[:_MEAS, :]


def _new_track(track_id: int, detection: Detection) -> _Track:
    mean = np.zeros(_STATE, dtype=np.float64)
    mean[:_MEAS] = _box_to_meas(detection.box)
    return _Track(
        track_id=track_id,
        det_class=detection.det_class,
        hits=1,
        missed=0,
        mean=mean,
        cov=_initial_covariance(),
        predicted=detection.box,
    )


def _predict_all(tracks: Sequence[_Track]) -> None:
    for track in tracks:
        track.mean, track.cov = _predict_motion(track.mean, track.cov)
        track.predicted = _state_to_box(track.mean)


def _iou(left: Box, right: Box) -> float:
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


def _pairwise_iou(tracks: Sequence[Box], detections: Sequence[Box]) -> npt.NDArray[np.float64]:
    overlaps = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    for row, track_box in enumerate(tracks):
        for col, det_box in enumerate(detections):
            overlaps[row, col] = _iou(track_box, det_box)
    return overlaps


def _linear_sum_assignment(cost: npt.NDArray[np.float64]) -> tuple[list[int], list[int]]:
    """Hungarian assignment from ``scipy.optimize.linear_sum_assignment``.

    scipy 1.17.1 ships no ``py.typed`` marker, so a direct import is untyped
    under ``mypy --strict``.
    """
    optimize = importlib.import_module("scipy.optimize")
    row_ind, col_ind = optimize.linear_sum_assignment(cost)
    return (
        [int(index) for index in row_ind.tolist()],
        [int(index) for index in col_ind.tolist()],
    )


def _gate(overlaps: npt.NDArray[np.float64], match_iou: float) -> _Assignment:
    rows, cols = _linear_sum_assignment(1.0 - overlaps)
    pairs: list[tuple[int, int]] = []
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    for row, col in zip(rows, cols, strict=True):
        if float(overlaps[row, col]) < match_iou:
            continue
        pairs.append((row, col))
        used_tracks.add(row)
        used_dets.add(col)
    track_count = int(overlaps.shape[0])
    det_count = int(overlaps.shape[1])
    free_tracks = [index for index in range(track_count) if index not in used_tracks]
    free_dets = [index for index in range(det_count) if index not in used_dets]
    return _Assignment(pairs, free_tracks, free_dets)


def _match(tracks: Sequence[Box], detections: Sequence[Box], match_iou: float) -> _Assignment:
    if len(tracks) == 0 or len(detections) == 0:
        return _Assignment([], list(range(len(tracks))), list(range(len(detections))))
    return _gate(_pairwise_iou(tracks, detections), match_iou)


def _split_scores(
    detections: Sequence[Detection],
    det_class: DetClass,
    high_thresh: float,
    low_thresh: float,
) -> tuple[list[Detection], list[Detection]]:
    high: list[Detection] = []
    low: list[Detection] = []
    for detection in detections:
        if detection.det_class is not det_class:
            continue
        if detection.score > high_thresh:
            high.append(detection)
        elif detection.score >= low_thresh:
            low.append(detection)
    return high, low


def _boxes(detections: Sequence[Detection]) -> list[Box]:
    return [detection.box for detection in detections]


def _predicted(tracks: Sequence[_Track]) -> list[Box]:
    return [track.predicted for track in tracks]


def _observation(track_id: int, detection: Detection) -> TrackObservation:
    return TrackObservation(
        camera_id=detection.camera_id,
        track_id=track_id,
        det_class=detection.det_class,
        frame_number=detection.frame_number,
        timestamp_seconds=detection.timestamp_seconds,
        box=detection.box,
        score=detection.score,
    )


def _emit_if_confirmed(
    track: _Track,
    detection: Detection,
    min_hits: int,
    observations: list[TrackObservation],
) -> None:
    if track.hits >= min_hits:
        observations.append(_observation(track.track_id, detection))


def _consume(
    tracks: Sequence[_Track],
    detections: Sequence[Detection],
    pairs: Sequence[tuple[int, int]],
    min_hits: int,
    observations: list[TrackObservation],
) -> None:
    for track_index, det_index in pairs:
        detection = detections[det_index]
        track = tracks[track_index]
        track.mean, track.cov = _correct_motion(track.mean, track.cov, detection.box)
        track.hits += 1
        track.missed = 0
        _emit_if_confirmed(track, detection, min_hits, observations)


def _mark_missed(tracks: Sequence[_Track], unmatched: Sequence[int]) -> None:
    for index in unmatched:
        tracks[index].missed += 1


def _birth(
    tracks: list[_Track],
    detections: Sequence[Detection],
    unmatched: Sequence[int],
    next_id: int,
    min_hits: int,
    observations: list[TrackObservation],
) -> int:
    for det_index in unmatched:
        detection = detections[det_index]
        track = _new_track(next_id, detection)
        tracks.append(track)
        next_id += 1
        _emit_if_confirmed(track, detection, min_hits, observations)
    return next_id


def _associate_class(
    tracks: list[_Track],
    detections: Sequence[Detection],
    det_class: DetClass,
    high_thresh: float,
    low_thresh: float,
    match_iou: float,
    min_hits: int,
    next_id: int,
    observations: list[TrackObservation],
) -> int:
    pool = [track for track in tracks if track.det_class is det_class]
    high, low = _split_scores(detections, det_class, high_thresh, low_thresh)
    first = _match(_predicted(pool), _boxes(high), match_iou)
    _consume(pool, high, first.pairs, min_hits, observations)
    remain = [pool[index] for index in first.unmatched_tracks]
    second = _match(_predicted(remain), _boxes(low), match_iou)
    _consume(remain, low, second.pairs, min_hits, observations)
    _mark_missed(remain, second.unmatched_tracks)
    return _birth(tracks, high, first.unmatched_dets, next_id, min_hits, observations)


def _track_id_of(observation: TrackObservation) -> int:
    return observation.track_id


def _associate_frame(
    tracks: list[_Track],
    detections: Sequence[Detection],
    high_thresh: float,
    low_thresh: float,
    match_iou: float,
    min_hits: int,
    next_id: int,
) -> tuple[list[TrackObservation], int]:
    observations: list[TrackObservation] = []
    for det_class in DetClass:
        next_id = _associate_class(
            tracks,
            detections,
            det_class,
            high_thresh,
            low_thresh,
            match_iou,
            min_hits,
            next_id,
            observations,
        )
    observations.sort(key=_track_id_of)
    return observations, next_id


def _keep(track: _Track, max_age_frames: int, min_hits: int) -> bool:
    if track.missed == 0:
        return True
    if track.hits < min_hits:
        return False
    return track.missed <= max_age_frames


class ByteTracker:
    """Two-stage IoU tracker for one camera.

    The module docstring records the association order and the default value
    of each argument.
    """

    def __init__(
        self,
        high_thresh: float = DEFAULT_HIGH_THRESH,
        low_thresh: float = DEFAULT_LOW_THRESH,
        match_iou: float = DEFAULT_MATCH_IOU,
        max_age_frames: int = DEFAULT_MAX_AGE_FRAMES,
        min_hits: int = DEFAULT_MIN_HITS,
    ) -> None:
        params = _parameters(high_thresh, low_thresh, match_iou, max_age_frames, min_hits)
        self._high_thresh = params.high_thresh
        self._low_thresh = params.low_thresh
        self._match_iou = params.match_iou
        self._max_age_frames = params.max_age_frames
        self._min_hits = params.min_hits
        self._tracks: list[_Track] = []
        self._next_id = _FIRST_TRACK_ID

    def reset(self) -> None:
        """Drop this camera's tracks and start ids at 1 again."""
        self._tracks = []
        self._next_id = _FIRST_TRACK_ID

    def update(self, detections: Sequence[Detection]) -> list[TrackObservation]:
        """Return this camera's confirmed tracks for one frame of detections."""
        _predict_all(self._tracks)
        observations, self._next_id = _associate_frame(
            self._tracks,
            detections,
            self._high_thresh,
            self._low_thresh,
            self._match_iou,
            self._min_hits,
            self._next_id,
        )
        self._tracks = [
            track for track in self._tracks if _keep(track, self._max_age_frames, self._min_hits)
        ]
        return observations
