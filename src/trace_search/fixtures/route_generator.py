"""Seeded flat-rectangle routes for six fake cameras.

``generate_route_fixture`` writes ``CAM-1.mp4`` .. ``CAM-N.mp4`` (mp4v),
``cameras.json``, ``topology.json``, ``truth.json``, and ``manifest.json``
under ``out_dir``. Figures are upright rectangles with two flat colour
bands. There is no head, no limbs, and no text. One figure copies another
figure's colour pair and walks the other corridor, which has no transit edge.

Figures sit in separate horizontal lanes with a background gap between them,
and they do not share one timetable. Each figure slides across its lane, so
the 10 frames ``perceptual_hashes`` samples are not the same picture. The
background stays a uniform dark colour. Nothing else is drawn.

The same seed writes byte-identical files. mp4v on opencv-python-headless
4.10.0.84 (Lavf59.27.100) already stores container creation and modification
times as 0. ``_zero_mp4_times`` sets those fields to 0 again so a writer that
embeds the wall clock cannot change the hash.

``RouteFixture.sample_frame`` is the BGR buffer passed to the encoder for one
occupied frame. mp4v is lossy, so a palette check belongs on that buffer.
The drawn box uses a half-open pixel range ``[x1, x2)`` x ``[y1, y2)``.

``figure_id`` labels a shape in this generated clip. It is not a person id.
Outputs are temp files and are never committed. ``out_dir`` must resolve
outside this repository.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from trace_search.fixtures.content import perceptual_hashes, sha256_file
from trace_search.ingest.camera_registry import CameraConfig, CameraRegistry
from trace_search.matching.topology import CameraTopology, TransitEdge

__all__ = ["RouteFixture", "generate_route_fixture"]

_MIN_SEED = 0
_MAX_SEED = 2_147_483_647
_MIN_CAMERAS = 4
_MAX_CAMERAS = 12
_MIN_FIGURES = 2
_MAX_FIGURES = 8
_MIN_FPS = 1
_MAX_FPS = 60
_MIN_SECONDS = 1
_MAX_SECONDS = 120
_MIN_EDGE = 64
_MAX_EDGE = 1920
_MIN_FRAMES = 10
_WINDOW_MIN = 0.5
_WINDOW_MAX = 8.0
_FRAME_EPS = 1e-3
_MAX_STEP = 4
_BACKGROUND_BGR = (16, 16, 16)
_SOURCE_URL = "https://github.com/shitanshusharma/trace-search"
_CONTAINER_BOXES = frozenset({b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"})
_TIMED_BOXES = frozenset({b"mvhd", b"tkhd", b"mdhd"})
_COLOUR_PAIRS: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    ((0, 0, 255), (255, 0, 0)),
    ((0, 255, 0), (255, 0, 255)),
    ((255, 255, 0), (0, 128, 255)),
    ((255, 0, 255), (0, 255, 255)),
    ((0, 165, 255), (128, 0, 128)),
    ((255, 128, 0), (0, 255, 128)),
    ((192, 0, 64), (64, 0, 192)),
    ((0, 192, 64), (192, 192, 0)),
)


@dataclass(frozen=True)
class RouteFixture:
    """Paths written by one call, plus the pre-encode sample frame.

    ``sample_frame`` is BGR uint8, the buffer for ``sample_camera_id`` at
    ``sample_frame_index`` before mp4v. ``palette_bgr`` includes the
    background and every band colour drawn in the clip.
    """

    out_dir: Path
    clip_paths: tuple[Path, ...]
    cameras_path: Path
    topology_path: Path
    truth_path: Path
    manifest_path: Path
    sample_frame: npt.NDArray[np.uint8]
    sample_camera_id: str
    sample_frame_index: int
    background_bgr: tuple[int, int, int]
    palette_bgr: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class _Args:
    out_dir: Path
    seed: int
    n_cameras: int
    n_figures: int
    fps: int
    width: int
    height: int
    seconds: int
    n_frames: int


@dataclass(frozen=True)
class _Visit:
    camera_index: int
    t_entry: float
    t_exit: float


@dataclass(frozen=True)
class _Box:
    camera_index: int
    frame_index: int
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class _Figure:
    figure_index: int
    upper_bgr: tuple[int, int, int]
    lower_bgr: tuple[int, int, int]
    distractor: bool
    visits: tuple[_Visit, ...]
    boxes: tuple[_Box, ...]


@dataclass(frozen=True)
class _Plan:
    args: _Args
    offsets: tuple[float, ...]
    edges: tuple[TransitEdge, ...]
    figures: tuple[_Figure, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _require_path(out_dir: object) -> Path:
    if not isinstance(out_dir, Path):
        raise TypeError(f"out_dir must be a Path, got {type(out_dir).__name__}")
    return out_dir


def _reject_bool(label: str, value: object) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an int, got {value!r}")


def _require_int(label: str, value: object, low: int, high: int) -> int:
    """Reject a bool before the int check. ``bool`` is a subclass of ``int``."""
    _reject_bool(label, value)
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an int, got {value!r}")
    if value < low:
        raise ValueError(f"{label} must be in {low}..{high}, got {value}")
    if value > high:
        raise ValueError(f"{label} must be in {low}..{high}, got {value}")
    return value


def _require_size(size: object) -> tuple[int, int]:
    if not isinstance(size, tuple):
        raise TypeError(f"size must be a (width, height) tuple, got {size!r}")
    if len(size) != 2:
        raise ValueError(f"size must be a (width, height) pair, got {size!r}")
    width = _require_int("width", size[0], _MIN_EDGE, _MAX_EDGE)
    height = _require_int("height", size[1], _MIN_EDGE, _MAX_EDGE)
    return width, height


def _require_frames(seconds: int, fps: int) -> int:
    frames = seconds * fps
    if frames < _MIN_FRAMES:
        raise ValueError(f"seconds * fps must be >= {_MIN_FRAMES}, got {frames}")
    return frames


def _resolved_outside_repo(out_dir: Path) -> Path:
    """Resolve ``..`` and symlinks, then refuse the repository tree."""
    resolved = out_dir.expanduser().resolve()
    root = _repo_root()
    if resolved == root:
        raise ValueError(f"out_dir is inside the repository: {resolved}")
    if root in resolved.parents:
        raise ValueError(f"out_dir is inside the repository: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"out_dir must be a directory, got {resolved}")
    return resolved


def _validate(
    out_dir: object,
    seed: object,
    n_cameras: object,
    n_figures: object,
    fps: object,
    size: object,
    seconds: object,
) -> _Args:
    path = _require_path(out_dir)
    seed_i = _require_int("seed", seed, _MIN_SEED, _MAX_SEED)
    cameras = _require_int("n_cameras", n_cameras, _MIN_CAMERAS, _MAX_CAMERAS)
    figures = _require_int("n_figures", n_figures, _MIN_FIGURES, _MAX_FIGURES)
    fps_i = _require_int("fps", fps, _MIN_FPS, _MAX_FPS)
    width, height = _require_size(size)
    seconds_i = _require_int("seconds", seconds, _MIN_SECONDS, _MAX_SECONDS)
    frames = _require_frames(seconds_i, fps_i)
    resolved = _resolved_outside_repo(path)
    return _Args(resolved, seed_i, cameras, figures, fps_i, width, height, seconds_i, frames)


def _camera_id(index: int) -> str:
    return f"CAM-{index + 1}"


def _figure_id(index: int) -> str:
    return f"F{index + 1}"


def _clock_offsets(n_cameras: int) -> tuple[float, ...]:
    """Spread planted offsets from -2.0 s to +3.0 s across the cameras."""
    step = 5.0 / float(n_cameras - 1)
    return tuple(-2.0 + step * float(index) for index in range(n_cameras))


def _zones(n_cameras: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    split = n_cameras // 2
    return tuple(range(split)), tuple(range(split, n_cameras))


def _edge(left: int, right: int) -> TransitEdge:
    return TransitEdge(_camera_id(left), _camera_id(right), _WINDOW_MIN, _WINDOW_MAX)


def _edges(n_cameras: int) -> tuple[TransitEdge, ...]:
    """Two corridors. Cameras in different corridors are not adjacent."""
    found: list[TransitEdge] = []
    for zone in _zones(n_cameras):
        pairs = zip(zone, zone[1:])
        for left, right in pairs:
            found.append(_edge(left, right))
            found.append(_edge(right, left))
    return tuple(found)


def _frames_in_visit(
    start: float,
    end: float,
    offset: float,
    fps: int,
    n_frames: int,
) -> tuple[int, ...]:
    found: list[int] = []
    for frame_index in range(n_frames):
        corrected = frame_index / float(fps) + offset
        if start <= corrected < end:
            found.append(frame_index)
    return tuple(found)


def _schedule_path(
    path: tuple[int, ...],
    offsets: tuple[float, ...],
    seconds: int,
    fps: int,
) -> tuple[tuple[int, float, float], ...] | None:
    frame = 1.0 / float(fps)
    dwell = max(frame, min(1.0, float(seconds) / float(len(path) + 1)))
    visits: list[tuple[int, float, float]] = []
    n_frames = seconds * fps
    for camera in path:
        low = offsets[camera]
        high = low + float(seconds)
        if visits:
            start = max(low, visits[-1][2] + _WINDOW_MIN)
        else:
            start = low
        end = start + dwell
        if end > high:
            end = high
        frames = _frames_in_visit(start, end, offsets[camera], fps, n_frames)
        if not frames:
            return None
        if visits and start - visits[-1][2] < _WINDOW_MIN:
            return None
        visits.append((camera, start, end))
    return tuple(visits)


def _fit_path(
    zone: tuple[int, ...],
    offsets: tuple[float, ...],
    seconds: int,
    fps: int,
) -> tuple[_Visit, ...]:
    length = len(zone)
    while length >= 2:
        planned = _schedule_path(zone[:length], offsets, seconds, fps)
        if planned is not None:
            return tuple(_Visit(camera, start, end) for camera, start, end in planned)
        length -= 1
    raise RuntimeError("validated camera count and duration did not produce a route")


def _colours(
    figure_index: int,
    seed: int,
    route_count: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if figure_index == route_count:
        source = 0
    else:
        source = figure_index
    shift = seed % len(_COLOUR_PAIRS)
    return _COLOUR_PAIRS[(source + shift) % len(_COLOUR_PAIRS)]


def _sample_indices(n_frames: int) -> tuple[int, ...]:
    """Frame indices ``perceptual_hashes`` reads. Truncation matches NumPy."""
    count = min(10, n_frames)
    chosen = np.linspace(0, n_frames - 1, count, dtype=int)
    return tuple(int(index) for index in chosen)


def _lane_layout(height: int, count: int) -> tuple[int, int, int]:
    """Gap, even lane height, and a margin so a lane does not touch the frame edge."""
    for margin in (8, 4, 2, 0):
        inner = height - 2 * margin
        if inner < count * 2:
            continue
        gap = 8
        while gap >= 0:
            usable = inner - gap * (count - 1)
            fig_h = usable // count
            if fig_h % 2 == 1:
                fig_h -= 1
            fits = fig_h >= 2 and fig_h * count + gap * (count - 1) <= inner
            if fits:
                return gap, fig_h, margin
            gap -= 2
    raise RuntimeError("validated frame height did not fit the figures")


def _fig_width(width: int) -> int:
    """Wide enough that a short slide still changes a sampled pHash."""
    fig_w = max(16, width // 8)
    if width - fig_w < 16:
        fig_w = max(8, width - 16)
    if fig_w % 2 == 1:
        fig_w -= 1
    return max(fig_w, 8)


def _origin_y(height: int, count: int, fig_h: int, gap: int, margin: int) -> int:
    used = count * fig_h + (count - 1) * gap
    extra = height - 2 * margin - used
    return margin + max(extra, 0) // 2


def _max_x(width: int, fig_w: int) -> int:
    """Right edge of the slide, four pixels in from the frame when it fits."""
    margin = 4 if width - fig_w >= 12 else 0
    return max(0, width - fig_w - margin)


def _stepped_x(
    frame_index: int,
    n_frames: int,
    max_x: int,
    camera_index: int,
    n_cameras: int,
) -> int:
    """Constant step of at most four pixels. Short clips use this path."""
    step = min(_MAX_STEP, max(1, max_x // (n_frames - 1)))
    span = step * (n_frames - 1)
    spare = max(0, max_x - span)
    phase = 0
    if n_cameras > 1 and spare > 0:
        phase = camera_index * spare // (n_cameras - 1)
        phase -= phase % 2
    return phase + step * frame_index


def _held_x(
    frame_index: int,
    n_frames: int,
    max_x: int,
    camera_index: int,
    n_cameras: int,
) -> int:
    """Hold a column for the sampled frame so mp4v is not mid-slide."""
    samples = _sample_indices(n_frames)
    phase_room = min(8 * max(n_cameras - 1, 0), max(0, max_x // 4))
    span = max(0, max_x - phase_room)
    phase = 0
    if n_cameras > 1 and phase_room > 0:
        phase = camera_index * phase_room // (n_cameras - 1)
    active = 0
    for slot, sample in enumerate(samples):
        previous = samples[slot - 1] if slot else 0
        lead = 0 if sample - previous <= 2 else 2
        if frame_index >= sample - lead:
            active = slot
    slots = max(1, len(samples) - 1)
    return phase + span * active // slots


def _x_at(
    frame_index: int,
    n_frames: int,
    width: int,
    fig_w: int,
    seed: int,
    camera_index: int,
    n_cameras: int,
) -> int:
    """An odd seed walks the other way. Each camera starts at a different column."""
    limit = _max_x(width, fig_w)
    if n_frames <= 1 or limit <= 0:
        return 0
    if _MAX_STEP * (n_frames - 1) <= limit:
        along = _stepped_x(frame_index, n_frames, limit, camera_index, n_cameras)
    else:
        along = _held_x(frame_index, n_frames, limit, camera_index, n_cameras)
    if seed % 2 == 1:
        return limit - along
    return along


def _zone_slots(
    zone: tuple[int, ...],
    offsets: tuple[float, ...],
    fps: int,
    n_frames: int,
) -> list[tuple[float, int]]:
    slots: list[tuple[float, int]] = []
    for camera in zone:
        offset = offsets[camera]
        for frame_index in _sample_indices(n_frames):
            slots.append((frame_index / float(fps) + offset, camera))
    slots.sort()
    return slots


def _cameras_between(origin: int, dest: int) -> tuple[int, ...]:
    step = 1 if dest > origin else -1
    return tuple(range(origin + step, dest, step))


def _latest_bridge(
    camera: int,
    entry: float,
    offsets: tuple[float, ...],
    fps: int,
    n_frames: int,
    seconds: int,
) -> tuple[float, float] | None:
    """One frame on ``camera`` that finishes at least ``_WINDOW_MIN`` before ``entry``."""
    offset = offsets[camera]
    latest = entry - _WINDOW_MIN - _FRAME_EPS
    instant: float | None = None
    for frame_index in range(n_frames - 1, -1, -1):
        candidate = frame_index / float(fps) + offset
        if offset <= candidate <= latest:
            instant = candidate
            break
    if instant is None:
        return None
    end = instant + _FRAME_EPS
    if entry - end > _WINDOW_MAX:
        end = entry - _WINDOW_MIN
    high = offset + float(seconds)
    if end > high or instant >= end:
        return None
    return instant, end


def _exit_time(last_instant: float, high: float, next_start: float) -> float | None:
    latest = min(high, next_start - _WINDOW_MIN)
    earliest = next_start - _WINDOW_MAX
    if latest <= last_instant:
        return None
    exit_at = max(earliest, last_instant + _FRAME_EPS)
    if exit_at > latest:
        return None
    return exit_at


def _last_instant(
    camera: int,
    start: float,
    end: float,
    offsets: tuple[float, ...],
    fps: int,
    n_frames: int,
) -> float | None:
    frames = _frames_in_visit(start, end, offsets[camera], fps, n_frames)
    if not frames:
        return None
    return frames[-1] / float(fps) + offsets[camera]


def _open_span(
    spans: list[list[float]],
    camera: int,
    instant: float,
    offsets: tuple[float, ...],
    seconds: int,
) -> bool:
    low = offsets[camera]
    high = low + float(seconds)
    if instant < low or instant >= high:
        return False
    end = min(high, instant + _FRAME_EPS)
    if instant >= end:
        return False
    spans.append([float(camera), instant, end])
    return True


def _extend_open_span(
    spans: list[list[float]],
    instant: float,
    offsets: tuple[float, ...],
    seconds: int,
) -> bool:
    camera = int(spans[-1][0])
    start = spans[-1][1]
    high = offsets[camera] + float(seconds)
    if instant < start or instant >= high:
        return False
    new_end = min(high, instant + _FRAME_EPS)
    if new_end > spans[-1][2]:
        spans[-1][2] = new_end
    return True


def _append_transit(
    spans: list[list[float]],
    zone: tuple[int, ...],
    camera: int,
    instant: float,
    offsets: tuple[float, ...],
    fps: int,
    n_frames: int,
    seconds: int,
) -> bool:
    origin = int(spans[-1][0])
    if origin not in zone or camera not in zone:
        return False
    mids = _cameras_between(origin, camera)
    high = offsets[camera] + float(seconds)
    end = min(high, instant + _FRAME_EPS)
    if instant >= end:
        return False
    planned: list[tuple[int, float, float]] = [(camera, instant, end)]
    for mid in reversed(mids):
        bridge = _latest_bridge(mid, planned[-1][1], offsets, fps, n_frames, seconds)
        if bridge is None:
            return False
        planned.append((mid, bridge[0], bridge[1]))
    prev_cam = int(spans[-1][0])
    prev_start = spans[-1][1]
    prev_end = spans[-1][2]
    last_instant = _last_instant(prev_cam, prev_start, prev_end, offsets, fps, n_frames)
    if last_instant is None:
        return False
    exit_at = _exit_time(last_instant, offsets[prev_cam] + float(seconds), planned[-1][1])
    if exit_at is None or not prev_start <= last_instant < exit_at:
        return False
    spans[-1][2] = exit_at
    for mid_cam, start, stop in reversed(planned):
        spans.append([float(mid_cam), start, stop])
    return True


def _cap_end(end: float, high: float, next_start: float | None) -> float:
    limit = high
    if next_start is not None:
        limit = min(limit, next_start - _WINDOW_MIN)
        if next_start - limit < _WINDOW_MIN:
            limit = math.nextafter(limit, -math.inf)
    if limit > end:
        return limit
    return end


def _stretch_spans(
    spans: list[list[float]],
    offsets: tuple[float, ...],
    seconds: int,
) -> tuple[tuple[int, float, float], ...]:
    stretched: list[tuple[int, float, float]] = []
    for index, span in enumerate(spans):
        camera = int(span[0])
        start = span[1]
        end = span[2]
        high = offsets[camera] + float(seconds)
        next_start = spans[index + 1][1] if index + 1 < len(spans) else None
        stretched.append((camera, start, _cap_end(end, high, next_start)))
    return tuple(stretched)


def _spans_are_linked(spans: tuple[tuple[int, float, float], ...]) -> bool:
    pairs = zip(spans, spans[1:])
    for previous, following in pairs:
        if abs(previous[0] - following[0]) != 1:
            return False
        gap = following[1] - previous[2]
        if gap < _WINDOW_MIN - 1e-9 or gap > _WINDOW_MAX + 1e-9:
            return False
    return True


def _visits_from_spans(
    spans: tuple[tuple[int, float, float], ...],
    offsets: tuple[float, ...],
    fps: int,
    n_frames: int,
) -> tuple[_Visit, ...] | None:
    if len(spans) < 2 or not _spans_are_linked(spans):
        return None
    visits: list[_Visit] = []
    for camera, start, end in spans:
        frames = _frames_in_visit(start, end, offsets[camera], fps, n_frames)
        if not frames:
            return None
        if end > offsets[camera] + float(n_frames) / float(fps) + 1e-9:
            return None
        visits.append(_Visit(camera, start, end))
    return tuple(visits)


def _cover_zone(
    zone: tuple[int, ...],
    offsets: tuple[float, ...],
    seconds: int,
    fps: int,
    n_frames: int,
) -> tuple[_Visit, ...] | None:
    """One walk that is present at every sampled frame of ``zone``.

    Returns None when the transit windows cannot hold that walk. Callers then
    use a shorter route. An empty sample frame hashes like every other empty
    frame, so this walk is what keeps the 10 pHashes distinct.
    """
    spans: list[list[float]] = []
    for instant, camera in _zone_slots(zone, offsets, fps, n_frames):
        if not spans:
            opened = _open_span(spans, camera, instant, offsets, seconds)
            if not opened:
                return None
            continue
        if int(spans[-1][0]) == camera:
            extended = _extend_open_span(spans, instant, offsets, seconds)
            if not extended:
                return None
            continue
        moved = _append_transit(spans, zone, camera, instant, offsets, fps, n_frames, seconds)
        if not moved:
            return None
    if not spans:
        return None
    return _visits_from_spans(_stretch_spans(spans, offsets, seconds), offsets, fps, n_frames)


def _pair_zone(zone: tuple[int, ...], index: int) -> tuple[int, ...]:
    if len(zone) < 3:
        return zone
    start = index % (len(zone) - 1)
    return zone[start : start + 2]


def _shift_visits(
    visits: tuple[_Visit, ...],
    delay: float,
    offsets: tuple[float, ...],
    seconds: int,
    fps: int,
    n_frames: int,
) -> tuple[_Visit, ...] | None:
    if delay == 0:
        return visits
    shifted: list[_Visit] = []
    for visit in visits:
        start = visit.t_entry + delay
        end = visit.t_exit + delay
        low = offsets[visit.camera_index]
        high = low + float(seconds)
        if end > high:
            start -= end - high
            end = high
        if start < low:
            return None
        frames = _frames_in_visit(start, end, low, fps, n_frames)
        if not frames:
            return None
        shifted.append(_Visit(visit.camera_index, start, end))
    for previous, following in zip(shifted, shifted[1:]):
        gap = following.t_entry - previous.t_exit
        if gap < _WINDOW_MIN or gap > _WINDOW_MAX:
            return None
    return tuple(shifted)


def _route_key(visits: tuple[_Visit, ...]) -> tuple[tuple[int, float, float], ...]:
    return tuple(
        (visit.camera_index, round(visit.t_entry, 6), round(visit.t_exit, 6)) for visit in visits
    )


def _delayed_until_distinct(
    base: tuple[_Visit, ...],
    offsets: tuple[float, ...],
    seconds: int,
    fps: int,
    n_frames: int,
    index: int,
    taken: set[tuple[tuple[int, float, float], ...]],
) -> tuple[_Visit, ...]:
    step = 1.0 / float(fps)
    attempt = index
    while attempt < index + 8:
        shifted = _shift_visits(base, attempt * step, offsets, seconds, fps, n_frames)
        attempt += 1
        if shifted is None:
            continue
        key = _route_key(shifted)
        if key in taken:
            continue
        return shifted
    return base


def _fallback_route(
    zone: tuple[int, ...],
    offsets: tuple[float, ...],
    seconds: int,
    fps: int,
    n_frames: int,
    index: int,
    taken: set[tuple[tuple[int, float, float], ...]],
) -> tuple[_Visit, ...]:
    try:
        base = _fit_path(_pair_zone(zone, index), offsets, seconds, fps)
    except RuntimeError:
        base = _fit_path(zone, offsets, seconds, fps)
    return _delayed_until_distinct(base, offsets, seconds, fps, n_frames, index, taken)


def _routes_for_zone(
    zone: tuple[int, ...],
    offsets: tuple[float, ...],
    seconds: int,
    fps: int,
    n_frames: int,
    count: int,
) -> tuple[tuple[_Visit, ...], ...]:
    # A zone that cannot host a two-camera walk is refused here, before files.
    _fit_path(zone, offsets, seconds, fps)
    covered = _cover_zone(zone, offsets, seconds, fps, n_frames)
    taken: set[tuple[tuple[int, float, float], ...]] = set()
    routes: list[tuple[_Visit, ...]] = []
    for index in range(count):
        if index == 0 and covered is not None:
            route = covered
        else:
            route = _fallback_route(zone, offsets, seconds, fps, n_frames, index, taken)
        taken.add(_route_key(route))
        routes.append(route)
    return tuple(routes)


def _boxes_for(
    visits: tuple[_Visit, ...],
    args: _Args,
    offsets: tuple[float, ...],
    y1: int,
    fig_w: int,
    fig_h: int,
) -> tuple[_Box, ...]:
    boxes: list[_Box] = []
    for visit in visits:
        frames = _frames_in_visit(
            visit.t_entry,
            visit.t_exit,
            offsets[visit.camera_index],
            args.fps,
            args.n_frames,
        )
        for frame_index in frames:
            x1 = _x_at(
                frame_index,
                args.n_frames,
                args.width,
                fig_w,
                args.seed,
                visit.camera_index,
                args.n_cameras,
            )
            boxes.append(
                _Box(
                    visit.camera_index,
                    frame_index,
                    x1,
                    y1,
                    x1 + fig_w,
                    y1 + fig_h,
                )
            )
    return tuple(boxes)


def _build_figure(
    index: int,
    route_count: int,
    visits: tuple[_Visit, ...],
    args: _Args,
    offsets: tuple[float, ...],
    y1: int,
    fig_w: int,
    fig_h: int,
) -> _Figure:
    upper, lower = _colours(index, args.seed, route_count)
    return _Figure(
        figure_index=index,
        upper_bgr=upper,
        lower_bgr=lower,
        distractor=index == route_count,
        visits=visits,
        boxes=_boxes_for(visits, args, offsets, y1, fig_w, fig_h),
    )


def _build_plan(args: _Args) -> _Plan:
    offsets = _clock_offsets(args.n_cameras)
    left_zone, right_zone = _zones(args.n_cameras)
    left_count = args.n_figures // 2
    right_count = args.n_figures - left_count
    gap, fig_h, margin = _lane_layout(args.height, args.n_figures)
    fig_w = _fig_width(args.width)
    origin = _origin_y(args.height, args.n_figures, fig_h, gap, margin)
    left = _routes_for_zone(left_zone, offsets, args.seconds, args.fps, args.n_frames, left_count)
    right = _routes_for_zone(
        right_zone, offsets, args.seconds, args.fps, args.n_frames, right_count
    )
    routes = left + right
    route_count = args.n_figures - 1
    figures = tuple(
        _build_figure(
            index,
            route_count,
            routes[index],
            args,
            offsets,
            origin + index * (fig_h + gap),
            fig_w,
            fig_h,
        )
        for index in range(args.n_figures)
    )
    return _Plan(args=args, offsets=offsets, edges=_edges(args.n_cameras), figures=figures)


def _camera_config(index: int, offset: float) -> CameraConfig:
    return CameraConfig(
        camera_id=_camera_id(index),
        brand="synthetic",
        site_map_x=float(index * 10),
        site_map_y=0.0,
        clock_offset_seconds=offset,
    )


def _registry(plan: _Plan) -> CameraRegistry:
    registry = CameraRegistry()
    for index, offset in enumerate(plan.offsets):
        registry.register(_camera_config(index, offset))
    return registry


def _paint(
    frame: npt.NDArray[np.uint8],
    box: _Box,
    upper: tuple[int, int, int],
    lower: tuple[int, int, int],
) -> None:
    mid = box.y1 + (box.y2 - box.y1) // 2
    frame[box.y1 : mid, box.x1 : box.x2] = upper
    frame[mid : box.y2, box.x1 : box.x2] = lower


def _index_boxes(
    plan: _Plan,
) -> dict[tuple[int, int], tuple[tuple[_Box, _Figure], ...]]:
    grouped: dict[tuple[int, int], list[tuple[_Box, _Figure]]] = {}
    for figure in plan.figures:
        for box in figure.boxes:
            grouped.setdefault((box.camera_index, box.frame_index), []).append((box, figure))
    return {key: tuple(value) for key, value in grouped.items()}


def _render_frame(
    plan: _Plan,
    camera_index: int,
    frame_index: int,
    indexed: dict[tuple[int, int], tuple[tuple[_Box, _Figure], ...]],
) -> npt.NDArray[np.uint8]:
    frame = np.full(
        (plan.args.height, plan.args.width, 3),
        _BACKGROUND_BGR,
        dtype=np.uint8,
    )
    for box, figure in indexed.get((camera_index, frame_index), ()):
        _paint(frame, box, figure.upper_bgr, figure.lower_bgr)
    return frame


def _open_writer(path: Path, fps: int, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    return writer


def _box_header(data: bytearray, pos: int, end: int) -> tuple[int, int]:
    size = int.from_bytes(data[pos : pos + 4], "big")
    if size == 1:
        large = int.from_bytes(data[pos + 8 : pos + 16], "big")
        return large, 16
    if size == 0:
        return end - pos, 8
    return size, 8


def _zero_timed_box(data: bytearray, payload: int) -> None:
    if payload >= len(data):
        return
    version = data[payload]
    if version == 0 and payload + 12 <= len(data):
        data[payload + 4 : payload + 12] = b"\x00" * 8
        return
    if version == 1 and payload + 20 <= len(data):
        data[payload + 4 : payload + 20] = b"\x00" * 16


def _zero_boxes(data: bytearray, start: int, end: int) -> None:
    pos = start
    while pos + 8 <= end:
        size, header = _box_header(data, pos, end)
        if size < header or pos + size > end:
            return
        kind = bytes(data[pos + 4 : pos + 8])
        payload = pos + header
        if kind in _CONTAINER_BOXES:
            _zero_boxes(data, payload, pos + size)
        elif kind in _TIMED_BOXES:
            _zero_timed_box(data, payload)
        pos += size


def _zero_mp4_times(path: Path) -> None:
    """Set mvhd, tkhd, and mdhd creation and modification times to 0."""
    data = bytearray(path.read_bytes())
    _zero_boxes(data, 0, len(data))
    path.write_bytes(data)


def _write_clip(
    path: Path,
    plan: _Plan,
    camera_index: int,
    indexed: dict[tuple[int, int], tuple[tuple[_Box, _Figure], ...]],
) -> None:
    writer = _open_writer(path, plan.args.fps, plan.args.width, plan.args.height)
    try:
        for frame_index in range(plan.args.n_frames):
            writer.write(_render_frame(plan, camera_index, frame_index, indexed))
    finally:
        writer.release()
    _zero_mp4_times(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def _truth_payload(plan: _Plan) -> dict[str, Any]:
    figures: list[dict[str, Any]] = []
    for figure in plan.figures:
        shares = _figure_id(0) if figure.distractor else None
        figures.append(
            {
                "boxes": [
                    {
                        "camera_id": _camera_id(box.camera_index),
                        "frame_index": box.frame_index,
                        "x1": box.x1,
                        "x2": box.x2,
                        "y1": box.y1,
                        "y2": box.y2,
                    }
                    for box in figure.boxes
                ],
                "distractor": figure.distractor,
                "figure_id": _figure_id(figure.figure_index),
                "lower_bgr": list(figure.lower_bgr),
                "shares_colours_with": shares,
                "upper_bgr": list(figure.upper_bgr),
                "visits": [
                    {
                        "camera_id": _camera_id(visit.camera_index),
                        "t_entry": visit.t_entry,
                        "t_exit": visit.t_exit,
                    }
                    for visit in figure.visits
                ],
            }
        )
    return {
        "background_bgr": list(_BACKGROUND_BGR),
        "figures": figures,
        "fps": plan.args.fps,
        "height": plan.args.height,
        "seconds": plan.args.seconds,
        "seed": plan.args.seed,
        "width": plan.args.width,
    }


def _seed_text(plan: _Plan, camera_index: int) -> str:
    payload = {
        "camera_index": camera_index,
        "fps": plan.args.fps,
        "height": plan.args.height,
        "n_cameras": plan.args.n_cameras,
        "n_figures": plan.args.n_figures,
        "seconds": plan.args.seconds,
        "seed": plan.args.seed,
        "width": plan.args.width,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _clip_row(plan: _Plan, camera_index: int, digest: str, phashes: list[str]) -> dict[str, Any]:
    fixture_id = f"FX-CODE-{101 + camera_index}"
    clip_name = f"CAM-{camera_index + 1}.mp4"
    return {
        "fixture_id": fixture_id,
        "generator": "generate_route_fixture",
        "generator_license": "MIT",
        "generator_version": "1.0.0",
        "license": "MIT",
        "name": f"FX-CODE route clip {fixture_id}",
        "path": clip_name,
        "phash": phashes,
        "seed": _seed_text(plan, camera_index),
        "sha256": digest,
        "source_url": _SOURCE_URL,
        "type": "code_fixture",
        "use": "CODE_TEST",
        "version": "1.0.0",
        "notes": (
            "Seeded flat two-band rectangles for one fake camera. "
            "CODE_TEST. Temp file, never committed."
        ),
    }


def _write_manifest(out_dir: Path, plan: _Plan) -> None:
    entries: list[dict[str, Any]] = []
    for camera_index in range(plan.args.n_cameras):
        path = out_dir / f"CAM-{camera_index + 1}.mp4"
        entries.append(
            _clip_row(
                plan,
                camera_index,
                sha256_file(path),
                perceptual_hashes(path),
            )
        )
    _write_json(
        out_dir / "manifest.json",
        {
            "entries": entries,
            "last_updated_ist": "2026-10-27",
            "manifest_version": "1.0",
        },
    )


def _write_outputs(out_dir: Path, plan: _Plan) -> None:
    indexed = _index_boxes(plan)
    for camera_index in range(plan.args.n_cameras):
        _write_clip(out_dir / f"CAM-{camera_index + 1}.mp4", plan, camera_index, indexed)
    registry = _registry(plan)
    _write_json(
        out_dir / "cameras.json",
        {"cameras": [camera.to_dict() for camera in registry.list_cameras()]},
    )
    topology = CameraTopology(registry, plan.edges)
    _write_json(out_dir / "topology.json", topology.to_dict())
    _write_json(out_dir / "truth.json", _truth_payload(plan))
    _write_manifest(out_dir, plan)


def _palette(plan: _Plan) -> tuple[tuple[int, int, int], ...]:
    colours: list[tuple[int, int, int]] = [_BACKGROUND_BGR]
    for figure in plan.figures:
        if figure.upper_bgr not in colours:
            colours.append(figure.upper_bgr)
        if figure.lower_bgr not in colours:
            colours.append(figure.lower_bgr)
    return tuple(colours)


def _sample_point(plan: _Plan) -> tuple[int, int]:
    figure = plan.figures[0]
    box = figure.boxes[0]
    return box.camera_index, box.frame_index


def _result(out_dir: Path, plan: _Plan) -> RouteFixture:
    camera_index, frame_index = _sample_point(plan)
    sample = _render_frame(plan, camera_index, frame_index, _index_boxes(plan))
    clips = tuple(out_dir / f"CAM-{index + 1}.mp4" for index in range(plan.args.n_cameras))
    return RouteFixture(
        out_dir=out_dir,
        clip_paths=clips,
        cameras_path=out_dir / "cameras.json",
        topology_path=out_dir / "topology.json",
        truth_path=out_dir / "truth.json",
        manifest_path=out_dir / "manifest.json",
        sample_frame=sample,
        sample_camera_id=_camera_id(camera_index),
        sample_frame_index=frame_index,
        background_bgr=_BACKGROUND_BGR,
        palette_bgr=_palette(plan),
    )


def generate_route_fixture(
    out_dir: Path,
    seed: int = 0,
    n_cameras: int = 6,
    n_figures: int = 4,
    fps: int = 10,
    size: tuple[int, int] = (640, 360),
    seconds: int = 30,
) -> RouteFixture:
    """Write one seeded route fixture and return its paths.

    Validation runs before any file is created. ``out_dir`` is resolved,
    including ``..`` and symlinks, and must sit outside the repository.
    ``seed``, ``n_cameras``, ``n_figures``, ``fps``, each side of ``size``,
    and ``seconds`` are ints. A bool is rejected. ``seconds * fps`` must be
    at least 10 so each clip can carry 10 perceptual hashes.

    Ranges: seed ``0..2147483647``, cameras ``4..12``, figures ``2..8``,
    fps ``1..60``, seconds ``1..120``, and each side of ``size`` ``64..1920``.
    """
    args = _validate(out_dir, seed, n_cameras, n_figures, fps, size, seconds)
    plan = _build_plan(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_outputs(args.out_dir, plan)
    return _result(args.out_dir, plan)
