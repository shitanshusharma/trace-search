"""Seeded flat-rectangle routes for six fake cameras.

``generate_route_fixture`` writes ``CAM-1.mp4`` .. ``CAM-N.mp4`` (mp4v),
``cameras.json``, ``topology.json``, ``truth.json``, and ``manifest.json``
under ``out_dir``. Figures are upright rectangles with two flat colour
bands. There is no head, no limbs, and no text. One figure copies another
figure's colour pair and walks the other corridor, which has no transit edge.

Each camera also draws one flat two-band rectangle in a reserved column on
the right. Its vertical position is the frame index. The rectangle does not
cover figure pixels. It is recorded under ``marks`` in ``truth.json``. The
temp manifest stores 10 pHashes, and the schema requires those 10 values to
be unique, so a camera with only a still background cannot be hashed as-is.

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
_MARK_WIDTH = 16
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


def _fig_size(width: int, height: int, n_figures: int) -> tuple[int, int]:
    fig_h = min(32, height // n_figures)
    if fig_h % 2 == 1:
        fig_h -= 1
    fig_w = min(16, width // 5)
    return fig_w, fig_h


def _x_positions(count: int, width: int, fig_w: int, seed: int) -> tuple[int, ...]:
    """Horizontal positions in the area left of the frame-index mark."""
    usable = width - _MARK_WIDTH
    start = seed % 5
    end = usable - fig_w
    if seed % 2 == 1:
        start, end = end, start
    if count <= 1:
        return (start,)
    span = end - start
    return tuple(start + span * index // (count - 1) for index in range(count))


def _mark_pair(seed: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return _COLOUR_PAIRS[(seed + 3) % len(_COLOUR_PAIRS)]


def _mark_height(height: int) -> int:
    block = height // 2
    if block % 2 == 1:
        block -= 1
    return block


def _mark_box(frame_index: int, n_frames: int, width: int, height: int) -> _Box:
    block = _mark_height(height)
    travel = height - block
    y1 = frame_index * travel // (n_frames - 1)
    x1 = width - _MARK_WIDTH
    return _Box(-1, frame_index, x1, y1, width, y1 + block)


def _boxes_for(
    visits: tuple[_Visit, ...],
    args: _Args,
    offsets: tuple[float, ...],
    figure_index: int,
    fig_w: int,
    fig_h: int,
) -> tuple[_Box, ...]:
    y1 = figure_index * fig_h
    y2 = y1 + fig_h
    boxes: list[_Box] = []
    for visit in visits:
        frames = _frames_in_visit(
            visit.t_entry,
            visit.t_exit,
            offsets[visit.camera_index],
            args.fps,
            args.n_frames,
        )
        xs = _x_positions(len(frames), args.width, fig_w, args.seed)
        for frame_index, x1 in zip(frames, xs):
            boxes.append(
                _Box(
                    visit.camera_index,
                    frame_index,
                    x1,
                    y1,
                    x1 + fig_w,
                    y2,
                )
            )
    return tuple(boxes)


def _build_figure(
    index: int,
    route_count: int,
    visits: tuple[_Visit, ...],
    args: _Args,
    offsets: tuple[float, ...],
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
        boxes=_boxes_for(visits, args, offsets, index, fig_w, fig_h),
    )


def _build_plan(args: _Args) -> _Plan:
    offsets = _clock_offsets(args.n_cameras)
    left_zone, right_zone = _zones(args.n_cameras)
    left = _fit_path(left_zone, offsets, args.seconds, args.fps)
    right = _fit_path(right_zone, offsets, args.seconds, args.fps)
    fig_w, fig_h = _fig_size(args.width, args.height, args.n_figures)
    route_count = args.n_figures - 1
    figures = tuple(
        _build_figure(
            index,
            route_count,
            right if index == route_count else left,
            args,
            offsets,
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


def _paint_mark(frame: npt.NDArray[np.uint8], plan: _Plan, frame_index: int) -> None:
    box = _mark_box(frame_index, plan.args.n_frames, plan.args.width, plan.args.height)
    upper, lower = _mark_pair(plan.args.seed)
    _paint(frame, box, upper, lower)


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
    _paint_mark(frame, plan, frame_index)
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


def _mark_records(plan: _Plan) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for camera_index in range(plan.args.n_cameras):
        for frame_index in range(plan.args.n_frames):
            box = _mark_box(frame_index, plan.args.n_frames, plan.args.width, plan.args.height)
            records.append(
                {
                    "camera_id": _camera_id(camera_index),
                    "frame_index": frame_index,
                    "x1": box.x1,
                    "x2": box.x2,
                    "y1": box.y1,
                    "y2": box.y2,
                }
            )
    return records


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
    upper, lower = _mark_pair(plan.args.seed)
    return {
        "background_bgr": list(_BACKGROUND_BGR),
        "figures": figures,
        "fps": plan.args.fps,
        "height": plan.args.height,
        "mark_lower_bgr": list(lower),
        "mark_upper_bgr": list(upper),
        "marks": _mark_records(plan),
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
    upper, lower = _mark_pair(plan.args.seed)
    if upper not in colours:
        colours.append(upper)
    if lower not in colours:
        colours.append(lower)
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
