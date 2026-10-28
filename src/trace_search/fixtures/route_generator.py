"""Seeded flat-rectangle routes for six fake cameras.

``generate_route_fixture`` writes ``CAM-1.mp4`` .. ``CAM-N.mp4`` (mp4v),
``cameras.json``, ``topology.json``, ``truth.json``, and ``manifest.json``
under ``out_dir``. Figures are upright rectangles with two flat colour
bands. There is no head, no limbs, and no text. The last figure copies
figure 0's colours and walks the other corridor. Corridors are not joined.

Each figure crosses its corridor once. A visit runs to the end of the
clip, so it lasts nearly the whole recording, and the figure moves across
the frame the entire time. One seeded ``numpy`` Generator picks the colour
shift, the direction of that move, and a per-camera entrance of at most one
second. Clocks step by the clip length plus one second, so those single
passes still land inside the transit window. Sampled frames therefore hold
the rectangle at a different x. Nothing else is drawn.

Only configs that keep that walk inside the transit window are accepted.
``(320, 240)`` and ``(640, 360)`` are accepted on every schedule.
``(1280, 720)`` is accepted for 6 cameras at 10 fps and 30 seconds.
The same seed writes byte-identical files. mp4v on opencv-python-headless
4.10.0.84 (Lavf59.27.100) already stores container times as 0.
``_zero_mp4_times`` sets them to 0 again.

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
_MAX_CAMERAS = 6
_MIN_FIGURES = 2
_MAX_FIGURES = 8
_MIN_FPS = 10
_MAX_FPS = 30
_MIN_SECONDS = 30
_MAX_SECONDS = 40
_MIN_EDGE = 240
_MAX_EDGE = 1280
_WINDOW_MIN = 0.5
_WINDOW_MAX = 8.0
_GAP_SECONDS = 1.0
_GUTTER = 8
_HASH_SAMPLES = 10
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
    start_frame: int
    stop_frame: int
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


@dataclass(frozen=True)
class _Variation:
    """Seed draws. The same seed replays the same Generator stream."""

    reverse: bool
    shift: int
    jitter: tuple[int, ...]


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


def _is_small_frame(width: int, height: int) -> bool:
    if width != 320:
        return False
    return height == 240


def _is_default_frame(width: int, height: int) -> bool:
    if width != 640:
        return False
    return height == 360


def _is_hd_frame(width: int, height: int) -> bool:
    if width != 1280:
        return False
    return height == 720


def _is_hd_schedule(n_cameras: int, seconds: int) -> bool:
    """720p stays on the default schedule so every accepted clip is encoded in tests."""
    if n_cameras != 6:
        return False
    return seconds == 30


def _require_known_size(width: int, height: int, n_cameras: int, seconds: int) -> None:
    if _is_small_frame(width, height):
        return
    if _is_default_frame(width, height):
        return
    if _is_hd_frame(width, height) and _is_hd_schedule(n_cameras, seconds):
        return
    raise ValueError(
        "size must be (320, 240) or (640, 360), or (1280, 720) "
        f"for 6 cameras and 30 seconds, got {(width, height)!r}"
    )


def _ten_fps_span(fps: int, seconds: int) -> bool:
    if fps != 10:
        return False
    if seconds == 30:
        return True
    return seconds == 40


def _thirty_fps_clip(n_cameras: int, fps: int, seconds: int) -> bool:
    if n_cameras != 4:
        return False
    if fps != 30:
        return False
    return seconds == 30


def _require_schedule(n_cameras: int, fps: int, seconds: int) -> None:
    if _ten_fps_span(fps, seconds):
        return
    if _thirty_fps_clip(n_cameras, fps, seconds):
        return
    raise ValueError(
        f"unsupported route schedule for {n_cameras} cameras at {fps} fps and {seconds} seconds"
    )


def _reject_repo(resolved: Path) -> None:
    root = _repo_root()
    if resolved == root:
        raise ValueError(f"out_dir is inside the repository: {resolved}")
    if root in resolved.parents:
        raise ValueError(f"out_dir is inside the repository: {resolved}")


def _reject_file(resolved: Path) -> None:
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"out_dir must be a directory, got {resolved}")


def _resolved_outside_repo(out_dir: Path) -> Path:
    """Resolve ``..`` and symlinks, then refuse the repository tree."""
    resolved = out_dir.expanduser().resolve()
    _reject_repo(resolved)
    _reject_file(resolved)
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
    _require_known_size(width, height, cameras, seconds_i)
    _require_schedule(cameras, fps_i, seconds_i)
    resolved = _resolved_outside_repo(path)
    return _Args(
        resolved,
        seed_i,
        cameras,
        figures,
        fps_i,
        width,
        height,
        seconds_i,
        seconds_i * fps_i,
    )


def _camera_id(index: int) -> str:
    return f"CAM-{index + 1}"


def _figure_id(index: int) -> str:
    return f"F{index + 1}"


def _clock_offsets(n_cameras: int, seconds: int) -> tuple[float, ...]:
    """Planted clocks. Each step is the clip length plus the transit gap.

    A visit that fills the file is then one gap later on the next camera,
    before any seeded entrance delay is added.
    """
    step = float(seconds) + _GAP_SECONDS
    return tuple(-2.0 + step * float(index) for index in range(n_cameras))


def _zones(n_cameras: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    split = n_cameras // 2
    return tuple(range(split)), tuple(range(split, n_cameras))


def _left_figure_count(n_figures: int) -> int:
    return n_figures // 2


def _edge(left: int, right: int) -> TransitEdge:
    return TransitEdge(_camera_id(left), _camera_id(right), _WINDOW_MIN, _WINDOW_MAX)


def _edges(n_cameras: int) -> tuple[TransitEdge, ...]:
    """Two corridors. Each link is the single direction the march walks."""
    found: list[TransitEdge] = []
    for zone in _zones(n_cameras):
        for left, right in zip(zone, zone[1:]):
            found.append(_edge(left, right))
    return tuple(found)


def _sample_indices(n_frames: int) -> tuple[int, ...]:
    """Frame indices ``perceptual_hashes`` reads. Truncation matches NumPy."""
    chosen = np.linspace(0, n_frames - 1, _HASH_SAMPLES, dtype=int)
    return tuple(int(index) for index in chosen)


def _jitter_limit(fps: int, n_frames: int) -> int:
    """At most one second, and early enough that only frame 0 can be empty."""
    second = _sample_indices(n_frames)[1]
    return min(fps, second - 1)


def _slot_start(sample: int, n_frames: int, dwell: int) -> int:
    latest = n_frames - dwell
    if sample < latest:
        return sample
    return latest


def _visit_start(jitter: int, n_frames: int) -> int:
    """Seeded entrance. The clamp keeps a one-frame visit inside the clip."""
    return _slot_start(jitter, n_frames, 1)


def _stamp(offset: float, frame: int, fps: int) -> float:
    return offset + frame / float(fps)


def _make_visit(
    camera: int,
    start: int,
    stop: int,
    offsets: tuple[float, ...],
    fps: int,
) -> _Visit:
    return _Visit(
        camera,
        start,
        stop,
        _stamp(offsets[camera], start, fps),
        _stamp(offsets[camera], stop, fps),
    )


def _variation(seed: int, n_cameras: int, fps: int, n_frames: int) -> _Variation:
    """Draw every seed-dependent choice from one Generator, in this order."""
    rng = np.random.default_rng(seed)
    reverse = int(rng.integers(0, 2)) == 1
    shift = int(rng.integers(0, len(_COLOUR_PAIRS)))
    limit = _jitter_limit(fps, n_frames)
    drawn = rng.integers(0, limit + 1, size=n_cameras)
    jitter = tuple(int(value) for value in drawn)
    return _Variation(reverse, shift, jitter)


def _march(
    zone: tuple[int, ...],
    jitter: tuple[int, ...],
    offsets: tuple[float, ...],
    fps: int,
    n_frames: int,
) -> tuple[_Visit, ...]:
    """One pass. Each camera keeps the figure until the last frame."""
    found: list[_Visit] = []
    for camera in zone:
        start = _visit_start(jitter[camera], n_frames)
        found.append(_make_visit(camera, start, n_frames, offsets, fps))
    return tuple(found)


def _colours(
    figure_index: int,
    shift: int,
    n_figures: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    source = figure_index
    if figure_index == n_figures - 1:
        source = 0
    return _COLOUR_PAIRS[(source + shift) % len(_COLOUR_PAIRS)]


def _lane_geometry(width: int, height: int, count: int) -> tuple[int, int, int, tuple[int, ...]]:
    remain = height - _GUTTER * (count + 1)
    fig_h = remain // count
    fig_h -= fig_h % 4
    fig_w = fig_h // 2
    span = width - fig_w - 2 * _GUTTER
    step = span // 9
    tops = tuple(_GUTTER + index * (fig_h + _GUTTER) for index in range(count))
    return fig_w, fig_h, step, tops


def _x_at(frame: int, n_frames: int, step: int, reverse: bool, camera: int) -> int:
    """Slide across the lane. Odd cameras face the other way."""
    span = step * 9
    progress = frame / (n_frames - 1)
    if reverse != (camera % 2 == 1):
        progress = 1.0 - progress
    shift = (camera // 2) * (step // 2)
    travel = span - (step // 2) * 2
    return _GUTTER + shift + int(progress * travel)


def _boxes_for(
    visits: tuple[_Visit, ...],
    figure_index: int,
    fig_w: int,
    fig_h: int,
    step: int,
    tops: tuple[int, ...],
    reverse: bool,
    n_frames: int,
) -> tuple[_Box, ...]:
    top = tops[figure_index]
    found: list[_Box] = []
    for visit in visits:
        for frame_index in range(visit.start_frame, visit.stop_frame):
            x_left = _x_at(frame_index, n_frames, step, reverse, visit.camera_index)
            found.append(
                _Box(
                    visit.camera_index,
                    frame_index,
                    x_left,
                    top,
                    x_left + fig_w,
                    top + fig_h,
                )
            )
    return tuple(found)


def _build_figures(
    args: _Args,
    offsets: tuple[float, ...],
    variation: _Variation,
) -> tuple[_Figure, ...]:
    zones = _zones(args.n_cameras)
    left = _left_figure_count(args.n_figures)
    fig_w, fig_h, step, tops = _lane_geometry(args.width, args.height, args.n_figures)
    figures: list[_Figure] = []
    for index in range(args.n_figures):
        zone = zones[0] if index < left else zones[1]
        visits = _march(zone, variation.jitter, offsets, args.fps, args.n_frames)
        upper, lower = _colours(index, variation.shift, args.n_figures)
        figures.append(
            _Figure(
                index,
                upper,
                lower,
                index == args.n_figures - 1,
                visits,
                _boxes_for(
                    visits,
                    index,
                    fig_w,
                    fig_h,
                    step,
                    tops,
                    variation.reverse,
                    args.n_frames,
                ),
            )
        )
    return tuple(figures)


def _build_plan(args: _Args) -> _Plan:
    offsets = _clock_offsets(args.n_cameras, args.seconds)
    variation = _variation(args.seed, args.n_cameras, args.fps, args.n_frames)
    figures = _build_figures(args, offsets, variation)
    return _Plan(args, offsets, _edges(args.n_cameras), figures)


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
        if size < header:
            raise ValueError(f"mp4 box size {size} is shorter than its {header}-byte header")
        if pos + size > end:
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
    boxes = plan.figures[0].boxes
    box = boxes[len(boxes) // 2]
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
    and ``seconds`` are ints. A bool is rejected.

    Accepted schedules are 4, 5, or 6 cameras at 10 fps for 30 or 40 seconds,
    plus 4 cameras at 30 fps for 30 seconds. ``n_figures`` is ``2..8`` on
    every schedule. ``(320, 240)`` and ``(640, 360)`` are accepted on every
    schedule. ``(1280, 720)`` is accepted for 6 cameras at 10 fps and 30
    seconds, which includes 8 figures at 720p. Seed stays ``0..2147483647``.
    Colours, the direction of the walk across the frame, and each camera's
    entrance come from ``numpy.random.default_rng(seed)``.
    """
    args = _validate(out_dir, seed, n_cameras, n_figures, fps, size, seconds)
    plan = _build_plan(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_outputs(args.out_dir, plan)
    return _result(args.out_dir, plan)
