"""Seeded route fixtures: one-pass walks, palette, detection, and the licence row.

Generation uses ``tmp_path`` only. The pixel check reads
``RouteFixture.sample_frame``, the BGR buffer before mp4v, because the
codec is lossy. A temp copy of the licence manifest, not the repo file,
carries the planted ``SMOKE_ONLY`` generator row.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

import scripts.check_licenses as check_licenses
from trace_search.detection.scripted_figures import scripted_figures_detector
from trace_search.fixtures import route_generator
from trace_search.fixtures.content import perceptual_hashes
from trace_search.fixtures.guards import (
    FixtureSeparationError,
    LoaderContext,
    check_fixture_allowed,
)
from trace_search.fixtures.route_generator import RouteFixture, generate_route_fixture
from trace_search.fixtures.separation import SeparationCheckError, check_separation
from trace_search.ingest.camera_registry import CameraRegistry
from trace_search.ingest.video_source import Frame
from trace_search.matching.topology import CameraTopology

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / "licenses" / "manifest.json"
_SCHEMA = _REPO / "licenses" / "manifest_schema.json"
_SOURCE = _REPO / "src" / "trace_search" / "fixtures" / "route_generator.py"
_ROW_NAME = "src/trace_search/fixtures/route_generator.py"
_BASE = "8af465e2e3ab3e3f9159bfd993cd0082bec62430"
_SCHEDULES: tuple[tuple[int, int, int], ...] = (
    (4, 10, 30),
    (4, 10, 40),
    (4, 30, 30),
    (5, 10, 30),
    (5, 10, 40),
    (6, 10, 30),
    (6, 10, 40),
)
_ACCEPTED: tuple[tuple[int, int, int, int, tuple[int, int]], ...] = (
    tuple(
        (n_cameras, n_figures, fps, seconds, size)
        for n_cameras, fps, seconds in _SCHEDULES
        for n_figures in (2, 4)
        for size in ((320, 240), (640, 360))
    )
    + tuple((4, n_figures, 10, 30, (320, 240)) for n_figures in (3, 5, 6, 7, 8))
    + (
        (5, 3, 10, 30, (320, 240)),
        (6, 8, 10, 30, (1280, 720)),
    )
)
_FIXTURES: dict[tuple[Any, ...], RouteFixture] = {}


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    """Fail the caller if a pure-Python walk does not return."""

    def handler(signum: int, frame: object) -> None:
        raise TimeoutError(f"call exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _small(out: Path, **overrides: Any) -> RouteFixture:
    params: dict[str, Any] = {
        "seed": 0,
        "n_cameras": 4,
        "n_figures": 2,
        "fps": 10,
        "size": (320, 240),
        "seconds": 30,
    }
    params.update(overrides)
    return generate_route_fixture(out, **params)


def _once(factory: pytest.TempPathFactory, **overrides: Any) -> RouteFixture:
    params: dict[str, Any] = {
        "seed": 0,
        "n_cameras": 6,
        "n_figures": 4,
        "fps": 10,
        "size": (640, 360),
        "seconds": 30,
    }
    params.update(overrides)
    key = (
        params["seed"],
        params["n_cameras"],
        params["n_figures"],
        params["fps"],
        params["size"],
        params["seconds"],
    )
    cached = _FIXTURES.get(key)
    if cached is not None:
        return cached
    fixture = generate_route_fixture(factory.mktemp("route"), **params)
    _FIXTURES[key] = fixture
    return fixture


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _manifest() -> dict[str, Any]:
    return _load(_MANIFEST)


def _generator_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest["entries"]
    assert isinstance(entries, list)
    return [
        entry for entry in entries if isinstance(entry, dict) and entry.get("type") == "generator"
    ]


def _route_row() -> dict[str, Any]:
    rows = [row for row in _generator_rows(_manifest()) if row.get("name") == _ROW_NAME]
    assert len(rows) == 1
    return rows[0]


def _cam1_x(truth: dict[str, Any]) -> list[int]:
    return [box["x1"] for box in truth["figures"][0]["boxes"] if box["camera_id"] == "CAM-1"]


def _assert_distractor(fixture: RouteFixture) -> None:
    truth = _load(fixture.truth_path)
    topology = CameraTopology.from_dict(_load(fixture.topology_path))
    figures = truth["figures"]
    primary = figures[0]
    distractor = figures[-1]
    assert primary["figure_id"] == "F1"
    assert primary["distractor"] is False
    assert primary["shares_colours_with"] is None
    assert distractor["distractor"] is True
    assert distractor["upper_bgr"] == primary["upper_bgr"]
    assert distractor["lower_bgr"] == primary["lower_bgr"]
    assert distractor["shares_colours_with"] == primary["figure_id"]
    assert primary["upper_bgr"] != primary["lower_bgr"]
    left = {visit["camera_id"] for visit in primary["visits"]}
    right = {visit["camera_id"] for visit in distractor["visits"]}
    assert left.isdisjoint(right)
    for origin in left:
        for destination in right:
            assert topology.window(origin, destination) is None
            assert topology.window(destination, origin) is None


def _assert_one_pass(visits: list[dict[str, Any]]) -> None:
    cameras = [str(visit["camera_id"]) for visit in visits]
    assert cameras == sorted(cameras)
    assert len(cameras) == len(set(cameras))
    assert 2 <= len(cameras) <= 3
    numbers = [int(camera.removeprefix("CAM-")) for camera in cameras]
    assert numbers[-1] - numbers[0] == len(numbers) - 1


def _assert_box_shape(box: dict[str, Any], truth: dict[str, Any]) -> None:
    assert 0 <= box["x1"] < box["x2"] <= truth["width"]
    assert 0 <= box["y1"] < box["y2"] <= truth["height"]
    assert box["y2"] - box["y1"] > box["x2"] - box["x1"]
    assert (box["y2"] - box["y1"]) % 2 == 0
    assert (box["x2"] - box["x1"]) % 2 == 0


def _assert_motion(boxes: list[dict[str, Any]]) -> None:
    xs = [int(box["x1"]) for box in boxes]
    assert xs[0] != xs[-1]
    assert abs(xs[-1] - xs[0]) >= 80
    assert len(set(xs)) >= 10
    deltas = [nxt - prev for prev, nxt in zip(xs, xs[1:])]
    rising = all(delta >= 0 for delta in deltas)
    falling = all(delta <= 0 for delta in deltas)
    assert rising or falling
    middle = xs[len(xs) // 2]
    assert middle != xs[0]
    assert middle != xs[-1]


def _visit_boxes(
    figure: dict[str, Any],
    visit: dict[str, Any],
    fps: int,
    offset: float,
) -> list[dict[str, Any]]:
    return [
        box
        for box in figure["boxes"]
        if box["camera_id"] == visit["camera_id"]
        and visit["t_entry"] <= box["frame_index"] / fps + offset < visit["t_exit"]
    ]


def _assert_dwells(fixture: RouteFixture) -> None:
    truth = _load(fixture.truth_path)
    registry = CameraRegistry.load(fixture.cameras_path)
    fps = int(truth["fps"])
    seconds = int(truth["seconds"])
    for figure in truth["figures"]:
        visits = figure["visits"]
        _assert_one_pass(visits)
        for visit in visits:
            assert visit["t_exit"] - visit["t_entry"] >= 2.0
            camera = registry.get(visit["camera_id"])
            assert camera is not None
            delay = visit["t_entry"] - camera.clock_offset_seconds
            assert 0.0 <= delay <= 1.0
            assert visit["t_exit"] == camera.clock_offset_seconds + seconds
            frames = _visit_boxes(figure, visit, fps, camera.clock_offset_seconds)
            assert len(frames) >= fps * 2
            assert frames[-1]["frame_index"] == fps * seconds - 1
            for box in frames:
                _assert_box_shape(box, truth)
            _assert_motion(frames)


def _assert_edges(fixture: RouteFixture) -> None:
    truth = _load(fixture.truth_path)
    payload = _load(fixture.topology_path)
    topology = CameraTopology.from_dict(payload)
    walked: set[tuple[str, str]] = set()
    tables: set[tuple[tuple[str, float, float], ...]] = set()
    for figure in truth["figures"]:
        visits = figure["visits"]
        tables.add(
            tuple((visit["camera_id"], visit["t_entry"], visit["t_exit"]) for visit in visits)
        )
        for previous, following in zip(visits, visits[1:]):
            gap = following["t_entry"] - previous["t_exit"]
            assert gap >= 1.0
            assert gap <= 8.0
            assert topology.allows(
                previous["camera_id"],
                previous["t_exit"],
                following["camera_id"],
                following["t_entry"],
            )
            walked.add((previous["camera_id"], following["camera_id"]))
    assert len(tables) == 2
    for edge in payload["edges"]:
        assert edge["min_seconds"] == 0.5
        assert edge["max_seconds"] == 8.0
        assert (edge["from_camera"], edge["to_camera"]) in walked
        assert topology.window(edge["to_camera"], edge["from_camera"]) is None


def _assert_clip_row(index: int, entry: dict[str, Any], clip: Path) -> None:
    assert entry["path"] == f"CAM-{index + 1}.mp4"
    assert entry["path"] == clip.name
    assert entry["use"] == "CODE_TEST"
    assert entry["type"] == "code_fixture"
    assert entry["fixture_id"] == f"FX-CODE-{101 + index}"
    assert entry["license"] == "MIT"
    assert entry["generator_license"] == "MIT"
    assert entry["generator"] == "generate_route_fixture"
    assert entry["generator_version"] == "1.0.0"
    assert entry["version"] == "1.0.0"
    assert entry["source_url"] == "https://github.com/shitanshusharma/trace-search"
    assert "two-band" in entry["notes"]
    assert f'"camera_index":{index}' in entry["seed"]
    assert entry["sha256"] == _sha(clip)
    assert len(entry["phash"]) == 10
    assert len(set(entry["phash"])) == 10


def _assert_decoded_clip(clip: Path, truth: dict[str, Any]) -> None:
    capture = cv2.VideoCapture(str(clip))
    try:
        assert capture.get(cv2.CAP_PROP_FPS) == float(truth["fps"])
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == truth["fps"] * truth["seconds"]
        assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == truth["width"]
        assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == truth["height"]
        assert b"mp4v" in clip.read_bytes()
    finally:
        capture.release()


def _assert_clips(fixture: RouteFixture) -> None:
    manifest = _load(fixture.manifest_path)
    assert manifest["manifest_version"] == "1.0"
    assert manifest["last_updated_ist"] == "2026-10-27"
    assert check_licenses.validate_schema(manifest, _SCHEMA) == []
    truth = _load(fixture.truth_path)
    entries = manifest["entries"]
    assert len(entries) == len(fixture.clip_paths)
    assert entries[0]["phash"] != entries[1]["phash"]
    for index, (entry, clip) in enumerate(zip(entries, fixture.clip_paths)):
        _assert_clip_row(index, entry, clip)
        _assert_decoded_clip(clip, truth)


def _assert_closed_walk(fixture: RouteFixture) -> None:
    _assert_distractor(fixture)
    _assert_dwells(fixture)
    _assert_edges(fixture)
    _assert_clips(fixture)


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> RouteFixture:
    return _once(
        tmp_path_factory,
        n_cameras=4,
        n_figures=2,
        fps=10,
        size=(320, 240),
        seconds=30,
    )


@pytest.fixture(scope="module")
def defaults(tmp_path_factory: pytest.TempPathFactory) -> RouteFixture:
    return _once(tmp_path_factory)


_COLOUR_TABLE = (
    ((0, 0, 255), (255, 0, 0)),
    ((0, 255, 0), (255, 0, 255)),
    ((255, 255, 0), (0, 128, 255)),
    ((255, 0, 255), (0, 255, 255)),
    ((0, 165, 255), (128, 0, 128)),
    ((255, 128, 0), (0, 255, 128)),
    ((192, 0, 64), (64, 0, 192)),
    ((0, 192, 64), (192, 192, 0)),
)


def test_colour_table_shifts_and_the_last_figure_copies() -> None:
    for shift in range(len(_COLOUR_TABLE)):
        for source in range(3):
            got = route_generator._colours(source, shift, 4)
            assert got == _COLOUR_TABLE[(source + shift) % len(_COLOUR_TABLE)]
            assert got[0] != got[1]
        assert route_generator._colours(3, shift, 4) == _COLOUR_TABLE[shift]


def test_offsets_zones_and_figure_counts() -> None:
    assert route_generator._clock_offsets(4, 30) == (-2.0, 29.0, 60.0, 91.0)
    assert route_generator._clock_offsets(5, 30) == (-2.0, 29.0, 60.0, 91.0, 122.0)
    assert route_generator._clock_offsets(6, 30) == (-2.0, 29.0, 60.0, 91.0, 122.0, 153.0)
    assert route_generator._clock_offsets(6, 40) == (-2.0, 39.0, 80.0, 121.0, 162.0, 203.0)
    assert route_generator._zones(4) == ((0, 1), (2, 3))
    assert route_generator._zones(5) == ((0, 1), (2, 3, 4))
    assert route_generator._zones(6) == ((0, 1, 2), (3, 4, 5))
    assert route_generator._left_figure_count(2) == 1
    assert route_generator._left_figure_count(3) == 1
    assert route_generator._left_figure_count(4) == 2
    assert route_generator._left_figure_count(5) == 2
    assert route_generator._left_figure_count(8) == 4
    assert route_generator._is_default_schedule(6, 30)
    assert not route_generator._is_default_schedule(4, 30)
    assert not route_generator._is_default_schedule(6, 40)
    assert route_generator._pair_or_quad(2)
    assert route_generator._pair_or_quad(4)
    assert not route_generator._pair_or_quad(3)
    assert not route_generator._pair_or_quad(8)


def test_samples_slots_and_frozen_records() -> None:
    assert route_generator._sample_indices(300) == (0, 33, 66, 99, 132, 166, 199, 232, 265, 299)
    assert route_generator._sample_indices(400) == (0, 44, 88, 133, 177, 221, 266, 310, 354, 399)
    assert route_generator._jitter_limit(10, 300) == 10
    assert route_generator._jitter_limit(30, 900) == 30
    assert route_generator._slot_start(33, 300, 5) == 33
    assert route_generator._slot_start(299, 300, 5) == 295
    assert route_generator._slot_start(295, 300, 5) == 295
    assert route_generator._visit_start(5, 300) == 5
    assert route_generator._visit_start(0, 300) == 0
    for cls in (
        route_generator.RouteFixture,
        route_generator._Args,
        route_generator._Visit,
        route_generator._Box,
        route_generator._Figure,
        route_generator._Plan,
        route_generator._Variation,
    ):
        assert cls.__dataclass_params__.frozen
    route_generator._require_schedule(4, 10, 40)
    route_generator._require_schedule(6, 10, 30)
    route_generator._require_schedule(4, 30, 30)


def test_lane_step_and_forward_edges() -> None:
    assert route_generator._lane_geometry(320, 240, 2) == (54, 108, 27, (8, 124))
    assert route_generator._lane_geometry(320, 240, 4) == (24, 48, 31, (8, 64, 120, 176))
    assert route_generator._lane_geometry(640, 360, 4) == (40, 80, 64, (8, 96, 184, 272))
    assert route_generator._lane_geometry(320, 240, 8) == (
        10,
        20,
        32,
        (8, 36, 64, 92, 120, 148, 176, 204),
    )
    assert route_generator._lane_geometry(1280, 720, 8) == (
        40,
        80,
        136,
        (8, 96, 184, 272, 360, 448, 536, 624),
    )
    pairs = {
        (edge.from_camera, edge.to_camera, edge.min_seconds, edge.max_seconds)
        for edge in route_generator._edges(4)
    }
    assert ("CAM-1", "CAM-2", 0.5, 8.0) in pairs
    assert ("CAM-2", "CAM-1", 0.5, 8.0) not in pairs
    assert ("CAM-3", "CAM-4", 0.5, 8.0) in pairs
    assert ("CAM-4", "CAM-3", 0.5, 8.0) not in pairs
    assert ("CAM-2", "CAM-3", 0.5, 8.0) not in pairs
    six = {(edge.from_camera, edge.to_camera) for edge in route_generator._edges(6)}
    assert six == {
        ("CAM-1", "CAM-2"),
        ("CAM-2", "CAM-3"),
        ("CAM-4", "CAM-5"),
        ("CAM-5", "CAM-6"),
    }


def test_x_at_follows_the_lane_step() -> None:
    assert route_generator._x_at(0, 300, 27, False, 0) == 8
    assert route_generator._x_at(299, 300, 27, False, 0) == 225
    assert route_generator._x_at(0, 300, 27, True, 0) == 225
    assert route_generator._x_at(299, 300, 27, True, 0) == 8
    assert route_generator._x_at(150, 300, 27, False, 0) == 116
    assert route_generator._x_at(5, 300, 27, True, 0) == 221
    assert route_generator._x_at(0, 300, 27, False, 1) == 225
    assert route_generator._x_at(3, 300, 27, True, 1) == 10
    assert route_generator._x_at(0, 300, 27, False, 2) == 21
    assert route_generator._x_at(299, 300, 27, False, 2) == 238
    assert route_generator._x_at(0, 300, 27, False, 3) == 238
    assert route_generator._x_at(299, 300, 27, False, 3) == 21
    assert route_generator._x_at(0, 300, 27, False, 4) == 34
    assert route_generator._x_at(299, 300, 27, False, 4) == 251
    assert route_generator._x_at(0, 300, 27, False, 5) == 251
    assert route_generator._x_at(299, 300, 27, False, 5) == 34


def test_signature_defaults_match_the_task() -> None:
    signature = inspect.signature(generate_route_fixture)
    assert signature.parameters["seed"].default == 0
    assert signature.parameters["n_cameras"].default == 6
    assert signature.parameters["n_figures"].default == 4
    assert signature.parameters["fps"].default == 10
    assert signature.parameters["size"].default == (640, 360)
    assert signature.parameters["seconds"].default == 30


def test_two_subprocesses_write_the_same_bytes(tmp_path: Path) -> None:
    program = "\n".join(
        [
            "import hashlib, sys",
            "from pathlib import Path",
            "from trace_search.fixtures.route_generator import generate_route_fixture",
            "fixture = generate_route_fixture(",
            "    Path(sys.argv[1]), seed=0, n_cameras=4, n_figures=2,",
            "    fps=10, size=(320, 240), seconds=30,",
            ")",
            "print(hashlib.sha256(fixture.clip_paths[0].read_bytes()).hexdigest())",
        ]
    )

    def run(dirname: str) -> str:
        completed = subprocess.run(
            [sys.executable, "-c", program, str(tmp_path / dirname)],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    assert run("left") == run("right")


def test_a_different_seed_changes_the_clip_bytes(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    left = _once(tmp_path_factory, n_cameras=4, n_figures=2, size=(320, 240))
    right = _once(tmp_path_factory, seed=1, n_cameras=4, n_figures=2, size=(320, 240))
    assert _sha(left.clip_paths[0]) != _sha(right.clip_paths[0])
    assert _sha(left.truth_path) != _sha(right.truth_path)


def test_colours_and_direction_follow_the_generator(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    left = _load(_once(tmp_path_factory, n_cameras=4, n_figures=2, size=(320, 240)).truth_path)
    right = _load(
        _once(tmp_path_factory, seed=2, n_cameras=4, n_figures=2, size=(320, 240)).truth_path
    )
    forward = _load(
        _once(tmp_path_factory, seed=3, n_cameras=4, n_figures=2, size=(320, 240)).truth_path
    )
    assert left["figures"][0]["upper_bgr"] != right["figures"][0]["upper_bgr"]
    assert left["figures"][0]["lower_bgr"] != right["figures"][0]["lower_bgr"]
    assert route_generator._variation(0, 4, 10, 300).reverse
    assert route_generator._variation(2, 4, 10, 300).reverse
    assert not route_generator._variation(3, 4, 10, 300).reverse
    assert _cam1_x(left)[0] > _cam1_x(left)[-1]
    assert _cam1_x(right)[0] > _cam1_x(right)[-1]
    assert _cam1_x(forward)[0] < _cam1_x(forward)[-1]


def test_six_cameras_use_planted_offsets_and_fx_code_ids(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    fixture = _once(tmp_path_factory, n_cameras=6, n_figures=4, size=(320, 240))
    registry = CameraRegistry.load(fixture.cameras_path)
    offsets = [registry.get(f"CAM-{index}").clock_offset_seconds for index in range(1, 7)]
    assert offsets == [-2.0, 29.0, 60.0, 91.0, 122.0, 153.0]
    assert [camera.site_map_x for camera in registry.list_cameras()] == [
        0.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ]
    assert all(
        camera.site_map_y == 0.0 and camera.brand == "synthetic"
        for camera in registry.list_cameras()
    )
    figures = _load(fixture.truth_path)["figures"]
    groups = [{visit["camera_id"] for visit in figure["visits"]} for figure in figures]
    assert groups == [
        {"CAM-1", "CAM-2", "CAM-3"},
        {"CAM-1", "CAM-2", "CAM-3"},
        {"CAM-4", "CAM-5", "CAM-6"},
        {"CAM-4", "CAM-5", "CAM-6"},
    ]
    assert figures[1]["upper_bgr"] != figures[0]["upper_bgr"]
    assert figures[2]["upper_bgr"] != figures[0]["upper_bgr"]
    assert figures[1]["upper_bgr"] != figures[2]["upper_bgr"]
    box = figures[0]["boxes"][0]
    assert box["y2"] - box["y1"] == 48
    assert box["x2"] - box["x1"] == 24
    assert box["y1"] == 8
    _assert_closed_walk(fixture)


def test_five_cameras_keep_the_floor_split(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    fixture = _once(tmp_path_factory, n_cameras=5, n_figures=3, size=(320, 240))
    figures = _load(fixture.truth_path)["figures"]
    cameras = [{visit["camera_id"] for visit in figure["visits"]} for figure in figures]
    assert cameras[0] == {"CAM-1", "CAM-2"}
    assert cameras[1] == {"CAM-3", "CAM-4", "CAM-5"}
    assert cameras[2] == {"CAM-3", "CAM-4", "CAM-5"}
    registry = CameraRegistry.load(fixture.cameras_path)
    offsets = [registry.get(f"CAM-{index}").clock_offset_seconds for index in range(1, 6)]
    assert offsets == [-2.0, 29.0, 60.0, 91.0, 122.0]
    _assert_closed_walk(fixture)


def test_default_fixture_walks_each_corridor_once(defaults: RouteFixture) -> None:
    _assert_closed_walk(defaults)
    figures = _load(defaults.truth_path)["figures"]
    box = figures[0]["boxes"][0]
    assert box["y2"] - box["y1"] == 80
    assert box["x2"] - box["x1"] == 40
    groups = [{visit["camera_id"] for visit in figure["visits"]} for figure in figures]
    assert groups[0] == groups[1] == {"CAM-1", "CAM-2", "CAM-3"}
    assert groups[2] == groups[3] == {"CAM-4", "CAM-5", "CAM-6"}
    assert [visit["camera_id"] for visit in figures[0]["visits"]] == ["CAM-1", "CAM-2", "CAM-3"]


def test_small_fixture_is_upright_and_moves(small: RouteFixture) -> None:
    _assert_closed_walk(small)
    box = _load(small.truth_path)["figures"][0]["boxes"][0]
    assert box["y2"] - box["y1"] == 108
    assert box["x2"] - box["x1"] == 54
    assert box["y1"] == 8
    assert box["x1"] == 221
    second = _load(small.truth_path)["figures"][1]["boxes"][0]
    assert second["y1"] == 124


def test_seed_zero_enters_late_and_keeps_its_colours(small: RouteFixture) -> None:
    truth = _load(small.truth_path)
    frames = {
        item["frame_index"] for item in truth["figures"][0]["boxes"] if item["camera_id"] == "CAM-1"
    }
    variation = route_generator._variation(0, 4, 10, 300)
    upper, lower = route_generator._colours(0, variation.shift, 2)
    xs = _cam1_x(truth)
    assert xs[0] == 221
    assert xs[-1] == 8
    assert variation.jitter[0] == 5
    assert min(frames) == 5
    assert 0 not in frames
    assert 299 in frames
    assert len(set(xs)) > 10
    assert truth["seed"] == 0
    assert truth["background_bgr"] == [16, 16, 16]
    assert truth["figures"][0]["upper_bgr"] == list(upper)
    assert truth["figures"][0]["lower_bgr"] == list(lower)


def test_each_figure_walks_its_corridor_once(
    small: RouteFixture,
    defaults: RouteFixture,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A route is one pass of multi-second visits, not a bounce between two cameras."""
    _assert_dwells(small)
    _assert_dwells(defaults)
    four = _load(small.truth_path)["figures"][0]["visits"]
    six = _load(defaults.truth_path)["figures"][0]["visits"]
    assert [visit["camera_id"] for visit in four] == ["CAM-1", "CAM-2"]
    assert [visit["camera_id"] for visit in six] == ["CAM-1", "CAM-2", "CAM-3"]
    assert all(visit["t_exit"] - visit["t_entry"] >= 2.0 for visit in four)
    assert all(visit["t_exit"] - visit["t_entry"] >= 2.0 for visit in six)
    xs = _cam1_x(_load(small.truth_path))
    assert xs[0] != xs[-1]
    assert len(set(xs)) >= 10
    later = _once(tmp_path_factory, seed=1, n_cameras=4, n_figures=2, size=(320, 240))
    other = _load(later.truth_path)["figures"][0]["visits"][0]["t_entry"]
    assert four[0]["t_entry"] != other


@pytest.mark.parametrize(
    ("n_cameras", "n_figures", "fps", "seconds", "size"),
    _ACCEPTED,
)
def test_every_accepted_config_passes_the_schema(
    tmp_path_factory: pytest.TempPathFactory,
    n_cameras: int,
    n_figures: int,
    fps: int,
    seconds: int,
    size: tuple[int, int],
) -> None:
    fixture = _once(
        tmp_path_factory,
        n_cameras=n_cameras,
        n_figures=n_figures,
        fps=fps,
        seconds=seconds,
        size=size,
    )
    _assert_closed_walk(fixture)


def _generator_accepts(
    n_cameras: int,
    n_figures: int,
    fps: int,
    seconds: int,
    size: tuple[int, int],
) -> bool:
    try:
        route_generator._require_int("n_cameras", n_cameras, 4, 6)
        route_generator._require_int("n_figures", n_figures, 2, 8)
        route_generator._require_int("fps", fps, 10, 30)
        route_generator._require_int("seconds", seconds, 30, 40)
        width, height = route_generator._require_size(size)
        route_generator._require_known_size(width, height, n_cameras, n_figures, fps, seconds)
        route_generator._require_schedule(n_cameras, fps, seconds)
    except (TypeError, ValueError):
        return False
    return True


def test_accepted_grid_matches_the_predicates() -> None:
    sizes = (
        (320, 240),
        (640, 360),
        (1280, 720),
        (640, 480),
        (640, 720),
        (1279, 720),
        (1280, 1080),
        (239, 240),
    )
    found: set[tuple[int, int, int, int, tuple[int, int]]] = set()
    for n_cameras in range(3, 8):
        for n_figures in range(1, 10):
            for fps in range(9, 32):
                for seconds in range(29, 42):
                    for size in sizes:
                        if _generator_accepts(n_cameras, n_figures, fps, seconds, size):
                            found.add((n_cameras, n_figures, fps, seconds, size))
    assert found == set(_ACCEPTED)
    assert len(_ACCEPTED) == len(found)


def test_eight_figures_at_720p_are_accepted(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    fixture = _once(
        tmp_path_factory,
        seed=123,
        n_cameras=6,
        n_figures=8,
        fps=10,
        size=(1280, 720),
        seconds=30,
    )
    _assert_closed_walk(fixture)
    truth = _load(fixture.truth_path)
    assert truth["seed"] == 123
    assert truth["width"] == 1280
    assert truth["height"] == 720
    assert len(truth["figures"]) == 8
    groups = [{visit["camera_id"] for visit in figure["visits"]} for figure in truth["figures"]]
    assert groups[:4] == [{"CAM-1", "CAM-2", "CAM-3"}] * 4
    assert groups[4:] == [{"CAM-4", "CAM-5", "CAM-6"}] * 4


def test_thirty_fps_and_forty_second_clips_walk_once(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    thirty = _once(
        tmp_path_factory,
        n_cameras=4,
        n_figures=2,
        fps=30,
        seconds=30,
        size=(320, 240),
    )
    forty = _once(
        tmp_path_factory,
        n_cameras=4,
        n_figures=2,
        fps=10,
        seconds=40,
        size=(320, 240),
    )
    _assert_closed_walk(thirty)
    _assert_closed_walk(forty)
    assert _load(thirty.truth_path)["fps"] == 30
    assert _load(forty.truth_path)["seconds"] == 40


def test_sample_frame_uses_only_the_palette(small: RouteFixture) -> None:
    truth = _load(small.truth_path)
    allowed = {tuple(truth["background_bgr"])}
    for figure in truth["figures"]:
        allowed.add(tuple(figure["upper_bgr"]))
        allowed.add(tuple(figure["lower_bgr"]))
    frame = small.sample_frame
    found = {tuple(int(channel) for channel in pixel) for pixel in frame.reshape(-1, 3)}
    assert found <= allowed
    assert tuple(truth["background_bgr"]) in found
    assert small.background_bgr in small.palette_bgr
    assert small.palette_bgr[0] == (16, 16, 16)
    assert len(small.palette_bgr) == len(set(small.palette_bgr))
    assert set(small.palette_bgr) == allowed


def test_sample_frame_matches_the_middle_box(small: RouteFixture) -> None:
    truth = _load(small.truth_path)
    primary = truth["figures"][0]
    middle = primary["boxes"][len(primary["boxes"]) // 2]
    assert small.sample_frame_index == 3
    assert small.sample_camera_id == "CAM-2"
    assert middle["frame_index"] == small.sample_frame_index
    assert middle["camera_id"] == small.sample_camera_id
    assert small.sample_frame_index != primary["boxes"][0]["frame_index"]
    box = next(
        item
        for item in primary["boxes"]
        if item["camera_id"] == small.sample_camera_id
        and item["frame_index"] == small.sample_frame_index
    )
    frame = small.sample_frame
    mid = box["y1"] + (box["y2"] - box["y1"]) // 2
    upper = tuple(int(channel) for channel in frame[box["y1"], box["x1"]])
    lower = tuple(int(channel) for channel in frame[mid, box["x1"]])
    assert upper == tuple(primary["upper_bgr"])
    assert lower == tuple(primary["lower_bgr"])
    background = np.array(truth["background_bgr"], dtype=np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=bool)
    for item in truth["figures"]:
        for recorded in item["boxes"]:
            same = (
                recorded["camera_id"] == small.sample_camera_id
                and recorded["frame_index"] == small.sample_frame_index
            )
            if not same:
                continue
            mask[recorded["y1"] : recorded["y2"], recorded["x1"] : recorded["x2"]] = True
    outside = frame.reshape(-1, 3)[~mask.reshape(-1)]
    assert np.all(outside == background)


def test_code_test_clips_pass_unit_test_and_are_refused_for_training(small: RouteFixture) -> None:
    manifest = small.manifest_path
    assert check_licenses.validate_schema(_load(manifest), _SCHEMA) == []
    for clip in small.clip_paths:
        check_fixture_allowed(clip, LoaderContext.UNIT_TEST, manifest)
        with pytest.raises(FixtureSeparationError, match="CODE_TEST"):
            check_fixture_allowed(clip, LoaderContext.TRAINING, manifest)
        with pytest.raises(FixtureSeparationError, match="CODE_TEST"):
            check_fixture_allowed(clip, LoaderContext.EVALUATION, manifest)


def test_flipping_a_generated_row_to_smoke_only_fails_the_schema(small: RouteFixture) -> None:
    """``code_fixture.use`` is const ``CODE_TEST``, so ``SMOKE_ONLY`` is refused."""
    manifest = _load(small.manifest_path)
    manifest["entries"][0]["use"] = "SMOKE_ONLY"
    errors = check_licenses.validate_schema(manifest, _SCHEMA)
    assert errors
    assert any("CODE_TEST" in error and "entries.0.use" in error for error in errors)


def _size_message(
    got: str,
    cameras: int = 4,
    figures: int = 2,
    seconds: int = 30,
) -> str:
    return (
        rf"^size {got} is not accepted for {cameras} cameras, "
        rf"{figures} figures, and {seconds} seconds$"
    )


def _reject(tmp_path: Path, error: type[BaseException], match: str, **overrides: Any) -> None:
    out = tmp_path / "rejected"
    with pytest.raises(error, match=match):
        _small(out, **overrides)
    assert not out.exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"seed": True}, r"seed must be an int, got True"),
        ({"seed": False}, r"seed must be an int, got False"),
        ({"seed": 1.5}, r"seed must be an int, got 1\.5"),
        ({"seed": "0"}, r"seed must be an int, got '0'"),
        ({"n_cameras": True}, r"n_cameras must be an int, got True"),
        ({"n_cameras": 4.0}, r"n_cameras must be an int, got 4\.0"),
        ({"n_figures": True}, r"n_figures must be an int, got True"),
        ({"n_figures": "4"}, r"n_figures must be an int, got '4'"),
        ({"fps": True}, r"fps must be an int, got True"),
        ({"fps": 10.0}, r"fps must be an int, got 10\.0"),
        ({"seconds": True}, r"seconds must be an int, got True"),
        ({"seconds": 30.0}, r"seconds must be an int, got 30\.0"),
    ],
)
def test_bool_and_non_int_arguments_raise_type_error(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    _reject(tmp_path, TypeError, message, **overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"seed": -1}, r"^seed must be in 0\.\.2147483647, got -1$"),
        ({"seed": 2_147_483_648}, r"^seed must be in 0\.\.2147483647, got 2147483648$"),
        ({"n_cameras": 3}, r"^n_cameras must be in 4\.\.6, got 3$"),
        ({"n_cameras": 7}, r"^n_cameras must be in 4\.\.6, got 7$"),
        ({"n_cameras": 12}, r"^n_cameras must be in 4\.\.6, got 12$"),
        ({"n_figures": 1}, r"^n_figures must be in 2\.\.8, got 1$"),
        ({"n_figures": 9}, r"^n_figures must be in 2\.\.8, got 9$"),
        ({"fps": 0}, r"^fps must be in 10\.\.30, got 0$"),
        ({"fps": 9}, r"^fps must be in 10\.\.30, got 9$"),
        ({"fps": 31}, r"^fps must be in 10\.\.30, got 31$"),
        ({"seconds": 0}, r"^seconds must be in 30\.\.40, got 0$"),
        ({"seconds": 29}, r"^seconds must be in 30\.\.40, got 29$"),
        ({"seconds": 41}, r"^seconds must be in 30\.\.40, got 41$"),
        (
            {"fps": 20},
            r"^unsupported route schedule for 4 cameras at 20 fps and 30 seconds$",
        ),
        (
            {"fps": 30, "seconds": 40},
            r"^unsupported route schedule for 4 cameras at 30 fps and 40 seconds$",
        ),
        (
            {"n_cameras": 6, "n_figures": 4, "fps": 30},
            r"^unsupported route schedule for 6 cameras at 30 fps and 30 seconds$",
        ),
        (
            {"n_cameras": 5, "n_figures": 2, "fps": 30},
            r"^unsupported route schedule for 5 cameras at 30 fps and 30 seconds$",
        ),
        (
            {"n_cameras": 5, "n_figures": 3, "fps": 30},
            _size_message(r"\(320, 240\)", cameras=5, figures=3),
        ),
        (
            {"seconds": 36},
            r"^unsupported route schedule for 4 cameras at 10 fps and 36 seconds$",
        ),
        (
            {"seconds": 40, "fps": 30},
            r"^unsupported route schedule for 4 cameras at 30 fps and 40 seconds$",
        ),
    ],
)
def test_out_of_range_arguments_raise_value_error(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    _reject(tmp_path, ValueError, message, **overrides)


@pytest.mark.parametrize(
    ("size", "error", "message"),
    [
        ((320,), ValueError, r"size must be a \(width, height\) pair"),
        ((320, 240, 1), ValueError, r"size must be a \(width, height\) pair"),
        ([320, 240], TypeError, r"size must be a \(width, height\) tuple"),
        (None, TypeError, r"size must be a \(width, height\) tuple"),
        ((239, 240), ValueError, r"^width must be in 240\.\.1280, got 239$"),
        ((1281, 720), ValueError, r"^width must be in 240\.\.1280, got 1281$"),
        ((320, 239), ValueError, r"^height must be in 240\.\.1280, got 239$"),
        ((320, 1281), ValueError, r"^height must be in 240\.\.1280, got 1281$"),
        ((True, 240), TypeError, r"width must be an int, got True"),
        ((320, True), TypeError, r"height must be an int, got True"),
        ((320.0, 240), TypeError, r"width must be an int, got 320\.0"),
        ((320, 240.0), TypeError, r"height must be an int, got 240\.0"),
        ((320, 360), ValueError, _size_message(r"\(320, 360\)")),
        ((640, 240), ValueError, _size_message(r"\(640, 240\)")),
        ((480, 360), ValueError, _size_message(r"\(480, 360\)")),
        ((240, 240), ValueError, _size_message(r"\(240, 240\)")),
        ((640, 480), ValueError, _size_message(r"\(640, 480\)")),
        ((640, 640), ValueError, _size_message(r"\(640, 640\)")),
        ((400, 360), ValueError, _size_message(r"\(400, 360\)")),
        ((320, 320), ValueError, _size_message(r"\(320, 320\)")),
        ((1280, 1080), ValueError, _size_message(r"\(1280, 1080\)")),
        ((1280, 360), ValueError, _size_message(r"\(1280, 360\)")),
        ((720, 1280), ValueError, _size_message(r"\(720, 1280\)")),
    ],
)
def test_size_guards(
    tmp_path: Path,
    size: object,
    error: type[BaseException],
    message: str,
) -> None:
    _reject(tmp_path, error, message, size=size)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"size": (1280, 720)}, _size_message(r"\(1280, 720\)")),
        (
            {"n_cameras": 5, "n_figures": 3, "size": (1280, 720)},
            _size_message(r"\(1280, 720\)", cameras=5, figures=3),
        ),
        (
            {"n_cameras": 6, "n_figures": 4, "size": (1280, 720)},
            _size_message(r"\(1280, 720\)", cameras=6, figures=4),
        ),
        (
            {"n_cameras": 6, "n_figures": 8, "seconds": 40, "size": (1280, 720)},
            _size_message(r"\(1280, 720\)", cameras=6, figures=8, seconds=40),
        ),
        ({"fps": 30, "size": (1280, 720)}, _size_message(r"\(1280, 720\)")),
        (
            {"n_cameras": 6, "n_figures": 8, "size": (640, 360)},
            _size_message(r"\(640, 360\)", cameras=6, figures=8),
        ),
        (
            {"n_cameras": 6, "n_figures": 4, "size": (640, 480)},
            _size_message(r"\(640, 480\)", cameras=6, figures=4),
        ),
        (
            {"n_cameras": 4, "n_figures": 8, "seconds": 40, "size": (320, 240)},
            _size_message(r"\(320, 240\)", figures=8, seconds=40),
        ),
        (
            {"n_cameras": 4, "n_figures": 8, "fps": 20, "size": (320, 240)},
            _size_message(r"\(320, 240\)", figures=8),
        ),
        (
            {"n_cameras": 4, "n_figures": 8, "fps": 30, "size": (320, 240)},
            _size_message(r"\(320, 240\)", figures=8),
        ),
        (
            {"n_cameras": 6, "n_figures": 8, "size": (640, 720)},
            _size_message(r"\(640, 720\)", cameras=6, figures=8),
        ),
        (
            {"n_cameras": 6, "n_figures": 8, "size": (1279, 720)},
            _size_message(r"\(1279, 720\)", cameras=6, figures=8),
        ),
    ],
)
def test_720p_requires_six_cameras_and_thirty_seconds(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    _reject(tmp_path, ValueError, message, **overrides)


def test_seed_max_is_accepted(tmp_path: Path) -> None:
    fixture = _small(tmp_path / "seed-max", seed=2_147_483_647)
    assert fixture.clip_paths[0].is_file()


def test_out_dir_must_be_a_path(tmp_path: Path) -> None:
    out_dir: object = str(tmp_path / "text")
    with pytest.raises(TypeError, match=r"out_dir must be a Path, got str"):
        generate_route_fixture(out_dir, n_cameras=4, n_figures=2, size=(320, 240))


def test_repo_root_and_a_child_and_dotdot_are_refused() -> None:
    with pytest.raises(ValueError, match="inside the repository"):
        _small(_REPO)
    with pytest.raises(ValueError, match="inside the repository"):
        _small(_REPO / "src")
    disguised = _REPO / "src" / ".." / "tests"
    with pytest.raises(ValueError, match="inside the repository"):
        _small(disguised)


def test_grandchild_of_the_repo_is_refused() -> None:
    target = _REPO / "src" / "trace_search" / "nested"
    with pytest.raises(ValueError, match="inside the repository"):
        _small(target)
    assert not target.exists()


def test_symlink_into_the_repo_is_refused(tmp_path: Path) -> None:
    link = tmp_path / "into-repo"
    link.symlink_to(_REPO / "src", target_is_directory=True)
    with pytest.raises(ValueError, match="inside the repository"):
        _small(link)
    assert not any(_REPO.joinpath("src").glob("CAM-*.mp4"))


def test_dotdot_that_resolves_outside_is_allowed(tmp_path: Path) -> None:
    target = tmp_path / "a" / ".." / "outside"
    fixture = _small(target)
    assert fixture.clip_paths[0].is_file()
    assert fixture.out_dir == (tmp_path / "outside").resolve()


def test_a_file_out_dir_is_refused(tmp_path: Path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        _small(blocked)


def test_relative_path_outside_the_repo_is_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    fixture = _small(Path("relative-out"))
    assert (tmp_path / "relative-out" / "CAM-1.mp4").is_file()
    assert fixture.out_dir == (tmp_path / "relative-out").resolve()


def test_tilde_is_expanded_before_the_repo_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(_REPO)
    fixture = _small(Path("~/route-out"))
    assert (home / "route-out" / "CAM-1.mp4").is_file()
    assert fixture.out_dir == (home / "route-out").resolve()


def test_nested_out_dir_is_created(tmp_path: Path) -> None:
    fixture = _small(tmp_path / "nest" / "deeper")
    assert fixture.clip_paths[0].is_file()


def test_a_closed_writer_raises_before_any_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Closed:
        fourcc = staticmethod(lambda *_args: 0)

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def write(self, _frame: object) -> None:
            raise AssertionError("write")

        def release(self) -> None:
            return None

    setattr(_Closed, "isOpened", lambda _self: False)
    monkeypatch.setattr(cv2, "VideoWriter", _Closed)
    out = tmp_path / "closed"
    with pytest.raises(RuntimeError, match="video writer"):
        _small(out)
    assert not any(out.glob("CAM-*.mp4"))


def test_version_0_container_times_are_cleared(tmp_path: Path) -> None:
    """A version-0 mvhd with a non-zero stamp is cleared.

    opencv-python-headless 4.10.0.84 already writes those fields as 0.
    This plants a stamp so removing the clear changes the file.
    """
    fixture = _small(tmp_path / "stamp")
    clip = fixture.clip_paths[0]
    data = bytearray(clip.read_bytes())
    pos = data.find(b"mvhd")
    assert pos > 0
    payload = pos + 4
    assert data[payload] == 0
    data[payload + 4 : payload + 12] = b"\x11" * 8
    clip.write_bytes(data)
    route_generator._zero_mp4_times(clip)
    cleared = clip.read_bytes()
    assert cleared[payload + 4 : payload + 12] == b"\x00" * 8


def test_generation_does_not_change_git_status(tmp_path: Path) -> None:
    before = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=_REPO,
        text=True,
    )
    _small(tmp_path / "clean")
    after = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=_REPO,
        text=True,
    )
    assert after == before


def test_json_is_indented_and_sorted(small: RouteFixture) -> None:
    text = small.cameras_path.read_text(encoding="utf-8")
    assert text.startswith('{\n  "cameras"')
    assert text.endswith("\n")
    assert text.index('"brand"') < text.index('"camera_id"')
    truth = small.truth_path.read_text(encoding="utf-8")
    assert truth.startswith('{\n  "background_bgr"')
    assert truth.endswith("\n")
    with pytest.raises(ValueError, match="Out of range float values"):
        route_generator._write_json(small.out_dir / "nan.json", {"value": float("nan")})


def test_the_encoder_is_released(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    released: list[bool] = []
    real = cv2.VideoWriter

    class _Proxy:
        def __init__(self, inner: cv2.VideoWriter) -> None:
            self._inner = inner

        def write(self, frame: object) -> None:
            self._inner.write(frame)

        def release(self) -> None:
            released.append(True)
            self._inner.release()

    setattr(_Proxy, "isOpened", lambda self: bool(self._inner.isOpened()))

    def _factory(*args: object, **kwargs: object) -> _Proxy:
        return _Proxy(real(*args, **kwargs))

    _factory.fourcc = real.fourcc
    monkeypatch.setattr(cv2, "VideoWriter", _factory)
    _small(tmp_path / "released")
    assert released


def test_an_existing_directory_is_accepted(tmp_path: Path) -> None:
    out = tmp_path / "already"
    out.mkdir()
    fixture = _small(out)
    assert fixture.clip_paths[0].is_file()


def test_variation_replays_one_generator_stream() -> None:
    rng = np.random.default_rng(np.random.SeedSequence([0, 0]))
    reverse = int(rng.integers(0, 2)) == 1
    shift = int(rng.integers(0, 8))
    jitter = tuple(int(value) for value in rng.integers(0, 11, size=6))
    got = route_generator._variation(0, 6, 10, 300)
    assert got.reverse is reverse
    assert got.shift == shift
    assert got.jitter == jitter
    assert jitter == (5, 2, 3, 0, 0, 0)
    wide = route_generator._variation(16, 6, 10, 300)
    assert wide.jitter == (2, 3, 1, 10, 7, 5)
    assert 10 in wide.jitter
    short = route_generator._variation(0, 4, 10, 300)
    assert short.jitter == (5, 2, 3, 0)
    fast = np.random.default_rng(np.random.SeedSequence([0, 0]))
    assert int(fast.integers(0, 2)) == 1
    assert int(fast.integers(0, 8)) == shift
    drawn = tuple(int(value) for value in fast.integers(0, 31, size=4))
    assert route_generator._variation(0, 4, 30, 900).jitter == drawn
    assert route_generator._variation(3, 6, 10, 300).shift == 0


def test_seeds_congruent_mod_8_write_different_clips(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    digests = [
        _sha(
            _once(
                tmp_path_factory,
                seed=seed,
                n_cameras=4,
                n_figures=2,
                size=(320, 240),
            ).clip_paths[0]
        )
        for seed in (0, 8, 16, 2_147_483_640, 2840, 15536)
    ]
    assert len(set(digests)) == 6


def test_direction_is_not_the_seed_parity(tmp_path_factory: pytest.TempPathFactory) -> None:
    reversed_walk = _once(tmp_path_factory, n_cameras=4, n_figures=2, size=(320, 240))
    forward = _once(tmp_path_factory, seed=3, n_cameras=4, n_figures=2, size=(320, 240))
    same_parity = _once(tmp_path_factory, seed=1, n_cameras=4, n_figures=2, size=(320, 240))
    assert _cam1_x(_load(reversed_walk.truth_path))[0] == 221
    assert _cam1_x(_load(forward.truth_path))[0] == 10
    assert _cam1_x(_load(same_parity.truth_path))[0] == 222
    assert len(set(_cam1_x(_load(reversed_walk.truth_path)))) > 10


def test_numpy_seed_is_rejected_before_any_file(tmp_path: Path) -> None:
    out = tmp_path / "numpy-seed"
    with pytest.raises(TypeError, match="seed"):
        _small(out, seed=np.int64(1))
    assert not out.exists()


def test_manifest_seed_string_includes_the_seed(small: RouteFixture) -> None:
    entry = _load(small.manifest_path)["entries"][0]
    assert entry["seed"] == (
        '{"camera_index":0,"fps":10,"height":240,"n_cameras":4,'
        '"n_figures":2,"seconds":30,"seed":0,"width":320}'
    )


def test_each_clip_keeps_its_own_phashes(small: RouteFixture) -> None:
    manifest = _load(small.manifest_path)
    seen: list[list[str]] = []
    for entry, clip in zip(manifest["entries"], small.clip_paths):
        digest = perceptual_hashes(clip)
        assert entry["phash"] == digest
        assert len(digest) == 10
        assert len(set(digest)) == 10
        seen.append(digest)
    assert seen[0] != seen[1]


def test_decoded_frame_uses_width_then_height(small: RouteFixture) -> None:
    capture = cv2.VideoCapture(str(small.clip_paths[0]))
    try:
        ok, frame = capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    assert ok
    assert frame is not None
    assert frame.shape[:2] == (240, 320)
    assert fps == 10.0
    assert b"mp4v" in small.clip_paths[0].read_bytes()


def _nearest_is_background(
    image: np.ndarray[Any, np.dtype[np.uint8]],
    background: tuple[int, int, int],
    colours: list[tuple[int, int, int]],
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    pixels = image.astype(np.int32)
    bg = np.array(background, dtype=np.int32)
    dist_bg = np.sum((pixels - bg) ** 2, axis=2)
    nearest = np.ones(image.shape[:2], dtype=bool)
    for colour in colours:
        if colour == background:
            continue
        dist = np.sum((pixels - np.array(colour, dtype=np.int32)) ** 2, axis=2)
        nearest &= dist_bg <= dist
    return nearest


def _assert_band_interiors(
    pixels: np.ndarray[Any, np.dtype[np.int32]],
    dist_bg: np.ndarray[Any, np.dtype[np.int32]],
    box: dict[str, Any],
) -> None:
    """mp4v smears the outer pixel and the band boundary. The interiors stay flat."""
    mid = box["y1"] + (box["y2"] - box["y1"]) // 2
    bands = (
        (box["y1"] + 2, mid - 2, box["upper"]),
        (mid + 2, box["y2"] - 2, box["lower"]),
    )
    x1 = box["x1"] + 2
    x2 = box["x2"] - 2
    for top, bottom, colour in bands:
        region = pixels[top:bottom, x1:x2]
        local = dist_bg[top:bottom, x1:x2]
        dist = np.sum((region - np.array(colour, dtype=np.int32)) ** 2, axis=2)
        assert np.all(dist < local)


def test_decoded_pixels_follow_the_truth_boxes(small: RouteFixture) -> None:
    """mp4v shifts flat colours by a few levels, so the check is the nearest one."""
    truth = _load(small.truth_path)
    background = tuple(int(channel) for channel in truth["background_bgr"])
    palette = [background]
    by_camera: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for figure in truth["figures"]:
        upper = tuple(int(channel) for channel in figure["upper_bgr"])
        lower = tuple(int(channel) for channel in figure["lower_bgr"])
        for colour in (upper, lower):
            if colour not in palette:
                palette.append(colour)
        for box in figure["boxes"]:
            recorded = dict(box)
            recorded["upper"] = upper
            recorded["lower"] = lower
            by_camera.setdefault(box["camera_id"], {}).setdefault(box["frame_index"], []).append(
                recorded
            )
    for clip in small.clip_paths:
        capture = cv2.VideoCapture(str(clip))
        try:
            for frame_index in range(int(truth["fps"]) * int(truth["seconds"])):
                ok, image = capture.read()
                assert ok and image is not None
                boxes = by_camera.get(clip.stem, {}).get(frame_index, [])
                nearest = _nearest_is_background(image, background, palette)
                pixels = image.astype(np.int32)
                dist_bg = np.sum((pixels - np.array(background, dtype=np.int32)) ** 2, axis=2)
                halo = np.zeros(image.shape[:2], dtype=bool)
                for box in boxes:
                    _assert_band_interiors(pixels, dist_bg, box)
                    y1, y2 = max(0, box["y1"] - 2), min(image.shape[0], box["y2"] + 2)
                    x1, x2 = max(0, box["x1"] - 2), min(image.shape[1], box["x2"] + 2)
                    halo[y1:y2, x1:x2] = True
                assert np.all(nearest[~halo])
        finally:
            capture.release()


def _box_iou(
    truth_box: tuple[float, float, float, float],
    det_box: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = truth_box
    bx1, by1, bx2, by2 = det_box
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    area = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / area


def _match_truth_boxes(fixture: RouteFixture) -> tuple[int, int, int]:
    truth = _load(fixture.truth_path)
    expected: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
    for figure in truth["figures"]:
        for box in figure["boxes"]:
            key = (box["camera_id"], box["frame_index"])
            expected.setdefault(key, []).append(
                (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))
            )
    detector = scripted_figures_detector()
    matched = 0
    total = 0
    false_positives = 0
    fps = float(truth["fps"])
    for clip in fixture.clip_paths:
        camera_id = clip.stem
        capture = cv2.VideoCapture(str(clip))
        try:
            frame_index = 0
            while True:
                ok, image = capture.read()
                if not ok or image is None:
                    break
                frame = Frame(
                    image=image,
                    timestamp_seconds=frame_index / fps,
                    frame_number=frame_index,
                    camera_id=camera_id,
                )
                detections = detector.detect(frame)
                used: set[int] = set()
                for truth_box in expected.get((camera_id, frame_index), []):
                    total += 1
                    for det_index, detection in enumerate(detections):
                        if det_index in used:
                            continue
                        if _box_iou(truth_box, detection.box) >= 0.5:
                            used.add(det_index)
                            matched += 1
                            break
                false_positives += len(detections) - len(used)
                frame_index += 1
        finally:
            capture.release()
    return matched, total, false_positives


def test_scripted_figures_match_every_truth_box(
    defaults: RouteFixture,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Decoded defaults, and one other seed. IoU is one to one. No extras."""
    other = _once(
        tmp_path_factory,
        seed=123,
        n_cameras=4,
        n_figures=2,
        fps=10,
        size=(320, 240),
        seconds=30,
    )
    matched, total, false_positives = _match_truth_boxes(defaults)
    assert matched == total == 3580
    assert false_positives == 0
    matched, total, false_positives = _match_truth_boxes(other)
    assert matched == total == 1177
    assert false_positives == 0


def _stamp_file(path: Path, blob: bytes) -> bytes:
    path.write_bytes(blob)
    route_generator._zero_mp4_times(path)
    return path.read_bytes()


def _timed_box(kind: bytes, version: int, stamp: bytes) -> bytes:
    payload = bytes([version, 0, 0, 0]) + stamp
    return (8 + len(payload)).to_bytes(4, "big") + kind + payload


def _container(kind: bytes, children: bytes) -> bytes:
    return (8 + len(children)).to_bytes(4, "big") + kind + children


def test_container_time_forms_are_cleared(tmp_path: Path) -> None:
    """Version 1, a 64-bit size, and a size of 0 all lose their stamps."""
    version_1 = bytes([1, 0, 0, 0]) + b"\x22" * 16 + b"\x33" * 20
    box = (8 + len(version_1)).to_bytes(4, "big") + b"mvhd" + version_1
    cleared = _stamp_file(tmp_path / "v1.mp4", box)
    assert cleared[12:28] == b"\x00" * 16

    inner_payload = bytes([0, 0, 0, 0]) + b"\x44" * 8 + b"\x00" * 4
    inner = (8 + len(inner_payload)).to_bytes(4, "big") + b"mvhd" + inner_payload
    total = 16 + len(inner)
    moov = (1).to_bytes(4, "big") + b"moov" + total.to_bytes(8, "big") + inner
    cleared = _stamp_file(tmp_path / "large.mp4", moov)
    assert cleared[28:36] == b"\x00" * 8

    body = bytes([0, 0, 0, 0]) + b"\x55" * 8
    blob = (0).to_bytes(4, "big") + b"mvhd" + body
    cleared = _stamp_file(tmp_path / "to-end.mp4", blob)
    assert cleared[12:20] == b"\x00" * 8

    short = (100).to_bytes(4, "big") + b"mvhd" + b"\x66" * 4
    assert _stamp_file(tmp_path / "short.mp4", short) == short
    oversize = (40).to_bytes(4, "big") + b"mvhd" + bytes([0, 0, 0, 0]) + b"\x11" * 8
    assert _stamp_file(tmp_path / "oversize.mp4", oversize) == oversize
    short_version_0 = (19).to_bytes(4, "big") + b"mvhd" + bytes([0, 0, 0, 0]) + b"\x11" * 7
    assert _stamp_file(tmp_path / "short-v0.mp4", short_version_0) == short_version_0
    short_version_1 = (27).to_bytes(4, "big") + b"mvhd" + bytes([1, 0, 0, 0]) + b"\x22" * 15
    assert _stamp_file(tmp_path / "short-v1.mp4", short_version_1) == short_version_1
    tiny = (1).to_bytes(4, "big") + b"mvhd" + b"\x01" * 2
    assert _stamp_file(tmp_path / "tiny.mp4", tiny) == tiny
    header_only = (0).to_bytes(4, "big") + b"mvhd"
    assert _stamp_file(tmp_path / "header.mp4", header_only) == header_only
    other = bytes([2, 0, 0, 0]) + b"\x77" * 16
    other_box = (8 + len(other)).to_bytes(4, "big") + b"mvhd" + other
    assert _stamp_file(tmp_path / "other.mp4", other_box) == other_box


def test_tkhd_and_mdhd_inside_every_container_are_cleared(tmp_path: Path) -> None:
    """Non-zero tkhd, mdhd, and a version-1 stamp sit under the container chain."""
    timed = b"".join(
        [
            _timed_box(b"tkhd", 0, b"\x11" * 8),
            _timed_box(b"mdhd", 1, b"\x22" * 16),
            _timed_box(b"mvhd", 0, b"\x33" * 8),
        ]
    )
    free = (12).to_bytes(4, "big") + b"free" + b"\xab\xcd\xef\x01"
    blob = free + timed
    for kind in (b"edts", b"stbl", b"minf", b"mdia", b"trak", b"moov"):
        blob = _container(kind, blob)
    cleared = _stamp_file(tmp_path / "chain.mp4", blob)
    assert b"\x11" * 8 not in cleared
    assert b"\x22" * 16 not in cleared
    assert b"\x33" * 8 not in cleared
    assert b"\xab\xcd\xef\x01" in cleared


def test_edge_of_buffer_container_sizes_keep_their_meaning(tmp_path: Path) -> None:
    """A stamp that ends on the last byte is cleared. A size that runs past is not."""
    exact = _timed_box(b"mvhd", 1, b"\x22" * 16)
    assert b"\x22" * 16 not in _stamp_file(tmp_path / "exact-v1.mp4", exact)

    body = bytes([0, 0, 0, 0]) + b"\x55" * 8
    inner = (0).to_bytes(4, "big") + b"mvhd" + body
    shifted = (8).to_bytes(4, "big") + b"free" + inner
    cleared = _stamp_file(tmp_path / "shifted-zero.mp4", shifted)
    assert b"\x55" * 8 not in cleared
    assert b"free" in cleared

    plain = _timed_box(b"mvhd", 0, b"\x11" * 8)
    fit = 16 + len(plain)
    large = (1 << 56) + fit
    huge = (1).to_bytes(4, "big") + b"moov" + large.to_bytes(8, "big") + plain
    assert b"\x11" * 8 in _stamp_file(tmp_path / "high-byte.mp4", huge)


def test_short_box_size_boundary_is_eight_bytes(tmp_path: Path) -> None:
    """Seven bytes is not a box. Eight bytes of a 64-bit size 0 raises."""
    seven = (0).to_bytes(4, "big") + b"mvh"
    path = tmp_path / "seven.mp4"
    path.write_bytes(seven)
    route_generator._zero_mp4_times(path)
    assert path.read_bytes() == seven
    eight = (1).to_bytes(4, "big") + b"free"
    path.write_bytes(eight)
    with _time_limit(5), pytest.raises(ValueError, match="shorter than its 16-byte header"):
        route_generator._zero_mp4_times(path)
    assert path.read_bytes() == eight


def test_a_narrow_box_raises_before_the_walk_continues(tmp_path: Path) -> None:
    blob = (4).to_bytes(4, "big") + b"mvhd" + b"\x22" * 8
    path = tmp_path / "narrow-direct.mp4"
    path.write_bytes(blob)
    with pytest.raises(ValueError, match="shorter than its 8-byte header"):
        route_generator._zero_mp4_times(path)
    assert path.read_bytes() == blob


def test_a_short_box_size_raises_instead_of_hanging(tmp_path: Path) -> None:
    """A 64-bit size of 0 used to spin. A 32-bit size of 4 is shorter than a header."""
    program = "\n".join(
        [
            "import sys",
            "from pathlib import Path",
            "from trace_search.fixtures.route_generator import _zero_mp4_times",
            "try:",
            "    _zero_mp4_times(Path(sys.argv[1]))",
            "except ValueError as exc:",
            "    print(exc)",
            "    raise SystemExit(0)",
            "raise SystemExit(1)",
        ]
    )
    zero_large = (1).to_bytes(4, "big") + b"free" + (0).to_bytes(8, "big") + b"\x11" * 8
    narrow = (4).to_bytes(4, "big") + b"mvhd" + b"\x22" * 8
    for name, blob in (("zero-large.mp4", zero_large), ("narrow.mp4", narrow)):
        path = tmp_path / name
        path.write_bytes(blob)
        completed = subprocess.run(
            [sys.executable, "-c", program, str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert completed.returncode == 0
        assert "shorter than" in completed.stdout
        assert path.read_bytes() == blob


def test_repo_root_resolves_a_symlinked_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.symlink_to(_REPO, target_is_directory=True)
    linked = checkout / "src" / "trace_search" / "fixtures" / "route_generator.py"
    assert linked.resolve() == _SOURCE.resolve()
    assert Path(str(linked)).parents[3] != _REPO.resolve()
    monkeypatch.setattr(route_generator, "__file__", str(linked))
    assert route_generator._repo_root() == _REPO.resolve()
    leaked = _REPO / "out-probe"
    try:
        with pytest.raises(ValueError, match="inside the repository"):
            generate_route_fixture(
                checkout / "out-probe",
                seed=0,
                n_cameras=4,
                n_figures=2,
                fps=10,
                size=(320, 240),
                seconds=30,
            )
        assert not leaked.exists()
    finally:
        if leaked.exists():
            shutil.rmtree(leaked)


def test_licence_row_matches_the_source_and_has_no_use_tag() -> None:
    rows = _generator_rows(_manifest())
    assert len(rows) == 2
    row = _route_row()
    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()
    assert row["sha256"] == digest
    assert row["type"] == "generator"
    assert row["license"] == "MIT"
    assert row["generator_license"] == "MIT"
    assert row["version"] == _BASE
    assert row["seed"] == "0..2147483647"
    assert row["source_url"] == "https://github.com/shitanshusharma/trace-search"
    assert "use" not in row
    assert "fixture_id" not in row
    notes = str(row["notes"])
    assert "temp files, never committed" in notes
    assert "in-repo generator row" in notes
    assert "version is the base commit" in notes
    assert "not a content pin" in notes
    assert "sha256" in notes
    assert check_licenses.validate_schema(_manifest(), _SCHEMA) == []
    assert check_licenses.check_manifest_licenses(_manifest()) == []


def test_planted_smoke_only_generator_row_fails_closed(tmp_path: Path) -> None:
    """The committed generator row passes the licence checks.

    The schema enum allows ``SMOKE_ONLY`` on a generator row, so the refusal
    is the separation check: a ``SMOKE_ONLY`` row with no pHash list fails
    closed. The planted file is a temp copy.
    """
    planted = _manifest()
    found = False
    for entry in planted["entries"]:
        if entry.get("name") == _ROW_NAME and entry.get("type") == "generator":
            entry["use"] = "SMOKE_ONLY"
            found = True
    assert found
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(planted), encoding="utf-8")
    with pytest.raises(SeparationCheckError, match="SMOKE_ONLY"):
        check_separation(path, repo_root=_REPO)
