"""Seeded 6-camera route fixtures: files, truth, palette, and the licence row.

Generation uses ``tmp_path`` only. The pixel check reads
``RouteFixture.sample_frame``, the BGR buffer before mp4v, because the
codec is lossy. A temp copy of the licence manifest, not the repo file,
carries the planted ``SMOKE_ONLY`` generator row.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import pytest

import scripts.check_licenses as check_licenses
from trace_search.fixtures import route_generator
from trace_search.fixtures.guards import (
    FixtureSeparationError,
    LoaderContext,
    check_fixture_allowed,
)
from trace_search.fixtures.route_generator import RouteFixture, generate_route_fixture
from trace_search.fixtures.separation import SeparationCheckError, check_separation
from trace_search.ingest.camera_registry import CameraRegistry
from trace_search.matching.topology import CameraTopology

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / "licenses" / "manifest.json"
_SCHEMA = _REPO / "licenses" / "manifest_schema.json"
_SOURCE = _REPO / "src" / "trace_search" / "fixtures" / "route_generator.py"
_ROW_NAME = "src/trace_search/fixtures/route_generator.py"
_BASE = "8af465e2e3ab3e3f9159bfd993cd0082bec62430"


def _small(out: Path, **overrides: Any) -> RouteFixture:
    params: dict[str, Any] = {
        "seed": 0,
        "n_cameras": 4,
        "n_figures": 2,
        "fps": 10,
        "size": (64, 64),
        "seconds": 1,
    }
    params.update(overrides)
    return generate_route_fixture(out, **params)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(out: Path) -> dict[str, str]:
    return {path.name: _sha(path) for path in sorted(out.iterdir()) if path.is_file()}


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


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> RouteFixture:
    return _small(tmp_path_factory.mktemp("route-small"))


def test_signature_defaults_match_the_task() -> None:
    signature = inspect.signature(generate_route_fixture)
    assert signature.parameters["seed"].default == 0
    assert signature.parameters["n_cameras"].default == 6
    assert signature.parameters["n_figures"].default == 4
    assert signature.parameters["fps"].default == 10
    assert signature.parameters["size"].default == (640, 360)
    assert signature.parameters["seconds"].default == 30


def test_same_seed_is_byte_identical_and_a_later_second_still_matches(tmp_path: Path) -> None:
    first = _small(tmp_path / "first")
    time.sleep(1.1)
    second = _small(tmp_path / "second")
    assert _hashes(first.out_dir) == _hashes(second.out_dir)
    assert first.sample_frame.tobytes() == second.sample_frame.tobytes()
    names = set(_hashes(first.out_dir))
    assert names == {
        "CAM-1.mp4",
        "CAM-2.mp4",
        "CAM-3.mp4",
        "CAM-4.mp4",
        "cameras.json",
        "topology.json",
        "truth.json",
        "manifest.json",
    }


def test_a_different_seed_changes_the_clip_bytes(tmp_path: Path) -> None:
    left = _small(tmp_path / "seed-0", seed=0)
    right = _small(tmp_path / "seed-1", seed=1)
    assert _sha(left.clip_paths[0]) != _sha(right.clip_paths[0])
    assert _sha(left.truth_path) != _sha(right.truth_path)
    assert _sha(left.manifest_path) != _sha(right.manifest_path)


def test_six_cameras_use_planted_offsets_and_fx_code_ids(tmp_path: Path) -> None:
    fixture = _small(tmp_path / "six", n_cameras=6, n_figures=4)
    registry = CameraRegistry.load(fixture.cameras_path)
    offsets = [registry.get(f"CAM-{index}").clock_offset_seconds for index in range(1, 7)]
    assert offsets == [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
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
    manifest = _load(fixture.manifest_path)
    entries = manifest["entries"]
    assert [entry["fixture_id"] for entry in entries] == [
        "FX-CODE-101",
        "FX-CODE-102",
        "FX-CODE-103",
        "FX-CODE-104",
        "FX-CODE-105",
        "FX-CODE-106",
    ]
    for entry, clip in zip(entries, fixture.clip_paths):
        assert entry["type"] == "code_fixture"
        assert entry["use"] == "CODE_TEST"
        assert entry["generator"] == "generate_route_fixture"
        assert entry["generator_license"] == "MIT"
        assert entry["sha256"] == _sha(clip)
        assert len(entry["phash"]) == 10
        assert all(len(item) == 16 for item in entry["phash"])


def test_truth_gaps_follow_topology_and_boxes_stay_inside(small: RouteFixture) -> None:
    truth = _load(small.truth_path)
    topology = CameraTopology.from_dict(_load(small.topology_path))
    registry = CameraRegistry.load(small.cameras_path)
    figures = truth["figures"]
    assert len(figures) == 2
    primary = figures[0]
    distractors = [figure for figure in figures if figure["distractor"]]
    assert len(distractors) == 1
    distractor = distractors[0]
    assert distractor["upper_bgr"] == primary["upper_bgr"]
    assert distractor["lower_bgr"] == primary["lower_bgr"]
    assert distractor["shares_colours_with"] == primary["figure_id"]
    left = {visit["camera_id"] for visit in primary["visits"]}
    right = {visit["camera_id"] for visit in distractor["visits"]}
    assert left.isdisjoint(right)
    for origin in left:
        for destination in right:
            assert topology.window(origin, destination) is None
            assert topology.window(destination, origin) is None
    width = truth["width"]
    height = truth["height"]
    fps = truth["fps"]
    for figure in figures:
        visits = figure["visits"]
        assert len(visits) >= 2
        pairs = zip(visits, visits[1:])
        for previous, following in pairs:
            assert topology.allows(
                previous["camera_id"],
                previous["t_exit"],
                following["camera_id"],
                following["t_entry"],
            )
        for visit in visits:
            matched = [box for box in figure["boxes"] if box["camera_id"] == visit["camera_id"]]
            assert matched
            camera = registry.get(visit["camera_id"])
            assert camera is not None
            for box in matched:
                assert 0 <= box["x1"] < box["x2"] <= width
                assert 0 <= box["y1"] < box["y2"] <= height
                corrected = box["frame_index"] / fps + camera.clock_offset_seconds
                assert visit["t_entry"] <= corrected < visit["t_exit"]


def test_sample_frame_uses_only_the_palette_and_exact_bands(small: RouteFixture) -> None:
    """Palette check is the pre-encode buffer. mp4v is not a pixel source."""
    truth = _load(small.truth_path)
    allowed = {tuple(truth["background_bgr"])}
    allowed.add(tuple(truth["mark_upper_bgr"]))
    allowed.add(tuple(truth["mark_lower_bgr"]))
    for figure in truth["figures"]:
        allowed.add(tuple(figure["upper_bgr"]))
        allowed.add(tuple(figure["lower_bgr"]))
    frame = small.sample_frame
    found = {tuple(int(channel) for channel in pixel) for pixel in frame.reshape(-1, 3)}
    assert found <= allowed
    assert tuple(truth["background_bgr"]) in found
    primary = truth["figures"][0]
    assert tuple(primary["upper_bgr"]) in found
    assert tuple(primary["lower_bgr"]) in found
    box = next(
        item
        for item in primary["boxes"]
        if item["camera_id"] == small.sample_camera_id
        and item["frame_index"] == small.sample_frame_index
    )
    mid = box["y1"] + (box["y2"] - box["y1"]) // 2
    upper = tuple(int(channel) for channel in frame[box["y1"], box["x1"]])
    lower = tuple(int(channel) for channel in frame[mid, box["x1"]])
    assert upper == tuple(primary["upper_bgr"])
    assert lower == tuple(primary["lower_bgr"])
    background = tuple(truth["background_bgr"])
    for y_index, row in enumerate(frame):
        for x_index, pixel in enumerate(row):
            colour = tuple(int(channel) for channel in pixel)
            if colour == background:
                continue
            boxes = list(truth["marks"])
            for figure in truth["figures"]:
                boxes.extend(figure["boxes"])
            inside = False
            for item in boxes:
                same_frame = (
                    item["camera_id"] == small.sample_camera_id
                    and item["frame_index"] == small.sample_frame_index
                )
                if not same_frame:
                    continue
                if item["x1"] <= x_index < item["x2"] and item["y1"] <= y_index < item["y2"]:
                    inside = True
            assert inside


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


def test_twelve_cameras_still_walk_an_edge(tmp_path: Path) -> None:
    fixture = _small(tmp_path / "twelve", n_cameras=12, n_figures=2)
    truth = _load(fixture.truth_path)
    topology = CameraTopology.from_dict(_load(fixture.topology_path))
    registry = CameraRegistry.load(fixture.cameras_path)
    assert len(fixture.clip_paths) == 12
    for figure in truth["figures"]:
        visits = figure["visits"]
        assert len(visits) >= 2
        for visit in visits:
            assert any(box["camera_id"] == visit["camera_id"] for box in figure["boxes"])
        for previous, following in zip(visits, visits[1:]):
            assert topology.allows(
                previous["camera_id"],
                previous["t_exit"],
                following["camera_id"],
                following["t_entry"],
            )
        for visit in visits:
            camera = registry.get(visit["camera_id"])
            assert camera is not None
            assert visit["t_exit"] <= camera.clock_offset_seconds + truth["seconds"]


def test_odd_lane_height_still_draws_inside_the_frame(tmp_path: Path) -> None:
    fixture = _small(tmp_path / "three", n_figures=3)
    truth = _load(fixture.truth_path)
    assert len(truth["figures"]) == 3
    for figure in truth["figures"]:
        for box in figure["boxes"]:
            assert box["y2"] <= truth["height"]
            assert (box["y2"] - box["y1"]) % 2 == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"seed": True}, "seed"),
        ({"seed": False}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"seed": "0"}, "seed"),
        ({"n_cameras": True}, "n_cameras"),
        ({"n_cameras": 4.0}, "n_cameras"),
        ({"n_figures": True}, "n_figures"),
        ({"n_figures": "4"}, "n_figures"),
        ({"fps": True}, "fps"),
        ({"fps": 10.0}, "fps"),
        ({"seconds": True}, "seconds"),
        ({"seconds": 1.0}, "seconds"),
    ],
)
def test_bool_and_non_int_arguments_raise_type_error(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    out = tmp_path / "rejected"
    with pytest.raises(TypeError, match=message):
        _small(out, **overrides)
    assert not out.exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": -1},
        {"seed": 2_147_483_648},
        {"n_cameras": 3},
        {"n_cameras": 13},
        {"n_figures": 1},
        {"n_figures": 9},
        {"fps": 0},
        {"fps": 61},
        {"seconds": 0},
        {"seconds": 121},
        {"seconds": 1, "fps": 9},
    ],
)
def test_out_of_range_arguments_raise_value_error(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    out = tmp_path / "rejected"
    with pytest.raises(ValueError):
        _small(out, **overrides)
    assert not out.exists()


@pytest.mark.parametrize(
    ("size", "error"),
    [
        ((64,), ValueError),
        ((64, 64, 1), ValueError),
        ([64, 64], TypeError),
        (None, TypeError),
        ((63, 64), ValueError),
        ((64, 63), ValueError),
        ((1921, 64), ValueError),
        ((64, 1921), ValueError),
        ((True, 64), TypeError),
        ((64, True), TypeError),
        ((64.0, 64), TypeError),
        ((64, 64.0), TypeError),
    ],
)
def test_size_guards(tmp_path: Path, size: object, error: type[BaseException]) -> None:
    out = tmp_path / "rejected"
    with pytest.raises(error):
        _small(out, size=size)
    assert not out.exists()


def test_boundary_values_are_accepted(tmp_path: Path) -> None:
    _small(tmp_path / "seed-max", seed=2_147_483_647)
    _small(tmp_path / "figures", n_figures=8)
    _small(tmp_path / "fps", fps=60, seconds=1)
    _small(tmp_path / "seconds", fps=1, seconds=10)
    _small(tmp_path / "wide", size=(1920, 64))
    _small(tmp_path / "tall", size=(64, 1920))
    _small(tmp_path / "long", seconds=120, fps=1)


def test_out_dir_must_be_a_path(tmp_path: Path) -> None:
    out_dir: object = str(tmp_path / "text")
    with pytest.raises(TypeError, match="out_dir"):
        generate_route_fixture(
            out_dir,
            seed=0,
            n_cameras=4,
            n_figures=2,
            fps=10,
            size=(64, 64),
            seconds=1,
        )


def test_repo_root_and_a_child_and_dotdot_are_refused() -> None:
    with pytest.raises(ValueError, match="inside the repository"):
        _small(_REPO)
    with pytest.raises(ValueError, match="inside the repository"):
        _small(_REPO / "src")
    disguised = _REPO / "src" / ".." / "tests"
    with pytest.raises(ValueError, match="inside the repository"):
        _small(disguised)


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


def test_an_existing_directory_is_accepted(tmp_path: Path) -> None:
    out = tmp_path / "already"
    out.mkdir()
    fixture = _small(out)
    assert fixture.clip_paths[0].is_file()


def test_odd_seed_reverses_the_walk(tmp_path: Path) -> None:
    even = _small(tmp_path / "even", seed=0)
    odd = _small(tmp_path / "odd", seed=1)

    def xs(fixture: RouteFixture) -> list[int]:
        truth = _load(fixture.truth_path)
        boxes = [box for box in truth["figures"][0]["boxes"] if box["camera_id"] == "CAM-1"]
        return [box["x1"] for box in boxes]

    even_x = xs(even)
    odd_x = xs(odd)
    assert even_x[0] < even_x[-1]
    assert odd_x[0] > odd_x[-1]


def test_odd_frame_height_keeps_an_even_mark(tmp_path: Path) -> None:
    fixture = _small(tmp_path / "tall-odd", size=(64, 66))
    mark = _load(fixture.truth_path)["marks"][0]
    assert (mark["y2"] - mark["y1"]) % 2 == 0
    assert mark["y2"] <= 66


def test_a_one_camera_prefix_is_not_a_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def schedule(
        path: tuple[int, ...],
        offsets: tuple[float, ...],
        _seconds: int,
        _fps: int,
    ) -> tuple[tuple[int, float, float], ...] | None:
        if len(path) >= 2:
            return None
        camera = path[0]
        start = offsets[camera]
        return ((camera, start, start + 0.2),)

    monkeypatch.setattr(route_generator, "_schedule_path", schedule)
    with pytest.raises(RuntimeError, match="did not produce a route"):
        _small(tmp_path / "one-camera")


def test_a_zone_that_never_fits_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(route_generator, "_schedule_path", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="did not produce a route"):
        _small(tmp_path / "unfit")


def _stamp_file(path: Path, blob: bytes) -> bytes:
    path.write_bytes(blob)
    route_generator._zero_mp4_times(path)
    return path.read_bytes()


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
    # Declared size runs past the buffer. The walk stops, so the stamp stays.
    oversize = (40).to_bytes(4, "big") + b"mvhd" + bytes([0, 0, 0, 0]) + b"\x11" * 8
    assert _stamp_file(tmp_path / "oversize.mp4", oversize) == oversize
    # One byte short of a version-0 stamp. A longer write would grow the file.
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
