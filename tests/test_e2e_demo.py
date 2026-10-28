"""End-to-end synthetic demo. Numbers are not evidence for any AC."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from scripts.run_e2e_demo import main
from trace_search.attributes.store import TrackAttributeStore
from trace_search.detection.types import DetClass
from trace_search.fixtures.guards import FixtureSeparationError
from trace_search.matching.route_metrics import TruthSegment
from trace_search.matching.tracklets import Tracklet
from trace_search.pipeline import e2e
from trace_search.pipeline.e2e import (
    ATTRIBUTE_DB_FILENAME,
    DERIVED_FROM,
    EMBEDDER_DIM,
    EMBEDDING_DB_FILENAME,
    FALSE_MERGE_MAX,
    IOU_MIN,
    LABEL,
    SEGMENT_PRECISION_MIN,
    SEGMENT_RECALL_MIN,
    TRACK_DB_FILENAME,
    DemoReport,
    FigureRecord,
    FigureReport,
    Schedule,
    TruthBox,
    assert_outside_repo,
    box_iou,
    choose_seed,
    clip_file,
    delete_run_rows,
    describe_cameras,
    metric_marks,
    new_workdir,
    remove_workdir,
    render_json,
    render_text,
    require_code_test_clip,
    route_from_seed,
    run_demo,
    score_demo,
    track_ownership,
)
from trace_search.reid.sqlite_vector_store import SqliteVectorStore
from trace_search.reid.types import AppearanceEmbedding
from trace_search.tracking.store import TrackStore
from trace_search.tracking.types import TrackObservation

_REPO = Path(__file__).resolve().parents[1]
_MEDIA = (
    "cameras.json",
    "topology.json",
    "truth.json",
    "manifest.json",
)
_STATUS_AT_IMPORT = subprocess.run(
    ["git", "status", "--porcelain"],
    check=True,
    capture_output=True,
    text=True,
    cwd=_REPO,
).stdout


@dataclass(frozen=True)
class _DemoRun:
    code: int
    text: str
    elapsed: float
    root: Path
    live: Path
    run_id: str
    remaining_tracks: int
    routes: dict[str, Any]
    cameras: dict[str, Any]
    truth: dict[str, Any]
    run: dict[str, Any]


def _porcelain() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    return result.stdout


def _line(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"{prefix} missing from:\n{text}")


def _metric(text: str, label: str) -> tuple[float, str]:
    line = _line(text, f"{label}=")
    value = float(line.split("=", 1)[1].split()[0])
    return value, line


def _observation(
    camera_id: str,
    track_id: int,
    frame_number: int,
    box: tuple[float, float, float, float],
    *,
    det_class: DetClass = DetClass.PERSON,
    timestamp_seconds: float = 0.0,
) -> TrackObservation:
    return TrackObservation(
        camera_id,
        track_id,
        det_class,
        frame_number,
        timestamp_seconds,
        box,
        0.95,
    )


def _tracklet(camera_id: str, track_id: int) -> Tracklet:
    box = (0.0, 0.0, 10.0, 10.0)
    return Tracklet(
        camera_id,
        track_id,
        DetClass.PERSON,
        "run",
        0.0,
        1.0,
        1,
        box,
        box,
        (0,),
    )


def _report(
    recall: float,
    precision: float,
    false_merge: float,
    *,
    figures: tuple[FigureReport, ...] = (),
    skipped: tuple[str, ...] = (),
) -> DemoReport:
    return DemoReport("run-1", recall, precision, false_merge, figures, skipped, 1.25)


def _copy_tree(src: Path, dest: Path) -> None:
    shutil.copytree(src, dest)


def _media_copy(src: Path, dest: Path) -> None:
    dest.mkdir()
    for name in _MEDIA:
        shutil.copy(src / name, dest / name)
    for path in sorted(src.glob("CAM-*.mp4")):
        shutil.copy(path, dest / path.name)


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _plant_cross_figure(root: Path) -> None:
    """Add one out-of-window track that copies F1's embedding onto F2's lane.

    Lockstep figures already share a gap inside the transit window, so opening
    that window does not change their links. This row sits 50 seconds later.
    The closed window refuses it. The open window can take it.
    """
    routes = _read_json(root / "routes.json")
    truth = _read_json(root / "truth.json")
    run_id = _read_json(root / "run.json")["run_id"]
    assert isinstance(run_id, str)
    figures = routes["figures"]
    assert isinstance(figures, list)
    seed = next(figure for figure in figures if figure["figure_id"] == "F1")
    segments = seed["segments"]
    assert isinstance(segments, list)
    origin = segments[0]
    nxt = segments[1]
    partner = next(figure for figure in truth["figures"] if figure["figure_id"] == "F2")
    box = next(item for item in partner["boxes"] if item["camera_id"] == nxt["camera_id"])
    vectors = SqliteVectorStore(root / EMBEDDING_DB_FILENAME, EMBEDDER_DIM)
    total = vectors.count(DERIVED_FROM, run_id)
    probe = tuple(1.0 if index == 0 else 0.0 for index in range(EMBEDDER_DIM))
    hits = vectors.search(probe, total, run_id=run_id, derived_from=DERIVED_FROM)
    source = next(
        item
        for item, _score in hits
        if item.camera_id == origin["camera_id"] and item.track_id == origin["track_id"]
    )
    attrs = TrackAttributeStore(root / ATTRIBUTE_DB_FILENAME)
    attr = attrs.read(str(origin["camera_id"]), int(origin["track_id"]), run_id)[-1]
    store = TrackStore(root / TRACK_DB_FILENAME)
    store.write(
        run_id,
        "planted",
        DERIVED_FROM,
        [
            _observation(
                str(nxt["camera_id"]),
                9001,
                int(box["frame_index"]),
                (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])),
                timestamp_seconds=float(origin["t_exit"]) + 50.0,
            )
        ],
    )
    attrs.write(
        str(nxt["camera_id"]),
        9001,
        run_id,
        DERIVED_FROM,
        attr.upper_color,
        attr.lower_color,
        attr.bag_on_track,
        attr.upper_confidence,
        attr.lower_confidence,
        attr.n_frames,
    )
    vectors.add(
        run_id,
        DERIVED_FROM,
        [
            AppearanceEmbedding(
                str(nxt["camera_id"]),
                9001,
                0,
                tuple(source.vector),
                source.embedder_name,
                source.placeholder,
            )
        ],
    )


@pytest.fixture(scope="session")
def demo_run(tmp_path_factory: pytest.TempPathFactory) -> _DemoRun:
    """One real pipeline. Snapshot the files, then the default run deletes them."""
    live = tmp_path_factory.mktemp("e2e-live")
    saved = tmp_path_factory.mktemp("e2e-saved")
    seen: dict[str, object] = {}
    real_delete = e2e.delete_run_rows

    def _wrapped(root: Path, run_id: str) -> None:
        shutil.copytree(root, saved, dirs_exist_ok=True)
        real_delete(root, run_id)
        seen["run_id"] = run_id
        seen["tracks"] = TrackStore(root / TRACK_DB_FILENAME).count(run_id=run_id)

    e2e.delete_run_rows = _wrapped
    buf = io.StringIO()
    try:
        import time

        started = time.perf_counter()
        code = main(["--workdir", str(live), "--seed", "0"], sink=buf)
        elapsed = time.perf_counter() - started
    finally:
        e2e.delete_run_rows = real_delete
    text = buf.getvalue()
    if code != 0 or not saved.is_dir():
        raise AssertionError(text)
    run_id = seen["run_id"]
    remaining = seen["tracks"]
    assert isinstance(run_id, str)
    assert isinstance(remaining, int)
    return _DemoRun(
        code,
        text,
        elapsed,
        saved,
        live,
        run_id,
        remaining,
        _read_json(saved / "routes.json"),
        _read_json(saved / "cameras.json"),
        _read_json(saved / "truth.json"),
        _read_json(saved / "run.json"),
    )


def test_main_passes_and_default_run_is_removed(demo_run: _DemoRun) -> None:
    assert demo_run.code == 0
    assert demo_run.elapsed < 180.0
    assert not demo_run.live.exists()
    assert demo_run.remaining_tracks == 0
    lines = demo_run.text.splitlines()
    assert lines[0] == LABEL
    assert "not evidence" in lines[0]
    assert "workdir=" not in demo_run.text
    assert "Traceback" not in demo_run.text
    for figure in demo_run.truth["figures"]:
        count = len(figure["visits"])
        assert _line(demo_run.text, f"{figure['figure_id']}: ") == (
            f"{figure['figure_id']}: {count}/{count}"
        )
    recall, recall_line = _metric(demo_run.text, "segment_recall")
    precision, precision_line = _metric(demo_run.text, "segment_precision")
    merges, merge_line = _metric(demo_run.text, "false_merge_rate")
    assert recall >= 0.80
    assert precision >= 0.90
    assert merges <= 0.05
    assert "PASS" in recall_line
    assert "PASS" in precision_line
    assert "PASS" in merge_line
    assert "(>= 0.80)" in recall_line
    assert "(>= 0.90)" in precision_line
    assert "(<= 0.05)" in merge_line
    assert _line(demo_run.text, "run time ").startswith("run time ")
    assert _line(demo_run.text, "run_id=") == f"run_id={demo_run.run_id}"
    assert _porcelain() == _STATUS_AT_IMPORT


def test_demo_decodes_real_clips_and_holds_clock_extremes(demo_run: _DemoRun) -> None:
    offsets = [float(camera["clock_offset_seconds"]) for camera in demo_run.cameras["cameras"]]
    assert min(offsets) == -2.0
    assert max(offsets) == 203.0
    frames = demo_run.run["frames"]
    assert demo_run.run["seconds"] == 40
    assert demo_run.run["n_cameras"] == 6
    assert demo_run.run["n_figures"] == 4
    assert demo_run.run["width"] == 320
    assert demo_run.run["height"] == 240
    assert set(frames) == {f"CAM-{index}" for index in range(1, 7)}
    assert all(count == 400 for count in frames.values())
    for camera_id in frames:
        clip = demo_run.root / f"{camera_id}.mp4"
        assert clip.is_file()
        assert clip.stat().st_size > 0
    tracks = TrackStore(demo_run.root / "tracks.sqlite")
    attributes = TrackAttributeStore(demo_run.root / "attributes.sqlite")
    vectors = SqliteVectorStore(demo_run.root / "embeddings.sqlite", EMBEDDER_DIM)
    assert tracks.count(run_id=demo_run.run_id, derived_from="CODE_TEST") > 0
    assert tracks.count(run_id=demo_run.run_id, derived_from="SMOKE") == 0
    assert attributes.count(run_id=demo_run.run_id, derived_from="CODE_TEST") > 0
    assert vectors.count("CODE_TEST", demo_run.run_id) > 0
    assert tracks.db_path == (demo_run.root / "tracks.sqlite").resolve()
    assert attributes.db_path == (demo_run.root / "attributes.sqlite").resolve()
    assert vectors.db_path == (demo_run.root / "embeddings.sqlite").resolve()


def test_lockstep_figures_keep_their_own_tracks(demo_run: _DemoRun) -> None:
    observations = e2e._read_observations(demo_run.root, demo_run.run_id)
    figures = e2e._load_figures(demo_run.root / "truth.json")
    owned = track_ownership(observations, e2e._all_boxes(figures))
    grouped: dict[tuple[tuple[str, float, float], ...], list[dict[str, Any]]] = {}
    truth_by_id = {figure["figure_id"]: figure for figure in demo_run.truth["figures"]}
    for figure in demo_run.routes["figures"]:
        visits = truth_by_id[figure["figure_id"]]["visits"]
        key = tuple(
            (visit["camera_id"], float(visit["t_entry"]), float(visit["t_exit"]))
            for visit in visits
        )
        grouped.setdefault(key, []).append(figure)
        assert figure["found"] == figure["truth"]
        assert figure["seed_camera"] is not None
        assert figure["seed_track"] is not None
        for segment in figure["segments"]:
            pair = (segment["camera_id"], segment["track_id"])
            assert owned[pair] == figure["figure_id"]
    assert len(grouped) == 2
    for group in grouped.values():
        assert len(group) == 2
        assert group[0]["seed_camera"] == group[1]["seed_camera"]
        assert group[0]["seed_track"] != group[1]["seed_track"]


def test_json_report_is_not_evidence(demo_run: _DemoRun) -> None:
    buf = io.StringIO()
    code = score_demo(
        demo_run.root,
        emit_json=True,
        elapsed_seconds=demo_run.elapsed,
        sink=buf,
    )
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert payload["evidence"] is False
    assert payload["notice"] == "not evidence"
    assert payload["label"] == LABEL
    assert payload["passed"] is True
    assert payload["results"] == {
        "false_merge_rate": "PASS",
        "segment_precision": "PASS",
        "segment_recall": "PASS",
    }
    assert payload["thresholds"]["segment_recall"] == 0.80
    assert payload["thresholds"]["segment_precision"] == 0.90
    assert payload["thresholds"]["false_merge_rate"] == 0.05
    assert payload["segment_recall"] >= 0.80
    assert payload["run_id"] == demo_run.run_id
    assert "Traceback" not in buf.getvalue()


def test_shrunk_window_fails_recall(demo_run: _DemoRun, tmp_path: Path) -> None:
    root = tmp_path / "shrunk"
    _copy_tree(demo_run.root, root)
    buf = io.StringIO()
    code = score_demo(root, shrink_first_window=True, sink=buf)
    text = buf.getvalue()
    assert code == 1
    assert text.splitlines()[0] == LABEL
    recall, line = _metric(text, "segment_recall")
    assert recall < 0.80
    assert "FAIL" in line
    assert "Traceback" not in text
    found_text, truth_text = _line(text, "F1: ").split(": ", 1)[1].split("/")
    assert int(found_text) < int(truth_text)


def test_open_window_fails_false_merge(demo_run: _DemoRun, tmp_path: Path) -> None:
    root = tmp_path / "open"
    _copy_tree(demo_run.root, root)
    _plant_cross_figure(root)
    closed = io.StringIO()
    closed_code = score_demo(root, sink=closed)
    assert closed_code == 0
    assert "PASS" in _metric(closed.getvalue(), "false_merge_rate")[1]
    opened = io.StringIO()
    code = score_demo(root, enforce_time_window=False, sink=opened)
    text = opened.getvalue()
    assert code == 1
    assert text.splitlines()[0] == LABEL
    merges, line = _metric(text, "false_merge_rate")
    assert merges > 0.05
    assert "FAIL" in line
    assert "Traceback" not in text


def test_empty_camera_is_skipped_and_its_rows_stay(demo_run: _DemoRun, tmp_path: Path) -> None:
    root = tmp_path / "empty"
    _copy_tree(demo_run.root, root)
    run_id = demo_run.run_id
    attrs = TrackAttributeStore(root / ATTRIBUTE_DB_FILENAME)
    vectors = SqliteVectorStore(root / EMBEDDING_DB_FILENAME, EMBEDDER_DIM)
    before_cam = len(attrs.read("CAM-1", run_id=run_id))
    attrs.write("CAM-EMPTY", 7, run_id, DERIVED_FROM, "red", "blue", False, 0.91, 0.92, 3)
    unit = tuple(1.0 if index == 0 else 0.0 for index in range(EMBEDDER_DIM))
    vectors.add(
        run_id,
        DERIVED_FROM,
        [AppearanceEmbedding("CAM-EMPTY", 7, 0, unit, "untrained-placeholder-v1", True)],
    )
    before_vector = vectors.count(DERIVED_FROM, run_id)
    reasons = describe_cameras(root, [("CAM-EMPTY", demo_run.root / "CAM-1.mp4")], run_id)
    assert len(reasons) == 1
    assert reasons[0].startswith("CAM-EMPTY: ")
    assert "no planned sample frame was read" in reasons[0]
    assert "Traceback" not in reasons[0]
    kept = attrs.read("CAM-EMPTY", 7, run_id)
    assert len(kept) == 1
    assert kept[0].upper_color == "red"
    assert kept[0].lower_color == "blue"
    assert len(attrs.read("CAM-1", run_id=run_id)) == before_cam
    assert vectors.count(DERIVED_FROM, run_id) == before_vector
    buf = io.StringIO()
    code = score_demo(root, skipped=reasons, sink=buf)
    text = buf.getvalue()
    assert text.splitlines()[0] == LABEL
    assert f"skipped {len(reasons)}: {reasons[0]}" in text
    assert "Traceback" not in text
    assert code == 0


def test_unexpected_describe_error_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = tmp_path / "CAM-1.mp4"
    clip.write_bytes(b"not a decoded clip")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("encoder")

    monkeypatch.setattr(e2e, "describe_tracks", _boom)
    with pytest.raises(RuntimeError, match="encoder"):
        describe_cameras(tmp_path, [("CAM-1", clip)], "run-x")


def test_non_code_test_is_refused_before_a_store_exists(demo_run: _DemoRun, tmp_path: Path) -> None:
    root = tmp_path / "tagged"
    _media_copy(demo_run.root, root)
    manifest = _read_json(root / "manifest.json")
    for entry in manifest["entries"]:
        entry["use"] = "SMOKE_ONLY"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    buf = io.StringIO()
    code = run_demo(root, generate=False, keep=True, sink=buf)
    text = buf.getvalue()
    assert code == 1
    assert "FORBIDDEN" in text
    assert "Traceback" not in text
    assert not (root / TRACK_DB_FILENAME).exists()
    assert not (root / ATTRIBUTE_DB_FILENAME).exists()
    assert not (root / EMBEDDING_DB_FILENAME).exists()
    assert f"workdir={root.resolve()}" in text


def test_missing_manifest_row_is_refused(demo_run: _DemoRun, tmp_path: Path) -> None:
    root = tmp_path / "unlisted"
    _media_copy(demo_run.root, root)
    manifest = _read_json(root / "manifest.json")
    manifest["entries"] = []
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    clip = root / "CAM-1.mp4"
    with pytest.raises(FixtureSeparationError, match="no manifest row"):
        require_code_test_clip(clip, root / "manifest.json")
    buf = io.StringIO()
    code = run_demo(root, generate=False, keep=True, sink=buf)
    assert code == 1
    assert "no manifest row" in buf.getvalue()
    assert not (root / TRACK_DB_FILENAME).exists()


def test_require_code_test_accepts_the_generated_clip(demo_run: _DemoRun) -> None:
    clip = demo_run.root / "CAM-1.mp4"
    fixture_id = require_code_test_clip(clip, demo_run.root / "manifest.json")
    assert fixture_id.startswith("FX-CODE-")


def test_missing_manifest_file_is_refused(demo_run: _DemoRun, tmp_path: Path) -> None:
    root = tmp_path / "no-manifest"
    _media_copy(demo_run.root, root)
    (root / "manifest.json").unlink()
    buf = io.StringIO()
    code = run_demo(root, generate=False, keep=True, sink=buf)
    assert code == 1
    assert "manifest not found" in buf.getvalue()
    assert not (root / TRACK_DB_FILENAME).exists()


def test_workdir_inside_the_repo_writes_nothing() -> None:
    target = _REPO / "e2e-not-a-workdir"
    before = _porcelain()
    buf = io.StringIO()
    code = main(["--workdir", str(target)], sink=buf)
    assert code == 1
    assert "inside the repository" in buf.getvalue()
    assert "Traceback" not in buf.getvalue()
    assert not target.exists()
    assert _porcelain() == before


def test_symlink_into_the_repo_is_refused(tmp_path: Path) -> None:
    link = tmp_path / "repo-link"
    link.symlink_to(_REPO)
    target = link / "nested-e2e"
    buf = io.StringIO()
    code = main(["--workdir", str(target)], sink=buf)
    assert code == 1
    assert "inside the repository" in buf.getvalue()
    assert not (_REPO / "nested-e2e").exists()
    assert not target.exists()


def test_bad_seed_exits_without_a_traceback(tmp_path: Path) -> None:
    target = tmp_path / "bad-seed"
    buf = io.StringIO()
    code = main(["--workdir", str(target), "--seed", "-1"], sink=buf)
    assert code == 1
    text = buf.getvalue()
    assert text.startswith("error:")
    assert "Traceback" not in text
    assert not target.exists()


def test_keep_leaves_the_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _prepare(root: Path, schedule: Schedule, *, generate: bool) -> tuple[str, ...]:
        del schedule, generate
        (root / "run.json").write_text(json.dumps({"run_id": "kept"}), encoding="utf-8")
        return ()

    def _score(root: Path, **kwargs: object) -> int:
        del root
        print(LABEL, file=kwargs["sink"])
        return 0

    monkeypatch.setattr(e2e, "_prepare", _prepare)
    monkeypatch.setattr(e2e, "score_demo", _score)
    target = tmp_path / "kept"
    target.mkdir()
    buf = io.StringIO()
    code = run_demo(target, keep=True, sink=buf)
    assert code == 0
    assert f"workdir={target.resolve()}" in buf.getvalue()
    assert (target / "run.json").is_file()


def test_default_temp_workdir_is_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, Path] = {}
    real = new_workdir

    def _new() -> Path:
        path = real()
        created["path"] = path
        return path

    def _prepare(root: Path, schedule: Schedule, *, generate: bool) -> tuple[str, ...]:
        del schedule, generate
        (root / "run.json").write_text(json.dumps({"run_id": "temp"}), encoding="utf-8")
        return ()

    def _score(root: Path, **kwargs: object) -> int:
        del root
        sink = kwargs["sink"]
        assert isinstance(sink, io.TextIOBase)
        print(LABEL, file=sink)
        return 0

    monkeypatch.setattr(e2e, "new_workdir", _new)
    monkeypatch.setattr(e2e, "_prepare", _prepare)
    monkeypatch.setattr(e2e, "score_demo", _score)
    buf = io.StringIO()
    code = run_demo(sink=buf)
    assert code == 0
    path = created["path"]
    assert not path.exists()
    assert path.name.startswith("trace-search-e2e-")
    assert _REPO not in path.parents
    assert "workdir=" not in buf.getvalue()


def test_cleanup_failure_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _prepare(root: Path, schedule: Schedule, *, generate: bool) -> tuple[str, ...]:
        del root, schedule, generate
        return ()

    def _score(root: Path, **kwargs: object) -> int:
        del root, kwargs
        return 0

    def _remove(path: Path) -> None:
        raise OSError(f"busy {path}")

    monkeypatch.setattr(e2e, "_prepare", _prepare)
    monkeypatch.setattr(e2e, "score_demo", _score)
    monkeypatch.setattr(e2e, "remove_workdir", _remove)
    buf = io.StringIO()
    code = run_demo(tmp_path, sink=buf)
    assert code == 1
    assert "OSError" in buf.getvalue()
    assert "Traceback" not in buf.getvalue()


def test_bad_run_json_does_not_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _prepare(root: Path, schedule: Schedule, *, generate: bool) -> tuple[str, ...]:
        del schedule, generate
        (root / "run.json").write_text("{}", encoding="utf-8")
        return ()

    def _score(root: Path, **kwargs: object) -> int:
        del root, kwargs
        return 0

    monkeypatch.setattr(e2e, "_prepare", _prepare)
    monkeypatch.setattr(e2e, "score_demo", _score)
    target = tmp_path / "bad-json"
    target.mkdir()
    buf = io.StringIO()
    code = run_demo(target, sink=buf)
    assert code == 1
    assert "Traceback" not in buf.getvalue()
    assert "run_id" in buf.getvalue()


def test_stdout_is_the_default_sink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _prepare(root: Path, schedule: Schedule, *, generate: bool) -> tuple[str, ...]:
        del schedule, generate
        (root / "run.json").write_text(json.dumps({"run_id": "out"}), encoding="utf-8")
        return ()

    def _score(root: Path, **kwargs: object) -> int:
        del root
        print("from-score", file=kwargs["sink"])
        return 0

    monkeypatch.setattr(e2e, "_prepare", _prepare)
    monkeypatch.setattr(e2e, "score_demo", _score)
    target = tmp_path / "stdout"
    target.mkdir()
    assert run_demo(target, keep=True) == 0
    captured = capsys.readouterr()
    assert "from-score" in captured.out
    assert f"workdir={target.resolve()}" in captured.out
    assert "Traceback" not in captured.err


def test_main_flag_forwarding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _fake(
        workdir: Path | None,
        *,
        schedule: Schedule,
        keep: bool,
        emit_json: bool,
        sink: object = None,
    ) -> int:
        seen["workdir"] = workdir
        seen["seed"] = schedule.seed
        seen["keep"] = keep
        seen["json"] = emit_json
        seen["sink"] = sink
        return 0

    monkeypatch.setattr("scripts.run_e2e_demo.run_demo", _fake)
    code = main(
        ["--workdir", str(tmp_path), "--seed", "4", "--keep", "--json"],
        sink=io.StringIO(),
    )
    assert code == 0
    assert seen["workdir"] == tmp_path
    assert seen["seed"] == 4
    assert seen["keep"] is True
    assert seen["json"] is True
    assert seen["sink"] is not None


def test_main_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _fake(
        workdir: Path | None,
        *,
        schedule: Schedule,
        keep: bool,
        emit_json: bool,
        sink: object = None,
    ) -> int:
        del sink
        seen["workdir"] = workdir
        seen["seed"] = schedule.seed
        seen["keep"] = keep
        seen["json"] = emit_json
        return 3

    monkeypatch.setattr("scripts.run_e2e_demo.run_demo", _fake)
    assert main([]) == 3
    assert seen == {"workdir": None, "seed": 0, "keep": False, "json": False}


def test_main_rejects_unknown_args() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--nope"])
    assert caught.value.code == 2


def test_thresholds_and_exit_code() -> None:
    assert SEGMENT_RECALL_MIN == 0.80
    assert SEGMENT_PRECISION_MIN == 0.90
    assert FALSE_MERGE_MAX == 0.05
    assert IOU_MIN == 0.5
    assert LABEL == "CODE_TEST: not evidence for any AC (AC-07 is DEFERRED)"
    passing = _report(0.80, 0.90, 0.05)
    assert metric_marks(0.80, 0.90, 0.05) == (True, True, True)
    text = render_text(passing)
    assert text.splitlines()[0] == LABEL
    assert "segment_recall=0.8000 (>= 0.80) PASS" in text
    assert "segment_precision=0.9000 (>= 0.90) PASS" in text
    assert "false_merge_rate=0.0500 (<= 0.05) PASS" in text
    assert "run time 1.25s" in text
    assert "skipped" not in text
    assert e2e._exit_code(passing) == 0
    low_recall = _report(0.7999, 0.90, 0.05)
    high_merge = _report(0.80, 0.90, 0.0501)
    low_precision = _report(0.80, 0.8999, 0.05)
    assert "segment_recall=0.7999 (>= 0.80) FAIL" in render_text(low_recall)
    assert "segment_precision=0.8999 (>= 0.90) FAIL" in render_text(low_precision)
    assert "false_merge_rate=0.0501 (<= 0.05) FAIL" in render_text(high_merge)
    assert e2e._exit_code(low_recall) == 1
    assert e2e._exit_code(low_precision) == 1
    assert e2e._exit_code(high_merge) == 1
    failed = json.loads(render_json(low_recall))
    assert failed["passed"] is False
    assert failed["evidence"] is False
    assert failed["results"]["segment_recall"] == "FAIL"
    assert metric_marks(0.7999, 0.90, 0.05) == (False, True, True)
    assert metric_marks(0.80, 0.8999, 0.05) == (True, False, True)
    assert metric_marks(0.80, 0.90, 0.0501) == (True, True, False)


def test_render_includes_figures_and_skips() -> None:
    figure = FigureReport("F1", 2, 3, "CAM-1", 4, (("CAM-1", 4, 0.0, 1.0),))
    report = _report(1.0, 1.0, 0.0, figures=(figure,), skipped=("CAM-9: source yielded no frames",))
    text = render_text(report)
    assert "F1: 2/3" in text
    assert "skipped 1: CAM-9: source yielded no frames" in text
    payload = json.loads(render_json(report))
    assert payload["evidence"] is False
    assert payload["notice"] == "not evidence"
    assert payload["figures"][0]["figure_id"] == "F1"
    assert payload["figures"][0]["segments"][0]["track_id"] == 4
    assert payload["skipped"] == ["CAM-9: source yielded no frames"]
    assert payload["passed"] is True


def test_box_iou_and_ownership() -> None:
    full = (0.0, 0.0, 10.0, 10.0)
    assert box_iou(full, full) == 1.0
    assert box_iou(full, (1.0, 0.0, 11.0, 10.0)) == pytest.approx(90.0 / 110.0)
    assert box_iou(full, (1.0, 20.0, 3.0, 22.0)) == 0.0
    assert box_iou(full, (20.0, 1.0, 22.0, 3.0)) == 0.0
    assert box_iou((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)) == 0.0
    edge = _observation("CAM-1", 1, 0, (0.0, 0.0, 2.0, 2.0))
    owned = track_ownership([edge], [TruthBox("F1", "CAM-1", 0, (0.0, 0.0, 2.0, 1.0))])
    assert owned[("CAM-1", 1)] == "F1"
    below = _observation("CAM-1", 1, 0, (0.0, 0.0, 2.0, 2.0))
    missed = track_ownership([below], [TruthBox("F1", "CAM-1", 0, (0.0, 0.0, 2.0, 0.9))])
    assert missed == {}
    tie = [
        _observation("CAM-1", 3, 0, full),
        _observation("CAM-1", 3, 1, (0.0, 20.0, 10.0, 30.0)),
    ]
    boxes = [
        TruthBox("B", "CAM-1", 0, full),
        TruthBox("A", "CAM-1", 1, (0.0, 20.0, 10.0, 30.0)),
    ]
    assert track_ownership(tie, boxes) == {}
    majority = [
        _observation("CAM-1", 4, 0, full),
        _observation("CAM-1", 4, 1, full),
        _observation("CAM-1", 4, 2, (0.0, 20.0, 10.0, 30.0)),
        _observation("CAM-1", 8, 0, full, det_class=DetClass.BACKPACK),
    ]
    votes = [
        TruthBox("F1", "CAM-1", 0, full),
        TruthBox("F1", "CAM-1", 1, full),
        TruthBox("F2", "CAM-1", 2, (0.0, 20.0, 10.0, 30.0)),
    ]
    assert track_ownership(majority, votes) == {("CAM-1", 4): "F1"}
    assert ("CAM-1", 8) not in track_ownership(majority, votes)
    weak = [
        _observation("CAM-1", 5, 0, full),
        _observation("CAM-1", 5, 1, (100.0, 100.0, 110.0, 110.0)),
        _observation("CAM-1", 5, 2, (120.0, 120.0, 130.0, 130.0)),
    ]
    assert track_ownership(weak, [TruthBox("F1", "CAM-1", 0, full)]) == {}
    split = [
        _observation("CAM-1", 6, 0, full),
        _observation("CAM-1", 6, 1, (100.0, 100.0, 110.0, 110.0)),
    ]
    assert track_ownership(split, [TruthBox("F1", "CAM-1", 0, full)]) == {}


def test_best_figure_and_majority_edges() -> None:
    box = (0.0, 0.0, 10.0, 10.0)
    assert e2e._best_figure(box, [("far", (100.0, 100.0, 110.0, 110.0))]) is None
    assert e2e._best_figure((0.0, 0.0, 2.0, 2.0), [("edge", (0.0, 0.0, 2.0, 1.0))]) == "edge"
    assert e2e._best_figure(box, [("first", box), ("second", box)]) == "first"
    assert e2e._best_figure(box, [("partial", (0.0, 0.0, 10.0, 6.0)), ("full", box)]) == "full"
    assert e2e._majority({}, 3) is None
    assert e2e._majority({"A": 1}, 0) is None
    assert e2e._majority({"A": 1, "B": 1}, 2) is None
    assert e2e._majority({"B": 1, "A": 2}, 3) == "A"


def test_choose_seed_breaks_ties_and_ignores_other_cameras() -> None:
    full = (0.0, 0.0, 10.0, 10.0)
    truth = [TruthBox("F1", "CAM-1", 0, full)]
    tied = [
        _observation("CAM-1", 5, 0, full),
        _observation("CAM-1", 2, 0, full),
    ]
    assert choose_seed(tied, "CAM-1", truth) == ("CAM-1", 2)
    higher = [
        _observation("CAM-1", 1, 0, (0.0, 0.0, 10.0, 6.0)),
        _observation("CAM-1", 4, 0, full),
    ]
    assert choose_seed(higher, "CAM-1", truth) == ("CAM-1", 4)
    assert choose_seed([_observation("CAM-2", 1, 0, full)], "CAM-1", truth) is None
    assert choose_seed([], "CAM-1", truth) is None
    assert e2e._better_seed(0.1, ("CAM-1", 3), -1.0, None) is True
    assert e2e._better_seed(0.9, ("CAM-1", 3), 0.2, ("CAM-1", 1)) is True
    assert e2e._better_seed(0.1, ("CAM-1", 3), 0.2, ("CAM-1", 1)) is False
    assert e2e._better_seed(0.2, ("CAM-1", 1), 0.2, ("CAM-1", 3)) is True
    assert e2e._better_seed(0.2, ("CAM-1", 4), 0.2, ("CAM-1", 3)) is False
    assert e2e._better_seed(0.1, ("CAM-1", 1), 0.2, ("CAM-1", 3)) is False


def test_peak_iou_stays_on_one_camera_and_frame() -> None:
    full = (0.0, 0.0, 10.0, 10.0)
    row = _observation("CAM-1", 1, 0, full)
    other_camera = TruthBox("F1", "CAM-2", 0, full)
    other_frame = TruthBox("F1", "CAM-1", 4, full)
    assert e2e._peak_iou([row], [other_camera, other_frame]) == 0.0
    assert e2e._peak_iou([row], [TruthBox("F1", "CAM-1", 0, full)]) == 1.0


def test_edited_window_keeps_each_branch() -> None:
    assert e2e._edited_window(1, 0.5, 8.0, True, True) == (0.5, 8.0)
    assert e2e._edited_window(0, 0.5, 8.0, False, False) == (0.0, e2e.OPEN_EDGE_MAX)
    assert e2e._edited_window(0, 0.5, 8.0, True, True) == (0.0, e2e.SHRUNK_EDGE_MAX)
    assert e2e._edited_window(0, 0.5, 8.0, False, True) == (0.0, e2e.SHRUNK_EDGE_MAX)


def test_skip_reasons_name_only_failed_cameras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _one(camera_id: str, *_args: object, **_kwargs: object) -> str | None:
        if camera_id == "CAM-OK":
            return None
        return "no planned sample frame was read"

    monkeypatch.setattr(e2e, "_describe_one", _one)
    reasons = describe_cameras(
        tmp_path,
        [("CAM-OK", tmp_path / "ok.mp4"), ("CAM-EMPTY", tmp_path / "empty.mp4")],
        "run-skip",
    )
    assert reasons == ("CAM-EMPTY: no planned sample frame was read",)


def test_value_error_is_skipped_and_other_errors_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = tmp_path / "CAM-1.mp4"
    clip.write_bytes(b"not a decoded clip")

    def _value(*_args: object, **_kwargs: object) -> None:
        raise ValueError("no planned sample frame was read")

    monkeypatch.setattr(e2e, "describe_tracks", _value)
    reasons = describe_cameras(tmp_path, [("CAM-1", clip)], "run-value")
    assert reasons == ("CAM-1: no planned sample frame was read",)


def test_fixture_check_runs_before_identify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    def _deny(*_args: object, **_kwargs: object) -> None:
        raise FixtureSeparationError("check-first")

    monkeypatch.setattr(e2e, "check_fixture_allowed", _deny)
    with pytest.raises(FixtureSeparationError, match="check-first"):
        require_code_test_clip(clip, manifest)


def test_route_from_seed_and_missing_route() -> None:
    track = _tracklet("CAM-1", 3)
    assert route_from_seed([track], [], None) is None
    assert route_from_seed([track], [], ("CAM-1", 9)) is None
    route = route_from_seed([track], [], ("CAM-1", 3))
    assert route is not None
    assert route.seed_track == 3
    assert len(route.segments) == 1
    figure = FigureRecord("F9", (TruthSegment("F9", "CAM-1", 0.0, 1.0),), ())
    report, built = e2e._one_figure(figure, [], [], [], {})
    assert built is None
    assert report.found == 0
    assert report.seed_camera is None
    assert report.segments == ()
    reports, routes, predicted, truth = e2e._figure_routes((figure,), [], [], [], {})
    assert routes == ()
    assert predicted == ()
    assert reports[0].found == 0
    assert truth == figure.visits


def test_delete_run_rows_keeps_other_runs(tmp_path: Path) -> None:
    root = tmp_path / "stores"
    root.mkdir()
    delete_run_rows(root, "missing")
    assert not (root / TRACK_DB_FILENAME).exists()
    store = TrackStore(root / TRACK_DB_FILENAME)
    attrs = TrackAttributeStore(root / ATTRIBUTE_DB_FILENAME)
    vectors = SqliteVectorStore(root / EMBEDDING_DB_FILENAME, EMBEDDER_DIM)
    row = _observation("CAM-1", 1, 0, (0.0, 0.0, 10.0, 10.0))
    store.write("keep-me", "fixture", DERIVED_FROM, [row])
    store.write("drop-me", "fixture", DERIVED_FROM, [row])
    attrs.write("CAM-1", 1, "keep-me", DERIVED_FROM, "red", "blue", False, 1.0, 1.0, 1)
    attrs.write("CAM-1", 1, "drop-me", DERIVED_FROM, "red", "blue", False, 1.0, 1.0, 1)
    unit = tuple(1.0 if index == 0 else 0.0 for index in range(EMBEDDER_DIM))
    vectors.add("keep-me", DERIVED_FROM, [AppearanceEmbedding("CAM-1", 1, 0, unit, "n", True)])
    vectors.add("drop-me", DERIVED_FROM, [AppearanceEmbedding("CAM-1", 1, 1, unit, "n", True)])
    delete_run_rows(root, "drop-me")
    assert store.count(run_id="drop-me") == 0
    assert store.count(run_id="keep-me") == 1
    assert attrs.count(run_id="drop-me") == 0
    assert attrs.count(run_id="keep-me") == 1
    assert vectors.count(DERIVED_FROM, "drop-me") == 0
    assert vectors.count(DERIVED_FROM, "keep-me") == 1
    only_tracks = tmp_path / "tracks-only"
    only_tracks.mkdir()
    TrackStore(only_tracks / TRACK_DB_FILENAME).write(
        "drop-me",
        "fixture",
        DERIVED_FROM,
        [row],
    )
    delete_run_rows(only_tracks, "drop-me")
    assert TrackStore(only_tracks / TRACK_DB_FILENAME).count(run_id="drop-me") == 0


def test_remove_and_reject_workdirs(tmp_path: Path) -> None:
    missing = tmp_path / "gone"
    remove_workdir(missing)
    assert not missing.exists()
    folder = tmp_path / "dir"
    folder.mkdir()
    (folder / "note.txt").write_text("x", encoding="utf-8")
    remove_workdir(folder)
    assert not folder.exists()
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        remove_workdir(file_path)
    nested = _REPO / "e2e-remove-probe"
    nested.mkdir()
    try:
        with pytest.raises(ValueError, match="inside the repository"):
            remove_workdir(nested)
        assert nested.is_dir()
    finally:
        if nested.exists():
            nested.rmdir()
    created = new_workdir()
    try:
        assert created.is_dir()
        assert _REPO not in created.parents
        assert created.name.startswith("trace-search-e2e-")
    finally:
        remove_workdir(created)
    with pytest.raises(ValueError, match="inside the repository"):
        assert_outside_repo(_REPO)
    assert assert_outside_repo(tmp_path) == tmp_path.resolve()


def test_clip_file_rejects_a_path(tmp_path: Path) -> None:
    assert clip_file(tmp_path, "CAM-1") == tmp_path / "CAM-1.mp4"
    with pytest.raises(ValueError, match="camera id"):
        clip_file(tmp_path, "")
    with pytest.raises(ValueError, match="camera id"):
        clip_file(tmp_path, ".")
    with pytest.raises(ValueError, match="camera id"):
        clip_file(tmp_path, "CAM-1/clip")
    with pytest.raises(ValueError, match="camera id"):
        clip_file(tmp_path, "..")


def test_json_helpers_reject_bad_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        e2e._as_object([], "item")
    with pytest.raises(ValueError, match="keys must be strings"):
        e2e._as_object({1: "a"}, "item")
    with pytest.raises(ValueError, match="must be a list"):
        e2e._as_list("no", "items")
    with pytest.raises(ValueError, match="non-empty string"):
        e2e._as_text("", "figure_id")
    with pytest.raises(ValueError, match="non-empty string"):
        e2e._as_text(1, "figure_id")
    with pytest.raises(ValueError, match="expected a number"):
        e2e._as_number(True)
    with pytest.raises(ValueError, match="expected a number"):
        e2e._as_number("1")
    assert e2e._as_number(2) == 2.0
    with pytest.raises(ValueError, match="frame_index"):
        e2e._as_frame(True)
    with pytest.raises(ValueError, match="frame_index"):
        e2e._as_frame(1.5)
    path = tmp_path / "truth.json"
    path.write_text(json.dumps({"figures": "no"}), encoding="utf-8")
    with pytest.raises(ValueError, match="figures"):
        e2e._load_figures(path)
    path.write_text(json.dumps({"figures": [{"figure_id": ""}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="figure_id"):
        e2e._load_figures(path)
    assert e2e._optional_run_id(tmp_path) is None
    (tmp_path / "run.json").write_text(json.dumps({"run_id": "abc"}), encoding="utf-8")
    assert e2e._read_run_id(tmp_path) == "abc"


def test_window_edits_cover_each_branch(demo_run: _DemoRun, tmp_path: Path) -> None:
    root = tmp_path / "windows"
    _copy_tree(demo_run.root, root)
    both = io.StringIO()
    code = score_demo(root, enforce_time_window=False, shrink_first_window=True, sink=both)
    assert both.getvalue().splitlines()[0] == LABEL
    assert code in (0, 1)
    closed = e2e._edited_window(1, 0.5, 8.0, True, True)
    opened = e2e._edited_window(0, 0.5, 8.0, False, False)
    shrunk = e2e._edited_window(0, 0.5, 8.0, True, True)
    assert closed == (0.5, 8.0)
    assert opened == (0.0, e2e.OPEN_EDGE_MAX)
    assert shrunk == (0.0, e2e.SHRUNK_EDGE_MAX)


def test_sources_do_not_call_banned_loaders() -> None:
    banned = (
        "load_reid_samples",
        "LoaderContext.TRAINING",
        "LoaderContext.EVALUATION",
        "LoaderContext.STUDY",
        "urllib",
        "requests",
        "http.client",
        "socket",
    )
    for relative in ("src/trace_search/pipeline/e2e.py", "scripts/run_e2e_demo.py"):
        text = (_REPO / relative).read_text(encoding="utf-8")
        for item in banned:
            assert item not in text
    script = (_REPO / "scripts/run_e2e_demo.py").read_text(encoding="utf-8")
    assert "name: str" not in script
    assert "name: str" not in (_REPO / "src/trace_search/pipeline/e2e.py").read_text(
        encoding="utf-8"
    )
