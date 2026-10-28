"""End-to-end synthetic demo: generate, ingest, link, and score routes.

One call runs T-E-9 clips through the built pipeline and compares each
figure's route to that clip's truth. The numbers are not evidence for any
acceptance criterion. AC-07 is DEFERRED.

Thresholds are fixed (segment recall >= 0.80, segment precision >= 0.90,
false-merge <= 0.05). They are not fitted to a run.

A camera whose appearance pass raises ``ValueError`` is skipped. The
message is counted and printed. That camera's existing rows are not
deleted. The traceback is not printed.

SQLite files live in the workdir at the retention registry names. The
workdir must resolve outside this repository. It must not be the
repository root, an ancestor of the repository, or the home directory.
Unless the caller keeps it, the run deletes its own ``run_id`` and
removes only paths that run created. A directory that already existed
is left in place, including the files that were already there.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from trace_search.attributes.attributes import AttributeExtractor, Box
from trace_search.attributes.store import TrackAttributeStore
from trace_search.detection.scripted_figures import scripted_figures_detector
from trace_search.detection.types import DetClass
from trace_search.fixtures.guards import (
    FixtureSeparationError,
    FixtureUseTag,
    LoaderContext,
    check_fixture_allowed,
    identify_fixture_by_content,
    load_manifest_fixtures,
)
from trace_search.fixtures.route_generator import generate_route_fixture
from trace_search.ingest.camera_registry import CameraConfig, CameraRegistry
from trace_search.ingest.file_ingest import FileVideoSource
from trace_search.matching.appearance import describe_tracks
from trace_search.matching.linker import Link, link_tracklets
from trace_search.matching.route import Route, RouteSegment, build_route
from trace_search.matching.route_metrics import (
    TruthSegment,
    false_merge_rate,
    matched_pairs,
    segment_precision,
    segment_recall,
)
from trace_search.matching.topology import CameraTopology, TransitEdge
from trace_search.matching.tracklets import Tracklet, load_tracklets
from trace_search.pipeline.camera_pipeline import run_camera
from trace_search.reid.placeholder import UntrainedPlaceholderEmbedder
from trace_search.reid.sqlite_vector_store import SqliteVectorStore
from trace_search.reid.types import AppearanceEmbedder
from trace_search.retention.registry import (
    ATTRIBUTE_DB_FILENAME,
    EMBEDDING_DB_FILENAME,
    TRACK_DB_FILENAME,
)
from trace_search.tracking.bytetrack import ByteTracker
from trace_search.tracking.store import TrackStore
from trace_search.tracking.types import Tracker, TrackObservation

__all__ = [
    "ATTRIBUTE_DB_FILENAME",
    "EMBEDDER_DIM",
    "EMBEDDING_DB_FILENAME",
    "FALSE_MERGE_MAX",
    "IOU_MIN",
    "LABEL",
    "OPEN_EDGE_MAX",
    "SEGMENT_PRECISION_MIN",
    "SEGMENT_RECALL_MIN",
    "SHRUNK_EDGE_MAX",
    "TRACK_DB_FILENAME",
    "DemoReport",
    "FigureReport",
    "Schedule",
    "TruthBox",
    "assert_outside_repo",
    "box_iou",
    "choose_seed",
    "clip_file",
    "delete_run_rows",
    "describe_cameras",
    "metric_marks",
    "new_workdir",
    "remove_workdir",
    "render_json",
    "render_text",
    "require_code_test_clip",
    "route_from_seed",
    "run_demo",
    "score_demo",
    "track_ownership",
]

# D-T6-2 (a). Fixed before the first run. Do not move these to pass a clip.
SEGMENT_RECALL_MIN = 0.80
SEGMENT_PRECISION_MIN = 0.90
FALSE_MERGE_MAX = 0.05

# Majority ownership counts a box only at this IoU or higher.
IOU_MIN = 0.5

# Planted topology edits used by tests. These are not metric thresholds.
SHRUNK_EDGE_MAX = 0.05
OPEN_EDGE_MAX = 1_000_000.0

LABEL = "CODE_TEST: not evidence for any AC (AC-07 is DEFERRED)"
DERIVED_FROM = "CODE_TEST"
EMBEDDER_DIM = 256
EMBEDDER_SEED = 0
WORKDIR_PREFIX = "trace-search-e2e-"
RUN_FILENAME = "run.json"
ROUTES_FILENAME = "routes.json"
CAMERAS_FILENAME = "cameras.json"
TOPOLOGY_FILENAME = "topology.json"
TRUTH_FILENAME = "truth.json"
MANIFEST_FILENAME = "manifest.json"

TrackKey = tuple[str, int]


@dataclass(frozen=True)
class Schedule:
    """Generator settings for one demo run.

    The default is 6 cameras, 4 figures, 320 by 240, 10 fps, and 40 seconds.
    That schedule is accepted by T-E-9. Its clocks run from -2 s to +203 s,
    and two figures share each corridor.
    """

    seed: int = 0
    n_cameras: int = 6
    n_figures: int = 4
    fps: int = 10
    width: int = 320
    height: int = 240
    seconds: int = 40


@dataclass(frozen=True)
class TruthBox:
    """One truth rectangle for one figure on one frame."""

    figure_id: str
    camera_id: str
    frame_index: int
    box: Box


@dataclass(frozen=True)
class FigureRecord:
    """Truth for one generated figure. ``figure_id`` is not a person id."""

    figure_id: str
    visits: tuple[TruthSegment, ...]
    boxes: tuple[TruthBox, ...]


@dataclass(frozen=True)
class FigureReport:
    """How many truth visits one figure got back, and which track seeded it."""

    figure_id: str
    found: int
    truth_count: int
    seed_camera: str | None
    seed_track: int | None
    segments: tuple[tuple[str, int, float, float], ...]


@dataclass(frozen=True)
class DemoReport:
    """Metrics for one scored run. ``evidence`` is never true."""

    run_id: str
    recall: float
    precision: float
    false_merge: float
    figures: tuple[FigureReport, ...]
    skipped: tuple[str, ...]
    elapsed_seconds: float


def assert_outside_repo(path: Path) -> Path:
    """Resolve ``path`` and refuse the repo, its ancestors, and home.

    ``~`` is expanded before resolving symlinks. The repository root, any
    descendant, any ancestor, and the home directory itself are refused.
    A directory under home that is none of those is allowed.
    """
    resolved = Path(path).expanduser().resolve()
    root = _repo_root()
    _refuse_repo_tree(resolved, root)
    _refuse_home(resolved)
    return resolved


def _refuse_repo_tree(resolved: Path, root: Path) -> None:
    if resolved == root or root in resolved.parents:
        raise ValueError(f"workdir is inside the repository: {resolved}")
    if resolved in root.parents:
        raise ValueError(f"workdir is an ancestor of the repository: {resolved}")


def _refuse_home(resolved: Path) -> None:
    if resolved == Path.home().resolve():
        raise ValueError(f"workdir is the home directory: {resolved}")


def new_workdir() -> Path:
    """Create a temp directory outside the repository."""
    return assert_outside_repo(Path(tempfile.mkdtemp(prefix=WORKDIR_PREFIX)))


def clip_file(root: Path, camera_id: str) -> Path:
    """Return the mp4 path for one camera. Refuse a camera id that is a path."""
    if camera_id in {"", ".", ".."} or camera_id != Path(camera_id).name:
        raise ValueError(f"camera id cannot name a clip file, got {camera_id!r}")
    return root / f"{camera_id}.mp4"


def require_code_test_clip(path: Path, manifest: Path) -> str:
    """Refuse the clip unless it is a ``CODE_TEST`` manifest row.

    ``check_fixture_allowed`` runs first. A missing row or any other use
    tag still fails closed, before ``FileVideoSource`` is constructed.
    """
    check_fixture_allowed(path, LoaderContext.UNIT_TEST, manifest)
    identified = identify_fixture_by_content(path, load_manifest_fixtures(manifest))
    if identified is None:
        raise FixtureSeparationError(
            f"FAIL-CLOSED: no manifest row for {path} (refused before the clip is opened)"
        )
    use_tag, fixture_id = identified
    if use_tag is not FixtureUseTag.CODE_TEST:
        raise FixtureSeparationError(
            f"Fixture {path} identified as {fixture_id} with use tag {use_tag.value} "
            "is FORBIDDEN in unit_test context (identified by SHA-256 or pHash)."
        )
    return fixture_id


def box_iou(left: Box, right: Box) -> float:
    """Intersection over union for two ``(x1, y1, x2, y2)`` boxes."""
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    overlap_w = min(left_x2, right_x2) - max(left_x1, right_x1)
    overlap_h = min(left_y2, right_y2) - max(left_y1, right_y1)
    if overlap_w <= 0.0 or overlap_h <= 0.0:
        return 0.0
    overlap = overlap_w * overlap_h
    left_area = (left_x2 - left_x1) * (left_y2 - left_y1)
    right_area = (right_x2 - right_x1) * (right_y2 - right_y1)
    return overlap / (left_area + right_area - overlap)


def track_ownership(
    observations: Sequence[TrackObservation],
    truth_boxes: Sequence[TruthBox],
) -> dict[TrackKey, str]:
    """Own each person track by the figure with a strict majority of boxes.

    A box votes only when its best truth IoU on that camera and frame is
    at least ``IOU_MIN``. An exact IoU tie votes for the lower figure id,
    so the owner does not depend on truth-file order. The winner needs
    more than half of the track's boxes. A tie of votes, or a track with
    no vote, is left out of the map.
    """
    by_frame = _truth_by_frame(truth_boxes)
    owned: dict[TrackKey, str] = {}
    for key, rows in _person_rows(observations).items():
        winner = _majority(_votes(rows, by_frame), len(rows))
        if winner is not None:
            owned[key] = winner
    return owned


def choose_seed(
    observations: Sequence[TrackObservation],
    camera_id: str,
    truth_boxes: Sequence[TruthBox],
) -> TrackKey | None:
    """Pick the person track with the highest IoU to ``truth_boxes``.

    Only ``camera_id`` is considered. The peak must be at least ``IOU_MIN``.
    A disjoint track is not a seed. Equal IoU keeps the smaller track id.
    """
    best_key: TrackKey | None = None
    best_score = -1.0
    for key, rows in _person_rows(observations).items():
        if key[0] != camera_id:
            continue
        score = _peak_iou(rows, truth_boxes)
        if score < IOU_MIN:
            continue
        if _better_seed(score, key, best_score, best_key):
            best_key = key
            best_score = score
    return best_key


def metric_marks(recall: float, precision: float, false_merge: float) -> tuple[bool, bool, bool]:
    """Compare one run to the fixed bars. Order is recall, precision, false-merge."""
    recall_ok = recall >= SEGMENT_RECALL_MIN
    precision_ok = precision >= SEGMENT_PRECISION_MIN
    merge_ok = false_merge <= FALSE_MERGE_MAX
    return (recall_ok, precision_ok, merge_ok)


def route_from_seed(
    tracklets: Sequence[Tracklet],
    links: Sequence[Link],
    key: TrackKey | None,
) -> Route | None:
    """Build the route for ``key``, or return None when the seed is missing."""
    if key is None:
        return None
    for item in tracklets:
        if (item.camera_id, item.track_id) == key:
            return build_route(item, tracklets, links)
    return None


def describe_cameras(
    data_root: Path,
    clips: Sequence[tuple[str, Path]],
    run_id: str,
) -> tuple[str, ...]:
    """Store appearance for each camera. Return one skip reason per failure.

    ``describe_tracks`` raises ``ValueError`` before it deletes rows when
    the camera has no sampled person, the source yields no frames, or the
    read misses every planned sample. That error is caught here. The
    camera's existing rows stay. Any other error propagates. The demo
    prints the reasons after the label line.
    """
    store = TrackStore(data_root / TRACK_DB_FILENAME)
    attributes = TrackAttributeStore(data_root / ATTRIBUTE_DB_FILENAME)
    vectors = SqliteVectorStore(data_root / EMBEDDING_DB_FILENAME, EMBEDDER_DIM)
    embedder = UntrainedPlaceholderEmbedder(dim=EMBEDDER_DIM, seed=EMBEDDER_SEED)
    extractor = AttributeExtractor()
    reasons: list[str] = []
    for camera_id, clip in clips:
        reason = _describe_one(
            camera_id,
            clip,
            store,
            attributes,
            vectors,
            embedder,
            extractor,
            run_id,
        )
        if reason is not None:
            reasons.append(f"{camera_id}: {reason}")
    return tuple(reasons)


def delete_run_rows(root: Path, run_id: str) -> None:
    """Delete rows for ``run_id`` in each store that already exists.

    A missing file is left missing. Other run ids stay.
    """
    _delete_track_rows(root, run_id)
    _delete_attribute_rows(root, run_id)
    _delete_embedding_rows(root, run_id)


def remove_workdir(path: Path) -> None:
    """Remove a directory that is safe to delete.

    The repository, its ancestors, its descendants, and the home directory
    are refused. ``run_demo`` calls this only for a directory that run
    created. A pre-existing workdir is trimmed by ``_remove_created``.
    """
    resolved = assert_outside_repo(path)
    if not resolved.exists():
        return
    if not resolved.is_dir():
        raise ValueError(f"workdir is not a directory: {resolved}")
    shutil.rmtree(resolved)


def render_text(report: DemoReport) -> str:
    """Text report. The label is the first line. Metrics name the fixed bars."""
    recall_ok, precision_ok, merge_ok = metric_marks(
        report.recall, report.precision, report.false_merge
    )
    lines = [LABEL]
    for figure in report.figures:
        lines.append(f"{figure.figure_id}: {figure.found}/{figure.truth_count}")
    lines.append(_metric_line("segment_recall", report.recall, ">=", SEGMENT_RECALL_MIN, recall_ok))
    lines.append(
        _metric_line(
            "segment_precision", report.precision, ">=", SEGMENT_PRECISION_MIN, precision_ok
        )
    )
    lines.append(
        _metric_line("false_merge_rate", report.false_merge, "<=", FALSE_MERGE_MAX, merge_ok)
    )
    lines.append(f"run time {report.elapsed_seconds:.2f}s")
    lines.append(f"run_id={report.run_id}")
    if report.skipped:
        lines.append(f"skipped {len(report.skipped)}: {'; '.join(report.skipped)}")
    return "\n".join(lines) + "\n"


def render_json(report: DemoReport) -> str:
    """JSON report. ``evidence`` is false. The label says this is not evidence."""
    recall_ok, precision_ok, merge_ok = metric_marks(
        report.recall, report.precision, report.false_merge
    )
    payload = {
        "evidence": False,
        "false_merge_rate": report.false_merge,
        "figures": [_figure_payload(figure) for figure in report.figures],
        "label": LABEL,
        "notice": "not evidence",
        "passed": bool(recall_ok and precision_ok and merge_ok),
        "results": {
            "false_merge_rate": "PASS" if merge_ok else "FAIL",
            "segment_precision": "PASS" if precision_ok else "FAIL",
            "segment_recall": "PASS" if recall_ok else "FAIL",
        },
        "run_id": report.run_id,
        "run_time_seconds": report.elapsed_seconds,
        "segment_precision": report.precision,
        "segment_recall": report.recall,
        "skipped": list(report.skipped),
        "thresholds": {
            "false_merge_rate": FALSE_MERGE_MAX,
            "segment_precision": SEGMENT_PRECISION_MIN,
            "segment_recall": SEGMENT_RECALL_MIN,
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def score_demo(
    root: Path,
    *,
    enforce_time_window: bool = True,
    shrink_first_window: bool = False,
    emit_json: bool = False,
    elapsed_seconds: float = 0.0,
    skipped: Sequence[str] = (),
    sink: TextIO | None = None,
) -> int:
    """Link, build one route per truth figure, print metrics, and return the exit code."""
    output = _sink(sink)
    report = _report(root, enforce_time_window, shrink_first_window, elapsed_seconds, skipped)
    _write_routes(root, report)
    if emit_json:
        print(render_json(report), file=output, end="")
    else:
        print(render_text(report), file=output, end="")
    return _exit_code(report)


def run_demo(
    workdir: Path | None = None,
    *,
    schedule: Schedule | None = None,
    keep: bool = False,
    emit_json: bool = False,
    generate: bool = True,
    enforce_time_window: bool = True,
    shrink_first_window: bool = False,
    sink: TextIO | None = None,
) -> int:
    """Generate (unless told not to), run the pipeline, score, and clean up.

    ``workdir`` None creates a temp directory. The repository, an ancestor,
    a descendant, and the home directory return 1 before any file is
    written. ``keep`` prints the workdir and leaves it in place. Otherwise
    this deletes the run's rows and removes only paths this run created.
    """
    output = _sink(sink)
    try:
        owned = _acquire_workdir(workdir)
    except ValueError as exc:
        _report_error(exc, output)
        return 1
    chosen = Schedule() if schedule is None else schedule
    started = time.perf_counter()
    try:
        skipped = _prepare(owned.root, chosen, generate=generate)
        code = score_demo(
            owned.root,
            enforce_time_window=enforce_time_window,
            shrink_first_window=shrink_first_window,
            emit_json=emit_json,
            elapsed_seconds=time.perf_counter() - started,
            skipped=skipped,
            sink=output,
        )
    except Exception as exc:
        _report_error(exc, output)
        code = 1
    if not _finish(owned, keep, output):
        return 1
    return code


@dataclass(frozen=True)
class _Workdir:
    """A workdir plus the paths that were already there."""

    root: Path
    created: bool
    prior: frozenset[Path]


def _acquire_workdir(workdir: Path | None) -> _Workdir:
    if workdir is None:
        return _Workdir(new_workdir(), True, frozenset())
    root = assert_outside_repo(workdir)
    if root.exists():
        return _existing_workdir(root)
    _make_workdir(root)
    return _Workdir(root, True, frozenset())


def _existing_workdir(root: Path) -> _Workdir:
    if not root.is_dir():
        raise ValueError(f"workdir is not a directory: {root}")
    return _Workdir(root, False, _snapshot(root))


def _make_workdir(root: Path) -> None:
    try:
        root.mkdir()
    except OSError as exc:
        raise ValueError(f"workdir cannot be created: {root}") from exc


def _snapshot(root: Path) -> frozenset[Path]:
    found = {Path(".")}
    for path in root.rglob("*"):
        found.add(path.relative_to(root))
    return frozenset(found)


def _finish(owned: _Workdir, keep: bool, sink: TextIO) -> bool:
    """Delete this run's paths unless ``keep`` is set. False means cleanup raised."""
    try:
        if keep:
            print(f"workdir={owned.root}", file=sink)
            return True
        _delete_owned_rows(owned.root)
        _remove_created(owned)
    except Exception as exc:
        _report_error(exc, sink)
        return False
    return True


def _delete_owned_rows(root: Path) -> None:
    run_id = _optional_run_id(root)
    if run_id is None:
        return
    delete_run_rows(root, run_id)


def _remove_created(owned: _Workdir) -> None:
    if owned.created:
        remove_workdir(owned.root)
        return
    _remove_new_paths(owned.root, owned.prior)


def _remove_new_paths(root: Path, prior: frozenset[Path]) -> None:
    created = [path for path in root.rglob("*") if path.relative_to(root) not in prior]
    created.sort(key=lambda path: len(path.parts), reverse=True)
    for path in created:
        _remove_one(path)


def _remove_one(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink()
        return
    path.rmdir()


def _prepare(root: Path, schedule: Schedule, *, generate: bool) -> tuple[str, ...]:
    if generate:
        _generate(root, schedule)
    manifest = root / MANIFEST_FILENAME
    registry = CameraRegistry.load(root / CAMERAS_FILENAME)
    clips = _identified_clips(root, manifest, registry)
    run_id = str(uuid.uuid4())
    frames = _ingest(root, clips, run_id)
    pairs = [(camera.camera_id, path) for camera, _fixture_id, path in clips]
    reasons = describe_cameras(root, pairs, run_id)
    _write_run(root, run_id, schedule, frames)
    return reasons


def _identified_clips(
    root: Path,
    manifest: Path,
    registry: CameraRegistry,
) -> tuple[tuple[CameraConfig, str, Path], ...]:
    """Identify every clip before a store or a video source is opened."""
    found: list[tuple[CameraConfig, str, Path]] = []
    for camera in registry.list_cameras():
        path = clip_file(root, camera.camera_id)
        fixture_id = require_code_test_clip(path, manifest)
        found.append((camera, fixture_id, path))
    return tuple(found)


def _generate(root: Path, schedule: Schedule) -> None:
    generate_route_fixture(
        root,
        seed=schedule.seed,
        n_cameras=schedule.n_cameras,
        n_figures=schedule.n_figures,
        fps=schedule.fps,
        size=(schedule.width, schedule.height),
        seconds=schedule.seconds,
    )


def _ingest(
    root: Path,
    clips: Sequence[tuple[CameraConfig, str, Path]],
    run_id: str,
) -> dict[str, int]:
    store = TrackStore(root / TRACK_DB_FILENAME)
    detector = scripted_figures_detector()
    frames: dict[str, int] = {}
    for camera, fixture_id, path in clips:
        source = FileVideoSource(camera.camera_id, path, start_timestamp=0.0)
        stats = run_camera(
            source,
            detector,
            _new_tracker,
            store,
            run_id,
            fixture_id,
            DERIVED_FROM,
            clock_offset_seconds=camera.clock_offset_seconds,
        )
        frames[camera.camera_id] = stats.frames
    return frames


def _new_tracker() -> Tracker:
    return ByteTracker()


def _describe_one(
    camera_id: str,
    clip: Path,
    store: TrackStore,
    attributes: TrackAttributeStore,
    vectors: SqliteVectorStore,
    embedder: UntrainedPlaceholderEmbedder,
    extractor: AttributeExtractor,
    run_id: str,
) -> str | None:
    source = FileVideoSource(camera_id, clip, start_timestamp=0.0)
    people = load_tracklets(store, run_id, camera_ids=(camera_id,), det_class=DetClass.PERSON)
    bags = _bag_tracklets(store, run_id, camera_id)
    checked: AppearanceEmbedder = embedder
    try:
        describe_tracks(
            source,
            people,
            bags,
            store,
            run_id,
            extractor,
            checked,
            attributes,
            vectors,
            DERIVED_FROM,
        )
    except ValueError as exc:
        return str(exc)
    return None


def _bag_tracklets(store: TrackStore, run_id: str, camera_id: str) -> tuple[Tracklet, ...]:
    found: list[Tracklet] = []
    for det_class in (DetClass.BACKPACK, DetClass.HANDBAG):
        found.extend(load_tracklets(store, run_id, camera_ids=(camera_id,), det_class=det_class))
    return tuple(found)


def _report(
    root: Path,
    enforce_time_window: bool,
    shrink_first_window: bool,
    elapsed_seconds: float,
    skipped: Sequence[str],
) -> DemoReport:
    run_id = _read_run_id(root)
    figures = _load_figures(root / TRUTH_FILENAME)
    observations = _read_observations(root, run_id)
    topology = _topology(root / TOPOLOGY_FILENAME, enforce_time_window, shrink_first_window)
    tracklets = load_tracklets(TrackStore(root / TRACK_DB_FILENAME), run_id)
    links = _links(root, run_id, tracklets, topology)
    ownership = track_ownership(observations, _all_boxes(figures))
    figure_reports, routes, predicted, truth = _figure_routes(
        figures, observations, tracklets, links, ownership
    )
    recall, precision, merges = _metric_values(predicted, truth, routes, ownership)
    return DemoReport(
        run_id, recall, precision, merges, figure_reports, tuple(skipped), elapsed_seconds
    )


def _figure_routes(
    figures: Sequence[FigureRecord],
    observations: Sequence[TrackObservation],
    tracklets: Sequence[Tracklet],
    links: Sequence[Link],
    ownership: Mapping[TrackKey, str],
) -> tuple[
    tuple[FigureReport, ...], tuple[Route, ...], tuple[RouteSegment, ...], tuple[TruthSegment, ...]
]:
    reports: list[FigureReport] = []
    routes: list[Route] = []
    predicted: list[RouteSegment] = []
    truth: list[TruthSegment] = []
    for figure in figures:
        report, route = _one_figure(figure, observations, tracklets, links, ownership)
        reports.append(report)
        truth.extend(figure.visits)
        if route is None:
            continue
        routes.append(route)
        predicted.extend(route.segments)
    return tuple(reports), tuple(routes), tuple(predicted), tuple(truth)


def _one_figure(
    figure: FigureRecord,
    observations: Sequence[TrackObservation],
    tracklets: Sequence[Tracklet],
    links: Sequence[Link],
    ownership: Mapping[TrackKey, str],
) -> tuple[FigureReport, Route | None]:
    first = figure.visits[0]
    boxes = tuple(box for box in figure.boxes if box.camera_id == first.camera_id)
    key = choose_seed(observations, first.camera_id, boxes)
    route = route_from_seed(tracklets, links, key)
    found = _found_for(figure.figure_id, route, figure.visits, ownership)
    segments = _segment_tuples(route)
    camera_id, track_id = _seed_pair(key)
    report = FigureReport(
        figure.figure_id,
        found,
        len(figure.visits),
        camera_id,
        track_id,
        segments,
    )
    return report, route


def _found_for(
    figure_id: str,
    route: Route | None,
    visits: Sequence[TruthSegment],
    ownership: Mapping[TrackKey, str],
) -> int:
    if route is None:
        return 0
    pairs = matched_pairs(route.segments, visits, ownership)
    return len(pairs)


def _seed_pair(key: TrackKey | None) -> tuple[str | None, int | None]:
    if key is None:
        return None, None
    return key[0], key[1]


def _segment_tuples(route: Route | None) -> tuple[tuple[str, int, float, float], ...]:
    if route is None:
        return ()
    return tuple(
        (segment.camera_id, segment.track_id, segment.t_entry, segment.t_exit)
        for segment in route.segments
    )


def _metric_values(
    predicted: Sequence[RouteSegment],
    truth: Sequence[TruthSegment],
    routes: Sequence[Route],
    ownership: Mapping[TrackKey, str],
) -> tuple[float, float, float]:
    return (
        segment_recall(predicted, truth, ownership),
        segment_precision(predicted, truth, ownership),
        false_merge_rate(routes, ownership),
    )


def _links(
    root: Path,
    run_id: str,
    tracklets: Sequence[Tracklet],
    topology: CameraTopology,
) -> list[Link]:
    vectors = SqliteVectorStore(root / EMBEDDING_DB_FILENAME, EMBEDDER_DIM)
    attributes = TrackAttributeStore(root / ATTRIBUTE_DB_FILENAME)
    return list(
        link_tracklets(
            tracklets,
            topology,
            vectors,
            attributes,
            run_id=run_id,
            derived_from=DERIVED_FROM,
        )
    )


def _topology(path: Path, enforce_time_window: bool, shrink_first_window: bool) -> CameraTopology:
    topology = CameraTopology.from_dict(_load_json(path))
    if enforce_time_window and not shrink_first_window:
        return topology
    return _retimed(topology, enforce_time_window, shrink_first_window)


def _retimed(
    topology: CameraTopology,
    enforce_time_window: bool,
    shrink_first_window: bool,
) -> CameraTopology:
    raw = topology.to_dict()
    registry = CameraRegistry()
    for item in _as_list(raw.get("cameras"), "cameras"):
        registry.register(CameraConfig.from_dict(_as_object(item, "camera")))
    edges: list[TransitEdge] = []
    for index, item in enumerate(_as_list(raw.get("edges"), "edges")):
        edge = _as_object(item, "edge")
        low, high = _edited_window(
            index,
            _as_number(edge.get("min_seconds")),
            _as_number(edge.get("max_seconds")),
            enforce_time_window,
            shrink_first_window,
        )
        edges.append(
            TransitEdge(
                _as_text(edge.get("from_camera"), "from_camera"),
                _as_text(edge.get("to_camera"), "to_camera"),
                low,
                high,
            )
        )
    return CameraTopology(registry, edges)


def _edited_window(
    index: int,
    low: float,
    high: float,
    enforce_time_window: bool,
    shrink_first_window: bool,
) -> tuple[float, float]:
    if shrink_first_window and index == 0:
        return (0.0, SHRUNK_EDGE_MAX)
    if not enforce_time_window:
        return (0.0, OPEN_EDGE_MAX)
    return (low, high)


def _read_observations(root: Path, run_id: str) -> list[TrackObservation]:
    registry = CameraRegistry.load(root / CAMERAS_FILENAME)
    store = TrackStore(root / TRACK_DB_FILENAME)
    rows: list[TrackObservation] = []
    for camera in registry.list_cameras():
        rows.extend(store.read(camera.camera_id, run_id))
    return rows


def _all_boxes(figures: Sequence[FigureRecord]) -> tuple[TruthBox, ...]:
    found: list[TruthBox] = []
    for figure in figures:
        found.extend(figure.boxes)
    return tuple(found)


def _load_figures(path: Path) -> tuple[FigureRecord, ...]:
    payload = _as_object(_load_json(path), path.name)
    return tuple(_figure(item) for item in _as_list(payload.get("figures"), "figures"))


def _figure(value: object) -> FigureRecord:
    raw = _as_object(value, "figure")
    figure_id = _as_text(raw.get("figure_id"), "figure_id")
    visits = tuple(_visit(figure_id, item) for item in _as_list(raw.get("visits"), "visits"))
    boxes = tuple(_truth_box(figure_id, item) for item in _as_list(raw.get("boxes"), "boxes"))
    return FigureRecord(figure_id, visits, boxes)


def _visit(figure_id: str, value: object) -> TruthSegment:
    raw = _as_object(value, "visit")
    return TruthSegment(
        figure_id,
        _as_text(raw.get("camera_id"), "camera_id"),
        _as_number(raw.get("t_entry")),
        _as_number(raw.get("t_exit")),
    )


def _truth_box(figure_id: str, value: object) -> TruthBox:
    raw = _as_object(value, "box")
    box = (
        _as_number(raw.get("x1")),
        _as_number(raw.get("y1")),
        _as_number(raw.get("x2")),
        _as_number(raw.get("y2")),
    )
    return TruthBox(
        figure_id,
        _as_text(raw.get("camera_id"), "camera_id"),
        _as_frame(raw.get("frame_index")),
        box,
    )


def _write_run(root: Path, run_id: str, schedule: Schedule, frames: Mapping[str, int]) -> None:
    payload: dict[str, object] = {
        "embedder_dim": EMBEDDER_DIM,
        "frames": dict(frames),
        "fps": schedule.fps,
        "height": schedule.height,
        "n_cameras": schedule.n_cameras,
        "n_figures": schedule.n_figures,
        "run_id": run_id,
        "seconds": schedule.seconds,
        "seed": schedule.seed,
        "width": schedule.width,
    }
    _write_json(root / RUN_FILENAME, payload)


def _write_routes(root: Path, report: DemoReport) -> None:
    payload: dict[str, object] = {
        "figures": [_figure_payload(figure) for figure in report.figures],
        "run_id": report.run_id,
    }
    _write_json(root / ROUTES_FILENAME, payload)


def _figure_payload(figure: FigureReport) -> dict[str, object]:
    return {
        "figure_id": figure.figure_id,
        "found": figure.found,
        "seed_camera": figure.seed_camera,
        "seed_track": figure.seed_track,
        "segments": [
            {
                "camera_id": camera_id,
                "t_entry": t_entry,
                "t_exit": t_exit,
                "track_id": track_id,
            }
            for camera_id, track_id, t_entry, t_exit in figure.segments
        ],
        "truth": figure.truth_count,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_run_id(root: Path) -> str:
    payload = _as_object(_load_json(root / RUN_FILENAME), RUN_FILENAME)
    return _as_text(payload.get("run_id"), "run_id")


def _optional_run_id(root: Path) -> str | None:
    path = root / RUN_FILENAME
    if not path.is_file():
        return None
    return _read_run_id(root)


def _delete_track_rows(root: Path, run_id: str) -> None:
    path = root / TRACK_DB_FILENAME
    if not path.is_file():
        return
    TrackStore(path).delete_where(run_id=run_id)


def _delete_attribute_rows(root: Path, run_id: str) -> None:
    path = root / ATTRIBUTE_DB_FILENAME
    if not path.is_file():
        return
    TrackAttributeStore(path).delete_where(run_id=run_id)


def _delete_embedding_rows(root: Path, run_id: str) -> None:
    path = root / EMBEDDING_DB_FILENAME
    if not path.is_file():
        return
    SqliteVectorStore(path, EMBEDDER_DIM).delete_where(run_id=run_id)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    parsed: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings")
        parsed[key] = item
    return parsed


def _as_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return list(value)


def _as_text(value: object, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _as_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number, got {value!r}")
    return float(value)


def _as_frame(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"frame_index must be an int, got {value!r}")
    return value


def _person_rows(
    observations: Sequence[TrackObservation],
) -> dict[TrackKey, tuple[TrackObservation, ...]]:
    grouped: dict[TrackKey, list[TrackObservation]] = {}
    for row in observations:
        if row.det_class is not DetClass.PERSON:
            continue
        grouped.setdefault((row.camera_id, row.track_id), []).append(row)
    return {key: tuple(rows) for key, rows in grouped.items()}


def _truth_by_frame(
    boxes: Sequence[TruthBox],
) -> dict[tuple[str, int], tuple[tuple[str, Box], ...]]:
    grouped: dict[tuple[str, int], list[tuple[str, Box]]] = {}
    for item in boxes:
        grouped.setdefault((item.camera_id, item.frame_index), []).append(
            (item.figure_id, item.box)
        )
    return {key: tuple(rows) for key, rows in grouped.items()}


def _votes(
    rows: Sequence[TrackObservation],
    by_frame: Mapping[tuple[str, int], Sequence[tuple[str, Box]]],
) -> dict[str, int]:
    votes: dict[str, int] = {}
    for row in rows:
        figure_id = _best_figure(row.box, by_frame.get((row.camera_id, row.frame_number), ()))
        if figure_id is None:
            continue
        votes[figure_id] = votes.get(figure_id, 0) + 1
    return votes


def _best_figure(box: Box, candidates: Sequence[tuple[str, Box]]) -> str | None:
    """Best IoU at or above ``IOU_MIN``. An exact tie keeps the lower figure id."""
    chosen: str | None = None
    best = IOU_MIN
    for figure_id, truth_box in candidates:
        score = box_iou(box, truth_box)
        if score < IOU_MIN:
            continue
        if _prefer_figure(score, figure_id, best, chosen):
            chosen = figure_id
            best = score
    return chosen


def _prefer_figure(score: float, figure_id: str, best: float, chosen: str | None) -> bool:
    if chosen is None or score > best:
        return True
    if score < best:
        return False
    return figure_id < chosen


def _majority(votes: Mapping[str, int], total: int) -> str | None:
    if total <= 0 or len(votes) == 0:
        return None
    winner = min(votes, key=lambda figure_id: (-votes[figure_id], figure_id))
    if votes[winner] * 2 <= total:
        return None
    return winner


def _peak_iou(rows: Sequence[TrackObservation], boxes: Sequence[TruthBox]) -> float:
    best = 0.0
    for row in rows:
        for truth in boxes:
            if truth.camera_id != row.camera_id or truth.frame_index != row.frame_number:
                continue
            score = box_iou(row.box, truth.box)
            if score > best:
                best = score
    return best


def _better_seed(score: float, key: TrackKey, best_score: float, best_key: TrackKey | None) -> bool:
    if best_key is None:
        return True
    if score > best_score:
        return True
    if score < best_score:
        return False
    return key[1] < best_key[1]


def _metric_line(label: str, value: float, operator: str, limit: float, ok: bool) -> str:
    state = "PASS" if ok else "FAIL"
    return f"{label}={value:.4f} ({operator} {limit:.2f}) {state}"


def _exit_code(report: DemoReport) -> int:
    recall_ok, precision_ok, merge_ok = metric_marks(
        report.recall, report.precision, report.false_merge
    )
    if recall_ok and precision_ok and merge_ok:
        return 0
    return 1


def _sink(sink: TextIO | None) -> TextIO:
    if sink is None:
        return sys.stdout
    return sink


def _report_error(exc: BaseException, sink: TextIO) -> None:
    print(f"error: {type(exc).__name__}: {exc}", file=sink)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
