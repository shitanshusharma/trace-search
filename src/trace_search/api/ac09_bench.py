"""AC-09 latency harness for a synthetic FX-CODE index.

The ``ac09`` case is the gate: 6 cameras times 2 hours, 300 person tracks
per camera-hour (3,600 tracks), one 256-d placeholder vector per track,
one attribute row per track, and 20 observations per track. The
``stress-6000x512`` case is the larger probe and is reported only.

Crops stay in memory. Rows use ``derived_from='CODE_TEST'``. No video is
read or written. The printed numbers are an engineering run. AC-09 is
passed only by QA from the frozen SHA.

``search_p50_s`` and ``search_p95_s`` are the slower of the attribute
search and the crop search at that percentile. The attribute and crop
figures are also reported on their own.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
import numpy.typing as npt

from trace_search.api.search import SearchService
from trace_search.attributes.store import TrackAttributeStore
from trace_search.audit.log import AuditLog, init_hmac_key
from trace_search.detection.types import DetClass
from trace_search.ingest.camera_registry import CameraConfig, CameraRegistry
from trace_search.matching.linker import link_tracklets
from trace_search.matching.route import build_route
from trace_search.matching.topology import CameraTopology, TransitEdge
from trace_search.matching.tracklets import Tracklet, load_tracklets
from trace_search.reid.placeholder import UntrainedPlaceholderEmbedder
from trace_search.reid.sqlite_vector_store import SqliteVectorStore
from trace_search.reid.types import AppearanceEmbedding
from trace_search.reid.vector_store import InMemoryVectorStore
from trace_search.tracking.store import TrackStore, _db_path_outside_repo
from trace_search.tracking.types import TrackObservation

_NOTE = "Eng dev run. AC-09 is passed only by QA from the frozen SHA."
_PRD_SEARCH = "Search latency p95 ≤2 s for an attribute or crop query"
_PRD_ROUTE = "Route build p95 ≤10 s"
_PRD_INDEX = (
    "a synthetic FX-CODE index equivalent to 12 camera-hours "
    "(track and embedding counts stated in proof)"
)
_ORIGIN = "CODE_TEST"
_SEARCH_P95_S = 2.0
_ROUTE_P95_S = 10.0
_CAMERAS = 6
_HOURS_PER_CAMERA = 2.0
_OBS_PER_TRACK = 20
_ATTRIBUTE_CALLS = 100
_CROP_CALLS = 100
_ROUTE_BUILDS = 20
_AC09_PER_HOUR = 300
_AC09_DIM = 256
_STRESS_PER_CAMERA = 1000
_STRESS_PER_HOUR = 500
_STRESS_DIM = 512
_INDEX_DIM = _AC09_DIM
_FRAME_GAP_S = 0.05
_TRANSIT_MAX_S = 60.0
_BOX = (1.0, 2.0, 11.0, 21.0)
_COLORS = ("red", "blue", "black", "green", "gray", "white")
_EMBEDDER_LABEL = "untrained-placeholder-v1"
_CPUINFO = Path("/proc/cpuinfo")
_MEMINFO = Path("/proc/meminfo")
_KIB_PER_GB = 1024.0 * 1024.0
_HEX = "0123456789abcdef"

_CASE_AC09 = "ac09"
_CASE_STRESS = "stress-6000x512"
_DEFAULT_CASES = (_CASE_AC09, _CASE_STRESS)


class _Thresholds(TypedDict):
    search_p95_s: float
    route_p95_s: float


class _PrdText(TypedDict):
    search: str
    route: str
    index: str


class _Hardware(TypedDict):
    cpu: str
    cores: int
    ram_gb: float
    python: str


class _StoreBuild(TypedDict):
    store: str
    one_batch_s: float
    one_row_s: float
    one_row_over_one_batch: float
    one_row_within_3x: bool


class _IndexBuild(TypedDict):
    tracks: int
    embedding_dim: int
    stores: list[_StoreBuild]


class CaseResult(TypedDict):
    """One measured case. AC-09 is not marked passed."""

    criterion: str
    case: str
    camera_hours: float
    fixture_id: str
    tracks: int
    observations: int
    observations_per_track: int
    embeddings: int
    embedding_dim: int
    cameras: int
    attribute_search_calls: int
    crop_search_calls: int
    route_builds: int
    attribute_search_p50_s: float
    attribute_search_p95_s: float
    crop_search_p50_s: float
    crop_search_p95_s: float
    search_p50_s: float
    search_p95_s: float
    route_p50_s: float
    route_p95_s: float
    attribute_search_samples_s: list[float]
    crop_search_samples_s: list[float]
    route_samples_s: list[float]
    thresholds: _Thresholds
    prd: _PrdText
    index_build: _IndexBuild
    hardware: _Hardware
    sha: str
    note: str


@dataclass(frozen=True)
class CasePlan:
    """Counts for one case at one scale. ``camera_hours`` follows the track rate."""

    case: str
    fixture_id: str
    cameras: int
    hours_per_camera: float
    tracks_per_camera: int
    tracks_per_camera_hour: int
    observations_per_track: int
    embedding_dim: int
    attribute_calls: int
    crop_calls: int
    route_builds: int
    seed: int

    @property
    def tracks(self) -> int:
        """Person tracks in the index."""
        return self.cameras * self.tracks_per_camera

    @property
    def observations(self) -> int:
        """Observation rows. Each track contributes ``observations_per_track``."""
        return self.tracks * self.observations_per_track

    @property
    def embeddings(self) -> int:
        """One placeholder vector per track."""
        return self.tracks

    @property
    def camera_hours(self) -> float:
        """Tracks divided by the decided tracks-per-camera-hour rate."""
        return self.tracks / self.tracks_per_camera_hour

    @property
    def run_id(self) -> str:
        """Store run id for this case. It is not a person id."""
        return f"run-{self.case}"


class _VectorAdder(Protocol):
    """The ``add`` method both vector stores implement."""

    def add(
        self,
        run_id: str,
        derived_from: str,
        embeddings: Sequence[AppearanceEmbedding],
    ) -> int:
        """Store one batch and return how many rows were stored."""


@dataclass(frozen=True)
class _Spec:
    case: str
    fixture_id: str
    tracks_per_camera: int
    tracks_per_camera_hour: int
    embedding_dim: int
    seed: int


_SPECS: dict[str, _Spec] = {
    _CASE_AC09: _Spec(
        case=_CASE_AC09,
        fixture_id="FX-CODE-209",
        tracks_per_camera=_AC09_PER_HOUR * int(_HOURS_PER_CAMERA),
        tracks_per_camera_hour=_AC09_PER_HOUR,
        embedding_dim=_AC09_DIM,
        seed=209,
    ),
    _CASE_STRESS: _Spec(
        case=_CASE_STRESS,
        fixture_id="FX-CODE-210",
        tracks_per_camera=_STRESS_PER_CAMERA,
        tracks_per_camera_hour=_STRESS_PER_HOUR,
        embedding_dim=_STRESS_DIM,
        seed=210,
    ),
}


def case_plan(case: str, scale: float = 1.0) -> CasePlan:
    """Return the counts for ``case`` after ``scale``. No store is opened."""
    checked = _require_scale(scale)
    spec = _SPECS[_require_case_name(case)]
    return _plan_from_spec(spec, checked)


def run_cases(
    work_dir: Path,
    *,
    scale: float = 1.0,
    cases: Sequence[str] | None = None,
) -> list[CaseResult]:
    """Build each case, time searches and route builds, and return JSON objects.

    ``work_dir`` must already exist and must sit outside the repository.
    SQLite files are created under it. The directory is left in place.
    """
    checked_scale = _require_scale(scale)
    root = _require_work_dir(work_dir)
    selected = _require_cases(cases)
    hardware = _hardware()
    sha = _head_sha(_repo_root())
    results = [_run_case(root, case_plan(case, checked_scale)) for case in selected]
    index_build = _measure_index_build(root, case_plan(_CASE_AC09, checked_scale).tracks)
    return [_with_index_build(item, index_build, hardware, sha) for item in results]


def _plan_from_spec(spec: _Spec, scale: float) -> CasePlan:
    return CasePlan(
        case=spec.case,
        fixture_id=spec.fixture_id,
        cameras=_CAMERAS,
        hours_per_camera=_HOURS_PER_CAMERA,
        tracks_per_camera=_scale_tracks(spec.tracks_per_camera, scale),
        tracks_per_camera_hour=spec.tracks_per_camera_hour,
        observations_per_track=_OBS_PER_TRACK,
        embedding_dim=spec.embedding_dim,
        attribute_calls=_ATTRIBUTE_CALLS,
        crop_calls=_CROP_CALLS,
        route_builds=_ROUTE_BUILDS,
        seed=spec.seed,
    )


def _scale_tracks(full: int, scale: float) -> int:
    """Scale a track count. A fraction that rounds to zero still builds one track."""
    scaled = int(round(full * scale))
    if scaled < 1:
        return 1
    return scaled


def _require_scale(scale: object) -> float:
    """A finite number in ``(0, 1]``. ``bool`` is rejected."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError(f"scale must be a real number in (0, 1], got {scale!r}")
    number = float(scale)
    if not math.isfinite(number) or number <= 0.0 or number > 1.0:
        raise ValueError(f"scale must be in (0, 1], got {scale!r}")
    return number


def _require_case_name(case: object) -> str:
    if not isinstance(case, str) or case not in _SPECS:
        raise ValueError(f"unknown case {case!r}")
    return case


def _require_cases(cases: object) -> tuple[str, ...]:
    """Case ids. ``None`` selects both. A bare str is not a list of ids."""
    if cases is None:
        return _DEFAULT_CASES
    if isinstance(cases, str) or not isinstance(cases, Sequence):
        raise TypeError(f"cases must be a sequence of case ids, got {type(cases).__name__}")
    if len(cases) == 0:
        raise ValueError("cases must contain at least one case id")
    chosen: list[str] = []
    for item in cases:
        label = _require_case_name(item)
        if label in chosen:
            raise ValueError(f"duplicate case {label!r}")
        chosen.append(label)
    return tuple(chosen)


def _require_work_dir(path: object) -> Path:
    """An existing directory outside the repository."""
    if not isinstance(path, Path):
        raise TypeError(f"work_dir must be a Path, got {type(path).__name__}")
    if not path.exists():
        raise ValueError(f"work dir does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"work dir must be a directory: {path}")
    try:
        resolved = _db_path_outside_repo(path)
    except ValueError as exc:
        raise ValueError(f"work dir must sit outside the repo tree: {path}") from exc
    return resolved


def _percentile(samples: Sequence[float], percent: float) -> float:
    """Linear percentile. ``percent`` is in ``[0, 100]``."""
    if isinstance(percent, bool) or not isinstance(percent, (int, float)):
        raise TypeError(f"percent must be a real number in [0, 100], got {percent!r}")
    level = float(percent)
    if not math.isfinite(level) or level < 0.0 or level > 100.0:
        raise ValueError(f"percent must be in [0, 100], got {percent!r}")
    ordered = _finite_samples(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (level / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _finite_samples(samples: Sequence[float]) -> list[float]:
    if len(samples) == 0:
        raise ValueError("percentile needs at least one sample")
    ordered: list[float] = []
    for item in samples:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"sample must be a real number, got {item!r}")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"sample must be finite, got {item!r}")
        ordered.append(number)
    ordered.sort()
    return ordered


def _within_3x(one_row_s: float, one_batch_s: float) -> bool:
    """True when the one-row build is at most three times the one-batch build."""
    _require_build_time("one_row_s", one_row_s)
    _require_build_time("one_batch_s", one_batch_s)
    return one_row_s <= one_batch_s * 3.0


def _build_ratio(one_row_s: float, one_batch_s: float) -> float:
    """``one_row_s / one_batch_s``. A zero batch time is refused."""
    _require_build_time("one_row_s", one_row_s)
    batch = _require_build_time("one_batch_s", one_batch_s)
    if batch == 0.0:
        raise ValueError("one-batch build time must be > 0")
    return one_row_s / batch


def _require_build_time(label: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and >= 0, got {value!r}")
    return number


def _repo_root() -> Path:
    """Repository root. This file lives at ``src/trace_search/api``."""
    return Path(__file__).resolve().parents[3]


def _head_sha(repo: Path) -> str:
    """``git rev-parse HEAD``. The value is not typed by hand."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return _require_sha(completed.returncode, completed.stdout)


def _require_sha(return_code: int, stdout: str) -> str:
    if return_code != 0:
        raise ValueError("git rev-parse HEAD failed")
    text = stdout.strip()
    if len(text) != 40 or any(char not in _HEX for char in text):
        raise ValueError("git rev-parse HEAD did not return a commit sha")
    return text


def _hardware() -> _Hardware:
    return _Hardware(
        cpu=_cpu_model(),
        cores=_core_count(),
        ram_gb=_ram_gb(),
        python=_python_version(),
    )


def _python_version() -> str:
    parts = sys.version_info
    return f"{parts.major}.{parts.minor}.{parts.micro}"


def _core_count() -> int:
    count = os.cpu_count()
    if count is None or count < 1:
        raise ValueError("cpu count is unavailable")
    return count


def _cpu_model(cpuinfo: Path | None = None) -> str:
    path = _CPUINFO if cpuinfo is None else cpuinfo
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    return _cpu_model_from_text(text)


def _cpu_model_from_text(text: str) -> str:
    for line in text.splitlines():
        if not line.startswith("model name"):
            continue
        value = line.partition(":")[2].strip()
        if value != "":
            return value
    return "unknown"


def _ram_gb(meminfo: Path | None = None) -> float:
    path = _MEMINFO if meminfo is None else meminfo
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"meminfo cannot be read: {path}") from exc
    return _ram_gb_from_text(text)


def _ram_gb_from_text(text: str) -> float:
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            return _kib_to_gb(_memtotal_kib(line))
    raise ValueError("MemTotal is missing from meminfo")


def _memtotal_kib(line: str) -> int:
    parts = line.split()
    if len(parts) < 2:
        raise ValueError("MemTotal is missing a size")
    try:
        kib = int(parts[1])
    except ValueError as exc:
        raise ValueError("MemTotal is not an integer") from exc
    if kib <= 0:
        raise ValueError(f"MemTotal must be positive, got {kib}")
    return kib


def _kib_to_gb(kib: int) -> float:
    return kib / _KIB_PER_GB


def _run_case(root: Path, plan: CasePlan) -> CaseResult:
    work = root / plan.case
    work.mkdir(parents=True, exist_ok=True)
    world = _build_world(work, plan)
    attribute = _time_attribute_searches(world)
    crop = _time_crop_searches(world)
    route = _time_routes(world)
    return _result(plan, attribute, crop, route)


def _build_world(work: Path, plan: CasePlan) -> _World:
    tracks = TrackStore(work / "tracks.sqlite")
    attributes = TrackAttributeStore(work / "attributes.sqlite")
    vectors = InMemoryVectorStore(plan.embedding_dim)
    _write_tracks(tracks, plan)
    _write_attributes(attributes, plan)
    vectors.add(plan.run_id, _ORIGIN, _embeddings(plan))
    service = _service(work, tracks, attributes, vectors, plan.embedding_dim)
    camera_ids = _camera_ids(plan.cameras)
    tracklets = tuple(load_tracklets(tracks, plan.run_id))
    return _World(
        plan=plan,
        tracks=tracks,
        attributes=attributes,
        vectors=vectors,
        service=service,
        camera_ids=camera_ids,
        tracklets=tracklets,
        topology=_topology(camera_ids),
    )


@dataclass
class _World:
    plan: CasePlan
    tracks: TrackStore
    attributes: TrackAttributeStore
    vectors: InMemoryVectorStore
    service: SearchService
    camera_ids: tuple[str, ...]
    tracklets: tuple[Tracklet, ...]
    topology: CameraTopology


def _service(
    work: Path,
    tracks: TrackStore,
    attributes: TrackAttributeStore,
    vectors: InMemoryVectorStore,
    dim: int,
) -> SearchService:
    key_path = work / "audit.hmac"
    init_hmac_key(key_path)
    audit = AuditLog(work / "audit.jsonl", key_path=key_path)
    audit.initialize()
    embedder = UntrainedPlaceholderEmbedder(dim=dim, seed=0)
    return SearchService(tracks, attributes, vectors, embedder, audit)


def _write_tracks(store: TrackStore, plan: CasePlan) -> None:
    store.write(plan.run_id, plan.fixture_id, _ORIGIN, _observations(plan))


def _write_attributes(store: TrackAttributeStore, plan: CasePlan) -> None:
    for camera_id in _camera_ids(plan.cameras):
        for track_id in range(plan.tracks_per_camera):
            _write_one_attribute(store, plan, camera_id, track_id)


def _write_one_attribute(
    store: TrackAttributeStore,
    plan: CasePlan,
    camera_id: str,
    track_id: int,
) -> None:
    store.write(
        camera_id,
        track_id,
        plan.run_id,
        _ORIGIN,
        _COLORS[track_id % len(_COLORS)],
        "black",
        track_id % 2 == 0,
        0.9,
        0.8,
        plan.observations_per_track,
    )


def _observations(plan: CasePlan) -> list[TrackObservation]:
    rows: list[TrackObservation] = []
    span = plan.hours_per_camera * 3600.0
    step = span / plan.tracks_per_camera
    for camera_id in _camera_ids(plan.cameras):
        for track_id in range(plan.tracks_per_camera):
            rows.extend(
                _track_rows(camera_id, track_id, track_id * step, plan.observations_per_track)
            )
    return rows


def _track_rows(
    camera_id: str,
    track_id: int,
    start: float,
    observations: int,
) -> list[TrackObservation]:
    return [
        TrackObservation(
            camera_id=camera_id,
            track_id=track_id,
            det_class=DetClass.PERSON,
            frame_number=frame,
            timestamp_seconds=start + frame * _FRAME_GAP_S,
            box=_BOX,
            score=0.9,
        )
        for frame in range(observations)
    ]


def _embeddings(plan: CasePlan) -> list[AppearanceEmbedding]:
    return _unit_embeddings(
        plan.cameras,
        plan.tracks_per_camera,
        plan.embedding_dim,
        plan.seed,
    )


def _unit_embeddings(
    cameras: int,
    tracks_per_camera: int,
    dim: int,
    seed: int,
) -> list[AppearanceEmbedding]:
    total = cameras * tracks_per_camera
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((total, dim))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    unit = np.asarray(raw / norms, dtype=np.float64)
    rows: list[AppearanceEmbedding] = []
    cursor = 0
    for camera_id in _camera_ids(cameras):
        for track_id in range(tracks_per_camera):
            vector = tuple(float(value) for value in unit[cursor])
            cursor += 1
            rows.append(
                AppearanceEmbedding(
                    camera_id=camera_id,
                    track_id=track_id,
                    frame_number=0,
                    vector=vector,
                    embedder_name=_EMBEDDER_LABEL,
                    placeholder=True,
                )
            )
    return rows


def _camera_ids(count: int) -> tuple[str, ...]:
    return tuple(f"cam-{index}" for index in range(count))


def _topology(camera_ids: Sequence[str]) -> CameraTopology:
    registry = CameraRegistry()
    for index, camera_id in enumerate(camera_ids):
        registry.register(CameraConfig(camera_id, "hikvision", float(index), 0.0, 0.0))
    edges = [
        TransitEdge(camera_ids[index], camera_ids[index + 1], 0.0, _TRANSIT_MAX_S)
        for index in range(len(camera_ids) - 1)
    ]
    return CameraTopology(registry, edges)


def _window(plan: CasePlan) -> tuple[float, float]:
    return (0.0, plan.hours_per_camera * 3600.0)


def _time_attribute_searches(world: _World) -> list[float]:
    samples: list[float] = []
    query: dict[str, object] = {"upper_color": "red"}
    for _index in range(world.plan.attribute_calls):
        samples.append(_time_call(lambda: _attribute_search(world, query)))
    return samples


def _attribute_search(world: _World, query: Mapping[str, object]) -> None:
    world.service.by_attributes(
        "bench-operator",
        "case-ac09",
        "latency-bench",
        query,
        _window(world.plan),
        world.camera_ids,
        limit=50,
        run_id=world.plan.run_id,
    )


def _time_crop_searches(world: _World) -> list[float]:
    crops = _random_crops(world.plan.crop_calls, world.plan.seed + 1)
    samples: list[float] = []
    for crop in crops:
        samples.append(_time_crop(world, crop))
    return samples


def _time_crop(world: _World, crop: npt.NDArray[np.uint8]) -> float:
    started = time.perf_counter()
    _crop_search(world, crop)
    return time.perf_counter() - started


def _crop_search(world: _World, crop: npt.NDArray[np.uint8]) -> None:
    world.service.by_example_crop(
        "bench-operator",
        "case-ac09",
        "latency-bench",
        crop,
        _window(world.plan),
        world.camera_ids,
        k=50,
        run_id=world.plan.run_id,
    )


def _random_crops(count: int, seed: int) -> list[npt.NDArray[np.uint8]]:
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, 256, size=(count, 8, 8, 3), dtype=np.uint8)
    crops: list[npt.NDArray[np.uint8]] = []
    for index in range(count):
        crops.append(np.array(drawn[index], dtype=np.uint8, copy=True))
    return crops


def _time_routes(world: _World) -> list[float]:
    if len(world.tracklets) == 0:
        raise ValueError("route build needs at least one tracklet")
    samples: list[float] = []
    for _index in range(world.plan.route_builds):
        samples.append(_time_call(lambda: _one_route(world)))
    return samples


def _one_route(world: _World) -> None:
    links = link_tracklets(
        world.tracklets,
        world.topology,
        world.vectors,
        world.attributes,
        run_id=world.plan.run_id,
        derived_from=_ORIGIN,
    )
    build_route(world.tracklets[0], world.tracklets, links)


def _time_call(call: Callable[[], object]) -> float:
    started = time.perf_counter()
    call()
    return time.perf_counter() - started


def _result(
    plan: CasePlan,
    attribute: list[float],
    crop: list[float],
    route: list[float],
) -> CaseResult:
    attribute_p50 = _percentile(attribute, 50)
    attribute_p95 = _percentile(attribute, 95)
    crop_p50 = _percentile(crop, 50)
    crop_p95 = _percentile(crop, 95)
    route_p50 = _percentile(route, 50)
    route_p95 = _percentile(route, 95)
    return CaseResult(
        criterion="AC-09",
        case=plan.case,
        camera_hours=plan.camera_hours,
        fixture_id=plan.fixture_id,
        tracks=plan.tracks,
        observations=plan.observations,
        observations_per_track=plan.observations_per_track,
        embeddings=plan.embeddings,
        embedding_dim=plan.embedding_dim,
        cameras=plan.cameras,
        attribute_search_calls=len(attribute),
        crop_search_calls=len(crop),
        route_builds=len(route),
        attribute_search_p50_s=attribute_p50,
        attribute_search_p95_s=attribute_p95,
        crop_search_p50_s=crop_p50,
        crop_search_p95_s=crop_p95,
        search_p50_s=max(attribute_p50, crop_p50),
        search_p95_s=max(attribute_p95, crop_p95),
        route_p50_s=route_p50,
        route_p95_s=route_p95,
        attribute_search_samples_s=attribute,
        crop_search_samples_s=crop,
        route_samples_s=route,
        thresholds=_Thresholds(search_p95_s=_SEARCH_P95_S, route_p95_s=_ROUTE_P95_S),
        prd=_PrdText(search=_PRD_SEARCH, route=_PRD_ROUTE, index=_PRD_INDEX),
        index_build=_empty_index_build(),
        hardware=_empty_hardware(),
        sha="",
        note=_NOTE,
    )


def _empty_index_build() -> _IndexBuild:
    """Placeholder until the shared index-build measurement is attached."""
    return _IndexBuild(tracks=0, embedding_dim=_INDEX_DIM, stores=[])


def _empty_hardware() -> _Hardware:
    return _Hardware(cpu="", cores=0, ram_gb=0.0, python="")


def _with_index_build(
    item: CaseResult,
    index_build: _IndexBuild,
    hardware: _Hardware,
    sha: str,
) -> CaseResult:
    item["index_build"] = index_build
    item["hardware"] = hardware
    item["sha"] = sha
    return item


def _measure_index_build(root: Path, tracks: int) -> _IndexBuild:
    """Time a 256-d index in one batch and in one-row batches, per store."""
    per_camera = _tracks_per_camera(tracks)
    work = root / "index-build"
    work.mkdir(parents=True, exist_ok=True)
    rows = _unit_embeddings(_CAMERAS, per_camera, _INDEX_DIM, seed=1)
    stores = [
        _measure_memory(rows),
        _measure_sqlite(work, rows),
    ]
    return _IndexBuild(tracks=len(rows), embedding_dim=_INDEX_DIM, stores=stores)


def _tracks_per_camera(tracks: int) -> int:
    """Split ``tracks`` evenly across the six cameras."""
    if isinstance(tracks, bool) or not isinstance(tracks, int):
        raise TypeError(f"tracks must be an int, got {tracks!r}")
    if tracks < _CAMERAS or tracks % _CAMERAS != 0:
        raise ValueError(f"tracks must be a positive multiple of {_CAMERAS}, got {tracks}")
    return tracks // _CAMERAS


def _measure_memory(rows: Sequence[AppearanceEmbedding]) -> _StoreBuild:
    dim = _INDEX_DIM
    one_batch = _time_add(InMemoryVectorStore(dim), rows, each=False)
    one_row = _time_add(InMemoryVectorStore(dim), rows, each=True)
    return _store_build("in_memory", one_batch, one_row)


def _measure_sqlite(work: Path, rows: Sequence[AppearanceEmbedding]) -> _StoreBuild:
    dim = _INDEX_DIM
    batch_store = SqliteVectorStore(work / "index-batch.sqlite", dim)
    row_store = SqliteVectorStore(work / "index-rows.sqlite", dim)
    one_batch = _time_add(batch_store, rows, each=False)
    one_row = _time_add(row_store, rows, each=True)
    return _store_build("sqlite", one_batch, one_row)


def _time_add(store: _VectorAdder, rows: Sequence[AppearanceEmbedding], *, each: bool) -> float:
    started = time.perf_counter()
    if each:
        for row in rows:
            store.add("run-index-build", _ORIGIN, (row,))
    else:
        store.add("run-index-build", _ORIGIN, rows)
    return time.perf_counter() - started


def _store_build(label: str, one_batch_s: float, one_row_s: float) -> _StoreBuild:
    return _StoreBuild(
        store=label,
        one_batch_s=one_batch_s,
        one_row_s=one_row_s,
        one_row_over_one_batch=_build_ratio(one_row_s, one_batch_s),
        one_row_within_3x=_within_3x(one_row_s, one_batch_s),
    )
