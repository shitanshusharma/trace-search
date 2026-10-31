"""AC-09 harness at 1% size. No latency threshold is asserted."""

from __future__ import annotations

import json
import math
import socket
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trace_search.api import ac09_bench as bench
from trace_search.api.ac09_bench import case_plan, run_cases
from trace_search.reid.sqlite_vector_store import SqliteVectorStore
from trace_search.reid.vector_store import InMemoryVectorStore

_SCALE = 0.01
_KEYS = frozenset(
    {
        "criterion",
        "case",
        "camera_hours",
        "fixture_id",
        "tracks",
        "observations",
        "observations_per_track",
        "embeddings",
        "embedding_dim",
        "cameras",
        "attribute_search_calls",
        "crop_search_calls",
        "route_builds",
        "attribute_search_p50_s",
        "attribute_search_p95_s",
        "crop_search_p50_s",
        "crop_search_p95_s",
        "search_p50_s",
        "search_p95_s",
        "route_p50_s",
        "route_p95_s",
        "attribute_search_samples_s",
        "crop_search_samples_s",
        "route_samples_s",
        "thresholds",
        "prd",
        "index_build",
        "hardware",
        "sha",
        "note",
    }
)


@pytest.fixture(autouse=True)
def _refuse_socket_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests refuse socket.socket.connect. Local SQLite does not connect."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("ac09 bench tests refuse socket.socket.connect")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def test_socket_connect_is_refused() -> None:
    with pytest.raises(OSError, match="refuse socket.socket.connect"):
        socket.socket().connect(("127.0.0.1", 9))


@pytest.fixture(scope="module")
def scaled_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[bench.CaseResult]]:
    """One 1% run of both cases. The gate index is not built here."""
    plan = case_plan("ac09", scale=_SCALE)
    assert plan.tracks == 36
    assert plan.tracks < 100
    work = tmp_path_factory.mktemp("ac09-bench")
    report = run_cases(work, scale=_SCALE)
    return work, report


def _by_case(report: list[bench.CaseResult]) -> dict[str, bench.CaseResult]:
    return {str(item["case"]): item for item in report}


def test_scaled_run_has_both_cases_and_the_json_keys(
    scaled_run: tuple[Path, list[bench.CaseResult]],
) -> None:
    work, report = scaled_run
    assert bench._repo_root() not in work.parents
    assert [item["case"] for item in report] == ["ac09", "stress-6000x512"]
    for item in report:
        assert set(item) == _KEYS
        assert item["criterion"] == "AC-09"
        assert item["note"] == "Eng dev run. AC-09 is passed only by QA from the frozen SHA."
        assert item["thresholds"] == {"search_p95_s": 2.0, "route_p95_s": 10.0}
        assert item["prd"] == {
            "search": "Search latency p95 ≤2 s for an attribute or crop query",
            "route": "Route build p95 ≤10 s",
            "index": (
                "a synthetic FX-CODE index equivalent to 12 camera-hours "
                "(track and embedding counts stated in proof)"
            ),
        }
        assert "passed" not in item


def test_scaled_counts_match_the_plan(
    scaled_run: tuple[Path, list[bench.CaseResult]],
) -> None:
    _work, report = scaled_run
    found = _by_case(report)
    ac09 = case_plan("ac09", scale=_SCALE)
    stress = case_plan("stress-6000x512", scale=_SCALE)
    assert ac09.tracks == 36
    assert ac09.observations == 720
    assert ac09.camera_hours == pytest.approx(0.12)
    assert stress.tracks == 60
    assert stress.observations == 1200
    assert stress.embedding_dim == 512
    assert stress.camera_hours == pytest.approx(0.12)
    for plan in (ac09, stress):
        item = found[plan.case]
        assert item["fixture_id"] == plan.fixture_id
        assert item["cameras"] == 6
        assert item["tracks"] == plan.tracks
        assert item["observations"] == plan.observations
        assert item["observations_per_track"] == 20
        assert item["embeddings"] == plan.embeddings
        assert item["embedding_dim"] == plan.embedding_dim
        assert item["camera_hours"] == pytest.approx(plan.camera_hours)
        assert item["attribute_search_calls"] == 100
        assert item["crop_search_calls"] == 100
        assert item["route_builds"] == 20


def test_percentiles_come_from_the_samples_and_p95_is_at_least_p50(
    scaled_run: tuple[Path, list[bench.CaseResult]],
) -> None:
    """Counts and ordering only. The 2 s and 10 s thresholds are not applied."""
    _work, report = scaled_run
    for item in report:
        attribute = item["attribute_search_samples_s"]
        crop = item["crop_search_samples_s"]
        route = item["route_samples_s"]
        assert len(attribute) == 100
        assert len(crop) == 100
        assert len(route) == 20
        assert item["attribute_search_p50_s"] == pytest.approx(bench._percentile(attribute, 50))
        assert item["attribute_search_p95_s"] == pytest.approx(bench._percentile(attribute, 95))
        assert item["crop_search_p50_s"] == pytest.approx(bench._percentile(crop, 50))
        assert item["crop_search_p95_s"] == pytest.approx(bench._percentile(crop, 95))
        assert item["route_p50_s"] == pytest.approx(bench._percentile(route, 50))
        assert item["route_p95_s"] == pytest.approx(bench._percentile(route, 95))
        assert item["attribute_search_p95_s"] >= item["attribute_search_p50_s"]
        assert item["crop_search_p95_s"] >= item["crop_search_p50_s"]
        assert item["route_p95_s"] >= item["route_p50_s"]
        assert item["search_p50_s"] == max(
            item["attribute_search_p50_s"], item["crop_search_p50_s"]
        )
        assert item["search_p95_s"] == max(
            item["attribute_search_p95_s"], item["crop_search_p95_s"]
        )
        assert item["search_p95_s"] >= item["search_p50_s"]


def test_index_build_reports_both_stores(
    scaled_run: tuple[Path, list[bench.CaseResult]],
) -> None:
    work, report = scaled_run
    expected_tracks = case_plan("ac09", scale=_SCALE).tracks
    for item in report:
        index = item["index_build"]
        assert index["tracks"] == expected_tracks
        assert index["embedding_dim"] == 256
        assert [store["store"] for store in index["stores"]] == ["in_memory", "sqlite"]
        for store in index["stores"]:
            assert store["one_batch_s"] > 0.0
            assert store["one_row_s"] > 0.0
            assert store["one_row_over_one_batch"] == pytest.approx(
                store["one_row_s"] / store["one_batch_s"]
            )
            assert store["one_row_within_3x"] is (store["one_row_s"] <= store["one_batch_s"] * 3.0)
    batch = work / "index-build" / "index-batch.sqlite"
    rows = work / "index-build" / "index-rows.sqlite"
    assert _embedding_rows(batch) == expected_tracks
    assert _embedding_rows(rows) == expected_tracks
    assert _origins(batch) == {"CODE_TEST"}
    assert _origins(rows) == {"CODE_TEST"}


def test_sqlite_rows_match_the_scaled_index(
    scaled_run: tuple[Path, list[bench.CaseResult]],
) -> None:
    work, _report = scaled_run
    ac09 = case_plan("ac09", scale=_SCALE)
    stress = case_plan("stress-6000x512", scale=_SCALE)
    assert _table_count(work / "ac09" / "tracks.sqlite", "track_observations") == ac09.observations
    assert _table_count(work / "stress-6000x512" / "tracks.sqlite", "track_observations") == (
        stress.observations
    )
    assert _distinct(work / "ac09" / "tracks.sqlite", "derived_from") == {"CODE_TEST"}
    assert _distinct(work / "stress-6000x512" / "tracks.sqlite", "derived_from") == {"CODE_TEST"}
    assert _table_count(work / "ac09" / "attributes.sqlite", "track_attributes") == ac09.tracks
    assert _table_count(work / "stress-6000x512" / "attributes.sqlite", "track_attributes") == (
        stress.tracks
    )
    assert _line_count(work / "ac09" / "audit.jsonl") == 200
    assert _line_count(work / "stress-6000x512" / "audit.jsonl") == 200


def test_scaled_run_records_hardware_and_the_checked_out_sha(
    scaled_run: tuple[Path, list[bench.CaseResult]],
) -> None:
    _work, report = scaled_run
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    for item in report:
        hardware = item["hardware"]
        assert hardware["cpu"] != ""
        assert hardware["cores"] >= 1
        assert hardware["ram_gb"] > 0.0
        assert hardware["python"].startswith("3.11.")
        assert item["sha"] == head


def test_ac09_plan_is_the_twelve_camera_hour_gate() -> None:
    plan = case_plan("ac09")
    assert plan.case == "ac09"
    assert plan.fixture_id == "FX-CODE-209"
    assert plan.cameras == 6
    assert plan.hours_per_camera == 2.0
    assert plan.tracks_per_camera == 600
    assert plan.tracks_per_camera_hour == 300
    assert plan.tracks == 3600
    assert plan.observations_per_track == 20
    assert plan.observations == 72000
    assert plan.embeddings == 3600
    assert plan.embedding_dim == 256
    assert plan.camera_hours == 12.0
    assert plan.attribute_calls == 100
    assert plan.crop_calls == 100
    assert plan.route_builds == 20
    assert plan.run_id == "run-ac09"


def test_stress_plan_is_six_thousand_tracks_at_512() -> None:
    plan = case_plan("stress-6000x512")
    assert plan.fixture_id == "FX-CODE-210"
    assert plan.cameras == 6
    assert plan.tracks_per_camera == 1000
    assert plan.tracks_per_camera_hour == 500
    assert plan.tracks == 6000
    assert plan.observations == 120000
    assert plan.embeddings == 6000
    assert plan.embedding_dim == 512
    assert plan.camera_hours == 12.0
    assert plan.run_id == "run-stress-6000x512"


def test_tiny_scale_still_builds_one_track_per_camera() -> None:
    plan = case_plan("ac09", scale=1e-4)
    assert plan.tracks_per_camera == 1
    assert plan.tracks == 6


def test_scale_rounds_to_the_nearest_track_count() -> None:
    assert bench._scale_tracks(600, 0.003) == 2


@pytest.mark.parametrize(
    "scale",
    [True, False, "0.01", None, object()],
)
def test_scale_rejects_non_numbers(scale: object) -> None:
    with pytest.raises(TypeError, match="scale"):
        case_plan("ac09", scale=scale)  # type: ignore[arg-type]


@pytest.mark.parametrize("scale", [0, 0.0, -0.1, 1.01, math.nan, math.inf, -math.inf])
def test_scale_rejects_out_of_range(scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        case_plan("ac09", scale=scale)


def test_scale_accepts_one_and_a_fraction() -> None:
    assert case_plan("ac09", scale=1).tracks == 3600
    assert case_plan("ac09", scale=1.0).tracks == 3600
    assert bench._require_scale(1) == 1.0


@pytest.mark.parametrize("case", ["nope", "", 1, None, "AC09"])
def test_unknown_case_is_rejected(case: object) -> None:
    with pytest.raises(ValueError, match="unknown case"):
        bench._require_case_name(case)


def test_cases_default_and_reject_a_bare_string() -> None:
    assert bench._require_cases(None) == ("ac09", "stress-6000x512")
    assert bench._require_cases(("stress-6000x512",)) == ("stress-6000x512",)
    with pytest.raises(TypeError, match="got str"):
        bench._require_cases("ac09")
    with pytest.raises(TypeError, match="sequence"):
        bench._require_cases(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        bench._require_cases(())
    with pytest.raises(ValueError, match="duplicate"):
        bench._require_cases(("ac09", "ac09"))
    with pytest.raises(ValueError, match="unknown case"):
        bench._require_cases(("ac09", "other"))


def test_work_dir_guards(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="does not exist"):
        run_cases(missing, scale=_SCALE, cases=("ac09",))
    stored = tmp_path / "file.sqlite"
    stored.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a directory"):
        run_cases(stored, scale=_SCALE, cases=("ac09",))
    inside = Path(__file__).resolve().parent
    with pytest.raises(ValueError, match="work dir must sit outside"):
        run_cases(inside, scale=_SCALE, cases=("ac09",))
    with pytest.raises(TypeError, match="work_dir"):
        run_cases("not-a-path", scale=_SCALE)  # type: ignore[arg-type]


def test_percentile_endpoints_and_interpolation() -> None:
    assert bench._percentile([3.0, 1.0, 2.0], 0) == 1.0
    assert bench._percentile([3.0, 1.0, 2.0], 100) == 3.0
    assert bench._percentile([1.0], 50) == 1.0
    assert bench._percentile([1.0], 95) == 1.0
    samples = [index / 100.0 for index in range(100)]
    assert bench._percentile(samples, 50) == pytest.approx(0.495)
    assert bench._percentile(samples, 95) == pytest.approx(0.9405)
    assert bench._percentile(samples, 95) >= bench._percentile(samples, 50)
    assert bench._percentile([1.0, 3.0], 95) == pytest.approx(2.9)


@pytest.mark.parametrize("percent", [True, "50", None])
def test_percentile_rejects_a_bad_percent(percent: object) -> None:
    with pytest.raises(TypeError, match="percent"):
        bench._percentile([1.0], percent)  # type: ignore[arg-type]


@pytest.mark.parametrize("percent", [-0.1, 100.1, math.nan, math.inf])
def test_percentile_rejects_a_percent_outside_the_unit_interval(percent: float) -> None:
    with pytest.raises(ValueError, match="percent"):
        bench._percentile([1.0], percent)


def test_percentile_rejects_empty_and_non_finite_samples() -> None:
    with pytest.raises(ValueError, match="at least one"):
        bench._percentile([], 50)
    with pytest.raises(TypeError, match="sample"):
        bench._percentile([True], 50)  # type: ignore[list-item]
    with pytest.raises(TypeError, match="sample"):
        bench._percentile(["1.0"], 50)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="finite"):
        bench._percentile([math.nan], 50)
    with pytest.raises(ValueError, match="finite"):
        bench._percentile([math.inf], 50)


def test_build_ratio_and_three_times_bound() -> None:
    assert bench._build_ratio(3.0, 1.0) == 3.0
    assert bench._within_3x(3.0, 1.0) is True
    assert bench._within_3x(3.0 + 1e-9, 1.0) is False
    assert bench._within_3x(0.0, 0.0) is True
    with pytest.raises(ValueError, match="one-batch"):
        bench._build_ratio(1.0, 0.0)
    with pytest.raises(ValueError, match=">="):
        bench._within_3x(-1.0, 1.0)
    with pytest.raises(TypeError, match="one_row_s"):
        bench._build_ratio(True, 1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="one_row_s"):
        bench._within_3x(True, 1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="one_batch_s"):
        bench._within_3x(1.0, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one_batch_s"):
        bench._within_3x(1.0, -1.0)
    with pytest.raises(ValueError, match="finite"):
        bench._require_build_time("one_batch_s", math.nan)


def test_sha_guard() -> None:
    assert bench._require_sha(0, "a" * 40 + "\n") == "a" * 40
    with pytest.raises(ValueError, match="failed"):
        bench._require_sha(1, "a" * 40)
    with pytest.raises(ValueError, match="commit sha"):
        bench._require_sha(0, "ab")
    with pytest.raises(ValueError, match="commit sha"):
        bench._require_sha(0, "A" * 40)
    with pytest.raises(ValueError, match="commit sha"):
        bench._require_sha(0, "g" * 40)
    with pytest.raises(ValueError, match="commit sha"):
        bench._require_sha(0, "")


def test_default_cpu_and_ram_read_proc() -> None:
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    assert bench._cpu_model() == bench._cpu_model_from_text(cpuinfo)
    assert bench._ram_gb() == pytest.approx(bench._ram_gb_from_text(meminfo))
    assert bench._core_count() >= 1
    assert bench._python_version().startswith("3.11.")


def test_cpu_model_parser(tmp_path: Path) -> None:
    assert bench._cpu_model_from_text("processor\t: 0\n") == "unknown"
    assert bench._cpu_model_from_text("model name\t: \n") == "unknown"
    assert bench._cpu_model_from_text("model name\t: Test CPU\n") == "Test CPU"
    assert bench._cpu_model_from_text("processor\t: 0\nmodel name\t: Real CPU\n") == "Real CPU"
    assert bench._cpu_model_from_text("model\t: decoy\nmodel name\t: Real CPU\n") == "Real CPU"
    missing = tmp_path / "no-cpuinfo"
    assert bench._cpu_model(missing) == "unknown"
    present = tmp_path / "cpuinfo"
    present.write_text("model name\t: Boxed CPU\n", encoding="utf-8")
    assert bench._cpu_model(present) == "Boxed CPU"


def test_ram_parser(tmp_path: Path) -> None:
    assert bench._ram_gb_from_text("MemTotal: 2097152 kB\n") == pytest.approx(2.0)
    assert bench._memtotal_kib("MemTotal: 1024") == 1024
    assert bench._memtotal_kib("MemTotal: 1 kB") == 1
    assert bench._kib_to_gb(2 * 1024 * 1024) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="missing from meminfo"):
        bench._ram_gb_from_text("MemFree: 1 kB\n")
    with pytest.raises(ValueError, match="missing a size"):
        bench._memtotal_kib("MemTotal:")
    with pytest.raises(ValueError, match="not an integer"):
        bench._memtotal_kib("MemTotal: lots kB")
    with pytest.raises(ValueError, match="positive"):
        bench._memtotal_kib("MemTotal: 0 kB")
    with pytest.raises(ValueError, match="positive"):
        bench._memtotal_kib("MemTotal: -4 kB")
    missing = tmp_path / "no-meminfo"
    with pytest.raises(ValueError, match="cannot be read"):
        bench._ram_gb(missing)
    present = tmp_path / "meminfo"
    present.write_text("MemTotal: 1048576 kB\n", encoding="utf-8")
    assert bench._ram_gb(present) == pytest.approx(1.0)


def test_core_count_rejects_an_empty_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench.os, "cpu_count", lambda: None)
    with pytest.raises(ValueError, match="unavailable"):
        bench._core_count()
    monkeypatch.setattr(bench.os, "cpu_count", lambda: 0)
    with pytest.raises(ValueError, match="unavailable"):
        bench._core_count()
    monkeypatch.setattr(bench.os, "cpu_count", lambda: 1)
    assert bench._core_count() == 1


def test_tracks_must_divide_across_cameras() -> None:
    assert bench._tracks_per_camera(36) == 6
    assert bench._tracks_per_camera(6) == 1
    with pytest.raises(TypeError, match="tracks"):
        bench._tracks_per_camera(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="multiple"):
        bench._tracks_per_camera(5)
    with pytest.raises(ValueError, match="multiple"):
        bench._tracks_per_camera(8)
    with pytest.raises(ValueError, match="multiple"):
        bench._tracks_per_camera(0)
    with pytest.raises(ValueError, match="multiple"):
        bench._tracks_per_camera(-6)


def test_one_row_add_stores_each_embedding_alone() -> None:
    rows = bench._unit_embeddings(6, 1, 256, seed=3)

    class _Recorder:
        def __init__(self) -> None:
            self.batches: list[int] = []

        def add(self, run_id: str, derived_from: str, embeddings: object) -> int:
            assert run_id == "run-index-build"
            assert derived_from == "CODE_TEST"
            batch = list(embeddings)  # type: ignore[arg-type]
            self.batches.append(len(batch))
            return len(batch)

    each = _Recorder()
    bench._time_add(each, rows, each=True)
    assert each.batches == [1, 1, 1, 1, 1, 1]
    whole = _Recorder()
    bench._time_add(whole, rows, each=False)
    assert whole.batches == [6]


def test_result_uses_the_slower_search_and_quotes_the_prd() -> None:
    plan = case_plan("ac09", scale=_SCALE)
    attribute = [0.0, 1.0]
    crop = [0.0, 0.2]
    route = [1.0, 1.5, 9.0]
    item = bench._result(plan, attribute, crop, route)
    assert item["attribute_search_p95_s"] == pytest.approx(bench._percentile(attribute, 95))
    assert item["crop_search_p50_s"] == pytest.approx(bench._percentile(crop, 50))
    assert item["route_p95_s"] == pytest.approx(bench._percentile(route, 95))
    assert item["search_p95_s"] == max(item["attribute_search_p95_s"], item["crop_search_p95_s"])
    assert item["search_p50_s"] == max(item["attribute_search_p50_s"], item["crop_search_p50_s"])
    assert item["search_p95_s"] == item["attribute_search_p95_s"]
    slower_crop = bench._result(plan, [0.0, 1.0], [0.0, 4.0], [1.0])
    assert slower_crop["crop_search_p95_s"] > slower_crop["attribute_search_p95_s"]
    assert slower_crop["search_p95_s"] == slower_crop["crop_search_p95_s"]
    assert slower_crop["search_p50_s"] == max(
        slower_crop["attribute_search_p50_s"], slower_crop["crop_search_p50_s"]
    )
    assert item["note"] == "Eng dev run. AC-09 is passed only by QA from the frozen SHA."
    assert item["thresholds"] == {"search_p95_s": 2.0, "route_p95_s": 10.0}
    assert item["prd"]["search"] == "Search latency p95 ≤2 s for an attribute or crop query"
    assert item["prd"]["route"] == "Route build p95 ≤10 s"
    assert item["criterion"] == "AC-09"
    assert item["sha"] == ""
    assert item["index_build"]["tracks"] == 0


def test_head_sha_refuses_a_directory_that_is_not_a_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="failed"):
        bench._head_sha(tmp_path)


def test_head_sha_matches_git() -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert bench._head_sha(bench._repo_root()) == head


def test_route_build_refuses_an_empty_catalogue() -> None:
    world = bench._World(
        plan=case_plan("ac09", scale=_SCALE),
        tracks=None,  # type: ignore[arg-type]
        attributes=None,  # type: ignore[arg-type]
        vectors=None,  # type: ignore[arg-type]
        service=None,  # type: ignore[arg-type]
        camera_ids=("cam-0",),
        tracklets=(),
        topology=None,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="at least one tracklet"):
        bench._time_routes(world)


def test_main_forwards_arguments_and_prints_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.bench_ac09 as script

    seen: dict[str, object] = {}

    def fake(work: Path, *, scale: float, cases: object) -> list[dict[str, str]]:
        seen["work"] = work
        seen["scale"] = scale
        seen["cases"] = cases
        return [{"case": "ac09"}]

    monkeypatch.setattr(script, "run_cases", fake)
    assert script.main(["--scale", "0.01", "--case", "ac09", "--work-dir", str(tmp_path)]) == 0
    assert seen == {"work": tmp_path, "scale": 0.01, "cases": ["ac09"]}
    printed = json.loads(capsys.readouterr().out)
    assert printed == [{"case": "ac09"}]


def test_main_removes_its_temporary_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.bench_ac09 as script

    seen: dict[str, Path] = {}

    def fake(work: Path, *, scale: float, cases: object) -> list[object]:
        seen["work"] = work
        assert work.exists()
        assert scale == 0.01
        assert cases is None
        return []

    monkeypatch.setattr(script, "run_cases", fake)
    assert script.main(["--scale", "0.01"]) == 0
    assert not seen["work"].exists()
    assert capsys.readouterr().out.strip() == "[]"


def test_script_entry_rejects_a_bad_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy

    saved = sys.modules.pop("scripts.bench_ac09", None)
    monkeypatch.setattr(sys, "argv", ["bench_ac09.py", "--scale", "nope"])
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.bench_ac09", run_name="__main__")
        assert caught.value.code != 0
    finally:
        if saved is not None:
            sys.modules["scripts.bench_ac09"] = saved


def test_main_default_scale_uses_a_temp_dir_outside_the_repo(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.bench_ac09 as script

    seen: dict[str, object] = {}

    def fake(work: Path, *, scale: float, cases: object) -> list[dict[str, str]]:
        seen["work"] = work
        seen["scale"] = scale
        seen["cases"] = cases
        resolved = work.resolve()
        repo = bench._repo_root().resolve()
        assert work.is_dir()
        assert resolved != Path.cwd().resolve()
        assert repo != resolved
        assert repo not in resolved.parents
        return [{"case": "ac09"}]

    monkeypatch.setattr(script, "run_cases", fake)
    assert script.main([]) == 0
    printed = capsys.readouterr().out
    assert printed.endswith("\n")
    assert json.loads(printed) == [{"case": "ac09"}]
    assert seen["scale"] == 1.0
    assert seen["cases"] is None
    work = seen["work"]
    assert isinstance(work, Path)
    assert not work.exists()


def test_main_rejects_a_bad_scale_before_a_run(tmp_path: Path) -> None:
    import scripts.bench_ac09 as script

    with pytest.raises(SystemExit):
        script.main(["--scale", "nope", "--work-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="unknown case"):
        script.main(["--scale", "0.01", "--case", "nope", "--work-dir", str(tmp_path)])


def test_one_track_per_camera_is_offset_along_the_chain() -> None:
    plan = case_plan("ac09", scale=1e-4)
    assert bench._track_start(plan, 0, 0) == 0.0
    assert bench._track_start(plan, 1, 0) == 10.0
    assert bench._track_start(plan, 5, 0) == 50.0


@pytest.mark.parametrize(
    "plan",
    [
        case_plan("ac09", scale=0.01),
        case_plan("ac09", scale=0.003),
        case_plan("ac09"),
        case_plan("stress-6000x512"),
    ],
    ids=["small", "two-tracks", "full", "stress"],
)
def test_last_track_on_the_last_camera_ends_at_the_window(plan: bench.CasePlan) -> None:
    """The last cam-5 track ends on the window, and its start is span minus slack.

    Slack is the chain offset plus the track duration. The last camera adds
    that offset back, so its start is ``span - slack + chain``.
    """
    assert plan.tracks_per_camera >= 2
    if plan.tracks_per_camera == 2:
        assert plan.case == "ac09"
    span = plan.hours_per_camera * 3600.0
    duration = (plan.observations_per_track - 1) * 0.05
    chain = (plan.cameras - 1) * 10.0
    slack = chain + duration
    last_track = plan.tracks_per_camera - 1
    start = bench._track_start(plan, plan.cameras - 1, last_track)
    end = start + duration
    window_end = bench._window(plan)[1]
    assert bench._track_start(plan, 0, last_track) == pytest.approx(span - slack)
    assert start == pytest.approx(span - slack + chain)
    assert end <= window_end
    assert end == pytest.approx(window_end)
    assert bench._schedule_slack(plan) == pytest.approx(slack)


def test_timers_subtract_the_start_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter((10.0, 10.25, 20.0, 20.5, 30.0, 30.75))
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(ticks))
    assert bench._time_call(lambda: None) == pytest.approx(0.25)
    monkeypatch.setattr(bench, "_crop_search", lambda _world, _crop: [])
    assert bench._time_crop(None, np.zeros((8, 8, 3), dtype=np.uint8)) == pytest.approx(0.5)  # type: ignore[arg-type]

    class _Store:
        def add(self, run_id: str, derived_from: str, embeddings: object) -> int:
            return len(tuple(embeddings))  # type: ignore[arg-type]

    rows = bench._unit_embeddings(1, 1, 8, seed=1)
    assert bench._time_add(_Store(), rows, each=False) == pytest.approx(0.75)


def test_one_batch_is_one_add_and_one_row_is_one_add_per_track(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = bench._unit_embeddings(6, 1, 256, seed=4)
    memory: list[int] = []
    sqlite_batches: list[int] = []
    real_memory = InMemoryVectorStore.add
    real_sqlite = SqliteVectorStore.add

    def spy_memory(
        self: InMemoryVectorStore,
        run_id: str,
        derived_from: str,
        embeddings: object,
    ) -> int:
        batch = tuple(embeddings)  # type: ignore[arg-type]
        memory.append(len(batch))
        return real_memory(self, run_id, derived_from, batch)

    def spy_sqlite(
        self: SqliteVectorStore,
        run_id: str,
        derived_from: str,
        embeddings: object,
    ) -> int:
        batch = tuple(embeddings)  # type: ignore[arg-type]
        sqlite_batches.append(len(batch))
        return real_sqlite(self, run_id, derived_from, batch)

    monkeypatch.setattr(InMemoryVectorStore, "add", spy_memory)
    monkeypatch.setattr(SqliteVectorStore, "add", spy_sqlite)
    measured_memory = bench._measure_memory(rows)
    measured_sqlite = bench._measure_sqlite(tmp_path, rows)
    assert memory == [6, 1, 1, 1, 1, 1, 1]
    assert sqlite_batches == [6, 1, 1, 1, 1, 1, 1]
    assert measured_memory["store"] == "in_memory"
    assert measured_sqlite["store"] == "sqlite"


def test_run_cases_defaults_to_scale_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scales: list[float] = []
    real_plan = bench.case_plan

    def recording(case: str, scale: float = 1.0) -> bench.CasePlan:
        scales.append(scale)
        return real_plan(case, _SCALE)

    def fake_run(_root: Path, plan: bench.CasePlan) -> bench.CaseResult:
        return bench._result(plan, [0.1, 0.2], [0.1, 0.4], [0.2, 0.3])

    def fake_index(_root: Path, tracks: int) -> bench._IndexBuild:
        return bench._IndexBuild(tracks=tracks, embedding_dim=256, stores=[])

    monkeypatch.setattr(bench, "case_plan", recording)
    monkeypatch.setattr(bench, "_run_case", fake_run)
    monkeypatch.setattr(bench, "_measure_index_build", fake_index)
    monkeypatch.setattr(bench, "_head_sha", lambda _repo: "a" * 40)
    report = run_cases(tmp_path)
    assert scales == [1.0, 1.0, 1.0]
    assert [item["case"] for item in report] == ["ac09", "stress-6000x512"]
    assert report[0]["sha"] == "a" * 40


def test_small_index_search_and_route_do_the_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One attribute hit per six tracks, six cameras, and a linked route.

    Scale 0.02 is 12 tracks per camera, so a second red track starts after
    60 s. No latency threshold is asserted.
    """
    plan = case_plan("ac09", scale=0.02)
    assert plan.tracks_per_camera == 12
    assert plan.tracks == 72
    world = bench._build_world(tmp_path, plan)
    assert world.vectors.count("CODE_TEST", plan.run_id) == plan.tracks
    assert bench._window(plan) == (0.0, 7200.0)
    assert world.topology.successors("cam-0") == ("cam-1",)
    assert world.topology.successors("cam-4") == ("cam-5",)
    assert world.topology.successors("cam-5") == ()
    red_ids = [track_id for track_id in range(plan.tracks_per_camera) if track_id % 6 == 0]
    found = bench._attribute_search(world, {"upper_color": "red"})
    assert len(found) == plan.cameras * len(red_ids)
    assert {hit.camera_id for hit in found} == {f"cam-{index}" for index in range(6)}
    assert min(hit.t_start for hit in found) == pytest.approx(0.0)
    assert max(hit.t_start for hit in found) > 60.0
    early = [hit for hit in found if hit.camera_id == "cam-1" and hit.track_id == 0]
    assert len(early) == 1
    assert early[0].t_start == pytest.approx(10.0)
    assert early[0].t_end - early[0].t_start == pytest.approx(0.95)
    assert early[0].score is None
    crops = bench._random_crops(2, plan.seed + 1)
    assert crops[0].shape == (8, 8, 3)
    assert not np.array_equal(crops[0], crops[1])
    ranked = bench._crop_search(world, crops[0])
    assert len(ranked) == 50
    assert {hit.camera_id for hit in ranked} == {f"cam-{index}" for index in range(6)}
    scores = [hit.score for hit in ranked]
    assert all(isinstance(score, float) for score in scores)
    assert max(score for score in scores if score is not None) > min(
        score for score in scores if score is not None
    )
    route = bench._one_route(world)
    assert route.seed_camera == "cam-0"
    assert len(route.segments) >= 2
    assert {segment.camera_id for segment in route.segments} >= {"cam-0", "cam-1"}
    timed_queries: list[tuple[dict[str, object], int]] = []
    real_attribute = bench._attribute_search

    def spy_attribute(active: bench._World, query: dict[str, object]) -> list[object]:
        hits = real_attribute(active, query)
        timed_queries.append((dict(query), len(hits)))
        return hits

    monkeypatch.setattr(bench, "_attribute_search", spy_attribute)
    world.plan = replace(world.plan, attribute_calls=1, route_builds=1)
    bench._time_attribute_searches(world)
    assert timed_queries == [({"upper_color": "red"}, plan.cameras * len(red_ids))]
    loads = {"n": 0}
    link_counts: list[int] = []
    real_load = bench.load_tracklets
    real_link = bench.link_tracklets

    def spy_load(*args: object, **kwargs: object) -> object:
        loads["n"] += 1
        return real_load(*args, **kwargs)  # type: ignore[arg-type]

    def spy_link(*args: object, **kwargs: object) -> list[object]:
        links = real_link(*args, **kwargs)  # type: ignore[arg-type]
        link_counts.append(len(links))
        return links

    monkeypatch.setattr(bench, "load_tracklets", spy_load)
    monkeypatch.setattr(bench, "link_tracklets", spy_link)
    world.plan = replace(world.plan, route_builds=1)
    samples = bench._time_routes(world)
    assert len(samples) == 1
    assert loads["n"] == 1
    assert len(link_counts) == 1
    assert link_counts[0] > 0
    assert _n_frames(tmp_path / "attributes.sqlite") == {20}
    assert _timestamp(tmp_path / "tracks.sqlite", "cam-1", 0, 0) == pytest.approx(10.0)


def _table_count(path: Path, table: str) -> int:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(f"select count(*) from {table}").fetchone()
    finally:
        connection.close()
    assert row is not None
    return int(row[0])


def _distinct(path: Path, column: str) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(f"select distinct {column} from track_observations").fetchall()
    finally:
        connection.close()
    return {str(item[0]) for item in rows}


def _embedding_rows(path: Path) -> int:
    return _table_count(path, "appearance_embeddings")


def _origins(path: Path) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "select distinct derived_from from appearance_embeddings"
        ).fetchall()
    finally:
        connection.close()
    return {str(item[0]) for item in rows}


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _n_frames(path: Path) -> set[int]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute("select distinct n_frames from track_attributes").fetchall()
    finally:
        connection.close()
    return {int(row[0]) for row in rows}


def _timestamp(path: Path, camera_id: str, track_id: int, frame_number: int) -> float:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            """
            select timestamp_seconds from track_observations
            where camera_id = ? and track_id = ? and frame_number = ?
            """,
            (camera_id, track_id, frame_number),
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return float(row[0])
